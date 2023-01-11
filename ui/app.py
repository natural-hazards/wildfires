import io
import sys

from jinja2 import Template

import ee as earthengine
import folium

from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView

from ui.dialog import QAreaDialog

from procs.geom import RectangleArea
from map.folium import FoliumMap
from utils.timer import elapsed_timer


class UIApp(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        self._folium_map = None
        self._select_ds = None

        earthengine.Initialize()

        self.__initUI()

    def __initUI(self) -> None:

        WINDOW_WIDTH = 1000
        WINDOW_HEIGHT = 800

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()

        self._web_view = QWebEngineView()

        # download request slot
        self._web_view.page().profile().downloadRequested.connect(
            self.__handleDownloadRequest
        )

        self.__loadMap()
        # self.__loadFireCollections_CCI()
        self.__loadFireCollections_MTBS()

        # add layer control
        layer_control = folium.LayerControl(autoZIndex=False)
        self._folium_map.map.add_child(layer_control)

        #
        self._button_add_area = QPushButton('Add area')
        self._button_add_area.clicked.connect(self.__addRectangleArea)

        self._button_export_map = QPushButton('Export map')
        self._button_export_map.clicked.connect(self.__exportMap)

        hbox.addWidget(self._button_add_area)
        hbox.addWidget(self._button_export_map)

        # add widgets
        vbox.addLayout(hbox)
        vbox.addWidget(self._web_view)

        self.setLayout(vbox)
        self.setGeometry(500, 500, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.setWindowTitle('Select areas')
        self.__renderMap()
        self.show()

    def __loadMap(self) -> None:

        AK_COORDINATES = (60.160507, -153.369141)

        MAP_SHAPE = (800, 500)
        ZOOM_START = 4

        self._folium_map = FoliumMap(location=AK_COORDINATES, shape=MAP_SHAPE, zoom_start=ZOOM_START)

    def __loadFireCollections_CCI(self) -> None:

        from earthengine import ds_fire

        # load FireCII v5.1 collection
        CONFIDENCE_LEVEL = 70
        OPACITY = .7

        visualisation_params = {
            'min': CONFIDENCE_LEVEL,
            'max': 100,
            'opacity': OPACITY,
            'palette': ['red', 'orange', 'yellow']
        }

        ds_name = 'FireCCI v5.1 (all)'
        with elapsed_timer('Loading {}'.format(ds_name)):
            burn_area = ds_fire.EarthEngineFireDatasets.FireCII.getBurnArea(CONFIDENCE_LEVEL)
            map = self._folium_map.map
            map.addGoogleEarthEngineLayer(burn_area, visualisation_params, ds_name, show=False)

        for year in range(2001, 2021):
            start_date = '{}-01-01'.format(year)
            end_date = '{}-01-01'.format(year + 1)
            ds_name = 'FireCII v5.1 ({})'.format(year)

            # loading fire collection
            with elapsed_timer('Loading {}'.format(ds_name)):
                burn_area = ds_fire.EarthEngineFireDatasets.FireCII.getBurnArea(CONFIDENCE_LEVEL, start_date, end_date)
                map = self._folium_map.map
                map.addGoogleEarthEngineLayer(burn_area, visualisation_params, ds_name, show=False)

        self.__renderMap()

    def __loadFireCollections_MTBS(self) -> None:

        from earthengine import ds_fire

        # load MTBS fire collection
        SEVERITY_FROM = 3
        SEVERITY_TO = 4
        OPACITY = .7

        visualisation_params = {
            'min': SEVERITY_FROM,
            'max': SEVERITY_TO,
            'opacity': OPACITY,
            'palette': ['#FFFF00', '#FF0000']
        }

        ds_name = 'MTBS (all)'
        with elapsed_timer('Loading {}'.format(ds_name)):
            burn_area = ds_fire.EarthEngineFireDatasets.FireMTBS.getBurnArea(SEVERITY_FROM, SEVERITY_TO)
            map = self._folium_map.map
            map.addGoogleEarthEngineLayer(burn_area, visualisation_params, ds_name, show=False)

        for year in range(2001, 2021):
            start_date = '{}-01-01'.format(year)
            end_date = '{}-01-01'.format(year + 1)
            ds_name = 'MTBS ({})'.format(year)

            # loading fire collection
            with elapsed_timer('Loading {}'.format(ds_name)):
                burn_area = ds_fire.EarthEngineFireDatasets.FireMTBS.getBurnArea(SEVERITY_FROM, SEVERITY_TO, start_date, end_date)
                map = self._folium_map.map
                map.addGoogleEarthEngineLayer(burn_area, visualisation_params, ds_name, show=False)

        self.__renderMap()

    def __renderMap(self) -> None:

        io_bytes = io.BytesIO()

        self._folium_map.map.save(io_bytes, close_file=False)
        self._web_view.setHtml(io_bytes.getvalue().decode())

    def __exportMap(self) -> None:

        def captureImage(fn: str) -> None:

            size = self._web_view.contentsRect()
            img = QImage(size.width(), size.height(), QImage.Format_ARGB32)

            self._web_view.render(img)
            img.save(fn)

        dialog_title = 'Save map as image'
        path, _ = QFileDialog.getSaveFileName(self, dialog_title, 'map.png')

        if path:
            captureImage(path)

    def __addRectangleArea(self) -> None:

        def convert_distance(v, unit) -> float:

            if unit == 0:  # m
                return v
            elif unit == 1:  # km
                return v * 1000.
            elif unit == 2:  # px
                return v * 500.

        def draw_rectangle(bounds) -> None:

            # js injection to folium map (draw rectangle)
            map = self._folium_map.map

            js = Template(
                """
                var map = {{map}}
                
                var params = {{bounds}}; 
                var rectangle = new L.Rectangle(params);
                
                drawnItems.addLayer(rectangle)
                """
            ).render(map=map.get_name(), bounds=bounds)

            # render page
            self._web_view.page().runJavaScript(js)

        # open dialog to specify area size
        dialog = QAreaDialog(self)
        if dialog.exec():
            parms_area = dialog.getAreaGeometry()

            rectangle_area = RectangleArea()
            rectangle_area.center = (parms_area['center_lat'], parms_area['center_lon'])
            rectangle_area.width = convert_distance(parms_area['area_width'], parms_area['unit'])
            rectangle_area.height = convert_distance(parms_area['area_height'], parms_area['unit'])

            draw_rectangle(rectangle_area.bounds)

    def __handleDownloadRequest(self, request) -> None:

        dialog_title = 'Save GeoJSON File'
        path, _ = QFileDialog.getSaveFileName(self, dialog_title, request.suggestedFileName())

        if path:
            request.setPath(path)
            request.accept()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    ui = UIApp()

    sys.exit(app.exec_())
