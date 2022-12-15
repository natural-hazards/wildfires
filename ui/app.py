import io
import sys

from jinja2 import Template

import ee as earthengine
import folium

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

        vbox = QVBoxLayout(self)
        hbox = QHBoxLayout(self)

        self._web_view = QWebEngineView()

        # download request slot
        self._web_view.page().profile().downloadRequested.connect(
            self.__handleDownloadRequest
        )

        self.__loadMap()
        # self.__loadFireCollections()

        # add layer control
        layer_control = folium.LayerControl(autoZIndex=False)
        self._folium_map.map.add_child(layer_control)

        #
        button_add_area = QPushButton('Add area')
        button_add_area.clicked.connect(self.__addRectangleArea)

        button_export_map = QPushButton('Export map')
        button_export_map.clicked.connect(self.__exportMap)

        hbox.addWidget(button_add_area)
        hbox.addWidget(button_export_map)

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

        self._folium_map = FoliumMap(location=AK_COORDINATES,
                                     shape=MAP_SHAPE,
                                     zoom_start=ZOOM_START)

    def __loadFireCollections(self) -> None:

        from earthengine import ds

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
            burn_area = ds.EarthEngineFireDatasets.FireCII.getBurnArea(CONFIDENCE_LEVEL)
            self._folium_map.map.addGoogleEarthEngineLayer(burn_area,
                                                           visualisation_params,
                                                           ds_name,
                                                           show=False)

        for year in range(2001, 2021):
            start_date = '{}-01-01'.format(year)
            end_date = '{}-01-01'.format(year + 1)
            ds_name = 'FireCII v5.1 ({})'.format(year)

            # loading fire collection
            with elapsed_timer('Loading {}'.format(ds_name)):
                burn_area = ds.EarthEngineFireDatasets.FireCII.getBurnArea(CONFIDENCE_LEVEL, start_date, end_date)

                self._folium_map.map.addGoogleEarthEngineLayer(burn_area,
                                                               visualisation_params,
                                                               ds_name,
                                                               show=False)

        self.__renderMap()

    def __renderMap(self) -> None:

        io_bytes = io.BytesIO()
        self._folium_map.map.save(io_bytes, close_file=False)

        self._web_view.setHtml(io_bytes.getvalue().decode())

    def __exportMap(self) -> None:

        print('Export map')

    def __addRectangleArea(self) -> None:

        def convert_distance(v, unit) -> float:

            if unit == 0:  # m
                return v
            elif unit == 1:  # km
                return v * 1000.
            elif unit == 2:  # px
                return v * 250.

        def draw_rectangle(bounds) -> None:
            # injection to folium
            js = Template(
                """
                var map = {{map}}
                
                var params = {{bounds}}; 
                var polygon = new L.Polygon(params);
                
                drawnItems.addLayer(polygon)
                """
            ).render(map=self._folium_map.map.get_name(), bounds=bounds)
            # run rendering
            self._web_view.page().runJavaScript(js)

        # run dialog
        dialog = QAreaDialog(self)
        if dialog.exec():
            parms_area = dialog.getAreaGeometry()

            rectangle_area = RectangleArea()
            rectangle_area.center = (parms_area['center_lat'], parms_area['center_lon'])
            rectangle_area.width = convert_distance(parms_area['area_width'], parms_area['unit'])
            rectangle_area.height = convert_distance(parms_area['area_height'], parms_area['unit'])

            draw_rectangle(rectangle_area.bounds)

    def __handleDownloadRequest(self, request) -> None:

        path, _ = QFileDialog.getSaveFileName(self, "Save GeoJSON File", request.suggestedFileName())

        if path:
            request.setPath(path)
            request.accept()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.aboutQt()
    ui = UIApp()
    sys.exit(app.exec_())
