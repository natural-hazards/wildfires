import io
import sys

import ee as earthengine
import folium

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView

from map.folium import FoliumMap


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

        self._web_view = QWebEngineView()

        self.__loadMap()
        self.__loadFireCollections()

        # add layer control
        layer_control = folium.LayerControl(autoZIndex=False)
        self._folium_map.map.add_child(layer_control)

        # add widgets
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

        visualisation_params = {
            'min': CONFIDENCE_LEVEL,
            'max': 100,
            'opacity': .7,
            'palette': ['red', 'orange', 'yellow']
        }

        burn_area = ds.EarthEngineFireDataset.FireCII.getBurnArea(CONFIDENCE_LEVEL)
        self._folium_map.map.addGoogleEarthEngineLayer(burn_area, visualisation_params, 'FireCCI v5.1')

        self.__renderMap()

    def __renderMap(self) -> None:

        io_bytes = io.BytesIO()
        self._folium_map.map.save(io_bytes, close_file=False)

        self._web_view.setHtml(io_bytes.getvalue().decode())


if __name__ == "__main__":

    app = QApplication(sys.argv)
    okno = UIApp()
    sys.exit(app.exec_())
