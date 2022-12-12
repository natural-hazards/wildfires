import io
import sys

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView

from map.folium import FoliumMap


class UIApp(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        self._folium_map = None

        self.__initUI()

    def __initUI(self) -> None:

        WINDOW_WIDTH = 1000
        WINDOW_HEIGHT = 800

        vbox = QVBoxLayout(self)

        self._web_view = QWebEngineView()
        self.__loadMap()

        vbox.addWidget(self._web_view)
        self.setLayout(vbox)
        self.setGeometry(500, 500, WINDOW_WIDTH, WINDOW_HEIGHT)

        # self.setWindowTitle('')
        self.show()

    def __loadMap(self) -> None:

        AK_COORDINATES = (60.160507, -153.369141)

        MAP_SHAPE = (800, 500)
        ZOOM_START = 4

        self._folium_map = FoliumMap(location=AK_COORDINATES, shape=MAP_SHAPE, zoom_start=ZOOM_START)

        io_bytes = io.BytesIO()
        self._folium_map.map.save(io_bytes, close_file=False)

        self._web_view.setHtml(io_bytes.getvalue().decode())

    def __loadDataset(self) -> None:

        raise NotImplementedError


if __name__ == "__main__":

    app = QApplication(sys.argv)
    okno = UIApp()
    sys.exit(app.exec_())
