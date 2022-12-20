import re

from PyQt5.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout
from PyQt5.QtWidgets import QLineEdit, QLabel, QMessageBox
from PyQt5.QtGui import QDoubleValidator

from ui.validator import ValidLatitude


class QAreaDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.__initUI()

    def __initUI(self) -> None:

        # area center (group)
        self._group_area_center = QGroupBox('Area center', self)

        self._center_layout = QHBoxLayout(self)
        self._group_area_center.setLayout(self._center_layout)

        self._center_lat = QLineEdit(self)
        self._center_lat.setValidator(ValidLatitude())
        self._center_lat.textChanged[str].connect(self.__latitudeOnChanged)

        self._center_lon = QLineEdit(self)
        self._center_lon.setValidator(QDoubleValidator(-180., 180., 15))

        self._center_layout.addWidget(QLabel('Latitude:'))
        self._center_layout.addWidget(self._center_lat)
        self._center_layout.addWidget(QLabel('Longitude:'))
        self._center_layout.addWidget(self._center_lon)

        # area size (group)
        self._group_area_size = QGroupBox('Area size', self)

        self._size_layout = QHBoxLayout(self)
        self._group_area_size.setLayout(self._size_layout)

        self._area_width = QLineEdit(self)
        self._area_width.setValidator(QDoubleValidator(0.25, 4e5, 2))

        self._area_height = QLineEdit(self)
        self._area_height.setValidator(QDoubleValidator(0.25, 4e5, 2))

        # combo box (distance)
        self._cb_unit = QComboBox()
        self._cb_unit.addItem('m')
        self._cb_unit.addItem('km')
        self._cb_unit.addItem('px')
        self._cb_unit.setCurrentIndex(1)

        self._size_layout.addWidget(QLabel('Width:'))
        self._size_layout.addWidget(self._area_width)
        self._size_layout.addWidget(QLabel('Height:'))
        self._size_layout.addWidget(self._area_height)
        self._size_layout.addWidget(self._cb_unit)

        # cancel and ok buttons
        self._button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        # define layout
        layout = QFormLayout()
        layout.addRow(self._group_area_center)
        layout.addRow(self._group_area_size)
        layout.addRow(self._button_box)

        self.setLayout(layout)
        self.setWindowTitle('Add area')

        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

    def __latitudeOnChanged(self, text):

        re_float = r'[-+]?(?:\d*\.*\d+)'
        re_pattern = r'{}\,\s*{}'.format(re_float, re_float)

        if re.fullmatch(re_pattern, text):
            lat_lon = text.split(',')

            lat = lat_lon[0].strip()
            lon = lat_lon[1].strip()

            self._center_lat.setText(lat)
            self._center_lon.setText(lon)
            self._center_lon.setFocus()

    @staticmethod
    def __openAlertDialog(text_alert: str) -> None:

        msg_box = QMessageBox()
        msg_box.setText(text_alert)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setWindowTitle('Missing data')
        msg_box.exec()

    def accept(self):

        text_alert = ''

        if self._center_lat.text() == '' and self._center_lon.text() == '':
            text_alert = 'Missing center (latitude and longitude)'
        elif self._center_lat.text() == '':
            text_alert = 'Missing center (latitude)'
        elif self._center_lon.text() == '':
            text_alert = 'Missing center (longitude)'

        if self._area_width.text() == '' and self._area_height.text() == '':
            text_alert = '{}\nMissing area (width and height)'.format(text_alert)
        elif self._area_width.text() == '':
            text_alert = '{}\nMissing area (width)'.format(text_alert)
        elif self._area_height.text() == '':
            text_alert = '{}\nMissing area (height)'.format(text_alert)

        if text_alert == '':
            super().accept()
        else:
            self.__openAlertDialog(text_alert)

    def getAreaGeometry(self) -> dict:

        geometry = {}

        center_lat = float(self._center_lat.text())
        geometry['center_lat'] = center_lat

        center_lon = float(self._center_lon.text())
        geometry['center_lon'] = center_lon

        area_width = float(self._area_width.text())
        geometry['area_width'] = area_width

        area_height = float(self._area_height.text())
        geometry['area_height'] = area_height

        id_unit = self._cb_unit.currentIndex()
        geometry['unit'] = id_unit

        return geometry

