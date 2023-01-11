
from enum import Enum
from PyQt5.QtWidgets import QComboBox, QDialog, QFormLayout, QHBoxLayout, QDialogButtonBox


class UIPrelude(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self.__initUI()

    def __initUI(self) -> None:

        hbox = QHBoxLayout()

        self._cb_unit = QComboBox()
        self._cb_unit.addItem('FireCCI v5.1')
        self._cb_unit.addItem('MTBS')
        self._cb_unit.setCurrentIndex(0)

        hbox.addWidget(self._cb_unit)

        # cancel and ok buttons
        self._button_box = QDialogButtonBox(QDialogButtonBox.Ok)

        # define layout
        layout = QFormLayout()
        layout.addRow(hbox)
        layout.addRow(self._button_box)

        self.setLayout(layout)
        self.setWindowTitle('Select data set')

        # set listeners
        self._button_box.accepted.connect(self.accept)

    def getSelectedCollectionID(self) -> int:

        return self._cb_unit.currentIndex()
