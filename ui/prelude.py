
from PyQt5.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel
from PyQt5.QtWidgets import QGroupBox

from earthengine.ds import FireCIIAvailability


class UIPrelude(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self._cb_years = None

        self.__initUI()

    def __initUI(self) -> None:

        hbox = QHBoxLayout()

        self._group_dataset = QGroupBox('Collection')
        self._hbox_collection = QHBoxLayout()

        self._cb_collection = QComboBox()
        self._cb_collection.addItem('FireCCI v5.1')
        self._cb_collection.addItem('MTBS')
        self._cb_collection.setCurrentIndex(0)

        self._hbox_collection.addWidget(self._cb_collection)
        self._group_dataset.setLayout(self._hbox_collection)

        self._group_period = QGroupBox('Period')
        self._hbox_period = QHBoxLayout()

        self._cb_period = QComboBox()
        self._cb_period.addItem('Years')
        self._cb_period.addItem('Months')
        self._cb_period.setCurrentIndex(0)

        self._hbox_period.addWidget(self._cb_period, 0)
        self._group_period.setLayout(self._hbox_period)

        hbox.addWidget(self._group_dataset)
        hbox.addWidget(self._group_period)

        # cancel and ok buttons
        self._button_box = QDialogButtonBox(QDialogButtonBox.Ok)

        # define layout
        layout = QFormLayout()
        layout.addRow(hbox)
        layout.addRow(self._button_box)

        self.setLayout(layout)
        self.setWindowTitle('Select data set')

        # combo box
        self._cb_collection.currentIndexChanged.connect(self.collectionSelectionChanged)
        self._cb_period.currentIndexChanged.connect(self.periodSelectionChanged)

        # set listeners
        self._button_box.accepted.connect(self.accept)

    def getSelectedCollectionID(self) -> int:

        return self._cb_collection.currentIndex()

    def getSelectedPeriodID(self) -> int:

        return self._cb_period.currentIndex()

    def getSelectedYear(self) -> int:

        if self._cb_years is not None:

            return int(self._cb_years.currentText())

    def collectionSelectionChanged(self):

        coll_index = self._cb_collection.currentIndex()
        cb_item_cnt = self._cb_period.count()

        if coll_index == 0:  # FireCII collection

            if cb_item_cnt == 1:
                self._cb_period.addItem('Months')
                self._cb_period.setCurrentIndex(0)

        elif coll_index == 1:  # MTBS collection

            if cb_item_cnt == 2:

                self._cb_period.removeItem(1)

                if self._hbox_period.count() > 1:

                    self._hbox_period.itemAt(1).widget().deleteLater()
                    self._hbox_period.itemAt(2).widget().deleteLater()
                    self._cb_years = None

    def periodSelectionChanged(self):

        coll_index = self._cb_collection.currentIndex()

        if coll_index == 0:

            if self._cb_period.currentIndex() == 1:

                text_label = QLabel('of')
                self._hbox_period.addWidget(text_label, 1)

                # add year ranges
                self._cb_years = QComboBox()

                YEAR_BEGIN = FireCIIAvailability.BEGIN.value
                YEAR_END = FireCIIAvailability.END.value
                self._cb_years.addItems(['{}'.format(y) for y in range(YEAR_BEGIN, YEAR_END + 1)])

                self._hbox_period.addWidget(self._cb_years, 2)

            elif self._cb_period.currentIndex() == 0:

                self._hbox_period.itemAt(1).widget().deleteLater()
                self._hbox_period.itemAt(2).widget().deleteLater()
                self._cb_years = None
