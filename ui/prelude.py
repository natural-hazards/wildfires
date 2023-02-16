
from PyQt5.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel
from PyQt5.QtWidgets import QGroupBox

from earthengine.ds import FireCIIAvailability, FireLabelsCollectionID
from utils.time import TimePeriod


class UIPrelude(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)

        self._cb_years = None

        # group collection
        self._group_collection = None  # collection group
        self._hbox_collection = None  # layout
        self._cb_collection = None  # combo box for collection selection

        # group time period
        self._group_time_period = None  # time period group
        self._hbox_time_period = None  # layout
        self._cb_years = None  # combo box for year selection

        # button box
        self._button_box = None

        self.__initUI()

    def __initUI_GroupCollection(self) -> QGroupBox:

        self._group_collection = QGroupBox('Collection')
        self._hbox_collection = QHBoxLayout()

        self._cb_collection = QComboBox()
        self._cb_collection.addItem('FireCCI v5.1')
        self._cb_collection.addItem('MTBS')
        self._cb_collection.setCurrentIndex(0)

        self._hbox_collection.addWidget(self._cb_collection)
        self._group_collection.setLayout(self._hbox_collection)

        return self._group_collection

    def __initUI_GroupTimePeriod(self) -> QGroupBox:

        self._group_time_period = QGroupBox('Time Period')
        self._hbox_time_period = QHBoxLayout()

        self._cb_time_period = QComboBox()
        self._cb_time_period.addItem('Years')
        self._cb_time_period.addItem('Months')
        self._cb_time_period.setCurrentIndex(0)

        self._hbox_time_period.addWidget(self._cb_time_period, 0)
        self._group_time_period.setLayout(self._hbox_time_period)

        return self._group_time_period

    def __initUI(self) -> None:

        hbox = QHBoxLayout()

        group_collection = self.__initUI_GroupCollection()
        group_time_period = self.__initUI_GroupTimePeriod()

        # set up layout
        hbox.addWidget(group_collection)
        hbox.addWidget(group_time_period)

        # cancel and ok buttons
        self._button_box = QDialogButtonBox(QDialogButtonBox.Ok)

        # dialog layout
        layout = QFormLayout()
        layout.addRow(hbox)
        layout.addRow(self._button_box)

        self.setLayout(layout)
        self.setWindowTitle('Select data set')

        # set listeners
        self._cb_collection.currentIndexChanged.connect(
            self.collectionSelectionChanged
        )
        self._cb_time_period.currentIndexChanged.connect(
            self.periodSelectionChanged
        )

        self._button_box.accepted.connect(
            self.accept
        )

    def __addYearSelectionComboBox(self) -> None:

        text_label = QLabel('of')
        self._hbox_time_period.addWidget(text_label, 1)

        # add year ranges
        self._cb_years = QComboBox()

        YEAR_BEGIN = FireCIIAvailability.BEGIN.value
        YEAR_END = FireCIIAvailability.END.value
        self._cb_years.addItems(['{}'.format(y) for y in range(YEAR_BEGIN, YEAR_END + 1)])

        self._hbox_time_period.addWidget(self._cb_years, 2)

    def __removeYearSelectionComboBox(self) -> None:

        self._hbox_time_period.itemAt(1).widget().deleteLater()
        self._hbox_time_period.itemAt(2).widget().deleteLater()
        self._cb_years = None

    def getSelectedCollectionID(self) -> int:

        return self._cb_collection.currentIndex()

    def getSelectedPeriodID(self) -> TimePeriod:

        return TimePeriod(self._cb_time_period.currentIndex())

    def getSelectedYear(self) -> int:

        if self._cb_years is not None:

            return int(self._cb_years.currentText())

    def collectionSelectionChanged(self):

        coll_index = self._cb_collection.currentIndex()
        cb_item_cnt = self._cb_time_period.count()  # this is just for sure to avoid wrong states of the application

        if coll_index == FireLabelsCollectionID.ESA_FIRE_CII.value:  # FireCII collection (ESA)

            if cb_item_cnt == 1:
                self._cb_time_period.addItem('Months')
                self._cb_time_period.setCurrentIndex(0)

        elif coll_index == FireLabelsCollectionID.MTBS.value:  # MTBS collection

            if cb_item_cnt == 2:

                self._cb_time_period.removeItem(1)

                if self._hbox_time_period.count() > 1:

                    self.__removeYearSelectionComboBox()

    def periodSelectionChanged(self):

        coll_index = self._cb_collection.currentIndex()

        if coll_index == 0:

            if self._cb_time_period.currentIndex() == TimePeriod.MONTHS.value:

                self.__addYearSelectionComboBox()

            elif self._cb_time_period.currentIndex() == TimePeriod.YEARS.value:

                self.__removeYearSelectionComboBox()
