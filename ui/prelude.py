
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtWidgets import QGroupBox, QSlider

from earthengine.ds import FireCIIAvailability, FireLabelsCollectionID, MTBSSeverity
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

        # uncertainty for wildfire detection
        self._group_fire_uncertainty = None
        self._hbox_uncertainty = None

        # uncertainty for FireCII ESA collection
        self._label_firecii_uncertainty_level = None
        self._label_firecii_uncertainty_level_value = None

        self._slider_firecii_uncertainty = None

        # uncertainty for MTBS collection
        self._label_mtbs_uncertainty_level_from = None
        self._label_mtbs_uncertainty_level_to = None

        self._cb_mtbs_severity_from = None
        self._cb_mtbs_severity_to = None

        # button box
        self._button_box = None

        # set size
        DIALOG_WIDTH = 600
        DIALOG_HEIGHT = 220
        self.setFixedSize(DIALOG_WIDTH, DIALOG_HEIGHT)

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

    def __addUncertaintyWidgets_FireCII(self) -> QHBoxLayout:

        CONFIDENCE_LEVEL = 70

        self._label_firecii_uncertainty_level = QLabel('Confidence level')
        self._label_firecii_uncertainty_level.setMargin(0)
        self._label_firecii_uncertainty_level.setFixedWidth(120)

        self._label_firecii_uncertainty_level_value = QLabel('{}%'.format(CONFIDENCE_LEVEL))
        self._label_firecii_uncertainty_level_value.setMargin(0)
        self._label_firecii_uncertainty_level_value.setMaximumWidth(40)

        # slider
        self._slider_firecii_uncertainty = QSlider(Qt.Horizontal)
        self._slider_firecii_uncertainty.setMinimum(50)
        self._slider_firecii_uncertainty.setMaximum(100)
        self._slider_firecii_uncertainty.setValue(CONFIDENCE_LEVEL)
        self._slider_firecii_uncertainty.setTickInterval(1)

        self._hbox_uncertainty.addWidget(self._label_firecii_uncertainty_level, 0)
        self._hbox_uncertainty.addWidget(self._label_firecii_uncertainty_level_value, 1)
        self._hbox_uncertainty.addWidget(self._slider_firecii_uncertainty, 2)

        # set slot
        self._slider_firecii_uncertainty.valueChanged.connect(
            self.__slotUncertaintyLevelChanged
        )

        return self._hbox_uncertainty

    def __removeUncertaintyWidgets_FireCII(self) -> None:

        for i in range(3):
            self._hbox_uncertainty.itemAt(i).widget().deleteLater()

    def __addUncertaintyWidgets_MTBS(self) -> QHBoxLayout:

        self._label_mtbs_uncertainty_level_from = QLabel('Severity from')
        self._label_mtbs_uncertainty_level_from.setMargin(0)
        self._label_mtbs_uncertainty_level_from.setMaximumWidth(80)
        self._label_mtbs_uncertainty_level_to = QLabel('to HIGH.')

        # from combo box
        self._cb_mtbs_severity_from = QComboBox()
        for severity in MTBSSeverity:
            self._cb_mtbs_severity_from.addItem(severity.name)

        # add widgets to layout
        self._hbox_uncertainty.addWidget(self._label_mtbs_uncertainty_level_from, 0)
        self._hbox_uncertainty.addWidget(self._cb_mtbs_severity_from, 1)
        self._hbox_uncertainty.addWidget(self._label_mtbs_uncertainty_level_to, 2)

        return self._hbox_uncertainty

    def __removeUncertaintyWidgets_MTBS(self) -> None:

        for i in range(3):
            self._hbox_uncertainty.itemAt(i).widget().deleteLater()

    def __initUI_GroupFireLevelUncertainty(self) -> QGroupBox:

        self._group_fire_uncertainty = QGroupBox('Fire Uncertainty')
        self._hbox_uncertainty = QHBoxLayout()

        # FireCII fire uncertainty layout
        hbox_fire_uncertainty = self.__addUncertaintyWidgets_FireCII()

        # set group box layout
        self._group_fire_uncertainty.setLayout(hbox_fire_uncertainty)

        return self._group_fire_uncertainty

    def __initUI_GroupFireLevelUncertainty_MTBS(self) -> QGroupBox:

        pass

    def __initUI(self) -> None:

        hbox = QHBoxLayout()

        group_collection = self.__initUI_GroupCollection()
        group_time_period = self.__initUI_GroupTimePeriod()
        group_fire_uncertainty = self.__initUI_GroupFireLevelUncertainty()

        # set up layout (horizontal box)
        hbox.addWidget(group_collection)
        hbox.addWidget(group_time_period)

        # cancel and ok buttons
        self._button_box = QDialogButtonBox(QDialogButtonBox.Ok)

        # dialog layout
        layout = QFormLayout()
        layout.addRow(hbox)
        layout.addRow(group_fire_uncertainty)
        layout.addRow(self._button_box)

        self.setLayout(layout)
        self.setWindowTitle('Select data set')

        # set listeners
        self._cb_collection.currentIndexChanged.connect(
            self.__slotCollectionSelectionChanged
        )
        self._cb_time_period.currentIndexChanged.connect(
            self.__slotPeriodSelectionChanged
        )

        self._button_box.accepted.connect(
            self.accept
        )

    def __addYearSelectionComboBox(self) -> None:

        text_label = QLabel('of')
        text_label.setFixedWidth(20)
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

    # events

    def closeEvent(self, evnt):

        super(UIPrelude, self).closeEvent(evnt)

    # slots

    def __slotUncertaintyLevelChanged(self):

        value_level = self._slider_firecii_uncertainty.value()
        self._label_firecii_uncertainty_level_value.setText('{}%'.format(value_level))

    def __slotCollectionSelectionChanged(self):

        coll_index = self._cb_collection.currentIndex()
        cb_item_cnt = self._cb_time_period.count()  # this is just for sure to avoid wrong states of the application

        if coll_index == FireLabelsCollectionID.ESA_FIRE_CII.value:  # FireCII collection (ESA)

            if cb_item_cnt == 1:
                self._cb_time_period.addItem('Months')
                self._cb_time_period.setCurrentIndex(0)

            # change widgets (FireCII -> MTBS)
            if self._hbox_uncertainty.count() > 1:
                self.__removeUncertaintyWidgets_MTBS()
            self.__addUncertaintyWidgets_FireCII()

        elif coll_index == FireLabelsCollectionID.MTBS.value:  # MTBS collection

            if cb_item_cnt == 2:

                self._cb_time_period.removeItem(1)

                if self._hbox_time_period.count() > 1:
                    self.__removeYearSelectionComboBox()

            # change widgets (FireCII -> MTBS)
            if self._hbox_uncertainty.count() > 1:
                self.__removeUncertaintyWidgets_FireCII()
            self.__addUncertaintyWidgets_MTBS()
        else:
            raise RuntimeError('Something gots wrong!')

    def __slotPeriodSelectionChanged(self):

        coll_index = self._cb_collection.currentIndex()

        if coll_index == 0:

            if self._cb_time_period.currentIndex() == TimePeriod.MONTHS.value:

                self.__addYearSelectionComboBox()

            elif self._cb_time_period.currentIndex() == TimePeriod.YEARS.value:

                self.__removeYearSelectionComboBox()
