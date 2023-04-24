import gc
import os

from enum import Enum
from typing import Union

import numpy as np

from mlfire.data.view import DatasetView, FireLabelsViewOpt, SatImgViewOpt
from mlfire.earthengine.collections import ModisIndex
from mlfire.earthengine.collections import FireLabelsCollection
from mlfire.earthengine.collections import MTBSSeverity, MTBSRegion

# utils imports
from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')


class TrainTestValSplitOpt(Enum):

    SHUFFLE_SPLIT = 0
    IMG_HORIZONTAL_SPLIT = 1
    IMG_VERTICAL_SPLIT = 2


class DataAdapterTS(DatasetView):

    def __init__(self,
                 lst_satimgs: Union[tuple[str], list[str]],
                 lst_labels: Union[tuple[str], list[str]],
                 ds_start_date: lazy_import('datetime').date = None,
                 ds_end_date: lazy_import('datetime').date = None,
                 # transformer options
                 train_test_val_opt: TrainTestValSplitOpt = TrainTestValSplitOpt.SHUFFLE_SPLIT,
                 test_ratio: float = 0.33,
                 val_ratio: float = None,
                 # view options
                 modis_collection: ModisIndex = ModisIndex.REFLECTANCE,
                 label_collection: FireLabelsCollection = FireLabelsCollection.MTBS,
                 cci_confidence_level: int = 70,
                 mtbs_severity_from: MTBSSeverity = MTBSSeverity.LOW,
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA,
                 ndvi_view_threshold: float = None,
                 satimg_view_opt: SatImgViewOpt = SatImgViewOpt.NATURAL_COLOR,
                 labels_view_opt: FireLabelsViewOpt = FireLabelsViewOpt.LABEL) -> None:

        super().__init__(
            lst_satimgs=lst_satimgs,
            lst_labels=lst_labels,
            modis_collection=modis_collection,
            label_collection=label_collection,
            cci_confidence_level=cci_confidence_level,
            mtbs_severity_from=mtbs_severity_from,
            mtbs_region=mtbs_region,
            ndvi_view_threshold=ndvi_view_threshold,
            satimg_view_opt=satimg_view_opt,
            labels_view_opt=labels_view_opt
        )

        self._ds_start_date = None
        self.ds_start_date = ds_start_date

        self._ds_end_date = None
        self.ds_end_date = ds_end_date

        # transformer options
        self._train_test_val_opt = None
        self.train_test_val_opt = train_test_val_opt

        # test data set options
        self._test_ratio = None
        self.test_ratio = test_ratio

        # validation data set options
        self._val_ratio = None
        self.val_ratio = val_ratio

    def _reset(self):

        super()._reset()

    # properties
    @property
    def train_test_val_opt(self) -> TrainTestValSplitOpt:

        return self._train_test_val_opt

    @train_test_val_opt.setter
    def train_test_val_opt(self, flg: TrainTestValSplitOpt) -> None:

        if flg == self._train_test_val_opt:
            return

        self._reset()
        self._train_test_val_opt = flg

    @property
    def test_ratio(self) -> float:

        return self._test_ratio

    @test_ratio.setter
    def test_ratio(self, val: float) -> None:

        if self._test_ratio == val:
            return

        self._reset()
        self._test_ratio = val

    @property
    def val_ratio(self) -> float:

        return self._val_ratio

    @val_ratio.setter
    def val_ratio(self, val: float) -> None:

        if self._val_ratio == val:
            return

        self._reset()
        self._val_ratio = val

    # TODO start/end index

    @property
    def ds_start_date(self) -> lazy_import('datetime').date:

        return self._ds_start_date

    @ds_start_date.setter
    def ds_start_date(self, val_date: lazy_import('datetime').date) -> None:

        if self._ds_start_date == val_date:
            return

        # clean up
        del self._ds_training; self._ds_training = None
        del self._ds_test; self._ds_test = None
        gc.collect()

        self._ds_start_date = val_date

    @property
    def ds_end_date(self) -> lazy_import('datetime').date:

        return self._ds_end_date

    @ds_end_date.setter
    def ds_end_date(self, val_date: lazy_import('datetime').date) -> None:

        if self._ds_end_date == val_date:
            return

        # clean up
        del self._ds_training; self._ds_training = None
        del self._ds_test; self._ds_test = None
        gc.collect()

        self._ds_end_date = val_date

    # load labels

    def __loadLabels_MTBS(self) -> _np.ndarray:

        datetime = lazy_import('datetime')

        start_label_date = datetime.date(year=self.ds_start_date.year, month=1, day=1)
        start_label_index = int(self._df_dates_labels.index[self._df_dates_labels['Date'] == start_label_date][0])

        if self.ds_start_date != self.ds_end_date:
            end_label_date = datetime.date(year=self.ds_end_date.year, month=1, day=1)
            end_label_index = int(self._df_dates_labels.index[self._df_dates_labels['Date'] == end_label_date][0])
            id_bands = (start_label_index, end_label_index)
        else:
            id_bands = start_label_index

        # get fire severity
        rs_severity = self._readFireSeverity_MTBS(id_bands=id_bands)

        # convert severity to labels
        c1 = rs_severity >= self.mtbs_severity_from.value; c2 = rs_severity <= MTBSSeverity.HIGH.value
        label = _np.logical_and(c1, c2).astype(_np.float32)
        # set not observed pixels
        label[rs_severity == MTBSSeverity.NON_MAPPED_AREA.value] = _np.nan

        return label

    def __loadLabels_CCI(self) -> _np.ndarray:

        datetime = lazy_import('datetime')

        start_label_date = datetime.date(year=self.ds_start_date.year, month=self.ds_start_date.month, day=1)
        start_label_index = int(self._df_dates_labels.index[self._df_dates_labels['Date'] == start_label_date][0])

        if self.ds_start_date != self.ds_end_date:
            end_label_date = datetime.date(year=self.ds_end_date.year, month=self.ds_end_date.month, day=1)
            end_label_index = int(self._df_dates_labels.index[self._df_dates_labels['Date'] == end_label_date][0])
            id_bands = (start_label_index, end_label_index)
        else:
            id_bands = start_label_index

        # get fire confidence level
        rs_cl, rs_flags = self._readFireConfidenceLevel_CCI(id_bands=id_bands)

        # convert severity to labels
        labels = (rs_cl >= self.cci_confidence_level).astype(_np.float32)
        # set not observed pixels
        PIXEL_NOT_OBSERVED = -1
        labels[rs_flags == PIXEL_NOT_OBSERVED] = _np.nan

        return labels

    def __loadLabels(self) -> _np.ndarray:

        if self.label_collection == FireLabelsCollection.MTBS:
            return self.__loadLabels_MTBS()
        elif self.label_collection == FireLabelsCollection.CCI:
            return self.__loadLabels_CCI()
        else:
            raise NotImplementedError

    def __loadSatImg_TS(self) -> _np.ndarray:

        pass

    # time series transformation
    def __transformTimeseries(self) -> None:

        pass

    def createDataset(self) -> None:

        if self._ds_training:
            return

        if not self._labels_processed:
            # processing descriptions of bands related to fire labels and obtain dates from them
            try:
                self._processMetaData_LABELS()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to labels!')

        if not self._satimgs_processed:
            # process descriptions of bands related to satellite images and obtain dates from them
            try:
                self._processMetaData_SATELLITE_IMG()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to satellite images!')

        try:
            labels = self.__loadLabels()
        except IOError or ValueError:
            pass

        try:
            ts_satimgs = self.__loadSatImg_TS()
        except IOError or ValueError:
            pass

        # TODO split data set to training and test (if required)

        try:
            self.__transformTimeseries()
        except ValueError:
            pass

    def getTrainingDataset(self):

        pass

    def getTestDataset(self):

        pass

    def getValidationDataset(self):

        pass


# use cases
if __name__ == '__main__':

    DATA_DIR = 'data/tifs'
    PREFIX_IMG = 'ak_reflec_january_december_{}_100km'

    LABEL_COLLECTION = FireLabelsCollection.MTBS
    # LABEL_COLLECTION = FireLabelsCollection.CCI
    STR_LABEL_COLLECTION = LABEL_COLLECTION.name.lower()

    lst_satimgs = []
    lst_labels = []

    CCI_CONFIDENCE_LEVEL = 70

    for year in range(2004, 2006):
        PREFIX_IMG_YEAR = PREFIX_IMG.format(year)

        fn_satimg = os.path.join(DATA_DIR, '{}_epsg3338_area_0.tif'.format(PREFIX_IMG_YEAR))
        lst_satimgs.append(fn_satimg)

        fn_labels = os.path.join(DATA_DIR, '{}_epsg3338_area_0_{}_labels.tif'.format(PREFIX_IMG_YEAR, STR_LABEL_COLLECTION))
        lst_labels.append(fn_labels)

    transformer_ts = DataAdapterTS(
        lst_satimgs=lst_satimgs,
        lst_labels=lst_labels,
        label_collection=LABEL_COLLECTION,
        mtbs_severity_from=MTBSSeverity.LOW,
        cci_confidence_level=CCI_CONFIDENCE_LEVEL,
    )

    # set dates
    index_begin_date = 0
    index_end_date = -1

    start_date = transformer_ts.satimg_dates.iloc[index_begin_date]['Date']
    transformer_ts.ds_start_date = start_date
    end_date = transformer_ts.satimg_dates.iloc[index_begin_date]['Date']
    transformer_ts.ds_end_date = end_date

    transformer_ts.createDataset()
