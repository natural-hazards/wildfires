import gc
import os

from enum import Enum
from typing import Union

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


class TransformerTS(DatasetView):

    def __init__(self,
                 lst_satimgs: Union[tuple[str], list[str]],
                 lst_labels: Union[tuple[str], list[str]],
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

        self._ds_training = None
        self._ds_test = None
        self._ds_validation = None

        # transformer options
        self._train_test_val_opt = None
        self.train_test_val_opt = train_test_val_opt

        # test data set options
        self._test_ratio = None
        self.test_ratio = test_ratio

        # validation data set options
        self._val_ratio = None
        self.validation_ratio = val_ratio

    def _reset(self):

        super()._reset()

        # data sets
        del self._ds_training; self._ds_training = None
        del self._ds_test; self._ds_test = None
        del self._ds_validation; self._ds_validation = None

        gc.collect()  # invoke garbage collector

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
    def validation_ratio(self) -> float:

        return self._val_ratio

    @validation_ratio.setter
    def validation_ratio(self, val: float) -> None:

        if self._val_ratio == val:
            return

        self._reset()
        self._val_ratio = val

    # time series transformation
    def __transformTimeseries(self) -> None:

        pass

    def __createDataset(self) -> None:

        pass

    def getTrainingDataset(self):

        pass

    def getTestDataset(self):

        pass

    def getValidationDataset(self):

        pass


# use cases
if __name__ == '__main__':

    pass
