import gc
import os

from enum import Enum
from typing import Union

from mlfire.data.view import DatasetView, FireLabelsViewOpt, SatImgViewOpt
from mlfire.earthengine.collections import ModisIndex
from mlfire.earthengine.collections import FireLabelsCollection
from mlfire.earthengine.collections import MTBSSeverity, MTBSRegion
from mlfire.features.pca import FactorOP

# utils imports
from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')


class DatasetSplitOpt(Enum):

    SHUFFLE_SPLIT = 0
    IMG_HORIZONTAL_SPLIT = 1
    IMG_VERTICAL_SPLIT = 2


class DatasetTransformOP(Enum):

    NONE = 0
    SCALE = 1
    STANDARTIZE_ZSCORE = 2
    PCA = 4
    SAVITZKY_GOLAY = 8


class DataAdapterTS(DatasetView):

    def __init__(self,
                 lst_satimgs: Union[tuple[str], list[str]],
                 lst_labels: Union[tuple[str], list[str]],
                 ds_start_date: lazy_import('datetime').date = None,
                 ds_end_date: lazy_import('datetime').date = None,
                 # transformer options
                 ds_split_opt: DatasetSplitOpt = DatasetSplitOpt.SHUFFLE_SPLIT,
                 test_ratio: float = 0.33,
                 val_ratio: float = 0.,
                 # transformation operation
                 transform_ops: Union[tuple[DatasetTransformOP], list[DatasetTransformOP]] = (DatasetTransformOP.NONE,),
                 savgol_polyorder: int = 1,
                 savgol_winlen: int = 5,
                 pca_nfactors: int = 2,
                 pca_ops: list[FactorOP] = (FactorOP.USER_SET,),
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

        # transformation options
        self._train_test_val_opt = None
        self.train_test_val_opt = ds_split_opt

        self._lst_transform_ops = None
        self._transform_ops = DatasetTransformOP.NONE.value
        self.transform_ops = transform_ops

        self._savgol_polyorder = None
        self.savgol_polyorder = savgol_polyorder

        self._savgol_winlen = None
        self.savgol_winlen = savgol_winlen

        self._pca_nfactors = 0
        self.pca_nfactors = pca_nfactors

        self._lst_pca_ops = None
        self._pca_ops = 0
        self.pca_ops = pca_ops

        self._lst_extractors_pca = []

        # test data set options
        self._test_ratio = None
        self.test_ratio = test_ratio

        # validation data set options
        self._val_ratio = None
        self.val_ratio = val_ratio

    def _reset(self):

        super()._reset()

        if hasattr(self, '_lst_extractors_pca'):
            del self._lst_extractors_pca; self._lst_extractors_pca = []
            gc.collect()  # invoke garbage collector

    # properties
    @property
    def train_test_val_opt(self) -> DatasetSplitOpt:

        return self._train_test_val_opt

    @train_test_val_opt.setter
    def train_test_val_opt(self, flg: DatasetSplitOpt) -> None:

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

    """
    Transform options
    """

    @property
    def transform_ops(self) -> list[DatasetTransformOP]:

        return self._lst_transform_ops

    @transform_ops.setter
    def transform_ops(self, lst_ops: list[DatasetTransformOP]) -> None:

        if self._lst_transform_ops == lst_ops:
            return

        self._reset()

        self._transform_ops = 0
        self._lst_transform_ops = lst_ops
        for op in lst_ops: self._transform_ops |= op.value

    @property
    def savgol_polyorder(self) -> int:

        return self._savgol_polyorder

    @savgol_polyorder.setter
    def savgol_polyorder(self, order: int) -> None:

        if self._savgol_polyorder == order:
            return

        self._reset()
        self._savgol_polyorder = order

    @property
    def savgol_winlen(self) -> int:

        return self._savgol_winlen

    @savgol_winlen.setter
    def savgol_winlen(self, winlen: int) -> None:

        if self._savgol_winlen == winlen:
            return

        self._reset()
        self._savgol_winlen = winlen

    @property
    def pca_nfactors(self) -> int:

        return self._nfactors_pca

    @pca_nfactors.setter
    def pca_nfactors(self, n) -> None:

        if self._pca_nfactors == n:
            return

        self._reset()
        self._nfactors_pca = n

    @property
    def pca_ops(self) -> list[FactorOP]:

        return self._lst_pca_ops

    @pca_ops.setter
    def pca_ops(self, lst_ops: list[FactorOP]) -> None:

        if self._lst_pca_ops == lst_ops:
            return

        self._reset()

        self._lst_pca_ops = lst_ops
        self._pca_ops = 0
        # set ops flag
        for op in lst_ops: self._pca_ops |= op.value

    # load labels

    def __loadLabels_MTBS(self) -> _np.ndarray:

        datetime = lazy_import('datetime')

        # TODO check start and end date

        start_label_date = datetime.date(year=self.ds_start_date.year, month=1, day=1)
        start_label_index = int(self._df_dates_labels.index[self._df_dates_labels['Date'] == start_label_date][0])

        if self.ds_start_date != self.ds_end_date:

            end_label_date = datetime.date(year=self.ds_end_date.year, month=1, day=1)
            end_label_index = int(self._df_dates_labels.index[self._df_dates_labels['Date'] == end_label_date][0])
            id_bands = range(start_label_index, end_label_index + 1)

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

        # TODO check start and end date

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

        del rs_cl, rs_flags
        gc.collect()

        return labels

    def __loadLabels(self) -> _np.ndarray:

        if self.label_collection == FireLabelsCollection.MTBS:
            return self.__loadLabels_MTBS()
        elif self.label_collection == FireLabelsCollection.CCI:
            return self.__loadLabels_CCI()
        else:
            raise NotImplementedError

    def __loadSatImg_REFLECTANCE_ALL_BANDS(self) -> _np.ndarray:

        len_ds = len(self._ds_satimgs)

        if len_ds > 1:

            lst_img_ts = []
            for band_id in range(len_ds):
                lst_img_ts.append(self._ds_satimgs[band_id].ReadAsArray())
            img_ts = _np.concatenate(lst_img_ts)

            del lst_img_ts
            gc.collect()

        else:

            img_ts = self._ds_satimgs[0].ReadAsArray()

        return img_ts

    def __loadSatImg_REFLECTANCE_SELECTED_RANGE(self, start_img_id: int, end_img_id: int) -> _np.ndarray:

        NBANDS_MODIS = 7
        lst_img_ts = []

        for img_id in range(start_img_id, end_img_id + 1):

            gc.collect()

            id_ds, start_band_id = self._map_start_satimgs[img_id]
            ds_satimg = self._ds_satimgs[id_ds]

            for band_id in range(0, NBANDS_MODIS):
                lst_img_ts.append(ds_satimg.GetRasterBand(start_band_id + band_id).ReadAsArray())

        img_ts = _np.array(lst_img_ts)
        del lst_img_ts; gc.collect()

        return img_ts

    def __loadSatImg_REFLECTANCE(self) -> _np.ndarray:

        start_img_id = self._df_dates_satimgs.index[self._df_dates_satimgs['Date'] == self.ds_start_date][0]
        end_img_id = self._df_dates_satimgs.index[self._df_dates_satimgs['Date'] == self.ds_end_date][0]

        if end_img_id - start_img_id + 1 == len(self._df_dates_satimgs['Date']):

            ts_imgs = self.__loadSatImg_REFLECTANCE_ALL_BANDS()

        else:

            ts_imgs = self.__loadSatImg_REFLECTANCE_SELECTED_RANGE(start_img_id=start_img_id, end_img_id=end_img_id)

        ts_imgs = ts_imgs.astype(_np.float32)

        return ts_imgs

    def __loadSatImg_TS(self) -> _np.ndarray:

        start_date = self.ds_start_date
        if start_date not in self._df_dates_satimgs['Date'].values:
            raise AttributeError('Start date does not correspond any band!')

        end_date = self.ds_end_date
        if end_date not in self._df_dates_satimgs['Date'].values:
            raise AttributeError('End date does not correspond any band!')

        if self.modis_collection == ModisIndex.REFLECTANCE:
            return self.__loadSatImg_REFLECTANCE()
        else:
            raise NotImplementedError

    """
    Time series transformation
    """
    @staticmethod
    def __transformTimeseries_STANDARTIZE_ZSCORE(ts_imgs: _np.ndarray) -> _np.ndarray:

        NBANDS = 7  # TODO implement for additional indexes such as NDVI and LST

        for band_id in range(NBANDS):
            # get band
            img_band = ts_imgs[:, band_id::NBANDS]

            # compute standard deviation
            std_band = _np.std(img_band)
            if std_band == 0.: continue

            # standardizing series of satellite images per band
            ts_imgs[:, band_id::NBANDS] = (img_band - _np.mean(img_band)) / std_band

        return ts_imgs

    def __transformTimeseries(self, ts_imgs: _np.ndarray) -> _np.ndarray:

        NBANDS = 7  # TODO implement for additional indexes such as NDVI and LST

        # scale data using MODIS scale factor
        # see https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1
        if self._transform_ops & DatasetTransformOP.SCALE.value == DatasetTransformOP.SCALE.value:

            ts_imgs /= 1e-4  # scale factor for LST is 0.02

        # standardize data to have zero mean and unit standard deviation using z-score
        if self._transform_ops & DatasetTransformOP.STANDARTIZE_ZSCORE.value == DatasetTransformOP.STANDARTIZE_ZSCORE.value:

            ts_imgs = self.__transformTimeseries_STANDARTIZE_ZSCORE(ts_imgs=ts_imgs)

        if self._transform_ops & DatasetTransformOP.SAVITZKY_GOLAY.value == DatasetTransformOP.SAVITZKY_GOLAY.value:

            scipy_signal = lazy_import('scipy.signal')

            for band_id in range(NBANDS):
                # apply Savitzky–Golay filter, parameters are user-specified
                ts_imgs[:, band_id::NBANDS] = scipy_signal.savgol_filter(
                    ts_imgs[:, band_id::NBANDS],
                    window_length=self.savgol_winlen,
                    polyorder=self.savgol_polyorder
                )

        return ts_imgs

    def __transformTimeseries_PCA_FIT(self, ts_imgs: _np.ndarray) -> None:

        # lazy imports
        features_pca = lazy_import('mlfire.features.pca')
        copy = lazy_import('copy')

        NBANDS = 7  # TODO as a class attribute

        if DatasetTransformOP.STANDARTIZE_ZSCORE not in self._lst_transform_ops:
            ts_imgs = self.__transformTimeseries_STANDARTIZE_ZSCORE(ts_imgs=ts_imgs)

        # build up PCA projection for each band

        lst_extractors = []
        nlatent_factors_found = 0

        for band_id in range(NBANDS):

            extractor_pca = features_pca.TransformPCA(
                train_ds=ts_imgs[:, band_id::NBANDS],
                factor_ops=self._lst_pca_ops,
                nlatent_factors=self.pca_ops,
                verbose=True
            )

            extractor_pca.fit()

            nlatent_factors_found = max(nlatent_factors_found, extractor_pca.nlatent_factors)
            lst_extractors.append(extractor_pca)

        # retrain PCA feature extractors

        for band_id in range(NBANDS):

            extractor_pca = lst_extractors[band_id]

            # retrain if required
            if extractor_pca.nlatent_factors < nlatent_factors_found:
                extractor_pca.nlatent_factors_user = nlatent_factors_found

                # explicitly required number of latent factors
                if isinstance(self._lst_transform_ops, tuple):
                    mod_lst_pca_ops = list(self._lst_pca_ops)
                else:
                    mod_lst_pca_ops = copy.deepcopy(self._lst_transform_ops)
                if FactorOP.TEST_CUMSUM in mod_lst_pca_ops: mod_lst_pca_ops.remove(FactorOP.TEST_CUMSUM)
                if FactorOP.TEST_BARTLETT in mod_lst_pca_ops: mod_lst_pca_ops.remove(FactorOP.TEST_BARTLETT)
                if FactorOP.USER_SET not in mod_lst_pca_ops: mod_lst_pca_ops.append(FactorOP.USER_SET)
                extractor_pca.factor_ops = mod_lst_pca_ops

                # retrain extractor
                extractor_pca.fit()

        self._lst_extractors_pca = lst_extractors

    def __transformTimeseries_PCA(self, ts_imgs: _np.ndarray, standardize=False) -> _np.ndarray:

        if standardize:

            self.__transformTimeseries_STANDARTIZE_ZSCORE(ts_imgs=ts_imgs)

        NBANDS = 7
        NFACTORS = self._lst_extractors_pca[0].nlatent_factors

        nsamples_test = ts_imgs.shape[0]
        reduced_ts = _np.zeros(shape=(nsamples_test, NBANDS * NFACTORS), dtype=ts_imgs.dtype)

        for band_id in range(NBANDS):
            transformer_pca = self._lst_extractors_pca[band_id]
            reduced_ts[:, band_id::NBANDS] = transformer_pca.transform(ts_imgs[:, band_id::NBANDS])

        # clean up and invoke garbage collector
        del ts_imgs; gc.collect()

        return reduced_ts

    """
    Data set split into training, test and validation
    """

    def __splitDataset_SHUFFLE_SPLIT(self, ts_imgs: _np.ndarray, labels: _np.ndarray):

        # lazy import
        model_selection = lazy_import('sklearn.model_selection')

        # convert to 3D (spatial in time) multi-spectral image to time series related to pixels
        ts_imgs = ts_imgs.reshape((ts_imgs.shape[0], -1)).T
        # reshape labels to be 1D vector
        labels = labels.reshape(-1)

        if self.test_ratio > 0.:

            ts_imgs_train, ts_imgs_test, labels_train, labels_test = model_selection.train_test_split(
                ts_imgs,
                labels,
                test_size=self.test_ratio,
                random_state=42
            )

        else:

            ts_imgs_train = ts_imgs; labels_train = labels
            ts_imgs_test = None; labels_test = None

        if self.val_ratio > 0.:

            ts_imgs_train, ts_imgs_val, labels_train, labels_val = model_selection.train_test_split(
                ts_imgs_train,
                labels_train,
                test_size=self.val_ratio,
                random_state=42
            )

        else:

            ts_imgs_val = labels_val = None

        if self.test_ratio > 0. and self.val_ratio > 0.:
            return ts_imgs_train, ts_imgs_test, ts_imgs_val, labels_train, labels_test, ts_imgs_val
        elif self.test_ratio > 0.:
            return ts_imgs_train, ts_imgs_test, labels_train, labels_test
        elif self.val_ratio > 0.:
            return ts_imgs_train, ts_imgs_val, labels_train, labels_val

    def __splitDataset_HORIZONTAL_SPLIT(self, ts_imgs: _np.ndarray, labels: _np.ndarray):

        if self.test_ratio > 0.:

            _, rows, _ = ts_imgs.shape
            hi_rows = int(rows * (1. - self.test_ratio))

            ts_imgs_train = ts_imgs[:, :hi_rows, :]; labels_train = labels[:hi_rows, :]
            ts_imgs_test = ts_imgs[:, hi_rows:, :]; labels_test = labels[hi_rows:, :]

        else:

            ts_imgs_train = ts_imgs; labels_train = labels
            ts_imgs_test = None; labels_test = None

        if self.val_ratio > 0.:

            _, rows, _ = ts_imgs_train.shape
            hi_rows = int(rows * (1. - self.val_ratio))

            ts_imgs_val = ts_imgs_train[:, hi_rows:, :]; labels_val = labels_train[hi_rows:, :]
            ts_imgs_train = ts_imgs_train[:, :hi_rows, :]; labels_train = labels_train[:hi_rows, :]

        else:

            ts_imgs_val = labels_val = None

        if self.test_ratio > 0. and self.val_ratio > 0.:
            return ts_imgs_train, ts_imgs_test, ts_imgs_val, labels_train, labels_test, ts_imgs_val
        elif self.test_ratio > 0.:
            return ts_imgs_train, ts_imgs_test, labels_train, labels_test
        elif self.val_ratio > 0.:
            return ts_imgs_train, ts_imgs_val, labels_train, labels_val

    def __splitDataset_VERTICAL_SPLIT(self, ts_imgs: _np.ndarray, labels: _np.ndarray):

        if self.test_ratio > 0.:

            _, _, cols = ts_imgs.shape
            hi_cols = int(cols * (1. - self.test_ratio))

            ts_imgs_train = ts_imgs[:, :, :hi_cols]; labels_train = labels[:, :hi_cols]
            ts_imgs_test = ts_imgs[:, :, hi_cols:]; labels_test = labels[:, hi_cols:]

        else:

            ts_imgs_train = ts_imgs; labels_train = labels
            ts_imgs_test = None; labels_test = None

        if self.val_ratio > 0.:

            _, _, cols = ts_imgs_train.shape
            hi_cols = int(cols * (1. - self.val_ratio))

            ts_imgs_val = ts_imgs_train[:, :, hi_cols:]; labels_val = labels_train[:, hi_cols:]
            ts_imgs_train = ts_imgs_train[:, :, :hi_cols]; labels_train = labels_train[:, :hi_cols]

        else:

            ts_imgs_val = labels_val = None

        if self.test_ratio > 0. and self.val_ratio > 0.:
            return ts_imgs_train, ts_imgs_test, ts_imgs_val, labels_train, labels_test, ts_imgs_val
        elif self.test_ratio > 0.:
            return ts_imgs_train, ts_imgs_test, labels_train, labels_test
        elif self.val_ratio > 0.:
            return ts_imgs_train, ts_imgs_val, labels_train, labels_val
        else:
            return ts_imgs_train, labels_train

    def __splitDataset(self, ts_imgs: _np.ndarray, labels: _np.ndarray):

        if self.train_test_val_opt == DatasetSplitOpt.SHUFFLE_SPLIT:
            ds = self.__splitDataset_SHUFFLE_SPLIT(ts_imgs=ts_imgs, labels=labels)
        elif self.train_test_val_opt == DatasetSplitOpt.IMG_HORIZONTAL_SPLIT:
            ds = self.__splitDataset_HORIZONTAL_SPLIT(ts_imgs=ts_imgs, labels=labels)
        elif self.train_test_val_opt == DatasetSplitOpt.IMG_VERTICAL_SPLIT:
            ds = self.__splitDataset_VERTICAL_SPLIT(ts_imgs=ts_imgs, labels=labels)
        else:
            raise NotImplementedError

        return ds

    def createDataset(self) -> None:

        if self._ds_training:
            return

        if not self._labels_processed:
            # processing descriptions of bands related to fire labels and obtain dates from them
            try:
                self._processMetaData_LABELS()
            except IOError or ValueError:
                raise RuntimeError('Cannot process meta data related to labels!')

        if not self._satimgs_processed:
            # process descriptions of bands related to satellite images and obtain dates from them
            try:
                self._processMetaData_SATELLITE_IMG()
            except IOError or ValueError:
                raise RuntimeError('Cannot process meta data related to satellite images!')

        try:
            labels = self.__loadLabels()
        except IOError or ValueError or NotImplementedError:
            raise RuntimeError('Cannot load labels!')

        try:
            ts_imgs = self.__loadSatImg_TS()
        except IOError or ValueError or NotImplementedError:
            raise RuntimeError('Cannot load series of satellite images!')

        # TODO add NDVI and land surface temperature
        # TODO clean data set if needed
        lst_ds = list(self.__splitDataset(ts_imgs=ts_imgs, labels=labels))

        try:
            for id_ds in range(len(lst_ds) // 2):
                ts_imgs = lst_ds[id_ds]; tmp_shape = None

                if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
                    # save original shape of series
                    tmp_shape = ts_imgs.shape
                    # reshape image to time series related to pixels
                    ts_imgs = ts_imgs.reshape((ts_imgs.shape[0], -1)).T

                ts_imgs = self.__transformTimeseries(ts_imgs)

                if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
                    # reshape back to series of satellite images
                    lst_ds[id_ds] = ts_imgs.T.reshape(tmp_shape)

                gc.collect()  # invoke garbage collector
        except ValueError:
            pass

        if self._transform_ops & DatasetTransformOP.PCA.value == DatasetTransformOP.PCA.value:

            ts_imgs_train = lst_ds[0]
            tmp_shape = None

            if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
                # save original shape of series
                tmp_shape = ts_imgs_train.shape
                # reshape image to time series related to pixels
                ts_imgs_train = ts_imgs_train.reshape((ts_imgs_train.shape[0], -1)).T

            self.__transformTimeseries_PCA_FIT(ts_imgs_train)
            ts_imgs = self.__transformTimeseries_PCA(ts_imgs=ts_imgs_train)

            if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
                lst_ds[0] = ts_imgs.T.reshape(ts_imgs.shape[1], tmp_shape[1], tmp_shape[2])

            # transform test and validation data set
            for id_ds in range(1, len(lst_ds) // 2):

                # save original shape of series
                ts_imgs = lst_ds[id_ds]
                tmp_shape = None

                if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
                    tmp_shape = ts_imgs.shape
                    ts_imgs = ts_imgs.T.reshape((ts_imgs.shape[0], -1)).T

                standardize = False if DatasetTransformOP.STANDARTIZE_ZSCORE in self._lst_transform_ops else True
                ts_imgs = self.__transformTimeseries_PCA(ts_imgs=ts_imgs, standardize=standardize)

                if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
                    lst_ds[id_ds] = ts_imgs.T.reshape(ts_imgs.shape[1], tmp_shape[1], tmp_shape[2])

        if self.test_ratio > 0 and self.val_ratio > 0:

            self._ds_training = (lst_ds[0], lst_ds[3])
            self._ds_test = (lst_ds[1], lst_ds[4])
            self._ds_val = (lst_ds[2], lst_ds[5])

        else:

            self._ds_training = (lst_ds[0], lst_ds[2]) if len(lst_ds) > 2 else tuple(lst_ds)

            if self.test_ratio > 0.:
                self._ds_test = (lst_ds[1], lst_ds[3])
            elif self.val_ratio > 0.:
                self._ds_val = (lst_ds[1], lst_ds[3])

    def getTrainingDataset(self) -> _np.ndarray:

        if not self._ds_training:
            self.createDataset()

        return self._ds_training

    def getTestDataset(self) -> _np.ndarray:

        if not self._ds_test and self.test_ratio > 0.:
            self.createDataset()

        return self._ds_test

    def getValidationDataset(self) -> _np.ndarray:

        if not self._ds_val and self.val_ratio > 0.:
            self.createDataset()

        return self._ds_val


# use cases
if __name__ == '__main__':

    DATA_DIR = 'data/tifs'
    PREFIX_IMG = 'ak_reflec_january_december_{}_100km'

    LABEL_COLLECTION = FireLabelsCollection.MTBS
    # LABEL_COLLECTION = FireLabelsCollection.CCI
    STR_LABEL_COLLECTION = LABEL_COLLECTION.name.lower()

    DS_SPLIT_OPT = DatasetSplitOpt.IMG_VERTICAL_SPLIT
    TEST_RATIO = 1. / 3.  # split data set to training and test sets in ratio 2 : 1
    VAL_RATIO = 1. / 3.  # split training data set to new training and validation data sets in ratio 2 : 1

    TRANSFORM_OPS = [DatasetTransformOP.PCA, DatasetTransformOP.SAVITZKY_GOLAY]
    PCA_OPS = [FactorOP.TEST_CUMSUM]

    lst_satimgs = []
    lst_labels = []

    CCI_CONFIDENCE_LEVEL = 70

    for year in range(2004, 2006):
        PREFIX_IMG_YEAR = PREFIX_IMG.format(year)

        fn_satimg = os.path.join(DATA_DIR, '{}_epsg3338_area_0.tif'.format(PREFIX_IMG_YEAR))
        lst_satimgs.append(fn_satimg)

        fn_labels = os.path.join(DATA_DIR, '{}_epsg3338_area_0_{}_labels.tif'.format(PREFIX_IMG_YEAR, STR_LABEL_COLLECTION))
        lst_labels.append(fn_labels)

    adapter_ts = DataAdapterTS(
        lst_satimgs=lst_satimgs,
        lst_labels=lst_labels,
        label_collection=LABEL_COLLECTION,
        mtbs_severity_from=MTBSSeverity.LOW,
        cci_confidence_level=CCI_CONFIDENCE_LEVEL,
        # transformation options
        transform_ops=TRANSFORM_OPS,
        pca_ops=PCA_OPS,
        # data set split options
        ds_split_opt=DS_SPLIT_OPT,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO
    )

    # set dates
    index_begin_date = 0
    index_end_date = -1

    print(adapter_ts.satimg_dates)

    start_date = adapter_ts.satimg_dates.iloc[index_begin_date]['Date']
    adapter_ts.ds_start_date = start_date
    end_date = adapter_ts.satimg_dates.iloc[index_end_date]['Date']
    adapter_ts.ds_end_date = end_date

    adapter_ts.createDataset()
