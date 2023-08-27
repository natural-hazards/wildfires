# TODO rename like pix_ts.py (not use spatial information)

import gc
import os

from enum import Enum
from typing import Union

import numpy as np  # TODO remove

from mlfire.data.view import DatasetView, FireLabelsViewOpt, SatImgViewOpt
from mlfire.earthengine.collections import ModisCollection
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
    STANDARTIZE_ZSCORE = 1
    PCA = 2
    PCA_PER_BAND = 4
    SAVITZKY_GOLAY = 8
    NOT_PROCESS_UNCHARTED_PIXELS = 16


class VegetationIndex(Enum):

    NONE = 0
    EVI = 2
    EVI2 = 4
    NDVI = 8


class DataAdapterTS(DatasetView):

    def __init__(self,
                 # sources - labels and satellite data (reflectance and temperature)
                 lst_labels: Union[tuple[str], list[str]],
                 lst_satdata_reflectance: Union[tuple[str], list[str], None] = None,
                 lst_satdata_temperature: Union[tuple[str], list[str], None] = None,
                 # TODO comment
                 ds_start_date: lazy_import('datetime').date = None,
                 ds_end_date: lazy_import('datetime').date = None,
                 # transformer options
                 ds_split_opt: DatasetSplitOpt = DatasetSplitOpt.SHUFFLE_SPLIT,
                 test_ratio: float = .33,
                 val_ratio: float = .0,
                 # add vegetation index
                 vegetation_index: Union[tuple[VegetationIndex], list[VegetationIndex]] = (VegetationIndex.NONE,),
                 # transformation operation
                 transform_ops: Union[tuple[DatasetTransformOP], list[DatasetTransformOP]] = (DatasetTransformOP.NONE,),
                 # TODO comment
                 savgol_polyorder: int = 1,
                 savgol_winlen: int = 5,
                 # TODO comment
                 pca_nfactors: int = 2,
                 pca_ops: list[FactorOP] = (FactorOP.USER_SET,),
                 pca_retained_variance: float = 0.95,
                 # view options
                 satdata_select_opt: ModisCollection = ModisCollection.REFLECTANCE,
                 label_collection: FireLabelsCollection = FireLabelsCollection.MTBS,
                 cci_confidence_level: int = 70,
                 mtbs_severity_from: MTBSSeverity = MTBSSeverity.LOW,
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA,
                 ndvi_view_threshold: float = None,
                 satimg_view_opt: SatImgViewOpt = SatImgViewOpt.NATURAL_COLOR,
                 labels_view_opt: FireLabelsViewOpt = FireLabelsViewOpt.LABEL) -> None:

        # TODO inheritance from DatasetBase?
        super().__init__(
            lst_labels=lst_labels,
            lst_satdata_reflectance=lst_satdata_reflectance,
            lst_satdata_temperature=lst_satdata_temperature,
            satdata_select_opt=satdata_select_opt,
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

        # vegetation index

        self._lst_vegetation_index = None
        self._vi_ops = VegetationIndex.NONE.value
        self.vegetation_index = vegetation_index

        # Savitzky-Golay filter properties

        self._savgol_polyorder = None
        self.savgol_polyorder = savgol_polyorder

        self._savgol_winlen = None
        self.savgol_winlen = savgol_winlen

        # principal component properties

        self._pca_nfactors = 0
        self.pca_nfactors = pca_nfactors

        self._lst_pca_ops = None
        self._pca_ops = 0
        self.pca_ops = pca_ops

        self._pca_retained_variance = None
        self.pca_retained_variance = pca_retained_variance

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
    Vegetation index options
    """

    @property
    def vegetation_index(self) -> Union[list[VegetationIndex], tuple[VegetationIndex]]:

        return self._lst_vegetation_index

    @vegetation_index.setter
    def vegetation_index(self, lst_vi: Union[list[VegetationIndex], tuple[VegetationIndex]]) -> None:

        if self.vegetation_index == lst_vi:
            return

        self._reset()

        self._vi_ops = 0
        self._lst_vegetation_index = lst_vi
        for op in lst_vi: self._vi_ops |= op.value

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

        return self._pca_nfactors

    @pca_nfactors.setter
    def pca_nfactors(self, n) -> None:

        if self._pca_nfactors == n:
            return

        self._reset()
        self._pca_nfactors = n

    @property
    def pca_retained_variance(self) -> float:

        return self._pca_retained_variance

    @pca_retained_variance.setter
    def pca_retained_variance(self, val: float) -> None:

        if val == self._pca_retained_variance:
            return

        self._reset()
        self._pca_retained_variance = val

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
    # TODO move to loader?

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
            id_bands = range(start_label_index, end_label_index + 1)

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

    # load reflectance
    # TODO move to loader

    def __loadSatImg_REFLECTANCE_ALL_BANDS(self) -> _np.ndarray:

        len_ds = len(self._ds_satdata_reflectance)

        if len_ds > 1:

            rows = self._ds_satdata_reflectance[0].RasterYSize; cols = self._ds_satdata_reflectance[0].RasterXSize
            nrasters = self._ds_satdata_reflectance[0].RasterCount

            for id_img in range(1, len_ds):

                tmp_img = self._ds_satdata_reflectance[id_img]

                if rows != tmp_img.RasterYSize or cols != tmp_img.RasterXSize:
                    raise RuntimeError('Inconsistent shape among sources!')

                nrasters += tmp_img.RasterCount

            # allocate an empty array
            satimg_ts = _np.empty(shape=(rows, cols, nrasters), dtype=_np.float32)
            rstart = rend = 0

            for id_img in range(len_ds):

                gc.collect()  # invoke garbage collector

                rend += self._ds_satdata_reflectance[id_img].RasterCount

                tmp_img = self._ds_satdata_reflectance[id_img].ReadAsArray()
                satimg_ts[:, :, rstart:rend] = _np.moveaxis(tmp_img, 0, -1)

                rstart = rend

        else:

            satimg_ts = self._ds_satdata_reflectance[0].ReadAsArray()
            satimg_ts = _np.moveaxis(satimg_ts, 0, -1)
            satimg_ts = satimg_ts.astype(_np.float32)

        return satimg_ts

    def __loadSatImg_REFLECTANCE_SELECTED_RANGE(self, start_id_img: int, end_id_img: int) -> _np.ndarray:

        NBANDS_MODIS = 7
        rgn = range(start_id_img, end_id_img + 1)

        rows = self._ds_satdata_reflectance[0].RasterYSize; cols = self._ds_satdata_reflectance[0].RasterXSize
        nbands = NBANDS_MODIS * len(rgn)

        satimg_ts = _np.empty(shape=(rows, cols, nbands), dtype=_np.float32)
        band_pos = 0

        for id_img in rgn:

            id_img, start_id_band = self._map_start_satimgs[id_img]
            satimg = self._ds_satdata_reflectance[id_img]

            for id_band in range(0, NBANDS_MODIS):

                satimg_ts[:, :, band_pos] = satimg.GetRasterBand(start_id_band + id_band).ReadAsArray()
                band_pos += 1

        # invoke garbage collector
        gc.collect()

        return satimg_ts

    def __loadSatImg_REFLECTANCE(self) -> _np.ndarray:

        start_img_id = self._df_dates_reflectance.index[self._df_dates_reflectance['Date'] == self.ds_start_date][0]
        end_img_id = self._df_dates_reflectance.index[self._df_dates_reflectance['Date'] == self.ds_end_date][0]

        if end_img_id - start_img_id + 1 == len(self._df_dates_reflectance['Date']):
            satimg_ts = self.__loadSatImg_REFLECTANCE_ALL_BANDS()
        else:
            satimg_ts = self.__loadSatImg_REFLECTANCE_SELECTED_RANGE(start_id_img=start_img_id, end_id_img=end_img_id)

        # scale pixel values using MODIS scale factor (0.0001)
        # see https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1
        satimg_ts /= 1e4

        # TODO #bands related to MODIS as constant
        self._nfeatures_ts = 7

        return satimg_ts

    def __loadSatImg_TS(self) -> _np.ndarray:

        start_date = self.ds_start_date
        if start_date not in self._df_dates_reflectance['Date'].values: raise AttributeError('Start date does not correspond any band!')

        end_date = self.ds_end_date
        if end_date not in self._df_dates_reflectance['Date'].values: raise AttributeError('End date does not correspond any band!')

        if self.modis_collection == ModisCollection.REFLECTANCE:
            return self.__loadSatImg_REFLECTANCE()
        else:
            raise NotImplementedError

    """
    Preprocessing 
    """

    def __transformTimeseries_STANDARTIZE_ZSCORE(self, ts_imgs: _np.ndarray) -> _np.ndarray:

        stats = lazy_import('scipy.stats')

        NFEATURES_TS = self._nfeatures_ts  # TODO implement for additional indexes such as NDVI and LST

        for band_id in range(NFEATURES_TS):

            img_band = ts_imgs[:, band_id::NFEATURES_TS]

            # check if standard deviation is greater than 0
            std_band = _np.std(img_band)
            if std_band == 0.: continue

            ts_imgs[:, band_id::NFEATURES_TS] = stats.zscore(img_band, axis=1)

        return ts_imgs

    def __transformTimeseries(self, ts_imgs: _np.ndarray) -> _np.ndarray:

        utils_time = lazy_import('mlfire.utils.time')

        NFEATURES_TS = self._nfeatures_ts  # implement #features ts as property

        # standardize data to have zero mean and unit standard deviation using z-score
        if self._transform_ops & DatasetTransformOP.STANDARTIZE_ZSCORE.value == DatasetTransformOP.STANDARTIZE_ZSCORE.value:

            with utils_time.elapsed_timer('Standardize time series pixels using z-score'):

                ts_imgs = self.__transformTimeseries_STANDARTIZE_ZSCORE(ts_imgs=ts_imgs)

        if self._transform_ops & DatasetTransformOP.SAVITZKY_GOLAY.value == DatasetTransformOP.SAVITZKY_GOLAY.value:

            scipy_signal = lazy_import('scipy.signal')

            for band_id in range(NFEATURES_TS):
                # apply Savitzky–Golay filter, parameters are user-specified
                ts_imgs[:, band_id::NFEATURES_TS] = scipy_signal.savgol_filter(
                    ts_imgs[:, band_id::NFEATURES_TS],
                    window_length=self.savgol_winlen,
                    polyorder=self.savgol_polyorder
                )

        return ts_imgs

    """
    Preprocessing (principal component analysis) 
    """

    def __transformTimeseries_PCA_FIT_PER_BAND(self, ts_imgs: _np.ndarray) -> None:

        # lazy imports
        features_pca = lazy_import('mlfire.features.pca')
        copy = lazy_import('copy')

        NFEATURES_TS = self._nfeatures_ts  # TODO as a property

        # build up PCA projection for each band

        lst_extractors = []
        nlatent_factors_found = 0

        for band_id in range(NFEATURES_TS):

            extractor_pca = features_pca.TransformPCA(
                train_ds=ts_imgs[:, band_id::NFEATURES_TS],
                factor_ops=self._lst_pca_ops,
                nlatent_factors=self.pca_nfactors,
                retained_variance=self.pca_retained_variance,
                verbose=True
            )

            extractor_pca.fit()

            nlatent_factors_found = max(nlatent_factors_found, extractor_pca.nlatent_factors)
            lst_extractors.append(extractor_pca)

        # refit PCA feature extractors

        for band_id in range(NFEATURES_TS):

            extractor_pca = lst_extractors[band_id]

            # retrain if required
            if extractor_pca.nlatent_factors < nlatent_factors_found:
                extractor_pca.nlatent_factors_user = nlatent_factors_found

                # explicitly required number of latent factors
                if isinstance(self._lst_transform_ops, tuple):
                    mod_lst_pca_ops = list(self._lst_pca_ops)
                else:
                    mod_lst_pca_ops = copy.deepcopy(self._lst_transform_ops)
                if FactorOP.CUMULATIVE_EXPLAINED_VARIANCE in mod_lst_pca_ops: mod_lst_pca_ops.remove(FactorOP.CUMULATIVE_EXPLAINED_VARIANCE)
                if FactorOP.USER_SET not in mod_lst_pca_ops: mod_lst_pca_ops.append(FactorOP.USER_SET)
                extractor_pca.factor_ops = mod_lst_pca_ops

                # retrain extractor
                extractor_pca.fit()

        self._lst_extractors_pca = lst_extractors
        self._pca_nfactors = nlatent_factors_found

    def __transformTimeseries_PCA_FIT_ALL_BANDS(self, ts_imgs: _np.ndarray) -> None:

        # lazy imports
        features_pca = lazy_import('mlfire.features.pca')

        extractor_pca = features_pca.TransformPCA(
            train_ds=ts_imgs,
            factor_ops=self._lst_pca_ops,
            nlatent_factors=self.pca_nfactors,
            retained_variance=self.pca_retained_variance,
            verbose=True
        )

        extractor_pca.fit()
        self._lst_extractors_pca = [extractor_pca]
        self._pca_nfactors = extractor_pca.nlatent_factors

    def __transformTimeseries_PCA_FIT(self, ts_imgs: _np.ndarray) -> _np.ndarray:

        utils_time = lazy_import('mlfire.utils.time')

        if DatasetTransformOP.STANDARTIZE_ZSCORE not in self._lst_transform_ops:
            ts_imgs = self.__transformTimeseries_STANDARTIZE_ZSCORE(ts_imgs=ts_imgs)

        with utils_time.elapsed_timer('Transforming data using PCA'):

            if self._transform_ops & DatasetTransformOP.PCA.value == DatasetTransformOP.PCA.value:
                return self.__transformTimeseries_PCA_FIT_ALL_BANDS(ts_imgs=ts_imgs)
            elif self._transform_ops & DatasetTransformOP.PCA_PER_BAND.value == DatasetTransformOP.PCA_PER_BAND.value:
                return self.__transformTimeseries_PCA_FIT_PER_BAND(ts_imgs=ts_imgs)
            else:
                raise NotImplementedError

    def __transformTimeseries_PCA_TRANSFORM_PER_BAND(self, ts_imgs: _np.ndarray) -> _np.ndarray:

        NFEATURES_TS = self._nfeatures_ts
        NFACTORS = self._lst_extractors_pca[0].nlatent_factors

        nsamples = ts_imgs.shape[0]
        reduced_ts = _np.zeros(shape=(nsamples, NFEATURES_TS * NFACTORS), dtype=ts_imgs.dtype)

        for band_id in range(NFEATURES_TS):
            transformer_pca = self._lst_extractors_pca[band_id]
            reduced_ts[:, band_id::NFEATURES_TS] = transformer_pca.transform(ts_imgs[:, band_id::NFEATURES_TS])

        # clean up and invoke garbage collector
        del ts_imgs; gc.collect()

        return reduced_ts

    def __transformTimeseries_PCA_TRANSFORM_ALL_BANDS(self, ts_imgs: _np.ndarray) -> _np.ndarray:

        transformer_pca = self._lst_extractors_pca[0]
        reduced_ts = transformer_pca.transform(ts_imgs)

        # clean up and invoke garbage collector
        del ts_imgs; gc.collect()

        return reduced_ts

    def __transformTimeseries_PCA_TRANSFORM(self, ts_imgs: _np.ndarray, standardize: bool = False) -> _np.ndarray:

        if standardize: self.__transformTimeseries_STANDARTIZE_ZSCORE(ts_imgs=ts_imgs)

        if self._transform_ops & DatasetTransformOP.PCA.value == DatasetTransformOP.PCA.value:
            return self.__transformTimeseries_PCA_TRANSFORM_ALL_BANDS(ts_imgs=ts_imgs)
        elif self._transform_ops & DatasetTransformOP.PCA_PER_BAND.value == DatasetTransformOP.PCA_PER_BAND.value:
            return self.__transformTimeseries_PCA_TRANSFORM_PER_BAND(ts_imgs=ts_imgs)
        else:
            raise NotImplementedError

    def __transformTimeseries_PCA(self, ds_imgs: list) -> list:

        FLG_UNCHARTED_PIXS = DatasetTransformOP.NOT_PROCESS_UNCHARTED_PIXELS.value
        NFEATURES_TS = self._nfeatures_ts
        nds = len(ds_imgs) // 2

        ts_imgs = ds_imgs[0]; tmp_shape = None

        if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
            tmp_shape = ts_imgs.shape
            ts_imgs = ts_imgs.reshape((-1, tmp_shape[2]))

        if self._transform_ops & FLG_UNCHARTED_PIXS == FLG_UNCHARTED_PIXS:
            labels = ds_imgs[nds]
            if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT: labels = labels.reshape(-1)

            # fit
            mask = ~_np.isnan(labels)
            self.__transformTimeseries_PCA_FIT(ts_imgs[mask, :])

            # transform
            NFEATURES_TS *= self.pca_nfactors
            tmp_imgs = np.empty(shape=(ts_imgs.shape[0], NFEATURES_TS), dtype=ts_imgs.dtype); tmp_imgs[:, :] = _np.nan

            tmp_imgs[mask, :] = self.__transformTimeseries_PCA_TRANSFORM(ts_imgs=ts_imgs[mask, :])
            del ts_imgs; gc.collect()  # invoke garbage collector

            ts_imgs = tmp_imgs

            del mask; gc.collect()  # invoke garbage collector
        else:
            self.__transformTimeseries_PCA_FIT(ts_imgs)
            ts_imgs = self.__transformTimeseries_PCA_TRANSFORM(ts_imgs=ts_imgs)

        if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
            ts_imgs = ts_imgs.reshape((tmp_shape[0], tmp_shape[1], ts_imgs.shape[1]))

        ds_imgs[0] = ts_imgs

        # clean up
        gc.collect()  # invoke garbage collector

        # transform test and validation data set
        for id_ds in range(1, len(ds_imgs) // 2):

            ts_imgs = ds_imgs[id_ds]

            if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
                tmp_shape = ts_imgs.shape
                ts_imgs = ts_imgs.reshape((-1, tmp_shape[2]))

            standardize = False if DatasetTransformOP.STANDARTIZE_ZSCORE in self._lst_transform_ops else True

            if self._transform_ops & FLG_UNCHARTED_PIXS == FLG_UNCHARTED_PIXS:
                labels = ds_imgs[nds + id_ds]
                if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT: labels = labels.reshape(-1)

                # transform
                mask = ~_np.isnan(labels)

                tmp_imgs = np.empty(shape=(ts_imgs.shape[0], NFEATURES_TS), dtype=ts_imgs.dtype)
                tmp_imgs[:, :] = _np.nan

                tmp_imgs[mask, :] = self.__transformTimeseries_PCA_TRANSFORM(ts_imgs=ts_imgs[mask, :])
                del ts_imgs; gc.collect()  # invoke garbage collector

                ts_imgs = tmp_imgs

                del mask; gc.collect()  # invoke garbage collector
            else:
                ts_imgs = self.__transformTimeseries_PCA_TRANSFORM(ts_imgs=ts_imgs, standardize=standardize)

            if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
                #
                ts_imgs = ts_imgs.reshape((tmp_shape[0], tmp_shape[1], ts_imgs.shape[1]))

            ds_imgs[id_ds] = ts_imgs

        return ds_imgs

    def __preprocessingSatelliteImages(self, ds_imgs: list) -> list:

        FLG_UNCHARTED_PIXS = DatasetTransformOP.NOT_PROCESS_UNCHARTED_PIXELS.value
        nds = len(ds_imgs) // 2

        try:
            for id_ds in range(nds):

                ts_imgs = ds_imgs[id_ds]; tmp_shape = None

                if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
                    # save original shape of series
                    tmp_shape = ts_imgs.shape
                    # reshape image to time series related to pixels
                    ts_imgs = ts_imgs.reshape((-1, tmp_shape[2]))

                if self._transform_ops & FLG_UNCHARTED_PIXS == FLG_UNCHARTED_PIXS:
                    labels = ds_imgs[id_ds + nds]

                    if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT: labels = labels.reshape(-1)
                    mask = ~_np.isnan(labels)

                    ts_imgs[mask, :] = self.__transformTimeseries(ts_imgs[mask, :])
                    ts_imgs[~mask, :] = _np.nan

                    del mask; gc.collect()  # invoke garbage collector
                else:
                    ts_imgs = self.__transformTimeseries(ts_imgs)

                if self.train_test_val_opt != DatasetSplitOpt.SHUFFLE_SPLIT:
                    # reshape back to series of satellite images
                    ts_imgs = ts_imgs.reshape(tmp_shape)

                ds_imgs[id_ds] = ts_imgs

            gc.collect()  # invoke garbage collector
        except ValueError:
            pass

        """
        Dimensionality reduction using principal component analysis 
        """

        if self._transform_ops & DatasetTransformOP.PCA.value == DatasetTransformOP.PCA.value or \
                self._transform_ops & DatasetTransformOP.PCA_PER_BAND.value == DatasetTransformOP.PCA_PER_BAND.value:

            ds_imgs = self.__transformTimeseries_PCA(ds_imgs=ds_imgs)

        return ds_imgs

    """
    Vegetation index
    TODO move to loader
    """

    def __addVegetationIndex_EVI(self, ts_imgs: _np.ndarray, labels: _np.ndarray) -> _np.ndarray:

        NFEATURES_TS = self._nfeatures_ts

        ee_collection = lazy_import('mlfire.earthengine.collections')
        ModisReflectanceSpectralBands = ee_collection.ModisReflectanceSpectralBands

        ref_blue = ts_imgs[:, :, (ModisReflectanceSpectralBands.BLUE.value - 1)::NFEATURES_TS]
        ref_nir = ts_imgs[:, :, (ModisReflectanceSpectralBands.NIR.value - 1)::NFEATURES_TS]
        ref_red = ts_imgs[:, :, (ModisReflectanceSpectralBands.RED.value - 1)::NFEATURES_TS]

        _np.seterr(divide='ignore', invalid='ignore')

        # constants
        L = 1.; G = 2.5; C1 = 6.; C2 = 7.5

        evi = G * _np.divide(ref_nir - ref_red, ref_nir + C1 * ref_red - C2 * ref_blue + L)

        evi_infs = _np.isinf(evi)
        evi_nans = _np.isnan(evi)

        ninfs = _np.count_nonzero(evi_infs)
        nnans = _np.count_nonzero(evi_nans)

        if ninfs > 0:
            print(f'#inf values = {ninfs} in EVI. The will be removed from data set!')

            labels[_np.any(evi_infs, axis=2)] = _np.nan
            evi = _np.where(evi_infs, _np.nan, evi)

        if nnans > 0:
            print(f'#NaN values = {nnans} in EVI. The will be removed from data set!')

            labels[_np.any(evi_nans, axis=2)] = _np.nan

        ts_imgs = _np.insert(ts_imgs, range(NFEATURES_TS, ts_imgs.shape[2] + 1, NFEATURES_TS), evi, axis=2)

        # clean up and invoke garbage collector
        del evi; gc.collect()

        self._nfeatures_ts += 1

        return ts_imgs, labels

    def __addVegetationIndex_EVI2(self, ts_imgs: _np.ndarray, labels: _np.ndarray) -> [_np.ndarray, _np.ndarray]:

        NFEATURES_TS = self._nfeatures_ts

        ee_collection = lazy_import('mlfire.earthengine.collections')
        ModisReflectanceSpectralBands = ee_collection.ModisReflectanceSpectralBands

        ref_nir = ts_imgs[:, :, (ModisReflectanceSpectralBands.NIR.value - 1)::NFEATURES_TS]
        ref_red = ts_imgs[:, :, (ModisReflectanceSpectralBands.RED.value - 1)::NFEATURES_TS]

        _np.seterr(divide='ignore', invalid='ignore')

        # compute EVI using 2 bands (nir and red)
        evi2 = 2.5 * _np.divide(ref_nir - ref_red, ref_nir + 2.4 * ref_red + 1)

        evi2_infs = _np.isinf(evi2)
        evi2_nans = _np.isnan(evi2)

        ninfs = _np.count_nonzero(evi2_infs)
        nnans = _np.count_nonzero(evi2_nans)

        if ninfs > 0:
            print(f'#inf values = {ninfs} in EVI2. The will be removed from data set!')

            labels[_np.any(evi2_infs, axis=2)] = _np.nan
            evi2 = _np.where(evi2_infs, _np.nan, evi2)

        if nnans > 0:
            print(f'#NaN values = {nnans} in EVI2. The will be removed from data set!')

            labels[_np.any(evi2_nans, axis=2)] = _np.nan

        # add features
        ts_imgs = _np.insert(ts_imgs, range(NFEATURES_TS, ts_imgs.shape[2] + 1, NFEATURES_TS), evi2, axis=2)

        # clean up and invoke garbage collector
        del evi2; gc.collect()

        self._nfeatures_ts += 1

        return ts_imgs, labels

    def __addVegetationIndex_NDVI(self, ts_imgs: _np.ndarray, labels: _np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:

        NFEATURES_TS = self._nfeatures_ts

        ee_collection = lazy_import('mlfire.earthengine.collections')
        ModisReflectanceSpectralBands = ee_collection.ModisReflectanceSpectralBands

        ref_nir = ts_imgs[:, :, (ModisReflectanceSpectralBands.NIR.value - 1)::NFEATURES_TS]
        ref_red = ts_imgs[:, :, (ModisReflectanceSpectralBands.RED.value - 1)::NFEATURES_TS]

        _np.seterr(divide='ignore', invalid='ignore')

        # compute NDVI
        ndvi = _np.divide(ref_nir - ref_red, ref_nir + ref_red)

        ndvi_infs = _np.isinf(ndvi)
        ndvi_nans = _np.isnan(ndvi)

        ninfs = _np.count_nonzero(ndvi_infs)
        nnans = _np.count_nonzero(ndvi_nans)

        if ninfs > 0:
            print(f'#inf values = {ninfs} in NDVI. The will be removed from data set!')

            labels[_np.any(ndvi_infs, axis=2)] = _np.nan
            ndvi = _np.where(ndvi_infs, _np.nan, ndvi)

        if nnans > 0:
            print(f'#NaN values = {nnans} in NDVI. The will be removed from data set!')

            labels[_np.any(ndvi_nans, axis=2)] = _np.nan

        # add to features
        ts_imgs = _np.insert(ts_imgs, range(NFEATURES_TS, ts_imgs.shape[2] + 1, NFEATURES_TS), ndvi, axis=2)

        # clean up and invoke garbage collector
        del ndvi; gc.collect()

        self._nfeatures_ts += 1

        return ts_imgs, labels

    def __addVegetationIndex(self, ts_imgs: _np.ndarray, labels: _np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:

        # https://en.wikipedia.org/wiki/Enhanced_vegetation_index
        # https://lpdaac.usgs.gov/documents/621/MOD13_User_Guide_V61.pdf

        out_ts_imgs = ts_imgs; out_labels = labels

        if self._vi_ops & VegetationIndex.NDVI.value == VegetationIndex.NDVI.value:
            out_ts_imgs, out_labels = self.__addVegetationIndex_NDVI(ts_imgs=out_ts_imgs, labels=out_labels)

        if self._vi_ops & VegetationIndex.EVI.value == VegetationIndex.EVI.value:
            out_ts_imgs, out_labels = self.__addVegetationIndex_EVI(ts_imgs=out_ts_imgs, labels=labels)

        if self._vi_ops & VegetationIndex.EVI2.value == VegetationIndex.EVI2.value:
            out_ts_imgs, out_labels = self.__addVegetationIndex_EVI2(ts_imgs=out_ts_imgs, labels=labels)

        return out_ts_imgs, out_labels

    """
    Data set split into training, test, and validation
    """

    def __splitDataset_SHUFFLE_SPLIT(self, ts_imgs: _np.ndarray, labels: _np.ndarray) -> list:

        # lazy import
        model_selection = lazy_import('sklearn.model_selection')

        # convert to 3D (spatial in time) multi-spectral image to time series related to pixels
        ts_imgs = ts_imgs.reshape((-1, ts_imgs.shape[2]))
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
            return [ts_imgs_train, ts_imgs_test, ts_imgs_val, labels_train, labels_test, ts_imgs_val]
        elif self.test_ratio > 0.:
            return [ts_imgs_train, ts_imgs_test, labels_train, labels_test]
        elif self.val_ratio > 0.:
            return [ts_imgs_train, ts_imgs_val, labels_train, labels_val]
        else:
            return [ts_imgs, labels]

    def __splitDataset_HORIZONTAL_SPLIT(self, satimg_ts: _np.ndarray, labels: _np.ndarray) -> list:

        if self.test_ratio > 0.:

            rows, _, _ = satimg_ts.shape
            hi_rows = int(rows * (1. - self.test_ratio))

            satimg_ts_train = satimg_ts[:hi_rows, :, :]; labels_train = labels[:hi_rows, :]
            satimg_ts_test = satimg_ts[hi_rows:, :, :]; labels_test = labels[hi_rows:, :]

        else:

            satimg_ts_train = satimg_ts; labels_train = labels
            satimg_ts_test = None; labels_test = None

        if self.val_ratio > 0.:

            rows, _, _ = satimg_ts_train.shape
            hi_rows = int(rows * (1. - self.val_ratio))

            satimg_ts_val = satimg_ts_train[hi_rows:, :, :]; labels_val = labels_train[hi_rows:, :]
            satimg_ts_train = satimg_ts_train[:hi_rows, :, :]; labels_train = labels_train[:hi_rows, :]

        else:

            satimg_ts_val = labels_val = None

        if self.test_ratio > 0. and self.val_ratio > 0.:
            return [satimg_ts_train, satimg_ts_test, satimg_ts_val, labels_train, labels_test, labels_val]
        elif self.test_ratio > 0.:
            return [satimg_ts_train, satimg_ts_test, labels_train, labels_test]
        elif self.val_ratio > 0.:
            return [satimg_ts_train, satimg_ts_val, labels_train, labels_val]
        else:
            return [satimg_ts_train, labels_train]

    def __splitDataset_VERTICAL_SPLIT(self, satimg_ts: _np.ndarray, labels: _np.ndarray) -> list:

        if self.test_ratio > 0.:

            _, cols, _ = satimg_ts.shape
            hi_cols = int(cols * (1. - self.test_ratio))

            ts_imgs_train = satimg_ts[:, :hi_cols, :]; labels_train = labels[:, :hi_cols]
            ts_imgs_test = satimg_ts[:, hi_cols:, :]; labels_test = labels[:, hi_cols:]

        else:

            ts_imgs_train = satimg_ts; labels_train = labels
            ts_imgs_test = None; labels_test = None

        if self.val_ratio > 0.:

            _, cols, _ = ts_imgs_train.shape
            hi_cols = int(cols * (1. - self.val_ratio))

            ts_imgs_val = ts_imgs_train[:, hi_cols:, :]; labels_val = labels_train[:, hi_cols:]
            ts_imgs_train = ts_imgs_train[:, :hi_cols, :]; labels_train = labels_train[:, :hi_cols]

        else:

            ts_imgs_val = labels_val = None

        if self.test_ratio > 0. and self.val_ratio > 0.:
            return [ts_imgs_train, ts_imgs_test, ts_imgs_val, labels_train, labels_test, labels_val]
        elif self.test_ratio > 0.:
            return [ts_imgs_train, ts_imgs_test, labels_train, labels_test]
        elif self.val_ratio > 0.:
            return [ts_imgs_train, ts_imgs_val, labels_train, labels_val]
        else:
            return [ts_imgs_train, labels_train]

    def __splitDataset(self, ts_imgs: _np.ndarray, labels: _np.ndarray) -> list:

        if self.train_test_val_opt == DatasetSplitOpt.SHUFFLE_SPLIT:
            ds = self.__splitDataset_SHUFFLE_SPLIT(ts_imgs=ts_imgs, labels=labels)
        elif self.train_test_val_opt == DatasetSplitOpt.IMG_HORIZONTAL_SPLIT:
            ds = self.__splitDataset_HORIZONTAL_SPLIT(satimg_ts=ts_imgs, labels=labels)
        elif self.train_test_val_opt == DatasetSplitOpt.IMG_VERTICAL_SPLIT:
            ds = self.__splitDataset_VERTICAL_SPLIT(satimg_ts=ts_imgs, labels=labels)
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

        if labels.shape != ts_imgs.shape[0:2]:
            raise RuntimeError('Inconsistent shape between satellite images and labels!')

        # vegetation index
        if self._vi_ops >= VegetationIndex.NONE.value:
            ts_imgs, labels = self.__addVegetationIndex(ts_imgs=ts_imgs, labels=labels)

        # TODO fill nan values
        lst_ds = self.__splitDataset(ts_imgs=ts_imgs, labels=labels)

        lst_ds = self.__preprocessingSatelliteImages(ds_imgs=lst_ds)

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

    def getTrainingDataset(self) -> tuple:

        if not self._ds_training: self.createDataset()
        return self._ds_training

    def getTestDataset(self) -> tuple:

        if not self._ds_test and self.test_ratio > 0.: self.createDataset()
        return self._ds_test

    def getValidationDataset(self) -> _np.ndarray:

        if not self._ds_val and self.val_ratio > 0.: self.createDataset()
        return self._ds_val


# use cases
if __name__ == '__main__':

    VAR_DATA_DIR = 'data/tifs'

    VAR_PREFIX_REFLECTANCE_IMG = 'ak_reflec_january_december_{year}_100km'
    VAR_PREFIX_TEMPSURFACE_IMG = 'ak_lst_january_december_{year}_100km'

    VAR_PREFIX_LABEL = 'ak_january_december_{year}_100km'

    # VAR_LABEL_COLLECTION = FireLabelsCollection.MTBS
    VAR_LABEL_COLLECTION = FireLabelsCollection.CCI
    VAR_STR_LABEL_COLLECTION = VAR_LABEL_COLLECTION.name.lower()

    VAR_DS_SPLIT_OPT = DatasetSplitOpt.IMG_VERTICAL_SPLIT
    VAR_TEST_RATIO = 1. / 3.  # split data set to training and test sets in ratio 2 : 1
    VAR_VALIDATION_RATIO = 1. / 3.  # split training data set to new training and validation data sets in ratio 2 : 1

    VAR_TRANSFORM_OPS = [DatasetTransformOP.PCA_PER_BAND, DatasetTransformOP.NOT_PROCESS_UNCHARTED_PIXELS]
    VAR_PCA_OPS = [FactorOP.CUMULATIVE_EXPLAINED_VARIANCE]

    VAR_LST_SATIMGS_REFLECTANCE = []
    VAR_LST_SATIMGS_TEMPSURFACE = []
    VAR_LST_LABELS = []

    VAR_CCI_CONFIDENCE_LEVEL = 70

    for year in range(2004, 2006):

        VAR_PREFIX_REFLECTANCE_IMG_YEAR = VAR_PREFIX_REFLECTANCE_IMG.format(year=year)
        VAR_PREFIX_TEMPSURFACE_IMG_YEAR = VAR_PREFIX_TEMPSURFACE_IMG.format(year=year)

        VAR_PREFIX_LABEL_IMG_YEAR = VAR_PREFIX_LABEL.format(year=year)

        VAR_FN_SATIMG_REFLECTANCE = os.path.join(VAR_DATA_DIR, '{}_epsg3338_area_0.tif'.format(VAR_PREFIX_REFLECTANCE_IMG_YEAR))
        VAR_LST_SATIMGS_REFLECTANCE.append(VAR_FN_SATIMG_REFLECTANCE)

        VAR_FN_SATIMG_TEMPSURFACE = os.path.join(VAR_DATA_DIR, '{}_epsg3338_area_0.tif'.format(VAR_PREFIX_TEMPSURFACE_IMG_YEAR))
        VAR_LST_SATIMGS_TEMPSURFACE.append(VAR_FN_SATIMG_TEMPSURFACE)

        VAR_FN_LABELS = '{}_epsg3338_area_0_{}_labels.tif'.format(VAR_PREFIX_LABEL_IMG_YEAR, VAR_STR_LABEL_COLLECTION)
        VAR_FN_LABELS = os.path.join(VAR_DATA_DIR, VAR_FN_LABELS)
        VAR_LST_LABELS.append(VAR_FN_LABELS)

    print(VAR_LST_SATIMGS_REFLECTANCE)
    print(VAR_LST_SATIMGS_TEMPSURFACE)

    adapter_ts = DataAdapterTS(
        # sources
        lst_satdata_reflectance=VAR_LST_SATIMGS_REFLECTANCE,
        lst_satdata_temperature=VAR_LST_SATIMGS_TEMPSURFACE,
        lst_labels=VAR_LST_LABELS,
        # TODO comment
        label_collection=VAR_LABEL_COLLECTION,
        mtbs_severity_from=MTBSSeverity.LOW,
        cci_confidence_level=VAR_CCI_CONFIDENCE_LEVEL,
        # transformation options
        transform_ops=VAR_TRANSFORM_OPS,
        pca_ops=VAR_PCA_OPS,
        # data set split options
        ds_split_opt=VAR_DS_SPLIT_OPT,  # TODO rename -> split_ops
        test_ratio=VAR_TEST_RATIO,
        val_ratio=VAR_VALIDATION_RATIO
    )

    # set dates
    VAR_INDEX_BEGIN_DATE = 0
    VAR_INDEX_END_DATE = -1

    print(adapter_ts.dates_reflectance)
    print(adapter_ts.dates_tempsurface)

    # TODO comment
    VAR_START_DATE = adapter_ts.dates_reflectance.iloc[VAR_INDEX_BEGIN_DATE]['Date']
    adapter_ts.ds_start_date = VAR_START_DATE
    VAR_END_DATE = adapter_ts.dates_reflectance.iloc[VAR_INDEX_END_DATE]['Date']
    adapter_ts.ds_end_date = VAR_END_DATE

    # TODO comment
    adapter_ts.createDataset()
