
import gc

from enum import Enum
from typing import Union

# TODO comment
from mlfire.earthengine.collections import MTBSRegion, MTBSSeverity

from mlfire.data.fuze import VegetationIndexSelectOpt, LIST_VEGETATION_SELECT_OPT
from mlfire.data.loader import FireMapSelectOpt, SatDataSelectOpt
from mlfire.data.view import SatImgViewOpt, FireMapsViewOpt

from mlfire.data.fuze import SatDataFuze
from mlfire.data.view import SatDataView

from mlfire.features.pca import TransformPCA, FactorOP  # TODO rename module mlfire.extractors
from mlfire.features.pca import LIST_PCA_FACTOR_OPT

# utils imports
from mlfire.utils.const import LIST_STRINGS, LIST_NDARRAYS

from mlfire.utils.time import elapsed_timer
from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')
_sk_model_selection = lazy_import('sklearn.model_selection')

_scipy_stats = lazy_import('scipy.stats')
_scipy_signal = lazy_import('scipy.signal')


class SatDataSplitOpt(Enum):

    SHUFFLE_SPLIT = 0
    IMG_HORIZONTAL_SPLIT = 1
    IMG_VERTICAL_SPLIT = 2


class SatDataPreprocessOpt(Enum):

    NONE = 0
    STANDARTIZE_ZSCORE = 1
    PCA = 2
    PCA_PER_BAND = 4  # TODO rename PCA_PER_FEATURE and set 3
    SAVITZKY_GOLAY = 8
    NOT_PROCESS_UNCHARTED_PIXELS = 16
    # ALL?

    def __and__(self, other):

        if isinstance(other, SatDataPreprocessOpt):
            return SatDataPreprocessOpt(self.value & other.value)
        elif isinstance(other, int):
            return SatDataPreprocessOpt(self.value & other)
        else:
            err_msg = f'unsuported operand type(s) for &: {type(self)} and {type(other)}'
            raise TypeError(err_msg)

    def __or__(self, other):  # TODO remove?

        if isinstance(other, SatDataPreprocessOpt):
            return SatDataPreprocessOpt(self.value | other.value)
        elif isinstance(other, int):
            return SatDataPreprocessOpt(self.value | other)
        else:
            err_msg = f'unsuported operand type(s) for |: {type(self)} and {type(other)}'
            raise TypeError(err_msg)

    def __eq__(self, other):

        if isinstance(other, SatDataPreprocessOpt):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            return False


# defines
LIST_PREPROCESS_SATDATA_OPT = Union[
    None, SatDataPreprocessOpt, tuple[SatDataPreprocessOpt], list[SatDataPreprocessOpt]
]


class SatDataAdapterTS(SatDataFuze, SatDataView):

    def __init__(self,
                 lst_firemaps: LIST_STRINGS,
                 # lst_firemaps_test = None
                 lst_satdata_reflectance: LIST_STRINGS = None,
                 # lst_satdata_reflectance_test = None
                 lst_satdata_temperature: LIST_STRINGS = None,
                 # lst_satdata_temperature_test = None
                 # TODO comment
                 opt_select_firemap: FireMapSelectOpt = FireMapSelectOpt.MTBS,
                 # TODO comment
                 opt_select_satdata: Union[SatDataSelectOpt, list[SatDataSelectOpt]] = SatDataSelectOpt.ALL,
                 # TODO comment
                 select_timestamps: Union[list, tuple, None] = None,
                 # TODO comment
                 cci_confidence_level: int = 70,
                 # TODO comment
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA,
                 mtbs_min_severity: MTBSSeverity = MTBSSeverity.LOW,
                 # TODO comment
                 lst_vegetation_add: LIST_VEGETATION_SELECT_OPT = (VegetationIndexSelectOpt.NONE,), # TODO rename
                 # TODO comment
                 opt_split_satdata: SatDataSplitOpt = SatDataSplitOpt.SHUFFLE_SPLIT,
                 test_ratio: float = .33,
                 val_ratio: float = .0,
                 # TODO comment
                 opt_preprocess_satdata: LIST_PREPROCESS_SATDATA_OPT = (SatDataPreprocessOpt.STANDARTIZE_ZSCORE,),
                 # TODO comment
                 savgol_polyorder: int = 1,
                 savgol_winlen: int = 5,
                 # TODO comment
                 opt_pca_factor: LIST_PCA_FACTOR_OPT = (FactorOP.USER_SET,),  # TODO rename
                 pca_retained_variance: float = 0.95,
                 pca_nfactors: int = 2,
                 # view
                 ndvi_view_threshold: Union[float, None] = None,
                 satimg_view_opt: SatImgViewOpt = SatImgViewOpt.NATURAL_COLOR,
                 firemaps_view_opt: FireMapsViewOpt = FireMapsViewOpt.LABEL,
                 # TODO comment
                 estimate_time: bool = True,
                 random_state: int = 42):

        SatDataView.__init__(
            self,
            lst_firemaps=None,
            ndvi_view_threshold=ndvi_view_threshold,
            satimg_view_opt=satimg_view_opt,
            labels_view_opt=firemaps_view_opt
        )

        SatDataFuze.__init__(
            self,
            lst_firemaps=lst_firemaps,
            lst_satdata_reflectance=lst_satdata_reflectance,
            lst_satdata_temperature=lst_satdata_temperature,
            opt_select_firemap=opt_select_firemap,
            opt_select_satdata=opt_select_satdata,
            select_timestamps=select_timestamps,
            cci_confidence_level=cci_confidence_level,
            mtbs_region=mtbs_region,
            mtbs_min_severity=mtbs_min_severity,
            lst_vegetation_add=lst_vegetation_add,
            estimate_time=estimate_time
        )

        self.__lst_satdata = None
        self.__lst_firemaps = None

        self._ds_training = None
        self._ds_test = None
        self._ds_val = None

        # TODO comment

        self.__lst_preprocess_satdata = None; self.__satdata_opt = -1
        self.opt_preprocess_satdata = opt_preprocess_satdata

        self.__savgol_polyorder = None
        self.savgol_polyorder = savgol_polyorder

        self.__savgol_winlen = None
        self.savgol_winlen = savgol_winlen

        # TODO comment

        self.__lst_pca_ops = None
        self.__pca_ops = 0
        self.opt_pca_factor = opt_pca_factor

        self.__pca_retained_variance = None
        self.pca_retained_variance = pca_retained_variance

        self.__pca_nfactors = None
        self.pca_nfactors = pca_nfactors

        self.__lst_extractors = None

        # TODO comment

        self.__opt_split_satdata = None
        self.opt_split_satdata = opt_split_satdata

        self.__test_ratio = None
        self.test_ratio = test_ratio

        self.__val_ratio = None
        self.val_ratio = val_ratio

        self.__random_state = None
        self.random_state = random_state

    @property
    def opt_preprocess_satdata(self) -> tuple[SatDataPreprocessOpt]:

        return self.__lst_preprocess_satdata

    @opt_preprocess_satdata.setter
    def opt_preprocess_satdata(self, opt: LIST_PREPROCESS_SATDATA_OPT):
        # check type of input argument
        if opt is None: return

        cnd_check = isinstance(opt, tuple) | isinstance(opt, list)
        cnd_check = cnd_check & isinstance(opt[0], SatDataPreprocessOpt)
        cnd_check = cnd_check | isinstance(opt, SatDataPreprocessOpt)

        if not cnd_check:
            err_msg = f'unsupported input type: {type(opt)}'
            raise TypeError(err_msg)

        self._reset()

        if isinstance(opt, SatDataPreprocessOpt):
            self.__satdata_opt = opt.value
            self.__lst_preprocess_satdata = (opt,)
        else:
            self.__satdata_opt = 0
            self.__lst_preprocess_satdata = tuple(opt)
            for op in opt: self.__satdata_opt |= op.value

    @property
    def opt_split_satdata(self) -> SatDataSplitOpt:

        return self.__opt_split_satdata

    @opt_split_satdata.setter
    def opt_split_satdata(self, op: SatDataSplitOpt) -> None:

        if self.__opt_split_satdata == op:
            return

        self._reset()
        self.__opt_split_satdata = op

    @property
    def test_ratio(self) -> float:

        return self.__test_ratio

    @test_ratio.setter
    def test_ratio(self, val: float) -> None:

        if self.__test_ratio == val:
            return

        self._reset()
        self.__test_ratio = val

    @property
    def val_ratio(self) -> float:

        return self.__val_ratio

    @val_ratio.setter
    def val_ratio(self, val: float) -> None:

        if self.__val_ratio == val:
            return

        self._reset()
        self.__val_ratio = val

    @property
    def random_state(self) -> int:

        return self.__random_state

    @random_state.setter
    def random_state(self, state: int) -> None:

        if self.__random_state == state:
            return

        self._reset()
        self.__random_state = state

    @property
    def savgol_polyorder(self) -> int:

        return self.__savgol_polyorder

    @savgol_polyorder.setter
    def savgol_polyorder(self, order: int) -> None:

        if self.__savgol_polyorder == order:
            return

        self._reset()
        self.__savgol_polyorder = order

    @property
    def savgol_winlen(self) -> int:

        return self.__savgol_winlen

    @savgol_winlen.setter
    def savgol_winlen(self, winlen: int) -> None:

        if self.__savgol_winlen == winlen:
            return

        self._reset()
        self.__savgol_winlen = winlen

    @property
    def opt_pca_factor(self) -> tuple[FactorOP]:

        return self.__lst_pca_ops

    @opt_pca_factor.setter
    def opt_pca_factor(self, opt: LIST_PCA_FACTOR_OPT) -> None:
        # TODO is this necessary or move this implementation to PCA

        # check type of input argument
        if opt is None: return

        cnd_check = isinstance(opt, tuple) | isinstance(opt, list)
        cnd_check = cnd_check & isinstance(opt[0], FactorOP)
        cnd_check = cnd_check | isinstance(opt, FactorOP)

        if not cnd_check:
            err_msg = f'unsupported input type: {type(opt)}'
            raise TypeError(err_msg)

        cnd_not_list = isinstance(opt, FactorOP)
        if cnd_not_list & ((opt,) == self.__lst_pca_ops):
            return
        elif opt == self.__lst_pca_ops:
            return

        self._reset()

        if cnd_not_list:
            self.__pca_ops = opt.value
            self.__lst_pca_ops = (opt,)
        else:
            self.__pca_ops = 0
            self.__lst_pca_ops = tuple(opt)
            for op in opt: self.__pca_ops |= op.value   # TODO fix later

    @property
    def pca_nfactors(self) -> int:

        return self.__pca_nfactors

    @pca_nfactors.setter
    def pca_nfactors(self, n: int) -> None:  # TODO rename input argument to val
        # TODO check type of input argument
        if self.__pca_nfactors == n:
            return

        self._reset()
        self.__pca_nfactors = n

    @property
    def pca_retained_variance(self) -> float:

        return self.__pca_retained_variance

    @pca_retained_variance.setter
    def pca_retained_variance(self, val: float) -> None:
        # TODO check type of input argument
        if val == self.__pca_retained_variance:
            return

        self._reset()
        self.__pca_retained_variance = val

    def _reset(self) -> None:

        SatDataFuze._reset(self)

        if hasattr(self, '__lst_satdata'): del self.__lst_satdata; self.__lst_satdata = None
        if hasattr(self, '__lst_firemaps'): del self.__lst_firemaps; self.__lst_firemaps = None

        if hasattr(self, '_ds_training'): del self._ds_training; self._ds_training = None
        if hasattr(self, '_ds_test'): del self._ds_test; self._ds_test = None
        if hasattr(self, '_ds_val'): del self._ds_val; self._ds_val = None

        if hasattr(self, '__lst_extractors'): del self.__lst_extractors; self.__lst_extractors = None

        # TODO check

        # clean up
        gc.collect()

    """
    TODO comment
    """

    def __preprocess_STANDARTIZE(self, np_satdata: _np.ndarray) -> _np.ndarray:

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        len_features = len(self.features)
        for feature_id in range(len_features):
            sub_img = np_satdata[:, feature_id::len_features]

            # check if standard deviation is greater than 0
            std_band = _np.std(sub_img)
            if std_band == 0.: continue

            np_satdata[:, feature_id::len_features] = _scipy_stats.zscore(sub_img, axis=1)

        return np_satdata

    def __preproess_FILTER_SAVITZKY_GOLAY(self, np_satdata: _np.ndarray) -> _np.ndarray:

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        len_features = len(self.features)
        for feature_id in range(len_features):
            sub_img = np_satdata[:, feature_id::len_features]

            # apply Savitzky–Golay filter, parameters are user-defined
            np_satdata[:, feature_id::len_features] = _scipy_signal.savgol_filter(
                sub_img,
                window_length=self.savgol_winlen,
                polyorder=self.savgol_polyorder
            )

        return np_satdata

    """
    TODO comment
    """

    def __preprocess_PCA_FIT_ALL_FEATURES(self, np_satdata: _np.ndarray) -> tuple:

        if self.__lst_extractors is not None: return self.__lst_extractors

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        extractor_pca = TransformPCA(
            train_ds=np_satdata,
            factor_ops=self.opt_pca_factor,
            nlatent_factors=self.pca_nfactors,
            retained_variance=self.pca_retained_variance
            # verbose=True
        )
        extractor_pca.fit()

        return (extractor_pca,)

    def __preprocess_PCA_FIT_PER_FEATURE(self, np_satdata: _np.ndarray) -> tuple:

        if self.__lst_extractors is not None: return self.__lst_extractors

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        lst_extractors = []
        nfactors = 0

        # transforming initial satellite data using PCA
        len_features = len(self.features)
        for feature_id in range(len_features):
            sub_img = np_satdata[:, feature_id::len_features]

            extractor_pca = TransformPCA(
                train_ds=sub_img,
                factor_ops=self.opt_pca_factor,
                nlatent_factors=self.pca_nfactors,
                retained_variance=self.pca_retained_variance
                # verbose=True
            )
            extractor_pca.fit()

            # determine max latent factors among features
            nfactors = max(nfactors, extractor_pca.nlatent_factors)
            lst_extractors.append(extractor_pca)

        # refit transformation matrix of PCA for max latent factors among features
        for feature_id in range(len_features):
            extractor_pca = lst_extractors[feature_id]

            if extractor_pca.nlatent_factors < nfactors:
                extractor_pca.nlatent_factors_user = nfactors

                mod_opt_pca = list(self.opt_pca_factor)
                if FactorOP.CUMULATIVE_EXPLAINED_VARIANCE & self.__pca_ops == FactorOP.CUMULATIVE_EXPLAINED_VARIANCE:
                    mod_opt_pca.remove(FactorOP.CUMULATIVE_EXPLAINED_VARIANCE)
                if FactorOP.USER_SET & self.__pca_ops != FactorOP.USER_SET:
                    mod_opt_pca.append(FactorOP.USER_SET)

                extractor_pca.factor_ops = mod_opt_pca
                extractor_pca.fit()

        # convert list of extractor to tuple
        return tuple(lst_extractors)  # lst_extractors

    def __preprocess_PCA_FIT(self, np_satdata: _np.ndarray) -> tuple:

        if self.__lst_extractors is not None: return self.__lst_extractors

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if SatDataPreprocessOpt.PCA_PER_BAND & self.__satdata_opt == SatDataPreprocessOpt.PCA_PER_BAND:
            return self.__preprocess_PCA_FIT_PER_FEATURE(np_satdata=np_satdata)
        else:
            return self.__preprocess_PCA_FIT_ALL_FEATURES(np_satdata=np_satdata)

    def __preprocess_PCA_TRANSFORM_PER_FEATURE(self, np_satdata: _np.ndarray) -> _np.ndarray:

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        len_px = np_satdata.shape[0]; len_features = len(self.features)
        nfactors = self.__lst_extractors[0].nlatent_factors  # TODO this as property

        # TODO alloc with mem map
        proj_shape = (len_px, nfactors * len_features)
        proj_satdata = _np.zeros(proj_shape, dtype=_np.float32)

        for feature_id in range(len_features):
            pca_extractor = self.__lst_extractors[feature_id]

            sub_img = np_satdata[:, feature_id::len_features]
            proj_satdata[:, feature_id::len_features] = pca_extractor.transform(sub_img)

        return proj_satdata

    def __preprocess_PCA_TRANSFORM_ALL_FEATURES(self, np_satdata: _np.ndarray) -> _np.ndarray:

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        raise NotImplementedError

    def __preprocess_PCA_TRANSFORM(self, np_satdata: _np.ndarray) -> _np.ndarray:

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if SatDataPreprocessOpt.PCA_PER_BAND & self.__satdata_opt == SatDataPreprocessOpt.PCA_PER_BAND:
            return self.__preprocess_PCA_TRANSFORM_PER_FEATURE(np_satdata=np_satdata)
        else:
            raise NotImplementedError
            # return self.__preprocess_PCA_FIT_ALL_FEATURES(np_satdata=np_satdata)

    def __preprocess_PCA(self, lst_satdata: LIST_NDARRAYS, lst_firemaps: LIST_NDARRAYS) -> LIST_NDARRAYS:

        # TODO check input argument
        # TODO don't forget on uncharted pixels

        if lst_satdata[0] is None:
            # TODO error
            pass

        # TODO comment
        cnd_reshape = self.opt_split_satdata != SatDataSplitOpt.SHUFFLE_SPLIT

        # TODO comment
        cnd_not_process = SatDataPreprocessOpt.NOT_PROCESS_UNCHARTED_PIXELS & self.__satdata_opt  # TODO rename
        cnd_not_process = cnd_not_process == SatDataPreprocessOpt.NOT_PROCESS_UNCHARTED_PIXELS

        tmp_shape = None
        lst_satdata = list(lst_satdata)

        for id_ds, (np_satdata, np_firemaps) in enumerate(zip(lst_satdata, lst_firemaps)):
            # TODO comment
            if np_satdata is None: continue

            if cnd_reshape:
                tmp_shape = np_satdata.shape
                np_satdata = np_satdata.reshape(-1, tmp_shape[2])

            if cnd_not_process:
                np_firemaps = np_firemaps.reshape(-1); np_mask = ~_np.isnan(np_firemaps)
                _np_satdata = np_satdata[np_mask, :]  # Developer note (Marek): this creates copy of an array
            else:
                _np_satdata = np_satdata

            # transform
            if id_ds == 0:
                # TODO elapsed time
                self.__lst_extractors = self.__preprocess_PCA_FIT(np_satdata=_np_satdata)
                # TODO check if this is right and then latent factors as property
                self.__pca_nfactors = self.__lst_extractors[0].nlatent_factors
            # fit
            # TODO elapsed time
            _np_satdata = self.__preprocess_PCA_TRANSFORM(np_satdata=_np_satdata)
            if cnd_reshape:
                tmp_shape = (tmp_shape[0], tmp_shape[1], _np_satdata.shape[1])
                _np_satdata = _np_satdata.reshape(tmp_shape)
            lst_satdata[id_ds] = _np_satdata

            # clean up memory
            gc.collect()

        return lst_satdata

    def __preprocess(self, lst_satdata: LIST_NDARRAYS, lst_firemaps: LIST_NDARRAYS) -> LIST_NDARRAYS:

        # TODO check input arguments
        # TODO return only time series

        # TODO comment
        cnd_reshape = self.opt_split_satdata != SatDataSplitOpt.SHUFFLE_SPLIT

        # TODO comment
        cnd_not_process = SatDataPreprocessOpt.NOT_PROCESS_UNCHARTED_PIXELS & self.__satdata_opt  # TODO rename
        cnd_not_process = cnd_not_process == SatDataPreprocessOpt.NOT_PROCESS_UNCHARTED_PIXELS

        # TODO comment
        cnd_pca = SatDataPreprocessOpt.PCA & self.__satdata_opt == SatDataPreprocessOpt.PCA
        cnd_pca |= SatDataPreprocessOpt.PCA_PER_BAND & self.__satdata_opt == SatDataPreprocessOpt.PCA_PER_BAND

        # TODO comment
        cnd_zscore = SatDataPreprocessOpt.STANDARTIZE_ZSCORE & self.__satdata_opt
        cnd_zscore = cnd_zscore == SatDataPreprocessOpt.STANDARTIZE_ZSCORE

        for np_satdata, np_firemaps in zip(lst_satdata, lst_firemaps):
            if np_satdata is None: continue

            if cnd_reshape:
                tmp_shape = np_satdata.shape
                np_satdata = np_satdata.reshape(-1, tmp_shape[2])

            # not handle uncharted pixels in preprocessing
            if cnd_not_process:
                np_firemaps = np_firemaps.reshape(-1); np_mask = ~_np.isnan(np_firemaps)
                np_satdata_ = np_satdata[np_mask, :]  # Developer note (Marek): this creates copy of an array
            else:
                np_satdata_ = np_satdata

            # standardization time series using zscore
            if cnd_zscore or cnd_pca:
                msg = 'standardize time series using z-score'
                with elapsed_timer(msg=msg, enable=self.estimate_time):
                    np_satdata[:, :] = self.__preprocess_STANDARTIZE(np_satdata=np_satdata_)

            # filtering using Savitzky-golay filter
            if SatDataPreprocessOpt.SAVITZKY_GOLAY & self.__satdata_opt == SatDataPreprocessOpt.SAVITZKY_GOLAY:
                msg = 'filtering data using Savitzky-Golay'
                with elapsed_timer(msg=msg, enable=self.estimate_time):
                    np_satdata[:, :] = self.__preproess_FILTER_SAVITZKY_GOLAY(np_satdata=np_satdata_)

        """
        Global feature extraction using principal component analysis (PCA)
        """

        if cnd_pca:
            lst_satdata = self.__preprocess_PCA(lst_satdata=lst_satdata, lst_firemaps=lst_firemaps)

        return lst_satdata

    """
    TODO comment
    """

    def __splitData_SHUFFLE(self, satdata: _np.ndarray, firemaps: _np.ndarray) -> (LIST_NDARRAYS, LIST_NDARRAYS):

        if not isinstance(satdata, _np.ndarray):
            err_msg = f'unsupported type of argument #1: {type(satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if not isinstance(firemaps, _np.ndarray):
            err_msg = f'unsupported type of argument #2: {type(firemaps)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        satdata = satdata.reshape((-1, satdata.shape[2]))
        firemaps = firemaps.reshape(-1)

        if self.test_ratio > 0.:
            satdata_train, satdata_test, firemaps_train, firemaps_test = _sk_model_selection.train_test_split(
                satdata,
                firemaps,
                test_size=self.test_ratio,
                random_state=self.random_state
            )
        else:
            satdata_train = satdata; firemaps_train = firemaps
            satdata_test = None; firemaps_test = None

        if self.val_ratio > 0.:
            satdata_train, satdata_val, firemaps_train, firemaps_val = _sk_model_selection.train_test_split(
                satdata,
                firemaps,
                test_size=self.val_ratio,
                random_state=self.random_state
            )
        else:
            satdata_val = firemaps_val = None

        satdata = (satdata_train, satdata_test, satdata_val)
        firemaps = (firemaps_train, firemaps_test, firemaps_val)

        return satdata, firemaps

    def __splitData_HORIZONTAL(self, satdata: _np.ndarray, firemaps: _np.ndarray) -> (LIST_NDARRAYS, LIST_NDARRAYS):

        if not isinstance(satdata, _np.ndarray):
            err_msg = f'unsupported type of argument #1: {type(satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if not isinstance(firemaps, _np.ndarray):
            err_msg = f'unsupported type of argument #2: {type(firemaps)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if self.test_ratio > 0.:
            rows, _, _ = satdata.shape
            hi_rows = int(rows * (1. - self.test_ratio))

            satdata_train = satdata[:hi_rows, :, :]; firemaps_train = firemaps[:hi_rows, :]
            satdata_test = satdata[hi_rows:, :, :]; firemaps_test = firemaps[hi_rows:, :]
        else:
            satdata_train = satdata; firemaps_train = firemaps
            satdata_test = None; firemaps_test = None

        if self.val_ratio > 0.:
            rows, _, _ = satdata.shape
            hi_rows = int(rows * (1. - self.val_ratio))

            satdata_val = satdata_train[hi_rows:, :, :]; firemaps_val = firemaps_train[hi_rows:, :]
            satdata_train = satdata_train[:hi_rows, :, :]; firemaps_train = firemaps_train[:hi_rows, :]
        else:
            satdata_val = firemaps_val = None

        satdata = (satdata_train, satdata_test, satdata_val)
        firemaps = (firemaps_train, firemaps_test, firemaps_val)

        return satdata, firemaps

    def __splitData_VERTICAL(self, satdata: _np.ndarray, firemaps: _np.ndarray) -> (LIST_NDARRAYS, LIST_NDARRAYS):

        if not isinstance(satdata, _np.ndarray):
            err_msg = f'unsupported type of argument #1: {type(satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if not isinstance(firemaps, _np.ndarray):
            err_msg = f'unsupported type of argument #2: {type(firemaps)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if self.test_ratio > 0.:
            _, cols, _ = satdata.shape
            hi_cols = int(cols * (1. - self.test_ratio))

            satdata_train = satdata[:, :hi_cols, :]; firemaps_train = firemaps[:, :hi_cols]
            satdata_test = satdata[:, hi_cols:, :]; firemaps_test = firemaps[:, hi_cols:]
        else:
            satdata_train = satdata; firemaps_train = firemaps
            satdata_test = None; firemaps_test = None

        if self.val_ratio > 0.:
            _, cols, _ = satdata_train.shape
            hi_cols = int(cols * (1. - self.val_ratio))

            satdata_val = satdata_train[:, hi_cols:, :]; firemaps_val = firemaps_train[:, hi_cols:]
            satdata_train = satdata_train[:, :hi_cols, :]; firemaps_train = firemaps_train[:, :hi_cols]
        else:
            satdata_val = firemaps_val = None

        satdata = (satdata_train, satdata_test, satdata_val)
        firemaps = (firemaps_train, firemaps_test, firemaps_val)

        return satdata, firemaps

    def __splitData(self, satdata: _np.ndarray, firemaps: _np.ndarray) -> (LIST_NDARRAYS, LIST_NDARRAYS):

        if not isinstance(satdata, _np.ndarray):
            err_msg = f'unsupported type of argument #1: {type(satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if not isinstance(firemaps, _np.ndarray):
            err_msg = f'unsupported type of argument #2: {type(firemaps)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if self.opt_split_satdata == SatDataSplitOpt.SHUFFLE_SPLIT:
            return self.__splitData_SHUFFLE(satdata=satdata, firemaps=firemaps)
        elif self.opt_split_satdata == SatDataSplitOpt.IMG_HORIZONTAL_SPLIT:
            return self.__splitData_HORIZONTAL(satdata=satdata, firemaps=firemaps)
        else:
            return self.__splitData_VERTICAL(satdata=satdata, firemaps=firemaps)

    """
    TODO comment
    """

    def createDatasets(self) -> None:

        if self._ds_training is not None: return

        self.fuzeData()  # load and combine satellite data, and load fire maps either

        lst_satdata, lst_firemaps = self.__splitData(satdata=self._np_satdata, firemaps=self._np_firemaps)
        lst_satdata = self.__preprocess(lst_satdata=lst_satdata, lst_firemaps=lst_firemaps)

        self._ds_training = (lst_satdata[0], lst_firemaps[0])
        if self.test_ratio > 0.: self._ds_test = (lst_satdata[1], lst_firemaps[1])
        if self.val_ratio > 0.: self._ds_val = (lst_satdata[2], lst_firemaps[2])

        # TODO set list of satellite data and fire maps

    def getTrainingDataset(self) -> tuple[_np.ndarray, _np.ndarray]:

        if self._ds_training is None: self.createDatasets()

        return self._ds_training

    def getTestDataset(self) -> tuple[_np.ndarray, _np.ndarray]:

        if self.test_ratio == 0:
            pass  # TODO warning

        if self._ds_test is None: self.createDatasets()
        return self._ds_test

    def getValDataset(self) -> tuple[_np.ndarray, _np.ndarray]:

        if self.val_ratio == 0:
            pass  # TODO warning

        if self._ds_val is None: self.createDatasets()
        return self._ds_val


if __name__ == '__main__':

    _os = lazy_import('os')

    VAR_DATA_DIR = 'data/tifs'

    VAR_PREFIX_IMG_REFLECTANCE = 'ak_reflec_january_december_{}_100km'
    VAR_PREFIX_IMG_TEMPERATURE = 'ak_lst_january_december_{}_100km'
    VAR_PREFIX_IMG_FIREMAPS = 'ak_january_december_{}_100km'

    VAR_LST_REFLECTANCE = []
    VAR_LST_TEMPERATURE = []
    VAR_LST_FIREMAPS = []

    for year in range(2004, 2006):
        VAR_PREFIX_IMG_REFLECTANCE_YEAR = VAR_PREFIX_IMG_REFLECTANCE.format(year)
        VAR_PREFIX_IMG_TEMPERATURE_YEAR = VAR_PREFIX_IMG_TEMPERATURE.format(year)

        VAR_PREFIX_IMG_FIREMAPS_YEAR = VAR_PREFIX_IMG_FIREMAPS.format(year)

        fn_satimg_reflec = f'{VAR_PREFIX_IMG_REFLECTANCE_YEAR}_epsg3338_area_0.tif'
        fn_satimg_reflec = _os.path.join(VAR_DATA_DIR, fn_satimg_reflec)
        VAR_LST_REFLECTANCE.append(fn_satimg_reflec)

        fn_satimg_temperature = f'{VAR_PREFIX_IMG_TEMPERATURE_YEAR}_epsg3338_area_0.tif'
        fn_satimg_temperature = _os.path.join(VAR_DATA_DIR, fn_satimg_temperature)
        VAR_LST_TEMPERATURE.append(fn_satimg_temperature)

        fn_labels_mtbs = '{}_epsg3338_area_0_mtbs_labels.tif'.format(VAR_PREFIX_IMG_FIREMAPS_YEAR)
        fn_labels_mtbs = _os.path.join(VAR_DATA_DIR, fn_labels_mtbs)
        VAR_LST_FIREMAPS.append(fn_labels_mtbs)

    # setup of data set loader
    dataset_loader = SatDataAdapterTS(
        lst_firemaps=VAR_LST_FIREMAPS,
        lst_satdata_reflectance=VAR_LST_REFLECTANCE,
        lst_satdata_temperature=VAR_LST_TEMPERATURE,
        opt_split_satdata=SatDataSplitOpt.IMG_HORIZONTAL_SPLIT,
        lst_vegetation_add=[VegetationIndexSelectOpt.EVI, VegetationIndexSelectOpt.EVI2, VegetationIndexSelectOpt.NDVI],
        opt_select_satdata=SatDataSelectOpt.ALL,
        estimate_time=True
    )

    print(dataset_loader.timestamps_firemaps)
    print(dataset_loader.timestamps_satdata)

    VAR_START_DATE = dataset_loader.timestamps_satdata.iloc[0]['Timestamps']
    VAR_END_DATE = dataset_loader.timestamps_satdata.iloc[-1]['Timestamps']

    dataset_loader.selected_timestamps = (VAR_START_DATE, VAR_END_DATE)
    print(dataset_loader.shape_satdata)

    dataset_loader.createDatasets()
    ds_train = dataset_loader.getTrainingDataset()

    # print(ds_train[0].shape)
    # print(ds_train[0])
