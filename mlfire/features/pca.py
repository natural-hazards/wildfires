
from enum import Enum, auto
from typing import Union

# utils
from mlfire.utils.device import Device
from mlfire.utils.functool import lazy_import, optional_import

# lazy imports
_np = lazy_import('numpy')
_stats = lazy_import('scipy.stats')
_sklearn_decomposition = lazy_import('sklearn.decomposition')

# optional imports
_cuml_common = optional_import('cuml.common')
_rapids_decomposition = optional_import('cuml.decomposition.pca')


class FactorOP(Enum):  # TODO rename

    NONE = 0
    USER_SET = 1
    CUMULATIVE_EXPLAINED_VARIANCE = 2
    ALL = 3

    def __and__(self, other):
        # TODO improve implementation
        if isinstance(other, FactorOP):
            return FactorOP(self.value & other.value)
        elif isinstance(other, int):
            return FactorOP(self.value & other)
        else:
            raise NotImplementedError

    def __or__(self, other):
        # TODO improve implementation
        return FactorOP(self.value | other.value)

    # TODO or
    # TODO eq
    # TODO str


# defines
LIST_PCA_FACTOR_OPT = Union[
    FactorOP, list[FactorOP], tuple[FactorOP], None
]


class ExtractorPCA(object):

    def __init__(self,
                 ds: _np.ndarray,
                 opt_factor: LIST_PCA_FACTOR_OPT = (FactorOP.USER_SET,),
                 nfactors: int = 2,
                 retained_variance: float = 0.95,
                 device: Device = Device.CPU,
                 verbose: bool = True,
                 random_state: int = 42) -> None:

        self.__pca = None

        # initial data set
        self.__ds = None
        self.initial_ds = ds

        self.__lst_factor_ops = None
        self.__factor_ops = 0
        self.opt_factor = opt_factor

        # TODO comment
        self.__user_nfactors = nfactors
        self.__factors = -1

        # TODO comment
        self.__retained_variance = None
        self.retained_variance = retained_variance

        # device
        self.__device = None
        self.device = device

        # random state
        self.__random_state = None
        self.random_state = random_state

        # verbose
        self.__verbose = False   # TODO change to estimate time
        self.verbose = verbose

        self.__is_trained = False

    def __del__(self):
        del self.__pca

    @property
    def initial_ds(self) -> _np.ndarray:
        return self.__ds

    @initial_ds.setter
    def initial_ds(self, ds: _np.ndarray) -> None:
        if self.__ds is not None and (self.__ds == ds).all():
            return

        self.__reset()
        self.__ds = ds

    @property
    def opt_factor(self) -> tuple[FactorOP, ...]:
        return self.__lst_factor_ops

    @opt_factor.setter
    def opt_factor(self, ops: LIST_PCA_FACTOR_OPT):

        # TODO reimplement

        if self.__lst_factor_ops == ops:
            return

        self.__reset()

        flg = 0
        for op in ops: flg |= op.value

        self.__lst_factor_ops = ops
        self.__factor_ops = flg

        self.__is_trained = False

    @property
    def nfactors_user(self) -> int:

        return self.__user_nfactors

    @nfactors_user.setter
    def nfactors_user(self, n: int) -> None:

        if self.__user_nfactors == n:
            return

        self.__reset()

        self.__user_nfactors = n
        self.__is_trained = False

    @property
    def nfactors(self) -> int:

        if not self.__is_trained:
            self.fit()

        return self.__factors

    @property
    def retained_variance(self) -> float:
        return self.__retained_variance

    @retained_variance.setter
    def retained_variance(self, val: float) -> None:
        if self.__retained_variance == val:
            return

        self.__reset()
        self.__retained_variance = val

    @property
    def explained_variance_ratio(self) -> _np.ndarray:
        if self.__pca is None:
            raise RuntimeError('PCA trasformation is not performed!')
        return self.__pca.explained_variance_ratio_

    @property
    def device(self) -> Device:
        return self.__device

    @device.setter
    def device(self, dev: Device) -> None:
        if self.__device == dev:
            return

        self.__reset()
        self.__device = dev

    @property
    def random_state(self) -> int:
        return self.__random_state

    @random_state.setter
    def random_state(self, state) -> None:

        if self.__random_state == state:
            return

        self.__reset()
        self.__random_state = state

    @property
    def verbose(self) -> bool:
        return self.__verbose

    @verbose.setter
    def verbose(self, flg: bool) -> None:

        self.__verbose = flg

    def __reset(self):
        del self.__pca; self.__pca = None
        self.__is_trained = False

    def __estimateLatentFactors(self) -> None:

        if self.__pca is None:
            raise RuntimeError('PCA trasformation is not performed!')

        var_ratio = self.__pca.explained_variance_ratio_
        cumsum_var_ratio = _np.cumsum(var_ratio)
        self.__factors = _np.argmax(cumsum_var_ratio >= self.retained_variance)

        # if self.verbose:
        msg = f'PCA, found {self.__factors} latent factor'
        if self.__factors > 1: msg = f'{msg}s'
        msg = f'{msg} using cumulative explained variance'
      
        print(msg)

    def fit(self) -> None:

        if self.__is_trained:
            return

        if self.initial_ds is None:
            raise RuntimeError('Training data set is not specified!')

        # TODO comment
        nfactors = self.nfactors_user if self.__factor_ops & FactorOP.USER_SET.value == FactorOP.USER_SET.value else None

        if self.device == Device.GPU:
            _cuml_common.logger.set_level(0)
            self.__pca = _rapids_decomposition.PCA(
                n_components=nfactors,
                svd_solver='auto',
                random_state=self.random_state
            )
        else:
            self.__pca = _sklearn_decomposition.PCA(
                n_components=nfactors,
                svd_solver='auto',
                random_state=self.random_state
            )

        # TODO comment
        self.__pca.fit(self.initial_ds)

        if self.__factor_ops & FactorOP.USER_SET.value == FactorOP.USER_SET.value:
            self.__factors = self.nfactors_user
        else:
            self.__estimateLatentFactors()

        self.__is_trained = True

    def fit_transform(self, ds: _np.ndarray):
        self.fit()
        return self.transform(ds)

    def transform(self, ds: _np.ndarray):
        return self.__pca.transform(ds)[:, :self.__factors]
