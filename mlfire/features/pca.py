
from enum import Enum
from typing import Union

# utils
from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')
_stats = lazy_import('scipy.stats')
_sklearn_decomposition = lazy_import('sklearn.decomposition')


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


class TransformPCA(object):  # TODO rename -> ExtractorPCA

    def __init__(self,
                 train_ds: _np.ndarray,  # rename init ds
                 nlatent_factors: int = 2,
                 factor_ops: LIST_PCA_FACTOR_OPT = (FactorOP.USER_SET,),
                 retained_variance: float = 0.95,
                 verbose: bool = True) -> None:

        self._pca = None

        # training data set
        self._ds = train_ds
        self.training_dataset = train_ds

        # latent factor properties
        self._user_nlatent_factors = nlatent_factors
        self._nlatent_factors = -1

        self._lst_factor_ops = None
        self._factor_ops = 0
        self.factor_ops = factor_ops

        # set significance level for Bartlett test
        self._retained_variance = None
        self.retained_variance = retained_variance

        # verbose
        self._verbose = False # change to estimate time
        self.verbose = verbose

        self._is_trained = False

    @property
    def training_dataset(self) -> _np.ndarray:

        return self._ds

    @training_dataset.setter
    def training_dataset(self, ds: _np.ndarray) -> None:

        if self._ds is not None and (self._ds == ds).all():
            return

        self.__reset()
        self._ds = ds

    @property
    def nlatent_factors_user(self) -> int:

        return self._user_nlatent_factors

    @nlatent_factors_user.setter
    def nlatent_factors_user(self, n: int) -> None:

        if self._user_nlatent_factors == n:
            return

        self.__reset()

        self._user_nlatent_factors = n
        self._is_trained = False

    @property
    def factor_ops(self) -> list[FactorOP]:

        return self._lst_factor_ops

    @factor_ops.setter
    def factor_ops(self, ops: list[FactorOP]):

        # TODO reimplement

        if self._lst_factor_ops == ops:
            return

        self.__reset()

        flg = 0
        for op in ops: flg |= op.value

        self._lst_factor_ops = ops
        self._factor_ops = flg

        self._is_trained = False

    @property
    def explained_variance_ratio(self) -> _np.ndarray:

        if self._pca is None:
            raise RuntimeError('PCA trasformation is not performed!')

        return self._pca.explained_variance_ratio_

    @property
    def nlatent_factors(self) -> int:

        if not self._is_trained:
            self.fit()

        return self._nlatent_factors

    @property
    def retained_variance(self) -> float:

        return self._retained_variance

    @retained_variance.setter
    def retained_variance(self, val: float) -> None:

        if self._retained_variance == val:
            return

        self.__reset()
        self._retained_variance = val

    @property
    def verbose(self) -> bool:

        return self._verbose

    @verbose.setter
    def verbose(self, flg: bool) -> None:

        self._verbose = flg

    def __reset(self):

        del self._pca; self._pca = None
        self._is_trained = False

    def __estimateLatentFactors(self) -> None:

        if self._pca is None:
            raise RuntimeError('PCA trasformation is not performed!')

        var_ratio = self._pca.explained_variance_ratio_
        cumsum_var_ratio = _np.cumsum(var_ratio)
        self._nlatent_factors = _np.argmax(cumsum_var_ratio >= self.retained_variance)

        # if self.verbose:
        msg = 'PCA, found {} latent factor using cumulative explained variance'.format(self._nlatent_factors)
        if self._nlatent_factors > 1: msg = f'{msg}s'
        print(msg)

    def fit(self) -> None:

        if self._is_trained:
            return

        if self.training_dataset is None:
            raise RuntimeError('Training data set is not specified!')

        # fit data transformation
        n_components = self.nlatent_factors_user if self._factor_ops & FactorOP.USER_SET.value == FactorOP.USER_SET.value else None

        self._pca = _sklearn_decomposition.PCA(n_components=n_components, svd_solver='auto', random_state=42)
        self._pca.fit(self.training_dataset)

        if self._factor_ops & FactorOP.USER_SET.value == FactorOP.USER_SET.value:
            self._nlatent_factors = self.nlatent_factors_user
        else:
            self.__estimateLatentFactors()

        self._is_trained = True

    def fit_transform(self, ds):

        self.fit()
        return self.transform(ds)

    def transform(self, ds):

        return self._pca.transform(ds)[:, :self._nlatent_factors]
