import numpy as np  # TODO to lazy import

from enum import Enum
from sklearn.decomposition import PCA as learnPCA

# lazy imports


class FactorOP(Enum):

    NONE = 0
    USER_SET = 1
    IGNORE_FIRST = 2
    TEST_CUMSUM = 4
    TEST_BARTLETT = 8


class TransformPCA(object):

    def __init__(self,
                 train_ds: list,
                 nlatent_factors=2,
                 factor_ops: list[FactorOP] = (FactorOP.USER_SET,),
                 verbose: bool = True) -> None:

        self._pca = None

        # training data set
        self._ds = train_ds
        self.training_dataset = train_ds

        # latent factor
        self._user_nlatent_factors = nlatent_factors
        self._nlatent_factor = -1

        self._lst_factor_ops = None
        self._factor_ops = 0
        self.factor_ops = factor_ops

        # verbose
        self._verbose = False
        self.verbose = verbose

        self._is_trained = False

    @property
    def training_dataset(self) -> list:

        return self._ds

    @training_dataset.setter
    def training_dataset(self, ds: list) -> None:

        # TODO implement
        # if self._ds == ds:
        #     return

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

        if self._lst_factor_ops == ops:
            return

        self.__reset()

        flg = 0
        for op in ops: flg |= op.value

        self._lst_factor_ops = ops
        self._factor_ops = flg

        self._is_trained = False

    @property
    def explained_variance(self) -> np.ndarray:

        return self._pca.explained_variance_ratio_

    @property
    def nlatent_factors(self) -> int:

        if not self._is_trained:
            self.fit()

        return self._nlatent_factor

    @property
    def verbose(self) -> bool:

        return self._verbose

    @verbose.setter
    def verbose(self, flg: bool) -> None:

        self._verbose = flg

    def __reset(self):

        del self._pca; self._pca = None
        self._is_trained = False

    def fit(self) -> None:

        if self._is_trained:
            return

        if self.training_dataset is None:
            raise AttributeError('Training data set is not specified!')

        n_components = self._user_nlatent_factors if self._factor_ops & FactorOP.USER_SET.value == FactorOP.USER_SET.value else None
        if self._factor_ops & FactorOP.USER_SET.value == FactorOP.USER_SET.value and self._factor_ops & FactorOP.IGNORE_FIRST.value == FactorOP.IGNORE_FIRST.value:
            n_components += 1

        # training data transformation
        self._pca = learnPCA(n_components=n_components, svd_solver='full')
        self._pca.fit(self.training_dataset)

        #
        if self._factor_ops & FactorOP.USER_SET.value == FactorOP.USER_SET.value:
            self._nlatent_factor = self.nlatent_factors_user
        elif self._factor_ops & FactorOP.TEST_CUMSUM.value == FactorOP.TEST_CUMSUM.value:
            variance_ratio = self._pca.explained_variance_ratio_
            cs_variance_ratio = np.cumsum(variance_ratio)
            self._nlatent_factor = np.argmax(cs_variance_ratio >= 0.999)

            if self.verbose:
                msg = 'PCA cumsum test, found explainable {} latent factor'.format(self._nlatent_factor)
                if self._nlatent_factor > 1: msg = f'{msg}s'
                print(msg)
        elif self._factor_ops & FactorOP.TEST_BARTLETT.value == FactorOP.TEST_BARTLETT.value:
            # https://www.sciencedirect.com/science/article/pii/S0047259X05000813?ref=cra_js_challenge&fr=RR-1
            # TODO implement
            pass

        self._is_trained = True

    def fit_transform(self, ds):

        self.fit()
        return self.transform(ds)

    def transform(self, ds):

        return self._pca.transform(ds)[:, :self._nlatent_factor]
