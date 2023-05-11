import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

from mlfire.utils.functool import lazy_import

# lazy import
_np = lazy_import('numpy')


class AucRoc(object):

    def __init__(self, labels_true, labels_pred):

        # TODO check if labels true and labels pred has same nan values

        self._labels_true = labels_true.reshape(-1)
        self._labels_true = self._labels_true[~_np.isnan(self._labels_true)]

        self._labels_pred = labels_pred.reshape(-1)
        self._labels_pred = self._labels_pred[~_np.isnan(self._labels_pred)]

        self._fpr = None
        self._tpr = None

        self._auc = None

    @property
    def auc(self) -> float:

        if self._auc is None:
            self.__compute()

        return self._auc

    @property
    def fpr(self) -> float:

        if self._fpr is None:
            self.__compute()

        return self._fpr

    @property
    def tpr(self) -> float:

        if self._tpr is None:
            self.__compute()

        return self._tpr

    def __reset(self) -> None:

        del self._fpr, self._tpr
        self._fpr = self._tpr = None

        self._auc = None

    def __compute(self) -> None:

        self._fpr, self._tpr, _ = roc_curve(self._labels_true.reshape(-1), self._labels_pred.reshape(-1))
        self._auc = auc(self._fpr, self._tpr)

    def plot(self) -> None:

        fpr = self.fpr
        tpr = self.tpr

        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            label='AUC ROC = {0:.4f}'.format(self._auc),
        )
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.title('Receiver operating characteristic')
        plt.legend(loc='lower right')
