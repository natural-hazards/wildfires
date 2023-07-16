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

        self._fpr, self._tpr, _ = roc_curve(self._labels_true, self._labels_pred)
        self._auc = auc(self._fpr, self._tpr)

    def plot(self, ax=None) -> None:

        fpr = self.fpr
        tpr = self.tpr

        if ax is None:

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

            plt.show()

        else:

            ax.plot(
                fpr,
                tpr,
                color='darkorange',
                label='AUC ROC = {0:.4f}'.format(self._auc),
            )
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')

            ax.set_title('Receiver operating characteristic')
            ax.legend(loc='lower right')


# test
if __name__ == '__main__':

    _np = lazy_import('numpy')
    _plt_pylab = lazy_import('matplotlib.pylab')
    _utils_cmat = lazy_import('mlfire.utils.cmat')

    y_true = _np.asarray([1, 0, 1, 1, 0])
    y_pred = _np.asarray([0, 0, 1, 1, 0])

    auc_roc = AucRoc(labels_true=y_true, labels_pred=y_pred)
    auc_roc.plot()
