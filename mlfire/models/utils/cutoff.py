
from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')
_sk_metrics = lazy_import('sklearn.metrics')


def sensivity_specifity_cutoff(y_true, y_pred):

    if len(y_true.shape) == 2:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

    mask = ~_np.isnan(y_true)

    fpr, tpr, thresholds = _sk_metrics.roc_curve(y_true[mask], y_pred[mask])
    idx = _np.argmax(tpr - fpr)

    return thresholds[idx]
