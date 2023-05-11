from typing import Union

from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')
_xgboost = lazy_import('xgboost')


def report(labels_pred: _np.ndarray,
           labels_true: _np.ndarray) -> None:

    imblearn_metrics = lazy_import('imblearn.metrics')
    sklearn_metrics = lazy_import('sklearn.metrics')

    print('Classification report:')
    print(sklearn_metrics.classification_report(
        labels_true[~_np.isnan(labels_true)], labels_pred[~_np.isnan(labels_true)]
    ))

    print('Classification report (imbalanced)')
    print(imblearn_metrics.classification_report_imbalanced(
        labels_true[~_np.isnan(labels_true)], labels_pred[~_np.isnan(labels_true)]
    ))


def predict(xgb: _xgboost.XGBClassifier,
            ds: Union[tuple[_np.ndarray], list[_np.ndarray]],
            with_report: bool = False) -> _np.ndarray:

    ts_img: _np.ndarray = ds[0]
    labels: _np.ndarray = ds[1]

    ts_shape = ts_img.shape

    ts_pixels = ts_img.reshape(ts_shape[0], -1).T
    labels = labels.reshape(-1)

    # remove uncharted pixels represented as time series
    ts_pixels = ts_pixels[~_np.isnan(labels)]

    labels_pred = _np.empty(shape=labels.shape, dtype=labels.dtype); labels_pred[:] = _np.nan
    labels_pred[~_np.isnan(labels)] = xgb.predict(ts_pixels)

    if with_report: report(labels_pred=labels_pred, labels_true=labels)

    # change back to original shape
    labels_pred = labels_pred.reshape(ts_shape[1:3])

    return labels_pred
