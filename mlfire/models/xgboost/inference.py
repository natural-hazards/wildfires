from typing import Union

from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')
_xgboost = lazy_import('xgboost')


def report(labels_true: _np.ndarray,
           labels_pred: _np.ndarray) -> None:

    imblearn_metrics = lazy_import('imblearn.metrics')
    sklearn_metrics = lazy_import('sklearn.metrics')

    print('Classification report:\n')
    print(sklearn_metrics.classification_report(
        labels_true[~_np.isnan(labels_true)], labels_pred[~_np.isnan(labels_true)]
    ))

    print('\nClassification report (imbalanced):\n')
    print(imblearn_metrics.classification_report_imbalanced(
        labels_true[~_np.isnan(labels_true)], labels_pred[~_np.isnan(labels_true)]
    ))

    iou_score = sklearn_metrics.jaccard_score(y_true=labels_true[~_np.isnan(labels_true)], y_pred=labels_pred[~_np.isnan(labels_true)])
    print('\nIoU (intersection over union): {:.2f}'.format(iou_score))


def plot_aucroc(labels_true: _np.ndarray,
                labels_pred: _np.ndarray,
                ax=None) -> None:

    roc = lazy_import('mlfire.utils.roc')

    auc_roc = roc.AucRoc(labels_true=labels_true, labels_pred=labels_pred)
    auc_roc.plot(ax=ax)


def plot_cmat(labels_true: _np.ndarray,
              labels_pred: _np.ndarray,
              ax=None) -> None:

    cmat = lazy_import('mlfire.utils.cmat')

    cm = cmat.ConfusionMatrix(
        y_true=labels_true[~_np.isnan(labels_true)],
        y_pred=labels_pred[~_np.isnan(labels_true)],
        labels=['Background', 'Fire']
    )

    cm.plot(ax=ax)


def predict(xgb: _xgboost.XGBClassifier,
            ds: Union[tuple[_np.ndarray], list[_np.ndarray]],
            with_aucroc: bool = False,
            with_cmat: bool = False,
            with_report: bool = False) -> _np.ndarray:

    utils_report = lazy_import('mlfire.models.utils.report')

    ts_img: _np.ndarray = ds[0]
    labels: _np.ndarray = ds[1]

    ts_shape = ts_img.shape

    ts_pixels = ts_img.reshape(-1, ts_shape[2])
    labels = labels.reshape(-1)

    # remove uncharted pixels represented as time series
    ts_pixels = ts_pixels[~_np.isnan(labels)]

    labels_pred = _np.empty(shape=labels.shape, dtype=labels.dtype); labels_pred[:] = _np.nan
    labels_pred[~_np.isnan(labels)] = xgb.predict(ts_pixels)

    utils_report.show_report(
        labels_true=labels,
        labels_pred=labels_pred,
        with_aucroc=with_aucroc,
        with_cmat=with_cmat,
        with_report=with_report
    )

    labels_pred = labels_pred.reshape(ts_shape[0:2])

    return labels_pred
