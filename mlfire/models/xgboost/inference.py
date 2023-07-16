from typing import Union

from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')
_xgboost = lazy_import('xgboost')


def predict(xgb: _xgboost.XGBClassifier,
            ds: Union[tuple[_np.ndarray], list[_np.ndarray]],
            show_report: bool = False,
            report_with_aucroc: bool = False,
            report_with_cmat: bool = False,
            report_with_scores: bool = False) -> _np.ndarray:

    ts_img: _np.ndarray = ds[0]
    labels: _np.ndarray = ds[1]

    ts_shape = ts_img.shape

    ts_pixels = ts_img.reshape(-1, ts_shape[2])
    labels = labels.reshape(-1)

    # remove uncharted pixels represented as time series
    ts_pixels = ts_pixels[~_np.isnan(labels)]

    labels_pred = _np.empty(shape=labels.shape, dtype=labels.dtype); labels_pred[:] = _np.nan
    labels_pred[~_np.isnan(labels)] = xgb.predict(ts_pixels)

    if show_report:

        utils_report = lazy_import('mlfire.models.utils.report')

        utils_report.show_report(
            labels_true=labels,
            labels_pred=labels_pred,
            with_aucroc=report_with_aucroc,
            with_cmat=report_with_cmat,
            with_scores=report_with_scores
        )

    labels_pred = labels_pred.reshape(ts_shape[0:2])

    return labels_pred
