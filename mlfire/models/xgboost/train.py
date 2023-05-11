from typing import Union

from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')
_xgboost = lazy_import('xgboost')

_time = lazy_import('mlfire.utils.time')


def trainSegmentationModel(xgb: _xgboost.XGBClassifier,
                           ds: Union[tuple[_np.ndarray], list[_np.ndarray]]) -> None:

    ts_img: _np.ndarray = ds[0]
    labels: _np.ndarray = ds[1]

    ts_shape = ts_img.shape

    ts_pixels = ts_img.reshape(ts_shape[0], -1).T
    labels = labels.reshape(-1)

    # remove uncharted pixels represented as time series
    ts_pixels = ts_pixels[~_np.isnan(labels)]
    labels = labels[~_np.isnan(labels)]

    # train model
    with _time.elapsed_timer('Training XGBoost model'):
        xgb.fit(ts_pixels, labels)
