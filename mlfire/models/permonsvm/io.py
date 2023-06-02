
from mlfire.utils.functool import lazy_import

_io_utils = lazy_import('mlfire.utils.io')
_np = lazy_import('numpy')


def loadResult(fn_pred: str, fn_known: str) -> _np.ndarray:

    mask = _io_utils.loadArrayHDF5(fn_known, 'mask')
    pred = _io_utils.loadArrayHDF5(fn_pred, 'y_predictions')

    y_pred = _np.empty(shape=mask.shape, dtype=pred.dtype); y_pred[:] = _np.nan
    y_pred[~mask] = pred

    return y_pred


if __name__ == '__main__':

    fn_test = '../../../data/h5/mtbs/ak_modis_2004_2005_100km_test.h5'
    fn_pred = '../../../output/permonsvm/ak_modis_2004_2005_100km_test_predictions.h5'

    y_pred = loadResult(fn_pred=fn_pred, fn_known=fn_test)
    print(y_pred.shape)
