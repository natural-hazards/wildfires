
from mlfire.utils.functool import lazy_import

_io = lazy_import('os')

_io_utils = lazy_import('mlfire.utils.io')
_np = lazy_import('numpy')


def loadResult(fn_pred: str, fn_mask: str) -> _np.ndarray:

    if not _io.path.exists(fn_pred):
        raise IOError('File {} not exist!'.format(fn_pred))

    if not _io.path.exists(fn_mask):
        raise IOError('File {} not exist!'.format(fn_mask))

    mask = _io_utils.loadArrayHDF5(fn_mask, 'mask')
    pred = _io_utils.loadArrayHDF5(fn_pred, 'y_predictions')

    y_pred = _np.empty(shape=mask.shape, dtype=pred.dtype); y_pred[:] = _np.nan
    y_pred[~mask] = pred

    return y_pred


def loadSource(fn: str) -> _np.ndarray:

    if not _io.path.exists(fn):
        raise IOError('File {} not exist!'.format(fn))

    mask = _io_utils.loadArrayHDF5(fn, 'mask')
    labels = _io_utils.loadArrayHDF5(fn, 'y')

    y = _np.empty(shape=mask.shape, dtype=labels.dtype); y[:] = _np.nan
    y[~mask] = labels

    return y


if __name__ == '__main__':

    VAR_FN_TEST = 'data/h5/mtbs/ak_modis_2004_2005_100km_labels_test.h5'
    VAR_FN_PRED = 'output/permonsvm/ak_modis_2004_2005_100km_test_labels_predictions.h5'

    VAR_Y_TEST = loadSource(fn=VAR_FN_TEST)
    VAR_Y_PRED = loadResult(fn_pred=VAR_FN_PRED, fn_mask=VAR_FN_TEST)

    print(VAR_Y_TEST.shape)
    print(VAR_Y_PRED.shape)
