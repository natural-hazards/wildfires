
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

    fn_test = '../../../data/h5/mtbs/ak_modis_2004_2005_100km_test.h5'
    fn_pred = '../../../output/permonsvm/ak_modis_2004_2005_100km_test_predictions.h5'

    y_test = loadSource(fn=fn_test)
    y_pred = loadResult(fn_pred=fn_pred, fn_mask=fn_test)

    print(y_test.shape)
    print(y_pred.shape)
