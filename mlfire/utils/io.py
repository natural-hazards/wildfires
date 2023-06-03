
from enum import Enum
from typing import Union

from mlfire.utils.functool import lazy_import
from mlfire.utils.functool import optional_import

# lazy imports
_np = lazy_import('numpy')
_scipy_sparse = lazy_import('scipy.sparse')

# optional imports
io_h5py = optional_import('h5py')
io_petsc = optional_import('PetscBinaryIO')


class FileFormat(Enum):

    PETSC_BINARY = 1
    HDF5 = 2


def saveDataset_PETSC_BINARY(ds: Union[tuple[_np.ndarray, _np.ndarray], list[_np.ndarray, _np.ndarray]], fn) -> None:

    raise NotImplementedError

    # ts = ds[0]
    # labels = ds[1]
    #
    # ts_pixels = ts.reshape((-1, ds[0].shape[2]))
    # labels = labels.reshape(-1)
    #
    # vec_labels = labels.view(io_petsc.Vec)
    # # mat_ts = ts.view(PetscBinaryIO.MatDense) this is not supported yet
    # mat_ts = _scipy_sparse.csr_matrix(ts_pixels)
    # petsc_ds = (mat_ts, vec_labels)
    #
    # io = io_petsc.PetscBinaryIO()
    # io.writeBinaryFile(fn, petsc_ds)


def saveDataset_HDF5(ds: Union[tuple[_np.ndarray, _np.ndarray], list[_np.ndarray, _np.ndarray]], fn: str) -> None:

    ts = ds[0]
    labels = ds[1]

    # store binary mask
    mask = _np.isnan(labels)

    # reshape data set
    if len(ds[0].shape) > 2:
        ts_pixels = ts.reshape((-1, ds[0].shape[2]))
        labels = labels.reshape(-1)
    else:
        ts_pixels = ts

    ts_pixels_drop_nans = ts_pixels[~mask.reshape(-1), :]
    labels_drop_nans = labels[~mask.reshape(-1)]

    with io_h5py.File(fn, 'w') as hf:

        str_type = 'double'
        attr_name = 'MATLAB_class'

        # store matrix of feature vectors
        ts_pixels_drop_nans = _np.transpose(ts_pixels_drop_nans)

        hfds = hf.create_dataset('X', shape=ts_pixels_drop_nans.shape, dtype=_np.float64, data=ts_pixels_drop_nans)
        ascii_type = io_h5py.string_dtype('ascii', 6)
        hfds.attrs[attr_name] = _np.array(str_type.encode('ascii'), dtype=ascii_type)

        # store vector of labels
        hf.create_dataset('y', shape=labels_drop_nans.shape, dtype=_np.float64, data=labels_drop_nans)

        # store mask of labels
        hf.create_dataset('mask', shape=mask.shape, dtype=_np.bool_, data=mask)


def saveDataset(ds: tuple[_np.ndarray], fn: str, file_format: FileFormat = FileFormat.HDF5) -> None:

    ts_img = ds[0]
    labels = ds[1]

    if file_format == FileFormat.PETSC_BINARY:
        saveDataset_PETSC_BINARY(ds=(ts_img, labels), fn=fn)
    elif file_format == FileFormat.HDF5:
        saveDataset_HDF5(ds=(ts_img, labels), fn=fn)
    else:
        raise NotImplementedError


def loadArrayHDF5(fn: str, obj_name: str) -> _np.ndarray:

    with io_h5py.File(fn, 'r') as hf:

        y = _np.array(hf[obj_name][:])

    return y
