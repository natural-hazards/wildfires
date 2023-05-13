
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


def saveDatasetT_PETSC_BINARY(ds: Union[tuple[_np.ndarray, _np.ndarray], list[_np.ndarray, _np.ndarray]], fn) -> None:

    ts = ds[0]
    labels = ds[1]

    vec_labels = labels.view(io_petsc.Vec)
    # mat_ts = ts.view(PetscBinaryIO.MatDense) this is not supported yet
    mat_ts = _scipy_sparse.csr_matrix(ts)
    petsc_ds = (mat_ts, vec_labels)

    io = io_petsc.PetscBinaryIO()
    io.writeBinaryFile(fn, petsc_ds)


def saveDataset_HDF5(ds: Union[tuple[_np.ndarray, _np.ndarray], list[_np.ndarray, _np.ndarray]], fn: str) -> None:

    ts = ds[0]
    labels = ds[1]

    with io_h5py.File(fn, 'w') as hf:

        str_type = 'double'
        attr_name = 'MATLAB_class'

        # create matrix of feature vectors
        ts = _np.transpose(ts)

        hfds = hf.create_dataset('X', shape=ts.shape, dtype=_np.float64, data=ts)
        ascii_type = io_h5py.string_dtype('ascii', 6)
        hfds.attrs[attr_name] = _np.array(str_type.encode('ascii'), dtype=ascii_type)

        # create vector of labels
        hf.create_dataset('y', shape=labels.shape, dtype=_np.float64, data=labels)


def saveDataset(ds: tuple[_np.ndarray], fn: str, file_format: FileFormat = FileFormat.HDF5) -> None:

    ts_img = ds[0]; labels = ds[1]

    # reshape
    ts_pixels = ts_img.reshape(ts_img.shape[0], -1).T
    labels = labels.reshape(-1)

    if file_format == FileFormat.PETSC_BINARY:
        saveDatasetT_PETSC_BINARY(ds=(ts_pixels, labels), fn=fn)
    elif file_format == FileFormat.HDF5:
        saveDataset_HDF5(ds=(ts_pixels, labels), fn=fn)
    else:
        raise NotImplementedError
