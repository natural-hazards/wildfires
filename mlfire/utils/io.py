import numpy as np

from scipy import sparse


def saveDatasetToPetscBinary(ds: tuple[np.ndarray], fn) -> None:

    import PetscBinaryIO

    ts = ds[0]
    labels = ds[1]

    vec_labels = labels.view(PetscBinaryIO.Vec)
    # mat_ts = ts.view(PetscBinaryIO.MatDense) this is not supported yet
    mat_ts = sparse.csr_matrix(ts)
    petsc_ds = (mat_ts, vec_labels)

    io = PetscBinaryIO.PetscBinaryIO()
    io.writeBinaryFile(fn, petsc_ds)


def saveDatasetToHDF5(ds: tuple[np.ndarray], fn) -> None:

    import h5py

    ts = ds[0]
    labels = ds[1]

    with h5py.File(fn, 'w') as hf:

        str_type = 'double'
        attr_name = 'MATLAB_class'

        # create matrix of feature vectors
        ts = np.transpose(ts)

        hfds = hf.create_dataset('X', shape=ts.shape, dtype=np.float64, data=ts)
        ascii_type = h5py.string_dtype('ascii', 6)
        hfds.attrs[attr_name] = np.array(str_type.encode('ascii'), dtype=ascii_type)

        # create vector of labels
        hf.create_dataset('y', shape=labels.shape, dtype=np.float64, data=labels)
