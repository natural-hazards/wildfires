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
