
from enum import Enum
from typing import Union

from mlfire.utils.functool import lazy_import
from mlfire.utils.functool import optional_import

# lazy imports
_colors = lazy_import('mlfire.utils.colors')

_gdal = lazy_import('osgeo.gdal')
_osr = lazy_import('osgeo.osr')

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
    # firemaps = ds[1]
    #
    # ts_pixels = ts.reshape((-1, ds[0].shape[2]))
    # firemaps = firemaps.reshape(-1)
    #
    # vec_labels = firemaps.view(io_petsc.Vec)
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

        # store vector of firemaps
        hf.create_dataset('y', shape=labels_drop_nans.shape, dtype=_np.float64, data=labels_drop_nans)

        # store mask of firemaps
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


def saveGeoTiff(fn_output: str, firemap: _np.ndarray, firemap_test: _np.ndarray, transform, projection,
                show_uncharted_pixels=False, reproject: bool = True) -> None:

    # TODO horizontal split

    rows_train, cols = firemap.shape
    rows_test, cols = firemap_test.shape

    Colors = _colors.Colors

    # create geotif data set
    driver = _gdal.GetDriverByName('GTiff')

    options = ('PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF')
    ds_firemap = driver.Create(fn_output, cols, rows_train + rows_test, 3, _gdal.GDT_UInt16, options=options)

    srs = _osr.SpatialReference()
    srs.ImportFromWkt(projection)

    # define colors
    BACKGROUD_COLOR_1 = Colors.GRAY_COLOR.value
    BACKGROUD_COLOR_2 = Colors.WHITE_COLOR.value
    BACKGROUD_COLOR_TRAIN = [int(0.5 * BACKGROUD_COLOR_2[i] + 0.5 * Colors.GREEN_COLOR.value[i]) for i in range(3)]

    # numpy array
    np_firemap = _np.empty((rows_train + rows_test, cols, 3))
    np_firemap[:] = BACKGROUD_COLOR_1

    # training
    np_firemap[:rows_train, :][firemap == 0, :] = BACKGROUD_COLOR_TRAIN
    np_firemap[:rows_train, :][firemap == 1, :] = Colors.RED_COLOR.value
    np_firemap[:rows_train, :][_np.isnan(firemap), :] = (0, 0, 0) if show_uncharted_pixels else BACKGROUD_COLOR_TRAIN

    # TODO validation data set

    # test
    np_firemap[rows_train:, :][firemap_test == 0, :] = Colors.GRAY_COLOR.value
    np_firemap[rows_train:, :][firemap_test == 1, :] = Colors.RED_COLOR.value
    np_firemap[rows_train:, :][_np.isnan(firemap_test), :] = (0, 0, 0) if show_uncharted_pixels else Colors.GRAY_COLOR.value

    for i in range(3): ds_firemap.GetRasterBand(i + 1).WriteArray(np_firemap[:, :, i])

    ds_firemap.SetGeoTransform(transform)
    ds_firemap.SetProjection(srs.ExportToWkt())
    ds_firemap.FlushCache()
    ds_firemap = None

    # reproject for Google Earth
    if reproject:
        kwargs = {'format': 'GTiff', 'dstSRS': 'EPSG:4326', 'dstAlpha': True}
        ds_firemap = _gdal.Warp(fn_output, fn_output, **kwargs)
        ds_firemap.FlushCache()
        ds_firemap = None

