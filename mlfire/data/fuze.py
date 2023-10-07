
import gc

from enum import Enum
from typing import Union

# TODO comment
from mlfire.earthengine.collections import MTBSRegion, MTBSSeverity

from mlfire.data.loader import SatDataLoader
from mlfire.data.loader import FireMapSelectOpt, SatDataSelectOpt
from mlfire.data.loader import _NFEATURES_REFLECTANCE

# TODO comment
from mlfire.utils.const import LIST_STRINGS
from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')

_ee_collection = lazy_import('mlfire.earthengine.collections')

_ModisReflectanceSpectralBands = _ee_collection.ModisReflectanceSpectralBands


class VegetationIndexSelectOpt(Enum):

    NONE = 0
    EVI = 2
    EVI2 = 4
    NDVI = 8

    def __and__(self, other):

        if isinstance(other, VegetationIndexSelectOpt):
            return VegetationIndexSelectOpt(self.value & other.value)
        elif isinstance(other, int):
            return VegetationIndexSelectOpt(self.value & other)
        else:
            err_msg = f'unsuported operand type(s) for &: {type(self)} and {type(other)}'
            raise TypeError(err_msg)

    def __or__(self, other):

        if isinstance(other, VegetationIndexSelectOpt):
            return VegetationIndexSelectOpt(self.value & other.value)
        elif isinstance(other, int):
            return VegetationIndexSelectOpt(self.value & other)
        else:
            err_msg = f'unsuported operand type(s) for |: {type(self)} and {type(other)}'
            raise TypeError(err_msg)

    def __eq__(self, other):

        if isinstance(other, VegetationIndexSelectOpt):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            return False

    def __hash__(self):

        return self.value


# defines
LIST_VEGETATION_SELECT_OPT = Union[
    VegetationIndexSelectOpt, tuple[VegetationIndexSelectOpt], list[VegetationIndexSelectOpt]
]


class SatDataFuze(SatDataLoader):

    def __init__(self,
                 lst_firemaps: LIST_STRINGS,
                 lst_satdata_reflectance: LIST_STRINGS = None,
                 lst_satdata_temperature: LIST_STRINGS = None,
                 # TODO comment
                 opt_select_satdata: SatDataSelectOpt = SatDataSelectOpt.ALL,  # TODO change argument type
                 # TODO comment
                 opt_select_firemap: FireMapSelectOpt = FireMapSelectOpt.MTBS,
                 # TODO comment
                 select_timestamps: Union[list, tuple, None] = None,
                 # TODO comment
                 cci_confidence_level: int = 70,
                 # TODO comment
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA,
                 # TODO comment
                 mtbs_min_severity: MTBSSeverity = MTBSSeverity.LOW,
                 # TODO comment
                 lst_vegetation_add: LIST_VEGETATION_SELECT_OPT = (VegetationIndexSelectOpt.NONE,),
                 # TODO comment
                 estimate_time: bool = True):

        SatDataLoader.__init__(
            self,
            lst_firemaps=lst_firemaps,
            lst_satdata_reflectance=lst_satdata_reflectance,
            lst_satdata_temperature=lst_satdata_temperature,
            opt_select_firemap=opt_select_firemap,
            opt_select_satdata=opt_select_satdata,
            select_timestamps=select_timestamps,
            cci_confidence_level=cci_confidence_level,
            mtbs_region=mtbs_region,
            mtbs_min_severity=mtbs_min_severity,
            estimate_time=estimate_time
        )

        self._lst_vegetation_index = None; self._vi_ops = -1  # TODO private
        self.lst_vegetation_add = lst_vegetation_add  # TODO rename

    @property
    def lst_vegetation_add(self) -> LIST_VEGETATION_SELECT_OPT:

        return self._lst_vegetation_index

    @lst_vegetation_add.setter
    def lst_vegetation_add(self, vi: LIST_VEGETATION_SELECT_OPT) -> None:
        # check type of input argument
        if vi is None: return

        cnd_check = isinstance(vi, tuple) | isinstance(vi, list)
        cnd_check = cnd_check & isinstance(vi[0], VegetationIndexSelectOpt)
        cnd_check = cnd_check | isinstance(vi, VegetationIndexSelectOpt)

        if not cnd_check:
            err_msg = f'unsupported input type: {type(vi)}'
            raise TypeError(err_msg)

        self._reset()

        if isinstance(vi, VegetationIndexSelectOpt):
            self._vi_ops = vi.value
            self._lst_vegetation_index = (vi,)
        else:
            self._vi_ops = 0
            self._lst_vegetation_index = vi
            for op in vi: self._vi_ops |= op.value

    """
    Vegetation
    """

    @staticmethod
    def __computeVegetationIndex_EVI(reflec: _np.ndarray, firemaps: _np.ndarray) -> _np.ndarray:

        if not isinstance(reflec, _np.ndarray):
            err_msg = f'unsupported type of argument #1: {type(reflec)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if not isinstance(firemaps, _np.ndarray):
            err_msg = f'unsupported type of argument #2: {type(firemaps)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        rs_blue = reflec[:, :, (_ModisReflectanceSpectralBands.BLUE - 1)::_NFEATURES_REFLECTANCE]  # TODO rename
        rs_nir = reflec[:, :, (_ModisReflectanceSpectralBands.NIR - 1)::_NFEATURES_REFLECTANCE]  # TODO rename
        rs_red = reflec[:, :, (_ModisReflectanceSpectralBands.RED - 1)::_NFEATURES_REFLECTANCE]  # TODO rename

        _np.seterr(divide='ignore', invalid='ignore')

        # constants
        L = 1.; G = 2.5; C1 = 6.; C2 = 7.5

        evi = G * _np.divide(rs_nir - rs_red, rs_nir + C1 * rs_red - C2 * rs_blue + L)

        evi_infs = _np.isinf(evi)
        evi_nans = _np.isnan(evi)

        ninfs = _np.count_nonzero(evi_infs)
        nnans = _np.count_nonzero(evi_nans)

        if ninfs > 0:
            msg = f'#inf values = {ninfs} in EVI. The will be removed from data set!'
            print(msg)

            firemaps[_np.any(evi_infs, axis=2)] = _np.nan
            evi = _np.where(evi_infs, _np.nan, evi)

        if nnans > 0:
            msg = f'#NaN values = {nnans} in EVI. These values will be removed from data set!'
            print(msg)

            firemaps[_np.any(evi_nans, axis=2)] = _np.nan

        return evi

    @staticmethod
    def __computeVegetationIndex_EVI2(reflec: _np.ndarray, firemaps: _np.ndarray) -> (_np.ndarray, _np.ndarray):

        if not isinstance(reflec, _np.ndarray):
            err_msg = f'unsupported type of argument #1: {type(reflec)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if not isinstance(firemaps, _np.ndarray):
            err_msg = f'unsupported type of argument #2: {type(firemaps)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        ref_nir = reflec[:, :, (_ModisReflectanceSpectralBands.NIR - 1)::_NFEATURES_REFLECTANCE]  # TODO rename
        ref_red = reflec[:, :, (_ModisReflectanceSpectralBands.RED - 1)::_NFEATURES_REFLECTANCE]  # TODO rename

        _np.seterr(divide='ignore', invalid='ignore')

        # compute EVI using 2 bands (nir and red)
        evi2 = 2.5 * _np.divide(ref_nir - ref_red, ref_nir + 2.4 * ref_red + 1)

        evi2_infs = _np.isinf(evi2)
        evi2_nans = _np.isnan(evi2)

        ninfs = _np.count_nonzero(evi2_infs)
        nnans = _np.count_nonzero(evi2_nans)

        if ninfs > 0:
            msg = f'#inf values = {ninfs} in EVI2. The will be removed from data set!'
            print(msg)

            firemaps[_np.any(evi2_infs, axis=2)] = _np.nan
            evi2 = _np.where(evi2_infs, _np.nan, evi2)

        if nnans > 0:
            msg = f'#NaN values = {nnans} in EVI2. The will be removed from data set!'
            print(msg)

            firemaps[_np.any(evi2_nans, axis=2)] = _np.nan

        return evi2

    @staticmethod
    def __computeVegetationIndex_NDVI(reflec: _np.ndarray, firemaps: _np.ndarray) -> (_np.ndarray, _np.ndarray):

        if not isinstance(reflec, _np.ndarray):
            err_msg = f'unsupported type of argument #1: {type(reflec)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if not isinstance(firemaps, _np.ndarray):
            err_msg = f'unsupported type of argument #2: {type(firemaps)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        ref_nir = reflec[:, :, (_ModisReflectanceSpectralBands.NIR - 1)::_NFEATURES_REFLECTANCE]
        ref_red = reflec[:, :, (_ModisReflectanceSpectralBands.RED - 1)::_NFEATURES_REFLECTANCE]

        _np.seterr(divide='ignore', invalid='ignore')

        # compute NDVI
        ndvi = _np.divide(ref_nir - ref_red, ref_nir + ref_red)

        ndvi_infs = _np.isinf(ndvi)
        ndvi_nans = _np.isnan(ndvi)

        ninfs = _np.count_nonzero(ndvi_infs)
        nnans = _np.count_nonzero(ndvi_nans)

        if ninfs > 0:
            msg = f'#inf values = {ninfs} in NDVI. The will be removed from data set!'
            print(msg)

            firemaps[_np.any(ndvi_infs, axis=2)] = _np.nan
            ndvi = _np.where(ndvi_infs, _np.nan, ndvi)

        if nnans > 0:
            msg = f'#NaN values = {nnans} in NDVI. The will be removed from data set!'
            print(msg)

            firemaps[_np.any(ndvi_nans, axis=2)] = _np.nan

        return ndvi

    def __addVegetationFeatures(self) -> None:

        """
        https://en.wikipedia.org/wiki/Enhanced_vegetation_index
        https://lpdaac.usgs.gov/documents/621/MOD13_User_Guide_V61.pdf
        """

        cnd_reflectance_sel = self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE
        cnd_reflectance_sel &= self.lst_satdata_reflectance is not None

        cnd_temperature_sel = self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE
        cnd_temperature_sel &= self.lst_satdata_temperature is not None

        if not cnd_reflectance_sel:
            if self.lst_satdata_reflectance is not None:
                rows = self._rs_rows_satdata; cols = self._rs_cols_satdata
                len_features = len(self.selected_timestamps_satdata)
                len_features *= _NFEATURES_REFLECTANCE

                shape_reflectance = (len_features, rows, cols)
                np_satdata_reflectance = _np.empty(shape=shape_reflectance)

                self._loadGeoTIFF_DATASETS_SATDATA(opt_select=SatDataSelectOpt.REFLECTANCE)
                self._loadSatData_IMPL(
                    ds_satdata=self._ds_satdata_reflectance,
                    np_satdata=np_satdata_reflectance,
                    opt_select=SatDataSelectOpt.REFLECTANCE
                )

                np_satdata_reflectance = _np.moveaxis(np_satdata_reflectance, 0, -1)

                # clean up
                del self._ds_satdata_reflectance; self._ds_satdata_reflectance = None
                gc.collect()
            else:
                err_msg = 'satellite data (reflectance) is not set'
                raise FileNotFoundError(err_msg)
        else:
            np_satdata_reflectance = self._np_satdata_reflectance

        idx_start = 0
        if cnd_reflectance_sel: idx_start += _NFEATURES_REFLECTANCE
        if cnd_temperature_sel: idx_start += 1

        step_ts = idx_start
        if VegetationIndexSelectOpt.EVI & self._vi_ops == VegetationIndexSelectOpt.EVI: step_ts += 1
        if VegetationIndexSelectOpt.EVI2 & self._vi_ops == VegetationIndexSelectOpt.EVI2: step_ts += 1
        if VegetationIndexSelectOpt.NDVI & self._vi_ops == VegetationIndexSelectOpt.NDVI: step_ts += 1

        if VegetationIndexSelectOpt.EVI & self._vi_ops == VegetationIndexSelectOpt.EVI:
            out_evi = self._np_satdata[:, :, idx_start::step_ts]
            out_evi[:, :, :] = self.__computeVegetationIndex_EVI(reflec=np_satdata_reflectance, firemaps=self._np_firemaps)
            idx_start += 1
            gc.collect()

        if VegetationIndexSelectOpt.EVI2 & self._vi_ops == VegetationIndexSelectOpt.EVI2:
            out_evi2 = self._np_satdata[:, :, idx_start::step_ts]
            out_evi2[:, :, :] = self.__computeVegetationIndex_EVI2(reflec=np_satdata_reflectance, firemaps=self._np_firemaps)
            idx_start += 1
            gc.collect()

        if VegetationIndexSelectOpt.NDVI & self._vi_ops == VegetationIndexSelectOpt.NDVI:
            out_ndvi = self._np_satdata[:, :, idx_start::step_ts]
            out_ndvi[:, :, :] = self.__computeVegetationIndex_NDVI(reflec=np_satdata_reflectance, firemaps=self._np_firemaps)
            gc.collect()

        if not cnd_temperature_sel:
            del np_satdata_reflectance; gc.collect()

    def fuzeData(self) -> None:  # TODO rename method

        set_indexes = {VegetationIndexSelectOpt.EVI, VegetationIndexSelectOpt.EVI2, VegetationIndexSelectOpt.NDVI}
        extra_bands = 0

        if self._vi_ops > 0: extra_bands = len(set_indexes & set(self._lst_vegetation_index))

        self._processMetadata_SATDATA()
        self.loadSatData(extra_bands=extra_bands)

        self._processMetaData_FIREMAPS()
        self.loadFiremaps()

        self.__addVegetationFeatures()

    """
    
    """

    @property
    def len_ts_satdata(self) -> int:

        len_ts = super().len_ts_satdata
        len_timestamps = len(self.selected_timestamps_satdata)

        if VegetationIndexSelectOpt.EVI & self._vi_ops == VegetationIndexSelectOpt.EVI:
            len_ts += len_timestamps
        if VegetationIndexSelectOpt.EVI2 & self._vi_ops == VegetationIndexSelectOpt.EVI2:
            len_ts += len_timestamps
        if VegetationIndexSelectOpt.NDVI & self._vi_ops == VegetationIndexSelectOpt.NDVI:
            len_ts += len_timestamps

        return len_ts


if __name__ == '__main__':

    import os

    VAR_DATA_DIR = 'data/tifs'

    VAR_PREFIX_IMG_REFLECTANCE = 'ak_reflec_january_december_{}_100km'
    VAR_PREFIX_IMG_TEMPERATURE = 'ak_lst_january_december_{}_100km'
    VAR_PREFIX_IMG_LABELS = 'ak_january_december_{}_100km'

    VAR_LST_SATIMGS_REFLECTANCE = []
    VAR_LST_SATIMGS_TEMPERATURE = []
    VAR_LST_FIREMAPS = []

    ADD_VEGETATION = [VegetationIndexSelectOpt.NDVI, VegetationIndexSelectOpt.EVI, VegetationIndexSelectOpt.EVI2]
    # ADD_VEGETATION = (VegetationIndexSelectOpt.NONE,)

    for year in range(2004, 2006):
        VAR_PREFIX_IMG_REFLECTANCE_YEAR = VAR_PREFIX_IMG_REFLECTANCE.format(year)
        VAR_PREFIX_IMG_TEMPERATURE_YEAR = VAR_PREFIX_IMG_TEMPERATURE.format(year)
        VAR_PREFIX_IMG_LABELS_YEAR = VAR_PREFIX_IMG_LABELS.format(year)

        fn_satimg_reflec = '{}_epsg3338_area_0.tif'.format(VAR_PREFIX_IMG_REFLECTANCE_YEAR)
        fn_satimg_reflec = os.path.join(VAR_DATA_DIR, fn_satimg_reflec)
        VAR_LST_SATIMGS_REFLECTANCE.append(fn_satimg_reflec)

        fn_satimg_temperature = '{}_epsg3338_area_0.tif'.format(VAR_PREFIX_IMG_TEMPERATURE_YEAR)
        fn_satimg_temperature = os.path.join(VAR_DATA_DIR, fn_satimg_temperature)
        VAR_LST_SATIMGS_TEMPERATURE.append(fn_satimg_temperature)

        fn_labels_mtbs = '{}_epsg3338_area_0_mtbs_labels.tif'.format(VAR_PREFIX_IMG_LABELS_YEAR)
        fn_labels_mtbs = os.path.join(VAR_DATA_DIR, fn_labels_mtbs)
        VAR_LST_FIREMAPS.append(fn_labels_mtbs)

    # setup of data set loader
    dataset_fuzion = SatDataFuze(
        lst_firemaps=VAR_LST_FIREMAPS,
        lst_satdata_reflectance=VAR_LST_SATIMGS_REFLECTANCE,
        lst_satdata_temperature=VAR_LST_SATIMGS_TEMPERATURE,
        lst_vegetation_add=ADD_VEGETATION,
        opt_select_satdata=SatDataSelectOpt.ALL
    )

    VAR_START_DATE = dataset_fuzion.timestamps_satdata.iloc[0]['Timestamps']
    VAR_END_DATE = dataset_fuzion.timestamps_satdata.iloc[-1]['Timestamps']
    dataset_fuzion.select_timestamps = (VAR_START_DATE, VAR_END_DATE)

    dataset_fuzion.fuzeData()
