
import gc
import itertools

from enum import Enum
from typing import Union

from mlfire.earthengine.collections import MTBSRegion, MTBSSeverity

from mlfire.data.loader import SatDataLoader
from mlfire.data.loader import FireMapSelectOpt, SatDataSelectOpt
from mlfire.data.loader import _NFEATURES_REFLECTANCE

from mlfire.utils.const import LIST_STRINGS
from mlfire.utils.functool import lazy_import
from mlfire.utils.time import elapsed_timer

# lazy imports
_np = lazy_import('numpy')
_ee_collection = lazy_import('mlfire.earthengine.collections')

# lazy import (classes)
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

    def __or__(self, other):  # TODO remove?
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

    def __str__(self) -> str:
        return self.name.lower()


# defines
LIST_VEGETATION_SELECT_OPT = Union[
    VegetationIndexSelectOpt, tuple[VegetationIndexSelectOpt, ...], list[VegetationIndexSelectOpt, ...], None
]


class SatDataFuze(SatDataLoader):

    def __init__(self,
                 lst_firemaps: LIST_STRINGS,
                 lst_satdata_reflectance: LIST_STRINGS = None,
                 lst_satdata_temperature: LIST_STRINGS = None,
                 # TODO comment
                 opt_select_satdata: SatDataSelectOpt = SatDataSelectOpt.ALL,
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

        self.__shape_satdata = None
        self.__lst_features = None

        self.__lst_vegetation_index = None; self.__vi_ops = -1
        self.lst_vegetation_add = lst_vegetation_add  # TODO rename

    @property
    def lst_vegetation_add(self) -> tuple[VegetationIndexSelectOpt, ...]:

        return self.__lst_vegetation_index

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
            self.__vi_ops = vi.value
            self.__lst_vegetation_index = (vi,)
        else:
            self.__vi_ops = 0
            self.__lst_vegetation_index = tuple(vi)
            for op in vi: self.__vi_ops |= op.value

    def _reset(self) -> None:

        SatDataLoader._reset(self)

        if hasattr(self, '__lst_features'): del self.__lst_features; self.__lst_features = None
        if hasattr(self, '__shape_satdata'): del self.__shape_satdata; self.__shape_satdata = None

    """
    Vegetation
    """

    @staticmethod
    def __computeVegetationIndex_EVI(reflec: _np.ndarray, firemaps: _np.ndarray) -> _np.ndarray:
        # check arguments
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
            msg = f'#inf values = {ninfs} in EVI. These values will be removed from data set!'
            Warning(msg)

            firemaps[_np.any(evi_infs, axis=2)] = _np.nan
            evi = _np.where(evi_infs, _np.nan, evi)

        if nnans > 0:
            msg = f'#NaN values = {nnans} in EVI. These values will be removed from data set!'
            Warning(msg)

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
            msg = f'#inf values = {ninfs} in EVI2. These values will be removed from data set!'
            Warning(msg)

            firemaps[_np.any(evi2_infs, axis=2)] = _np.nan
            evi2 = _np.where(evi2_infs, _np.nan, evi2)

        if nnans > 0:
            msg = f'#NaN values = {nnans} in EVI2. These values will be removed from data set!'
            Warning(msg)

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
            msg = f'#inf values = {ninfs} in NDVI. These values will be removed from data set!'
            Warning(msg)

            firemaps[_np.any(ndvi_infs, axis=2)] = _np.nan
            ndvi = _np.where(ndvi_infs, _np.nan, ndvi)

        if nnans > 0:
            msg = f'#NaN values = {nnans} in NDVI. These values will be removed from data set!'
            Warning(msg)

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
                # TODO raise error
                pass

            len_timestamps = len(self.selected_timestamps_satdata); nfeatures = int(_NFEATURES_REFLECTANCE)
            rows, cols, _, _ = self.shape_satdata

            shape_reflectance = (len_timestamps * nfeatures, rows, cols)
            # TODO alloc with memory map
            np_reflec = _np.empty(shape=shape_reflectance)

            # TODO incorporate into ds_satdata_reflectance property
            self._loadGeoTIFF_DATASETS_SATDATA(opt_select=SatDataSelectOpt.REFLECTANCE)

            msg = 'loading satellite data (temperature)'
            with elapsed_timer(msg=msg, enable=self.estimate_time):
                np_reflec = self._loadSatData_IMPL(
                    ds_satdata=self._ds_satdata_reflectance,
                    np_satdata=np_reflec,
                    layout_layers=self._layout_layers_reflectance,
                    nfeatures=nfeatures
                )

            np_reflec = _np.moveaxis(np_reflec, 0, -1); np_reflec *= 1e-4
        else:
            _, _, len_timestamps, len_features = self.shape_satdata
            end_selection = int(_NFEATURES_REFLECTANCE)

            # TODO comment
            idx = list(itertools.chain(
                *[range(i * len_features, i * len_features + end_selection) for i in range(len_timestamps)]
            ))

            np_reflec = self._np_satdata[:, :, idx]

        # TODO use satdata_shape and improve implementation
        idx_start = 0
        if cnd_reflectance_sel: idx_start += _NFEATURES_REFLECTANCE
        if cnd_temperature_sel: idx_start += 1

        step_ts = idx_start
        if VegetationIndexSelectOpt.EVI & self.__vi_ops == VegetationIndexSelectOpt.EVI: step_ts += 1
        if VegetationIndexSelectOpt.EVI2 & self.__vi_ops == VegetationIndexSelectOpt.EVI2: step_ts += 1
        if VegetationIndexSelectOpt.NDVI & self.__vi_ops == VegetationIndexSelectOpt.NDVI: step_ts += 1

        if VegetationIndexSelectOpt.EVI & self.__vi_ops == VegetationIndexSelectOpt.EVI:
            out_evi = self._np_satdata[:, :, idx_start::step_ts]
            out_evi[:, :, :] = self.__computeVegetationIndex_EVI(reflec=np_reflec, firemaps=self._np_firemaps)
            idx_start += 1
            gc.collect()

        if VegetationIndexSelectOpt.EVI2 & self.__vi_ops == VegetationIndexSelectOpt.EVI2:
            out_evi2 = self._np_satdata[:, :, idx_start::step_ts]
            out_evi2[:, :, :] = self.__computeVegetationIndex_EVI2(reflec=np_reflec, firemaps=self._np_firemaps)
            idx_start += 1
            gc.collect()

        if VegetationIndexSelectOpt.NDVI & self.__vi_ops == VegetationIndexSelectOpt.NDVI:
            out_ndvi = self._np_satdata[:, :, idx_start::step_ts]
            out_ndvi[:, :, :] = self.__computeVegetationIndex_NDVI(reflec=np_reflec, firemaps=self._np_firemaps)
            gc.collect()

        if not cnd_reflectance_sel:
            del self._layout_layers_reflectance; del self._ds_satdata_reflectance
            del np_reflec

            # clean up
            gc.collect()

    def fuzeData(self) -> None:  # TODO rename method

        self._processMetadata_SATDATA()
        self.loadSatData()

        self._processMetaData_FIREMAPS()
        self.loadFiremaps()

        if self.__vi_ops > 0: self.__addVegetationFeatures()

    """
    Shape (satellite data)  
    """

    @property
    def features(self) -> tuple:

        if self.__lst_features is not None: return self.__lst_features

        lst_features = list(super().features)
        if VegetationIndexSelectOpt.EVI & self.__vi_ops == VegetationIndexSelectOpt.EVI:
            lst_features.append(str(VegetationIndexSelectOpt.EVI))
        if VegetationIndexSelectOpt.EVI2 & self.__vi_ops == VegetationIndexSelectOpt.EVI2:
            lst_features.append(str(VegetationIndexSelectOpt.EVI2))
        if VegetationIndexSelectOpt.NDVI & self.__vi_ops == VegetationIndexSelectOpt.NDVI:
            lst_features.append(str(VegetationIndexSelectOpt.NDVI))

        self.__lst_features = tuple(lst_features)
        return self.__lst_features


if __name__ == '__main__':

    import os

    VAR_DATA_DIR = 'data/tifs'

    VAR_PREFIX_IMG_REFLECTANCE = 'ak_reflec_january_december_{}_100km'
    VAR_PREFIX_IMG_TEMPERATURE = 'ak_lst_january_december_{}_100km'
    VAR_PREFIX_IMG_LABELS = 'ak_january_december_{}_100km'

    VAR_LST_SATIMGS_REFLECTANCE = []
    VAR_LST_SATIMGS_TEMPERATURE = []
    VAR_LST_FIREMAPS = []

    # ADD_VEGETATION = [VegetationIndexSelectOpt.NDVI, VegetationIndexSelectOpt.EVI, VegetationIndexSelectOpt.EVI2]
    ADD_VEGETATION = (VegetationIndexSelectOpt.NONE,)

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
        # lst_satdata_temperature=VAR_LST_SATIMGS_TEMPERATURE,
        lst_vegetation_add=ADD_VEGETATION,
        opt_select_satdata=SatDataSelectOpt.ALL
    )

    VAR_START_DATE = dataset_fuzion.timestamps_satdata.iloc[0]['Timestamps']
    VAR_END_DATE = dataset_fuzion.timestamps_satdata.iloc[-1]['Timestamps']

    dataset_fuzion.selected_timestamps = (VAR_START_DATE, VAR_END_DATE)

    print(dataset_fuzion.shape_satdata)
    print(dataset_fuzion.shape_firemap)
    print(dataset_fuzion.features)

    dataset_fuzion.fuzeData()
