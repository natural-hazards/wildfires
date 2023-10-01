
import gc

from enum import Enum
from typing import Union

#
from mlfire.data.loader import _NFEATURES_REFLECTANCE
from mlfire.data.loader import SatDataLoader, SatDataSelectOpt

#
from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')

_ee_collection = lazy_import('mlfire.earthengine.collections')

_ModisReflectanceSpectralBands = _ee_collection.ModisReflectanceSpectralBands


class VegetationIndex(Enum):

    NONE = 0
    EVI = 2
    EVI2 = 4
    NDVI = 8

    def __and__(self, other):
        if isinstance(other, VegetationIndex):
            return VegetationIndex(self.value & other.value)
        elif isinstance(other, int):
            return VegetationIndex(self.value & other)
        else:
            raise NotImplementedError

    def __eq__(self, other):
        # TODO check type
        if not isinstance(other, VegetationIndex):
            raise TypeError

        return self.value == other.value


class SatDataFuze(SatDataLoader):

    def __init__(self,
                 lst_firemaps: Union[tuple[str], list[str], None],
                 lst_satdata_reflectance: Union[tuple[str], list[str], None] = None,
                 lst_satdata_temperature: Union[tuple[str], list[str], None] = None,
                 lst_vegetation_add: Union[tuple[VegetationIndex], list[VegetationIndex]] = (VegetationIndex.NONE,),
                 opt_select_satdata: Union[SatDataSelectOpt, list[SatDataSelectOpt]] = SatDataSelectOpt.ALL,
                 ):

        SatDataLoader.__init__(
            self,
            lst_firemaps=lst_firemaps,
            lst_satdata_reflectance=lst_satdata_reflectance,
            lst_satdata_temperature=lst_satdata_temperature,
            opt_select_satdata=opt_select_satdata
        )

        self._lst_vegetation_index = None
        self._vi_ops = VegetationIndex.NONE.value
        self.lst_vegetation_add = lst_vegetation_add

    @property
    def lst_vegetation_add(self) -> Union[list[VegetationIndex], tuple[VegetationIndex]]:

        return self._lst_vegetation_index

    @lst_vegetation_add.setter
    def lst_vegetation_add(self, lst_vi: Union[list[VegetationIndex], tuple[VegetationIndex]]) -> None:

        if self.lst_vegetation_add == lst_vi:
            return

        self._reset()

        self._vi_ops = 0
        self._lst_vegetation_index = lst_vi
        for op in lst_vi: self._vi_ops |= op.value

    """
    Vegetation
    """

    @staticmethod
    def __computeVegetationIndex_EVI(reflec: _np.ndarray, labels: _np.ndarray) -> _np.ndarray:

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

            labels[_np.any(evi_infs, axis=2)] = _np.nan
            evi = _np.where(evi_infs, _np.nan, evi)

        if nnans > 0:
            msg = f'#NaN values = {nnans} in EVI. These values will be removed from data set!'
            print(msg)

            labels[_np.any(evi_nans, axis=2)] = _np.nan

        return evi

    @staticmethod
    def __computeVegetationIndex_EVI2(reflec: _np.ndarray, labels: _np.ndarray) -> (_np.ndarray, _np.ndarray):

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

            labels[_np.any(evi2_infs, axis=2)] = _np.nan
            evi2 = _np.where(evi2_infs, _np.nan, evi2)

        if nnans > 0:
            msg = f'#NaN values = {nnans} in EVI2. The will be removed from data set!'
            print(msg)

            labels[_np.any(evi2_nans, axis=2)] = _np.nan

        return evi2

    @staticmethod
    def __computeVegetationIndex_NDVI(reflec: _np.ndarray, labels: _np.ndarray) -> (_np.ndarray, _np.ndarray):

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
            print(f'#inf values = {ninfs} in NDVI. The will be removed from data set!')

            labels[_np.any(ndvi_infs, axis=2)] = _np.nan
            ndvi = _np.where(ndvi_infs, _np.nan, ndvi)

        if nnans > 0:
            print(f'#NaN values = {nnans} in NDVI. The will be removed from data set!')

            labels[_np.any(ndvi_nans, axis=2)] = _np.nan

        return ndvi

    def __addVegetationProperties(self):

        """
        https://en.wikipedia.org/wiki/Enhanced_vegetation_index
        https://lpdaac.usgs.gov/documents/621/MOD13_User_Guide_V61.pdf
        """

        cnd_reflectance_sel = self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE
        cnd_temperature_sel = self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE

        if not cnd_reflectance_sel:
            if self.lst_satdata_reflectance is not None:
                rows = self.rs_rows; cols = self.rs_cols
                np_satdata_reflectance = _np.empty((_NFEATURES_REFLECTANCE * self._ntimestamps, rows, cols))

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
                raise TypeError   # is this right error?
        else:
            np_satdata_reflectance = self._np_satdata_reflectance

        idx_start = 0
        if cnd_reflectance_sel: idx_start += _NFEATURES_REFLECTANCE
        if cnd_temperature_sel: idx_start += 1

        step_ts = idx_start
        if VegetationIndex.EVI & self._vi_ops == VegetationIndex.EVI: step_ts += 1
        if VegetationIndex.EVI2 & self._vi_ops == VegetationIndex.EVI2: step_ts += 1
        if VegetationIndex.NDVI & self._vi_ops == VegetationIndex.NDVI: step_ts += 1

        if VegetationIndex.EVI & self._vi_ops == VegetationIndex.EVI:
            out_evi = self._np_satdata[:, :, idx_start::step_ts]
            out_evi[:, :, :] = self.__computeVegetationIndex_EVI(reflec=np_satdata_reflectance, labels=self._np_firemaps)
            idx_start += 1
            gc.collect()

        if VegetationIndex.EVI2 & self._vi_ops == VegetationIndex.EVI2:
            out_evi2 = self._np_satdata[:, :, idx_start::step_ts]
            out_evi2[:, :, :] = self.__computeVegetationIndex_EVI2(reflec=np_satdata_reflectance, labels=self._np_firemaps)
            idx_start += 1
            gc.collect()

        if VegetationIndex.NDVI & self._vi_ops == VegetationIndex.NDVI:
            out_ndvi = self._np_satdata[:, :, idx_start::step_ts]
            out_ndvi[:, :, :] = self.__computeVegetationIndex_NDVI(reflec=np_satdata_reflectance, labels=self._np_firemaps)
            idx_start += 1
            gc.collect()

        if not cnd_temperature_sel:
            del np_satdata_reflectance; gc.collect()

    def fuzeData(self) -> None:

        set_indexes = {VegetationIndex.EVI, VegetationIndex.EVI2, VegetationIndex.NDVI}
        extra_bands = 0

        if self._vi_ops > 0: extra_bands = len(set_indexes & set(self._lst_vegetation_index))

        self._processMetadata_SATDATA()
        self.loadSatData(extra_bands=extra_bands)

        self._processMetaData_FIREMAPS()
        self.loadFiremaps()

        self.__addVegetationProperties()

    """
    
    """

    @property
    def len_satdata_ts(self) -> int:

        len_ts = super().len_satdata_ts

        if VegetationIndex.EVI & self._vi_ops == VegetationIndex.EVI: len_ts += self._ntimestamps
        if VegetationIndex.EVI2 & self._vi_ops == VegetationIndex.EVI2: len_ts += self._ntimestamps
        if VegetationIndex.NDVI & self._vi_ops == VegetationIndex.NDVI: len_ts += self._ntimestamps

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

    ADD_VEGETATION = [VegetationIndex.NDVI, VegetationIndex.EVI, VegetationIndex.EVI2]
    # ADD_VEGETATION = (VegetationIndex.NONE,)

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
        opt_select_satdata=SatDataSelectOpt.NONE
    )

    VAR_START_DATE = dataset_fuzion.timestamps_satdata.iloc[0]['Timestamps']
    VAR_END_DATE = dataset_fuzion.timestamps_satdata.iloc[-1]['Timestamps']
    dataset_fuzion.select_timestamps = (VAR_START_DATE, VAR_END_DATE)

    dataset_fuzion.fuzeData()
