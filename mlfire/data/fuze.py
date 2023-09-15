
import gc

from enum import Enum
from typing import Union

#
from mlfire.data.loader import SatDataLoader

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

    def __add__(self, other) -> int:
        # check type
        raise NotImplementedError

    def __sub__(self, other):
        # check type
        raise NotImplementedError


class SatDataFuze(SatDataLoader):

    def __init__(self,
                 lst_firemaps: Union[tuple[str], list[str], None],
                 lst_satdata_reflectance: Union[tuple[str], list[str], None] = None,
                 lst_satdata_temperature: Union[tuple[str], list[str], None] = None,
                 lst_vegetation_ops: Union[tuple[VegetationIndex], list[VegetationIndex]] = (VegetationIndex.NONE,),  # TODO rename
                 ):

        SatDataLoader.__init__(
            self,
            lst_firemaps=lst_firemaps,
            lst_satdata_reflectance=lst_satdata_reflectance,
            lst_satdata_temperature=lst_satdata_temperature
        )

        self._lst_vegetation_index = None
        self._vi_ops = VegetationIndex.NONE.value
        self.vegetation_index = lst_vegetation_ops

    @property
    def vegetation_index(self) -> Union[list[VegetationIndex], tuple[VegetationIndex]]:

        return self._lst_vegetation_index

    @vegetation_index.setter
    def vegetation_index(self, lst_vi: Union[list[VegetationIndex], tuple[VegetationIndex]]) -> None:

        if self.vegetation_index == lst_vi:
            return

        self._reset()

        self._vi_ops = 0
        self._lst_vegetation_index = lst_vi  # TODO remove?
        for op in lst_vi: self._vi_ops |= op.value

    """
    Vegetation
    """

    def __computeVegetationIndex_EVI(self, satdata_reflec: _np.ndarray, labels: _np.ndarray) -> (_np.ndarray, _np.ndarray):
        # TODO rename arguments

        raise NotImplementedError

        # NFEATURES_RELFEC = 7
        #
        # ref_blue = satdata_reflec[:, :, (_ModisReflectanceSpectralBands.BLUE.value - 1)::NFEATURES_RELFEC]
        # ref_nir = satdata_reflec[:, :, (_ModisReflectanceSpectralBands.NIR.value - 1)::NFEATURES_RELFEC]
        # ref_red = satdata_reflec[:, :, (_ModisReflectanceSpectralBands.RED.value - 1)::NFEATURES_RELFEC]
        #
        # _np.seterr(divide='ignore', invalid='ignore')
        #
        # # constants
        # L = 1.; G = 2.5; C1 = 6.; C2 = 7.5
        #
        # evi = G * _np.divide(ref_nir - ref_red, ref_nir + C1 * ref_red - C2 * ref_blue + L)
        #
        # evi_infs = _np.isinf(evi)
        # evi_nans = _np.isnan(evi)
        #
        # ninfs = _np.count_nonzero(evi_infs)
        # nnans = _np.count_nonzero(evi_nans)
        #
        # if ninfs > 0:
        #     msg = f'#inf values = {ninfs} in EVI. The will be removed from data set!'
        #     print(msg)
        #
        #     labels[_np.any(evi_infs, axis=2)] = _np.nan
        #     evi = _np.where(evi_infs, _np.nan, evi)
        #
        # if nnans > 0:
        #     msg = f'#NaN values = {nnans} in EVI. The will be removed from data set!'
        #     print(msg)
        #
        #     labels[_np.any(evi_nans, axis=2)] = _np.nan
        #
        # ts_imgs = _np.insert(ts_imgs, range(NFEATURES_TS, ts_imgs.shape[2] + 1, NFEATURES_TS), evi, axis=2)
        #
        # # clean up and invoke garbage collector
        # del evi; gc.collect()
        #
        # self._nfeatures_ts += 1
        #
        # return ts_imgs, labels

    def __computeVegetationIndexes(self):

        pass

    def fuzeData(self) -> None:  # rename

        self.loadSatData()  # TODO extra features for vegetation indexes and infrared
        self.loadFiremaps()
        # TODO load labels
        self.__computeVegetationIndexes()


if __name__ == '__main__':

    import os

    VAR_DATA_DIR = 'data/tifs'

    VAR_PREFIX_IMG_REFLECTANCE = 'ak_reflec_january_december_{}_100km'
    VAR_PREFIX_IMG_TEMPERATURE = 'ak_lst_january_december_{}_100km'
    VAR_PREFIX_IMG_LABELS = 'ak_january_december_{}_100km'

    VAR_LST_SATIMGS_REFLECTANCE = []
    VAR_LST_SATIMGS_TEMPERATURE = []
    VAR_LST_FIREMAPS = []

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
        lst_satdata_temperature=VAR_LST_SATIMGS_TEMPERATURE
    )

    VAR_START_DATE = dataset_fuzion.timestamps_reflectance.iloc[0]['Timestamps']
    VAR_END_DATE = dataset_fuzion.timestamps_reflectance.iloc[-1]['Timestamps']
    dataset_fuzion.select_timestamps = (VAR_START_DATE, VAR_END_DATE)

    dataset_fuzion.fuzeData()
