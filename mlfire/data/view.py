import gc
import os

from enum import Enum
from typing import Union

from mlfire.data.loader import DatasetLoader, SatDataSelectOpt
from mlfire.earthengine.collections import ModisCollection, ModisReflectanceSpectralBands
from mlfire.earthengine.collections import FireLabelCollection
from mlfire.earthengine.collections import MTBSSeverity, MTBSRegion

# utils imports
from mlfire.utils.functool import lazy_import
from mlfire.utils.utils_string import band2date_firecci
from mlfire.utils.plots import imshow

# lazy imports
_np = lazy_import('numpy')


class SatImgViewOpt(Enum):

    CIR = 'Color Infrared, Vegetation'
    EVI = 'EVI'
    EVI2 = 'EVI2'
    NATURAL_COLOR = 'Natural Color'
    NDVI = 'NVDI'
    SHORTWAVE_INFRARED1 = 'Shortwave Infrared using SWIR1'
    SHORTWAVE_INFRARED2 = 'Shortwave Infrared using SWIR2'


class FireLabelsViewOpt(Enum):

    LABEL = 1
    CONFIDENCE_LEVEL = 2
    SEVERITY = 3


class DatasetView(DatasetLoader):  # TODO rename -> SatDataView

    def __init__(self,
                 lst_firemaps: Union[tuple[str], list[str]],
                 lst_satdata_reflectance: Union[tuple[str], list[str], None] = None,
                 lst_satdata_temperature: Union[tuple[str], list[str], None] = None,
                 opt_select_satdata: SatDataSelectOpt = SatDataSelectOpt.ALL,
                 opt_select_firemap: FireLabelCollection = FireLabelCollection.MTBS,  # TODO rename -> WildfireMapCollection
                 # TODO comment
                 cci_confidence_level: int = 70,
                 # TODO comment
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA,
                 mtbs_min_severity: MTBSSeverity = MTBSSeverity.LOW,
                 # TODO comment
                 ndvi_view_threshold: Union[float, None] = None,
                 satimg_view_opt: SatImgViewOpt = SatImgViewOpt.NATURAL_COLOR,
                 labels_view_opt: FireLabelsViewOpt = FireLabelsViewOpt.LABEL) -> None:

        super().__init__(
            lst_firemaps=lst_firemaps,
            lst_satdata_reflectance=lst_satdata_reflectance,
            lst_satdata_temperature=lst_satdata_temperature,
            opt_select_satdata=opt_select_satdata,
            opt_select_firemap=opt_select_firemap,
            cci_confidence_level=cci_confidence_level,
            mtbs_region=mtbs_region,
            mtbs_min_severity=mtbs_min_severity
        )

        self._satimg_view_opt = None
        self.satimg_view_opt = satimg_view_opt

        self._labels_view_opt = None
        self.labels_view_opt = labels_view_opt

        self._ndvi_view_thrs = None
        self.ndvi_view_threshold = ndvi_view_threshold

    @property
    def satimg_view_opt(self) -> SatImgViewOpt:

        return self._satimg_view_opt

    @satimg_view_opt.setter
    def satimg_view_opt(self, opt: SatImgViewOpt) -> None:

        if self._satimg_view_opt == opt: return
        self._satimg_view_opt = opt

    @property
    def labels_view_opt(self) -> FireLabelsViewOpt:

        return self._labels_view_opt

    @labels_view_opt.setter
    def labels_view_opt(self, opt: FireLabelsViewOpt) -> None:

        if self._labels_view_opt == opt: return
        self._labels_view_opt = opt

    @property
    def ndvi_view_threshold(self) -> float:

        return self._ndvi_view_thrs

    @ndvi_view_threshold.setter
    def ndvi_view_threshold(self, thrs: float) -> None:

        if thrs is None:
            self._ndvi_view_thrs = -1
            return

        if thrs < 0 or thrs > 1:
            raise ValueError('Threshold value must be between 0 and 1')

        self._ndvi_view_thrs = thrs

    """
    Display functionality (MULTISPECTRAL SATELLITE IMAGE, MODIS)
    """

    def __getSatelliteImageArray_MODIS_CIR(self, img_id: int) -> _np.ndarray:

        id_ds, start_band_id = self._map_layout_relectance[img_id]
        ds_satimg = self._ds_satdata_reflectance[id_ds]

        # display CIR representation of MODIS input (color infrared - vegetation)
        # https://eos.com/make-an-analysis/color-infrared/
        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_nir = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.RED.value - 1
        ref_red = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.GREEN.value - 1
        ref_green = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        # get image in range 0 - 255 per channel
        cir_img = _np.empty(shape=ref_nir.shape + (3,), dtype=_np.uint8)
        cir_img[:, :, 0] = (ref_nir + 100.) / 16100. * 255.
        cir_img[:, :, 1] = (ref_red + 100.) / 16100. * 255.
        cir_img[:, :, 2] = (ref_green + 100.) / 16100. * 255.

        return cir_img

    def __getSatelliteImageArray_MODIS_EVI(self, img_id: int) -> _np.ndarray:

        cmap = lazy_import('mlfire.utils.cmap')

        id_ds, start_band_id = self._map_layout_relectance[img_id]
        ds_satimg = self._ds_satdata_reflectance[id_ds]

        band_id = start_band_id + ModisReflectanceSpectralBands.BLUE.value - 1
        ref_blue = ds_satimg.GetRasterBand(band_id).ReadAsArray() / 1e4

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_nir = ds_satimg.GetRasterBand(band_id).ReadAsArray() / 1e4

        band_id = start_band_id + ModisReflectanceSpectralBands.RED.value - 1
        ref_red = ds_satimg.GetRasterBand(band_id).ReadAsArray() / 1e4

        _np.seterr(divide='ignore')

        # constants
        L = 1.; G = 2.5; C1 = 6.; C2 = 7.5

        # EVI computation
        evi = G * _np.divide(ref_nir - ref_red, ref_nir + C1 * ref_red - C2 * ref_blue + L)
        ninf = _np.count_nonzero(evi == _np.inf)
        if ninf > 0:
            print(f'#inf values = {ninf} in NDVI')
            evi = _np.where(evi == _np.inf, -99, evi)

        # Colour palette and thresholding inspired by
        # https://developers.google.com/earth-engine/datasets/catalog/MODIS_MOD09GA_006_EVI#description
        # Credit: Zachary Langford @langfordzl
        lst_colors = [
            '#ffffff', '#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', '#74a901',
            '#66a000', '#529400', '#3e8601', '#207401', '#056201', '#004c00', '#023b01',
            '#012e01', '#011d01', '#011301'
        ]

        # get EVI2 heat map
        cmap_helper = cmap.CMapHelper(lst_colors=lst_colors, vmin=.2, vmax=.8)
        img_evi = cmap_helper.getRGBA(evi)[:, :, :-1]
        img_evi[evi == -99, :] = [0, 0, 0]

        return img_evi

    def __getSatelliteImageArray_MODIS_EVI2(self, img_id: int):

        cmap = lazy_import('mlfire.utils.cmap')

        id_ds, start_band_id = self._map_layout_relectance[img_id]
        ds_satimg = self._ds_satdata_reflectance[id_ds]

        ref_red = ds_satimg.GetRasterBand(start_band_id).ReadAsArray() / 1e4

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_nir = ds_satimg.GetRasterBand(band_id).ReadAsArray() / 1e4

        evi2 = 2.5 * _np.divide(ref_nir - ref_red, ref_nir + 2.4 * ref_red + 1.)
        ninf = _np.count_nonzero(evi2 == _np.inf)
        if ninf > 0:
            print(f'#inf values = {ninf} in NDVI')
            evi2 = _np.where(evi2 == _np.inf, -99, evi2)

        # Colour palette and thresholding inspired by
        # https://developers.google.com/earth-engine/datasets/catalog/MODIS_MOD09GA_006_EVI#description
        # Credit: Zachary Langford @langfordzl
        lst_colors = [
            '#ffffff', '#ce7e45', '#df923d', '#f1b555', '#fcd163', '#99b718', '#74a901',
            '#66a000', '#529400', '#3e8601', '#207401', '#056201', '#004c00', '#023b01',
            '#012e01', '#011d01', '#011301'
        ]

        # get EVI2 heat map
        cmap_helper = cmap.CMapHelper(lst_colors=lst_colors, vmin=.2, vmax=.8)
        img_evi2 = cmap_helper.getRGBA(evi2)[:, :, :-1]
        img_evi2[evi2 == -99, :] = [0, 0, 0]

        return img_evi2

    def __getSatelliteImageArray_MODIS_NATURAL_COLOR(self, img_id: int) -> _np.ndarray:

        id_ds, start_band_id = self._map_layout_relectance[img_id]
        ds_satimg = self._ds_satdata_reflectance[id_ds]

        ref_red = ds_satimg.GetRasterBand(start_band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.GREEN.value - 1
        ref_green = self._ds_satdata_reflectance[id_ds].GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.BLUE.value - 1
        ref_blue = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        # get image in range 0 - 255 per channel
        img = _np.empty(shape=ref_red.shape + (3,), dtype=_np.uint8)
        img[:, :, 0] = (ref_red + 100.) / 16100. * 255.
        img[:, :, 1] = (ref_green + 100.) / 16100. * 255.
        img[:, :, 2] = (ref_blue + 100.) / 16100. * 255.

        return img

    def __getSatelliteImageArray_NDVI(self, img_id: int) -> _np.ndarray:

        # lazy imports
        plt = lazy_import('matplotlib.pylab')

        id_ds, start_band_id = self._map_layout_relectance[img_id]
        ds_satimg = self._ds_satdata_reflectance[id_ds]

        # computing Normalized Difference Vegetation Index (NDVI)
        ref_red = ds_satimg.GetRasterBand(start_band_id).ReadAsArray() / 1e4

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_nir = ds_satimg.GetRasterBand(band_id).ReadAsArray() / 1e4

        # Colour palette and thresholding inspired by
        # https://www.neonscience.org/resources/learning-hub/tutorials/calc-ndvi-tiles-py
        # Credit: Zachary Langford @langfordzl
        cmap = plt.get_cmap(name='RdYlGn') if self.ndvi_view_threshold > -1. else plt.get_cmap(name='seismic')

        _np.seterr(divide='ignore')

        ndvi = _np.divide(ref_nir - ref_red, ref_nir + ref_red)
        ninf = _np.count_nonzero(ndvi == _np.inf)
        if ninf > 0:
            print(f'#inf values = {ninf} in NDVI')
            ndvi = _np.where(ndvi == _np.inf, -99, ndvi)

        img_ndvi = _np.uint8(cmap(ndvi)[:, :, :-1] * 255)
        if self.ndvi_view_threshold > -1: img_ndvi[ndvi < self.ndvi_view_threshold] = [255, 255, 255]
        img_ndvi[ndvi == -99] = [0, 0, 0]

        return img_ndvi

    def __getSatelliteImageArray_MODIS_SHORTWAVE_INFRARED_SWIR1(self, img_id: int) -> _np.ndarray:

        id_ds, start_band_id = self._map_layout_relectance[img_id]
        ds_satimg = self._ds_satdata_reflectance[id_ds]

        # https://eos.com/make-an-analysis/vegetation-analysis/
        band_id = start_band_id + ModisReflectanceSpectralBands.SWIR1.value - 1
        ref_swir1 = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_nir = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        ref_red = ds_satimg.GetRasterBand(start_band_id).ReadAsArray()

        # get image in range 0 - 255 per channel
        shortwave_ir1 = _np.empty(shape=ref_swir1.shape + (3,), dtype=_np.uint8)
        shortwave_ir1[:, :, 0] = (ref_swir1 + 100.) / 16100. * 255.
        shortwave_ir1[:, :, 1] = (ref_nir + 100.) / 16100. * 255.
        shortwave_ir1[:, :, 2] = (ref_red + 100.) / 16100. * 255.

        return shortwave_ir1

    def __getSatelliteImageArray_MODIS_SHORTWAVE_INFRARED_SWIR2(self, img_id: int) -> _np.ndarray:

        id_ds, start_band_id = self._map_layout_relectance[img_id]
        ds_satimg = self._ds_satdata_reflectance[id_ds]

        # https://eos.com/make-an-analysis/shortwave-infrared/
        band_id = start_band_id + ModisReflectanceSpectralBands.SWIR2.value - 1
        ref_swir2 = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_nir = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.RED.value - 1
        ref_red = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        # get image in range 0 - 255 per channel
        shortwave_ir2 = _np.empty(shape=ref_swir2.shape + (3,), dtype=_np.uint8)
        shortwave_ir2[:, :, 0] = (ref_swir2 + 100.) / 16100. * 255.
        shortwave_ir2[:, :, 1] = (ref_nir + 100.) / 16100. * 255.
        shortwave_ir2[:, :, 2] = (ref_red + 100.) / 16100. * 255.

        return shortwave_ir2

    def __getSatelliteImageArray_MODIS(self, img_id: int) -> _np.ndarray:

        if self.satimg_view_opt == SatImgViewOpt.NATURAL_COLOR:
            return self.__getSatelliteImageArray_MODIS_NATURAL_COLOR(img_id=img_id)
        elif self.satimg_view_opt == SatImgViewOpt.CIR:
            return self.__getSatelliteImageArray_MODIS_CIR(img_id=img_id)
        elif self.satimg_view_opt == SatImgViewOpt.EVI:
            return self.__getSatelliteImageArray_MODIS_EVI(img_id=img_id)
        elif self.satimg_view_opt == SatImgViewOpt.EVI2:
            return self.__getSatelliteImageArray_MODIS_EVI2(img_id=img_id)
        elif self.satimg_view_opt == SatImgViewOpt.NDVI:
            return self.__getSatelliteImageArray_NDVI(img_id=img_id)
        elif self.satimg_view_opt == SatImgViewOpt.SHORTWAVE_INFRARED1:
            return self.__getSatelliteImageArray_MODIS_SHORTWAVE_INFRARED_SWIR1(img_id=img_id)
        elif self.satimg_view_opt == SatImgViewOpt.SHORTWAVE_INFRARED2:
            return self.__getSatelliteImageArray_MODIS_SHORTWAVE_INFRARED_SWIR2(img_id=img_id)
        else:
            raise NotImplementedError

    def __getSatelliteImageArray(self, img_id: int) -> _np.ndarray:

        if self.modis_collection == ModisCollection.REFLECTANCE:
            return self.__getSatelliteImageArray_MODIS(img_id)
        else:
            raise NotImplementedError

    def __showSatImage_MODIS(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]],
                             brightness_factors: Union[tuple[float, float], list[float, float]], show: bool = True, ax=None) -> None:
        # lazy imports
        opencv = lazy_import('cv2')

        if not self._satdata_processed:
            try:
                self._processMetaData_SATELLITE_IMG()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to satellite images!')

        if id_img < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        if len(self) - 1 < id_img:
            raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands!'.format(len(self)))

        satimg = self.__getSatelliteImageArray(id_img)
        if brightness_factors is not None and \
                self.satimg_view_opt != SatImgViewOpt.NDVI and \
                self.satimg_view_opt != SatImgViewOpt.EVI and \
                self.satimg_view_opt != SatImgViewOpt.EVI2:

            # increase image brightness
            satimg = opencv.convertScaleAbs(satimg, alpha=brightness_factors[0], beta=brightness_factors[1])

        # figure title
        img_type = self.satimg_view_opt.value
        str_title = 'MODIS ({}, {})'.format(img_type, self.timestamps_reflectance.iloc[id_img]['Date'])

        if self.satimg_view_opt == SatImgViewOpt.NDVI and self.ndvi_view_threshold > -1:
            str_title = '{}, threshold={:.2f})'.format(str_title[:-1], self.ndvi_view_threshold)

        # show labels and binary mask related to localization of wildfires (CCI labels)
        imshow(src=satimg, title=str_title, figsize=figsize, show=show, ax=ax)

    def showSatImage(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                     brightness_factors: Union[tuple[float, float], list[float, float]] = (5., 5.), show: bool = True, ax=None) -> None:

        if self.modis_collection == ModisCollection.REFLECTANCE:
            self.__showSatImage_MODIS(id_img=id_img, figsize=figsize, brightness_factors=brightness_factors, show=show, ax=ax)
        else:
            raise NotImplementedError

    """
    Display functionality (LABELS)
    """

    def __readFireConfidenceLevel_RANGE_CCI(self, id_bands: range) -> (_np.ndarray, _np.ndarray):

        lst_cl = []
        lst_flags = []

        for id_band in id_bands:

            id_ds, id_rs = self._map_layout_firemaps[id_band]

            # get bands
            rs_cl = self._ds_firemaps[id_ds].GetRasterBand(id_rs)
            rs_flags = self._ds_firemaps[id_ds].GetRasterBand(id_rs + 1)

            # get descriptions
            dsc_cl = rs_cl.GetDescription()
            dsc_flags = rs_flags.GetDescription()

            if band2date_firecci(dsc_cl) != band2date_firecci(dsc_flags):
                raise ValueError('Dates between ConfidenceLevel and ObservedFlag bands are not same!')

            lst_flags.append(rs_flags.ReadAsArray())
            lst_cl.append(rs_cl.ReadAsArray())

        np_cl = _np.array(lst_cl)
        del lst_cl; gc.collect()  # invoke garbage collector

        np_cl_agg = _np.max(np_cl, axis=0)
        del np_cl; gc.collect()  # invoke garbage collector

        np_flags = _np.array(lst_flags)
        del lst_flags; gc.collect()  # invoke garbage collector

        np_flags_agg = _np.max(np_flags, axis=0); np_flags_agg[_np.any(np_flags == -1, axis=0)] = -1
        del np_flags; gc.collect()

        return np_cl_agg, np_flags_agg

    def _readFireConfidenceLevel_CCI(self, id_bands: Union[int, range]) -> (_np.ndarray, _np.ndarray):

        # TODO move to loader

        if isinstance(id_bands, range):

            return self.__readFireConfidenceLevel_RANGE_CCI(id_bands)

        elif isinstance(id_bands, int):

            id_ds, id_rs = self._map_layout_firemaps[id_bands]

            # get bands
            rs_cl = self._ds_firemaps[id_ds].GetRasterBand(id_rs)
            rs_flags = self._ds_firemaps[id_ds].GetRasterBand(id_rs + 1)

            # get descriptions
            dsc_cl = rs_cl.GetDescription()
            dsc_flags = rs_flags.GetDescription()

            if band2date_firecci(dsc_cl) != band2date_firecci(dsc_flags):
                raise ValueError('Dates between ConfidenceLevel and ObservedFlag bands are not same!')

            return rs_cl.ReadAsArray(), rs_flags.ReadAsArray()
        else:
            raise NotImplementedError

    def __getFireLabels_CCI(self, id_bands: Union[int, range], with_fire_mask=False, with_uncharted_areas: bool = True) -> \
            Union[_np.ndarray, tuple[_np.ndarray]]:

        # lazy imports
        colors = lazy_import('mlfire.utils.colors')

        rs_cl, rs_flags = self._readFireConfidenceLevel_CCI(id_bands=id_bands)
        mask_fires = None

        label = _np.empty(shape=rs_cl.shape + (3,), dtype=_np.uint8)
        label[:, :] = colors.Colors.GRAY_COLOR.value

        if with_fire_mask or self.labels_view_opt == FireLabelsViewOpt.LABEL: mask_fires = rs_cl >= self.cci_confidence_level

        if self.labels_view_opt == FireLabelsViewOpt.LABEL:

            label[mask_fires, :3] = colors.Colors.RED_COLOR.value

        elif self.labels_view_opt == FireLabelsViewOpt.CONFIDENCE_LEVEL:

            cmap = lazy_import('mlfire.utils.cmap')

            lst_colors = ['#ff0000', '#ff5a00', '#ffff00']
            cl_min = 50
            cl_max = 100

            cmap_helper = cmap.CMapHelper(lst_colors=lst_colors, vmin=cl_min, vmax=cl_max)

            for v in range(self.cci_confidence_level, cl_max + 1):
                c = [int(v * 255) for v in cmap_helper.getRGB(v)]
                label[rs_cl == v, :] = c
        else:
            raise AttributeError('Not supported option for viewing labels!')

        if with_uncharted_areas:
            PIXEL_NOT_OBSERVED = -1
            label[rs_flags == PIXEL_NOT_OBSERVED, :] = 0

        if with_fire_mask:
            return label, mask_fires
        else:
            del mask_fires; gc.collect()
            return label

    def __showFireLabels_CCI(self, id_bands: Union[int, range],  figsize: Union[tuple[float, float], list[float, float]],
                             show_uncharted_areas: bool = True, show: bool = True, ax=None) -> None:

        # lazy imports
        calendar = lazy_import('calendar')

        # put a region name to a plot title
        str_title = 'CCI labels'

        # get date or start/end dates
        if isinstance(id_bands, range):

            first_date = self.timestamps_wildfires.iloc[id_bands[0]]['Date']
            last_date = self.timestamps_wildfires.iloc[id_bands[-1]]['Date']

            if first_date.year == last_date.year:

                str_first_date = f'{calendar.month_name[first_date.month]}'
                str_last_date = f'{calendar.month_name[last_date.month]} {last_date.year}'

            else:

                str_first_date = f'{calendar.month_name[first_date.month]} {first_date.year}'
                str_last_date = f'{calendar.month_name[last_date.month]} {last_date.year}'

            str_date = '{} - {}'.format(str_first_date, str_last_date)

        elif isinstance(id_bands, int):

            label_date = self.timestamps_wildfires.iloc[id_bands]['Date']
            str_date = f'{calendar.month_name[label_date.month]} {label_date.year}'

        else:
            raise NotImplementedError

        # put date to a figure title
        str_title = '{} ({})'.format(str_title, str_date)

        # get fire labels
        labels = self.__getFireLabels_CCI(id_bands=id_bands, with_uncharted_areas=show_uncharted_areas)

        # show labels and binary mask related to localization of wildfires (CCI labels)
        imshow(src=labels, title=str_title, figsize=figsize, show=show, ax=ax)

    def __readFireSeverity_RANGE_MTBS(self, id_bands: range) -> _np.ndarray:

        lst_severity = []

        for id_band in id_bands:

            id_ds, id_rs = self._map_layout_firemaps[id_band]
            rs_severity = self._ds_firemaps[id_ds].GetRasterBand(id_rs).ReadAsArray()

            lst_severity.append(rs_severity)

        np_severity = _np.array(lst_severity)
        del lst_severity; gc.collect()  # invoke garbage collector

        np_severity_agg = _np.max(np_severity, axis=0)
        np_non_mapped = _np.any(np_severity == MTBSSeverity.NON_MAPPED_AREA.value, axis=0)
        np_severity_agg[np_non_mapped] = MTBSSeverity.NON_MAPPED_AREA.value

        # invoke garbage collector
        del np_severity, np_non_mapped
        gc.collect()

        return np_severity_agg

    def _readFireSeverity_MTBS(self, id_bands: Union[int, range]) -> lazy_import('numpy').ndarray:

        if isinstance(id_bands, range):
            return self.__readFireSeverity_RANGE_MTBS(id_bands)
        elif isinstance(id_bands, int):
            id_ds, id_rs = self._map_layout_firemaps[id_bands]
            return self._ds_firemaps[id_ds].GetRasterBand(id_rs).ReadAsArray()
        else:
            raise NotImplementedError

    def __getFireLabels_MTBS(self, id_bands: Union[int, range], with_fire_mask: bool = False, with_uncharted_areas: bool = True) -> \
            Union[_np.ndarray, tuple[_np.ndarray]]:

        # lazy imports
        colors = lazy_import('mlfire.utils.colors')

        # get fire severity
        rs_severity = self._readFireSeverity_MTBS(id_bands=id_bands)
        mask_fires = None

        label = _np.empty(shape=rs_severity.shape + (3,), dtype=_np.uint8)
        label[:, :] = colors.Colors.GRAY_COLOR.value

        if with_fire_mask or self.labels_view_opt == FireLabelsViewOpt.LABEL:

            c1 = rs_severity >= self.mtbs_severity_from.value; c2 = rs_severity <= MTBSSeverity.HIGH.value
            mask_fires = _np.logical_and(c1, c2)

        if self.labels_view_opt == FireLabelsViewOpt.LABEL:

            label[mask_fires, :] = colors.Colors.RED_COLOR.value

        elif self.labels_view_opt == FireLabelsViewOpt.SEVERITY:

            cmap = lazy_import('mlfire.utils.cmap')

            lst_colors = ['#ff0000', '#ff5a00', '#ffff00']
            severity_min = MTBSSeverity.LOW.value
            severity_max = MTBSSeverity.HIGH.value

            cmap_helper = cmap.CMapHelper(lst_colors=lst_colors, vmin=severity_min, vmax=severity_max)
            for v in range(self.mtbs_severity_from.value, severity_max + 1):
                c = [int(v * 255) for v in cmap_helper.getRGB(v)]
                label[rs_severity == v, :] = c

        else:
            raise AttributeError

        if with_uncharted_areas:

            mask_uncharted = _np.array(rs_severity == MTBSSeverity.NON_MAPPED_AREA.value)
            if len(mask_uncharted) != 0: label[mask_uncharted, :] = 0

        # return
        if with_fire_mask:
            return label, mask_fires
        else:
            del mask_fires; gc.collect()
            return label

    def __showFireLabels_MTBS(self, id_bands: Union[int, range], figsize: Union[tuple[float, float], list[float, float]],
                              show_uncharted_areas: bool = True, show: bool = True, ax=None) -> None:

        # put a region name to a plot title
        str_title = 'MTBS labels ({}'.format(self.mtbs_region.name)

        # get date or start/end dates
        if isinstance(id_bands, range):
            first_date = self.timestamps_wildfires.iloc[id_bands[0]]['Date']
            last_date = self.timestamps_wildfires.iloc[id_bands[-1]]['Date']
            str_date = '{}-{}'.format(first_date.year, last_date.year)
        elif isinstance(id_bands, int):
            label_date = self.timestamps_wildfires.iloc[id_bands]['Date']
            str_date = str(label_date.year)
        else:
            raise NotImplementedError

        # put date to a figure title
        str_title = '{} {})'.format(str_title, str_date)

        # get fire labels
        labels = self.__getFireLabels_MTBS(id_bands=id_bands, with_uncharted_areas=show_uncharted_areas)

        # show up labels and binary mask (non-) related to localization of wildfires (MTBS labels)
        imshow(src=labels, title=str_title, figsize=figsize, show=show, ax=ax)

    def showFireLabels(self, id_bands: Union[int, range], figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                       show_uncharted_areas: bool = True, show: bool = True, ax=None) -> None:

        if not self._firemaps_processed:
            # processing descriptions of bands related to fire labels and obtain dates from them
            try:
                self._processMetaData_FIREMAPS()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to labels!')

        if isinstance(id_bands, int):

            if id_bands < 0:
                raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')
            elif self.nbands_label - 1 < id_bands:
                raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(self.nbands_label))

        elif isinstance(id_bands, range):

            for id_band in (id_bands[0], id_bands[-1]):

                if id_band < 0:
                    raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')
                elif self.nbands_label - 1 < id_band:
                    raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(self.nbands_label))

        if self.label_collection == FireLabelCollection.CCI:
            self.__showFireLabels_CCI(id_bands=id_bands, figsize=figsize, show_uncharted_areas=show_uncharted_areas, show=show, ax=ax)
        elif self.label_collection == FireLabelCollection.MTBS:
            self.__showFireLabels_MTBS(id_bands=id_bands, figsize=figsize, show_uncharted_areas=show_uncharted_areas, show=show, ax=ax)
        else:
            raise NotImplementedError

    """
    Display functionality (MULTISPECTRAL SATELLITE IMAGE + LABELS)
    """

    def __getLabelsForSatImg_CCI(self, id_img: int) -> (_np.ndarray, _np.ndarray):

        # lazy import
        datetime = lazy_import('datetime')

        # get label date time
        date_satimg = self.timestamps_reflectance.iloc[id_img]['Date']
        date_label = datetime.date(year=date_satimg.year, month=date_satimg.month, day=1)

        # get index of corresponding labels
        label_index = int(self.timestamps_wildfires.index[self._df_timestamps_firemaps['Date'] == date_label][0])
        label, mask_fires = self.__getFireLabels_CCI(id_bands=label_index, with_fire_mask=True, with_uncharted_areas=False)

        return label, mask_fires

    def __getLabelsForSatImg_MTBS(self, id_img: int) -> (_np.ndarray, _np.ndarray):

        # lazy import
        datetime = lazy_import('datetime')

        date_satimg = self.timestamps_reflectance.iloc[id_img]['Date']
        date_label = datetime.date(year=date_satimg.year, month=1, day=1)

        # get index of corresponding labels
        label_index = int(self._df_timestamps_firemaps.index[self._df_timestamps_firemaps['Date'] == date_label][0])
        label, mask_fires = self.__getFireLabels_MTBS(id_bands=label_index, with_fire_mask=True, with_uncharted_areas=False)

        return label, mask_fires

    def __getLabelsForSatImg(self, id_img: int) -> (_np.ndarray, _np.ndarray):

        if self.label_collection == FireLabelCollection.CCI:
            return self.__getLabelsForSatImg_CCI(id_img)
        elif self.label_collection == FireLabelCollection.MTBS:
            return self.__getLabelsForSatImg_MTBS(id_img)
        else:
            raise NotImplementedError

    def __showSatImageWithFireLabels_MODIS(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                                           brightness_factors: Union[tuple[float, float], list[float, float]] = (5., 5.),
                                           show: bool = True, ax=None) -> None:
        # lazy import
        opencv = lazy_import('cv2')

        # get image numpy array
        satimg = self.__getSatelliteImageArray(id_img)

        # increase image brightness
        if brightness_factors is not None and self.satimg_view_opt != SatImgViewOpt.NDVI:
            # increase image brightness
            satimg = opencv.convertScaleAbs(satimg, alpha=brightness_factors[0], beta=brightness_factors[1])

        # get labels and confidence level/severity
        labels, mask = self.__getLabelsForSatImg(id_img)
        satimg[mask, :] = labels[mask, :]

        ref_date = self.timestamps_reflectance.iloc[id_img]['Date']
        img_type = self.satimg_view_opt.value
        str_title = 'MODIS ({}, {}, labels={})'.format(img_type, ref_date, self.label_collection.name)

        # show multi spectral image as RGB with additional information about localization of wildfires
        imshow(src=satimg, title=str_title, figsize=figsize, show=show, ax=ax)

    def showSatImageWithFireLabels(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                                   brightness_factors: Union[tuple[float, float], list[float, float]] = (5., 5.),
                                   show: bool = True, ax=None) -> None:

        if not self._firemaps_processed:
            # processing descriptions of bands related to fire labels and obtain dates from them
            try:
                self._processMetaData_FIREMAPS()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to labels!')

        if not self._satdata_processed:
            # process descriptions of bands related to satellite images and obtain dates from them
            try:
                self._processMetaData_SATELLITE_IMG()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to satellite images!')

        if self.modis_collection == ModisCollection.REFLECTANCE:
            self.__showSatImageWithFireLabels_MODIS(id_img=id_img, figsize=figsize, brightness_factors=brightness_factors, show=show, ax=ax)
        else:
            raise NotImplementedError


# use case examples
if __name__ == '__main__':

    VAR_DATA_DIR = 'data/tifs'

    VAR_PREFIX_IMG = 'ak_reflec_january_december_{}_100km'
    VAR_PREFIX_LABEL = 'ak_january_december_{}_100km'

    VAR_LABEL_COLLECTION = FireLabelCollection.MTBS
    # VAR_LABEL_COLLECTION = FireLabelCollection.CCI
    VAR_STR_LABEL_COLLECTION = VAR_LABEL_COLLECTION.name.lower()

    VAR_LST_SATIMGS = []
    VAR_LST_LABELS = []

    for year in range(2004, 2006):

        VAR_PREFIX_IMG_YEAR = VAR_PREFIX_IMG.format(year)
        VAR_PRFIX_LABEL_IMG_YEAR = VAR_PREFIX_LABEL.format(year)

        VAR_FN_SATIMG = '{}_epsg3338_area_0.tif'.format(VAR_PREFIX_IMG_YEAR)
        VAR_FN_SATIMG = os.path.join(VAR_DATA_DIR, VAR_FN_SATIMG)
        VAR_LST_SATIMGS.append(VAR_FN_SATIMG)

        VAR_FN_LABELS = '{}_epsg3338_area_0_{}_labels.tif'.format(VAR_PRFIX_LABEL_IMG_YEAR, VAR_STR_LABEL_COLLECTION)
        VAR_FN_LABELS = os.path.join(VAR_DATA_DIR, VAR_FN_LABELS)
        VAR_LST_LABELS.append(VAR_FN_LABELS)

    VAR_SATIMG_VIEW_OPT = SatImgViewOpt.NATURAL_COLOR
    # VAR_SATIMG_VIEW_OPT = SatImgViewOpt.CIR  # uncomment this line for viewing a satellite image in infrared
    # VAR_SATIMG_VIEW_OPT = SatImgViewOpt.NDVI  # uncomment this line for displaying NDVI using information from satellite image
    # VAR_SATIMG_VIEW_OPT = SatImgViewOpt.SHORTWAVE_INFRARED1  # uncomment this line for viewing a satellite image in infrared using SWIR1 band
    # VAR_SATIMG_VIEW_OPT = SatImgViewOpt.SHORTWAVE_INFRARED2  # uncomment this line for viewing a satellite image in infrared using SWIR2 band

    VAR_NDVI_THRESHOLD = 0.5
    VAR_CCI_CONFIDENCE_LEVEL = 70

    # VAR_LABELS_VIEW_OPT = FireLabelsViewOpt.CONFIDENCE_LEVEL if VAR_LABEL_COLLECTION == FireLabelCollection.CCI else FireLabelsViewOpt.SEVERITY
    VAR_LABELS_VIEW_OPT = FireLabelsViewOpt.LABEL  # uncomment this line for viewing fire labels instead of confidence level or severity

    # setup of data set loader
    dataset_view = DatasetView(
        lst_satdata_reflectance=VAR_LST_SATIMGS,
        lst_loc_fires=VAR_LST_LABELS,
        satimg_view_opt=VAR_SATIMG_VIEW_OPT,
        opt_select_firemap=VAR_LABEL_COLLECTION,
        labels_view_opt=VAR_LABELS_VIEW_OPT,
        mtbs_min_severity=MTBSSeverity.LOW,
        cci_confidence_level=VAR_CCI_CONFIDENCE_LEVEL,
        ndvi_view_threshold=VAR_NDVI_THRESHOLD if VAR_SATIMG_VIEW_OPT == SatImgViewOpt.NDVI else None
    )

    print('#ts = {}'.format(len(dataset_view)))
    print(dataset_view.timestamps_reflectance)
    print(dataset_view.timestamps_wildfires)

    dataset_view.showFireLabels(18 if VAR_LABEL_COLLECTION == FireLabelCollection.CCI else 1)
    dataset_view.showSatImage(70)
    dataset_view.showSatImageWithFireLabels(70)

    # labels aggregation
    dataset_view.showFireLabels(18 if VAR_LABEL_COLLECTION == FireLabelCollection.CCI else 0)
    dataset_view.showFireLabels(19 if VAR_LABEL_COLLECTION == FireLabelCollection.CCI else 1)
    dataset_view.showFireLabels(range(18, 20) if VAR_LABEL_COLLECTION == FireLabelCollection.CCI else range(0, 2))

    # view evi and evi2
    dataset_view.satimg_view_opt = SatImgViewOpt.EVI2
    dataset_view.showSatImage(70, show=True)
