
import gc
import datetime

from enum import Enum
from typing import Union

from mlfire.data.loader import SatDataLoader, SatDataSelectOpt, FireMapSelectOpt
from mlfire.earthengine.collections import ModisReflectanceSpectralBands
from mlfire.earthengine.collections import MTBSSeverity, MTBSRegion

# utils imports
from mlfire.utils.functool import lazy_import

from mlfire.utils.cmap import CMapHelper
from mlfire.utils.plots import imshow

# lazy imports
_np = lazy_import('numpy')
_opencv = lazy_import('cv2')


class SatImgViewOpt(Enum):   # TODO rename -> SatDataViewOpt

    CIR = 'Color Infrared, Vegetation'
    EVI = 'EVI'
    EVI2 = 'EVI2'
    NATURAL_COLOR = 'Natural Color'
    NDVI = 'NVDI'
    SHORTWAVE_INFRARED1 = 'Shortwave Infrared using SWIR1'
    SHORTWAVE_INFRARED2 = 'Shortwave Infrared using SWIR2'
    TEMPERATURE = 'Land surface temperature'

    def __str__(self) -> str:
        return self.value


class FireMapsViewOpt(Enum):  # TODO rename -> FireMapViewOpt

    LABEL = 1
    CONFIDENCE_LEVEL = 2
    SEVERITY = 3


class SatDataView(SatDataLoader):

    def __init__(self,
                 lst_firemaps: Union[tuple[str], list[str], None],
                 lst_satdata_reflectance: Union[tuple[str], list[str], None] = None,
                 lst_satdata_temperature: Union[tuple[str], list[str], None] = None,
                 opt_select_satdata: SatDataSelectOpt = SatDataSelectOpt.ALL,
                 opt_select_firemap: FireMapSelectOpt = FireMapSelectOpt.MTBS,
                 # TODO comment
                 cci_confidence_level: int = 70,
                 # TODO comment
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA,
                 mtbs_min_severity: MTBSSeverity = MTBSSeverity.LOW,
                 # TODO comment
                 ndvi_view_threshold: Union[float, None] = None,
                 # TODO comment
                 view_opt_satdata: SatImgViewOpt = SatImgViewOpt.NATURAL_COLOR,
                 view_opt_firemap: FireMapsViewOpt = FireMapsViewOpt.LABEL,
                 # TODO comment
                 estimate_time: bool = False) -> None:

        super().__init__(
            lst_firemaps=lst_firemaps,
            lst_satdata_reflectance=lst_satdata_reflectance,
            lst_satdata_temperature=lst_satdata_temperature,
            opt_select_satdata=opt_select_satdata,
            opt_select_firemap=opt_select_firemap,
            cci_confidence_level=cci_confidence_level,
            mtbs_region=mtbs_region,
            mtbs_min_severity=mtbs_min_severity,
            estimate_time=estimate_time
        )

        self.__view_opt_satdata = None
        self.view_opt_satdata = view_opt_satdata

        self.__view_opt_firemap = None
        self.view_opt_firemap = view_opt_firemap

        self._ndvi_view_thrs = None
        self.ndvi_view_threshold = ndvi_view_threshold

    @property
    def view_opt_satdata(self) -> SatImgViewOpt:

        return self.__view_opt_satdata

    @view_opt_satdata.setter
    def view_opt_satdata(self, opt: SatImgViewOpt) -> None:

        if self.__view_opt_satdata == opt: return
        self.__view_opt_satdata = opt

    @property
    def view_opt_firemap(self) -> FireMapsViewOpt:

        return self.__view_opt_firemap

    @view_opt_firemap.setter
    def view_opt_firemap(self, opt: FireMapsViewOpt) -> None:

        # TODO check input

        if self.__view_opt_firemap == opt: return
        self.__view_opt_firemap = opt

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

        # TODO check input argument

        if len(self._ds_satdata_reflectance) > 1:  # TODO is there another value
            id_ds, start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[id_ds]
        else:
            start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[0]

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

        # TODO check input argument

        if len(self._ds_satdata_reflectance) > 1:  # TODO is there another value
            id_ds, start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[id_ds]
        else:
            start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[0]

        band_id = start_band_id + ModisReflectanceSpectralBands.BLUE.value - 1
        ref_blue = ds_satimg.GetRasterBand(band_id).ReadAsArray() / 1e4

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_nir = ds_satimg.GetRasterBand(band_id).ReadAsArray() / 1e4

        band_id = start_band_id + ModisReflectanceSpectralBands.RED.value - 1
        ref_red = ds_satimg.GetRasterBand(band_id).ReadAsArray() / 1e4

        _np.seterr(divide='ignore', invalid='ignore')

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
        cmap_helper = CMapHelper(lst_colors=lst_colors, vmin=.2, vmax=.8)  # TODO fix argument
        img_evi = cmap_helper.getRGBA(evi)[:, :, :-1]
        img_evi[evi == -99, :] = [0, 0, 0]

        return img_evi

    def __getSatelliteImageArray_MODIS_EVI2(self, img_id: int):

        # TODO check input argument

        if len(self._ds_satdata_reflectance) > 1:  # TODO is there another value
            id_ds, start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[id_ds]
        else:
            start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[0]

        ref_red = ds_satimg.GetRasterBand(start_band_id).ReadAsArray() / 1e4

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_nir = ds_satimg.GetRasterBand(band_id).ReadAsArray() / 1e4

        _np.seterr(divide='ignore', invalid='ignore')

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
        cmap_helper = CMapHelper(lst_colors=lst_colors, vmin=.2, vmax=.8)
        img_evi2 = cmap_helper.getRGBA(evi2)[:, :, :-1]
        img_evi2[evi2 == -99, :] = [0, 0, 0]

        return img_evi2

    def __getSatelliteImageArray_MODIS_NATURAL_COLOR(self, img_id: int) -> _np.ndarray:

        if len(self._ds_satdata_reflectance) > 1:  # TODO is there another value
            id_ds, start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[id_ds]
        else:
            start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[0]

        ref_red = ds_satimg.GetRasterBand(start_band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.GREEN.value - 1
        ref_green = ds_satimg.GetRasterBand(band_id).ReadAsArray()

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
        plt = lazy_import('matplotlib.pylab')  # TODO remove

        if len(self._ds_satdata_reflectance) > 1:  # TODO is there another value
            id_ds, start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[id_ds]
        else:
            start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[0]

        # computing Normalized Difference Vegetation Index (NDVI)
        ref_red = ds_satimg.GetRasterBand(start_band_id).ReadAsArray() / 1e4

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_nir = ds_satimg.GetRasterBand(band_id).ReadAsArray() / 1e4

        # Colour palette and thresholding inspired by
        # https://www.neonscience.org/resources/learning-hub/tutorials/calc-ndvi-tiles-py
        # Credit: Zachary Langford @langfordzl
        cmap = plt.get_cmap(name='RdYlGn') if self.ndvi_view_threshold > -1. else plt.get_cmap(name='seismic')

        _np.seterr(divide='ignore', invalid='ignore')

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

        if len(self._ds_satdata_reflectance) > 1:  # TODO is there another value
            id_ds, start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[id_ds]
        else:
            start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[0]

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

        if len(self._ds_satdata_reflectance) > 1:  # TODO is there another value
            id_ds, start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[id_ds]
        else:
            start_band_id = self._layout_layers_reflectance[img_id]
            ds_satimg = self._ds_satdata_reflectance[0]

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

        if self.view_opt_satdata == SatImgViewOpt.NATURAL_COLOR:
            return self.__getSatelliteImageArray_MODIS_NATURAL_COLOR(img_id=img_id)
        elif self.view_opt_satdata == SatImgViewOpt.CIR:
            return self.__getSatelliteImageArray_MODIS_CIR(img_id=img_id)
        elif self.view_opt_satdata == SatImgViewOpt.EVI:
            return self.__getSatelliteImageArray_MODIS_EVI(img_id=img_id)
        elif self.view_opt_satdata == SatImgViewOpt.EVI2:
            return self.__getSatelliteImageArray_MODIS_EVI2(img_id=img_id)
        elif self.view_opt_satdata == SatImgViewOpt.NDVI:
            return self.__getSatelliteImageArray_NDVI(img_id=img_id)
        elif self.view_opt_satdata == SatImgViewOpt.SHORTWAVE_INFRARED1:
            return self.__getSatelliteImageArray_MODIS_SHORTWAVE_INFRARED_SWIR1(img_id=img_id)
        elif self.view_opt_satdata == SatImgViewOpt.SHORTWAVE_INFRARED2:
            return self.__getSatelliteImageArray_MODIS_SHORTWAVE_INFRARED_SWIR2(img_id=img_id)
        else:
            raise NotImplementedError

    def __getSatelliteImageArray(self, img_id: int) -> _np.ndarray:

        # TODO fix
        # if self.modis_collection == ModisCollection.REFLECTANCE:
        return self.__getSatelliteImageArray_MODIS(img_id)
        # else:
        #    raise NotImplementedError

    def __showSatImage_MODIS(self, id_img: int, figsize: Union[tuple[float, ...], list[float, ...]],
                             brightness_factors: Union[tuple[float, ...], list[float, ...]], show: bool = True, ax=None) -> None:
        # lazy imports
        opencv = lazy_import('cv2')  # TODO to preamble

        if not self._satdata_processed:
            try:
                self._processMetadata_SATDATA()  # TODO change to right method
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to satellite images!')

        if id_img < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        # TODO fix
        # if len(self) - 1 < id_img:
        #     raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands!'.format(len(self)))

        satimg = self.__getSatelliteImageArray(id_img)
        if brightness_factors is not None and \
                self.view_opt_satdata != SatImgViewOpt.NDVI and \
                self.view_opt_satdata != SatImgViewOpt.EVI and \
                self.view_opt_satdata != SatImgViewOpt.EVI2:

            # increase image brightness
            satimg = opencv.convertScaleAbs(satimg, alpha=brightness_factors[0], beta=brightness_factors[1])

        # figure title
        img_type = self.view_opt_satdata.value
        str_title = 'MODIS ({}, {})'.format(img_type, self.timestamps_reflectance.iloc[id_img]['Timestamps'])

        if self.view_opt_satdata == SatImgViewOpt.NDVI and self.ndvi_view_threshold > -1:
            str_title = '{}, threshold={:.2f})'.format(str_title[:-1], self.ndvi_view_threshold)

        # show a fire map and binary mask related to localization of wildfires (CCI collection)
        imshow(src=satimg, title=str_title, figsize=figsize, show=show, ax=ax)

    def showSatData(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                    brightness_factors: Union[tuple[float, float], list[float, float]] = (5., 5.), show: bool = True, ax=None) -> None:

        # if self.modis_collection == ModisCollection.REFLECTANCE:
        self.__showSatImage_MODIS(id_img=id_img, figsize=figsize, brightness_factors=brightness_factors, show=show, ax=ax)  # TODO reflectance
        # else:
        #    raise NotImplementedError

    """
    Display functionality (firemap)
    """

    def __getFireMap_CCI(self, id_bands: Union[int, range], with_fire_mask=False, with_uncharted_areas: bool = True) -> \
            Union[_np.ndarray, tuple[_np.ndarray]]:  # TODO rename that reflect the result is for visualization

        # lazy imports
        colors = lazy_import('mlfire.utils.colors')  # TODO move to preable

        rs_cl = self._processConfidenceLevel_CCI(rs_ids=id_bands)   #, rs_flags = self._readFireConfidenceLevel_CCI(id_bands=id_bands)
        mask_fires = None

        label = _np.empty(shape=rs_cl.shape + (3,), dtype=_np.uint8)
        label[:, :] = colors.Colors.GRAY_COLOR.value

        if with_fire_mask or self.view_opt_firemap == FireMapsViewOpt.LABEL: mask_fires = rs_cl >= self.cci_confidence_level

        if self.view_opt_firemap == FireMapsViewOpt.LABEL:

            label[mask_fires, :3] = colors.Colors.RED_COLOR.value

        elif self.view_opt_firemap == FireMapsViewOpt.CONFIDENCE_LEVEL:

            cmap = lazy_import('mlfire.utils.cmap')

            lst_colors = ['#ff0000', '#ff5a00', '#ffff00']
            cl_min = 50
            cl_max = 100

            cmap_helper = cmap.CMapHelper(lst_colors=lst_colors, vmin=cl_min, vmax=cl_max)

            for v in range(self.cci_confidence_level, cl_max + 1):
                c = [int(v * 255) for v in cmap_helper.getRGB(v)]
                label[rs_cl == v, :] = c
        else:
            raise AttributeError('Not supported option for viewing firemaps!')

        if with_uncharted_areas:
            PIXEL_NOT_OBSERVED = -1
            label[rs_cl == PIXEL_NOT_OBSERVED, :] = 0

        if with_fire_mask:
            return label, mask_fires
        else:
            del mask_fires; gc.collect()
            return label

    def __showFireMap_CCI(self, id_bands: Union[int, range], figsize: Union[tuple[float, float], list[float, float]],
                          show_uncharted_areas: bool = True, show: bool = True, ax=None) -> None:

        # lazy imports
        calendar = lazy_import('calendar')  # TODO move to begining of script

        # put a region name to a plot title
        str_title = 'CCI firemaps'

        # get date or start/end dates
        if isinstance(id_bands, range):

            first_date = self.timestamps_firemaps.iloc[id_bands[0]]['Timestamps']
            last_date = self.timestamps_firemaps.iloc[id_bands[-1]]['Timestamps']

            if first_date.year == last_date.year:
                str_first_date = f'{calendar.month_name[first_date.month]}'
                str_last_date = f'{calendar.month_name[last_date.month]} {last_date.year}'
            else:
                str_first_date = f'{calendar.month_name[first_date.month]} {first_date.year}'
                str_last_date = f'{calendar.month_name[last_date.month]} {last_date.year}'

            str_date = '{} - {}'.format(str_first_date, str_last_date)

        elif isinstance(id_bands, int):

            label_date = self.timestamps_firemaps.iloc[id_bands]['Timestamps']
            str_date = f'{calendar.month_name[label_date.month]} {label_date.year}'

        else:
            raise NotImplementedError

        # put date to a figure title
        str_title = f'{str_title} ({str_date})'

        # get fire map
        labels = self.__getFireMap_CCI(id_bands=id_bands, with_uncharted_areas=show_uncharted_areas)

        # show fire map and binary mask related to localization of wildfires (CCI collection)
        imshow(src=labels, title=str_title, figsize=figsize, show=show, ax=ax)

    def __getFireMap_MTBS(self, id_bands: Union[int, range], with_fire_mask: bool = False, with_uncharted_areas: bool = True) -> \
            Union[_np.ndarray, tuple[_np.ndarray]]:

        # lazy imports
        colors = lazy_import('mlfire.utils.colors')  # TODO move to beginning of script

        # get fire severity
        rs_severity = self._processSeverity_MTBS(rs_ids=id_bands)
        mask_fires = None

        label = _np.empty(shape=rs_severity.shape + (3,), dtype=_np.uint8)
        label[:, :] = colors.Colors.GRAY_COLOR.value

        if with_fire_mask or self.view_opt_firemap == FireMapsViewOpt.LABEL:
            c1 = rs_severity >= self.mtbs_min_severity.value; c2 = rs_severity <= MTBSSeverity.HIGH.value  # TODO rewrite
            mask_fires = _np.logical_and(c1, c2)

        if self.view_opt_firemap == FireMapsViewOpt.LABEL:
            label[mask_fires, :] = colors.Colors.RED_COLOR.value
        elif self.view_opt_firemap == FireMapsViewOpt.SEVERITY:
            cmap = lazy_import('mlfire.utils.cmap')

            lst_colors = ['#ff0000', '#ff5a00', '#ffff00']
            severity_min = MTBSSeverity.LOW.value
            severity_max = MTBSSeverity.HIGH.value

            cmap_helper = cmap.CMapHelper(lst_colors=lst_colors, vmin=severity_min, vmax=severity_max)
            for v in range(self.mtbs_min_severity.value, severity_max + 1):
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

    def __showFireMap_MTBS(self, id_bands: Union[int, range], figsize: Union[tuple[float, float], list[float, float]],
                           show_uncharted_areas: bool = True, show: bool = True, ax=None) -> None:

        # get date or range of dates
        if isinstance(id_bands, range):
            first_date = self.timestamps_firemaps.iloc[id_bands[0]]['Timestamps']
            last_date = self.timestamps_firemaps.iloc[id_bands[-1]]['Timestamps']
            str_date = '{}-{}'.format(first_date.year, last_date.year)
        elif isinstance(id_bands, int):
            label_date = self.timestamps_firemaps.iloc[id_bands]['Timestamps']
            str_date = str(label_date.year)
        else:
            raise NotImplementedError

        # put date to a figure title
        str_title = f'MTBS firemap ({self.mtbs_region}, {str_date})'

        # get a fire map
        firemap = self.__getFireMap_MTBS(id_bands=id_bands, with_uncharted_areas=show_uncharted_areas)

        # show up fire map and binary mask (non-) related to localization of wildfires (MTBS collection)
        imshow(src=firemap, title=str_title, figsize=figsize, show=show, ax=ax)

    def showFireMap(self, id_bands: Union[int, range], figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                    show_uncharted_areas: bool = True, show: bool = True, ax=None) -> None:

        # TODO check input arguments

        # TODO comment
        len_firemaps = len(self.timestamps_firemaps)

        if isinstance(id_bands, int):
            if id_bands < 0:
                raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')
            elif len_firemaps - 1 < id_bands:
                raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(len_firemaps))
        elif isinstance(id_bands, range):
            for id_band in (id_bands[0], id_bands[-1]):
                if id_band < 0:
                    raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')
                elif len_firemaps - 1 < id_band:
                    raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(len_firemaps))

        if self.opt_select_firemap == FireMapSelectOpt.CCI:
            self.__showFireMap_CCI(id_bands=id_bands, figsize=figsize, show_uncharted_areas=show_uncharted_areas, show=show, ax=ax)
        elif self.opt_select_firemap == FireMapSelectOpt.MTBS:
            self.__showFireMap_MTBS(id_bands=id_bands, figsize=figsize, show_uncharted_areas=show_uncharted_areas, show=show, ax=ax)
        else:
            raise NotImplementedError

    """
    TODO rename
    Display functionality (MULTISPECTRAL SATELLITE IMAGE + LABELS)
    """

    def __getFireMapForSatImg_CCI(self, id_img: int) -> tuple[_np.ndarray, ...]:

        # TODO check input argument

        # get a fire map timestamp (CCI)
        date_satdata = self.timestamps_reflectance.iloc[id_img]['Timestamps']
        date_firemap = datetime.date(year=date_satdata.year, month=date_satdata.month, day=1)

        # get index of corresponding fire map
        cnd = self._df_timestamps_firemaps['Timestamps'] == date_firemap
        firemap_index = int(self.timestamps_firemaps.index[cnd][0])

        firemap, mask = self.__getFireMap_CCI(
            id_bands=firemap_index, with_fire_mask=True, with_uncharted_areas=False
        )

        return firemap, mask

    def __getFireMapForSatImg_MTBS(self, id_img: int) -> tuple[_np.ndarray, ...]:

        # TODO check input argument

        # get a fire map timestamp (MTBS)
        date_satdata = self.timestamps_reflectance.iloc[id_img]['Timestamps']
        date_firemap = datetime.date(year=date_satdata.year, month=1, day=1)

        # get index of corresponding fire map
        cnd = self._df_timestamps_firemaps['Timestamps'] == date_firemap
        firemap_index = int(self._df_timestamps_firemaps.index[cnd][0])

        firemap, mask = self.__getFireMap_MTBS(
            id_bands=firemap_index, with_fire_mask=True, with_uncharted_areas=False
        )

        return firemap, mask

    def __getFireMapForSatImg(self, id_img: int) -> tuple[_np.ndarray, ...]:

        # TODO check input argument

        if self.opt_select_firemap == FireMapSelectOpt.CCI:
            return self.__getFireMapForSatImg_CCI(id_img)
        elif self.opt_select_firemap == FireMapSelectOpt.MTBS:
            return self.__getFireMapForSatImg_MTBS(id_img)
        else:
            raise NotImplementedError

    def __showSatDataWithFireMap(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                                 brightness_factors: Union[tuple[float, float], list[float, float]] = (5., 5.),
                                 show: bool = True, ax=None) -> None:

        # TODO check input arguments

        # get image numpy array
        satimg = self.__getSatelliteImageArray(id_img)

        # increase image brightness
        if brightness_factors is not None and self.view_opt_satdata != SatImgViewOpt.NDVI:
            # increase image brightness
            satimg = _opencv.convertScaleAbs(satimg, alpha=brightness_factors[0], beta=brightness_factors[1])

        # get fire map and confidence level/severity
        labels, mask = self.__getFireMapForSatImg(id_img)
        satimg[mask, :] = labels[mask, :]

        ref_date = self.timestamps_reflectance.iloc[id_img]['Timestamps']
        str_title = f'MOD09A1 ({self.view_opt_satdata}, {ref_date}, firemap={self.opt_select_firemap})'

        # show multi spectral image as RGB with additional information about localization of wildfires
        imshow(src=satimg, title=str_title, figsize=figsize, show=show, ax=ax)

    def showSatDataWithFireMap(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                               brightness_factors: Union[tuple[float, float], list[float, float]] = (5., 5.),
                               show: bool = True, ax=None) -> None:

        # TODO check input arguments

        if self.view_opt_satdata == SatImgViewOpt.TEMPERATURE:
            raise NotImplementedError
        else:
            self.__showSatDataWithFireMap(
                id_img=id_img, figsize=figsize, brightness_factors=brightness_factors, show=show, ax=ax
            )


# use case examples
if __name__ == '__main__':

    _os = lazy_import('os')

    VAR_DATA_DIR = 'data/tifs'

    VAR_PREFIX_IMG_REFLECTANCE = 'ak_reflec_january_december_{}_100km'
    VAR_PREFIX_IMG_TEMPERATURE = 'ak_lst_january_december_{}_100km'
    VAR_PREFIX_IMG_FIREMAPS = 'ak_january_december_{}_100km'

    VAR_LST_REFLECTANCE = []
    VAR_LST_TEMPERATURE = []
    VAR_LST_FIREMAPS = []

    for year in range(2004, 2006):
        VAR_PREFIX_IMG_REFLECTANCE_YEAR = VAR_PREFIX_IMG_REFLECTANCE.format(year)
        VAR_PREFIX_IMG_TEMPERATURE_YEAR = VAR_PREFIX_IMG_TEMPERATURE.format(year)

        VAR_PREFIX_IMG_FIREMAPS_YEAR = VAR_PREFIX_IMG_FIREMAPS.format(year)

        fn_satimg_reflec = f'{VAR_PREFIX_IMG_REFLECTANCE_YEAR}_epsg3338_area_0.tif'
        fn_satimg_reflec = _os.path.join(VAR_DATA_DIR, fn_satimg_reflec)
        VAR_LST_REFLECTANCE.append(fn_satimg_reflec)

        fn_satimg_temperature = f'{VAR_PREFIX_IMG_TEMPERATURE_YEAR}_epsg3338_area_0.tif'
        fn_satimg_temperature = _os.path.join(VAR_DATA_DIR, fn_satimg_temperature)
        VAR_LST_TEMPERATURE.append(fn_satimg_temperature)

        fn_labels_mtbs = '{}_epsg3338_area_0_mtbs_labels.tif'.format(VAR_PREFIX_IMG_FIREMAPS_YEAR)
        fn_labels_mtbs = _os.path.join(VAR_DATA_DIR, fn_labels_mtbs)
        VAR_LST_FIREMAPS.append(fn_labels_mtbs)

    VAR_OPT_SELECT_FIREMAP = FireMapSelectOpt.MTBS
    # VAR_LABEL_COLLECTION = FireMapSelectOpt.CCI

    VAR_SATDATA_VIEW_OPT = SatImgViewOpt.NATURAL_COLOR
    # VAR_SATIMG_VIEW_OPT = SatImgViewOpt.CIR  # uncomment this line for viewing a satellite image in infrared
    # VAR_SATIMG_VIEW_OPT = SatImgViewOpt.NDVI  # uncomment this line for displaying NDVI using information from satellite image
    # VAR_SATIMG_VIEW_OPT = SatImgViewOpt.SHORTWAVE_INFRARED1  # uncomment this line for viewing a satellite image in infrared using SWIR1 band
    # VAR_SATIMG_VIEW_OPT = SatImgViewOpt.SHORTWAVE_INFRARED2  # uncomment this line for viewing a satellite image in infrared using SWIR2 band

    VAR_NDVI_THRESHOLD = 0.5

    VAR_FIREMAP_VIEW_OPT = FireMapsViewOpt.CONFIDENCE_LEVEL if VAR_OPT_SELECT_FIREMAP == FireMapSelectOpt.CCI else FireMapsViewOpt.SEVERITY
    # VAR_FIREMAP_VIEW_OPT = FireMapsViewOpt.LABEL  # uncomment this line for viewing label instead of confidence level or severity

    VAR_CCI_CONFIDENCE_LEVEL = 70
    VAR_MTBS_MIN_SEVERITY = MTBSSeverity.LOW

    dataset_view = SatDataView(
        lst_firemaps=VAR_LST_FIREMAPS,
        lst_satdata_reflectance=VAR_LST_REFLECTANCE,
        lst_satdata_temperature=VAR_LST_TEMPERATURE,
        # selection of modis collection
        opt_select_satdata=SatDataSelectOpt.TEMPERATURE,  # TODO as variable
        # fire map collection
        opt_select_firemap=VAR_OPT_SELECT_FIREMAP,
        # TODO comment
        mtbs_min_severity=VAR_MTBS_MIN_SEVERITY if VAR_OPT_SELECT_FIREMAP == FireMapSelectOpt.MTBS else None,
        cci_confidence_level=VAR_CCI_CONFIDENCE_LEVEL if VAR_OPT_SELECT_FIREMAP == FireMapSelectOpt.CCI else None,
        #
        view_opt_satdata=VAR_SATDATA_VIEW_OPT,
        view_opt_firemap=VAR_FIREMAP_VIEW_OPT,
        # TODO comment
        ndvi_view_threshold=VAR_NDVI_THRESHOLD if VAR_SATDATA_VIEW_OPT == SatImgViewOpt.NDVI else None,
        estimate_time=True
    )

    #
    # print('#ts = {}'.format(dataset_view.len_ts_satdata))
    # print('#firemaps = {}'.format(dataset_view.len_ts_firemaps))

    print(dataset_view.timestamps_satdata)
    print(dataset_view.timestamps_firemaps)

    dataset_view.showFireMap(18 if VAR_OPT_SELECT_FIREMAP == FireMapSelectOpt.CCI else 1)
    dataset_view.view_opt_satdata = VAR_SATDATA_VIEW_OPT

    dataset_view.showSatData(70)
    dataset_view.showSatDataWithFireMap(70)

    # fire maps aggregation
    dataset_view.showFireMap(18 if VAR_OPT_SELECT_FIREMAP == FireMapSelectOpt.CCI else 0)
    dataset_view.showFireMap(19 if VAR_OPT_SELECT_FIREMAP == FireMapSelectOpt.CCI else 1)
    dataset_view.showFireMap(range(18, 20) if VAR_OPT_SELECT_FIREMAP == FireMapSelectOpt.CCI else range(0, 2))

    # view evi and evi2
    dataset_view.view_opt_satdata = SatImgViewOpt.EVI
    dataset_view.showSatData(70, show=True)

    dataset_view.view_opt_satdata = SatImgViewOpt.EVI2
    dataset_view.showSatData(70, show=True)
