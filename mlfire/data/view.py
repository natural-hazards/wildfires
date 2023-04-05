import gc
import os

from enum import Enum
from typing import Union

from mlfire.data.loader import DatasetLoader
from mlfire.earthengine.collections import ModisIndex, ModisReflectanceSpectralBands
from mlfire.earthengine.collections import FireLabelsCollection
from mlfire.earthengine.collections import MTBSSeverity, MTBSRegion

# utils imports
from mlfire.utils.functool import lazy_import
from mlfire.utils.utils_string import band2date_firecci
from mlfire.utils.plots import imshow


class SatImgViewOpt(Enum):

    CIR = 'Color Infrared, Vegetation'
    NATURAL_COLOR = 'Natural Color'
    NDVI = 'NVDI'
    SHORTWAVE_INFRARED1 = 'Shortwave Infrared using SWIR1'
    SHORTWAVE_INFRARED2 = 'Shortwave Infrared using SWIR2'


class FireLabelsViewOpt(Enum):

    LABEL = 1
    CONFIDENCE_LEVEL = 2
    SEVERITY = 3


class DatasetView(DatasetLoader):

    def __init__(self,
                 lst_satimgs: Union[tuple[str], list[str]],
                 lst_labels: Union[tuple[str], list[str]],
                 modis_collection: ModisIndex = ModisIndex.REFLECTANCE,
                 label_collection: FireLabelsCollection = FireLabelsCollection.MTBS,
                 cci_confidence_level: int = 70,
                 mtbs_severity_from: MTBSSeverity = MTBSSeverity.LOW,
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA,
                 satimg_view_opt: SatImgViewOpt = SatImgViewOpt.NATURAL_COLOR,
                 labels_view_opt: FireLabelsViewOpt = FireLabelsViewOpt.LABEL) -> None:

        super().__init__(
            lst_satimgs=lst_satimgs,
            lst_labels=lst_labels,
            modis_collection=modis_collection,
            label_collection=label_collection,
            cci_confidence_level=cci_confidence_level,
            mtbs_severity_from=mtbs_severity_from,
            mtbs_region=mtbs_region
        )

        self._satimg_view_opt = None
        self.satimg_view_opt = satimg_view_opt

        self._labels_view_opt = None
        self.labels_view_opt = labels_view_opt

    @property
    def satimg_view_opt(self) -> SatImgViewOpt:

        return self._satimg_view_opt

    @satimg_view_opt.setter
    def satimg_view_opt(self, opt: SatImgViewOpt) -> None:

        if self._satimg_view_opt == opt:
            return

        self._satimg_view_opt = opt

    @property
    def labels_view_opt(self) -> FireLabelsViewOpt:

        return self._labels_view_opt

    @labels_view_opt.setter
    def labels_view_opt(self, opt: FireLabelsViewOpt) -> None:

        if self._labels_view_opt == opt:
            return

        self._labels_view_opt = opt

    """
    Display functionality (MULTISPECTRAL SATELLITE IMAGE, MODIS)
    """

    def __getSatelliteImageArray_MODIS_CIR(self, img_id: int) -> lazy_import('numpy').ndarray:

        np = lazy_import('numpy')

        id_ds, start_band_id = self._map_start_satimgs[img_id]
        ds_satimg = self._ds_satimgs[id_ds]

        # display as CIR image (color infrared - vegetation)
        # https://eos.com/make-an-analysis/color-infrared/
        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_red = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.RED.value - 1
        ref_green = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.GREEN.value - 1
        ref_blue = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        # get image in range 0 - 255 per channel
        cir_img = np.empty(shape=ref_red.shape + (3,), dtype=np.uint8)
        cir_img[:, :, 0] = (ref_blue + 100.) / 16100. * 255.
        cir_img[:, :, 1] = (ref_green + 100.) / 16100. * 255.
        cir_img[:, :, 2] = (ref_red + 100.) / 16100. * 255.

        return cir_img

    def __getSatelliteImageArray_MODIS_NATURAL_COLOR(self, img_id: int) -> lazy_import('numpy').ndarray:

        np = lazy_import('numpy')

        id_ds, start_band_id = self._map_start_satimgs[img_id]
        ds_satimg = self._ds_satimgs[id_ds]

        ref_red = ds_satimg.GetRasterBand(start_band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.GREEN.value - 1
        ref_green = self._ds_satimgs[id_ds].GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.BLUE.value - 1
        ref_blue = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        # get image in range 0 - 255 per channel
        img = np.empty(shape=ref_red.shape + (3,), dtype=np.uint8)
        img[:, :, 0] = (ref_blue + 100.) / 16100. * 255.
        img[:, :, 1] = (ref_green + 100.) / 16100. * 255.
        img[:, :, 2] = (ref_red + 100.) / 16100. * 255.

        return img

    def __getSatelliteImageArray_NDVI(self, img_id: int) -> lazy_import('numpy').ndarray:

        np = lazy_import('numpy')
        mpl = lazy_import('matplotlib')
        plt = lazy_import('matplotlib.pylab')

        id_ds, start_band_id = self._map_start_satimgs[img_id]
        ds_satimg = self._ds_satimgs[id_ds]

        # computing Normalized Difference Vegetation Index (NDVI)
        ref_red = ds_satimg.GetRasterBand(start_band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_nir = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        cmap = plt.get_cmap(name='RdYlGn')
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)

        ndvi = (ref_nir - ref_red) / (ref_nir + ref_red)
        img_ndvi = np.uint8(cmap(norm(ndvi))[:, :, :-1] * 255)[:, :, ::-1]

        return img_ndvi

    def __getSatelliteImageArray_MODIS_SHORTWAVE_INFRARED_SWIR1(self, img_id: int) -> lazy_import('numpy').ndarray:

        np = lazy_import('numpy')

        id_ds, start_band_id = self._map_start_satimgs[img_id]
        ds_satimg = self._ds_satimgs[id_ds]

        # https://eos.com/make-an-analysis/vegetation-analysis/
        band_id = start_band_id + ModisReflectanceSpectralBands.SWIR1.value - 1
        ref_red = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_green = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.RED.value - 1
        ref_blue = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        # get image in range 0 - 255 per channel
        shortwave_ir1 = np.empty(shape=ref_red.shape + (3,), dtype=np.uint8)
        shortwave_ir1[:, :, 0] = (ref_blue + 100.) / 16100. * 255.
        shortwave_ir1[:, :, 1] = (ref_green + 100.) / 16100. * 255.
        shortwave_ir1[:, :, 2] = (ref_red + 100.) / 16100. * 255.

        return shortwave_ir1

    def __getSatelliteImageArray_MODIS_SHORTWAVE_INFRARED_SWIR2(self, img_id: int) -> lazy_import('numpy').ndarray:

        np = lazy_import('numpy')

        id_ds, start_band_id = self._map_start_satimgs[img_id]
        ds_satimg = self._ds_satimgs[id_ds]

        # https://eos.com/make-an-analysis/shortwave-infrared/
        band_id = start_band_id + ModisReflectanceSpectralBands.SWIR2.value - 1
        ref_red = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.NIR.value - 1
        ref_green = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        band_id = start_band_id + ModisReflectanceSpectralBands.RED.value - 1
        ref_blue = ds_satimg.GetRasterBand(band_id).ReadAsArray()

        # get image in range 0 - 255 per channel
        shortwave_ir2 = np.empty(shape=ref_red.shape + (3,), dtype=np.uint8)
        shortwave_ir2[:, :, 0] = (ref_blue + 100.) / 16100. * 255.
        shortwave_ir2[:, :, 1] = (ref_green + 100.) / 16100. * 255.
        shortwave_ir2[:, :, 2] = (ref_red + 100.) / 16100. * 255.

        return shortwave_ir2

    def __getSatelliteImageArray_MODIS(self, img_id: int) -> lazy_import('numpy').ndarray:

        if self.satimg_view_opt == SatImgViewOpt.NATURAL_COLOR:
            return self.__getSatelliteImageArray_MODIS_NATURAL_COLOR(img_id=img_id)
        elif self.satimg_view_opt == SatImgViewOpt.CIR:
            return self.__getSatelliteImageArray_MODIS_CIR(img_id=img_id)
        elif self.satimg_view_opt == SatImgViewOpt.NDVI:
            return self.__getSatelliteImageArray_NDVI(img_id=img_id)
        elif self.satimg_view_opt == SatImgViewOpt.SHORTWAVE_INFRARED1:
            return self.__getSatelliteImageArray_MODIS_SHORTWAVE_INFRARED_SWIR1(img_id=img_id)
        elif self.satimg_view_opt == SatImgViewOpt.SHORTWAVE_INFRARED2:
            return self.__getSatelliteImageArray_MODIS_SHORTWAVE_INFRARED_SWIR2(img_id=img_id)
        else:
            raise NotImplementedError

    def __getSatelliteImageArray(self, img_id: int) -> lazy_import('numpy').ndarray:

        if self.modis_collection == ModisIndex.REFLECTANCE:
            return self.__getSatelliteImageArray_MODIS(img_id)
        else:
            raise NotImplementedError

    def __showSatImage_MODIS(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]],
                             brightness_factors: Union[tuple[float, float], list[float, float]]) -> None:
        # lazy imports
        opencv = lazy_import('cv2')

        if not self._satimgs_processed:
            try:
                self._processMetaData_SATELLITE_IMG()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to satellite images!')

        if id_img < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        if len(self) - 1 < id_img:
            raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands!'.format(len(self)))

        satimg = self.__getSatelliteImageArray(id_img)
        if brightness_factors is not None and self.satimg_view_opt == SatImgViewOpt.NATURAL_COLOR:
            # increase image brightness
            satimg = opencv.convertScaleAbs(satimg, alpha=brightness_factors[0], beta=brightness_factors[1])

        # figure title
        img_type = self.satimg_view_opt.value
        str_title = 'MODIS ({}, {})'.format(img_type, self.satimg_dates.iloc[id_img]['Date'])

        # show labels and binary mask related to localization of wildfires (CCI labels)
        imshow(src=satimg, title=str_title, figsize=figsize, show=True)

    def showSatImage(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                     brightness_factors: Union[tuple[float, float], list[float, float]] = (5., 5.)) -> None:

        if self.modis_collection == ModisIndex.REFLECTANCE:
            self.__showSatImage_MODIS(id_img=id_img, figsize=figsize, brightness_factors=brightness_factors)
        else:
            raise NotImplementedError

    """
    Display functionality (LABELS)
    """

    def __getFireLabels_CCI(self, confidence_level: lazy_import('numpy').ndarray, with_fire_mask=False) -> \
            Union[lazy_import('numpy').ndarray, tuple[lazy_import('numpy').ndarray]]:

        # lazy imports
        colors = lazy_import('mlfire.utils.colors')
        np = lazy_import('numpy')

        label = np.empty(shape=confidence_level.shape + (3,), dtype=np.uint8)
        label[:, :] = colors.Colors.GRAY_COLOR.value
        mask_fires = None

        if with_fire_mask or self.labels_view_opt == FireLabelsViewOpt.CONFIDENCE_LEVEL:
            mask_fires = confidence_level >= self.cci_confidence_level

        if self.labels_view_opt == FireLabelsViewOpt.LABEL:

            label[mask_fires, :3] = colors.Colors.RED_COLOR.value

        elif self.labels_view_opt == FireLabelsViewOpt.CONFIDENCE_LEVEL:

            cmap = lazy_import('mlfire.utils.cmap')

            lst_colors = ['#ff0000', '#ff5a00', '#ffff00']
            cl_min = 50
            cl_max = 100

            cmap_helper = cmap.CMapHelper(lst_colors=lst_colors, vmin=cl_min, vmax=cl_max)

            for v in range(self.cci_confidence_level, cl_max + 1):
                c = [int(v * 255) for v in cmap_helper.getRGB(v)[::-1]]
                label[confidence_level == v, :] = c

        else:
            raise AttributeError('Not supported option for viewing labels!')

        if mask_fires is not None:
            return label, mask_fires
        else:
            del mask_fires; gc.collect()
            return label

    def __showFireLabels_CCI(self, id_band: int,  figsize: Union[tuple[float, float], list[float, float]]) -> None:

        # lazy imports
        calendar = lazy_import('calendar')
        np = lazy_import('numpy')

        PIXEL_NOT_BURNABLE = -1

        # read band as raster image
        id_ds, id_rs = self._map_band_id_label[id_band]

        # confidence level
        rs_cl = self._ds_labels[id_ds].GetRasterBand(id_rs)
        dsc_cl = rs_cl.GetDescription()

        # observed flag
        rs_mask = self._ds_labels[id_ds].GetRasterBand(id_rs + 1)
        dsc_mask = rs_mask.GetDescription()

        # check to avoid issues in following processing
        if band2date_firecci(dsc_cl) != band2date_firecci(dsc_mask):
            raise ValueError('Dates between ConfidenceLevel and ObservedFlag bands are not same!')

        confidence_level = rs_cl.ReadAsArray()
        mask = rs_mask.ReadAsArray()

        label = self.__getFireLabels_CCI(confidence_level=confidence_level)
        # show non-mapped areas in a black colour
        if np.max(mask) == 1: label[mask == PIXEL_NOT_BURNABLE, :] = 0

        label_date = self.label_dates.iloc[id_band]['Date']
        str_title = 'CCI labels ({}, {})'.format(label_date.year, calendar.month_name[label_date.month])

        # show labels and binary mask related to localization of wildfires (CCI labels)
        imshow(src=label, title=str_title, figsize=figsize, show=True)

    def __getFireLabels_MTBS(self, fire_severity: lazy_import('numpy').ndarray, with_fire_mask=False) -> \
            Union[lazy_import('numpy').ndarray, tuple[lazy_import('numpy').ndarray]]:

        # lazy imports
        colors = lazy_import('mlfire.utils.colors')
        np = lazy_import('numpy')

        label = np.empty(shape=fire_severity.shape + (3,), dtype=np.uint8)
        label[:, :] = colors.Colors.GRAY_COLOR.value
        mask_fires = None

        if with_fire_mask or self.labels_view_opt == FireLabelsViewOpt.LABEL:

            c1 = fire_severity >= self.mtbs_severity_from.value; c2 = fire_severity <= MTBSSeverity.HIGH.value
            mask_fires = np.logical_and(c1, c2)

        if self.labels_view_opt == FireLabelsViewOpt.LABEL:

            label[mask_fires, :] = colors.Colors.RED_COLOR.value

        elif self.labels_view_opt == FireLabelsViewOpt.SEVERITY:

            cmap = lazy_import('mlfire.utils.cmap')

            lst_colors = ['#ff0000', '#ffff00']
            severity_min = MTBSSeverity.LOW.value
            severity_max = MTBSSeverity.HIGH.value

            cmap_helper = cmap.CMapHelper(lst_colors=lst_colors, vmin=severity_min, vmax=severity_max)
            for v in range(self.mtbs_severity_from.value, severity_max + 1):
                c = [int(v * 255) for v in cmap_helper.getRGB(v)[::-1]]
                label[fire_severity == v, :] = c

        else:
            raise ArithmeticError

        # return
        if mask_fires is not None:
            return label, mask_fires
        else:
            del mask_fires; gc.collect()
            return label

    def __showFireLabels_MTBS(self, id_band: int, figsize: Union[tuple[float, float], list[float, float]]) -> None:

        np = lazy_import('numpy')

        # read band as raster image
        id_ds, id_rs = self._map_band_id_label[id_band]
        rs_severity = self._ds_labels[id_ds].GetRasterBand(id_rs).ReadAsArray()

        label = self.__getFireLabels_MTBS(rs_severity, with_fire_mask=False)

        # show up non-mapped areas as black
        mask_non_mapped = np.array(rs_severity == MTBSSeverity.NON_MAPPED_AREA.value)
        if len(mask_non_mapped) != 0: label[mask_non_mapped, :] = 0

        # get date
        label_date = self.label_dates.iloc[id_band]['Date']
        str_title = 'MTBS labels ({} {})'.format(self.mtbs_region.name, label_date.year)

        # show up labels and binary mask (non-) related to localization of wildfires (MTBS labels)
        imshow(src=label, title=str_title, figsize=figsize, show=True)

    def showFireLabels(self, id_band: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5)) -> None:

        if not self._labels_processed:
            # processing descriptions of bands related to fire labels and obtain dates from them
            try:
                self._processMetaData_LABELS()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to labels!')

        if id_band < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        if self.nbands_label - 1 < id_band:
            raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(self.nbands_label))

        if self.label_collection == FireLabelsCollection.CCI:
            self.__showFireLabels_CCI(id_band=id_band, figsize=figsize)
        elif self.label_collection == FireLabelsCollection.MTBS:
            self.__showFireLabels_MTBS(id_band=id_band, figsize=figsize)
        else:
            raise NotImplementedError

    """
    Display functionality (MULTISPECTRAL SATELLITE IMAGE + LABELS)
    """

    def __getLabelsForSatImg_CCI(self, id_img: int) -> (lazy_import('numpy').ndarray, lazy_import('numpy').ndarray):

        # lazy import
        datetime = lazy_import('datetime')

        # get label date time
        date_satimg = self.satimg_dates.iloc[id_img]['Date']
        date_satimg = datetime.date(year=date_satimg.year, month=date_satimg.month, day=1)

        # get index of corresponding labels
        label_index = self.label_dates.index[self._df_dates_labels['Date'] == date_satimg][0]
        id_ds, id_rs = self._map_band_id_label[label_index]

        confidence_level = self._ds_labels[id_ds].GetRasterBand(id_rs).ReadAsArray()
        label, mask_fires = self.__getFireLabels_CCI(confidence_level=confidence_level, with_fire_mask=True)

        return label, mask_fires

    def __getLabelsForSatImg_MTBS(self, id_img: int) -> (lazy_import('numpy').ndarray, lazy_import('numpy').ndarray):

        # lazy import
        datetime = lazy_import('datetime')

        date_satimg = self.satimg_dates.iloc[id_img]['Date']
        date_satimg = datetime.date(year=date_satimg.year, month=1, day=1)

        # get index of corresponding labels
        label_index = self._df_dates_labels.index[self._df_dates_labels['Date'] == date_satimg][0]
        id_ds, id_rs = self._map_band_id_label[label_index]

        # get severity
        severity_fire = self._ds_labels[id_ds].GetRasterBand(id_rs).ReadAsArray()
        label, mask_fires = self.__getFireLabels_MTBS(fire_severity=severity_fire, with_fire_mask=True)

        return label, mask_fires

    def __getLabelsForSatImg(self, id_img: int) -> (lazy_import('numpy').ndarray, lazy_import('numpy').ndarray):

        if self.label_collection == FireLabelsCollection.CCI:
            return self.__getLabelsForSatImg_CCI(id_img)
        elif self.label_collection == FireLabelsCollection.MTBS:
            return self.__getLabelsForSatImg_MTBS(id_img)
        else:
            raise NotImplementedError

    def __showSatImageWithFireLabels_REFLECTANCE(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                                                 brightness_factors: Union[tuple[float, float], list[float, float]] = (5., 5.)) -> None:
        # lazy import
        opencv = lazy_import('cv2')

        # get image numpy array
        satimg = self.__getSatelliteImageArray(id_img)

        # increase image brightness
        if brightness_factors is not None and self.satimg_view_opt == SatImgViewOpt.NATURAL_COLOR:
            # increase image brightness
            satimg = opencv.convertScaleAbs(satimg, alpha=brightness_factors[0], beta=brightness_factors[1])

        # get labels and confidence level/severity
        labels, mask = self.__getLabelsForSatImg(id_img)
        satimg[mask, :] = labels[mask, :]

        ref_date = self.satimg_dates.iloc[id_img]['Date']
        img_type = self.satimg_view_opt.value
        str_title = 'MODIS ({}, {}, labels={})'.format(img_type, ref_date, self.label_collection.name)

        # show multi spectral image as RGB with additional information about localization of wildfires
        imshow(src=satimg, title=str_title, figsize=figsize, show=True)

    def showSatImageWithFireLabels(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                                   brightness_factors: Union[tuple[float, float], list[float, float]] = (5., 5.)):

        if not self._labels_processed:
            # processing descriptions of bands related to fire labels and obtain dates from them
            try:
                self._processMetaData_LABELS()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to labels!')

        if not self._satimgs_processed:
            # process descriptions of bands related to satellite images and obtain dates from them
            try:
                self._processMetaData_SATELLITE_IMG()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to satellite images!')

        if self.modis_collection == ModisIndex.REFLECTANCE:
            self.__showSatImageWithFireLabels_REFLECTANCE(id_img=id_img, figsize=figsize, brightness_factors=brightness_factors)
        else:
            raise NotImplementedError


#
if __name__ == '__main__':

    DATA_DIR = 'data/tifs'
    # PREFIX_IMG = 'ak_reflec_january_december_{}_100km'
    PREFIX_IMG = 'ak_reflec_january_december_{}_850'

    LABEL_COLLECTION = FireLabelsCollection.MTBS
    # LABEL_COLLECTION = FireLabelsCollection.CCI
    STR_LABEL_COLLECTION = LABEL_COLLECTION.name.lower()

    lst_satimgs = []
    lst_labels = []

    for year in range(2004, 2005):

        PREFIX_IMG_YEAR = PREFIX_IMG.format(year)

        fn_satimg = os.path.join(DATA_DIR, '{}_epsg3338_area_0.tif'.format(PREFIX_IMG_YEAR))
        lst_satimgs.append(fn_satimg)

        fn_labels = os.path.join(DATA_DIR, '{}_epsg3338_area_0_{}_labels.tif'.format(PREFIX_IMG_YEAR, STR_LABEL_COLLECTION))
        lst_labels.append(fn_labels)

    SATIMG_VIEW_OPT = SatImgViewOpt.NDVI

    LABELS_VIEW_OPT = FireLabelsViewOpt.LABEL
    LABELS_VIEW_OPT = FireLabelsViewOpt.CONFIDENCE_LEVEL if LABEL_COLLECTION == FireLabelsCollection.CCI else FireLabelsViewOpt.SEVERITY

    # setup of data set loader
    dataset_view = DatasetView(
        lst_satimgs=lst_satimgs,
        lst_labels=lst_labels,
        satimg_view_opt=SATIMG_VIEW_OPT,
        label_collection=LABEL_COLLECTION,
        labels_view_opt=LABELS_VIEW_OPT,
        mtbs_severity_from=MTBSSeverity.LOW
    )

    print('#ts = {}'.format(len(dataset_view)))
    print(dataset_view.label_dates)

    dataset_view.showSatImage(30)

    # dataset_view.showFireLabels(18 if LABEL_COLLECTION == FireLabelsCollection.CCI else 1)
    # dataset_view.showSatImage(70)
    # dataset_view.showSatImageWithFireLabels(70)
