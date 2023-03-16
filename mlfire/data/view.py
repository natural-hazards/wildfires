import os

from typing import Union

from mlfire.data.loader import DatasetLoader
from mlfire.earthengine.collections import ModisIndex, ModisReflectanceSpectralBands
from mlfire.earthengine.collections import FireLabelsCollection, MTBSSeverity

# utils imports
from mlfire.utils.functool import lazy_import
from mlfire.utils.plots import imshow


class DatasetView(DatasetLoader):

    def __init__(self,
                 lst_satimgs: Union[tuple[str], list[str]],
                 lst_labels: Union[tuple[str], list[str]]):

        super().__init__(
            lst_satimgs,
            lst_labels
        )

    """
    Display functionality (MULTISPECTRAL SATELLITE IMAGE, MODIS)
    """

    def __getSatelliteImageArray_REFLECTANCE(self, img_id: int) -> lazy_import('numpy.ndarray'):

        np = lazy_import('numpy')

        id_ds, start_band_id = self._map_start_satimgs[img_id]
        ds_satimg = self._ds_satimgs[id_ds]

        ref_red = ds_satimg.GetRasterBand(start_band_id).ReadAsArray()
        band_id = start_band_id + ModisReflectanceSpectralBands.BLUE.value - 1
        ref_blue = ds_satimg.GetRasterBand(band_id).ReadAsArray()
        band_id = start_band_id + ModisReflectanceSpectralBands.GREEN.value - 1
        ref_green = self._ds_satimgs[id_ds].GetRasterBand(band_id).ReadAsArray()

        # get image in range 0 - 255 per channel
        img = np.zeros(shape=ref_red.shape + (3,), dtype=np.uint8)
        img[:, :, 0] = (ref_blue + 100.) / 16100. * 255.
        img[:, :, 1] = (ref_green + 100.) / 16100. * 255.
        img[:, :, 2] = (ref_red + 100.) / 16100. * 255.

        return img

    def __getSatelliteImageArray(self, img_id: int) -> lazy_import('numpy.ndarray'):

        if self.modis_collection == ModisIndex.REFLECTANCE:
            return self.__getSatelliteImageArray_REFLECTANCE(img_id)
        else:
            raise NotImplementedError

    def __showSatImage_REFLECTANCE(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]],
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

        img = self.__getSatelliteImageArray(id_img)  # increase image brightness
        enhanced_img = opencv.convertScaleAbs(img, alpha=brightness_factors[0], beta=brightness_factors[1])

        # figure title
        str_title = 'MODIS Reflectance ({})'.format(self._df_dates_satimgs.iloc[id_img]['Date'])

        # show labels and binary mask related to localization of wildfires (CCI labels)
        imshow(src=enhanced_img, title=str_title, figsize=figsize, show=True)

    def showSatImage(self, id_img: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5),
                     brightness_factors: Union[tuple[float, float], list[float, float]] = (5., 5.)) -> None:

        if self.modis_collection == ModisIndex.REFLECTANCE:
            self.__showSatImage_REFLECTANCE(id_img=id_img, figsize=figsize, brightness_factors=brightness_factors)
        else:
            raise NotImplementedError

    """
    Display functionality (LABELS)
    """

    def __showFireLabels_MTBS(self, id_band: int, figsize: Union[tuple[float, float], list[float, float]]) -> None:

        np = lazy_import('numpy')

        if not self._labels_processed:
            try:
                self._processMetaData_LABELS()
            except IOError or ValueError:
                raise IOError('Cannot process meta data related to labels!')

        if id_band < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        if self.nbands_label - 1 < id_band:
            raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(id_band))

        # read band as raster image
        id_ds, band_id = self._map_band_id_label[id_band]
        rs = self._ds_labels[id_ds].GetRasterBand(band_id).ReadAsArray()

        # create mask for non-mapping areas
        mask = np.zeros(shape=rs.shape)
        mask[rs == MTBSSeverity.NON_MAPPING_AREA.value] = 1

        # create label mask
        label = np.zeros(shape=rs.shape)
        label[np.logical_and(rs >= self.mtbs_severity_from.value, rs <= MTBSSeverity.HIGH.value)] = 1

        # create image for visualizing labels
        img = np.ones(shape=rs.shape + (3,), dtype=np.float32)
        img[label == 1, 0:2] = 0; img[label == 1, 2] = 1  # display area affected by a fire in a red colour
        if np.max(mask) == 1:
            img[mask == 1, :] = 0  # display non-mapping areas in a black colour

        # display labels
        label_date = self.label_dates.iloc[id_band]['Date']
        str_title = 'MTBS labels ({} {})'.format(self.mtbs_region.name, label_date.year)

        # show labels and binary mask related to localization of wildfires (MTBS labels)
        imshow(src=img, title=str_title, figsize=figsize, show=True)

    def showFireLabels(self, id_band: int, figsize: Union[tuple[float, float], list[float, float]] = (6.5, 6.5)) -> None:

        if self.label_collection == FireLabelsCollection.CCI:
            raise NotImplementedError
        elif self.label_collection == FireLabelsCollection.MTBS:
            self.__showFireLabels_MTBS(id_band=id_band, figsize=figsize)
        else:
            raise NotImplementedError


#
if __name__ == '__main__':

    DATA_DIR = 'data/tifs'
    PREFIX_IMG = 'ak_reflec_january_december_{}_100km'

    lst_satimgs = []
    lst_labels_mtbs = []

    for year in range(2004, 2006):

        PREFIX_IMG_YEAR = PREFIX_IMG.format(year)

        fn_satimg = os.path.join(DATA_DIR, '{}_epsg3338_area_0.tif'.format(PREFIX_IMG_YEAR))
        lst_satimgs.append(fn_satimg)

        fn_labels_mtbs = os.path.join(DATA_DIR, '{}_epsg3338_area_0_mtbs_labels.tif'.format(PREFIX_IMG_YEAR))
        lst_labels_mtbs.append(fn_labels_mtbs)

    # setup of data set loader
    dataset_view = DatasetView(
        lst_satimgs=lst_satimgs,
        lst_labels=lst_labels_mtbs
    )

    print(len(dataset_view))
    print(dataset_view.label_dates)

    dataset_view.showSatImage(70)
    dataset_view.showFireLabels(0)
