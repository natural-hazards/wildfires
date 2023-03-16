import os

import numpy as np

from typing import Union

from mlfire.data.loader import DatasetLoader
from mlfire.earthengine.collections import ModisIndex, ModisReflectanceSpectralBands

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

    def __getSatelliteImageArray_REFLECTANCE(self, img_id: int) -> np.ndarray:

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

    def __getSatelliteImageArray(self, img_id: int) -> np.ndarray:

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
                super()._processMetaData_SATELLITE_IMG()
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
    dataset_view.showSatImage(70)
