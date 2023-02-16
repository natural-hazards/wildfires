import os.path

import cv2 as opencv
import numpy as np

from osgeo import gdal


class DataAdapterTS(object):

    def __init__(self,
                 src_labels: str,
                 cci_confidence_level: int = None):

        self._ds_labels = None

        self._src_labels = None
        self.src_labels = src_labels

        self._cci_confidence_level = None
        self.cci_confidence_level = cci_confidence_level

    @property
    def src_labels(self) -> str:

        return self._src_labels

    @src_labels.setter
    def src_labels(self, fn: str) -> None:

        if self._src_labels == fn:
            return

        if not os.path.exists(fn):
            raise IOError('File {} does not exist!'.format(fn))

        self._src_labels = fn

    @property
    def cci_confidence_level(self) -> int:

        return self._cci_confidence_level

    @cci_confidence_level.setter
    def cci_confidence_level(self, level: int) -> None:

        if self._cci_confidence_level == level:
            return

        if level < 0 or level > 100:
            raise ValueError('Confidence level for FireCCI labels must be positive int between 0 and 100!')

        self._cci_confidence_level = level

    def __reset(self) -> None:

        pass

    @property
    def nbands_labels(self) -> int:

        if self._ds_labels is None:
            self.__loadGeoTIFF_LABELS()

        return self._ds_labels.RasterCount

    # io functionality

    def __loadGeoTIFF_LABELS(self) -> None:

        if self._src_labels is None:
            raise IOError('File related to labels is not set!')

        # load data set of labels from GeoTIFF
        self._ds_labels = gdal.Open(self.src_labels)

    # process functionality

    def __processLabels_ESA_CCI(self) -> None:

        pass

    def __processLabels(self) -> None:

        pass

    # display functionality

    def __imshow_label_CCI(self, band_id: int) -> None:

        if self._ds_labels is None:
            self.__loadGeoTIFF_LABELS()

        if band_id < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        if self.nbands_labels - 1 < band_id:
            raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(band_id))

        # read band as raster
        img = self._ds_labels.GetRasterBand(band_id + 1).ReadAsArray()

        # create binary mask
        img[img < self.cci_confidence_level] = 0
        img[img >= self.cci_confidence_level] = 1

        fn_label = os.path.basename(self.src_labels)
        opencv.imshow('Labels (band={})'.format(band_id), img.astype(np.float32))
        opencv.waitKey(0)

    def imshow_label(self, band_id: int) -> None:

        self.__imshow_label_CCI(band_id)


if __name__ == '__main__':

    src_labels = 'tutorials/ak_april_july_2004_500_epsg3338_area_0_labels.tif'
    adapter = DataAdapterTS(
        src_labels=src_labels,
        cci_confidence_level=70
    )
    adapter.imshow_label(2)
