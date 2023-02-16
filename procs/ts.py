import calendar
import datetime
import gc
import os.path

import cv2 as opencv
import numpy as np
import pandas as pd

from osgeo import gdal

from utils.utils_string import band2date_firecci


class DataAdapterTS(object):

    def __init__(self,
                 src_labels: str,
                 cci_confidence_level: int = None):

        self._ds_labels = None
        self._df_dates_labels = None

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

        del self._df_dates_labels; self._df_dates_labels = None
        gc.collect()

    @property
    def nbands_labels(self) -> int:

        if self._ds_labels is None:
            self.__loadGeoTIFF_LABELS()

        return self._ds_labels.RasterCount

    @property
    def dates_label(self) -> pd.DataFrame:

        if self._df_dates_labels is None:
            self.__getBandDates_LABEL_CCI()

        return self._df_dates_labels

    def getLabelBandDate(self, band_id: int):

        if self._df_dates_labels is None:
            self.__getBandDates_LABEL_CCI()

        band_date = self._df_dates_labels.iloc[band_id][0]
        return band_date

    # io functionality

    def __loadGeoTIFF_LABELS(self) -> None:

        if self._src_labels is None:
            raise IOError('File related to labels is not set!')

        # load data set of labels from GeoTIFF
        self._ds_labels = gdal.Open(self.src_labels)

    # process functionality

    def __getBandDates_LABEL_CCI(self) -> None:

        try:
            self.__loadGeoTIFF_LABELS()
        except IOError:
            raise IOError('Cannot load a label file ({})!'.format(self.src_labels))

        lst = []

        for raster_id in range(self.nbands_labels):

            rs_band = self._ds_labels.GetRasterBand(raster_id + 1)
            dsc_band = rs_band.GetDescription()

            lst_date = band2date_firecci(dsc_band).split('-')
            band_date = datetime.date(year=int(lst_date[0]), month=int(lst_date[1]), day=1)
            lst.append(band_date)

        df_dates = pd.DataFrame(lst)
        self._df_dates_labels = df_dates

    def __processLabels_CCI(self) -> None:

        pass

    def __processLabels(self) -> None:

        pass

    # display functionality

    def __imshow_label_CCI(self, band_id: int) -> None:

        if self._ds_labels is None:
            try:
                self.__loadGeoTIFF_LABELS()
            except IOError:
                raise IOError('Cannot load a label file ({})!'.format(self.src_labels))

        if band_id < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        if self.nbands_labels - 1 < band_id:
            raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(band_id))

        # read band as raster
        img = self._ds_labels.GetRasterBand(band_id + 1).ReadAsArray()

        # create binary mask
        img[img < self.cci_confidence_level] = 0  # not fire
        img[img >= self.cci_confidence_level] = 1  # fire

        band_date = self.getLabelBandDate(band_id)

        # display
        str_title = 'Labels ({}, {})'.format(band_date.year, calendar.month_name[band_date.month])
        opencv.imshow(str_title, img.astype(np.float32))
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
