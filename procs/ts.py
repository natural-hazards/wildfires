import calendar
import datetime
import gc
import os.path

import cv2 as opencv
import numpy as np
import pandas as pd

from osgeo import gdal

from earthengine.ds import FireLabelsCollection, MTBSRegion, MTBSSeverity
from utils.utils_string import band2date_firecci, band2date_mtbs


class DataAdapterTS(object):

    def __init__(self,
                 src_labels: str,
                 label_collection: FireLabelsCollection = FireLabelsCollection.CCI,
                 cci_confidence_level: int = None,
                 mtbs_severity_from: MTBSSeverity = MTBSSeverity.LOW,
                 mtbs_region: MTBSRegion = None):

        self._ds_labels = None
        self._df_dates_labels = None

        self._map_band_ids = None

        self._nbands_labels = -1

        # properties

        self._src_labels = None
        self.src_labels = src_labels

        self._label_collection = None
        self.label_collection = label_collection

        self._cci_confidence_level = None
        self.cci_confidence_level = cci_confidence_level

        self._mtbs_severity_from = None
        self.mtbs_severity_from = mtbs_severity_from

        self._mtbs_region = None
        self.mtbs_region = mtbs_region

        self._labels_processed = False

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
    def label_collection(self) -> FireLabelsCollection:

        return self._label_collection

    @label_collection.setter
    def label_collection(self, collection: FireLabelsCollection) -> None:

        if self.label_collection == collection:
            return

        self.__reset()
        self._label_collection = collection

    # FireCII properties

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

    # MTBS properties

    @property
    def mtbs_severity_from(self) -> MTBSSeverity:

        return self._mtbs_severity_from

    @mtbs_severity_from.setter
    def mtbs_severity_from(self, severity: MTBSSeverity) -> None:

        if self._mtbs_severity_from == severity:
            return

        self.__reset()
        self._mtbs_severity_from = severity

    @property
    def mtbs_region(self) -> MTBSRegion:

        return self._mtbs_region

    @mtbs_region.setter
    def mtbs_region(self, region: MTBSRegion) -> None:

        if self._mtbs_region == region:
            return

        self.__reset()
        self._mtbs_region = region

    def __reset(self) -> None:

        del self._df_dates_labels; self._df_dates_labels = None
        del self._map_band_ids; self._map_band_ids = None
        gc.collect()

        self._nbands_labels = -1
        self._labels_processed = False

    @property
    def nbands_labels(self) -> int:

        if not self._labels_processed:
            self.__processLabels()

        return self._nbands_labels

    @property
    def dates_label(self) -> pd.DataFrame:

        if self._df_dates_labels is None:
            self.__getBandDates_LABEL()

        return self._df_dates_labels

    def getLabelBandDate(self, band_id: int):

        # TODO reimplement

        if self._df_dates_labels is None:
            self.__getBandDates_LABEL()

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

        if self._ds_labels is None:
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
        del lst
        gc.collect()

        self._df_dates_labels = df_dates

    def __processBandDates_LABEL_MTBS(self) -> None:

        if self._ds_labels is None:
            try:
                self.__loadGeoTIFF_LABELS()
            except IOError:
                raise IOError('Cannot load a label file ({})!'.format(self.src_labels))

        lst = []
        nbands = self._ds_labels.RasterCount

        for raster_id in range(nbands):

            rs_band = self._ds_labels.GetRasterBand(raster_id + 1)
            dsc_band = rs_band.GetDescription()

            if self.mtbs_region.value in dsc_band:
                band_date = band2date_mtbs(dsc_band)
                lst.append(band_date)

        if not lst:
            raise ValueError('Label file does not containe any useful data!')

        df_dates = pd.DataFrame(lst)
        del lst
        gc.collect()

        self._df_dates_labels = df_dates

    def __getBandDates_LABEL(self) -> None:

        if self.label_collection == FireLabelsCollection.CCI:
            self.__getBandDates_LABEL_CCI()
        elif self.label_collection == FireLabelsCollection.MTBS:
            self.__processBandDates_LABEL_MTBS()
        else:
            raise NotImplementedError

    def __processLabels_CCI(self) -> None:

        pass

    def __processLabels_MTBS(self) -> None:

        if self._ds_labels is None:
            try:
                self.__loadGeoTIFF_LABELS()
            except IOError:
                raise IOError('Cannot load a label file ({})'.format(self.src_labels))

        # process date
        try:
            self.__processBandDates_LABEL_MTBS()
        except ValueError:
            raise ValueError('Cannot process date bands!')

        # determine bands and their ids for region selection
        if self._map_band_ids is None:
            del self._map_band_ids; self._map_band_ids = None
            gc.collect()  # invoke garbage collector

        map_band_ids = {}
        pos = 0

        # get all number of bands in GeoTIFF data set
        nbands = self._ds_labels.RasterCount

        for band_id, raster_id in enumerate(range(nbands)):

            rs_band = self._ds_labels.GetRasterBand(raster_id + 1)
            dsc_band = rs_band.GetDescription()

            # map id data set for selected region to band id
            if self.mtbs_region.value in dsc_band:
                map_band_ids[pos] = band_id + 1; pos += 1

        if not map_band_ids:
            raise ValueError('Label file {} does not contain any useful information for a selected region.')

        self._map_band_ids = map_band_ids
        self._nbands_labels = pos

        # set that everything is done
        self._labels_processed = True

    def __processLabels(self) -> None:

        if self.label_collection == FireLabelsCollection.CCI:
            self.__processLabels_CCI()
        elif self.label_collection == FireLabelsCollection.MTBS:
            self.__processLabels_MTBS()
        else:
            raise NotImplementedError

    # display functionality

    def __imshow_label_MTBS(self, band_id: int) -> None:

        if not self._labels_processed:
            try:
                self.__processLabels()
            except IOError:
                raise IOError('Cannot load a label file ({})!'.format(self.src_labels))

        if band_id < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        if self.nbands_labels - 1 < band_id:
            raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(band_id))

        # read band as raster image
        rs_band_id = self._map_band_ids[band_id]
        img = self._ds_labels.GetRasterBand(rs_band_id).ReadAsArray()

        # create mask for non-mapping areas
        mask = np.zeros(shape=img.shape)
        mask[img == MTBSSeverity.NON_MAPPING_AREA.value] = 1

        # create label mask
        label = np.zeros(shape=img.shape)
        label[np.logical_and(img >= self.mtbs_severity_from.value, img <= MTBSSeverity.HIGH.value)] = 1

        # create image for visualizing labels
        img = np.ones(shape=img.shape + (3,), dtype=np.float32)
        img[label == 1, 0:2] = 0; img[label == 1, 2] = 1  # display area affected by a fire in a red colour
        if np.max(mask) == 1:
            img[mask == 1, :] = 0  # display non-mapping areas in a black colour

        # display labels
        str_title = 'MTBS labels ({}, {})'.format(self.mtbs_region.name, self.dates_label.iloc[band_id][0])
        opencv.imshow(str_title, img)
        opencv.waitKey(0)

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

        # read band as raster image
        img = self._ds_labels.GetRasterBand(band_id + 1).ReadAsArray()

        # create label mask
        img[img < self.cci_confidence_level] = 0  # not fire
        img[img >= self.cci_confidence_level] = 1  # fire

        band_date = self.getLabelBandDate(band_id)

        # display
        str_title = 'Labels ({}, {})'.format(band_date.year, calendar.month_name[band_date.month])
        opencv.imshow(str_title, img.astype(np.float32))
        opencv.waitKey(0)

    def imshow_label(self, band_id: int) -> None:

        if self.label_collection == FireLabelsCollection.CCI:
            self.__imshow_label_CCI(band_id)
        elif self.label_collection == FireLabelsCollection.MTBS:
            self.__imshow_label_MTBS(band_id)
        else:
            raise NotImplementedError


if __name__ == '__main__':

    # fn_labels = 'tutorials/ak_april_july_2004_500_epsg3338_area_0_cci_labels.tif'
    fn_labels = 'tutorials/ak_april_july_2004_500_epsg3338_area_0_mtbs_labels.tif'

    # adapter
    # adapter = DataAdapterTS(
    #     src_labels=fn_labels,
    #     label_collection=FireLabelsCollection.CCI,
    #     cci_confidence_level=70
    # )

    adapter = DataAdapterTS(
        src_labels=fn_labels,
        label_collection=FireLabelsCollection.MTBS,
        mtbs_region=MTBSRegion.ALASKA,
    )

    # print dates and show label
    print(adapter.dates_label)
    adapter.imshow_label(0)
