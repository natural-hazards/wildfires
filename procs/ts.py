import calendar
import datetime
import gc
import os.path

import cv2 as opencv
import numpy as np
import pandas as pd

from osgeo import gdal

from earthengine.ds import FireLabelsCollection, ModisIndex, ModisReflectanceSpecralBands, MTBSRegion, MTBSSeverity
from utils.utils_string import band2date_firecci, band2date_mtbs, band2data_reflectance


class DataAdapterTS(object):

    def __init__(self,
                 src_satimg: str,
                 src_labels: str,
                 modis_collection: ModisIndex = ModisIndex.REFLECTANCE,
                 label_collection: FireLabelsCollection = FireLabelsCollection.CCI,
                 cci_confidence_level: int = None,
                 mtbs_severity_from: MTBSSeverity = MTBSSeverity.LOW,
                 mtbs_region: MTBSRegion = None):

        self._ds_satimg = None
        self._df_dates_satimg = None

        self._ds_labels = None
        self._df_dates_labels = None

        self._map_band_id_label = None
        self._map_start_satimgs = None

        self._nimages = -1
        self._nbands_labels = -1

        # properties (satellite image)

        self._src_satimg = None
        self.src_satimg = src_satimg

        self._modis_collection = None
        self.modis_collection = modis_collection

        # properties (labels)

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

        self._satimg_processed = False
        self._labels_processed = False

    @property
    def src_satimg(self) -> str:

        return self._src_satimg

    @src_satimg.setter
    def src_satimg(self, fn: str) -> None:

        if self._src_satimg == fn:
            return

        self.__reset()
        self._src_satimg = fn

    @property
    def modis_collection(self) -> ModisIndex:

        return self._modis_collection

    @modis_collection.setter
    def modis_collection(self, collection: ModisIndex) -> None:

        if self.modis_collection == collection:
            return

        self.__reset()
        self._modis_collection = collection

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

    """
    FireCII properties
    """

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

    """
    MTBS properties
    """

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
        del self._map_band_id_label; self._map_band_id_label = None
        gc.collect()

        del self._df_dates_satimg; self._df_dates_satimg = None
        del self._map_start_satimgs; self._map_start_satimgs = None

        self._nimages = -1
        self._nbands_labels = -1

        self._satimg_processed = False
        self._labels_processed = False

    @property
    def nimgs(self) -> int:

        if not self._satimg_processed:
            self.__processMetaData_SATELLITE_IMG()

        return self._nimages

    @property
    def satimg_dates(self) -> pd.DataFrame:

        if not self._satimg_processed:
            self.__processMetaData_SATELLITE_IMG()

        return self._df_dates_satimg

    @property
    def nbands_labels(self) -> int:

        if not self._labels_processed:
            self.__processLabels()

        return self._nbands_labels

    @property
    def label_dates(self) -> pd.DataFrame:

        if self._df_dates_labels is None:
            self.__processBandDates_LABEL()

        return self._df_dates_labels

    def getLabelBandDate(self, band_id: int):

        # TODO reimplement

        if self._df_dates_labels is None:
            self.__processBandDates_LABEL()

        band_date = self._df_dates_labels.iloc[band_id][0]
        return band_date

    """
    IO functionality
    """

    def __loadGeoTIFF_SATELLITE_IMG(self) -> None:

        if self._src_satimg is None:
            raise IOError('File related to modis data is not set!')

        try:
            self._ds_satimg = gdal.Open(self.src_satimg)
        except IOError:
            raise IOError('Cannot load source related to satellite image {}!'.format(self.src_satimg))

    def __loadGeoTIFF_LABELS(self) -> None:

        if self._src_labels is None:
            raise IOError('File related to labels is not set!')

        # load data set of labels from GeoTIFF
        self._ds_labels = gdal.Open(self.src_labels)

    """
    Processing meta data (SATELLITE IMAGE, MODIS)
    """

    def __processBandDates_SATIMG_MODIS_REFLECTANCE(self) -> None:

        nbands = self._ds_satimg.RasterCount

        unique_dates = set()

        for band_id in range(0, nbands):  # proportion of spectra is divided in 7 bands

            rs_band = self._ds_satimg.GetRasterBand(band_id + 1)
            band_dsc = rs_band.GetDescription()

            if '_sur_refl_' in band_dsc:
                band_date = band2data_reflectance(rs_band.GetDescription())
                unique_dates.add(band_date)

        if not unique_dates:
            raise ValueError('Label file does not contain any useful data!')

        df_dates = pd.DataFrame(sorted(unique_dates), columns=['Date'])
        del unique_dates
        gc.collect()

        self._df_dates_satimg = df_dates

    def __processBandDates_SATELLITE_IMG(self) -> None:

        if self._ds_satimg is None:
            try:
                self.__loadGeoTIFF_SATELLITE_IMG()
            except IOError:
                raise IOError('Cannot load a satellite imgs file ({})'.format(self.src_satimg))

        if self.modis_collection == ModisIndex.REFLECTANCE:
            self.__processBandDates_SATIMG_MODIS_REFLECTANCE()
        else:
            raise NotImplementedError

    def __processMetaData_SATIMG_MODIS_REFLECTANCE(self) -> None:

        if self._map_start_satimgs is not None:
            del self._map_start_satimgs; self._map_band_id_satimg = None
            gc.collect()

        map_start_satimg = {}
        pos = 0

        # get all number of bands in GeoTIFF data set
        nbands = self._ds_satimg.RasterCount
        last_date = 0

        for band_id, raster_id in enumerate(range(nbands)):

            rs_band = self._ds_satimg.GetRasterBand(raster_id + 1)
            band_dsc = rs_band.GetDescription()
            band_date = band2data_reflectance(band_dsc)

            if '_sur_refl_' in band_dsc and last_date != band_date:
                map_start_satimg[pos] = band_id + 1; pos += 1
                last_date = band_date

        if not map_start_satimg:
            raise ValueError('Sattelite image file does not contain any useful data!')

        self._map_start_satimgs = map_start_satimg
        self._nimages = pos

    def __processMetaData_SATELLITE_IMG(self) -> None:

        if self._ds_satimg is None:
            try:
                self.__loadGeoTIFF_SATELLITE_IMG()
            except IOError:
                raise IOError('Cannot load a label file ({})'.format(self.src_labels))

        if self._df_dates_satimg is None:
            try:
                self.__processBandDates_SATELLITE_IMG()
            except ValueError:
                raise ValueError('Cannot process dates related to bands for CCI collection!')

        if self.modis_collection == ModisIndex.REFLECTANCE:
            self.__processMetaData_SATIMG_MODIS_REFLECTANCE()
        else:
            raise NotImplementedError

        self._satimg_processed = True

    """
    Process functionality (LABELS)
    """

    def __processBandDates_LABEL_CCI(self) -> None:

        lst = []
        nbands = self._ds_labels.RasterCount

        for raster_id in range(nbands):

            rs_band = self._ds_labels.GetRasterBand(raster_id + 1)
            dsc_band = rs_band.GetDescription()

            if 'ConfidenceLevel' in dsc_band:

                band_date = band2date_firecci(dsc_band)
                lst.append(band_date)

        if not lst:
            raise ValueError('Label file does not contain any useful data!')

        df_dates = pd.DataFrame(lst, columns=['Date'])
        del lst
        gc.collect()

        self._df_dates_labels = df_dates

    def __processBandDates_LABEL_MTBS(self) -> None:

        lst = []
        nbands = self._ds_labels.RasterCount

        for raster_id in range(nbands):

            rs_band = self._ds_labels.GetRasterBand(raster_id + 1)
            dsc_band = rs_band.GetDescription()

            if self.mtbs_region.value in dsc_band:
                band_date = band2date_mtbs(dsc_band)
                lst.append(band_date)

        if not lst:
            raise ValueError('Label file does not contain any useful data!')

        df_dates = pd.DataFrame(lst, columns=['Date'])
        del lst
        gc.collect()

        self._df_dates_labels = df_dates

    def __processBandDates_LABEL(self) -> None:

        if self._ds_labels is None:
            try:
                self.__loadGeoTIFF_LABELS()
            except IOError:
                raise IOError('Cannot load a label file ({})!'.format(self.src_labels))

        if self.label_collection == FireLabelsCollection.CCI:
            self.__processBandDates_LABEL_CCI()
        elif self.label_collection == FireLabelsCollection.MTBS:
            self.__processBandDates_LABEL_MTBS()
        else:
            raise NotImplementedError

    def __processLabels_CCI(self) -> None:

        if self._map_band_id_label is None:
            del self._map_band_id_label; self._map_band_id_label = None
            gc.collect()

        map_band_ids = {}
        pos = 0

        # get all number of bands in GeoTIFF data set
        nbands = self._ds_labels.RasterCount

        for band_id, raster_id in enumerate(range(nbands)):

            rs_band = self._ds_labels.GetRasterBand(raster_id + 1)
            dsc_band = rs_band.GetDescription()

            # map id data set for selected region to band id
            if 'ConfidenceLevel' in dsc_band:
                map_band_ids[pos] = band_id + 1; pos += 1

        if not map_band_ids:
            raise ValueError('Label file {} does not contain any useful information for a selected region.')

        self._map_band_id_label = map_band_ids
        self._nbands_labels = pos

    def __processLabels_MTBS(self) -> None:

        # determine bands and their ids for region selection
        if self._map_band_id_label is None:
            del self._map_band_id_label; self._map_band_id_label = None
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

        self._map_band_id_label = map_band_ids
        self._nbands_labels = pos

    def __processLabels(self) -> None:

        if self._ds_labels is None:
            try:
                self.__loadGeoTIFF_LABELS()
            except IOError:
                raise IOError('Cannot load a label file ({})'.format(self.src_labels))

        if self._df_dates_labels is None:
            try:
                self.__processBandDates_LABEL()
            except ValueError:
                raise ValueError('Cannot process dates related to bands for CCI collection!')

        if self.label_collection == FireLabelsCollection.CCI:
            self.__processLabels_CCI()
        elif self.label_collection == FireLabelsCollection.MTBS:
            self.__processLabels_MTBS()
        else:
            raise NotImplementedError

        # set that everything is done
        self._labels_processed = True

    """
    Display functionality (SATELLITE IMAGE, MODIS)
    """

    def __imshow_SATELLITE_IMG_REFLECTANCE(self, img_id) -> None:

        if not self._satimg_processed:
            try:
                self.__processMetaData_SATELLITE_IMG()
            except IOError and ValueError:
                raise IOError('Cannot process the satellite image ({})'.format(self.src_satimg))

        if img_id < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        if self.nimgs - 1 < img_id:
            raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(self.nimgs))

        band_start = self._map_start_satimgs[img_id]

        ref_red = self._ds_satimg.GetRasterBand(band_start).ReadAsArray()
        band_id = band_start + ModisReflectanceSpecralBands.BLUE.value - 1
        ref_blue = self._ds_satimg.GetRasterBand(band_id).ReadAsArray()
        band_id = band_start + ModisReflectanceSpecralBands.GREEN.value - 1
        ref_green = self._ds_satimg.GetRasterBand(band_id).ReadAsArray()

        # get image
        img = np.zeros(shape=ref_red.shape + (3,), dtype=np.uint8)
        img[:, :, 0] = (ref_blue + 100.) / 16100. * 255.
        img[:, :, 1] = (ref_green + 100.) / 16100. * 255.
        img[:, :, 2] = (ref_red + 100.) / 16100. * 255.

        enhanced_img = opencv.convertScaleAbs(img, alpha=5., beta=5.)
        img = enhanced_img

        str_title = 'MODIS Reflectance ({})'.format(self.satimg_dates.iloc[img_id]['Date'])
        opencv.imshow(str_title, img)
        opencv.waitKey(0)

    def imshow(self, img_id: int) -> None:

        if self.modis_collection == ModisIndex.REFLECTANCE:
            self.__imshow_SATELLITE_IMG_REFLECTANCE(img_id)
        else:
            raise NotImplementedError

    """
    Display functionality (LABELS)
    """

    def __imshow_label_CCI(self, band_id: int) -> None:

        if not self._labels_processed:
            try:
                self.__processLabels()
            except IOError and ValueError:
                raise IOError('Cannot process the label file ({})!'.format(self.src_labels))

        if band_id < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        if self.nbands_labels - 1 < band_id:
            raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(self.nbands_labels))

        rs_band_id = self._map_band_id_label[band_id]

        # confidence level
        rs_cl = self._ds_labels.GetRasterBand(rs_band_id)
        dsc_cl = rs_cl.GetDescription()

        # observed flag
        rs_mask = self._ds_labels.GetRasterBand(rs_band_id + 1)
        dsc_mask = rs_mask.GetDescription()

        # checking for sure to avoid problems in subsequent processing
        if band2date_firecci(dsc_cl) != band2date_firecci(dsc_mask):
            raise ValueError('Dates between ConfidenceLevel and ObservedFlag bands are not same!')

        confidence_level = rs_cl.ReadAsArray()
        mask = rs_mask.ReadAsArray()

        confidence_level[confidence_level < self.cci_confidence_level] = 0
        confidence_level[confidence_level >= self.cci_confidence_level] = 1

        PIXEL_NOT_BURNABLE = -1
        mask[mask <= PIXEL_NOT_BURNABLE] = 1

        # label
        label = np.ones(shape=confidence_level.shape + (3,), dtype=np.float32)
        label[confidence_level == 1, 0:2] = 0; label[confidence_level == 1, 2] = 1  # display area affected by a fire in a red colour
        if np.max(mask) == 1:
            label[mask == 1, :] = 0  # display non-mapping areas in a black colour

        str_title = 'CCI labels ({})'.format(self.label_dates.iloc[band_id][0])
        opencv.imshow(str_title, label)
        opencv.waitKey(0)

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
        rs_band_id = self._map_band_id_label[band_id]
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
        str_title = 'MTBS labels ({}, {})'.format(self.mtbs_region.name, self.label_dates.iloc[band_id][0])
        opencv.imshow(str_title, img)
        opencv.waitKey(0)

    def imshow_label(self, band_id: int) -> None:

        if self.label_collection == FireLabelsCollection.CCI:
            self.__imshow_label_CCI(band_id)
        elif self.label_collection == FireLabelsCollection.MTBS:
            self.__imshow_label_MTBS(band_id)
        else:
            raise NotImplementedError


if __name__ == '__main__':

    fn_satimg = 'tutorials/ak_reflec_april_july_2004_500_epsg3338_area_0.tif'
    fn_labels_cci = 'tutorials/ak_april_july_2004_500_epsg3338_area_0_cci_labels.tif'
    fn_labels_mtbs = 'tutorials/ak_april_july_2004_500_epsg3338_area_0_mtbs_labels.tif'

    # adapter
    # adapter = DataAdapterTS(
    #     src_satimg=fn_satimg,
    #     src_labels=fn_labels_cci,
    #     label_collection=FireLabelsCollection.CCI,
    #     cci_confidence_level=70
    # )

    # print dates and show label
    # print(adapter.label_dates)
    # adapter.imshow_label(3)

    adapter = DataAdapterTS(
        src_satimg=fn_satimg,
        src_labels=fn_labels_mtbs,
        label_collection=FireLabelsCollection.MTBS,
        mtbs_region=MTBSRegion.ALASKA,
    )

    print(adapter.label_dates)
    print(adapter.satimg_dates)
    print(adapter.nimgs)
    adapter.imshow(14)
    # print(adapter.nimgs)

    # print dates and show satellite image

    # print dates and show label
    # print(adapter.label_dates)
    # adapter.imshow_label(0)
