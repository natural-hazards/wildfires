import calendar
import copy
import datetime
import gc
import os.path

import cv2 as opencv
import numpy as np
import pandas as pd

from enum import Enum

from osgeo import gdal

from scipy import stats
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

from earthengine.ds import FireLabelsCollection, ModisIndex, ModisReflectanceSpecralBands
from earthengine.ds import MTBSRegion, MTBSSeverity
from utils.utils_string import band2date_firecci, band2date_mtbs, band2data_reflectance

# time series transformation
from procs.fft import TransformFFT
from procs.pca import TransformPCA, FactorOP

# utils imports
from utils.time import elapsed_timer
from utils.plots import imshow


class DatasetTransformOP(Enum):

    NONE = 0
    STANDARTIZE_ZSCORE = 1
    FFT = 2
    PCA = 4
    SAVITZKY_GOLAY = 8


class DataAdapterTS(object):

    def __init__(self,
                 src_satimg: str,
                 src_labels: str,
                 # TODO src_satimg_test and src_labels_test
                 ds_start_date: datetime.date = None,
                 ds_end_date: datetime.date = None,
                 ds_test_ratio: float = 0.33,
                 transform_ops: list[DatasetTransformOP] = (DatasetTransformOP.NONE,),
                 nfactors_pca: int = None,
                 pca_ops: list[FactorOP] = (FactorOP.NONE,),
                 fft_nfeatures: int = None,
                 savgol_polyorder: int = 1,
                 savgol_winlen: int = 5,
                 modis_collection: ModisIndex = ModisIndex.REFLECTANCE,
                 label_collection: FireLabelsCollection = FireLabelsCollection.CCI,
                 cci_confidence_level: int = 70,
                 mtbs_severity_from: MTBSSeverity = MTBSSeverity.LOW,
                 mtbs_region: MTBSRegion = None,
                 verbose: bool = False):

        self._ds_satimg = None
        self._df_dates_satimg = None

        self._ds_labels = None
        self._df_dates_labels = None

        self._map_band_id_label = None
        self._map_start_satimgs = None

        self._nimages = -1
        self._nbands_labels = -1

        # training and test data set

        self._ds_training = None
        self._ds_test = None

        self._ds_test_ratio = None
        self.ds_test_ratio = ds_test_ratio

        # properties of data transformation

        self._lst_transform_ops = None
        self._transform_ops = 0
        self.transform_ops = transform_ops

        self._savgol_polyorder = None
        self.savgol_polyorder = savgol_polyorder

        self._savgol_winlen = None
        self.savgol_winlen = savgol_winlen

        self._fft_nfeatures = None
        self.fft_nfeatures = fft_nfeatures

        self._nfactors_pca = 0
        self.nfactors_pca = nfactors_pca

        self._lst_pca_ops = None
        self._pca_ops = 0
        self.pca_ops = pca_ops

        # properties of training and test data set

        self._ds_start_date = None
        if ds_start_date is not None: self.ds_start_date = ds_start_date

        self._ds_end_date = None
        if ds_end_date is not None: self._ds_end_date = ds_end_date

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

        # verbose
        self._verbose = False
        self.verbose = verbose

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
    Time series transformation properties
    """

    @property
    def transform_ops(self) -> list[DatasetTransformOP]:

        return self._lst_transform_ops

    @transform_ops.setter
    def transform_ops(self, lst_ops: list[DatasetTransformOP]) -> None:

        if self._lst_transform_ops == lst_ops:
            return

        self.__reset()

        self._transform_ops = 0
        self._lst_transform_ops = lst_ops
        for op in lst_ops: self._transform_ops |= op.value

    @property
    def savgol_polyorder(self) -> int:

        return self._savgol_polyorder

    @savgol_polyorder.setter
    def savgol_polyorder(self, order: int) -> None:

        if self._savgol_polyorder == order:
            return

        self.__reset()
        self._savgol_polyorder = order

    @property
    def savgol_winlen(self) -> int:

        return self._savgol_winlen

    @savgol_winlen.setter
    def savgol_winlen(self, winlen: int) -> None:

        if self._savgol_winlen == winlen:
            return

        self.__reset()
        self._savgol_winlen = winlen

    @property
    def fft_nfeatures(self) -> int:

        return self._fft_nfeatures

    @fft_nfeatures.setter
    def fft_nfeatures(self, n: int) -> None:

        if self._fft_nfeatures == n:
            return

        self.__reset()
        self._fft_nfeatures = n

    @property
    def nfactors_pca(self) -> int:

        return self._nfactors_pca

    @nfactors_pca.setter
    def nfactors_pca(self, n) -> None:

        if self._nfactors_pca == n:
            return

        self.__reset()
        self._nfactors_pca = n

    @property
    def pca_ops(self) -> list[FactorOP]:

        return self._lst_pca_ops

    @pca_ops.setter
    def pca_ops(self, lst_ops: list[FactorOP]) -> None:

        if self._lst_pca_ops == lst_ops:
            return

        self.__reset()

        self._lst_pca_ops = lst_ops
        self._pca_ops = 0
        # set ops flag
        for op in lst_ops: self._pca_ops |= op.value

    """
    Training and test data set properties
    """

    @property
    def ds_training(self) -> tuple:

        if self._ds_training is None:
            self.__createDatasets()

        return self._ds_training

    @property
    def ds_test(self) -> tuple:

        if self._ds_test is None:
            self.__createDatasets()

        return self._ds_test

    @property
    def ds_start_date(self) -> datetime.date:

        return self._ds_start_date

    @ds_start_date.setter
    def ds_start_date(self, d: datetime.date) -> None:

        if self._ds_start_date == d:
            return

        del self._ds_training; self._ds_training = None
        del self._ds_test; self._ds_test = None
        gc.collect()

        self._ds_start_date = d

    @property
    def ds_end_date(self) -> datetime.date:

        return self._ds_end_date

    @ds_end_date.setter
    def ds_end_date(self, d: datetime.date) -> None:

        if self._ds_end_date == d:
            return

        del self._ds_training; self._ds_training = None
        del self._ds_test; self._ds_test = None
        gc.collect()

        self._ds_end_date = d

    @property
    def ds_test_ratio(self) -> float:

        return self._ds_test_ratio

    @ds_test_ratio.setter
    def ds_test_ratio(self, ratio: float) -> None:

        if self._ds_test_ratio == ratio:
            return

        self.__reset()
        self._ds_test_ratio = ratio

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

        del self._df_dates_satimg; self._df_dates_satimg = None
        del self._map_start_satimgs; self._map_start_satimgs = None

        del self._ds_training; self._ds_training = None
        del self._ds_test; self._ds_test = None

        # invoke garbage collector
        gc.collect()

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

    @property
    def verbose(self) -> bool:

        return self._verbose

    @verbose.setter
    def verbose(self, flg: bool):

        self._verbose = flg

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

        with elapsed_timer('Processing band dates (satellite images, reflectance)'):
            # processing band dates related to multi spectral images (reflectance)
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
    Creating data set
    """

    def __loadLabels_CCI(self) -> (np.ndarray, np.ndarray):

        start_date = self.ds_start_date
        start_date = datetime.date(year=start_date.year, month=start_date.month, day=1)

        start_band_id = self._df_dates_labels.index[self._df_dates_labels['Date'] == start_date][0]
        start_band_id = self._map_band_id_label[start_band_id]

        end_date = self.ds_end_date
        end_date = datetime.date(year=end_date.year, month=end_date.month, day=1)

        end_band_id = self._df_dates_labels.index[self._df_dates_labels['Date'] == end_date][0]
        end_band_id = self._map_band_id_label[end_band_id]

        lst_confidence = []
        lst_masks = []

        for band_id in range(start_band_id, end_band_id + 1, 2):

            lst_confidence.append(self._ds_labels.GetRasterBand(band_id).ReadAsArray())
            lst_masks.append(self._ds_labels.GetRasterBand(band_id + 1).ReadAsArray())

        # mask
        np_mask = np.array(lst_masks)
        # clean up
        del lst_masks; gc.collect()

        np_mask[np_mask <= -1] = 1
        np_mask = np.logical_not(np.max(np_mask, axis=0))
        gc.collect()  # invoke garbage collector

        # convert confidence to labels (using threshold)
        np_confidence = np.array(lst_confidence)
        del lst_confidence; gc.collect()

        np_confidence = np.max(np_confidence, axis=0)
        gc.collect()  # invoke garbage collector

        np_confidence[np_confidence < self.cci_confidence_level] = 0
        np_confidence[np_confidence >= self.cci_confidence_level] = 1

        return np_confidence, np_mask

    def __loadLabels_MTBS(self) -> (np.ndarray, np.ndarray):

        start_date = self.ds_start_date
        start_date = datetime.date(year=start_date.year, month=1, day=1)

        start_band_id = self._df_dates_labels.index[self._df_dates_labels['Date'] == start_date][0]
        start_band_id = self._map_band_id_label[start_band_id]

        end_date = self.ds_end_date
        end_date = datetime.date(year=end_date.year, month=1, day=1)

        if start_date == end_date:
            end_band_id = start_band_id
        else:
            end_band_id = self._df_dates_labels.index[self._df_dates_labels['Date'] == end_date][0]
            end_band_id = self._map_band_id_label[end_band_id]

        lst_labels = []
        lst_masks = []

        for band_id in range(start_band_id, end_band_id + 1):

            # invoke garbage collector
            gc.collect()

            np_severity = self._ds_labels.GetRasterBand(band_id).ReadAsArray()

            np_label = np.zeros(shape=np_severity.shape, dtype=np_severity.dtype)
            np_label[np.logical_and(np_severity >= self.mtbs_severity_from.value, np_severity <= MTBSSeverity.HIGH.value)] = 1
            lst_labels.append(np_label)

            np_mask = np.ones(shape=np_severity.shape, dtype=np_severity.dtype)
            np_mask[np_severity == MTBSSeverity.NON_MAPPING_AREA.value] = 0
            lst_masks.append(np_mask)

        # create array of labels
        np_labels = np.array(lst_labels)
        del lst_labels; gc.collect()

        np_label = np.max(np_labels, axis=0)
        gc.collect()  # invoke garbage collector

        # create binary mask
        np_masks = np.array(lst_masks)
        del lst_masks; gc.collect()

        np_mask = np.max(np_masks, axis=0)
        gc.collect()

        return np_label, np_mask

    def __createLabels(self) -> (np.ndarray, np.ndarray):

        if not self._labels_processed:
            try:
                self.__processLabels()
            except IOError or ValueError:
                raise IOError('Cannot process the label file ({})!'.format(self.src_labels))

        if self.label_collection == FireLabelsCollection.CCI:
            return self.__loadLabels_CCI()
        elif self.label_collection == FireLabelsCollection.MTBS:
            return self.__loadLabels_MTBS()
        else:
            raise NotImplementedError

    def __loadTimeSeries_REFLECTANCE(self, mask: np.ndarray = None) -> np.ndarray:

        start_date = self.ds_start_date
        if start_date not in self._df_dates_satimg['Date'].values:
            raise AttributeError('Start date does not correspond any band!')

        end_date = self.ds_end_date
        if end_date not in self._df_dates_satimg['Date'].values:
            raise AttributeError('End date does not correspont any band!')

        start_band_id = self._df_dates_satimg['Date'].index[self._df_dates_satimg['Date'] == start_date][0]
        start_band_id = self._map_start_satimgs[start_band_id]

        end_band_id = self._df_dates_satimg['Date'].index[self._df_dates_satimg['Date'] == end_date][0]
        end_band_id = self._map_start_satimgs[end_band_id]

        lst_bands = []

        for img_start in range(start_band_id, end_band_id + 1, 7):
            # getting reflectance bands
            for band_id in range(img_start, img_start + 7):

                rs_band = self._ds_satimg.GetRasterBand(band_id)
                band_dsc = rs_band.GetDescription()

                with elapsed_timer('Loading spectral band {} (satellite image)'.format(band_dsc)):
                    np_band = rs_band.ReadAsArray()

                # TODO process nan values
                lst_bands.append(np_band)

        img_ts = np.array(lst_bands)
        del lst_bands; gc.collect()

        img_ts = img_ts.reshape((img_ts.shape[0], -1)).T
        ts_reflectance = img_ts[mask.reshape(-1) == 1, :].astype(np.float32)

        return ts_reflectance

    def __createTimeSeries(self, mask: np.ndarray = None) -> np.ndarray:

        if not self._satimg_processed:
            try:
                self.__processMetaData_SATELLITE_IMG()
            except IOError or ValueError:
                raise IOError('Cannot process the satellite image ({})'.format(self.src_satimg))

        if self.modis_collection == ModisIndex.REFLECTANCE:
            return self.__loadTimeSeries_REFLECTANCE(mask)
        else:
            raise NotImplementedError

    def __transformTimeSeries_REFLECTANCE(self, ts: np.ndarray) -> np.ndarray:

        nbands = 7

        # standardize data using z-score
        if self._transform_ops & DatasetTransformOP.STANDARTIZE_ZSCORE.value == DatasetTransformOP.STANDARTIZE_ZSCORE.value:

            with elapsed_timer('Standardizing data'):
                for band_id in range(nbands):
                    # TODO avoid time series with std = 0
                    ts[:, band_id::nbands] = stats.zscore(ts[:, band_id::nbands], axis=1)

        if self._transform_ops & DatasetTransformOP.SAVITZKY_GOLAY.value == DatasetTransformOP.SAVITZKY_GOLAY.value:

            with elapsed_timer('Smoothing time series using Savitzky Golay filter'):

                for band_id in range(nbands):
                    ts[:, band_id::nbands] = savgol_filter(
                        ts[:, band_id::nbands],
                        window_length=self.savgol_winlen,
                        polyorder=self.savgol_polyorder
                    )

        # Transforming data to frequency domain
        if self._transform_ops & DatasetTransformOP.FFT.value == DatasetTransformOP.FFT.value:

            with elapsed_timer('Transforming time series to frequency domain'):

                mod_ts = np.zeros(shape=(ts.shape[0], 7 * self.fft_nfeatures))

                for band_id in range(nbands):
                    # transforming data to frequency domain
                    ts_band = ts[:, band_id::nbands]
                    transformer_fft = TransformFFT(
                        nfeatures=self.fft_nfeatures
                    )
                    mod_ts[:, band_id::nbands] = transformer_fft.transform(ts_band)

                    # invoke garbage collector
                    gc.collect()

                del ts; gc.collect()
                ts = mod_ts

        return ts

    def __transformTimeSeries(self, ts: np.ndarray) -> np.ndarray:

        if self.modis_collection == ModisIndex.REFLECTANCE:
            return self.__transformTimeSeries_REFLECTANCE(ts)
        else:
            raise NotImplementedError

    # Principal component analysis

    def __principalCompoenentAnalysis_REFLECTANCE(self, ts_training: np.ndarray, ts_test: np.ndarray) -> (np.ndarray, np.ndarray):

        nbands = 7

        if DatasetTransformOP.STANDARTIZE_ZSCORE not in self._lst_transform_ops:
            # standardize using z-score before applying reduction using PCA
            with elapsed_timer('Standardizing data'):
                for ts in (ts_training, ts_test):
                    for band_id in range(nbands):
                        # TODO avoid time series with std = 0
                        ts[:, band_id::nbands] = stats.zscore(ts[:, band_id::nbands], axis=1)

        with elapsed_timer('Transforming data using PCA'):

            # transforming training and test data set

            lst_transformers = []
            nlatent_factors_found = 0

            for band_id in range(nbands):

                transformer_pca = TransformPCA(
                    train_ds=ts_training[:, band_id::nbands],
                    factor_ops=self._lst_pca_ops,
                    nlatent_factors=self.nfactors_pca,
                    verbose=True
                )

                transformer_pca.fit()

                nlatent_factors_found = max(nlatent_factors_found, transformer_pca.nlatent_factors)
                lst_transformers.append(transformer_pca)

            nsamples_training = ts_training.shape[0]
            reduced_ts_training = np.zeros(shape=(nsamples_training, nbands * nlatent_factors_found), dtype=ts_training.dtype)

            # transforming training data set
            for band_id in range(nbands):

                transformer_pca = lst_transformers[band_id]

                # retrain if required
                if transformer_pca.nlatent_factors < nlatent_factors_found:
                    transformer_pca.nlatent_factors_user = nlatent_factors_found

                    # explicitly required number of latent factors
                    if isinstance(self._lst_transform_ops, tuple):
                        mod_lst_pca_ops = list(self._lst_pca_ops)
                    else:
                        mod_lst_pca_ops = copy.deepcopy(self._lst_transform_ops)
                    if FactorOP.TEST_CUMSUM in mod_lst_pca_ops: mod_lst_pca_ops.remove(FactorOP.TEST_CUMSUM)
                    if FactorOP.TEST_BARTLETT in mod_lst_pca_ops: mod_lst_pca_ops.remove(FactorOP.TEST_BARTLETT)
                    if FactorOP.USER_SET not in mod_lst_pca_ops: mod_lst_pca_ops.append(FactorOP.USER_SET)
                    transformer_pca.factor_ops = mod_lst_pca_ops

                    # set test
                    transformer_pca.fit()

                # transform training and test data set
                reduced_ts_training[:, band_id::nbands] = transformer_pca.transform(ts_training[:, band_id::nbands])

            # clean up and invoke garbage collector
            del ts_training; gc.collect()

            if ts_test is not None:
                nsamples_test = ts_test.shape[0]
                reduced_ts_test = np.zeros(shape=(nsamples_test, nbands * nlatent_factors_found), dtype=ts_test.dtype)

                for band_id in range(nbands):
                    transformer_pca = lst_transformers[band_id]
                    reduced_ts_test[:, band_id::nbands] = transformer_pca.transform(ts_test[:, band_id::nbands])

                # clean up and invoke garbage collector
                del ts_test; gc.collect()
            else:
                reduced_ts_test = None

        return reduced_ts_training, reduced_ts_test

    def __principalComponentAnalysis(self, ts_training: np.ndarray, ts_test: np.ndarray) -> (np.ndarray, np.ndarray):

        if self.modis_collection == ModisIndex.REFLECTANCE:
            return self.__principalCompoenentAnalysis_REFLECTANCE(ts_training, ts_test)
        else:
            raise NotImplementedError

    def __createDatasets(self) -> None:

        if self._ds_training and self._ds_test:
            return

        # load labels
        try:
            with elapsed_timer('Creating labels'):
                labels, mask = self.__createLabels()
        except IOError:
            raise IOError('Cannot process the label file ({})!'.format(self.src_labels))

        # reshape labels to be 1D vector
        labels = labels.reshape(-1)[mask.reshape(-1) == 1]

        if self.verbose:
            nbackground_pixels = np.count_nonzero(labels == 0)
            percent_bp = nbackground_pixels / labels.shape[0]

            nfire_pixels = np.count_nonzero(labels == 1)
            percent_fp = nfire_pixels / labels.shape[0]

            print('#background pixels {} ({:.2f}%)'.format(nbackground_pixels, percent_bp * 100.))
            print('#fire pixels {} ({:.2f}%)'.format(nfire_pixels, percent_fp * 100.))

        # load time series used mask
        try:
            with elapsed_timer('Creating time series date set'):
                ts = self.__createTimeSeries(mask)
        except IOError:
            raise IOError('Cannot process the satellite image ({})'.format(self.src_satimg))

        # time series transformation
        ts = self.__transformTimeSeries(ts)

        # split data set into training and test
        if self._ds_test_ratio > 0.:
            ts_train, ts_test, labels_train, labels_test = train_test_split(
                ts,
                labels,
                test_size=self._ds_test_ratio,
                random_state=42
            )
        else:
            ts_train = ts
            labels_train = labels
            ts_test = None
            labels_test = None

        if self._transform_ops & DatasetTransformOP.PCA.value == DatasetTransformOP.PCA.value:
            ts_train, ts_test = self.__principalComponentAnalysis(ts_train, ts_test)

        self._ds_training = (ts_train, labels_train)
        self._ds_test = (ts_test, labels_test)

    """
    Display functionality (SATELLITE IMAGE, MODIS)
    """

    def __getSatelliteImageArray_REFLECTANCE(self, img_id: int) -> np.ndarray:

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

        return img

    def __getSatelliteImageArray(self, img_id: int) -> np.ndarray:

        if self.modis_collection == ModisIndex.REFLECTANCE:
            return self.__getSatelliteImageArray_REFLECTANCE(img_id)
        else:
            raise NotImplementedError

    def __imshow_SATELLITE_IMG_REFLECTANCE(self, img_id) -> None:

        if not self._satimg_processed:
            try:
                self.__processMetaData_SATELLITE_IMG()
            except IOError or ValueError:
                raise IOError('Cannot process the satellite image ({})'.format(self.src_satimg))

        if img_id < 0:
            raise ValueError('Wrong band indentificator! It must be value greater or equal to 0!')

        if self.nimgs - 1 < img_id:
            raise ValueError('Wrong band indentificator! GeoTIFF contains only {} bands.'.format(self.nimgs))

        img = self.__getSatelliteImageArray(img_id)
        # increase image brightness
        enhanced_img = opencv.convertScaleAbs(img, alpha=5., beta=5.)

        str_title = 'MODIS Reflectance ({})'.format(self.satimg_dates.iloc[img_id]['Date'])
        opencv.imshow(str_title, enhanced_img)
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
            except IOError or ValueError:
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

        label_date = self.label_dates.iloc[band_id]['Date']
        str_title = 'CCI labels ({}, {})'.format(label_date.year, calendar.month_name[label_date.month])

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
        label_date = self.label_dates.iloc[band_id]['Date']
        str_title = 'MTBS labels ({} {})'.format(self.mtbs_region.name, label_date.year)

        opencv.imshow(str_title, img)
        opencv.waitKey(0)

    def imshow_label(self, band_id: int) -> None:

        if self.label_collection == FireLabelsCollection.CCI:
            self.__imshow_label_CCI(band_id)
        elif self.label_collection == FireLabelsCollection.MTBS:
            self.__imshow_label_MTBS(band_id)
        else:
            raise NotImplementedError

    """
    Display functionality satellite image combined with labels
    """

    def __getLabelsForSatImgDate_CCI(self, img_id: int) -> np.ndarray:

        date_satimg = self._df_dates_satimg.iloc[img_id]['Date']
        date_satimg = datetime.date(year=date_satimg.year, month=date_satimg.month, day=1)

        # get index of corresponding labels
        label_index = self._df_dates_labels.index[self._df_dates_labels['Date'] == date_satimg][0]
        rs_band_id = self._map_band_id_label[label_index]

        confidence_level = self._ds_labels.GetRasterBand(rs_band_id).ReadAsArray()

        confidence_level[confidence_level < self.cci_confidence_level] = 0
        confidence_level[confidence_level >= self.cci_confidence_level] = 1

        return confidence_level

    def __getLabelsForSatImgDate_MTBS(self, img_id: int) -> np.ndarray:

        date_satimg = self._df_dates_satimg.iloc[img_id]['Date']
        date_satimg = datetime.date(year=date_satimg.year, month=1, day=1)

        # get index of corresponding labels
        label_index = self._df_dates_labels.index[self._df_dates_labels['Date'] == date_satimg][0]
        rs_band_id = self._map_band_id_label[label_index]

        # get severity
        severity_fire = self._ds_labels.GetRasterBand(rs_band_id).ReadAsArray()
        label = np.zeros(shape=severity_fire.shape)
        label[np.logical_and(severity_fire >= self.mtbs_severity_from.value, severity_fire <= MTBSSeverity.HIGH.value)] = 1

        return label

    def __getLabelsForSource(self, img_id: int) -> np.ndarray:

        if self.label_collection == FireLabelsCollection.CCI:
            return self.__getLabelsForSatImgDate_CCI(img_id)
        elif self.label_collection == FireLabelsCollection.MTBS:
            return self.__getLabelsForSatImgDate_MTBS(img_id)
        else:
            raise NotImplementedError

    def __imshow_SATELLITE_IMG_REFLECTANCE_WITH_LABELS(self, img_id: int):

        if not self._labels_processed:
            try:
                self.__processLabels()
            except IOError or ValueError:
                raise IOError('Cannot process the label file ({})!'.format(self.src_labels))

        if not self._satimg_processed:
            try:
                self.__processMetaData_SATELLITE_IMG()
            except IOError or ValueError:
                raise IOError('Cannot process the satellite image ({})'.format(self.src_satimg))

        satimg = self.__getSatelliteImageArray(img_id)
        # increase image brightness
        satimg = opencv.convertScaleAbs(satimg, alpha=5., beta=5.)

        labels = self.__getLabelsForSource(img_id)
        satimg[labels == 1, 0:2] = 0; satimg[labels == 1, 2] = 255

        ref_date = self.satimg_dates.iloc[img_id]['Date']
        str_title = 'MODIS Reflectance ({}, labels={})'.format(ref_date, self.label_collection.name)

        imshow(src=satimg, title=str_title, figsize=(6.5, 6.5), show=True)

    def imshow_with_labels(self, img_id: int) -> None:

        if self.modis_collection == ModisIndex.REFLECTANCE:
            self.__imshow_SATELLITE_IMG_REFLECTANCE_WITH_LABELS(img_id)
        else:
            raise NotImplementedError


if __name__ == '__main__':

    DATA_DIR = 'pipelines/ts/data/'

    # fn_satimg = os.path.join(DATA_DIR, 'ak_reflec_january_december_2004_500_epsg3338_area_0.tif')
    # fn_labels_cci = os.path.join(DATA_DIR, 'ak_april_july_2004_500_epsg3338_area_0_cci_labels.tif')
    # fn_labels_mtbs = os.path.join(DATA_DIR, 'ak_january_december_2004_500_epsg3338_area_0_mtbs_labels.tif')
    # fn_satimg = os.path.join(DATA_DIR, 'ak_reflec_january_december_2004_850_850_km_epsg3338_area_0.tif')
    # fn_labels_mtbs = os.path.join(DATA_DIR, 'ak_january_december_2004_850_850_km_epsg3338_area_0_mtbs_labels.tif')
    # fn_labels_cci = os.path.join(DATA_DIR, 'ak_january_december_2004_850_850_km_epsg3338_area_0_cci_labels.tif')

    # # adapter
    # adapter = DataAdapterTS(
    #     src_satimg=fn_satimg,
    #     src_labels=fn_labels_cci,
    #     label_collection=FireLabelsCollection.CCI,
    #     cci_confidence_level=70
    # )
    #
    # label_dates = adapter.label_dates
    # print(label_dates)
    # satimg_dates = adapter.satimg_dates
    # print(satimg_dates)
    #
    # print('start date {}'.format(adapter.satimg_dates.iloc[0]['Date']))
    # adapter.ds_start_date = adapter.satimg_dates.iloc[0]['Date']
    # print('end date {}'.format(adapter.satimg_dates.iloc[14]['Date']))
    # adapter.ds_end_date = adapter.satimg_dates.iloc[14]['Date']
    #
    # adapter.getDataset()

    adapter = DataAdapterTS(
        src_satimg=fn_satimg,
        src_labels=fn_labels_cci,
        label_collection=FireLabelsCollection.CCI,
        mtbs_region=MTBSRegion.ALASKA,
        cci_confidence_level=70,
        verbose=True
    )

    print('start date {}'.format(adapter.satimg_dates.iloc[0]['Date']))
    adapter.ds_start_date = adapter.satimg_dates.iloc[0]['Date']
    print('end date {}'.format(adapter.satimg_dates.iloc[14]['Date']))
    adapter.ds_end_date = adapter.satimg_dates.iloc[14]['Date']

    print(adapter.satimg_dates)
    print(adapter.label_dates)
    # adapter.getDataset()

    for i in range(20, 40):
        adapter.imshow_with_labels(i)
