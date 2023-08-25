import gc
import os

from enum import Enum
from typing import Union

import pandas as pd  # TODO remove

# collections
from mlfire.earthengine.collections import FireLabelsCollection, ModisCollection
from mlfire.earthengine.collections import MTBSRegion, MTBSSeverity

# import utils
from mlfire.utils.functool import lazy_import
from mlfire.utils.time import elapsed_timer
from mlfire.utils.utils_string import band2date_reflectance, band2date_tempsurface
from mlfire.utils.utils_string import band2date_firecci, band2date_mtbs

# lazy imports
_pd = lazy_import('pandas')


class CollectionOPS(Enum):  # TODO rename SourceSelection

    NONE = 0
    REFLECTANCE = 1
    SURFACE_TEMPERATURE = 2
    ALL = 3

    def __and__(self, other):
        return CollectionOPS(self.value & other.value)

    def __eq__(self, other):
        return self.value == other.value

    def __or__(self, other):
        return CollectionOPS(self.value | other.value)


class DatasetLoader(object):  # TODO rename to data set base

    def __init__(self,
                 lst_labels: Union[tuple[str], list[str]],
                 lst_satimgs_reflectance: Union[tuple[str], list[str], None] = None,
                 lst_satimgs_tempsurface: Union[tuple[str], list[str], None] = None,
                 # TODO comment
                 test_ratio: float = .33,
                 val_ratio: float = .0,
                 # TODO comment
                 modis_collection: ModisCollection = ModisCollection.REFLECTANCE,  # TODO change SourceSelection
                 # TODO add here vegetation indices and infrared bands
                 label_collection: FireLabelsCollection = FireLabelsCollection.MTBS,
                 # TODO comment
                 cci_confidence_level: int = 70,
                 mtbs_severity_from: MTBSSeverity = MTBSSeverity.LOW,
                 # TODO comment
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA) -> None:

        self._ds_satimgs_reflectance = None
        self._df_satimgs_reflectance = None

        self._ds_satimgs_tempsurface = None
        self._df_dates_tempsurface = None

        self._ds_labels = None
        self._df_dates_labels = None

        self._map_start_satimgs = None
        self._map_band_id_label = None

        self._nimgs = 0
        self._nbands_label = 0  # TODO rename

        # training, test, and validation data sets

        # TODO move to ts.py?

        self._nfeatures_ts = 0  # TODO rename

        self._ds_training = None
        self._ds_test = None
        self._ds_val = None

        self._test_ratio = None
        self.test_ratio = test_ratio

        self._val_ratio = None
        self.val_ratio = val_ratio

        # properties sources - reflectance, land surface temperature, and labels

        self._lst_satimgs_reflectance = None
        self.lst_satimgs_reflectance = lst_satimgs_reflectance

        self._lst_satimgs_tempsurface = None
        self.lst_satimgs_tempsurface = lst_satimgs_tempsurface

        self._modis_collection = None  # TODO rename
        self.modis_collection = modis_collection  # TODO rename

        self._satimgs_processed = False

        # properties source - labels

        self._lst_labels = None
        self.lst_labels = lst_labels

        self._label_collection = None
        self.label_collection = label_collection

        self._mtbs_region = None
        if label_collection == FireLabelsCollection.MTBS: self.mtbs_region = mtbs_region

        self._mtbs_severity_from = None
        if label_collection == FireLabelsCollection.MTBS: self.mtbs_severity_from = mtbs_severity_from

        self._cci_confidence_level = -1
        if label_collection == FireLabelsCollection.CCI: self.cci_confidence_level = cci_confidence_level

        self._labels_processed = False

    """
    Properties and setter related to sources
    """

    @property
    def modis_collection(self) -> ModisCollection:  # TODO change to list of type CollectionOPS

        return self._modis_collection

    @modis_collection.setter
    def modis_collection(self, collection: ModisCollection) -> None:

        if self.modis_collection == collection:
            return

        self._reset()  # clean up
        self._modis_collection = collection

    @property
    def lst_satimgs_reflectance(self) -> Union[tuple[str], list[str], None]:

        return self._lst_satimgs_reflectance

    @lst_satimgs_reflectance.setter
    def lst_satimgs_reflectance(self, lst_fn: Union[tuple[str], list[str]]) -> None:

        if self._lst_satimgs_reflectance == lst_fn:
            return

        for fn in lst_fn:
            if not os.path.exists(fn):
                raise IOError(f'File {fn} does not exist!')

        self._reset()  # clean up
        self._lst_satimgs_reflectance = lst_fn

    @property
    def lst_satimgs_tempsurface(self) -> Union[tuple[str], list[str]]:

        return self._lst_satimgs_tempsurface

    @lst_satimgs_tempsurface.setter
    def lst_satimgs_tempsurface(self, lst_fn: Union[tuple[str], list[str]]):

        if self._lst_satimgs_tempsurface == lst_fn:
            return

        for fn in lst_fn:
            if not os.path.exists(fn):
                raise IOError(f'File {fn} does not exitst!')

        self._reset()  # clean up
        self._lst_satimgs_tempsurface = lst_fn

    """
    Dates - reflectance, land surface temperature
    """

    @property
    def dates_reflectance(self) -> lazy_import('pandas').DataFrame:

        if self._df_satimgs_reflectance is None:
            self.__processDates_SATELLITE_IMGS_MODIS(selection=CollectionOPS.REFLECTANCE)

        return self._df_satimgs_reflectance

    @property
    def dates_tempsurface(self) -> lazy_import('pandas').DataFrame:

        if self._df_dates_tempsurface is None:
            self.__processDates_SATELLITE_IMGS_MODIS(selection=CollectionOPS.SURFACE_TEMPERATURE)

        return self._df_dates_tempsurface

    @property
    def label_dates(self) -> lazy_import('pandas.DataFrame'):

        if self._df_dates_labels is None:
            self.__processBandDates_LABEL()

        return self._df_dates_labels

    """
    FireCII labels properties
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
    MTBS labels properties
    """

    @property
    def mtbs_region(self) -> MTBSRegion:

        return self._mtbs_region

    @mtbs_region.setter
    def mtbs_region(self, region: MTBSRegion) -> None:

        if self._mtbs_region == region:
            return

        self._reset()
        self._mtbs_region = region

    @property
    def mtbs_severity_from(self) -> MTBSSeverity:

        return self._mtbs_severity_from

    @mtbs_severity_from.setter
    def mtbs_severity_from(self, severity: MTBSSeverity) -> None:

        if self._mtbs_severity_from == severity:
            return

        self._reset()
        self._mtbs_severity_from = severity

    """
    Labels related to wildfires 
    """

    @property
    def lst_labels(self) -> Union[tuple[str], list[str]]:

        return self._lst_labels

    @lst_labels.setter
    def lst_labels(self, lst_labels: Union[tuple[str], list[str]]) -> None:

        if self._lst_labels == lst_labels:
            return

        for fn in lst_labels:
            if not os.path.exists(fn):
                raise IOError('File {} does not exist!'.format(fn))

        self._reset()  # clean up
        self._lst_labels = lst_labels

    @property
    def label_collection(self) -> FireLabelsCollection:

        return self._label_collection

    @label_collection.setter
    def label_collection(self, collection: FireLabelsCollection) -> None:

        if self.label_collection == collection:
            return

        self._reset()  # clean up
        self._label_collection = collection

    @property
    def nbands_label(self) -> int:  # TODO rename?

        if not self._labels_processed:
            self._processMetaData_LABELS()

        return self._nbands_label

    """
    Training, test and validation data sets
    """

    @property
    def test_ratio(self) -> float:

        return self._test_ratio

    @test_ratio.setter
    def test_ratio(self, ratio: float) -> None:

        if self.test_ratio == ratio:
            return

        self._reset()  # clean up
        self._test_ratio = ratio

    @property
    def val_ratio(self) -> float:

        return self._val_ratio

    @val_ratio.setter
    def val_ratio(self, ratio: float) -> None:

        if self.val_ratio == ratio:
            return

        self._reset()
        self._val_ratio = ratio

    def _reset(self):

        del self._ds_training; self._ds_training = None
        del self._ds_test; self._ds_test = None
        del self._ds_val; self._ds_val = None

        del self._df_satimgs_reflectance; self._df_satimgs_reflectance = None
        del self._map_start_satimgs; self._map_start_satimgs = None

        del self._df_dates_tempsurface; self._df_dates_tempsurface = None
        # TODO map start?

        del self._df_dates_labels; self._df_dates_labels = None
        del self._map_band_id_label; self._map_band_id_label = None

        gc.collect()  # invoke garbage collector

        self._nfeatures_ts = 0  # TODO rename nbands_img?
        self._nimgs = 0
        self._nbands_label = 0

        # set flags to false
        self._satimgs_processed = False  # TODO rename?
        self._labels_processed = False

    """
    Load sources - reflectance, land surface temperature, or labels
    """

    @staticmethod
    def __loadGeoTIFF_SOURCES(lst_sources: Union[list[str], tuple[str]]) -> list:

        # lazy import
        gdal = lazy_import('osgeo.gdal')

        lst_ds = []

        for fn in lst_sources:
            try:
                ds = gdal.Open(fn)
            except IOError:
                raise RuntimeWarning(f'Cannot load source {fn}!')

            if ds is None:
                raise RuntimeWarning(f'Source {fn} is empty!')
            lst_ds.append(ds)

        return lst_ds

    def __loadGeoTIFF_REFLECTANCE(self) -> None:

        if self.lst_satimgs_reflectance is None:
            err_msg = 'Satellite data (reflectance) is not set!'
            raise TypeError(err_msg)

        del self._ds_satimgs_reflectance; gc.collect()
        self._ds_satimgs_reflectance = None

        self._ds_satimgs_reflectance = self.__loadGeoTIFF_SOURCES(self.lst_satimgs_reflectance)
        if not self._ds_satimgs_reflectance:
            err_msg = 'Satellite data (reflectance) was not loaded!'
            raise IOError(err_msg)

    def __loadGeoTIFF_SURFACE_TEMPERATURE(self) -> None:

        if self.lst_satimgs_tempsurface is None:
            err_msg = 'Satellite data (land surface temperature) is not set!'
            raise TypeError(err_msg)

        del self._ds_satimgs_tempsurface; gc.collect()
        self._ds_satimgs_tempsurface = None

        self._ds_satimgs_tempsurface = self.__loadGeoTIFF_SOURCES(self.lst_satimgs_tempsurface)
        if not self._ds_satimgs_tempsurface:
            err_msg = 'Satellite data (land surface temperature) was not loaded!'
            raise IOError(err_msg)

    def __loadGeoTIFF_LABELS(self) -> None:

        if not self.lst_labels or self.lst_labels is None:
            err_msg = 'Satellite data (labels - wildfires localization) is not set!'
            raise TypeError(err_msg)

        del self._ds_labels; gc.collect()
        self._ds_labels = None

        self._ds_labels = self.__loadGeoTIFF_SOURCES(self.lst_labels)
        if not self._ds_labels:
            err_msg = 'Satellite data (labels - wildfires localization) is not set!'
            raise IOError(err_msg)

    # TODO rename
    """
    Process meta data (MULTISPECTRAL SATELLITE IMAGE, MODIS)
    """

    def __processDates_MODIS_REFLECTANCE(self) -> None:

        if self._df_satimgs_reflectance is not None:
            return

        unique_dates = set()

        with elapsed_timer('Processing band dates (reflectance)'):

            for i, img_ds in enumerate(self._ds_satimgs_reflectance):
                for rs_id in range(0, img_ds.RasterCount):

                    rs_band = img_ds.GetRasterBand(rs_id + 1)
                    rs_dsc = rs_band.GetDescription()

                    if '_sur_refl_' in rs_dsc:
                        reflec_date = band2date_reflectance(rs_dsc)
                        unique_dates.add((reflec_date, i))

            if not unique_dates:
                err_msg = ''
                raise ValueError(err_msg)

            try:
                df_dates = pd.DataFrame(sorted(unique_dates), columns=['Date', 'Image ID'])
            except MemoryError:
                raise MemoryError

            # clean up
            del unique_dates; gc.collect()

        del self._df_satimgs_reflectance; gc.collect()
        self._df_satimgs_reflectance = df_dates

    def __processDates_MODIS_LAND_SURFACE_TEMPERATURE(self) -> None:

        if self._df_dates_tempsurface is not None:
            return

        lst_dates = []

        with elapsed_timer('Processing dates (land surface temperature)'):

            for i, img_ds in enumerate(self._ds_satimgs_tempsurface):
                for rs_id in range(0, img_ds.RasterCount):

                    rs_band = img_ds.GetRasterBand(rs_id + 1)
                    rs_dsc = rs_band.GetDescription()

                    if 'lst_day_1km' in rs_dsc.lower():
                        tempsurf_date = band2date_tempsurface(rs_dsc)
                        lst_dates.append((tempsurf_date, i))

            if not lst_dates:
                err_msg = 'Surface temperature sources do not contain any useful information about dates'
                raise ValueError(err_msg)

            try:
                df_dates = pd.DataFrame(sorted(lst_dates), columns=['Date', 'Image ID'])
            except MemoryError:
                err_msg = 'DataFrame was not created!'
                raise MemoryError(err_msg)

            # clean up
            del lst_dates; gc.collect()

        del self._df_dates_tempsurface; gc.collect()
        self._df_dates_tempsurface = df_dates

    def __processDates_SATELLITE_IMGS_MODIS(self, selection: CollectionOPS = CollectionOPS.ALL) -> None:

        # processing reflectance (MOD09A1)

        if self.lst_satimgs_reflectance is not None and (selection & CollectionOPS.REFLECTANCE == CollectionOPS.REFLECTANCE):

            if self._ds_satimgs_reflectance is None:
                try:
                    self.__loadGeoTIFF_REFLECTANCE()
                except IOError:
                    err_msg = 'Cannot load any of MOD09A1 sources (reflectance): {}'
                    err_msg = err_msg.format(self.lst_satimgs_reflectance)
                    raise IOError(err_msg)

            try:
                self.__processDates_MODIS_REFLECTANCE()
            except ValueError:
                err_msg = 'Cannot process dates of MOD09A1 sources (reflectance)!'
                raise ValueError(err_msg)

        # processing land surface temperature (MOD11A2)

        if self.lst_satimgs_tempsurface is not None and (selection & CollectionOPS.SURFACE_TEMPERATURE == CollectionOPS.SURFACE_TEMPERATURE):

            if self._ds_satimgs_tempsurface is None:
                try:
                    self.__loadGeoTIFF_SURFACE_TEMPERATURE()
                except IOError:
                    err_msg = 'Cannot load any of MOD11A2 sources (land surface temperature): {}'
                    err_msg = err_msg.format(self.lst_satimgs_tempsurface)
                    raise IOError(err_msg)

            try:
                self.__processDates_MODIS_LAND_SURFACE_TEMPERATURE()
            except ValueError:
                err_msg = 'Cannot process dates of MOD11A2 sources (land surface temperature)!'
                raise ValueError(err_msg)

    # meta data?

    def __processMultiSpectralBands_SATIMG_MODIS_REFLECTANCE(self):

        if self._map_start_satimgs is not None:
            del self._map_start_satimgs; self._map_band_id_satimg = None
            gc.collect()

        map_start_satimg = {}
        pos = 0

        with elapsed_timer('Processing multi spectral bands (satellite images, modis, reflectance)'):

            for id_ds, ds in enumerate(self._ds_satimgs_reflectance):

                last_date = 0  # reset last date
                nbands = ds.RasterCount  # get number of bands in GeoTIFF data set

                for band_id in range(nbands):

                    rs_band = ds.GetRasterBand(band_id + 1)
                    band_dsc = rs_band.GetDescription()
                    band_date = band2date_reflectance(band_dsc)

                    # determine where a multi spectral image begins
                    if '_sur_refl_' in band_dsc and last_date != band_date:
                        map_start_satimg[pos] = (id_ds, band_id + 1); pos += 1
                        last_date = band_date

        if not map_start_satimg:
            raise ValueError('Any satellite image do not contain any useful data!')

        self._map_start_satimgs = map_start_satimg
        self._nimgs = pos

    def _processMetaData_SATELLITE_IMG(self) -> None:

        if self._ds_satimgs_reflectance is None:
            try:
                self.__loadGeoTIFF_REFLECTANCE()
            except IOError:
                raise IOError('Cannot load any following satellite images: {}'.format(self.lst_satimgs_reflectance))

        if self._df_satimgs_reflectance is None:
            try:
                self.__processDates_SATELLITE_IMGS_MODIS()
            except ValueError:
                msg = 'Cannot process band dates of any following satellite images: {}'
                raise ValueError(msg.format(self.lst_satimgs_reflectance))

        try:
            if self.modis_collection == ModisCollection.REFLECTANCE:
                self.__processMultiSpectralBands_SATIMG_MODIS_REFLECTANCE()
            else:
                raise NotImplementedError
        except ValueError or NotImplementedError:
            raise ValueError('Cannot process multi spectral bands meta data!')

        self._satimgs_processed = True

    """
    Process meta data (LABELS)
    """

    def __processBandDates_LABEL_CCI(self) -> None:

        # lazy imports
        pd = lazy_import('pandas')

        if self._df_dates_labels is not None:
            del self._df_dates_labels; self._df_satimgs_reflectance = None
            gc.collect()

        lst = []

        with elapsed_timer('Processing band dates (labels, CCI)'):

            for id_ds, ds in enumerate(self._ds_labels):
                for band_id in range(ds.RasterCount):

                    rs_band = ds.GetRasterBand(band_id + 1)
                    dsc_band = rs_band.GetDescription()

                    if 'ConfidenceLevel' in dsc_band:
                        band_date = band2date_firecci(dsc_band)
                        lst.append((band_date, id_ds))

        if not lst:
            raise ValueError('Label file does not contain any useful data!')

        df_dates = pd.DataFrame(sorted(lst), columns=['Date', 'Image ID'])
        del lst; gc.collect()

        self._df_dates_labels = df_dates

    def __processBandDates_LABEL_MTBS(self) -> None:

        # lazy imports
        pd = lazy_import('pandas')

        if self._df_dates_labels is not None:
            del self._df_dates_labels; self._df_satimgs_reflectance = None
            gc.collect()

        lst = []

        with elapsed_timer('Processing band dates (labels, MTBS)'):

            for id_ds, ds in enumerate(self._ds_labels):
                for band_id in range(ds.RasterCount):

                    rs_band = ds.GetRasterBand(band_id + 1)
                    dsc_band = rs_band.GetDescription()

                    if self.mtbs_region.value in dsc_band:
                        band_date = band2date_mtbs(dsc_band)
                        lst.append((band_date, id_ds))

        if not lst:
            raise ValueError('Label file does not contain any useful data!')

        df_dates = pd.DataFrame(sorted(lst), columns=['Date', 'Image ID'])
        del lst; gc.collect()

        self._df_dates_labels = df_dates

    def __processBandDates_LABEL(self) -> None:

        if self._ds_labels is None:
            try:
                self.__loadGeoTIFF_LABELS()
            except IOError:
                raise IOError('Cannot load any following label sources: {}'.format(self.lst_labels))

        try:
            if self.label_collection == FireLabelsCollection.CCI:
                self.__processBandDates_LABEL_CCI()
            elif self.label_collection == FireLabelsCollection.MTBS:
                self.__processBandDates_LABEL_MTBS()
            else:
                raise NotImplementedError
        except ValueError or NotImplementedError:
            raise ValueError('Cannot process band dates related to labels ({})!'.format(self.label_collection.name))

    def __processLabels_CCI(self) -> None:

        if self._map_band_id_label is None:
            del self._map_band_id_label; self._map_band_id_label = None
            gc.collect()

        map_band_ids = {}
        pos = 0

        with elapsed_timer('Processing fire labels ({})'.format(self.label_collection.name)):

            for id_ds, ds in enumerate(self._ds_labels):
                for band_id in range(ds.RasterCount):

                    rs_band = ds.GetRasterBand(band_id + 1)
                    dsc_band = rs_band.GetDescription()

                    # map id data set for selected region to band id
                    if 'ConfidenceLevel' in dsc_band:
                        map_band_ids[pos] = (id_ds, band_id + 1); pos += 1

        if not map_band_ids:
            raise ValueError('Label file {} does not contain any useful information for a selected region.')

        self._map_band_id_label = map_band_ids
        self._nbands_label = pos

    def __processLabels_MTBS(self) -> None:

        # determine bands and their ids for region selection
        if self._map_band_id_label is None:
            del self._map_band_id_label; self._map_band_id_label = None
            gc.collect()  # invoke garbage collector

        map_band_ids = {}
        pos = 0

        with elapsed_timer('Processing fire labels ({})'.format(self.label_collection.name)):

            for id_ds, ds in enumerate(self._ds_labels):
                for band_id in range(ds.RasterCount):

                    rs_band = ds.GetRasterBand(band_id + 1)
                    dsc_band = rs_band.GetDescription()

                    # map id data set for selected region to band id
                    if self.mtbs_region.value in dsc_band:
                        map_band_ids[pos] = (id_ds, band_id + 1); pos += 1

        if not map_band_ids:
            msg = 'Any labels ({}) do not contain any useful information: {}'.format(self.label_collection.name, self._lst_labels)
            raise ValueError(msg)

        self._map_band_id_label = map_band_ids
        self._nbands_label = pos

    def _processMetaData_LABELS(self) -> None:

        if self._ds_labels is None:
            try:
                self.__loadGeoTIFF_LABELS()
            except IOError:
                raise IOError('Cannot load any following label sources: {}'.format(self.lst_labels))

        if self._df_dates_labels is None:
            try:
                self.__processBandDates_LABEL()
            except ValueError or AttributeError:
                raise ValueError('Cannot process band dates related labels ({})!'.format(self.label_collection.name))

        try:
            if self.label_collection == FireLabelsCollection.CCI:
                self.__processLabels_CCI()
            elif self.label_collection == FireLabelsCollection.MTBS:
                self.__processLabels_MTBS()
            else:
                raise NotImplementedError
        except ValueError or NotImplementedError:
            msg = 'Cannot process labels meta data!'
            raise ValueError(msg)

        # set that everything is done
        self._labels_processed = True

    """
    Magic methods
    """

    def __len__(self) -> int:

        if not self._satimgs_processed:
            self._processMetaData_SATELLITE_IMG()

        return self._nimgs


# tests
if __name__ == '__main__':

    VAR_DATA_DIR = 'data/tifs'

    VAR_PREFIX_IMG_REFLECTANCE = 'ak_reflec_january_december_{}_100km'
    VAR_PREFIX_IMG_LABELS = 'ak_january_december_{}_100km'

    VAR_LST_SATIMGS = []
    VAR_LST_LABELS_MTBS = []

    for year in range(2004, 2006):
        VAR_PREFIX_IMG_REFLECTANCE_YEAR = VAR_PREFIX_IMG_REFLECTANCE.format(year)
        VAR_PREFIX_IMG_LABELS_YEAR = VAR_PREFIX_IMG_LABELS.format(year)

        fn_satimg_reflec = '{}_epsg3338_area_0.tif'.format(VAR_PREFIX_IMG_REFLECTANCE_YEAR)
        fn_satimg_reflec = os.path.join(VAR_DATA_DIR, fn_satimg_reflec)
        VAR_LST_SATIMGS.append(fn_satimg_reflec)

        fn_labels_mtbs = '{}_epsg3338_area_0_mtbs_labels.tif'.format(VAR_PREFIX_IMG_LABELS_YEAR)
        fn_labels_mtbs = os.path.join(VAR_DATA_DIR, fn_labels_mtbs)
        VAR_LST_LABELS_MTBS.append(fn_labels_mtbs)

    # setup of data set loader
    dataset_loader = DatasetLoader(
        lst_satimgs_reflectance=VAR_LST_SATIMGS,
        lst_labels=VAR_LST_LABELS_MTBS
    )

    print(dataset_loader.label_dates)
