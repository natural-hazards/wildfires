import gc
import os

from typing import Union

# collections
from mlfire.earthengine.collections import FireLabelsCollection, ModisIndex
from mlfire.earthengine.collections import MTBSRegion, MTBSSeverity

# import utils
from mlfire.utils.functool import lazy_import
from mlfire.utils.time import elapsed_timer
from mlfire.utils.utils_string import band2date_reflectance
from mlfire.utils.utils_string import band2date_firecci, band2date_mtbs


class DatasetLoader(object):

    def __init__(self,
                 lst_satimgs: Union[tuple[str], list[str]],
                 lst_labels: Union[tuple[str], list[str]],
                 test_ratio: float = .33,
                 modis_collection: ModisIndex = ModisIndex.REFLECTANCE,
                 label_collection: FireLabelsCollection = FireLabelsCollection.MTBS,
                 cci_confidence_level: int = 70,
                 mtbs_severity_from: MTBSSeverity = MTBSSeverity.LOW,
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA):

        self._ds_satimgs = None
        self._df_dates_satimgs = None

        self._ds_labels = None
        self._df_dates_labels = None

        self._map_start_satimgs = None
        self._map_band_id_label = None

        self._nimgs = 0
        self._nbands_label = 0

        # training and test data set

        self._ds_training = None
        self._ds_test = None

        self._test_ratio = None
        self.test_ratio = test_ratio

        # properties (multi spectral images - MODIS)

        self._lst_satimgs = None
        self.lst_satimgs = lst_satimgs

        self._modis_collection = None
        self.modis_collection = modis_collection

        self._satimgs_processed = False

        # properties (labels)

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
    Multispectral images properties
    """

    @property
    def lst_satimgs(self) -> Union[tuple[str], list[str]]:

        return self._lst_satimgs

    @lst_satimgs.setter
    def lst_satimgs(self, lst_fn: Union[tuple[str], list[str]]) -> None:

        if self._lst_satimgs == lst_fn:
            return

        for fn in lst_fn:
            if not os.path.exists(fn):
                raise IOError('File {} does not exist!'.format(fn))

        self.__reset()  # clean up
        self._lst_satimgs = lst_fn

    @property
    def modis_collection(self) -> ModisIndex:

        return self._modis_collection

    @modis_collection.setter
    def modis_collection(self, collection: ModisIndex) -> None:

        if self.modis_collection == collection:
            return

        self.__reset()  # clean up
        self._modis_collection = collection

    @property
    def satimg_dates(self) -> lazy_import('pandas.DataFrame'):

        if self._df_dates_labels is None:
            self.__processBandDates_SATELLITE_IMGS()

        return self._df_dates_satimgs

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
    MTBS labels properties
    """

    @property
    def mtbs_region(self) -> MTBSRegion:

        return self._mtbs_region

    @mtbs_region.setter
    def mtbs_region(self, region: MTBSRegion) -> None:

        if self._mtbs_region == region:
            return

        self.__reset()
        self._mtbs_region = region

    @property
    def mtbs_severity_from(self) -> MTBSSeverity:

        return self._mtbs_severity_from

    @mtbs_severity_from.setter
    def mtbs_severity_from(self, severity: MTBSSeverity) -> None:

        if self._mtbs_severity_from == severity:
            return

        self.__reset()
        self._mtbs_severity_from = severity

    """
    Labels related to multispectral images
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

        self.__reset()  # clean up
        self._lst_labels = lst_labels

    @property
    def label_collection(self) -> FireLabelsCollection:

        return self._label_collection

    @label_collection.setter
    def label_collection(self, collection: FireLabelsCollection) -> None:

        if self.label_collection == collection:
            return

        self.__reset()  # clean up
        self._label_collection = collection

    @property
    def nbands_label(self) -> int:

        if not self._labels_processed:
            self._processMetaData_LABELS()

        return self._nbands_label

    @property
    def label_dates(self) -> lazy_import('pandas.DataFrame'):

        if self._df_dates_labels is None:
            self.__processBandDates_LABEL()

        return self._df_dates_labels

    """
    Training and test data sets
    """

    @property
    def test_ratio(self) -> float:

        return self._test_ratio

    @test_ratio.setter
    def test_ratio(self, ratio: float) -> None:

        if self.test_ratio == ratio:
            return

        self.__reset()  # clean up
        self._test_ratio = ratio

    def __reset(self):

        del self._ds_training; self._ds_training = None
        del self._ds_test; self._ds_test = None

        del self._df_dates_satimgs; self._df_dates_satimgs = None
        del self._map_start_satimgs; self._map_start_satimgs = None

        del self._df_dates_labels; self._df_dates_labels = None
        del self._map_band_id_label; self._map_band_id_label = None

        gc.collect()  # invoke garbage collector

        self._nimgs = 0
        self._nbands_label = 0

        # set flags to false
        self._satimgs_processed = False
        self._labels_processed = False

    """
    IO functionality
    """

    def __loadGeoTIFF_SATELLITE_IMGS(self) -> None:

        # lazy import
        gdal = lazy_import('osgeo.gdal')

        if self._lst_satimgs is None:
            raise IOError('Multispectral satellite data is not set!')

        del self._ds_satimgs; gc.collect()
        self._ds_satimgs = []

        for fn in self._lst_satimgs:
            try:
                ds = gdal.Open(fn)
            except IOError:
                IOError('Cannot load source {}!'.format(fn))
                continue

            self._ds_satimgs.append(ds)

        if len(self._ds_satimgs) == 0:
            raise IOError('Cannot load any of following sources {}!'.format(self._lst_satimgs))

    def __loadGeoTIFF_LABELS(self) -> None:

        # lazy import
        gdal = lazy_import('osgeo.gdal')

        if self._lst_labels is None:
            raise IOError('Labels related to satellite images are not set!')

        del self._ds_labels; gc.collect()
        self._ds_labels = []

        for fn in self._lst_labels:
            try:
                ds = gdal.Open(fn)
            except IOError:
                IOError('Cannot load source {}!'.format(fn))
                continue

            self._ds_labels.append(ds)

        if len(self._ds_labels) == 0:
            raise IOError('Cannot load any of following sources {}!'.format(self._lst_labels))

    """
    Process meta data (MULTISPECTRAL SATELLITE IMAGE, MODIS)
    """

    def __processBandDates_SATIMG_MODIS_REFLECTANCE(self) -> None:

        pd = lazy_import('pandas')

        nbands = 0
        unique_dates = set()

        with elapsed_timer('Processing band dates (satellite images, reflectance index)'):

            for id_ds, ds in enumerate(self._ds_satimgs):

                # determine number of bands for each data set
                n = ds.RasterCount; nbands += n

                # processing band dates related to multi spectral images (reflectance)
                for band_id in range(0, n):

                    rs_band = ds.GetRasterBand(band_id + 1)
                    band_dsc = rs_band.GetDescription()

                    if '_sur_refl_' in band_dsc:
                        band_date = band2date_reflectance(rs_band.GetDescription())
                        unique_dates.add((band_date, id_ds))

            if not unique_dates:
                raise ValueError('Multi spectral images do not contain any useful information about dates for band!')

            df_dates = pd.DataFrame(sorted(unique_dates), columns=['Date', 'Image ID'])
            # clean up
            del unique_dates; gc.collect()

        del self._df_dates_satimgs; gc.collect()
        self._df_dates_satimgs = df_dates

    def __processBandDates_SATELLITE_IMGS(self) -> None:

        if self._ds_satimgs is None:
            try:
                self.__loadGeoTIFF_SATELLITE_IMGS()
            except IOError:
                raise IOError('Cannot load any of following satellite images: {}'.format(self.lst_satimgs))

        # TODO try-except
        if self.modis_collection == ModisIndex.REFLECTANCE:
            self.__processBandDates_SATIMG_MODIS_REFLECTANCE()
        else:
            raise NotImplementedError

    def __processMultiSpectralBands_SATIMG_MODIS_REFLECTANCE(self):

        if self._map_start_satimgs is not None:
            del self._map_start_satimgs; self._map_band_id_satimg = None
            gc.collect()

        map_start_satimg = {}
        pos = 0

        with elapsed_timer('Processing multi spectral bands (satellite images, reflectance index)'):

            for id_ds, ds in enumerate(self._ds_satimgs):

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

        if self._ds_satimgs is None:
            try:
                self.__loadGeoTIFF_SATELLITE_IMGS()
            except IOError:
                raise IOError('Cannot load any following satellite images: {}'.format(self.lst_satimgs))

        if self._df_dates_satimgs is None:
            try:
                self.__processBandDates_SATELLITE_IMGS()
            except ValueError:
                msg = 'Cannot process band dates of any following satellite images: {}'
                raise ValueError(msg.format(self.lst_satimgs))

        try:
            if self.modis_collection == ModisIndex.REFLECTANCE:
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
            del self._df_dates_labels; self._df_dates_satimgs = None
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
            del self._df_dates_labels; self._df_dates_satimgs = None
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
    dataset_loader = DatasetLoader(
        lst_satimgs=lst_satimgs,
        lst_labels=lst_labels_mtbs
    )
