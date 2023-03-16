import gc
import os

from typing import Union

# TODO move to lazy loader
import pandas as pd
from osgeo import gdal

from mlfire.earthengine.collections import FireLabelsCollection, ModisIndex
from mlfire.utils.time import elapsed_timer
from mlfire.utils.utils_string import band2data_reflectance


class DatasetLoader(object):

    def __init__(self,
                 lst_satimgs: Union[tuple[str], list[str]],
                 lst_labels: Union[tuple[str], list[str]],
                 test_ratio: float = .33,
                 modis_collection: ModisIndex = ModisIndex.REFLECTANCE,
                 label_collection: FireLabelsCollection = FireLabelsCollection.MTBS):

        self._ds_satimgs = None
        self._df_dates_satimgs = None

        self._ds_labels = None
        self._df_dates_labels = None

        self._map_start_satimgs = None
        self._nimgs = -1

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

        gc.collect()  # invoke garbage collector

        self._nimgs = -1

        # set flags to false
        self._satimgs_processed = False
        self._labels_processed = False

    """
    IO functionality
    """

    def __loadGeoTIFF_SATELLITE_IMGS(self) -> None:

        if self._lst_satimgs is None:
            raise IOError('Multispectral satellite data is not set!')

        del self._ds_satimgs
        self._ds_satimgs = []

        for fn in self._lst_satimgs:
            try:
                ds = gdal.Open(fn)
            except IOError:
                IOError('Cannot load source {}!'.format(fn))
                continue

            self._ds_satimgs.append(ds)

        if len(self._ds_satimgs) == 0:
            raise IOError('Cannot load any of following sources {}!'.format(self._ds_satimgs))

    """
    Process meta data
    """

    def __processBandDates_SATIMG_MODIS_REFLECTANCE(self) -> None:

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
                        band_date = band2data_reflectance(rs_band.GetDescription())
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
                    band_date = band2data_reflectance(band_dsc)

                    # determine where a multi spectral image begins
                    if '_sur_refl_' in band_dsc and last_date != band_date:
                        map_start_satimg[pos] = (id_ds, band_id + 1); pos += 1
                        last_date = band_date

        if not map_start_satimg:
            raise ValueError('Sattelite image file does not contain any useful data!')

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

        if self.modis_collection == ModisIndex.REFLECTANCE:
            self.__processMultiSpectralBands_SATIMG_MODIS_REFLECTANCE()
        else:
            raise NotImplementedError

        self._satimgs_processed = True

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
