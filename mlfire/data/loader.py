import gc
import os

from enum import Enum
from typing import Union

# collections
from mlfire.earthengine.collections import FireLabelCollection, ModisCollection
from mlfire.earthengine.collections import MTBSRegion, MTBSSeverity

# import utils
from mlfire.utils.functool import lazy_import
from mlfire.utils.time import elapsed_timer
from mlfire.utils.utils_string import satdata_dsc2date
from mlfire.utils.utils_string import band2date_firecci, band2date_mtbs

# lazy imports
_pd = lazy_import('pandas')

# lazy imports - classes
PandasDataFrame = _pd.DataFrame


class SatDataSelectOpt(Enum):

    NONE = 0
    REFLECTANCE = 1
    SURFACE_TEMPERATURE = 2  # TODO rename -> TEMPERATURE
    ALL = 3

    def __and__(self, other):
        return SatDataSelectOpt(self.value & other.value)

    def __eq__(self, other):
        if other is None: return False
        return self.value == other.value

    def __or__(self, other):
        return SatDataSelectOpt(self.value | other.value)


class DatasetLoader(object):  # TODO rename to SatDataLoader and split for labels

    def __init__(self,
                 lst_labels_locfires: Union[tuple[str], list[str]],  # TODO rename
                 lst_satdata_reflectance: Union[tuple[str], list[str], None] = None,
                 lst_satdata_temperature: Union[tuple[str], list[str], None] = None,
                 # TODO add here vegetation indices and infrared bands
                 label_collection: FireLabelCollection = FireLabelCollection.MTBS,
                 # TODO comment
                 opt_select_satdata: Union[SatDataSelectOpt, list[SatDataSelectOpt]] = SatDataSelectOpt.ALL,
                 # TODO comment
                 cci_confidence_level: int = 70,
                 # TODO comment
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA,
                 mtbs_min_severity: MTBSSeverity = MTBSSeverity.LOW,
                 # TODO comment
                 test_ratio: float = .33,
                 val_ratio: float = .0,
                 # TODO comment
                 estimate_time: bool = True):

        self._ds_satdata_reflectance = None
        self._df_timestamps_reflectance = None

        self._ds_satdata_temperature = None
        self._df_timestamps_temperature = None

        self._ds_wildfires = None
        self._df_timestamps_wildfires = None

        self._map_layout_relectance = None
        self._map_layout_temperature = None

        self._map_band_id_label = None  # TODO rename

        self.__len_ts_reflectance = 0
        self.__len_ts_temperature = 0
        self._nbands_label = 0  # TODO rename

        # training, test, and validation data sets

        # TODO move to ts.py?

        self._nfeatures_ts = 0  # TODO rename

        self._ds_training = None
        self._ds_test = None
        self._ds_val = None

        self.__test_ratio = None
        self.test_ratio = test_ratio

        self.__val_ratio = None
        self.val_ratio = val_ratio

        # properties sources - reflectance, land surface temperature, and labels

        self.__lst_satdata_reflectance = None
        self.lst_satdata_reflectance = lst_satdata_reflectance

        self.__lst_satdata_temperature = None
        self.lst_satdata_temperature = lst_satdata_temperature

        self.__opt_select_satdata = None
        self.opt_select_satdata = opt_select_satdata

        self._satimgs_processed = False  # TODO rename

        # properties source - labels (wildfire locations)

        self.__lst_labels_wildfires = None
        self.lst_labels_wildfires = lst_labels_locfires

        self.__label_collection = None
        self.label_collection = label_collection

        self.__mtbs_region = None
        if label_collection == FireLabelCollection.MTBS: self.mtbs_region = mtbs_region

        self.__mtbs_min_severity = None
        if label_collection == FireLabelCollection.MTBS: self.mtbs_min_severity = mtbs_min_severity

        self.__cci_confidence_level = -1
        if label_collection == FireLabelCollection.CCI: self.cci_confidence_level = cci_confidence_level

        self._labels_processed = False  # TODO rename

        self.__estimate_time = None
        self.estimate_time = estimate_time

    """
    Satellite data (sources) - reflectance and temperature
    """

    @property
    def opt_select_satdata(self) -> SatDataSelectOpt:

        return self.__opt_select_satdata

    @opt_select_satdata.setter
    def opt_select_satdata(self, opt_select: SatDataSelectOpt) -> None:

        if self.opt_select_satdata == opt_select:
            return

        if isinstance(opt_select, (list, tuple)):
            _flgs = 0

            for opt in opt_select: _flgs |= opt
            opt_select = _flgs

        self._reset()  # clean up
        self.__opt_select_satdata = opt_select

    @property
    def lst_satdata_reflectance(self) -> Union[tuple[str], list[str], None]:

        return self.__lst_satdata_reflectance

    @lst_satdata_reflectance.setter
    def lst_satdata_reflectance(self, lst_fn: Union[tuple[str], list[str]]) -> None:

        if self.__lst_satdata_reflectance == lst_fn:
            return

        if not isinstance(lst_fn, (tuple, list)):
            err_msg = ''
            raise TypeError(err_msg)

        for fn in lst_fn:
            if not os.path.exists(fn):
                raise IOError(f'file {fn} does not exist')

        self._reset()  # clean up
        self.__lst_satdata_reflectance = lst_fn

    @property
    def lst_satdata_temperature(self) -> Union[tuple[str], list[str]]:

        return self.__lst_satdata_temperature

    @lst_satdata_temperature.setter
    def lst_satdata_temperature(self, lst_fn: Union[tuple[str], list[str]]):

        if self.__lst_satdata_temperature == lst_fn:
            return

        if not isinstance(lst_fn, (tuple, list)):
            err_msg = ''
            raise TypeError(err_msg)

        for fn in lst_fn:
            if not os.path.exists(fn):
                raise IOError(f'file {fn} does not exitst')

        self._reset()  # clean up
        self.__lst_satdata_temperature = lst_fn

    """
    Timestamps - reflectance, land surface temperature, and labels for wildfire localization
    """

    @property
    def timestamps_reflectance(self) -> PandasDataFrame:

        if self._df_timestamps_reflectance is None:
            self.__processTimestamps_SATDATA(select_opt=SatDataSelectOpt.REFLECTANCE)

            if self._df_timestamps_reflectance is None:
                err_msg = 'data frame containing timestamps (reflectance) was not created'
                raise TypeError(err_msg)

        return self._df_timestamps_reflectance

    @property
    def timestamps_temperature(self) -> PandasDataFrame:

        if self._df_timestamps_temperature is None:
            self.__processTimestamps_SATDATA(select_opt=SatDataSelectOpt.SURFACE_TEMPERATURE)

            if self._df_timestamps_temperature is None:
                err_msg = 'data frame containing timestamps (temperature) was not created'
                raise TypeError(err_msg)

        return self._df_timestamps_temperature

    @property
    def timestamps_locfire_labels(self) -> PandasDataFrame:

        if self._df_timestamps_wildfires is None:
            self.__processTimestamps_LABEL()

            if self._df_timestamps_wildfires is None:
                err_msg = 'data frame containing timestamps (fire location) was not created!'
                raise TypeError(err_msg)

        return self._df_timestamps_wildfires

    """
    FireCII labels properties
    """

    @property
    def cci_confidence_level(self) -> int:

        return self.__cci_confidence_level

    @cci_confidence_level.setter
    def cci_confidence_level(self, level: int) -> None:

        if self.__cci_confidence_level == level:
            return

        if level < 0 or level > 100:
            raise ValueError('confidence level for FireCCI labels must be positive int between 0 and 100')

        self.__cci_confidence_level = level

    """
    MTBS labels properties
    """

    @property
    def mtbs_region(self) -> MTBSRegion:

        return self.__mtbs_region

    @mtbs_region.setter
    def mtbs_region(self, region: MTBSRegion) -> None:

        if self.__mtbs_region == region:
            return

        self._reset()
        self.__mtbs_region = region

    @property
    def mtbs_min_severity(self) -> MTBSSeverity:

        return self.__mtbs_min_severity

    @mtbs_min_severity.setter
    def mtbs_min_severity(self, severity: MTBSSeverity) -> None:

        if self.__mtbs_min_severity == severity:
            return

        self._reset()
        self.__mtbs_min_severity = severity

    """
    Labels related to wildfires 
    """

    @property
    def lst_labels_wildfires(self) -> Union[tuple[str], list[str]]:

        return self.__lst_labels_wildfires

    @lst_labels_wildfires.setter
    def lst_labels_wildfires(self, lst_labels: Union[tuple[str], list[str]]) -> None:

        if self.__lst_labels_wildfires == lst_labels:
            return

        for fn in lst_labels:
            if not os.path.exists(fn):
                raise IOError(f'file {fn} does not exist!')

        self._reset()  # clean up
        self.__lst_labels_wildfires = lst_labels

    @property
    def label_collection(self) -> FireLabelCollection:

        return self.__label_collection

    @label_collection.setter
    def label_collection(self, collection: FireLabelCollection) -> None:

        if self.label_collection == collection:
            return

        self._reset()  # clean up
        self.__label_collection = collection

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

        return self.__test_ratio

    @test_ratio.setter
    def test_ratio(self, ratio: float) -> None:

        if self.test_ratio == ratio:
            return

        self._reset()  # clean up
        self.__test_ratio = ratio

    @property
    def val_ratio(self) -> float:

        return self.__val_ratio

    @val_ratio.setter
    def val_ratio(self, ratio: float) -> None:

        if self.val_ratio == ratio:
            return

        self._reset()
        self.__val_ratio = ratio

    # Time measure

    @property
    def estimate_time(self) -> bool:

        return self.__estimate_time

    @estimate_time.setter
    def estimate_time(self, flg: bool) -> None:

        self.__estimate_time = flg

    def _reset(self):

        del self._ds_training; self._ds_training = None
        del self._ds_test; self._ds_test = None
        del self._ds_val; self._ds_val = None

        del self._df_timestamps_reflectance; self._df_timestamps_reflectance = None
        del self._map_layout_relectance; self._map_layout_relectance = None

        del self._df_timestamps_temperature; self._df_timestamps_temperature = None
        del self._map_layout_temperature; self._map_layout_temperature = None

        del self._df_timestamps_wildfires; self._df_timestamps_wildfires = None
        del self._map_band_id_label; self._map_band_id_label = None

        gc.collect()  # invoke garbage collector

        self._nfeatures_ts = 0  # TODO rename nbands_img?
        self.__len_ts_reflectance = 0
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
                raise RuntimeWarning(f'cannot load source {fn}')

            if ds is None:
                raise RuntimeWarning(f'source {fn} is empty')
            lst_ds.append(ds)

        return lst_ds

    def __loadGeoTIFF_REFLECTANCE(self) -> None:

        if self.lst_satdata_reflectance is None:
            err_msg = 'satellite data (reflectance) is not set'
            raise TypeError(err_msg)

        del self._ds_satdata_reflectance; gc.collect()
        self._ds_satdata_reflectance = None

        self._ds_satdata_reflectance = self.__loadGeoTIFF_SOURCES(self.lst_satdata_reflectance)
        if not self._ds_satdata_reflectance:
            err_msg = 'satellite data (reflectance) was not loaded'
            raise IOError(err_msg)

    def __loadGeoTIFF_TEMPERATURE(self) -> None:

        if self.lst_satdata_temperature is None:
            err_msg = 'satellite data (temperature) is not set'
            raise TypeError(err_msg)

        del self._ds_satdata_temperature; gc.collect()
        self._ds_satdata_temperature = None

        self._ds_satdata_temperature = self.__loadGeoTIFF_SOURCES(self.lst_satdata_temperature)
        if not self._ds_satdata_temperature:
            err_msg = 'satellite data (temperature) was not loaded'
            raise IOError(err_msg)

    def __loadGeoTIFF_LABELS(self) -> None:

        if not self.lst_labels_wildfires or self.lst_labels_wildfires is None:
            err_msg = 'Satellite data (labels - wildfires localization) is not set!'
            raise TypeError(err_msg)

        del self._ds_wildfires; gc.collect()
        self._ds_wildfires = None

        self._ds_wildfires = self.__loadGeoTIFF_SOURCES(self.lst_labels_wildfires)
        if not self._ds_wildfires:
            err_msg = 'Satellite data (labels - wildfires localization) is not set!'
            raise IOError(err_msg)

    """
    Processing timestamps - satellite data (reflectance and temperature)  
    """

    def __processTimestamps_SATDATA_REFLECTANCE(self) -> None:

        if self._df_timestamps_reflectance is not None:
            return

        if self._ds_satdata_reflectance is None:
            try:
                self.__loadGeoTIFF_REFLECTANCE()
            except IOError:
                err_msg = 'cannot load any source - satellite data (reflectance): {}'
                err_msg = err_msg.format(self.lst_satdata_reflectance)
                raise IOError(err_msg)

        nsources = len(self.lst_satdata_reflectance)

        unique_dates = set()
        col_names = ['Timestamps', 'Image ID'] if nsources > 1 else ['Timestamps']

        with elapsed_timer(msg='Processing timestamps (reflectance)', enable=self.estimate_time):

            for i, img_ds in enumerate(self._ds_satdata_reflectance):
                for rs_id in range(0, img_ds.RasterCount):

                    rs_band = img_ds.GetRasterBand(rs_id + 1)
                    rs_dsc = rs_band.GetDescription()

                    if '_sur_refl_' in rs_dsc:
                        reflec_date = satdata_dsc2date(rs_dsc)
                        unique_dates.add((reflec_date, i) if nsources > 1 else reflec_date)

            if not unique_dates:
                err_msg = 'sources (reflectance) do not contain any useful information about timestamps'
                raise ValueError(err_msg)

            try:
                df_dates = _pd.DataFrame(sorted(unique_dates), columns=col_names)
            except MemoryError:
                err_msg = 'pandas data frame (reflectance) was not created'
                raise MemoryError(err_msg)

            # clean up
            del unique_dates; gc.collect()

        del self._df_timestamps_reflectance; gc.collect()
        self._df_timestamps_reflectance = df_dates

    def __processTimestamps_SATDATA_TEMPERATURE(self) -> None:

        if self._df_timestamps_temperature is not None:
            return

        if self._ds_satdata_temperature is None:
            try:
                self.__loadGeoTIFF_TEMPERATURE()
            except IOError:
                err_msg = 'cannot load any source - satellite data (temperature): {}'
                err_msg = err_msg.format(self.lst_satdata_temperature)
                raise IOError(err_msg)

        nsources = len(self.lst_satdata_temperature)

        lst_dates = []
        col_names = ['Timestamps', 'Image ID'] if nsources > 1 else ['Timestamps']

        with elapsed_timer(msg='Processing timestamps (temperature)', enable=self.estimate_time):

            for i, img_ds in enumerate(self._ds_satdata_temperature):
                for rs_id in range(0, img_ds.RasterCount):

                    rs_band = img_ds.GetRasterBand(rs_id + 1)
                    rs_dsc = rs_band.GetDescription()

                    if 'lst_day_1km' in rs_dsc.lower():
                        temperature_date = satdata_dsc2date(rs_dsc)
                        lst_dates.append((temperature_date, i) if nsources > 1 else temperature_date)

            if not lst_dates:
                err_msg = 'sources (temperature) do not contain any useful information about dates'
                raise ValueError(err_msg)

            try:
                df_dates = _pd.DataFrame(sorted(lst_dates), columns=col_names)
            except MemoryError:
                err_msg = 'pandas data frame (temperature) was not created'
                raise MemoryError(err_msg)

            # clean up
            del lst_dates; gc.collect()

        del self._df_timestamps_temperature; gc.collect()
        self._df_timestamps_temperature = df_dates

    def __processTimestamps_SATDATA(self, select_opt: SatDataSelectOpt = SatDataSelectOpt.ALL) -> None:

        # processing reflectance (MOD09A1)

        if self.lst_satdata_reflectance is not None and \
           (select_opt & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE):

            if self._ds_satdata_reflectance is None:
                try:
                    self.__loadGeoTIFF_REFLECTANCE()
                except IOError:
                    err_msg = 'cannot load any source - satellite data (reflectance): {}'
                    err_msg = err_msg.format(self.lst_satdata_reflectance)
                    raise IOError(err_msg)

            try:
                self.__processTimestamps_SATDATA_REFLECTANCE()
            except ValueError:
                err_msg = 'cannot process timestamps - satellite data (reflectance)'
                raise ValueError(err_msg)

        # processing land surface temperature (MOD11A2)

        if self.lst_satdata_temperature is not None and \
           (select_opt & SatDataSelectOpt.SURFACE_TEMPERATURE == SatDataSelectOpt.SURFACE_TEMPERATURE):

            if self._ds_satdata_temperature is None:
                try:
                    self.__loadGeoTIFF_TEMPERATURE()
                except IOError:
                    err_msg = 'cannot load any source - satellite data (temperature): {}'
                    err_msg = err_msg.format(self.lst_satdata_temperature)
                    raise IOError(err_msg)

            try:
                self.__processTimestamps_SATDATA_TEMPERATURE()
            except ValueError:
                err_msg = 'cannot process timestamps - satellite data (temperature)'
                raise ValueError(err_msg)

    """
    Processing metadata - reflectance and temperature
    """

    def __processLayersLayout_SATDATA_REFLECTANCE(self) -> None:

        if self._map_layout_relectance is not None:
            return

        if self._ds_satdata_reflectance is None:
            try:
                self.__loadGeoTIFF_REFLECTANCE()
            except IOError:
                err_msg = 'cannot load any source - satellite data (reflectance): {}'
                err_msg = err_msg.format(self.lst_satdata_reflectance)
                raise IOError(err_msg)

        map_layout_satdata = {}
        pos = 0

        nsources = len(self._ds_satdata_reflectance)

        with elapsed_timer('Processing layout of layers (reflectance)', enable=self.estimate_time):

            for i, img_ds in enumerate(self._ds_satdata_reflectance):
                last_date = 0  # reset value of last date
                for rs_id in range(0, img_ds.RasterCount):

                    rs_band = img_ds.GetRasterBand(rs_id + 1)
                    rs_dsc = rs_band.GetDescription()

                    reflec_date = satdata_dsc2date(rs_dsc)

                    if ('_sur_refl_' in rs_dsc.lower()) and (reflec_date != last_date):
                        map_layout_satdata[pos] = (i, rs_id + 1) if nsources > 1 else rs_id + 1; pos += 1
                        last_date = reflec_date

        if not map_layout_satdata:
            raise TypeError('satellite data (reflectance) do not contain any useful layer')

        self._map_layout_relectance = map_layout_satdata
        self.__len_ts_reflectance = pos

    def __processLayersLayout_SATDATA_TEMPERATURE(self) -> None:

        if self._map_layout_temperature is not None:
            return

        if self._ds_satdata_temperature is None:
            try:
                self.__loadGeoTIFF_TEMPERATURE()
            except IOError:
                err_msg = 'cannot load any source - satellite data (temperature): {}'
                err_msg = err_msg.format(self.lst_satdata_temperature)
                raise IOError(err_msg)

        map_layout_satdata = {}
        pos = 0

        nsources = len(self._ds_satdata_temperature)

        with elapsed_timer('Processing layout of layers (temperature)', enable=self.estimate_time):

            for i, img_ds in enumerate(self._ds_satdata_temperature):
                last_date = 0  # reset value of last date
                for rs_id in range(0, img_ds.RasterCount):

                    rs_band = img_ds.GetRasterBand(rs_id + 1)
                    rs_dsc = rs_band.GetDescription()

                    temperature_date = satdata_dsc2date(rs_dsc)

                    if ('lst_day_1km' in rs_dsc.lower()) and (temperature_date != last_date):
                        map_layout_satdata[pos] = (i, rs_id + 1) if nsources > 1 else rs_id + 1; pos += 1
                        last_date = temperature_date

        if not map_layout_satdata:
            raise TypeError('satellite data (temperature) do not contain any useful layer')

        self._map_layout_temperature = map_layout_satdata
        self.__len_ts_temperature = pos

    def _processMetadata_SATDATA(self) -> None:

        # processing reflectance (MOD09A1)

        if self.lst_satdata_reflectance is not None and \
           (self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE):

            if self._df_timestamps_reflectance is None:
                try:
                    self.__processTimestamps_SATDATA(select_opt=SatDataSelectOpt.REFLECTANCE)
                except IOError or ValueError:
                    err_msg = 'cannot process timestamps - satellite data (reflectance)'
                    raise TypeError(err_msg)

            try:
                self.__processLayersLayout_SATDATA_REFLECTANCE()
            except TypeError:
                err_msg = 'cannot process a layout of layers - satellite data (reflectance)'
                raise TypeError(err_msg)

        # processing land surface temperature (MOD11A2)

        if self.lst_satdata_temperature is not None and \
           (self.opt_select_satdata & SatDataSelectOpt.SURFACE_TEMPERATURE == SatDataSelectOpt.SURFACE_TEMPERATURE):

            if self._df_timestamps_temperature is None:
                try:
                    self.__processTimestamps_SATDATA_TEMPERATURE()
                except IOError or ValueError:
                    err_msg = 'cannot process timestamps - satellite data (temperature)'
                    raise TypeError(err_msg)

            try:
                self.__processLayersLayout_SATDATA_TEMPERATURE()
            except TypeError:
                err_msg = 'cannot process a layout of layers - satellite data (temperature)'
                raise TypeError(err_msg)

        # TODO comment
        self._satimgs_processed = True  # TODO rename

    """
    Process meta data (LABELS)
    """

    def __processBandDates_LABEL_CCI(self) -> None:

        if self._df_timestamps_wildfires is not None:
            del self._df_timestamps_wildfires; self._df_timestamps_reflectance = None
            gc.collect()

        lst = []

        with elapsed_timer('Processing band dates (labels, CCI)'):

            for id_ds, ds in enumerate(self._ds_wildfires):
                for band_id in range(ds.RasterCount):

                    rs_band = ds.GetRasterBand(band_id + 1)
                    dsc_band = rs_band.GetDescription()

                    if 'ConfidenceLevel' in dsc_band:
                        band_date = band2date_firecci(dsc_band)
                        lst.append((band_date, id_ds))

        if not lst:
            raise ValueError('Label file does not contain any useful data!')

        df_dates = _pd.DataFrame(sorted(lst), columns=['Date', 'Image ID'])
        del lst; gc.collect()

        self._df_timestamps_wildfires = df_dates

    def __processBandDates_LABEL_MTBS(self) -> None:

        if self._df_timestamps_wildfires is not None:
            del self._df_timestamps_wildfires; self._df_timestamps_reflectance = None
            gc.collect()

        lst = []

        with elapsed_timer('Processing band dates (labels, MTBS)'):

            for id_ds, ds in enumerate(self._ds_wildfires):
                for band_id in range(ds.RasterCount):

                    rs_band = ds.GetRasterBand(band_id + 1)
                    dsc_band = rs_band.GetDescription()

                    if self.mtbs_region.value in dsc_band:
                        band_date = band2date_mtbs(dsc_band)
                        lst.append((band_date, id_ds))

        if not lst:
            raise ValueError('Sources do not contain any useful data!')

        df_dates = _pd.DataFrame(sorted(lst), columns=['Date', 'Image ID'])
        del lst; gc.collect()

        self._df_timestamps_wildfires = df_dates

    def __processTimestamps_LABEL(self) -> None:

        if self._ds_wildfires is None:
            try:
                self.__loadGeoTIFF_LABELS()
            except IOError:
                raise IOError('Cannot load any following label sources: {}'.format(self.lst_labels_wildfires))

        try:
            if self.label_collection == FireLabelCollection.CCI:
                self.__processBandDates_LABEL_CCI()
            elif self.label_collection == FireLabelCollection.MTBS:
                self.__processBandDates_LABEL_MTBS()
            else:
                raise NotImplementedError
        except ValueError or NotImplementedError:
            raise ValueError('Cannot process band dates related to labels ({})!'.format(self.label_collection.name))

    """
    labels
    """

    def __processLabels_CCI(self) -> None:

        if self._map_band_id_label is None:
            del self._map_band_id_label; self._map_band_id_label = None
            gc.collect()

        map_band_ids = {}
        pos = 0

        with elapsed_timer('Processing fire labels ({})'.format(self.label_collection.name)):

            for id_ds, ds in enumerate(self._ds_wildfires):
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

        # determine bands and their ids for region select_opt
        if self._map_band_id_label is None:
            del self._map_band_id_label; self._map_band_id_label = None
            gc.collect()  # invoke garbage collector

        map_band_ids = {}
        pos = 0

        with elapsed_timer('Processing fire labels ({})'.format(self.label_collection.name)):

            for id_ds, ds in enumerate(self._ds_wildfires):
                for band_id in range(ds.RasterCount):

                    rs_band = ds.GetRasterBand(band_id + 1)
                    dsc_band = rs_band.GetDescription()

                    # map id data set for selected region to band id
                    if self.mtbs_region.value in dsc_band:
                        map_band_ids[pos] = (id_ds, band_id + 1); pos += 1

        if not map_band_ids:
            msg = 'Any labels ({}) do not contain any useful information: {}'.format(self.label_collection.name, self.__lst_labels_wildfires)
            raise ValueError(msg)

        self._map_band_id_label = map_band_ids
        self._nbands_label = pos

    def _processMetaData_LABELS(self) -> None:

        if self._ds_wildfires is None:
            try:
                self.__loadGeoTIFF_LABELS()
            except IOError:
                raise IOError('Cannot load any following label sources: {}'.format(self.lst_labels_wildfires))

        if self._df_timestamps_wildfires is None:
            try:
                self.__processTimestamps_LABEL()
            except ValueError or AttributeError:
                raise ValueError('Cannot process ban related labels ({})!'.format(self.label_collection.name))

        try:
            if self.label_collection == FireLabelCollection.CCI:
                self.__processLabels_CCI()
            elif self.label_collection == FireLabelCollection.MTBS:
                self.__processLabels_MTBS()
            else:
                raise NotImplementedError
        except ValueError or NotImplementedError:
            msg = 'Cannot process labels metadata!'
            raise ValueError(msg)

        # everything is done
        self._labels_processed = True

    """
    TODO comment
    """

    def getLengthTimeseries(self, opt_select: SatDataSelectOpt) -> int:

        if not isinstance(opt_select, SatDataSelectOpt):
            err_msg = f'getLengthTimeseries() argument must be SatDataSelectOpt, not \'{type(opt_select)}\''
            raise TypeError(err_msg)

        if opt_select & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE:
            self.__processLayersLayout_SATDATA_REFLECTANCE()
            return self.__len_ts_reflectance
        elif opt_select & SatDataSelectOpt.SURFACE_TEMPERATURE == SatDataSelectOpt.SURFACE_TEMPERATURE:
            self.__processLayersLayout_SATDATA_TEMPERATURE()
            return self.__len_ts_temperature
        else:
            raise NotImplementedError

    @property
    def len_ts(self) -> int:

        len_ts_reflectance = len_ts_temperature = length_ts = 0

        if self.lst_satdata_reflectance is not None:
            length_ts = len_ts_reflectance = self.getLengthTimeseries(opt_select=SatDataSelectOpt.REFLECTANCE)
        if self.lst_satdata_temperature is not None:
            length_ts = len_ts_temperature = self.getLengthTimeseries(opt_select=SatDataSelectOpt.SURFACE_TEMPERATURE)

        if self.lst_satdata_temperature is not None and self.lst_satdata_reflectance is not None:
            if len_ts_reflectance != len_ts_temperature:
                err_msg = ''
                raise ValueError(err_msg)

        return length_ts


if __name__ == '__main__':

    # TODO fix

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
        lst_satdata_reflectance=VAR_LST_SATIMGS,
        lst_loc_fires=VAR_LST_LABELS_MTBS
    )

    print(dataset_loader.timestamps_locfire_labels)
