# TODO rename -> loader.py -> load.py

import gc
import os

from enum import Enum
from typing import Union

# collections
from mlfire.earthengine.collections import MTBSRegion, MTBSSeverity
from mlfire.earthengine.collections import NFEATURES_REFLECTANCE as _NFEATURES_REFLECTANCE

# import utils
from mlfire.utils.functool import lazy_import
from mlfire.utils.time import elapsed_timer
from mlfire.utils.utils_string import satdata_dsc2date
from mlfire.utils.utils_string import band2date_firecci, band2date_mtbs

# lazy imports
_datetime = lazy_import('datetime')
_np = lazy_import('numpy')
_pd = lazy_import('pandas')

# lazy imports - classes
_PandasDataFrame = _pd.DataFrame


class SatDataSelectOpt(Enum):

    NONE = 0
    REFLECTANCE = 1
    TEMPERATURE = 2
    ALL = 3

    def __and__(self, other):
        return SatDataSelectOpt(self.value & other.value)

    def __eq__(self, other):
        if other is None: return False
        return self.value == other.value

    def __or__(self, other):
        return SatDataSelectOpt(self.value | other.value)


class FireMapSelectOpt(Enum):

    MTBS = 1
    CCI = 2

    def __eq__(self, other):
        if other is None: return False

        if not isinstance(other, FireMapSelectOpt):
            err_msg = ''
            raise TypeError(err_msg)

        return self.value == other.value

    def __str__(self):
        return self.name


class MetaDataSelectOpt(Enum):

    NONE = 0
    TIMESTAMPS = 1
    LAYOUT = 2
    ALL = 3

    def __and__(self, other):
        return SatDataSelectOpt(self.value & other.value)

    def __eq__(self, other):
        if other is None: return False
        return self.value == other.value

    def __or__(self, other):
        return SatDataSelectOpt(self.value | other.value)


class SatDataLoader(object):

    def __init__(self,
                 lst_firemaps: Union[tuple[str], list[str], None],
                 lst_satdata_reflectance: Union[tuple[str], list[str], None] = None,
                 lst_satdata_temperature: Union[tuple[str], list[str], None] = None,
                 # TODO comment
                 opt_select_firemap: FireMapSelectOpt = FireMapSelectOpt.MTBS,
                 # TODO comment
                 opt_select_satdata: Union[SatDataSelectOpt, list[SatDataSelectOpt]] = SatDataSelectOpt.ALL,
                 # TODO comment
                 select_timestamps: Union[list, tuple, None] = None,
                 # TODO comment
                 cci_confidence_level: int = 70,
                 # TODO comment
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA,
                 mtbs_min_severity: MTBSSeverity = MTBSSeverity.LOW,
                 # TODO comment
                 test_ratio: float = .33,  # TODO move to DataAdapterTS (ts.py)
                 val_ratio: float = .0,  # TODO move to DataAdapterTS (ts.py)
                 # TODO comment
                 estimate_time: bool = True):

        self._np_satdata = None
        self._np_firemaps = None

        self._ds_satdata_reflectance = None
        self._df_timestamps_reflectance = None
        self._np_satdata_reflectance = None

        self._ds_satdata_temperature = None
        self._df_timestamps_temperature = None
        self._np_satdata_temperature = None

        self._ds_firemaps = None
        self._df_timestamps_firemaps = None

        self._layout_layers_reflectance = None
        self._layout_layers_temperature = None

        self._map_layout_firemaps = None  # TODO rename -> _layout_layers_firemaps

        self.__len_ts_reflectance = 0
        self.__len_ts_temperature = 0
        self.__len_firemaps = 0

        # training, test, and validation data sets

        # TODO move to ts.py?

        self._nfeatures_ts = 0  # TODO rename or remove?

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

        self._satdata_processed = False  # TODO rename

        # properties source - labels (wildfire locations)

        self.__lst_firemap = None
        self.lst_firemaps = lst_firemaps

        self.__opt_select_firemap = None
        self.opt_select_firemap = opt_select_firemap

        self.__mtbs_region = None
        if opt_select_firemap == FireMapSelectOpt.MTBS: self.mtbs_region = mtbs_region

        self.__mtbs_min_severity = None
        if opt_select_firemap == FireMapSelectOpt.MTBS: self.mtbs_min_severity = mtbs_min_severity

        self.__cci_confidence_level = -1
        if opt_select_firemap == FireMapSelectOpt.CCI: self.cci_confidence_level = cci_confidence_level

        self._firemaps_processed = False  # TODO rename

        # TODO comment
        self.__select_timestamps = None
        self.select_timestamps = select_timestamps

        # TODO comment
        self.__ntimestamps = -1
        self.__timestamps_processed = False

        self.__estimate_time = None
        self.estimate_time = estimate_time

    """
    Satellite data (sources) - reflectance and temperature
    """

    @property
    def select_timestamps(self) -> Union[list, tuple, None]:

        return self.__select_timestamps

    @select_timestamps.setter
    def select_timestamps(self, timestamps: Union[list, tuple, None]) -> None:

        if self.__select_timestamps == timestamps:
            return

        self._reset()  # clean up
        self.__select_timestamps = timestamps

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
    def timestamps_reflectance(self) -> _PandasDataFrame:

        if self._df_timestamps_reflectance is None:
            self._processTimestamps_SATDATA(opt_select=SatDataSelectOpt.REFLECTANCE)

            if self._df_timestamps_reflectance is None:
                err_msg = 'data frame containing timestamps (reflectance) was not created'
                raise TypeError(err_msg)

        return self._df_timestamps_reflectance

    @property
    def timestamps_temperature(self) -> _PandasDataFrame:

        if self._df_timestamps_temperature is None:
            self._processTimestamps_SATDATA(opt_select=SatDataSelectOpt.TEMPERATURE)

            if self._df_timestamps_temperature is None:
                err_msg = 'data frame containing timestamps (temperature) was not created'
                raise TypeError(err_msg)

        return self._df_timestamps_temperature

    @property
    def timestamps_firemaps(self) -> _PandasDataFrame:

        if self._df_timestamps_firemaps is None:
            self.__processTimestamps_FIREMAPS()

            if self._df_timestamps_firemaps is None:
                err_msg = 'data frame containing timestamps (fire location) was not created!'
                raise TypeError(err_msg)

        return self._df_timestamps_firemaps

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
    def lst_firemaps(self) -> Union[tuple[str], list[str]]:

        return self.__lst_firemap

    @lst_firemaps.setter
    def lst_firemaps(self, lst_firemaps: Union[tuple[str], list[str]]) -> None:

        if self.__lst_firemap == lst_firemaps:
            return

        for fn in lst_firemaps:
            if not os.path.exists(fn):
                raise IOError(f'file {fn} does not exist!')

        self._reset()  # clean up
        self.__lst_firemap = lst_firemaps

    @property
    def opt_select_firemap(self) -> FireMapSelectOpt:

        return self.__opt_select_firemap

    @opt_select_firemap.setter
    def opt_select_firemap(self, opt_select: FireMapSelectOpt) -> None:

        if self.opt_select_firemap == opt_select:
            return

        self._reset()  # clean up
        self.__opt_select_firemap = opt_select

    @property
    def nbands_label(self) -> int:  # TODO rename?

        if not self._firemaps_processed:
            self._processMetaData_FIREMAPS()

        return self.__len_firemaps

    """
    Training, test and validation data sets TODO move to DataAdapterTS (ts.py) 
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

    """
    Private properties
    """

    @property
    def _ntimestamps(self) -> int:

        if self.__timestamps_processed: return self.__ntimestamps

        if self._df_timestamps_reflectance is not None:
            df_timestamps = self._df_timestamps_reflectance
        elif self._df_timestamps_temperature is not None:
            df_timestamps = self._df_timestamps_temperature
        else:
            err_msg = ''  # TODO error message
            raise TypeError(err_msg)

        if isinstance(self.select_timestamps[0], _datetime.date):
            begin_timestamp = self.select_timestamps[0]
            end_timestamp = self.select_timestamps[1]

            cond = (df_timestamps['Timestamps'] >= begin_timestamp) & (df_timestamps['Timestamps'] <= end_timestamp)
            self.__ntimestamps = len(df_timestamps[cond])
        else:
            # TODO type int
            raise NotImplementedError

        self.__timestamps_processed = True
        return self.__ntimestamps

    def _reset(self):

        del self._ds_training; self._ds_training = None # TODO move
        del self._ds_test; self._ds_test = None  # TODO move
        del self._ds_val; self._ds_val = None  # TODO move

        del self._df_timestamps_reflectance; self._df_timestamps_reflectance = None
        del self._layout_layers_reflectance; self._layout_layers_reflectance = None
        del self._ds_satdata_reflectance; self._ds_satdata_reflectance = None

        del self._df_timestamps_temperature; self._df_timestamps_temperature = None
        del self._layout_layers_temperature; self._layout_layers_temperature = None
        del self._ds_satdata_temperature; self._ds_satdata_temperature = None

        del self._df_timestamps_firemaps; self._df_timestamps_firemaps = None
        del self._map_layout_firemaps; self._map_layout_firemaps = None

        gc.collect()  # invoke garbage collector

        self.__len_ts_reflectance = 0
        self.__len_ts_temperature = 0
        self.__len_firemaps = 0

        self._nfeatures_ts = 0  # TODO rename nbands_img or remove?

        # TODO comment
        self.__ntimestamps = 0; self.__timestamps_processed = False

        # set flags to false
        self._satdata_processed = False
        self._firemaps_processed = False

    """
    Load sources - reflectance, land surface temperature, or labels
    """

    @staticmethod
    def __loadGeoTIFF_SOURCES(lst_sources: Union[list[str], tuple[str]]) -> list:  # TODO rename -> __loadGeoTIFF_DATASETS?

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

    def __loadGeoTIFF_REFLECTANCE(self) -> None:  # TODO rename -> __loadGeoTIFF_DATASETS_REFLECTANCE?

        if self.lst_satdata_reflectance is None:
            err_msg = 'satellite data (reflectance) is not set'
            raise TypeError(err_msg)

        del self._ds_satdata_reflectance; gc.collect()
        self._ds_satdata_reflectance = None

        self._ds_satdata_reflectance = self.__loadGeoTIFF_SOURCES(self.lst_satdata_reflectance)
        if not self._ds_satdata_reflectance:
            err_msg = 'satellite data (reflectance) was not loaded'
            raise IOError(err_msg)

    def __loadGeoTIFF_TEMPERATURE(self) -> None:  # TODO rename -> __loadGeoTIFF_DATASETS_TEMPERATURE?

        if self.lst_satdata_temperature is None:
            err_msg = 'satellite data (temperature) is not set'
            raise TypeError(err_msg)

        del self._ds_satdata_temperature; gc.collect()
        self._ds_satdata_temperature = None

        self._ds_satdata_temperature = self.__loadGeoTIFF_SOURCES(self.lst_satdata_temperature)
        if not self._ds_satdata_temperature:
            err_msg = 'satellite data (temperature) was not loaded'
            raise IOError(err_msg)

    def __loadGeoTIFF_FIREMAPS(self) -> None:

        if not self.lst_firemaps or self.lst_firemaps is None:
            err_msg = 'Satellite data (labels - wildfires localization) is not set!'  # TODO rename
            raise TypeError(err_msg)

        del self._ds_firemaps; gc.collect()
        self._ds_firemaps = None

        self._ds_firemaps = self.__loadGeoTIFF_SOURCES(self.lst_firemaps)
        if not self._ds_firemaps:
            err_msg = 'Satellite data (labels - wildfires localization) is not set!'  # TODO rename
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

        del self._df_timestamps_reflectance; gc.collect()  # TODO remove?
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

        self._df_timestamps_temperature = df_dates

    def _processTimestamps_SATDATA(self, opt_select: SatDataSelectOpt = SatDataSelectOpt.ALL) -> None:

        if not (isinstance(opt_select, SatDataSelectOpt)):
            err_msg = f'__processTimestamps_FIREMAPS() argument must be FireMapSelectOpt, not \'{type(opt_select)}\''
            raise TypeError(err_msg)

        # processing reflectance (MOD09A1)

        if self.lst_satdata_reflectance is not None and \
           (opt_select & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE):

            if self._ds_satdata_reflectance is None:
                try:
                    self.__loadGeoTIFF_REFLECTANCE()
                except IOError:
                    err_msg = 'cannot load any source - satellite data (reflectance): {}'
                    err_msg = err_msg.format(self.lst_satdata_reflectance)
                    raise IOError(err_msg)

            try:
                self.__processTimestamps_SATDATA_REFLECTANCE()
            except MemoryError or ValueError:
                err_msg = 'cannot process timestamps - satellite data (reflectance)'
                raise ValueError(err_msg)

        # processing land surface temperature (MOD11A2)

        if self.lst_satdata_temperature is not None and \
           (opt_select & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE):

            if self._ds_satdata_temperature is None:
                try:
                    self.__loadGeoTIFF_TEMPERATURE()
                except IOError:
                    err_msg = 'cannot load any source - satellite data (temperature): {}'
                    err_msg = err_msg.format(self.lst_satdata_temperature)
                    raise IOError(err_msg)

            try:
                self.__processTimestamps_SATDATA_TEMPERATURE()
            except MemoryError or ValueError:
                err_msg = 'cannot process timestamps - satellite data (temperature)'
                raise ValueError(err_msg)

    """
    Processing metadata and layout of layers - reflectance and temperature
    """

    def _processLayersLayout_SATDATA_REFLECTANCE(self) -> None:

        if self._layout_layers_reflectance is not None:
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

        self._layout_layers_reflectance = map_layout_satdata
        self.__len_ts_reflectance = pos

    def _processLayersLayout_SATDATA_TEMPERATURE(self) -> None:

        if self._layout_layers_temperature is not None:
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

                    if ('lst_day_1km' in rs_dsc.lower()) and (temperature_date != last_date):  # is condition for date necessary?
                        map_layout_satdata[pos] = (i, rs_id + 1) if nsources > 1 else rs_id + 1; pos += 1
                        last_date = temperature_date

        if not map_layout_satdata:
            raise TypeError('satellite data (temperature) do not contain any useful layer')

        self._layout_layers_temperature = map_layout_satdata
        self.__len_ts_temperature = pos

    def _processMetadata_SATDATA(self) -> None:

        # processing reflectance (MOD09A1)

        if self.lst_satdata_reflectance is not None and \
           (self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE):

            if self._df_timestamps_reflectance is None:
                try:
                    self._processTimestamps_SATDATA(opt_select=SatDataSelectOpt.REFLECTANCE)
                except IOError or ValueError:
                    err_msg = 'cannot process timestamps - satellite data (reflectance)'
                    raise TypeError(err_msg)

            try:
                self._processLayersLayout_SATDATA_REFLECTANCE()
            except TypeError:
                err_msg = 'cannot process a layout of layers - satellite data (reflectance)'
                raise TypeError(err_msg)

        # processing land surface temperature (MOD11A2)

        if self.lst_satdata_temperature is not None and \
           (self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE):

            if self._df_timestamps_temperature is None:
                try:
                    self._processTimestamps_SATDATA(opt_select=SatDataSelectOpt.TEMPERATURE)
                except IOError or ValueError:
                    err_msg = 'cannot process timestamps - satellite data (temperature)'
                    raise TypeError(err_msg)

            try:
                self._processLayersLayout_SATDATA_TEMPERATURE()
            except TypeError:
                err_msg = 'cannot process a layout of layers - satellite data (temperature)'
                raise TypeError(err_msg)

        # TODO comment
        self._satdata_processed = True

    """
    Processing fire maps - timestamps (MTBS and FireCCI)  
    """

    def __processTimestamps_FIREMAPS(self, opt_select: FireMapSelectOpt = None) -> None:

        if not (isinstance(opt_select, FireMapSelectOpt) or opt_select is None):
            err_msg = f'__processTimestamps_FIREMAPS() argument must be FireMapSelectOpt, not \'{type(opt_select)}\''
            raise TypeError(err_msg)

        if self._df_timestamps_firemaps is not None:
            return

        _opt_firemap = self.opt_select_firemap if opt_select is None else opt_select
        firemap_name = _opt_firemap.name.lower()

        if self._ds_firemaps is None:
            try:
                self.__loadGeoTIFF_FIREMAPS()
            except IOError:
                err_msg = 'cannot load any fire map ({}): {}'
                err_msg = err_msg.format(
                    firemap_name,
                    self.lst_firemaps
                )
                raise IOError(err_msg)

        nsources = len(self._ds_firemaps)

        lst_dates = []
        col_names = ['Timestamps', 'Image ID'] if nsources > 1 else ['Timestamps']

        proc_msg = '({} fire map{})'.format(firemap_name, 's' if nsources > 1 else '')
        proc_msg = f'Processing timestamps {proc_msg}'
        with elapsed_timer(proc_msg, enable=self.estimate_time):

            for i, img_ds in enumerate(self._ds_firemaps):
                last_timestamps = 0  # reset value of last date
                for rs_id in range(0, img_ds.RasterCount):

                    rs_band = img_ds.GetRasterBand(rs_id + 1)
                    rs_dsc = rs_band.GetDescription()

                    if _opt_firemap == FireMapSelectOpt.CCI and 'ConfidenceLevel' in rs_dsc:
                        firemap_timestamp = band2date_firecci(rs_dsc)
                    elif _opt_firemap == FireMapSelectOpt.MTBS and self.mtbs_region.value in rs_dsc:
                        firemap_timestamp = band2date_mtbs(rs_dsc)

                    if last_timestamps != firemap_timestamp:
                        lst_dates.append((firemap_timestamp, i) if nsources > 1 else firemap_timestamp)
                        last_timestamps = firemap_timestamp

            if not lst_dates:
                err_msg = 's' if nsources > 1 else ''
                err_msg = f'fire map{err_msg} ({firemap_name}) do not contain any useful layer'
                raise TypeError(err_msg)

            try:
                df_dates = _pd.DataFrame(sorted(lst_dates), columns=col_names)
            except MemoryError:
                err_msg = 'pandas data frame (timestamps fires) was not created'
                raise MemoryError(err_msg)

        self._df_timestamps_firemaps = df_dates

    """
    Processing fire maps - layout of layers (MTBS and FireCCI) 
    """

    def __processLayersLayout_FIREMAPS(self, opt_select: FireMapSelectOpt = None):

        if not (isinstance(opt_select, FireMapSelectOpt) or opt_select is None):
            err_msg = f'__processLayersLayout_FIREMAPS() argument must be FireMapSelectOpt, not \'{type(opt_select)}\''
            raise TypeError(err_msg)

        if self._map_layout_firemaps is not None:
            return

        _opt_firemap = self.opt_select_firemap if opt_select is None else opt_select
        firemap_name = _opt_firemap.name.lower()

        if self._ds_firemaps is None:
            try:
                self.__loadGeoTIFF_FIREMAPS()
            except IOError:
                err_msg = 'cannot load any fire map ({}): {}'
                err_msg = err_msg.format(
                    firemap_name,
                    self.lst_firemaps
                )
                raise IOError(err_msg)

        map_layout_firemaps = {}
        pos = 0

        nsources = len(self._ds_firemaps)

        proc_msg = '({} fire map{})'.format(firemap_name, 's' if nsources > 1 else '')
        proc_msg = f'Processing layout of layers {proc_msg}'
        with elapsed_timer(proc_msg, enable=self.estimate_time):

            for i, img_ds in enumerate(self._ds_firemaps):
                last_timestamps = 0  # reset value of last date
                for rs_id in range(0, img_ds.RasterCount):

                    rs_band = img_ds.GetRasterBand(rs_id + 1)
                    rs_dsc = rs_band.GetDescription()

                    if _opt_firemap == FireMapSelectOpt.CCI and 'ConfidenceLevel' in rs_dsc:
                        firemap_timestamp = band2date_firecci(rs_dsc)
                    elif _opt_firemap == FireMapSelectOpt.MTBS and self.mtbs_region.value in rs_dsc:
                        firemap_timestamp = band2date_mtbs(rs_dsc)

                    if firemap_timestamp != last_timestamps:
                        map_layout_firemaps[pos] = (i, rs_id + 1) if nsources > 1 else rs_id + 1; pos += 1
                        last_timestamps = firemap_timestamp

        if not map_layout_firemaps:
            err_msg = f'fire maps ({firemap_name}) do not contain any useful layer'
            raise TypeError(err_msg)

        self._map_layout_firemaps = map_layout_firemaps
        self.__len_firemaps = pos

    def _processMetaData_FIREMAPS(self, opt_select: MetaDataSelectOpt = MetaDataSelectOpt.ALL) -> None:  # TODO private?

        if self._firemaps_processed:
            return

        if not isinstance(opt_select, MetaDataSelectOpt):
            err_msg = f'_processMetaData_FIREMAPS() argument must be MetaDataSelectOpt, not \'{type(opt_select)}\''
            raise TypeError(err_msg)

        if opt_select & MetaDataSelectOpt.TIMESTAMPS == MetaDataSelectOpt.TIMESTAMPS:
            if self._df_timestamps_firemaps is None:
                try:
                    self.__processTimestamps_FIREMAPS(opt_select=self.opt_select_firemap)
                except IOError or ValueError:
                    err_msg = ''
                    raise TypeError(err_msg)

        if opt_select & MetaDataSelectOpt.LAYOUT == MetaDataSelectOpt.LAYOUT:
            if self._map_layout_firemaps is None:
                try:
                    self.__processLayersLayout_FIREMAPS(opt_select=self.opt_select_firemap)
                except TypeError:
                    firemap_name = self.opt_select_firemap.name.lower()
                    err_msg = f'cannot process a layout of layers - fire maps ({firemap_name})'
                    raise TypeError(err_msg)

        if opt_select & MetaDataSelectOpt.ALL == opt_select.ALL:
            self._firemaps_processed = True

    """
    Loading sources - reflectance and temperature
    """

    def __loadSatData_ALLOC(self, extra_features: int = 0) -> None:

        if not self._satdata_processed: self._processMetadata_SATDATA()

        # TODO remove and create self._df_timestamps after processing meta data
        if self._df_timestamps_reflectance is not None:
            df_timestamps = self._df_timestamps_reflectance
            ds_satdata = self._ds_satdata_reflectance
        elif self._df_timestamps_temperature is not None:
            df_timestamps = self._df_timestamps_temperature
            ds_satdata = self._ds_satdata_temperature
        else:
            err_msg = ''  # TODO error message
            raise TypeError(err_msg)

        cnd_reflectance = (self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE and
                           self._ds_satdata_reflectance is not None)
        cnd_temperature = (self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE and
                           self._ds_satdata_temperature is not None)

        nfeatures = extra_features
        if cnd_reflectance: nfeatures += _NFEATURES_REFLECTANCE
        if cnd_temperature: nfeatures += 1

        # TODO alloc large memory with memory map
        # TODO alloc for cases when reflectance is not set
        rows = ds_satdata[0].RasterYSize; cols = ds_satdata[0].RasterXSize
        self._np_satdata = _np.empty(shape=(nfeatures * self._ntimestamps, rows, cols), dtype=_np.float32)

        if cnd_reflectance:
            idx = [True] * _NFEATURES_REFLECTANCE + [False] * (nfeatures - _NFEATURES_REFLECTANCE)
            idx = idx * self._ntimestamps
            self._np_satdata_reflectance = self._np_satdata[idx, :, :]

        if cnd_temperature:
            idx = []
            if cnd_reflectance: idx += [False] * _NFEATURES_REFLECTANCE
            idx = idx + [True] + [False] * extra_features
            idx = idx * self._ntimestamps

            self._np_satdata_temperature = self._np_satdata[idx, :, :]

        # raise NotImplementedError

    def _loadSatData_SELECTED_RANGE(self) -> None:

        raise NotImplementedError

    def __loadSatData_ALL_RASTERS(self, ds_satdata, np_satdata: _np.ndarray, type_name: str) -> _np.ndarray:

        len_ds = len(ds_satdata)

        if len_ds > 1:
            msg = f'Loading satdata sources ({type_name})'
            with (elapsed_timer(msg=msg, enable=self.estimate_time)):

                rstart = rend = 0

                for i, img_ds in enumerate(ds_satdata):
                    msg = f'Loading data from img #{i} ({type_name})'
                    with elapsed_timer(msg=msg, enable=self.estimate_time):
                        rend += img_ds.RasterCount; np_satdata[rstart:rend, :, :] = img_ds.ReadAsArray()
                        rstart = rend
        else:
            msg = f'Loading satdata source ({type_name})'
            with elapsed_timer(msg=msg, enable=self.estimate_time):
                np_satdata = ds_satdata[0].ReadAsArray()
                np_satdata = _np.moveaxis(np_satdata, 0, -1)
                np_satdata = np_satdata.astype(_np.float32)

        return np_satdata

    def loadSatData(self, extra_bands: int = 0) -> None:  # TODO protected?

        # TODO check allocated
        if not self._satdata_processed: self._processMetadata_SATDATA()

        # TODO remove and create self._df_timestamps after processing meta data
        if self._df_timestamps_reflectance is not None:
            df_timestamps = self._df_timestamps_reflectance
        elif self._df_timestamps_temperature is not None:
            df_timestamps = self._df_timestamps_temperature
        else:
            err_msg = ''  # TODO error message
            raise TypeError(err_msg)

        self.__loadSatData_ALLOC(extra_features=extra_bands)

        # TODO check loaded
        # TODO check if reflectance and temperature timestamps are same
        if isinstance(self.select_timestamps[0], _datetime.date):
            begin_timestamp = self.select_timestamps[0]
            end_timestamp = self.select_timestamps[1]
        else:
            raise NotImplementedError

        #
        cnd_reflectance = (self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE and
                           self._ds_satdata_reflectance is not None)
        cnd_temperature = (self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE and
                           self._ds_satdata_temperature is not None)

        if (begin_timestamp == df_timestamps['Timestamps'].iloc[0] and
                end_timestamp == df_timestamps['Timestamps'].iloc[-1]):

            if cnd_reflectance:
                type_name = SatDataSelectOpt.REFLECTANCE.name.lower()
                self._np_satdata_reflectance = self.__loadSatData_ALL_RASTERS(
                    ds_satdata=self._ds_satdata_reflectance, np_satdata=self._np_satdata_reflectance, type_name=type_name
                )

            if cnd_temperature:
                type_name = SatDataSelectOpt.TEMPERATURE.name.lower()
                self._np_satdata_temperature = self.__loadSatData_ALL_RASTERS(
                    ds_satdata=self._ds_satdata_temperature, np_satdata=self._np_satdata_temperature, type_name=type_name
                )

        else:

            raise NotImplementedError

            # begin_cond = self._df_timestamps_reflectance['Date'] == begin_timestamp
            # end_cond = self._df_timestamps_reflectance['Date'] == end_timestamp
            #
            # rs_id = (
            #     self._df_timestamps_reflectance.index[begin_cond][0],
            #     self._df_timestamps_reflectance.index[end_cond][0]
            # )

        # TODO comment
        self._np_satdata = _np.moveaxis(self._np_satdata, 0, -1)
        if cnd_reflectance: self._np_satdata_reflectance = _np.moveaxis(self._np_satdata_reflectance, 0, -1)
        if cnd_temperature: self._np_satdata_temperature = _np.moveaxis(self._np_satdata_temperature, 0, -1)

    """
    Loading fire maps - MTBS and FireCCI 
    """

    def _processConfidenceLevel_CCI(self, rs_ids) -> _np.ndarray:

        if isinstance(rs_ids, int): rs_ids = [rs_ids]

        rows = self._ds_firemaps[0].RasterYSize; cols = self._ds_firemaps[1].RasterXSize
        nmaps = len(rs_ids)

        np_confidence = _np.empty(shape=(rows, cols, nmaps), dtype=_np.float32) if nmaps > 1 \
            else _np.empty(shape=(rows, cols), dtype=_np.float32)
        np_flags = _np.copy(np_confidence)

        for sr_id, rs_id in enumerate(rs_ids):
            ds_id, local_rs_id = self._map_layout_firemaps[rs_id]

            rs_cl = self._ds_firemaps[ds_id].GetRasterBand(local_rs_id)
            rs_flags = self._ds_firemaps[ds_id].GetRasterBand(local_rs_id + 1)

            # check date
            if band2date_firecci(rs_cl.GetDescription()) != band2date_firecci(rs_flags.GetDescription()):
                err_mgs = 'Dates between ConfidenceLevel and ObservedFlag bands are not same!'
                raise ValueError(err_mgs)

            if nmaps > 1:
                np_confidence[:, :, sr_id] = rs_cl.ReadAsArray()
                np_flags[:, :, sr_id] = rs_flags.ReadAsArray()
            else:
                np_confidence[:, :] = rs_cl.ReadAsArray()
                np_flags[:, :] = rs_flags.ReadAsArray()

        if nmaps > 1:
            np_uncharted = _np.any(np_flags == -1, axis=-1)
            np_confidence_agg = _np.max(np_confidence, axis=-1)  # TODO also mean?
            np_confidence = np_confidence_agg; gc.collect()
        else:
            np_uncharted = np_confidence == -1

        np_confidence[np_uncharted] = -1  # TODO set Nan

        return np_confidence

    def _processSeverity_MTBS(self, rs_ids) -> _np.ndarray:

        if isinstance(rs_ids, int): rs_ids = [rs_ids]
        # TODO check input

        if not self._firemaps_processed: self._processMetaData_FIREMAPS()

        rows = self._ds_firemaps[0].RasterYSize; cols = self._ds_firemaps[1].RasterXSize
        nmaps = len(rs_ids)

        np_severity = _np.empty(shape=(rows, cols, nmaps), dtype=_np.float32) if nmaps > 1 \
            else _np.empty(shape=(rows, cols), dtype=_np.float32)

        for sr_id, rs_id in enumerate(rs_ids):
            ds_id, local_rs_id = self._map_layout_firemaps[rs_id]
            if nmaps > 1:
                np_severity[:, :, sr_id] = self._ds_firemaps[ds_id].GetRasterBand(local_rs_id).ReadAsArray()
            else:
                np_severity[:, :] = self._ds_firemaps[ds_id].GetRasterBand(local_rs_id).ReadAsArray()

        if nmaps > 1:
            np_uncharted = _np.any(np_severity == MTBSSeverity.NON_MAPPED_AREA.value, axis=-1)  # MTBSSeverity.NON_MAPPED_AREA.value
            np_severity_agg = _np.max(np_severity, axis=-1)  # TODO mean
            np_severity = np_severity_agg; gc.collect()  # clean up
        else:
            np_uncharted = np_severity == MTBSSeverity.NON_MAPPED_AREA.value

        np_severity[np_uncharted] = MTBSSeverity.NON_MAPPED_AREA.value  # TODO set nan?
        del np_uncharted; gc.collect()  # clean up

        return np_severity

    def loadFiremaps(self):  # TODO protected?

        if self._np_firemaps is not None: return
        if not self._firemaps_processed: self._processMetaData_FIREMAPS()

        if isinstance(self.select_timestamps[0], _datetime.date):
            begin_timestamp = self.select_timestamps[0]; end_timestamp = self.select_timestamps[1]
            months = [0] * 2

            if self.opt_select_firemap == FireMapSelectOpt.MTBS:
                months[0] = months[1] = 1
            elif self.opt_select_firemap == FireMapSelectOpt.CCI:
                months[0] = begin_timestamp.month
                months[1] = end_timestamp.month

            begin_timestamp = _datetime.date(year=begin_timestamp.year, month=months[0], day=1)
            end_timestamp = _datetime.date(year=end_timestamp.year, month=months[1], day=1)

            cnd_begin_idx = self._df_timestamps_firemaps['Timestamps'] == begin_timestamp
            begin_idx = self._df_timestamps_firemaps.index[cnd_begin_idx][0]

            if begin_timestamp != end_timestamp:
                cnd_end_idx = self._df_timestamps_firemaps['Timestamps'] == end_timestamp
                end_idx = self._df_timestamps_firemaps.index[cnd_end_idx][0]
                rs_ids = range(begin_idx, end_idx + 1)
            else:
                rs_ids = begin_idx
        else:
            raise NotImplementedError

        if self.opt_select_firemap == FireMapSelectOpt.MTBS:
            np_severity = self._processSeverity_MTBS(rs_ids=rs_ids)

            # convert severity to labels
            c1 = np_severity >= self.mtbs_min_severity.value; c2 = np_severity <= MTBSSeverity.HIGH.value
            self._np_firemaps = _np.logical_and(c1, c2).astype(_np.float32)

            # clean up
            del np_severity; gc.collect()
        elif self.opt_select_firemap == FireMapSelectOpt.CCI:
            np_confidence = self._processConfidenceLevel_CCI(rs_ids=rs_ids)

            # convert confidence level to labels
            c1 = np_confidence >= self.cci_confidence_level
            self._np_firemaps = c1.astype(_np.float32)

            # clean up
            del np_confidence; gc.collect()

    """
    TODO comment
    """

    def getTimeseriesLength(self, opt_select: SatDataSelectOpt) -> int:

        if not isinstance(opt_select, SatDataSelectOpt):
            err_msg = f'getTimeseriesLength() argument must be SatDataSelectOpt, not \'{type(opt_select)}\''
            raise TypeError(err_msg)

        if opt_select & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE:
            self._processLayersLayout_SATDATA_REFLECTANCE()
            return self.__len_ts_reflectance
        elif opt_select & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE:
            self._processLayersLayout_SATDATA_TEMPERATURE()
            return self.__len_ts_temperature
        else:
            raise NotImplementedError

    @property
    def len_ts(self) -> int:  # TODO rename

        len_ts_reflectance = len_ts_temperature = length_ts = 0

        if self.lst_satdata_reflectance is not None:
            length_ts = len_ts_reflectance = self.getTimeseriesLength(opt_select=SatDataSelectOpt.REFLECTANCE)
        if self.lst_satdata_temperature is not None:
            length_ts = len_ts_temperature = self.getTimeseriesLength(opt_select=SatDataSelectOpt.TEMPERATURE)

        if self.lst_satdata_temperature is not None and self.lst_satdata_reflectance is not None:
            if len_ts_reflectance != len_ts_temperature:
                err_msg = ''
                raise ValueError(err_msg)

        return length_ts

    @property
    def len_firemaps(self) -> int:  # TODO rename?

        # TODO improve implementation

        if not self._map_layout_firemaps:
            try:
                self._processMetaData_FIREMAPS(opt_select=MetaDataSelectOpt.LAYOUT)
            except TypeError:
                err_msg = ''
                raise TypeError(err_msg)

        return self.__len_firemaps


if __name__ == '__main__':

    VAR_DATA_DIR = 'data/tifs'

    VAR_PREFIX_IMG_REFLECTANCE = 'ak_reflec_january_december_{}_100km'
    VAR_PREFIX_IMG_LABELS = 'ak_january_december_{}_100km'

    VAR_LST_SATIMGS = []
    VAR_LST_FIREMAPS = []

    for year in range(2004, 2006):

        VAR_PREFIX_IMG_REFLECTANCE_YEAR = VAR_PREFIX_IMG_REFLECTANCE.format(year)
        VAR_PREFIX_IMG_LABELS_YEAR = VAR_PREFIX_IMG_LABELS.format(year)

        fn_satimg_reflec = '{}_epsg3338_area_0.tif'.format(VAR_PREFIX_IMG_REFLECTANCE_YEAR)
        fn_satimg_reflec = os.path.join(VAR_DATA_DIR, fn_satimg_reflec)
        VAR_LST_SATIMGS.append(fn_satimg_reflec)

        fn_labels_mtbs = '{}_epsg3338_area_0_mtbs_labels.tif'.format(VAR_PREFIX_IMG_LABELS_YEAR)
        fn_labels_mtbs = os.path.join(VAR_DATA_DIR, fn_labels_mtbs)
        VAR_LST_FIREMAPS.append(fn_labels_mtbs)

    # setup of data set loader
    dataset_loader = SatDataLoader(
        lst_firemaps=VAR_LST_FIREMAPS,
        lst_satdata_reflectance=VAR_LST_SATIMGS,
        estimate_time=True
    )

    print(dataset_loader.timestamps_firemaps)

    VAR_START_DATE = dataset_loader.timestamps_reflectance.iloc[0]['Timestamps']
    VAR_END_DATE = dataset_loader.timestamps_reflectance.iloc[-1]['Timestamps']

    dataset_loader.select_timestamps = (VAR_START_DATE, VAR_END_DATE)

    dataset_loader.loadSatData()
