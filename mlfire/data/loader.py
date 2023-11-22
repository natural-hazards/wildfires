# TODO rename -> loader.py -> load.py

import gc
import itertools
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

_gdal = lazy_import('osgeo.gdal')
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
        if not isinstance(other, SatDataSelectOpt):
            err_msg = f'unsuported operand type(s) for &: {type(self)} and {type(other)}'
            raise TypeError(err_msg)
        
        return SatDataSelectOpt(self.value & other.value)

    def __or__(self, other):
        if not isinstance(other, SatDataSelectOpt):
            err_msg = f'unsuported operand type(s) for |: {type(self)} and {type(other)}'
            raise TypeError(err_msg)

        return SatDataSelectOpt(self.value | other.value)

    def __eq__(self, other):
        if other is None:
            return False
        elif not isinstance(other, SatDataSelectOpt):
            return False
        else:
            return self.value == other.value

    def __str__(self) -> str:
        return self.name.lower()


class SatDataFeatures(Enum):  # TODO or rename SatDataFeaturesName

    RED = 'red'  # visible (wave length 620–670nm)
    NIR = 'nir'  # near infra-red (wave length 841–876nm)
    BLUE = 'blue'  # visible (wave length 459–479nm)
    GREEN = 'green'  # visible (wave length 545–565nm)
    SWIR1 = 'swir1'  # short-wave infra-red (wave length 1230–1250nm)
    SWIR2 = 'swir2'  # short-wave infra-red (wave length 1628-1652nm)
    SWIR3 = 'swir3'  # short-wave infra-red (wave length 2105-2155nm)
    TEMPERATURE = 'temperature'  # TODO comment

    def __str__(self):
        return self.value

    # TODO and?
    # TODO eq


class FireMapSelectOpt(Enum):

    MTBS = 1
    CCI = 2

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        elif not isinstance(other, FireMapSelectOpt):
            return False
        else:
            return self.value == other.value

    def __str__(self) -> str:
        return self.name


class MetaDataSelectOpt(Enum):

    NONE = 0
    TIMESTAMPS = 1
    LAYOUT = 2
    ALL = 3

    def __and__(self, other):
        if not isinstance(other, MetaDataSelectOpt):
            err_msg = f'unsuported operand type(s) for &: {type(self)} and {type(other)}'
            raise TypeError(err_msg)
        return MetaDataSelectOpt(self.value & other.value)

    def __or__(self, other):
        if not isinstance(other, MetaDataSelectOpt):
            err_msg = f'unsuported operand type(s) for |: {type(self)} and {type(other)}'
            raise TypeError(err_msg)
        return MetaDataSelectOpt(self.value | other.value)

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        elif not isinstance(other, MetaDataSelectOpt):
            return False
        else:
            return self.value == other.value


class SatDataLoader(object):

    def __init__(self,
                 lst_firemaps: Union[tuple[str], list[str], None],
                 lst_satdata_reflectance: Union[tuple[str], list[str], None] = None,
                 lst_satdata_temperature: Union[tuple[str], list[str], None] = None,
                 # TODO comment
                 opt_select_firemap: FireMapSelectOpt = FireMapSelectOpt.MTBS,
                 # TODO comment
                 opt_select_satdata: Union[SatDataSelectOpt, list[SatDataSelectOpt], None] = SatDataSelectOpt.ALL,
                 # TODO comment
                 select_timestamps: Union[list, tuple, None] = None,  # TODO rename -> lst_selected_timestamps
                 # TODO comment
                 cci_confidence_level: int = 70,
                 # TODO comment
                 mtbs_region: MTBSRegion = MTBSRegion.ALASKA,
                 mtbs_min_severity: MTBSSeverity = MTBSSeverity.LOW,
                 # TODO comment
                 estimate_time: bool = True):

        self._np_satdata = None
        self._np_firemaps = None

        self._ds_satdata_reflectance = None
        self._df_timestamps_reflectance = None

        self._ds_satdata_temperature = None
        self._df_timestamps_temperature = None

        self._ds_firemaps = None
        self._df_timestamps_firemaps = None

        self._layout_layers_reflectance = None
        self._layout_layers_temperature = None
        self._layout_layers_firemaps = None  # TODO property

        self.__rs_rows_satadata = -1
        self.__rs_cols_satdata = -1

        self.__rs_rows_firemaps = -1
        self.__rs_cols_firemaps = -1

        self.__len_ts_reflectance = 0  # TODO remove?
        self.__len_ts_temperature = 0  # TODO remove?
        self.__len_ts_firemaps = 0  # TODO remove?

        self.__lst_features = None

        self.__shape_satdata = None
        self.__shape_firemaps = None

        # properties sources - reflectance and land surface temperature

        self.__lst_satdata_reflectance = None
        self.lst_satdata_reflectance = lst_satdata_reflectance

        self.__lst_satdata_temperature = None
        self.lst_satdata_temperature = lst_satdata_temperature

        self.__opt_select_satdata = None
        self.opt_select_satdata = opt_select_satdata

        self._satdata_processed = False  # TODO set as private

        # properties source - fire maps (wildfire locations)

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

        self._firemaps_processed = False  # TODO set as private

        # timestamps

        self.__df_timestamps = None  # TODO remove?

        self.__selected_timestamps = None  # TODO rename -> __lst_selected_timestamps
        self.selected_timestamps = select_timestamps  # -> __lst_selected_timestamps

        self.__ntimestamps = -1  # TODO remove
        self.__timestamps_processed = False

        # measure time

        self.__estimate_time = None
        self.estimate_time = estimate_time

    """
    Satellite data (sources) - reflectance and temperature
    """

    @property
    def selected_timestamps(self) -> Union[list, tuple, None]:  # TODO rename -> lst_selected_timestamps and return tuple

        return self.__selected_timestamps

    @selected_timestamps.setter
    def selected_timestamps(self, timestamps: Union[list, tuple, None]) -> None:  # TODO rename -> lst_selected_timestamps and return tuple

        # TODO check input

        if self.__selected_timestamps == timestamps:
            return

        self._reset()  # clean up
        self.__selected_timestamps = timestamps

    @property
    def opt_select_satdata(self) -> SatDataSelectOpt:  # TODO return tuple

        return self.__opt_select_satdata

    @opt_select_satdata.setter
    def opt_select_satdata(self, opt_select: SatDataSelectOpt) -> None:

        # TODO improve implementation
        # TODO check list

        if self.opt_select_satdata == opt_select:
            return

        if isinstance(opt_select, (list, tuple)):
            _flgs = 0

            for opt in opt_select: _flgs |= opt
            opt_select = _flgs

        self._reset()  # clean up
        self.__opt_select_satdata = opt_select

    @property
    def lst_satdata_reflectance(self) -> Union[tuple[str], list[str], None]:  # TODO return tuple

        return self.__lst_satdata_reflectance

    @lst_satdata_reflectance.setter
    def lst_satdata_reflectance(self, lst_fn: Union[tuple[str], list[str]]) -> None:

        # TODO check input type

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
    def lst_satdata_temperature(self) -> Union[tuple[str], list[str]]:  # TODO return tuple

        return self.__lst_satdata_temperature

    @lst_satdata_temperature.setter
    def lst_satdata_temperature(self, lst_fn: Union[tuple[str], list[str]]):

        # todo check input type

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
    Timestamps - reflectance, land surface temperature, and firemaps for wildfire localization
    """

    @property
    def timestamps_reflectance(self) -> _PandasDataFrame:

        # TODO check if reflectance is selected

        if self._df_timestamps_reflectance is None:
            self._processTimestamps_SATDATA(opt_select=SatDataSelectOpt.REFLECTANCE)

            if self._df_timestamps_reflectance is None:
                err_msg = 'data frame containing timestamps (reflectance) was not created'
                raise TypeError(err_msg)

        return self._df_timestamps_reflectance

    @property
    def timestamps_temperature(self) -> _PandasDataFrame:

        # TODO check if temperature is selected

        if self._df_timestamps_temperature is None:
            self._processTimestamps_SATDATA(opt_select=SatDataSelectOpt.TEMPERATURE)

            if self._df_timestamps_temperature is None:
                err_msg = 'data frame containing timestamps (temperature) was not created'
                raise TypeError(err_msg)

        return self._df_timestamps_temperature

    @property
    def timestamps_satdata(self) -> _PandasDataFrame:

        # TODO df_timestamps_satdata as attribute

        df_timestamps = None

        if self.opt_select_satdata == SatDataSelectOpt.NONE:
            if self.lst_satdata_reflectance is not None:
                df_timestamps = self.timestamps_reflectance
            elif self.lst_satdata_temperature is not None:
                df_timestamps = self.timestamps_temperature
            else:
                raise TypeError  # TODO check if this error right
        else:
            df_timestamps_reflectance = df_timestamps_temperature = None

            cnd_reflectance = self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE
            cnd_reflectance &= self.lst_satdata_reflectance is not None

            cnd_temperature = self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE
            cnd_temperature &= self.lst_satdata_temperature is not None

            if cnd_reflectance: df_timestamps = df_timestamps_reflectance = self.timestamps_reflectance
            if cnd_temperature: df_timestamps = df_timestamps_temperature = self.timestamps_temperature

            if cnd_reflectance and cnd_temperature:
                if not df_timestamps_temperature.equals(df_timestamps_reflectance):
                    raise TypeError  # TODO check error is right

        return df_timestamps  # TODO as private attribute

    def __getSelectedTimestamps_SATDATA_FROM_LIST(self, lst_timestamps_satdata: list) -> _PandasDataFrame:

        if not isinstance(lst_timestamps_satdata, (list, tuple)):
            raise NotImplementedError

        if len(lst_timestamps_satdata) != 2:
            raise NotImplementedError

        if isinstance(lst_timestamps_satdata[0], (_datetime.date, int)):
            timestamp_start = lst_timestamps_satdata[0]
            timestamp_end = lst_timestamps_satdata[1]

            if isinstance(lst_timestamps_satdata[0], int):
                timestamp_start = self.timestamps_satdata.iloc[timestamp_start]['Timestamps']
                timestamp_end = self.timestamps_satdata.iloc[timestamp_end]['Timestamps']

            cnd = timestamp_start == self.timestamps_satdata.iloc[0]['Timestamps']
            cnd &= timestamp_end == self.timestamps_satdata.iloc[-1]['Timestamps']

            if cnd:
                df_timestamps = self.timestamps_satdata
            else:
                df_timestamps = self.timestamps_satdata[
                    (self.timestamps_satdata['Timestamps'] >= timestamp_start) &
                    (self.timestamps_satdata['Timestamps'] <= timestamp_end)
                ]
        else:
            raise NotImplementedError

        return df_timestamps

    def __getSelectedTimestamps_SATDATA_FROM_LIST_LIST(self, lst_timestamps_satdata: list) -> _PandasDataFrame:

        if not isinstance(lst_timestamps_satdata, (list, tuple)):
            raise NotImplementedError

        df_timestamps = None
        for timestamps in lst_timestamps_satdata:
            df = self.__getSelectedTimestamps_SATDATA_FROM_LIST(
                lst_timestamps_satdata=timestamps
            )
            df_timestamps = df if df_timestamps is None else _pd.concat([df_timestamps, df])

        return df_timestamps

    @property
    def selected_timestamps_satdata(self) -> _PandasDataFrame:

        if not isinstance(self.selected_timestamps, (list, tuple)):
            raise NotImplementedError

        if isinstance(self.selected_timestamps[0], (int, _datetime.date)):
            df_timestamps = self.__getSelectedTimestamps_SATDATA_FROM_LIST(
                lst_timestamps_satdata=self.selected_timestamps)
        else:
            df_timestamps = self.__getSelectedTimestamps_SATDATA_FROM_LIST_LIST(
                lst_timestamps_satdata=self.selected_timestamps
            )

        # df_timestamps as private attribute
        return df_timestamps

    @property
    def timestamps_firemaps(self) -> _PandasDataFrame:

        if self._df_timestamps_firemaps is None:
            self.__processTimestamps_FIREMAPS()

            if self._df_timestamps_firemaps is None:
                err_msg = 'data frame containing timestamps (fire location) was not created!'
                raise TypeError(err_msg)

        return self._df_timestamps_firemaps

    """
    TODO comment
    """

    @property
    def _layout_layers_reflectance(self) -> Union[dict, None]:   # TODO rename

        if self.__layout_layers_reflectance is not None:
            return self.__layout_layers_reflectance

        # TODO try-except
        self.__processLayersLayout_SATDATA_REFLECTANCE()
        return self.__layout_layers_reflectance

    @_layout_layers_reflectance.setter
    def _layout_layers_reflectance(self, layout) -> None:
        # TODO check input type
        self.__layout_layers_reflectance = layout

    @_layout_layers_reflectance.deleter
    def _layout_layers_reflectance(self) -> None:
        del self.__layout_layers_reflectance
        self.__layout_layers_reflectance = None

    @property
    def _layout_layers_temperature(self) -> Union[dict, None]:  # TODO rename
        if self.__layout_layers_temperature is not None:
            return self.__layout_layers_temperature

        # TODO try-except
        self.__processLayersLayout_SATDATA_TEMPERATURE()
        return self.__layout_layers_temperature

    @_layout_layers_temperature.setter
    def _layout_layers_temperature(self, layout) -> None:
        # TODO check input type
        self.__layout_layers_temperature = layout

    @_layout_layers_temperature.deleter
    def _layout_layers_temperature(self) -> None:
        del self.__layout_layers_temperature
        self.__layout_layers_temperature = None

    """
    TODO comment
    """

    @property
    def _ds_satdata_reflectance(self):
        if self.__ds_satdata_reflectance is not None:
            return self.__ds_satdata_reflectance

        cnd_reflectance = self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE
        if ~cnd_reflectance: return None

        # TODO try-except
        self.__loadGeoTIFF_DATASETS_REFLECTANCE()
        return self.__ds_satdata_reflectance

    @_ds_satdata_reflectance.setter
    def _ds_satdata_reflectance(self, ds) -> None:
        # TODO check input
        self.__ds_satdata_reflectance = ds

    @_ds_satdata_reflectance.deleter
    def _ds_satdata_reflectance(self) -> None:
        del self.__ds_satdata_reflectance
        self.__ds_satdata_reflectance = None

    @property
    def _ds_satdata_temperature(self):
        if self.__ds_satdata_temperature is not None:
            return self.__ds_satdata_temperature

        cnd_temperature = self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE
        if ~cnd_temperature: return None

        # TODO try-except
        self.__loadGeoTIFF_DATASETS_TEMPERATURE()
        return self.__ds_satdata_temperature

    @_ds_satdata_temperature.setter
    def _ds_satdata_temperature(self, ds) -> None:
        # TODO check input type
        self.__ds_satdata_temperature = ds

    @_ds_satdata_temperature.deleter
    def _ds_satdata_temperature(self) -> None:
        del self.__ds_satdata_temperature
        self.__ds_satdata_temperature = None

    """
    FireCII (ESA firemaps) firemaps properties
    """

    @property
    def cci_confidence_level(self) -> int:

        return self.__cci_confidence_level

    @cci_confidence_level.setter
    def cci_confidence_level(self, level: int) -> None:

        # TODO check input type

        if self.__cci_confidence_level == level:
            return

        if level < 0 or level > 100:
            raise ValueError('confidence level for FireCCI firemaps must be positive int between 0 and 100')

        self.__cci_confidence_level = level

    """
    MTBS firemaps properties
    """

    @property
    def mtbs_region(self) -> MTBSRegion:

        return self.__mtbs_region

    @mtbs_region.setter
    def mtbs_region(self, region: MTBSRegion) -> None:

        # TODO check input type

        if self.__mtbs_region == region:
            return

        self._reset()
        self.__mtbs_region = region

    @property
    def mtbs_min_severity(self) -> MTBSSeverity:

        return self.__mtbs_min_severity

    @mtbs_min_severity.setter
    def mtbs_min_severity(self, severity: MTBSSeverity) -> None:

        # TODO check input type

        if self.__mtbs_min_severity == severity:
            return

        self._reset()
        self.__mtbs_min_severity = severity

    """
    Labels related to wildfires 
    """

    @property
    def lst_firemaps(self) -> Union[tuple[str], list[str]]:  # TODO return tuple

        return self.__lst_firemap

    @lst_firemaps.setter
    def lst_firemaps(self, lst_firemaps: Union[tuple[str], list[str]]) -> None:

        # TODO check input type

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

        # TODO check input type

        if self.opt_select_firemap == opt_select:
            return

        self._reset()  # clean up
        self.__opt_select_firemap = opt_select

    # Time measure

    @property
    def estimate_time(self) -> bool:

        return self.__estimate_time

    @estimate_time.setter
    def estimate_time(self, flg: bool) -> None:

        if not isinstance(flg, bool):
            return

        self.__estimate_time = flg

    def _reset(self):

        # TODO check if all values are set to default

        del self._df_timestamps_reflectance; self._df_timestamps_reflectance = None
        del self._layout_layers_reflectance; self._layout_layers_reflectance = None
        del self._ds_satdata_reflectance; self._ds_satdata_reflectance = None

        del self._df_timestamps_temperature; self._df_timestamps_temperature = None
        del self._layout_layers_temperature; self._layout_layers_temperature = None
        del self._ds_satdata_temperature; self._ds_satdata_temperature = None

        del self._df_timestamps_firemaps; self._df_timestamps_firemaps = None
        del self._layout_layers_firemaps; self._layout_layers_firemaps = None

        gc.collect()  # invoke garbage collector

        self.__len_ts_reflectance = 0
        self.__len_ts_temperature = 0
        self.__len_ts_firemaps = 0

        # TODO comment
        self.__ntimestamps = -1  # TODO remove
        self.__timestamps_processed = False

        # set flags to false
        self._satdata_processed = False  # TODO private?
        self._firemaps_processed = False  # TODO private?

    """
    Load sources - reflectance, land surface temperature, or firemaps
    """

    @staticmethod
    def __loadGeoTIFF_DATASETS(lst_sources: Union[list[str], tuple[str]]) -> list:  # TODO return tuple?

        # TODO check input type

        lst_ds = []
        for fn in lst_sources:
            try:
                ds = _gdal.Open(fn)
            except IOError:
                raise RuntimeWarning(f'cannot load source {fn}')

            if ds is None:
                raise RuntimeWarning(f'source {fn} is empty')
            lst_ds.append(ds)

        return lst_ds

    def __loadGeoTIFF_DATASETS_REFLECTANCE(self) -> None:

        if self.lst_satdata_reflectance is None:
            err_msg = 'satellite data (reflectance) is not set'
            raise TypeError(err_msg)

        del self._ds_satdata_reflectance; gc.collect()
        self._ds_satdata_reflectance = None

        self._ds_satdata_reflectance = self.__loadGeoTIFF_DATASETS(self.lst_satdata_reflectance)
        if not self._ds_satdata_reflectance:
            err_msg = 'satellite data (reflectance) was not loaded'
            raise IOError(err_msg)

    def __loadGeoTIFF_DATASETS_TEMPERATURE(self) -> None:

        if self.lst_satdata_temperature is None:
            err_msg = 'satellite data (temperature) is not set'
            raise TypeError(err_msg)

        del self._ds_satdata_temperature; gc.collect()
        self._ds_satdata_temperature = None

        self._ds_satdata_temperature = self.__loadGeoTIFF_DATASETS(self.lst_satdata_temperature)
        if not self._ds_satdata_temperature:
            err_msg = 'satellite data (temperature) was not loaded'
            raise IOError(err_msg)

    def _loadGeoTIFF_DATASETS_SATDATA(self, opt_select: SatDataSelectOpt):

        # TODO check input type

        if opt_select & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE:
            self.__loadGeoTIFF_DATASETS_REFLECTANCE()
        elif opt_select & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE:
            self.__loadGeoTIFF_DATASETS_TEMPERATURE()
        else:
            # TODO warning?
            pass

    def __loadGeoTIFF_DATASETS_FIREMAP(self) -> None:

        if not self.lst_firemaps or self.lst_firemaps is None:
            err_msg = 'firemap sources are not set!'
            raise TypeError(err_msg)

        del self._ds_firemaps; gc.collect()
        self._ds_firemaps = None

        self._ds_firemaps = self.__loadGeoTIFF_DATASETS(self.lst_firemaps)
        if not self._ds_firemaps:
            err_msg = 'satellite data (firemaps - wildfires localization) is not set!'  # TODO rename
            raise IOError(err_msg)

    """
    Processing timestamps - satellite data (reflectance and temperature)  
    """

    def __processTimestamps_SATDATA_REFLECTANCE(self) -> None:

        if self._df_timestamps_reflectance is not None:
            return

        if self._ds_satdata_reflectance is None:
            try:
                self.__loadGeoTIFF_DATASETS_REFLECTANCE()
            except IOError:
                err_msg = 'cannot load any source - satellite data (reflectance): {}'
                err_msg = err_msg.format(self.lst_satdata_reflectance)
                raise IOError(err_msg)

        nsources = len(self.lst_satdata_reflectance)

        unique_dates = set()
        col_names = ['Timestamps', 'Image ID'] if nsources > 1 else ['Timestamps']

        with elapsed_timer(msg='processing timestamps (reflectance)', enable=self.estimate_time):
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

        self._df_timestamps_reflectance = df_dates

    def __processTimestamps_SATDATA_TEMPERATURE(self) -> None:

        if self._df_timestamps_temperature is not None:
            return

        if self._ds_satdata_temperature is None:
            try:
                self.__loadGeoTIFF_DATASETS_TEMPERATURE()
            except IOError:
                err_msg = 'cannot load any source - satellite data (temperature): {}'
                err_msg = err_msg.format(self.lst_satdata_temperature)
                raise IOError(err_msg)

        nsources = len(self.lst_satdata_temperature)

        lst_dates = []
        col_names = ['Timestamps', 'Image ID'] if nsources > 1 else ['Timestamps']

        with elapsed_timer(msg='processing timestamps (temperature)', enable=self.estimate_time):
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
            err_msg = f'__processTimestamps_FIREMAPS() argument must be FireMapSelectOpt, not {type(opt_select)}'
            raise TypeError(err_msg)

        # processing reflectance (MOD09A1)

        if self.lst_satdata_reflectance is not None and \
           (opt_select & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE):

            if self._ds_satdata_reflectance is None:
                try:
                    self.__loadGeoTIFF_DATASETS_REFLECTANCE()
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
                    self.__loadGeoTIFF_DATASETS_TEMPERATURE()
                except IOError:
                    err_msg = 'cannot load any source - satellite data (temperature): {}'
                    err_msg = err_msg.format(self.lst_satdata_temperature)
                    raise IOError(err_msg)

            try:
                self.__processTimestamps_SATDATA_TEMPERATURE()
            except MemoryError or ValueError:
                err_msg = 'cannot process timestamps - satellite data (temperature)'
                raise ValueError(err_msg)

        # TODO set timestamps flag

    """
    Processing metadata and layout of layers - reflectance and temperature
    """

    def __processLayersLayout_SATDATA_REFLECTANCE(self) -> None:

        if self.__layout_layers_reflectance is not None:
            return

        if self._ds_satdata_reflectance is None:  # TODO remove
            try:
                self.__loadGeoTIFF_DATASETS_REFLECTANCE()
            except IOError:
                err_msg = 'cannot load any source - satellite data (reflectance): {}'
                err_msg = err_msg.format(self.lst_satdata_reflectance)
                raise IOError(err_msg)

        map_layout_satdata = {}
        pos = 0

        nsources = len(self._ds_satdata_reflectance)

        with elapsed_timer('processing layout of layers (reflectance)', enable=self.estimate_time):
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
        self.__len_ts_reflectance = pos  # TODO is this attribute necessary?

    def __processLayersLayout_SATDATA_TEMPERATURE(self) -> None:

        if self.__layout_layers_temperature is not None:
            return

        if self._ds_satdata_temperature is None:
            try:
                self.__loadGeoTIFF_DATASETS_TEMPERATURE()
            except IOError:
                err_msg = 'cannot load any source - satellite data (temperature): {}'
                err_msg = err_msg.format(self.lst_satdata_temperature)
                raise IOError(err_msg)

        map_layout_satdata = {}
        pos = 0

        nsources = len(self._ds_satdata_temperature)

        with elapsed_timer('processing layout of layers (temperature)', enable=self.estimate_time):
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
        self.__len_ts_temperature = pos  # TODO is this attribute necessary?

    def _processMetadata_SATDATA(self) -> None:

        if self._satdata_processed: return

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
                self.__processLayersLayout_SATDATA_REFLECTANCE()
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
                self.__processLayersLayout_SATDATA_TEMPERATURE()
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
                self.__loadGeoTIFF_DATASETS_FIREMAP()
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
        proc_msg = f'processing timestamps {proc_msg}'

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

        if self._layout_layers_firemaps is not None:
            return

        _opt_firemap = self.opt_select_firemap if opt_select is None else opt_select
        firemap_name = _opt_firemap.name.lower()

        if self._ds_firemaps is None:
            try:
                self.__loadGeoTIFF_DATASETS_FIREMAP()
            except IOError:
                err_msg = 'cannot load any fire map ({}): {}'
                err_msg = err_msg.format(
                    firemap_name,
                    self.lst_firemaps
                )
                raise IOError(err_msg)

        map_layout_firemaps = {}  # TODO rename
        pos = 0

        nsources = len(self._ds_firemaps)

        proc_msg = '({} fire map{})'.format(firemap_name, 's' if nsources > 1 else '')
        proc_msg = f'processing layout of layers {proc_msg}'

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

        self._layout_layers_firemaps = map_layout_firemaps
        self.__len_ts_firemaps = pos  # TODO is this attribute necessary

    def _processMetaData_FIREMAPS(self, opt_select: MetaDataSelectOpt = MetaDataSelectOpt.ALL) -> None:

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
                    err_msg = ''  # TODO error message
                    raise TypeError(err_msg)

        if opt_select & MetaDataSelectOpt.LAYOUT == MetaDataSelectOpt.LAYOUT:
            if self._layout_layers_firemaps is None:
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

    def __allocSatDataBuffer(self) -> None:

        if self._np_satdata is not None: return

        rows, cols, _, len_features = self.shape_satdata
        len_timestamps = len(self.selected_timestamps_satdata)

        np_shape = (len_timestamps * len_features, rows, cols)
        # TODO memmap?
        self._np_satdata = _np.empty(shape=np_shape, dtype=_np.float32)

    def __loadSatData_ALL_RASTERS(self, ds_satdata: list[_gdal.Dataset, ...], np_satdata: _np.ndarray) -> _np.ndarray:

        if not isinstance(ds_satdata, list) and not isinstance(ds_satdata[0], _gdal.Dataset):
            err_msg = ''  # TODO error message
            raise TypeError(err_msg)

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument #2: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        len_ds = len(ds_satdata)

        if len_ds > 1:
            rstart = rend = 0
            for i, img_ds in enumerate(ds_satdata):
                msg = f'loading data from img #{i}'
                with elapsed_timer(msg=msg, enable=self.estimate_time):
                    rend += img_ds.RasterCount; np_satdata[rstart:rend, :, :] = img_ds.ReadAsArray()
                    rstart = rend
        else:
            np_satdata[:, :, :] = ds_satdata[0].ReadAsArray()
            # np_satdata = _np.moveaxis(np_satdata, 0, -1)  # TODO possible error
            np_satdata = np_satdata.astype(_np.float32)

        return np_satdata

    def __loadSatData_FOR_SELECTED_TIMESTAMPS(self, ds_satdata: list[_gdal.Dataset, ...], np_satdata: _np.ndarray,
                                              layout_layers: dict, nfeatures: int) -> _np.ndarray:

        if not isinstance(ds_satdata, list) and not isinstance(ds_satdata[0], _gdal.Dataset):
            err_msg = ''  # TODO error message
            raise TypeError(err_msg)

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument #2: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if not isinstance(layout_layers, dict):
            err_msg = ''  # TODO error message
            raise TypeError(err_msg)

        if not isinstance(nfeatures, int):
            err_msg = ''
            raise TypeError(err_msg)

        lst_ds_ids = self.selected_timestamps_satdata['Image ID'].unique()
        pos = 0

        for ds_id in lst_ds_ids:
            cnd = self.selected_timestamps_satdata['Image ID'] == ds_id
            timestamps = _pd.to_datetime(self.selected_timestamps_satdata['Timestamps'][cnd])

            # get all years in image
            years = timestamps.dt.year.unique()
            for y in years:
                range_days = _pd.date_range(start=f'01/01/{y}', end=f'12/31/{y}', freq='8D')
                img_year = timestamps[timestamps[cnd].dt.year == y]

                if img_year.equals(range_days):
                    rend = pos + ds_satdata[ds_id].RasterCount
                    np_satdata[pos:rend, :, :] = ds_satdata[ds_id].ReadAsArray()
                else:
                    for i in img_year.index:
                        _, rs_id_start = layout_layers[i]
                        img_ds = ds_satdata[ds_id]
                        for feature_id in range(nfeatures):
                            np_satdata[pos, :, :] = img_ds.GetRasterBand(rs_id_start + feature_id).ReadAsArray()
                            pos += 1

    def _loadSatData_IMPL(self, ds_satdata: list[_gdal.Dataset, ...], np_satdata: _np.ndarray, layout_layers: dict, nfeatures: int) -> _np.ndarray:

        if not isinstance(ds_satdata, list) and not isinstance(ds_satdata[0], _gdal.Dataset):
            err_msg = ''
            raise TypeError(err_msg)

        if not isinstance(np_satdata, _np.ndarray):
            err_msg = f'unsupported type of argument #2: {type(np_satdata)}, this argument must be a numpy array.'
            raise TypeError(err_msg)

        if not isinstance(layout_layers, dict):
            err_msg = ''
            raise TypeError(err_msg)

        if not isinstance(nfeatures, int):
            err_msg = ''
            raise TypeError(err_msg)

        if self.timestamps_satdata['Timestamps'].equals(self.selected_timestamps_satdata['Timestamps']):
            return self.__loadSatData_ALL_RASTERS(ds_satdata=ds_satdata, np_satdata=np_satdata)
        else:
            return self.__loadSatData_FOR_SELECTED_TIMESTAMPS(
                ds_satdata=ds_satdata, np_satdata=np_satdata, layout_layers=layout_layers, nfeatures=nfeatures
            )

    def __loadSatData_REFLETANCE(self) -> None:  # TODO fix name -> __loadSatData_REFLECTANCE

        rows, cols, _, len_features = self.shape_satdata
        len_timestamps = len(self.selected_timestamps_satdata)

        nfeatures = int(_NFEATURES_REFLECTANCE)

        idx = list(itertools.chain(
            *[range(i * len_features, i * len_features + nfeatures) for i in range(len_timestamps)]
        ))

        msg = 'loading satellite data (reflectance)'
        with elapsed_timer(msg=msg, enable=self.estimate_time):
            self._np_satdata[idx, ...] = self._loadSatData_IMPL(
                ds_satdata=self._ds_satdata_reflectance,
                np_satdata=self._np_satdata[idx, :, :],
                layout_layers=self._layout_layers_reflectance,
                nfeatures=nfeatures
            )

        # scale pixel values using MODIS09 scale factor (0.0001) for reflectance
        # see https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1
        self._np_satdata[idx, :, :] *= 1e-4

    def __loadSatData_TEMPERATURE(self) -> None:

        cnd_reflectance = (self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE and
                           self._ds_satdata_reflectance is not None)

        _, _, _, len_features = self.shape_satdata
        len_timestamps = len(self.selected_timestamps_satdata)

        idx_start = _NFEATURES_REFLECTANCE if cnd_reflectance else 0
        idx = slice(idx_start, len_timestamps * len_features, len_features)

        msg = 'loading satellite data (temperature)'
        with elapsed_timer(msg=msg, enable=self.estimate_time):
            self._np_satdata[idx, ...] = self._loadSatData_IMPL(
                ds_satdata=self._ds_satdata_temperature,
                np_satdata=self._np_satdata[idx, :, :],
                layout_layers=self._layout_layers_temperature,
                nfeatures=1
            )

        # scale pixel values using MODIS11 scale factor (0.02) for temperature
        # see https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1
        self._np_satdata[idx, :, :] *= 1e-2

    def loadSatData(self) -> None:

        if self._np_satdata is not None: return

        # TODO comment
        # TODO try-except
        self.__allocSatDataBuffer()

        # conditions
        cnd_reflectance = (self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE and
                           self._ds_satdata_reflectance is not None)
        cnd_temperature = (self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE and
                           self._ds_satdata_temperature is not None)

        # TODO comment
        self._processMetadata_SATDATA()

        if cnd_reflectance:
            self.__loadSatData_REFLETANCE()
        if cnd_temperature:
            self.__loadSatData_TEMPERATURE()

        # TODO comment
        self._np_satdata = _np.moveaxis(self._np_satdata, 0, -1)

    """
    Loading fire maps - MTBS and FireCCI 
    """

    def _processConfidenceLevel_CCI(self, rs_ids) -> _np.ndarray:

        if isinstance(rs_ids, int): rs_ids = (rs_ids,)
        # TODO check input

        rows = self._ds_firemaps[0].RasterYSize; cols = self._ds_firemaps[0].RasterXSize
        nmaps = len(rs_ids)

        np_confidence = _np.empty(shape=(rows, cols, nmaps), dtype=_np.float32) if nmaps > 1 \
            else _np.empty(shape=(rows, cols), dtype=_np.float32)
        np_flags = _np.copy(np_confidence)

        for sr_id, rs_id in enumerate(rs_ids):
            ds_id, local_rs_id = self._layout_layers_firemaps[rs_id]

            rs_cl = self._ds_firemaps[ds_id].GetRasterBand(local_rs_id)
            rs_flags = self._ds_firemaps[ds_id].GetRasterBand(local_rs_id + 1)

            # check date
            if band2date_firecci(rs_cl.GetDescription()) != band2date_firecci(rs_flags.GetDescription()):
                err_mgs = 'dates between ConfidenceLevel and ObservedFlag bands are not same!'
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

        rows = self._ds_firemaps[0].RasterYSize; cols = self._ds_firemaps[0].RasterXSize  # TODO rows, cols as property
        nmaps = len(rs_ids)

        np_severity = _np.empty(shape=(rows, cols, nmaps), dtype=_np.float32) if nmaps > 1 \
            else _np.empty(shape=(rows, cols), dtype=_np.float32)

        for sr_id, rs_id in enumerate(rs_ids):
            if nmaps > 1:
                ds_id, local_rs_id = self._layout_layers_firemaps[rs_id]
                np_severity[:, :, sr_id] = self._ds_firemaps[ds_id].GetRasterBand(local_rs_id).ReadAsArray()
            else:
                local_rs_id = self._layout_layers_firemaps[rs_id]
                np_severity[:, :] = self._ds_firemaps[0].GetRasterBand(local_rs_id).ReadAsArray()

        if nmaps > 1:
            np_uncharted = _np.any(np_severity == MTBSSeverity.NON_MAPPED_AREA.value, axis=-1)  # MTBSSeverity.NON_MAPPED_AREA.value
            np_severity_agg = _np.max(np_severity, axis=-1)  # TODO mean
            np_severity = np_severity_agg; gc.collect()  # clean up
        else:
            np_uncharted = np_severity == MTBSSeverity.NON_MAPPED_AREA.value  # TODO use __eq__

        # TODO remove?
        np_severity[np_uncharted] = _np.nan  # MTBSSeverity.NON_MAPPED_AREA.value  # TODO set nan?
        del np_uncharted; gc.collect()  # clean up

        return np_severity

    def loadFiremaps(self):  # TODO protected?

        # TODO fix implementation

        if self._np_firemaps is not None: return
        if not self._firemaps_processed: self._processMetaData_FIREMAPS()

        # TODO fix this
        # if isinstance(self.selected_timestamps_satdata
        begin_timestamp = self.selected_timestamps_satdata.iloc[0]['Timestamps']
        end_timestamp = self.selected_timestamps_satdata.iloc[-1]['Timestamps']
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
            rs_ids = (begin_idx,)

        if self.opt_select_firemap == FireMapSelectOpt.MTBS:
            np_severity = self._processSeverity_MTBS(rs_ids=rs_ids)

            # convert severity to fire map (label)
            # TODO use __eq__, ...
            c1 = np_severity >= self.mtbs_min_severity.value; c2 = np_severity <= MTBSSeverity.HIGH.value
            uncharted = _np.isnan(np_severity)  # TODO rename

            self._np_firemaps = _np.logical_and(c1, c2).astype(_np.float32)
            self._np_firemaps[uncharted] = _np.nan

            # clean up
            del np_severity; gc.collect()
        elif self.opt_select_firemap == FireMapSelectOpt.CCI:
            np_confidence = self._processConfidenceLevel_CCI(rs_ids=rs_ids)

            # convert confidence level to firemaps
            c1 = np_confidence >= self.cci_confidence_level
            self._np_firemaps = c1.astype(_np.float32)
            # TODO add nans

            # clean up
            del np_confidence; gc.collect()

    """
    Shape (satellite data) 
    """

    @property
    def _rs_rows_satdata(self) -> int:

        if self.__rs_rows_satadata > -1: return self.__rs_rows_satadata

        cnd_reflectance_sel = self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE
        cnd_temperature_sel = self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE

        cnd_reflectance_lst_set = self.lst_satdata_reflectance is not None
        cnd_reflectance = (cnd_reflectance_lst_set & cnd_reflectance_sel)
        cnd_reflectance |= ~cnd_reflectance_sel & ~cnd_temperature_sel & cnd_reflectance_lst_set

        cnd_temperature = self.lst_satdata_temperature is not None

        if cnd_reflectance:
            if not self._ds_satdata_reflectance: self.__loadGeoTIFF_DATASETS_REFLECTANCE()
            self.__rs_rows_satadata = self._ds_satdata_reflectance[0].RasterYSize

            if not cnd_reflectance_sel:
                del self._ds_satdata_reflectance; self._ds_satdata_reflectance = None
                gc.collect()
        elif cnd_temperature:
            if not self._ds_satdata_temperature: self.__loadGeoTIFF_DATASETS_TEMPERATURE()
            self.__rs_rows_satadata = self._ds_satdata_temperature[0].RasterYSize

            if not cnd_temperature_sel:
                del self._ds_satdata_temperature; self._ds_satdata_temperature = None
                gc.collect()
        else:
            err_msg = 'satdata was not provided'
            raise FileNotFoundError(err_msg)

        return self.__rs_rows_satadata

    @property
    def _rs_cols_satdata(self) -> int:

        if self.__rs_cols_satdata > -1: return self.__rs_cols_satdata

        cnd_reflectance_sel = self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE
        cnd_temperature_sel = self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE

        cnd_reflectance_lst_set = self.lst_satdata_reflectance is not None
        cnd_reflectance = (cnd_reflectance_lst_set & cnd_reflectance_sel)
        cnd_reflectance |= ~cnd_reflectance_sel & ~cnd_temperature_sel & cnd_reflectance_lst_set

        cnd_temperature = self.lst_satdata_temperature is not None

        if cnd_reflectance:
            # TODO try-except
            if not self._ds_satdata_reflectance: self.__loadGeoTIFF_DATASETS_REFLECTANCE() # TODO remove
            self.__rs_cols_satdata = self._ds_satdata_reflectance[0].RasterXSize

            if not cnd_reflectance_sel:
                del self._ds_satdata_reflectance; self._ds_satdata_reflectance = None
                gc.collect()
        elif cnd_temperature:
            # TODO try-except
            if not self._ds_satdata_temperature: self.__loadGeoTIFF_DATASETS_TEMPERATURE()  # TODO remove
            self.__rs_cols_satdata = self._ds_satdata_temperature[0].RasterXSize

            if not cnd_temperature_sel:
                del self._ds_satdata_temperature; self._df_timestamps_temperature = None  # TODO check
                gc.collect()
        else:
            err_msg = 'satdata was not provided'
            raise FileNotFoundError(err_msg)

        return self.__rs_cols_satdata

    @property
    def features(self) -> tuple:

        if self.__lst_features is not None: return self.__lst_features

        cnd_reflectance_sel = self.opt_select_satdata & SatDataSelectOpt.REFLECTANCE == SatDataSelectOpt.REFLECTANCE
        cnd_reflectance_sel &= self.lst_satdata_reflectance is not None

        cnd_temperature_sel = self.opt_select_satdata & SatDataSelectOpt.TEMPERATURE == SatDataSelectOpt.TEMPERATURE
        cnd_temperature_sel &= self.lst_satdata_temperature is not None

        lst_features = []

        if cnd_reflectance_sel:
            lst_features.extend([
                str(SatDataFeatures.RED),   # visible (wave length 620–670nm)
                str(SatDataFeatures.NIR),   # near infra-red (wave length 841–876nm)
                str(SatDataFeatures.BLUE),   # visible (wave length 459–479nm)
                str(SatDataFeatures.GREEN),   # visible (wave length 545–565nm)
                str(SatDataFeatures.SWIR1),   # short-wave infra-red (wave length 1230–1250nm)
                str(SatDataFeatures.SWIR2),   # short-wave infra-red (wave length 1628-1652nm)
                str(SatDataFeatures.SWIR3),   # short-wave infra-red (wave length 2105-2155nm)
            ])

        if cnd_temperature_sel:
            lst_features.append(
                str(SatDataFeatures.TEMPERATURE)  # TODO comment
            )

        # convert to tuple
        self.__lst_features = tuple(lst_features) if lst_features is not None else None
        return self.__lst_features

    @property
    def shape_satdata(self) -> tuple:

        if self.__shape_satdata is not None: return self.__shape_satdata

        rows = self._rs_rows_satdata
        cols = self._rs_cols_satdata

        len_ts = len(self.timestamps_satdata)
        len_features = len(self.features) if self.features is not None else 0

        self.__shape_satdata = (rows, cols, len_ts, len_features)
        return self.__shape_satdata

    # TODO shape_selected_satdata

    """
    Shape (fire maps) 
    """

    @property
    def _rs_rows_firemap(self) -> int:

        if self.__rs_rows_firemaps > -1: return self.__rs_rows_firemaps

        if self._ds_firemaps is None: self.__loadGeoTIFF_DATASETS_FIREMAP()
        self.__rs_rows_firemaps = self._ds_firemaps[0].RasterYSize

        return self.__rs_rows_firemaps

    @property
    def _rs_cols_firemap(self) -> int:

        if self.__rs_cols_firemaps > -1: return self.__rs_cols_firemaps

        if self._ds_firemaps is None: self.__loadGeoTIFF_DATASETS_FIREMAP()
        self.__rs_cols_firemaps = self._ds_firemaps[0].RasterXSize

        return self.__rs_cols_firemaps

    @property
    def shape_firemap(self) -> tuple:

        if self.__shape_firemaps is not None: return self.__shape_firemaps

        rows = self._rs_rows_firemap; cols = self._rs_cols_firemap
        len_ts = len(self.timestamps_firemaps)

        self.__shape_firemaps = (rows, cols, len_ts)
        return self.__shape_firemaps


if __name__ == '__main__':

    VAR_DATA_DIR = 'data/tifs'

    VAR_PREFIX_IMG_REFLECTANCE = 'ak_reflec_january_december_{}_100km'
    VAR_PREFIX_IMG_TEMPERATURE = 'ak_lst_january_december_{}_100km'
    VAR_PREFIX_IMG_FIREMAPS = 'ak_january_december_{}_100km'

    VAR_LST_REFLECTANCE = []
    VAR_LST_TEMPERATURE = []
    VAR_LST_FIREMAPS = []

    for year in range(2004, 2006):

        VAR_PREFIX_IMG_REFLECTANCE_YEAR = VAR_PREFIX_IMG_REFLECTANCE.format(year)
        VAR_PREFIX_IMG_TEMPERATURE_YEAR = VAR_PREFIX_IMG_TEMPERATURE.format(year)
        VAR_PREFIX_IMG_LABELS_YEAR = VAR_PREFIX_IMG_FIREMAPS.format(year)

        fn_satimg_reflec = '{}_epsg3338_area_0.tif'.format(VAR_PREFIX_IMG_REFLECTANCE_YEAR)
        fn_satimg_reflec = os.path.join(VAR_DATA_DIR, fn_satimg_reflec)
        VAR_LST_REFLECTANCE.append(fn_satimg_reflec)

        fn_satimg_temperature = '{}_epsg3338_area_0.tif'.format(VAR_PREFIX_IMG_TEMPERATURE_YEAR)
        fn_satimg_temperature = os.path.join(VAR_DATA_DIR, fn_satimg_temperature)
        VAR_LST_TEMPERATURE.append(fn_satimg_temperature)

        fn_labels_mtbs = '{}_epsg3338_area_0_mtbs_labels.tif'.format(VAR_PREFIX_IMG_LABELS_YEAR)
        fn_labels_mtbs = os.path.join(VAR_DATA_DIR, fn_labels_mtbs)
        VAR_LST_FIREMAPS.append(fn_labels_mtbs)

    # setup of data set loader
    dataset_loader = SatDataLoader(
        lst_firemaps=VAR_LST_FIREMAPS,
        lst_satdata_reflectance=VAR_LST_REFLECTANCE,
        lst_satdata_temperature=VAR_LST_TEMPERATURE,
        opt_select_satdata=SatDataSelectOpt.ALL,
        estimate_time=True
    )

    VAR_START_IDX = 0
    VAR_END_IDX = -1

    VAR_START_DATE = dataset_loader.timestamps_satdata.iloc[VAR_START_IDX]['Timestamps']
    VAR_END_DATE = dataset_loader.timestamps_satdata.iloc[VAR_END_IDX]['Timestamps']

    print(dataset_loader.timestamps_satdata)

    dataset_loader.selected_timestamps = ((0, 45), (50, 70))
    # dataset_loader.selected_timestamps = (VAR_START_IDX, VAR_END_IDX)
    # dataset_loader.selected_timestamps = (VAR_START_DATE, VAR_END_DATE)

    # print(dataset_loader.timestamps_firemaps)

    print(dataset_loader.timestamps_satdata)
    print(dataset_loader.selected_timestamps_satdata)

    dataset_loader.loadSatData()

    # VAR_START_DATE = dataset_loader.timestamps_satdata.iloc[0]['Timestamps']
    # VAR_END_DATE = dataset_loader.timestamps_satdata.iloc[-1]['Timestamps']

    # dataset_loader.selected_timestamps = (VAR_START_DATE, VAR_END_DATE)
    # print(dataset_loader.selected_timestamps_satdata.iloc[0]['Timestamps'])
    #
    # print(dataset_loader.shape_satdata)
    # print(dataset_loader.shape_firemap)
    #
    # print(f'len ts: {len(dataset_loader.timestamps_satdata)}')
    # print(f'len ts: {len(dataset_loader.timestamps_firemaps)}')

    # print(dataset_loader.selected_timestamps_satdata)
    # dataset_loader.loadSatData()
