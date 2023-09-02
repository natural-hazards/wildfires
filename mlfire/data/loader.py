# TODO rename -> loader.py -> load.py

import gc
import os

from enum import Enum
from typing import Union

# collections
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


class DatasetLoader(object):  # TODO rename to SatDataLoad and split for labels?

    def __init__(self,
                 lst_firemaps: Union[tuple[str], list[str]],
                 lst_satdata_reflectance: Union[tuple[str], list[str], None] = None,
                 lst_satdata_temperature: Union[tuple[str], list[str], None] = None,
                 # TODO add here vegetation indices and infrared bands
                 opt_select_firemap: FireMapSelectOpt = FireMapSelectOpt.MTBS,
                 # TODO comment
                 opt_select_satdata: Union[SatDataSelectOpt, list[SatDataSelectOpt]] = SatDataSelectOpt.ALL,
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

        self._ds_satdata_reflectance = None
        self._df_timestamps_reflectance = None

        self._ds_satdata_temperature = None
        self._df_timestamps_temperature = None

        self._ds_firemaps = None
        self._df_timestamps_firemaps = None

        self._map_layout_relectance = None  # TODO rename -> _layers_layout_reflectance
        self._map_layout_temperature = None  # TODO rename -> _layers_layout_temperature

        self._map_layout_firemaps = None  # TODO rename

        self.__len_ts_reflectance = 0
        self.__len_ts_temperature = 0

        self.__len_firemaps = 0

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

        self._labels_processed = False  # TODO rename

        # TODO comment

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
    def timestamps_firemaps(self) -> PandasDataFrame:

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

        if not self._labels_processed:
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

    def _reset(self):

        del self._ds_training; self._ds_training = None
        del self._ds_test; self._ds_test = None
        del self._ds_val; self._ds_val = None

        del self._df_timestamps_reflectance; self._df_timestamps_reflectance = None
        del self._map_layout_relectance; self._map_layout_relectance = None

        del self._df_timestamps_temperature; self._df_timestamps_temperature = None
        del self._map_layout_temperature; self._map_layout_temperature = None

        del self._df_timestamps_firemaps; self._df_timestamps_firemaps = None
        del self._map_layout_firemaps; self._map_layout_firemaps = None

        gc.collect()  # invoke garbage collector

        self._nfeatures_ts = 0  # TODO rename nbands_img?
        self.__len_ts_reflectance = 0
        self.__len_ts_temperature = 0

        self.__len_firemaps = 0  # TODO rename

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

    def __processTimestamps_SATDATA_REFLECTANCE(self) -> None:  # TODO merge with __processTimestamps_SATDATA_TEMPERATURE

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

    def __processTimestamps_SATDATA_TEMPERATURE(self) -> None:  # TODO merge with __processTimestamps_SATDATA_REFLECTANCE

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

        del self._df_timestamps_temperature; gc.collect()  # TODO remove?
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
            except MemoryError or ValueError:
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
            except MemoryError or ValueError:
                err_msg = 'cannot process timestamps - satellite data (temperature)'
                raise ValueError(err_msg)

    """
    Processing metadata and layout of layers - reflectance and temperature
    """

    def __processLayersLayout_SATDATA_REFLECTANCE(self) -> None:  # TODO merge with __processLayersLayout_SATDATA_TEMPERATURE

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

    def __processLayersLayout_SATDATA_TEMPERATURE(self) -> None:  # TODO merge with __processLayersLayout_SATDATA_REFLECTANCE

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

                    if ('lst_day_1km' in rs_dsc.lower()) and (temperature_date != last_date):  # is condition for date necessary?
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
    Processing timestamps - fire maps (MTBS and FireCCI)  
    """

    def __processTimestamps_FIREMAPS_CCI(self) -> None:  # TODO merge with __processTimestamps_FIREMAPS_MTBS

        if self._df_timestamps_firemaps is not None:
            return

        if self._ds_firemaps is None:
            try:
                self.__loadGeoTIFF_FIREMAPS()
            except IOError:
                err_msg = 'cannot load any source - FireCCI maps: {}'
                err_msg = err_msg.format(self.lst_firemaps)
                raise IOError(err_msg)

        nsources = len(self.lst_firemaps)

        lst_dates = []
        col_names = ['Timestamps', 'Image ID'] if nsources > 1 else ['Timestamps']

        proc_msg = 'Processing timestamps (FireCCI map{})'.format('s' if nsources > 1 else '')
        with elapsed_timer(msg=proc_msg, enable=self.estimate_time):

            for i, img_ds in enumerate(self._ds_firemaps):
                for rs_id in range(0, img_ds.RasterCount):

                    rs_band = img_ds.GetRasterBand(rs_id + 1)
                    rs_dsc = rs_band.GetDescription()

                    if 'ConfidenceLevel' in rs_dsc:
                        map_date_mtbs = band2date_firecci(rs_dsc)
                        lst_dates.append((map_date_mtbs, i) if nsources > 1 else map_date_mtbs)

            if not lst_dates:
                err_msg = 'sources (FireCCI map{}) do not contain any useful information about dates'
                err_msg = err_msg.format('s' if nsources > 1 else '')
                raise ValueError(err_msg)

            try:
                df_dates = _pd.DataFrame(sorted(lst_dates), columns=col_names)
            except MemoryError:
                err_msg = 'pandas data frame (timestamps fires) was not created'
                raise MemoryError(err_msg)

            # clean up
            del lst_dates; gc.collect()

        self._df_timestamps_firemaps = df_dates

    def __processTimestamps_FIREMAPS_MTBS(self) -> None:  # TODO merge with __processTimestamps_FIREMAPS_CCI

        if self._df_timestamps_firemaps is not None:
            return

        if self._ds_firemaps is None:
            try:
                self.__loadGeoTIFF_FIREMAPS()
            except IOError:
                err_msg = 'cannot load any source - MTBS maps: {}'
                err_msg = err_msg.format(self.lst_firemaps)
                raise IOError(err_msg)

        nsources = len(self.lst_firemaps)

        lst_dates = []
        col_names = ['Timestamps', 'Image ID'] if nsources > 1 else ['Timestamps']

        proc_msg = 'Processing timestamps (MTBS fire map{})'.format('s' if nsources > 1 else '')
        with elapsed_timer(msg=proc_msg, enable=self.estimate_time):

            for i, img_ds in enumerate(self._ds_firemaps):
                for rs_id in range(0, img_ds.RasterCount):

                    rs_band = img_ds.GetRasterBand(rs_id + 1)
                    rs_dsc = rs_band.GetDescription()

                    if self.mtbs_region.value in rs_dsc:  # TODO last date?

                        map_date_mtbs = band2date_mtbs(rs_dsc)
                        lst_dates.append((map_date_mtbs, i) if nsources > 1 else map_date_mtbs)

            if not lst_dates:
                err_msg = 'sources (MTBS map{}) do not contain any useful information about dates'
                err_msg = err_msg.format('s' if nsources > 1 else '')
                raise ValueError(err_msg)

            try:
                df_dates = _pd.DataFrame(sorted(lst_dates), columns=col_names)
            except MemoryError:
                err_msg = 'pandas data frame (timestamps fires) was not created'
                raise MemoryError(err_msg)

            # clean up
            del lst_dates; gc.collect()

        self._df_timestamps_firemaps = df_dates

    def __processTimestamps_FIREMAPS(self) -> None:

        # TODO improve implementation

        if self._df_timestamps_firemaps is not None:
            return

        if self._ds_firemaps is None:
            try:
                self.__loadGeoTIFF_FIREMAPS()
            except IOError:
                err_msg = 'cannot load any source - fire maps ({}): {}'
                err_msg = err_msg.format(
                    self.opt_select_firemap.name.lower(),
                    self.lst_firemaps
                )
                raise IOError(err_msg)

        try:
            if self.opt_select_firemap == FireMapSelectOpt.CCI:
                self.__processTimestamps_FIREMAPS_CCI()
            else:
                self.__processTimestamps_FIREMAPS_MTBS()
        except MemoryError or ValueError:
            err_msg = ''
            raise TypeError(err_msg)

    """
    labels
    """

    def __processLayersLayout_FIREMAPS(self, opt_select: FireMapSelectOpt = None):

        if not (isinstance(opt_select, FireMapSelectOpt) or None):
            err_msg = f'__processLayersLayout_FIREMAPS() argument must be FireMapSelectOpt, not \'{type(opt_select)}\''
            raise TypeError(err_msg)

        if self._map_layout_firemaps is not None:
            return self._map_layout_firemaps

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

                    if self.opt_select_firemap == FireMapSelectOpt.CCI and 'ConfidenceLevel' in rs_dsc:
                        firemap_timestamp = band2date_firecci(rs_dsc)
                    elif self.opt_select_firemap == FireMapSelectOpt.MTBS and self.mtbs_region.value in rs_dsc:
                        firemap_timestamp = band2date_mtbs(rs_dsc)

                    if firemap_timestamp != last_timestamps:
                        map_layout_firemaps[pos] = (i, rs_id + 1) if nsources > 1 else rs_id + 1; pos += 1
                        last_timestamps = firemap_timestamp

        if not map_layout_firemaps:
            err_msg = f'fire maps ({firemap_name}) do not contain any useful layer'
            raise TypeError(err_msg)

        self._map_layout_firemaps = map_layout_firemaps
        self.__len_firemaps = pos

    def _processMetaData_FIREMAPS(self, opt_select: MetaDataSelectOpt = MetaDataSelectOpt.ALL) -> None:

        if not isinstance(opt_select, MetaDataSelectOpt):
            err_msg = f'_processMetaData_FIREMAPS() argument must be MetaDataSelectOpt, not \'{type(opt_select)}\''
            raise TypeError(err_msg)

        if opt_select & MetaDataSelectOpt.TIMESTAMPS == MetaDataSelectOpt.TIMESTAMPS:
            if self._df_timestamps_firemaps is None:
                try:
                    self.__processTimestamps_FIREMAPS()  # TODO add argument opt_select_firemaps
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

        self._labels_processed = True

    """
    TODO comment
    """

    def getTimeseriesLength(self, opt_select: SatDataSelectOpt) -> int:

        if not isinstance(opt_select, SatDataSelectOpt):
            err_msg = f'getTimeseriesLength() argument must be SatDataSelectOpt, not \'{type(opt_select)}\''
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
            length_ts = len_ts_reflectance = self.getTimeseriesLength(opt_select=SatDataSelectOpt.REFLECTANCE)
        if self.lst_satdata_temperature is not None:
            length_ts = len_ts_temperature = self.getTimeseriesLength(opt_select=SatDataSelectOpt.SURFACE_TEMPERATURE)

        if self.lst_satdata_temperature is not None and self.lst_satdata_reflectance is not None:
            if len_ts_reflectance != len_ts_temperature:
                err_msg = ''
                raise ValueError(err_msg)

        return length_ts

    @property
    def len_firemaps(self) -> int:

        if not self._map_layout_firemaps:
            try:
                self._processMetaData_FIREMAPS(opt_select=MetaDataSelectOpt.LAYOUT)
            except TypeError:
                err_msg = ''
                raise TypeError(err_msg)

        return self.__len_firemaps


if __name__ == '__main__':

    # TODO fix constructor arguments

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

    print(dataset_loader.timestamps_wildfires)
