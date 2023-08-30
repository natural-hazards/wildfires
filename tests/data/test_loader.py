import pytest

from enum import Enum

from mlfire.utils.functool import lazy_import

# lazy imports - modules
_datetime = lazy_import('datetime')
_os = lazy_import('os')

_pd = lazy_import('pandas')

_mlfire_data_loader = lazy_import('mlfire.data.loader')

# lazy imports - classes
_SatDataSelectOpt = _mlfire_data_loader.SatDataSelectOpt
_SatDataLoader = _mlfire_data_loader.DatasetLoader
_PandasDataFrame = _pd.DataFrame


class TestSatDataType(Enum):

    __test__ = False

    REFLECTANCE = 'reflec'
    TEMPERATURE = 'lst'

    def __str__(self):

        return self.value


def lst_satdata_year(type_satdata: TestSatDataType, year: int = 2004) -> tuple[str]:

    satdata_dir = _os.path.abspath('data/tifs')
    prefix_satdata = 'ak_{}_january_december_{}_100km'

    prefix_satdata_yr = prefix_satdata.format(type_satdata, year)

    fn_satimg = '{}_epsg3338_area_0.tif'.format(prefix_satdata_yr)
    fn_satimg = _os.path.join(satdata_dir, fn_satimg)
    fn_satimg = _os.path.join(satdata_dir, fn_satimg)

    return (fn_satimg,)


def lst_satdata_multi_year(type_satdata: TestSatDataType, range_year: range = range(2004, 2006)) -> list[str]:

    lst_satadata_reflec: list[str] = []

    satdata_dir = _os.path.abspath('data/tifs')
    prefix_satdata_reflectance = 'ak_{}_january_december_{}_100km' # TODO renmae

    for year in range_year:
        prefix_satdata_reflectance_yr = prefix_satdata_reflectance.format(type_satdata, year) # TODO rename

        fn_satimg_reflec = '{}_epsg3338_area_0.tif'.format(prefix_satdata_reflectance_yr)
        fn_satimg_reflec = _os.path.join(satdata_dir, fn_satimg_reflec)

        lst_satadata_reflec.append(fn_satimg_reflec)

    return lst_satadata_reflec


@pytest.fixture(scope='module')
def expected_satdata_timestamps_2004() -> _PandasDataFrame:

    lst_timestamps: list = []

    start_date = _datetime.date(year=2004, month=1, day=1)
    lst_timestamps.extend([
        start_date + i * _datetime.timedelta(days=8) for i in range(0, 46)
    ])

    return _pd.DataFrame(lst_timestamps, columns=('Timestamps',))


@pytest.fixture(scope='module')
def expected_satdata_timestamps_2004_2005() -> _PandasDataFrame:

    lst_dates: list[tuple] = []

    for year in range(2004, 2006):
        start_date = _datetime.date(year=year, month=1, day=1)
        lst_dates.extend([
            (start_date + i * _datetime.timedelta(days=8), 0 if year == 2004 else 1) for i in range(0, 46)
        ])

    return _pd.DataFrame(lst_dates, columns=('Timestamps', 'Image ID'))


@pytest.fixture(scope='module')
def satdata_reflectance_2004() -> tuple[str]:

    return lst_satdata_year(type_satdata=TestSatDataType.REFLECTANCE, year=2004)


@pytest.fixture(scope='module')
def satdata_reflectance_2004_2005() -> list[str]:

    return lst_satdata_multi_year(type_satdata=TestSatDataType.REFLECTANCE, range_year=range(2004, 2006))


@pytest.fixture(scope='module')
def satdata_temperature_2004() -> tuple[str]:

    return lst_satdata_year(type_satdata=TestSatDataType.TEMPERATURE, year=2004)


@pytest.fixture(scope='module')
def satdata_temperature_2004_2005() -> list[str]:

    return lst_satdata_multi_year(type_satdata=TestSatDataType.TEMPERATURE, range_year=range(2004, 2006))


"""
Testing functiononality for SatDataLoader (loader for satellite data - reflectance and temperature)
"""


@pytest.mark.data
def TEST_SatDataLoaderReflectanceTimestamps_2004(satdata_reflectance_2004, expected_satdata_timestamps_2004):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_reflectance=satdata_reflectance_2004,
        estimate_time=False
    )

    timestamps_reflectance = satdata_loader.timestamps_reflectance
    _pd.testing.assert_frame_equal(timestamps_reflectance, expected_satdata_timestamps_2004)


@pytest.mark.data
def TEST_SatDataLoaderReflectanceTimestamps_2004_2005(satdata_reflectance_2004_2005, expected_satdata_timestamps_2004_2005):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_reflectance=satdata_reflectance_2004_2005,
        estimate_time=False
    )

    timestamps_reflectance = satdata_loader.timestamps_reflectance
    _pd.testing.assert_frame_equal(timestamps_reflectance, expected_satdata_timestamps_2004_2005)


@pytest.mark.data
@pytest.mark.parametrize('len_ts', [46])
def TEST_SatDataLoaderReflectanceLengthTimeseries_2004(satdata_reflectance_2004, len_ts):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_reflectance=satdata_reflectance_2004,
        estimate_time=False
    )

    assert satdata_loader.getLengthTimeseries(opt_select=_SatDataSelectOpt.REFLECTANCE) == len_ts
    assert satdata_loader.len_ts == len_ts


@pytest.mark.data
@pytest.mark.parametrize('len_ts', [92])
def TEST_SatDataLoaderReflectanceLengthTimeseries_2004_2005(satdata_reflectance_2004_2005, len_ts):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_reflectance=satdata_reflectance_2004_2005,
        estimate_time=False
    )

    assert satdata_loader.getLengthTimeseries(opt_select=_SatDataSelectOpt.REFLECTANCE) == len_ts
    assert satdata_loader.len_ts == len_ts


@pytest.mark.data
@pytest.mark.parametrize('exception', [pytest.raises(TypeError)])
def TEST_SatDataLoaderReflectanceSourcesNotSet(satdata_temperature_2004, exception):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_temperature=satdata_temperature_2004,
        estimate_time=False
    )

    with exception:
        _ = satdata_loader.timestamps_reflectance


@pytest.mark.data
def TEST_SatDataLoaderTemperatureTimestamps_2004(satdata_temperature_2004, expected_satdata_timestamps_2004):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_temperature=satdata_temperature_2004,
        estimate_time=False
    )

    timestamps_temperature = satdata_loader.timestamps_temperature
    _pd.testing.assert_frame_equal(timestamps_temperature, expected_satdata_timestamps_2004)


@pytest.mark.data
def TEST_SatDataLoaderTemperatureTimestamps_2004_2005(satdata_temperature_2004_2005, expected_satdata_timestamps_2004_2005):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_temperature=satdata_temperature_2004_2005,
        estimate_time=False
    )

    timestamps_temperature = satdata_loader.timestamps_temperature
    _pd.testing.assert_frame_equal(timestamps_temperature, expected_satdata_timestamps_2004_2005)


@pytest.mark.data
@pytest.mark.parametrize('len_ts', [46])
def TEST_SatDataLoaderTemperatureLengthTimeseries_2004(satdata_temperature_2004, len_ts):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_temperature=satdata_temperature_2004,
        estimate_time=False
    )

    assert satdata_loader.getLengthTimeseries(opt_select=_SatDataSelectOpt.SURFACE_TEMPERATURE) == len_ts
    assert satdata_loader.len_ts == len_ts


@pytest.mark.data
@pytest.mark.parametrize('len_ts', [92])
def TEST_SatDataLoaderTemperatureLengthTimeseries_2004_2005(satdata_temperature_2004_2005, len_ts):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_temperature=satdata_temperature_2004_2005,
        estimate_time=False
    )

    assert satdata_loader.getLengthTimeseries(opt_select=_SatDataSelectOpt.SURFACE_TEMPERATURE) == len_ts
    assert satdata_loader.len_ts == len_ts


@pytest.mark.data
@pytest.mark.parametrize('exception', [pytest.raises(TypeError)])
def TEST_SatDataLoaderTemperatureSourcesNotSet(satdata_reflectance_2004, exception):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_reflectance=satdata_reflectance_2004,
        estimate_time=False
    )

    with exception:
        _ = satdata_loader.timestamps_temperature
