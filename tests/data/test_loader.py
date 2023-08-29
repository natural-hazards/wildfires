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


def lst_satdata(data_type: TestSatDataType) -> list[str]:

    lst_satadata_reflec: list[str] = []

    satdata_dir = _os.path.abspath('data/tifs')
    prefix_satdata_reflectance = 'ak_{}_january_december_{}_100km'

    for year in range(2004, 2006):
        prefix_satdata_reflectance_yr = prefix_satdata_reflectance.format(data_type, year)

        fn_satimg_reflec = '{}_epsg3338_area_0.tif'.format(prefix_satdata_reflectance_yr)
        fn_satimg_reflec = _os.path.join(satdata_dir, fn_satimg_reflec)

        lst_satadata_reflec.append(fn_satimg_reflec)

    return lst_satadata_reflec


@pytest.fixture(scope='module')
def expected_satdata_timestamps() -> _PandasDataFrame:

    lst_dates: list[tuple] = []

    for year in range(2004, 2006):
        start_date = _datetime.date(year=year, month=1, day=1)
        lst_dates.extend([
            (start_date + i * _datetime.timedelta(days=8), 0 if year == 2004 else 1) for i in range(0, 46)
        ])

    return _pd.DataFrame(lst_dates, columns=('Date', 'Image ID'))


@pytest.fixture(scope='module')
def satdata_reflectance() -> list[str]:

    return lst_satdata(TestSatDataType.REFLECTANCE)


@pytest.fixture(scope='module')
def satdata_temperature() -> list[str]:

    return lst_satdata(TestSatDataType.TEMPERATURE)


"""
Testing functiononality for SatDataLoader (loader for satellite data - reflectance and temperature)
"""


@pytest.mark.data
def TEST_SatDataLoaderReflectanceTimestamps(satdata_reflectance, expected_satdata_timestamps):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_reflectance=satdata_reflectance,
        estimate_time=False
    )

    timestamps_reflectance = satdata_loader.timestamps_reflectance
    _pd.testing.assert_frame_equal(timestamps_reflectance, expected_satdata_timestamps)


@pytest.mark.data
@pytest.mark.parametrize('len_ts', [92])
def TEST_SatDataLoaderReflectanceLengthTimeseries(satdata_reflectance, len_ts):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_reflectance=satdata_reflectance,
        estimate_time=False
    )

    assert satdata_loader.getLengthTimeseries(opt_select=_SatDataSelectOpt.REFLECTANCE) == len_ts
    assert satdata_loader.len_ts == len_ts


@pytest.mark.data
@pytest.mark.parametrize('exception', [pytest.raises(TypeError)])
def TEST_SatDataLoaderReflectanceSourcesNotSet(satdata_temperature, exception):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_temperature=satdata_temperature,
        estimate_time=False
    )

    with exception:
        _ = satdata_loader.timestamps_reflectance


@pytest.mark.data
def TEST_SatDataLoaderTemperatureTimestamps(satdata_temperature, expected_satdata_timestamps):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_temperature=satdata_temperature,
        estimate_time=False
    )

    timestamps_temperature = satdata_loader.timestamps_temperature
    _pd.testing.assert_frame_equal(timestamps_temperature, expected_satdata_timestamps)


@pytest.mark.data
@pytest.mark.parametrize('len_ts', [92])
def TEST_SatDataLoaderTemperatureLengthTimeseries(satdata_temperature, len_ts):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_temperature=satdata_temperature,
        estimate_time=False
    )

    assert satdata_loader.getLengthTimeseries(opt_select=_SatDataSelectOpt.SURFACE_TEMPERATURE) == len_ts
    assert satdata_loader.len_ts == len_ts


@pytest.mark.data
@pytest.mark.parametrize('exception', [pytest.raises(TypeError)])
def TEST_SatDataLoaderTemperatureSourcesNotSet(satdata_temperature, exception):

    satdata_loader = _SatDataLoader(
        lst_labels_locfires=None,
        lst_satdata_reflectance=satdata_temperature,
        estimate_time=False
    )

    with exception:
        _ = satdata_loader.timestamps_temperature
