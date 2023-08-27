import pytest

from enum import Enum

from mlfire.utils.functool import lazy_import

# lazy imports - modules
_datetime = lazy_import('datetime')
_os = lazy_import('os')

_pd = lazy_import('pandas')

_mlfire_data_loader = lazy_import('mlfire.data.loader')

# lazy imports - classes
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
Testing functiononality for SatDataLoader
"""


@pytest.mark.data
def test_satdata_loader_reflectance_timestamps(satdata_reflectance, expected_satdata_timestamps):

    satdata_loader = _SatDataLoader(
        lst_labels=None,
        lst_satdata_reflectance=satdata_reflectance
    )

    df_timestamps = satdata_loader.timestamps_reflectance
    _pd.testing.assert_frame_equal(df_timestamps, expected_satdata_timestamps)


@pytest.mark.data
def test_satdata_loader_temperature_timestamps(satdata_temperature, expected_satdata_timestamps):

    satdata_loader = _SatDataLoader(
        lst_labels=None,
        lst_satdata_temperature=satdata_temperature
    )

    # TODO check when ask from satdata are not set

    df_timestamps = satdata_loader.timestamps_temperature
    _pd.testing.assert_frame_equal(df_timestamps, expected_satdata_timestamps)


# TODO fire dates
