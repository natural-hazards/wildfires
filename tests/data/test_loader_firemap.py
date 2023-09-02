# TODO split to satellite data and fire maps

import pytest

from enum import Enum

from mlfire.utils.functool import lazy_import

# lazy imports - modules

_os = lazy_import('os')

_datetime = lazy_import('datetime')
_dateutil_relative = lazy_import('dateutil.relativedelta')

_pd = lazy_import('pandas')

_mlfire_data_loader = lazy_import('mlfire.data.loader')

# lazy imports - classes

_SatDataSelectOpt = _mlfire_data_loader.SatDataSelectOpt
_FireMapSelectOpt = _mlfire_data_loader.FireMapSelectOpt

_SatDataLoader = _mlfire_data_loader.DatasetLoader
_PandasDataFrame = _pd.DataFrame


class TestFireMapType(Enum):  # TODO remove?

    __test__ = False

    MTBS = 'mtbs'
    FIRECCI = 'cci'

    def __str__(self):

        return self.value


def lst_firemap_year(type_firemap: TestFireMapType, year: int = 2004) -> tuple[str]:

    satdata_dir = _os.path.abspath('data/tifs')
    prefix_firemap = 'ak_january_december_{}_100km'

    prefix_firemap_yr = prefix_firemap.format(year)

    fn_firemap = f'{prefix_firemap_yr}_epsg3338_area_0_{type_firemap}_labels.tif'
    fn_firemap = _os.path.join(satdata_dir, fn_firemap)
    fn_firemap = _os.path.join(satdata_dir, fn_firemap)

    return (fn_firemap,)


def lst_firemap_multi_year(type_firemap: TestFireMapType, range_year: range = range(2004, 2006)) -> list[str]:

    lst_firemap: list[str] = []

    firemap_dir = _os.path.abspath('data/tifs')
    prefix_firemap = 'ak_january_december_{}_100km'

    for year in range_year:
        prefix_firemap_yr = prefix_firemap.format(year)

        fn_firemap = f'{prefix_firemap_yr}_epsg3338_area_0_{type_firemap}_labels.tif'
        fn_firemap = _os.path.join(firemap_dir, fn_firemap)

        lst_firemap.append(fn_firemap)

    return lst_firemap


@pytest.fixture(scope='module')
def firemap_mtbs_2004() -> tuple[str]:

    return lst_firemap_year(type_firemap=TestFireMapType.MTBS, year=2004)


@pytest.fixture(scope='module')
def firemaps_mtbs_2004_2005() -> list[str]:

    return lst_firemap_multi_year(type_firemap=TestFireMapType.MTBS, range_year=range(2004, 2006))


@pytest.fixture(scope='module')
def firemaps_cci_2004() -> tuple[str]:

    return lst_firemap_year(type_firemap=TestFireMapType.FIRECCI, year=2004)


@pytest.fixture(scope='module')
def firemaps_cci_2004_2005() -> list[str]:

    return lst_firemap_multi_year(type_firemap=TestFireMapType.FIRECCI, range_year=range(2004, 2006))


@pytest.fixture(scope='module')
def expected_firemap_mtbs_timestamps_2004() -> _PandasDataFrame:

    lst_timestamps: tuple[_datetime.date] = (_datetime.date(year=2004, month=1, day=1), )
    return _pd.DataFrame(lst_timestamps, columns=('Timestamps',))


@pytest.fixture(scope='module')
def expected_firemap_mtbs_timestamps_2004_2005() -> _PandasDataFrame:

    lst_dates: list[_datetime.date] = []
    lst_dates.extend([(_datetime.date(year=year, month=1, day=1), year - 2004) for year in range(2004, 2006)])

    return _pd.DataFrame(lst_dates, columns=('Timestamps', 'Image ID'))


@pytest.fixture(scope='module')
def expected_firemap_cci_timestamps_2004() -> _PandasDataFrame:

    lst_timestamps: list[_datetime.date] = []

    start_date = _datetime.date(year=2004, month=1, day=1)
    lst_timestamps.extend([
        start_date + i * _dateutil_relative.relativedelta(months=1) for i in range(0, 12)
    ])

    return _pd.DataFrame(lst_timestamps, columns=('Timestamps',))


@pytest.fixture(scope='module')
def expected_firemap_cci_timestamps_2004_2005() -> _PandasDataFrame:

    lst_timestamps: list[_datetime.date] = []

    for year in range(2004, 2006):
        start_date = _datetime.date(year=year, month=1, day=1)
        lst_timestamps.extend([
            (start_date + i * _dateutil_relative.relativedelta(months=1), year - 2004) for i in range(0, 12)
        ])

    return _pd.DataFrame(lst_timestamps, columns=('Timestamps', 'Image ID'))


"""

"""


@pytest.mark.data
def TEST_SatDataLoader_TIMESTAMPS_FIREMAP_MTBS_2004(firemap_mtbs_2004, expected_firemap_mtbs_timestamps_2004):

    satdata_loader = _SatDataLoader(
        lst_firemaps=firemap_mtbs_2004,
        opt_select_firemap=_FireMapSelectOpt.MTBS,
        estimate_time=False
    )

    timestamps_firemaps = satdata_loader.timestamps_firemaps
    _pd.testing.assert_frame_equal(timestamps_firemaps, expected_firemap_mtbs_timestamps_2004)


@pytest.mark.data
def TEST_SatDataLoader_TIMESTAMPS_FIREMAP_MTBS_2004_2005(firemaps_mtbs_2004_2005, expected_firemap_mtbs_timestamps_2004_2005):

    satdata_loader = _SatDataLoader(
        lst_firemaps=firemaps_mtbs_2004_2005,
        opt_select_firemap=_FireMapSelectOpt.MTBS,
        estimate_time=False
    )

    timestamps_firemaps = satdata_loader.timestamps_firemaps
    _pd.testing.assert_frame_equal(timestamps_firemaps, expected_firemap_mtbs_timestamps_2004_2005)


@pytest.mark.data
@pytest.mark.parametrize('len_ts', [1])
def TEST_SatDataLoaderLenFiremaps_MTBS_2004(firemap_mtbs_2004, len_ts):

    satdata_loader = _SatDataLoader(
        lst_firemaps=firemap_mtbs_2004,
        opt_select_firemap=_FireMapSelectOpt.MTBS,
        estimate_time=False
    )

    assert satdata_loader.len_firemaps == len_ts


@pytest.mark.data
@pytest.mark.parametrize('len_ts', [2])
def TEST_SatDataLoaderLenFiremaps_MTBS_2004_2005(firemaps_mtbs_2004_2005, len_ts):

    satdata_loader = _SatDataLoader(
        lst_firemaps=firemaps_mtbs_2004_2005,
        opt_select_firemap=_FireMapSelectOpt.MTBS,
        estimate_time=False
    )

    assert satdata_loader.len_firemaps == len_ts


@pytest.mark.data
@pytest.mark.parametrize('exception', [pytest.raises(TypeError)])
def TEST_SatDataLoaderSourcesNotSet_FIREMAP_MTBS(exception):

    satdata_loader = _SatDataLoader(
        lst_firemaps=None,
        opt_select_firemap=_FireMapSelectOpt.MTBS,
        estimate_time=False
    )

    with exception:
        _ = satdata_loader.timestamps_firemaps


@pytest.mark.data
def TEST_SatDataLoader_TIMESTAMPS_FIREMAP_CCI_2004(firemaps_cci_2004, expected_firemap_cci_timestamps_2004):

    satdata_loader = _SatDataLoader(
        lst_firemaps=firemaps_cci_2004,
        opt_select_firemap=_FireMapSelectOpt.CCI,
        estimate_time=False
    )

    timestamps_firemaps = satdata_loader.timestamps_firemaps
    _pd.testing.assert_frame_equal(timestamps_firemaps, expected_firemap_cci_timestamps_2004)


@pytest.mark.data
def TEST_SatDataLoader_TIMESTAMPS_FIREMAP_CCI_2004_2005(firemaps_cci_2004_2005, expected_firemap_cci_timestamps_2004_2005):

    satdata_loader = _SatDataLoader(
        lst_firemaps=firemaps_cci_2004_2005,
        opt_select_firemap=_FireMapSelectOpt.CCI,
        estimate_time=False
    )

    timestamps_firemaps = satdata_loader.timestamps_firemaps
    _pd.testing.assert_frame_equal(timestamps_firemaps, expected_firemap_cci_timestamps_2004_2005)

@pytest.mark.data
@pytest.mark.parametrize('len_ts', [12])
def TEST_SatDataLoaderLenFiremaps_CCI_2004(firemaps_cci_2004, len_ts):

    satdata_loader = _SatDataLoader(
        lst_firemaps=firemaps_cci_2004,
        opt_select_firemap=_FireMapSelectOpt.CCI,
        estimate_time=False
    )

    assert satdata_loader.len_firemaps == len_ts


@pytest.mark.data
@pytest.mark.parametrize('len_ts', [24])
def TEST_SatDataLoaderLenFiremaps_CCI_2004_2005(firemaps_cci_2004_2005, len_ts):

    satdata_loader = _SatDataLoader(
        lst_firemaps=firemaps_cci_2004_2005,
        opt_select_firemap=_FireMapSelectOpt.CCI,
        estimate_time=False
    )

    assert satdata_loader.len_firemaps == len_ts


@pytest.mark.data
@pytest.mark.parametrize('exception', [pytest.raises(TypeError)])
def TEST_SatDataLoaderSourcesNotSet_FIREMAP_CCI(exception):

    satdata_loader = _SatDataLoader(
        lst_firemaps=None,
        opt_select_firemap=_FireMapSelectOpt.CCI,
        estimate_time=False
    )

    with exception:
        _ = satdata_loader.timestamps_firemaps
