import ee as earthengine

from enum import Enum


class ModisReflectanceSpectralBands(Enum):  # TODO rename

    RED = 1  # visible (wave length 620–670nm)
    NIR = 2  # near infra-red (wave length 841–876nm)
    BLUE = 3  # visible (wave length 459–479nm)
    GREEN = 4  # visible (wave length 545–565nm)
    SWIR1 = 5  # short-wave infra-red (wave length 1230–1250nm)
    SWIR2 = 6  # short-wave infra-red (wave length 1628-1652nm)
    SWIR3 = 7  # short-wave infra-red (wave length 2105-2155nm)

    def __add__(self, other) -> int:
        if not isinstance(other, int):
            err_msg = ''
            raise TypeError(err_msg)

        return self.value + other

    def __sub__(self, other) -> int:
        if not isinstance(other, int):
            err_msg = ''
            raise TypeError(err_msg)

        return self.value - other

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        elif isinstance(other, ModisReflectanceSpectralBands):
            return self == other
        else:
            raise NotImplementedError


class ModisCollection(Enum):

    LST = 'MODIS/061/MOD11A2'  # land surface temperature # TODO rename to LAND_TEMPERATURE
    REFLECTANCE = 'MODIS/061/MOD09A1'  # spectral reflectance bands


class FireLabelCollection(Enum):

    CCI = 'ESA/CCI/FireCCI/5_1'
    MTBS = 'USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1'


class FireLabelsCollectionID(Enum):

    CII = 0
    MTBS = 1


class FireCIIAvailability(Enum):

    BEGIN = 2001
    END = 2020


class MTBSSeverity(Enum):

    LOW = 2
    MODERATE = 3
    HIGH = 4
    INCREASE_GREENNESS = 5
    NON_MAPPED_AREA = 6  # TODO rename -> UNCHARTED_AREA


class MTBSRegion(Enum):

    ALASKA = 'AK'
    CONTINENTAL_USA = 'CONUS'
    HAWAI = 'HI'
    PUERTO_RICO = 'PR'


class EarthEngineFireDatasets(object):

    def __init__(self, ds_name):

        self._ds_name = ds_name

    # ESA fire cii dataset
    class FireCII:

        @staticmethod
        def getBurnArea(confidence_level, start_date='2001-01-01', end_date='2020-01-01'):

            def filter_confidence(value: float):
                def fn_filter(image):
                    mask = image.select('ConfidenceLevel').gte(value)
                    return image.updateMask(mask)

                return fn_filter

            ds_fire_cci = earthengine.ImageCollection('ESA/CCI/FireCCI/5_1')
            date_filter = earthengine.Filter.date(start_date, end_date)

            ds_fire_cci = ds_fire_cci.filter(date_filter)
            if confidence_level > 0:
                ds_fire_cci = ds_fire_cci.map(filter_confidence(confidence_level))
            ds_fire_cci = ds_fire_cci.select('ConfidenceLevel')
            burn_area = ds_fire_cci.max()

            return burn_area

    class FireMTBS:

        @staticmethod
        def getBurnArea(severity_from, severity_to, start_date='2001-01-01', end_date='2021-01-01'):

            def filter_severity(value_from: float, value_to: float):
                def fn_filter(image):
                    #
                    mask = image.select('Severity').gte(value_from)
                    mask = mask.lte(value_to)
                    return image.updateMask(mask)

                return fn_filter

            ds_fire_mtbs = earthengine.ImageCollection('USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1')
            date_filer = earthengine.Filter.date(start_date, end_date)

            ds_fire_mtbs = ds_fire_mtbs.filter(date_filer)
            if severity_from > 0:
                ds_fire_mtbs = ds_fire_mtbs.map(filter_severity(severity_from, severity_to))
            ds_fire_mtbs = ds_fire_mtbs.select('Severity')
            burn_area = ds_fire_mtbs.max()

            return burn_area
