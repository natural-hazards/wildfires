import ee as earthengine


class EarthEngineFireDatasets(object):

    def __init__(self, ds_name):

        self._ds_name = ds_name

    # ESA fire cii dataset
    class FireCII:

        @staticmethod
        def getBurnArea(confidence_level, start_date='2001-01-01', end_date='2020-01-01'):

            def filter_confidence(value: float):
                def fn_filter(image):
                    mask = image.select('ConfidenceLevel').gt(value)
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
