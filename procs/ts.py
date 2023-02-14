# https://www.youtube.com/watch?v=M5NygAGT5AI (semantic segmentation)
import os.path

import numpy as np
import pandas as pd

from enum import Enum

from osgeo import gdal

from earthengine.ds import MTBSRegion
from utils.utils_string import band2date_firecci, band2date_mtbs
from utils.time import elapsed_timer


class FillMissingValues(Enum):

    TS_SAVITZKY_GOLAY = 1
    IMFILL = 2


class DataAdapterTS(object):

    def __init__(self,
                 path_geotiff: str = None,
                 path_labels: str = None,
                 mtbs_region: MTBSRegion = None):

        self._ds_sources = None
        self._ds_labels = None

        self._path_img = None
        if path_geotiff is not None:
            self.path_img = path_geotiff

        self._label_path = None
        if path_labels is not None:
            self.path_labels = path_labels

        self._mtbs_region = None
        if mtbs_region is not None:
            self.mtbs_region = mtbs_region

        # dictionary contains time series
        self._dict_ts = None
        self._df_labels = None

        # dates pandas data frames
        self._df_dates_sources = None
        self._df_dates_labels = None

    def __len__(self) -> int:

        return len(self._dict_ts) if len(self._dict_ts) is not None else 0

    @property
    def path_img(self) -> str:

        return self._path_img

    @path_img.setter
    def path_img(self, fn: str) -> None:

        if not os.path.exists(fn):
            raise IOError('File {} does not exist!'.format(fn))

        path_source = os.path.abspath(fn)

        if self._path_img == path_source:
            return

        del self._ds_sources; self._ds_sources = None
        self._path_img = path_source

    @property
    def path_labels(self) -> str:

        return self._label_path

    @path_labels.setter
    def path_labels(self, fn: str) -> None:

        if not os.path.exists(fn):
            raise IOError('File {} does not exist!'.format(fn))

        path_source = os.path.abspath(fn)

        if self._label_path == path_source:
            return

        del self._ds_labels; self._ds_labels = None
        self._label_path = path_source

    @property
    def mtbs_region(self) -> MTBSRegion:

        return self._mtbs_region

    @mtbs_region.setter
    def mtbs_region(self, region: MTBSRegion) -> None:

        if self._mtbs_region == region:
            return

        self.__reset()
        self._mtbs_region = region

    @property
    def nrasters(self) -> int:

        if self._ds_sources is None:
            try:
                self.__loadGeoTiff_SOURCES()
            except IOError:
                print('Can not load geotiff {}'.format(self.path_img))
                return 0

        return self._ds_sources.RasterCount

    @property
    def raster_npixels(self) -> int:

        if self._ds_sources is None:
            try:
                self.__loadGeoTiff_SOURCES()
            except IOError:
                print('Can not load geotiff {}'.format(self.path_img))

        w, h = self._ds_sources.RasterXSize, self._ds_sources.RasterYSize
        return w * h

    @property
    def raster_shape(self) -> tuple:

        if self._ds_sources is None:
            try:
                self.__loadGeoTiff_SOURCES()
            except IOError:
                print('Can not load geotiff {}'.format(self.path_img))

        shape = (self._ds_sources.RasterXSize, self._ds_sources.RasterYSize)
        return shape

    @property
    def nlabels(self) -> int:

        if self._ds_labels is None:
            try:
                self.__loadGeoTIff_LABELS()
            except IOError:
                print('Can not load geotiff {}'.format(self.path_labels))

        return self._ds_labels.RasterCount

    def __reset(self) -> None:

        del self._ds_sources; self._ds_sources = None
        del self._dict_ts; self._dict_ts = None
        del self._df_dates_sources; self._df_dates_sources = None

    def __loadGeoTiff_SOURCES(self) -> None:

        if self._ds_sources is not None:
            self.__reset()

        if self.path_img is None:
            raise IOError('GeoTIFF file (source) is not set!')

        self._ds_sources = gdal.Open(self.path_img)

    def __loadGeoTIff_LABELS(self) -> None:

        if self._ds_labels is not None:
            self.__reset()

        if self.path_labels is None:
            raise IOError('GeoTIFF file (labels) is not set!')

        self._ds_labels = gdal.Open(self.path_labels)

    def __determineListDates_SOURCES(self) -> None:

        if self._ds_sources is None:
            try:
                self.__loadGeoTiff_SOURCES()
            except IOError:
                print('Can not load geotiff {}'.format(self.path_img))
                return

        lst = []
        for id_raster in range(self.nrasters):
            rs_band = self._ds_sources.GetRasterBand(id_raster + 1)
            dsc_band = rs_band.GetDescription()

            band_date = band2date_firecci(dsc_band)
            lst.append(band_date)

        lst = sorted(list(set(lst)))

        raster_from = np.linspace(start=0, stop=(len(lst) - 1) * 7, num=len(lst)).astype(int)
        raster_to = np.linspace(start=7, stop=(len(lst)) * 7, num=len(lst)).astype(int)

        df_dates = pd.DataFrame(lst)
        df_dates['begin'] = raster_from
        df_dates['end'] = raster_to

        self._df_dates_sources = df_dates

    def __determineListDates_LABELS(self) -> None:

        if self._ds_labels is None:
            try:
                self.__loadGeoTIff_LABELS()
            except IOError:
                print('Can not load geotiff {}'.format(self.path_labels))
                return

        lst = []
        for id_raster in range(self.nlabels):
            rs_band = self._ds_labels.GetRasterBand(id_raster + 1)
            dsc_band = rs_band.GetDescription()
            print(dsc_band)

            if 'Severity' in dsc_band:
                band_date = band2date_mtbs(dsc_band)
            else:
                band_date = band2date_firecci(dsc_band)
            lst.append(band_date)

        df_dates = pd.DataFrame(lst)
        self._df_dates_labels = df_dates

    @property
    def list_dates_imgs(self):

        if self._df_dates_sources is not None:
            return self._df_dates_sources

        self.__determineListDates_SOURCES()
        return self._df_dates_sources

    @property
    def list_dates_labels(self):

        if self._df_dates_labels is not None:
            return self._df_dates_labels

        self.__determineListDates_LABELS()
        return self._df_dates_labels

    def get_imraster(self, id_raster) -> np.ndarray:

        if self._ds_sources is None:
            self.__loadGeoTiff_SOURCES()

        # [red, near infra-red (nir), blue, green, short-wave infra-red (swir) 1, swir2, swir3]
        rs_band = self._ds_sources.GetRasterBand(id_raster + 1).ReadAsArray()
        return rs_band

    def get_ts(self, id_pixel) -> pd.DataFrame:

        if self._dict_ts is None:
            self.__create_ts_dict()

        df = pd.DataFrame.from_dict(self.__get_dict_ts(id_pixel))
        return df

    def get_labels(self, id_time) -> pd.DataFrame:

        if self._df_labels is None:
            self.__create_labels_dataframe()

        return self._df_labels[id_time]

    def __get_dict_ts(self, index) -> dict:

        if self._dict_ts is None:
            self.__create_ts_dict()

        return self._dict_ts[index]

    def __create_ts_dict(self) -> None:

        if self._ds_sources is None:
            try:
                self.__loadGeoTiff_SOURCES()
            except IOError:
                print('Can not load geotiff (sources) {}'.format(self.path_img))

        # sources

        npixels = self.raster_npixels
        dict_ts = {px_id: {'date': []} for px_id in range(npixels)}

        for i in range(self.nrasters):

            rs_band = self._ds_sources.GetRasterBand(i + 1)
            dsc_band = rs_band.GetDescription()

            if 'Severity' in dsc_band:
                band_date = band2date_firecci(dsc_band)
            else:
                band_date = band2date_mtbs(dsc_band)

            band_date = pd.Timestamp(band_date, unit='D')
            band_name = dsc_band.split('_')[-1]

            np_band = rs_band.ReadAsArray().flatten()

            for px_id, val in enumerate(np_band):

                if band_date not in dict_ts[px_id]['date']:
                    dict_ts[px_id]['date'].append(band_date)

                if band_name not in dict_ts[px_id]:
                    dict_ts[px_id][band_name] = []

                dict_ts[px_id][band_name].append(val)

        del self._dict_ts; self._dict_ts = dict_ts

    def __create_labels_dataframe(self) -> None:

        if self._ds_labels is None:
            try:
                self.__determineListDates_LABELS()
            except IOError:
                print('Ce not load geotiff (labels) {}'.format(self.path_labels))

        dict_labels = {}
        for i in range(self.nlabels):
            rs_band = self._ds_labels.GetRasterBand(i + 1)
            dsc_band = rs_band.GetDescription()

            if 'Severity' in dsc_band:
                band_date = band2date_mtbs(dsc_band)
            else:
                band_date = band2date_firecci(dsc_band)
            band_date = pd.Timestamp(band_date, unit='D')

            np_band = rs_band.ReadAsArray().flatten()
            dict_labels[band_date] = np_band

        del self._df_labels; self._df_labels = pd.DataFrame.from_dict(dict_labels)

    def load(self) -> None:

        if self._dict_ts is None:
            self.__create_ts_dict()

        if self._df_labels is None:
            self.__create_labels_dataframe()


# tests
if __name__ == '__main__':

    pass

    from PIL import Image, ImageEnhance

    # im_path = 'tutorials/ak_2004_500px2_epsg3338_area_0.tif'
    # label_path = 'tutorials/ak_1984_2021_500px2_epsg3338_area_0_labels.tif'
    #
    # ts = DataAdapterTS(path_geotiff=im_path, path_labels=label_path)
    # with elapsed_timer('Get date list (imgs)'):
    #     print(ts.list_dates_imgs)
    #
    # with elapsed_timer('Get date list (labels)'):
    #     print(ts.list_dates_labels)
    #
    # with elapsed_timer('Get labels pandas'):
    #     print(ts.get_labels(pd.Timestamp('2004-01')))
    #
    # print(ts.nlabels)

    # ts.__loadLabels()
    # with elapsed_timer('Get pandas serie px=0'):
    #     ts_px0 = ts.get_pandas(0)
    #
    # x = ts_px0['date']
    # y = ts_px0['b01']
    #
    # plt.plot(x, y)
    # plt.grid()
    # plt.xticks(rotation=15)
    # plt.show()

    # ts = DataAdapterTS(path_geotiff=im_path)
    #
    # [red, near infra-red (nir), blue, green, short-wave infra-red (swir) 1, swir2, swir3]
    # red = ts.get_imraster(0) / 1e4
    # blue = ts.get_imraster(2) / 1e4
    # green = ts.get_imraster(3) / 1e4
    #
    # # modis_img = (red + green + blue) / 3
    # modis_img = np.dstack((red, green, blue))
    # pil_img = Image.fromarray(np.uint8(modis_img * 255))
    #
    # enhancer = ImageEnhance.Brightness(pil_img)
    # factor = 20.2
    # im_output = enhancer.enhance(factor)
    #
    # enhancer = ImageEnhance.Contrast(im_output)
    # im_output = enhancer.enhance(.6)
    #
    # enhancer = ImageEnhance.Color(im_output)
    # im_output = enhancer.enhance(.85)
    # modis_img = ((modis_img - modis_img.min()) / (modis_img.max() - modis_img.min()))

    # plt.imshow(im_output, cmap='gray')
    # plt.show()

    # with elapsed_timer('Load'):
    #     ts.load()
    #
