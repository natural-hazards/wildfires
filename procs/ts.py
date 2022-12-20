import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enum import Enum

from osgeo import gdal

from utils.utils_string import band2date
from utils.timer import elapsed_timer


class FillMissingValues(Enum):

    TS_SAVITZKY_GOLAY = 1
    

class TimeSeries(object):

    def __init__(self,
                 path_geotiff: str = None,
                 path_labels: str = None):

        self._impath = None
        if path_geotiff is not None:
            self.path_geotiff = path_geotiff

        self._label_path = None
        if path_labels is not None:
            self.path_labels = path_labels

        self._ds = None
        self._dict_ts = None

        self._df_dates = None

    def __len__(self) -> int:

        return len(self._dict_ts) if len(self._dict_ts) is not None else 0

    @property
    def path_geotiff(self) -> str:

        return self._impath

    @path_geotiff.setter
    def path_geotiff(self, fn: str) -> None:

        if not os.path.exists(fn):
            raise IOError('File {} does not exist!'.format(fn))

        self._impath = os.path.abspath(fn)

    @property
    def path_labels(self) -> str:

        return self._label_path

    @path_labels.setter
    def path_labels(self, fn: str) -> None:

        if not os.path.exists(fn):
            raise IOError('File {} does not exist!'.format(fn))

        path_source = os.path.abspath(fn)

        if self._label_path != path_source:
            self._label_path = path_source

    @property
    def nrasters(self) -> int:

        if self._ds is None:
            try:
                self.__loadGeoTiff()
            except IOError:
                print('Can not load geotiff {}'.format(self.path_geotiff))

        return self._ds.RasterCount

    @property
    def npixels(self) -> int:

        if self._ds is None:
            try:
                self.__loadGeoTiff()
            except IOError:
                print('Can not load geotiff {}'.format(self.path_geotiff))

        w, h = self._ds.RasterXSize, self._ds.RasterYSize
        return w * h

    def __reset(self) -> None:

        del self._ds; self._ds = None
        del self._dict_ts; self._dict_ts = None
        del self._df_dates; self._df_dates = None

    def __loadGeoTiff(self) -> None:

        if self._ds is not None:
            self.__reset()

        if self.path_geotiff is None:
            raise IOError('GeoTIFF file is not set!')

        self._ds = gdal.Open(self.path_geotiff)

    def __determineListDates(self) -> None:

        lst = []
        for id_raster in range(self.nrasters):
            rs_band = self._ds.GetRasterBand(id_raster + 1)
            dsc_band = rs_band.GetDescription()

            band_date = band2date(dsc_band)
            lst.append(band_date)

        lst = sorted(list(set(lst)))

        raster_from = np.linspace(start=0, stop=(len(lst) - 1) * 7, num=len(lst)).astype(int)
        raster_to = np.linspace(start=7, stop=(len(lst)) * 7, num=len(lst)).astype(int)

        df_dates = pd.DataFrame(lst)
        df_dates['begin'] = raster_from
        df_dates['end'] = raster_to

        self._df_dates = df_dates

    @property
    def list_dates(self):

        if self._df_dates is not None:
            return self._df_dates

        if self._ds is None:
            try:
                self.__loadGeoTiff()
            except IOError:
                print('Can not load geotiff {}'.format(self.path_geotiff))

        self.__determineListDates()

        return self._df_dates

    def get_imraster(self, id_raster) -> np.ndarray:

        if self._ds is None:
            self.__loadGeoTiff()

        # [red, near infra-red (nir), blue, green, short-wave infra-red (swir) 1, swir2, swir3]
        rs_band = self._ds.GetRasterBand(id_raster + 1).ReadAsArray()
        return rs_band

    def get_pandas(self, id_pixel) -> pd.DataFrame:

        if self._dict_ts is None:
            self.load()

        df = pd.DataFrame.from_dict(self.get_dict(id_pixel))
        return df

    def get_dict(self, index) -> dict:

        if self._ds is None:
            self.load()

        return self._dict_ts[index]

    def load(self) -> None:

        if self._ds is None:
            try:
                self.__loadGeoTiff()
            except IOError:
                print('Can not load geotiff {}'.format(self.path_geotiff))

        npixels = self.npixels
        dict_ts = {px_id: {'date': []} for px_id in range(npixels)}

        for i in range(self.nrasters):

            rs_band = self._ds.GetRasterBand(i + 1)
            dsc_band = rs_band.GetDescription()

            band_date = band2date(dsc_band)
            band_date = pd.Timestamp(band_date, unit='D')
            band_name = dsc_band.split('_')[-1]

            np_band = rs_band.ReadAsArray().flatten()

            for px_id, val in enumerate(np_band):

                if band_date not in dict_ts[px_id]['date']:
                    dict_ts[px_id]['date'].append(band_date)

                if band_name not in dict_ts[px_id]:
                    dict_ts[px_id][band_name] = []

                dict_ts[px_id][band_name].append(val)

        self._dict_ts = dict_ts


# tests
if __name__ == '__main__':

    from PIL import Image, ImageEnhance

    im_path = 'tutorials/ak_2004_500px2_epsg3338_area_0.tif'

    ts = TimeSeries(path_geotiff=im_path)
    with elapsed_timer('Get date list'):
        print(ts.list_dates)

    with elapsed_timer('Get pandas serie px=0'):
        ts_px0 = ts.get_pandas(0)

    x = ts_px0['date']
    y = ts_px0['b01']

    plt.plot(x, y)
    plt.grid()
    plt.xticks(rotation=15)
    plt.show()

    # ts = TimeSeries(path_geotiff=im_path)
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
