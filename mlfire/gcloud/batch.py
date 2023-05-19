import os
import re

import ee as earthengine
import geojson

from enum import Enum

from mlfire.earthengine.collections import ModisCollection, FireLabelsCollection
from mlfire.utils.utils_string import getRandomString


class FileFormat(Enum):

    GeoTIFF = 'GeoTIFF'


class CRS(Enum):

    WSG84 = 'EPSG:4326'
    ALASKA_ALBERS = 'EPSG:3338'


class ExportData(Enum):

    NONE = 0
    SATIMG = 1
    LABEL = 2
    ALL = 3

    def __and__(self, other):

        return ExportData(self.value & other.value)


class EarthEngineBatch(object):

    def __init__(self,
                 file_json: str,
                 startdate: earthengine.Date = None,
                 enddate: earthengine.Date = None,
                 modis_index: ModisCollection = ModisCollection.REFLECTANCE,
                 labels_collection: FireLabelsCollection = FireLabelsCollection.CCI,
                 export: ExportData = ExportData.ALL,
                 resolution_per_pixel: int = 500,
                 crs: CRS = CRS.WSG84,
                 output_prefix: str = None,
                 output_format: FileFormat = FileFormat.GeoTIFF,
                 task_description: str = None,
                 gdrive_folder: str = None):

        self._polygons = None

        self._collection_img_modis = None
        self._collection_img_labels = None

        # properties related to exporting labels and satellite images
        self._json_rois = None
        self.file_json = file_json

        self.modis_index = None
        self._modis_index = modis_index

        self._labels_collection = None
        self.labels_collection = labels_collection

        self._export_flg = None
        self.export_flag = export

        self._startdate = None
        self.startdate = startdate

        self._enddate = None
        self.enddate = enddate

        # task properties
        self._crs = None
        self.crs = crs

        self._output_prefix = None
        if output_prefix is not None: self.output_prefix = output_prefix

        self._output_format = None
        self.output_format = output_format

        self._task_description = None
        self.task_description = task_description

        self._scale = None
        self.resolution_per_pixel = resolution_per_pixel

        self._gdrive_folder = None
        if gdrive_folder is not None: self.gdrive_folder = gdrive_folder

    @staticmethod
    def authenticate() -> None:

        earthengine.Authenticate()

    @staticmethod
    def initialize() -> None:

        earthengine.Initialize()

    @staticmethod
    def task_list() -> None:

        print('Running or ready task list:')
        lst_task = earthengine.batch.Task.list()

        for task in lst_task:

            if task.state == task.State.READY or task.state == task.State.RUNNING:
                print(task.name, '\t', '({})'.format(task.status()['description']))

    @staticmethod
    def task_cancel_all() -> None:

        earthengine.batch.Task.cancel()

    # properties

    @property
    def export_flag(self) -> ExportData:

        return self._export_flg

    @export_flag.setter
    def export_flag(self, flg) -> None:

        self._export_flg = flg

    @property
    def file_json(self) -> str:

        return self._json_rois

    @file_json.setter
    def file_json(self, fn: str) -> None:

        if not os.path.exists(fn):
            raise IOError('File {} does not exists'.format(fn))

        self.__reset()
        self._json_rois = fn

    @property
    def modis_index(self) -> ModisCollection:

        return self._modis_index

    @modis_index.setter
    def modis_index(self, index: ModisCollection) -> None:

        self._modis_index = index

    @property
    def labels_collection(self) -> FireLabelsCollection:

        return self._labels_collection

    @labels_collection.setter
    def labels_collection(self, collection: FireLabelsCollection) -> None:

        self._labels_collection = collection

    @property
    def startdate(self) -> earthengine.Date:

        return self._startdate

    @startdate.setter
    def startdate(self, date: earthengine.Date) -> None:

        self._startdate = date

    @property
    def enddate(self) -> earthengine.Date:

        return self._enddate

    @enddate.setter
    def enddate(self, date) -> None:

        self._enddate = date

    @property
    def output_prefix(self) -> str:

        return self._output_prefix

    @output_prefix.setter
    def output_prefix(self, prefix: str) -> None:

        if not re.match('^[A-Za-z0-9_-]*$', prefix):
            raise ValueError('Output prefix must contain only numbers, letters, dashes and underscores!')

        self._output_prefix = prefix

    @property
    def task_description(self) -> str:

        return self._task_description

    @task_description.setter
    def task_description(self, desc: str) -> None:

        self._task_description = desc

    @property
    def gdrive_folder(self) -> str:

        return self._gdrive_folder

    @gdrive_folder.setter
    def gdrive_folder(self, folder: str) -> None:

        if not re.match('^[A-Za-z0-9_-]*$', folder):
            raise ValueError('Folder name must contain only numbers, letters, dashes and underscores!')

        self._gdrive_folder = folder

    @property
    def output_format(self) -> FileFormat:

        return self._output_format

    @output_format.setter
    def output_format(self, f: FileFormat) -> None:

        self._output_format = f

    @property
    def crs(self) -> CRS:

        return self._crs

    @crs.setter
    def crs(self, crs: CRS) -> None:

        self._crs = crs

    @property
    def resolution_per_pixel(self) -> int:

        return self._scale

    @resolution_per_pixel.setter
    def resolution_per_pixel(self, s: int) -> None:

        if s <= 0:
            raise ValueError('Scale must be positive!')

        self._scale = s

    # methods

    def __reset(self) -> None:

        del self._polygons; self._polygons = None
        del self._collection_img_modis; self._collection_img_modis = None

    def __loadCollection(self) -> None:

        if self._collection_img_modis is not None:
            return

        self._collection_img_modis = earthengine.ImageCollection(self._modis_index.value)

        if self._modis_index == ModisCollection.REFLECTANCE:
            self._collection_img_modis = self._collection_img_modis.select('sur_refl.+')
        elif self._modis_index == ModisCollection.LST:
            self._collection_img_modis = self._collection_img_modis.select('LST_Day.+')
        elif self._modis_index == ModisCollection.EVI:
            self._collection_img_modis = self._collection_img_modis.select('EVI')
        elif self._collection_img_modis == ModisCollection.NDVI:
            self._collection_img_modis = self._collection_img_modis.select('NDVI')

    def __loadLabels(self) -> None:

        if self._collection_img_labels is not None:
            return

        self._collection_img_labels = earthengine.ImageCollection(self.labels_collection.value)

        if self.labels_collection == FireLabelsCollection.CCI:
            self._collection_img_labels = self._collection_img_labels.select('ConfidenceLevel', 'ObservedFlag')
        else:
            self._collection_img_labels = self._collection_img_labels.select('Severity')

    def __loadGeoJSON(self) -> dict:

        with open(self._json_rois) as f:
            fire_rois = geojson.load(f)

        return fire_rois

    def __loadROI(self) -> None:

        self.__reset()

        self._polygons = []
        features = self.__loadGeoJSON()['features']

        for feature in features:

            geometry = feature['geometry']

            coordinates = geometry['coordinates']
            shape_type = geometry['type']

            if shape_type == 'Polygon' or shape_type == 'Rectangle':
                polygon = earthengine.Geometry.Polygon(coordinates)
                self._polygons.append(polygon)

    def submit(self):

        if self._polygons is None:
            self.__loadROI()

        if self._collection_img_modis is None:
            self.__loadCollection()

        if self._collection_img_labels is None:
            self.__loadLabels()

        collection_filtered = self._collection_img_modis.filterDate(self.startdate, self.enddate)
        img_bands = collection_filtered.toBands()  # get multi spectral image

        labels_filtered = self._collection_img_labels.filterDate(self.startdate, self.enddate)
        labels_bands = labels_filtered.toBands()
        if self.labels_collection == FireLabelsCollection.CCI:
            # avoid inconsistent types
            labels_bands = labels_bands.toInt16()

        _prefix = '' if self.output_prefix is None else '{}_'.format(self.output_prefix)
        _folder = getRandomString(10) if self.gdrive_folder is None else self.gdrive_folder
        _desc = '{},'.format(self.task_description) if self.task_description is not None else ''

        print('Submitted jobs:')
        for i, area in enumerate(self._polygons):

            task_modis = None
            task_labels = None

            if self._export_flg & ExportData.SATIMG == ExportData.SATIMG:

                # define a job for exporting satellite images (bands)
                task_modis = earthengine.batch.Export.image.toDrive(
                    image=img_bands,
                    description='{}area_{}'.format(_desc, i),
                    scale=self.resolution_per_pixel,
                    region=area,
                    folder=_folder,
                    fileNamePrefix='{}area_{}'.format(_prefix, i),
                    fileFormat=self.output_format.value,
                    crs=self.crs.value
                )

            if self._export_flg & ExportData.LABEL == ExportData.LABEL:

                # get name related to collection name
                collection_name = self.labels_collection.name.lower()

                # define job for exporting labels
                task_labels = earthengine.batch.Export.image.toDrive(
                    image=labels_bands,
                    description='{}area_{}_{}_labels'.format(_desc, i, collection_name),
                    scale=self.resolution_per_pixel,
                    region=area,
                    folder=_folder,
                    fileNamePrefix='{}area_{}_{}_labels'.format(_prefix, i, collection_name),
                    fileFormat=self.output_format.value,
                    crs=self.crs.value
                )

            if self._export_flg & ExportData.SATIMG == ExportData.SATIMG:

                task_modis.start()
                print(task_modis.status)

            if self._export_flg & ExportData.LABEL == ExportData.LABEL:

                task_labels.start()
                print(task_labels.status)


# tests
if __name__ == '__main__':

    # initialize earth engine
    EarthEngineBatch.initialize()

    fn_json = 'data/jsons/ak_area_100km.geojson'

    for y in range(2005, 2006):

        start_date = earthengine.Date('{}-01-01'.format(y))
        end_date = earthengine.Date('{}-01-01'.format(y + 1))

        earthengine_batch = EarthEngineBatch(
            file_json=fn_json,
            startdate=start_date,
            enddate=end_date
        )
        earthengine_batch.export_flag = ExportData.LABEL

        earthengine_batch.crs = CRS.ALASKA_ALBERS
        earthengine_batch.resolution_per_pixel = 500  # pixel corresponds to resolution 500x500 meters

        earthengine_batch.task_description = 'MODIS-REFLECTANCE-AK-{}-JANUARY-DECEMBER-EPSG3338'.format(y)
        earthengine_batch.output_prefix = 'ak_reflec_january_december_{}_100km_epsg3338'.format(y)

        earthengine_batch.labels_collection = FireLabelsCollection.CCI
        earthengine_batch.modis_index = ModisCollection.REFLECTANCE

        earthengine_batch.gdrive_folder = 'AK_{}'.format(y)
        earthengine_batch.submit()

        # earthengine_batch.export_flag = ExportData.LABEL
        # earthengine_batch.labels_collection = FireLabelsCollection.CCI
        # earthengine_batch.submit()

        # print task list (running or ready)
        earthengine_batch.task_list()
