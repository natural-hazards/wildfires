
import os
import re

import ee as earthengine
import geojson

from enum import Enum

from utils.utils_string import getRandomString


class ModisIndex(Enum):

    LST = 'MODIS/006/MOD11A2'
    REFLECTANCE = 'MODIS/006/MOD09A1'
    EVI = 'MODIS/006/MOD13A1'
    NDVI = 'MODIS/006/MOD13A1'


class FireLabelsCollection(Enum):

    ESA_FIRE_CCI = 'ESA/CCI/FireCCI/5_1'
    MTBS = 'USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1'


class FileFormat(Enum):

    GeoTIFF = 'GeoTIFF'
    TFRecord = 'TFRecord'


class CRS(Enum):

    WSG84 = 'EPSG:4326'
    ALASKA_ALBERS = 'EPSG:3338'


class EarthEngineBatch(object):  # TODO set params in constructor

    def __init__(self):

        self._crs = CRS.WSG84

        self._json_rois = None
        self._polygons = None

        self._collection = None
        self._type_index = ModisIndex.REFLECTANCE

        self._labels = None
        self._labels_collection = FireLabelsCollection.ESA_FIRE_CCI

        self._start_date = None
        self._end_date = None

        # task
        self._output_prefix = None
        self._output_format = FileFormat.GeoTIFF

        self._task_description = None
        self._scale = 500

        self._gdrive_folder = None

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
    def type_index(self) -> ModisIndex:

        return self._type_index

    @type_index.setter
    def type_index(self, index: ModisIndex) -> None:

        self._type_index = index

    @property
    def labels_collection(self) -> FireLabelsCollection:

        return self._labels_collection

    @labels_collection.setter
    def labels_collection(self, collection: FireLabelsCollection) -> None:

        self._labels_collection = collection

    @property
    def start_date(self) -> earthengine.Date:

        return self._start_date

    @start_date.setter
    def start_date(self, date: earthengine.Date) -> None:

        self._start_date = date

    @property
    def end_date(self) -> earthengine.Date:

        return self._end_date

    @end_date.setter
    def end_date(self, date) -> None:

        self._end_date = date

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
    def scale(self) -> int:

        return self._scale

    @scale.setter
    def scale(self, s: int) -> None:

        if s <= 0:
            raise ValueError('Scale must be positive!')

        self._scale = s

    def __reset(self) -> None:

        del self._polygons; self._polygons = None
        del self._collection; self._collection = None

    def __loadCollection(self) -> None:

        if self._collection is not None:
            return

        self._collection = earthengine.ImageCollection(self._type_index.value)

        if self._type_index == ModisIndex.REFLECTANCE:
            self._collection = self._collection.select('sur_refl.+')
        elif self._type_index == ModisIndex.LST:
            self._collection = self._collection.select('LST_Day.+')
        elif self._type_index == ModisIndex.EVI:
            self._collection = self._collection.select('EVI')
        elif self._collection == ModisIndex.NDVI:
            self._collection = self._collection.select('NDVI')

    def __loadLabels(self) -> None:

        if self._labels is not None:
            return

        self._labels = earthengine.ImageCollection(self.labels_collection.value)

        if self.labels_collection == FireLabelsCollection.ESA_FIRE_CCI:
            self._labels = self._labels.select('ConfidenceLevel')
        else:
            self._labels = self._labels.select('Severity')

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

    def export(self):

        if self._polygons is None:
            self.__loadROI()

        if self._collection is None:
            self.__loadCollection()

        if self._labels is None:
            self.__loadLabels()

        collection_filtered = self._collection.filterDate(self.start_date, self.end_date)
        img_bands = collection_filtered.toBands()  # get multi spectral image

        labels_filtered = self._labels.filterDate(self.start_date, self.end_date)
        labels_bands = labels_filtered.toBands()

        _prefix = '' if self.output_prefix is None else '{}_'.format(self.output_prefix)
        _folder = getRandomString(10) if self.gdrive_folder is None else self.gdrive_folder
        _desc = '{},'.format(self.task_description) if self.task_description is not None else ''

        print('Submitted jobs:')
        for i, area in enumerate(self._polygons):

            task_modis = earthengine.batch.Export.image.toDrive(
                image=img_bands,
                description='{}area_{}'.format(_desc, i),
                scale=self.scale,
                region=area,
                folder=_folder,
                fileNamePrefix='{}area_{}'.format(_prefix, i),
                fileFormat=self.output_format.value,
                crs=self.crs.value
            )

            task_labels = earthengine.batch.Export.image.toDrive(
                image=labels_bands,
                description='{}area_{}_labels'.format(_desc, i),
                scale=self.scale,
                region=area,
                folder=_folder,
                fileNamePrefix='{}area_{}_labels'.format(_prefix, i),
                fileFormat=self.output_format.value,
                crs=self.crs.value
            )

            task_modis.start()
            print(task_modis.status)

            task_labels.start()
            print(task_labels.status)


# tests
if __name__ == '__main__':

    # initialize earth engine
    EarthEngineBatch.initialize()

    fn_json = 'tutorials/data.geojson'
    start_date = earthengine.Date('2004-01-01')
    end_date = earthengine.Date('2005-02-01')

    exporter = EarthEngineBatch()

    exporter.labels_collection = FireLabelsCollection.MTBS
    exporter.crs = CRS.ALASKA_ALBERS
    exporter.scale = 500  # pixel corresponds to resolution 500x500 meters

    exporter.task_description = 'MODIS-REFLECTANCE-AK-2004-EPSG3338'
    exporter.output_prefix = 'ak_2004_500px2_epsg3338'

    exporter.start_date = start_date
    exporter.end_date = end_date

    exporter.file_json = fn_json
    exporter.type_index = ModisIndex.REFLECTANCE

    exporter.gdrive_folder = 'AK_2004'
    exporter.export()

    # print task list (running or ready)
    exporter.task_list()
