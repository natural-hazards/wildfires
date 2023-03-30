import os

from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from mlfire.data.ts import DataAdapterTS, DatasetTransformOP
from mlfire.earthengine.collections import FireLabelsCollection, MTBSRegion

from mlfire.utils.io import saveDatasetToHDF5
from mlfire.utils.time import elapsed_timer

if __name__ == '__main__':

    DATA_DIR = 'data/tifs'
    OUTPUT_H5_DIR = 'data/h5/mtbs'
    DS_PREFIX = 'ak_modis_2005_100km_'

    fn_satimg = os.path.join(DATA_DIR, 'ak_reflec_january_december_2005_100km_epsg3338_area_0.tif')
    fn_labels_cci = os.path.join(DATA_DIR, 'ak_reflec_january_december_2005_100km_epsg3338_area_0_cci_labels.tif')
    fn_labels_mtbs = os.path.join(DATA_DIR, 'ak_reflec_january_december_2005_100km_epsg3338_area_0_mtbs_labels.tif')

    """
    Setup adapter (time series)
    """
    adapter = DataAdapterTS(
        src_satimg=fn_satimg,
        src_labels=fn_labels_mtbs,
        transform_ops=[DatasetTransformOP.STANDARTIZE_ZSCORE],
        mtbs_region=MTBSRegion.ALASKA,
        label_collection=FireLabelsCollection.MTBS,
        cci_confidence_level=70
    )

    index_begin_date = 0
    index_end_date = -1

    print('Data set start date {}'.format(adapter.satimg_dates.iloc[index_begin_date]['Date']))
    adapter.ds_start_date = adapter.satimg_dates.iloc[index_begin_date]['Date']

    print('Data set end date {}'.format(adapter.satimg_dates.iloc[index_end_date]['Date']))
    adapter.ds_end_date = adapter.satimg_dates.iloc[index_end_date]['Date']

    """
    Get training and test data set
    """
    with elapsed_timer('Get training/test data sets'):
        ds_training = adapter.ds_training
        ds_test = adapter.ds_test

    """
    Saving data to HDF5
    """
    with elapsed_timer('Save training data set'):
        fn_training = os.path.join(OUTPUT_H5_DIR, '{}training.h5'.format(DS_PREFIX))
        saveDatasetToHDF5(ds_training, fn_training)

    with elapsed_timer('Save test data set'):
        fn_test = os.path.join(OUTPUT_H5_DIR, '{}test.h5'.format(DS_PREFIX))
        saveDatasetToHDF5(ds_test, fn_test)

    """
    Training and test model using XGBoost
    """
    with elapsed_timer('Training model using XGBoost'):
        xgb = XGBClassifier(objective='binary:logistic')
        xgb.fit(ds_training[0], ds_training[1])

    """
    Inference using trained model
    """
    labels_pred = xgb.predict(ds_test[0])

    print('\n Report\n')
    print(classification_report(ds_test[1], labels_pred))
    print('\n Report (imbalanced)\n')
    print(classification_report_imbalanced(ds_test[1], labels_pred))

