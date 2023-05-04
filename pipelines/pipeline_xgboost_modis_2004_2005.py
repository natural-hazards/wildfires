import os

from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from mlfire.data.ts import DataAdapterTS, DatasetTransformOP, DatasetSplitOpt
from mlfire.earthengine.collections import FireLabelsCollection
from mlfire.earthengine.collections import MTBSSeverity, MTBSRegion
from mlfire.features.pca import FactorOP

from mlfire.utils.io import saveDatasetToHDF5
from mlfire.utils.time import elapsed_timer

# utils imports
from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')


if __name__ == '__main__':

    DATA_DIR = 'data/tifs'
    OUTPUT_H5_DIR = 'data/h5/mtbs'
    DS_PREFIX = 'ak_modis_2005_100km_'

    DATA_DIR = 'data/tifs'
    PREFIX_IMG = 'ak_reflec_january_december_{}_100km'

    LABEL_COLLECTION = FireLabelsCollection.MTBS
    # LABEL_COLLECTION = FireLabelsCollection.CCI
    STR_LABEL_COLLECTION = LABEL_COLLECTION.name.lower()

    DS_SPLIT_OPT = DatasetSplitOpt.SHUFFLE_SPLIT
    # DS_SPLIT_OPT = DatasetSplitOpt.IMG_VERTICAL_SPLIT
    # DS_SPLIT_OPT = DatasetSplitOpt.IMG_HORIZONTAL_SPLIT
    TEST_RATIO = 1. / 3.  # split data set to training and test sets in ratio 2 : 1

    TRANSFORM_OPS = [DatasetTransformOP.STANDARTIZE_ZSCORE]
    PCA_OPS = [FactorOP.TEST_CUMSUM]

    lst_satimgs = []
    lst_labels = []

    CCI_CONFIDENCE_LEVEL = 70

    for year in range(2004, 2006):

        PREFIX_IMG_YEAR = PREFIX_IMG.format(year)

        fn_satimg = os.path.join(DATA_DIR, '{}_epsg3338_area_0.tif'.format(PREFIX_IMG_YEAR))
        lst_satimgs.append(fn_satimg)

        fn_labels = os.path.join(DATA_DIR, '{}_epsg3338_area_0_{}_labels.tif'.format(PREFIX_IMG_YEAR, STR_LABEL_COLLECTION))
        lst_labels.append(fn_labels)

    fn_satimg = os.path.join(DATA_DIR, 'ak_reflec_january_december_2005_100km_epsg3338_area_0.tif')
    fn_labels_cci = os.path.join(DATA_DIR, 'ak_reflec_january_december_2005_100km_epsg3338_area_0_cci_labels.tif')
    fn_labels_mtbs = os.path.join(DATA_DIR, 'ak_reflec_january_december_2005_100km_epsg3338_area_0_mtbs_labels.tif')

    """
    Setup adapter (time series)
    """

    adapter_ts = DataAdapterTS(
        lst_satimgs=lst_satimgs,
        lst_labels=lst_labels,
        label_collection=LABEL_COLLECTION,
        mtbs_severity_from=MTBSSeverity.LOW,
        cci_confidence_level=CCI_CONFIDENCE_LEVEL,
        # transformation options
        transform_ops=TRANSFORM_OPS,
        pca_ops=PCA_OPS,
        # data set split options
        ds_split_opt=DS_SPLIT_OPT,
        test_ratio=TEST_RATIO
    )

    index_begin_date = 0
    index_end_date = -1

    print('Data set start date {}'.format(adapter_ts.satimg_dates.iloc[index_begin_date]['Date']))
    adapter_ts.ds_start_date = adapter_ts.satimg_dates.iloc[index_begin_date]['Date']

    print('Data set end date {}'.format(adapter_ts.satimg_dates.iloc[index_end_date]['Date']))
    adapter_ts.ds_end_date = adapter_ts.satimg_dates.iloc[index_end_date]['Date']

    """
    Create training, test and validation data sets
    """

    adapter_ts.createDataset()

    ds_train = adapter_ts.getTrainingDataset()
    ds_test = adapter_ts.getTestDataset()

    ts_train = ds_train[0]; label_train = ds_train[1]
    ts_test = ds_test[0]; label_test = ds_test[1]

    """
    Reshape
    """

    if len(ts_train.shape) == 3:

        tmp_shape_train = ds_train[0].shape
        tmp_ts_train = ds_train[0].T.reshape(tmp_shape_train[0], -1).T

        tmp_shape_test = ds_test[0].shape
        tmp_ts_test = ds_test[0].T.reshape(tmp_shape_test[0], -1).T

    else:

        tmp_ts_train = ds_train[0]
        tmp_ts_test = ds_test[0]

    label_train = label_train.reshape(-1)
    label_test = label_test.reshape(-1)

    """
    Remove NaN values    
    """

    ts_train = tmp_ts_train[~_np.isnan(label_train)]
    label_train = label_train[~_np.isnan(label_train)]

    ts_test = tmp_ts_test[~_np.isnan(label_test)]
    label_test = label_test[~_np.isnan(label_test)]

    """
    Train classifier
    """

    with elapsed_timer('Training model using XGBoost'):

        xgb = XGBClassifier(objective='binary:logistic')
        xgb.fit(ts_train, label_train)

    """
    Inference using trained model
    """

    labels_pred = xgb.predict(ts_test)

    print('\n Report\n')
    print(classification_report(label_test, labels_pred))
    print('\n Report (imbalanced)\n')
    print(classification_report_imbalanced(label_test, labels_pred))

    # TODO visualization

    """
    Saving data to HDF5
    """

    with elapsed_timer('Save training data set'):
        fn_training = os.path.join(OUTPUT_H5_DIR, '{}training.h5'.format(DS_PREFIX))
        saveDatasetToHDF5((ts_train, label_train), fn_training)

    with elapsed_timer('Save test data set'):
        fn_test = os.path.join(OUTPUT_H5_DIR, '{}test.h5'.format(DS_PREFIX))
        saveDatasetToHDF5((ts_test, label_test), fn_test)
