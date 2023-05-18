from mlfire.utils.functool import lazy_import

# lazy imports
_os = lazy_import('os')
_np = lazy_import('numpy')

# import CLI argument parser
_args = lazy_import('pipelines.args')

_data_ts = lazy_import('mlfire.data.ts')
_ee_collection = lazy_import('mlfire.earthengine.collections')
_features_pca = lazy_import('mlfire.features.pca')

_xgboost = lazy_import('xgboost')
_xgboost_inference = lazy_import('mlfire.models.xgboost.inference')
_xgboost_train = lazy_import('mlfire.models.xgboost.train')


if __name__ == '__main__':

    kwargs = _args.cli_argument_parser()

    DATA_DIR = kwargs['img_dir']
    DS_PREFIX = 'ak_modis_2005_100km_'
    PREFIX_IMG = 'ak_reflec_january_december_{}_100km'

    LABEL_COLLECTION = _ee_collection.FireLabelsCollection.MTBS
    # LABEL_COLLECTION = _ee_collection.FireLabelsCollection.CCI
    STR_LABEL_COLLECTION = LABEL_COLLECTION.name.lower()

    # DS_SPLIT_OPT = _data_ts.DatasetSplitOpt.SHUFFLE_SPLIT
    # DS_SPLIT_OPT = _data_ts.DatasetSplitOpt.IMG_VERTICAL_SPLIT
    DS_SPLIT_OPT = _data_ts.DatasetSplitOpt.IMG_HORIZONTAL_SPLIT
    TEST_RATIO = 1. / 3.  # split data set to training and test sets in ratio 2 : 1

    DatasetTransformOP = _data_ts.DatasetTransformOP
    FactorOP = _data_ts.FactorOP

    TRANSFORM_OPS = [DatasetTransformOP.STANDARTIZE_ZSCORE, DatasetTransformOP.PCA]
    PCA_OPS = [FactorOP.CUMULATIVE_EXPLAINED_VARIANCE]
    PCA_RETAINED_VARIANCE = .99

    lst_satimgs = []
    lst_labels = []

    CCI_CONFIDENCE_LEVEL = 70

    for year in range(2004, 2006):

        PREFIX_IMG_YEAR = PREFIX_IMG.format(year)

        fn_satimg = _os.path.join(DATA_DIR, '{}_epsg3338_area_0.tif'.format(PREFIX_IMG_YEAR))
        lst_satimgs.append(fn_satimg)

        fn_labels = _os.path.join(DATA_DIR, '{}_epsg3338_area_0_{}_labels.tif'.format(PREFIX_IMG_YEAR, STR_LABEL_COLLECTION))
        lst_labels.append(fn_labels)

    fn_satimg = _os.path.join(DATA_DIR, 'ak_reflec_january_december_2005_100km_epsg3338_area_0.tif')
    fn_labels_cci = _os.path.join(DATA_DIR, 'ak_reflec_january_december_2005_100km_epsg3338_area_0_cci_labels.tif')
    fn_labels_mtbs = _os.path.join(DATA_DIR, 'ak_reflec_january_december_2005_100km_epsg3338_area_0_mtbs_labels.tif')

    """
    Setup adapter (time series)
    """

    DataAdapterTS = _data_ts.DataAdapterTS
    MTBSSeverity = _ee_collection.MTBSSeverity

    adapter_ts = DataAdapterTS(
        lst_satimgs=lst_satimgs,
        lst_labels=lst_labels,
        label_collection=LABEL_COLLECTION,
        mtbs_severity_from=MTBSSeverity.LOW,
        cci_confidence_level=CCI_CONFIDENCE_LEVEL,
        # transformation options
        transform_ops=TRANSFORM_OPS,
        pca_ops=PCA_OPS,
        pca_retained_variance=PCA_RETAINED_VARIANCE,
        # data set split options
        ds_split_opt=DS_SPLIT_OPT,
        test_ratio=TEST_RATIO,
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

    xgb = _xgboost.XGBClassifier(objective='binary:logistic')
    _xgboost_train.trainSegmentationModel(xgb=xgb, ds=ds_train)

    print('\n============================\nInference on a train data set\n')

    _xgboost_inference.predict(
        xgb=xgb,
        ds=ds_train,
        with_report=True,
    )

    print('\n============================\nInference on a test data set\n')

    _xgboost_inference.predict(
        xgb=xgb,
        ds=ds_test,
        with_report=True,
    )
