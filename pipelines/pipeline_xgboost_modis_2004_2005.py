from mlfire.utils.functool import lazy_import

# lazy imports
_os = lazy_import('os')
_np = lazy_import('numpy')

# import CLI argument parser
_args = lazy_import('pipelines.args')

_data_loader = lazy_import('mlfire.data.loader')
_data_ts = lazy_import('mlfire.data.ts')
_data_fuze = lazy_import('mlfire.data.fuze')

_ee_collection = lazy_import('mlfire.earthengine.collections')
_features_pca = lazy_import('mlfire.features.pca')

_xgboost = lazy_import('xgboost')
_xgboost_inference = lazy_import('mlfire.models.xgboost.inference')
_xgboost_train = lazy_import('mlfire.models.xgboost.train')


if __name__ == '__main__':

    kwargs = _args.cli_argument_parser()
    DATA_DIR = kwargs['img_dir']

    PREFIX_REFLECTANCE_IMG = 'ak_reflec_january_december_{}_100km'
    PREFIX_FIREMAP_IMG = 'ak_january_december_{}_100km'

    FIREMAP_COLLECTION = _data_loader.FireMapSelectOpt.MTBS
    # LABEL_COLLECTION = _ee_collection.FireLabelCollection.CCI
    STR_LABEL_COLLECTION = FIREMAP_COLLECTION.name.lower()

    # DS_SPLIT_OPT = _data_ts.SatDataSplitOpt.SHUFFLE_SPLIT
    # DS_SPLIT_OPT = _data_ts.SatDataSplitOpt.IMG_VERTICAL_SPLIT
    DS_SPLIT_OPT = _data_ts.SatDataSplitOpt.IMG_HORIZONTAL_SPLIT
    TEST_RATIO = 1. / 3.  # split data set to training and test sets in ratio 2 : 1

    SatDataPreprocessOpt = _data_ts.SatDataPreprocessOpt
    FactorOP = _data_ts.FactorOP

    TRANSFORM_OPS = (
        SatDataPreprocessOpt.STANDARTIZE_ZSCORE,
        SatDataPreprocessOpt.PCA_PER_BAND,
        SatDataPreprocessOpt.NOT_PROCESS_UNCHARTED_PIXELS
    )
    PCA_OPS = (FactorOP.CUMULATIVE_EXPLAINED_VARIANCE,)
    PCA_RETAINED_VARIANCE = .90

    VegetationIndexSelectOpt = _data_fuze.VegetationIndexSelectOpt
    ADD_VI = (VegetationIndexSelectOpt.EVI,)

    lst_satdata = []
    lst_firemaps = []

    CCI_CONFIDENCE_LEVEL = 70

    for year in range(2004, 2006):

        PREFIX_REFLECTANCE_IMG_YEAR = PREFIX_REFLECTANCE_IMG.format(year)
        fn_satdata = '{}_epsg3338_area_0.tif'.format(PREFIX_REFLECTANCE_IMG_YEAR)
        fn_satdata = _os.path.join(DATA_DIR, fn_satdata)
        lst_satdata.append(fn_satdata)

        PREFIX_FIREMAP_IMG_YEAR = PREFIX_FIREMAP_IMG.format(year)
        fn_firemaps = f'{PREFIX_FIREMAP_IMG_YEAR}_epsg3338_area_0_{STR_LABEL_COLLECTION}_labels.tif'
        fn_firemaps = _os.path.join(DATA_DIR, fn_firemaps)
        lst_firemaps.append(fn_firemaps)

    """
    Setup satelitte data adapter 
    """

    SatDataSelectOpt = _data_loader.SatDataSelectOpt
    MTBSSeverity = _ee_collection.MTBSSeverity

    adapter_ts = _data_ts.SatDataAdapterTS(
        lst_firemaps=lst_firemaps,
        lst_satdata_reflectance=lst_satdata,
        # selection of modis collection
        opt_select_satdata=SatDataSelectOpt.REFLECTANCE,
        # fire maps
        opt_select_firemap=FIREMAP_COLLECTION,
        mtbs_min_severity=MTBSSeverity.LOW,
        # vegetation index
        lst_vegetation_add=ADD_VI,
        # transformation options
        opt_preprocess_satdata=TRANSFORM_OPS,
        opt_pca_factor=PCA_OPS,
        pca_retained_variance=PCA_RETAINED_VARIANCE,
        # data set split options
        opt_split_satdata=DS_SPLIT_OPT,
        test_ratio=TEST_RATIO,
    )

    id_start_date = 0
    start_timestamp = adapter_ts.timestamps_satdata.iloc[id_start_date]['Timestamps']
    print('Data set start date {}'.format(start_timestamp))

    id_end_date = -1
    end_timestamp = adapter_ts.timestamps_satdata.iloc[id_end_date]['Timestamps']
    print('Data set end date {}'.format(end_timestamp))

    """
    Create training, test and validation data sets
    """

    adapter_ts.selected_timestamps = (start_timestamp, end_timestamp)
    adapter_ts.createDatasets()

    ds_train = adapter_ts.getTrainingDataset()
    ds_test = adapter_ts.getTestDataset()

    xgb = _xgboost.XGBClassifier(objective='binary:logistic')
    _xgboost_train.trainSegmentationModel(xgb=xgb, ds=ds_train)

    print('\n============================\nInference on a train data set\n')

    _xgboost_inference.predict(
        xgb=xgb,
        ds=ds_train,
        show_report=True,
        report_with_scores=True
    )

    print('\n============================\nInference on a test data set\n')

    _xgboost_inference.predict(
        xgb=xgb,
        ds=ds_test,
        show_report=True,
        report_with_scores=True
    )
