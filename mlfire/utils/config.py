import yaml

from itertools import chain, combinations, product
from typing import Union

from mlfire.data.loader import SatDataSelectOpt
from mlfire.data.fuze import VegetationIndexSelectOpt
from mlfire.data.ts import SatDataPreprocessOpt


class FeatureDict(dict):

    def __setitem__(self, k, v):
        if isinstance(k, (SatDataSelectOpt, VegetationIndexSelectOpt, SatDataPreprocessOpt)):
            k = k.name.upper()
            super().__setitem__(k, v)
        else:
            raise KeyError(f'Color {k} is not valid')

    def __getitem__(self, k):
        if isinstance(k, (SatDataSelectOpt, VegetationIndexSelectOpt, SatDataPreprocessOpt)):
            k = k.name.upper()
        else:
            raise KeyError(f'Color {k} is not valid')

        return super().__getitem__(k)


#
_config_encode_dict = FeatureDict({
    SatDataSelectOpt.ALL: 'all',
    SatDataSelectOpt.TEMPERATURE: 'temperature',
    SatDataSelectOpt.REFLECTANCE: 'reflectance',
    VegetationIndexSelectOpt.EVI: 'evi',
    VegetationIndexSelectOpt.EVI2: 'evi2',
    VegetationIndexSelectOpt.NDVI: 'ndvi',
    SatDataPreprocessOpt.STANDARTIZE_ZSCORE: 'zscore',
    SatDataPreprocessOpt.SAVITZKY_GOLAY: 'savitzky_golay',
    SatDataPreprocessOpt.PCA: 'pca',
    SatDataPreprocessOpt.PCA_PER_BAND: 'pca_per_band',
    SatDataPreprocessOpt.NOT_PROCESS_UNCHARTED_PIXELS: 'not_process_uncharted_pixels'
})

#
_config_decode_dict = {
    # satellite data
    'all': SatDataSelectOpt.ALL,
    'temperature': SatDataSelectOpt.TEMPERATURE,
    'reflectance': SatDataSelectOpt.REFLECTANCE,
    # vegetation
    'evi': VegetationIndexSelectOpt.EVI,
    'evi2': VegetationIndexSelectOpt.EVI2,
    'ndvi': VegetationIndexSelectOpt.NDVI,
    # transform ops
    'zscore': SatDataPreprocessOpt.STANDARTIZE_ZSCORE,
    'savitzky_golay': SatDataPreprocessOpt.SAVITZKY_GOLAY,
    'pca': SatDataPreprocessOpt.PCA,
    'pca_per_band': SatDataPreprocessOpt.PCA_PER_BAND,
    'not_process_uncharted_pixels': SatDataPreprocessOpt.NOT_PROCESS_UNCHARTED_PIXELS,
}


def decodeConfigFile(config_dict) -> dict:

    decoded_config = {}

    if 'satdata' not in config_dict:
        raise IOError

    lst_satdata = []
    for k in config_dict['satdata']: lst_satdata.append(_config_decode_dict[k])
    decoded_config['satdata'] = tuple(lst_satdata)

    if 'vegetation' in config_dict:
        lst_vegetation: list[VegetationIndexSelectOpt, ...] = []
        for k in config_dict['vegetation']: lst_vegetation.append(_config_decode_dict[k])
        decoded_config['vegetation'] = tuple(lst_vegetation)
    else:
        decoded_config['vegetation'] = (VegetationIndexSelectOpt.NONE,)

    if 'opt_preprocess_satdata' in config_dict:
        lst_ops: list[SatDataPreprocessOpt, ...] = []
        for k in config_dict['opt_preprocess_satdata']: lst_ops.append(_config_decode_dict[k])
        decoded_config['opt_preprocess_satdata'] = tuple(lst_ops)
    else:
        decoded_config['opt_preprocess_satdata'] = (SatDataPreprocessOpt.NONE,)

    if 'savitzky_golay' in config_dict:
        if 'poly_order' in config_dict['savitzky_golay']:
            decoded_config['savgol_polyorder'] = config_dict['savitzky_golay']['polyorder']
        else:
            decoded_config['savgol_polyorder'] = 1

        if 'savgol_winlen' in config_dict['savitzky_golay']:
            decoded_config['savgol_winlen'] = config_dict['savitzky_golay']['winlen']
        else:
            decoded_config['savgol_winlen'] = 5
    else:
        decoded_config['savgol_polyorder'] = 1
        decoded_config['savgol_winlen'] = 5

    decoded_config['pca_retained_variance'] = config_dict['pca_retained_variance']
    return decoded_config


def createConfigFile(fn: str, opt_select_satdata: tuple, lst_vegetation: Union[tuple, None] = None,
                     opt_preprocess_satdata: Union[tuple, None] = None, savitzky_golay: Union[dict, None] = None,
                     pca_retained_var: Union[float, None] = 0.8) -> None:

    config_dict = {}

    config_satdata = []
    for s in opt_select_satdata: config_satdata.append(_config_encode_dict[s])
    config_dict['satdata'] = config_satdata

    if lst_vegetation is not None:
        config_vegetation = []
        for v in lst_vegetation: config_vegetation.append(_config_encode_dict[v])
        config_dict['vegetation'] = config_vegetation

    if opt_preprocess_satdata is not None:
        config_ops = []
        for t in opt_preprocess_satdata: config_ops.append(_config_decode_dict[t])
        config_dict['opt_preprocess_satdata'] = config_ops

    if savitzky_golay is not None:
        config_dict['savitzky_golay'] = savitzky_golay

    if pca_retained_var is not None:
        config_dict['pca_retained_variance'] = pca_retained_var

    # create file
    yaml.dump(config_dict, fn)


def loadConfigFile(config_file: str) -> dict:

    with open(config_file, 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config_dict = decodeConfigFile(config_dict)
    return config_dict


def powerset(iterable):
    """
    powerset([EVI, EVI2, NDVI]) --> () (EVI,) (EVI2,) (NDVI,) (EVI,EVI2) (EVI,NDVI) (EVI2,NDVI) (EVI,EVI2,NDVI)
    """

    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


if __name__ == '__main__':

    VAR_CONFIG_FILE = '/Users/marek/Playground/wildfires/pipelines/test_yaml.yaml'
    print(loadConfigFile(VAR_CONFIG_FILE))

    VAR_CONFIG_FILE = '/Users/marek/Playground/wildfires/pipelines/test_yaml2.yaml'

    VAR_OPT_SELECT_SATDATA = (SatDataSelectOpt.TEMPERATURE, SatDataSelectOpt.REFLECTANCE, SatDataSelectOpt.ALL)
    VAR_OPT_SELECT_VEGETATION = (VegetationIndexSelectOpt.EVI, VegetationIndexSelectOpt.EVI2, VegetationIndexSelectOpt.NDVI)
    VAR_OPT_TRANSFORM_OPS = (
        SatDataPreprocessOpt.STANDARTIZE_ZSCORE,
        SatDataPreprocessOpt.PCA_PER_BAND,
        SatDataPreprocessOpt.SAVITZKY_GOLAY
    )

    VAR_PCA_RETAINED_VARIANCES = (.7, .8, .9, .95)
    VAR_SAVGOL_POLYORDER_RANGE = range(1, 5)
    VAR_SAVGOL_WINLEN_RANGE = range(5, 11)

    VAR_OPT_TRANSFORM_OPS = powerset(VAR_OPT_TRANSFORM_OPS)
    VAR_OPT_SELECT_VEGETATION = powerset(VAR_OPT_SELECT_VEGETATION)

    VAR_OPT_TRANSFORM_VEGETATION = []

    for i in range(len(VAR_OPT_TRANSFORM_OPS)):
        for j in range(len(VAR_OPT_SELECT_VEGETATION)):
            opt_transform = VAR_OPT_TRANSFORM_OPS[i] if VAR_OPT_TRANSFORM_OPS[i] else (SatDataPreprocessOpt.NONE,)
            if {SatDataPreprocessOpt.PCA_PER_BAND, SatDataPreprocessOpt.STANDARTIZE_ZSCORE}.issubset(opt_transform):
                continue
            opt_vegetation = VAR_OPT_SELECT_VEGETATION[j] if VAR_OPT_SELECT_VEGETATION[j] else (VegetationIndexSelectOpt.NONE,)
            VAR_OPT_TRANSFORM_VEGETATION.append((opt_vegetation, opt_transform))

    lst_settings = []

    for opt_satdata in VAR_OPT_SELECT_SATDATA:
        for opt_tranveg in VAR_OPT_TRANSFORM_VEGETATION:

            lst_satdata = (opt_satdata,)

            lst_opt_vegetation = opt_tranveg[0]
            lst_opt_transform = opt_tranveg[1]

            settings = {
                'satdata': lst_satdata,
                'vegetation': lst_opt_vegetation,
                'preprocess_satdata': lst_opt_transform
            }

            if SatDataPreprocessOpt.PCA_PER_BAND in lst_opt_transform:
                for variance in VAR_PCA_RETAINED_VARIANCES:
                    settings_pca = settings.copy()
                    settings_pca['pca_retained_variance'] = variance
                    lst_settings.append(settings_pca)

            if SatDataPreprocessOpt.SAVITZKY_GOLAY in lst_opt_transform:
                settings = lst_settings[-1]
                lst_settings.pop()
                for p in VAR_SAVGOL_POLYORDER_RANGE:
                    for w in VAR_SAVGOL_WINLEN_RANGE:
                        savgol_filter_setting = {'polyorder': p, 'winlen': w}
                        _settings = settings.copy()
                        _settings['savitzky_golay'] = savgol_filter_setting
                        lst_settings.append(_settings)

    print('#config files', len(lst_settings))
