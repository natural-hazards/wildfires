import yaml

from mlfire.data.loader import SatDataSelectOpt
from mlfire.data.fuze import VegetationIndexSelectOpt
from mlfire.data.ts import SatDataPreprocessOpt

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

    if 'transform_ops' in config_dict:
        lst_ops: list[SatDataPreprocessOpt, ...] = []
        for k in config_dict['transform_ops']: lst_ops.append(_config_decode_dict[k])
        decoded_config['transform_ops'] = tuple(lst_ops)
    else:
        decoded_config['transform_ops'] = (SatDataPreprocessOpt.NONE,)

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

    decoded_config['pca_retrained_variance'] = config_dict['pca_retrained_variance']
    return decoded_config


def loadConfigFile(config_file: str) -> dict:

    with open(config_file, 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config_dict = decodeConfigFile(config_dict)
    return config_dict


if __name__ == '__main__':

    VAR_CONFIG_FILE = '/Users/marek/Playground/wildfires/pipelines/test_yaml.yaml'
    print(loadConfigFile(VAR_CONFIG_FILE))
