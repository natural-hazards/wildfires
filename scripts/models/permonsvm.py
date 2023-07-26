from mlfire.utils.functool import lazy_import

_enum = lazy_import('enum')


class SolverType(_enum.Enum):

    MPGP = 'mpgp'
    BLMVM = 'blmvm'


class ModelOutput(_enum.Enum):

    LABEL = 'label'
    PROBABILITY = 'probability'


class HyperParameterOptimization(_enum.Enum):

    GRID_SEARCH = 'grid_search'
    WARM_START = 'with_warm_start'


class LossType(_enum.Enum):

    L1 = 'L1'
    L2 = 'L2'


class ScoreType(_enum.Enum):

    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'F1'
    JACCARD_INDEX = 'jaccard'
    AUC_ROC = 'auc_roc'


class Device(_enum.Enum):

    CPU = 'cpu'
    CUDA = 'cuda'
    HIP = 'hip'
