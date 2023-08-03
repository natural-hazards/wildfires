
from mlfire.utils.functool import lazy_import

# lazy imports
_enum = lazy_import('enum')


class ScoreTypes(_enum.Enum):

    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'F1'
    JACCARD_INDEX = 'jaccard'
    AUC_ROC = 'auc_roc'
