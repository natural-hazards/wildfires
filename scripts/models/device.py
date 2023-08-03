
from mlfire.utils.functool import lazy_import

# lazy imports
_enum = lazy_import('enum')


class DeviceTypes(_enum.Enum):

    CPU = 'cpu'
    CUDA = 'cuda'
    HIP = 'hip'
