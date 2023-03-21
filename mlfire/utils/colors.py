from enum import Enum
from mlfire.utils.functool import lazy_import

np = lazy_import('numpy')


class Colors(Enum):

    GRAY_COLOR = [85., 90., 90.]
    RED_COLOR = [43, 75, 238]
