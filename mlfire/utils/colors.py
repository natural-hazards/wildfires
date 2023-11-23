from enum import Enum
from mlfire.utils.functool import lazy_import

np = lazy_import('numpy')


class Colors(Enum):

    GRAY_COLOR = (90, 90, 85)
    GREEN_COLOR = (119, 221, 119)
    RED_COLOR = (238, 75, 43)
    WHITE_COLOR = (255, 255, 255)

