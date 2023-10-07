
from typing import Union

# TODO comment
from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')

LIST_NDARRAYS = Union[list[_np.ndarray], tuple[_np.ndarray]]
LIST_STRINGS = Union[tuple[str], list[str], None]
