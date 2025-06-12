# Import submodules
from . import Flagging_module
from . import Utils_module
from . import Background_module
from . import Detection_module
from . import Cleaning_module
from . import Metrics_module
from . import Petrosian_module
from . import cleaning  # Cython-compiled extension module


__version__ = "0.1.0"

# Define public API
__all__ = [
    "Flagging_module",
    "Utils_module",
    "Background_module",
    "Detection_module",
    "Cleaning_module",
    "Metrics_module",
    "Petrosian_module",
    "cleaning",
]


