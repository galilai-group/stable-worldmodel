from .utils import *  # noqa: F403
from .dataset import *  # noqa: F403
from .format import (
    FORMATS,
    WRITE_MODES,
    Format,
    Writer,
    detect_format,
    get_format,
    list_formats,
    register_format,
    validate_write_mode,
)

# Importing the formats subpackage registers all built-in formats whose
# optional deps are installed.
from . import formats as _formats  # noqa: F401

# Re-export concrete readers/writers from their format modules so existing
# imports like `from stable_worldmodel.data import LanceDataset` keep working.
# Optional formats (hdf5, video) are re-exported only when their extras are
# installed; absent ones are simply not bound at module level.
from .formats.lance import LanceDataset, LanceWriter
from .formats.folder import FolderDataset, FolderWriter, ImageDataset
from .formats.lerobot import LeRobotAdapter

try:
    from .formats.hdf5 import HDF5Dataset, HDF5Writer  # noqa: F401
except ImportError:
    pass

try:
    from .formats.video import VideoDataset, VideoWriter  # noqa: F401
except ImportError:
    pass


__all__ = [
    'FORMATS',
    'Format',
    'FolderDataset',
    'FolderWriter',
    'ImageDataset',
    'LanceDataset',
    'LanceWriter',
    'LeRobotAdapter',
    'WRITE_MODES',
    'Writer',
    'detect_format',
    'get_format',
    'list_formats',
    'register_format',
    'validate_write_mode',
]
