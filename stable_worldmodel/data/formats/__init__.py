"""Built-in dataset formats. Importing this package registers each format
whose optional dependencies are available; the rest are silently skipped.

Lance and folder ship with the core install. HDF5, video, and lerobot
need their backing libraries (h5py, decord/imageio, lerobot); install
them together with the umbrella extra ``[format]``.
"""

from __future__ import annotations

import logging as _logging

from . import lance  # noqa: F401
from . import folder  # noqa: F401
from . import lerobot  # noqa: F401


def _try_import(modname: str) -> None:
    try:
        __import__(f'{__name__}.{modname}')
    except ImportError as exc:
        _logging.getLogger(__name__).debug(
            "format '%s' not registered: %s", modname, exc
        )


_try_import('hdf5')
_try_import('video')
