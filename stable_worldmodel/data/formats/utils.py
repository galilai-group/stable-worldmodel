"""Tiny helpers shared by built-in formats."""

from __future__ import annotations

import numpy as np


def is_image_column(vals) -> bool:
    """Return True if `vals` looks like a sequence of HxW image frames."""
    if not vals:
        return False
    sample = np.asarray(vals[0])
    return (
        sample.dtype == np.uint8
        and sample.ndim == 3
        and sample.shape[-1] in (1, 3)
    )
