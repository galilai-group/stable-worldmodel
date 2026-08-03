"""Dataset-source converters that stream external demonstrations into SWM.

Unlike :mod:`stable_worldmodel.data.formats`, converters understand a source
library's API (for example MineRL's trajectory iterator) and write a normal
SWM dataset.  The output can then be loaded, converted, merged, and inspected
through the existing format registry.
"""

from .minerl import MineRLConversionSummary, convert_minerl

__all__ = ['MineRLConversionSummary', 'convert_minerl']
