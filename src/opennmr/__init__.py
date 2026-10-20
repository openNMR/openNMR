"""
openNMR
========

Vendor-agnostic NMR import, processing, and plotting. Every vendor
importer in :mod:`opennmr.io` reads raw acquisition data and produces
the same :class:`opennmr.core.NMRSpectrum`, which
:mod:`opennmr.processing` and :mod:`opennmr.plotting` then operate on
uniformly, regardless of vendor or dimensionality.

    from opennmr import io, processing, plotting
    from opennmr.core import NMRSpectrum

    spectrum = io.load("/path/to/bruker/experiment")   # or an X-Pulse .jdx
    fig, ax = plotting.plot_1d_stack([spectrum])
"""

import logging as _logging


class _DropFontWeightNoise(_logging.Filter):
    """matplotlib logs "findfont: Failed to find font weight normal for
    Apple Chancery, now using 0." while it reads the fonts macOS ships
    (seen in a user's terminal, 2026-10-20). It is harmless -- the face is
    still usable -- and nothing a user can act on, so it is dropped.
    Every other font message (a family that is genuinely missing, say)
    still reaches the log."""

    def filter(self, record: _logging.LogRecord) -> bool:
        return "Failed to find font weight" not in record.getMessage()


_logging.getLogger("matplotlib.font_manager").addFilter(_DropFontWeightNoise())

from . import core
from . import io
from . import processing
from . import plotting
from . import notebook
from ._version import __version__

__all__ = ["core", "io", "processing", "plotting", "notebook", "__version__"]
