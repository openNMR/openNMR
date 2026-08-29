"""
nmrsuite
========

Vendor-agnostic NMR import, processing, and plotting. Every vendor
importer in :mod:`nmrsuite.io` reads raw acquisition data and produces
the same :class:`nmrsuite.core.NMRSpectrum`, which
:mod:`nmrsuite.processing` and :mod:`nmrsuite.plotting` then operate on
uniformly, regardless of vendor or dimensionality.

    from nmrsuite import io, processing, plotting
    from nmrsuite.core import NMRSpectrum

    spectrum = io.load("/path/to/bruker/experiment")   # or an X-Pulse .jdx
    fig, ax = plotting.plot_1d_stack([spectrum])
"""

from . import core
from . import io
from . import processing
from . import plotting
from ._version import __version__

__all__ = ["core", "io", "processing", "plotting", "__version__"]
