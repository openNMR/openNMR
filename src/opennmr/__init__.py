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

from . import core
from . import io
from . import processing
from . import plotting
from ._version import __version__

__all__ = ["core", "io", "processing", "plotting", "__version__"]
