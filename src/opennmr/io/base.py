"""
Vendor-reader registry and auto-detecting ``load()``.

Every vendor module (``opennmr.io.bruker``, ``opennmr.io.xpulse``, and
eventually JEOL/Varian) registers a detector + reader pair here. Adding
a new vendor later means writing one module that calls
:func:`register`, not restructuring this dispatcher.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from ..core.spectrum import NMRSpectrum

Detector = Callable[[str], bool]
Reader = Callable[..., NMRSpectrum]
Peeker = Callable[..., dict]


@dataclass
class VendorReader:
    name: str
    detect: Detector
    read: Reader
    #: Optional fast, metadata-only probe: shape/nucleus/etc. from a
    #: dataset's (tiny) parameter files alone, never touching its large
    #: data file(s). None for a vendor that hasn't implemented one yet
    #: -- peek() below falls back to a full load rather than requiring
    #: every vendor to support this from day one.
    peek: Peeker | None = None


_REGISTRY: dict[str, VendorReader] = {}


def register(name: str, detect: Detector, read: Reader, peek: Peeker | None = None) -> None:
    """Register a vendor. ``detect(path)`` returns True if ``path``
    looks like this vendor's data; ``read(path, **kwargs)`` returns an
    :class:`NMRSpectrum`; ``peek(path, **kwargs)``, if given, returns a
    metadata-only dict (see :func:`peek`)."""
    _REGISTRY[name] = VendorReader(name=name, detect=detect, read=read, peek=peek)


def registered_vendors() -> list[str]:
    return list(_REGISTRY)


def detect_vendor(path: str) -> str:
    """Return the name of the first registered vendor whose
    ``detect(path)`` returns True.

    Raises
    ------
    ValueError
        If no registered vendor recognises ``path``.
    """
    for name, vendor in _REGISTRY.items():
        if vendor.detect(path):
            return name
    raise ValueError(
        f"Could not auto-detect a vendor format for {path!r}. "
        f"Registered vendors: {registered_vendors()}. "
        f"Pass vendor= explicitly if you know the format."
    )


def load(path: str, vendor: str | None = None, **kwargs) -> NMRSpectrum:
    """Load ``path`` as an :class:`NMRSpectrum`, auto-detecting the
    vendor format unless ``vendor`` is given explicitly.

    ``**kwargs`` are forwarded to the resolved vendor's ``read()``
    (e.g. ``procno``, ``mass``, ``f1p``/``f2p`` for Bruker).
    """
    if vendor is None:
        vendor = detect_vendor(path)
    if vendor not in _REGISTRY:
        raise ValueError(
            f"Unknown vendor {vendor!r}. Registered vendors: {registered_vendors()}."
        )
    return _REGISTRY[vendor].read(path, **kwargs)


def peek(path: str, vendor: str | None = None, **kwargs) -> dict:
    """Learn what ``path`` holds (shape, nucleus, dimensionality, ...)
    without necessarily reading its large data file(s) -- for a Data-
    tab folder browser deciding whether/what a directory holds, or any
    other caller that just needs to know what's there before committing
    to a full :func:`load`.

    Falls back to a full ``load()`` (correct, just not fast) for a
    vendor that hasn't registered its own ``peek`` -- this is purely
    additive, so no existing vendor has to support it to keep working.
    """
    if vendor is None:
        vendor = detect_vendor(path)
    if vendor not in _REGISTRY:
        raise ValueError(
            f"Unknown vendor {vendor!r}. Registered vendors: {registered_vendors()}."
        )
    entry = _REGISTRY[vendor]
    if entry.peek is not None:
        return entry.peek(path, **kwargs)
    spectrum = entry.read(path, **kwargs)
    return {
        "vendor": spectrum.metadata.vendor,
        "ndim": spectrum.data.ndim,
        "shape": spectrum.data.shape,
        "nucleus": spectrum.metadata.nucleus,
        "nucleus_indirect": spectrum.metadata.nucleus_indirect,
        "pseudo_2d": spectrum.metadata.pseudo_2d,
    }


def _exists(*parts: str) -> Callable[[str], bool]:
    """Helper: build a detector that checks one or more paths (each
    relative to the candidate directory) all exist."""
    def detector(path: str) -> bool:
        return all(os.path.exists(os.path.join(path, p)) for p in parts)
    return detector
