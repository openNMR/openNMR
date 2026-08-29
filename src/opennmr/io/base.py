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


@dataclass
class VendorReader:
    name: str
    detect: Detector
    read: Reader


_REGISTRY: dict[str, VendorReader] = {}


def register(name: str, detect: Detector, read: Reader) -> None:
    """Register a vendor. ``detect(path)`` returns True if ``path``
    looks like this vendor's data; ``read(path, **kwargs)`` returns an
    :class:`NMRSpectrum`."""
    _REGISTRY[name] = VendorReader(name=name, detect=detect, read=read)


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


def _exists(*parts: str) -> Callable[[str], bool]:
    """Helper: build a detector that checks one or more paths (each
    relative to the candidate directory) all exist."""
    def detector(path: str) -> bool:
        return all(os.path.exists(os.path.join(path, p)) for p in parts)
    return detector
