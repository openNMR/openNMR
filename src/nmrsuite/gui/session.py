"""
In-memory session state for the GUI: which spectra are loaded.

Deliberately has no Qt import -- this is the part of the GUI package a
future browser-based frontend (the planned phase 2) can reuse directly,
since it's just bookkeeping over :mod:`nmrsuite.io`/:class:`NMRSpectrum`,
not anything Qt-specific.
"""

from __future__ import annotations

import dataclasses
import os

from .. import io
from ..core.registry import find_gamma
from ..core.spectrum import NMRSpectrum


class Session:
    """Holds the spectra loaded into one GUI session, in load order."""

    def __init__(self) -> None:
        self.spectra: list[NMRSpectrum] = []
        # Parallel to `spectra` -- how each one was loaded, so a later
        # override (e.g. pseudo_2d) can be applied by reloading with
        # the same call plus that one override, rather than needing a
        # separate reprocessing path for already-loaded data.
        self._load_info: list[dict] = []

    def add(self, path: str, vendor: str | None = None, **kwargs) -> NMRSpectrum:
        """Load ``path`` via :func:`nmrsuite.io.load` (auto-detecting
        the vendor unless ``vendor`` is given) and append it. Returns
        the loaded spectrum."""
        spectrum = io.load(path, vendor=vendor, **kwargs)
        resolved_vendor = vendor or spectrum.metadata.vendor
        self.spectra.append(spectrum)
        self._load_info.append({"path": path, "vendor": resolved_vendor, "kwargs": dict(kwargs)})
        return spectrum

    def reload(self, index: int, **override_kwargs) -> NMRSpectrum:
        """Re-load spectrum ``index`` from its original path, with
        ``override_kwargs`` merged into (and remembered alongside) the
        kwargs it was originally loaded with -- e.g. ``pseudo_2d=True``
        to reinterpret a 2D dataset without needing a from-scratch
        reprocessing path. Any nucleus override already applied (see
        :func:`set_nucleus`) is preserved across the reload."""
        info = self._load_info[index]
        info["kwargs"] = {**info["kwargs"], **override_kwargs}

        old_meta = self.spectra[index].metadata
        spectrum = io.load(info["path"], vendor=info["vendor"], **info["kwargs"])
        spectrum.metadata = dataclasses.replace(
            spectrum.metadata, nucleus=old_meta.nucleus, nucleus_indirect=old_meta.nucleus_indirect,
        )
        self.spectra[index] = spectrum
        return spectrum

    def get_load_kwargs(self, index: int) -> dict:
        """The kwargs spectrum ``index`` was most recently loaded/
        reloaded with -- e.g. so a GUI can restore reprocessing
        controls (FID window/zero-fill/etc.) to what's actually
        applied when the user reselects a previously-reprocessed
        spectrum, instead of resetting them to defaults every time."""
        return dict(self._load_info[index]["kwargs"])

    def remove(self, index: int) -> None:
        del self.spectra[index]
        del self._load_info[index]

    def clear(self) -> None:
        self.spectra.clear()
        self._load_info.clear()

    def set_nucleus(self, index: int, nucleus: str, indirect: bool = False) -> None:
        """Override the (direct, or ``indirect`` for F1) nucleus label
        recorded for spectrum ``index`` -- for cases the vendor's own
        recorded nucleus is simply wrong (e.g. a nuclide picked from an
        instrument's table only for its approximate frequency, on a
        magnet/console combination the table wasn't calibrated for)."""
        field = "nucleus_indirect" if indirect else "nucleus"
        self.spectra[index] = dataclasses.replace(
            self.spectra[index],
            metadata=dataclasses.replace(self.spectra[index].metadata, **{field: nucleus}),
        )

    def set_mass(self, index: int, mass: float) -> None:
        """Record a sample mass (g) for spectrum ``index`` -- never
        recorded by any vendor file, so this is the only way it's
        known. Used for mass-normalized multi-spectrum comparison."""
        self.spectra[index] = dataclasses.replace(
            self.spectra[index],
            metadata=dataclasses.replace(self.spectra[index].metadata, mass=mass),
        )

    def swap(self, i: int, j: int) -> None:
        """Swap the load-order position of spectra ``i`` and ``j`` --
        e.g. for reordering a multi-spectrum comparison."""
        self.spectra[i], self.spectra[j] = self.spectra[j], self.spectra[i]
        self._load_info[i], self._load_info[j] = self._load_info[j], self._load_info[i]

    def suggest_nucleus(self, index: int, indirect: bool = False) -> str | None:
        """Suggest the actual nucleus for spectrum ``index``, for the
        specific case of a nuclide chosen from an instrument table
        calibrated for a 500 MHz (11.75 T) console but actually run on
        a 100 MHz (2.35 T) magnet (see
        :func:`nmrsuite.core.registry.find_gamma`'s ``field='100'``) --
        picks the real nuclide whose gyromagnetic ratio best matches
        the recorded SFO1 once the field mismatch is accounted for.
        Returns None if there's nothing to suggest from (no recorded
        nucleus/SFO1, or the recorded nucleus isn't in the gamma
        table)."""
        meta = self.spectra[index].metadata
        nucleus = meta.nucleus_indirect if indirect else meta.nucleus
        sfo1 = meta.sfo1_indirect if indirect else meta.sfo1
        if not nucleus or not sfo1:
            return None
        result = find_gamma(nucleus, field="100", SFO1=sfo1)
        return result["Nuclide"] if result else None

    def label(self, index: int) -> str:
        """A short, human-readable label for spectrum ``index`` --
        folder name, nucleus, vendor, and procno."""
        spectrum = self.spectra[index]
        meta = spectrum.metadata
        name = os.path.basename(meta.source_path.rstrip("/")) if meta.source_path else f"spectrum {index}"
        bits = [name]
        if meta.nucleus:
            bits.append(meta.nucleus)
        bits.append(meta.vendor)
        if meta.procno is not None:
            bits.append(f"procno {meta.procno}")
        return " · ".join(bits)

    def __len__(self) -> int:
        return len(self.spectra)
