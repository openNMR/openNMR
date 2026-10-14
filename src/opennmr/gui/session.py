"""
In-memory session state for the GUI: which spectra are loaded.

Deliberately has no Qt import -- this is the part of the GUI package a
future browser-based frontend (the planned phase 2) can reuse directly,
since it's just bookkeeping over :mod:`opennmr.io`/:class:`NMRSpectrum`,
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
        """Load ``path`` via :func:`opennmr.io.load` (auto-detecting
        the vendor unless ``vendor`` is given) and append it. Returns
        the loaded spectrum."""
        spectrum = io.load(path, vendor=vendor, **kwargs)
        resolved_vendor = vendor or spectrum.metadata.vendor
        self.spectra.append(spectrum)
        self._load_info.append({"path": path, "vendor": resolved_vendor, "kwargs": dict(kwargs)})
        return spectrum

    def adopt(self, path: str, vendor: str, kwargs: dict, spectrum: NMRSpectrum) -> NMRSpectrum:
        """Append an *already-read* spectrum, recording how it was
        loaded exactly as :meth:`add` would.

        For the background loader: the worker thread does the read and
        hands the spectrum over, and this -- called on the GUI thread --
        is what puts it in the session. The two lists here are mutable
        shared state, so only one thread may ever touch them, which is
        why the worker cannot simply call :meth:`add` itself."""
        self.spectra.append(spectrum)
        self._load_info.append({"path": path, "vendor": vendor, "kwargs": dict(kwargs)})
        return spectrum

    @staticmethod
    def _path_key(path: str | None) -> str | None:
        """One canonical spelling of ``path``, for comparing two loads.

        ``realpath`` (not ``abspath``) so a symlinked data folder and
        its target are recognised as the same experiment, and a
        trailing slash -- which a folder picker adds and a drag-and-drop
        does not -- is stripped first so it cannot make two spellings of
        one Bruker experiment folder look different. Case is left alone:
        macOS and Windows are usually case-insensitive and Linux is not,
        and folding it here would wrongly merge two genuinely different
        folders on Linux.
        """
        if not path:
            return None
        return os.path.realpath(os.path.normpath(path.rstrip("/")))

    def index_for_path(self, path: str, vendor: str | None = None,
                       procno: int | None = None) -> int | None:
        """The index of the already-loaded spectrum read from ``path``,
        or ``None``.

        ``vendor`` : when given, only a match loaded by that same
        reader counts -- one folder can legitimately be opened by two
        vendors' readers (a Bruker directory that also holds a JCAMP
        export), and those are two different spectra, not a duplicate.

        Derived spectra (VOCS output, spectrum arithmetic, a resolved
        SCORE component) have no path at all and can never match.
        """
        key = self._path_key(path)
        if key is None:
            return None
        for index, info in enumerate(self._load_info):
            if self._path_key(info.get("path")) != key:
                continue
            if vendor is not None and info.get("vendor") != vendor:
                continue
            # ``procno``: only that processing of the experiment counts
            # (two proc numbers of one experiment are two spectra).
            if procno is not None and self.procno(index) != int(procno):
                continue
            return index
        return None

    def procno(self, index: int) -> int:
        """The proc number spectrum ``index`` was read with (1 if none)."""
        meta_procno = getattr(self.spectra[index].metadata, "procno", None)
        value = self._load_info[index]["kwargs"].get("procno", meta_procno)
        try:
            return int(value) if value is not None else 1
        except (TypeError, ValueError):
            return 1

    def reload(self, index: int, **override_kwargs) -> NMRSpectrum:
        """Re-load spectrum ``index`` from its original path, with
        ``override_kwargs`` merged into (and remembered alongside) the
        kwargs it was originally loaded with -- e.g. ``pseudo_2d=True``
        to reinterpret a 2D dataset without needing a from-scratch
        reprocessing path. Any nucleus override already applied (see
        :func:`set_nucleus`) is preserved across the reload."""
        info = self._load_info[index]
        info["kwargs"] = {**info["kwargs"], **override_kwargs}
        spectrum = self._load_with(index, info["kwargs"])
        self.spectra[index] = spectrum
        return spectrum

    def reset(self, index: int, **kwargs) -> NMRSpectrum:
        """Re-load spectrum ``index`` from its original path with
        exactly ``kwargs`` -- *replacing* the remembered load kwargs
        entirely, unlike :meth:`reload`'s merge. For a GUI "Reset"
        action discarding every live processing choice back to a
        specific known-good starting point (typically what the caller
        itself snapshotted right after the initial load), not
        layering one more override on top of whatever's accumulated
        since."""
        info = self._load_info[index]
        info["kwargs"] = dict(kwargs)
        spectrum = self._load_with(index, info["kwargs"])
        self.spectra[index] = spectrum
        return spectrum

    def preview(self, index: int, **override_kwargs) -> NMRSpectrum:
        """Like :meth:`reload`, but purely a scratch re-read -- doesn't
        replace the stored spectrum or its remembered load kwargs.
        For a GUI live-preview of "what would committing this change
        actually produce" (e.g. the Processing tab's own dual FID/
        spectrum view, live as a not-yet-applied window/zero-fill/etc.
        choice changes) that the user hasn't clicked "Apply Processing"
        for yet -- :meth:`reload` would be the wrong tool here, since
        it immediately commits."""
        kwargs = {**self._load_info[index]["kwargs"], **override_kwargs}
        return self._load_with(index, kwargs)

    def _load_with(self, index: int, kwargs: dict) -> NMRSpectrum:
        info = self._load_info[index]
        old_meta = self.spectra[index].metadata
        reload_kwargs = dict(kwargs)
        # Every reload/reset/preview on this slot re-reads the *same*
        # info["path"] -- only kwargs changes -- and Bruker's raw,
        # digital-filter-removed FID doesn't depend on any of read_2d's
        # own parameters except path itself (see its docstring). So
        # whatever fid_raw the previous load already cached is always
        # still valid here, and passing it back in skips re-reading a
        # (often hundreds-of-MB) ser file and redoing an ~2.4s digital-
        # filter removal on every single reprocess action. Gated on
        # vendor: no other reader declares (or needs) this kwarg, and
        # an unexpected one would raise a TypeError if it leaked through.
        # A drift-corrected load keeps the corrected FIDs as fid_raw (what
        # in-memory previews reprocess from) and the file's own as
        # fid_raw_uncorrected: the cache must be the latter, or read_2d
        # would correct the already-corrected FIDs a second time.
        cached = old_meta.raw.get("fid_raw_uncorrected")
        if cached is None:
            cached = old_meta.raw.get("fid_raw")
        if old_meta.vendor == "bruker" and cached is not None:
            reload_kwargs["_raw_fid_cache"] = cached
        spectrum = io.load(info["path"], vendor=info["vendor"], **reload_kwargs)
        spectrum.metadata = dataclasses.replace(
            spectrum.metadata, nucleus=old_meta.nucleus, nucleus_indirect=old_meta.nucleus_indirect,
        )
        return spectrum

    def get_load_kwargs(self, index: int) -> dict:
        """The kwargs spectrum ``index`` was most recently loaded/
        reloaded with -- e.g. so a GUI can restore reprocessing
        controls (FID window/zero-fill/etc.) to what's actually
        applied when the user reselects a previously-reprocessed
        spectrum, instead of resetting them to defaults every time."""
        return dict(self._load_info[index]["kwargs"])

    def replace_data(self, index: int, data, data_imag=None) -> NMRSpectrum:
        """Swap spectrum ``index``'s data arrays, keeping everything
        else about it.

        For a processing step that *transforms* a spectrum rather than
        producing a new one -- reference deconvolution is the first.
        Deliberately not :meth:`add_derived`: that sets
        ``source_path`` to a display label, which would cut the result
        off from the experiment's own files. A reference-deconvolved
        eNMR series still has to be able to find its ``vdlist`` and
        ``diff.xml``, so the metadata has to survive intact.

        The original is still one click away: this session's own
        Reset re-reads from the file, and the load kwargs are
        untouched.
        """
        spectrum = self.spectra[index]
        self.spectra[index] = dataclasses.replace(
            spectrum, data=data, data_imag=data_imag)
        return self.spectra[index]

    def add_derived(self, spectrum: NMRSpectrum, label: str) -> int:
        """Append a spectrum computed from already-loaded data (VOCS,
        spectrum arithmetic, ...) rather than read from a file --
        there's no path/reload kwargs for it, so :meth:`reload`/
        :meth:`preview` are simply not meaningful for this entry.
        ``metadata.vendor`` is overwritten to ``"derived"`` (and
        ``source_path`` to ``label``, used by :meth:`label` for
        display) so GUI controls that need a raw FID to reprocess from
        (gated via ``MainWindow._fid_reprocessable``, which only
        recognises the real vendor names) correctly stay hidden for
        it. Returns the new spectrum's session index."""
        # Where it came from, before source_path is overwritten with the
        # label: the experiment folder it was derived from (a slice, a
        # CPMG sum, a VOCS correction), or None for a simulation. Kept
        # here rather than in metadata.raw, which a derived spectrum
        # can share with its parent. Used to suggest where Save
        # Progress puts it (io.history.default_dataset_folder).
        origin = spectrum.metadata.source_path
        if not (origin and os.path.exists(origin)):
            origin = None
        origin_vendor = spectrum.metadata.vendor
        metadata = dataclasses.replace(spectrum.metadata, vendor="derived", source_path=label)
        spectrum = dataclasses.replace(spectrum, metadata=metadata)
        self.spectra.append(spectrum)
        self._load_info.append({"path": None, "vendor": "derived", "kwargs": {},
                                "origin": origin, "origin_vendor": origin_vendor})
        return len(self.spectra) - 1

    def info(self, index: int) -> dict:
        """The live bookkeeping dict for spectrum ``index`` (how it was
        loaded, where a derived one came from, its processing history
        -- see :mod:`opennmr.io.history`). Moves with the spectrum on
        :meth:`swap`/:meth:`remove`, and survives :meth:`reload`/
        :meth:`reset`, which replace only its ``kwargs``."""
        return self._load_info[index]

    def rebase(self, index: int, spectrum: NMRSpectrum, path: str, vendor: str) -> None:
        """Make spectrum ``index`` a disk-backed one read from ``path``
        -- a derived spectrum just saved as an openNMR dataset folder
        (:func:`opennmr.io.history.save_spectrum_folder`). Its other
        bookkeeping (history, origin) is kept."""
        self.spectra[index] = spectrum
        info = self._load_info[index]
        info.update({"path": path, "vendor": vendor, "kwargs": {}})

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
        :func:`opennmr.core.registry.find_gamma`'s ``field='100'``) --
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
        result = find_gamma(nucleus, field="100", SFO1=sfo1, quiet=True)
        return result["Nuclide"] if result else None

    def label(self, index: int) -> str:
        """A short, human-readable label for spectrum ``index`` --
        folder name, nucleus, vendor, and procno."""
        spectrum = self.spectra[index]
        meta = spectrum.metadata
        # A name the user typed wins outright. One override, honoured
        # here, reaches everything that names a spectrum: the Data-tab
        # list, the Compare legend, a DOSY overlay's own legend, the
        # processing report. Explicit user request: "I would like the
        # ability to manipulate things like the legend name".
        override = (meta.raw or {}).get("display_label")
        if isinstance(override, str) and override.strip():
            return override.strip()
        if meta.vendor == "derived" and meta.source_path:
            # A derived spectrum's source_path is a *label* that
            # add_derived put there, not a path -- basename()ing it
            # silently truncates at any "/" the label happens to
            # contain. Found with a resolved SCORE component labelled
            # with its diffusion coefficient: "... (2.50e-10 m^2/s)"
            # came out as "s)". Any derived label with a slash in it --
            # a ratio from spectrum arithmetic, a unit -- had the same
            # problem.
            name = meta.source_path
        elif meta.source_path:
            name = os.path.basename(meta.source_path.rstrip("/"))
        else:
            name = f"spectrum {index}"
        bits = [name]
        if meta.nucleus:
            bits.append(meta.nucleus)
        bits.append(meta.vendor)
        if meta.procno is not None:
            bits.append(f"procno {meta.procno}")
        return " · ".join(bits)

    def __len__(self) -> int:
        return len(self.spectra)
