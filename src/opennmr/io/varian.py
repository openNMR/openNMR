"""
Varian/Agilent (VnmrJ) importer.

Like X-Pulse, Varian hands back only a raw time-domain FID -- there's
no vendor-processed fallback the way Bruker's TopSpin ``1r``/``1i``
are -- so every spectrum goes through this project's shared
:mod:`opennmr.processing.ft` pipeline (apodize -> zero-fill -> FFT ->
phase -> baseline) in Python.

The raw binary ``fid``/``procpar`` read is :mod:`opennmr.io.varian_binary`,
this project's own -- no ``nmrglue``. It is pinned by a test asserting
**byte-identical** output against ``nmrglue`` on the real datasets in
``test_data``, which is the standard this kind of format code has to
meet: a subtle error in header offsets or real/imaginary interleaving
produces a plausible-looking but wrong spectrum rather than a crash.

**Shift referencing** (:func:`_ppm_axis`) follows VnmrJ's own
definition: the spectrum is centred on ``sfrq`` and 0 ppm sits at
``reffrq``. ``sfrq`` is the transmitter frequency and *already
includes* ``tof``, so ``tof`` plays no part. Until 2026-10-22 it was
subtracted a second time, which moved every spectrum by
``-tof/sfrq`` ppm: -155.6 ppm on a 27Al spectrum with ``tof = 20 kHz``,
-1.0 ppm on 1H with ``tof = 499.8 Hz``. ``rfl``/``rfp`` (VnmrJ's
reference-line position and value) are not read because ``reffrq``
already encodes them: VnmrJ sets ``reffrq = sfrq - (sw/2 - rfl +
rfp)/1e6``, and the two agree to 1e-7 ppm on real data. ``setref``
derives ``reffrq`` from the lock (``H1reffrq`` times the nucleus's
IUPAC Xi ratio), so this is lock-based referencing. Checked against
known 27Al shifts (AlCl4-/Al2Cl7- at 103/97 ppm, pinned by
``tests/test_varian.py``) and against TopSpin's own ``vconv``
conversion of the same data.
"""

from __future__ import annotations

import os
import re

import numpy as np

from . import base
from . import varian_binary
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from ..processing import ft as processing_ft


_NUCLEUS_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _normalize_nucleus(tn: str | None) -> str | None:
    """Varian's ``tn`` ("Li7", "H1") is symbol-then-number -- every
    other vendor here (and :func:`opennmr.core.registry.find_gamma`)
    uses number-then-symbol ("7Li", "1H"), so swap it rather than
    silently breaking nucleus-based lookups/overrides downstream."""
    if not tn:
        return None
    m = _NUCLEUS_RE.match(tn.strip())
    if m:
        symbol, number = m.groups()
        return f"{number}{symbol}"
    return tn


def _procpar_value(procpar: dict, key: str, default=None, cast=float):
    """One value out of nmrglue's ``procpar`` dict (every entry is
    ``{"values": [str, ...], ...}``, even scalar parameters)."""
    entry = procpar.get(key)
    if not entry or not entry.get("values"):
        return default
    try:
        return cast(entry["values"][0])
    except (TypeError, ValueError):
        return default


def _ppm_axis(freq_hz: np.ndarray, sfrq_hz: float, reffrq_hz: float) -> np.ndarray:
    """ppm for each point of a spectrum whose ``freq_hz`` is the
    offset from the spectrum's centre, as :func:`opennmr.processing.ft.
    fft_spectrum` returns it.

    VnmrJ's definition: the centre is at ``sfrq`` and 0 ppm at
    ``reffrq``, so a point sits at ``sfrq + freq_hz`` and its shift is
    measured from ``reffrq``, in units of ``reffrq``. ``tof`` is
    deliberately absent -- ``sfrq`` already includes it (see this
    module's docstring)."""
    return (np.asarray(freq_hz) + (sfrq_hz - reffrq_hz)) / reffrq_hz * 1e6


# ============================================================
# 1D
# ============================================================

def read_1d(path: str, mass: float = 1.0,
            window: str = "exponential", window_kwargs: dict | None = None,
            zero_fill: int | None = None,
            fid_shift: float = 0.0, align_echo: bool = False,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward",
            baseline_correct: bool = False, normalize: bool = False,
            autophase: bool = True, phase_mode: str = "apc0",
            phase0: float = 0.0, phase1: float = 0.0, pivot_frac: float = 0.5, phase0_inc: float = 0.1,
            phase1_max: float = 20.0, phase1_improvement: float = 0.15,
            phase_penalty: str = "magnitude") -> NMRSpectrum:
    """Read a Varian/Agilent 1D FID directory (``fid`` + ``procpar``)
    and process it in Python.

    ``window``/``window_kwargs``/``zero_fill``/``fid_shift``/
    ``align_echo``/``linear_predict_*``/``baseline_correct``/
    ``normalize``/``autophase``/``phase_mode``/``phase0``/``phase1``/
    ``phase0_inc``/``phase1_max``/``phase1_improvement``/
    ``phase_penalty`` : identical meaning to
    :func:`opennmr.io.xpulse.read_1d`'s same-named parameters -- both
    vendors hand back a raw FID and share this project's one
    processing pipeline (:mod:`opennmr.processing.ft`).
    """
    dic, fid = varian_binary.read(path)
    procpar = dic["procpar"]
    fid = np.asarray(fid, dtype=np.complex128)
    fid_raw = fid  # the truly raw FID -- see metadata.raw["fid_raw"]'s own comment below
    original_length = len(fid)

    sw = _procpar_value(procpar, "sw")
    if not sw:
        raise ValueError(f"No spectral width (sw) recorded in {path!r}'s procpar.")
    dt = 1.0 / sw

    sfrq_hz = _procpar_value(procpar, "sfrq", default=1.0) * 1e6  # observed freq
    reffrq_hz = _procpar_value(procpar, "reffrq", default=sfrq_hz / 1e6) * 1e6  # ppm=0 reference
    nucleus = _normalize_nucleus(_procpar_value(procpar, "tn", default=None, cast=str))
    ns = _procpar_value(procpar, "nt", default=1.0)
    np_points = _procpar_value(procpar, "np", default=2 * len(fid), cast=int) // 2

    if fid_shift:
        fid = processing_ft.roll_fid(fid, fid_shift)
    if align_echo:
        fid = processing_ft.align_echo(fid)
    if linear_predict_points:
        fid = processing_ft.linear_predict(
            fid, order=linear_predict_order, n_predict=linear_predict_points,
            direction=linear_predict_direction,
        )
        np_points = len(fid)

    fid = processing_ft.apodize(fid, dt, window=window, **(window_kwargs or {}))

    if isinstance(zero_fill, int) and zero_fill > len(fid):
        fid = processing_ft.zero_fill(fid, zero_fill)
    else:
        fid = fid[:np_points]

    freq_hz, spectrum = processing_ft.fft_spectrum(fid, dt)
    freq_ppm = _ppm_axis(freq_hz, sfrq_hz, reffrq_hz)

    if autophase:
        if phase_mode == "joint":
            phase0, phase1 = processing_ft.autophase01(
                spectrum, freq_ppm, phase0_range=(-180, 180), phase0_inc=phase0_inc,
                phase1_max=phase1_max, phase1_improvement=phase1_improvement,
                penalty=phase_penalty, pivot_frac=0.5,
            )
        elif phase_mode == "apc0":
            phase0 = processing_ft.apc0(spectrum, freq_ppm, phase0_inc)
        else:
            raise ValueError("phase_mode must be 'apc0' or 'joint'")

    spectrum = processing_ft.apply_phase(spectrum, freq_ppm, phase0, phase1, pivot_frac=pivot_frac)

    if baseline_correct:
        spectrum = processing_ft.baseline_correct(freq_ppm, spectrum)

    spectrum = spectrum / mass / ns  # match the other vendor readers: data is always per-scan

    if normalize:
        maxval = np.max(np.abs(spectrum))
        if maxval > 0:
            spectrum = spectrum / maxval

    metadata = NMRMetadata(
        vendor="varian",
        nucleus=nucleus,
        sfo1=sfrq_hz / 1e6,
        bf1=reffrq_hz / 1e6,
        # Centre relative to bf1, so sfo1 = bf1 + o1 (TopSpin's vconv
        # stores O1 the same way). Not ``tof``: that is relative to
        # VnmrJ's own base frequency, which no field here records.
        o1=sfrq_hz - reffrq_hz,
        ns=ns,
        source_path=path,
        probe=_procpar_value(procpar, "probe", default=None, cast=str),
        pulse_program=_procpar_value(procpar, "seqfil", default=None, cast=str),
        raw={
            "procpar": procpar, "phase0": phase0, "phase1": phase1,
            "fid": fid, "fid_dt": dt, "fid_original_length": original_length,
            # The *truly* raw FID -- straight off disk, before even
            # shift/align/LP, let alone windowing/zero-fill. See
            # opennmr.io.bruker's matching "fid_raw" comment: lets the
            # GUI recompute a live preview of a not-yet-applied
            # processing choice entirely in memory.
            "fid_raw": fid_raw,
        },
    )
    axis = NMRAxis(kind="frequency", ppm=freq_ppm, hz=freq_hz)

    return NMRSpectrum(
        data=np.real(spectrum),
        axes=[axis],
        metadata=metadata,
        data_imag=np.imag(spectrum),
    )


# ============================================================
# 2D
# ============================================================

def read_2d(path: str, mass: float = 1.0, pseudo_2d: bool | None = None,
            window: str = "exponential", window_kwargs: dict | None = None,
            zero_fill: int | None = None,
            fid_shift: float = 0.0, align_echo: bool = False,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward",
            autophase: bool = True, phase_mode: str = "apc0",
            phase0: float = 0.0, phase1: float = 0.0, pivot_frac: float = 0.5, phase0_inc: float = 0.1,
            phase1_max: float = 20.0, phase1_improvement: float = 0.15,
            phase_penalty: str = "magnitude",
            drift_correction: dict | None = None) -> NMRSpectrum:
    """Read a Varian/Agilent 2D FID directory.

    **Preliminary**: correctly assembling Varian's States-hypercomplex
    F1 dimension depends on the specific ``array``/``phase``
    acquisition scheme used (this reads it via ``ni`` x 2 phase steps,
    the standard States layout, confirmed structurally against real
    ``test_data/varian_2d`` -- but not yet checked against a known-good
    2D lineshape from another source). Treat with real caution before
    trusting a fitted result from this path -- see ``TODO.md``.

    ``pseudo_2d`` : ``None`` (default) treats every arrayed 2D as
    pseudo-2D (F1 gets no FFT, no known-frequency-dimension case is
    supported yet) -- pass ``False`` once true F1-frequency (COSY/
    HSQC-style) support is added.
    """
    # as_2d=True: this function does its own reshaping (below, based
    # on ni/the standard States-hypercomplex layout) rather than
    # nmrglue's own shape/torder auto-detection, which both warns and
    # falls back to unreshaped raw data for this kind of phase-arrayed
    # acquisition anyway.
    dic, fid = varian_binary.read(path, as_2d=True)
    procpar = dic["procpar"]

    ni = _procpar_value(procpar, "ni", cast=int)
    sw = _procpar_value(procpar, "sw")
    if not sw:
        raise ValueError(f"No spectral width (sw) recorded in {path!r}'s procpar.")
    dt = 1.0 / sw

    n_direct_points = fid.shape[-1]
    n_traces = fid.shape[0] if fid.ndim == 2 else int(np.prod(fid.shape[:-1]))
    if ni and n_traces == 2 * ni:
        # Standard States-hypercomplex layout: 2 phase-cycled traces
        # (real/imaginary F1 components) per t1 increment.
        fid = np.asarray(fid, dtype=np.complex128).reshape(ni, 2, n_direct_points)
        fid_f1 = fid[:, 0, :] + 1j * fid[:, 1, :]
    else:
        # No recognised hypercomplex layout -- fall back to reading
        # every trace as its own row (still usable as a pseudo-2D
        # series, just without a properly assembled complex F1).
        fid_f1 = np.asarray(fid, dtype=np.complex128).reshape(n_traces, n_direct_points)

    sfrq_hz = _procpar_value(procpar, "sfrq", default=1.0) * 1e6
    reffrq_hz = _procpar_value(procpar, "reffrq", default=sfrq_hz / 1e6) * 1e6
    nucleus = _normalize_nucleus(_procpar_value(procpar, "tn", default=None, cast=str))
    ns = _procpar_value(procpar, "nt", default=1.0)

    # Linear field-drift correction of the raw FIDs, row by row in
    # acquisition order (``drift_correction={"start_hz":, "end_hz":}``;
    # see notebook.drift_correction). Rows are separate experiments, so
    # only the direct-dimension correction applies.
    drift_record = None
    if drift_correction:
        from ..notebook import drift_correction as _dc
        fid_f1 = _dc.correct_pseudo_2d_fids(
            fid_f1, dt, start_hz=drift_correction.get("start_hz", 0.0),
            end_hz=drift_correction.get("end_hz"), shifts_hz=drift_correction.get("shifts_hz"),
            endpoint=bool(drift_correction.get("endpoint", True)))
        drift_record = _dc.pseudo_2d_record(fid_f1.shape[0], drift_correction, "Varian")

    # fid_shift/align_echo/linear_predict are all 1D-only too -- same
    # per-row reasoning as apodize/zero_fill/fft_spectrum below.
    if fid_shift or align_echo or linear_predict_points:
        def _pre_process_row(row):
            if fid_shift:
                row = processing_ft.roll_fid(row, fid_shift)
            if align_echo:
                row = processing_ft.align_echo(row)
            if linear_predict_points:
                row = processing_ft.linear_predict(
                    row, order=linear_predict_order, n_predict=linear_predict_points,
                    direction=linear_predict_direction,
                )
            return row
        fid_f1 = np.array([_pre_process_row(row) for row in fid_f1])

    # apodize/zero_fill/fft_spectrum are all 1D-only (weights/fftfreq
    # sized off len(fid), fftshift would also reorder the row/F1 axis
    # of a 2D array) -- apply row by row, matching bruker.read_2d's
    # own per-row windowing pattern.
    fid_f2 = np.apply_along_axis(
        lambda row: processing_ft.apodize(row, dt, window=window, **(window_kwargs or {})), 1, fid_f1,
    )
    if isinstance(zero_fill, int) and zero_fill > fid_f2.shape[-1]:
        fid_f2 = np.apply_along_axis(lambda row: processing_ft.zero_fill(row, zero_fill), 1, fid_f2)

    freq_hz = None
    spectrum_rows = []
    for row in fid_f2:
        freq_hz, spec_row = processing_ft.fft_spectrum(row, dt)
        spectrum_rows.append(spec_row)
    spectrum = np.array(spectrum_rows)
    freq_ppm = _ppm_axis(freq_hz, sfrq_hz, reffrq_hz)

    if autophase:
        # Phase against the summed (all-rows) spectrum, then apply the
        # same correction to every row -- matches bruker.read_2d's own
        # "one phase for the whole F2 dimension" behaviour.
        summed = np.sum(spectrum, axis=0)
        if phase_mode == "joint":
            phase0, phase1 = processing_ft.autophase01(
                summed, freq_ppm, phase0_range=(-180, 180), phase0_inc=phase0_inc,
                phase1_max=phase1_max, phase1_improvement=phase1_improvement,
                penalty=phase_penalty, pivot_frac=0.5,
            )
        elif phase_mode == "apc0":
            phase0 = processing_ft.apc0(summed, freq_ppm, phase0_inc)
        else:
            raise ValueError("phase_mode must be 'apc0' or 'joint'")

    spectrum = np.apply_along_axis(
        lambda row: processing_ft.apply_phase(row, freq_ppm, phase0, phase1, pivot_frac=pivot_frac), 1, spectrum,
    )
    spectrum = spectrum / mass / ns

    y_index = np.arange(1, spectrum.shape[0] + 1)
    y_axis = NMRAxis(kind="index", index=y_index)
    f2_axis = NMRAxis(kind="frequency", ppm=freq_ppm, hz=freq_hz)

    metadata = NMRMetadata(
        vendor="varian",
        nucleus=nucleus,
        sfo1=sfrq_hz / 1e6,
        bf1=reffrq_hz / 1e6,
        # Centre relative to bf1, so sfo1 = bf1 + o1 (TopSpin's vconv
        # stores O1 the same way). Not ``tof``: that is relative to
        # VnmrJ's own base frequency, which no field here records.
        o1=sfrq_hz - reffrq_hz,
        ns=ns,
        source_path=path,
        pseudo_2d=True,
        probe=_procpar_value(procpar, "probe", default=None, cast=str),
        pulse_program=_procpar_value(procpar, "seqfil", default=None, cast=str),
        raw={
            "procpar": procpar, "phase0": phase0, "phase1": phase1,
            "fid": fid_f2, "fid_dt": dt,
            **({"drift_correction": drift_record} if drift_record is not None else {}),
        },
    )

    return NMRSpectrum(
        data=np.real(spectrum),
        axes=[y_axis, f2_axis],
        metadata=metadata,
        data_imag=np.imag(spectrum),
    )


# ============================================================
# Vendor registration
# ============================================================

def _detect(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "procpar")) and os.path.exists(os.path.join(path, "fid"))


def _read_auto(path: str, **kwargs) -> NMRSpectrum:
    """Vendor-registry entry point: ``ni`` (indirect-dimension
    increments, always recorded even for a 1D experiment, as ``1``)
    distinguishes 1D from 2D without needing to read the (potentially
    large) binary FID itself just to check its shape."""
    procpar = varian_binary.read_procpar(os.path.join(path, "procpar"))
    ni = _procpar_value(procpar, "ni", default=1, cast=int)
    if ni and ni > 1:
        return read_2d(path, **kwargs)
    kwargs.pop("drift_correction", None)      # a 1D has no series to correct
    return read_1d(path, **kwargs)


def peek(path: str) -> dict:
    """Metadata-only probe: shape/nucleus/dimensionality from
    ``procpar`` alone, never touching the binary ``fid``.

    Without this, :func:`opennmr.io.base.peek` falls back to a full
    ``load()`` -- which for the Data-tab folder browser meant reading
    and *processing* (apodize, FFT, autophase, baseline) every
    experiment in a folder just to print "7Li, 1D, 1000". Measured at
    ~28 ms per experiment against ~1 ms here.

    Shape is derived the way the readers derive it, and a test pins
    that agreement (``test_peek_agrees_with_load_on_every_dataset``):
    ``np`` counts real+imaginary points, so the acquired dimension is
    ``np // 2``, and ``ni`` is the indirect increment count -- the
    same parameter :func:`_read_auto` already uses to choose between
    :func:`read_1d` and :func:`read_2d` without reading the FID.
    """
    procpar = varian_binary.read_procpar(os.path.join(path, "procpar"))
    np_points = _procpar_value(procpar, "np", default=0, cast=int) or 0
    ni = _procpar_value(procpar, "ni", default=1, cast=int) or 1
    nucleus = _normalize_nucleus(_procpar_value(procpar, "tn", default=None, cast=str))
    # Same source read_1d/read_2d name as the pulse programme.
    pulse_program = _procpar_value(procpar, "seqfil", default=None, cast=str)
    fast = int(np_points) // 2
    if ni > 1:
        # read_2d always builds an index (not frequency) indirect axis
        # -- an arrayed Varian 2D is treated as a pseudo-2D, see its
        # own `pseudo_2d=True`. Reported here to match.
        return {
            "vendor": "varian", "ndim": 2, "shape": (int(ni), fast),
            "nucleus": nucleus,
            "nucleus_indirect": _normalize_nucleus(
                _procpar_value(procpar, "dn", default=None, cast=str)),
            "pseudo_2d": True,
            "pulse_program": pulse_program,
        }
    return {
        "vendor": "varian", "ndim": 1, "shape": (fast,),
        "nucleus": nucleus, "nucleus_indirect": None, "pseudo_2d": False,
        "pulse_program": pulse_program,
    }


base.register("varian", _detect, _read_auto, peek=peek)
