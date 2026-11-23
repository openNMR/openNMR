"""
TecMag (TNMR / Orchestra) importer: ``.tnt`` files and the "Data Point
Listing" text files TNMR exports.

Written from the published TNMR file layout ("A1 - TNMR File Format",
distributed with TNMR) with numpy only -- no other package is needed to read
the files.

What is read
------------
``.tnt`` (binary)
    ``TNT1.nnn`` magic, then tagged sections. ``TMAG`` (1024 bytes: the
    acquisition parameters) and ``DATA`` (complex64, Fortran order, shape
    ``actual_npts``) are used; the rest (``TMG2`` display settings, ``PSEQ``
    the pulse programme) is not needed and is skipped. ``npts[0]`` is the
    direct dimension; ``npts[1] > 1`` makes the file a series of FIDs, read
    as a **pseudo-2D** (the indirect axis is the record number: the file
    holds no delay or gradient list a reader could trust -- give those in
    Dynamics, as for any other series). Three or four dimensions are not
    read (the error says so).

``.txt`` (TNMR *Data Point Listing*)
    ``Data Point Listing of File ...``, ``Starting Point: 1 Ending Point: N``
    (and, for a series, ``Starting 2D Record: 1 Ending Record: M``), a
    ``Real  Imag  usec`` header, then one line per point, the records one
    after the other. The listing has **no observe frequency, nucleus or
    spectral width other than the time column**: the dwell time comes from
    the time column, and ppm needs ``nucleus`` with ``sfo1`` (MHz) or
    ``field`` (see :func:`opennmr.io.simpson.resolve_frequency`); without
    them the axis is in Hz and says so.

Both are **raw FIDs**, processed by :mod:`opennmr.processing.ft` exactly like
any measured FID (window, zero-fill, truncation, linear prediction, phase)
and re-processed by the same Processing controls.

Conventions to know
-------------------
* The spectrum is the standard FFT of the FID as stored. If lines come out
  mirrored against TNMR's own display, pass ``conjugate=True`` (the
  receiver convention is not recorded in the file in a form that can be
  relied on).
* TNMR's digital-receiver group delay (``nDigitalCenter``) is **not**
  removed; use the FID shift control if a first-order phase remains.
* The observe frequency is ``ob_freq[0]`` (MHz), the nucleus ``nuclei`` for
  the observe channel (``"H1"`` is read as ``"1H"``), the spectral width
  ``sw[0]`` (Hz). ``reference_ppm`` is the shift the carrier stands for (0
  unless you know otherwise).
"""

from __future__ import annotations

import os
import re

import numpy as np

from . import base
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from ..processing import ft as processing_ft

RAW_KEY = "tecmag"

_MAGIC_RE = re.compile(rb"^TNT1\.\d\d\d$")
_LISTING_HEAD = "Data Point Listing of File"

_TLV = np.dtype([("tag", "S4"), ("flag", "<u4"), ("length", "<u4")])

#: ``TMAG`` section: the acquisition parameters (the "space" fields are padding).
_TMAG = np.dtype([
    ("npts", "<i4", 4), ("actual_npts", "<i4", 4), ("acq_points", "<i4"),
    ("npts_start", "<i4", 4), ("scans", "<i4"), ("actual_scans", "<i4"),
    ("dummy_scans", "<i4"), ("repeat_times", "<i4"), ("sadimension", "<i4"),
    ("samode", "<i4"),
    ("magnet_field", "<f8"), ("ob_freq", "<f8", 4), ("base_freq", "<f8", 4),
    ("offset_freq", "<f8", 4), ("ref_freq", "<f8"), ("NMR_frequency", "<f8"),
    ("obs_channel", "<i2"), ("space2", "S42"),
    ("sw", "<f8", 4), ("dwell", "<f8", 4), ("filter", "<f8"),
    ("experiment_time", "<f8"), ("acq_time", "<f8"), ("last_delay", "<f8"),
    ("spectrum_direction", "<i2"), ("hardware_sideband", "<i2"), ("Taps", "<i2"),
    ("Type", "<i2"), ("bDigRec", "<u4"), ("nDigitalCenter", "<i4"),
    ("space3", "S16"),
    ("transmitter_gain", "<i2"), ("receiver_gain", "<i2"),
    ("NumberOfReceivers", "<i2"), ("RG2", "<i2"), ("receiver_phase", "<f8"),
    ("space4", "S4"),
    ("set_spin_rate", "<u2"), ("actual_spin_rate", "<u2"),
    ("lock_field", "<i2"), ("lock_power", "<i2"), ("lock_gain", "<i2"),
    ("lock_phase", "<i2"), ("lock_freq_mhz", "<f8"), ("lock_ppm", "<f8"),
    ("H2O_freq_ref", "<f8"), ("space5", "S16"),
    ("set_temperature", "<f8"), ("actual_temperature", "<f8"),
    ("shim_units", "<f8"), ("shims", "<i2", 36), ("shim_FWHM", "<f8"),
    ("HH_dcpl_attn", "<i2"), ("DF_DN", "<i2"), ("F1_tran_mode", "<i2", 7),
    ("dec_BW", "<i2"), ("grd_orientation", "S4"), ("LatchLP", "<i4"),
    ("grd_Theta", "<f8"), ("grd_Phi", "<f8"), ("space6", "S264"),
    ("start_time", "<u4"), ("finish_time", "<u4"), ("elapsed_time", "<i4"),
    ("date", "S32"), ("nuclei", "S16", 4), ("sequence", "S32"),
    ("lock_solvent", "S16"), ("lock_nucleus", "S16"),
])


# ---------------------------------------------------------------------------
# Recognising the files
# ---------------------------------------------------------------------------

def _head(path: str, n: int) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(n)
    except OSError:
        return b""


def is_tnt(path: str) -> bool:
    """A TNMR binary file: ``TNT1.nnn`` at the start."""
    return os.path.isfile(path) and bool(_MAGIC_RE.match(_head(path, 8)))


def is_listing(path: str) -> bool:
    """A TNMR "Data Point Listing" text file."""
    return (os.path.isfile(path)
            and _head(path, len(_LISTING_HEAD)).decode("latin1") == _LISTING_HEAD)


def _detect(path: str) -> bool:
    return is_tnt(path) or is_listing(path)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _nucleus(text) -> str | None:
    """``"H1"`` -> ``"1H"``, ``"Li7"`` -> ``"7Li"``; ``"1H"`` and anything not
    of that form are returned as they are (``None`` for an empty name)."""
    if isinstance(text, bytes):
        text = text.decode("latin1")
    text = str(text).strip().strip("\x00")
    if not text:
        return None
    match = re.fullmatch(r"([A-Za-z]{1,2})(\d{1,3})", text)
    if match:
        return f"{match.group(2)}{match.group(1)[0].upper()}{match.group(1)[1:].lower()}"
    return text


def parse_tnt(path: str) -> dict:
    """``{"params": {...TMAG...}, "data": complex array shaped like
    ``actual_npts`` (trailing length-1 dimensions dropped, at least 1D),
    "version": "TNT1.008"}``."""
    with open(path, "rb") as handle:
        blob = handle.read()
    if not _MAGIC_RE.match(blob[:8]):
        raise ValueError(f"{path} is not a TNMR (.tnt) file (no 'TNT1.nnn' at the start).")
    sections: dict[str, tuple[int, int]] = {}
    position = 8
    while position + _TLV.itemsize <= len(blob):
        tag, _flag, length = np.frombuffer(blob, _TLV, count=1, offset=position)[0]
        position += _TLV.itemsize
        if int(length) > len(blob) - position:
            break         # past the end: the pulse-programme section's length field is not one
        sections[tag.decode("latin1")] = (position, int(length))
        position += int(length)
    for needed in ("TMAG", "DATA"):
        if needed not in sections:
            raise ValueError(f"{path}: the {needed} section is missing; not a readable TNMR file.")
    offset, length = sections["TMAG"]
    if length != _TMAG.itemsize:
        raise ValueError(f"{path}: TMAG section is {length} bytes, expected {_TMAG.itemsize}.")
    tmag = np.frombuffer(blob, _TMAG, count=1, offset=offset)[0]
    params = {name: tmag[name] for name in _TMAG.names if not name.startswith("space")}
    shape = [int(n) for n in params["actual_npts"]]
    count = int(np.prod(shape))
    offset, length = sections["DATA"]
    if length != count * 8:
        raise ValueError(f"{path}: DATA holds {length // 8} complex points, but the header says "
                         f"{shape} = {count}.")
    data = np.frombuffer(blob, "<c8", count=count, offset=offset).astype(np.complex128)
    data = data.reshape(shape, order="F")
    while data.ndim > 1 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 2:
        # Stored with the direct dimension fastest: (points, records) -> (records, points).
        data = np.ascontiguousarray(data.T)
    return {"params": params, "data": data, "version": blob[:8].decode("ascii")}


def parse_listing(path: str) -> dict:
    """``{"data": complex array (points,) or (records, points), "dt": s,
    "title": the listing's own file line}`` from a Data Point Listing."""
    with open(path, "r", encoding="latin1") as handle:
        lines = handle.read().splitlines()
    if not lines or not lines[0].startswith(_LISTING_HEAD):
        raise ValueError(f"{path} is not a TNMR Data Point Listing.")
    points = records = None
    start = None
    for number, line in enumerate(lines[:12]):
        text = line.strip()
        match = re.match(r"Starting Point:\s*(\d+)\s+Ending Point:\s*(\d+)", text)
        if match:
            points = int(match.group(2)) - int(match.group(1)) + 1
        match = re.match(r"Starting 2D Record:\s*(\d+)\s+Ending Record:\s*(\d+)", text)
        if match:
            records = int(match.group(2)) - int(match.group(1)) + 1
        if text.lower().startswith("real"):
            start = number + 1
            break
    if start is None or points is None:
        raise ValueError(f"{path}: no 'Real Imag usec' table found in the listing.")
    columns = np.array([[float(v) for v in line.split()[:3]] for line in lines[start:] if line.strip()])
    if columns.ndim != 2 or columns.shape[1] < 3:
        raise ValueError(f"{path}: the table needs three columns (real, imaginary, time in usec).")
    records = records or 1
    if columns.shape[0] != points * records:
        raise ValueError(f"{path}: {columns.shape[0]} data lines, but the header says "
                         f"{points} points x {records} record(s).")
    time_us = columns[:points, 2]
    if points < 2 or not np.all(np.diff(time_us) > 0):
        raise ValueError(f"{path}: the time column does not increase; the dwell time is unknown.")
    steps = np.diff(time_us)
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
        raise ValueError(f"{path}: the time column is not evenly spaced.")
    data = (columns[:, 0] + 1j * columns[:, 1]).reshape(records, points)
    return {"data": data[0] if records == 1 else data, "dt": float(steps[0]) * 1e-6,
            "title": lines[0][len(_LISTING_HEAD):].strip()}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _row_spectrum(row, dt, *, window, window_kwargs, zero_fill, truncate_points, fid_shift,
                  align_echo, echo_offset, lp_points, lp_order, lp_direction):
    """One FID through the shared pipeline: ``(processed fid, n_acquired)``."""
    prepared = processing_ft.prepare_fid(
        row, fid_shift=fid_shift, align_echo_enabled=align_echo, echo_offset=echo_offset,
        truncate=truncate_points, linear_predict_points=lp_points,
        linear_predict_order=lp_order, linear_predict_direction=lp_direction)
    n_acquired = len(prepared)
    prepared = processing_ft.apodize(prepared, dt, window=window, **(window_kwargs or {}))
    if zero_fill is True:
        target = 1 << max(int(np.ceil(np.log2(max(len(prepared), 1)))), 0)
    elif isinstance(zero_fill, int) and not isinstance(zero_fill, bool):
        target = zero_fill
    else:
        target = None
    if target is not None and target > len(prepared):
        prepared = processing_ft.zero_fill(prepared, target)
    elif target is None:
        prepared = prepared[:max(n_acquired, 1)]
    return prepared


def read(path: str, *, nucleus: str | None = None, field=None, sfo1: float | None = None,
         reference_ppm: float = 0.0, conjugate: bool = False,
         window: str = "exponential", window_kwargs: dict | None = None,
         zero_fill: int | bool | None = None, truncate: int | bool | None = None,
         fid_shift: float = 0.0, align_echo: bool = False, echo_offset: float = 0.0,
         linear_predict_points: int = 0, linear_predict_order: int = 20,
         linear_predict_direction: str = "forward",
         autophase: bool = True, phase_mode: str = "apc0",
         phase0: float = 0.0, phase1: float = 0.0, pivot_frac: float = 0.5,
         phase0_inc: float = 0.1, phase1_max: float = 20.0, phase1_improvement: float = 0.15,
         phase_penalty: str = "magnitude", mass: float = 1.0,
         **_ignored) -> NMRSpectrum:
    """A TecMag ``.tnt`` file or Data Point Listing as an :class:`NMRSpectrum`
    -- 1D, or a pseudo-2D when it holds several FIDs. See the module
    docstring. ``nucleus`` / ``sfo1`` / ``field`` are only needed for a
    listing (a ``.tnt`` carries them; giving them overrides the file)."""
    from .simpson import resolve_frequency, _frequency_axis
    path = os.fspath(path)
    params = None
    if is_tnt(path):
        parsed = parse_tnt(path)
        params = parsed["params"]
        data = parsed["data"]
        if data.ndim > 2:
            raise ValueError(f"{path}: a {data.ndim}-dimensional TNMR file is not read; "
                             "only 1D and series of FIDs (2D) are.")
        dwell = float(params["dwell"][0])
        sw_hz = float(params["sw"][0])
        dt = dwell if dwell > 0 else (1.0 / sw_hz if sw_hz > 0 else 0.0)
        if dt <= 0:
            raise ValueError(f"{path}: no dwell time or spectral width in the header.")
        index = int(params["obs_channel"]) - 1
        names = [_nucleus(n) for n in params["nuclei"]]
        file_nucleus = names[index] if 0 <= index < len(names) and names[index] else names[0]
        file_sfo1 = float(params["ob_freq"][0]) or None
        title = os.path.basename(path)
        kind = "tnt"
    elif is_listing(path):
        parsed = parse_listing(path)
        data, dt = parsed["data"], parsed["dt"]
        file_nucleus = file_sfo1 = None
        title = os.path.basename(path)
        kind = "listing"
    else:
        raise ValueError(f"{path} is neither a TNMR .tnt file nor a Data Point Listing.")

    series = data.ndim == 2
    rows = np.atleast_2d(data)                       # (records, points)
    nucleus = nucleus or file_nucleus
    observe, _b0 = resolve_frequency(nucleus, field, sfo1 if sfo1 else file_sfo1)
    fid_raw = np.conj(rows) if conjugate else rows
    fid_raw = np.asarray(fid_raw, dtype=np.complex128) / (mass or 1.0)
    original_length = fid_raw.shape[1]

    if truncate is True:
        truncate_points = None
    elif isinstance(truncate, int) and not isinstance(truncate, bool):
        truncate_points = truncate
    else:
        truncate_points = None
    processed = [_row_spectrum(
        row, dt, window=window, window_kwargs=window_kwargs, zero_fill=zero_fill,
        truncate_points=truncate_points, fid_shift=fid_shift, align_echo=align_echo,
        echo_offset=echo_offset, lp_points=linear_predict_points, lp_order=linear_predict_order,
        lp_direction=linear_predict_direction) for row in fid_raw]
    fids = np.array(processed)
    spectra = []
    freq_hz = None
    for row in fids:
        freq_hz, spec_row = processing_ft.fft_spectrum(row, dt)
        spectra.append(spec_row)
    spectrum = np.array(spectra)
    ppm, hz, used, label = _frequency_axis(freq_hz, observe, reference_ppm)

    autophased = bool(autophase)
    if autophase:
        summed = spectrum.sum(axis=0)
        if phase_mode == "joint":
            phase0, phase1 = processing_ft.autophase01(
                summed, ppm, phase0_range=(-180, 180), phase0_inc=phase0_inc,
                phase1_max=phase1_max, phase1_improvement=phase1_improvement,
                penalty=phase_penalty, pivot_frac=0.5)
        elif phase_mode == "apc0":
            phase0 = processing_ft.apc0(summed, ppm, phase0_inc)
        else:
            raise ValueError("phase_mode must be 'apc0' or 'joint'")
    spectrum = np.array([processing_ft.apply_phase(row, ppm, phase0, phase1, pivot_frac=pivot_frac)
                         for row in spectrum])

    from .xpulse import _phase_metadata
    asked = {"format": kind, "nucleus": nucleus, "sfo1": observe, "reference_ppm": reference_ppm,
             "conjugate": bool(conjugate), "records": int(rows.shape[0]),
             "dwell_time_s": dt, "spectral_width_Hz": 1.0 / dt}
    if params is not None:
        asked.update({"tnmr_version": parsed["version"], "scans": int(params["scans"]),
                      "sequence": params["sequence"].decode("latin1").strip(),
                      "date": params["date"].decode("latin1").strip(),
                      "temperature_K": float(params["actual_temperature"]),
                      "magnet_field_T": float(params["magnet_field"]),
                      "receiver_gain": int(params["receiver_gain"]),
                      "acq_time_s": float(params["acq_time"]),
                      "last_delay_s": float(params["last_delay"])})
    metadata = NMRMetadata(
        vendor="tecmag", nucleus=nucleus, sfo1=used, bf1=used,
        o1=(reference_ppm * observe) if observe else 0.0, ns=None, source_path=path,
        pseudo_2d=series,
        pulse_program=(asked.get("sequence") or None),
        raw={RAW_KEY: asked, "axis_is_hz": observe is None, "title": title,
             **_phase_metadata(phase0, phase1, autophased, pivot_frac),
             "fid": fids if series else fids[0],
             "fid_raw": fid_raw if series else fid_raw[0],
             "fid_dt": dt, "fid_original_length": original_length})
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=hz, label=label)
    if not series:
        return NMRSpectrum(data=np.real(spectrum[0]), axes=[axis], metadata=metadata,
                           data_imag=np.imag(spectrum[0]))
    y_axis = NMRAxis(kind="index", index=np.arange(1, spectrum.shape[0] + 1))
    return NMRSpectrum(data=np.real(spectrum), axes=[y_axis, axis], metadata=metadata,
                       data_imag=np.imag(spectrum))


def peek(path: str, **_ignored) -> dict:
    """Shape and nucleus from the header (a ``.tnt``) or the listing's header lines."""
    if is_tnt(path):
        parsed = parse_tnt(path)
        params, shape = parsed["params"], parsed["data"].shape
        index = int(params["obs_channel"]) - 1
        names = [_nucleus(n) for n in params["nuclei"]]
        nucleus = names[index] if 0 <= index < len(names) and names[index] else names[0]
    else:
        listing = parse_listing(path)
        shape, nucleus = listing["data"].shape, None
    return {"vendor": "tecmag", "ndim": len(shape), "shape": tuple(shape), "nucleus": nucleus,
            "nucleus_indirect": None, "pseudo_2d": len(shape) == 2}


base.register("tecmag", _detect, read, peek=peek)
