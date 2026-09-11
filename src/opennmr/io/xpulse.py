"""
X-Pulse importer.

Unlike Bruker/TopSpin, X-Pulse hands back a raw time-domain FID encoded
in a JCAMP-DX file (ASDF/AFFN pseudo-digit compressed data table) --
apodization, zero-filling, FFT, phasing and baseline correction all
happen here in Python, via :mod:`opennmr.processing.ft`.

The ASDF/AFFN decoding in :func:`_read_xpulse_jcamp` is ported close to
unchanged from ``leonmr.read_xpulse_jcamp`` -- it's intricate
pseudo-digit decoding with a documented, hard-won regex fix (see the
comment on ``time_re`` below), not worth rewriting from scratch.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import re
from warnings import warn

import numpy as np
from scipy.interpolate import interp1d

from . import base
from . import cloud
from ..processing import ft as processing_ft
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum


# ============================================================
# ASDF/AFFN JCAMP-DX decoding (ported from leonmr.read_xpulse_jcamp)
# ============================================================

#: ASDF squeeze/diff/dup pseudo-digit alphabet (JCAMP-DX 6.0's own
#: compressed-data-table encoding) -- shared by every X-Pulse reader
#: below (the plain-1D two-DATATABLE format and the NTUPLES/PAGE 2D
#: format both use it identically per data line, just organized into a
#: different overall file structure).
_ASDF_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
_ASDF_SQZ_DIGITS = {
    "@": "0", "A": "1", "B": "2", "C": "3", "D": "4", "E": "5",
    "F": "6", "G": "7", "H": "8", "I": "9",
    "a": "-1", "b": "-2", "c": "-3", "d": "-4", "e": "-5",
    "f": "-6", "g": "-7", "h": "-8", "i": "-9",
}
_ASDF_DIF_DIGITS = {
    "%": "0", "J": "1", "K": "2", "L": "3", "M": "4", "N": "5",
    "O": "6", "P": "7", "Q": "8", "R": "9",
    "j": "-1", "k": "-2", "l": "-3", "m": "-4", "n": "-5",
    "o": "-6", "p": "-7", "q": "-8", "r": "-9",
}
_ASDF_DUP_DIGITS = {
    "S": "1", "T": "2", "U": "3", "V": "4", "W": "5",
    "X": "6", "Y": "7", "Z": "8", "s": "9",
}


def _asdf_append_value(data, value_to_append, isdif):
    if isdif and data:
        data.append(data[-1] + value_to_append)
    else:
        data.append(value_to_append)


def _asdf_finish_value(valuestr, currentmode, prev_value_to_append, data):
    try:
        if currentmode == 1:  # squeeze
            if prev_value_to_append is not None:
                _asdf_append_value(data, *prev_value_to_append)
            return (float(valuestr), False), True
        elif currentmode == 2:  # diff
            if prev_value_to_append is not None:
                _asdf_append_value(data, *prev_value_to_append)
            return (float(valuestr), True), True
        elif currentmode == 3:  # duplicate
            if prev_value_to_append is None:
                warn("Parse error: DUP entry without preceding value")
                return None, False
            dupcount = int(valuestr)
            for _ in range(dupcount):
                _asdf_append_value(data, *prev_value_to_append)
            return None, True
        else:  # mode 0 (first)
            return None, True
    except ValueError:
        return None, False


def _parse_pseudo(datalines):
    data = []
    currentmode = 0
    valuestr = []
    skip_checkpoint = False
    value_to_append = None

    for dataline in datalines:
        if not dataline:
            continue
        first_of_line = True
        for char in dataline.strip():
            if char in _ASDF_DIGITS:
                valuestr.append(char)
                continue
            try:
                valuechar = _ASDF_SQZ_DIGITS[char]
                newmode = 1
            except KeyError:
                try:
                    valuechar = _ASDF_DIF_DIGITS[char]
                    newmode = 2
                except KeyError:
                    try:
                        valuechar = _ASDF_DUP_DIGITS[char]
                        newmode = 3
                    except KeyError:
                        warn(f"Unknown pseudo-digit: {char}")
                        return None
            valuestr = "".join(valuestr)
            previous_is_dif = currentmode == 2 or (currentmode == 3 and value_to_append and value_to_append[1])
            if not skip_checkpoint:
                value_to_append, success = _asdf_finish_value(valuestr, currentmode, value_to_append, data)
                if not success:
                    warn("Data parsing failed")
                    return None
            skip_checkpoint = first_of_line and previous_is_dif
            currentmode = newmode
            valuestr = [valuechar]
            first_of_line = False
    if not skip_checkpoint:
        valuestr = "".join(valuestr)
        value_to_append, success = _asdf_finish_value(valuestr, currentmode, value_to_append, data)
        if not success:
            warn("Data parsing failed at last")
            return None
    if value_to_append is not None:
        _asdf_append_value(data, *value_to_append)
    return data


def _parse_affn_pac(datalines):
    value_re = re.compile(r"([+-]?\d+\.?\d*|[+-]?\.\d+)([eE][+-]?\d+)?")
    data = []
    for dataline in datalines:
        if not dataline.strip():
            continue
        for match in value_re.finditer(dataline):
            base_str = match.group(1)
            exp = match.group(2)
            try:
                value = float(base_str + (exp if exp is not None else ""))
            except ValueError:
                warn(f"Data parsing failed at line: {dataline}")
                return None
            data.append(value)
    return data


def _parse_data(datastring, varform):
    datalines = datastring.split("\n")
    if not datalines:
        return None
    varform = varform.strip().upper()
    if varform == "ASDF":
        data = _parse_pseudo(datalines)
    elif varform == "AFFN":
        data = _parse_affn_pac(datalines)
    else:
        warn(f"Unknown VARFORM: {varform}, trying ASDF")
        data = _parse_pseudo(datalines)
        if data is None:
            data = _parse_affn_pac(datalines)
    if data is None:
        return None
    return np.asarray(data, dtype="float64"), "R"


def _read_xpulse_jcamp(filepath: str, verbose: bool = False, timegrid_warning: bool = False) -> dict:
    """Parse an X-Pulse JCAMP-DX file into a raw complex FID plus the
    instrument/experiment parameter dicts it carries.

    Returns a dict with keys ``"complex"``, ``"real"``, ``"imag"``,
    ``"time"``, ``"params"`` (instrument parameters, ``##$KEY=value``
    lines), ``"exp_params"`` (experiment parameters from the
    ``$$SPINFLOW`` block).
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    params: dict = {}
    exp_params: dict = {}
    page_data: dict = {}
    current_page = None
    in_datatable = False
    in_spinflow_block = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("$$"):
            body = line[2:].strip().lower()
            if "spinflow" in body and "start" in body:
                in_spinflow_block = True
                continue
            if "spinflow" in body and "end" in body:
                in_spinflow_block = False
                continue
            if in_spinflow_block and "=" in line:
                k, v = line[2:].split("=", 1)
                k, v = k.strip(), v.strip()
                try:
                    exp_params[k] = [float(x) for x in v.split(",")] if "," in v else float(v)
                except ValueError:
                    exp_params[k] = v
            continue
        if line.upper().startswith("##"):
            in_datatable = False
            m = re.match(r"##([^=]+)=\s*(.*)", line)
            if m:
                key, val = m.group(1).strip().upper(), m.group(2).strip()
                params[key] = val
                if key == "PAGE" and "N=" in val:
                    try:
                        current_page = int(val.split("N=")[1].split()[0])
                    except (IndexError, ValueError):
                        current_page = None
                    if current_page is not None:
                        page_data.setdefault(current_page, [])
                elif key in ("DATATABLE", "XYDATA"):
                    in_datatable = True
            continue
        if in_datatable and current_page is not None:
            page_data.setdefault(current_page, []).append(line)

    varform_str = params.get("VARFORM", "AFFN, ASDF, ASDF, AFFN")
    varforms = [v.strip() for v in varform_str.split(",")]
    varform_real = varforms[1] if len(varforms) > 1 else "ASDF"
    varform_imag = varforms[2] if len(varforms) > 2 else "ASDF"

    # "NP" is the plain-1D exp_params key; some exports (e.g. a 2D
    # sequence's per-slice raw FID, F1/F2 acquisition points recorded
    # separately) instead only record "NP2" for the direct dimension's
    # point count -- fall back to that before the hardcoded default.
    expected_points = int(exp_params.get("NP", exp_params.get("NP2", 512)))

    real_lines = page_data.get(1, [])
    imag_lines = page_data.get(2, [])

    # Regex to extract time values (first float in each line). Requires
    # the SIGN to be present on any exponent ([eE][+-]\d+, mandatory not
    # optional): ASDF's squeeze/diff/dup pseudo-digit alphabet never
    # produces a literal '+'/'-' character, so a sign-bearing exponent
    # can only be a genuine one -- never a collision with ASDF data that
    # happens to start with 'e'/'E' (a real line can read e.g.
    # "0.065856E81855k04961..." where "0.065856" is the time and
    # "E81855..." is squeeze-encoded DATA, not "0.065856E8"). An earlier,
    # optional-sign version of this regex corrupted exactly that case.
    time_re = re.compile(r"^([+-]?\d+\.?\d*|[+-]?\.\d+)([eE][+-]\d+)?")

    time_real = []
    real_data_lines = []
    for line in real_lines:
        time_match = time_re.match(line)
        if time_match:
            time_real.append(float(time_match.group(0)))
            real_data_lines.append(line[time_match.end():])
        else:
            real_data_lines.append(line)

    time_imag = []
    imag_data_lines = []
    for line in imag_lines:
        time_match = time_re.match(line)
        if time_match:
            time_imag.append(float(time_match.group(0)))
            imag_data_lines.append(line[time_match.end():])
        else:
            imag_data_lines.append(line)

    real_parsed = _parse_data("\n".join(real_data_lines), varform_real)
    imag_parsed = _parse_data("\n".join(imag_data_lines), varform_imag) if imag_data_lines else None

    if real_parsed is None:
        raise RuntimeError("Could not parse real data from page 1")

    real, _ = real_parsed
    imag, _ = imag_parsed if imag_parsed else (np.zeros(len(real)), "I")

    if len(real) != expected_points:
        warn(f"Real data points mismatch: expected {expected_points}, got {len(real)}. Truncating/padding.")
        if len(real) > expected_points:
            real = real[:expected_points]
        else:
            real = np.pad(real, (0, expected_points - len(real)), mode='constant')

    if len(imag) != expected_points:
        warn(f"Imag data points mismatch: expected {expected_points}, got {len(imag)}. Truncating/padding.")
        if len(imag) > expected_points:
            imag = imag[:expected_points]
        else:
            imag = np.pad(imag, (0, expected_points - len(imag)), mode='constant')

    def build_time_array(time_values, counts, last):
        time_full = []
        for i, t in enumerate(time_values):
            count = counts[i] if i < len(counts) else 1
            if i == len(time_values) - 1:
                next_t = last
            else:
                next_t = time_values[i + 1] if i + 1 < len(time_values) else last
            time_segment = np.linspace(t, next_t, count, endpoint=False)
            time_full.extend(time_segment)
        return np.array(time_full)

    try:
        last = float(params.get("LAST", "0").split(",")[0])
    except ValueError:
        last = max(time_real[-1], time_imag[-1]) if time_real and time_imag else 0.0

    def count_data_points(lines_, varform):
        counts = []
        for line in lines_:
            if varform == "ASDF":
                parsed = _parse_pseudo([line])
                counts.append(len(parsed) if parsed is not None else 0)
            elif varform == "AFFN":
                parsed = _parse_affn_pac([line])
                counts.append(len(parsed) if parsed is not None else 0)
            else:
                counts.append(1)
        return counts

    real_counts = count_data_points(real_data_lines, varform_real)
    imag_counts = count_data_points(imag_data_lines, varform_imag)

    time_real_full = build_time_array(time_real, real_counts, last)
    time_imag_full = build_time_array(time_imag, imag_counts, last)

    def validate_time_spacing(time_array, label):
        if len(time_array) < 2:
            warn(f"Not enough {label} time points to validate spacing.")
            return
        spacing = np.diff(time_array)
        if not np.allclose(spacing, spacing[0], rtol=1e-6) and timegrid_warning:
            warn(f"{label} time grid is not equally spaced! Max diff: {np.max(np.abs(spacing - spacing[0]))}")

    validate_time_spacing(time_real_full, "Real")
    validate_time_spacing(time_imag_full, "Imaginary")

    if len(time_real_full) != len(real):
        if timegrid_warning:
            warn(f"Real time/data length mismatch: time={len(time_real_full)}, data={len(real)}. Using uniform time grid.")
        time_real_full = np.linspace(time_real[0], last, len(real))

    if len(time_imag_full) != len(imag):
        if timegrid_warning:
            warn(f"Imag time/data length mismatch: time={len(time_imag_full)}, data={len(imag)}. Using uniform time grid.")
        time_imag_full = np.linspace(time_imag[0], last, len(imag))

    n_points = max(len(real), len(imag))
    try:
        first = float(params.get("FIRST", "0").split(",")[0])
    except ValueError:
        first = 0.0
    uniform_time = np.linspace(first, last, n_points)

    real = interp1d(time_real_full, real, kind='linear', fill_value='extrapolate')(uniform_time)
    imag = interp1d(time_imag_full, imag, kind='linear', fill_value='extrapolate')(uniform_time)

    try:
        factor_str = params.get("FACTOR", "1, 1, 1, 1")
        factors = [float(x.strip()) for x in factor_str.split(",")]
        real = real * (factors[1] if len(factors) > 1 else 1.0)
        imag = imag * (factors[2] if len(factors) > 2 else 1.0)
    except (ValueError, IndexError):
        pass

    if verbose:
        print(f"Expected points: {expected_points}, Real: {len(real)}, Imag: {len(imag)}")
        print(f"Uniform time: {uniform_time[0]:.6f} to {uniform_time[-1]:.6f} s ({len(uniform_time)} points)")

    return {
        "complex": real + 1j * imag,
        "real": real,
        "imag": imag,
        "time": uniform_time,
        "params": params,
        "exp_params": exp_params,
    }


# ============================================================
# NMRSpectrum-producing reader
# ============================================================

def _phase_metadata(phase0: float, phase1: float, autophased: bool,
                    pivot_frac: float = 0.5) -> dict:
    """Where a just-applied phase belongs in ``metadata.raw``.

    A phase the reader **computed** (autophase) is a *recorded* base:
    it describes this file as read, and every later rebuild from the
    raw FID has to reapply it before anything the user adds on top. A
    phase the **caller supplied** is already the user's own, and goes
    in the user slot so it is not double-counted.

    Keyed through :data:`opennmr.io.bruker.RECORDED_PHASE_KEYS` /
    ``USER_PHASE_KEYS`` rather than literal strings, so the two vendors
    cannot drift apart on which key means what -- the GUI reads both
    through the same accessors.

    **The bug this fixes.** X-Pulse files record no phase of their own,
    so the base is whatever autophase finds at read time -- and that
    was being stored in the *user* slot, where a saved sidecar's own
    ``phase0`` overwrote it on load. The rebuild-from-raw then applied
    the user's correction with no base under it. Measured on
    ``test_data/xpulse_PFG/4``: autophase finds -177.2 deg (real-part
    peak 0.992 of the magnitude peak); the sidecar holds the user's
    +38.33; reloading applied 38.33 alone and gave 0.118, i.e.
    essentially unphased. Recorded + user restores 0.92. Reported as
    the phase "not working on the PFG experiment", and as needing to
    "jump start it by changing the phase".
    """
    from . import bruker

    recorded0, recorded1 = bruker.RECORDED_PHASE_KEYS["f2"]
    user0, user1 = bruker.USER_PHASE_KEYS["f2"]
    if autophased:
        # **Converted, not copied.** The recorded slot means TopSpin's
        # index ramp applied as exp(+i*phi); autophase returns a
        # pivot-model phase applied as exp(-i*phi). Writing the raw
        # value here made the two cancel on every rebuild from the raw
        # FID -- see bruker.recorded_phase_from_pivot_model for the
        # measurement.
        phc0, phc1 = bruker.recorded_phase_from_pivot_model(phase0, phase1, pivot_frac)
        return {recorded0: phc0, recorded1: phc1, user0: 0.0, user1: 0.0}
    return {recorded0: 0.0, recorded1: 0.0, user0: float(phase0), user1: float(phase1)}


def read_1d(path: str, *, window: str = "exponential", window_kwargs: dict | None = None,
            zero_fill: int | bool | None = None, truncate: int | bool | None = None,
            fid_shift: float = 0.0, align_echo: bool = False, echo_offset: float = 0.0,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward",
            baseline_correct: bool | None = None, normalize: bool = False,
            autophase: bool = True, phase_mode: str = "apc0",
            phase0: float = 0.0, phase1: float = 0.0, pivot_frac: float = 0.5, phase0_inc: float = 0.1,
            phase1_max: float = 20.0, phase1_improvement: float = 0.15,
            phase_penalty: str = "magnitude", verbose: bool = False,
            timegrid_warning: bool = False) -> NMRSpectrum:
    """Import an X-Pulse JCAMP-DX FID and process it (apodize,
    zero-fill, FFT, phase, baseline-correct) into an
    :class:`NMRSpectrum`.

    ``window``/``window_kwargs`` : passed to
    :func:`opennmr.processing.ft.apodize` (default: exponential with
    no line broadening, i.e. no-op -- pass e.g.
    ``window_kwargs={"lb": 5.0}`` for 5 Hz line broadening, or
    ``window="hann"`` for a Hann window instead).

    ``truncate``/``zero_fill`` : ``True`` uses the value recorded in
    the file (``TDEFF``/``SI``); an ``int`` sets it explicitly; ``None``
    (default) does neither -- except when ``align_echo=True``, which
    always caps the effective truncation at the echo-top's own wrap
    point (see ``align_echo`` below), regardless of what ``truncate``
    (or its absence) would otherwise imply.

    ``fid_shift``/``align_echo`` run first (before ``truncate``, not
    after -- see the code comment at the truncate step for why order
    matters for a full-echo acquisition), then ``truncate``, then
    ``linear_predict_points``, all before apodization -- see
    :mod:`opennmr.processing.ft`'s ``roll_fid``/``align_echo``/
    ``linear_predict`` for what each does. Useful for full-echo
    (rather than one-sided decaying) acquisitions, common on
    wideline/quadrupolar X-Pulse experiments. ``align_echo`` also caps
    ``truncate`` at the echo top's own wrap point (``len(fid) -
    ceil(echo_top)``) -- neither an unspecified ``truncate`` nor the
    file's own recorded ``TDEFF`` reliably stop short of that point on
    their own, which otherwise lets the wrapped pre-echo portion (a
    mirror image of the post-echo decay, for a symmetric echo) back in
    and produces an FID that "looks like a time-reversal" once FFT'd.
    Pass an explicit ``truncate`` shorter than that if an even shorter
    effective FID is wanted; a longer one is silently capped rather
    than honoured, since including any part of the wrap is essentially
    never desired.

    ``baseline_correct`` : ``None`` (default) picks ``not align_echo``
    -- ported from ``leonmr``'s ``xpulse_image_import``, which forces
    this off when echo-aligning: a straight-line baseline fit is a
    poor model for the top-hat-like profiles that come out of an
    echo-aligned gradient-echo/imaging acquisition (confirmed on real
    ``test_data/xpulse_MRI`` data -- leaving the default linear
    baseline-correct *on* alongside ``align_echo=True`` measurably
    worsens the result, from 52% negative-going points with neither
    correction to 81% with both -- 41% with ``align_echo`` alone).
    Pass ``True``/``False`` explicitly to override either way.

    ``phase_mode`` : ``"apc0"`` (default, zero-order-only autophase) or
    ``"joint"`` (:func:`opennmr.processing.ft.autophase01`, zero- and
    first-order). Only used when ``autophase=True``; ignored otherwise
    (``phase0``/``phase1`` are then applied as given).
    """

    if baseline_correct is None:
        baseline_correct = not align_echo

    fid_data = _read_xpulse_jcamp(path, verbose=verbose, timegrid_warning=timegrid_warning)
    fid = np.asarray(fid_data["complex"], dtype=np.complex128)
    fid_raw = fid  # the truly raw FID -- see metadata.raw["fid_raw"]'s own comment below
    original_length = len(fid)  # NP: acquired points, before truncate/shift/align/LP/window/zero-fill
    params = fid_data["params"]
    exp_params = fid_data["exp_params"]

    np_points = int(exp_params.get("NP", exp_params.get("NP2", len(fid))))
    sw = float(exp_params.get("Filter", params.get("$SWEEPWIDTH", 0.0)))
    if sw <= 0:
        raise ValueError("Spectral width (sw) must be > 0 to compute dt")
    dt = 1.0 / sw
    sfo1_hz = float(params.get("$SFO1", 1.0)) * 1e6
    sf_hz = float(params.get("$SF", 1.0)) * 1e6
    o1_hz = float(params.get("$O1", 1.0))

    # is True checked ahead of isinstance(..., int): bool is an int
    # subclass in Python, so isinstance(True, int) is True -- checking
    # isinstance() first would make the "use the recorded TDEFF/SI"
    # branch below unreachable whenever truncate/zero_fill=True is
    # passed.
    if truncate is True:
        tdeff_points = int(exp_params.get("TDEFF", len(fid)))
    elif isinstance(truncate, int):
        tdeff_points = truncate
    else:
        tdeff_points = None

    # Shift -> echo-align -> truncate (capped at the echo-wrap point)
    # -> linear-predict, in that exact order -- see
    # processing.ft.prepare_fid's own docstring for why the order (and
    # the wrap-point cap specifically) matters; this used to be inline
    # here, now shared with the GUI's own live Processing-preview so
    # there's exactly one implementation of it.
    fid = processing_ft.prepare_fid(
        fid, fid_shift=fid_shift, align_echo_enabled=align_echo, echo_offset=echo_offset,
        truncate=tdeff_points,
        linear_predict_points=linear_predict_points, linear_predict_order=linear_predict_order,
        linear_predict_direction=linear_predict_direction,
    )

    if linear_predict_points:
        # The truncate-back-to-np_points fallback below exists to undo
        # zero_fill's own padding when no zero_fill was requested -- it
        # must not also undo the extension linear prediction just added.
        np_points = len(fid)

    fid = processing_ft.apodize(fid, dt, window=window, **(window_kwargs or {}))

    if zero_fill is True:
        si_points = int(exp_params.get("SI", np_points))
    elif isinstance(zero_fill, int):
        si_points = zero_fill
    else:
        si_points = None
    if si_points is not None and si_points > len(fid):
        fid = processing_ft.zero_fill(fid, si_points)
    elif np_points is not None:
        fid = fid[:np_points]

    freq_hz, spectrum = processing_ft.fft_spectrum(fid, dt)
    freq_ppm = (freq_hz - o1_hz - (sf_hz - sfo1_hz)) / sfo1_hz * 1e6

    # Whether the phase about to be applied was *computed here* or
    # supplied by the caller -- see _phase_metadata.
    autophased = bool(autophase)
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

    if normalize:
        maxval = np.max(np.abs(spectrum))
        if maxval > 0:
            spectrum = spectrum / maxval

    nucleus = re.sub(r"\W+", "", params.get(".OBSERVENUCLEUS", "") or "") or None

    metadata = NMRMetadata(
        vendor="xpulse",
        nucleus=nucleus,
        sfo1=sfo1_hz / 1e6,
        bf1=sf_hz / 1e6,
        o1=o1_hz,
        source_path=path,
        probe=exp_params.get("Probe"), pulse_program=params.get(".PULSESEQUENCE"),
        raw={
            "params": params, "exp_params": exp_params,
            # Recorded (autophase's own base) vs user phase -- see
            # _phase_metadata for why the split matters on reload.
            **_phase_metadata(phase0, phase1, autophased, pivot_frac),
            # The fully processed (pre-FFT) FID -- an optional GUI
            # preview of what FID-domain processing actually did.
            "fid": fid, "fid_dt": dt, "fid_original_length": original_length,
            # The *truly* raw FID -- straight out of the file, before
            # even fid_shift/align_echo/truncate/linear-predict, let
            # alone windowing/zero-fill. Lets the GUI recompute a live
            # preview of a not-yet-applied processing choice entirely
            # in memory (via processing.ft.prepare_fid + apodize/
            # zero_fill/fft_spectrum/apply_phase, the exact same steps
            # this function itself just ran) -- unlike "fid" above,
            # which already has whatever was chosen at *this* load
            # baked in, so isn't safe to reprocess with a *different*
            # choice without double-applying the old one.
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
# Genuine 2D (NTUPLES-format JCAMP-DX, e.g. HSQC): a
# ``##DATACLASS= NTUPLES``, ``##NUMDIM= 2`` file with one PAGE-pair
# per t1 (F1) increment -- a "(T2++(R..R))" real page immediately
# followed by a "(T2++(I..I))" imaginary page, both sharing the same
# recorded ``T1=`` value. Distinct from read_array's arrayN/-subfolder
# assembly (always pseudo-2D, no true F1 FFT) -- this is a genuinely
# hypercomplex-encoded 2D dataset.
# ============================================================

#: Any letter from the ASDF squeeze/diff/dup pseudo-digit alphabet,
#: excluding e/E (also legitimate in plain AFFN scientific notation,
#: e.g. "1.23e-4") -- genuine AFFN decimal text never contains any
#: other letter, so this cleanly distinguishes the two by content.
_ASDF_LETTER_RE = re.compile(r"[A-DF-Za-df-z@%]")


def _detect_xpulse_datatable_varform(lines: list[str]) -> str:
    """"ASDF" if any line contains a pseudo-digit letter, else "AFFN"
    -- used instead of trusting a NTUPLES file's own declared VARFORM,
    confirmed unreliable for the R/I columns on a real export (see
    :func:`_read_xpulse_ntuples_2d`'s own docstring)."""
    return "ASDF" if any(_ASDF_LETTER_RE.search(line) for line in lines) else "AFFN"


def _parse_xpulse_timegrid_block(lines: list[str], varform: str, t_first: float, t_last: float,
                                  timegrid_warning: bool = False) -> np.ndarray:
    """Parse one time-checkpoint-prefixed, ASDF/AFFN pseudo-digit
    DATATABLE block into a real-valued array on a uniform time grid
    spanning ``[t_first, t_last]`` -- the same reconstruction
    :func:`_read_xpulse_jcamp` already does for a plain 1D file's own
    two DATATABLE blocks (each data line there starts with a time
    checkpoint too), generalized to work on one block at a time so
    :func:`_read_xpulse_ntuples_2d` can call it once per PAGE."""
    time_re = re.compile(r"^([+-]?\d+\.?\d*|[+-]?\.\d+)([eE][+-]\d+)?")
    times: list[float] = []
    data_lines: list[str] = []
    for line in lines:
        m = time_re.match(line)
        if m:
            times.append(float(m.group(0)))
            data_lines.append(line[m.end():])
        else:
            data_lines.append(line)

    parsed = _parse_data("\n".join(data_lines), varform)
    if parsed is None:
        raise RuntimeError("Could not parse an X-Pulse NTUPLES DATATABLE block")
    values, _ = parsed

    varform_upper = varform.strip().upper()

    def _count(line: str) -> int:
        p = _parse_pseudo([line]) if varform_upper == "ASDF" else _parse_affn_pac([line])
        return len(p) if p is not None else 0

    counts = [_count(line) for line in data_lines]

    time_full: list[float] = []
    for i, t in enumerate(times):
        count = counts[i] if i < len(counts) else 1
        next_t = times[i + 1] if i + 1 < len(times) else t_last
        time_full.extend(np.linspace(t, next_t, count, endpoint=False))
    time_full_arr = np.array(time_full)

    if len(time_full_arr) != len(values):
        if timegrid_warning:
            warn(f"time/data length mismatch: time={len(time_full_arr)}, data={len(values)}. Using uniform time grid.")
        time_full_arr = np.linspace(times[0] if times else t_first, t_last, len(values))

    uniform_time = np.linspace(t_first, t_last, len(values))
    return interp1d(time_full_arr, values, kind="linear", fill_value="extrapolate")(uniform_time)


def _read_xpulse_ntuples_2d(filepath: str, timegrid_warning: bool = False) -> dict:
    """Parse an X-Pulse NTUPLES-format 2D JCAMP-DX file into a raw,
    row-paired complex FID array (row ``2k`` real, row ``2k+1``
    imaginary, for F1 increment ``k`` -- the same convention
    :func:`opennmr.processing.ft.decode_hypercomplex_f1` already
    expects for Bruker's States/States-TPPI data; this file's own
    ``##.ACQUISITION SCHEME= TPPI`` plus the real/imag page-pairing
    confirmed by direct inspection is the identical encoding) plus the
    parameters needed to process both dimensions.

    Duplicate-named header keys (``##$SF=``/``##$SFO1=`` each appear
    twice, once per dimension, disambiguated only by a trailing
    ``$$ f1``/``$$ f2`` inline comment -- confirmed by direct
    inspection, not assumed) are captured separately rather than via a
    single last-value-wins dict, which would otherwise silently lose
    F2's own values. ``##FIRST=``/``##LAST=`` similarly appear both as
    a genuine global (pre-PAGE) declaration and, separately, as an
    always-``0,0,0,0`` per-PAGE placeholder in standard NTUPLES
    layout -- only the first (global) occurrence is used here, since
    per-row time reconstruction instead comes from each data line's
    own leading time checkpoint, exactly as the plain-1D reader
    already does.

    Returns a dict with keys ``"complex"`` (2D array, row-paired),
    ``"params"``, ``"exp_params"``, ``"t1_values"`` (the raw, not
    zero-based, per-page-pair T1 values recorded in the file, purely
    for diagnostics -- not used for F1 axis construction, since the
    increments are uniform and ``exp_params["SW1"]`` already gives the
    true F1 dwell time directly).
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    params: dict = {}
    exp_params: dict = {}
    pages: list[dict] = []
    current_page = None
    in_datatable = False
    in_spinflow_block = False
    global_first: list[float] | None = None
    global_last: list[float] | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("$$"):
            body = line[2:].strip().lower()
            if "spinflow" in body and "start" in body:
                in_spinflow_block = True
                continue
            if "spinflow" in body and "end" in body:
                in_spinflow_block = False
                continue
            if in_spinflow_block and "=" in line:
                k, v = line[2:].split("=", 1)
                k, v = k.strip(), v.strip()
                try:
                    exp_params[k] = [float(x) for x in v.split(",")] if "," in v else float(v)
                except ValueError:
                    exp_params[k] = v
            continue
        if not line.upper().startswith("##"):
            if in_datatable and current_page is not None:
                current_page["lines"].append(line)
            continue

        in_datatable = False
        m = re.match(r"##([^=]+)=\s*(.*)", line)
        if not m:
            continue
        key, val = m.group(1).strip().upper(), m.group(2).strip()

        if key == "PAGE":
            t1_match = re.search(r"T1\s*=\s*([-+0-9.eE]+)", val)
            current_page = {"t1": float(t1_match.group(1)) if t1_match else None, "kind": None, "lines": []}
            pages.append(current_page)
        elif key == "DATATABLE":
            in_datatable = True
            if current_page is not None:
                current_page["kind"] = "I" if "(I..I)" in val.upper() else "R"
        elif key in ("FIRST", "LAST") and current_page is None:
            # Global (pre-PAGE) declaration only -- see this function's
            # own docstring for why the per-PAGE occurrence is ignored.
            values = [float(x.strip()) for x in val.split(",")]
            if key == "FIRST":
                global_first = values
            else:
                global_last = values
        elif "$$" in val:
            # Duplicate-named, dimension-tagged key (SF/SFO1) -- keep
            # both, keyed by the trailing comment.
            body, _, tag = val.partition("$$")
            tag = tag.strip().lower()
            params[f"{key}_{tag.upper()}" if tag in ("f1", "f2") else key] = body.strip()
        else:
            params[key] = val

    if len(pages) < 2 or len(pages) % 2 != 0:
        raise RuntimeError(f"Expected an even number of NTUPLES PAGE blocks (real/imag pairs), got {len(pages)}")
    if global_first is None or global_last is None:
        raise RuntimeError("NTUPLES file has no global ##FIRST=/##LAST= declaration")

    # NOT params.get("VARFORM"): confirmed by direct inspection against
    # the real test file that its own ``##VARFORM= AFFN, AFFN, AFFN,
    # AFFN`` declaration is simply wrong for the R/I (FID amplitude)
    # columns -- their actual data lines are unambiguously ASDF
    # squeeze/diff/dup pseudo-digit text (the same alphabet the plain-
    # 1D reader already decodes), not plain decimal numbers; trusting
    # the header produced garbage values (fragments of pseudo-digit
    # text mis-read as AFFN numerals) and a wrong per-line point count.
    # Detected from the data's own content instead -- ASDF's alphabet
    # is all letters (bar scientific-notation e/E), which never appear
    # in genuine AFFN decimal text -- so this self-corrects regardless
    # of what any future NTUPLES export's own header happens to claim.
    t2_first, t2_last = global_first[1], global_last[1]
    varform_r = _detect_xpulse_datatable_varform(pages[0]["lines"])
    varform_i = _detect_xpulse_datatable_varform(pages[1]["lines"])

    rows = []
    t1_values = []
    for real_page, imag_page in zip(pages[0::2], pages[1::2]):
        if real_page["kind"] != "R" or imag_page["kind"] != "I":
            raise RuntimeError("NTUPLES PAGE pair isn't in the expected (real, imaginary) order")
        real = _parse_xpulse_timegrid_block(real_page["lines"], varform_r, t2_first, t2_last, timegrid_warning)
        imag = _parse_xpulse_timegrid_block(imag_page["lines"], varform_i, t2_first, t2_last, timegrid_warning)
        rows.append(real)
        rows.append(imag)
        t1_values.append(real_page["t1"])

    n2 = max(len(row) for row in rows)
    rows = [row if len(row) == n2 else np.pad(row, (0, n2 - len(row))) for row in rows]

    return {
        "complex": np.array(rows, dtype=np.float64),  # row-paired real/imag, not yet combined -- see docstring
        "params": params,
        "exp_params": exp_params,
        "t1_values": np.array(t1_values, dtype=np.float64),
    }


def read_2d(path: str, *, window: str = "exponential", window_kwargs: dict | None = None,
            zero_fill: int | bool | None = None, zero_fill_f1: int | bool | None = None,
            truncate_f1: int | bool | None = None,
            autophase: bool = True, phase_mode: str = "apc0",
            phase0: float = 0.0, phase1: float = 0.0, pivot_frac: float = 0.5, phase0_inc: float = 0.1,
            timegrid_warning: bool = False) -> NMRSpectrum:
    """Read an X-Pulse NTUPLES-format genuine 2D dataset (e.g. HSQC) --
    :func:`_read_xpulse_ntuples_2d` for the raw row-paired FID, then
    F2 apodize/zero-fill/FFT per row (the same steps
    :func:`read_1d` uses, just looped), then
    :func:`opennmr.processing.ft.decode_hypercomplex_f1` (this file's
    own ``##.ACQUISITION SCHEME= TPPI`` plus its real/imag PAGE-pairing,
    confirmed by direct inspection, is States-TPPI's exact row-pairing
    convention) before F1's own FFT.

    ``zero_fill_f1`` : ``True`` zero-fills F1 to ``NP1`` rounded up to
    the next power of 2 (X-Pulse has no recorded "SI" for F1 the way
    ``proc2s`` does for Bruker -- there's nothing processed to recall);
    an ``int`` sets it explicitly; ``None`` (default) leaves F1 at its
    native ``NP1`` row-pair count. ``truncate_f1`` : like ``TD`` --
    ``True`` uses the file's own recorded ``NP1``; an ``int`` uses
    that many F1 increments (row *pairs*, i.e. real+imag together);
    ``None`` (default) uses every increment actually present. Applied
    to the row-paired data *before* :func:`decode_hypercomplex_f1`, so
    an odd request is rounded down to the nearest whole pair rather
    than truncating a real row without its matching imaginary one.

    Unlike :func:`read_1d`, no ``fid_shift``/``align_echo``/
    ``linear_predict`` here -- these interact with F1's own row-pairing
    in ways not yet worked out for this format -- scoped out rather
    than guessed at.

    **F1's own ppm-axis referencing is a best-effort, not a verified
    convention.** Both dimensions' ppm axes here use the file's own
    ``TxPPM``/``TxPPMX`` fields (H/X transmitter offset, directly in
    ppm) as the FFT axis's centre -- chosen over reproducing
    :func:`read_1d`'s own ``$O1``/``$SF``/``$SFO1``-based Hz formula
    because this NTUPLES export has no ``$O1`` field for either
    dimension at all (confirmed by direct inspection) and using
    ``exp_params["O1"]`` (a *different*, seemingly unrelated field
    also named "O1") in that formula's place produced a 1H axis
    spanning roughly -60 to -30 ppm -- not just off, physically
    impossible for real 1H shifts, so treated as confirmation that
    field wasn't the right one, not just an untried alternative.
    ``TxPPM``/``TxPPMX`` centre both axes in a physically sensible
    range, but the specific peak positions recovered from the one real
    test dataset available (Quinine HSQC) couldn't be independently
    checked against a reference spectrum, and the strongest peak
    recovered sits at a 1H shift (~21 ppm) too far downfield to be
    real -- likely an unphased artifact riding on real signal, not
    proof the referencing itself is wrong, but not proof it's right
    either. Treat this axis as unverified until checked against a
    known reference spectrum -- see TODO.md.
    """
    fid_data = _read_xpulse_ntuples_2d(path, timegrid_warning=timegrid_warning)
    raw_rows = np.asarray(fid_data["complex"], dtype=np.complex128)  # row-paired: 2k real, 2k+1 imag
    params = fid_data["params"]
    exp_params = fid_data["exp_params"]

    sw2 = float(exp_params.get("Filter", 0.0))
    if sw2 <= 0:
        raise ValueError("Spectral width (F2) must be > 0 to compute dt")
    dt2 = 1.0 / sw2
    sfo1_hz = float(params.get("$SFO1_F2", params.get("$SFO1", 1.0))) * 1e6
    sfo1_hz_f1 = float(params.get("$SFO1_F1", 1.0)) * 1e6
    tx_ppm_f2 = float(exp_params.get("TxPPM", 0.0))
    tx_ppm_f1 = float(exp_params.get("TxPPMX", 0.0))

    sw1 = float(exp_params.get("SW1", 0.0))
    if sw1 <= 0:
        raise ValueError("Spectral width (F1) must be > 0 to compute dt")
    dt1 = 1.0 / sw1

    # truncate_f1 -- applied to row *pairs*, before decode_hypercomplex_f1.
    n1_native = raw_rows.shape[0] // 2
    if truncate_f1 is True:
        td1 = int(exp_params.get("NP1", n1_native))
    elif isinstance(truncate_f1, int):
        td1 = truncate_f1
    else:
        td1 = None
    if td1 is not None and 0 < td1 < n1_native:
        raw_rows = raw_rows[: 2 * td1]
    fid_raw = raw_rows

    def _process_row(row):
        windowed = processing_ft.apodize(row, dt2, window=window, **(window_kwargs or {}))
        if zero_fill is True:
            si_points = int(exp_params.get("NP2", len(windowed)))
        elif isinstance(zero_fill, int):
            si_points = zero_fill
        else:
            si_points = None
        if si_points is not None and si_points > len(windowed):
            windowed = processing_ft.zero_fill(windowed, si_points)
        freq_hz, spectrum_row = processing_ft.fft_spectrum(windowed, dt2)
        return freq_hz, spectrum_row

    freq_hz = None
    f2_rows = []
    for row in raw_rows:
        freq_hz, spectrum_row = _process_row(row)
        f2_rows.append(spectrum_row)
    spectrum = np.array(f2_rows)
    freq_ppm = tx_ppm_f2 + freq_hz / (sfo1_hz / 1e6)

    autophased = bool(autophase)
    if autophase:
        mean_spectrum = spectrum.mean(axis=0)
        if phase_mode == "apc0":
            phase0 = processing_ft.apc0(mean_spectrum, freq_ppm, phase0_inc)
        else:
            raise ValueError("phase_mode must be 'apc0'")
    if phase0 or phase1:
        spectrum = processing_ft.apply_phase(spectrum, freq_ppm, phase0, phase1, pivot_frac=pivot_frac)

    # F1 (indirect): hypercomplex decode (States-TPPI -- see this
    # function's own docstring) then optional zero-fill then FFT.
    spectrum = processing_ft.decode_hypercomplex_f1(spectrum, fn_mode=5)
    if zero_fill_f1 is True:
        si1 = int(2 ** np.ceil(np.log2(max(spectrum.shape[0], 1))))
    elif isinstance(zero_fill_f1, int):
        si1 = zero_fill_f1
    else:
        si1 = None
    if si1 is not None and si1 > spectrum.shape[0]:
        spectrum = np.apply_along_axis(lambda col: processing_ft.zero_fill(col, si1), 0, spectrum)
    freq_hz_f1, spectrum = processing_ft.fft_spectrum(spectrum.T, dt1)
    spectrum = spectrum.T
    freq_ppm_f1 = tx_ppm_f1 + freq_hz_f1 / (sfo1_hz_f1 / 1e6)

    ns = float(exp_params.get("NS", 1.0))
    spectrum = spectrum / ns  # match read_1d's own convention: data is always per-scan

    nucleus_f2 = re.sub(r"\W+", "", params.get(".OBSERVENUCLEUS", "") or "") or None
    nucleus_f1 = re.sub(r"\W+", "", params.get("$INDIRECTNUCLEUS", "") or "") or None

    metadata = NMRMetadata(
        vendor="xpulse",
        nucleus=nucleus_f2, nucleus_indirect=nucleus_f1,
        sfo1=sfo1_hz / 1e6, sfo1_indirect=sfo1_hz_f1 / 1e6,
        source_path=path,
        probe=exp_params.get("Probe"), pulse_program=params.get(".PULSESEQUENCE"),
        raw={
            "params": params, "exp_params": exp_params,
            # Recorded (autophase's own base) vs user phase -- see
            # _phase_metadata for why the split matters on reload.
            **_phase_metadata(phase0, phase1, autophased, pivot_frac),
            "fn_mode": 5,
            "fid": f2_rows, "fid_dt": dt2, "fid_original_length": raw_rows.shape[1],
            "fid_dt_f1": dt1,
            "fid_raw": fid_raw,
        },
    )
    y_axis = NMRAxis(kind="frequency", ppm=freq_ppm_f1, hz=freq_hz_f1)
    x_axis = NMRAxis(kind="frequency", ppm=freq_ppm, hz=freq_hz)

    return NMRSpectrum(
        data=np.real(spectrum), axes=[y_axis, x_axis], metadata=metadata, data_imag=np.imag(spectrum),
    )


# ============================================================
# Arrayed experiments (e.g. PFG): one pseudo-2D spectrum out of
# several array#/ subfolders, instead of N separate 1D matches
# ============================================================

_ARRAY_DIR_RE = re.compile(r"^array(\d+)$", re.IGNORECASE)


def _find_array_dirs(path: str) -> list[tuple[int, str]]:
    """(step number, dir path) for every ``arrayN`` subfolder directly
    under ``path``, sorted by ``N`` -- X-Pulse's convention for an
    arrayed acquisition (one step, e.g. one gradient strength or one
    delay, per subfolder)."""
    found = []
    try:
        entries = os.scandir(path)
    except OSError:
        return []
    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            m = _ARRAY_DIR_RE.match(entry.name)
            if m:
                found.append((int(m.group(1)), entry.path))
    found.sort(key=lambda t: t[0])
    return found


def _find_array_step_jdx(array_dir: str) -> str | None:
    """The one ``.jdx``/``.dx`` file directly inside an ``arrayN``
    folder -- the averaged spectrum for that step (``..._avg.jdx``).
    Deliberately not recursive: each array folder's own ``raw/``
    subfolder holds the individual per-scan repeats that combine into
    this one averaged file, and isn't independently useful data (see
    ``MainWindow._XPULSE_RAW_SCAN_RE``, the same distinction the GUI's
    own folder scan already makes)."""
    try:
        entries = os.scandir(array_dir)
    except OSError:
        return None
    candidates = sorted(
        e.path for e in entries
        if e.is_file(follow_symlinks=False) and os.path.splitext(e.name)[1].lower() in (".jdx", ".dx")
    )
    return candidates[0] if candidates else None


def _read_gradient_relative(path: str) -> np.ndarray | None:
    """``parameters/G1_valueslist.txt`` -- one relative gradient
    strength (a fraction, not an absolute value in any physical unit
    -- there's no gradient calibration recorded in the acquired data
    itself) per line, one line per array step, X-Pulse's own PFG
    convention. ``None`` if the file isn't present (not every arrayed
    experiment is a gradient series)."""
    list_path = os.path.join(path, "parameters", "G1_valueslist.txt")
    if not os.path.exists(list_path):
        return None
    try:
        with open(list_path) as f:
            values = [float(line.strip()) for line in f if line.strip()]
    except (OSError, ValueError):
        return None
    return np.array(values, dtype=float)


def read_array(path: str, **kwargs) -> NMRSpectrum:
    """Read an X-Pulse *arrayed* experiment -- a parent folder holding
    one ``arrayN/`` subfolder per acquisition step (a PFG
    gradient-strength series, or any other stepped/arrayed pulse
    sequence) -- as ONE pseudo-2D :class:`NMRSpectrum`, one row per
    step, in step order. Without this, opening the parent folder finds
    N independent 1D spectra instead of the one series they actually
    are.

    ``**kwargs`` are forwarded to :func:`read_1d` for every step (the
    same window/zero-fill/align-echo/etc. overrides), so this composes
    with the rest of the FID-reprocessing pipeline exactly like a
    single-file X-Pulse load.

    Relative gradient strengths recorded in the experiment's own
    ``parameters/G1_valueslist.txt`` (if present) are stashed in
    ``metadata.raw["gradient_relative"]``, one value per row in the
    same order -- lets a PFG series be fit against directly, without
    needing an absolute gradient calibration the data doesn't record.
    """
    array_dirs = _find_array_dirs(path)
    if not array_dirs:
        raise ValueError(f"No array#/ subfolders found under {path!r}.")

    slices = []
    for step, array_dir in array_dirs:
        jdx_path = _find_array_step_jdx(array_dir)
        if jdx_path is None:
            raise ValueError(f"No .jdx/.dx file found directly under {array_dir!r} (step {step}).")
        slices.append(read_1d(jdx_path, **kwargs))

    n_points = len(slices[0].data)
    if any(len(s.data) != n_points for s in slices):
        raise ValueError(
            "All array steps must produce the same number of points after "
            "processing -- got mismatched sizes across steps (check that "
            "zero_fill/truncate settings are consistent)."
        )

    first = slices[0]
    data = np.stack([s.data for s in slices])
    data_imag = (
        np.stack([s.data_imag for s in slices]) if all(s.data_imag is not None for s in slices) else None
    )
    fid_stack = np.stack([s.metadata.raw["fid"] for s in slices])
    fid_raw_stack = (
        np.stack([s.metadata.raw["fid_raw"] for s in slices])
        if all(len(s.metadata.raw["fid_raw"]) == len(slices[0].metadata.raw["fid_raw"]) for s in slices)
        else None  # acquisition length genuinely differed across steps -- rare, no single-array shape to stack into
    )

    f1_axis = NMRAxis(kind="index", index=np.arange(1, len(slices) + 1))
    f2_axis = first.axes[0]

    metadata = dataclasses.replace(
        first.metadata,
        pseudo_2d=True,
        source_path=path,
        raw={
            **first.metadata.raw,
            "fid": fid_stack,
            "fid_raw": fid_raw_stack,
            "gradient_relative": _read_gradient_relative(path),
        },
    )

    return NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata, data_imag=data_imag)


# ============================================================
# Vendor registration
# ============================================================

def _detect(path: str) -> bool:
    """A single ``.jdx``/``.dx`` file whose header contains an
    X-Pulse-flavoured JCAMP-DX marker, OR a folder holding ``arrayN``
    subfolders (an arrayed experiment -- see :func:`read_array`)."""
    if os.path.isdir(path):
        return bool(_find_array_dirs(path))
    if not os.path.isfile(path):
        return False
    if os.path.splitext(path)[1].lower() not in (".jdx", ".dx"):
        return False
    # A cloud placeholder is accepted on its extension alone. Reading
    # even the first 4 KB of one makes the sync client materialise the
    # *whole* file (file providers fetch files, not byte ranges), so
    # scanning a OneDrive folder of these would download every one of
    # them just to decide what to list. Being optimistic here is the
    # cheap mistake: a `.jdx` that turns out not to be X-Pulse fails
    # at load, by which point it has been downloaded anyway -- whereas
    # reading now costs the download unconditionally, for every file,
    # before the user has asked for anything.
    if cloud.is_dataless(path):
        return True
    try:
        with open(path, "r", errors="ignore") as f:
            head = f.read(4096)
    except OSError:
        return False
    return "##$SFO1=" in head or "$$SPINFLOW" in head.upper()


def _is_ntuples_2d_file(path: str) -> bool:
    """Whether ``path`` is a genuine hypercomplex 2D NTUPLES file
    (:func:`read_2d`), not a plain 1D FID (:func:`read_1d`) -- both
    are single-file X-Pulse JCAMP-DX using the same ``##DATACLASS=
    NTUPLES`` data-table format for their FID either way, so that
    field alone doesn't distinguish them (real bug: it's present even
    on a plain 1D file, e.g. ``test_data/xpulse_MRI/14/*_avg.jdx``,
    which was misdetected as 2D and sent into :func:`read_2d`, which
    then failed on this file's non-existent F1 spectral width -- "must
    be > 0 to compute dt"). ``##NUMDIM= 2`` (vs. ``= 1`` for a plain
    FID) is the field that actually says so."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
    except OSError:
        return False
    return re.search(r"##NUMDIM\s*=\s*2\b", head, re.IGNORECASE) is not None


def _read_auto(path: str, **kwargs) -> NMRSpectrum:
    """Vendor-registry entry point: an arrayed-experiment folder reads
    as one pseudo-2D spectrum (:func:`read_array`); a single NTUPLES
    file reads as a genuine 2D spectrum (:func:`read_2d`); any other
    single file reads as one 1D spectrum (:func:`read_1d`)."""
    if os.path.isdir(path):
        return read_array(path, **kwargs)
    if _is_ntuples_2d_file(path):
        # read_2d's own kwarg set is a strict subset of read_1d's
        # (no fid_shift/align_echo/linear_predict/truncate -- see its
        # own docstring for why) -- drop whatever it doesn't accept
        # rather than erroring on a caller that always passes read_1d's
        # full kwarg set (e.g. the GUI's own uniform per-vendor call).
        accepted = set(inspect.signature(read_2d).parameters)
        return read_2d(path, **{k: v for k, v in kwargs.items() if k in accepted})
    return read_1d(path, **kwargs)


base.register("xpulse", _detect, _read_auto)
