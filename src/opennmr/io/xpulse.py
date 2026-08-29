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
import os
import re
from warnings import warn

import numpy as np
from scipy.interpolate import interp1d

from . import base
from ..processing import ft as processing_ft
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum


# ============================================================
# ASDF/AFFN JCAMP-DX decoding (ported from leonmr.read_xpulse_jcamp)
# ============================================================

def _read_xpulse_jcamp(filepath: str, verbose: bool = False, timegrid_warning: bool = False) -> dict:
    """Parse an X-Pulse JCAMP-DX file into a raw complex FID plus the
    instrument/experiment parameter dicts it carries.

    Returns a dict with keys ``"complex"``, ``"real"``, ``"imag"``,
    ``"time"``, ``"params"`` (instrument parameters, ``##$KEY=value``
    lines), ``"exp_params"`` (experiment parameters from the
    ``$$SPINFLOW`` block).
    """

    _DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
    _SQZ_DIGITS = {
        "@": "0", "A": "1", "B": "2", "C": "3", "D": "4", "E": "5",
        "F": "6", "G": "7", "H": "8", "I": "9",
        "a": "-1", "b": "-2", "c": "-3", "d": "-4", "e": "-5",
        "f": "-6", "g": "-7", "h": "-8", "i": "-9",
    }
    _DIF_DIGITS = {
        "%": "0", "J": "1", "K": "2", "L": "3", "M": "4", "N": "5",
        "O": "6", "P": "7", "Q": "8", "R": "9",
        "j": "-1", "k": "-2", "l": "-3", "m": "-4", "n": "-5",
        "o": "-6", "p": "-7", "q": "-8", "r": "-9",
    }
    _DUP_DIGITS = {
        "S": "1", "T": "2", "U": "3", "V": "4", "W": "5",
        "X": "6", "Y": "7", "Z": "8", "s": "9",
    }

    def _append_value(data, value_to_append, isdif):
        if isdif and data:
            data.append(data[-1] + value_to_append)
        else:
            data.append(value_to_append)

    def _finish_value(valuestr, currentmode, prev_value_to_append, data):
        try:
            if currentmode == 1:  # squeeze
                if prev_value_to_append is not None:
                    _append_value(data, *prev_value_to_append)
                return (float(valuestr), False), True
            elif currentmode == 2:  # diff
                if prev_value_to_append is not None:
                    _append_value(data, *prev_value_to_append)
                return (float(valuestr), True), True
            elif currentmode == 3:  # duplicate
                if prev_value_to_append is None:
                    warn("Parse error: DUP entry without preceding value")
                    return None, False
                dupcount = int(valuestr)
                for _ in range(dupcount):
                    _append_value(data, *prev_value_to_append)
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
                if char in _DIGITS:
                    valuestr.append(char)
                    continue
                try:
                    valuechar = _SQZ_DIGITS[char]
                    newmode = 1
                except KeyError:
                    try:
                        valuechar = _DIF_DIGITS[char]
                        newmode = 2
                    except KeyError:
                        try:
                            valuechar = _DUP_DIGITS[char]
                            newmode = 3
                        except KeyError:
                            warn(f"Unknown pseudo-digit: {char}")
                            return None
                valuestr = "".join(valuestr)
                previous_is_dif = currentmode == 2 or (currentmode == 3 and value_to_append and value_to_append[1])
                if not skip_checkpoint:
                    value_to_append, success = _finish_value(valuestr, currentmode, value_to_append, data)
                    if not success:
                        warn("Data parsing failed")
                        return None
                skip_checkpoint = first_of_line and previous_is_dif
                currentmode = newmode
                valuestr = [valuechar]
                first_of_line = False
        if not skip_checkpoint:
            valuestr = "".join(valuestr)
            value_to_append, success = _finish_value(valuestr, currentmode, value_to_append, data)
            if not success:
                warn("Data parsing failed at last")
                return None
        if value_to_append is not None:
            _append_value(data, *value_to_append)
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

def read_1d(path: str, *, window: str = "exponential", window_kwargs: dict | None = None,
            zero_fill: int | bool | None = None, truncate: int | bool | None = None,
            fid_shift: float = 0.0, align_echo: bool = False,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward",
            baseline_correct: bool | None = None, normalize: bool = False,
            autophase: bool = True, phase_mode: str = "apc0",
            phase0: float = 0.0, phase1: float = 0.0, phase0_inc: float = 0.1,
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

    if fid_shift:
        fid = processing_ft.roll_fid(fid, fid_shift)

    # Deliberately computed/applied BEFORE the truncate step below,
    # not after: align_echo rolls the echo top to index 0, which wraps
    # the pre-echo portion (dead time, then the rise up to the echo
    # top -- roughly a mirror image of the post-echo decay, for a
    # symmetric full echo) around to the tail of the array. Only a
    # truncate applied to the POST-roll array, short enough to stop
    # before reaching the wrap point, actually discards it -- get this
    # backwards (as this used to be) and the "aligned" FID has a
    # mirror-image copy of its own decay spliced onto its tail -- an
    # FID that "looks like a time-reversal" once FFT'd. Matches
    # leonmr's xpulse_image_import, which rolls the full FID first and
    # only then hands it to process_fid()'s own TDEFF truncation.
    echo_wrap_limit = None
    if align_echo:
        top = processing_ft.find_echo_top(fid)
        fid = processing_ft.align_echo(fid)
        # Neither the file's own recorded TDEFF nor an unspecified
        # `truncate` reliably stops short of the wrap point on its own
        # (confirmed on real data: TDEFF is set for the *original*,
        # pre-roll acquisition -- e.g. "stop before the noise floor" --
        # not recomputed relative to where the echo top ends up after
        # rolling, so it can easily extend past the wrap point and
        # let the mirrored tail back in). Capping here, unconditionally
        # rather than only when a truncate was explicitly requested,
        # is what actually fixes the artifact by default -- ceil() is
        # deliberately conservative, so even a fractional echo-top
        # position never leaves a single wrapped sample in.
        echo_wrap_limit = len(fid) - int(np.ceil(top))

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

    if echo_wrap_limit is not None:
        tdeff_points = echo_wrap_limit if tdeff_points is None else min(tdeff_points, echo_wrap_limit)

    if tdeff_points is not None and tdeff_points < len(fid):
        fid = fid[:tdeff_points]

    if linear_predict_points:
        fid = processing_ft.linear_predict(
            fid, order=linear_predict_order, n_predict=linear_predict_points,
            direction=linear_predict_direction,
        )
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

    spectrum = processing_ft.apply_phase(spectrum, freq_ppm, phase0, phase1)

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
        raw={
            "params": params, "exp_params": exp_params, "phase0": phase0, "phase1": phase1,
            # The fully processed (pre-FFT) FID -- an optional GUI
            # preview of what FID-domain processing actually did.
            "fid": fid, "fid_dt": dt, "fid_original_length": original_length,
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

    f1_axis = NMRAxis(kind="index", index=np.arange(1, len(slices) + 1))
    f2_axis = first.axes[0]

    metadata = dataclasses.replace(
        first.metadata,
        pseudo_2d=True,
        source_path=path,
        raw={
            **first.metadata.raw,
            "fid": fid_stack,
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
    try:
        with open(path, "r", errors="ignore") as f:
            head = f.read(4096)
    except OSError:
        return False
    return "##$SFO1=" in head or "$$SPINFLOW" in head.upper()


def _read_auto(path: str, **kwargs) -> NMRSpectrum:
    """Vendor-registry entry point: an arrayed-experiment folder reads
    as one pseudo-2D spectrum (:func:`read_array`); a single file reads
    as one 1D spectrum (:func:`read_1d`)."""
    if os.path.isdir(path):
        return read_array(path, **kwargs)
    return read_1d(path, **kwargs)


base.register("xpulse", _detect, _read_auto)
