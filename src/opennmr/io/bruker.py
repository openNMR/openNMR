"""
Bruker importer.

``read_1d``/``read_2d`` read the raw ``fid``/``ser`` file and
reconstruct the spectrum in Python, using ``nmrglue`` for the
genuinely Bruker-specific DSP that's risky to hand-roll (raw binary
reading, digital-filter/group-delay removal, and windowing that
matches Bruker's own ``WDW``-coded window functions) -- FFT and
phasing go through the same :mod:`opennmr.processing.ft` primitives
used for every other vendor, so there is one consistent FFT/phase
convention across the whole package rather than a second one specific
to Bruker.

Only ``WDW`` in ``{0 (none), 1 (EM), 3 (SINE), 4 (QSINE)}`` is
currently supported for the raw-FID path (see
:func:`_apply_bruker_window`) -- these cover routine 1D processing;
GM and less common windows raise ``NotImplementedError`` rather than
silently guessing at an unverified parameter mapping. ``read_1d``/
``read_2d`` catch exactly that error around the windowing step only
and continue with the FID left unwindowed -- the point of the raw-FID
path is to stay on it (raw FID in, full reprocessing available) even
when one specific recorded parameter isn't understood yet, not to
abandon it for TopSpin's already-processed, non-reprocessable output
just because of that. The user can pick a different window from the
Processing tab immediately (or extend :func:`_apply_bruker_window`
once the WDW code's parameters are known) rather than losing raw-FID
reprocessing entirely. Flagged via
``metadata.raw["window_unsupported"]`` so callers (the GUI) can tell
the user why the spectrum isn't windowed as recorded, rather than it
being silent.

``read_1d_processed``/``read_2d_processed`` are a fallback, and a
cross-check: they read TopSpin's own already-processed output
directly, porting ``nmrtools2``'s original ``NMR1dimport``/
``NMR2dimport`` logic close to unchanged and independently of
``nmrglue``, so agreement between the two paths on the same dataset is
itself evidence the raw-FID reprocessing above is correct (see the
project plan's Verification section) -- not just two call paths that
happen to share a library underneath. That cross-check is exactly how
the note below was found and confirmed against real data, not guessed.

_CONJUGATE_QUADRATURE_NOTE: Bruker's raw quadrature detection encodes
a "positive" chemical shift with the opposite sense to
:mod:`opennmr.processing.ft`'s ``fft_spectrum``/axis convention (that
convention was validated against X-Pulse's own real,
independently-working formula -- see ``io/xpulse.py`` -- so it's kept
as-is there rather than "fixed" globally). Left uncorrected, this
first showed up as an unexplained referencing offset that differed
per dataset (confirmed on two real 2D datasets: PFG diffusion and a
Hahn-echo T2 series) rather than a clean mirror, because the
mislabelled data was also compared against a nonzero-centred axis;
mirrored around a near-zero-ppm window it instead reads as an exact
sign flip (e.g. a real 31P spectrum read out at +144.5 ppm instead of
-144.5). Conjugating the FID before windowing/FFT is the fix,
confirmed by comparing REAL parts (not just magnitude) against
``read_2d_processed`` on real data -- reversing the output array
alone matches on magnitude too, but only conjugating the input also
gets the phase relationship right. Applied once, on the complex data
straight out of ``_remove_digital_filter``, so it covers F2 -- and,
since Bruker's raw complex encoding is presumably one consistent
convention library-wide, F1 too, though only F2 has actually been
checked against known-correct output so far (every real dataset
available while building this was pseudo-2D, which never exercises
the F1 FFT branch in :func:`read_2d`).
"""

from __future__ import annotations

import os

import numpy as np
import nmrglue as ng

from . import base
from .jcampdx import parse_jcampdx_params
from ..processing import ft as processing_ft
from ..core.registry import find_gamma
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum

_WDW_NAMES = {0: "none", 1: "em", 2: "gm", 3: "sine", 4: "qsine"}


def _remove_digital_filter(dic: dict, data: np.ndarray) -> np.ndarray:
    """``ng.bruker.remove_digital_filter``, skipped when ``DECIM==1``
    (no oversampling -- common for wideline/quadrupolar acquisitions
    needing a very large SW, e.g. a Hahn-echo on a low-gamma nucleus).
    There is then no digital filter to remove, and ``DECIM=1`` isn't
    in nmrglue's own DECIM/DSPFVS lookup table -- it raises
    ``ValueError: decim not in lookup table`` rather than treating
    this as a no-op."""
    decim = dic.get("acqus", {}).get("DECIM")
    if decim is not None and float(decim) == 1.0:
        return data
    return ng.bruker.remove_digital_filter(dic, data)


def _guess_raw_2d_shape(dic: dict, ser_path: str) -> tuple[int, int]:
    """The raw ``ser`` file's true shape ``(n_rows, n_complex_per_row)``,
    determined directly from ``TD``/``DTYPA``/the file's own size --
    not from ``nmrglue.bruker.guess_shape``, which needs the pulse
    program's own loop/increment structure to handle anything beyond
    the simplest case (this project deliberately skips reading it --
    see :func:`read_2d`'s own comment) and, even with it enabled, has
    a real bug for float64 (``DTYPA=2``) data: it rounds the per-row
    word count using the correct 128-word convention but then converts
    that to bytes with a hardcoded ``*4`` (int32's own size) instead
    of ``*8``, so its own file-size-based guess is never actually
    validated.

    Row *length* (``TD`` rounded up to the block size the acquisition
    memory actually uses -- 128 float64 words / 1024 bytes for
    ``DTYPA=2``, 256 int32 words / 1024 bytes otherwise, matching the
    documented convention for the direct dimension) is confirmed
    empirically via autocorrelation on real raw ``ser`` data (a
    genuine periodicity in the FID envelope, not just a formula) on
    two independent real States-TPPI datasets, and matches nmrglue's
    own current guess exactly on the two Bruker 2D datasets this
    project's test suite already exercises successfully
    (``bruker_T2/11``, ``bruker_PFG/150``) -- this isn't a change in
    behaviour for data that already worked.

    Row *count*: ``acqu2s``'s own ``TD`` directly -- confirmed by
    reading a real compiled pulse program's own ``mc``-macro expansion
    for a States-TPPI dataset directly (not guessed): the labelled
    loop structure writes exactly ``TD`` total rows (``trunc(TD/2)``
    t1 steps, each writing a real/imaginary States pair of 2 rows).
    On real data this doesn't always land on the file's own exact byte
    count -- a small, so-far-unexplained handful of extra rows beyond
    ``TD`` even on acquisitions with a clean (non-aborted-and-restarted)
    audit trail, see TODO.md's own write-up of this investigation.
    Rather than fail, falls back to ``file_size // row_bytes`` (floor
    division, always in-bounds) whenever ``TD`` itself doesn't divide
    the file exactly, so a real acquisition with this small
    discrepancy still loads -- capturing every genuinely-present row
    -- instead of crashing outright the way relying on ``TD`` alone
    (and nmrglue's own guess) both currently do."""
    acqus = dic["acqus"]
    td0 = int(float(acqus["TD"]))
    td1 = int(float(dic.get("acqu2s", {}).get("TD", 0))) or None
    dtypa = int(float(acqus.get("DTYPA", 0)))
    word_bytes = 8 if dtypa == 2 else 4
    block = 128 if dtypa == 2 else 256
    per_row_words = -(-td0 // block) * block  # ceil to the nearest block
    row_bytes = per_row_words * word_bytes

    fsize = os.path.getsize(ser_path)
    if td1 and td1 * row_bytes == fsize:
        n_rows = td1
    else:
        n_rows = fsize // row_bytes  # floor -- always a valid, in-bounds reshape
    return n_rows, per_row_words // 2


def _read_raw_2d_data(path: str, dic: dict) -> np.ndarray:
    """Read the raw ``ser`` file into a complex ``(n_rows,
    n_complex_per_row)`` array, via ``nmrglue.bruker.read`` itself
    (same call every other reader in this module uses, and the same
    mocking seam every test in this project's suite already relies on)
    but with the shape it reshapes to coming from
    :func:`_guess_raw_2d_shape` instead of ``nmrglue``'s own internal
    guess (see that function's own docstring for why).

    ``ng.bruker.read``'s own ``.reshape()`` demands an *exact* fit --
    it falls back to returning the flat, unshaped array (with its own
    warning) otherwise, which is exactly what happens on a real
    dataset where :func:`_guess_raw_2d_shape`'s own floor-division
    fallback was needed (the file's byte count doesn't exactly match
    ``TD``'s own expected total -- see that function's docstring).
    Truncating and reshaping that flat array ourselves in that case is
    always safe: the target shape's element count was already derived
    from this exact array's own length, by construction.

    A large-looking leftover here (tens of percent of one row) is not
    by itself evidence of a wrong guess -- confirmed on two independent
    real ``exsy_echo3.enb`` datasets (``test_data/.../2011``, 53% of one
    row left over, and ``.../2019``, 42%, both in the same acquisition
    session): both guesses' own F2 projections match ``read_2d_processed``'s
    trusted top-5 peak positions closely, and ``2019``'s raw path has
    been extensively used and cross-validated against independent
    references elsewhere in this project's own history. An earlier
    version of this function raised on exactly this condition, which
    would have silently broken raw-FID reprocessing for the
    already-validated ``2019`` dataset -- reverted once that was found;
    see TODO.md for the full investigation, including why autocorrelation
    and "does some other row count divide the file exactly" both turned
    out to be unreliable signals for this specific pulse sequence
    family."""
    ser_path = os.path.join(path, "ser")
    n_rows, n_complex_per_row = _guess_raw_2d_shape(dic, ser_path)
    return _read_bruker_binary(ser_path, dic, (n_rows, n_complex_per_row))


def _read_bruker_binary(binary_path: str, dic: dict, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a raw Bruker ``ser``/``fid`` file into a complex array.

    Replaces a ``nmrglue.bruker.read(..., shape=...)`` call. Same
    result -- verified byte-identical against it on every real dataset
    in this project's ``test_data`` -- but without routing through
    ``nmrglue``'s own reshape, which *warns* and returns the flat,
    unshaped array whenever the requested shape doesn't divide the file
    exactly::

        UserWarning: (39583744,) cannot be shaped into (1002, 39488)

    That warning was cosmetic (this module immediately truncated and
    reshaped the flat array itself, which is always safe here because
    the shape's own element count is derived from this exact array's
    length by construction -- see :func:`_guess_raw_2d_shape`), but it
    surfaced to users on real EXSY data and read as a genuine error
    when it wasn't one. Reading the file directly means the
    "leftover tail" case is simply the normal path, with no warning to
    explain away.

    Deliberately narrow in scope: this reads the *binary* file only.
    ``nmrglue`` is still used, on purpose, for the parts that are
    genuinely hard to get right and that it has already validated
    against real Bruker output -- digital-filter/group-delay removal
    (:func:`_remove_digital_filter`) and the window functions
    (:func:`_apply_bruker_window`). Parameter files were never
    ``nmrglue``'s job here: :mod:`opennmr.io.jcampdx` has parsed those
    from scratch since the beginning.

    Endianness and data type follow the same rules ``nmrglue`` itself
    applies (confirmed by reading its source, not assumed):
    big-endian only when ``acqus``'s ``BYTORDA == 1``, and float64 only
    when ``DTYPA == 2``; int32 otherwise.

    ``shape`` : ``(n_rows, n_complex_per_row)`` for a 2D ``ser``, or
    ``None`` for a 1D ``fid`` (returns a flat complex array). A file
    holding more data than ``shape`` needs is truncated to fit; one
    holding less raises, rather than silently returning a short array.
    """
    acqus = dic.get("acqus", {}) if isinstance(dic, dict) else {}
    big_endian = acqus.get("BYTORDA") == 1
    is_float = acqus.get("DTYPA") == 2
    dtype = np.dtype(("f8" if is_float else "i4")).newbyteorder(">" if big_endian else "<")

    flat = np.fromfile(binary_path, dtype=dtype)
    # Interleaved real/imaginary pairs -- drop a trailing unpaired
    # value rather than letting the complex view fail on an odd count.
    complex_data = flat[: (flat.size // 2) * 2].astype(np.float64).view(np.complex128)

    if shape is None:
        return complex_data
    n_rows, n_complex_per_row = shape
    needed = n_rows * n_complex_per_row
    if complex_data.size < needed:
        raise ValueError(
            f"{binary_path} holds {complex_data.size} complex points, fewer than the "
            f"{needed} required for shape {shape}."
        )
    return complex_data[:needed].reshape(shape)


def apply_recorded_phase(spectrum: np.ndarray, phc0: float, phc1: float, axis: int = -1) -> np.ndarray:
    """Apply TopSpin's own recorded ``PHC0``/``PHC1`` in TopSpin's own
    convention: ``phi(i) = PHC0 + PHC1 * i/(n-1)`` across ``axis``,
    multiplied in as ``exp(+i*phi)``.

    **Measured, not assumed.** An earlier round tried feeding the
    recorded values through :func:`opennmr.processing.ft.apply_phase`
    (pivot-based, normalised over the *ppm* range) and found it made
    agreement with TopSpin dramatically worse -- which was correct
    evidence about that model, and was wrongly read at the time as
    "the recorded constants don't transfer". They transfer exactly;
    the convention was simply different.

    The convention here was measured directly rather than fitted:
    ``bruker_T2/11`` and ``bruker_PFG/150`` are processed by TopSpin
    with ``xf2`` only (``proc2s``'s ``SI`` equals its ``TD``, so F1 was
    never transformed), and both store a ``2ii`` block alongside
    ``2rr`` that is the F2 *imaginary* partner -- so
    ``2rr + 1j*2ii`` is TopSpin's own complex F2 spectrum, and the
    per-point ratio against this module's own unphased output gives the
    applied phase with no fitting at all. On ``bruker_PFG/150`` that
    ratio has constant magnitude (15.997..16.003 -- a pure phase
    difference, the 16 being ``NC_proc`` scaling) and a first-order fit
    residual of **0.001 degrees**, yielding ``p0 = -10.250`` against a
    recorded ``PHC0 = 349.742`` (identical mod 360) and
    ``p1 = -80.115`` against a recorded ``PHC1 = -80.100``.

    Applying it end-to-end lifts whole-array agreement with TopSpin
    from **+0.513 to +0.9999** (``bruker_PFG/150``) and **+0.816 to
    +0.9972** (``bruker_T2/11``).

    ``axis`` : ``-1`` for F2 (the direct dimension, the last axis),
    ``0`` for F1 -- the ramp always runs across the *transformed*
    dimension's own points.

    **Not a second phase model.** This is TopSpin's own parameters
    *converted into* the one model this project uses --
    :func:`opennmr.processing.ft.apply_phase` -- rather than an
    independent implementation of phasing. The two are exactly
    equivalent (verified to 5e-16, not approximately):

        ``apply_recorded_phase(s, PHC0, PHC1)``
          ==  ``apply_phase(s, ppm, -PHC0, +PHC1, pivot_frac=1.0)``

    for the descending ppm axis every reader in this project produces.
    ``pivot_frac=1.0`` puts the pivot at the *first* point of the array
    -- which is what an index ramp starting at ``PHC0`` means -- and
    the ``-PHC0`` accounts for ``apply_phase``'s own ``exp(-i*phi)``
    against this convention's ``exp(+i*phi)``.

    Keeping the conversion behind a named function (rather than
    scattering the sign and pivot flips across call sites) is the point:
    the mapping is measured, not obvious, and belongs in exactly one
    place.
    """
    n = spectrum.shape[axis]
    if n < 2 or (not phc0 and not phc1):
        return spectrum
    moved = np.moveaxis(spectrum, axis, -1)
    # A synthetic descending axis: apply_phase normalises over the
    # axis's own *range*, so only its direction and linearity matter,
    # not its units -- which keeps this correct for an F1 axis carrying
    # slice indices as readily as one carrying ppm.
    axis_values = np.linspace(1.0, 0.0, n)
    phased = processing_ft.apply_phase(moved, axis_values, -phc0, phc1, pivot_frac=1.0)
    return np.moveaxis(phased, -1, axis)


#: ``metadata.raw`` keys holding the phase TopSpin itself recorded, per
#: dimension. Kept **separate** from ``phase0``/``phase1`` (and their
#: ``_f1`` siblings), which mean the *user's* own phase: the two are
#: expressed in different models -- see :func:`apply_recorded_phase` --
#: so merging them into one number would make neither reproducible.
#: ``apply_dimension_phase`` is the single place they are combined.
RECORDED_PHASE_KEYS = {
    "f2": ("recorded_phase0", "recorded_phase1"),
    "f1": ("recorded_phase0_f1", "recorded_phase1_f1"),
}
USER_PHASE_KEYS = {
    "f2": ("phase0", "phase1"),
    "f1": ("phase0_f1", "phase1_f1"),
}


def recorded_phase(raw: dict, dim: str = "f2") -> tuple[float, float]:
    """This spectrum's recorded ``(PHC0, PHC1)`` for ``dim``
    (``"f2"``/``"f1"``), or ``(0.0, 0.0)``. One accessor so the reader,
    the GUI's live rebuild and the phase display can't disagree about
    which keys hold it."""
    key0, key1 = RECORDED_PHASE_KEYS[dim]
    return float(raw.get(key0, 0.0) or 0.0), float(raw.get(key1, 0.0) or 0.0)


def apply_dimension_phase(data: np.ndarray, ppm: np.ndarray, recorded: tuple[float, float],
                           user: tuple[float, float], pivot_frac: float = 0.5) -> np.ndarray:
    """Apply a dimension's full phase to ``data``'s **last axis**:
    the file's own recorded phase first, then the user's own on top.

    The single composition point for the two different phase models
    this project has to hold at once, so every path that rebuilds a
    spectrum from its raw FID produces the same answer:

    * **recorded** -- TopSpin's ``PHC0``/``PHC1`` in TopSpin's own
      index-ramp convention (:func:`apply_recorded_phase`). Reproduces
      what the file says the spectrum should look like.
    * **user** -- whatever the operator has dialled in since, in the
      pivot-based model (:func:`opennmr.processing.ft.apply_phase`)
      that makes interactive first-order phasing rotate about a chosen
      point rather than about index 0.

    Order matters and is not arbitrary: the recorded phase describes
    the file's own state, so it belongs underneath; the user's phase is
    an adjustment *relative to* that state, which is exactly what the
    Phase controls show.

    Callers handle their own orientation -- transpose so the dimension
    of interest is last (F1 paths already do), matching
    ``processing.ft``'s own last-axis convention.
    """
    data = apply_recorded_phase(data, recorded[0], recorded[1], axis=-1)
    if user[0] or user[1]:
        data = processing_ft.apply_phase(data, ppm, user[0], user[1], pivot_frac=pivot_frac)
    return data


def _apply_bruker_window(fid: np.ndarray, procs: dict, sw_hz: float) -> np.ndarray:
    """Apply the window TopSpin recorded in ``procs`` (``WDW``, with
    ``LB``/``SSB`` as needed), via ``nmrglue.proc_base``."""
    wdw = int(procs.get("WDW", 0))
    if wdw == 0:
        return fid
    if wdw == 1:
        lb_hz = float(procs.get("LB", 0.0))
        return ng.proc_base.em(fid, lb=lb_hz / sw_hz)
    if wdw in (3, 4):
        ssb = float(procs.get("SSB", 0.0))
        off = 1.0 / ssb if ssb >= 2 else 0.0
        power = 1.0 if wdw == 3 else 2.0
        return ng.proc_base.sp(fid, off=off, end=1.0, pow=power)
    raise NotImplementedError(
        f"Bruker window WDW={wdw} ({_WDW_NAMES.get(wdw, 'unknown')}) is not yet "
        f"supported for raw-FID reprocessing (supported: 0/none, 1/EM, 3/SINE, "
        f"4/QSINE). Use read_1d_processed/read_2d_processed instead, or extend "
        f"_apply_bruker_window once you know this WDW code's parameters."
    )


# Bruker's own PH_mod codes, from procs/proc2s -- how TopSpin itself
# displays this dataset.
_PH_MOD_NONE = 0     # "no"  -- no phase correction
_PH_MOD_PK = 1       # "pk"  -- apply the recorded PHC0/PHC1
_PH_MOD_MC = 2       # "mc"  -- magnitude calculation
_PH_MOD_PS = 3       # "ps"  -- power spectrum

_PH_MOD_NAMES = {
    _PH_MOD_NONE: "no (no phase correction)",
    _PH_MOD_PK: "pk (recorded PHC0/PHC1)",
    _PH_MOD_MC: "mc (magnitude)",
    _PH_MOD_PS: "ps (power)",
}


def _resolve_magnitude_mode(magnitude: bool | None, *procs_dicts) -> str | None:
    """Decide whether to produce a magnitude/power spectrum, and which.

    Returns ``"magnitude"``, ``"power"``, or ``None`` (leave the
    spectrum phase-sensitive).

    ``magnitude`` is the caller's explicit override: ``True`` forces a
    magnitude spectrum for *any* dataset (the GUI's own "Magnitude
    mode" toggle -- useful for a badly-phased or hard-to-phase
    spectrum regardless of what the vendor recorded), ``False`` forces
    it off even when the file asks for it, and ``None`` (the default)
    follows whatever TopSpin itself recorded in ``PH_mod``.

    Real, and the reason this exists: magnitude mode is genuinely used
    in this project's own test data -- 9 Bruker datasets record
    ``PH_mod=2``, mostly gradient-COSY experiments (e.g.
    ``exam_CMCse_3/12``, ``cosygpmfppqf``). Ignoring it made those load
    as an unphased, dispersive-looking phase-sensitive spectrum that
    correlated **+0.01** with TopSpin's own processed output, against
    **+0.86** for the magnitude of the same complex data -- which is
    what the "contours are inverted / don't match" report turned out to
    be about for that class of file.

    Any of the supplied parameter dicts asking for magnitude/power is
    enough (for a 2D, F2's ``procs`` and F1's ``proc2s`` are both
    consulted, since either dimension can carry the flag -- the COSY
    above records it on F1 while F2 says ``no``). Pass only the dicts
    that actually apply: a pseudo-2D's ``proc2s`` describes an indirect
    dimension that was never transformed, so it shouldn't vote.

    ``PH_mod=3`` (power) is supported for completeness but is
    **unverified against real data** -- no dataset available here
    records it, unlike magnitude.
    """
    if magnitude is not None:
        return "magnitude" if magnitude else None
    for procs in procs_dicts:
        if not procs:
            continue
        try:
            ph_mod = int(float(procs.get("PH_mod", _PH_MOD_NONE) or _PH_MOD_NONE))
        except (TypeError, ValueError):
            continue
        if ph_mod == _PH_MOD_MC:
            return "magnitude"
        if ph_mod == _PH_MOD_PS:
            return "power"
    return None


def _apply_magnitude_mode(spectrum: np.ndarray, mode: str | None) -> np.ndarray:
    """``abs()`` for ``"magnitude"``, ``abs()**2`` for ``"power"``,
    unchanged for ``None``. Applied at the very *end* of processing,
    after windowing/FFT/phasing in every dimension -- which is where it
    was verified to reproduce TopSpin's own output (taking it earlier,
    e.g. between F2's and F1's transforms, is a different and wrong
    quantity)."""
    if mode == "magnitude":
        return np.abs(spectrum)
    if mode == "power":
        return np.abs(spectrum) ** 2
    return spectrum


def _read_procs(datapath: str, procno: int, proc2: bool = False) -> dict:
    filename = "proc2s" if proc2 else "procs"
    return parse_jcampdx_params(os.path.join(datapath, "pdata", str(procno), filename))


def _ppm_hz_axis(o1: float, sf: float, obs: float, sw: float, si: int) -> tuple[np.ndarray, np.ndarray]:
    """The ppm/Hz axis formula used throughout ``nmrtools2`` -- kept
    identical here (for both the raw-FID and processed-fallback paths)
    so a raw-vs-processed comparison on the same dataset is a
    comparison of intensities on a matched axis, not also two
    different axis constructions."""
    sr = (sf - obs) * 1e6
    true_centre = o1 - sr
    x_hz = np.linspace(true_centre + sw / 2, true_centre - sw / 2, num=int(si))
    x_ppm = x_hz / sf
    return x_ppm, x_hz


# ============================================================
# Raw FID -- 1D
# ============================================================

def read_1d(path: str, procno: int = 1, mass: float = 1.0, f1p: float = 0.0, f2p: float = 0.0,
            autophase: bool = True, phase_mode: str = "recorded", phase0: float = 0.0, phase1: float = 0.0,
            pivot_frac: float = 0.5,
            phase0_inc: float = 0.1, phase1_max: float = 20.0, phase1_improvement: float = 0.15,
            phase_penalty: str = "magnitude",
            window: str | None = None, window_kwargs: dict | None = None, zero_fill: int | None = None,
            fid_shift: float = 0.0, align_echo: bool = False,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward",
            magnitude: bool | None = None) -> NMRSpectrum:
    """Read the raw ``fid`` and reprocess it using the parameters
    TopSpin itself recorded in ``procs`` (window, ``SI``, and --
    ``phase_mode="recorded"``, the default -- the phase TopSpin itself
    already applied, ``PHC0``/``PHC1``).

    ``phase_mode`` : ``"recorded"`` (default) reproduces whatever
    phase was already showing in TopSpin for this file -- either from
    a user's own careful manual phasing, or TopSpin's own automatic
    processing, both real prior work this shouldn't silently discard
    (a user who already prepared/phased their data should not have to
    redo it just to open the file here). ``"apc0"``/``"joint"`` guess
    a fresh phase from scratch instead (zero-order grid search, or
    joint zero+first-order) -- useful for a raw/never-processed
    acquisition, or to deliberately re-derive a phase independent of
    whatever TopSpin recorded.

    ``pivot_frac`` : where ``phase1`` pivots (0..1 across the ppm
    range, 0.5 = centre) -- matters whenever ``phase1`` is nonzero,
    including a caller-supplied one with ``autophase=False`` (e.g. the
    GUI reprocessing from raw FID with a manually-dialed-in phase, so
    the result matches exactly rather than re-pivoting at centre).

    ``f1p``/``f2p`` : ppm crop window applied after processing (order
    doesn't matter; both zero imports the full spectrum), matching
    ``nmrtools2``'s ``NMR1dimport``.

    ``window``/``window_kwargs`` : override TopSpin's own recorded
    ``WDW``/``LB``/``SSB`` with one of :data:`opennmr.processing.ft`'s
    generic window functions instead (``None``, the default, keeps
    TopSpin's own choice). ``zero_fill`` similarly overrides the
    recorded ``SI`` (``None`` keeps it).

    ``fid_shift``/``align_echo``/``linear_predict_points`` : applied to
    the raw FID, in that order, before windowing -- see
    :mod:`opennmr.processing.ft`'s ``roll_fid``/``align_echo``/
    ``linear_predict`` for what each does.
    """
    # read_pulseprogram=False: we never use dic["pprog"], and reading
    # it is slow to fail (and warns) whenever the pulse program can't
    # be found/parsed (renamed/custom sequences, older datasets, ...).
    # It also feeds nmrglue's own shape-guessing for 2D+ data (see
    # read_2d) -- not a concern here since 1D shape-guessing doesn't
    # need it, but skipped uniformly for the same reason either way.
    # Parameters via nmrglue's own acqus reader, but the binary `fid`
    # via this module's :func:`_read_bruker_binary` -- same reasoning
    # (and same verified-identical result) as the 2D path in
    # :func:`_read_raw_2d_data`. 1D has no shape to guess, so it never
    # produced the reshape warning that motivated the 2D change; done
    # here too so both paths read their binary the same single way
    # rather than splitting into two conventions that could drift.
    dic = ng.bruker.read_acqus_file(path)
    data = _read_bruker_binary(os.path.join(path, "fid"), dic)
    fid = _remove_digital_filter(dic, data).astype(np.complex128)
    fid = np.conj(fid)  # see _CONJUGATE_QUADRATURE_NOTE below
    fid_raw = fid  # the truly raw FID -- see metadata.raw["fid_raw"]'s own comment below
    original_length = len(fid)  # NP: acquired points, before shift/align/LP/window/zero-fill

    acqus = dic["acqus"]
    procs = _read_procs(path, procno)

    ns = float(acqus.get("NS", 1.0))
    obs = float(acqus.get("BF1"))
    o1 = float(acqus.get("O1"))
    sf = float(procs.get("SF"))
    sw = float(procs.get("SW_p"))
    si = int(zero_fill) if zero_fill is not None else int(procs.get("SI", len(fid)))
    dt = 1.0 / sw

    # Shift -> echo-align -> linear-predict (no truncate/TDeff concept
    # for Bruker -- see processing.ft.prepare_fid's own docstring for
    # the full step-ordering rationale, shared with X-Pulse's read_1d
    # and the GUI's own live Processing-preview).
    fid = processing_ft.prepare_fid(
        fid, fid_shift=fid_shift, align_echo_enabled=align_echo,
        linear_predict_points=linear_predict_points, linear_predict_order=linear_predict_order,
        linear_predict_direction=linear_predict_direction,
    )

    window_unsupported_note = None
    if window is not None:
        fid = processing_ft.apodize(fid, dt, window=window, **(window_kwargs or {}))
    else:
        try:
            fid = _apply_bruker_window(fid, procs, sw)
        except NotImplementedError as exc:
            # Stay on the raw-FID path (full reprocessing still
            # available -- see the module docstring) rather than
            # abandoning it for TopSpin's own, non-reprocessable
            # output just because this one recorded window isn't
            # understood yet; the FID is simply left unwindowed.
            window_unsupported_note = str(exc)
    fid = processing_ft.zero_fill(fid, si)

    _, spectrum = processing_ft.fft_spectrum(fid, dt)
    x_ppm, x_hz = _ppm_hz_axis(o1, sf, obs, sw, si)

    recorded_f2 = (0.0, 0.0)
    if autophase:
        if phase_mode == "joint":
            phase0, phase1 = processing_ft.autophase01(
                spectrum, x_ppm, phase1_max=phase1_max, phase1_improvement=phase1_improvement,
                penalty=phase_penalty, phase0_inc=phase0_inc,
            )
        elif phase_mode == "apc0":
            phase0 = processing_ft.apc0(spectrum, x_ppm, phase0_inc)
        elif phase_mode == "recorded":
            # Kept as the *recorded* phase, in TopSpin's own model,
            # rather than folded into phase0/phase1 (the user's own, in
            # the pivot model) -- see apply_dimension_phase for why the
            # two must not be merged into one number.
            recorded_f2 = (float(procs.get("PHC0", 0.0)), float(procs.get("PHC1", 0.0)))
        else:
            raise ValueError("phase_mode must be 'apc0', 'joint', or 'recorded'")

    spectrum = apply_dimension_phase(spectrum, x_ppm, recorded_f2, (phase0, phase1), pivot_frac=pivot_frac)
    spectrum = spectrum / mass / ns

    if f1p != 0 or f2p != 0:
        lo, hi = sorted((f1p, f2p))
        keep = (x_ppm >= lo) & (x_ppm <= hi)
        x_ppm, x_hz, spectrum = x_ppm[keep], x_hz[keep], spectrum[keep]

    metadata = NMRMetadata(
        vendor="bruker", nucleus=acqus.get("NUC1"), sfo1=sf, bf1=obs, o1=o1, ns=ns,
        source_path=path, procno=procno, probe=acqus.get("PROBHD"), pulse_program=acqus.get("PULPROG"),
        raw={
            "acqus": acqus, "procs": procs, "phase0": phase0, "phase1": phase1,
            "recorded_phase0": recorded_f2[0], "recorded_phase1": recorded_f2[1],
            # The exact (shift/echo-align/LP/windowed/zero-filled) FID
            # fed into the FFT above -- an optional GUI preview of what
            # FID-domain processing actually did, kept alongside (not
            # instead of) the frequency-domain result, which stays the
            # default view.
            "fid": fid, "fid_dt": dt, "fid_original_length": original_length,
            # The *truly* raw FID -- straight off disk (post digital-
            # filter removal only), before even shift/align/LP, let
            # alone windowing/zero-fill. Lets the GUI recompute a live
            # preview of a not-yet-applied processing choice entirely
            # in memory (processing.ft.prepare_fid + apodize/zero_fill/
            # fft_spectrum/apply_phase, the same steps this function
            # itself just ran) -- unlike "fid" above, which already has
            # whatever was chosen at *this* load baked in, so isn't
            # safe to reprocess with a *different* choice without
            # double-applying the old one.
            "fid_raw": fid_raw,
            **({"window_unsupported": window_unsupported_note} if window_unsupported_note else {}),
        },
    )
    axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)
    # Magnitude/power last of all, after phasing -- see
    # _apply_magnitude_mode. Only F2's own procs votes for a 1D.
    magnitude_mode = _resolve_magnitude_mode(magnitude, procs)
    if magnitude_mode:
        spectrum = _apply_magnitude_mode(spectrum, magnitude_mode)
        metadata.raw["magnitude_mode"] = magnitude_mode
        metadata.raw["magnitude_mode_source"] = "requested" if magnitude is not None else "recorded (PH_mod)"
        # A magnitude/power spectrum has no meaningful phase left, so
        # there is no imaginary part to retain -- and data_imag=None is
        # already how the GUI knows to disable its phase controls
        # ("Phase adjustment needs a spectrum with retained imaginary
        # data"), rather than offering a correction that cannot do
        # anything.
        return NMRSpectrum(data=np.real(spectrum), axes=[axis], metadata=metadata, data_imag=None)
    return NMRSpectrum(data=np.real(spectrum), axes=[axis], metadata=metadata, data_imag=np.imag(spectrum))


# ============================================================
# Raw FID -- 2D
# ============================================================

def read_2d(path: str, procno: int = 1, homonuclear: bool = False, pseudo_2d: bool | None = None,
            autophase: bool = True, phase_mode: str = "recorded", phase0_inc: float = 0.1,
            phase0: float = 0.0, phase1: float = 0.0, pivot_frac: float = 0.5,
            window: str | None = None, window_kwargs: dict | None = None, zero_fill: int | None = None,
            window_f1: str | None = None, window_kwargs_f1: dict | None = None,
            zero_fill_f1: int | None = None, truncate_f1: int | None = None,
            fid_shift: float = 0.0, align_echo: bool = False,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward",
            magnitude: bool | None = None) -> NMRSpectrum:
    """Read the raw ``ser`` and reprocess it (F2 then F1) using the
    parameters TopSpin recorded in ``procs``/``proc2s``.

    Raw ``ser`` shape comes from :func:`_guess_raw_2d_shape`/
    :func:`_read_raw_2d_data` -- not ``nmrglue.bruker.read``'s own
    shape-guessing, which needs the pulse program's own loop/increment
    structure for anything beyond the simplest case (this project
    deliberately skips reading it) and has a real DTYPA-related bug of
    its own even when it is read (see :func:`_guess_raw_2d_shape`'s
    own docstring for both, and TODO.md for the real-data investigation
    that found them). This is the higher-risk half of the raw-FID path
    (more moving parts than 1D) -- validate against
    :func:`read_2d_processed` on real data before trusting it for
    phase-sensitive work (see the project plan's Verification section).
    Performs States/States-TPPI/Echo-Antiecho hypercomplex row-pair
    decoding before the F1 FFT when ``acqu2s``'s own ``FnMODE`` calls
    for it -- see :func:`opennmr.processing.ft.decode_hypercomplex_f1`
    for the exact per-mode recipe. States-TPPI is validated against 5
    real Bruker datasets; Echo-Antiecho is now also validated against a
    real dataset (``test_data/Bruker Example Data/Cyclosporine/203``,
    2026-09-02 -- see this function's own ``fn_mode == 6`` comment
    below and :func:`opennmr.processing.ft.decode_hypercomplex_f1`'s
    docstring for what that took).

    ``pseudo_2d`` : whether F1 is a non-frequency indirect dimension
    (no FFT along that axis -- a slice/experiment index instead), i.e.
    a relaxation/diffusion series rather than a conventional 2D
    spectrum. ``None`` (default) auto-detects from ``acqu2s``'s
    ``NUC1`` (Bruker's own convention: ``<off>`` for pseudo-2D). Pass
    ``True``/``False`` explicitly to override the detection -- useful
    since it depends on ``acqu2s`` being present and conventionally
    populated, which isn't guaranteed for every pulse sequence/Bruker
    software version.

    ``autophase``/``phase_mode`` : zero-order only, uniformly across
    the whole F2 dimension -- ``phase_mode="recorded"`` (default)
    reproduces F2's own recorded ``PHC0`` (see :func:`read_1d`'s own
    ``phase_mode`` docs for the full "don't discard the user's own
    processing" rationale, identical here); ``"apc0"`` guesses fresh
    via a grid search instead. ``phase0``/``phase1``/``pivot_frac`` :
    applied (via :func:`opennmr.processing.ft.apply_phase`, uniformly
    across F2) whenever ``autophase=False`` -- lets a manually-dialed-in
    phase correction (from the GUI's Phase tab) survive an "Apply
    Processing" click instead of being silently reset. If both
    ``autophase=True`` and a nonzero ``phase1`` are given, ``phase0``
    is still recomputed per ``phase_mode`` but ``phase1``/``pivot_frac``
    are applied on top of it (neither ``recorded`` nor ``apc0`` touch
    ``phase1`` here -- F1's own phase, when relevant, is a separate
    concern this function doesn't otherwise address).

    ``window``/``window_kwargs``/``zero_fill``/``fid_shift``/
    ``align_echo``/``linear_predict_points`` : see :func:`read_1d` --
    identical meaning, but only ever applied along F2 (the direct
    dimension, one row at a time).

    ``window_f1``/``window_kwargs_f1`` : same idea for F1 (the indirect
    dimension, one column -- i.e. one already-F2-processed row's worth
    of complex points across every t1 increment -- at a time), applied
    *after* :func:`opennmr.processing.ft.decode_hypercomplex_f1` and
    before F1's own zero-fill/FFT, same ordering as F2's own
    window-then-zero-fill-then-FFT. ``None`` (default, matching
    ``window``'s own convention) reproduces TopSpin's own recorded
    ``proc2s`` ``WDW``/``LB``/``SSB`` via :func:`_apply_bruker_window`
    -- **this was previously silently unimplemented** (this docstring
    used to claim F1 "always follows TopSpin's own recorded proc2s",
    which the code never actually did -- F1 got no apodization
    whatsoever, recorded or otherwise) **and is now applied**, found
    via the same real Echo-Antiecho dataset that surfaced the
    ``fn_mode == 6`` conjugate fix above (its own ``proc2s`` records
    ``WDW=4``/``SSB=2``, QSINE, same as F2). An unsupported recorded
    ``proc2s`` ``WDW`` code degrades the same way F2's own does --
    leaves F1 unwindowed and records
    ``metadata.raw["window_unsupported_f1"]`` -- rather than failing
    the whole load.

    ``zero_fill_f1`` : ``None`` (default) uses ``proc2s``'s own
    recorded ``SI`` (TopSpin's own F1 zero-fill target), falling back
    to the row count actually present if that's missing; an ``int``
    overrides it explicitly. ``truncate_f1`` : ``None`` (default) uses
    every row/row-pair present; an ``int`` keeps only the first that
    many F1 increments -- applied to raw row *pairs* (not individual
    rows) whenever ``FnMODE`` calls for hypercomplex decoding, so a
    request never separates a real row from its matching imaginary
    one, same convention as X-Pulse's own
    :func:`opennmr.io.xpulse.read_2d`.
    """
    # Shape comes from _guess_raw_2d_shape/_read_raw_2d_data (own
    # TD/DTYPA/file-size-based logic, see their own docstrings) rather
    # than nmrglue's own shape-guessing -- that needs the pulse
    # program's own loop/increment structure for anything beyond the
    # simplest case, and even reading it doesn't resolve every real
    # pulse program's own loop-variable naming (see TODO.md). Still
    # uses nmrglue for the acqus/acqu2s parameter dicts themselves
    # (read_acqus_file, no pulse-program dependency) and the actual
    # digital-filter removal below (remove_digital_filter).
    dic = ng.bruker.read_acqus_file(path)
    data = _read_raw_2d_data(path, dic)
    data = _remove_digital_filter(dic, data).astype(np.complex128)
    data = np.conj(data)  # see _CONJUGATE_QUADRATURE_NOTE below
    fid_raw = data  # the truly raw FID (every row/slice) -- see metadata.raw["fid_raw"]'s own comment below
    original_length = data.shape[1]  # NP (F2): acquired points, before shift/align/LP/window/zero-fill

    acqus = dic["acqus"]
    procs = _read_procs(path, procno)
    proc2s = _read_procs(path, procno, proc2=True)

    obs = float(acqus.get("BF1"))
    o1 = float(acqus.get("O1"))
    sf = float(procs.get("SF"))
    sw = float(procs.get("SW_p"))
    si2 = int(zero_fill) if zero_fill is not None else int(procs.get("SI", data.shape[1]))
    ns = float(acqus.get("NS", 1.0))

    nuc2 = acqus.get("NUC1") if homonuclear else dic.get("acqu2s", {}).get("NUC1")
    if pseudo_2d is None:
        pseudo_2d = (nuc2 == "off")

    if homonuclear:
        obs_1, o1_1, sf_1, sw_1 = obs, o1, sf, sw
    else:
        acqu2s = dic.get("acqu2s", {})
        obs_1 = float(acqu2s.get("BF1", obs))
        o1_1 = float(acqu2s.get("O1", o1))
        sf_1 = float(proc2s.get("SF", sf))
        sw_1 = float(proc2s.get("SW_p", sw))
    # Governs whether/how F1 row-pairs get combined into genuine
    # hypercomplex points below, before F1's own FFT -- see
    # processing.ft.decode_hypercomplex_f1's own docstring for the
    # exact per-mode recipe and what's/isn't validated against real
    # data. Defaults to 1 (QF) -- a plain complex-per-row F1, needing
    # no row-pairing -- matching every genuine (non-pseudo) 2D dataset
    # actually in this project's test_data that doesn't record FnMODE
    # explicitly.
    fn_mode = int(dic.get("acqu2s", {}).get("FnMODE", 1))

    # Echo-Antiecho (FnMODE 6, a.k.a. Rance-Kay) needs one more thing
    # *before* F2 processing even starts: the "echo" (even-indexed) raw
    # row of each pair must be conjugated *and time-reversed* first.
    # This was flagged in decode_hypercomplex_f1's own docstring as
    # unverified/best-effort (no real Echo-Antiecho data existed in
    # this project to check against) -- verified against a real
    # dataset (test_data/Bruker Example Data/Cyclosporine/203,
    # FnMODE=6) in two rounds, the first of which shipped a real,
    # user-reported bug (worth understanding, not just the fix itself):
    #
    # Round 1 found that the plain sum/diff combine (no raw-row
    # transform at all) reproduced every real peak at its correct
    # position but *also* produced an equal-magnitude ghost at that
    # peak's F1-mirror position (ratio ~0.8-1.2x the genuine peak) --
    # exactly the "mirror across the spectrum's centre" failure mode
    # that docstring predicted. Conjugating the even-indexed raw row
    # (no reversal) dropped that ghost to ~0.005-0.02 (noise floor) on
    # F1 -- shipped as the fix. **But conjugating a raw time-domain row
    # doesn't just fix the F1/hypercomplex phase relationship -- FFT
    # doesn't commute with conjugation (FFT(conj(x))[k] ==
    # conj(FFT(x)[-k])), so it also *reverses that row's own F2
    # frequency axis*, an unavoidable side effect neither round 1's
    # verification nor its own reasoning checked for** (it only
    # measured the F1-mirror ratio, not an F2 one) -- confirmed
    # directly: the lone-conjugate fix's own F1 ratio was excellent
    # (~0.01) but its F2 ratio was ~0.95, an equal-magnitude ghost at
    # each peak's F2-mirror position instead. Real user report caught
    # this ("I'm pretty sure the echo-anti echo data is now mirrored
    # in x") after the lone-conjugate version had already shipped.
    #
    # Round 2's fix: conjugate *and* reverse the even row (order
    # doesn't matter -- conjugation is elementwise, reversal is a pure
    # reordering, so they commute) -- the two frequency-reversals
    # (one from the conjugate, one from the explicit reversal) cancel,
    # leaving F2 untouched, while the conjugate's own value-negation
    # still supplies the hypercomplex relationship F1 needs. Verified
    # on the same real dataset: both F1 and F2 mirror ratios land at
    # ~0.005-0.01 (noise floor) across the 10 strongest real peaks,
    # simultaneously -- checked together this time, specifically
    # because round 1's own gap was "fixed and verified one axis
    # without checking the other actually stayed fixed too." A copy,
    # not an in-place transform, so ``metadata.raw["fid_raw"]``
    # (assigned above from this same ``data`` binding) stays genuinely
    # unmodified.
    if fn_mode == 6:
        data = data.copy()
        data[0::2] = np.conj(data[0::2])[:, ::-1]

    # ----- F2 (direct dimension): [shift/align/LP], window, zero-fill, FFT per row -----
    # Same shared step (and step order) as read_1d/xpulse.read_1d --
    # see processing.ft.prepare_fid's own docstring. Skipped entirely
    # (real, measured regression, not a style choice: ~1200x slower
    # for a synthetic 32x16384 benchmark, confirmed before fixing it)
    # when none of these three are actually requested -- the common
    # case for a fresh load -- since np.apply_along_axis pays real
    # per-row Python-level function-call overhead even when the
    # function it calls is a no-op for every row, unlike a plain
    # `data` reference. This one-line guard is exactly what the
    # pre-shared-prepare_fid code already did (three separate `if`s,
    # one per step) -- lost when they were combined into one call.
    if fid_shift or align_echo or linear_predict_points:
        # align_echo's own auto-truncation caps each row at its own
        # len(row) - ceil(echo_top) (see prepare_fid's docstring) --
        # fine for a single 1D FID, but here every row is independently
        # capped by np.apply_along_axis's per-row prepare_fid call, and
        # a real (non-imaging) pseudo-2D series' rows rarely have a
        # genuine echo shape at all, so find_echo_top lands on a
        # different, essentially noise-driven index each time. The
        # resulting per-row lengths then disagree, which
        # np.apply_along_axis cannot reconcile into one array (raises
        # ValueError: could not broadcast...) -- found on real T2 data
        # (bruker_T2/11) via Align Echo, which every Bruker 2D/pseudo-
        # 2D spectrum offers unconditionally regardless of whether its
        # rows are actually echo-shaped. Precompute one shared cap (the
        # minimum across every row) and pass it in as an explicit
        # truncate= for all of them, so every row ends up the same
        # final length regardless of where each one's own echo top
        # happened to land.
        truncate_cap = None
        if align_echo:
            echo_wrap_limits = [
                row.shape[0] - int(np.ceil(processing_ft.find_echo_top(row))) for row in data
            ]
            truncate_cap = min(echo_wrap_limits)
        fid_f2 = np.apply_along_axis(
            lambda row: processing_ft.prepare_fid(
                row, fid_shift=fid_shift, align_echo_enabled=align_echo, truncate=truncate_cap,
                linear_predict_points=linear_predict_points, linear_predict_order=linear_predict_order,
                linear_predict_direction=linear_predict_direction,
            ),
            1, data,
        )
    else:
        fid_f2 = data
    window_unsupported_note = None
    if window is not None:
        # Whole-array calls, not np.apply_along_axis -- window/zero_fill/
        # fft_spectrum all operate on the last axis (F2's own axis
        # already), batched across every row in one call. Confirmed via
        # profiling (real 2048x65536 EXSY load, 9.3s total) that these
        # six apply_along_axis calls (F2's three plus F1's three below)
        # accounted for 67% of read_2d's own wall time, almost entirely
        # np.apply_along_axis's own per-call Python overhead across
        # 66,000+ single-row/column calls rather than actual FFT cost --
        # see TODO.md's "Big-data loading performance" entry.
        fid_f2 = processing_ft.apodize(fid_f2, 1.0 / sw, window=window, **(window_kwargs or {}))
    else:
        try:
            fid_f2 = _apply_bruker_window(fid_f2, procs, sw)
        except NotImplementedError as exc:
            # Same reasoning as read_1d -- stay on the raw-FID path
            # (full reprocessing still available) rather than
            # abandoning it; the same WDW code applies to every row,
            # so this fails (or doesn't) for the whole batch at once.
            window_unsupported_note = str(exc)
    fid_f2 = processing_ft.zero_fill(fid_f2, si2)
    _, spectrum = processing_ft.fft_spectrum(fid_f2, 1.0 / sw)

    # F2's ppm axis, computed now (F1 processing below only changes
    # axis 0's length, never F2's point count) since a phase1/pivot
    # correction needs it.
    x_ppm, x_hz = _ppm_hz_axis(o1, sf, obs, sw, spectrum.shape[1])
    recorded_f2 = (0.0, 0.0)
    if autophase:
        if phase_mode == "recorded":
            # F2's recorded phase, kept separate from the user's own --
            # see apply_dimension_phase. Note PHC1 is now honoured here;
            # it used to be dropped entirely for 2D (read_1d always read
            # both), which left every 2D missing TopSpin's own
            # first-order term.
            recorded_f2 = (float(procs.get("PHC0", 0.0)), float(procs.get("PHC1", 0.0)))
        elif phase_mode == "apc0":
            phase0 = processing_ft.apc0(spectrum.mean(axis=0), x_ppm, phase0_inc)
        else:
            raise ValueError("phase_mode must be 'apc0' or 'recorded'")
    spectrum = apply_dimension_phase(spectrum, x_ppm, recorded_f2, (phase0, phase1), pivot_frac=pivot_frac)

    # ----- F1 (indirect dimension): hypercomplex decode, zero-fill, FFT along axis 0 (skipped for pseudo-2D) -----
    window_unsupported_note_f1 = None
    recorded_f1 = (0.0, 0.0)
    if pseudo_2d:
        y_index = np.arange(1, spectrum.shape[0] + 1)
        y_axis = NMRAxis(kind="index", index=y_index)
    else:
        # truncate_f1 -- applied to raw row *pairs* when FnMODE calls
        # for hypercomplex decoding (fn_mode 4/5/6: 2 raw rows per t1
        # increment), else directly (fn_mode 0-3: 1 raw row per
        # increment already, e.g. every genuine 2D dataset actually in
        # this project's own test_data, which is all FnMODE 1/QF) --
        # same convention as X-Pulse's own read_2d.
        if truncate_f1 is not None:
            n1_native = spectrum.shape[0] // 2 if fn_mode in (4, 5, 6) else spectrum.shape[0]
            if 0 < truncate_f1 < n1_native:
                raw_row_count = 2 * truncate_f1 if fn_mode in (4, 5, 6) else truncate_f1
                spectrum = spectrum[:raw_row_count]
        # Row-pair decode (States/States-TPPI/Echo-Antiecho) *before*
        # zero-fill/FFT -- a no-op for FnMODE 0-3 (see
        # decode_hypercomplex_f1's own docstring). Halves the row count
        # for FnMODE 4/5/6, so proc2s's own recorded SI (already in
        # terms of the *decoded* point count, matching what TopSpin
        # itself zero-fills to) is used as-is absent an explicit
        # zero_fill_f1 override, but the fallback -- for when SI isn't
        # recorded at all -- must fall back to the decoded row count,
        # not the raw pre-decode one.
        # Normalise Bruker's own F1 quadrature sense to the canonical
        # one before decoding, rather than teaching the shared decoder
        # about vendors: negating the sine-modulated rows is exactly
        # `a - 1j*b` (see decode_hypercomplex_f1's own docstring for how
        # that form was measured against TopSpin).
        # States / States-TPPI only (4, 5). Echo-Antiecho (6) uses a
        # different recombination that this sign is not part of --
        # applying it there flipped real EA data from +0.395 to
        # -0.380, caught by the real-data check rather than the
        # suite, which passed regardless.
        if fn_mode in (4, 5):
            spectrum = spectrum.copy()
            spectrum[1::2] *= -1
        spectrum = processing_ft.decode_hypercomplex_f1(spectrum, fn_mode)
        # Transpose so F1 is the last axis for the same whole-array
        # calls used for F2 above (window/zero_fill/fft_spectrum all
        # operate on shape[-1]), then transpose back -- avoids
        # np.apply_along_axis's per-column loop (65536 separate calls
        # on this project's real EXSY benchmark, each processing a
        # short column, so per-call overhead dominated even more here
        # than in F2's own loop).
        spectrum = spectrum.T
        if window_f1 is not None:
            spectrum = processing_ft.apodize(spectrum, 1.0 / sw_1, window=window_f1, **(window_kwargs_f1 or {}))
        else:
            try:
                spectrum = _apply_bruker_window(spectrum, proc2s, sw_1)
            except NotImplementedError as exc:
                # Same reasoning as F2's own WDW fallback above -- stay
                # on the raw-FID path (full reprocessing still
                # available) rather than abandoning it.
                window_unsupported_note_f1 = str(exc)
        si1 = int(zero_fill_f1) if zero_fill_f1 is not None else int(proc2s.get("SI", spectrum.shape[-1]))
        fid_f1 = processing_ft.zero_fill(spectrum, si1)
        _, spectrum = processing_ft.fft_spectrum(fid_f1, 1.0 / sw_1)
        spectrum = spectrum.T
        if fn_mode not in (4, 5, 6):
            # Real bug, found and fixed: for FnMODE 0-3 (QF and friends
            # -- decode_hypercomplex_f1 is a no-op for these, see its
            # own docstring), each row's own raw complex value is used
            # directly as F1's time-domain point. That value already
            # carries the module-level _CONJUGATE_QUADRATURE_NOTE fix
            # (np.conj(data), applied once up front) -- correct for F2
            # (confirmed against real data), but that fix's own
            # docstring explicitly flagged F1 as *assumed*, not
            # verified, "since every real dataset available while
            # building this was pseudo-2D, which never exercises the F1
            # FFT branch". This is the first genuine (non-pseudo) QF-
            # mode 2D dataset to actually exercise it, and the
            # assumption doesn't hold: conjugating a signal before its
            # own FFT reverses that FFT's own frequency-bin order
            # (FFT(conj(x))[k] == conj(FFT(x)[-k])), so F1 comes out
            # mirrored -- confirmed directly against real COSY data
            # (test_data/Bruker Example Data/exam_CMCse_3/12,
            # PULPROG cosygpmfppqf, FnMODE 1): read_2d_processed's own
            # (trusted) diagonal peaks sit at up to ~135x higher
            # intensity once F1 is reversed to match, at every one of 8
            # independently-checked positions along the diagonal (a
            # homonuclear COSY's own diagonal peaks are always at
            # F1==F2, an unambiguous, dataset-independent check). The
            # hypercomplex modes (4/5/6) don't need this: their own row-
            # pair recombination (decode_hypercomplex_f1) evidently
            # already produces the correct F1 direction as a side
            # effect of its own construction (already validated
            # end-to-end against real States-TPPI/Echo-Antiecho data
            # elsewhere in this project's history) -- this fix is
            # deliberately scoped to leave that path untouched.
            spectrum = spectrum[::-1, :]
        # F1's own recorded phase. This dimension previously had no
        # recorded-phase step at all -- a real gap, reported as "the
        # indirect phase is for sure not loading across from
        # processing". Applied directly rather than through
        # apply_dimension_phase: this reader takes no F1 *user* phase
        # (the GUI owns that, and applies both together on its own
        # rebuild path), so there is nothing here to compose with.
        if autophase and phase_mode == "recorded":
            recorded_f1 = (float(proc2s.get("PHC0", 0.0)), float(proc2s.get("PHC1", 0.0)))
            spectrum = apply_recorded_phase(spectrum, recorded_f1[0], recorded_f1[1], axis=0)
        y_ppm, y_hz = _ppm_hz_axis(o1_1, sf_1, obs_1, sw_1, si1)
        y_axis = NMRAxis(kind="frequency", ppm=y_ppm, hz=y_hz)

    x_axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)
    spectrum = spectrum / ns  # match read_1d's convention: data is always per-scan

    metadata = NMRMetadata(
        vendor="bruker", nucleus=acqus.get("NUC1"), nucleus_indirect=nuc2,
        sfo1=sf, sfo1_indirect=(None if pseudo_2d else sf_1), bf1=obs, o1=o1, ns=ns,
        source_path=path, procno=procno, pseudo_2d=pseudo_2d, homonuclear=homonuclear,
        probe=acqus.get("PROBHD"), pulse_program=acqus.get("PULPROG"),
        raw={
            "acqus": acqus, "procs": procs, "proc2s": proc2s, "phase0": phase0, "phase1": phase1,
            "recorded_phase0": recorded_f2[0], "recorded_phase1": recorded_f2[1],
            "recorded_phase0_f1": recorded_f1[0], "recorded_phase1_f1": recorded_f1[1],
            # F1's own hypercomplex encoding (see
            # processing.ft.decode_hypercomplex_f1) -- retrievable so a
            # live GUI recompute that redoes F1's FFT (any F2-only
            # control change, for a conventional 2D) can redo this
            # decode too, rather than silently reverting to plain
            # complex-per-row once F2's own controls change.
            "fn_mode": fn_mode,
            # F2's fully processed (pre-FFT) FID, one row per F1
            # position -- an optional GUI preview of FID-domain
            # processing; F1 isn't included (its own FFT, when not
            # pseudo-2D, isn't overridable the way F2's is -- see
            # this function's docstring).
            "fid": fid_f2, "fid_dt": 1.0 / sw, "fid_original_length": original_length,
            # F1's own dt -- lets a live, in-memory recompute redo F1's
            # FFT (needed whenever F2's own reprocessing changes, since
            # F1's FFT operates across the rows F2's own FFT just
            # produced) without re-deriving sw_1 from the ppm axis.
            # None for pseudo-2D, where F1 has no FFT at all.
            "fid_dt_f1": (None if pseudo_2d else 1.0 / sw_1),
            # The *truly* raw FID, every row -- see read_1d's own
            # "fid_raw" comment for why this is cached separately from
            # "fid" above (safe to reprocess a *different*, not-yet-
            # committed choice from, without double-applying the one
            # already baked into "fid").
            "fid_raw": fid_raw,
            **({"window_unsupported": window_unsupported_note} if window_unsupported_note else {}),
            **({"window_unsupported_f1": window_unsupported_note_f1} if window_unsupported_note_f1 else {}),
        },
    )
    # Magnitude/power last of all, after both dimensions' own
    # windowing/FFT/phasing -- see _apply_magnitude_mode for why the
    # position matters. Both dimensions' parameter sets get a vote,
    # since either can carry the flag (the real COSY that motivated
    # this records mc on F1 while F2 says "no") -- but a pseudo-2D's
    # proc2s describes an indirect dimension that was never
    # transformed, so it is deliberately excluded there.
    magnitude_mode = _resolve_magnitude_mode(magnitude, procs, None if pseudo_2d else proc2s)
    if magnitude_mode:
        spectrum = _apply_magnitude_mode(spectrum, magnitude_mode)
        metadata.raw["magnitude_mode"] = magnitude_mode
        metadata.raw["magnitude_mode_source"] = "requested" if magnitude is not None else "recorded (PH_mod)"
        # See read_1d's own matching comment: no meaningful phase
        # remains, and data_imag=None is how the GUI disables phasing.
        return NMRSpectrum(data=np.real(spectrum), axes=[y_axis, x_axis], metadata=metadata, data_imag=None)
    return NMRSpectrum(data=np.real(spectrum), axes=[y_axis, x_axis], metadata=metadata, data_imag=np.imag(spectrum))


# ============================================================
# Already-processed fallback (also the raw-FID path's cross-check)
# ============================================================

def read_1d_processed(path: str, procno: int = 1, mass: float = 1.0, f1p: float = 0.0, f2p: float = 0.0,
                       heteronuclear_2d_slice: bool = False, slice_dimension: int = 2,
                       magnitude: bool | None = None) -> NMRSpectrum:
    """Read TopSpin's own already-processed ``1r``/``1i`` directly --
    ported from ``nmrtools2``'s ``NMR1dimport``."""
    real_path = os.path.join(path, "pdata", str(procno), "1r")
    imag_path = os.path.join(path, "pdata", str(procno), "1i")
    acqus_path = os.path.join(path, "acqus")

    if not os.path.exists(real_path):
        # Matches read_2d_processed's own 2rr guard -- this is the last
        # fallback _read_auto tries, so without a clear message here an
        # experiment with neither raw nor processed data fails with a
        # bare errno-2 from np.fromfile. Found by sweeping every Bruker
        # dataset in test_data: `.../2010` has acqus and a pdata/1
        # directory but no fid, no ser and no 1r -- an aborted or
        # never-processed acquisition.
        raise FileNotFoundError(
            f"No processed 1r data at {real_path}, and no raw fid/ser either -- this "
            f"experiment has parameter files but no acquired or processed spectrum "
            f"(an aborted or never-processed acquisition)."
        )

    if heteronuclear_2d_slice:
        if slice_dimension not in (2, 3):
            raise ValueError("slice_dimension must be 2 or 3")
        axis_acqus_path = os.path.join(path, f"acqu{slice_dimension}s")
    else:
        axis_acqus_path = acqus_path

    real = np.fromfile(real_path, dtype="int32", count=-1).astype(np.float64)
    imag = np.fromfile(imag_path, dtype="int32", count=-1).astype(np.float64)

    axis_meta = parse_jcampdx_params(axis_acqus_path)
    acq_meta = parse_jcampdx_params(acqus_path)

    nuc = axis_meta.get("NUC1")
    obs = float(axis_meta.get("BF1"))
    sfo1 = float(axis_meta.get("SFO1"))
    o1 = float(axis_meta.get("O1"))
    ns = float(acq_meta.get("NS", 1.0))

    procs = _read_procs(path, procno)
    sw = float(procs.get("SW_p"))
    si = float(procs.get("SI"))
    sf = float(procs.get("SF"))
    nc_proc = float(procs.get("NC_proc", 0.0))

    x_ppm, x_hz = _ppm_hz_axis(o1, sf, obs, sw, si)
    real = real * 2 ** nc_proc
    imag = imag * 2 ** nc_proc

    if f1p != 0 or f2p != 0:
        lo, hi = sorted((f1p, f2p))
        keep = (x_ppm >= lo) & (x_ppm <= hi)
        x_ppm, x_hz, real, imag = x_ppm[keep], x_hz[keep], real[keep], imag[keep]

    real = real / mass / ns
    imag = imag / mass / ns

    metadata = NMRMetadata(
        vendor="bruker", nucleus=nuc, sfo1=sf, bf1=obs, o1=o1, ns=ns, source_path=path, procno=procno,
        probe=acq_meta.get("PROBHD"), pulse_program=acq_meta.get("PULPROG"),
        raw={"acqus": axis_meta, "procs": procs},
    )
    axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)
    # Magnitude/power applies just the same to TopSpin's own already-
    # processed data as to the raw-FID path -- same PH_mod, same
    # explicit override -- so the toggle behaves identically whichever
    # way a dataset happened to load. (Found the hard way: without
    # this, `_read_auto` forwarded `magnitude=` into a reader that
    # didn't accept it and raised TypeError on every processed-fallback
    # dataset.)
    magnitude_mode = _resolve_magnitude_mode(magnitude, procs)
    if magnitude_mode:
        metadata.raw["magnitude_mode"] = magnitude_mode
        metadata.raw["magnitude_mode_source"] = "requested" if magnitude is not None else "recorded (PH_mod)"
        # See read_2d_processed's own matching block for why a missing
        # imaginary part is normal here rather than a defect.
        source = real + 1j * imag if imag is not None else real.astype(complex)
        if imag is None and magnitude is None:
            return NMRSpectrum(data=real, axes=[axis], metadata=metadata, data_imag=None)
        combined = _apply_magnitude_mode(source, magnitude_mode)
        return NMRSpectrum(data=np.real(combined), axes=[axis], metadata=metadata, data_imag=None)
    return NMRSpectrum(data=real, axes=[axis], metadata=metadata, data_imag=imag)


def read_2d_processed(path: str, procno: int = 1, homonuclear: bool = False,
                       pseudo_2d: bool | None = None, magnitude: bool | None = None) -> NMRSpectrum:
    """Read TopSpin's own already-processed ``2rr`` directly -- ported
    from ``nmrtools2``'s ``NMR2dimport``, including its Bruker XDIM
    submatrix-block reconstruction and ``GRPDLY`` roll, both of which
    apply only to this processed-data storage format (not to raw
    ``ser`` data, which :func:`read_2d` never needs them for).

    ``pseudo_2d`` : see :func:`read_2d` -- ``None`` (default)
    auto-detects from ``acqu2s``'s ``NUC1``; pass ``True``/``False``
    to override.
    """
    real_path = os.path.join(path, "pdata", str(procno), "2rr")
    imag_path = os.path.join(path, "pdata", str(procno), "2ri")
    acqus_path = os.path.join(path, "acqus")
    acqu2s_path = os.path.join(path, "acqu2s")

    acqus = parse_jcampdx_params(acqus_path)
    procs = _read_procs(path, procno)

    obs = float(acqus.get("BF1"))
    o1 = float(acqus.get("O1"))
    sf = float(procs.get("SF"))
    sw = float(procs.get("SW_p"))
    si1_val = float(procs.get("SI"))
    xdim_f2 = int(procs.get("XDIM", 0))
    grpdly = float(procs.get("GRPDLY", 0.0))

    # F1's own processed-data block size (SI/XDIM) always comes from
    # the real on-disk proc2s, regardless of homonuclear -- that flag
    # only says whether F1's *chemical-shift-axis* parameters
    # (nucleus/BF1/O1/SF/SW) should be looked up independently from
    # acqu2s/proc2s or borrowed from F2's own acqus/procs. Reading
    # proc2s only in the `else` branch (as this used to) silently
    # discarded a real SI/XDIM whenever homonuclear=True was passed
    # for a genuine 2D dataset that happens to observe the same
    # nucleus on both axes -- found on real 2H EXSY data, where F1's
    # SI (1024) was replaced by F2's own SI (65536), reshaping the
    # 2rr block into a shape many times too large to fit the file.
    try:
        proc2s = _read_procs(path, procno, proc2=True)
    except FileNotFoundError:
        proc2s = {}

    if homonuclear:
        nuc2, obs_1, o1_1, sf_1, sw_1 = acqus.get("NUC1"), obs, o1, sf, sw
    else:
        acqu2s = parse_jcampdx_params(acqu2s_path)
        nuc2 = acqu2s.get("NUC1")
        obs_1 = float(acqu2s.get("BF1"))
        o1_1 = float(acqu2s.get("O1"))
        sf_1 = float(proc2s.get("SF"))
        sw_1 = float(proc2s.get("SW_p"))

    si2_val = float(proc2s.get("SI", si1_val))
    xdim_f1 = int(proc2s.get("XDIM", 1)) if proc2s else 1

    if pseudo_2d is None:
        pseudo_2d = (nuc2 == "off")

    x_ppm, x_hz = _ppm_hz_axis(o1, sf, obs, sw, si1_val)
    if pseudo_2d:
        y_index = np.arange(1, int(si2_val) + 1)
        y_axis = NMRAxis(kind="index", index=y_index)
    else:
        y_ppm, y_hz = _ppm_hz_axis(o1_1, sf_1, obs_1, sw_1, si2_val)
        y_axis = NMRAxis(kind="frequency", ppm=y_ppm, hz=y_hz)

    si1, si2 = int(si1_val), int(si2_val)

    def _load_block(file_path: str) -> np.ndarray | None:
        if not os.path.exists(file_path):
            return None
        raw = np.fromfile(file_path, dtype="<i4")
        if xdim_f1 == 1:
            block = raw.reshape([si2, si1])
        else:
            submatrix_rows = si2 // xdim_f1
            submatrix_cols = si1 // xdim_f2
            submatrix_number = submatrix_cols * submatrix_rows
            blocks = np.array(np.array_split(raw, submatrix_number))
            blocks = np.reshape(blocks, (submatrix_rows, submatrix_cols, -1))
            block = np.vstack([
                np.hstack([np.reshape(c, (xdim_f1, xdim_f2)) for c in b])
                for b in blocks
            ])
        if grpdly != 0:
            block = np.roll(block, -int(np.floor(grpdly)), axis=1)
        return block.astype(np.float64)

    ns = float(acqus.get("NS", 1.0))
    spectrum = _load_block(real_path)
    if spectrum is None:
        # `_load_block` returns None for a missing file, which is the
        # right answer for the *optional* imaginary blocks (2ri/2ii --
        # a magnitude-mode dataset legitimately has none) but not for
        # the real block this function exists to read. Without this,
        # a `pdata/N` directory holding only parameter files produced
        # `AttributeError: 'NoneType' object has no attribute 'ndim'`
        # from deep inside NMRSpectrum's own constructor, with nothing
        # naming the file or the experiment. Real, pre-existing, and
        # found by sweeping every Bruker dataset in test_data:
        # `Bruker Example Data/Cyclosporine/206/pdata/1` has procs,
        # proc2s, clevels and a thumbnail but no 2rr at all.
        raise FileNotFoundError(
            f"No processed 2rr data at {real_path} -- this experiment's pdata/{procno} "
            f"directory has parameter files but no processed spectrum. Process it in "
            f"TopSpin first, pick a different procno, or load the raw ser instead."
        )
    spectrum_imag = _load_block(imag_path)
    if spectrum is not None:
        spectrum = spectrum / ns  # match read_1d_processed's convention: data is always per-scan
    if spectrum_imag is not None:
        spectrum_imag = spectrum_imag / ns

    metadata = NMRMetadata(
        vendor="bruker", nucleus=acqus.get("NUC1"), nucleus_indirect=nuc2,
        sfo1=sf, sfo1_indirect=(None if pseudo_2d else sf_1), bf1=obs, o1=o1, ns=ns,
        source_path=path, procno=procno, pseudo_2d=pseudo_2d, homonuclear=homonuclear,
        probe=acqus.get("PROBHD"), pulse_program=acqus.get("PULPROG"),
        raw={"acqus": acqus, "procs": procs, "proc2s": proc2s},
    )
    x_axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)
    # See read_1d_processed's own matching block. proc2s votes too,
    # except for a pseudo-2D whose indirect dimension was never
    # transformed -- same rule as read_2d.
    magnitude_mode = _resolve_magnitude_mode(magnitude, procs, None if pseudo_2d else proc2s)
    if magnitude_mode:
        metadata.raw["magnitude_mode"] = magnitude_mode
        metadata.raw["magnitude_mode_source"] = "requested" if magnitude is not None else "recorded (PH_mod)"
        if spectrum_imag is not None:
            combined = _apply_magnitude_mode(spectrum + 1j * spectrum_imag, magnitude_mode)
            return NMRSpectrum(data=np.real(combined), axes=[y_axis, x_axis], metadata=metadata, data_imag=None)
        # No imaginary part on disk. For a file that *records*
        # magnitude mode this is the normal case, not a defect --
        # TopSpin doesn't write a 2ri for one, because the data it
        # wrote is already the magnitude (confirmed on the real COSY:
        # every value in its 2rr is non-negative). So the mode is
        # recorded for the UI's benefit and the data is passed through
        # untouched rather than having abs() applied a second time.
        # An explicit user request still gets abs() applied, since a
        # real-only array is all there is to work with.
        if magnitude is not None:
            return NMRSpectrum(
                data=_apply_magnitude_mode(spectrum.astype(complex), magnitude_mode).real,
                axes=[y_axis, x_axis], metadata=metadata, data_imag=None,
            )
        return NMRSpectrum(data=spectrum, axes=[y_axis, x_axis], metadata=metadata, data_imag=None)
    return NMRSpectrum(data=spectrum, axes=[y_axis, x_axis], metadata=metadata, data_imag=spectrum_imag)


# ============================================================
# Variable-delay lists
# ============================================================

def read_vdlist(datapath: str, vdlist_name: str = "vdlist") -> dict:
    """Read a Bruker variable-delay list (``vdlist``), as used by
    pseudo-2D relaxation experiments to label each slice with the
    delay actually used. Ported unchanged from ``nmrtools2``.

    ``datapath`` : the Bruker experiment directory (``vdlist_name`` is
    then read from directly inside it, or ``lists/vd/<vdlist_name>``),
    or a direct path to a delay-list file anywhere on disk.

    Returns a dict with ``"name"``, ``"raw"`` (original strings, e.g.
    ``"10m"``), and ``"seconds"`` (ndarray).
    """
    if os.path.isfile(datapath):
        list_path = datapath
        vdlist_name = os.path.basename(datapath)
    else:
        candidates = [
            os.path.join(datapath, vdlist_name),
            os.path.join(datapath, "lists", "vd", vdlist_name),
        ]
        list_path = next((p for p in candidates if os.path.exists(p)), None)
        if list_path is None:
            raise FileNotFoundError(
                f"Could not find vdlist '{vdlist_name}' in {datapath} "
                f"(looked in: {candidates})."
            )

    unit_scale = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "s": 1.0}
    raw, seconds = [], []

    with open(list_path, "r", errors="ignore") as inp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            raw.append(line)
            if line[-1] in unit_scale:
                seconds.append(float(line[:-1]) * unit_scale[line[-1]])
            else:
                seconds.append(float(line))

    return {"name": vdlist_name, "raw": raw, "seconds": np.array(seconds, dtype=float)}


def read_vclist(datapath: str, vclist_name: str = "vclist") -> dict:
    """Read a Bruker variable-counter list (``vclist``) -- integer loop
    counters (not time-suffixed, unlike ``vdlist``) used by some PFG
    diffusion pulse sequences (e.g. gradient-pulse-train or
    convection-compensated variants) to step something other than a
    plain per-slice delay. :func:`read_vdlist`'s own parser already
    handles unsuffixed plain numbers (its ``else`` branch), so this is
    a thin, discoverable wrapper around it rather than new parsing --
    the ``"seconds"`` key is still returned by that shared parser, but
    is a misnomer here: read the raw integer counts from ``"raw"``,
    or ``"seconds"`` if they're being (mis)used as a plain numeric
    series for lack of anywhere better to put them.
    """
    return read_vdlist(datapath, vdlist_name=vclist_name)


def read_diff_xml(path: str, nucleus: str | None = None) -> dict:
    """Read a Bruker diffusion (PFG) experiment's ``diff.xml`` --
    delta/DELTA (converted ms -> s), the expected diffusion
    coefficient, and, per gradient step, both the gradient strength
    and TopSpin's own precomputed b-value.

    Ported from ``leonmr.diff_params_import`` -- the key thing it gets
    right that a hand-rolled ``b = (gamma*g*delta)^2*(DELTA-delta/3)``
    calculation risks getting subtly wrong is using TopSpin's *own*
    b-values (``xValues/List``'s 4th number per gradient step) rather
    than recomputing them, so this only needs `gradient`/`delta`/
    `DELTA` for display -- the fit itself (see
    ``processing.relaxation``'s ``stejskal_tanner`` model) uses ``B``
    directly.

    ``nucleus`` : optional -- if given, ``gamma`` (10^7 rad/s/T) is
    looked up via :func:`opennmr.core.registry.find_gamma` and
    included for reference; the b-values above already have gamma
    baked in from TopSpin's own calculation, so nothing in the actual
    fit depends on this.

    Returns a dict: ``gradient`` (T/m, ascending with step), ``B``
    (s/m^2), ``delta``, ``DELTA`` (s), ``expD`` (m^2/s, TopSpin's own
    guess -- a reasonable initial value for a fit), and ``gamma``
    (None if ``nucleus`` wasn't given).
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(os.path.join(path, "diff.xml"))
    root = tree.getroot()

    delta = float(root.find(".//delta").text) / 1000.0  # ms -> s
    DELTA = float(root.find(".//DELTA").text) / 1000.0  # ms -> s
    expD = float(root.find(".//exDiffCoff").text)

    x_values = [float(v) for v in root.find(".//xValues/List").text.split()]
    gradient = [g / 100.0 for g in x_values[1::4]]  # G/cm -> T/m
    b_values = x_values[3::4]  # s/m^2, TopSpin's own calculation

    gamma = find_gamma(nucleus)["gamma"] if nucleus else None

    return {
        "gradient": np.array(gradient, dtype=float), "B": np.array(b_values, dtype=float),
        "delta": delta, "DELTA": DELTA, "expD": expD, "gamma": gamma,
    }


# ============================================================
# Vendor registration
# ============================================================

def _detect(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "acqus")) and (
        os.path.exists(os.path.join(path, "fid")) or os.path.exists(os.path.join(path, "ser"))
        or os.path.isdir(os.path.join(path, "pdata"))
    )


def _read_auto(path: str, **kwargs) -> NMRSpectrum:
    """Vendor-registry entry point: prefer the raw-FID path, falling
    back to already-processed data when no raw ``fid``/``ser`` is
    present at all -- see the module docstring for why an unsupported
    window function no longer routes through here (``read_1d``/
    ``read_2d`` handle that themselves now, staying on the raw-FID
    path) -- or, for 2D specifically, when the raw ``ser``'s shape
    genuinely can't be determined at all (``_guess_raw_2d_shape``
    needs ``acqus["TD"]`` at minimum -- missing on a malformed/
    incomplete parameter set; there is no raw shape to "stay on" the
    way an unsupported window has a FID to stay on, so falling back is
    the only usable option). Flagged via
    ``metadata.raw["read_2d_shape_guess_failed"]`` so callers (the
    GUI) can tell the user why raw-FID reprocessing isn't available,
    rather than it being silent.

    Every branch that returns already-processed data instead of the
    raw-FID path -- including the two below that always did, silently
    -- also tags the more general ``metadata.raw["loaded_from_processed"]``
    with a short reason, so the GUI can show a "you're looking at
    processed, not raw, data" note uniformly regardless of *why* (no
    raw file at all vs. a raw file whose shape couldn't be determined).

    Catches ``KeyError`` (``_guess_raw_2d_shape``'s own ``acqus["TD"]``
    lookup, its only way to fail) and, defensively, ``IndexError``
    (this project's own raw-shape resolution replaced the ``nmrglue``-
    shape-guess-then-``.reshape()``-then-``data.shape[1]`` chain that
    used to be the real source of this -- see TODO.md's Bruker raw-
    ``ser`` investigation -- so this shouldn't actually trigger any
    more, but costs nothing to keep). Deliberately not the broader
    ``ValueError``: something else going wrong inside ``read_2d``
    should surface to the caller normally (found one for real once:
    ``align_echo``'s auto-truncation disagreeing row-to-row -- fixed
    directly, not by routing it through this fallback), not be
    silently swallowed into a fallback that discards whatever raw-FID
    processing was actually requested. Also deliberately not a broad
    catch for "the guessed shape left an oddly large fraction of the
    file unaccounted for" -- tried once, reverted, see
    :func:`_read_raw_2d_data`'s own docstring for why that turned out
    to be a false signal, not a real one, for this vendor's data."""
    has_ser = os.path.exists(os.path.join(path, "ser"))
    has_fid = os.path.exists(os.path.join(path, "fid"))
    has_2rr = os.path.exists(os.path.join(path, "pdata", str(kwargs.get("procno", 1)), "2rr"))
    # `pseudo_2d` is meaningless for the 1D readers (neither accepts it),
    # but a caller can't know whether a given path is 1D or 2D without
    # doing the very read it's trying to parameterise -- the branch
    # below is the first point where that's actually known. Dropping it
    # here lets the GUI pass its pseudo-2D preference in on the *first*
    # load rather than reading the file once to discover it's 2D and
    # then reading it a second time with the right flag (which cost a
    # discarded 5.19s genuine-2D reconstruction per load on the real
    # EXSY benchmark -- see TODO.md's duplication-trace entry).
    one_d_kwargs = {k: v for k, v in kwargs.items() if k != "pseudo_2d"}
    if has_ser:
        try:
            return read_2d(path, **kwargs)
        except (IndexError, KeyError) as exc:
            if not has_2rr:
                raise
            # read_2d_processed has no raw-FID-reprocessing parameters
            # (window/zero_fill/fid_shift/phase.../...) -- the GUI
            # always sends them once it's seen `ser` on disk (the same
            # check this function just made), so only the parameters
            # read_2d_processed actually accepts are forwarded here.
            processed_kwargs = {
                k: v for k, v in kwargs.items() if k in ("procno", "homonuclear", "pseudo_2d", "magnitude")
            }
            spectrum = read_2d_processed(path, **processed_kwargs)
            spectrum.metadata.raw["read_2d_shape_guess_failed"] = str(exc)
            spectrum.metadata.raw["loaded_from_processed"] = str(exc)
            return spectrum
    if has_fid:
        return read_1d(path, **one_d_kwargs)
    if has_2rr:
        spectrum = read_2d_processed(path, **kwargs)
        spectrum.metadata.raw["loaded_from_processed"] = "No raw ser file found for this experiment."
        return spectrum
    spectrum = read_1d_processed(path, **one_d_kwargs)
    spectrum.metadata.raw["loaded_from_processed"] = "No raw fid file found for this experiment."
    return spectrum


base.register("bruker", _detect, _read_auto)
