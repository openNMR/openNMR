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
            autophase: bool = True, phase_mode: str = "apc0", phase0: float = 0.0, phase1: float = 0.0,
            pivot_frac: float = 0.5,
            phase0_inc: float = 0.1, phase1_max: float = 20.0, phase1_improvement: float = 0.15,
            phase_penalty: str = "magnitude",
            window: str | None = None, window_kwargs: dict | None = None, zero_fill: int | None = None,
            fid_shift: float = 0.0, align_echo: bool = False,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward") -> NMRSpectrum:
    """Read the raw ``fid`` and reprocess it using the parameters
    TopSpin itself recorded in ``procs`` (window, ``SI``, and recorded
    ``PHC0``/``PHC1`` unless ``autophase=True`` overrides them).

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
    dic, data = ng.bruker.read(path, read_pulseprogram=False)
    fid = _remove_digital_filter(dic, data).astype(np.complex128)
    fid = np.conj(fid)  # see _CONJUGATE_QUADRATURE_NOTE below
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

    if fid_shift:
        fid = processing_ft.roll_fid(fid, fid_shift)
    if align_echo:
        fid = processing_ft.align_echo(fid)
    if linear_predict_points:
        fid = processing_ft.linear_predict(
            fid, order=linear_predict_order, n_predict=linear_predict_points,
            direction=linear_predict_direction,
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

    if autophase:
        if phase_mode == "joint":
            phase0, phase1 = processing_ft.autophase01(
                spectrum, x_ppm, phase1_max=phase1_max, phase1_improvement=phase1_improvement,
                penalty=phase_penalty, phase0_inc=phase0_inc,
            )
        elif phase_mode == "apc0":
            phase0 = processing_ft.apc0(spectrum, x_ppm, phase0_inc)
        elif phase_mode == "recorded":
            phase0 = float(procs.get("PHC0", 0.0))
            phase1 = float(procs.get("PHC1", 0.0))
        else:
            raise ValueError("phase_mode must be 'apc0', 'joint', or 'recorded'")

    spectrum = processing_ft.apply_phase(spectrum, x_ppm, phase0, phase1, pivot_frac=pivot_frac)
    spectrum = spectrum / mass / ns

    if f1p != 0 or f2p != 0:
        lo, hi = sorted((f1p, f2p))
        keep = (x_ppm >= lo) & (x_ppm <= hi)
        x_ppm, x_hz, spectrum = x_ppm[keep], x_hz[keep], spectrum[keep]

    metadata = NMRMetadata(
        vendor="bruker", nucleus=acqus.get("NUC1"), sfo1=sf, bf1=obs, o1=o1, ns=ns,
        source_path=path, procno=procno,
        raw={
            "acqus": acqus, "procs": procs, "phase0": phase0, "phase1": phase1,
            # The exact (shift/echo-align/LP/windowed/zero-filled) FID
            # fed into the FFT above -- an optional GUI preview of what
            # FID-domain processing actually did, kept alongside (not
            # instead of) the frequency-domain result, which stays the
            # default view.
            "fid": fid, "fid_dt": dt, "fid_original_length": original_length,
            **({"window_unsupported": window_unsupported_note} if window_unsupported_note else {}),
        },
    )
    axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)
    return NMRSpectrum(data=np.real(spectrum), axes=[axis], metadata=metadata, data_imag=np.imag(spectrum))


# ============================================================
# Raw FID -- 2D
# ============================================================

def read_2d(path: str, procno: int = 1, homonuclear: bool = False, pseudo_2d: bool | None = None,
            autophase: bool = True, phase0_inc: float = 0.1,
            phase0: float = 0.0, phase1: float = 0.0, pivot_frac: float = 0.5,
            window: str | None = None, window_kwargs: dict | None = None, zero_fill: int | None = None,
            fid_shift: float = 0.0, align_echo: bool = False,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward") -> NMRSpectrum:
    """Read the raw ``ser`` and reprocess it (F2 then F1) using the
    parameters TopSpin recorded in ``procs``/``proc2s``.

    Handles routine complex 2D processing via ``nmrglue``'s own
    shape-guessing and hypercomplex (States/TPPI) assembly. This is
    the higher-risk half of the raw-FID path (more moving parts than
    1D) -- validate against :func:`read_2d_processed` on real data
    before trusting it for phase-sensitive work (see the project
    plan's Verification section).

    ``pseudo_2d`` : whether F1 is a non-frequency indirect dimension
    (no FFT along that axis -- a slice/experiment index instead), i.e.
    a relaxation/diffusion series rather than a conventional 2D
    spectrum. ``None`` (default) auto-detects from ``acqu2s``'s
    ``NUC1`` (Bruker's own convention: ``<off>`` for pseudo-2D). Pass
    ``True``/``False`` explicitly to override the detection -- useful
    since it depends on ``acqu2s`` being present and conventionally
    populated, which isn't guaranteed for every pulse sequence/Bruker
    software version.

    ``autophase`` : zero-order-only (``apc0``, uniformly across the
    whole F2 dimension), same as before. ``phase0``/``phase1``/
    ``pivot_frac`` : applied (via :func:`opennmr.processing.ft.apply_phase`,
    uniformly across F2) whenever ``autophase=False`` -- lets a
    manually-dialed-in phase correction (from the GUI's Phase tab)
    survive an "Apply Processing" click instead of being silently reset
    to 0/re-autophased. If both ``autophase=True`` and a nonzero
    ``phase1`` are given, ``phase0`` is still recomputed by ``apc0``
    but ``phase1``/``pivot_frac`` are applied on top of it (``apc0``
    itself has no first-order component).

    ``window``/``window_kwargs``/``zero_fill``/``fid_shift``/
    ``align_echo``/``linear_predict_points`` : see :func:`read_1d` --
    identical meaning, but only ever applied along F2 (the direct
    dimension, one row at a time). F1 apodization/zero-fill still
    always follows TopSpin's own recorded ``proc2s`` -- not exposed as
    an override here, since indirect-dimension processing is normally
    fixed at acquisition rather than hand-tuned per view.
    """
    # read_pulseprogram=False: we never use dic["pprog"], and reading
    # it is slow to fail (and warns) whenever the pulse program can't
    # be found/parsed (renamed/custom sequences, older datasets, ...).
    # nmrglue's own shape-guessing (used below, since shape=None) also
    # falls back to a simpler, purely TD/acqu2s-TD-based guess when
    # dic["pprog"] is absent -- skipping the pulse program removes
    # dependence on parsing it at all, not just the warning.
    dic, data = ng.bruker.read(path, read_pulseprogram=False)
    data = _remove_digital_filter(dic, data).astype(np.complex128)
    data = np.conj(data)  # see _CONJUGATE_QUADRATURE_NOTE below
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
    si1 = int(proc2s.get("SI", data.shape[0]))

    # ----- F2 (direct dimension): [shift/align/LP], window, zero-fill, FFT per row -----
    fid_f2 = data
    if fid_shift:
        fid_f2 = np.apply_along_axis(lambda row: processing_ft.roll_fid(row, fid_shift), 1, fid_f2)
    if align_echo:
        fid_f2 = np.apply_along_axis(processing_ft.align_echo, 1, fid_f2)
    if linear_predict_points:
        fid_f2 = np.apply_along_axis(
            lambda row: processing_ft.linear_predict(
                row, order=linear_predict_order, n_predict=linear_predict_points,
                direction=linear_predict_direction,
            ),
            1, fid_f2,
        )
    window_unsupported_note = None
    if window is not None:
        fid_f2 = np.apply_along_axis(
            lambda row: processing_ft.apodize(row, 1.0 / sw, window=window, **(window_kwargs or {})), 1, fid_f2,
        )
    else:
        try:
            fid_f2 = np.apply_along_axis(lambda row: _apply_bruker_window(row, procs, sw), 1, fid_f2)
        except NotImplementedError as exc:
            # Same reasoning as read_1d -- stay on the raw-FID path
            # (full reprocessing still available) rather than
            # abandoning it; the same WDW code applies to every row,
            # so this fails (or doesn't) on the first one either way.
            window_unsupported_note = str(exc)
    fid_f2 = np.apply_along_axis(lambda row: processing_ft.zero_fill(row, si2), 1, fid_f2)
    spectrum = np.apply_along_axis(lambda row: processing_ft.fft_spectrum(row, 1.0 / sw)[1], 1, fid_f2)

    # F2's ppm axis, computed now (F1 processing below only changes
    # axis 0's length, never F2's point count) since a phase1/pivot
    # correction needs it.
    x_ppm, x_hz = _ppm_hz_axis(o1, sf, obs, sw, spectrum.shape[1])
    if autophase:
        phase0 = processing_ft.apc0(spectrum.mean(axis=0), x_ppm, phase0_inc)
    if phase0 or phase1:
        spectrum = processing_ft.apply_phase(spectrum, x_ppm, phase0, phase1, pivot_frac=pivot_frac)

    # ----- F1 (indirect dimension): zero-fill, FFT along axis 0 (skipped for pseudo-2D) -----
    if pseudo_2d:
        y_index = np.arange(1, spectrum.shape[0] + 1)
        y_axis = NMRAxis(kind="index", index=y_index)
    else:
        fid_f1 = np.apply_along_axis(lambda col: processing_ft.zero_fill(col, si1), 0, spectrum)
        spectrum = np.apply_along_axis(lambda col: processing_ft.fft_spectrum(col, 1.0 / sw_1)[1], 0, fid_f1)
        y_ppm, y_hz = _ppm_hz_axis(o1_1, sf_1, obs_1, sw_1, si1)
        y_axis = NMRAxis(kind="frequency", ppm=y_ppm, hz=y_hz)

    x_axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)
    spectrum = spectrum / ns  # match read_1d's convention: data is always per-scan

    metadata = NMRMetadata(
        vendor="bruker", nucleus=acqus.get("NUC1"), nucleus_indirect=nuc2,
        sfo1=sf, sfo1_indirect=(None if pseudo_2d else sf_1), bf1=obs, o1=o1, ns=ns,
        source_path=path, procno=procno, pseudo_2d=pseudo_2d, homonuclear=homonuclear,
        raw={
            "acqus": acqus, "procs": procs, "proc2s": proc2s, "phase0": phase0, "phase1": phase1,
            # F2's fully processed (pre-FFT) FID, one row per F1
            # position -- an optional GUI preview of FID-domain
            # processing; F1 isn't included (its own FFT, when not
            # pseudo-2D, isn't overridable the way F2's is -- see
            # this function's docstring).
            "fid": fid_f2, "fid_dt": 1.0 / sw, "fid_original_length": original_length,
            **({"window_unsupported": window_unsupported_note} if window_unsupported_note else {}),
        },
    )
    return NMRSpectrum(data=np.real(spectrum), axes=[y_axis, x_axis], metadata=metadata, data_imag=np.imag(spectrum))


# ============================================================
# Already-processed fallback (also the raw-FID path's cross-check)
# ============================================================

def read_1d_processed(path: str, procno: int = 1, mass: float = 1.0, f1p: float = 0.0, f2p: float = 0.0,
                       heteronuclear_2d_slice: bool = False, slice_dimension: int = 2) -> NMRSpectrum:
    """Read TopSpin's own already-processed ``1r``/``1i`` directly --
    ported from ``nmrtools2``'s ``NMR1dimport``."""
    real_path = os.path.join(path, "pdata", str(procno), "1r")
    imag_path = os.path.join(path, "pdata", str(procno), "1i")
    acqus_path = os.path.join(path, "acqus")

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
        raw={"acqus": axis_meta, "procs": procs},
    )
    axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)
    return NMRSpectrum(data=real, axes=[axis], metadata=metadata, data_imag=imag)


def read_2d_processed(path: str, procno: int = 1, homonuclear: bool = False,
                       pseudo_2d: bool | None = None) -> NMRSpectrum:
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

    if homonuclear:
        nuc2, obs_1, o1_1, sf_1, sw_1 = acqus.get("NUC1"), obs, o1, sf, sw
        proc2s = {}
    else:
        acqu2s = parse_jcampdx_params(acqu2s_path)
        proc2s = _read_procs(path, procno, proc2=True)
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
    spectrum_imag = _load_block(imag_path)
    if spectrum is not None:
        spectrum = spectrum / ns  # match read_1d_processed's convention: data is always per-scan
    if spectrum_imag is not None:
        spectrum_imag = spectrum_imag / ns

    metadata = NMRMetadata(
        vendor="bruker", nucleus=acqus.get("NUC1"), nucleus_indirect=nuc2,
        sfo1=sf, sfo1_indirect=(None if pseudo_2d else sf_1), bf1=obs, o1=o1, ns=ns,
        source_path=path, procno=procno, pseudo_2d=pseudo_2d, homonuclear=homonuclear,
        raw={"acqus": acqus, "procs": procs, "proc2s": proc2s},
    )
    x_axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)
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
    back to already-processed data only when no raw ``fid``/``ser`` is
    present at all -- see the module docstring for why an unsupported
    window function no longer routes through here (``read_1d``/
    ``read_2d`` handle that themselves now, staying on the raw-FID
    path)."""
    has_ser = os.path.exists(os.path.join(path, "ser"))
    has_fid = os.path.exists(os.path.join(path, "fid"))
    if has_ser:
        return read_2d(path, **kwargs)
    if has_fid:
        return read_1d(path, **kwargs)
    if os.path.exists(os.path.join(path, "pdata", str(kwargs.get("procno", 1)), "2rr")):
        return read_2d_processed(path, **kwargs)
    return read_1d_processed(path, **kwargs)


base.register("bruker", _detect, _read_auto)
