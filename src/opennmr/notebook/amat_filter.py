"""
The spectral filter for 2D aMAT spectra at fast MAS (explicit user request,
2026-11-13), as described in the Supporting Information, section S7, of
Angew. Chem. Int. Ed. 2024, e202408704.

Why it works
------------
The aMAT time-domain signal is (SI eq. 1)

    S(t1, t2) = Σ_k s_k exp(−i ω_iso t1) exp[−i (ω_iso t2 + k ω_rot t2)] :

the indirect dimension is modulated by the isotropic shift only, the
direct one by the whole MAS spectrum. So in the 2D spectrum S(ν1, ν2),
every horizontal trace S(ν1′, ν2) can carry real signal only at

    ν2 = ν1′ + n ν_rot ,  n an integer

-- its own centreband and spinning sidebands. Anything else on that trace
(the "ghosts" of much more intense signals, which hide weak ones such as
the para-proton at −37 ppm in the SI's example) is an artefact.

The filter
----------
For each F1 position ν1 = ν_c, a 1D pulse train f(ν2; ν_c) with period
ν_rot, centred at ν_c, covering F2 (SI Figure S7); each pulse is ``width``
wide and, as in the SI, shaped by a Hann window (``shape="hann"``) to
emphasise the signal at its centre, or flat (``shape="rect"``). The
trains of every F1 row form the 2D filter F(ν1, ν2), whose central ridge
lies on the diagonal ν1 = ν2 (SI Figure S8c); the spectrum is multiplied
by it (both its real and imaginary parts, so the phase is untouched).

The pulse width
---------------
The SI shows ν_rot/3 (the default). It can be chosen to remove as much of
the artefacts and noise as possible, but **must be broader than the
homogeneous linewidth 1/(πT2′)**, or real signal is cut away. With a Hann
pulse the part above half height is width/2 wide, so the check here is
width/2 ≥ 1/(πT2′) (width ≥ linewidth for a flat pulse); a narrower pulse
is reported, not refused. Give the linewidth (``linewidth_hz``) or T2′
(``t2_prime_s``) to have it checked.

Time domain
-----------
F depends on ν2 − ν1 only and is periodic in it, so multiplying by it is,
in the time domain, a sum of copies of S shifted by whole rotor periods
along t2 and back along t1, weighted by the Fourier coefficients of one
pulse. The t1 increment of an aMAT experiment is not a fraction of the
rotor period that makes those shifts land on the sampled points, so the
time-domain form needs interpolation -- which is this same Fourier
multiplication. The SI applies it in the frequency domain; so does this
module (on a processed 2D).

What it cannot remove: distortions of a trace at the true signal positions
(the SI notes minor distortions remaining, e.g. for the ortho proton).

    from opennmr.notebook import amat_filter as af
    filtered = af.apply_amat_filter(spectrum, nu_rot_hz=111e3)
    fig = af.plot_amat_trace(spectrum, filtered, f1_ppm=-37)
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["pulse_train", "amat_filter_map", "apply_amat_filter", "amat_trace",
           "plot_amat_trace", "plot_amat_filter_map", "SHAPES", "REFERENCE"]

#: Pulse shapes: the SI's Hann-truncated pulse and a flat one.
SHAPES = {"hann": "Hann-shaped pulses (as in the SI)", "rect": "Flat (rectangular) pulses"}

REFERENCE = "Angew. Chem. Int. Ed. 2024, e202408704, Supporting Information S7"


def pulse_train(nu, centre, nu_rot, width, shape: str = "hann") -> np.ndarray:
    """f(ν; ν_c): pulses of ``width`` at ``centre`` + n ``nu_rot`` (all n),
    on the points ``nu`` (any unit, the same for every argument). 1 at each
    pulse centre; 0 between pulses. ``centre`` may be an array broadcasting
    against ``nu`` (one train per row)."""
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {tuple(SHAPES)}.")
    nu_rot, width = float(nu_rot), float(width)
    if nu_rot <= 0:
        raise ValueError("The spinning rate must be positive.")
    if not 0 < width <= nu_rot:
        raise ValueError("The pulse width must be positive and no wider than the spinning "
                         "rate (the pulses would overlap).")
    offset = np.asarray(nu, dtype=float) - np.asarray(centre, dtype=float)
    d = offset - np.round(offset / nu_rot) * nu_rot        # distance to the nearest pulse
    inside = np.abs(d) <= width / 2.0
    if shape == "rect":
        return inside.astype(float)
    return np.where(inside, np.cos(np.pi * d / width) ** 2, 0.0)


def amat_filter_map(f1_ppm, f2_ppm, *, nu_rot_hz, sf_mhz, width_hz=None, shape="hann",
                    f1_offset_ppm=0.0) -> np.ndarray:
    """The 2D filter F(ν1, ν2) (rows = F1) for the axes given (ppm):
    each F1 row is the pulse train centred at that row's F1 position
    (+ ``f1_offset_ppm``). ``sf_mhz`` converts Hz to ppm (the observed
    nucleus' spectrometer frequency). ``width_hz`` defaults to ν_rot/3."""
    sf = float(sf_mhz)
    if sf <= 0:
        raise ValueError("The spectrometer frequency is needed to turn Hz into ppm.")
    width_hz = float(nu_rot_hz) / 3.0 if width_hz is None else float(width_hz)
    f1 = np.asarray(f1_ppm, dtype=float)[:, None] + float(f1_offset_ppm)
    f2 = np.asarray(f2_ppm, dtype=float)[None, :]
    return pulse_train(f2, f1, float(nu_rot_hz) / sf, width_hz / sf, shape)


# ---------------------------------------------------------------------------

def _is_dict(obj) -> bool:
    return isinstance(obj, dict)


def _axes(spectrum):
    """(F1 ppm, F2 ppm, F2 spectrometer MHz, nucleus, raw metadata, data)."""
    if _is_dict(spectrum):
        if spectrum.get("pseudo_2d"):
            raise ValueError("A pseudo-2D series has no isotropic F1: the aMAT filter needs an "
                             "aMAT 2D.")
        data = np.asarray(spectrum["spectrum"], dtype=float)
        f1 = np.asarray(spectrum["y_ppm"], dtype=float)
        f2 = np.asarray(spectrum["x_ppm"], dtype=float)
        sf = float(spectrum.get("SF1") or 0.0)
        nuc1, nuc2 = spectrum.get("NUC1"), spectrum.get("NUC2")
        homo = bool(spectrum.get("homonuclear")) or nuc2 in (None, nuc1)
        raw = {"acqus": spectrum.get("acqus") or {}}
        nucleus = nuc1
    else:
        if spectrum.ndim != 2:
            raise ValueError("The aMAT filter needs a 2D spectrum.")
        if spectrum.metadata.pseudo_2d or spectrum.axes[0].kind != "frequency":
            raise ValueError("A pseudo-2D series has no isotropic F1: the aMAT filter needs an "
                             "aMAT 2D.")
        meta = spectrum.metadata
        data = np.asarray(spectrum.data, dtype=float)
        f1 = np.asarray(spectrum.axes[0].ppm, dtype=float)
        f2 = np.asarray(spectrum.axes[1].ppm, dtype=float)
        sf = float(meta.sfo1 or 0.0)
        from .two_d import canonical_nucleus
        n1, n2 = canonical_nucleus(meta.nucleus_indirect), canonical_nucleus(meta.nucleus)
        homo = bool(meta.homonuclear) or n1 is None or n1 == n2
        raw = meta.raw or {}
        nucleus = meta.nucleus
    if data.shape != (len(f1), len(f2)):
        raise ValueError(f"The data ({data.shape}) do not match the axes ({len(f1)} F1 × "
                         f"{len(f2)} F2 points).")
    if not homo:
        raise ValueError("aMAT is homonuclear: F1 must be the isotropic shift of the same "
                         "nucleus as F2.")
    return f1, f2, sf, nucleus, raw, data


def recorded_mas_rate(spectrum) -> float | None:
    """The MAS rate (Hz) recorded with the experiment (Bruker MASR), if any."""
    raw = spectrum if _is_dict(spectrum) else (spectrum.metadata.raw or {})
    acqus = (raw.get("acqus") or {}) if isinstance(raw, dict) else {}
    try:
        rate = float(acqus.get("MASR", 0) or 0)
    except (TypeError, ValueError):
        return None
    return rate if rate > 0 else None


def apply_amat_filter(spectrum, nu_rot_hz=None, *, width_hz=None, width_fraction=1.0 / 3.0,
                      shape: str = "hann", linewidth_hz=None, t2_prime_s=None,
                      f1_offset_ppm: float = 0.0, source_label: str | None = None):
    """The aMAT spectrum multiplied by its 2D pulse-train filter.

    Parameters
    ----------
    spectrum : NMRSpectrum or dict
        A processed aMAT 2D (F1 the isotropic, "infinite-MAS" dimension; F2
        the MAS spectrum), from :func:`opennmr.io.load` or
        :func:`opennmr.notebook.two_d.NMR2dimport`. The result is the same
        type, with the filter's log in ``metadata.raw["amat_filter"]`` (or
        ``["amat_filter"]``).
    nu_rot_hz : float
        The MAS rate (Hz); default the recorded one (Bruker MASR).
    width_hz, width_fraction : float
        The width of each pulse: ``width_hz``, or ``width_fraction`` × ν_rot
        (default 1/3, as in the SI's Figure S7).
    shape : "hann" or "rect"
        Hann-shaped pulses (the SI's) or flat ones.
    linewidth_hz, t2_prime_s : float
        The homogeneous linewidth 1/(πT2′) (Hz), or T2′ (s), to check the
        width against: the pulses must be broader than it.
    f1_offset_ppm : float
        Moves every pulse train by this much relative to its F1 row: for an
        F1 axis referenced (or scaled) slightly differently from F2. The
        ridge of the centre pulses should lie on the diagonal ν1 = ν2.
    """
    f1, f2, sf, nucleus, raw, data = _axes(spectrum)
    if nu_rot_hz is None:
        nu_rot_hz = recorded_mas_rate(spectrum)
        if nu_rot_hz is None:
            raise ValueError("Give the MAS rate (none is recorded with this spectrum).")
    nu_rot_hz = float(nu_rot_hz)
    if nu_rot_hz <= 0:
        raise ValueError("The MAS rate must be positive.")
    if sf <= 0:
        raise ValueError("The spectrometer frequency of F2 is unknown, so Hz cannot be turned "
                         "into ppm.")
    if width_hz is None:
        width_hz = float(width_fraction) * nu_rot_hz
    width_hz = float(width_hz)
    if t2_prime_s is not None and linewidth_hz is None:
        if float(t2_prime_s) <= 0:
            raise ValueError("T2′ must be positive.")
        linewidth_hz = 1.0 / (np.pi * float(t2_prime_s))
    F = amat_filter_map(f1, f2, nu_rot_hz=nu_rot_hz, sf_mhz=sf, width_hz=width_hz, shape=shape,
                        f1_offset_ppm=f1_offset_ppm)

    warnings_list = []
    passband = width_hz / 2.0 if shape == "hann" else width_hz
    if linewidth_hz is not None:
        if passband < float(linewidth_hz):
            warnings_list.append(
                f"The pulses are {'half-height ' if shape == 'hann' else ''}{passband:.4g} Hz "
                f"wide, narrower than the homogeneous linewidth {float(linewidth_hz):.4g} Hz "
                "(1/πT2′): real signal is cut. Widen the pulses.")
    dx = abs(float(f2[1] - f2[0])) * sf if len(f2) > 1 else 0.0
    if dx and width_hz < 3 * dx:
        warnings_list.append(f"Each pulse spans fewer than 3 F2 points ({width_hz / dx:.2g}): "
                             "zero-fill F2 more, or widen the pulses.")
    lo, hi = min(f2.min(), f2.max()), max(f2.min(), f2.max())
    f1_shifted = f1 + f1_offset_ppm
    outside = np.mean((f1_shifted < lo) | (f1_shifted > hi)) if len(f1) else 0.0
    if outside > 0.5:
        warnings_list.append(
            f"{outside:.0%} of the F1 axis lies outside the F2 range: F1 should be the "
            "isotropic shift on the same ppm scale as F2 (check its referencing, or use the "
            "F1 offset).")

    out_data = data * F
    imag = None
    if not _is_dict(spectrum) and spectrum.data_imag is not None:
        imag = np.asarray(spectrum.data_imag, dtype=float) * F
    removed = float(np.sum(np.abs(data) * (1.0 - F)) / max(np.sum(np.abs(data)), 1e-300))
    shape_text = "Hann-shaped" if shape == "hann" else "flat"
    log_lines = [
        f"aMAT spectral filter of {source_label or 'the selected 2D'} ({REFERENCE}): each "
        f"F1 row multiplied by a train of {shape_text} pulses {width_hz / 1e3:.4g} kHz wide "
        f"({width_hz / nu_rot_hz:.3g} ν_rot) every ν_rot = {nu_rot_hz / 1e3:.6g} kHz, centred "
        f"on the row's F1 position"
        + (f" + {f1_offset_ppm:g} ppm" if f1_offset_ppm else "") + " (ridge on ν1 = ν2).",
        f"{nucleus or ''} F2 frequency {sf:.6g} MHz: ν_rot = {nu_rot_hz / sf:.4g} ppm, pulse "
        f"width {width_hz / sf:.4g} ppm.".strip(),
        f"Removed {removed:.1%} of the summed |intensity| (artefacts and noise between the "
        "allowed positions)."]
    if linewidth_hz is not None:
        log_lines.append(f"Homogeneous linewidth 1/(πT2′) = {float(linewidth_hz):.4g} Hz; pulse "
                         f"{'half-height ' if shape == 'hann' else ''}width {passband:.4g} Hz.")
    log_lines.extend(f"Warning: {w}" for w in warnings_list)
    record = {"nu_rot_hz": nu_rot_hz, "width_hz": width_hz, "shape": shape,
              "linewidth_hz": None if linewidth_hz is None else float(linewidth_hz),
              "f1_offset_ppm": float(f1_offset_ppm), "removed_fraction": removed,
              "warnings": warnings_list, "source": source_label, "reference": REFERENCE,
              "log": "\n".join(log_lines)}

    if _is_dict(spectrum):
        out = dict(spectrum)
        out.update(spectrum=out_data, amat_filter=record)
        return out
    from ..core.spectrum import NMRSpectrum
    meta = spectrum.metadata
    new_raw = dict(meta.raw or {})
    new_raw["amat_filter"] = record
    metadata = dataclasses.replace(meta, raw=new_raw)
    return NMRSpectrum(data=out_data, axes=list(spectrum.axes), metadata=metadata,
                       data_imag=imag)


# ---------------------------------------------------------------------------
# Looking at it
# ---------------------------------------------------------------------------

def amat_trace(spectrum, filtered, f1_ppm) -> dict:
    """The horizontal trace at ``f1_ppm`` (the nearest F1 row) before and
    after filtering, with that row's pulse train: ``{"f1_ppm", "x_ppm",
    "before", "after", "train"}`` (as in the SI's Figure S8b/d)."""
    f1, f2, sf, _nuc, _raw, data = _axes(spectrum)
    rec = filtered["amat_filter"] if _is_dict(filtered) else filtered.metadata.raw["amat_filter"]
    row = int(np.argmin(np.abs(f1 - float(f1_ppm))))
    after = np.asarray(filtered["spectrum"] if _is_dict(filtered) else filtered.data,
                       dtype=float)[row]
    train = pulse_train(f2, f1[row] + rec["f1_offset_ppm"], rec["nu_rot_hz"] / sf,
                        rec["width_hz"] / sf, rec["shape"])
    return {"f1_ppm": float(f1[row]), "x_ppm": f2, "before": data[row], "after": after,
            "train": train}


def plot_amat_trace(spectrum, filtered, f1_ppm, figure=None, fontsize=11.0,
                    colours=("0.55", "C0", "C3"), show_train=True):
    """The trace at ``f1_ppm`` before (grey) and after (blue) filtering,
    with the pulse train (red) drawn above it (SI Figure S8b/d). Returns
    the figure."""
    import matplotlib.pyplot as plt
    tr = amat_trace(spectrum, filtered, f1_ppm)
    fig = figure if figure is not None else plt.figure(figsize=(6.0, 3.6))
    ax = fig.add_subplot(111)
    x = tr["x_ppm"]
    top = float(np.max(np.abs(tr["before"]))) or 1.0
    ax.plot(x, tr["before"] / top, color=colours[0], lw=0.8, label="Before")
    ax.plot(x, tr["after"] / top, color=colours[1], lw=1.0, label="Filtered")
    if show_train:
        ax.plot(x, 1.15 + 0.3 * tr["train"], color=colours[2], lw=0.9, label="Filter")
    ax.set_xlim(max(x), min(x))
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(r"$\delta$ (ppm)", fontsize=fontsize)
    ax.set_title(f"Trace at $\\delta_1$ = {tr['f1_ppm']:.2f} ppm", fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.9)
    ax.legend(frameon=False, fontsize=fontsize * 0.8, loc="center left",
              bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


def plot_amat_filter_map(spectrum, filtered, figure=None, fontsize=11.0, cmap="Greys"):
    """The 2D filter (SI Figure S8c): its centre-pulse ridge on the
    diagonal ν1 = ν2, drawn in red. Returns the figure."""
    import matplotlib.pyplot as plt
    f1, f2, sf, _nuc, _raw, _data = _axes(spectrum)
    rec = filtered["amat_filter"] if _is_dict(filtered) else filtered.metadata.raw["amat_filter"]
    F = amat_filter_map(f1, f2, nu_rot_hz=rec["nu_rot_hz"], sf_mhz=sf, width_hz=rec["width_hz"],
                        shape=rec["shape"], f1_offset_ppm=rec["f1_offset_ppm"])
    fig = figure if figure is not None else plt.figure(figsize=(5.0, 4.2))
    ax = fig.add_subplot(111)
    ax.pcolormesh(f2, f1, F, cmap=cmap, shading="auto", rasterized=True)
    ridge = f1 + rec["f1_offset_ppm"]
    keep = (ridge >= min(f2)) & (ridge <= max(f2))
    ax.plot(ridge[keep], f1[keep], color="C3", lw=1.2)
    ax.set_title(r"Filter; red: the centre pulses ($\nu_1 = \nu_2$)", fontsize=fontsize * 0.9)
    ax.set_xlim(max(f2), min(f2))
    ax.set_ylim(max(f1), min(f1))
    ax.set_xlabel(r"$\delta_2$ (ppm)", fontsize=fontsize)
    ax.set_ylabel(r"$\delta_1$, isotropic (ppm)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize * 0.9)
    fig.tight_layout()
    return fig
