"""
Reference deconvolution (FIDDLE) -- correcting a lineshape distortion
that every peak in the spectrum shares.

The idea, due to Morris and co-workers: pick one isolated resonance
whose *true* shape you are willing to assert, measure the shape it
actually came out as, and the ratio of the two is the instrument's
distortion. Divide it out of the whole spectrum and every peak is
corrected at once.

It works because the distortion is **common-mode**. A field
inhomogeneity -- a bad shim, a sample that is not spinning, or (the
case this was built for) the electrodes in an eNMR cell -- multiplies
every resonance's FID by the *same* envelope, because they all sit in
the same magnet at the same time. So the reference carries a complete
record of what happened to everything else.

Mechanically it is a division in the time domain::

    correction = fid_ideal / fid_measured
    fid_corrected = fid_whole * correction

and the entire difficulty is that ``fid_measured`` decays into noise.
Past the point where the reference has died away, that division is
noise over noise -- unbounded gain on nothing. :func:`correction` is
therefore about the *regularisation*, not the division; see its own
docstring and :data:`REGULARISATIONS`.

Its input must be an honest spectrum
------------------------------------
Every function here assumes ``spectrum_row`` is the Fourier transform of
a **causal FID** -- because the whole method runs by inverse-
transforming it back to one. That holds for a spectrum this package
computed from a raw FID. It does **not** hold for a vendor's
already-processed spectrum: Bruker's ``2rr``/``2ii`` has a digital-
filter group delay (``GRPDLY``, 76 points on the eNMR data here), and
apodisation and baseline correction have already been folded in, so
inverse-transforming it does not give back the FID the maths assumes.

Measured, so nobody has to rediscover it: applied to a processed
``2rr`` the water line at 4.925 ppm went from a peak of 2397 spanning
0.0159 ppm to a peak of 16 spanning 0.536 ppm -- the line destroyed
rather than sharpened. **Feed this the raw FID path**
(``metadata.raw["fid_raw"]`` is retained for every vendor), not a
processed spectrum. That wiring does not exist yet; see TODO.md.

What this does not fix
----------------------
* **Anything not common to the whole spectrum.** Convection, or the
  electro-osmotic flow in an eNMR cell, modulates phase per *scan*;
  it is not one fixed envelope and no reference can carry it.
* **An overlapped reference.** The isolated-resonance requirement is
  real: whatever else is inside the chosen region is treated as part
  of the reference's own lineshape and gets divided out of everything.
* **A wrong target.** The corrected linewidth is whatever you ask for.
  Asking for one much narrower than the acquisition supports does not
  reveal detail; it amplifies noise until the fit looks sharp and
  means nothing.
"""

from __future__ import annotations

import numpy as np

from . import ft as processing_ft
from . import fitting as processing_fitting

#: How the reciprocal ``1 / fid_measured`` is tamed where the measured
#: reference FID has decayed into noise. **This choice is the method**
#: -- the division itself is one line.
#:
#: * ``"wiener"`` -- ``conj(m) / (|m|^2 + eps * max|m|^2)``. Reduces to
#:   ``1/m`` where the reference is strong and rolls smoothly to zero
#:   where it is not, with no threshold to place. The default, because
#:   it needs one parameter and degrades gracefully rather than
#:   cliff-edged.
#: * ``"truncate"`` -- plain ``1/m`` while ``|m|`` is above
#:   ``eps * max|m|``, and 1 (no correction) after. Sharper and easier
#:   to reason about, at the cost of a discontinuity that rings.
#: * ``"none"`` -- the bare reciprocal. Present so the failure it
#:   causes can be demonstrated rather than described; not for real
#:   data.
REGULARISATIONS = ("wiener", "truncate", "none")


def isolate_reference(spectrum_row, freq_ppm, region: tuple[float, float]) -> np.ndarray:
    """The FID of just the resonance inside ``region``.

    Zeroes the spectrum everywhere else and inverse-transforms what is
    left. ``region`` is ``(low_ppm, high_ppm)`` in either order.

    Zeroing rather than cutting: the FID has to come back the same
    length as the original so it can divide into it point for point,
    and a windowed-out region is exactly a resonance with everything
    else removed.
    """
    spectrum_row = np.asarray(spectrum_row)
    freq_ppm = np.asarray(freq_ppm, dtype=float)
    if spectrum_row.shape[-1] != freq_ppm.size:
        raise ValueError(
            f"{spectrum_row.shape[-1]} spectral points against {freq_ppm.size} shifts."
        )
    low, high = (min(region), max(region))
    mask = (freq_ppm >= low) & (freq_ppm <= high)
    if not mask.any():
        raise ValueError(
            f"No spectral points between {low:g} and {high:g} ppm – check the "
            f"reference region against the spectrum's own range "
            f"({freq_ppm.min():g} to {freq_ppm.max():g} ppm)."
        )
    isolated = np.where(mask, spectrum_row, 0.0)
    return processing_ft.ifft_spectrum(isolated)


def ideal_reference(freq_ppm, shift: float, fwhm: float, amplitude: float = 1.0,
                     lineshape: str = "lorentzian", eta: float = 0.5) -> np.ndarray:
    """The FID a reference resonance *should* have had: the chosen
    lineshape at ``shift`` with width ``fwhm`` (both in ppm).

    **Built in the time domain, not by inverse-transforming a
    lineshape**, and that is not a shortcut -- it is the only correct
    construction. Inverse-transforming a *real* Lorentzian profile
    (the absorption shape :mod:`opennmr.processing.fitting` draws)
    gives a signal symmetric about ``t = 0``: half of it wraps around
    to the end of the array, and dividing by it corrupts exactly the
    late points the regularisation is trying to protect. Measured
    while building this -- a distortion that broadened a line 4x was
    only half-corrected until the ideal FID was made causal.

    An FID is causal: it starts at ``t = 0`` and decays. A Lorentzian
    of width ``w`` is ``exp(-pi*w*t)``; a Gaussian of width ``w`` is
    ``exp(-(pi*w*t)^2 / (4 ln 2))``; ``pseudo_voigt`` mixes the two by
    ``eta`` the same way the fitting module's own does. Position comes
    in as ``exp(-2i*pi*shift*t)``, the sign matching
    :func:`opennmr.processing.ft.fft_spectrum`'s own axis convention
    (which puts a ``+f`` FID at ``-f``).

    Time is measured in the units conjugate to ``freq_ppm``: the grid
    is ``k / sweep_ppm``, so a width given in ppm and a position given
    in ppm both mean what they say without needing a dwell time or a
    spectrometer frequency passed in.
    """
    freq_ppm = np.asarray(freq_ppm, dtype=float)
    if fwhm <= 0:
        raise ValueError(f"Target FWHM must be positive, got {fwhm:g}.")
    n = freq_ppm.size
    sweep = float(np.abs(freq_ppm[-1] - freq_ppm[0])) * n / max(n - 1, 1)
    if sweep <= 0:
        raise ValueError("The ppm axis has zero span – cannot build a reference FID.")
    t = np.arange(n) / sweep

    lorentz = np.exp(-np.pi * fwhm * t)
    gauss = np.exp(-((np.pi * fwhm * t) ** 2) / (4.0 * np.log(2.0)))
    if lineshape == "lorentzian":
        envelope = lorentz
    elif lineshape == "gaussian":
        envelope = gauss
    elif lineshape == "pseudo_voigt":
        envelope = eta * lorentz + (1.0 - eta) * gauss
    else:
        raise ValueError(
            f"Unknown lineshape {lineshape!r}; expected 'lorentzian', 'gaussian' "
            f"or 'pseudo_voigt'."
        )
    return amplitude * np.exp(-2j * np.pi * shift * t) * envelope


def correction(fid_measured, fid_ideal, method: str = "wiener", eps: float = 1e-3):
    """The time-domain correction ``fid_ideal / fid_measured``, damped
    where the measured reference has decayed into noise.

    **The damping is the method.** An undamped reciprocal is not a
    smaller version of this -- it is unusable: past the reference's own
    decay the denominator is noise, and the correction grows without
    bound, multiplying the late FID (where the real signal has also
    decayed) by whatever that noise happened to be. What comes out
    looks sharp and is not a spectrum of the sample.

    ``eps`` is relative to the reference's own maximum, so it means the
    same thing whatever scale the data is on. Larger is safer and
    broader: it backs off sooner, leaving a less-corrected but quieter
    result.

    Returns the complex correction, same length as the inputs.
    """
    if method not in REGULARISATIONS:
        raise ValueError(
            f"Unknown regularisation {method!r}. Available: {list(REGULARISATIONS)}"
        )
    measured = np.asarray(fid_measured)
    ideal = np.asarray(fid_ideal)
    if measured.shape != ideal.shape:
        raise ValueError(
            f"Reference FIDs differ in shape: measured {measured.shape}, ideal {ideal.shape}."
        )
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps:g}.")

    magnitude = np.abs(measured)
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak <= 0:
        raise ValueError(
            "The isolated reference FID is entirely zero – the reference region "
            "holds no signal."
        )

    if method == "none":
        with np.errstate(divide="ignore", invalid="ignore"):
            result = ideal / measured
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    if method == "truncate":
        keep = magnitude >= eps * peak
        result = np.ones_like(ideal, dtype=complex)
        with np.errstate(divide="ignore", invalid="ignore"):
            divided = ideal / measured
        result[keep] = divided[keep]
        return np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)

    # Wiener: conj(m) / (|m|^2 + eps*max|m|^2). Where |m| is large this
    # is 1/m; where it is small the added floor dominates and the whole
    # expression rolls to zero instead of diverging.
    floor = eps * peak ** 2
    return ideal * np.conj(measured) / (np.abs(measured) ** 2 + floor)


def reference_deconvolve(spectrum_row, freq_ppm, region: tuple[float, float], *,
                          fwhm: float, shift: float | None = None,
                          lineshape: str = "lorentzian", eta: float = 0.5,
                          method: str = "wiener", eps: float = 1e-3) -> dict:
    """Reference-deconvolve one spectrum against a resonance in
    ``region``.

    ``shift`` : where the target line is centred, in ppm. ``None``
    (default) uses the measured position of the reference (see
    :func:`reference_position`), so the correction does not silently
    move the spectrum; pass a value when the reference's true position
    is known, which beats any estimate from a distorted line.

    ``fwhm`` : the target linewidth in ppm. There is no default,
    deliberately -- it is the one number that decides what the output
    looks like, and a plausible-looking default would get copied
    without thought.

    Returns a dict with ``spectrum`` (the corrected row),
    ``correction``, ``fid_measured``, ``fid_ideal``, ``shift`` (the one
    actually used) and the settings, so a caller can show *why* a
    result looks the way it does rather than only the result.
    """
    spectrum_row = np.asarray(spectrum_row)
    freq_ppm = np.asarray(freq_ppm, dtype=float)

    fid_measured = isolate_reference(spectrum_row, freq_ppm, region)
    if shift is None:
        shift = reference_position(spectrum_row, freq_ppm, region)
    # Matched at t = 0, where both FIDs are just "how much reference
    # there is" before either has decayed -- so the correction changes
    # lineshape and leaves the overall scale alone. A reference
    # deconvolution should not rescale the spectrum by whatever the
    # target amplitude happened to be.
    amplitude = float(np.abs(fid_measured[0])) or 1.0
    fid_ideal = ideal_reference(freq_ppm, shift, fwhm, amplitude=amplitude,
                                 lineshape=lineshape, eta=eta)

    factor = correction(fid_measured, fid_ideal, method=method, eps=eps)
    fid_whole = processing_ft.ifft_spectrum(spectrum_row.astype(complex))
    corrected = processing_ft.fft_spectrum(fid_whole * factor, 1.0)[1]

    return {
        "spectrum": corrected, "correction": factor,
        "fid_measured": fid_measured, "fid_ideal": fid_ideal,
        "shift": float(shift), "fwhm": float(fwhm), "lineshape": lineshape,
        "method": method, "eps": float(eps), "region": (min(region), max(region)),
    }


def reference_position(spectrum_row, freq_ppm, region: tuple[float, float]) -> float:
    """Where the measured reference actually sits, in ppm -- the
    **maximum** of its magnitude.

    Used as the default target position so a correction changes
    lineshape without moving the spectrum.

    **An intensity-weighted centroid was tried first and is worse**,
    which is the whole reason this is a maximum and not the one-liner
    it looks like it should be. The distortions this module exists to
    correct are asymmetric: they drag a long tail out to one side, and
    a centroid follows the tail. Measured on a synthetic line whose
    true position is 0.000 ppm, distorted by a decaying field wobble:
    the magnitude maximum lands at -0.024, a centroid over points
    above half-height at -0.107, and the full-region centroid at
    -0.283 -- an order of magnitude worse than the maximum, and enough
    to visibly shift a corrected spectrum.

    Magnitude rather than the real part, so a reference that is not
    perfectly phased still reports where the line is rather than where
    its positive lobe is. Pass ``shift`` to
    :func:`reference_deconvolve` explicitly when the reference's true
    position is known (TMS at 0, a solvent peak) -- that is better than
    any estimate from a distorted line.
    """
    spectrum_row = np.asarray(spectrum_row)
    freq_ppm = np.asarray(freq_ppm, dtype=float)
    low, high = (min(region), max(region))
    mask = (freq_ppm >= low) & (freq_ppm <= high)
    magnitude = np.abs(spectrum_row[mask])
    if not magnitude.size or float(magnitude.max()) <= 0:
        return float((low + high) / 2.0)
    return float(freq_ppm[mask][int(np.argmax(magnitude))])


def _region_area(spectrum_row, freq_ppm, region: tuple[float, float]) -> float:
    """The reference's own integrated magnitude, for matching the
    target's amplitude to it."""
    spectrum_row = np.asarray(spectrum_row)
    freq_ppm = np.asarray(freq_ppm, dtype=float)
    low, high = (min(region), max(region))
    mask = (freq_ppm >= low) & (freq_ppm <= high)
    area = float(np.abs(np.trapezoid(np.abs(spectrum_row[mask]), freq_ppm[mask])))
    return area if area > 0 else 1.0
