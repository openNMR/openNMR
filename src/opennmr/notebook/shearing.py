"""
Shearing and scaling of 2-D NMR spectra (affine transformations of the
t1/t2 time domain), for MQMAS, STMAS, DAS, 2D J-resolved, 2D PASS/MAT
and any other experiment whose 2D signal follows a known "symmetry
pathway".

Follows Grandinetti *et al.*, *Prog. NMR Spectrosc.* **59** (2011)
121-196, Section 5 (Eqs. 186-216). The step-by-step derivation and the
numerical checks behind every sign used here are in
``SCRIPT_openNMR_Shearing_20261009.ipynb``.

How to use it (notebook)::

    from opennmr.io import load
    from opennmr.notebook import shearing

    s = load("path/to/mqmas_expno")                     # an NMRSpectrum
    plan = shearing.make_shear_plan("mqmas", nucleus="27Al", mq_order=3)
    print(plan.describe())                               # what F1 will mean
    scan = shearing.estimate_shear(s)                     # check against the data
    print(scan["summary"])
    sheared = shearing.shear_2d(s, plan)                  # a new NMRSpectrum
    print(sheared.metadata.raw["shear"]["log"])

``shear_2d`` also takes the notebook's own 2D dict
(:func:`opennmr.notebook.two_d.NMR2dimport`) and returns one, so the
result goes straight into :func:`~opennmr.notebook.two_d.NMR2Dplotter`.

What the operations are
-----------------------
A 2D signal whose frequency in each period is a linear combination of
the same interaction frequencies,

    nu1 = sum_k x1_k Omega_k ,    nu2 = sum_k x2_k Omega_k ,

can be made to drop one component ``a`` from F1 by a **shear parallel to
t2**, ``s'(t1, t2) = s(t1, t2 - kappa_t2 t1)``, with
``kappa_t2 = a1 / a2`` (Eq. 196). In frequency this moves every peak
``(nu1, nu2) -> (nu1 - kappa_t2 nu2, nu2)``. A **shear parallel to t1**,
``s'(t1, t2) = s(t1 - kappa_t1 t2, t2)``, moves
``(nu1, nu2) -> (nu1, nu2 - kappa_t1 nu1)`` and drops a component ``b``
from F2 when ``kappa_t1 = b2 / b1'`` (``b1'`` being F1's coefficient of
``b`` after any first shear). **Scaling** a time axis by ``s`` divides
that frequency axis by ``s``; it changes what the axis *reports*, never
the data.

Both shears are applied exactly (any sub-point amount) with the Fourier
shift theorem, in the mixed domain the operation actually belongs to:
the shear parallel to t2 is a frequency shift of each F1 interferogram
by ``-kappa_t2 * nu2`` (one per F2 column), the shear parallel to t1 a
shift of each F2 spectrum by ``-kappa_t1 * nu1`` (one per F1 row).
``nu`` is always the frequency **relative to the carrier**, read from
the FFT bins, never from a re-referenced ppm axis (a ppm zero that is
not the carrier would add ``kappa * nu_ref`` to every F1 position).

Echo-antiecho and States data
-----------------------------
The echo (p) and antiecho (n) pathways evolve with opposite senses in
t1, so in their raw frames they need opposite shears (kappa and
-kappa). Shearing the *recombined* spectrum does exactly that: the
recombination conjugates one pathway (openNMR's FnMODE 6 decode is
``1j * (conj(p) - n)``, :func:`opennmr.processing.ft.decode_hypercomplex_f1`;
States is the sum of the two), and the shear is linear, so one ramp on
the combined data is one ramp on n and its conjugate on p. Checked in
``tests/test_shearing.py`` from separate p and n rows through the real
decoder: identical to shearing each with its own kappa and then
recombining (to 1e-15), while giving both the same kappa is wrong by
the full scale of the data. What does break it is anything row-by-row
before the shear, such as Align Echo (warned about).

Split-t1
--------
Split-t1 sequences (Brown & Wimperis) share t1 between the MQ (or
satellite) coherence, 1/(1 + |kappa|) of it, and the CT, |kappa|/(1 +
|kappa|), so each crystallite's anisotropy refocuses within t1: the data
arrive sheared and in the unified scaling. ``make_shear_plan(...,
split_t1=True)`` therefore applies no shear and only scales/references
F1 with the effective t1 coefficients (x_MQ - kappa x_CT)/(1 + |kappa|).
Applying the ordinary plan to such data would shear it twice.
:func:`estimate_shear` recognises such data (best kappa near zero).

Conventions in this module
--------------------------
* Arrays are ``(F1, F2)``; frequency axes descend (high frequency at
  index 0), as everywhere in openNMR.
* Coefficients and kappas are in the frame of the axes **as displayed**
  (TopSpin's sense, which openNMR's readers reproduce). Whether F1 is
  displayed with the same sense as the physical coherence frequency
  depends on the quadrature scheme and the receiver phase cycle, so it
  is a parameter (``f1_sign``), and :func:`estimate_shear` measures it
  from the data. This is the practical answer to "which sign of kappa?".
* For quadrupolar nuclei the coefficients come from the standard
  second-order level functions (Amoureux & Fernandez 1998; Man 1998):
  for a coherence ``|m_a><m_b|`` of order ``p = m_a - m_b`` each
  coefficient is ``E(m_b) - E(m_a)``, normalised so that the observed
  central transition (p = -1) has positive coefficients, with

  ``E_cs(m) = m``,
  ``E_0(m) = m [I(I+1) - 3 m^2]``          (isotropic quadrupolar shift),
  ``E_2(m) = m [8 I(I+1) - 12 m^2 - 3]``   (rank-2 anisotropy),
  ``E_4(m) = m [18 I(I+1) - 34 m^2 - 5]``  (rank-4 anisotropy).

  These reproduce the literature ratios: kappa = -7/9 (I = 3/2, 3Q),
  19/12 (I = 5/2, 3Q), -8/9 (I = 3/2, ST1), and the F1 isotropic
  factors 17/8 (I = 3/2) and 17/31 (I = 5/2) in the "unified" scaling.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "level_functions", "coherence_coefficients", "symmetric_coherence",
    "satellite_coherence", "ShearPlan", "make_shear_plan", "EXPERIMENTS",
    "SCALE_CONVENTIONS", "spin_of", "shear_2d", "estimate_shear",
    "sideband_positions", "isotropic_from_peak", "plot_shear_scan",
    "plot_sideband_markers", "preset_kappas",
]

#: Experiments :func:`make_shear_plan` knows, with a one-line meaning.
EXPERIMENTS = {
    "mqmas": "MQMAS (quadrupolar, MAS): shear parallel to t2 removes the rank-4 "
             "anisotropy from F1",
    "stmas": "STMAS (quadrupolar, MAS): satellite in t1, central transition in t2",
    "das": "DAS (quadrupolar, two angles): rank-4 and rank-2 refocused together; "
           "optional second shear centres every F2 pattern on zero",
    "jres": "2D J-resolved, homonuclear (solution): the 45-degree tilt, shear "
            "parallel to t1 removes J from F2",
    "pass": "2D PASS sideband separation (MAS, constant-time pulse timing in t1): shear "
            "parallel to t1 moves every sideband under its isotropic peak; F1 holds the "
            "sideband order x MAS rate, F2 the isotropic shift",
    "mat": "Magic-angle turning / hopping (MAT): the anisotropy is refocused during t1 "
           "(hops between the three magic-angle orientations) and the isotropic shift is "
           "not, so F1 holds the isotropic shift; shear parallel to t1 removes it from F2, "
           "which then holds the anisotropic pattern about zero",
    "shifting_p": "Shifting-p echo (static, e.g. 2H paramagnetic; Walder 2015, Aleksis 2019): the "
                  "shift refocuses on the ridge t2 = t1 and the first-order quadrupolar "
                  "interaction at t2 = 0; shear parallel to t1 puts the shift in F1 and the "
                  "quadrupolar pattern in F2",
    "shifting_d": "Shifting-d echo, also the double/quadruple adiabatic versions (static, e.g. "
                  "2H paramagnetic; Walder 2015, Aleksis 2019): the shift refocuses at t2 = 0 "
                  "and the first-order quadrupolar interaction on t2 = t1; shear parallel to "
                  "t1 puts the quadrupolar pattern in F1 and the shift in F2",
    "custom_kappa": "Custom: give the shear ratios (and scaling) directly",
    "custom_coefficients": "Custom: give the pathway coefficients a1, a2 (and/or "
                           "b1, b2) and the chemical-shift coefficients",
}

#: F1 scaling conventions (``convention=`` of :func:`make_shear_plan`).
SCALE_CONVENTIONS = {
    "cs": "Chemical-shift scale: F1 divided by its chemical-shift coefficient, so "
          "delta_F1 = delta_CS + B * delta_QIS (B = -10/17 for every MQMAS/STMAS "
          "on a CT); Millot & Man 2002 / Amoureux 'C_z' convention",
    "unified": "Unified: F1 divided by 1 + |kappa_t2| (Grandinetti 2011, Eq. 197); "
               "gives delta_F1 = 17/8 delta_CS + ... for I = 3/2",
    "none": "No scaling: F1 left in the sheared coherence's own Hz",
}


# ---------------------------------------------------------------------------
# Pathway coefficients
# ---------------------------------------------------------------------------

def spin_of(nucleus) -> float | None:
    """Nuclear spin of ``nucleus`` (any spelling openNMR reads, e.g.
    ``"27Al"``, ``"Al27"``), or ``None`` if unknown."""
    if nucleus is None:
        return None
    from .two_d import canonical_nucleus
    from .utils import _GAMMA_DF
    name = canonical_nucleus(nucleus) or str(nucleus)
    row = _GAMMA_DF.loc[_GAMMA_DF["Nuclide"] == name]
    if row.empty:
        return None
    try:
        return float(row["Spin"].iloc[0])
    except (TypeError, ValueError):
        return None


def level_functions(spin: float, m: float) -> dict:
    """The level functions ``E(m)`` (see the module docstring) for one
    ``m`` level of a spin ``I``: keys ``cs``, ``qis``, ``q2``, ``q4``."""
    i = float(spin)
    ii = i * (i + 1.0)
    return {
        "cs": m,
        "qis": m * (ii - 3.0 * m * m),
        "q2": m * (8.0 * ii - 12.0 * m * m - 3.0),
        "q4": m * (18.0 * ii - 34.0 * m * m - 5.0),
    }


def coherence_coefficients(spin: float, m_a: float, m_b: float) -> dict:
    """Frequency coefficients of the coherence ``|m_a><m_b|`` (order
    ``p = m_a - m_b``): ``E(m_b) - E(m_a)`` for each component, so the
    observed central transition (``m_a=-1/2, m_b=+1/2``, p = -1) has
    ``cs = 1`` and positive quadrupolar coefficients."""
    if abs(m_a) > spin + 1e-9 or abs(m_b) > spin + 1e-9:
        raise ValueError(f"m = {m_a}, {m_b} is outside a spin {spin}.")
    ea, eb = level_functions(spin, m_a), level_functions(spin, m_b)
    return {k: eb[k] - ea[k] for k in ea}


def symmetric_coherence(spin: float, p: int) -> dict:
    """Coefficients of the symmetric (multiple-quantum) coherence of
    order ``p`` (odd; ``p = -1`` is the central transition)."""
    if p % 2 == 0 or abs(p) > 2 * spin + 1e-9:
        raise ValueError(f"A symmetric coherence of a spin {spin} needs an odd |p| <= {2 * spin:g}; got {p}.")
    return coherence_coefficients(spin, p / 2.0, -p / 2.0)


def satellite_coherence(spin: float, order: int = 1, p: int = -1) -> dict:
    """Coefficients of the ``order``-th satellite transition
    (``order=1``: m = +-3/2 <-> +-1/2) as a single-quantum coherence of
    order ``p`` (+-1). Both satellites of a pair have the same
    coefficients (the odd level functions make them so)."""
    if p not in (-1, 1):
        raise ValueError("A satellite transition is single-quantum: p = -1 or +1.")
    m_low, m_high = order - 0.5, order + 0.5
    if m_high > spin + 1e-9:
        raise ValueError(f"A spin {spin} has no satellite transition of order {order}.")
    c = coherence_coefficients(spin, m_low, m_high)          # p = -1
    return c if p == -1 else {k: -v for k, v in c.items()}


def _p2(theta_deg: float) -> float:
    c = math.cos(math.radians(theta_deg))
    return 0.5 * (3 * c * c - 1)


def _p4(theta_deg: float) -> float:
    c2 = math.cos(math.radians(theta_deg)) ** 2
    return (35 * c2 * c2 - 30 * c2 + 3) / 8.0


def preset_kappas(spin: float | None) -> list[dict]:
    """Every standard shear ratio ``kappa_t2`` for a spin (MQMAS echo
    orders up to 2I, STMAS satellites), as ``{"label", "kappa"}`` --
    what :func:`estimate_shear` compares a measured ratio against. A
    spin-1/2 or unknown spin gets the solution/sideband tilts instead."""
    out = []
    if spin and spin > 0.5:
        ct = symmetric_coherence(spin, -1)
        for order in range(3, int(round(2 * spin)) + 1, 2):
            for p in (-order, order):
                mq = symmetric_coherence(spin, p)
                if mq["q4"] * ct["q4"] < 0:
                    out.append({"label": f"MQMAS {order}Q echo (I = {_frac(spin)})",
                                "kappa": mq["q4"] / ct["q4"]})
        for order in range(1, int(round(spin - 0.5)) + 1):
            st = _echo_satellite(spin, order, ct)
            out.append({"label": f"STMAS ST{order} echo (I = {_frac(spin)})",
                        "kappa": st["q4"] / ct["q4"]})
    return out


def _echo_satellite(spin: float, order: int, ct: dict, pathway: str = "echo") -> dict:
    """The satellite coherence (p1 = -1 or +1) whose rank-4 anisotropy
    has the opposite sign to the CT's (``"echo"``) or the same sign."""
    for p in (-1, 1):
        st = satellite_coherence(spin, order, p)
        if (st["q4"] * ct["q4"] < 0) == (pathway == "echo"):
            st["p"] = p
            return st
    st = satellite_coherence(spin, order, -1)
    st["p"] = -1
    return st


def _frac(x: float) -> str:
    """``1.5 -> '3/2'``, ``2.0 -> '2'``, else a short decimal."""
    for den in (1, 2, 3, 4, 6, 8, 9, 12, 16, 17, 31):
        num = x * den
        if abs(num - round(num)) < 1e-9:
            num = int(round(num))
            return f"{num}" if den == 1 else f"{num}/{den}"
    return f"{x:.4g}"


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass
class ShearPlan:
    """What :func:`shear_2d` will do, and what the result's axes mean.

    Every coefficient is in the displayed-axis frame (see the module
    docstring). ``a`` is the component removed from F1 (shear parallel
    to t2), ``b`` the one removed from F2 (shear parallel to t1); ``cs``
    and ``qis`` are the chemical-shift and isotropic-quadrupolar-shift
    coefficients in (t1, t2), used only to reference and describe the
    result. Construct with :func:`make_shear_plan`.
    """

    kappa_t2: float = 0.0
    kappa_t1: float = 0.0
    scale_f1: float = 1.0
    scale_f2: float = 1.0
    cs: tuple = (1.0, 1.0)
    qis: tuple | None = None
    a: tuple | None = None
    b: tuple | None = None
    experiment: str = "custom_kappa"
    convention: str = "none"
    f1_sign: int = 1
    spin: float | None = None
    label: str = ""
    notes: list = field(default_factory=list)

    # -- what the result means --------------------------------------
    @property
    def cs_after(self) -> tuple:
        """Chemical-shift coefficients (F1, F2) after both shears,
        before scaling."""
        c1 = self.cs[0] - self.kappa_t2 * self.cs[1]
        c2 = self.cs[1] - self.kappa_t1 * c1
        return c1, c2

    @property
    def qis_after(self) -> tuple | None:
        if self.qis is None:
            return None
        q1 = self.qis[0] - self.kappa_t2 * self.qis[1]
        q2 = self.qis[1] - self.kappa_t1 * q1
        return q1, q2

    @property
    def f1_cs_factor(self) -> float:
        """``A`` in delta_F1 = A delta_CS + B delta_QIS(CT)."""
        return self.cs_after[0] / self.scale_f1

    @property
    def f1_qis_factor(self) -> float | None:
        """``B`` in delta_F1 = A delta_CS + B delta_QIS(CT), with
        delta_QIS(CT) the central transition's isotropic quadrupolar
        shift as seen in F2. ``None`` when the plan has no QIS."""
        q = self.qis_after
        if q is None or not self.qis[1]:
            return None
        return q[0] / (self.scale_f1 * self.qis[1])

    @property
    def f2_cs_factor(self) -> float:
        return self.cs_after[1] / self.scale_f2

    def sw_factor(self) -> tuple:
        """Factors the spectral widths are divided by: (|s1|, |s2|)."""
        return abs(self.scale_f1), abs(self.scale_f2)

    def describe(self) -> str:
        """Plain-language account of the plan: the ratios, the scaling,
        and what each axis of the result reports."""
        lines = [self.label or EXPERIMENTS.get(self.experiment, self.experiment)]
        if self.kappa_t2:
            src = (f" = a1/a2 = {self.a[0]:.6g}/{self.a[1]:.6g}" if self.a else "")
            lines.append(f"Shear parallel to t2: kappa_t2 = {self.kappa_t2:+.6f}"
                         f" ({_frac(self.kappa_t2)}){src}. Every peak moves "
                         "(nu1, nu2) -> (nu1 - kappa_t2*nu2, nu2).")
        if self.kappa_t1:
            lines.append(f"Shear parallel to t1: kappa_t1 = {self.kappa_t1:+.6f}"
                         f" ({_frac(self.kappa_t1)}). Every peak moves "
                         "(nu1, nu2) -> (nu1, nu2 - kappa_t1*nu1).")
        if not self.kappa_t1 and not self.kappa_t2:
            lines.append("No shear (scaling only).")
        conv = {"cs": "chemical-shift scale", "unified": "unified, 1 + |kappa_t2|",
                "none": "no scaling"}.get(self.convention, "custom factor")
        lines.append(f"F1 scaled by 1/{abs(self.scale_f1):.6g} ({conv})"
                     + ("" if self.scale_f1 > 0 else "; F1 reversed so the axis still descends")
                     + (f"; F2 scaled by 1/{abs(self.scale_f2):.6g}" if abs(self.scale_f2) != 1 else "")
                     + ".")
        a_f1 = self.f1_cs_factor
        b_f1 = self.f1_qis_factor
        if abs(self.cs_after[0]) < 1e-12:
            lines.append("F1 holds no chemical shift (a J or sideband dimension, in Hz about "
                         f"zero); F2 reports {self.f2_cs_factor:.6g} delta_CS.")
        elif self.cs[0] or self.cs[1]:
            text = f"F1 reports delta_F1 = {a_f1:.6g} delta_CS"
            if b_f1 is not None:
                text += f" {'+' if b_f1 >= 0 else '-'} {abs(b_f1):.6g} delta_QIS(CT)"
                text += f"  (A = {_frac(a_f1)}, B = {_frac(b_f1)})"
            lines.append(text + ".")
            if abs(self.f2_cs_factor - 1) > 1e-9:
                lines.append(f"F2 reports {self.f2_cs_factor:.6g} delta_CS.")
        if self.f1_sign < 0:
            lines.append("F1 is taken as displayed with the opposite sense to the coherence "
                         "frequency (f1_sign = -1), so every t1 coefficient is negated.")
        lines.extend(self.notes)
        return "\n".join(lines)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.update(cs_after=self.cs_after, qis_after=self.qis_after,
                 f1_cs_factor=self.f1_cs_factor, f1_qis_factor=self.f1_qis_factor)
        return d


def _f1_scale(convention, c1_after: float, kappa_t2: float) -> float:
    """The signed F1 scale for a convention. The sign follows the
    chemical-shift coefficient, so F1 always increases with delta_CS
    as F2 does (the data are reversed if that needs it)."""
    sign = -1.0 if c1_after < 0 else 1.0
    if isinstance(convention, (int, float)) and not isinstance(convention, bool):
        if not convention:
            raise ValueError("A custom F1 scale factor cannot be zero.")
        return sign * abs(float(convention))
    if convention == "cs":
        if abs(c1_after) < 1e-12:
            raise ValueError(
                "The chemical-shift scale needs a chemical shift in F1, and this "
                "plan has none after shearing (e.g. a J-resolved or sideband F1). "
                "Use convention='none' or a number.")
        return c1_after
    if convention == "unified":
        return sign * (1.0 + abs(kappa_t2))
    if convention == "none":
        return sign
    raise ValueError(f"Unknown scaling convention {convention!r}; use one of "
                     f"{list(SCALE_CONVENTIONS)} or a number.")


def default_f1_sign(spectrum=None, experiment: str = "mqmas") -> int:
    """The F1 direction (``f1_sign``) to start from for ``spectrum``.

    TopSpin's own ``xfshear`` (MQMAS/STMAS) always applies the phase
    that moves a peak by ``nu1 + R nu2`` in the displayed Bruker F1,
    whatever the quadrature mode. In openNMR's convention that is
    ``f1_sign = -1`` for the p = 0 -> +3 -> -1 echo: checked on a
    27Al 3QMAS of YAG (States-TPPI, I = 5/2) against TopSpin's own
    ``xfshear`` result, where the correlation of the two sheared
    spectra is 0.94 with -1 and 0.22 with +1 (sites smeared into
    diagonal streaks), and peaks at kappa = 19/12 exactly. So Bruker
    MQMAS/STMAS data default to -1; everything else (other vendors,
    other experiments) to +1. Check against the data
    (:func:`estimate_shear`) where in doubt."""
    if spectrum is None or experiment not in ("mqmas", "stmas"):
        return 1
    if _is_dict(spectrum):      # NMR2dimport's dict is always TopSpin data
        return -1
    return -1 if getattr(spectrum.metadata, "vendor", None) == "bruker" else 1


def make_shear_plan(experiment: str = "mqmas", *, spin: float | None = None,
                    nucleus: str | None = None, mq_order: int = 3,
                    pathway: str = "echo", satellite: int = 1,
                    das_angles: tuple = (37.38, 79.19), second_shear: bool = False,
                    split_t1: bool = False,
                    f1_sign: int | None = None, convention=None, spectrum=None,
                    kappa_t2: float | None = None, kappa_t1: float | None = None,
                    a: tuple | None = None, b: tuple | None = None,
                    cs: tuple | None = None, qis: tuple | None = None,
                    scale_f2: float = 1.0) -> ShearPlan:
    """Build a :class:`ShearPlan` for a known experiment, or from your
    own numbers.

    Parameters
    ----------
    experiment : str
        One of :data:`EXPERIMENTS`.
    spin, nucleus : quadrupolar spin I (or a nucleus to look it up from).
        Needed for ``mqmas``, ``stmas`` and ``das``.
    mq_order : int
        |p1| for MQMAS (3, 5, 7, 9; at most 2I).
    pathway : ``"echo"`` (default; the pathway whose rank-4 anisotropy
        has the opposite sign to the CT's and so refocuses) or
        ``"antiecho"``.
    satellite : int
        STMAS satellite order (1: +-3/2 <-> +-1/2; 2: +-5/2 <-> +-3/2).
    das_angles : (theta1, theta2) in degrees, for ``das``.
    second_shear : bool
        DAS only: also shear parallel to t1 so each site's F2 pattern is
        centred on zero (removes the isotropic part from F2).
    split_t1 : bool
        MQMAS/STMAS only: the data were acquired with a split-t1
        sequence (Brown & Wimperis), in which t1 is shared between the
        MQ (or satellite) coherence, for 1/(1 + |kappa|) of it, and the
        CT, for |kappa|/(1 + |kappa|), so the anisotropy refocuses within
        t1 and the spectrum arrives already sheared -- and already in the
        unified scaling. No shear is applied; F1 is scaled and referenced
        with the effective t1 coefficients
        ``x_eff = (x_MQ - kappa x_CT) / (1 + |kappa|)``. Assumes the F1
        dwell (Bruker IN0) is the whole t1 increment.
    spectrum : the 2D the plan is for (optional). With ``f1_sign`` left at
        ``None`` it decides the F1 direction, see :func:`default_f1_sign`
        (TopSpin data: -1 for MQMAS/STMAS; +1 otherwise).
    f1_sign : +1 or -1, or ``None`` (default: from ``spectrum``, else +1)
        +1 when F1 is displayed with the same sense as the t1
        coherence's frequency, -1 when reversed (receiver/quadrature
        convention). :func:`estimate_shear` tells you which.
    convention : ``"cs"``, ``"unified"``, ``"none"`` or a number
        How F1 is scaled; see :data:`SCALE_CONVENTIONS`. Default (None):
        ``"cs"``, or ``"none"`` for experiments with no chemical shift in
        F1 (J-resolved, PASS/MAT).
    kappa_t2, kappa_t1 : custom shear ratios (``custom_kappa``).
    a, b : (t1, t2) coefficients of the component to remove from F1 / F2
        (``custom_coefficients``).
    cs, qis : (t1, t2) chemical-shift / isotropic-quadrupolar-shift
        coefficients, for referencing and describing F1. Defaults: the
        experiment's own; (1, 1) and none for the custom modes.
    scale_f2 : float
        Divide F2 by this (rarely wanted; default 1).
    """
    exp = experiment.lower()
    if exp not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; use one of {list(EXPERIMENTS)}.")
    if f1_sign is None:
        f1_sign = default_f1_sign(spectrum, exp)
    if f1_sign not in (1, -1):
        raise ValueError("f1_sign must be +1 or -1.")
    if spin is None and nucleus is not None:
        spin = spin_of(nucleus)
    notes: list[str] = []
    k_t2 = k_t1 = 0.0
    a_c = b_c = None
    cs_c, qis_c = (1.0, 1.0), None
    label = EXPERIMENTS[exp]

    def need_spin():
        if spin is None or spin <= 0.5:
            raise ValueError(f"{exp.upper()} needs a quadrupolar spin (I > 1/2); give spin= or "
                             f"nucleus= (got {spin}).")

    if exp == "mqmas":
        need_spin()
        if mq_order % 2 == 0 or mq_order < 3 or mq_order > 2 * spin + 1e-9:
            raise ValueError(f"MQ order must be odd, >= 3 and <= 2I = {2 * spin:g}; got {mq_order}.")
        ct = symmetric_coherence(spin, -1)
        choices = [symmetric_coherence(spin, p) for p in (-mq_order, mq_order)]
        echo = [c for c in choices if c["q4"] * ct["q4"] < 0]
        anti = [c for c in choices if c["q4"] * ct["q4"] > 0]
        mq = (echo if pathway == "echo" else anti)[0]
        p1 = -mq_order if mq is choices[0] else mq_order
        a_c, cs_c, qis_c = (mq["q4"], ct["q4"]), (mq["cs"], ct["cs"]), (mq["qis"], ct["qis"])
        label = (f"MQMAS {mq_order}Q, I = {_frac(spin)}, {pathway} pathway "
                 f"(p = 0 -> {p1:+d} -> -1)")
        if pathway != "echo":
            notes.append("The antiecho pathway does not refocus the anisotropy; shearing it "
                         "alone is unusual (normally echo and antiecho are combined first).")
    elif exp == "stmas":
        need_spin()
        ct = symmetric_coherence(spin, -1)
        st = _echo_satellite(spin, satellite, ct, pathway)
        a_c, cs_c, qis_c = (st["q4"], ct["q4"]), (st["cs"], ct["cs"]), (st["qis"], ct["qis"])
        label = (f"STMAS ST{satellite} (I = {_frac(spin)}), {pathway} pathway "
                 f"(satellite p = {st['p']:+d} in t1, CT p = -1 in t2)")
        notes.append("STMAS assumes an exact magic angle and rotor-synchronised t1: the "
                     "first-order satellite broadening is not removed by shearing.")
    elif exp == "das":
        need_spin()
        ct = symmetric_coherence(spin, -1)
        t1_ang, t2_ang = das_angles
        a_c = (ct["q4"] * _p4(t1_ang), ct["q4"] * _p4(t2_ang))
        cs_c, qis_c = (ct["cs"], ct["cs"]), (ct["qis"], ct["qis"])
        label = f"DAS {t1_ang:g} / {t2_ang:g} deg (I = {_frac(spin)})"
        if second_shear:
            b_c = cs_c          # the isotropic part (CS and QIS share the CT's ratio)
        r2 = _p2(t1_ang) / _p2(t2_ang) if _p2(t2_ang) else float("inf")
        r4 = a_c[0] / a_c[1] if a_c[1] else float("inf")
        if abs(r2 - r4) > 1e-3:
            notes.append(f"These angles do not refocus rank 2 and rank 4 at the same "
                         f"t2/t1 ratio ({-r2:.4g} vs {-r4:.4g}); after the shear the rank-2 "
                         "anisotropy is still in F1.")
    elif exp == "mat":
        # Grandinetti et al., Prog. NMR Spectrosc. 59 (2011) 121, Sec. 6:
        # MAT is the discrete trajectory (magic-angle hopping, three
        # equally spaced orientations on the magic cone) that averages a
        # rank-2 anisotropy to zero. The isotropic shift (S.p_I) evolves
        # through t1 and t2 alike (coefficients 1, 1); the anisotropic
        # frequencies refocus within the t1 cycle and evolve in t2 only
        # (0, 1) -- the opposite assignment to PASS, where the isotropic
        # shift is the (0, 1) component and the sidebands (1, 1).
        cs_c, b_c = (1.0, 1.0), (1.0, 1.0)
        notes.append("MAT: F1 reports the isotropic shift (referenced like an ordinary "
                     "F1) and F2 the anisotropic pattern centred on zero, i.e. the "
                     "isotropic shift is taken out of F2. This assumes the isotropic shift "
                     "evolves for the whole t1 increment and the anisotropy is fully "
                     "refocused in it. If your sequence lets the isotropic shift evolve for "
                     "only a fraction f of each increment, use custom_coefficients with "
                     "b = cs = (f, 1). If the pattern comes out mirrored, use f1_sign = -1.")
    elif exp in ("shifting_p", "shifting_d"):
        # Walder et al., J. Chem. Phys. 142 (2015) 014201 and Aleksis et al.,
        # Solid State NMR 101 (2019) 51: two symmetry pathways, the shift
        # (p_I) and the first-order quadrupolar (d_I) interaction. In the
        # p-echo experiment the p_I echo is at t2 = t1 and the d_I echo at
        # t2 = 0, so p_I has the (t1, t2) coefficients (-1, 1) and d_I (0, 1);
        # the d-echo experiment swaps them. A unit shear parallel to t1 then
        # leaves the (-1, 1) component in F1 only.
        if exp == "shifting_p":
            cs_c, b_c = (-1.0, 1.0), (-1.0, 1.0)
        else:
            cs_c, b_c = (0.0, 1.0), (-1.0, 1.0)
            if convention is None:
                convention = "none"
        notes.append(("Shifting-p: F1 holds the paramagnetic shift (referenced as an ordinary "
                      "F1) and F2 the first-order quadrupolar pattern about zero, in Hz. "
                      if exp == "shifting_p" else
                      "Shifting-d: F1 holds the first-order quadrupolar pattern about zero "
                      "(Hz, unreferenced) and F2 the paramagnetic shift. ")
                     + "The t2 origin must be the end of the last pulse (acquire the whole "
                     "echo, remove the digital-filter group delay, no echo alignment) and t1 "
                     "must be the real evolution time of each increment. Adiabatic "
                     "(double/quadruple SHAP) d-echo data share this geometry. If the "
                     "correlation looks mirrored in F1, use f1_sign = -1.")
    elif exp in ("jres", "pass"):
        cs_c, b_c = (0.0, 1.0), (1.0, 1.0)
        if convention is None:
            convention = "none"
        if exp == "pass":
            notes.append("PASS (Grandinetti et al. 2011, Sec. 3.4.3): F1 holds the sideband "
                         "order x MAS rate, F2 the isotropic shift; if the sidebands move "
                         "away from (not onto) the isotropic peaks, use f1_sign = -1.")
            notes.append("t1 is sampled N times per rotor period, so F1 repeats every "
                         "N x MAS rate: sidebands of order beyond +-N/2 fold back in F1 and "
                         "are then sheared by the wrong amount. Use N at least as large as "
                         "the number of sidebands (5 pulse PASS: N = 8-16 typical).")
    elif exp == "custom_kappa":
        k_t2 = float(kappa_t2 or 0.0)
        k_t1 = float(kappa_t1 or 0.0)
        cs_c = tuple(float(v) for v in (cs if cs is not None else (1.0, 1.0)))
        qis_c = tuple(float(v) for v in qis) if qis is not None else None
    else:   # custom_coefficients
        if a is None and b is None:
            raise ValueError("Give a=(a1, a2) and/or b=(b1, b2).")
        a_c = tuple(float(v) for v in a) if a is not None else None
        b_c = tuple(float(v) for v in b) if b is not None else None
        cs_c = tuple(float(v) for v in (cs if cs is not None else (1.0, 1.0)))
        qis_c = tuple(float(v) for v in qis) if qis is not None else None

    if exp not in ("custom_kappa", "custom_coefficients"):
        if cs is not None:
            cs_c = tuple(float(v) for v in cs)
        if qis is not None:
            qis_c = tuple(float(v) for v in qis)

    # F1's displayed sense: negate every t1 coefficient.
    def flip(pair):
        return None if pair is None else (f1_sign * pair[0], pair[1])
    if exp == "custom_kappa":
        k_t2, k_t1 = f1_sign * k_t2, f1_sign * k_t1
        cs_c, qis_c = flip(cs_c), flip(qis_c)
    else:
        a_c, b_c, cs_c, qis_c = flip(a_c), flip(b_c), flip(cs_c), flip(qis_c)
        if a_c is not None:
            if not a_c[1]:
                raise ValueError("a2 is zero: the component to remove from F1 is not in t2, "
                                 "so no shear parallel to t2 can remove it.")
            k_t2 = a_c[0] / a_c[1]
        if b_c is not None:
            b1_after = b_c[0] - k_t2 * b_c[1]
            if abs(b1_after) < 1e-12:
                raise ValueError("After the first shear the component b is no longer in F1, "
                                 "so a shear parallel to t1 cannot remove it from F2.")
            k_t1 = b_c[1] / b1_after

    if split_t1:
        if exp not in ("mqmas", "stmas"):
            raise ValueError("split_t1 applies to MQMAS and STMAS only.")
        # The sequence has done the shear (and the unified scaling) itself:
        # what evolves in F1 is the time-weighted mix of the two coherences.
        f = 1.0 + abs(k_t2)
        frac_mq, frac_ct = 1.0 / f, abs(k_t2) / f

        def eff(pair):
            return None if pair is None else ((pair[0] - k_t2 * pair[1]) / f, pair[1])
        cs_c, qis_c = eff(cs_c), eff(qis_c)
        notes.append(f"Split-t1: acquired already sheared (t1 shared {_frac(frac_mq)} "
                     f"{'MQ' if exp == 'mqmas' else 'satellite'} : {_frac(frac_ct)} CT for "
                     f"kappa = {_frac(k_t2)}), so no shear is applied; F1 is only scaled and "
                     "referenced. This assumes the F1 dwell (IN0) is the whole t1 increment; "
                     "if it is only the MQ part, use a custom F1 factor.")
        label += " – split-t1 acquisition"
        a_c, k_t2 = None, 0.0
    if convention is None:
        convention = "cs"
    plan = ShearPlan(kappa_t2=k_t2, kappa_t1=k_t1, cs=cs_c, qis=qis_c, a=a_c, b=b_c,
                     experiment=exp, f1_sign=f1_sign, spin=spin, label=label, notes=notes,
                     convention=convention if isinstance(convention, str) else "custom")
    plan.scale_f1 = _f1_scale(convention, plan.cs_after[0], k_t2)
    if exp == "shifting_d" and convention == "none":
        # F1 holds the quadrupolar component, not the shift: its sign (not
        # the shift's, which is absent from F1) decides the axis direction,
        # so that F1 reads the quadrupolar frequency with its own sign.
        plan.scale_f1 = -1.0 if b_c[0] - k_t2 * b_c[1] < 0 else 1.0
    c2 = plan.cs_after[1]
    plan.scale_f2 = (-1.0 if c2 < 0 else 1.0) * abs(float(scale_f2 or 1.0))
    return plan


# ---------------------------------------------------------------------------
# Data access: NMRSpectrum or the notebook's 2D dict
# ---------------------------------------------------------------------------

def _is_dict(obj) -> bool:
    return isinstance(obj, dict)


def _midpoint(hz: np.ndarray) -> float:
    return 0.5 * (float(hz[0]) + float(hz[-1]))


def _sw_from(raw: dict, dim: str, span: float, n: int) -> tuple[float, str]:
    """Spectral width (Hz) of one dimension: the recorded value when the
    reader kept one, else the axis span (openNMR's axes span exactly SW).
    A recorded value disagreeing with the axis by more than a few
    points' worth falls back to the axis, which is what the data were
    built on."""
    candidates = []
    if dim == "f2":
        if raw.get("fid_dt"):
            candidates.append((1.0 / float(raw["fid_dt"]), "acquisition dwell"))
        for key, src in (("procs", "SW_p"), ("acqus", "SW_h")):
            d = raw.get(key) or {}
            if d.get(src):
                candidates.append((float(d[src]), f"{key} {src}"))
    else:
        if raw.get("fid_dt_f1"):
            candidates.append((1.0 / float(raw["fid_dt_f1"]), "acquisition dwell (F1)"))
        d = raw.get("proc2s") or {}
        if d.get("SW_p"):
            candidates.append((float(d["SW_p"]), "proc2s SW_p"))
    for value, src in candidates:
        if value > 0 and abs(value - span) <= 4 * span / max(n, 1):
            return value, src
    return span, "axis span"


def _imag_partner_axis(raw: dict, vendor: str | None) -> int | None:
    """Which axis ``data_imag`` is the quadrature partner along: 0 (F1)
    for hypercomplex data (States, States-TPPI, echo-antiecho; and
    TopSpin's ``2ri``), 1 (F2) for QF-type data, ``None`` if unknown."""
    if "imag_partner_axis" in raw:
        return raw["imag_partner_axis"]
    if "fid_dt_f1" in raw:          # read from the raw ser by openNMR
        return 0 if raw.get("fn_mode") in (4, 5, 6) else 1
    if vendor == "bruker" and raw.get("proc2s") is not None:
        return 0                    # TopSpin's 2rr + 2ri
    return None


def _view(obj) -> dict:
    """Everything the shear needs, from either input type."""
    if _is_dict(obj):
        if obj.get("pseudo_2d"):
            raise ValueError("A pseudo-2D series has no frequency F1 to shear.")
        data = np.asarray(obj["spectrum"], dtype=float)
        x_hz, y_hz = np.asarray(obj["x_Hz"], float), np.asarray(obj["y_Hz"], float)
        if (obj.get("SI1") and len(x_hz) != int(obj["SI1"])) or (
                obj.get("SI2") and len(y_hz) != int(obj["SI2"])):
            raise ValueError("This 2D was imported with display limits (f1l/f1r/f2l/f2r), "
                             "which cut the axes; shearing needs the whole spectrum. "
                             "Re-import with the limits left at 0.")
        sf2, sf1 = float(obj["SF1"]), float(obj.get("SF2") or obj["SF1"])
        raw = {}
        sw2 = float(obj.get("SW1") or abs(x_hz[0] - x_hz[-1]))
        sw1 = float(obj.get("SW2") or abs(y_hz[0] - y_hz[-1]))
        return dict(data=data, imag=None, partner=None, x_hz=x_hz, y_hz=y_hz,
                    x_ppm=np.asarray(obj["x_ppm"], float), y_ppm=np.asarray(obj["y_ppm"], float),
                    sf1=sf1, sf2=sf2, sw1=sw1, sw2=sw2, sw_src=("SW2", "SW1"), raw=raw,
                    homonuclear=bool(obj.get("homonuclear")) or obj.get("NUC1") == obj.get("NUC2"),
                    nucleus=obj.get("NUC1"), vendor="bruker")
    spec = obj
    if spec.ndim != 2:
        raise ValueError("Shearing needs a 2D spectrum.")
    if spec.metadata.pseudo_2d or spec.axes[0].kind != "frequency":
        raise ValueError("A pseudo-2D series has no frequency F1 to shear.")
    meta = spec.metadata
    raw = meta.raw or {}
    sf2 = float(meta.sfo1 or 1.0)
    sf1 = float(meta.sfo1_indirect or meta.sfo1 or 1.0)
    ax1, ax2 = spec.axes
    y_ppm = np.asarray(ax1.ppm, float)
    x_ppm = np.asarray(ax2.ppm, float)
    y_hz = np.asarray(ax1.hz, float) if ax1.hz is not None else y_ppm * sf1
    x_hz = np.asarray(ax2.hz, float) if ax2.hz is not None else x_ppm * sf2
    n1, n2 = spec.data.shape
    sw2, src2 = _sw_from(raw, "f2", abs(x_hz[0] - x_hz[-1]), n2)
    sw1, src1 = _sw_from(raw, "f1", abs(y_hz[0] - y_hz[-1]), n1)
    from .two_d import canonical_nucleus
    nuc, nuc_i = canonical_nucleus(meta.nucleus), canonical_nucleus(meta.nucleus_indirect)
    homonuclear = bool(meta.homonuclear) or nuc_i is None or nuc_i == nuc
    return dict(data=np.asarray(spec.data, float),
                imag=None if spec.data_imag is None else np.asarray(spec.data_imag, float),
                partner=_imag_partner_axis(raw, meta.vendor), x_hz=x_hz, y_hz=y_hz,
                x_ppm=x_ppm, y_ppm=y_ppm, sf1=sf1, sf2=sf2, sw1=sw1, sw2=sw2,
                sw_src=(src1, src2), raw=raw, homonuclear=homonuclear, nucleus=nuc,
                vendor=meta.vendor)


def _bin_offsets(n: int, sw: float) -> np.ndarray:
    """Carrier-relative frequency (Hz) of each point of a descending,
    fftshifted axis of ``n`` points -- the frequencies actually in the
    data (openNMR's ``fft_spectrum`` order)."""
    return -np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sw))


# ---------------------------------------------------------------------------
# The exact fractional shift
# ---------------------------------------------------------------------------

def _shift_lines(x: np.ndarray, axis: int, shift_pts: np.ndarray, *, complex_route: bool,
                 t0_pts: float = 0.0) -> np.ndarray:
    """Circularly shift every line of ``x`` along ``axis`` by its own
    (fractional) number of points, towards higher index, with the
    Fourier shift theorem. ``shift_pts`` has one value per line (the
    other axis).

    ``complex_route`` : ``x`` is complex along ``axis`` (its quadrature
    partner is present), so its conjugate domain is the physical time
    signal, sampled at ``(m + t0_pts) * dwell`` for m = 0..N-1. The ramp
    then is exactly the time-domain shear, with no zero-fill needed.
    Otherwise ``x`` is real: its conjugate domain is the time signal
    plus its mirror image at negative times, so the ramp uses signed
    times and the result is real. That is exact as long as the two do
    not overlap, i.e. the dimension was zero-filled at least twofold
    (see :func:`_mirror_overlap`).
    """
    x = np.moveaxis(np.asarray(x), axis, -1)
    n = x.shape[-1]
    y = np.fft.ifft(x, axis=-1)
    if complex_route:
        m = np.arange(n, dtype=float) + t0_pts
        ramp = np.exp(2j * np.pi * np.outer(shift_pts, m) / n)
    else:
        ms = np.fft.fftfreq(n) * n
        mt = np.sign(ms) * (np.abs(ms) + t0_pts)
        ramp = np.exp(2j * np.pi * np.outer(shift_pts, mt) / n)
        if n % 2 == 0:
            ramp[:, n // 2] = np.cos(np.pi * shift_pts)   # Nyquist: keep it Hermitian
    out = np.fft.fft(y * ramp, axis=-1)
    if not complex_route and not np.iscomplexobj(x):
        out = out.real
    return np.moveaxis(out, -1, axis)


def _mirror_overlap(plane: np.ndarray, axis: int) -> float:
    """Signal at the very middle of the conjugate (time) domain along
    ``axis`` -- time +-N/2, where the signal and its mirror image meet
    -- relative to the maximum. Zero when the dimension was zero-filled
    at least twofold (an FID of N/2 points ends one point short of it),
    which is what the real-only shift needs: the signed-time ramp is
    exact for any signal confined to |t| < N/2, whole echoes included,
    and wrong for one that wraps past it."""
    y = np.abs(np.fft.ifft(np.moveaxis(plane, axis, -1), axis=-1))
    n = y.shape[-1]
    peak = float(y.max()) or 1.0
    mid = y[..., n // 2] if n % 2 == 0 else np.maximum(y[..., n // 2], y[..., n // 2 + 1])
    return float(np.max(mid) / peak)


def _phase_factor(view: dict, axis: int) -> np.ndarray:
    """The phase already applied along ``axis`` of the data, as one
    unit complex factor per point (data = unphased x factor).

    Built from, in order: a phase baked in by an earlier shear
    (``baked_phase_f1/f2``, radians per point); TopSpin's recorded
    PHC0/PHC1 (openNMR's own record for data read from the raw ser, or
    ``procs``/``proc2s`` for TopSpin's processed 2rr/2ri); and the
    user's own phase (``phase0``/``phase1``[``_f1``]). The same
    composition as the GUI's rebuild, :func:`io.bruker.apply_dimension_phase`."""
    from ..io import bruker as io_bruker
    raw = view["raw"]
    dim = "f1" if axis == 0 else "f2"
    ppm = view["y_ppm"] if axis == 0 else view["x_ppm"]
    n = len(ppm)
    baked = raw.get(f"baked_phase_{dim}")
    k0, k1 = io_bruker.RECORDED_PHASE_KEYS[dim]
    if baked is not None or k0 in raw or k1 in raw or "fid_dt" in raw:
        recorded = io_bruker.recorded_phase(raw, dim)
    else:   # TopSpin's processed data: its recorded phase is in procs/proc2s
        params = raw.get("proc2s" if dim == "f1" else "procs") or {}
        try:
            recorded = (float(params.get("PHC0", 0.0) or 0.0), float(params.get("PHC1", 0.0) or 0.0))
        except (TypeError, ValueError):
            recorded = (0.0, 0.0)
        if view.get("vendor") != "bruker":
            recorded = (0.0, 0.0)
    u0, u1 = ("phase0_f1", "phase1_f1") if dim == "f1" else ("phase0", "phase1")
    user = (float(raw.get(u0, 0.0) or 0.0), float(raw.get(u1, 0.0) or 0.0))
    factor = io_bruker.apply_dimension_phase(np.ones(n, complex), ppm, recorded, user, pivot_frac=0.5)
    if baked is not None and len(baked) == n:
        factor = factor * np.exp(1j * np.asarray(baked, float))
    return factor


def _varies(factor: np.ndarray) -> float:
    """Spread (degrees) of a phase factor across the axis -- non-zero
    when it holds a first-order term."""
    ang = np.unwrap(np.angle(factor))
    return float(np.degrees(ang.max() - ang.min())) if ang.size else 0.0


def _along(vec: np.ndarray, axis: int) -> np.ndarray:
    return vec[:, None] if axis == 0 else vec[None, :]


def _shear_planes(view: dict, axis: int, shift_pts: np.ndarray, t0_pts: float,
                  warnings_out: list, routes: dict, name: str):
    """Shift ``view``'s data (and imaginary plane) along ``axis``.
    Uses the exact complex route when ``data_imag`` is the partner
    along this axis, taking that dimension's phase off before the shift
    and putting it back after (a first-order phase does not commute
    with a shift); otherwise shifts each real plane on its own."""
    data, imag = view["data"], view["imag"]
    dim = "f1" if axis == 0 else "f2"
    factor = view["phase"][axis]
    if imag is not None and view["partner"] == axis:
        z = (data + 1j * imag) * _along(np.conj(factor), axis)
        z = _shift_lines(z, axis, shift_pts, complex_route=True, t0_pts=t0_pts)
        z = z * _along(factor, axis)
        view["data"], view["imag"] = z.real, z.imag
        routes[name] = "complex (exact; quadrature partner present)"
        return
    overlap = _mirror_overlap(data, axis)
    if overlap > 0.05:
        warnings_out.append(
            f"{name}: {dim.upper()} has only its real part here and its time signal reaches "
            f"half-way along the time domain ({overlap:.0%} of the maximum): it was "
            f"zero-filled less than twofold, or an earlier shear moved the echo that far. "
            f"The shift then mixes the signal with its mirror image. Zero-fill "
            f"{dim.upper()} further (at least twofold, more for a second shear), or load the "
            "data with its imaginary part.")
    spread = _varies(factor)
    if spread > 0.5:
        warnings_out.append(
            f"{name}: {dim.upper()} carries a first-order phase ({spread:.1f} deg across the "
            "spectrum), which a shift along it cannot carry without the imaginary part; expect "
            f"a phase error that varies across the spectrum. Shear before first-order phasing "
            f"{dim.upper()}, or keep its imaginary part.")
    view["data"] = _shift_lines(data, axis, shift_pts, complex_route=False, t0_pts=t0_pts)
    if imag is not None:
        view["imag"] = _shift_lines(imag, axis, shift_pts, complex_route=False, t0_pts=t0_pts)
    routes[name] = "real planes (exact when zero-filled at least twofold)"


# ---------------------------------------------------------------------------
# The public operation
# ---------------------------------------------------------------------------

def shear_2d(spectrum, plan: ShearPlan, *, t1_0: float = 0.0, t2_0: float = 0.0,
             source_label: str | None = None):
    """Shear and scale a 2D spectrum according to ``plan``.

    Parameters
    ----------
    spectrum : NMRSpectrum or dict
        A conventional (not pseudo) 2D: an :class:`~opennmr.core.spectrum.NMRSpectrum`
        from :func:`opennmr.io.load`, or the dict from
        :func:`~opennmr.notebook.two_d.NMR2dimport` (imported without
        display limits). The result is the same type.
    plan : ShearPlan
        From :func:`make_shear_plan`.
    t1_0, t2_0 : float
        Time (s) of the first point of t1 / t2, if not zero. ``t1_0``
        matters for the shear parallel to t2: ignoring a non-zero first
        increment leaves an F1 phase that changes across F2 by
        ``2 pi kappa_t2 nu2 t1_0``.
    source_label : str
        Name of the input, for the log.

    Returns
    -------
    The sheared and scaled spectrum. Its log -- what was done, the
    routes used, any warnings, and what F1 now reports -- is in
    ``metadata.raw["shear"]`` (NMRSpectrum) or ``["shear"]`` (dict).
    """
    view = _view(spectrum)
    view["phase"] = (_phase_factor(view, 0), _phase_factor(view, 1))
    n1, n2 = view["data"].shape
    warn_list: list[str] = []
    routes: dict = {}
    nu1 = _bin_offsets(n1, view["sw1"])
    nu2 = _bin_offsets(n2, view["sw2"])

    if view["raw"].get("align_echo") and plan.kappa_t2:
        warn_list.append(
            "Each row was rolled to its own echo top (Align Echo) when this 2D was processed. "
            "That moves every t1 increment – and, with echo-antiecho data, the p- and "
            "n-type rows differently – by its own amount, which is a row-dependent shear of "
            "its own; the shear here assumes every row shares one t2 origin. Reprocess "
            "without Align Echo (use one F2 first-order phase for the echo delay) and shear "
            "that.")
    if view["raw"].get("magnitude_mode"):
        warn_list.append("This spectrum is in magnitude mode: the shear interpolates a "
                         "non-phase-sensitive surface, so line shapes are approximate. "
                         "Shear the phase-sensitive spectrum where possible.")

    if plan.kappa_t2:
        shift = plan.kappa_t2 * nu2 * n1 / view["sw1"]
        _shear_planes(view, 0, shift, t1_0 * view["sw1"], warn_list, routes, "shear parallel to t2")
    if plan.kappa_t1:
        shift = plan.kappa_t1 * nu1 * n2 / view["sw2"]
        _shear_planes(view, 1, shift, t2_0 * view["sw2"], warn_list, routes, "shear parallel to t1")

    # --- relabel (scaling) ---------------------------------------------
    c1_after, c2_after = plan.cs_after
    s1, s2 = plan.scale_f1, plan.scale_f2
    c_ref2 = _midpoint(view["x_hz"])                  # F2 carrier, in the displayed Hz frame
    if view["homonuclear"]:
        c_ref1, sf_ref1 = c_ref2, view["sf2"]
    else:
        c_ref1, sf_ref1 = _midpoint(view["y_hz"]), view["sf1"]
    rel1 = view["y_hz"] - _midpoint(view["y_hz"])
    rel2 = view["x_hz"] - c_ref2
    y_hz = rel1 / s1 + (c1_after / s1) * c_ref1
    x_hz = rel2 / s2 + (c2_after / s2) * c_ref2
    data, imag = view["data"], view["imag"]
    # The phase each axis carries, in the result's point order (an
    # axis reversed with its quadrature partner is conjugated too).
    baked = [np.angle(view["phase"][0]), np.angle(view["phase"][1])]
    if s1 < 0:
        y_hz, data = y_hz[::-1], data[::-1]
        imag = None if imag is None else imag[::-1]
        baked[0] = baked[0][::-1]
        if view["partner"] == 0 and imag is not None:
            imag = -imag        # reversing a complex axis conjugates it
            baked[0] = -baked[0]
    if s2 < 0:
        x_hz, data = x_hz[::-1], data[:, ::-1]
        imag = None if imag is None else imag[:, ::-1]
        baked[1] = baked[1][::-1]
        if view["partner"] == 1 and imag is not None:
            imag = -imag
            baked[1] = -baked[1]
    y_ppm, x_ppm = y_hz / sf_ref1, x_hz / view["sf2"]
    data = np.ascontiguousarray(data)
    imag = None if imag is None else np.ascontiguousarray(imag)

    sw1_new, sw2_new = view["sw1"] / abs(s1), view["sw2"] / abs(s2)
    log_lines = [f"Shear/scale of {source_label or 'the selected 2D'}:", plan.describe()]
    log_lines.append(f"Spectral widths used: F1 {view['sw1']:.6g} Hz ({view['sw_src'][0]}), "
                     f"F2 {view['sw2']:.6g} Hz ({view['sw_src'][1]}); after scaling F1 "
                     f"{sw1_new:.6g} Hz, F2 {sw2_new:.6g} Hz.")
    if t1_0 or t2_0:
        log_lines.append(f"First points: t1_0 = {t1_0 * 1e6:.4g} us, t2_0 = {t2_0 * 1e6:.4g} us.")
    for name, route in routes.items():
        log_lines.append(f"{name}: {route}.")
    log_lines.extend(f"Warning: {w}" for w in warn_list)
    record = {
        "plan": plan.as_dict(), "t1_0": t1_0, "t2_0": t2_0, "routes": routes,
        "warnings": warn_list, "sw_before": (view["sw1"], view["sw2"]),
        "sw_after": (sw1_new, sw2_new), "source": source_label, "log": "\n".join(log_lines),
    }
    for w in warn_list:
        warnings.warn(w, stacklevel=2)

    if _is_dict(spectrum):
        out = dict(spectrum)
        out.update(spectrum=data, x_ppm=x_ppm, y_ppm=y_ppm, x_Hz=x_hz, y_Hz=y_hz,
                   SW1=sw2_new, SW2=sw1_new, SF2=sf_ref1, shear=record)
        return out

    from ..core.spectrum import NMRAxis, NMRSpectrum
    meta = spectrum.metadata
    raw = {k: v for k, v in (meta.raw or {}).items()
           if not k.startswith("fid") and not k.startswith("slice_phase")
           and k not in ("phase0", "phase1", "phase0_f1", "phase1_f1",
                         "recorded_phase0", "recorded_phase1", "recorded_phase0_f1",
                         "recorded_phase1_f1", "_read_used_saved_processing")}
    raw["shear"] = record
    raw["imag_partner_axis"] = view["partner"] if imag is not None else None
    # The recorded/user phase keys were dropped above (a reader-style
    # rebuild must not apply them again); what they did is still in the
    # data, so it is kept here per point for a later shear to undo.
    raw["baked_phase_f1"], raw["baked_phase_f2"] = baked
    metadata = dataclasses.replace(meta, raw=raw, sfo1_indirect=sf_ref1)
    axes = [NMRAxis(kind="frequency", ppm=y_ppm, hz=y_hz),
            NMRAxis(kind="frequency", ppm=x_ppm, hz=x_hz)]
    return NMRSpectrum(data=data, axes=axes, metadata=metadata, data_imag=imag)


# ---------------------------------------------------------------------------
# Measuring the ratio from the data
# ---------------------------------------------------------------------------

def _sharpness(plane: np.ndarray, axis_kept: int) -> float:
    """Inverse participation ratio of the projection onto ``axis_kept``
    (larger = sharper). The projection sums the signed data before
    taking its magnitude, so truncation wiggles along the other axis
    cancel instead of smearing the projection as they are sheared
    (summing |data| made a short-t1 J-resolved spectrum look sharpest
    unsheared)."""
    p = np.abs(plane.sum(axis=1 - axis_kept))
    total = p.sum()
    return float((p * p).sum() / (total * total)) if total else 0.0


def estimate_shear(spectrum, direction: str = "t2", *, kappa_range=(-4.0, 4.0),
                   n_coarse: int = 161, n_fine: int = 41, threshold: float = 0.03,
                   spin: float | None = None) -> dict:
    """Measure the shear ratio from the data: the kappa that makes the
    projection onto F1 (``direction="t2"``) or F2 (``"t1"``) sharpest.
    That is the ratio whose shear lines every ridge up with an axis.

    Compares the result against the standard ratios for the nucleus's
    spin (both signs: a match with the negative of a standard value
    means F1 is displayed reversed, so use ``f1_sign=-1``).

    Only the columns (rows) holding signal above ``threshold`` of the
    maximum are used -- each is sheared independently, so this is exact
    and fast.

    Returns a dict: ``kappa`` (best), ``kappas``/``scores`` (the scan),
    ``matches`` (standard ratios sorted by closeness, each with the
    ``f1_sign`` it implies), ``prominence`` (how far the best score
    stands above the scan's median, 0-1) and ``summary`` (plain text).
    Caveats: a distribution of chemical shifts runs along a different
    slope from the quadrupolar ridge and pulls the estimate towards it;
    the sign check is the robust part.
    """
    view = _view(spectrum)
    if spin is None:
        spin = spin_of(view["nucleus"])
    data = view["data"]
    n1, n2 = data.shape
    peak = float(np.abs(data).max()) or 1.0
    if direction == "t2":
        keep = np.flatnonzero(np.abs(data).max(axis=0) > threshold * peak)
        plane = data[:, keep]
        nu = _bin_offsets(n2, view["sw2"])[keep]
        factor = n1 / view["sw1"]
        axis, kept_axis = 0, 0
    elif direction == "t1":
        keep = np.flatnonzero(np.abs(data).max(axis=1) > threshold * peak)
        plane = data[keep, :]
        nu = _bin_offsets(n1, view["sw1"])[keep]
        factor = n2 / view["sw2"]
        axis, kept_axis = 1, 1
    else:
        raise ValueError("direction must be 't2' (shear parallel to t2) or 't1'.")
    if plane.size == 0:
        raise ValueError("No signal above the threshold to measure a shear from.")

    def score(k):
        sheared = _shift_lines(plane, axis, k * nu * factor, complex_route=False)
        return _sharpness(sheared, kept_axis)

    kappas = np.linspace(kappa_range[0], kappa_range[1], n_coarse)
    scores = np.array([score(k) for k in kappas])
    best = int(np.argmax(scores))
    step = kappas[1] - kappas[0]
    fine = np.linspace(kappas[best] - step, kappas[best] + step, n_fine)
    fine_scores = np.array([score(k) for k in fine])
    k_best = float(fine[int(np.argmax(fine_scores))])
    all_k = np.concatenate([kappas, fine])
    all_s = np.concatenate([scores, fine_scores])
    order = np.argsort(all_k)
    all_k, all_s = all_k[order], all_s[order]
    med = float(np.median(all_s))
    top = float(all_s.max())
    prominence = (top - med) / top if top else 0.0

    matches = []
    if direction == "t2":
        for item in preset_kappas(spin):
            for sign in (1, -1):
                ref = sign * item["kappa"]
                matches.append({"label": item["label"], "kappa": ref, "f1_sign": sign,
                                "difference": abs(k_best - ref),
                                "relative": abs(k_best - ref) / abs(ref)})
        matches.sort(key=lambda m: m["difference"])
    else:
        # The J-resolved / PASS / MAT tilt is +-1 by construction; what the
        # data have to settle is the sign. Compared directly, because a
        # tilt also turns F1 linewidth into F2 linewidth, so with short t1
        # the unsheared projection can score as high as the right tilt.
        for sign in (1, -1):
            matches.append({"label": "45-degree tilt (J-resolved / PASS / MAT)",
                            "kappa": float(sign), "f1_sign": sign, "score": score(float(sign)),
                            "difference": abs(k_best - sign), "relative": abs(k_best - sign)})
        matches.sort(key=lambda m: -m["score"])

    text = [f"The sharpest {'F1' if direction == 't2' else 'F2'} projection is at "
            f"kappa_{direction} = {k_best:+.4f} (scan {kappa_range[0]:g} to {kappa_range[1]:g}, "
            f"step {step:.3g}, refined to {(fine[1] - fine[0]):.2g})."]
    if prominence < 0.2:
        text.append("The scan is flat, so the data do not pin the ratio down (little "
                    "anisotropy, or noise); rely on the experiment's own value.")
    if abs(k_best) < 2 * step:
        text.append("The best ratio is close to zero: the data look already sheared, as a "
                    "split-t1 acquisition is – choose split-t1 (make_shear_plan(..., "
                    "split_t1=True)) so that F1 is only scaled and referenced, not sheared "
                    "again. (The F1 direction cannot be read from a ridge that is already "
                    "flat; if the sites come out in the wrong F1 order, use f1_sign = -1.)")
    if direction == "t1":
        m = matches[0]
        other = matches[1]
        ratio = m["score"] / other["score"] if other["score"] else float("inf")
        text.append(f"Of the two tilts, kappa_t1 = {m['kappa']:+.0f} gives the sharper F2 "
                    f"projection ({ratio:.2f}x the other), so "
                    + ("F1 is displayed with the usual sense (f1_sign = +1)." if m["f1_sign"] > 0
                       else "F1 is displayed reversed: use f1_sign = -1.")
                    + (" The difference is small; check the result by eye." if ratio < 1.2 else ""))
    elif matches:
        m = matches[0]
        close = m["relative"] < 0.05
        text.append(f"Closest standard value: {m['label']}, kappa = {m['kappa']:+.4f} "
                    f"({_frac(abs(m['kappa']))}), {m['relative']:.1%} away"
                    + ("." if close else " – not a close match; check the experiment and spin."))
        if close:
            text.append("F1 is displayed with the same sense as the coherence frequency "
                        "(f1_sign = +1)." if m["f1_sign"] > 0 else
                        "F1 is displayed reversed relative to the coherence frequency, so "
                        "use f1_sign = -1 (the sign of every t1 coefficient flips).")
    return {"kappa": k_best, "direction": direction, "kappas": all_k, "scores": all_s,
            "matches": matches, "prominence": prominence, "spin": spin,
            "summary": " ".join(text)}


# ---------------------------------------------------------------------------
# Sidebands and isotropic decoding
# ---------------------------------------------------------------------------

def sideband_positions(plan: ShearPlan, mas_hz: float, *, orders_f1=range(-3, 4),
                       orders_f2=range(-3, 4), sw: tuple | None = None) -> list[dict]:
    """Where the spinning sidebands of a peak land after the shear and
    scaling, relative to its centreband, in Hz of the result's axes.

    A sideband of order ``k`` in t1 and ``j`` in t2 sits at
    ``(nu1 + k nr, nu2 + j nr)`` before the shear, so afterwards it is
    offset by ``dF1 = (k - kappa_t2 j) nr / s1`` and
    ``dF2 = (j - kappa_t1 (k - kappa_t2 j)) nr / s2`` -- at non-integer
    multiples of the spinning rate whenever kappa is not an integer.

    ``sw`` = (SW1, SW2) of the result (Hz): each offset is also given
    folded into +-SW/2 (``df1_folded``), which is where it appears when
    t1 was rotor-synchronised (F1 sidebands of the t1 period then fold
    onto the centreband, but those carried in from F2 do not).
    """
    out = []
    for k in orders_f1:
        for j in orders_f2:
            if k == 0 and j == 0:
                continue
            d1 = (k - plan.kappa_t2 * j) * mas_hz
            d2 = (j - plan.kappa_t1 * d1 / mas_hz) * mas_hz if mas_hz else 0.0
            d1 /= plan.scale_f1
            d2 /= plan.scale_f2
            item = {"k": k, "j": j, "df1": d1, "df2": d2}
            if sw is not None:
                w1, w2 = sw
                item["df1_folded"] = (d1 + w1 / 2) % w1 - w1 / 2 if w1 else d1
                item["df2_folded"] = (d2 + w2 / 2) % w2 - w2 / 2 if w2 else d2
            out.append(item)
    return out


def isotropic_from_peak(plan: ShearPlan, delta_f1: float, delta_f2_cog: float, *,
                        spin: float | None = None, larmor_mhz: float | None = None) -> dict:
    """The isotropic chemical shift and quadrupolar product of a site,
    from its F1 position and its F2 centre of gravity in the sheared,
    scaled spectrum (ppm).

    Solves delta_F1 = A delta_CS + B delta_QIS and
    delta_F2(cog) = delta_CS + delta_QIS for delta_CS and delta_QIS
    (the CT's isotropic quadrupolar shift; the anisotropy averages to
    zero in the centre of gravity). With ``spin`` and ``larmor_mhz``
    also gives P_Q = C_Q sqrt(1 + eta^2/3) (MHz) from
    delta_QIS = -(3/40) (P_Q/nu0)^2 [I(I+1) - 3/4] / [I^2 (2I-1)^2] 1e6.
    """
    a = plan.f1_cs_factor
    b = plan.f1_qis_factor
    if b is None:
        raise ValueError("This plan has no isotropic quadrupolar shift in it (not MQMAS/STMAS/DAS).")
    c2 = plan.f2_cs_factor
    q2 = plan.qis_after[1] / (plan.scale_f2 * plan.qis[1])
    det = a * q2 - b * c2
    if abs(det) < 1e-12:
        raise ValueError("F1 and F2 report the same combination; delta_CS and delta_QIS "
                         "cannot be separated with this plan.")
    d_cs = (delta_f1 * q2 - b * delta_f2_cog) / det
    d_qis = (a * delta_f2_cog - c2 * delta_f1) / det
    out = {"delta_cs": d_cs, "delta_qis": d_qis}
    spin = spin if spin is not None else plan.spin
    if spin and larmor_mhz:
        i = float(spin)
        factor = (3.0 / 40.0) * (i * (i + 1) - 0.75) / (i * i * (2 * i - 1) ** 2)
        pq2 = -d_qis * 1e-6 / factor * larmor_mhz ** 2
        out["PQ_MHz"] = math.sqrt(pq2) if pq2 > 0 else float("nan")
        if pq2 < 0:
            out["note"] = ("delta_QIS came out positive, which a second-order quadrupolar "
                           "shift cannot be: check the F2 centre of gravity and the plan.")
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_shear_scan(scan: dict, ax=None, *, show_matches: int = 4, fontsize: float = 10):
    """The sharpness scan from :func:`estimate_shear`, with the best
    value and the nearest standard ratios marked."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.0))
    s = np.asarray(scan["scores"])
    ax.plot(scan["kappas"], s / s.max(), color="#1f4e79", lw=1.2)
    ax.axvline(scan["kappa"], color="#c0392b", lw=1.0,
               label=fr"best $\kappa$ = {scan['kappa']:+.3f}")
    for m in scan["matches"][:show_matches]:
        ax.axvline(m["kappa"], color="0.5", lw=0.8, ls="--")
        ax.annotate(f"{m['kappa']:+.3f}", (m["kappa"], 0.02), xycoords=("data", "axes fraction"),
                    rotation=90, fontsize=fontsize * 0.7, color="0.35", ha="right", va="bottom")
    sub = "t_2" if scan["direction"] == "t2" else "t_1"
    ax.set_xlabel(_italic_label(fr"$\kappa^{{({sub})}}$"), fontsize=fontsize)
    ax.set_ylabel(_italic_label("Projection sharpness (norm.)"), fontsize=fontsize)
    ax.tick_params(labelsize=fontsize * 0.9)
    ax.legend(frameon=False, fontsize=fontsize * 0.8)
    return ax


def plot_sideband_markers(ax, positions: list[dict], centre_f1: float, centre_f2: float, *,
                          sf1: float = 1.0, sf2: float = 1.0, folded: bool = False,
                          f1_window: tuple | None = None, f2_window: tuple | None = None,
                          color: str = "#c0392b", fontsize: float = 7):
    """Mark :func:`sideband_positions` around a centreband at
    (``centre_f1``, ``centre_f2``) ppm on a 2D plot whose axes are in
    ppm (``sf1``/``sf2`` convert the Hz offsets).

    ``f1_window`` = the F1 axis's (lowest, highest) ppm: positions are
    then aliased into it, as they appear in the spectrum (F1 is
    sampled, so a sideband outside the window folds back in).
    ``f2_window`` drops markers outside F2's range. ``folded`` uses the
    offsets folded about the peak instead."""
    key1 = "df1_folded" if folded else "df1"
    key2 = "df2_folded" if folded else "df2"
    for p in positions:
        x = centre_f2 + p.get(key2, p["df2"]) / sf2
        y = centre_f1 + p.get(key1, p["df1"]) / sf1
        if f1_window is not None:
            lo, hi = min(f1_window), max(f1_window)
            span = hi - lo
            if span > 0:
                y = (y - lo) % span + lo
        if f2_window is not None and not (min(f2_window) <= x <= max(f2_window)):
            continue
        ax.plot(x, y, marker="+", color=color, ms=6, mew=1.0, ls="none")
        ax.annotate(f"{p['k']},{p['j']}", (x, y), textcoords="offset points", xytext=(3, 3),
                    fontsize=fontsize, color=color)
    return ax


def _italic_label(text):
    """Quantity symbols italic (plotting.labels.italicise_symbols)."""
    from opennmr.plotting.labels import italicise_symbols
    return italicise_symbols(text)
