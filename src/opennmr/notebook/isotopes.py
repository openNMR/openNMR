"""
Several nuclei on one spectrum: where another nucleus's lines fall on
the observed nucleus's ppm scale, and how an isotope of the same element
carries the same chemistry.

Two situations, one mechanism
-----------------------------
* **Isotope partners** -- 47Ti and 49Ti (Larmor frequencies 266 ppm
  apart at any field), 63Cu/65Cu, 85Rb/87Rb, ... Both isotopes sit in the
  same site, so they share the chemistry: isotropic shift, CSA (ppm),
  the EFG (eta_Q, and C_Q in proportion to the quadrupole moment Q),
  J (in proportion to gamma) and, at natural abundance, the number of
  nuclei. A partner is *linked* to its parent site
  (:meth:`~opennmr.notebook.combined_hamiltonian.Site.isotope_partner`);
  any parameter can be "unlinked" to give the partner its own value.
* **Foreign signals** -- a 63Cu line from the coil on a 23Na spectrum.
  A site of another nucleus with no link: it has its own parameters and
  is only *placed* on the observed scale.

Placing a nucleus on another's scale
------------------------------------
A line at ``delta`` ppm on nucleus Y's own scale has frequency
``Xi_Y (1 + delta 1e-6)`` (Xi in units of the 1H TMS frequency). On the
observed nucleus X's scale it is therefore at

    delta_X = (R (1 + delta 1e-6) - 1) 1e6,     R = Xi_Y / Xi_X,

and the inverse maps the observed axis onto Y's own scale -- which is
how a Y site is simulated on an X spectrum: on its own scale, at its
own Larmor frequency (R times the observed one), then laid on the X
axis point for point.

Xi are the IUPAC 2001 recommended frequency ratios
(:data:`opennmr.notebook.utils._XI_PERCENT`, a column of ``_GAMMA_DF``).
They include the reference compounds' shielding, which the ratio of
gyromagnetic ratios does not (47Ti/49Ti: 30 ppm different; 63Cu/23Na:
~2000 ppm). Where a nucleus has no tabulated Xi, the gamma ratio is used
and the result says so (``source == "gamma"``): the placement is then
only as good as that.

This assumes the observed spectrum is referenced to its nucleus's IUPAC
primary reference (directly, or indirectly through Xi). If it was
referenced to a secondary compound, the other nucleus is off by that
compound's shift -- fit it, or correct the referencing first.
"""

import re

import numpy as np

from .utils import _GAMMA_DF, find_isotope_info

__all__ = ["element_of", "xi_percent", "frequency_ratio", "to_observed_ppm",
           "from_observed_ppm", "isotope_partners", "quadrupole_moment", "signal_per_nucleus",
           "partner_factors", "describe_placement"]


def element_of(nucleus):
    """'49Ti' -> 'Ti'."""
    m = re.fullmatch(r"\s*(\d+)\s*([A-Za-z]+)\s*", str(nucleus))
    if not m:
        raise KeyError(f"{nucleus!r} is not a nuclide such as '47Ti'.")
    return m.group(2)


def _row(nucleus):
    rows = _GAMMA_DF[_GAMMA_DF["Nuclide"] == str(nucleus).strip()]
    if rows.empty:
        raise KeyError(f"No entry for nuclide {nucleus!r} in the nuclear data table.")
    return rows.iloc[0]


def xi_percent(nucleus):
    """IUPAC frequency ratio Xi (%) of ``nucleus``, or None if not tabulated."""
    value = float(_row(nucleus)["Frequency Ratio Xi (%)"])
    return None if not np.isfinite(value) else value


def frequency_ratio(nucleus, observe):
    """``(R, source)``: the frequency of ``nucleus``'s 0 ppm over that of
    ``observe``'s 0 ppm, from the IUPAC Xi ratios (``source='Xi'``), or,
    when either is untabulated, from the gyromagnetic ratios
    (``source='gamma'``, which ignores the reference compounds' shielding)."""
    if str(nucleus) == str(observe):
        return 1.0, "same"
    a, b = xi_percent(nucleus), xi_percent(observe)
    if a is not None and b is not None:
        return a / b, "Xi"
    ga = abs(float(find_isotope_info(nucleus)["gamma"]))
    gb = abs(float(find_isotope_info(observe)["gamma"]))
    return ga / gb, "gamma"


def to_observed_ppm(delta, nucleus, observe):
    """Shift(s) ``delta`` on ``nucleus``'s own scale, as ppm on ``observe``'s."""
    R, _src = frequency_ratio(nucleus, observe)
    return (R * (1.0 + np.asarray(delta, dtype=float) * 1e-6) - 1.0) * 1e6


def from_observed_ppm(x_ppm, nucleus, observe):
    """The observed axis ``x_ppm`` (``observe``'s ppm) on ``nucleus``'s own scale."""
    R, _src = frequency_ratio(nucleus, observe)
    return ((1.0 + np.asarray(x_ppm, dtype=float) * 1e-6) / R - 1.0) * 1e6


def isotope_partners(nucleus):
    """The other NMR-active, naturally occurring isotopes of ``nucleus``'s
    element, most abundant first ('47Ti' -> ['49Ti'])."""
    element = element_of(nucleus)
    out = []
    for nuc, spin, ab in zip(_GAMMA_DF["Nuclide"], _GAMMA_DF["Spin"],
                             _GAMMA_DF["Natural Abundance (%)"]):
        nuc = str(nuc)
        if nuc == str(nucleus) or element_of(nuc) != element:
            continue
        try:
            if float(spin) > 0 and float(ab) > 0:
                out.append((float(ab), nuc))
        except (TypeError, ValueError):
            continue
    return [nuc for _ab, nuc in sorted(out, reverse=True)]


def quadrupole_moment(nucleus):
    """Q in fm² (signed), or None for a spin-1/2."""
    text = str(_row(nucleus)["Quadrupole Moment (fm^2)"]).strip()
    try:
        return float(text)
    except ValueError:
        return None


def signal_per_nucleus(nucleus):
    """Relative NMR signal of one nucleus of this isotope at a fixed field:
    |gamma|^3 I(I+1) -- the Curie magnetisation (gamma^2 I(I+1)) times the
    voltage it induces (gamma B0) -- summed over every transition, as a
    site's ``amount`` counts it."""
    info = find_isotope_info(nucleus)
    gamma = abs(float(info["gamma"])) / 1e6
    spin = float(info["spin"])
    return gamma ** 3 * spin * (spin + 1)


def partner_factors(parent, partner):
    """How a linked partner's parameters follow its parent's:

    ``C_Q``    x Q(partner)/Q(parent)  (same EFG; |ratio|, as |C_Q| is fitted)
    ``J``      x gamma(partner)/gamma(parent)  (same reduced coupling, signed)
    ``amount`` x abundance x |gamma|^3 I(I+1), partner over parent (same
               site, natural abundance, uniform excitation)
    ``ppm``    1 (shift, CSA and broadening are in ppm, the same for both)

    Returns a dict of these, plus ``frequency_ratio`` and its ``source``."""
    p0, p1 = find_isotope_info(parent), find_isotope_info(partner)
    q0, q1 = quadrupole_moment(parent), quadrupole_moment(partner)
    cq = abs(q1 / q0) if q0 and q1 else None
    amount = (p1["abundance"] * signal_per_nucleus(partner)) / (
        p0["abundance"] * signal_per_nucleus(parent)) if p0["abundance"] > 0 else None
    R, src = frequency_ratio(partner, parent)
    return {"C_Q": cq, "J": float(p1["gamma"]) / float(p0["gamma"]), "amount": amount,
            "ppm": 1.0, "frequency_ratio": R, "source": src}


def describe_placement(nucleus, observe):
    """One line for a log: where ``nucleus``'s 0 ppm falls on ``observe``'s scale."""
    R, src = frequency_ratio(nucleus, observe)
    zero = (R - 1.0) * 1e6
    how = {"Xi": "IUPAC frequency ratios",
           "gamma": "gyromagnetic ratios (no IUPAC frequency ratio tabulated – reference "
                    "shielding ignored, so the position is approximate)",
           "same": "same nucleus"}[src]
    return (f"{nucleus} on the {observe} scale: its 0 ppm is at {zero:+.2f} ppm "
            f"(ratio {R:.9f}, from the {how}).")
