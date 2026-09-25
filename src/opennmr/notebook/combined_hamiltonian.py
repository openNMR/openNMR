"""
Combined Hamiltonian: one site, any combination of interactions.

A :class:`Site` is an observed nucleus with an isotropic shift, to which
interactions are *added* -- no function per combination::

    al = (Site("27Al", delta_iso=12.0, label="Al(1)")
          .add_quadrupolar(C_Q=4.2, eta_Q=0.35)             # MHz
          .add_csa(aniso=40.0, asym=0.2, euler=(0, 90, 0))  # ppm, deg
          .add_jcoupling("31P", n=2, J_Hz=15.0))
    spec = simulate_combined([al], ppm_axis, larmor_Hz=130.3e6, mas_Hz=15e3,
                             transitions="ct", order="second")

The same site can be written as a dict (``Site.to_dict`` /
``Site.from_dict``) -- what the GUI's Simulate > Combined Hamiltonian
section stores, and what the simulation's log records -- and
:func:`combined_to_python` writes the Python that reproduces a GUI
simulation in a notebook.

Physics (high field, every interaction small next to the Zeeman term)
---------------------------------------------------------------------
Each transition's frequency is a sum of contributions:

* **Zeeman + isotropic shift**: nu0 (1 + delta_iso 1e-6).
* **CSA** (first order): nu0 1e-6 b.D.b, D the traceless shielding
  anisotropy in the site frame and b the B0 direction. Identical for
  every transition of the site. Its second-order and CSA-quadrupolar
  cross terms are of order (1e-6 nu0 D)^2/nu0 and are neglected
  (below 1e-6 of the CSA itself).
* **Quadrupolar**: first order, second order, or exact diagonalisation
  of H_Z + H_Q -- see simulations' QUADRUPOLAR NMR section. With
  ``order='exact'`` the exactness refers to the quadrupolar part; CSA
  and J are still added at first order, which is exact to the
  precision above.
* **J (heteronuclear, weak coupling)**: an isotropic splitting,
  independent of orientation, so every line becomes the first-order
  multiplet of simulations.combine_jcouplings (natural abundance and
  partner spin included). Homonuclear, strongly coupled systems need
  notebook.spin_system instead.

**Frame and Euler angles.** The site frame is the EFG principal-axis
frame when there is a quadrupolar interaction, otherwise the CSA
principal-axis frame. The CSA tensor sits in the EFG frame as
D = R diag(d_xx, d_yy, d_zz) R^T with R = Rz(alpha) Ry(beta) Rz(gamma)
(active, ZYZ, degrees): (beta, alpha) are the polar angles of the CSA's
unique axis (d_zz) in the EFG frame and gamma turns the CSA about that
axis. All zero: the two tensors are coincident.

Under MAS the whole trajectory of each crystallite goes through the
same gamma-COMPUTE machinery as the quadrupolar simulator, so the
centreband lineshape and every sideband of every combination come
out together. Powder averaging uses one octant of the sphere when
that is exact (no CSA, or a CSA whose axes coincide with the EFG's --
D2h symmetry); a non-coincident CSA breaks that symmetry and the full
sphere is used automatically.
"""

import copy
import json

import numpy as np

from .utils import find_isotope_info
from .simulations import (
    csa_definition, combine_jcouplings, zcw_angles, get_powder_grid,
    _apply_broadening, _bin_sticks, _select_transitions, quadrupolar_transition_table,
    _perturbative_offsets, _exact_offsets, _rotor_frame, _mas_prepare, _mas_sticks,
    QUADRUPOLAR_ORDERS,
)

INTERACTION_TYPES = ("csa", "quadrupolar", "jcoupling")
_ALIASES = {"csa": "csa", "shielding": "csa", "chemical_shift_anisotropy": "csa",
            "quadrupolar": "quadrupolar", "quad": "quadrupolar", "efg": "quadrupolar",
            "jcoupling": "jcoupling", "j": "jcoupling", "j_coupling": "jcoupling",
            "scalar": "jcoupling"}


# ============================================================
# Site
# ============================================================

def _rotation_zyz(alpha_deg, beta_deg, gamma_deg):
    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    a, b, g = np.deg2rad([alpha_deg, beta_deg, gamma_deg])
    return rz(a) @ ry(b) @ rz(g)


class Site:
    """
    One observed site: a nucleus, an isotropic shift, and a list of
    interactions added with :meth:`add_csa`, :meth:`add_quadrupolar`,
    :meth:`add_jcoupling` or the generic :meth:`add` (each returns the
    site, so calls chain).

    Parameters
    ----------
    nucleus : str
        The observed nuclide, e.g. '27Al' (sets the spin I).
    delta_iso : float
        Isotropic chemical shift, ppm.
    label : str, optional
    amount : float
        Relative number of nuclei (the site's integrated intensity, all
        transitions together, before any transition selection).
    fwhm : float
        Broadening FWHM, ppm (0 = none).
    lineshape : 'gaussian' | 'lorentzian' | 'pseudo_voigt'; eta its mixing.
    """

    def __init__(self, nucleus, delta_iso=0.0, label=None, amount=1.0, fwhm=1.0,
                 lineshape="gaussian", eta=0.0, spin=None):
        self.nucleus = str(nucleus)
        self.spin = float(spin) if spin is not None else float(find_isotope_info(nucleus)["spin"])
        self.delta_iso = float(delta_iso)
        self.label = label or f"{self.nucleus} site"
        self.amount = float(amount)
        self.fwhm = float(fwhm)
        self.lineshape = lineshape
        self.eta = float(eta)
        self.interactions = []

    # ---- adding / removing interactions --------------------------------

    def add(self, kind, **params):
        """Add an interaction by name: 'csa', 'quadrupolar' or 'jcoupling'
        (aliases 'quad', 'j', ...), with that method's parameters."""
        key = _ALIASES.get(str(kind).strip().lower().replace("-", "_").replace(" ", "_"))
        if key is None:
            raise ValueError(f"Unknown interaction {kind!r}; use one of {INTERACTION_TYPES}.")
        return {"csa": self.add_csa, "quadrupolar": self.add_quadrupolar,
                "jcoupling": self.add_jcoupling}[key](**params)

    def add_csa(self, aniso, asym=0.0, convention="Haeberlen", euler=(0.0, 0.0, 0.0)):
        """Chemical-shift anisotropy. ``aniso``/``asym``: Delta/eta
        (Haeberlen) or span/skew (Herzfeld-Berger), ppm. ``euler``:
        (alpha, beta, gamma) degrees of the CSA frame in the EFG frame
        (ignored without a quadrupolar interaction). One per site."""
        if self.get("csa"):
            raise ValueError(f"{self.label} already has a CSA; a site has one shielding tensor "
                             "(edit it, or remove it first).")
        if convention not in ("Haeberlen", "Herzfeld-Berger"):
            raise ValueError("convention must be 'Haeberlen' or 'Herzfeld-Berger'.")
        euler = tuple(float(a) for a in euler)
        if len(euler) != 3:
            raise ValueError("euler must be (alpha, beta, gamma) in degrees.")
        self.interactions.append({"type": "csa", "aniso": float(aniso), "asym": float(asym),
                                  "convention": convention, "euler": euler})
        return self

    def add_quadrupolar(self, C_Q, eta_Q=0.0):
        """Quadrupolar coupling: ``C_Q`` in MHz, ``eta_Q`` 0..1. One per site."""
        if self.spin < 1:
            raise ValueError(f"{self.nucleus} (I = {self.spin:g}) has no quadrupole moment.")
        if self.get("quadrupolar"):
            raise ValueError(f"{self.label} already has a quadrupolar interaction; a site has "
                             "one EFG (edit it, or remove it first).")
        if not 0.0 <= float(eta_Q) <= 1.0:
            raise ValueError("eta_Q must lie between 0 and 1.")
        self.interactions.append({"type": "quadrupolar", "C_Q": float(C_Q),
                                  "eta_Q": float(eta_Q)})
        return self

    def add_jcoupling(self, nuclide, n=1, J_Hz=0.0, abundance=None):
        """A group of ``n`` equivalent partner nuclei (``nuclide``, e.g.
        '31P') with scalar coupling ``J_Hz``. Add as many groups as
        needed; natural abundance is taken from the table unless given."""
        find_isotope_info(nuclide)  # validates the name
        group = {"type": "jcoupling", "nuclide": str(nuclide), "n": int(n), "J_Hz": float(J_Hz)}
        if abundance is not None:
            group["abundance"] = float(abundance)
        self.interactions.append(group)
        return self

    def remove(self, which):
        """Remove interaction ``which``: its index, or a type name (removes
        every interaction of that type)."""
        if isinstance(which, (int, np.integer)):
            del self.interactions[int(which)]
        else:
            key = _ALIASES.get(str(which).lower(), which)
            self.interactions = [i for i in self.interactions if i["type"] != key]
        return self

    def get(self, kind):
        """The interactions of one type (a list)."""
        key = _ALIASES.get(str(kind).lower(), kind)
        return [i for i in self.interactions if i["type"] == key]

    def copy(self, label=None):
        site = copy.deepcopy(self)
        if label is not None:
            site.label = label
        return site

    # ---- description -----------------------------------------------------

    def to_dict(self):
        return {"nucleus": self.nucleus, "spin": self.spin, "delta_iso": self.delta_iso,
                "label": self.label, "amount": self.amount, "fwhm": self.fwhm,
                "lineshape": self.lineshape, "eta": self.eta,
                "interactions": copy.deepcopy(self.interactions)}

    @classmethod
    def from_dict(cls, d):
        site = cls(d["nucleus"], delta_iso=d.get("delta_iso", 0.0), label=d.get("label"),
                   amount=d.get("amount", 1.0), fwhm=d.get("fwhm", 1.0),
                   lineshape=d.get("lineshape", "gaussian"), eta=d.get("eta", 0.0),
                   spin=d.get("spin"))
        for inter in d.get("interactions", []):
            params = {k: v for k, v in inter.items() if k != "type"}
            site.add(inter["type"], **params)
        return site

    def describe(self):
        """One line: the site and every interaction, with units."""
        bits = [f"{self.label} ({self.nucleus}, I = {self.spin:g}): δiso {self.delta_iso:g} ppm"]
        for i in self.interactions:
            if i["type"] == "csa":
                names = ("Δ", "η") if i["convention"] == "Haeberlen" else ("span", "skew")
                text = f"CSA {names[0]} {i['aniso']:g} ppm, {names[1]} {i['asym']:g}"
                if self.get("quadrupolar") and any(i["euler"]):
                    text += " at Euler ({:g}, {:g}, {:g})°".format(*i["euler"])
                bits.append(text)
            elif i["type"] == "quadrupolar":
                bits.append(f"quadrupolar C_Q {i['C_Q']:g} MHz, η_Q {i['eta_Q']:g}")
            else:
                bits.append(f"J to {i['n']}×{i['nuclide']} {i['J_Hz']:g} Hz")
        bits.append(f"amount {self.amount:g}, FWHM {self.fwhm:g} ppm {self.lineshape}")
        return "; ".join(bits)

    def __repr__(self):
        return f"Site({self.describe()})"


def as_sites(sites):
    """Accept a Site, a dict, or a list of either; return a list of Sites."""
    if isinstance(sites, (Site, dict)):
        sites = [sites]
    return [s if isinstance(s, Site) else Site.from_dict(s) for s in sites]


# ============================================================
# Frequencies
# ============================================================

def _csa_frame_tensor(site):
    """Traceless CSA tensor (ppm) in the site frame, or None."""
    csa = site.get("csa")
    if not csa:
        return None
    c = csa[0]
    principal = np.diag(csa_definition(0.0, c["aniso"], c["asym"], convention=c["convention"]))
    D = np.diag(principal - principal.mean())
    if site.get("quadrupolar"):
        R = _rotation_zyz(*c["euler"])
        D = R @ D @ R.T
    return D


def _octant_is_exact(site):
    D = _csa_frame_tensor(site)
    if D is None:
        return True
    off = D - np.diag(np.diag(D))
    return bool(np.max(np.abs(off)) <= 1e-9 * max(np.max(np.abs(D)), 1e-30))


def _site_frequencies(site, larmor_Hz, b, rows, order):
    """Offsets from nu0 (Hz, excluding delta_iso and J) and intensities
    (normalised so all 2I transitions of the site sum to 1) of the
    selected transitions, for B0 directions ``b`` (..., 3)."""
    m = np.array([r["m"] for r in rows])
    total_weight = sum(r["weight"] for r in quadrupolar_transition_table(site.spin))
    quad = site.get("quadrupolar")
    shape = b.shape[:-1] + (len(m),)
    if quad:
        C_Q = quad[0]["C_Q"] * 1e6
        if order == "exact":
            nu, inten = _exact_offsets(site.spin, C_Q, quad[0]["eta_Q"], larmor_Hz, b, m)
        else:
            nu = _perturbative_offsets(site.spin, C_Q, quad[0]["eta_Q"], larmor_Hz, b, m, order)
            inten = np.broadcast_to(np.array([r["weight"] for r in rows]), shape)
    else:
        nu = np.zeros(shape)
        inten = np.broadcast_to(np.array([r["weight"] for r in rows]), shape)
    D = _csa_frame_tensor(site)
    if D is not None:
        nu = nu + (larmor_Hz * 1e-6 * np.einsum("...i,ij,...j->...", b, D, b))[..., None]
    return nu, inten / total_weight


def site_transition_frequencies(site, larmor_Hz, theta, phi, order="second", transitions="all"):
    """Per-orientation transition frequencies (offset from nu0, Hz,
    including delta_iso but not J) and intensities of one site, for B0
    at polar angles (theta, phi) in the site frame. For inspecting a
    crystallite or checking the engine."""
    site = as_sites(site)[0]
    rows = _rows(site, transitions)
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    b = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    nu, inten = _site_frequencies(site, larmor_Hz, b, rows, order)
    return {"offset_Hz": nu + site.delta_iso * 1e-6 * larmor_Hz, "intensity": inten,
            "label": [r["label"] for r in rows], "group": [r["group"] for r in rows]}


def _rows(site, transitions):
    rows = _select_transitions(site.spin, transitions)
    if not rows:
        raise ValueError(f"{site.label}: I = {site.spin:g} has no transitions matching "
                         f"{transitions!r} (a spin-1/2 has only the one, 'ct' or 'all').")
    return rows


def _j_lines(site, larmor_Hz):
    groups = [{k: v for k, v in g.items() if k != "type"} for g in site.get("jcoupling")]
    if not groups:
        return np.array([0.0]), np.array([1.0])
    f, w = combine_jcouplings(groups)
    return np.asarray(f) / larmor_Hz * 1e6, np.asarray(w)


def _powder(site, n_zcw, use_symmetry, static):
    """(theta, phi, weights, octant). ``n_zcw`` sets the orientation
    *density*: n_zcw points in one octant when that is exact; otherwise
    the full sphere -- with 8x as many points for a static pattern, so
    its sharp edges are sampled as finely (measured: a tilted CSA +
    quadrupolar static pattern was 15% off at 4181 full-sphere points,
    3% at 8x that). A MAS centreband/sideband pattern converges with far
    fewer, so it keeps n_zcw."""
    if use_symmetry and _octant_is_exact(site):
        theta, phi, _ = zcw_angles(8 * int(n_zcw))
        keep = (theta <= np.pi / 2) & (phi < np.pi / 2)
        theta, phi = theta[keep], phi[keep]
        return theta, phi, np.full(theta.size, 1.0 / theta.size), True
    theta, phi, w = zcw_angles(int(n_zcw) * (8 if static and use_symmetry else 1))
    return theta, phi, w, False


# ============================================================
# Simulation
# ============================================================

def _site_sticks(site, ppm_axis, larmor_Hz, mas_Hz, order, transitions, beta_r, n_zcw,
                 n_gamma=None, n_time=None, max_n_time=65536, use_symmetry=True):
    """One site's unbroadened spectrum per transition group, for amount 1:
    ``({group: sticks binned on ppm_axis}, powder info)``. Everything
    except the broadening and the amount, which are applied afterwards
    (so a fit can change them without re-simulating)."""
    rows = _rows(site, transitions)
    groups = [r["group"] for r in rows]
    comp = {g: np.zeros(len(ppm_axis)) for g in groups}
    j_ppm, j_w = _j_lines(site, larmor_Hz)
    theta, phi, wts, octant = _powder(site, n_zcw, use_symmetry, static=not mas_Hz)
    if not mas_Hz:
        b = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                      np.cos(theta)], -1)
        nu, inten = _site_frequencies(site, larmor_Hz, b, rows, order)
        position = site.delta_iso + nu / larmor_Hz * 1e6
        weight = wts[:, None] * inten
        for dj, wj in zip(j_ppm, j_w):
            for col, g in enumerate(groups):
                comp[g] += _bin_sticks(ppm_axis, position[:, col] + dj, wj * weight[:, col])
        return comp, {"orientations": len(wts), "octant": octant}
    g_n = int(n_gamma) if n_gamma else (32 if order == "exact" else 16)
    r, e1, e2 = _rotor_frame(theta, phi)
    gam = 2.0 * np.pi * np.arange(g_n) / g_n
    b = (np.cos(beta_r) * r[:, None, :]
         + np.sin(beta_r) * (np.cos(gam)[None, :, None] * e1[:, None, :]
                             + np.sin(gam)[None, :, None] * e2[:, None, :]))
    nu, inten = _site_frequencies(site, larmor_Hz, b, rows, order)
    inten = inten.mean(axis=1)
    prep = _mas_prepare(nu, mas_Hz, g_n, n_time, max_n_time)
    for _a, _b, centre, sb_Hz, weight in _mas_sticks(prep, inten, wts):
        position = site.delta_iso + (centre[:, None, :] + sb_Hz[None, :, None]) / larmor_Hz * 1e6
        for dj, wj in zip(j_ppm, j_w):
            for col, g in enumerate(groups):
                comp[g] += _bin_sticks(ppm_axis, position[:, :, col] + dj, wj * weight[:, :, col])
    return comp, {"orientations": len(wts), "octant": octant, "n_gamma": g_n,
                  "n_time": prep["n_time"]}


def _broaden_site(site, comp, ppm_axis):
    return {g: site.amount * (_apply_broadening(c, ppm_axis, fwhm=site.fwhm,
                                                lineshape=site.lineshape, eta=site.eta)
                              if site.fwhm > 0 else c)
            for g, c in comp.items()}


def simulate_combined(sites, ppm_axis, larmor_Hz=None, B0_T=None, mas_Hz=0.0, order="second",
                      transitions="all", rotor_angle="magic", n_zcw=None, n_gamma=None,
                      n_time=None, max_n_time=65536, use_symmetry=True):
    """
    Powder spectrum of one or more sites, each with any combination of
    CSA, quadrupolar and J interactions, static (``mas_Hz`` = 0) or
    under MAS.

    Parameters
    ----------
    sites : Site, dict, or a list of them. All must observe the same
        nucleus (it is one spectrum).
    ppm_axis : uniformly spaced ppm axis (either direction).
    larmor_Hz or B0_T : the observe frequency |gamma B0|, or the field.
    mas_Hz : spinning rate, Hz; 0 for static.
    order : quadrupolar level of theory, 'first' | 'second' | 'exact'.
    transitions : 'all' | 'ct' | 'st' | group names -- applied to every
        site (a spin-1/2 site has only its one transition).
    rotor_angle : 'magic' or degrees.
    n_zcw : powder orientation density per site (default 75025 static --
        cheap, and fine enough that a 0.4 ppm-broadened pattern shows no
        powder ripple (0.5% vs 7.5% at 10946) -- 4181 MAS, where the
        pattern converges much faster): that many points in one octant
        when that is exact
        (see the module notes); otherwise the full sphere, with 8x as
        many for a static pattern so the density is the same.
    n_gamma, n_time, max_n_time : MAS sampling, as quadrupolar_mas_spectrum.

    Returns
    -------
    dict: ``x_ppm``, ``real_spec`` (total), ``normalised_real_spec``,
    ``sites`` ({label: that site's broadened spectrum}),
    ``components`` ({label: {transition group: spectrum}}),
    ``settings`` (everything needed to reproduce it), ``site_dicts``,
    and ``log`` (one line per site, then the settings).
    """
    sites = as_sites(sites)
    if not sites:
        raise ValueError("Give at least one site.")
    nuclei = {s.nucleus for s in sites}
    if len(nuclei) > 1:
        raise ValueError(f"All sites must observe the same nucleus (one spectrum); got {sorted(nuclei)}.")
    labels = [s.label for s in sites]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Site labels must be unique, got {labels}.")
    if larmor_Hz is None:
        if B0_T is None:
            raise ValueError("Give larmor_Hz or B0_T.")
        larmor_Hz = abs(float(find_isotope_info(sites[0].nucleus)["gamma"])) * float(B0_T)
    larmor_Hz = float(larmor_Hz)
    if larmor_Hz <= 0:
        raise ValueError("larmor_Hz must be > 0.")
    if order not in QUADRUPOLAR_ORDERS:
        raise ValueError(f"order must be one of {QUADRUPOLAR_ORDERS}.")
    mas_Hz = float(mas_Hz or 0.0)
    if mas_Hz < 0:
        raise ValueError("mas_Hz must be >= 0.")
    if n_zcw is None:
        n_zcw = 4181 if mas_Hz else 75025
    ppm_axis = np.asarray(ppm_axis, dtype=float)
    beta_r = (np.arccos(1.0 / np.sqrt(3.0)) if rotor_angle == "magic"
              else np.deg2rad(float(rotor_angle)))

    site_spectra, components, notes = {}, {}, []
    used = {}
    for site in sites:
        comp, powder = _site_sticks(site, ppm_axis, larmor_Hz, mas_Hz, order, transitions,
                                    beta_r, n_zcw, n_gamma, n_time, max_n_time, use_symmetry)
        used[site.label] = powder
        broadened = _broaden_site(site, comp, ppm_axis)
        components[site.label] = broadened
        site_spectra[site.label] = np.sum(list(broadened.values()), axis=0)
        if not powder["octant"] and use_symmetry:
            notes.append(f"{site.label}: CSA not coincident with the EFG, so the full sphere "
                         "was averaged (no octant symmetry)")

    total = np.sum(list(site_spectra.values()), axis=0)
    peak = float(np.max(total)) if total.size else 0.0
    settings = {"larmor_Hz": larmor_Hz, "mas_Hz": mas_Hz, "order": order,
                "transitions": transitions, "rotor_angle": rotor_angle, "n_zcw": int(n_zcw),
                "n_gamma": n_gamma, "use_symmetry": use_symmetry}
    regime = ("static" if not mas_Hz else
              f"MAS {mas_Hz / 1e3:g} kHz" + ("" if rotor_angle == "magic" else f" at {rotor_angle}°"))
    log_lines = [s.describe() for s in sites]
    log_lines.append(
        f"{regime}, nu0 = {larmor_Hz / 1e6:.7g} MHz, quadrupolar {order}-order"
        .replace("exact-order", "exact") + f", transitions {transitions}, "
        + ", ".join(f"{k}: {v['orientations']} orientations"
                    + (" (octant)" if v["octant"] else "") for k, v in used.items()) + ".")
    log_lines += notes
    return {"x_ppm": ppm_axis, "real_spec": total,
            "normalised_real_spec": total / peak if peak > 0 else total,
            "sites": site_spectra, "components": components, "settings": settings,
            "site_dicts": [s.to_dict() for s in sites], "log": "\n".join(log_lines),
            "powder": used}


def combined_axis_limits(sites, larmor_Hz, transitions="all", margin=0.15, n=2584, mas_Hz=0.0,
                         order="second", threshold=2e-3):
    """ppm limits framing every site: static -- its full second-order
    frequency range, widened by the J multiplet; MAS -- wherever a quick,
    coarse MAS simulation has more than ``threshold`` of its maximum
    (spinning sidebands of a CSA spun slower than its width lie *outside*
    the static range, so the static range alone would clip them). Either
    way padded by ``margin`` of the range and five broadening FWHMs."""
    sites = as_sites(sites)
    theta, phi, _ = zcw_angles(n)
    b = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    lows, highs, fwhm = [], [], 0.0
    for site in sites:
        nu, _ = _site_frequencies(site, larmor_Hz, b, _rows(site, transitions), "second")
        j_ppm, _w = _j_lines(site, larmor_Hz)
        ppm = site.delta_iso + nu / larmor_Hz * 1e6
        lows.append(ppm.min() + j_ppm.min())
        highs.append(ppm.max() + j_ppm.max())
        fwhm = max(fwhm, site.fwhm)
    low, high = min(lows), max(highs)
    if mas_Hz:
        spacing = float(mas_Hz) / larmor_Hz * 1e6
        wide = np.linspace(low - 20 * spacing - 10 * fwhm, high + 20 * spacing + 10 * fwhm, 16384)
        spec = simulate_combined(sites, wide, larmor_Hz, mas_Hz=mas_Hz, order=order,
                                 transitions=transitions, n_zcw=377)["real_spec"]
        keep = np.flatnonzero(spec > threshold * spec.max())
        if keep.size:
            low, high = float(wide[keep[0]]), float(wide[keep[-1]])
    pad = max(margin * (high - low), 5.0 * fwhm, 1.0)
    return low - pad, high + pad


def combined_to_python(sites, ppm_axis=None, **settings):
    """The notebook code that reproduces a combined simulation -- the GUI
    offers it as 'Copy as Python', so a result can be rerun, changed or
    fitted from a notebook."""
    sites = as_sites(sites)
    lines = ["import numpy as np",
             "from opennmr.notebook.combined_hamiltonian import Site, simulate_combined", "",
             "sites = ["]
    for s in sites:
        head = (f"    Site({s.nucleus!r}, delta_iso={s.delta_iso!r}, label={s.label!r}, "
                f"amount={s.amount!r}, fwhm={s.fwhm!r}, lineshape={s.lineshape!r}"
                + (f", eta={s.eta!r}" if s.lineshape == "pseudo_voigt" else "") + ")")
        calls = []
        for i in s.interactions:
            if i["type"] == "csa":
                calls.append(f".add_csa(aniso={i['aniso']!r}, asym={i['asym']!r}, "
                             f"convention={i['convention']!r}, euler={tuple(i['euler'])!r})")
            elif i["type"] == "quadrupolar":
                calls.append(f".add_quadrupolar(C_Q={i['C_Q']!r}, eta_Q={i['eta_Q']!r})")
            else:
                extra = f", abundance={i['abundance']!r}" if "abundance" in i else ""
                calls.append(f".add_jcoupling({i['nuclide']!r}, n={i['n']!r}, "
                             f"J_Hz={i['J_Hz']!r}{extra})")
        lines.append(head + "".join(f"\n        {c}" for c in calls) + ",")
    lines.append("]")
    if ppm_axis is not None:
        ax = np.asarray(ppm_axis)
        lines.append(f"ppm_axis = np.linspace({float(ax[0])!r}, {float(ax[-1])!r}, {ax.size})")
    else:
        lines.append("ppm_axis = np.linspace(-100, 100, 4096)")
    args = ", ".join(f"{k}={v!r}" for k, v in settings.items() if v is not None)
    lines.append(f"spec = simulate_combined(sites, ppm_axis, {args})")
    return "\n".join(lines)


def sites_to_json(sites):
    return json.dumps([s.to_dict() for s in as_sites(sites)], indent=2, ensure_ascii=False)


def sites_from_json(text):
    return [Site.from_dict(d) for d in json.loads(text)]


def plot_combined(result, nuclide="", show_sites=True, ax=None, pl_width=3.375,
                  pl_height=3.375, fontsize=12, font=None, linewidth=1.2, fname=None,
                  figure=None, xlims=None):
    """Total spectrum (and each site, dashed) on a descending ppm axis,
    colour-blind-safe. ``figure``: draw into an existing Figure (the
    GUI's Simulation plot). ``xlims``: (low, high) ppm to show, e.g. the
    centreband of a MAS spectrum whose axis also holds sidebands."""
    import matplotlib.pyplot as plt
    from .plotting import _apply_axis_style
    from .utils import NUC_label
    from ..plotting.accessibility import ordered_qualitative_colours
    if font:
        plt.rcParams["font.family"] = font
    if ax is None:
        fig = figure if figure is not None else plt.figure(figsize=(pl_width, pl_height))
        ax = fig.subplots()
    else:
        fig = ax.figure
    x = result["x_ppm"]
    scale = float(np.max(result["real_spec"])) or 1.0
    colours = ordered_qualitative_colours("okabe_ito")
    ax.plot(x, result["real_spec"] / scale, color=colours[0], lw=linewidth, label="Total")
    if show_sites and len(result["sites"]) > 1:
        for i, (label, spec) in enumerate(result["sites"].items()):
            ax.plot(x, spec / scale, lw=linewidth * 0.9, ls=(0, (4, 2)),
                    color=colours[1 + i % (len(colours) - 1)], label=label)
        ax.legend(frameon=False, fontsize=fontsize * 0.8)
    _apply_axis_style(ax, "in", 4.0, 1.0, 4.0, None, fontsize, False)
    lo, hi = (min(xlims), max(xlims)) if xlims else (min(x), max(x))
    ax.set_xlim(hi, lo)
    if xlims:                      # scale to what is shown
        shown = (x >= lo) & (x <= hi)
        top = max(float(np.max(result["real_spec"][shown])) / scale, 1e-12)
        ax.set_ylim(-0.03 * top, 1.05 * top)
    ax.set_xlabel(NUC_label(nuclide, x_unit="ppm") if nuclide else "Shift (ppm)",
                  fontsize=fontsize)
    if fname:
        fig.savefig(fname, dpi=300, bbox_inches="tight")
    return ax
