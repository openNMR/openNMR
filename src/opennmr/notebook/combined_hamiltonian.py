"""
CSA, Quad & J-coupling: one site, any combination of interactions.

A :class:`Site` is an observed nucleus with an isotropic shift, to which
interactions are *added* -- no function per combination::

    al = (Site("27Al", delta_iso=12.0, label="Al(1)")
          .add_quadrupolar(C_Q=4.2, eta_Q=0.35)             # MHz
          .add_csa(aniso=40.0, asym=0.2, euler=(0, 90, 0))  # ppm, deg
          .add_jcoupling("31P", n=2, J_Hz=15.0))
    spec = simulate_combined([al], ppm_axis, larmor_Hz=130.3e6, mas_Hz=15e3,
                             transitions="ct", order="second")

The same site can be written as a dict (``Site.to_dict`` /
``Site.from_dict``) -- what the GUI's Simulate > CSA, Quad & J-coupling
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

    A site may observe a different nucleus from the spectrum's (a 49Ti
    site on a 47Ti spectrum, a 63Cu coil signal on a 23Na one): it is
    simulated on its own ppm scale and placed on the observed one
    through the IUPAC frequency ratios (:mod:`opennmr.notebook.isotopes`).
    ``link`` (see :meth:`link_to`, :meth:`isotope_partner`) ties an
    isotope's site to another isotope's, sharing its chemistry.
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
        #: None, or {"to": parent label, "free": [parameters it keeps its own]}
        self.link = None

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

    # ---- isotope partners ------------------------------------------------

    def link_to(self, parent, free=()):
        """Share ``parent``'s chemistry (``parent``: a Site or its label):
        this site -- another isotope of the same element -- then takes
        parent's shift, CSA, Euler angles, eta_Q, broadening and line
        shape; C_Q scaled by the quadrupole moments, J by the
        gyromagnetic ratios and the amount by natural abundance and
        receptivity (:func:`opennmr.notebook.isotopes.partner_factors`).
        ``free``: parameter names it keeps its own instead (e.g.
        ``["amount"]`` when a selective pulse excites the two isotopes
        differently, ``["fwhm"]`` when a distribution of EFGs dominates
        the width). Returns the site."""
        to = parent.label if isinstance(parent, Site) else str(parent)
        self.link = {"to": to, "free": [str(n) for n in free]}
        return self

    def unlink(self):
        """Give the site its own parameters again (keeps its current values)."""
        self.link = None
        return self

    def isotope_partner(self, nucleus=None, label=None, free=()):
        """A new site of another isotope of this site's element, linked
        to this one (see :meth:`link_to`), with the derived values filled
        in. ``nucleus``: default the element's other NMR-active isotope
        (it must have exactly one, e.g. 47Ti -> 49Ti, 63Cu -> 65Cu)."""
        from .isotopes import isotope_partners
        if self.link:
            raise ValueError(f"{self.label} is itself linked to {self.link['to']}; add the "
                             "partner to that site instead.")
        if nucleus is None:
            others = isotope_partners(self.nucleus)
            if len(others) != 1:
                raise ValueError(f"{self.nucleus}: name the partner isotope -- the element has "
                                 f"{len(others)} other NMR-active isotopes ({others}).")
            nucleus = others[0]
        partner = Site(nucleus, label=label or f"{self.label} ({nucleus})")
        partner.link_to(self, free=free)
        resolved = resolve_link_dicts([self.to_dict(), partner.to_dict()])[1]
        return Site.from_dict(resolved)

    def copy(self, label=None):
        site = copy.deepcopy(self)
        if label is not None:
            site.label = label
        return site

    # ---- description -----------------------------------------------------

    def to_dict(self):
        d = {"nucleus": self.nucleus, "spin": self.spin, "delta_iso": self.delta_iso,
             "label": self.label, "amount": self.amount, "fwhm": self.fwhm,
             "lineshape": self.lineshape, "eta": self.eta,
             "interactions": copy.deepcopy(self.interactions)}
        if self.link:
            d["link"] = copy.deepcopy(self.link)
        return d

    @classmethod
    def from_dict(cls, d):
        site = cls(d["nucleus"], delta_iso=d.get("delta_iso", 0.0), label=d.get("label"),
                   amount=d.get("amount", 1.0), fwhm=d.get("fwhm", 1.0),
                   lineshape=d.get("lineshape", "gaussian"), eta=d.get("eta", 0.0),
                   spin=d.get("spin"))
        for inter in d.get("interactions", []):
            params = {k: v for k, v in inter.items() if k != "type"}
            site.add(inter["type"], **params)
        link = d.get("link")
        if link and link.get("to"):
            site.link_to(link["to"], free=link.get("free") or ())
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
        if self.link:
            own = self.link.get("free") or []
            bits.append(f"linked to {self.link['to']}"
                        + (f" (own {', '.join(own)})" if own else ""))
        return "; ".join(bits)

    def __repr__(self):
        return f"Site({self.describe()})"


def as_sites(sites):
    """Accept a Site, a dict, or a list of either; return a list of Sites."""
    if isinstance(sites, (Site, dict)):
        sites = [sites]
    return [s if isinstance(s, Site) else Site.from_dict(s) for s in sites]


# ============================================================
# Several nuclei: isotope links and placement on the observed scale
# ============================================================

def resolve_link_dicts(dicts):
    """Site dicts with every linked site's values derived from its
    parent's (:meth:`Site.link_to`), keeping its own values for the names
    in its link's ``free``. Unlinked sites pass through unchanged (not
    copied). Raises ValueError for a link that cannot hold."""
    if not any(d.get("link") for d in dicts):
        return list(dicts)
    from .combined_fitting import site_parameter_names, _get, _set
    from .isotopes import element_of, partner_factors
    by_label = {d["label"]: d for d in dicts}
    out = []
    for d in dicts:
        link = d.get("link")
        if not link:
            out.append(d)
            continue
        parent = by_label.get(link.get("to"))
        who = f"{d['label']} ({d['nucleus']})"
        if parent is None or parent is d:
            raise ValueError(f"{who} is linked to {link.get('to')!r}, which is not another "
                             f"site ({list(by_label)}).")
        if parent.get("link"):
            raise ValueError(f"{who} is linked to {parent['label']}, which is itself linked: "
                             "link both isotopes to the same parent site.")
        if parent["nucleus"] == d["nucleus"]:
            raise ValueError(f"{who} is linked to a site of the same nucleus -- a link ties "
                             "two isotopes; for a copy of a site, duplicate it instead.")
        if element_of(parent["nucleus"]) != element_of(d["nucleus"]):
            raise ValueError(f"{who} cannot share {parent['label']}'s chemistry: only isotopes of "
                             f"one element do ({parent['nucleus']} is not "
                             f"{element_of(d['nucleus'])}). Remove the link to give it its own "
                             "parameters.")
        f = partner_factors(parent["nucleus"], d["nucleus"])
        new = copy.deepcopy(parent)
        new.update(nucleus=d["nucleus"], spin=float(find_isotope_info(d["nucleus"])["spin"]),
                   label=d["label"], link=copy.deepcopy(link))
        if new["spin"] < 1:
            new["interactions"] = [i for i in new["interactions"] if i["type"] != "quadrupolar"]
        for inter in new["interactions"]:
            if inter["type"] == "quadrupolar":
                if f["C_Q"] is None:
                    new["interactions"] = [i for i in new["interactions"]
                                           if i["type"] != "quadrupolar"]
                    break
                inter["C_Q"] = abs(float(inter["C_Q"])) * f["C_Q"]
            elif inter["type"] == "jcoupling":
                inter["J_Hz"] = float(inter["J_Hz"]) * f["J"]
        free = list(link.get("free") or [])
        if f["amount"] is None and "amount" not in free:
            raise ValueError(f"{who}: {d['nucleus']} has no natural abundance to scale the "
                             "amount by -- give the partner its own amount (free 'amount').")
        if f["amount"] is not None:
            new["amount"] = float(parent.get("amount", 1.0)) * f["amount"]
        names = site_parameter_names(new)
        bad = [n for n in free if n not in names]
        if bad:
            raise ValueError(f"{who}: cannot keep its own {bad}; a linked {d['nucleus']} site "
                             f"has {names}.")
        for name in free:
            try:
                _set(new, name, _get(d, name))
            except (KeyError, IndexError):
                pass        # the partner has no own value yet: keep the derived one
        out.append(new)
    return out


def resolve_links(sites):
    """Sites with every link resolved (see :func:`resolve_link_dicts`)."""
    sites = as_sites(sites)
    if not any(s.link for s in sites):
        return sites
    return [Site.from_dict(d) for d in resolve_link_dicts([s.to_dict() for s in sites])]


def observed_nucleus(sites, observe=None):
    """The spectrum's nucleus: ``observe`` if given, else the first
    unlinked site's."""
    if observe:
        find_isotope_info(str(observe))
        return str(observe)
    sites = as_sites(sites)
    first = next((s for s in sites if not s.link), sites[0] if sites else None)
    if first is None:
        raise ValueError("Give at least one site.")
    return first.nucleus


def site_axes(sites, ppm_axis, larmor_Hz, observe):
    """Per site: (its own ppm scale over the observed ``ppm_axis``, its
    own Larmor frequency) -- the observed ones for the observed nucleus."""
    from .isotopes import frequency_ratio, from_observed_ppm
    out = []
    for s in as_sites(sites):
        if s.nucleus == observe:
            out.append((ppm_axis, larmor_Hz))
        else:
            R, _src = frequency_ratio(s.nucleus, observe)
            out.append((from_observed_ppm(ppm_axis, s.nucleus, observe), larmor_Hz * R))
    return out


def placement_notes(sites, observe, ppm_axis=None):
    """Log lines for every nucleus other than ``observe``: where its 0 ppm
    falls, and whether each of its sites lies outside ``ppm_axis``."""
    from .isotopes import describe_placement, to_observed_ppm
    sites = as_sites(sites)
    notes = [describe_placement(n, observe)
             for n in dict.fromkeys(s.nucleus for s in sites) if n != observe]
    if ppm_axis is not None and len(ppm_axis):
        lo, hi = float(np.min(ppm_axis)), float(np.max(ppm_axis))
        for s in sites:
            if s.nucleus != observe:
                at = float(to_observed_ppm(s.delta_iso, s.nucleus, observe))
                if not lo - 5 * s.fwhm <= at <= hi + 5 * s.fwhm:
                    notes.append(f"{s.label} ({s.nucleus}) falls at {at:.1f} ppm on the "
                                 f"{observe} scale, outside the axis ({lo:g} to {hi:g} ppm).")
    return notes


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
                      n_time=None, max_n_time=65536, use_symmetry=True, observe=None):
    """
    Powder spectrum of one or more sites, each with any combination of
    CSA, quadrupolar and J interactions, static (``mas_Hz`` = 0) or
    under MAS.

    Parameters
    ----------
    sites : Site, dict, or a list of them. Sites of another nucleus (an
        isotope partner, a background signal) are simulated on their own
        scale and placed on the observed one by the IUPAC frequency
        ratios; linked sites take their parent's chemistry (Site.link_to).
    ppm_axis : uniformly spaced ppm axis (either direction), on the
        observed nucleus's scale.
    larmor_Hz or B0_T : the observe frequency |gamma B0|, or the field.
    observe : the spectrum's nucleus (default: the first unlinked site's).
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
    labels = [s.label for s in sites]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Site labels must be unique, got {labels}.")
    observe = observed_nucleus(sites, observe)
    sites = resolve_links(sites)
    if larmor_Hz is None:
        if B0_T is None:
            raise ValueError("Give larmor_Hz or B0_T.")
        larmor_Hz = abs(float(find_isotope_info(observe)["gamma"])) * float(B0_T)
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

    site_spectra, components = {}, {}
    notes = placement_notes(sites, observe, ppm_axis)
    used = {}
    for site, (axis, nu0) in zip(sites, site_axes(sites, ppm_axis, larmor_Hz, observe)):
        comp, powder = _site_sticks(site, axis, nu0, mas_Hz, order, transitions,
                                    beta_r, n_zcw, n_gamma, n_time, max_n_time, use_symmetry)
        used[site.label] = powder
        broadened = _broaden_site(site, comp, axis)
        components[site.label] = broadened
        site_spectra[site.label] = np.sum(list(broadened.values()), axis=0)
        if not powder["octant"] and use_symmetry:
            notes.append(f"{site.label}: CSA not coincident with the EFG, so the full sphere "
                         "was averaged (no octant symmetry)")

    total = np.sum(list(site_spectra.values()), axis=0)
    peak = float(np.max(total)) if total.size else 0.0
    settings = {"larmor_Hz": larmor_Hz, "mas_Hz": mas_Hz, "order": order,
                "transitions": transitions, "rotor_angle": rotor_angle, "n_zcw": int(n_zcw),
                "n_gamma": n_gamma, "use_symmetry": use_symmetry, "observe": observe}
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
                         order="second", threshold=2e-3, observe=None):
    """ppm limits framing every site: static -- its full second-order
    frequency range, widened by the J multiplet; MAS -- wherever a quick,
    coarse MAS simulation has more than ``threshold`` of its maximum
    (spinning sidebands of a CSA spun slower than its width lie *outside*
    the static range, so the static range alone would clip them). Either
    way padded by ``margin`` of the range and five broadening FWHMs."""
    from .isotopes import frequency_ratio, to_observed_ppm
    observe = observed_nucleus(sites, observe)
    sites = resolve_links(sites)
    theta, phi, _ = zcw_angles(n)
    b = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    lows, highs, fwhm = [], [], 0.0
    for site in sites:
        nu0 = larmor_Hz * frequency_ratio(site.nucleus, observe)[0]
        nu, _ = _site_frequencies(site, nu0, b, _rows(site, transitions), "second")
        j_ppm, _w = _j_lines(site, nu0)
        ppm = site.delta_iso + nu / nu0 * 1e6
        ends = to_observed_ppm([ppm.min() + j_ppm.min(), ppm.max() + j_ppm.max()],
                               site.nucleus, observe)
        lows.append(float(ends[0]))
        highs.append(float(ends[1]))
        fwhm = max(fwhm, site.fwhm)
    low, high = min(lows), max(highs)
    if mas_Hz:
        spacing = float(mas_Hz) / larmor_Hz * 1e6
        wide = np.linspace(low - 20 * spacing - 10 * fwhm, high + 20 * spacing + 10 * fwhm, 16384)
        spec = simulate_combined(sites, wide, larmor_Hz, mas_Hz=mas_Hz, order=order,
                                 transitions=transitions, n_zcw=377,
                                 observe=observe)["real_spec"]
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
        if s.link:
            free = f", free={list(s.link.get('free') or [])!r}" if s.link.get("free") else ""
            calls.append(f".link_to({s.link['to']!r}{free})")
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
    from ..plotting.accessibility import default_series_colours
    if font:
        plt.rcParams["font.family"] = font
    if ax is None:
        fig = figure if figure is not None else plt.figure(figsize=(pl_width, pl_height))
        ax = fig.subplots()
    else:
        fig = ax.figure
    x = result["x_ppm"]
    scale = float(np.max(result["real_spec"])) or 1.0
    # The total in black; the sites in a palette that is colour-blind
    # friendly for that many (measured -- accessibility.default_palette).
    ax.plot(x, result["real_spec"] / scale, color="#000000", lw=linewidth, label="Total")
    if show_sites and len(result["sites"]) > 1:
        colours = default_series_colours(len(result["sites"]))
        for i, (label, spec) in enumerate(result["sites"].items()):
            ax.plot(x, spec / scale, lw=linewidth * 0.9, ls=(0, (4, 2)),
                    color=colours[i], label=label)
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
