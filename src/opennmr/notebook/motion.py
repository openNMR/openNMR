"""
Motional lineshapes of anisotropic interactions: reorientation of the
CSA, quadrupolar and dipolar tensors, hops between inequivalent sites,
or both at once -- static, under MAS at any rate relative to the spinning speed,
and after a Hahn or quadrupolar (solid) echo.

(The notebook counterpart of Dynamics > Anisotropic motion; Simulate is
the source of the interactions: any :class:`~opennmr.notebook.
combined_hamiltonian.Site` can be given a motion.)

Model
-----
A *state* is ``(environment, g1, g2, ...)``:

* an **environment** is a Simulate site (its delta_iso, CSA, quadrupolar
  coupling) placed in a common *molecular frame* by three Euler angles,
  with optional heteronuclear dipolar couplings along given directions;
* each **rotational mode** ``g`` is a set of rotations with a jump
  network (n-fold ring jumps, a two-site flip, rotational diffusion on
  a ring, tetrahedral/octahedral reorientation, or your own).

Hops between environments (the same hop table as
:mod:`opennmr.notebook.exchange`: barriers, attempt frequencies or
direct rates) and every mode are independent processes, so the full rate
matrix is their Kronecker sum, and the populations are the product of
each factor's equilibrium populations. The tensor of state
``(e, g1, ..., gM)`` is ``R T_e R^T`` with ``R = R_gM ... R_g2 R_g1``:
**modes are listed innermost first, and every axis is given in the
molecular frame** (a methyl spinning on a chain that itself turns about
the chain axis: the methyl mode first, the chain mode second).

Lineshape
---------
For a crystallite with B0 along ``b`` (molecular frame) each state has a
frequency ``nu_s(b)`` and the transverse magnetisation obeys

    dM/dt = (i*2*pi*nu - R2 + K) M

(the Bloch-McConnell equation of :mod:`opennmr.notebook.exchange`, now
with orientation-dependent frequencies). It is solved exactly -- a
batched eigen-decomposition per orientation -- and averaged over a ZCW
powder set. The first-order frequencies are linear in each tensor:

    nu = nu0*1e-6*(delta_iso + b.D.b)                       CSA
       - 1.5*K*(2m-1)*b.Q.b,  K = C_Q/(2I(2I-1))            quadrupolar
       +/- b.Dd.b                                          dipolar (spin-1/2 partner)

with ``D`` traceless in ppm, ``Q = C_Q*V/V_zz`` in Hz and
``Dd = D_dip*(3/2)(u u^T - 1/3)`` in Hz (``D_dip = mu0 g_I g_S h /
(4 pi^2 r^3)`` -- the *edge* of the static pattern is +/-D_dip; 23.3 kHz
for a 1H-13C bond of 1.09 A). The same tensors sum linearly over a fast
motion, which gives the **motionally averaged tensor**
(:func:`averaged_tensors`, :func:`effective_site`) and the fast-limit
spectrum, static or MAS (:func:`averaged_spectrum`).

Validity (all reported in the result's ``log``)
-----------------------------------------------
* Each transition and each dipolar spin state is an independent
  Bloch-McConnell problem: the rate must be much smaller than both the
  Larmor frequency and (for I > 1/2 satellites) the quadrupolar
  splitting. Motion at ~nu_Q mixes transitions; this model does not
  describe it.
* Quadrupolar terms are first order by default; ``order='second'`` adds the
  second-order shifts (the central transition's powder pattern). Second-order
  frequencies are taken state by state, which describes motion slower than
  nu_Q; for motion much faster than nu_Q the EFG averages first
  (:func:`averaged_spectrum`, ``order='second'``). The two fast limits differ
  (:func:`limit_spectra` returns both).
* The partner of a dipolar coupling is a static spin-1/2 (no partner
  relaxation, no decoupling); homonuclear dipolar networks are not
  modelled.
* Dynamic MAS and echoes use a time-domain engine (steps of a rotor period, exact
  exchange within a step, frequencies averaged over the step): its accuracy is set by
  ``steps_per_period`` (default: no sideband aliases) and it needs the FID to decay, so
  every line is at least ``min_fwhm_points`` wide. A single-state model under MAS
  without echo keeps the exact sideband engine. Echo pulses are ideal and
  instantaneous; transitions are independent (no coherence transfer between
  transitions).
"""

from __future__ import annotations

import copy
import itertools
import math
import warnings

import numpy as np

from .combined_hamiltonian import Site, as_sites, _csa_frame_tensor, _rotation_zyz
from .exchange import arrhenius_rate, equilibrium_populations, exchange_regime, rate_matrix
from .simulations import (
    _apply_broadening, _bin_sticks, _efg_principal, _mas_prepare, _mas_sticks,
    _perturbative_offsets, _rotor_frame, _select_transitions, convert_csa_parameters,
    quadrupolar_transition_table,
    zcw_angles,
)
from .utils import find_isotope_info

__all__ = [
    "MotionSite", "Mode", "MotionModel", "ring_jump", "flip", "ring_diffusion",
    "random_reorientation", "custom_mode", "cone_jump", "dipolar_coupling_Hz",
    "axis_vector", "rotation_about_axis", "euler_from_rotation",
    "interaction_site", "rotation_model",
    "motion_spectrum", "motion_rate_series", "averaged_tensors", "effective_site",
    "averaged_spectrum", "limit_spectra", "plot_motion_series", "mode_from_spec",
    "model_from_spec", "motion_axis_limits", "motion_overlay", "motion_overlay_series", "plot_motion_overlay",
    "plot_motion_overlay_stack", "plot_motion_limits",
    "motion_notebook_code",
]

#: Planck constant (J s) and mu0/4pi (T m / A), exact in the 2019 SI.
_H = 6.62607015e-34
_MU0_4PI = 1.0e-7


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------

def axis_vector(axis):
    """A unit vector from ``'x'|'y'|'z'``, ``(polar, azimuth)`` in degrees,
    or a 3-vector."""
    if isinstance(axis, str):
        table = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
        key = axis.strip().lower()
        if key not in table:
            raise ValueError(f"axis {axis!r}: use 'x', 'y', 'z', (polar, azimuth) degrees or a 3-vector.")
        return np.array(table[key])
    a = np.asarray(axis, dtype=float).ravel()
    if a.size == 2:
        th, ph = np.deg2rad(a)
        v = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    elif a.size == 3:
        v = a.copy()
    else:
        raise ValueError("An axis is 'x'/'y'/'z', (polar, azimuth) in degrees or a 3-vector.")
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("An axis cannot be the zero vector.")
    return v / norm


def rotation_about_axis(axis, angle_rad):
    """Rotation matrix (Rodrigues) of ``angle_rad`` about ``axis``."""
    u = axis_vector(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    ux = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return c * np.eye(3) + s * ux + (1.0 - c) * np.outer(u, u)


def euler_from_rotation(R):
    """ZYZ Euler angles (degrees) of a proper rotation, as
    :func:`_rotation_zyz` builds it: ``R = Rz(a) Ry(b) Rz(g)``. ``b`` comes
    from ``atan2`` (``acos`` loses ~1e-8 rad of precision near 0 and 180
    degrees). At ``b`` = 0 or 180 degrees only a combination of ``a`` and
    ``g`` exists; ``a`` is then set to 0."""
    R = np.asarray(R, dtype=float)
    s = math.hypot(R[0, 2], R[1, 2])
    b = math.atan2(s, R[2, 2])
    if s < 1e-12:
        a = 0.0
        g = (math.atan2(R[1, 0], R[0, 0]) if R[2, 2] > 0
             else math.atan2(R[0, 1], -R[0, 0]))
    else:
        a = math.atan2(R[1, 2], R[0, 2])
        g = math.atan2(R[2, 1], -R[2, 0])
    return tuple(float(x) for x in np.rad2deg([a, b, g]))


def dipolar_coupling_Hz(nucleus, partner, r_A):
    """Dipolar coupling constant ``D = mu0 g_I g_S h / (4 pi^2 r^3)`` in Hz
    (``g`` the magnetogyric ratios in Hz/T, ``r`` in angstrom): the
    half-width of the static two-spin pattern's edges is D."""
    gi = abs(float(find_isotope_info(nucleus)["gamma"]))
    gs = abs(float(find_isotope_info(partner)["gamma"]))
    r = float(r_A) * 1e-10
    if r <= 0:
        raise ValueError("The internuclear distance must be positive.")
    return _MU0_4PI * gi * gs * _H / r ** 3


def _principal_axes(T, order):
    """Eigenvalues ``(xx, yy, zz)`` and the right-handed matrix whose
    columns are those axes, for a symmetric traceless ``T``. ``order``:
    ``'csa'`` (|zz| >= |xx| >= |yy|, Haeberlen) or ``'efg'``
    (|zz| >= |yy| >= |xx|)."""
    T = 0.5 * (np.asarray(T, float) + np.asarray(T, float).T)
    vals, vecs = np.linalg.eigh(T)
    idx = list(np.argsort(-np.abs(vals), kind="stable"))
    zz = idx[0]
    big, small = idx[1], idx[2]
    xx, yy = (big, small) if order == "csa" else (small, big)
    E = np.column_stack([vecs[:, xx], vecs[:, yy], vecs[:, zz]])
    if np.linalg.det(E) < 0:
        E[:, 0] = -E[:, 0]
    return np.array([vals[xx], vals[yy], vals[zz]]), E


# ----------------------------------------------------------------------
# Environments
# ----------------------------------------------------------------------

class MotionSite:
    """One environment: a Simulate :class:`Site` (shift, CSA, quadrupolar
    coupling; J couplings are ignored) placed in the molecular frame by
    ``euler`` = (alpha, beta, gamma) degrees (the site frame -- the EFG
    frame, or the CSA frame without a quadrupolar interaction -- in the
    molecular frame), plus optional dipolar couplings to spin-1/2
    partners added with :meth:`add_dipolar`.
    """

    def __init__(self, site, euler=(0.0, 0.0, 0.0), dipolar=()):
        site = as_sites(site)[0].copy()
        if site.get("jcoupling"):
            warnings.warn(f"{site.label}: J couplings are not part of the motional model and "
                          "are ignored.", UserWarning, stacklevel=2)
            site.remove("jcoupling")
        self.site = site
        euler = tuple(float(a) for a in euler)
        if len(euler) != 3:
            raise ValueError("euler must be (alpha, beta, gamma) in degrees.")
        self.euler = euler
        self.dipolar = []
        for d in dipolar:
            self.add_dipolar(**d)

    label = property(lambda self: self.site.label)
    nucleus = property(lambda self: self.site.nucleus)
    spin = property(lambda self: self.site.spin)

    def add_dipolar(self, partner, D_Hz=None, r_A=None, vector=(0.0, 0.0, 1.0)):
        """A dipolar coupling to a spin-1/2 ``partner`` (e.g. '1H'): give
        ``D_Hz`` or the distance ``r_A`` (angstrom), and the internuclear
        ``vector`` in the *site* frame (3-vector or (polar, azimuth)
        degrees). Heteronuclear only."""
        info = find_isotope_info(partner)
        if abs(float(info["spin"]) - 0.5) > 1e-9:
            raise ValueError(f"Dipolar partner {partner} has I = {info['spin']}: only spin-1/2 "
                             "partners are modelled.")
        if str(partner) == self.nucleus:
            raise ValueError("Homonuclear dipolar couplings are not modelled (a spin-pair "
                             "problem, not a tensor one); give a different nucleus.")
        if (D_Hz is None) == (r_A is None):
            raise ValueError("Give exactly one of D_Hz or r_A.")
        D = float(D_Hz) if D_Hz is not None else dipolar_coupling_Hz(self.nucleus, partner, r_A)
        self.dipolar.append({"partner": str(partner), "D_Hz": D,
                             "vector": [float(x) for x in axis_vector(vector)]})
        return self

    def tensors(self):
        """Molecular-frame tensors: ``csa`` (ppm, traceless, or None),
        ``quad`` (Hz = C_Q V/V_zz, or None) and ``dip`` (list of Hz tensors)."""
        R = _rotation_zyz(*self.euler)
        csa = _csa_frame_tensor(self.site)
        quad = None
        q = self.site.get("quadrupolar")
        if q:
            quad = R @ (q[0]["C_Q"] * 1e6 * np.diag(_efg_principal(q[0]["eta_Q"]))) @ R.T
        dip = []
        for d in self.dipolar:
            u = R @ np.asarray(d["vector"])
            dip.append(d["D_Hz"] * 1.5 * (np.outer(u, u) - np.eye(3) / 3.0))
        return {"csa": None if csa is None else R @ csa @ R.T, "quad": quad, "dip": dip}

    def to_dict(self):
        return {"site": self.site.to_dict(), "euler": list(self.euler),
                "dipolar": copy.deepcopy(self.dipolar)}

    @classmethod
    def from_dict(cls, d):
        out = cls(Site.from_dict(d["site"]), euler=d.get("euler", (0, 0, 0)))
        for dip in d.get("dipolar", []):
            out.add_dipolar(**dip)
        return out


def _as_motion_site(s):
    if isinstance(s, MotionSite):
        return s
    if isinstance(s, dict) and "site" in s:
        return MotionSite.from_dict(s)
    return MotionSite(s)


# ----------------------------------------------------------------------
# Rotational modes
# ----------------------------------------------------------------------

def _closure(generators, limit=96):
    """The finite rotation group generated by ``generators``."""
    group = [np.eye(3)]
    frontier = [np.eye(3)]
    while frontier:
        new = []
        for g in frontier:
            for h in generators:
                m = h @ g
                if not any(np.allclose(m, x, atol=1e-9) for x in group):
                    group.append(m)
                    new.append(m)
                    if len(group) > limit:
                        raise ValueError("The generators do not give a finite group.")
        frontier = new
    return group


class Mode:
    """A rotational mode: ``rotations`` (n, 3, 3) and a jump network.

    ``rate`` is ``{'k_Hz': k}`` (a temperature-independent rate *per
    allowed jump*) or ``{'Ea_eV': Ea, 'nu0_Hz': nu0}`` (Arrhenius). For
    ``rate_is='D'`` (ring diffusion) the number is the rotational
    diffusion rate ``D`` (s^-1: the decay rate of the m = 1 harmonic)
    instead. ``model``: 'nearest' (each state jumps to its two
    neighbours on the ring), 'forward' / 'backward' (one way round only), 'random' (to every other state) or 'custom'
    (``pattern[i, j]`` = relative rate i -> j). ``periodic=False`` makes the
    'nearest' / 'forward' / 'backward' networks an open chain (an arc of positions
    that does not close on itself) instead of a ring.
    """

    def __init__(self, kind, rotations, model="nearest", rate=None, pattern=None,
                 label=None, rate_is="k", axis=None, periodic=True):
        self.kind = str(kind)
        #: True when the last position is also a neighbour of the first (a closed ring);
        #: False for an open arc of positions (nearest-neighbour hops stop at the ends).
        self.periodic = bool(periodic)
        self.rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
        self.model = model
        if model not in ("nearest", "random", "custom", "forward", "backward"):
            raise ValueError("model must be 'nearest', 'random', 'forward', 'backward' or 'custom'.")
        self.pattern = None if pattern is None else np.asarray(pattern, dtype=float)
        if model == "custom":
            n = len(self.rotations)
            if self.pattern is None or self.pattern.shape != (n, n):
                raise ValueError("A custom mode needs an (n, n) pattern of relative rates.")
        self.rate = dict(rate or {"k_Hz": 0.0})
        self.rate_is = rate_is
        self.label = label or kind
        self.axis = None if axis is None else [float(x) for x in axis_vector(axis)]

    @property
    def n(self):
        return len(self.rotations)

    def rate_value(self, T_K=None, k_override=None):
        """The per-jump rate (s^-1) -- or D for ring diffusion -- at ``T_K``."""
        if k_override is not None:
            return float(k_override)
        if self.rate.get("k_Hz") is not None:
            return float(self.rate["k_Hz"])
        if T_K is None:
            raise ValueError(f"Mode {self.label!r} has an Arrhenius rate: give the temperature T_K.")
        return float(arrhenius_rate(self.rate.get("nu0_Hz", 1e12), self.rate["Ea_eV"], T_K))

    def rate_matrix(self, T_K=None, k_override=None):
        n = self.n
        k = self.rate_value(T_K, k_override)
        if k < 0:
            raise ValueError(f"Mode {self.label!r} has a negative rate.")
        if self.rate_is == "D":                      # per-jump rate from D (m = 1 decay)
            k = k / (2.0 * (1.0 - math.cos(2.0 * math.pi / n)))
        K = np.zeros((n, n))
        if self.model == "random":
            rel = np.ones((n, n)) - np.eye(n)
        elif self.model == "custom":
            rel = self.pattern.copy()
            np.fill_diagonal(rel, 0.0)
        else:
            rel = np.zeros((n, n))
            for i in range(n):
                if self.model == "forward":          # one-way: i -> i+1 only (anticlockwise
                    steps = (i + 1,)                 # seen from +axis, the right-hand sense)
                elif self.model == "backward":
                    steps = (i - 1,)
                else:
                    steps = (i + 1, i - 1)
                for j in steps:
                    if self.periodic:
                        j %= n
                    elif not 0 <= j < n:             # open arc: no hop past either end
                        continue
                    if j != i:
                        rel[i, j] = 1.0
        K += k * rel.T                               # K[j, i] = rate i -> j
        K -= np.diag(K.sum(axis=0))
        return K

    def to_dict(self):
        return {"kind": self.kind, "rotations": self.rotations.tolist(), "model": self.model,
                "rate": dict(self.rate), "rate_is": self.rate_is, "label": self.label,
                "pattern": None if self.pattern is None else self.pattern.tolist(),
                "axis": self.axis, "periodic": self.periodic}

    @classmethod
    def from_dict(cls, d):
        return cls(d["kind"], d["rotations"], d.get("model", "nearest"), d.get("rate"),
                   d.get("pattern"), d.get("label"), d.get("rate_is", "k"), d.get("axis"),
                   periodic=d.get("periodic", True))


def _rate_dict(k_Hz=None, Ea_eV=None, nu0_Hz=1e12):
    if (k_Hz is None) == (Ea_eV is None):
        raise ValueError("Give either k_Hz or Ea_eV (with nu0_Hz).")
    return {"k_Hz": float(k_Hz)} if k_Hz is not None else {"Ea_eV": float(Ea_eV),
                                                          "nu0_Hz": float(nu0_Hz)}


def ring_jump(n=3, axis="z", k_Hz=None, Ea_eV=None, nu0_Hz=1e12, model="nearest", label=None,
              step_deg=None):
    """``n``-site jumps about ``axis`` (e.g. a methyl or a ring flip about its axis).
    ``k_Hz`` is the rate per allowed jump: to each neighbour for 'nearest', to each other
    site for 'random', one way round for 'forward' (anticlockwise looking down the axis)
    or 'backward' (clockwise).

    The positions are ``0, step, 2*step, ...`` degrees about the axis. By default
    ``step_deg = 360/n``, a closed ring. Give a smaller ``step_deg`` for an *arc* (e.g.
    three positions 20 degrees apart: a libration or a wobble); the neighbour hops then
    stop at its ends instead of closing the ring (``n * step_deg`` may not exceed 360)."""
    n = int(n)
    if n < 2:
        raise ValueError("A ring needs at least two sites.")
    periodic = True
    step = 360.0 / n
    if step_deg is not None:
        step = float(step_deg)
        if step <= 0.0 or n * step > 360.0 + 1e-9:
            raise ValueError(f"{n} positions {step:g}° apart do not fit in one turn: need "
                             "0 < step and n × step <= 360°.")
        periodic = abs(n * step - 360.0) < 1e-9
    if not periodic and model in ("forward", "backward"):
        raise ValueError("One-way hops need a closed ring (n × step = 360°): on an open arc "
                         "everything would end up in the last position. Use 'nearest' (both "
                         "ways) for an arc.")
    rots = [rotation_about_axis(axis, math.radians(step * j)) for j in range(n)]
    default = f"{n}-site jump" if periodic else f"{n}-position arc ({step:g}° steps)"
    return Mode("ring_jump", rots, model, _rate_dict(k_Hz, Ea_eV, nu0_Hz),
                label=label or default, axis=axis, periodic=periodic)


def flip(angle_deg=180.0, axis="z", k_Hz=None, Ea_eV=None, nu0_Hz=1e12, label=None):
    """A two-site flip: the identity and a rotation of ``angle_deg`` about
    ``axis`` (a 180-degree flip, or any two-site jump)."""
    rots = [np.eye(3), rotation_about_axis(axis, math.radians(angle_deg))]
    return Mode("flip", rots, "random", _rate_dict(k_Hz, Ea_eV, nu0_Hz),
                label=label or f"{angle_deg:g}° flip", axis=axis)


def ring_diffusion(axis="z", D_Hz=None, Ea_eV=None, nu0_Hz=1e12, n=24, label=None):
    """Rotational diffusion about ``axis`` as an ``n``-site ring of
    nearest-neighbour jumps (default 24: the rank-2 decay rates are within
    ~2 % of continuous diffusion; at least 5 are needed). ``D_Hz``: the
    rotational diffusion rate, so that the m = 1 harmonic decays as
    exp(-D t) and the m = 2 (tensor) harmonic as exp(-4 D t)."""
    n = int(n)
    if n < 5:
        raise ValueError("A diffusion ring needs at least 5 sites to carry rank-2 tensors.")
    rots = [rotation_about_axis(axis, 2.0 * math.pi * j / n) for j in range(n)]
    return Mode("ring_diffusion", rots, "nearest", _rate_dict(D_Hz, Ea_eV, nu0_Hz),
                label=label or "ring diffusion", rate_is="D", axis=axis)


def random_reorientation(group="octahedral", k_Hz=None, Ea_eV=None, nu0_Hz=1e12, label=None):
    """Reorientation among the 24 ('octahedral') or 12 ('tetrahedral')
    rotations of a cubic group, every pair connected at ``k_Hz``. Rank-2
    tensors average to zero over either group (isotropic motion)."""
    key = str(group).lower()
    if key.startswith("oct"):
        gens = [rotation_about_axis("z", math.pi / 2), rotation_about_axis("x", math.pi / 2)]
    elif key.startswith("tet"):
        gens = [rotation_about_axis((1, 1, 1), 2 * math.pi / 3), rotation_about_axis("z", math.pi)]
    else:
        raise ValueError("group must be 'octahedral' (24) or 'tetrahedral' (12).")
    rots = _closure(gens)
    return Mode("random_reorientation", rots, "random", _rate_dict(k_Hz, Ea_eV, nu0_Hz),
                label=label or f"{key} reorientation")


def custom_mode(rotations, pattern=None, k_Hz=None, Ea_eV=None, nu0_Hz=1e12, label=None):
    """Your own rotations (3x3 matrices or (alpha, beta, gamma) degrees) and a
    ``pattern`` of relative rates (``pattern[i, j]``: i -> j, scaled by the rate;
    default: every pair equal)."""
    rots = []
    for r in rotations:
        r = np.asarray(r, dtype=float)
        rots.append(_rotation_zyz(*r) if r.shape == (3,) else r)
    n = len(rots)
    if pattern is None:
        pattern = np.ones((n, n)) - np.eye(n)
    return Mode("custom", rots, "custom", _rate_dict(k_Hz, Ea_eV, nu0_Hz), pattern=pattern,
                label=label or "custom motion")


# ----------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------

def _kron_sum(mats):
    """Kronecker sum of independent processes' rate matrices."""
    dims = [m.shape[0] for m in mats]
    total = int(np.prod(dims))
    K = np.zeros((total, total))
    for i, m in enumerate(mats):
        left = np.eye(int(np.prod(dims[:i]))) if i else np.eye(1)
        right = np.eye(int(np.prod(dims[i + 1:]))) if i + 1 < len(dims) else np.eye(1)
        K += np.kron(np.kron(left, m), right)
    return K


def _factor_populations(K):
    if not np.any(K):
        return np.full(K.shape[0], 1.0 / K.shape[0])
    return equilibrium_populations(K)


class MotionModel:
    """Environments, hops between them and rotational modes.

    Parameters
    ----------
    sites : MotionSite / Site / dict, or a list of them. All must observe
        the same nucleus and use the same line-shape family.
    hops : hop dicts of :func:`opennmr.notebook.exchange.rate_matrix`
        (indices into ``sites``); None for no hopping.
    modes : :class:`Mode` list, **innermost first** (see the module notes).
    couplings : jumps that change the environment and the orientation *together*
        (correlated motion, which the independent hops and modes cannot express):
        dicts ``{"from": state, "to": state, ...}`` where a state is its index in
        :meth:`state_table` or a tuple ``(environment, g1, g2, ...)``, with
        ``k_Hz`` (and optionally ``k_rev_Hz``, default equal) or
        ``Ea_eV`` (optionally ``Ea_rev_eV``) with ``nu0_Hz`` -- as a hop. They
        add to the rates of the independent processes; with any coupling the
        populations come from the whole rate matrix (detailed balance if the
        barriers give it). ``hop_scale`` scales couplings too.
    """

    MAX_STATES = 64

    def __init__(self, sites, hops=None, modes=(), couplings=None):
        if isinstance(sites, (MotionSite, Site, dict)):
            sites = [sites]
        self.sites = [_as_motion_site(s) for s in sites]
        if not self.sites:
            raise ValueError("Give at least one environment.")
        labels = [s.label for s in self.sites]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Environment labels must be unique, got {labels}.")
        nuc = {s.nucleus for s in self.sites}
        if len(nuc) != 1:
            raise ValueError(f"All environments must observe the same nucleus, got {sorted(nuc)}.")
        if len({s.site.lineshape for s in self.sites}) != 1:
            raise ValueError("All environments must share one line-shape family "
                             "('gaussian' or 'lorentzian').")
        shape = self.sites[0].site.lineshape
        if shape not in ("gaussian", "lorentzian"):
            raise ValueError(f"lineshape {shape!r} is not available for motional lineshapes: use "
                             "'gaussian' or 'lorentzian' (the Lorentzian part comes from the "
                             "dynamics itself).")
        self.nucleus = self.sites[0].nucleus
        self.spin = self.sites[0].spin
        self.hops = [dict(h) for h in (hops or [])]
        self.modes = [m if isinstance(m, Mode) else Mode.from_dict(m) for m in modes]
        self.couplings = [dict(c) for c in (couplings or [])]
        for c in self.couplings:                      # validate now, not at the first spectrum
            i, j = self._state_index(c["from"]), self._state_index(c["to"])
            if i == j:
                raise ValueError("A coupling must join two different states.")
        if self.n_states > self.MAX_STATES:
            warnings.warn(f"{self.n_states} states: the lineshape will be slow "
                          f"(more than {self.MAX_STATES}). Use fewer jump sites.",
                          UserWarning, stacklevel=2)

    # -- sizes ---------------------------------------------------------
    @property
    def n_env(self):
        return len(self.sites)

    @property
    def n_states(self):
        return self.n_env * int(np.prod([m.n for m in self.modes])) if self.modes else self.n_env

    def state_table(self):
        """One ``(environment label, mode state indices)`` per state, in order."""
        dims = [self.n_env] + [m.n for m in self.modes]
        return [(self.sites[idx[0]].label, idx[1:]) for idx in np.ndindex(*dims)]

    def _state_index(self, spec):
        """The index of a state given as an int or ``(environment, g1, g2, ...)``."""
        dims = [self.n_env] + [m.n for m in self.modes]
        if isinstance(spec, (int, np.integer)):
            idx = int(spec)
        else:
            spec = tuple(int(x) for x in spec)
            if len(spec) != len(dims) or any(not 0 <= v < d for v, d in zip(spec, dims)):
                raise ValueError(f"State {spec} does not fit the model's dimensions {tuple(dims)} "
                                 "(environment, then one index per mode).")
            idx = int(np.ravel_multi_index(spec, dims))
        if not 0 <= idx < self.n_states:
            raise ValueError(f"State index {idx} is outside 0..{self.n_states - 1}.")
        return idx

    # -- rates ---------------------------------------------------------
    def rate_factors(self, T_K=None, k_modes=None, hop_scale=1.0):
        """Per-process rate matrices ``[K_hops, K_mode1, ...]``."""
        k_modes = k_modes or {}
        if self.hops and T_K is None and any(h.get("k_Hz") is None for h in self.hops):
            raise ValueError("The hop table has Arrhenius barriers: give the temperature T_K "
                             "(or direct rates 'k_Hz').")
        if self.hops:
            K_env = hop_scale * rate_matrix(self.n_env, self.hops, T_K)
        else:
            K_env = np.zeros((self.n_env, self.n_env))
        mats = [K_env]
        for i, m in enumerate(self.modes):
            mats.append(m.rate_matrix(T_K, k_modes.get(i, k_modes.get(m.label))))
        return mats

    def rate_matrix(self, T_K=None, k_modes=None, hop_scale=1.0):
        """The full rate matrix (Kronecker sum plus any couplings), ``K[j, i]`` = rate i -> j."""
        K = _kron_sum(self.rate_factors(T_K, k_modes, hop_scale))
        for c in self.couplings:
            i, j = self._state_index(c["from"]), self._state_index(c["to"])
            if c.get("k_Hz") is not None:
                kf = float(c["k_Hz"])
                kr = float(c["k_rev_Hz"]) if c.get("k_rev_Hz") is not None else kf
            else:
                if T_K is None:
                    raise ValueError("A coupling has Arrhenius barriers: give the temperature T_K "
                                     "(or direct rates 'k_Hz').")
                nu0 = float(c.get("nu0_Hz", 1e12))
                Ea = float(c["Ea_eV"])
                Ea_r = float(c["Ea_rev_eV"]) if c.get("Ea_rev_eV") is not None else Ea
                kf, kr = float(arrhenius_rate(nu0, Ea, T_K)), float(arrhenius_rate(nu0, Ea_r, T_K))
            kf, kr = hop_scale * kf, hop_scale * kr
            if kf < 0 or kr < 0:
                raise ValueError("A coupling has a negative rate.")
            K[j, i] += kf
            K[i, i] -= kf
            K[i, j] += kr
            K[j, j] -= kr
        return K

    def populations(self, T_K=None, k_modes=None, hop_scale=1.0):
        """Equilibrium populations: the product of each process's own, or (with
        couplings) the null vector of the whole rate matrix."""
        if self.couplings:
            return _factor_populations(self.rate_matrix(T_K, k_modes, hop_scale))
        p = np.ones(1)
        for K in self.rate_factors(T_K, k_modes, hop_scale):
            p = np.kron(p, _factor_populations(K))
        return p

    # -- tensors -------------------------------------------------------
    def state_rotations(self):
        """(S, 3, 3): ``R_gM ... R_g2 R_g1`` of every state (identity with no modes)."""
        out = []
        dims = [self.n_env] + [m.n for m in self.modes]
        for idx in np.ndindex(*dims):
            R = np.eye(3)
            for m, g in zip(self.modes, idx[1:]):
                R = m.rotations[g] @ R
            out.append(R)
        return np.array(out)

    def state_data(self):
        """Per-state tensors in the molecular frame (see module notes)."""
        env = [s.tensors() for s in self.sites]
        n_part = max(len(s.dipolar) for s in self.sites)
        dims = [self.n_env] + [m.n for m in self.modes]
        e_idx = np.array([idx[0] for idx in np.ndindex(*dims)])
        R = self.state_rotations()

        def rotate(mats):                     # mats: (E, 3, 3) -> (S, 3, 3)
            return np.einsum("sij,sjk,slk->sil", R, mats[e_idx], R)

        def stack(key, k=None):
            out = []
            for e in env:
                t = e[key] if k is None else (e["dip"][k] if k < len(e["dip"]) else None)
                out.append(np.zeros((3, 3)) if t is None else t)
            return np.array(out)

        has_csa = any(e["csa"] is not None for e in env)
        has_quad = any(e["quad"] is not None for e in env)
        R_env = [_rotation_zyz(*s.euler) for s in self.sites]
        quad_params = []
        for si, e in enumerate(e_idx):
            q = self.sites[e].site.get("quadrupolar")
            quad_params.append(None if not q else
                               (q[0]["C_Q"] * 1e6, q[0]["eta_Q"], R[si] @ R_env[e]))
        return {
            "quad_params": quad_params,
            "diso": np.array([s.site.delta_iso for s in self.sites])[e_idx],
            "csa": rotate(stack("csa")) if has_csa else None,
            "quad": rotate(stack("quad")) if has_quad else None,
            "dip": [rotate(stack("dip", k)) for k in range(n_part)],
            "fwhm_ppm": np.array([s.site.fwhm for s in self.sites])[e_idx],
            "amount": float(sum(s.site.amount for s in self.sites)),
            "e_idx": e_idx,
        }

    def lines(self, transitions="all"):
        """The independent lines: quadrupolar transitions x dipolar spin states."""
        has_quad = any(s.site.get("quadrupolar") for s in self.sites)
        n_part = max(len(s.dipolar) for s in self.sites)
        I = self.spin
        if has_quad:
            rows = _select_transitions(I, transitions)
            total = sum(r["weight"] for r in quadrupolar_transition_table(I))
        else:
            rows, total = [None], 1.0
        out = []
        for row in rows:
            for signs in itertools.product((1.0, -1.0), repeat=n_part):
                w = (row["weight"] / total if row else 1.0) * 0.5 ** n_part
                q = -1.5 * (2.0 * row["m"] - 1.0) / (2.0 * I * (2.0 * I - 1.0)) if row else 0.0
                label = (row["label"] if row else "line") + (
                    "" if not n_part else " " + "".join("+" if s > 0 else "−" for s in signs))
                out.append({"weight": w, "qcoef": q, "signs": signs, "label": label,
                            "group": row["group"] if row else "",
                            "m": row["m"] if row else None})
        return out

    def describe(self):
        lines = [s.site.describe() + (f"; placed at Euler ({s.euler[0]:g}, {s.euler[1]:g}, "
                                      f"{s.euler[2]:g})°" if any(s.euler) else "")
                 for s in self.sites]
        for s in self.sites:
            for d in s.dipolar:
                lines.append(f"{s.label}: dipolar to {d['partner']}, D = {d['D_Hz'] / 1e3:.4g} kHz")
        for h in self.hops:
            lines.append("hop " + ", ".join(f"{k} {v}" for k, v in h.items()))
        for i, m in enumerate(self.modes, 1):
            rate = (f"{m.rate['k_Hz']:.4g} s⁻¹" if m.rate.get("k_Hz") is not None else
                    f"Ea {m.rate['Ea_eV']:g} eV, ν0 {m.rate['nu0_Hz']:.3g} Hz")
            lines.append(f"mode {i} (innermost first): {m.label}, {m.n} states, {m.model}, "
                         f"{'D' if m.rate_is == 'D' else 'k per jump'} {rate}")
        table = self.state_table()
        for c in self.couplings:
            i, j = self._state_index(c["from"]), self._state_index(c["to"])
            rates = ", ".join(f"{k} {v}" for k, v in c.items() if k not in ("from", "to"))
            lines.append(f"coupled jump {table[i][0]}{table[i][1]} → {table[j][0]}{table[j][1]}: "
                         f"{rates}")
        lines.append(f"{self.n_states} states in total.")
        return lines

    def to_dict(self):
        return {"sites": [s.to_dict() for s in self.sites], "hops": copy.deepcopy(self.hops),
                "modes": [m.to_dict() for m in self.modes],
                "couplings": copy.deepcopy(self.couplings)}

    @classmethod
    def from_dict(cls, d):
        return cls([MotionSite.from_dict(s) for s in d["sites"]], d.get("hops"),
                   [Mode.from_dict(m) for m in d.get("modes", [])], d.get("couplings"))


def cone_jump(site, half_angle_deg, n_sites=3, k_Hz=None, Ea_eV=None, nu0_Hz=1e12,
              model="nearest"):
    """The Weblab-style case: the site's unique (z) axis jumps among
    ``n_sites`` positions on a cone of ``half_angle_deg``. Returns a
    :class:`MotionModel` with the site tilted by the half-angle from the
    cone axis (molecular z) and an ``n_sites`` ring about z."""
    ms = MotionSite(site, euler=(0.0, float(half_angle_deg), 0.0))
    return MotionModel([ms], modes=[ring_jump(n_sites, "z", k_Hz, Ea_eV, nu0_Hz, model,
                                              label=f"{n_sites}-site cone jump")])


# ----------------------------------------------------------------------
# Frequencies
# ----------------------------------------------------------------------

def _forms(data, b):
    """Bilinear forms ``b.T.b`` per state: csa (S, n) ppm, quad (S, n) Hz,
    dipolar list of (S, n) Hz."""
    def form(T):
        return None if T is None else np.einsum("ni,sij,nj->sn", b, T, b, optimize=True)
    return form(data["csa"]), form(data["quad"]), [form(T) for T in data["dip"]]


def _line_frequencies(data, forms, lines, larmor_Hz, n, quad=True):
    """(L, S, n) offsets from nu0 in Hz, delta_iso included (``n`` orientations)."""
    csa, quad_form, dip = forms
    if not quad:
        quad_form = None
    base = data["diso"][:, None] + (np.zeros((len(data["diso"]), n)) if csa is None else csa)
    nu = np.broadcast_to(larmor_Hz * 1e-6 * base, (len(lines),) + base.shape).copy()
    for li, ln in enumerate(lines):
        if quad_form is not None and ln["qcoef"]:
            nu[li] += ln["qcoef"] * quad_form
        for k, sgn in enumerate(ln["signs"]):
            if dip[k] is not None:
                nu[li] += sgn * dip[k]
    return nu


def _frequencies(data, b, lines, larmor_Hz, order="first", spin=None):
    """(L, S, n) offsets from nu0 in Hz for B0 directions ``b`` (n, 3), at
    first order (every term linear in a tensor) or second order (the
    quadrupolar frequencies from the perturbation expressions of
    :mod:`simulations`, state by state)."""
    if order not in ("first", "second"):
        raise ValueError("order must be 'first' or 'second'.")
    n = len(b)
    nu = _line_frequencies(data, _forms(data, b), lines, larmor_Hz, n, quad=(order == "first"))
    if order == "second":
        ms = sorted({l["m"] for l in lines if l["m"] is not None})
        if ms:
            col = {m: k for k, m in enumerate(ms)}
            for si, qp in enumerate(data.get("quad_params") or []):
                if qp is None:
                    continue
                C, eta, Reff = qp
                off = _perturbative_offsets(float(spin), float(C), float(eta), larmor_Hz,
                                            b @ Reff, np.array(ms), "second")     # (n, len(ms))
                for li, ln in enumerate(lines):
                    if ln["m"] is not None:
                        nu[li, si] += off[:, col[ln["m"]]]
    return nu


def _hemisphere(n_orient):
    theta, phi, _ = zcw_angles(int(n_orient))
    keep = theta <= np.pi / 2 + 1e-12
    theta, phi = theta[keep], phi[keep]
    b = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    return b, np.full(len(theta), 1.0 / len(theta))


def _resolve_larmor(model, larmor_Hz, B0_T):
    if larmor_Hz is None:
        if B0_T is None:
            raise ValueError("Give larmor_Hz or B0_T.")
        larmor_Hz = abs(float(find_isotope_info(model.nucleus)["gamma"])) * float(B0_T)
    larmor_Hz = float(larmor_Hz)
    if larmor_Hz <= 0:
        raise ValueError("larmor_Hz must be > 0.")
    return larmor_Hz


def _uniform_axis(ppm_axis):
    x = np.asarray(ppm_axis, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("ppm_axis must be a 1D uniformly spaced axis.")
    d = np.diff(x)
    if not np.allclose(d, d[0], rtol=1e-6, atol=abs(d[0]) * 1e-6):
        raise ValueError("ppm_axis must be uniformly spaced.")
    return x


def _gaussian_smooth(spec, ppm_axis, fwhm_ppm):
    if fwhm_ppm <= 0:
        return spec
    from scipy.ndimage import gaussian_filter1d
    sigma = fwhm_ppm / (2.0 * math.sqrt(2.0 * math.log(2.0))) / abs(ppm_axis[1] - ppm_axis[0])
    return gaussian_filter1d(spec, sigma, mode="constant")


def _validity_notes(model, lines, larmor_Hz, K, transitions, nu_all, order="first"):
    notes = []
    kmax = float(np.max(-np.diag(K))) if K.size else 0.0
    quad_sites = [x for x in model.sites if x.site.get("quadrupolar")]
    if quad_sites and model.spin > 0.5:
        half = not float(model.spin).is_integer()
        if order == "first" and half and any(l["group"] == "CT" for l in lines):
            notes.append("⚠ The central transition has no first-order quadrupolar shift and "
                         "this is a first-order simulation: it is drawn at its δiso "
                         "(use order='second' for the second-order lineshape).")
        if order == "second":
            notes.append("Second-order quadrupolar frequencies are taken state by state: valid for "
                         "motion slower than ν_Q. Much faster motion averages the EFG first "
                         "(averaged_spectrum(order='second')).")
        I = model.spin
        nu_q = max(abs(float(x.site.get("quadrupolar")[0]["C_Q"])) * 1e6 for x in quad_sites
                   ) * 3.0 / (2.0 * I * (2.0 * I - 1.0))
        if nu_q > 0 and kmax > 0.1 * nu_q:
            notes.append(f"⚠ The fastest exit rate ({kmax:.3g} s⁻¹) is above 10 % of the "
                         f"quadrupolar frequency scale ν_Q ({nu_q:.3g} Hz): motion this fast "
                         "mixes transitions, which this model (independent transitions) "
                         "does not describe.")
    if kmax > 0.01 * larmor_Hz:
        notes.append(f"⚠ The fastest exit rate ({kmax:.3g} s⁻¹) is not small against the "
                     f"Larmor frequency ({larmor_Hz:.3g} Hz): the secular (high-field) "
                     "treatment fails.")
    spread = float(np.max(nu_all.max(axis=1) - nu_all.min(axis=1))) if nu_all.size else 0.0
    notes.append(f"Regime: the fastest exit rate is {kmax:.3g} s⁻¹ against a frequency spread "
                 f"of up to {spread:.3g} Hz between states → "
                 f"{exchange_regime(kmax, spread)} on the NMR time scale.")
    return notes, spread, kmax


# ----------------------------------------------------------------------
# Time-domain engine: dynamics under MAS and echo sequences
# ----------------------------------------------------------------------
#
# The same independent-line Bloch-McConnell model as the static engine, solved in the
# time domain: with ``a`` the vector of the state amplitudes of one line,
#
#     da/dt = (i 2 pi nu_s(t) - R2_s + K) a ,      FID(t) = sum_s a_s(t) .
#
# Under MAS nu_s(t) is periodic with the rotor period. One period is cut into N
# steps; in each step the frequencies are replaced by their average over the step
# (so a motionless model is exact) and the propagator exp(L dt) is exact. The rotor
# phase of the crystallite (gamma) is a cyclic shift of the step sequence, so the
# gamma average costs nothing in frequencies: n_gamma starts are propagated. The
# FID of one rotor period is stored and the following periods are obtained by
# repeated multiplication with the one-period propagator; the powder sum is taken on
# the FID, which is then Fourier transformed once.
#
# An echo is the same propagation with an ideal pulse in the middle: ``hahn`` inverts
# the coherence order (a -> a*, same line), ``solid`` is the quadrupolar echo that
# swaps a transition with its mirror transition (m -> 1 - m, no conjugation).
# Acquisition starts at the echo top, 2 tau after the first pulse.

def _batched_expm(M):
    """Matrix exponential of a stack (..., S, S): scaling and squaring, Taylor series."""
    S = M.shape[-1]
    norm = float(np.max(np.sum(np.abs(M), axis=-1))) if M.size else 0.0
    s = int(math.ceil(math.log2(norm / 0.5))) if norm > 0.5 else 0
    A = M / (2.0 ** s)
    E = np.broadcast_to(np.eye(S, dtype=complex), M.shape).copy()
    term = E.copy()
    for k in range(1, 15):
        term = (term @ A) / k
        E += term
    for _ in range(s):
        E = E @ E
    return E


def _parse_echo(echo):
    if echo is None or echo is False:
        return None
    if isinstance(echo, (tuple, list)) and len(echo) == 2:
        kind, tau = echo
    elif isinstance(echo, dict):
        kind = echo.get("kind", "hahn")
        tau = echo.get("tau_s", None)
        if tau is None and echo.get("tau_us") is not None:
            tau = float(echo["tau_us"]) * 1e-6
    else:
        raise ValueError("Give the echo as {'kind': 'hahn' | 'solid', 'tau_s': seconds} "
                         "(or ('hahn', tau_s)).")
    kind = str(kind).lower()
    if kind not in ("hahn", "solid"):
        raise ValueError("echo kind must be 'hahn' (pi pulse) or 'solid' (quadrupolar echo).")
    if tau is None or not float(tau) > 0:
        raise ValueError("The echo delay tau_s must be > 0.")
    return {"kind": kind, "tau_s": float(tau)}


def _mirror_lines(lines):
    """Index of each line's mirror transition (m -> 1 - m, same dipolar signs)."""
    out = []
    for ln in lines:
        if ln["m"] is None:
            raise ValueError("The solid (quadrupolar) echo needs a quadrupolar nucleus.")
        j = [k for k, o in enumerate(lines)
             if o["m"] is not None and abs(o["m"] - (1.0 - ln["m"])) < 1e-9
             and o["signs"] == ln["signs"]]
        if not j:
            raise ValueError("The solid echo needs every transition and its mirror transition: "
                             "use transitions='all' (a quadrupolar echo is for I = 1).")
        out.append(j[0])
    return out


def _td_spectrum(model, x, larmor_Hz, K, p, data, lines, order, R2, mas_Hz, rotor_angle,
                 n_orient, n_gamma, echo, steps_per_period, notes, max_points=2 ** 19):
    """Spectrum (sum of the points = the amount in the window) from the time-domain
    engine, plus an info dict. See the notes above."""
    S = model.n_states
    L = len(lines)
    mas = float(mas_Hz or 0.0)
    f_axis = x * 1e-6 * larmor_Hz
    df_out = abs(f_axis[1] - f_axis[0])
    bh, _ = _hemisphere(2584)
    nu_h = _frequencies(data, bh, lines, larmor_Hz, order, model.spin)
    fmin, fmax = float(nu_h.min()), float(nu_h.max())
    fc = 0.5 * (fmin + fmax)
    sw_need = max(1.3 * (fmax - fmin) + 4.0 * mas, 16.0 * df_out)

    info = {}
    if mas > 0:
        Ng = int(n_gamma)
        N = int(steps_per_period) if steps_per_period else int(math.ceil(sw_need / mas))
        N = max(Ng, Ng * int(math.ceil(N / Ng)))
        if N > 2048:
            notes.append(f"⚠ The spectral range needs {N} steps per rotor period; limited to 2048: "
                         "far spinning sidebands may alias. Use a faster spinning speed or a "
                         "smaller axis.")
            N = 2048 - 2048 % Ng
        dt = 1.0 / (N * mas)
        os_ = 4
        beta = (math.acos(1.0 / math.sqrt(3.0)) if rotor_angle == "magic"
                else math.radians(float(rotor_angle)))
        theta, phi, wts = zcw_angles(int(n_orient))
        r, e1, e2 = _rotor_frame(theta, phi)
        w_or = wts / wts.sum()
        n = len(theta)
        phases = 2.0 * np.pi * (np.arange(N * os_) + 0.5) / (N * os_)
    else:
        Ng, N, os_ = 1, 1, 1
        bh2, w_or = _hemisphere(int(n_orient))
        n = len(w_or)
        dt = 1.0 / sw_need
    ech = _parse_echo(echo)
    n_tau = 0
    if ech is not None:
        if mas > 0:
            n_tau = max(1, int(round(ech["tau_s"] / dt)))
            if n_tau % N:
                notes.append(f"The echo delay is not a whole number of rotor periods "
                             f"({n_tau / N:.3g}): interactions are refocused only if the echo "
                             "is rotor-synchronised.")
        else:
            n_tau = max(1, int(math.ceil(ech["tau_s"] * sw_need)))
            dt = ech["tau_s"] / n_tau
        tau_actual = n_tau * dt
        if abs(tau_actual - ech["tau_s"]) > 1e-9 * max(ech["tau_s"], 1e-9) and mas > 0:
            notes.append(f"The echo delay was rounded to {tau_actual * 1e6:.4g} μs (a multiple of "
                         f"the time step {dt * 1e6:.4g} μs).")
        mirror = _mirror_lines(lines) if ech["kind"] == "solid" else list(range(L))
        if ech["kind"] == "solid" and (np.ptp(data["diso"]) > 0 or data["csa"] is not None):
            notes.append("The solid echo does not refocus shifts and shift anisotropies: they "
                         "keep evolving for 2τ and the real part is plotted as is.")
    else:
        mirror = list(range(L))
    sw = 1.0 / dt if mas == 0 else N * mas
    t_dec = math.log(1000.0) / max(float(np.min(R2)), 1e-9)
    n_per = int(math.ceil(t_dec / (N * dt)))
    if n_per * N > max_points:
        n_per = max(2, max_points // N)
        notes.append(f"⚠ The FID was limited to {n_per * N} points ({n_per * N * dt * 1e3:.3g} ms): "
                     "the narrowest line may be truncated (wiggles). Use a coarser axis or "
                     "broaden the lines.")
    A = K.astype(complex) - np.diag(R2)
    idx = np.arange(S)
    gsteps = np.arange(Ng) * (N // Ng)
    fid = np.zeros((n_per, N), dtype=complex)
    chunk = max(1, int(2.5e6 // (N * S * (S * L + Ng))))
    with np.errstate(all="ignore"):
        for start in range(0, n, chunk):
            sl = slice(start, min(n, start + chunk))
            if mas > 0:
                bb = (math.cos(beta) * r[sl, None, :]
                      + math.sin(beta) * (np.cos(phases)[None, :, None] * e1[sl, None, :]
                                          + np.sin(phases)[None, :, None] * e2[sl, None, :]))
                c = bb.shape[0]
                nu = _frequencies(data, bb.reshape(-1, 3), lines, larmor_Hz, order, model.spin)
                wbar = nu.reshape(L, S, c, N, os_).mean(axis=-1)
            else:
                bb = bh2[sl]
                c = bb.shape[0]
                wbar = _frequencies(data, bb, lines, larmor_Hz, order, model.spin)[..., None]
            wbar = wbar - fc
            w_c = w_or[sl]
            P = []
            for li in range(L):
                Mx = np.broadcast_to(A, (c, N, S, S)).copy()
                Mx[..., idx, idx] += 2j * np.pi * wbar[li].transpose(1, 2, 0)
                P.append(_batched_expm(Mx * dt))
            for li in range(L):
                lb, la = li, mirror[li]
                x0 = np.broadcast_to((p * lines[li]["weight"]).astype(complex), (c, Ng, S)).copy()
                x0 *= (w_c / Ng)[:, None, None]
                j0 = gsteps
                if n_tau:
                    for i in range(n_tau):
                        Pm = P[lb][:, (j0 + i) % N]
                        x0 = np.einsum("cgij,cgj->cgi", Pm, x0)
                    if ech["kind"] == "hahn":
                        x0 = np.conj(x0)
                    for i in range(n_tau):
                        Pm = P[la][:, (j0 + n_tau + i) % N]
                        x0 = np.einsum("cgij,cgj->cgi", Pm, x0)
                    j0 = (j0 + 2 * n_tau) % N
                Phi = np.broadcast_to(np.eye(S, dtype=complex), (c, Ng, S, S)).copy()
                W = np.empty((c, Ng, N, S), dtype=complex)
                for jj in range(N):
                    W[:, :, jj, :] = Phi.sum(axis=2)
                    Phi = np.einsum("cgij,cgjk->cgik", P[la][:, (j0 + jj) % N], Phi)
                U = Phi
                xv = x0
                for ip in range(n_per):
                    fid[ip] += np.einsum("cgjs,cgs->j", W, xv)
                    xv = np.einsum("cgij,cgj->cgi", U, xv)
    s = fid.reshape(-1)
    M = len(s)
    nfft = 1 << int(math.ceil(math.log2(max(2 * M, 4.0 * sw / df_out))))
    nfft = min(nfft, 1 << 22)
    F = np.fft.fft(s, nfft)
    sp = 2.0 * np.real(dt * (F - s[0] / 2.0))
    fr = np.fft.fftshift(np.fft.fftfreq(nfft, dt))
    spec = np.interp(f_axis - fc, fr, np.fft.fftshift(sp), left=0.0, right=0.0)
    spec = spec * df_out * data["amount"]
    info = {"steps_per_period": N, "dt_s": dt, "n_periods": n_per, "fid_points": M,
            "spectral_width_Hz": sw, "n_orient": int(n), "n_gamma": int(Ng),
            "carrier_Hz": fc, "tail": float(abs(s[-1]) / max(abs(s[0]), 1e-300))}
    if ech is not None:
        info["echo"] = {"kind": ech["kind"], "tau_s": n_tau * dt,
                        "intensity": float(np.real(s[0]))}
    return spec, info


# ----------------------------------------------------------------------
# Dynamic static spectrum
# ----------------------------------------------------------------------

def motion_spectrum(model, ppm_axis, larmor_Hz=None, B0_T=None, T_K=None, mas_Hz=0.0,
                    transitions="all", k_modes=None, hop_scale=1.0, n_orient=None,
                    gauss_fwhm_ppm=0.0, min_fwhm_points=2.0, order="first", echo=None,
                    rotor_angle="magic", n_gamma=16, steps_per_period=None):
    """Powder lineshape of ``model`` at temperature ``T_K`` (or the given
    rates), with exact Bloch-McConnell dynamics: static, under MAS (``mas_Hz`` > 0,
    rates comparable to the spinning speed included) and/or after an echo.

    Parameters
    ----------
    model : :class:`MotionModel`.
    ppm_axis : uniformly spaced ppm axis (either direction).
    larmor_Hz or B0_T : observe frequency or field.
    T_K : temperature for Arrhenius rates and barrier-based populations.
    k_modes : ``{mode index or label: per-jump rate}`` replacing a mode's own rate.
    hop_scale : multiplies every hop rate (both directions: detailed
        balance is kept).
    transitions : 'all' | 'ct' | 'st' (quadrupolar nuclei).
    n_orient : ZCW density (static: the hemisphere is used, B0 and -B0 are
        equivalent; default 4181). Under MAS the default is 987 rotor-axis directions
        (the rotor phase is averaged separately by ``n_gamma`` starts).
    mas_Hz, rotor_angle : spinning speed and rotor angle ('magic' or degrees). Dynamic MAS
        (more than one state) and any echo use the time-domain engine (see the notes
        above ``_batched_expm``): exact exchange within each time step, the
        frequencies averaged over the step; a model with a single state under MAS keeps
        the exact sideband engine.
    echo : ``{'kind': 'hahn' | 'solid', 'tau_s': seconds}``. ``hahn`` is an ideal pi pulse
        (refocuses every frequency term, loses what the motion destroys); ``solid`` is the
        quadrupolar echo (I = 1: refocuses the first-order quadrupolar coupling, not the
        shifts). The spectrum is that of the echo from its top (2 tau after the first
        pulse); ``result['td']['echo']['intensity']`` is the echo amplitude as a fraction
        of the initial one. Under MAS give tau as a whole number of rotor periods.
    n_gamma, steps_per_period : rotor phases averaged (default 16) and time steps per
        rotor period (default: enough that no sideband aliases; a multiple of n_gamma).
    gauss_fwhm_ppm : extra Gaussian broadening, ppm.
    order : quadrupolar level of theory, 'first' or 'second' (state by state: see
        the module notes).
    min_fwhm_points : the lines are evaluated exactly on the axis, so a line
        narrower than the point spacing would be missed (area lost). Every
        state's Lorentzian width is therefore at least this many points (and the
        log says so); use a finer axis for narrower lines.

    Returns a dict: ``x_ppm``, ``real_spec`` (the sum of the points equals
    the total amount, as in Simulate), ``normalised_real_spec``,
    ``populations``, ``states``, ``K``, ``regime``, ``log``, ``notes``,
    ``settings``.
    """
    mas = float(mas_Hz or 0.0)
    if mas < 0:
        raise ValueError("mas_Hz must be >= 0.")
    echo_spec = _parse_echo(echo)
    if mas > 0 and model.n_states == 1 and echo_spec is None:
        return averaged_spectrum(model, ppm_axis, larmor_Hz, B0_T, T_K, mas_Hz=mas,
                                 rotor_angle=rotor_angle, transitions=transitions,
                                 k_modes=k_modes, hop_scale=hop_scale,
                                 gauss_fwhm_ppm=gauss_fwhm_ppm, order=order)
    use_td = mas > 0 or echo_spec is not None
    larmor_Hz = _resolve_larmor(model, larmor_Hz, B0_T)
    x = _uniform_axis(ppm_axis)
    f = x * 1e-6 * larmor_Hz
    omega = 2.0 * np.pi * f
    df = abs(f[1] - f[0])

    K = model.rate_matrix(T_K, k_modes, hop_scale)
    p = model.populations(T_K, k_modes, hop_scale)
    data = model.state_data()
    lines = model.lines(transitions)
    S = model.n_states
    n_given = n_orient
    n_orient = 4181 if n_orient is None else n_orient
    b, _w = _hemisphere(n_orient)
    n = len(b)
    nu = _frequencies(data, b, lines, larmor_Hz, order, model.spin)  # (L, S, n)

    lorentz = model.sites[0].site.lineshape == "lorentzian"
    fwhm_ppm = data["fwhm_ppm"]
    R2 = math.pi * fwhm_ppm * 1e-6 * larmor_Hz if lorentz else np.zeros(S)
    gauss_ppm = 0.0 if lorentz else float(p @ fwhm_ppm)
    notes, spread, kmax = _validity_notes(model, lines, larmor_Hz, K, transitions,
                                          nu.reshape(-1, n), order)
    floor = math.pi * float(min_fwhm_points) * df                      # R2 = pi * FWHM(Hz)
    if np.any(R2 < floor):
        notes.append(f"Lines narrower than {min_fwhm_points:g} points were widened to "
                     f"{min_fwhm_points * df:.3g} Hz FWHM (the axis step is {df:.3g} Hz); use a "
                     "finer ppm axis for narrower lines.")
        R2 = np.maximum(R2, floor)
    fwhms = [s.site.fwhm for s in model.sites]
    if not lorentz and (max(fwhms) - min(fwhms)) > 0.01 * max(max(fwhms), 1e-12):
        notes.append("Gaussian environments with different widths: the population-weighted mean "
                     f"({gauss_ppm:.3g} ppm) is applied to the whole spectrum.")

    chunk = max(1, int(3.0e6 // (S * len(x))))
    spec = np.zeros(len(x))
    eye_idx = np.arange(S)
    td = None
    if use_td:
        n_td = 987 if (mas > 0 and n_given is None) else (n_orient if mas > 0 else int(n_orient))
        spec, td = _td_spectrum(model, x, larmor_Hz, K, p, data, lines, order, R2, mas,
                                rotor_angle, n_td, n_gamma, echo_spec, steps_per_period, notes)
    with np.errstate(all="ignore"):
        for li, ln in enumerate([] if use_td else lines):
            for start in range(0, n, chunk):
                nuc = nu[li][:, start:start + chunk].T                   # (c, S)
                c = nuc.shape[0]
                L = np.zeros((c, S, S), dtype=complex)
                L += K
                L[:, eye_idx, eye_idx] += 2j * np.pi * nuc - R2
                lam, V = np.linalg.eig(L)
                right = np.linalg.solve(V, np.broadcast_to(p, (c, S))[..., None].astype(complex))[..., 0]
                coef = V.sum(axis=1) * right                             # (c, S)
                resp = 1.0 / (1j * omega[None, None, :] - lam[:, :, None])
                spec += ln["weight"] / n * np.real(np.einsum("ck,ckf->f", coef, resp))
    if not np.all(np.isfinite(spec)):
        raise FloatingPointError("The lineshape came out non-finite (a defective rate matrix?). "
                                 "Change a rate slightly or check the model.")
    if not use_td:
        spec = spec * 2.0 * df * data["amount"]                          # sum of points = amount
    spec = _gaussian_smooth(spec, x, np.hypot(gauss_ppm, gauss_fwhm_ppm))

    width_Hz = (np.mean(R2) + kmax) / math.pi + np.hypot(gauss_ppm, gauss_fwhm_ppm) * 1e-6 * larmor_Hz
    if not use_td and spread > 0 and spread / math.sqrt(n) > width_Hz:
        notes.append(f"⚠ The powder average may look grainy: neighbouring orientations differ by "
                     f"~{spread / math.sqrt(n):.3g} Hz but the lines are only ~{width_Hz:.3g} Hz "
                     "wide. Raise n_orient or add Gaussian broadening.")
    peak = float(np.max(spec)) if spec.size else 0.0
    temp = "T = %g K" % T_K if T_K is not None else "rates as given"
    if td is None:
        head = (f"static, ν0 = {larmor_Hz / 1e6:.7g} MHz, {len(lines)} line(s), {n} orientations "
                f"(hemisphere of a ZCW set of {int(n_orient)}), " + temp + ".")
    else:
        what = (f"MAS {mas / 1e3:g} kHz" if mas > 0 else "static")
        if echo_spec:
            what += f", {echo_spec['kind']} echo (τ = {td['echo']['tau_s'] * 1e6:.4g} μs; echo " \
                    f"amplitude {td['echo']['intensity']:.3g} of the initial)"
        head = (f"time domain, {what}, ν0 = {larmor_Hz / 1e6:.7g} MHz, {len(lines)} line(s), "
                f"{td['n_orient']} orientations × {td['n_gamma']} rotor phases, "
                f"{td['steps_per_period']} steps/period (Δt = {td['dt_s'] * 1e6:.4g} μs), "
                f"{td['fid_points']} FID points, " + temp + ".")
        if td["tail"] > 5e-3:
            notes.append(f"⚠ The FID is still at {td['tail']:.2g} of its start at the end: "
                         "the spectrum may show truncation wiggles.")
    log = model.describe() + [head] + notes
    return {"x_ppm": x, "real_spec": spec, "normalised_real_spec": spec / peak if peak > 0 else spec,
            "populations": p, "states": model.state_table(), "K": K, "regime":
            exchange_regime(kmax, spread), "log": "\n".join(log), "notes": notes,
            "settings": {"larmor_Hz": larmor_Hz, "T_K": T_K, "mas_Hz": mas,
                         "rotor_angle": rotor_angle, "echo": echo_spec,
                         "transitions": transitions, "order": order, "n_orient": int(n_orient),
                         "k_modes": dict(k_modes or {}), "hop_scale": hop_scale,
                         "gauss_fwhm_ppm": gauss_fwhm_ppm},
            "td": td, "model": model.to_dict()}


def motion_rate_series(model, rates, which="mode:0", ppm_axis=None, include_limits=True,
                       **kwargs):
    """:func:`motion_spectrum` over a list of ``rates``. ``which``:
    ``'mode:<index or label>'`` (that mode's per-jump rate, s^-1),
    ``'hops'`` (``hop_scale`` -- a multiplier on the hop table),
    ``'all'`` (every mode's rate set to the rate; hops left alone) or
    ``'T'`` (the "rates" are temperatures in K: Arrhenius rates and barrier-based
    populations are evaluated at each).

    With ``include_limits`` the slow-limit (population-weighted static
    patterns) and fast-limit (motionally averaged tensor) spectra are
    returned beside them, so the series can be read against the two ends.
    """
    rates = [float(r) for r in rates]
    if not rates:
        raise ValueError("Give at least one rate.")
    if ppm_axis is None:
        raise ValueError("Give a ppm_axis (all spectra share it).")
    out = []
    for r in rates:
        kw = dict(kwargs)
        if which == "T":
            kw["T_K"] = r
        elif which == "hops":
            kw["hop_scale"] = r
        elif which == "all":
            kw["k_modes"] = {i: r for i in range(len(model.modes))}
        elif str(which).startswith("mode:"):
            key = str(which)[5:]
            kw["k_modes"] = {(int(key) if key.lstrip("-").isdigit() else key): r}
        else:
            raise ValueError("which must be 'mode:<index or label>', 'hops', 'all' or 'T'.")
        res = motion_spectrum(model, ppm_axis, **kw)
        res["rate"] = r
        out.append(res)
    result = {"x_ppm": out[0]["x_ppm"], "rates": rates, "which": which,
              "spectra": [r["real_spec"] for r in out], "results": out}
    if include_limits:
        lim_kw = {k: v for k, v in kwargs.items()
                  if k in ("larmor_Hz", "B0_T", "T_K", "transitions", "gauss_fwhm_ppm", "mas_Hz",
                           "order", "rotor_angle")}
        if which == "T":
            lim_kw.pop("T_K", None)
            lim_kw["T_K"] = rates[0]
            result["limits_at_T_K"] = rates[0]
        result["limits"] = limit_spectra(model, ppm_axis, **lim_kw)
    return result


# ----------------------------------------------------------------------
# Single-state engine (static and MAS) and the limits
# ----------------------------------------------------------------------

def _select_state(data, s):
    return {"diso": data["diso"][s:s + 1], "csa": None if data["csa"] is None else data["csa"][s:s + 1],
            "quad": None if data["quad"] is None else data["quad"][s:s + 1],
            "dip": [d[s:s + 1] for d in data["dip"]], "fwhm_ppm": data["fwhm_ppm"][s:s + 1],
            "amount": data["amount"], "quad_params": data["quad_params"][s:s + 1]}


def _single_state_pattern(nu_fn, lines, ppm_axis, larmor_Hz, mas_Hz=0.0, rotor_angle="magic",
                          n_orient=None, n_gamma=16, n_time=None, max_n_time=65536):
    """Unbroadened (stick) pattern, summed over the lines, amount 1, of the
    frequencies ``nu_fn(b)`` -> (L, n) for B0 directions ``b`` (n, 3);
    static, or under MAS (a ZCW set of rotor axes, ``n_gamma`` rotor phases)."""
    out = np.zeros(len(ppm_axis))
    if not mas_Hz:
        b, w = _hemisphere(n_orient or 46368)
        nu = nu_fn(b)                                                   # (L, n)
        for li, ln in enumerate(lines):
            out += _bin_sticks(ppm_axis, nu[li] / larmor_Hz * 1e6, ln["weight"] * w)
        return out
    beta = (math.acos(1.0 / math.sqrt(3.0)) if rotor_angle == "magic"
            else math.radians(float(rotor_angle)))
    theta, phi, wts = zcw_angles(int(n_orient or 4181))
    r, e1, e2 = _rotor_frame(theta, phi)
    gam = 2.0 * np.pi * np.arange(n_gamma) / n_gamma
    b = (np.cos(beta) * r[:, None, :] + np.sin(beta) * (np.cos(gam)[None, :, None] * e1[:, None, :]
                                                       + np.sin(gam)[None, :, None] * e2[:, None, :]))
    N = len(theta)
    nu = nu_fn(b.reshape(-1, 3)).reshape(len(lines), N, n_gamma).transpose(1, 2, 0)   # (N, G, L)
    inten = np.broadcast_to(np.array([l["weight"] for l in lines]), (N, len(lines)))
    prep = _mas_prepare(nu, mas_Hz, n_gamma, n_time, max_n_time)
    for _a, _b, centre, sb_Hz, weight in _mas_sticks(prep, inten, wts):
        position = (centre[:, None, :] + sb_Hz[None, :, None]) / larmor_Hz * 1e6
        for li in range(len(lines)):
            out += _bin_sticks(ppm_axis, position[:, :, li], weight[:, :, li])
    return out


def _broaden(pattern, ppm_axis, fwhm_ppm, lineshape, eta, gauss_fwhm_ppm):
    spec = pattern
    if fwhm_ppm > 0:
        spec = _apply_broadening(spec, ppm_axis, fwhm=fwhm_ppm, lineshape=lineshape, eta=eta)
    return _gaussian_smooth(spec, ppm_axis, gauss_fwhm_ppm)


def _efg_parameters(Q):
    """``(C_Q, eta_Q, E)`` of a symmetric traceless EFG tensor in Hz: the
    signed C_Q (the largest principal value), eta_Q and the right-handed
    PAS matrix (columns x, y, z in the tensor's own frame)."""
    (vx, vy, vz), E = _principal_axes(Q, "efg")
    eta = float(min(max((vx - vy) / vz, 0.0), 1.0)) if abs(vz) > 1e-9 else 0.0
    return float(vz), eta, E


def _average_data(data, p):
    """The population-weighted average of every tensor: one state."""
    def avg(T):
        return None if T is None else np.einsum("s,sij->ij", p, T)[None]
    quad = avg(data["quad"])
    return {"diso": np.array([float(p @ data["diso"])]), "csa": avg(data["csa"]),
            "quad": quad, "dip": [avg(T) for T in data["dip"]],
            "fwhm_ppm": np.array([float(p @ data["fwhm_ppm"])]), "amount": data["amount"],
            "quad_params": [None if quad is None else _efg_parameters(quad[0])]}


def averaged_tensors(model, T_K=None, k_modes=None, hop_scale=1.0):
    """The motionally averaged tensors (fast limit): the population-
    weighted average of every state's tensor, diagonalised.

    Returns a dict: ``delta_iso`` (ppm), ``csa`` (``delta_aniso`` = Haeberlen Δ
    ppm, ``eta``, ``span``, ``skew`` -- Herzfeld-Berger/Mason --,
    ``euler_mol``, ``reduction``), ``quad`` (``C_Q`` Hz signed, ``eta_Q``,
    ``euler_mol``, ``reduction``), ``dipolar`` (list: ``D_Hz``, ``eta``,
    ``euler_mol``, ``reduction``), ``euler_csa_in_efg``, ``populations``,
    ``log``. ``reduction`` is the averaged anisotropy over the population-
    weighted mean of the states' static anisotropies (None when that is zero).
    """
    data = model.state_data()
    p = model.populations(T_K, k_modes, hop_scale)
    avg = _average_data(data, p)
    out = {"delta_iso": float(avg["diso"][0]), "populations": p, "csa": None, "quad": None,
           "dipolar": [], "euler_csa_in_efg": (0.0, 0.0, 0.0)}
    log = [f"Motionally averaged tensors: {model.n_states} states, "
           f"populations from {'detailed balance' if model.hops else 'equal weights'}."]
    log.append(f"δiso = {out['delta_iso']:.5g} ppm (population-weighted).")

    def reduction(avg_value, static_values):
        stat = float(np.sum(p * np.abs(static_values)))
        return float(abs(avg_value) / stat) if stat > 1e-12 else None

    E_q = None
    if avg["quad"] is not None:
        (vx, vy, vz), E_q = _principal_axes(avg["quad"][0], "efg")
        C = float(vz)
        eta = float(min(max((vx - vy) / vz, 0.0), 1.0)) if abs(vz) > 1e-9 else 0.0
        static_C = np.array([np.linalg.eigvalsh(q)[np.argmax(np.abs(np.linalg.eigvalsh(q)))]
                             for q in data["quad"]])
        out["quad"] = {"C_Q": C, "eta_Q": eta, "euler_mol": euler_from_rotation(E_q),
                       "reduction": reduction(C, static_C)}
        log.append(f"Quadrupolar: C_Q = {C / 1e6:.5g} MHz, η_Q = {eta:.4g}"
                   + (f" ({out['quad']['reduction']:.3g} of the static value)"
                      if out["quad"]["reduction"] is not None else "") + ".")
    if avg["csa"] is not None:
        (dx, dy, dz), E_c = _principal_axes(avg["csa"][0], "csa")
        eta = float(min(max((dy - dx) / dz, 0.0), 1.0)) if abs(dz) > 1e-9 else 0.0
        static_dz = np.array([_principal_axes(c, "csa")[0][2] for c in data["csa"]])
        entry = {"delta_aniso": float(dz), "eta": eta, "euler_mol": euler_from_rotation(E_c),
                 "reduction": reduction(dz, static_dz)}
        if abs(dz) > 1e-9:
            _iso, span, skew = convert_csa_parameters(0.0, dz, eta, "Haeberlen", "Herzfeld-Berger")
            entry.update({"span": float(span), "skew": float(skew)})
        else:
            entry.update({"span": 0.0, "skew": 0.0})
        out["csa"] = entry
        log.append(f"CSA: Δ = {dz:.5g} ppm, η = {eta:.4g} (Ω = {entry['span']:.5g} ppm, "
                   f"κ = {entry['skew']:.4g})"
                   + (f" ({entry['reduction']:.3g} of the static value)"
                      if entry["reduction"] is not None else "") + ".")
        if E_q is not None:
            rel = E_q.T @ E_c
            out["euler_csa_in_efg"] = euler_from_rotation(rel)
    for k, Td in enumerate(avg["dip"]):
        (dx, dy, dz), E_d = _principal_axes(Td[0], "csa")
        eta = float(min(max((dy - dx) / dz, 0.0), 1.0)) if abs(dz) > 1e-9 else 0.0
        static_dz = np.array([_principal_axes(t, "csa")[0][2] if np.any(t) else 0.0
                              for t in data["dip"][k]])
        out["dipolar"].append({"D_Hz": float(dz), "eta": eta, "euler_mol": euler_from_rotation(E_d),
                               "reduction": reduction(dz, static_dz)})
        log.append(f"Dipolar {k + 1}: D = {dz / 1e3:.5g} kHz, η = {eta:.4g}"
                   + (f" ({out['dipolar'][-1]['reduction']:.3g} of the static value)"
                      if out["dipolar"][-1]["reduction"] is not None else "") + ".")
    if len(model.sites) > 1 and not model.hops:
        log.append("Several environments without hops: the average assumes they interconvert "
                   "(equal populations).")
    out["log"] = "\n".join(log)
    return out


def effective_site(model, T_K=None, k_modes=None, hop_scale=1.0, label="motionally averaged"):
    """A Simulate :class:`Site` carrying the averaged shift, CSA and quadrupolar
    coupling -- ready for ``simulate_combined`` (static or MAS), or to send to
    Fit as starting values. Dipolar couplings cannot be held by a Site: they
    are in :func:`averaged_tensors` and in :func:`averaged_spectrum`."""
    eff = averaged_tensors(model, T_K, k_modes, hop_scale)
    first = model.sites[0].site
    site = Site(model.nucleus, delta_iso=eff["delta_iso"], label=label,
                amount=float(sum(s.site.amount for s in model.sites)),
                fwhm=float(model.populations(T_K, k_modes, hop_scale)
                           @ model.state_data()["fwhm_ppm"]),
                lineshape=first.lineshape, eta=first.eta)
    if eff["quad"] is not None and abs(eff["quad"]["C_Q"]) > 1e-6:
        site.add_quadrupolar(eff["quad"]["C_Q"] / 1e6, eff["quad"]["eta_Q"])
    if eff["csa"] is not None and abs(eff["csa"]["delta_aniso"]) > 1e-9:
        site.add_csa(eff["csa"]["delta_aniso"], eff["csa"]["eta"], "Haeberlen",
                     euler=eff["euler_csa_in_efg"] if site.get("quadrupolar") else (0.0, 0.0, 0.0))
    if any(abs(d["D_Hz"]) > 1e-9 for d in eff["dipolar"]):
        warnings.warn("Dipolar couplings cannot be carried by a Simulate site; use "
                      "averaged_spectrum() for the full fast-limit spectrum.", UserWarning,
                      stacklevel=2)
    return site


def averaged_spectrum(model, ppm_axis, larmor_Hz=None, B0_T=None, T_K=None, mas_Hz=0.0,
                      rotor_angle="magic", transitions="all", k_modes=None, hop_scale=1.0,
                      gauss_fwhm_ppm=0.0, n_orient=None, order="first"):
    """The spectrum of the motionally averaged tensors, static (``mas_Hz`` = 0) or
    under MAS, with the dipolar couplings and the quadrupolar coupling at
    first or second ``order``. The tensors -- the EFG included -- are
    averaged *before* the second-order shift is computed, which is the limit of
    motion much faster than nu_Q (:func:`limit_spectra` also returns the
    frequency average, the limit for motion slower than nu_Q)."""
    larmor_Hz = _resolve_larmor(model, larmor_Hz, B0_T)
    x = _uniform_axis(ppm_axis)
    data = model.state_data()
    p = model.populations(T_K, k_modes, hop_scale)
    avg = _average_data(data, p)
    lines = model.lines(transitions)
    notes = []
    if order == "first" and any(l["group"] == "CT" for l in lines) and avg["quad"] is not None \
            and model.spin > 0.5 and not float(model.spin).is_integer():
        notes.append("⚠ First-order quadrupolar only: the central transition has no first-order "
                     "shift and is drawn at δiso (use order='second').")
    if order == "second" and avg["quad"] is not None:
        notes.append("Second-order shifts are computed from the averaged EFG: valid for motion "
                     "much faster than ν_Q.")
    pattern = _single_state_pattern(
        lambda b: _frequencies(avg, b, lines, larmor_Hz, order, model.spin)[:, 0, :], lines, x,
        larmor_Hz, float(mas_Hz or 0.0), rotor_angle, n_orient)
    first = model.sites[0].site
    spec = data["amount"] * _broaden(pattern, x, float(avg["fwhm_ppm"][0]), first.lineshape,
                                     first.eta, gauss_fwhm_ppm)
    peak = float(np.max(spec)) if spec.size else 0.0
    eff = averaged_tensors(model, T_K, k_modes, hop_scale)
    regime = ("static" if not mas_Hz else f"MAS {float(mas_Hz) / 1e3:g} kHz")
    log = [eff["log"], f"Averaged-tensor spectrum, {regime}, ν0 = {larmor_Hz / 1e6:.7g} MHz, "
           f"quadrupolar {order} order."] + notes
    return {"x_ppm": x, "real_spec": spec, "normalised_real_spec": spec / peak if peak > 0 else spec,
            "effective": eff, "populations": p, "log": "\n".join(log), "notes": notes,
            "settings": {"larmor_Hz": larmor_Hz, "T_K": T_K, "mas_Hz": float(mas_Hz or 0.0),
                         "rotor_angle": rotor_angle, "transitions": transitions, "order": order}}


def limit_spectra(model, ppm_axis, larmor_Hz=None, B0_T=None, T_K=None, mas_Hz=0.0,
                  rotor_angle="magic", transitions="all", k_modes=None, hop_scale=1.0,
                  gauss_fwhm_ppm=0.0, n_orient=None, order="first"):
    """The ends of any rate series, static or under MAS:

    * ``static`` (rate → 0): the population-weighted sum of every state's own pattern;
    * ``fast`` (rate → ∞ of :func:`motion_spectrum`): each orientation's frequency is
      the population-weighted mean of the states' frequencies;
    * ``fast_tensor``: the tensors averaged first (:func:`averaged_spectrum`) -- the
      same as ``fast`` at first order, and the limit of motion much faster than
      nu_Q at second order, where the two differ.
    """
    larmor_Hz = _resolve_larmor(model, larmor_Hz, B0_T)
    x = _uniform_axis(ppm_axis)
    data = model.state_data()
    p = model.populations(T_K, k_modes, hop_scale)
    lines = model.lines(transitions)
    first = model.sites[0].site
    mas = float(mas_Hz or 0.0)
    static = np.zeros(len(x))
    for s in range(model.n_states):
        if p[s] <= 0:
            continue
        d = _select_state(data, s)
        pat = _single_state_pattern(
            lambda b, d=d: _frequencies(d, b, lines, larmor_Hz, order, model.spin)[:, 0, :],
            lines, x, larmor_Hz, mas, rotor_angle, n_orient)
        static += p[s] * _broaden(pat, x, float(d["fwhm_ppm"][0]), first.lineshape, first.eta,
                                  gauss_fwhm_ppm)
    static *= data["amount"]
    pat = _single_state_pattern(
        lambda b: np.einsum("s,lsn->ln", p, _frequencies(data, b, lines, larmor_Hz, order,
                                                          model.spin)),
        lines, x, larmor_Hz, mas, rotor_angle, n_orient)
    fast = data["amount"] * _broaden(pat, x, float(p @ data["fwhm_ppm"]), first.lineshape,
                                     first.eta, gauss_fwhm_ppm)
    tensor = averaged_spectrum(model, x, larmor_Hz, None, T_K, mas, rotor_angle, transitions,
                               k_modes, hop_scale, gauss_fwhm_ppm, n_orient, order)["real_spec"]
    return {"x_ppm": x, "static": static, "fast": fast, "fast_tensor": tensor}


# ----------------------------------------------------------------------
# Plot and notebook code
# ----------------------------------------------------------------------

def _nucleus_math(nucleus):
    """``"2H"`` -> ``"$^{2}$H"`` (mass number as a superscript, element upright)."""
    text = str(nucleus or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    letters = "".join(ch for ch in text if ch.isalpha())
    return f"$^{{{digits}}}$" + letters if digits and letters else text


def delta_label(nucleus=None):
    """Shift-axis label with an italic delta and the nucleus' mass number as a superscript."""
    return f"$\\delta$({_nucleus_math(nucleus)}) (ppm)" if nucleus else r"$\delta$ (ppm)"


def format_rate_label(k, unit=True):
    """``1e5`` -> ``$1\\times10^{5}$ s$^{-1}$`` (a coefficient only when it is not 1)."""
    k = float(k)
    if k == 0:
        return "0" + (" s$^{-1}$" if unit else "")
    mant, exp = f"{k:.2e}".split("e")
    exp = int(exp)
    mant = f"{float(mant):.3g}"
    if -2 <= exp <= 2:
        body = f"{k:.3g}"
    elif mant == "1":
        body = f"$10^{{{exp}}}$"
    else:
        body = f"${mant}\\times10^{{{exp}}}$"
    return body + (" s$^{-1}$" if unit else "")


def plot_motion_series(series, ax=None, offset=1.1, colours=None, label_fmt=None,
                       show_limits=True, nucleus=None, xlim=None, linewidth=1.0,
                       text_size="small"):
    """The spectra of :func:`motion_rate_series` stacked (slowest at the bottom),
    with the slow and fast limits as dashed traces beneath and above.
    Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    from ..plotting.accessibility import default_series_colours, series_colours

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.375, 4.5))
    else:
        fig = ax.figure
    rates = list(series["rates"])
    spectra = list(series["spectra"])
    x = series["x_ppm"]
    if label_fmt is None:
        label_fmt = {"T": "{k:.0f} K", "hops": "$\\times${k:.3g}"}.get(series.get("which"), None)
    n = len(spectra)
    if colours is None:
        colours = default_series_colours(n, ordered=True)
    elif isinstance(colours, str):
        colours = series_colours(n, colours)
    order = np.argsort(rates)
    shift = 1 if (show_limits and "limits" in series) else 0
    scale = max(float(np.max(np.abs(s))) for s in spectra) or 1.0
    if shift:
        lim = series["limits"]
        ln0, = ax.plot(x, lim["static"] / scale, "--", color="0.45", lw=0.8)
        ln0._opennmr_baseline = 0.0
    for row, i in enumerate(order, start=shift):
        ln, = ax.plot(x, spectra[i] / scale + row * offset, "-", color=colours[i % len(colours)],
                      lw=linewidth)
        ln._opennmr_baseline = row * offset
        ax.annotate(format_rate_label(rates[i]) if label_fmt is None else label_fmt.format(k=rates[i]), xy=(1.0, row * offset + 0.05),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=text_size, color=colours[i % len(colours)])
    if shift:
        ln1, = ax.plot(x, lim["fast"] / scale + (n + shift) * offset, "--", color="0.45", lw=0.8)
        ln1._opennmr_baseline = (n + shift) * offset
    ax.set_xlim(xlim if xlim is not None else (float(np.max(x)), float(np.min(x))))
    ax.set_xlabel(delta_label(nucleus))
    ax.yaxis.set_visible(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------
# Building models from plain data, and overlay on an experimental spectrum
# ----------------------------------------------------------------------

_MODE_ARGS = {
    "ring_jump": ("n", "axis", "model", "step_deg"),
    "flip": ("angle_deg", "axis"),
    "ring_diffusion": ("axis", "D_Hz", "n"),
    "random_reorientation": ("group",),
}


def mode_from_spec(spec):
    """A :class:`Mode` from a plain dict -- what a table (the GUI) or a saved file
    holds: ``{'kind': 'ring_jump' | 'flip' | 'ring_diffusion' | 'random_reorientation',
    'n' / 'angle_deg' / 'group': ..., 'axis': 'z' | 'x,y,z' | (polar, azimuth),
    'k_Hz': ... or 'Ea_eV': ..., 'nu0_Hz': ..., 'label': ...}``. A :class:`Mode` is
    returned unchanged."""
    if isinstance(spec, Mode):
        return spec
    spec = dict(spec)
    kind = spec.get("kind")
    if kind not in _MODE_ARGS:
        raise ValueError(f"Unknown mode kind {kind!r}: use one of {sorted(_MODE_ARGS)}.")
    kw = {k: spec[k] for k in _MODE_ARGS[kind] if spec.get(k) not in (None, "")}
    if isinstance(kw.get("axis"), str) and "," in kw["axis"]:
        parts = [float(v) for v in kw["axis"].replace(";", ",").split(",") if v.strip()]
        kw["axis"] = tuple(parts)
    for key in ("k_Hz", "Ea_eV", "nu0_Hz"):
        if spec.get(key) not in (None, ""):
            kw[key] = float(spec[key])
    if kind == "ring_diffusion":
        if "k_Hz" in kw:
            kw["D_Hz"] = kw.pop("k_Hz")
        if "n" in kw:
            kw["n"] = int(kw["n"])
    if kind == "ring_jump" and "n" in kw:
        kw["n"] = int(kw["n"])
    if kind == "ring_jump" and "step_deg" in kw:
        kw["step_deg"] = float(kw["step_deg"])
    if kind == "flip" and "angle_deg" in kw:
        kw["angle_deg"] = float(kw["angle_deg"])
    if spec.get("label"):
        kw["label"] = str(spec["label"])
    return {"ring_jump": ring_jump, "flip": flip, "ring_diffusion": ring_diffusion,
            "random_reorientation": random_reorientation}[kind](**kw)


def model_from_spec(sites, modes=(), hops=None, couplings=None):
    """A :class:`MotionModel` from ``sites`` (Simulate :class:`Site`s, :class:`MotionSite`s or
    their dicts) and ``modes`` given as :class:`Mode`s or the dicts of
    :func:`mode_from_spec`."""
    return MotionModel(list(sites), hops, [mode_from_spec(m) for m in modes], couplings)


#: Sense of rotation as users say it -> the Mode model. Looking down the axis from its tip
#: (+z towards the viewer), 'anticlockwise' is the right-hand (positive) sense.
_SENSE = {"both": "nearest", "either": "nearest", "back and forth": "nearest",
          "nearest": "nearest", "anticlockwise": "forward", "counterclockwise": "forward",
          "forward": "forward", "clockwise": "backward", "backward": "backward",
          "any": "random", "random": "random", "all": "random"}


def interaction_site(nucleus, delta_iso=0.0, label=None, csa=None, quad=None, dipolar=(),
                     euler=(0.0, 0.0, 0.0), fwhm=1.0, lineshape="lorentzian"):
    """A :class:`MotionSite` built directly from its tensors (no Simulate needed).

    ``csa`` = ``(delta_aniso_ppm, eta)`` (Haeberlen); ``quad`` = ``(C_Q_MHz, eta_Q)``;
    ``dipolar`` = ``[("1H", {"r_A": 1.09}), ("1H", {"D_Hz": 23300})...]`` or the short form
    ``("1H", 1.09)`` meaning a distance in angstrom. The interaction's principal axes
    (the EFG frame, or the CSA frame without a quadrupolar coupling) sit at ``euler`` =
    (alpha, beta, gamma) degrees from the frame of the first rotation axis -- ``beta`` is
    the angle between that axis and the principal z axis (for a C-D bond along z, the
    angle between the axis and the bond). The dipolar vectors are along the site z axis
    unless a ``"vector"`` is given in the dict."""
    site = Site(nucleus, delta_iso, label=label, fwhm=fwhm, lineshape=lineshape)
    if quad is not None:
        site.add_quadrupolar(float(quad[0]), float(quad[1]) if len(quad) > 1 else 0.0)
    if csa is not None:
        site.add_csa(float(csa[0]), float(csa[1]) if len(csa) > 1 else 0.0)
    out = MotionSite(site, euler=euler)
    for partner, how in dipolar:
        kw = dict(how) if isinstance(how, dict) else {"r_A": float(how)}
        out.add_dipolar(partner, **kw)
    return out


def rotation_model(sites, axes, hops=None, euler=None):
    """A :class:`MotionModel` for tensors that rotate about one or several axes, each
    with its own positions, sense and rate -- the notebook counterpart of the Dynamics
    rotation set-up.

    ``sites``: a :class:`MotionSite` (see :func:`interaction_site`), a Simulate
    :class:`Site` or a list of them. ``euler``: (alpha, beta, gamma) degrees placing the
    interaction's principal axes in the frame of axis 1 (plain Sites only; a MotionSite
    keeps its own).

    ``axes``: a list of dicts, **axis 1 first (innermost: the one nearest the
    interaction)**. Axis 1 *is* the reference direction (z); every further axis is given
    by ``polar`` (its angle from axis 1) and ``azimuth`` (degrees, about axis 1, from the
    x axis of the reference frame). Keys:

    * ``kind``: 'ring_jump' (default), 'flip' or 'ring_diffusion';
    * ``n``: number of positions (ring_jump), ``step_deg``: angle between positions
      (default 360/n: a closed ring; smaller for an open arc), ``angle_deg`` (flip);
    * ``sense``: 'both' (default), 'anticlockwise', 'clockwise' or 'any' (hops to every
      other position);
    * ``k_Hz`` (rate per hop) or ``Ea_eV`` with ``nu0_Hz`` -- every axis has its own;
    * ``label``.

    Axes are independent processes, combined as in :class:`MotionModel`: a methyl turning
    about the C-C axis (axis 1) on a chain that turns about another axis (axis 2) is
    ``[{"n": 3, "k_Hz": 1e8}, {"n": 2, "polar": 55, "k_Hz": 1e4}]``."""
    if isinstance(sites, (MotionSite, Site, dict)):
        sites = [sites]
    sites = [_as_motion_site(s) for s in sites]
    if euler is not None:
        if len(sites) != 1:
            raise ValueError("euler places one interaction; give a MotionSite per site "
                             "(interaction_site(..., euler=...)) for several.")
        sites[0].euler = tuple(float(a) for a in euler)
    if not axes:
        raise ValueError("Give at least one rotation axis.")
    modes = []
    for i, ax in enumerate(axes):
        spec = dict(ax)
        kind = spec.pop("kind", "ring_jump")
        polar = float(spec.pop("polar", 0.0))
        azimuth = float(spec.pop("azimuth", 0.0))
        sense = spec.pop("sense", "both")
        if i == 0:
            if polar or azimuth:
                raise ValueError("Axis 1 is the reference direction: leave out polar/azimuth "
                                 "and place the interaction with euler instead.")
            spec["axis"] = "z"
        else:
            spec["axis"] = (polar, azimuth)
        if kind == "ring_jump":
            key = str(sense).strip().lower()
            if key not in _SENSE:
                raise ValueError(f"sense {sense!r}: use 'both', 'anticlockwise', 'clockwise' "
                                 "or 'any'.")
            spec["model"] = _SENSE[key]
        spec["kind"] = kind
        if "label" not in spec:
            spec["label"] = None
        modes.append(mode_from_spec({k: v for k, v in spec.items() if v is not None}))
    return MotionModel(sites, hops, modes)


def motion_overlay_series(model, spectra, per_spectrum_temperature=True, **kwargs):
    """:func:`motion_overlay` for several experimental spectra, e.g. a variable-temperature
    series. ``spectra``: dicts ``{'ppm', 'y', 'label' (optional), 'T_K' (optional)}``.
    With ``per_spectrum_temperature`` each spectrum is simulated at its own ``T_K``
    (Arrhenius rates and barrier-based populations follow); the other keywords are shared.
    Returns ``{'overlays': [...], 'labels': [...], 'T_K': [...], 'log': str}``."""
    overlays, labels, temps = [], [], []
    for i, s in enumerate(spectra):
        kw = dict(kwargs)
        T = s.get("T_K")
        if per_spectrum_temperature and T is not None:
            kw["T_K"] = float(T)
        ov = motion_overlay(model, s["ppm"], s["y"], **kw)
        overlays.append(ov)
        labels.append(s.get("label") or (f"{float(T):g} K" if T is not None else f"spectrum {i + 1}"))
        temps.append(None if T is None else float(T))
    log = "\n".join(f"{lab}: scale {o['scale']:.4g}, rms residual {100 * o['rms']:.2f} %"
                    for lab, o in zip(labels, overlays))
    return {"overlays": overlays, "labels": labels, "T_K": temps, "log": log}


def plot_motion_overlay_stack(series, ax=None, offset=1.1, nucleus=None, data_colour="0.15",
                              colours=None, linewidth=1.2, text_size="small"):
    """The result of :func:`motion_overlay_series` as a stack (first at the bottom): each
    experimental spectrum thin and dark, its simulation coloured, each normalised to its own
    maximum and labelled. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    from ..plotting.accessibility import default_series_colours

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.375, 4.0))
    else:
        fig = ax.figure
    ovs = series["overlays"]
    colours = colours or default_series_colours(len(ovs), ordered=True)
    lo, hi = np.inf, -np.inf
    for i, (ov, lab) in enumerate(zip(ovs, series["labels"])):
        norm = float(np.max(np.abs(ov["y"]))) or 1.0
        base = i * offset
        l1, = ax.plot(ov["x_ppm"], ov["y"] / norm + base, "-", color=data_colour,
                      lw=0.8 * linewidth, label="experiment" if i == 0 else None)
        l2, = ax.plot(ov["x_ppm"], ov["sim"] / norm + base, "-", color=colours[i % len(colours)],
                      lw=linewidth, label="simulation" if i == 0 else None)
        l1._opennmr_baseline = l2._opennmr_baseline = base
        ax.annotate(lab, xy=(1.0, base + 0.05), xycoords=("axes fraction", "data"), ha="right",
                    va="bottom", fontsize=text_size, color=colours[i % len(colours)])
        lo, hi = min(lo, float(np.min(ov["x_ppm"]))), max(hi, float(np.max(ov["x_ppm"])))
    ax.set_xlim(hi, lo)
    ax.set_xlabel(delta_label(nucleus))
    ax.yaxis.set_visible(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig, ax


def motion_axis_limits(model, larmor_Hz=None, B0_T=None, order="first", transitions="all",
                       mas_Hz=0.0, gauss_fwhm_ppm=0.0, margin=0.1):
    """``(high, low)`` ppm that frame every state's pattern (all lines, the sidebands under
    MAS), padded by ``margin`` of its width and a few line widths."""
    larmor_Hz = _resolve_larmor(model, larmor_Hz, B0_T)
    b, _w = _hemisphere(1000)
    nu = _frequencies(model.state_data(), b, model.lines(transitions), larmor_Hz, order,
                      model.spin) / larmor_Hz * 1e6
    lo, hi = float(nu.min()), float(nu.max())
    width = float(np.max(model.state_data()["fwhm_ppm"])) + float(gauss_fwhm_ppm)
    pad = margin * (hi - lo) + 3.0 * width + 2.0 * float(mas_Hz or 0.0) / larmor_Hz * 1e6
    if hi - lo <= 0 and pad <= 0:
        pad = 1.0
    return hi + pad, lo - pad


def motion_overlay(model, ppm, y, fit_offset=True, window=None, n_points=2048, **kwargs):
    """Simulate ``model`` (any :func:`motion_spectrum` keyword: ``larmor_Hz``, ``T_K``,
    ``k_modes``, ``hop_scale``, ``mas_Hz``, ``echo``, ``order``, ``gauss_fwhm_ppm`` ...) and
    scale it onto the experimental spectrum ``(ppm, y)``: ``y ~ scale * sim + offset`` by
    linear least squares over ``window`` = ``(lo, hi)`` ppm (default the whole axis),
    ``offset`` only with ``fit_offset``. The simulation is computed on ``n_points`` points
    across the window and interpolated onto the experimental points.

    Returns ``x_ppm, y, sim`` (scaled, on the experimental points), ``residual``, ``scale``,
    ``offset``, ``rms`` (relative to the experimental maximum), ``simulation`` (the
    :func:`motion_spectrum` result) and ``log``.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    if ppm.shape != y.shape or ppm.ndim != 1:
        raise ValueError("ppm and y must be 1D arrays of the same length.")
    lo, hi = (float(np.min(ppm)), float(np.max(ppm))) if window is None else (
        float(min(window)), float(max(window)))
    keep = (ppm >= lo) & (ppm <= hi)
    if keep.sum() < 8:
        raise ValueError("Fewer than 8 experimental points lie in the window.")
    xs, ys = ppm[keep], y[keep]
    axis = np.linspace(hi, lo, int(n_points))
    res = motion_spectrum(model, axis, **kwargs)
    order = np.argsort(res["x_ppm"])
    base = np.interp(xs, res["x_ppm"][order], res["real_spec"][order])
    if fit_offset:
        A = np.column_stack([base, np.ones_like(base)])
        (scale, offset), *_ = np.linalg.lstsq(A, ys, rcond=None)
    else:
        denom = float(base @ base)
        scale, offset = (float(base @ ys) / denom if denom > 0 else 0.0), 0.0
    sim = scale * base + offset
    resid = ys - sim
    peak = float(np.max(np.abs(ys))) or 1.0
    rms = float(np.sqrt(np.mean(resid ** 2)) / peak)
    log = res["log"] + (f"\nScaled onto the experiment over {lo:.4g} to {hi:.4g} ppm "
                        f"({int(keep.sum())} points): scale {scale:.4g}"
                        + (f", offset {offset:.4g}" if fit_offset else "")
                        + f"; rms residual {100 * rms:.2f} % of the maximum.")
    return {"x_ppm": xs, "y": ys, "sim": sim, "residual": resid, "scale": float(scale),
            "offset": float(offset), "rms": rms, "simulation": res, "log": log,
            "window": (lo, hi)}


def plot_motion_overlay(overlay, ax=None, nucleus=None, data_colour="0.15", colour=None,
                        show_residual=True, linewidth=1.2, text_size="small", label=None,
                        xlim=None):
    """The experimental spectrum (thin, dark) with the scaled simulation (coloured) from
    :func:`motion_overlay`, and the residual (grey, offset below). Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    from ..plotting.accessibility import default_series_colours

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.375, 2.6))
    else:
        fig = ax.figure
    colour = colour or default_series_colours(1, ordered=True)[0]
    x, y, sim, res = overlay["x_ppm"], overlay["y"], overlay["sim"], overlay["residual"]
    scale = float(np.max(np.abs(y))) or 1.0
    ax.plot(x, y / scale, "-", color=data_colour, lw=0.8 * linewidth, label="experiment")
    ax.plot(x, sim / scale, "-", color=colour, lw=linewidth, label=label or "simulation")
    if show_residual:
        ax.plot(x, res / scale - 0.15, "-", color="0.55", lw=0.7 * linewidth, label="residual")
    ax.set_xlim(xlim if xlim is not None else (float(np.max(x)), float(np.min(x))))
    ax.set_xlabel(delta_label(nucleus))
    ax.yaxis.set_visible(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=text_size, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=3, handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout()
    return fig, ax


def plot_motion_limits(limits, ax=None, nucleus=None, linewidth=1.2, text_size="small",
                       colours=None):
    """The three reference spectra of :func:`limit_spectra` on one axis: the slow limit
    (no motion on the time scale), the fast limit (frequencies averaged) and the
    motionally averaged tensor. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    from ..plotting.accessibility import default_series_colours

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.375, 2.6))
    else:
        fig = ax.figure
    colours = colours or default_series_colours(3, ordered=False)
    x = limits["x_ppm"]
    scale = max(float(np.max(limits[k])) for k in ("static", "fast", "fast_tensor")) or 1.0
    for key, label, c, ls in (("static", "slow limit", colours[0], "-"),
                              ("fast", "fast (frequency average)", colours[1], "-"),
                              ("fast_tensor", "fast (tensor average)", colours[2], "--")):
        ax.plot(x, limits[key] / scale, ls, color=c, lw=linewidth, label=label)
    ax.set_xlim(float(np.max(x)), float(np.min(x)))
    ax.set_xlabel(delta_label(nucleus))
    ax.yaxis.set_visible(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=text_size, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=2, handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout()
    return fig, ax


def motion_notebook_code(model, rates=None, which="mode:0", larmor_MHz=None, T_K=None,
                         mas_Hz=None, echo=None, ppm_range=None, extra=None,
                         overlay=None):
    """Python that rebuilds ``model`` and runs the rate (or temperature) series -- what
    the GUI's 'Copy notebook code' gives. ``extra``: more :func:`motion_spectrum`
    keywords as a dict (``order``, ``transitions``, ``gauss_fwhm_ppm``, ...);
    ``ppm_range`` = (high, low); ``overlay``: a dict of the single-spectrum keywords
    (``k_modes``, ``hop_scale``, ``T_K`` ...) to add a :func:`motion_overlay` block
    for an experimental spectrum you load yourself."""
    import json
    spec = json.dumps(model.to_dict())
    lines = ["import json, numpy as np",
             "from opennmr.notebook import motion",
             f"model = motion.MotionModel.from_dict(json.loads({spec!r}))"]
    larmor = "larmor_Hz=%g" % (larmor_MHz * 1e6) if larmor_MHz else "B0_T=18.8"
    hi, lo = ppm_range if ppm_range else (100.0, -100.0)
    kw = ""
    if mas_Hz:
        kw += f", mas_Hz={float(mas_Hz):g}"
    if echo:
        kw += f", echo={echo!r}"
    for k, v in (extra or {}).items():
        kw += f", {k}={v!r}"
    if rates is not None:
        lines += [f"ppm = np.linspace({hi:g}, {lo:g}, 2048)",
                  f"series = motion.motion_rate_series(model, {list(rates)!r}, which={which!r}, "
                  f"ppm_axis=ppm, {larmor}" + (f", T_K={T_K:g}" if T_K and which != "T" else "")
                  + kw + ")",
                  "motion.plot_motion_series(series)"]
    if overlay is not None:
        okw = "".join(f", {k}={v!r}" for k, v in overlay.items())
        lines += ["",
                  "# Your experimental spectrum: ppm_exp, y_exp = ... (1D arrays, e.g. from",
                  "# opennmr.notebook's loaders), then scale the simulation onto it:",
                  f"overlay = motion.motion_overlay(model, ppm_exp, y_exp, {larmor}{kw}{okw})",
                  "print(overlay['log'])",
                  "motion.plot_motion_overlay(overlay)"]
    return "\n".join(lines)
