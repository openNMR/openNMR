"""
Strongly coupled (higher-order) J-coupled spectra: exact simulation and
fitting of an isotropic spin system in solution.

Where :func:`nmrtools.simulations.jcoupled_spectrum` builds a first-order
multiplet (each coupling splits every line symmetrically; valid when
|delta_nu| >> |J|), this module diagonalises the full Hamiltonian, so
roofing, AB quartets, ABX and AA'BB' patterns, second-order line
positions and "virtual coupling" all come out as they are measured.

Model
-----
A spin system is a list of spins -- each with a label, a nuclide, a
shift (ppm) and optionally a count ``n`` of *magnetically equivalent*
copies -- and a set of scalar couplings J (Hz) between labelled spins.
In the rotating frame of each nuclide (Hz)::

    H = sum_i nu_i Iz_i
        + sum_{i<j, same nuclide}      J_ij (Ix_i Ix_j + Iy_i Iy_j + Iz_i Iz_j)
        + sum_{i<j, different nuclide} J_ij  Iz_i Iz_j

with nu_i = shift_i (ppm) x Larmor frequency of that nuclide x 1e-6.
Couplings between different nuclides are always weak (their Larmor
frequencies differ by MHz), which is exact to within parts in 10^5 and
is what makes a heteronuclear partner's own shift irrelevant. Spins of
any I are allowed.

The Hamiltonian commutes with the total Iz of every nuclide, so it is
built and diagonalised block by block (one block per combination of
total m per nuclide) -- a 10-spin-1/2 homonuclear system is at most a
252 x 252 problem rather than 1024 x 1024.

Observed spectrum: transitions of F- = sum over observed spins of I-_i,
intensity |<b|F-|a>|^2 (equal populations: the high-temperature limit),
frequency E_a - E_b. Intensities are normalised so each observed spin
contributes 1 in total (sum rule), so a site's integral is its number of
nuclei.

Every spin in the system is taken to be the NMR-active isotope named:
the system describes one isotopomer. For the natural-abundance
satellites of a dilute partner (13C, 29Si, 195Pt...) use the first-order
coupling groups in :mod:`nmrtools.simulations`, or add the spin here and
scale the result by that isotopomer's abundance yourself.

Workflow
--------
::

    spins = [{"label": "A", "nuclide": "1H", "shift_ppm": 7.30},
             {"label": "B", "nuclide": "1H", "shift_ppm": 7.25},
             {"label": "X", "nuclide": "31P", "shift_ppm": 0.0}]
    couplings = [{"a": "A", "b": "B", "J_Hz": 8.0},
                 {"a": "A", "b": "X", "J_Hz": 12.0}]
    spec = spin_system_spectrum(spins, couplings, B0_tesla=9.395,
                                ppm_axis=np.linspace(7.0, 7.6, 20000),
                                fwhm_Hz=0.8)

and :func:`fit_spin_system_spectrum` to fit shifts, Js, linewidth and
height to a measured spectrum.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import find_isotope_info
from .simulations import multiplet_lineshape
from ..core import fit_progress as _fit_progress

#: Largest Hamiltonian block diagonalised before refusing (dense eigh on
#: an N x N block costs ~N^3; 3500 is a few seconds).
MAX_BLOCK_DIM = 3500


# ============================================================================
# Building the system
# ============================================================================
def _normalise_spins(spins) -> List[dict]:
    out, seen = [], set()
    for i, s in enumerate(spins):
        label = str(s.get("label", f"S{i + 1}"))
        if label in seen:
            raise ValueError(f"Two spins are labelled '{label}'.")
        seen.add(label)
        nuclide = str(s["nuclide"])
        info = find_isotope_info(nuclide)
        if info["spin"] <= 0:
            raise ValueError(f"{label}: {nuclide} has no nuclear spin.")
        n = int(s.get("n", 1))
        if n < 1:
            raise ValueError(f"{label}: n must be at least 1.")
        out.append({"label": label, "nuclide": nuclide, "shift_ppm": float(s["shift_ppm"]),
                    "n": n, "I": float(info["spin"]), "gamma": float(info["gamma"])})
    if not out:
        raise ValueError("A spin system needs at least one spin.")
    return out


def _normalise_couplings(couplings, labels) -> Dict[Tuple[str, str], float]:
    out = {}
    items = couplings.items() if isinstance(couplings, dict) else (
        ((c["a"], c["b"]), c["J_Hz"]) for c in couplings)
    for (a, b), J in items:
        a, b = str(a), str(b)
        if a not in labels or b not in labels:
            raise ValueError(f"Coupling {a}-{b}: no spin labelled "
                             f"{a if a not in labels else b}.")
        if a == b:
            raise ValueError(f"Coupling {a}-{a}: a spin cannot couple to itself (the J "
                             f"within an equivalent group has no effect on the spectrum).")
        key = tuple(sorted((a, b)))
        if key in out:
            raise ValueError(f"Coupling {key[0]}-{key[1]} is given twice.")
        out[key] = float(J)
    return out


def expand_spin_system(spins, couplings):
    """Expand each spin with ``n`` equivalent copies into explicit spins.

    Returns (expanded, J) where ``expanded`` is a list of dicts (label,
    parent label, nuclide, I, gamma, shift_ppm) and ``J`` an (N, N)
    symmetric array in Hz. Copies of one spin share its shift and all its
    couplings (magnetic equivalence); the J between copies is set to 0,
    which is exact since it does not affect the spectrum."""
    spins = _normalise_spins(spins)
    labels = [s["label"] for s in spins]
    J_pairs = _normalise_couplings(couplings, set(labels))
    expanded = []
    for s in spins:
        for k in range(s["n"]):
            expanded.append({**s, "parent": s["label"],
                             "label": s["label"] if s["n"] == 1 else f"{s['label']}{k + 1}"})
    N = len(expanded)
    J = np.zeros((N, N))
    for i, j in itertools.combinations(range(N), 2):
        key = tuple(sorted((expanded[i]["parent"], expanded[j]["parent"])))
        J[i, j] = J[j, i] = J_pairs.get(key, 0.0)
    return expanded, J


def larmor_frequencies(expanded, B0_tesla) -> np.ndarray:
    """|gamma| B0 (Hz) of every expanded spin."""
    return np.array([abs(s["gamma"]) * B0_tesla for s in expanded])


def block_sizes(spins, couplings=()) -> Dict[tuple, int]:
    """Dimension of each Hamiltonian block (keyed by total m per nuclide)
    without building anything -- to check a system is tractable."""
    expanded, _J = expand_spin_system(spins, couplings)
    nuclides = sorted({s["nuclide"] for s in expanded})
    sizes: Dict[tuple, int] = {(): 1}
    for s in expanded:
        ms = np.arange(-s["I"], s["I"] + 0.5, 1.0)
        k = nuclides.index(s["nuclide"])
        new: Dict[tuple, int] = {}
        for key, count in sizes.items():
            key = key if key else (0.0,) * len(nuclides)
            for m in ms:
                nk = list(key)
                nk[k] = round(nk[k] + m, 6)
                nk = tuple(nk)
                new[nk] = new.get(nk, 0) + count
        sizes = new
    return sizes


# ============================================================================
# Diagonalisation and transitions
# ============================================================================
def _raise_coef(I, m):
    return np.sqrt(max(I * (I + 1) - m * (m + 1), 0.0))


def _lower_coef(I, m):
    return np.sqrt(max(I * (I + 1) - m * (m - 1), 0.0))


def spin_system_transitions(spins, couplings, B0_tesla, observe=None,
                            intensity_threshold=1e-9, merge_within_Hz=1e-7):
    """Exact transitions of the observed nuclide.

    Parameters
    ----------
    spins : list of dict
        ``label``, ``nuclide``, ``shift_ppm``, optional ``n`` (magnetically
        equivalent copies, default 1).
    couplings : list of dict (``a``, ``b``, ``J_Hz``) or dict
        {(label_a, label_b): J_Hz}. Pairs not listed are uncoupled.
    B0_tesla : float
        Field; sets each nuclide's Larmor frequency, which converts shifts
        to Hz (and so sets how strongly coupled each pair is).
    observe : str, optional
        Observed nuclide; default the nuclide of the first spin.
    intensity_threshold : float
        Lines weaker than this fraction of the strongest are dropped.
    merge_within_Hz : float
        Lines closer than this are summed into one at their
        intensity-weighted mean position (degenerate transitions, and --
        for large systems -- lines far closer than any linewidth).

    Returns
    -------
    dict with ``stick_ppm`` (observed-nuclide shift scale), ``stick_Hz``
    (offset from that nuclide's reference frequency), ``intensity`` (each
    observed spin contributing 1 in total), ``larmor_Hz`` (of the observed
    nuclide), ``n_observed`` (observed spins after expansion) and
    ``max_block`` (largest block diagonalised).
    """
    expanded, J = expand_spin_system(spins, couplings)
    observe = observe or expanded[0]["nuclide"]
    obs_idx = [i for i, s in enumerate(expanded) if s["nuclide"] == observe]
    if not obs_idx:
        raise ValueError(f"No {observe} spin in the system to observe.")
    N = len(expanded)
    nuclides = sorted({s["nuclide"] for s in expanded})
    kind = np.array([nuclides.index(s["nuclide"]) for s in expanded])
    obs_kind = nuclides.index(observe)
    I = np.array([s["I"] for s in expanded])
    nu = np.array([s["shift_ppm"] for s in expanded]) * larmor_frequencies(expanded, B0_tesla) * 1e-6
    same = kind[:, None] == kind[None, :]

    sizes = block_sizes(spins, couplings)
    biggest = max(sizes.values())
    if biggest > MAX_BLOCK_DIM:
        raise ValueError(
            f"This spin system needs a {biggest} x {biggest} block (limit {MAX_BLOCK_DIM}): "
            f"too many strongly coupled spins. Split it into weakly coupled parts, or treat "
            f"equivalent groups with 'n'.")

    # Basis: every product state, grouped into blocks by total m per nuclide.
    m_lists = [np.arange(I[i], -I[i] - 0.5, -1.0) for i in range(N)]
    blocks: Dict[tuple, list] = {}
    for state in itertools.product(*m_lists):
        state = np.array(state)
        key = tuple(round(float(state[kind == k].sum()), 6) for k in range(len(nuclides)))
        blocks.setdefault(key, []).append(state)

    pairs = [(i, j) for i, j in itertools.combinations(range(N), 2) if J[i, j] != 0.0]
    flip_pairs = [(i, j) for i, j in pairs if same[i, j]]

    eig = {}
    for key, states in blocks.items():
        S = np.array(states)
        index = {tuple(st): k for k, st in enumerate(map(tuple, S))}
        dim = len(states)
        H = np.zeros((dim, dim))
        diag = S @ nu
        for i, j in pairs:
            diag = diag + J[i, j] * S[:, i] * S[:, j]
        H[np.arange(dim), np.arange(dim)] = diag
        for i, j in flip_pairs:
            for k, st in enumerate(S):
                # (J/2)(I+_i I-_j + I-_i I+_j): raise i, lower j (and the
                # reverse, from the (j, i) ordering below)
                for a, b in ((i, j), (j, i)):
                    if st[a] < I[a] - 1e-9 and st[b] > -I[b] + 1e-9:
                        new = st.copy()
                        new[a] += 1
                        new[b] -= 1
                        k2 = index[tuple(new)]
                        H[k2, k] += 0.5 * J[i, j] * _raise_coef(I[a], st[a]) * _lower_coef(I[b], st[b])
        E, V = np.linalg.eigh(H)
        eig[key] = (E, V, S, index)

    freqs, ints = [], []
    for key, (E_hi, V_hi, S_hi, _idx) in eig.items():
        low_key = list(key)
        low_key[obs_kind] = round(low_key[obs_kind] - 1.0, 6)
        low_key = tuple(low_key)
        if low_key not in eig:
            continue
        E_lo, V_lo, _S_lo, idx_lo = eig[low_key]
        F = np.zeros((len(E_lo), len(E_hi)))
        for k, st in enumerate(S_hi):
            for i in obs_idx:
                if st[i] > -I[i] + 1e-9:
                    new = st.copy()
                    new[i] -= 1
                    F[idx_lo[tuple(new)], k] += _lower_coef(I[i], st[i])
        T = V_lo.T @ F @ V_hi
        freqs.append((E_hi[None, :] - E_lo[:, None]).ravel())
        ints.append((T * T).ravel())
    freqs = np.concatenate(freqs) if freqs else np.array([])
    ints = np.concatenate(ints) if ints else np.array([])

    # Each observed spin contributes Tr(I+ I-) x (other states) in total;
    # normalise that to 1 per spin.
    dim_total = int(np.prod([2 * i + 1 for i in I]))
    I_obs = I[obs_idx[0]]
    ints = ints / ((2.0 / 3.0) * I_obs * (I_obs + 1) * dim_total)

    keep = ints > intensity_threshold * (ints.max() if ints.size else 1.0)
    freqs, ints = _merge_lines(freqs[keep], ints[keep], tol_Hz=merge_within_Hz)
    larmor_obs = abs(find_isotope_info(observe)["gamma"]) * B0_tesla
    return {"stick_Hz": freqs, "stick_ppm": freqs / larmor_obs * 1e6, "intensity": ints,
            "larmor_Hz": larmor_obs, "n_observed": len(obs_idx), "max_block": biggest,
            "observe": observe}


def _merge_lines(freqs, ints, tol_Hz=1e-7):
    """Sum lines closer than tol_Hz (degenerate transitions), placing each
    merged line at the intensity-weighted mean -- continuous in the
    parameters, unlike rounding to a grid. Chains of lines each within
    tol_Hz of the next merge together, so a group spans at most
    (lines - 1) x tol_Hz."""
    if freqs.size == 0:
        return freqs, ints
    order = np.argsort(freqs)
    f, w = freqs[order], ints[order]
    groups = np.concatenate(([0], np.cumsum(np.diff(f) > tol_Hz)))
    W = np.bincount(groups, weights=w)
    Fm = np.bincount(groups, weights=w * f) / np.where(W > 0, W, 1)
    return Fm, W


# ============================================================================
# Spectra
# ============================================================================
def spin_system_spectrum(spins, couplings, B0_tesla, ppm_axis, fwhm_Hz=1.0,
                         lineshape="lorentzian", eta=0.0, observe=None, scale="area"):
    """Broadened spectrum of the observed nuclide.

    fwhm_Hz : linewidth of every line (Hz).
    scale : 'area' -- the spectrum sums to ~(number of observed spins) over
        the axis points, as :func:`nmrtools.simulations.jcoupled_spectrum`'s
        does per site; 'peak' -- each unit-intensity line has height 1.

    Returns a dict like jcoupled_spectrum's: ``x_ppm``, ``real_spec``,
    ``normalised_real_spec``, ``stick_ppm``, ``stick_intensity``, plus
    ``transitions`` (the full :func:`spin_system_transitions` result).
    """
    # Lines within fwhm/1000 of each other are drawn as one (their
    # weighted mean): invisible at any linewidth, and it keeps a
    # 12-spin system (~10^6 transitions) quick to broaden.
    t = spin_system_transitions(spins, couplings, B0_tesla, observe=observe,
                                intensity_threshold=1e-7,
                                merge_within_Hz=max(1e-7, fwhm_Hz / 1000.0))
    x = np.asarray(ppm_axis, dtype=float)
    fwhm_ppm = fwhm_Hz / t["larmor_Hz"] * 1e6
    if scale == "area":
        spacing = abs(x[1] - x[0]) if x.size > 1 else 1.0
        y = spacing * multiplet_lineshape(x, t["stick_ppm"], t["intensity"], fwhm_ppm,
                                          lineshape=lineshape, eta=eta, normalise="area")
    elif scale == "peak":
        y = multiplet_lineshape(x, t["stick_ppm"], t["intensity"], fwhm_ppm,
                                lineshape=lineshape, eta=eta, normalise="peak")
    else:
        raise ValueError("scale must be 'area' or 'peak'.")
    peak = np.max(np.abs(y)) if y.size else 0.0
    return {"x_ppm": x, "real_spec": y, "normalised_real_spec": y / peak if peak else y,
            "stick_ppm": t["stick_ppm"], "stick_intensity": t["intensity"], "transitions": t}


def coupling_strengths(spins, couplings, B0_tesla) -> List[dict]:
    """For every coupled pair of the same nuclide: |delta_nu| / |J|.
    Below ~10 the first-order picture visibly fails; below ~3 it is
    qualitatively wrong. Pairs of different nuclides are always weak."""
    s = {x["label"]: x for x in _normalise_spins(spins)}
    out = []
    for (a, b), J in _normalise_couplings(couplings, set(s)).items():
        if s[a]["nuclide"] != s[b]["nuclide"] or J == 0:
            continue
        larmor = abs(s[a]["gamma"]) * B0_tesla
        dnu = abs(s[a]["shift_ppm"] - s[b]["shift_ppm"]) * larmor * 1e-6
        out.append({"a": a, "b": b, "J_Hz": J, "delta_nu_Hz": dnu, "ratio": dnu / abs(J)})
    return out


# ============================================================================
# Fitting
# ============================================================================
def _system_parameters(spins, couplings):
    """Flat parameter layout: every spin's shift, every coupling's J,
    then fwhm_Hz, amplitude, b0, b1."""
    spins = _normalise_spins(spins)
    labels = [s["label"] for s in spins]
    pairs = _normalise_couplings(couplings, set(labels))
    names = [f"shift {l} (ppm)" for l in labels] + [f"J {a}-{b} (Hz)" for a, b in pairs]
    names += ["fwhm (Hz)", "amplitude", "baseline: b0", "baseline: b1"]
    return spins, labels, list(pairs.items()), names


def fit_spin_system_spectrum(results, spins, couplings, B0_tesla=None, fwhm_Hz=1.0,
                             lineshape="lorentzian", eta=0.0, fit_regions=None,
                             fixed=None, bounds=None, observe=None,
                             baseline_regions=None, baseline_sigma=2.5,
                             baseline_max_iter=10, max_iterations=None):
    """Fit a spin system's shifts, Js, linewidth and height to a spectrum.

    The baseline offset (b0) is fitted with everything else; its slope is
    held at a robust estimate.

    Parameters
    ----------
    results : dict
        As from :func:`nmrtools.one_d.NMR1dimport` (x_ppm, real_spectrum,
        SFO1 in MHz, nuclide).
    spins, couplings :
        Starting values -- see :func:`spin_system_transitions`.
    B0_tesla : float, optional
        Default: SFO1 / |gamma| of the observed nuclide.
    fwhm_Hz : float
        Starting linewidth, shared by all lines.
    fit_regions : [(lo, hi), ...] in ppm, optional; default whole spectrum.
    fixed : dict, optional
        Parameter name -> True to hold it, names as in ``info["names"]``:
        ``"shift A (ppm)"``, ``"J A-B (Hz)"`` (labels in sorted order),
        ``"fwhm (Hz)"``, ``"amplitude"``.
    bounds : dict, optional
        Parameter name -> (low, high).
    max_iterations : int, optional
        Least-squares iterations allowed (each costs one model evaluation
        plus one per free parameter). ``None`` (default) uses the app-wide
        limit, :func:`opennmr.core.fit_progress.max_iterations` (2000 unless
        changed). A fit that runs out raises
        :class:`~opennmr.core.fit_progress.FitNotConverged`, whose message
        says what to try next.

    Signs: unlike a first-order multiplet, a strongly coupled spectrum
    can depend on the *relative* signs of couplings (ABX, AA'XX'), so J is
    not bounded to be positive; the starting sign selects the solution.

    Returns
    -------
    popt, pcov (full; zero rows/cols for fixed parameters), is_free, info
    -- ``info`` holds ``names``, ``labels``, ``pairs``, ``B0_tesla``,
    ``observe``, ``lineshape``, ``eta``, ``x_ref``, ``fit_regions``,
    ``warnings`` and is what :func:`evaluate_spin_system_fit` needs.

    The model is ``amplitude * sum_k w_k * L(x - x_k) + baseline``, with
    w_k the transition intensities (each observed spin totalling 1) and L
    a peak-height-1 line: ``amplitude`` is the height one isolated
    single-nucleus line would have.
    """
    from scipy.optimize import curve_fit
    from .fitting import build_region_mask, estimate_baseline_robust, expand_pcov

    spins_n, labels, pairs, names = _system_parameters(spins, couplings)
    observe = observe or spins_n[0]["nuclide"]
    if B0_tesla is None:
        B0_tesla = float(results["SFO1"]) * 1e6 / abs(find_isotope_info(observe)["gamma"])
    fixed = dict(fixed or {})
    bounds = dict(bounds or {})
    unknown = [k for k in list(fixed) + list(bounds) if k not in names]
    if unknown:
        raise ValueError(f"Unknown parameter(s) {unknown}; names are {names[:-2]}.")

    x_all = np.asarray(results["x_ppm"], dtype=float)
    y_raw = np.asarray(results["real_spectrum"], dtype=float)
    scale = float(np.max(np.abs(y_raw))) or 1.0
    order = np.argsort(x_all)
    x_sorted, y_sorted = x_all[order], y_raw[order] / scale
    mask = build_region_mask(x_sorted, fit_regions)
    x_fit, y_fit = x_sorted[mask], y_sorted[mask]
    x_ref = float(np.mean(x_fit))
    b0, b1, _ = estimate_baseline_robust(x_sorted, y_sorted, baseline_regions=baseline_regions,
                                         sigma=baseline_sigma, max_iter=baseline_max_iter,
                                         x_ref=x_ref)
    n_s, n_j = len(labels), len(pairs)

    def unpack(p):
        sp = [{**{k: s[k] for k in ("label", "nuclide", "n")}, "shift_ppm": p[i]}
              for i, s in enumerate(spins_n)]
        cp = {pair: p[n_s + k] for k, (pair, _J) in enumerate(pairs)}
        return sp, cp, p[n_s + n_j], p[n_s + n_j + 1]

    def shape(x, p):
        sp, cp, fwhm_hz, _amp = unpack(p)
        t = spin_system_transitions(sp, cp, B0_tesla, observe=observe,
                                    merge_within_Hz=max(1e-7, abs(fwhm_hz) / 1e4))
        return multiplet_lineshape(x, t["stick_ppm"], t["intensity"],
                                   abs(fwhm_hz) / t["larmor_Hz"] * 1e6, lineshape=lineshape,
                                   eta=eta, normalise="peak")

    fixed_note: list = []
    p0 = np.array([s["shift_ppm"] for s in spins_n] + [J for _pair, J in pairs]
                  + [float(fwhm_Hz), 1.0, b0, b1], dtype=float)
    # Starting height: data maximum over the fit window / the model's own.
    s0 = shape(x_fit, p0)
    p0[n_s + n_j + 1] = (np.max(y_fit - (b0 + b1 * (x_fit - x_ref))) / s0.max()
                         if s0.size and s0.max() > 0 else 1.0)
    is_free = np.array([not fixed.get(n, False) for n in names])
    # Baseline: the offset is fitted, the slope held. A robust estimate
    # made beforehand and held fixed (as the first-order fitter does)
    # reads Lorentzian tails as baseline -- 0.3 % of the height inside a
    # multiplet's window -- and the Js then bend to compensate.
    is_free[-1] = False
    # A spin's shift only shapes the observed spectrum if the spin is of
    # the observed nuclide or strongly coupled -- i.e. coupled to another
    # spin of its own nuclide. Otherwise (a 31P partner of 1H, say) it is
    # undetermined, so it is held rather than left to wander.
    nuc = {s["label"]: s["nuclide"] for s in spins_n}
    for i, s in enumerate(spins_n):
        homo = any(J != 0 and s["label"] in pair and nuc[pair[0]] == nuc[pair[1]]
                   for pair, J in pairs)
        if is_free[i] and s["nuclide"] != observe and not homo:
            is_free[i] = False
            warnings_held = f"{names[i]} held: it does not affect the {observe} spectrum."
            fixed_note.append(warnings_held)
    lower = np.full(p0.size, -np.inf)
    upper = np.full(p0.size, np.inf)
    lower[n_s + n_j] = 1e-4                           # fwhm (Hz)
    lower[n_s + n_j + 1] = 0.0                        # amplitude
    for name, (lo, hi) in bounds.items():
        k = names.index(name)
        lower[k], upper[k] = lo, hi
        if name == "amplitude":
            lower[k], upper[k] = lo / scale, hi / scale
    warnings = list(fixed_note)
    for k in np.flatnonzero(is_free):
        if not lower[k] <= p0[k] <= upper[k]:
            moved = float(np.clip(p0[k], lower[k], upper[k]))
            warnings.append(f"{names[k]}: start {p0[k]:g} outside its bounds; started at {moved:g}.")
            p0[k] = moved

    def model(x, *free):
        p = p0.copy()
        p[is_free] = free
        return p[n_s + n_j + 1] * shape(x_fit, p) + p[-2] + p[-1] * (x_fit - x_ref)

    # Absolute finite-difference steps: shifts 1e-6 ppm, J/fwhm 1e-4 Hz.
    floors = [1e-6] * n_s + [1e-4] * n_j + [1e-4, 0.0, 0.0, 0.0]
    steps = np.array([f / max(1.0, abs(v)) if f else np.sqrt(np.finfo(float).eps)
                      for f, v in zip(floors, p0)])
    # max_nfev in scipy counts residual evaluations but not the finite-
    # difference Jacobian's (n_free more per iteration), so a start that
    # wanders could otherwise cost ~n_free x 20000 diagonalisations. This
    # budget is in iterations' worth of work; tolerances are tight so a
    # converged fit is converged to far below any linewidth.
    n_free = int(is_free.sum())
    popt_free, pcov_free = _fit_progress.limited_curve_fit(
        _fit_progress.tracked(model, y_fit, n_free=n_free),
        x_fit, y_fit, p0=p0[is_free], bounds=(lower[is_free], upper[is_free]),
        diff_step=steps[is_free], max_iterations=max_iterations,
        ftol=1e-12, xtol=1e-12, gtol=1e-12, label="The spin-system fit")
    del n_free
    popt = p0.copy()
    popt[is_free] = popt_free
    unit = np.ones(popt.size)
    unit[n_s + n_j + 1] = scale
    unit[-2:] = scale
    popt = popt * unit
    pcov_free = np.asarray(pcov_free) * np.outer(unit[is_free], unit[is_free])
    pcov = expand_pcov(pcov_free, is_free, popt.size)
    popt[n_s + n_j] = abs(popt[n_s + n_j])
    info = {"names": names, "labels": labels, "pairs": [pair for pair, _J in pairs],
            "scale": scale,
            "spins": spins_n, "B0_tesla": float(B0_tesla), "observe": observe,
            "lineshape": lineshape, "eta": eta, "x_ref": x_ref,
            "fit_regions": fit_regions, "warnings": warnings}
    return popt, pcov, is_free, info


def fitted_system(popt, info):
    """(spins, couplings, fwhm_Hz, amplitude) from a fit, in the input
    format, for re-simulation or as the next starting point."""
    n_s, n_j = len(info["labels"]), len(info["pairs"])
    spins = [{"label": s["label"], "nuclide": s["nuclide"], "n": s["n"],
              "shift_ppm": float(popt[i])} for i, s in enumerate(info["spins"])]
    couplings = [{"a": a, "b": b, "J_Hz": float(popt[n_s + k])}
                 for k, (a, b) in enumerate(info["pairs"])]
    return spins, couplings, float(popt[n_s + n_j]), float(popt[n_s + n_j + 1])


def evaluate_spin_system_fit(x_ppm, popt, info, include_baseline=True):
    """The fitted model at ``x_ppm`` (any points)."""
    spins, couplings, fwhm_hz, amp = fitted_system(popt, info)
    t = spin_system_transitions(spins, couplings, info["B0_tesla"], observe=info["observe"])
    x = np.asarray(x_ppm, dtype=float)
    y = amp * multiplet_lineshape(x, t["stick_ppm"], t["intensity"],
                                  fwhm_hz / t["larmor_Hz"] * 1e6, lineshape=info["lineshape"],
                                  eta=info["eta"], normalise="peak")
    if include_baseline:
        y = y + popt[-2] + popt[-1] * (x - info["x_ref"])
    return y


def _canonical_signs(popt, info, observe):
    """popt with, for each spin not of the observed nuclide, the signs of
    all its couplings flipped if needed so the largest is positive -- the
    one sign the observed spectrum cannot fix."""
    p = np.array(popt, dtype=float)
    n_s = len(info["labels"])
    nuc = {s["label"]: s["nuclide"] for s in info["spins"]}
    for label, nuclide in nuc.items():
        if nuclide == observe:
            continue
        idx = [n_s + k for k, pair in enumerate(info["pairs"]) if label in pair]
        if idx and p[max(idx, key=lambda i: abs(p[i]))] < 0:
            p[idx] = -p[idx]
    return p


def _exchange_starts(spins, couplings, B0_tesla, observe):
    """Couplings-and-shifts starts with dnu_AB and dJ/2 exchanged, for every
    pair of observed-nuclide spins sharing an unobserved partner X."""
    nuc = {s["label"]: s["nuclide"] for s in spins}
    shift = {s["label"]: s["shift_ppm"] for s in spins}
    J = {tuple(sorted((c["a"], c["b"]))): c["J_Hz"] for c in couplings}
    larmor = abs(find_isotope_info(observe)["gamma"]) * B0_tesla
    starts = []
    obs = [l for l in nuc if nuc[l] == observe]
    for x in [l for l in nuc if nuc[l] != observe]:
        for a, b in itertools.combinations(obs, 2):
            jax, jbx = J.get(tuple(sorted((a, x))), 0.0), J.get(tuple(sorted((b, x))), 0.0)
            if jax == 0.0 and jbx == 0.0:
                continue
            nu_a, nu_b = shift[a] * larmor * 1e-6, shift[b] * larmor * 1e-6
            mean, dnu = 0.5 * (nu_a + nu_b), nu_a - nu_b
            jsum, dj = jax + jbx, jax - jbx
            dnu_new, dj_new = 0.5 * dj, 2.0 * dnu
            new_shift = dict(shift)
            new_shift[a] = (mean + 0.5 * dnu_new) / larmor * 1e6
            new_shift[b] = (mean - 0.5 * dnu_new) / larmor * 1e6
            new_J = dict(J)
            new_J[tuple(sorted((a, x)))] = 0.5 * (jsum + dj_new)
            new_J[tuple(sorted((b, x)))] = 0.5 * (jsum - dj_new)
            starts.append((new_shift, new_J))
    return starts


def _fit_rmse(results, popt, info):
    from .fitting import build_region_mask
    x = np.asarray(results["x_ppm"], dtype=float)
    y = np.asarray(results["real_spectrum"], dtype=float)
    mask = build_region_mask(x, info["fit_regions"]) if info["fit_regions"] else np.ones(x.size, bool)
    r = y[mask] - evaluate_spin_system_fit(x[mask], popt, info)
    return float(np.sqrt(np.mean(r * r)))


def equivalent_solutions(results, popt, info, fwhm_Hz=None, fixed=None, bounds=None,
                         tolerance=0.02, max_starts=12, pcov=None, is_free=None):
    """Look for other parameter sets that fit the data as well as ``popt``.

    Strongly coupled spectra can be genuinely ambiguous: the classic case
    is ABX observed on A and B, where two different sets of (delta_A,
    delta_B, J_AX, J_BX) give *identical* AB spectra -- only the X
    spectrum tells them apart. A single fit returns one of them, with
    nothing to say the other exists.

    The ambiguity has an exact form. For each state of X the AB part is an
    AB quartet whose splitting fixes only |dnu_AB +/- dJ/2| (dJ = J_AX -
    J_BX), so exchanging dnu_AB with dJ/2 -- keeping nu_A + nu_B, J_AX +
    J_BX and J_AB -- gives the same spectrum. This refits from that
    exchange for every pair of observed spins of one nuclide sharing an
    unobserved coupling partner (exact for ABX, a starting point for
    larger systems), and keeps every distinct result whose RMS residual
    is within ``tolerance`` (fractional) of the best.

    Solutions that differ only by flipping the signs of *all* couplings
    to one unobserved spin are the same solution -- that overall sign can
    never be measured from the observed nucleus -- and are not listed
    twice.

    Returns a list of dicts ``{"popt", "pcov", "is_free", "rmse",
    "differences"}``, the input solution first (with the ``pcov`` /
    ``is_free`` passed in, if any). ``differences`` names the parameters that differ from
    it. A list of one means no alternative was found among the starts
    tried (not a proof that none exists).
    """
    spins, couplings, fwhm_fit, _amp = fitted_system(popt, info)
    observe = info["observe"]
    starts = _exchange_starts(spins, couplings, info["B0_tesla"], observe)
    base_rmse = _fit_rmse(results, popt, info)
    found = [{"popt": np.asarray(popt), "pcov": pcov, "is_free": is_free, "rmse": base_rmse,
              "differences": []}]
    scale = float(np.max(np.abs(results["real_spectrum"]))) or 1.0
    for shift_start, J_start in starts[:max_starts]:
        cp = [{"a": a, "b": b, "J_Hz": J} for (a, b), J in J_start.items()]
        sp = [{**s, "shift_ppm": shift_start[s["label"]]} for s in spins]
        try:
            p, _c, _f, inf = fit_spin_system_spectrum(
                results, sp, cp, B0_tesla=info["B0_tesla"], fwhm_Hz=fwhm_Hz or fwhm_fit,
                lineshape=info["lineshape"], eta=info["eta"], fit_regions=info["fit_regions"],
                fixed=fixed, bounds=bounds, observe=observe)
        except Exception:  # noqa: BLE001 -- a start that does not converge is just skipped
            continue
        rmse = _fit_rmse(results, p, inf)
        best = min(s["rmse"] for s in found)
        if rmse > best * (1 + tolerance) + 1e-6 * scale:
            continue
        p = _canonical_signs(p, inf, observe)
        diffs = [name for name, a, b in zip(info["names"], _canonical_signs(popt, info, observe), p)
                 if not name.startswith("baseline") and name != "amplitude"
                 and abs(a - b) > (1e-4 if "ppm" in name else 0.05)]
        if not diffs:
            continue
        if any(all(abs(a - b) <= (1e-4 if "ppm" in n else 0.05)
                   for n, a, b in zip(info["names"], _canonical_signs(s["popt"], info, observe), p)
                   if not n.startswith("baseline") and n != "amplitude") for s in found):
            continue
        found.append({"popt": p, "pcov": _c, "is_free": _f, "rmse": rmse, "differences": diffs})
    return found
