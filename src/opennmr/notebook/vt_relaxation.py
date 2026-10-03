"""
Relaxation and linewidths against temperature: BPP / Arrhenius plots,
activation energies, and Hendrickson-Bray motional narrowing.

Typical use (a T1 series at several temperatures, one field)::

    from opennmr.notebook.vt_relaxation import (
        vt_dataset, fit_vt_relaxation, plot_arrhenius)

    data = vt_dataset(T_K=[250, 275, 300, 325, 350, 375, 400],
                      value=[0.91, 0.40, 0.21, 0.16, 0.19, 0.30, 0.52],   # T1 in s
                      error=[0.02] * 7, larmor_MHz=194.4, quantity="T1",
                      label="7Li, 11.7 T")
    fit = fit_vt_relaxation([data], mechanism="dipolar")
    print(fit["log"])
    fig, ax = plot_arrhenius(fit)          # ln(T1) vs 1000/T, temperature on top

Models
------
The correlation time follows an Arrhenius law ``tau_c = tau0 exp(Ea/kB T)``
(or, for polymers and glasses, Vogel-Fulcher-Tammann
``tau_c = tau0 exp(B/kB(T - T0))``). The spectral density is BPP's
Lorentzian, optionally made asymmetric (Kuchler/Heitjans form used for
ion conductors, ``J(w) = tau / (1 + (w tau)^beta)``, ``1 < beta <= 2``;
``beta = 2`` is BPP; the low-temperature flank then has slope
``(beta - 1) Ea``):

* T1:   ``R1  = A [J(w0) + 4 J(2 w0)]``
* T2:   ``R2  = A/2 [3 J(0) + 5 J(w0) + 2 J(2 w0)]``
* T1rho:``R1r = A/2 [3 J(2 w1) + 5 J(w0) + 2 J(2 w0)]``

``A`` is a free dipolar constant ``C`` (s^-2), or, for a quadrupolar
nucleus, ``(3 pi^2/50) (2I+3)/(I^2(2I-1)) (1 + eta^2/3) CQ^2`` with
``CQ`` in Hz (so that the extreme-narrowing limit is the textbook
``(3 pi^2/10) ... CQ^2 tau_c``). Several independent processes (e.g.
two hopping pathways, each giving its own rate maximum) add as rates,
``R = sum_i R_i + R_bg``, with an optional temperature-independent
background ``R_bg`` (paramagnetic impurities, a slow spin diffusion
floor).

Datasets at several Larmor frequencies are fitted *jointly*: ``tau0``,
``Ea`` (and ``beta``) describe the motion and are shared; the amplitude
is shared too unless ``per_dataset_amplitude=True``. Two fields are the
strongest test of a BPP model: the low-temperature flank must move with
``w0^beta`` while the high-temperature flank must not move at all.

What is not offered, and why
----------------------------
* A "BPP + quadrupolar" model with one correlation time. Both terms
  have the *same* temperature dependence (``A_dip F(tau) + A_Q F(tau)``),
  so only their sum is measurable and the two amplitudes are perfectly
  correlated. (Adding their *logarithms*, as some tools do, multiplies
  the rates, which is not a relaxation model at all.) Use one mechanism,
  or two processes with their own ``tau0``/``Ea``.
* BPP for T2 below the onset of motional narrowing. BPP's ``R2`` keeps
  growing as motion freezes, where the real linewidth plateaus at the
  rigid-lattice value. Fit linewidths there with
  :func:`fit_hendrickson_bray` instead.

Fits are least squares on ``ln R`` (equivalently ``ln T``), weighted by
the relative uncertainties when given, from several deterministic
starting points (the surface has distinct local minima, especially for
multiple processes); distinct solutions that fit almost as well are
reported, with AIC, AICc and BIC for model comparison and an optional
residual bootstrap.

All of this works the same in a notebook and in the GUI
(Dynamics > Relaxation vs temperature), which calls these functions.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "K_B_EV", "vt_dataset", "load_vt_csv", "spectral_density", "relaxation_rate",
    "quadrupolar_prefactor", "VTModel", "fit_vt_relaxation", "evaluate_vt_fit",
    "compare_vt_models", "fit_arrhenius", "hendrickson_bray", "fit_hendrickson_bray",
    "waugh_fedin_Ea", "rate_maximum_temperature", "plot_arrhenius", "plot_vs_temperature",
    "plot_hendrickson_bray", "vt_notebook_code",
]

#: Boltzmann constant in eV/K (CODATA 2018, exact).
K_B_EV = 8.617333262e-5

#: Quantities and how they are named on plots.
QUANTITIES = {
    "T1": ("T_1", "R_1"),
    "T2": ("T_2", "R_2"),
    "T1rho": (r"T_{1\rho}", r"R_{1\rho}"),
    "linewidth": (r"\Delta\nu", r"\Delta\nu"),
}


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def vt_dataset(T_K, value, error=None, larmor_MHz=None, quantity="T1", label=None,
               spinlock_kHz=None, value_is_rate=False, temperature_unit="K",
               value_unit="s") -> dict:
    """One series of relaxation times (or rates, or linewidths) against
    temperature, at one Larmor frequency.

    ``value`` in seconds (``value_unit`` 's' or 'ms'), or s^-1 with
    ``value_is_rate=True``; linewidths in Hz (``value_unit`` 'Hz' or
    'kHz'). ``temperature_unit`` 'K' or 'C'. ``spinlock_kHz`` is needed
    for T1rho. Points with a missing or non-positive value or temperature
    are dropped (and counted in ``dropped``)."""
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {tuple(QUANTITIES)}, not {quantity!r}.")
    T = np.asarray(T_K, dtype=float).ravel()
    if str(temperature_unit).upper().lstrip("°") == "C":
        T = T + 273.15
    v = np.asarray(value, dtype=float).ravel()
    if T.size != v.size:
        raise ValueError(f"{T.size} temperatures but {v.size} values.")
    e = (np.full(v.size, np.nan) if error is None
         else np.asarray(error, dtype=float).ravel())
    if e.size != v.size:
        raise ValueError(f"{v.size} values but {e.size} uncertainties.")
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "Hz": 1.0, "kHz": 1e3}.get(value_unit)
    if scale is None:
        raise ValueError(f"Unknown value unit {value_unit!r}.")
    if quantity == "linewidth" or value_is_rate:
        v, e = v * scale, e * scale
        if value_is_rate and quantity != "linewidth":
            # Store times: a rate's relative error is the time's.
            e = e / v ** 2
            v = 1.0 / v
    else:
        v, e = v * scale, e * scale
    good = np.isfinite(T) & np.isfinite(v) & (T > 0) & (v > 0)
    order = np.argsort(T[good])
    if quantity == "T1rho" and not spinlock_kHz:
        raise ValueError("A T1rho dataset needs the spin-lock field (spinlock_kHz).")
    return {
        "T_K": T[good][order], "value": v[good][order], "error": e[good][order],
        "larmor_MHz": float(larmor_MHz) if larmor_MHz else None, "quantity": quantity,
        "label": label or (f"{float(larmor_MHz):g} MHz" if larmor_MHz else quantity),
        "spinlock_kHz": float(spinlock_kHz) if spinlock_kHz else None,
        "dropped": int((~good).sum()),
    }


def load_vt_csv(path, quantity="T1", larmor_MHz=None, **kwargs) -> list:
    """Datasets from a CSV with columns ``T`` (K), ``value``, and
    optionally ``error``, ``larmor_MHz`` and ``label`` (a header row is
    required; names are case-insensitive, and ``T1``/``T2``/``width`` are
    accepted for ``value``). Rows are grouped into one dataset per
    (label, Larmor frequency)."""
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def col(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_T = col("T", "T_K", "temperature", "temp")
    c_v = col("value", "T1", "T2", "T1rho", "width", "linewidth", "fwhm")
    if c_T is None or c_v is None:
        raise ValueError(f"{path}: need a temperature column (T) and a value column; "
                         f"found {list(df.columns)}.")
    c_e = col("error", "err", "sigma", "uncertainty")
    c_f = col("larmor_MHz", "larmor", "frequency_MHz", "MHz")
    c_l = col("label", "series", "name")
    df = df.copy()
    df["_label"] = df[c_l].astype(str) if c_l else "data"
    df["_larmor"] = df[c_f] if c_f else (larmor_MHz if larmor_MHz else np.nan)
    out = []
    for (label, f), g in df.groupby(["_label", "_larmor"], dropna=False, sort=False):
        f = None if (f is None or (isinstance(f, float) and math.isnan(f))) else float(f)
        out.append(vt_dataset(g[c_T], g[c_v], g[c_e] if c_e else None, larmor_MHz=f,
                              quantity=quantity, label=label if c_l else None, **kwargs))
    return out


# ----------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------

def spectral_density(omega, tau, beta=2.0):
    """``J(w) = tau / (1 + (w tau)^beta)`` (BPP for ``beta = 2``)."""
    tau = np.asarray(tau, dtype=float)
    return tau / (1.0 + np.abs(np.asarray(omega, dtype=float) * tau) ** beta)


def quadrupolar_prefactor(spin, eta=0.0) -> float:
    """``(3 pi^2/50) (2I+3)/(I^2(2I-1)) (1 + eta^2/3)``: multiply by
    ``CQ^2`` (Hz^2) and ``[J(w0) + 4J(2w0)]`` for the quadrupolar R1."""
    I = float(spin)
    if I <= 0.5:
        raise ValueError(f"A quadrupolar mechanism needs spin > 1/2 (got {I:g}).")
    return (3.0 * math.pi ** 2 / 50.0) * (2 * I + 3) / (I ** 2 * (2 * I - 1)) * (1 + eta ** 2 / 3.0)


def _F(quantity, w0, tau, beta, w1=None):
    if quantity == "T1":
        return spectral_density(w0, tau, beta) + 4 * spectral_density(2 * w0, tau, beta)
    if quantity == "T2":
        return 0.5 * (3 * tau + 5 * spectral_density(w0, tau, beta)
                      + 2 * spectral_density(2 * w0, tau, beta))
    if quantity == "T1rho":
        if not w1:
            raise ValueError("T1rho needs the spin-lock frequency.")
        return 0.5 * (3 * spectral_density(2 * w1, tau, beta) + 5 * spectral_density(w0, tau, beta)
                      + 2 * spectral_density(2 * w0, tau, beta))
    raise ValueError(f"No BPP rate for quantity {quantity!r}.")


def correlation_time(T_K, tau0, Ea_eV, law="arrhenius", T0_K=0.0):
    """``tau0 exp(Ea / kB T)``, or VFT ``tau0 exp(Ea / kB (T - T0))``
    (``Ea`` then being the VFT pseudo-activation energy ``B`` in eV)."""
    T = np.asarray(T_K, dtype=float)
    if law == "vft":
        dT = np.maximum(T - float(T0_K), 1e-6)
        x = Ea_eV / (K_B_EV * dT)
    else:
        x = Ea_eV / (K_B_EV * T)
    return tau0 * np.exp(np.clip(x, -700, 700))


def relaxation_rate(T_K, larmor_MHz, tau0, Ea_eV, amplitude, quantity="T1", beta=2.0,
                    spinlock_kHz=None, law="arrhenius", T0_K=0.0):
    """One process's rate (s^-1). ``amplitude`` is the full prefactor
    ``A`` (s^-2) -- ``C`` for a dipolar mechanism, or
    ``quadrupolar_prefactor(I, eta) * CQ_Hz**2``."""
    w0 = 2 * math.pi * float(larmor_MHz) * 1e6
    w1 = 2 * math.pi * float(spinlock_kHz) * 1e3 if spinlock_kHz else None
    tau = correlation_time(T_K, tau0, Ea_eV, law, T0_K)
    return amplitude * _F(quantity, w0, tau, beta, w1)


def rate_maximum_temperature(larmor_MHz, tau0, Ea_eV, beta=2.0, law="arrhenius", T0_K=0.0):
    """Temperature (K) of the R1 maximum: where ``w0 tau_c`` takes the
    value that maximises ``J(w0) + 4 J(2 w0)`` (0.616 for BPP)."""
    from scipy.optimize import minimize_scalar

    w0 = 2 * math.pi * float(larmor_MHz) * 1e6
    res = minimize_scalar(lambda lx: -(spectral_density(1.0, 10 ** lx, beta)
                                       + 4 * spectral_density(2.0, 10 ** lx, beta)),
                          bounds=(-3, 3), method="bounded")
    tau_star = 10 ** res.x / w0
    if tau_star <= tau0:
        return float("nan")
    x = math.log(tau_star / tau0)
    return Ea_eV / (K_B_EV * x) + (T0_K if law == "vft" else 0.0)


# ----------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------

class VTModel:
    """What :func:`fit_vt_relaxation` fits.

    ``mechanism`` 'dipolar' (amplitude ``C``, s^-2) or 'quadrupolar'
    (amplitude ``CQ_MHz``, with ``spin`` and ``eta``); ``n_processes``
    independent motions (rates add); ``asymmetric`` frees ``beta``;
    ``background`` adds a constant rate ``R_bg``; ``law`` 'arrhenius' or
    'vft' (adds a shared ``T0_K``); ``per_dataset_amplitude`` gives each
    dataset its own amplitude in a joint fit."""

    def __init__(self, mechanism="dipolar", n_processes=1, asymmetric=False, background=False,
                 law="arrhenius", spin=None, eta=0.0, per_dataset_amplitude=False):
        if mechanism not in ("dipolar", "quadrupolar"):
            raise ValueError("mechanism must be 'dipolar' or 'quadrupolar'.")
        if mechanism == "quadrupolar" and not spin:
            raise ValueError("A quadrupolar mechanism needs the nuclear spin.")
        if int(n_processes) < 1:
            raise ValueError("n_processes must be at least 1.")
        if law not in ("arrhenius", "vft"):
            raise ValueError("law must be 'arrhenius' or 'vft'.")
        self.mechanism = mechanism
        self.n_processes = int(n_processes)
        self.asymmetric = bool(asymmetric)
        self.background = bool(background)
        self.law = law
        self.spin = float(spin) if spin else None
        self.eta = float(eta)
        self.per_dataset_amplitude = bool(per_dataset_amplitude)

    @property
    def amp_name(self):
        return "C" if self.mechanism == "dipolar" else "CQ_MHz"

    def describe(self) -> str:
        parts = ["BPP" if not self.asymmetric else "asymmetric BPP (free beta)",
                 "dipolar" if self.mechanism == "dipolar"
                 else f"quadrupolar (I = {self.spin:g}, eta = {self.eta:g})"]
        if self.n_processes > 1:
            parts.append(f"{self.n_processes} independent processes")
        if self.law == "vft":
            parts.append("VFT correlation time")
        if self.background:
            parts.append("+ constant background rate")
        return ", ".join(parts)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("mechanism", "n_processes", "asymmetric",
                                               "background", "law", "spin", "eta",
                                               "per_dataset_amplitude")}

    def param_names(self, n_datasets=1):
        """Parameter names, in order. Suffix ``_k`` numbers the process
        (only when there is more than one); ``@d`` the dataset for
        per-dataset amplitudes."""
        names = []
        for k in range(self.n_processes):
            sfx = f"_{k + 1}" if self.n_processes > 1 else ""
            if self.per_dataset_amplitude and n_datasets > 1:
                names += [f"{self.amp_name}{sfx}@{d + 1}" for d in range(n_datasets)]
            else:
                names.append(f"{self.amp_name}{sfx}")
            names += [f"tau0_s{sfx}", f"Ea_eV{sfx}"]
            if self.asymmetric:
                names.append(f"beta{sfx}")
        if self.law == "vft":
            names.append("T0_K")
        if self.background:
            if self.per_dataset_amplitude and n_datasets > 1:
                names += [f"R_bg@{d + 1}" for d in range(n_datasets)]
            else:
                names.append("R_bg")
        return names


def _amplitude(model, value):
    """Prefactor A (s^-2) from a fitted amplitude parameter."""
    if model.mechanism == "dipolar":
        return value
    return quadrupolar_prefactor(model.spin, model.eta) * (value * 1e6) ** 2


def _process_params(model, params, d, k):
    sfx = f"_{k + 1}" if model.n_processes > 1 else ""
    key = f"{model.amp_name}{sfx}"
    amp = params.get(f"{key}@{d + 1}", params.get(key))
    return (amp, params[f"tau0_s{sfx}"], params[f"Ea_eV{sfx}"],
            params.get(f"beta{sfx}", 2.0))


def model_rate(model, params, dataset, T_K, d=0, components=False):
    """Total rate (s^-1) of ``model`` for ``dataset`` (index ``d``) at
    ``T_K``; with ``components``, also each process's rate and the
    background."""
    q = dataset["quantity"]
    total = np.zeros_like(np.asarray(T_K, dtype=float))
    parts = []
    for k in range(model.n_processes):
        amp, tau0, Ea, beta = _process_params(model, params, d, k)
        r = relaxation_rate(T_K, dataset["larmor_MHz"], tau0, Ea, _amplitude(model, amp), q,
                            beta, dataset.get("spinlock_kHz"), model.law, params.get("T0_K", 0.0))
        parts.append(r)
        total = total + r
    if model.background:
        bg = params.get(f"R_bg@{d + 1}", params.get("R_bg", 0.0))
        total = total + bg
        parts.append(np.full_like(total, bg))
    return (total, parts) if components else total


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------

def _info_criteria(chi2, n, k, known_sigma):
    if known_sigma:
        base = chi2
    else:
        base = n * math.log(max(chi2, 1e-300) / n)
    aic = base + 2 * k
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else float("inf")
    bic = base + k * math.log(n)
    return aic, aicc, bic


def _initial_guess(model, datasets):
    """Heuristic start: the hottest rate maximum gives tau_c = 0.616/w0
    there, flanks give Ea."""
    d0 = datasets[0]
    T = d0["T_K"]
    R = 1.0 / d0["value"]
    i = int(np.argmax(R))
    w0 = 2 * math.pi * d0["larmor_MHz"] * 1e6
    Ea = 0.3
    if T.size >= 3:
        x = 1.0 / (K_B_EV * T)
        lo = slice(0, max(i + 1, 2))
        hi = slice(min(i, T.size - 2), T.size)
        slopes = []
        for sl in (lo, hi):
            if x[sl].size >= 2 and np.ptp(x[sl]) > 0:
                slopes.append(abs(np.polyfit(x[sl], np.log(R[sl]), 1)[0]))
        if slopes:
            Ea = float(np.clip(max(slopes), 0.03, 2.0))
    T_peak = T[i]
    tau_peak = 0.616 / w0
    tau0 = tau_peak * math.exp(-Ea / (K_B_EV * T_peak))
    A = R[i] * w0 / 1.43 if d0["quantity"] == "T1" else R[i] / max(tau_peak, 1e-30)
    return Ea, tau0, A, T_peak


def _amp_param_from_A(model, A):
    if model.mechanism == "dipolar":
        return A
    return math.sqrt(max(A, 1e-300) / quadrupolar_prefactor(model.spin, model.eta)) / 1e6


def _start_points(model, datasets, n_starts):
    """Deterministic starting points: the heuristic guess, then a
    Sobol set over (temperature of each process's maximum, Ea)."""
    from scipy.stats import qmc

    Ea_g, tau0_g, A_g, T_peak = _initial_guess(model, datasets)
    T_all = np.concatenate([d["T_K"] for d in datasets])
    T_lo, T_hi = float(T_all.min()), float(T_all.max())
    span = max(T_hi - T_lo, 1.0)
    w0 = 2 * math.pi * datasets[0]["larmor_MHz"] * 1e6
    n_p = model.n_processes
    R_bg_g = float(np.min(1.0 / np.concatenate([d["value"] for d in datasets]))) * 0.1
    starts = []

    def make(peaks, Eas, amps):
        p = {}
        for k in range(n_p):
            sfx = f"_{k + 1}" if n_p > 1 else ""
            tau0 = (0.616 / w0) * math.exp(-Eas[k] / (K_B_EV * peaks[k]))
            if model.law == "vft":
                tau0 = (0.616 / w0) * math.exp(-Eas[k] / (K_B_EV * max(peaks[k] - 0.5 * T_lo, 1.0)))
            p[f"{model.amp_name}{sfx}"] = _amp_param_from_A(model, amps[k])
            p[f"tau0_s{sfx}"] = tau0
            p[f"Ea_eV{sfx}"] = Eas[k]
            if model.asymmetric:
                p[f"beta{sfx}"] = 1.8
        if model.law == "vft":
            p["T0_K"] = 0.5 * T_lo
        if model.background:
            p["R_bg"] = R_bg_g
        return p

    if n_p == 1:
        starts.append(make([T_peak], [Ea_g], [A_g]))
    else:
        peaks = list(np.linspace(T_lo + 0.15 * span, T_hi - 0.15 * span, n_p))
        starts.append(make(peaks, [Ea_g] * n_p, [A_g / n_p] * n_p))
    m = max(1, int(math.ceil(math.log2(max(n_starts, 2)))))
    u = qmc.Sobol(2 * n_p, scramble=True, seed=1234).random_base2(m)[: max(n_starts - 1, 0)]
    for row in u:
        peaks = [T_lo - 0.3 * span + 1.6 * span * row[2 * k] for k in range(n_p)]
        peaks = [max(p, 5.0) for p in peaks]
        Eas = [0.05 * (30.0 ** row[2 * k + 1]) for k in range(n_p)]  # 0.05 - 1.5 eV, log-spaced
        starts.append(make(peaks, Eas, [A_g / n_p] * n_p))
    return starts


def _canonical_order(model, free, x):
    """``x`` with whole processes permuted into increasing Ea (only
    possible when every process's parameters are all free or all
    fixed alike; otherwise left as it is)."""
    n_p = model.n_processes
    groups = []
    for k in range(n_p):
        sfx = f"_{k + 1}"
        groups.append([i for i, n in enumerate(free)
                       if n.split("@")[0].endswith(sfx) and not n.startswith(("T0_K", "R_bg"))])
    if len({len(g) for g in groups}) != 1 or not groups[0]:
        return x
    stems = [[free[i].split("@")[0][: -len(f"_{k + 1}")] + ("@" + free[i].split("@")[1]
              if "@" in free[i] else "") for i in g] for k, g in enumerate(groups)]
    if any(st != stems[0] for st in stems):
        return x
    ea_idx = [next((i for i in g if free[i].startswith("Ea_eV")), None) for g in groups]
    if any(i is None for i in ea_idx):
        return x
    order = np.argsort([x[i] for i in ea_idx])
    y = np.array(x, dtype=float)
    for new_k, old_k in enumerate(order):
        for a, b in zip(groups[new_k], groups[old_k]):
            y[a] = x[b]
    return y


def _pack(model, names, params, n_datasets):
    """Internal (free-search) vector: log10 for positive scale
    parameters, linear for Ea/beta/T0."""
    out = []
    for name in names:
        base = name.split("@")[0]
        v = params.get(name, params.get(base))
        out.append(math.log10(v) if _is_log(base) else v)
    return np.array(out, dtype=float)


def _is_log(base):
    return base.startswith(("C", "tau0_s", "R_bg"))


def _bounds(names, T_min):
    lo, hi = [], []
    for name in names:
        base = name.split("@")[0]
        if base.startswith("tau0_s"):
            lo.append(-25.0), hi.append(-3.0)
        elif base.startswith("Ea_eV"):
            lo.append(0.005), hi.append(5.0)
        elif base.startswith("beta"):
            lo.append(1.0 + 1e-3), hi.append(2.0)
        elif base == "T0_K":
            lo.append(0.0), hi.append(max(T_min - 1.0, 1.0))
        elif base.startswith("R_bg"):
            lo.append(-12.0), hi.append(8.0)
        elif base.startswith("CQ_MHz"):
            lo.append(-4.0), hi.append(3.0)
        else:  # C
            lo.append(-5.0), hi.append(25.0)
    return np.array(lo), np.array(hi)


def _unpack(names, x):
    out = {}
    for name, v in zip(names, x):
        base = name.split("@")[0]
        out[name] = 10.0 ** v if _is_log(base) else float(v)
    return out


def fit_vt_relaxation(datasets, mechanism="dipolar", n_processes=1, asymmetric=False,
                      background=False, law="arrhenius", spin=None, eta=0.0,
                      per_dataset_amplitude=False, fixed=None, p0=None, n_starts=24,
                      n_bootstrap=0, T_range=None, model: VTModel | None = None,
                      seed=0) -> dict:
    """Fit relaxation times against temperature with a BPP-type model.

    Parameters
    ----------
    datasets : list of :func:`vt_dataset` dicts (or one). All must be
        the same quantity (T1, T2 or T1rho) and give ``larmor_MHz``;
        several fields are fitted jointly.
    mechanism, n_processes, asymmetric, background, law, spin, eta,
    per_dataset_amplitude : see :class:`VTModel` (or pass ``model``).
    fixed : dict of parameter -> value held fixed (names from
        ``VTModel.param_names``, e.g. ``{"beta": 2.0}`` -- already fixed
        unless ``asymmetric`` -- or ``{"Ea_eV": 0.3}``).
    p0 : dict of starting values; replaces the first automatic start.
    n_starts : deterministic starting points (local minima are common).
    n_bootstrap : residual-bootstrap refits for 16th-84th percentile
        intervals (0 = off; 200 is a reasonable number).
    T_range : (T_min, T_max) in K to restrict the fit to.

    Returns
    -------
    dict with ``params`` and ``errors`` (1-sigma from the covariance),
    ``ci`` (bootstrap 16-84 % intervals, if asked), ``model`` (dict),
    ``datasets`` (the fitted data), ``chi2``, ``redchi``, ``aic``,
    ``aicc``, ``bic``, ``n_points``, ``n_free``, ``rivals`` (other
    distinct minima within 2 of the best AICc), ``T_max`` (temperature of
    each process's rate maximum, per dataset), ``correlation``
    (free-parameter correlation matrix) and ``log``.
    """
    if isinstance(datasets, dict):
        datasets = [datasets]
    datasets = [dict(d) for d in datasets]
    if not datasets:
        raise ValueError("Give at least one dataset.")
    quantities = {d["quantity"] for d in datasets}
    if len(quantities) != 1:
        raise ValueError(f"All datasets must be the same quantity; got {sorted(quantities)}.")
    quantity = quantities.pop()
    if quantity == "linewidth":
        raise ValueError("Linewidths are fitted with fit_hendrickson_bray, not a BPP model.")
    for d in datasets:
        if not d.get("larmor_MHz"):
            raise ValueError(f"Dataset {d['label']!r} has no Larmor frequency; BPP needs it.")
    if T_range is not None:
        lo_T, hi_T = sorted(map(float, T_range))
        for d in datasets:
            keep = (d["T_K"] >= lo_T) & (d["T_K"] <= hi_T)
            for key in ("T_K", "value", "error"):
                d[key] = d[key][keep]
    model = model or VTModel(mechanism, n_processes, asymmetric, background, law, spin, eta,
                             per_dataset_amplitude)
    names = model.param_names(len(datasets))
    fixed = dict(fixed or {})
    if not model.asymmetric:
        pass  # beta is not a parameter; spectral_density defaults to 2
    unknown = [k for k in fixed if k not in names]
    if unknown:
        raise ValueError(f"Cannot fix {unknown}: this model's parameters are {names}.")
    free = [n for n in names if n not in fixed]
    n_pts = int(sum(d["T_K"].size for d in datasets))
    if n_pts <= len(free):
        raise ValueError(f"{n_pts} points cannot determine {len(free)} free parameters "
                         f"({', '.join(free)}). Fix some, or use a simpler model.")

    lnR = np.concatenate([np.log(1.0 / d["value"]) for d in datasets])
    rel = np.concatenate([d["error"] / d["value"] for d in datasets])
    known_sigma = bool(np.all(np.isfinite(rel)) and np.all(rel > 0))
    sig = rel if known_sigma else np.ones_like(lnR)
    T_min = float(min(d["T_K"].min() for d in datasets))
    lo, hi = _bounds(free, T_min)

    def full_params(x):
        p = _unpack(free, x)
        p.update(fixed)
        return p

    def residuals(x, target=lnR):
        p = full_params(x)
        model_ln = np.concatenate([np.log(np.maximum(model_rate(model, p, d, d["T_K"], i), 1e-300))
                                   for i, d in enumerate(datasets)])
        return (model_ln - target) / sig

    starts = _start_points(model, datasets, n_starts)
    if p0:
        first = dict(starts[0])
        first.update(p0)
        starts[0] = first
    solutions = []
    for s in starts:
        s = dict(s)
        if model.per_dataset_amplitude and len(datasets) > 1:
            for n in names:
                if "@" in n:
                    s.setdefault(n, s.get(n.split("@")[0]))
        x0 = np.clip(_pack(model, free, s, len(datasets)), lo + 1e-9, hi - 1e-9)
        try:
            sol = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac", max_nfev=4000)
        except (ValueError, FloatingPointError):
            continue
        if np.all(np.isfinite(sol.fun)):
            solutions.append(sol)
    if not solutions:
        raise RuntimeError("Every start failed; check the data and Larmor frequencies.")
    if model.n_processes > 1:
        # Processes are interchangeable: order each solution by Ea so
        # that a relabelled copy of the same minimum is not reported as
        # a rival, and process 1 is always the lowest barrier.
        for sol in solutions:
            sol.x = _canonical_order(model, free, sol.x)
    solutions.sort(key=lambda s: float(np.sum(s.fun ** 2)))
    best = solutions[0]
    chi2 = float(np.sum(best.fun ** 2))
    k = len(free)
    dof = max(n_pts - k, 1)
    redchi = chi2 / dof
    aic, aicc, bic = _info_criteria(chi2, n_pts, k, known_sigma)

    # Covariance in the internal (log) space, then to the reported one.
    params = full_params(best.x)
    errors = {n: float("nan") for n in names}
    corr = None
    try:
        J = best.jac
        cov = np.linalg.pinv(J.T @ J)
        if not known_sigma:
            cov = cov * redchi
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(sd, sd)
        for n, x, e in zip(free, best.x, sd):
            base = n.split("@")[0]
            errors[n] = float(10 ** x * math.log(10) * e) if _is_log(base) else float(e)
    except np.linalg.LinAlgError:
        pass

    # Distinct rival minima: as good within 2 AICc, but a different Ea.
    rivals = []
    for sol in solutions[1:]:
        c2 = float(np.sum(sol.fun ** 2))
        _a, ac, _b = _info_criteria(c2, n_pts, k, known_sigma)
        if ac - aicc > 2.0:
            break
        p = full_params(sol.x)
        if any(abs(p[n] - params[n]) > 0.02 * max(abs(params[n]), 1e-9)
               for n in free if n.split("@")[0].startswith("Ea_eV")):
            if not any(all(abs(p[n] - r["params"][n]) <= 0.02 * max(abs(p[n]), 1e-9)
                           for n in free if n.startswith("Ea")) for r in rivals):
                rivals.append({"params": p, "delta_aicc": ac - aicc, "chi2": c2})

    ci = {}
    if n_bootstrap and n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        fit_ln = lnR + best.fun * sig
        res = best.fun * sig
        draws = []
        for _ in range(int(n_bootstrap)):
            target = fit_ln + rng.choice(res, size=res.size, replace=True)
            try:
                sb = least_squares(lambda x: residuals(x, target), best.x, bounds=(lo, hi),
                                   x_scale="jac", max_nfev=2000)
                draws.append(full_params(sb.x))
            except (ValueError, FloatingPointError):
                continue
        for n in free:
            vals = np.array([dct[n] for dct in draws])
            if vals.size:
                ci[n] = (float(np.percentile(vals, 16)), float(np.percentile(vals, 84)))

    T_max = []
    for i, d in enumerate(datasets):
        row = []
        for kk in range(model.n_processes):
            _amp, tau0, Ea, beta = _process_params(model, params, i, kk)
            row.append(rate_maximum_temperature(d["larmor_MHz"], tau0, Ea, beta, model.law,
                                                params.get("T0_K", 0.0))
                       if quantity == "T1" else float("nan"))
        T_max.append(row)

    result = {
        "params": params, "errors": errors, "ci": ci, "free": free, "fixed": fixed,
        "model": model.as_dict(), "model_text": model.describe(), "quantity": quantity,
        "datasets": datasets, "chi2": chi2, "redchi": redchi, "aic": aic, "aicc": aicc,
        "bic": bic, "n_points": n_pts, "n_free": k, "known_sigma": known_sigma,
        "rivals": rivals, "T_max": T_max, "correlation": corr, "success": bool(best.success),
        "n_starts": len(starts),
    }
    result["log"] = _vt_report(result)
    return result


def _fmt(v, e=None):
    if e is None or not np.isfinite(e) or e <= 0:
        return f"{v:.4g}"
    exponent = int(math.floor(math.log10(e)))
    leading = e / 10 ** exponent
    # Two significant figures in the uncertainty when it starts with 1 or
    # 2, one otherwise (the usual rounding convention).
    digits = max(0, -exponent + (1 if leading < 3 else 0))
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.3e} ± {e:.1e}"
    return f"{v:.{digits}f} ± {e:.{max(digits, 0)}f}" if digits < 8 else f"{v:.4g} ± {e:.2g}"


def _vt_report(r) -> str:
    q = r["quantity"]
    lines = [f"{q} against temperature: {r['model_text']}.",
             f"{len(r['datasets'])} dataset(s): " + "; ".join(
                 f"{d['label']} ({d['larmor_MHz']:g} MHz, {d['T_K'].size} points, "
                 f"{d['T_K'].min():.0f}-{d['T_K'].max():.0f} K)" for d in r["datasets"]),
             ("Weighted by the given uncertainties" if r["known_sigma"]
              else "No uncertainties given: unweighted, errors scaled by the scatter")
             + f"; best of {r['n_starts']} starting points.",
             f"chi2 = {r['chi2']:.4g}, reduced chi2 = {r['redchi']:.3g}, "
             f"AICc = {r['aicc']:.2f}, BIC = {r['bic']:.2f} (n = {r['n_points']}, "
             f"k = {r['n_free']})."]
    for n, v in r["params"].items():
        tag = " (fixed)" if n in r["fixed"] else ""
        line = f"  {n} = {_fmt(v, r['errors'].get(n))}{tag}"
        if n in r["ci"]:
            line += f"   [bootstrap 16-84 %: {r['ci'][n][0]:.4g} to {r['ci'][n][1]:.4g}]"
        if n.startswith("Ea_eV"):
            line += f"   ({v * 96.485:.1f} kJ/mol)"
        lines.append(line)
    for d, row in zip(r["datasets"], r["T_max"]):
        good = [t for t in row if np.isfinite(t)]
        if good:
            lines.append(f"  {q} minimum (rate maximum) for {d['label']}: "
                         + ", ".join(f"{t:.0f} K" for t in good))
            outside = [t for t in good if not d["T_K"].min() <= t <= d["T_K"].max()]
            if outside:
                lines.append("    (" + ", ".join(f"{t:.0f} K" for t in outside)
                             + " is outside the measured range: only one flank of that "
                             "process is seen, so its amplitude and tau0 are extrapolated "
                             "and correlated.)")
    if r["model"]["asymmetric"]:
        lines.append("  Low-temperature flank slope = (beta - 1) Ea; high-temperature = Ea.")
    if r["correlation"] is not None and len(r["free"]) > 1:
        c = np.array(r["correlation"])
        iu = np.triu_indices_from(c, 1)
        worst = np.nanargmax(np.abs(c[iu])) if c[iu].size else None
        if worst is not None and np.isfinite(c[iu][worst]) and abs(c[iu][worst]) > 0.95:
            a, b = r["free"][iu[0][worst]], r["free"][iu[1][worst]]
            lines.append(f"  Note: {a} and {b} are strongly correlated (r = {c[iu][worst]:.3f}); "
                         "the data constrain a combination of them better than either.")
    if r["rivals"]:
        lines.append(f"  {len(r['rivals'])} other distinct solution(s) fit nearly as well "
                     "(within 2 AICc):")
        for rv in r["rivals"][:5]:
            Eas = ", ".join(f"{n} = {v:.3g}" for n, v in rv["params"].items() if n.startswith("Ea"))
            lines.append(f"    dAICc = {rv['delta_aicc']:.2f}: {Eas}")
    if r["redchi"] > 4 and r["known_sigma"]:
        lines.append("  Reduced chi2 well above 1: the model misses the data by more than "
                     "the uncertainties allow (another process? asymmetric flanks?).")
    return "\n".join(lines)


def evaluate_vt_fit(fit, T_K=None, n_points=300, dataset=0, components=False):
    """The fitted curve for one dataset: ``(T_K, rate)`` (and the
    per-process rates with ``components``), on ``n_points`` evenly
    spaced in 1/T across the data unless ``T_K`` is given."""
    model = VTModel(**fit["model"])
    d = fit["datasets"][dataset]
    if T_K is None:
        x = 1.0 / d["T_K"]
        pad = 0.05 * (x.max() - x.min() if x.size > 1 else x.max())
        T_K = 1.0 / np.linspace(x.max() + pad, max(x.min() - pad, 1e-6), n_points)
    T_K = np.asarray(T_K, dtype=float)
    out = model_rate(model, fit["params"], d, T_K, dataset, components=components)
    return (T_K,) + (out if components else (out,))


def compare_vt_models(datasets, models, **kwargs) -> list:
    """Fit each of ``models`` (dicts of :func:`fit_vt_relaxation`
    keywords, e.g. ``[{}, {"asymmetric": True}, {"n_processes": 2}]``)
    and return the fits sorted by AICc, each with ``delta_aicc`` and an
    Akaike weight ``weight``."""
    fits = []
    for m in models:
        kw = dict(kwargs)
        kw.update(m)
        try:
            fits.append(fit_vt_relaxation(datasets, **kw))
        except (ValueError, RuntimeError) as exc:
            fits.append({"model_text": str(m), "error": str(exc), "aicc": float("inf")})
    fits.sort(key=lambda f: f["aicc"])
    best = fits[0]["aicc"]
    ws = np.array([math.exp(-0.5 * (f["aicc"] - best)) if np.isfinite(f["aicc"]) else 0.0
                   for f in fits])
    ws = ws / ws.sum() if ws.sum() > 0 else ws
    for f, w in zip(fits, ws):
        f["delta_aicc"] = f["aicc"] - best
        f["weight"] = float(w)
    return fits


def fit_arrhenius(T_K, value, error=None, is_rate=False, T_range=None) -> dict:
    """Straight line through ``ln(rate)`` against ``1/kB T``: the
    activation energy from one flank of a BPP curve (or from any rate:
    hop rates, conductivity x T, ...).

    ``value`` is a time (rate = 1/value) unless ``is_rate``. Returns
    ``Ea_eV`` (the magnitude of the slope) with its 1-sigma error,
    ``ln_A`` (intercept), ``slope``, ``flank`` ('low-temperature' when
    the rate rises with temperature -- the slow-motion side of a T1
    minimum -- or 'high-temperature') and ``log``."""
    T = np.asarray(T_K, dtype=float)
    v = np.asarray(value, dtype=float)
    keep = np.isfinite(T) & np.isfinite(v) & (T > 0) & (v > 0)
    if T_range is not None:
        a, b = sorted(map(float, T_range))
        keep &= (T >= a) & (T <= b)
    if keep.sum() < 2:
        raise ValueError("An Arrhenius line needs at least two points in range.")
    T, v = T[keep], v[keep]
    rate = v if is_rate else 1.0 / v
    x = 1.0 / (K_B_EV * T)
    y = np.log(rate)
    w = None
    if error is not None:
        e = np.asarray(error, dtype=float)[keep]
        if np.all(np.isfinite(e)) and np.all(e > 0):
            w = v / e  # 1/sigma of ln
    if keep.sum() == 2:
        slope, intercept = np.polyfit(x, y, 1)
        cov = np.full((2, 2), np.nan)
    else:
        (slope, intercept), cov = np.polyfit(x, y, 1, w=w, cov="unscaled" if w is not None else True)
    e_slope = float(np.sqrt(cov[0, 0])) if np.all(np.isfinite(cov)) else float("nan")
    flank = "low-temperature" if slope < 0 else "high-temperature"
    res = {"Ea_eV": float(abs(slope)), "Ea_err_eV": e_slope, "ln_A": float(intercept),
           "slope": float(slope), "flank": flank, "T_K": T, "rate": rate,
           "n_points": int(T.size)}
    res["log"] = (f"Arrhenius line through {T.size} points, {T.min():.0f}-{T.max():.0f} K: "
                  f"Ea = {_fmt(abs(slope), e_slope)} eV ({abs(slope) * 96.485:.1f} kJ/mol), "
                  f"ln A = {intercept:.3f}; the rate {'rises' if slope < 0 else 'falls'} with "
                  f"temperature ({flank} flank of a T1 minimum).")
    return res


# ----------------------------------------------------------------------
# Hendrickson-Bray
# ----------------------------------------------------------------------

def hendrickson_bray(T_K, A, B, Ea_eV, D):
    """Hendrickson & Bray (J. Magn. Reson. 9, 341 (1973)) motional
    narrowing: ``dnu(T) = A / (1 + (A/B - 1) exp(-Ea/kB T)) + D``.

    ``A + D`` is the rigid-lattice (low-temperature) width, ``D`` the
    fully narrowed (high-temperature) width, and ``B`` the width the
    exponential term would reach at infinite temperature (it sets where
    narrowing starts; ``B << A``)."""
    T = np.asarray(T_K, dtype=float)
    return A / (1.0 + (A / B - 1.0) * np.exp(np.clip(-Ea_eV / (K_B_EV * T), -700, 700))) + D


def waugh_fedin_Ea(T_onset_K) -> float:
    """Waugh-Fedin estimate ``Ea [eV] = 1.617e-3 T_onset [K]``
    (``Ea = 0.156 T`` kJ/mol), from the temperature at which motional
    narrowing begins. A rough rule of thumb, not a fit."""
    return 1.617e-3 * float(T_onset_K)


def fit_hendrickson_bray(T_K, width_Hz, error=None, fixed=None, p0=None, n_starts=24) -> dict:
    """Fit linewidths against temperature with :func:`hendrickson_bray`.

    ``fixed`` / ``p0`` : dicts over ``A``, ``B``, ``Ea_eV``, ``D``.
    Returns ``params``, ``errors``, ``redchi``, ``aicc``, ``T_onset_K``
    (where the width has fallen 10 % of the way from rigid-lattice to
    narrowed), ``T_inflection_K`` (half-way), ``waugh_fedin_Ea_eV`` (the
    rule of thumb at ``T_onset_K``, for comparison) and ``log``."""
    from scipy.stats import qmc

    T = np.asarray(T_K, dtype=float)
    y = np.asarray(width_Hz, dtype=float)
    keep = np.isfinite(T) & np.isfinite(y) & (T > 0)
    T, y = T[keep], y[keep]
    e = None
    if error is not None:
        e = np.asarray(error, dtype=float)[keep]
        if not (np.all(np.isfinite(e)) and np.all(e > 0)):
            e = None
    names = ["A", "B", "Ea_eV", "D"]
    fixed = dict(fixed or {})
    free = [n for n in names if n not in fixed]
    if T.size <= len(free):
        raise ValueError(f"{T.size} points cannot determine {len(free)} free parameters; "
                         "the Hendrickson-Bray equation has four (A, B, Ea, D).")
    order = np.argsort(T)
    T, y = T[order], y[order]
    if e is not None:
        e = e[order]
    sig = e if e is not None else np.ones_like(y)
    D0 = float(np.min(y))
    A0 = max(float(np.max(y)) - D0, 1e-6)
    # internal: log10 A, log10 B, Ea, D
    lo = {"A": -6, "B": -40, "Ea_eV": 0.005, "D": -abs(A0) * 10}
    hi = {"A": 12, "B": 12, "Ea_eV": 5.0, "D": float(np.max(y)) * 10 + 1}

    def to_int(n, v):
        return math.log10(v) if n in ("A", "B") else v

    def from_int(n, v):
        return 10 ** v if n in ("A", "B") else v

    def full(x):
        p = {n: from_int(n, v) for n, v in zip(free, x)}
        p.update(fixed)
        return p

    def resid(x):
        p = full(x)
        return (hendrickson_bray(T, p["A"], p["B"], p["Ea_eV"], p["D"]) - y) / sig

    lo_v = np.array([lo[n] for n in free], float)
    hi_v = np.array([hi[n] for n in free], float)
    starts = []
    base = {"A": A0, "B": A0 * 1e-3, "Ea_eV": 0.3, "D": D0}
    if p0:
        base.update(p0)
    starts.append(base)
    # Sobol over (Ea, the temperature of the narrowing midpoint): the
    # midpoint T_m satisfies (A/B - 1) exp(-Ea/kB T_m) = 1.
    u = qmc.Sobol(2, scramble=True, seed=7).random_base2(5)[: max(n_starts - 1, 0)]
    for a, b in u:
        Ea = 0.03 * (50 ** a)
        Tm = T.min() + b * (T.max() - T.min())
        B = A0 / (1 + math.exp(min(Ea / (K_B_EV * Tm), 600)))
        starts.append({"A": A0, "B": max(B, 1e-12), "Ea_eV": Ea, "D": D0})
    sols = []
    for s in starts:
        x0 = np.clip(np.array([to_int(n, s[n]) for n in free]), lo_v + 1e-9, hi_v - 1e-9)
        try:
            sols.append(least_squares(resid, x0, bounds=(lo_v, hi_v), x_scale="jac",
                                      max_nfev=4000))
        except (ValueError, FloatingPointError):
            pass
    if not sols:
        raise RuntimeError("Every start failed.")
    best = min(sols, key=lambda s: float(np.sum(s.fun ** 2)))
    p = full(best.x)
    chi2 = float(np.sum(best.fun ** 2))
    k = len(free)
    dof = max(T.size - k, 1)
    redchi = chi2 / dof
    errors = {n: float("nan") for n in names}
    try:
        cov = np.linalg.pinv(best.jac.T @ best.jac) * (1.0 if e is not None else redchi)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        for n, x, s_ in zip(free, best.x, sd):
            errors[n] = 10 ** x * math.log(10) * s_ if n in ("A", "B") else float(s_)
    except np.linalg.LinAlgError:
        pass
    aic, aicc, bic = _info_criteria(chi2, T.size, k, e is not None)
    ratio = p["A"] / p["B"] - 1.0

    def T_at(frac):
        # A/(1+r exp(-Ea/kT)) = (1-frac) A  ->  r exp(-Ea/kT) = frac/(1-frac)
        if ratio <= 0:
            return float("nan")
        val = (frac / (1 - frac)) / ratio
        return p["Ea_eV"] / (K_B_EV * -math.log(val)) if 0 < val < 1 else float("nan")

    T_on, T_mid = T_at(0.1), T_at(0.5)
    res = {"params": p, "errors": errors, "chi2": chi2, "redchi": redchi, "aic": aic,
           "aicc": aicc, "bic": bic, "T_K": T, "width_Hz": y, "error": e, "fixed": fixed,
           "T_onset_K": T_on, "T_inflection_K": T_mid,
           "waugh_fedin_Ea_eV": waugh_fedin_Ea(T_on) if np.isfinite(T_on) else float("nan"),
           "success": bool(best.success)}
    lines = [f"Hendrickson-Bray fit of {T.size} linewidths, {T.min():.0f}-{T.max():.0f} K "
             + ("(weighted)" if e is not None else "(unweighted)") + ":",
             f"  Ea = {_fmt(p['Ea_eV'], errors['Ea_eV'])} eV ({p['Ea_eV'] * 96.485:.1f} kJ/mol)",
             f"  rigid-lattice width A + D = {p['A'] + p['D']:.4g} Hz; narrowed width D = "
             f"{_fmt(p['D'], errors['D'])} Hz; B = {p['B']:.3g} Hz",
             f"  narrowing onset (10 %) {T_on:.0f} K, midpoint {T_mid:.0f} K; reduced chi2 = "
             f"{redchi:.3g}"]
    if np.isfinite(res["waugh_fedin_Ea_eV"]):
        lines.append(f"  Waugh-Fedin rule of thumb at the onset: Ea ~ "
                     f"{res['waugh_fedin_Ea_eV']:.3f} eV (an estimate, for comparison only)")
    for n in fixed:
        lines.append(f"  {n} held at {fixed[n]:g}")
    res["log"] = "\n".join(lines)
    return res


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def _inverse(scale):
    """``x -> scale / x``, safe at x <= 0 (self-inverse)."""
    def f(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, scale / np.where(x > 0, x, 1.0), np.inf)
    return f


def _x_transform(x_axis):
    if x_axis == "1000/T":
        return (lambda T: 1000.0 / np.asarray(T, float)), _inverse(1000.0), r"1000/$T$ (K$^{-1}$)"
    if x_axis == "1/kBT":
        return ((lambda T: 1.0 / (K_B_EV * np.asarray(T, float))), _inverse(1.0 / K_B_EV),
                r"1/$k_\mathregular{B}T$ (eV$^{-1}$)")
    if x_axis == "T":
        return (lambda T: np.asarray(T, float)), (lambda x: x), r"$T$ (K)"
    raise ValueError("x_axis must be '1000/T', '1/kBT' or 'T'.")


def _y_transform(y_axis, quantity):
    sym_t, sym_r = QUANTITIES[quantity]
    if y_axis == "ln_time":
        return (lambda t: np.log(t)), rf"ln(${sym_t}$ / s)", False
    if y_axis == "ln_rate":
        return (lambda t: np.log(1.0 / t)), rf"ln(${sym_r}$ / s$^{{-1}}$)", False
    if y_axis == "log10_time":
        return (lambda t: np.log10(t)), rf"log$_{{10}}$(${sym_t}$ / s)", False
    if y_axis == "log10_rate":
        return (lambda t: np.log10(1.0 / t)), rf"log$_{{10}}$(${sym_r}$ / s$^{{-1}}$)", False
    if y_axis == "time":
        return (lambda t: t), rf"${sym_t}$ (s)", True
    if y_axis == "rate":
        return (lambda t: 1.0 / t), rf"${sym_r}$ (s$^{{-1}}$)", True
    raise ValueError(f"Unknown y_axis {y_axis!r}.")


def _add_top_axis(ax, x_axis, top_axis):
    if top_axis in (None, "none", "") or x_axis == "T":
        return None
    if x_axis == "1000/T":
        to_T = _inverse(1000.0)
    else:
        to_T = _inverse(1.0 / K_B_EV)
    if top_axis in ("C", "°C"):
        def fwd(x):
            return to_T(x) - 273.15

        def inv(tc):
            return to_T(np.asarray(tc, float) + 273.15)
        label = r"$T$ (°C)"
    else:
        fwd, inv = to_T, to_T
        label = r"$T$ (K)"
    sec = ax.secondary_xaxis("top", functions=(fwd, inv))
    sec.set_xlabel(label)
    # Round-number temperatures across the visible range.
    lo, hi = sorted(ax.get_xlim())
    Ts = np.sort(fwd(np.array([max(lo, 1e-9), hi])))
    step = _nice_step(Ts[1] - Ts[0])
    ticks = np.arange(math.ceil(Ts[0] / step) * step, Ts[1] + 0.5 * step, step)
    sec.set_xticks(ticks)
    return sec


def _nice_step(span, target=5):
    raw = max(span, 1e-9) / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _style_axes(ax):
    try:
        from .plotting import apply_tick_style
        apply_tick_style(ax)
    except Exception:  # noqa: BLE001 -- styling is cosmetic
        pass


def plot_arrhenius(fit=None, datasets=None, y_axis="ln_time", x_axis="1000/T", top_axis="K",
                   show_components=True, ax=None, colours=None, markers="osD^v<>",
                   show_legend=True, annotate_Ea=True, markersize=5.0, linewidth=1.2,
                   text_size="small"):
    """BPP/Arrhenius plot: ``ln(T1)`` (``y_axis``: 'ln_time', 'ln_rate',
    'log10_time', 'log10_rate', 'time', 'rate') against ``1000/T``
    (``x_axis``: '1000/T', '1/kBT' or 'T'), with the actual temperature
    on a second axis on top (``top_axis`` 'K', 'C' or None).

    Pass a fit from :func:`fit_vt_relaxation` (its data and curves are
    drawn), or just ``datasets``. Multi-process fits also show each
    process dashed (``show_components``). ``text_size`` sizes the Ea
    annotation and legend. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt

    from ..plotting.accessibility import default_series_colours, series_colours

    if fit is None and datasets is None:
        raise ValueError("Give a fit or datasets to plot.")
    if isinstance(datasets, dict):
        datasets = [datasets]
    data = fit["datasets"] if fit is not None else datasets
    quantity = data[0]["quantity"]
    if quantity == "linewidth":
        raise ValueError("Plot linewidths with plot_hendrickson_bray.")
    fx, _inv, xlabel = _x_transform(x_axis)
    fy, ylabel, log_scale = _y_transform(y_axis, quantity)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
    else:
        fig = ax.figure
    n = len(data)
    if colours is None:
        colours = default_series_colours(max(n, 1))
    elif isinstance(colours, str):
        colours = series_colours(n, colours)
    y_seen = []
    for i, d in enumerate(data):
        c = colours[i % len(colours)]
        x = fx(d["T_K"])
        y = fy(d["value"])
        y_seen.append(y)
        err = None
        e = d.get("error")
        if e is not None and np.all(np.isfinite(e)) and not log_scale and y_axis != "time":
            scale = 1.0 if y_axis.startswith("ln") else 1.0 / math.log(10)
            err = scale * e / d["value"]
        elif e is not None and np.all(np.isfinite(e)) and y_axis == "time":
            err = e
        # Markers as a plain line (so the legend entry is the marker,
        # and survives a legend being rebuilt), error bars beside it.
        if err is not None:
            ax.errorbar(x, y, yerr=err, fmt="none", ecolor=c, elinewidth=0.8, capsize=2,
                        zorder=3)
        ax.plot(x, y, markers[i % len(markers)], color=c, mfc="none" if i else c,
                ms=markersize, ls="none", label=d["label"], zorder=4)
        if fit is not None:
            T_c, total, parts = evaluate_vt_fit(fit, dataset=i, components=True)
            ax.plot(fx(T_c), fy(1.0 / total), "-", color=c, lw=linewidth, zorder=2)
            y_seen.append(fy(1.0 / total))
            if show_components and len(parts) > 1:
                for part in parts:
                    good = part > 0
                    ax.plot(fx(T_c[good]), fy(1.0 / part[good]), "--", color=c,
                            lw=0.8 * linewidth, alpha=0.7, zorder=1)
    if log_scale:
        ax.set_yscale("log")
    else:
        # Components can run far off the data (a process whose maximum
        # is outside the range); the axes follow the data and total fit.
        allv = np.concatenate([np.ravel(v) for v in y_seen])
        allv = allv[np.isfinite(allv)]
        if allv.size:
            lo_y, hi_y = float(allv.min()), float(allv.max())
            pad = 0.08 * (hi_y - lo_y or 1.0)
            ax.set_ylim(lo_y - pad, hi_y + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if fit is not None and annotate_Ea:
        Eas = [(n_, v) for n_, v in fit["params"].items() if n_.startswith("Ea_eV")]
        text = "\n".join(
            (rf"$E_\mathregular{{a{n_[5:].replace('_', ',')}}}$" if n_ != "Ea_eV"
             else r"$E_\mathregular{a}$")
            + f" = {_fmt(v, fit['errors'].get(n_))} eV" for n_, v in Eas)
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=text_size, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    _style_axes(ax)
    _add_top_axis(ax, x_axis, top_axis)
    if show_legend and n > 1:
        ax.legend(frameon=False, fontsize=text_size, loc="lower right")
    fig.tight_layout()
    return fig, ax


def plot_vs_temperature(datasets=None, fit=None, temperature_unit="K", log_y=True, ax=None,
                        colours=None, markers="osD^v<>", markersize=5.0, linewidth=1.2):
    """Relaxation times (or linewidths) against temperature itself,
    with the fitted curve if a fit is given. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt

    from ..plotting.accessibility import default_series_colours, series_colours

    if fit is not None and "datasets" in fit:
        datasets = fit["datasets"]
    if isinstance(datasets, dict):
        datasets = [datasets]
    if not datasets:
        raise ValueError("Give datasets or a fit.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
    else:
        fig = ax.figure
    shift = -273.15 if str(temperature_unit).lstrip("°").upper() == "C" else 0.0
    n = len(datasets)
    colours = (default_series_colours(max(n, 1)) if colours is None
               else series_colours(n, colours) if isinstance(colours, str) else colours)
    q = datasets[0]["quantity"]
    sym = QUANTITIES[q][0]
    for i, d in enumerate(datasets):
        c = colours[i % len(colours)]
        e = d.get("error")
        yerr = e if (e is not None and np.all(np.isfinite(e))) else None
        if yerr is not None:
            ax.errorbar(d["T_K"] + shift, d["value"], yerr=yerr, fmt="none", ecolor=c,
                        elinewidth=0.8, capsize=2, zorder=3)
        ax.plot(d["T_K"] + shift, d["value"], markers[i % len(markers)], color=c,
                mfc="none" if i else c, ms=markersize, ls="none", label=d["label"], zorder=4)
        if fit is not None and q != "linewidth":
            T_c, total = evaluate_vt_fit(fit, dataset=i)
            order = np.argsort(T_c)
            ax.plot(T_c[order] + shift, 1.0 / total[order], "-", color=c, lw=linewidth)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$T$ (°C)" if shift else r"$T$ (K)")
    ax.set_ylabel(rf"${sym}$ (Hz)" if q == "linewidth" else rf"${sym}$ (s)")
    _style_axes(ax)
    if n > 1:
        ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    return fig, ax


def plot_hendrickson_bray(fit, ax=None, temperature_unit="K", colour=None, unit="Hz",
                          show_onset=True, markersize=5.0, linewidth=1.2, text_size="small"):
    """Linewidths against temperature with the Hendrickson-Bray curve
    from :func:`fit_hendrickson_bray`. ``unit`` 'Hz' or 'kHz'. Returns
    ``(fig, ax)``."""
    import matplotlib.pyplot as plt

    from ..plotting.accessibility import default_series_colours

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
    else:
        fig = ax.figure
    c = colour or default_series_colours(1)[0]
    shift = -273.15 if str(temperature_unit).lstrip("°").upper() == "C" else 0.0
    div = 1e3 if unit == "kHz" else 1.0
    T, y, e = fit["T_K"], fit["width_Hz"], fit.get("error")
    ax.errorbar(T + shift, y / div, yerr=None if e is None else e / div, fmt="o", color=c,
                ms=markersize, lw=0, elinewidth=0.8, capsize=2, zorder=3)
    Tc = np.linspace(T.min() - 0.05 * np.ptp(T), T.max() + 0.05 * np.ptp(T), 400)
    p = fit["params"]
    ax.plot(Tc + shift, hendrickson_bray(Tc, p["A"], p["B"], p["Ea_eV"], p["D"]) / div, "-",
            color=c, lw=linewidth, zorder=2)
    if show_onset and np.isfinite(fit.get("T_onset_K", np.nan)):
        ax.axvline(fit["T_onset_K"] + shift, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel(r"$T$ (°C)" if shift else r"$T$ (K)")
    ax.set_ylabel(rf"$\Delta\nu$ ({unit})")
    ax.text(0.97, 0.97, rf"$E_\mathregular{{a}}$ = {_fmt(p['Ea_eV'], fit['errors']['Ea_eV'])} eV",
            transform=ax.transAxes, ha="right", va="top", fontsize=text_size)
    _style_axes(ax)
    fig.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------
# Notebook code
# ----------------------------------------------------------------------

def vt_notebook_code(datasets, fit_kwargs=None, plot_kwargs=None, hendrickson=False) -> str:
    """Python reproducing a GUI fit in a notebook."""
    lines = ["import numpy as np",
             "from opennmr.notebook.vt_relaxation import (",
             "    vt_dataset, fit_vt_relaxation, fit_hendrickson_bray,",
             "    plot_arrhenius, plot_hendrickson_bray)", ""]
    lines.append("datasets = [")
    for d in datasets:
        err = d.get("error")
        err_txt = (repr([round(float(v), 10) for v in err])
                   if err is not None and np.all(np.isfinite(err)) else "None")
        lines.append(
            f"    vt_dataset(T_K={[round(float(t), 4) for t in d['T_K']]!r},\n"
            f"               value={[float(v) for v in d['value']]!r},\n"
            f"               error={err_txt}, larmor_MHz={d.get('larmor_MHz')!r},\n"
            f"               quantity={d['quantity']!r}, label={d['label']!r}"
            + (f", spinlock_kHz={d['spinlock_kHz']!r}" if d.get("spinlock_kHz") else "")
            + "),")
    lines.append("]")
    if hendrickson:
        d = datasets[0]
        lines += ["fit = fit_hendrickson_bray(datasets[0]['T_K'], datasets[0]['value'], "
                  "datasets[0]['error']"
                  + (f", fixed={(fit_kwargs or {}).get('fixed')!r}"
                     if (fit_kwargs or {}).get("fixed") else "") + ")",
                  "print(fit['log'])", "fig, ax = plot_hendrickson_bray(fit)"]
        return "\n".join(lines)
    kw = ", ".join(f"{k}={v!r}" for k, v in (fit_kwargs or {}).items())
    pk = ", ".join(f"{k}={v!r}" for k, v in (plot_kwargs or {}).items())
    lines += [f"fit = fit_vt_relaxation(datasets{', ' + kw if kw else ''})",
              "print(fit['log'])",
              f"fig, ax = plot_arrhenius(fit{', ' + pk if pk else ''})"]
    return "\n".join(lines)
