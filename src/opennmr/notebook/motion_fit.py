"""Fitting motional lineshapes: variable-temperature (VT) series and single spectra.

Two routes, which answer different questions and are best used together:

* :func:`fit_motion_rates` -- the *model-light* route. Every spectrum gets its own rate
  (a one-parameter fit, with a scan so it cannot start in the wrong place) and the
  Arrhenius line is drawn through the rates afterwards (:func:`arrhenius_regression`).
  Each rate can be inspected, and spectra in the slow or fast limit -- whose lineshape does
  not depend on the rate -- are reported as undetermined and left out of the line rather
  than silently pulling it. It cannot, by itself, give the geometry.
* :func:`fit_motion` -- the *global* route. One model (Arrhenius barriers, cone angles, tensor
  values, ...) is fitted to all the spectra at once, each simulated at its own temperature.
  Use it for the final numbers, with starting values from the first route.

What the fits do that a plain least-squares call does not:

* Each spectrum's scale (and a constant offset) is solved linearly at every step, so it
  never needs a guess and never correlates with the motion.
* An Arrhenius rate ``nu0 exp(-Ea/kB T)`` is fitted as ``Ea`` and ``log10 k(T_ref)`` at a
  reference temperature in the middle of the data (``mode1.log10_k_ref``): ``Ea`` and
  ``log10 nu0`` are almost perfectly correlated -- the data only fix the rate near the
  temperatures measured, and ``nu0`` is an extrapolation to 1/T = 0 -- so fitting them
  directly is badly conditioned. ``nu0`` is reported with the error that follows from the
  covariance, which is usually much larger than the error on ``k(T_ref)``.
* Uncertainties come from the Jacobian at the solution, scaled by the reduced chi-square.
  They are the *formal* errors: they assume independent, equally noisy points and a model
  that is right, which a lineshape fit with phase, baseline and model errors does not
  deserve. Read them as a lower bound and look at the correlation matrix and at
  ``multistart``.
* ``n_starts`` runs coarse fits from Latin-hypercube starting points inside the bounds
  before the final fit and reports how the starts ended, so a single run that settled in a
  local minimum shows up as a spread of costs.

Parameter names (1-based, as :func:`opennmr.notebook.exchange.fit_exchange_spectra`)::

    mode{i}.Ea_eV  mode{i}.log10_nu0  mode{i}.log10_k_ref  mode{i}.log10_k
    hop{h}.Ea_eV   hop{h}.Ea_rev_eV   hop{h}.log10_nu0     hop{h}.log10_k_ref  hop{h}.log10_k
    site{j}.alpha  site{j}.beta       site{j}.gamma        (the Euler angles, degrees)
    site{j}.delta_iso  site{j}.fwhm  site{j}.C_Q  site{j}.eta_Q  site{j}.aniso  site{j}.asym
    site{j}.dipolar{n}.D_Hz          gauss_fwhm_ppm

``log10_k`` is a rate that does not depend on temperature (for spectra at one temperature, or
a series of one sample at one temperature); ``log10_k_ref`` and ``Ea_eV`` are an Arrhenius
rate. The rate parameters other than the one being fitted keep the model's values.

Not here (yet): relaxation (T1, T2) as a second observable. A lineshape alone fixes the
rate only where the spectrum is sensitive to it -- about 0.1 to 10 times the frequency
spread of the interaction -- so the barrier of a process whose rate is within that window
at just one or two temperatures is poorly determined; ``fit_motion`` says so through the
errors and the correlation matrix instead of hiding it.
"""

from __future__ import annotations

import copy
import math
import time
import warnings

import numpy as np

from .exchange import K_B_EV
from .motion import (MotionModel, motion_overlay_series, motion_spectrum)

__all__ = ["fit_parameter_list", "fit_motion", "fit_motion_rates", "arrhenius_regression",
           "plot_motion_fit", "plot_motion_rates", "motion_fit_code"]

_LN10 = math.log(10.0)

_RATE_KEYS = ("Ea_eV", "Ea_rev_eV", "log10_nu0", "log10_k_ref", "log10_k")
_SITE_KEYS = ("alpha", "beta", "gamma", "delta_iso", "fwhm", "C_Q", "eta_Q", "aniso", "asym")


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

def _parse_name(name: str):
    """``'mode1.Ea_eV'`` -> ``('mode', 0, None, 'Ea_eV')``; ``'site2.dipolar1.D_Hz'`` ->
    ``('site', 1, 0, 'D_Hz')``; ``'gauss_fwhm_ppm'`` -> ``('gauss', None, None, name)``."""
    if name == "gauss_fwhm_ppm":
        return "gauss", None, None, name
    parts = name.split(".")
    head = parts[0]
    for kind in ("mode", "hop", "site"):
        if head.startswith(kind) and head[len(kind):].isdigit():
            index = int(head[len(kind):]) - 1
            if index < 0:
                break
            if kind == "site" and len(parts) == 3 and parts[1].startswith("dipolar") \
                    and parts[1][7:].isdigit() and parts[2] == "D_Hz":
                return kind, index, int(parts[1][7:]) - 1, "D_Hz"
            if len(parts) == 2:
                key = parts[1]
                allowed = _SITE_KEYS if kind == "site" else _RATE_KEYS
                if key in allowed:
                    return kind, index, None, key
            break
    raise ValueError(f"Unknown fit parameter {name!r}; see opennmr.notebook.motion_fit for the "
                     "list (mode1.Ea_eV, mode1.log10_k_ref, site1.beta, ...).")


def _default_bounds(name: str, value: float):
    key = name.split(".")[-1]
    if key in ("Ea_eV", "Ea_rev_eV"):
        return 0.0, 3.0
    if key == "log10_nu0":
        return 6.0, 20.0
    if key in ("log10_k_ref", "log10_k"):
        return -3.0, 12.0
    if key == "beta":
        return 0.0, 180.0
    if key in ("alpha", "gamma"):
        return -360.0, 360.0
    if key in ("fwhm", "C_Q", "gauss_fwhm_ppm"):
        return 0.0, np.inf
    if key == "eta_Q":
        return 0.0, 1.0
    if key == "asym":
        return -1.0, 1.0
    return -np.inf, np.inf


def _site(d, i):
    if not 0 <= i < len(d["sites"]):
        raise ValueError(f"There is no environment {i + 1}.")
    return d["sites"][i]


def _interaction(site_dict, kinds):
    for it in site_dict["site"].get("interactions", []):
        if it.get("type") in kinds:
            return it
    return None


def _read(d, name, T_ref):
    """The current value of parameter ``name`` in the model dict ``d``."""
    kind, i, j, key = _parse_name(name)
    if kind == "gauss":
        return 0.0
    if kind == "mode":
        if not 0 <= i < len(d["modes"]):
            raise ValueError(f"There is no mode {i + 1}.")
        rate = d["modes"][i]["rate"]
        return _read_rate(rate, key, name, T_ref, f"Mode {i + 1}")
    if kind == "hop":
        if not 0 <= i < len(d["hops"]):
            raise ValueError(f"There is no hop {i + 1}.")
        hop = d["hops"][i]
        if key == "Ea_rev_eV":
            return float(hop["Ea_rev_eV"]) if hop.get("Ea_rev_eV") is not None else float(
                hop["Ea_eV"])
        return _read_rate(hop, key, name, T_ref, f"Hop {i + 1}")
    s = _site(d, i)
    if key in ("alpha", "beta", "gamma"):
        return float(s["euler"]["abg".index(key[0])])
    if key == "D_Hz":
        if not 0 <= j < len(s["dipolar"]):
            raise ValueError(f"Environment {i + 1} has no dipolar coupling {j + 1}.")
        return float(s["dipolar"][j]["D_Hz"])
    if key in ("delta_iso", "fwhm"):
        return float(s["site"].get(key, 0.0) or 0.0)
    if key in ("C_Q", "eta_Q"):
        q = _interaction(s, ("quadrupolar",))
        if q is None:
            raise ValueError(f"Environment {i + 1} has no quadrupolar interaction ({name}).")
        return float(q[key])
    c = _interaction(s, ("csa",))
    if c is None:
        raise ValueError(f"Environment {i + 1} has no CSA interaction ({name}).")
    return float(c[key])


def _read_rate(rate, key, name, T_ref, what):
    if key == "Ea_eV":
        if rate.get("Ea_eV") is None:
            raise ValueError(f"{what} has a fixed rate k: give it an Arrhenius rate (Ea and nu0) "
                             f"to fit {name}, or fit its log10_k.")
        return float(rate["Ea_eV"])
    if key == "log10_nu0":
        if rate.get("Ea_eV") is None:
            raise ValueError(f"{what} has a fixed rate k: there is no attempt frequency to fit.")
        return math.log10(float(rate.get("nu0_Hz", 1e12)))
    if key == "log10_k_ref":
        if rate.get("Ea_eV") is None:
            raise ValueError(f"{what} has a fixed rate k: give it an Arrhenius rate (Ea and nu0) "
                             f"to fit {name}.")
        if T_ref is None:
            raise ValueError("log10_k_ref needs a reference temperature (T_ref_K).")
        return math.log10(float(rate.get("nu0_Hz", 1e12))) - float(rate["Ea_eV"]) / (
            K_B_EV * T_ref * _LN10)
    if key == "log10_k":
        k = rate.get("k_Hz")
        if k is None:
            if T_ref is None:
                raise ValueError("log10_k of an Arrhenius rate needs a temperature.")
            k = float(rate.get("nu0_Hz", 1e12)) * math.exp(-float(rate["Ea_eV"]) / (K_B_EV * T_ref))
        return math.log10(max(float(k), 1e-300))
    raise ValueError(f"{name} does not apply here.")


def _apply(d, values, T_ref):
    """Write ``values`` (name -> number) into the model dict ``d``; returns the extra Gaussian
    broadening (ppm) if ``gauss_fwhm_ppm`` is among them, else None."""
    extra = None
    order = sorted(values, key=lambda n: (n.endswith(("log10_k_ref", "log10_k")), n))
    for name in order:
        v = float(values[name])
        kind, i, j, key = _parse_name(name)
        if kind == "gauss":
            extra = max(v, 0.0)
        elif kind in ("mode", "hop"):
            holder = d["modes"][i]["rate"] if kind == "mode" else d["hops"][i]
            if key == "Ea_eV":
                holder["Ea_eV"] = v
            elif key == "Ea_rev_eV":
                holder["Ea_rev_eV"] = v
            elif key == "log10_nu0":
                holder["nu0_Hz"] = 10.0 ** v
            elif key == "log10_k_ref":
                Ea = float(holder["Ea_eV"])
                holder["nu0_Hz"] = 10.0 ** v * math.exp(Ea / (K_B_EV * T_ref))
            else:                                           # log10_k
                for drop in ("Ea_eV", "Ea_rev_eV", "nu0_Hz"):
                    holder.pop(drop, None)
                holder["k_Hz"] = 10.0 ** v
                if kind == "hop":
                    holder["k_rev_Hz"] = 10.0 ** v
        else:
            s = d["sites"][i]
            if key in ("alpha", "beta", "gamma"):
                euler = list(s["euler"])
                euler["abg".index(key[0])] = v
                s["euler"] = euler
            elif key == "D_Hz":
                s["dipolar"][j]["D_Hz"] = v
            elif key in ("delta_iso", "fwhm"):
                s["site"][key] = max(v, 0.0) if key == "fwhm" else v
            elif key in ("C_Q", "eta_Q"):
                _interaction(s, ("quadrupolar",))[key] = v
            else:
                _interaction(s, ("csa",))[key] = v
    return extra


def fit_parameter_list(model, T_ref_K=None) -> list[dict]:
    """The parameters of ``model`` that can be fitted, for building a table: a list of
    ``{'name', 'label', 'value', 'min', 'max', 'default_free'}``. ``default_free`` marks the
    ones usually fitted (the Arrhenius pair of each activated process, or the rate of a fixed
    one). ``T_ref_K`` (for ``log10_k_ref``) defaults to 300 K."""
    T_ref = 300.0 if T_ref_K is None else float(T_ref_K)
    d = model.to_dict()
    out = []

    def add(name, label, free=False):
        try:
            value = _read(d, name, T_ref)
        except ValueError:
            return
        lo, hi = _default_bounds(name, value)
        out.append({"name": name, "label": label, "value": value, "min": lo, "max": hi,
                    "default_free": free})

    for i, m in enumerate(d["modes"], 1):
        tag = f"mode {i} ({m.get('label') or m['kind']})"
        if m["rate"].get("Ea_eV") is not None:
            add(f"mode{i}.Ea_eV", f"{tag}: Ea (eV)", True)
            add(f"mode{i}.log10_k_ref", f"{tag}: log10 k at T_ref (s⁻¹)", True)
            add(f"mode{i}.log10_nu0", f"{tag}: log10 ν₀ (s⁻¹)")
        else:
            add(f"mode{i}.log10_k", f"{tag}: log10 k (s⁻¹)", True)
    for h, hop in enumerate(d["hops"], 1):
        tag = f"hop {h}"
        if hop.get("Ea_eV") is not None:
            add(f"hop{h}.Ea_eV", f"{tag}: Ea (eV)", True)
            add(f"hop{h}.log10_k_ref", f"{tag}: log10 k at T_ref (s⁻¹)", True)
            add(f"hop{h}.log10_nu0", f"{tag}: log10 ν₀ (s⁻¹)")
            add(f"hop{h}.Ea_rev_eV", f"{tag}: Ea reverse (eV)")
        else:
            add(f"hop{h}.log10_k", f"{tag}: log10 k (s⁻¹)", True)
    for j, s in enumerate(d["sites"], 1):
        tag = f"environment {j} ({s['site'].get('label', '')})"
        for key, label in (("alpha", "α (°)"), ("beta", "β (°)"), ("gamma", "γ (°)"),
                           ("delta_iso", "δiso (ppm)"), ("fwhm", "FWHM")):
            add(f"site{j}.{key}", f"{tag}: {label}")
        for key, label in (("C_Q", "C_Q (MHz)"), ("eta_Q", "η_Q"), ("aniso", "CSA anisotropy"),
                           ("asym", "CSA asymmetry")):
            add(f"site{j}.{key}", f"{tag}: {label}")
        for n, dip in enumerate(s["dipolar"], 1):
            add(f"site{j}.dipolar{n}.D_Hz", f"{tag}: dipolar to {dip['partner']} D (Hz)")
    out.append({"name": "gauss_fwhm_ppm", "label": "extra Gaussian broadening (ppm)",
                "value": 0.0, "min": 0.0, "max": np.inf, "default_free": False})
    return out


def _normalise_parameters(model, parameters, T_ref):
    """``parameters`` -> ordered list of ``{'name','value','min','max','vary'}``."""
    d = model.to_dict()
    if isinstance(parameters, (list, tuple)):
        parameters = {n: {} for n in parameters}
    if not parameters:
        raise ValueError("Choose at least one parameter to fit.")
    out = []
    for name, spec in dict(parameters).items():
        _parse_name(name)
        if isinstance(spec, (int, float)):
            spec = {"value": float(spec)}
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            spec = {"min": spec[0], "max": spec[1]}
        spec = dict(spec or {})
        value = spec.get("value")
        if value is None:
            value = _read(d, name, T_ref)
        lo, hi = _default_bounds(name, float(value))
        lo = spec.get("min", lo)
        hi = spec.get("max", hi)
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        if not lo <= float(value) <= hi:
            raise ValueError(f"{name}: the starting value {float(value):g} is outside its bounds "
                             f"({lo:g}, {hi:g}).")
        out.append({"name": name, "value": float(value), "min": lo, "max": hi,
                    "vary": bool(spec.get("vary", True))})
    return out


# ----------------------------------------------------------------------
# The residual problem
# ----------------------------------------------------------------------

def _prepare_spectra(spectra, n_points, window=None):
    if isinstance(spectra, dict):
        spectra = [spectra]
    if not spectra:
        raise ValueError("Give at least one spectrum to fit.")
    prepared = []
    for k, s in enumerate(spectra):
        ppm = np.asarray(s["ppm"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        if ppm.shape != y.shape or ppm.ndim != 1:
            raise ValueError("Each spectrum needs 1D 'ppm' and 'y' arrays of the same length.")
        lo, hi = float(ppm.min()), float(ppm.max())
        w = s.get("window", window)
        if w is not None:
            lo, hi = float(min(w)), float(max(w))
        keep = (ppm >= lo) & (ppm <= hi)
        if keep.sum() < 8:
            raise ValueError(f"Spectrum {k + 1} has fewer than 8 points in its window.")
        T = s.get("T_K")
        label = s.get("label") or (f"{float(T):g} K" if T is not None else f"spectrum {k + 1}")
        ys = y[keep]
        norm = float(np.max(np.abs(ys))) or 1.0
        prepared.append({"x": ppm[keep], "y": ys / norm, "norm": norm, "window": (lo, hi),
                         "T_K": None if T is None else float(T), "label": label,
                         "axis": np.linspace(hi, lo, int(n_points)),
                         "ppm_full": ppm, "y_full": y,
                         "sigma": (None if s.get("sigma") is None else float(s["sigma"]) / norm)})
    return prepared


def _resolve_kwargs(spectra, kw):
    kw = dict(kw)
    if kw.get("larmor_Hz") is None and kw.get("B0_T") is None:
        for s in (spectra if not isinstance(spectra, dict) else [spectra]):
            if s.get("larmor_MHz"):
                kw["larmor_Hz"] = float(s["larmor_MHz"]) * 1e6
                break
    if kw.get("larmor_Hz") is None and kw.get("B0_T") is None:
        raise ValueError("Give larmor_Hz (or B0_T) – or 'larmor_MHz' in the spectra.")
    return kw


def _grid_note(model, prepared, kw):
    """A warning line when the simulation grid is too coarse for the narrowest intrinsic line
    (the engine then widens the lines to two grid points, which biases the rates)."""
    try:
        fwhm = float(np.min(model.state_data()["fwhm_ppm"])) + float(kw.get("gauss_fwhm_ppm") or 0.0)
    except (KeyError, ValueError):
        return None
    floor = float(kw.get("min_fwhm_points", 2.0))
    worst = max(abs(p["axis"][0] - p["axis"][1]) for p in prepared)
    if fwhm > 0 and floor * worst > fwhm:
        need = int(math.ceil(max(abs(p["axis"][0] - p["axis"][-1]) for p in prepared)
                             * floor / fwhm))
        return (f"⚠ The simulation grid ({worst:.3g} ppm per point) is coarser than "
                f"{floor:g} points across the narrowest intrinsic line ({fwhm:.3g} ppm): lines are "
                f"widened to that, which biases the rates. Use n_points of about {need} or more.")
    return None


class _Problem:
    """Simulates the model at each spectrum's temperature, solves each spectrum's scale and
    offset linearly, and returns the residuals the optimiser works on."""

    def __init__(self, model, prepared, names, fixed, T_ref, kw, fit_offset):
        self.base = model.to_dict()
        self.prepared = prepared
        self.names = names
        self.fixed = fixed                 # name -> value held (not varied)
        self.T_ref = T_ref
        self.kw = kw
        self.fit_offset = fit_offset
        self.n_calls = 0
        self.sim_seconds = 0.0

    def model_at(self, x):
        d = copy.deepcopy(self.base)
        values = dict(self.fixed)
        values.update(zip(self.names, x))
        extra = _apply(d, values, self.T_ref)
        return MotionModel.from_dict(d), extra

    def simulate(self, x):
        model, extra = self.model_at(x)
        out = []
        for p in self.prepared:
            kw = dict(self.kw)
            if extra is not None:
                kw["gauss_fwhm_ppm"] = extra
            if p["T_K"] is not None:
                kw["T_K"] = p["T_K"]
            t0 = time.perf_counter()
            res = motion_spectrum(model, p["axis"], **kw)
            self.sim_seconds += time.perf_counter() - t0
            self.n_calls += 1
            order = np.argsort(res["x_ppm"])
            base = np.interp(p["x"], res["x_ppm"][order], res["real_spec"][order])
            if self.fit_offset:
                A = np.column_stack([base, np.ones_like(base)])
                coef, *_ = np.linalg.lstsq(A, p["y"], rcond=None)
                scale, offset = float(coef[0]), float(coef[1])
            else:
                denom = float(base @ base)
                scale = float(base @ p["y"]) / denom if denom > 0 else 0.0
                offset = 0.0
            out.append({"base": base, "scale": scale, "offset": offset,
                        "sim": scale * base + offset, "result": res})
        return out, model

    def residuals(self, x):
        try:
            sims, _model = self.simulate(x)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            return np.full(sum(p["x"].size for p in self.prepared), 1e3)
        parts = []
        for p, s in zip(self.prepared, sims):
            r = p["y"] - s["sim"]
            if p["sigma"]:
                r = r / p["sigma"]
            parts.append(r)
        return np.concatenate(parts)


def _errors(sol, n_params, n_points, n_linear):
    """(errors, covariance, correlation, condition number, redchi) at ``sol``."""
    chi2 = float(np.sum(sol.fun ** 2))
    dof = max(n_points - n_params - n_linear, 1)
    redchi = chi2 / dof
    J = sol.jac
    nan = np.full(n_params, np.nan)
    try:
        JtJ = J.T @ J
        cov = np.linalg.pinv(JtJ) * redchi
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(err, err)
        sv = np.linalg.svd(J * (1.0 / np.maximum(np.linalg.norm(J, axis=0), 1e-300)),
                           compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        return err, cov, corr, cond, redchi
    except np.linalg.LinAlgError:
        return nan, np.full((n_params, n_params), np.nan), np.full((n_params, n_params),
                                                                   np.nan), float("inf"), redchi


def fit_motion(model, spectra, parameters, T_ref_K=None, fit_offset=True, n_points=1024,
               n_starts=0, search_nfev=25, seed=0, max_nfev=None, window=None, **kwargs) -> dict:
    """Fit ``model`` to ``spectra`` -- a VT series, or a single spectrum -- with every spectrum
    simulated at its own temperature.

    Parameters
    ----------
    model : :class:`~opennmr.notebook.motion.MotionModel`, with starting values.
    spectra : dicts ``{'ppm', 'y', 'T_K' (needed for Arrhenius rates), 'label', 'larmor_MHz',
        'window' (ppm, lo/hi), 'sigma' (noise s.d., y units)}``.
    parameters : names to fit (see this module's notes), or ``{name: value | (min, max) |
        {'value', 'min', 'max', 'vary'}}``. ``vary=False`` holds a parameter at ``value``.
    T_ref_K : reference temperature of ``log10_k_ref`` (default: the temperature the data
        spread evenly around in 1/T, which decorrelates it from Ea).
    fit_offset : solve a constant offset per spectrum as well as the scale.
    n_points : points of each simulation (interpolated onto the data).
    n_starts : extra coarse fits from Latin-hypercube starts inside the bounds (needs finite
        bounds on the fitted parameters; ``search_nfev`` evaluations each) before the final fit.
    kwargs : :func:`~opennmr.notebook.motion.motion_spectrum` options shared by all spectra
        (``larmor_Hz``, ``mas_Hz``, ``echo``, ``order``, ``transitions``, ``n_orient``, ...).

    Returns a dict: ``params``, ``errors`` (1-sigma, formal: see the module notes), ``model``
    (the fitted model), ``overlay`` (data and fit per spectrum, for plotting), ``derived``
    (e.g. ``mode1.log10_nu0`` and ``mode1.nu0_Hz`` with errors), ``correlation``, ``condition``,
    ``redchi``, ``starts`` and ``log``.
    """
    from ..core import fit_progress as _fit_progress

    kw = _resolve_kwargs(spectra, kwargs)
    prepared = _prepare_spectra(spectra, n_points, window)
    temps = sorted({p["T_K"] for p in prepared if p["T_K"] is not None})
    if T_ref_K is None:
        T_ref_K = (1.0 / float(np.mean([1.0 / t for t in temps]))) if temps else None
    T_ref = None if T_ref_K is None else float(T_ref_K)
    plist = _normalise_parameters(model, parameters, T_ref)
    free = [p for p in plist if p["vary"]]
    fixed = {p["name"]: p["value"] for p in plist if not p["vary"]}
    names = [p["name"] for p in free]
    if not names:
        raise ValueError("Every parameter is held fixed: nothing to fit.")
    for h in range(1, len(model.hops) + 2):
        pair = {f"hop{h}.Ea_eV", f"hop{h}.log10_nu0"}
        if pair <= set(names) and len(temps) < 2:
            raise ValueError(f"Hop {h}: Ea and the attempt frequency cannot both be fitted at "
                             "one temperature; only the rate is defined. Fit log10_k.")
    for i in range(1, len(model.modes) + 1):
        pair = {f"mode{i}.Ea_eV", f"mode{i}.log10_nu0"}
        if pair <= set(names) and len(temps) < 2:
            raise ValueError(f"Mode {i}: Ea and the attempt frequency cannot both be fitted at "
                             "one temperature; only the rate is defined. Fit log10_k.")
        if {f"mode{i}.log10_nu0", f"mode{i}.log10_k_ref"} <= set(names):
            raise ValueError(f"Mode {i}: log10_nu0 and log10_k_ref are the same freedom in "
                             "different clothes; choose one.")
    if any(n.endswith(".log10_k") for n in names) and len(temps) > 1:
        raise ValueError("log10_k is a rate that does not depend on temperature, and the spectra "
                         "are at different temperatures. Fit Ea and log10_k_ref (an Arrhenius "
                         "rate) or use fit_motion_rates for a rate per spectrum.")
    if any(("Ea_eV" in n or "log10_nu0" in n or "log10_k_ref" in n) for n in names) and not temps:
        raise ValueError("Arrhenius parameters need the temperature of each spectrum ('T_K').")
    if any(p["T_K"] is None for p in prepared) and temps:
        raise ValueError("Give every spectrum a temperature, or none.")

    x0 = np.array([p["value"] for p in free], dtype=float)
    lo = np.array([p["min"] for p in free], dtype=float)
    hi = np.array([p["max"] for p in free], dtype=float)
    pad = 1e-9 * np.maximum(1.0, np.abs(x0))
    x0 = np.clip(x0, lo + np.where(np.isfinite(lo), pad, 0), hi - np.where(np.isfinite(hi), pad, 0))
    problem = _Problem(model, prepared, names, fixed, T_ref, kw, fit_offset)

    t0 = time.perf_counter()
    problem.simulate(x0)            # not guarded: a model that cannot be simulated says why
    one_eval = time.perf_counter() - t0
    log = [f"Motional lineshape fit: {len(prepared)} spectrum/spectra"
           + (f" ({', '.join(p['label'] for p in prepared)})" if len(prepared) <= 8 else "")
           + f", {int(sum(p['x'].size for p in prepared))} points, {len(names)} fitted "
           f"parameter(s) + {len(prepared) * (2 if fit_offset else 1)} linear scale/offset values.",
           f"One residual evaluation (all spectra) takes {one_eval:.2g} s; a fit needs about "
           f"{(len(names) + 1)} of them per iteration."]
    note = _grid_note(model, prepared, kw)
    if note:
        log.append(note)
    if T_ref is not None and any(n.endswith("log10_k_ref") for n in names):
        log.append(f"Arrhenius rates are fitted as Ea and log10 k at T_ref = {T_ref:.1f} K "
                   "(1/T at the middle of the data).")

    starts = []
    best_x = x0
    if n_starts and n_starts > 0:
        finite = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        if not finite.any():
            warnings.warn("n_starts needs finite bounds on the fitted parameters; ignored.",
                          UserWarning, stacklevel=2)
        else:
            from scipy.optimize import least_squares
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=int(finite.sum()), seed=seed)
            unit = sampler.random(int(n_starts))
            trials = [x0.copy()]
            for u in unit:
                x = x0.copy()
                x[finite] = lo[finite] + u * (hi[finite] - lo[finite])
                trials.append(x)
            for k, xs in enumerate(trials):
                _fit_progress.check_cancelled()
                _fit_progress.report_stage(f"Search start {k + 1} of {len(trials)}")
                try:
                    s = least_squares(problem.residuals, xs, bounds=(lo, hi), x_scale="jac",
                                      max_nfev=int(search_nfev))
                    starts.append({"start": xs.tolist(), "x": s.x.tolist(),
                                   "cost": float(np.sum(s.fun ** 2))})
                except (ValueError, ArithmeticError):
                    continue
            if starts:
                best = min(starts, key=lambda s: s["cost"])
                best_x = np.array(best["x"])
                near = sum(1 for s in starts if s["cost"] <= 1.05 * best["cost"])
                log.append(f"Search: {len(starts)} coarse fits ({int(search_nfev)} evaluations "
                           f"each); {near} ended within 5 % of the best cost "
                           f"({best['cost']:.4g}; the first was your own starting point at "
                           f"{starts[0]['cost']:.4g}). "
                           + ("Several starts agree." if near > 1 else
                              "Only one start reached it: the landscape has other minima; "
                              "check the result against the data."))

    _fit_progress.report_stage("Final fit")
    sol = _fit_progress.limited_least_squares(
        problem.residuals, best_x, bounds=(lo, hi), x_scale="jac",
        max_iterations=max_nfev, raise_on_limit=False, label="The motion fit")
    n_points_total = int(sum(p["x"].size for p in prepared))
    n_linear = len(prepared) * (2 if fit_offset else 1)
    err, cov, corr, cond, redchi = _errors(sol, len(names), n_points_total, n_linear)
    xs = np.asarray(sol.x, dtype=float)
    sims, fitted = problem.simulate(xs)
    params = dict(zip(names, map(float, xs)))
    errors = dict(zip(names, map(float, err)))
    at_bound = [n for n, v, a, b in zip(names, xs, lo, hi)
                if (np.isfinite(a) and abs(v - a) <= 1e-6 * max(1.0, abs(a)))
                or (np.isfinite(b) and abs(v - b) <= 1e-6 * max(1.0, abs(b)))]

    derived = {}
    for i in range(1, len(model.modes) + 1):
        _derive(derived, f"mode{i}", names, xs, cov, T_ref)
    for h in range(1, len(model.hops) + 1):
        _derive(derived, f"hop{h}", names, xs, cov, T_ref)

    overlay_kw = {k: v for k, v in kw.items() if k != "T_K"}
    overlay = motion_overlay_series(fitted, [
        {"ppm": p["ppm_full"], "y": p["y_full"], "T_K": p["T_K"], "label": p["label"]}
        for p in prepared], fit_offset=fit_offset, n_points=n_points, window=window,
        **overlay_kw)

    log.append(f"{'Converged' if sol.success else 'Did NOT converge'}: {sol.message} "
               f"Reduced chi-square (data normalised to each spectrum's maximum) = {redchi:.4g}.")
    for name in names:
        v, e = params[name], errors[name]
        note = ""
        if name.endswith("log10_k_ref"):
            note = f"  (k = {10 ** v:.4g} s^-1 at {T_ref:.1f} K)"
        elif name.endswith("log10_nu0"):
            note = f"  (nu0 = {10 ** v:.3g} s^-1)"
        elif name.endswith("log10_k"):
            note = f"  (k = {10 ** v:.4g} s^-1)"
        log.append(f"  {name} = {v:.6g} ± {e:.2g}{note}" + ("  [AT A BOUND]" if name in at_bound
                                                             else ""))
    for key, (v, e) in derived.items():
        log.append(f"  {key} = {v:.6g} ± {e:.2g}" + (f"  (nu0 = {10 ** v:.3g} s^-1)"
                                                    if key.endswith("log10_nu0") else ""))
    if len(names) > 1 and np.all(np.isfinite(corr)):
        big = [(names[a], names[b], corr[a, b]) for a in range(len(names))
               for b in range(a) if abs(corr[a, b]) > 0.95]
        for a, b, c in big:
            log.append(f"  ⚠ {a} and {b} are correlated ({c:+.3f}): only a combination of them "
                       "is determined; their separate errors are not independent.")
    if cond > 1e6:
        log.append(f"  ⚠ The problem is badly conditioned (condition number {cond:.2g}): at least "
                   "one parameter is not determined by these spectra.")
    if at_bound:
        log.append("  ⚠ " + ", ".join(at_bound) + " ended at a bound: widen it, or the parameter "
                   "is not constrained.")
    if getattr(sol, "not_converged", None) is not None:
        log.append(str(sol.not_converged))
    log.append("Errors are formal (Jacobian, scaled by the reduced chi-square): they assume "
               "independent points and a correct model, and are a lower bound.")
    for lab, s in zip((p["label"] for p in prepared), overlay["overlays"]):
        log.append(f"  {lab}: scale {s['scale']:.4g}, rms residual {100 * s['rms']:.2f} % of the "
                   "maximum.")
    return {"params": params, "errors": errors, "free": names, "fixed": fixed,
            "bounds": {n: (float(a), float(b)) for n, a, b in zip(names, lo, hi)},
            "at_bound": at_bound, "derived": derived, "model": fitted, "overlay": overlay,
            "covariance": cov, "correlation": corr, "condition": cond, "redchi": redchi,
            "chi2": float(np.sum(sol.fun ** 2)), "n_points": n_points_total,
            "success": bool(sol.success), "starts": starts, "T_ref_K": T_ref,
            "T_K": [p["T_K"] for p in prepared], "labels": [p["label"] for p in prepared],
            "kwargs": {k: v for k, v in kw.items()}, "seconds_per_evaluation": one_eval,
            "log": "\n".join(log)}


def _derive(out, tag, names, x, cov, T_ref):
    """log10 nu0 (and its error) from Ea and log10 k(T_ref), where both were fitted."""
    a, b = f"{tag}.Ea_eV", f"{tag}.log10_k_ref"
    if a in names and b in names and T_ref:
        ia, ib = names.index(a), names.index(b)
        c = 1.0 / (K_B_EV * T_ref * _LN10)
        value = float(x[ib] + c * x[ia])
        var = cov[ib, ib] + c * c * cov[ia, ia] + 2 * c * cov[ia, ib]
        out[f"{tag}.log10_nu0"] = (value, float(math.sqrt(max(var, 0.0))) if np.isfinite(var)
                                   else float("nan"))


# ----------------------------------------------------------------------
# A rate for every spectrum, then the Arrhenius line
# ----------------------------------------------------------------------

def arrhenius_regression(T_K, k_Hz, sigma_log10k=None, min_sigma=0.05) -> dict:
    """Straight line ``log10 k = log10 nu0 - Ea / (ln10 kB T)`` through rates by weighted least
    squares (``sigma_log10k``: errors in decades, floored at ``min_sigma`` so that no single
    point decides the line). Returns ``Ea_eV``, ``Ea_err_eV``, ``log10_nu0``,
    ``log10_nu0_err``, ``covariance``, ``redchi`` (None for two points), ``n`` and ``log``.
    The errors of Ea and nu0 are strongly correlated: quote Ea and k at a temperature in the
    middle of the data, and nu0 only as an extrapolation."""
    T = np.asarray(T_K, dtype=float)
    k = np.asarray(k_Hz, dtype=float)
    keep = np.isfinite(T) & np.isfinite(k) & (T > 0) & (k > 0)
    if keep.sum() < 2:
        raise ValueError("An Arrhenius line needs at least two rates.")
    T, k = T[keep], k[keep]
    y = np.log10(k)
    x = 1.0 / (K_B_EV * T * _LN10)                    # slope = -Ea
    if sigma_log10k is None:
        w = np.ones_like(y)
    else:
        s = np.maximum(np.asarray(sigma_log10k, dtype=float)[keep], float(min_sigma))
        w = 1.0 / s
    A = np.column_stack([-x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    n = int(T.size)
    resid = (y - A @ coef) * w
    dof = n - 2
    redchi = float(np.sum(resid ** 2) / dof) if dof > 0 else None
    scale = redchi if (redchi is not None and sigma_log10k is None) else (
        max(redchi, 1.0) if redchi is not None else float("nan"))
    cov = np.linalg.pinv((A * w[:, None]).T @ (A * w[:, None])) * scale if dof > 0 else np.full(
        (2, 2), np.nan)
    Ea, lnu = float(coef[0]), float(coef[1])
    e_Ea, e_nu = float(math.sqrt(max(cov[0, 0], 0))), float(math.sqrt(max(cov[1, 1], 0)))
    log = (f"Arrhenius line through {n} rates, {T.min():.0f} to {T.max():.0f} K: Ea = "
           f"{Ea:.4g} ± {e_Ea:.2g} eV ({Ea * 96.485:.1f} kJ/mol), log10 nu0 = {lnu:.3g} ± "
           f"{e_nu:.2g}"
           + (f", reduced chi-square {redchi:.3g}" if redchi is not None else "")
           + ". Ea and nu0 are strongly correlated.")
    return {"Ea_eV": Ea, "Ea_err_eV": e_Ea, "log10_nu0": lnu, "log10_nu0_err": e_nu,
            "covariance": cov, "redchi": redchi, "n": n, "T_K": T, "k_Hz": k, "log": log}


def fit_motion_rates(model, spectra, mode=1, k_range=(1e-1, 1e10), n_scan=None,
                     fit_offset=True, n_points=1024, window=None, delta_chi2=9.0,
                     **kwargs) -> dict:
    """The rate of one mode (``mode``, 1-based) fitted to each spectrum separately, then the
    Arrhenius line through the rates (:func:`arrhenius_regression`).

    Each spectrum is scanned on a grid of rates (half a decade apart over ``k_range``) and the
    best refined, so the answer does not depend on a starting guess. A rate counts as
    *determined* when moving it a decade either way worsens the fit by more than
    ``delta_chi2`` (default 9: three sigma) in units of the fit's own scatter; spectra in the
    slow or the fast limit, where the lineshape does not depend on the rate, are not, and are
    left out of the line (with their rate bounds reported). Everything else in the model
    stays at its value; the temperature of a spectrum only matters for the line afterwards,
    so ``T_K`` is optional here.

    Returns ``k_Hz`` (per spectrum), ``log10_k``, ``log10_k_err``, ``determined``, ``T_K``,
    ``arrhenius`` (None with fewer than two determined rates), ``overlay`` (data and fitted
    spectra, each with its own rate) and ``log``.
    """
    from scipy.optimize import least_squares

    from ..core import fit_progress as _fit_progress

    kw = _resolve_kwargs(spectra, kwargs)
    prepared = _prepare_spectra(spectra, n_points, window)
    d0 = model.to_dict()
    i = int(mode)
    if not 1 <= i <= len(d0["modes"]):
        raise ValueError(f"There is no mode {i}.")
    lo_k, hi_k = math.log10(k_range[0]), math.log10(k_range[1])
    if not lo_k < hi_k:
        raise ValueError("k_range must be (lowest, highest) with lowest < highest.")
    grid = np.arange(lo_k, hi_k + 1e-9, 0.5 if n_scan is None else (hi_k - lo_k) / (n_scan - 1))
    name = f"mode{i}.log10_k"
    rows, fits = [], []
    log = [f"Rate of mode {i} ({d0['modes'][i - 1].get('label') or d0['modes'][i - 1]['kind']}) "
           f"fitted to each of {len(prepared)} spectra: scan of {len(grid)} rates from "
           f"{k_range[0]:.3g} to {k_range[1]:.3g} s^-1, then refinement."]
    note = _grid_note(model, prepared, kw)
    if note:
        log.append(note)
    for p in prepared:
        _fit_progress.check_cancelled()
        _fit_progress.report_stage(f"Rate for {p['label']}")
        problem = _Problem(model, [p], [name], {}, None, {k: v for k, v in kw.items()
                                                          if k != "T_K"}, fit_offset)
        problem.simulate(np.array([grid[0]]))      # not guarded: say why it cannot be simulated
        costs = []
        for g in grid:
            r = problem.residuals(np.array([g]))
            costs.append(float(r @ r))
        g0 = float(grid[int(np.argmin(costs))])
        a, b = max(g0 - 1.0, lo_k - 1.0), min(g0 + 1.0, hi_k + 1.0)
        sol = least_squares(problem.residuals, np.array([g0]), bounds=([a], [b]), x_scale=1.0,
                            diff_step=1e-3)
        g = float(sol.x[0])
        best = float(sol.fun @ sol.fun)
        n = p["x"].size
        dof = max(n - 3, 1)
        s2 = best / dof
        up = problem.residuals(np.array([g + 1.0]))
        dn = problem.residuals(np.array([g - 1.0]))
        d_up, d_dn = (float(up @ up) - best) / s2, (float(dn @ dn) - best) / s2
        determined = bool(d_up > delta_chi2 and d_dn > delta_chi2 and a + 1e-9 < g < b - 1e-9
                          and lo_k <= g <= hi_k)
        J = sol.jac
        err = float(math.sqrt(s2 / max(float((J.T @ J)[0, 0]), 1e-300)))
        sims, fitted = problem.simulate(np.array([g]))
        rows.append({"label": p["label"], "T_K": p["T_K"], "log10_k": g, "err": err,
                     "determined": determined, "d_up": d_up, "d_dn": d_dn,
                     "cost": best})
        fits.append((p, fitted, sims[0]))
        state = "" if determined else "  (NOT determined: " + (
            "the lineshape is insensitive to the rate here – slow or fast limit" if min(
                d_up, d_dn) <= delta_chi2 else "at the edge of the scan range") + ")"
        log.append(f"  {p['label']}: k = {10 ** g:.4g} s^-1 (log10 k = {g:.3f} ± {err:.2g}){state}")
    T = np.array([r["T_K"] if r["T_K"] is not None else np.nan for r in rows])
    k = 10.0 ** np.array([r["log10_k"] for r in rows])
    det = np.array([r["determined"] for r in rows])
    arrhenius = None
    usable = det & np.isfinite(T)
    if usable.sum() >= 2:
        arrhenius = arrhenius_regression(T[usable], k[usable],
                                         np.array([r["err"] for r in rows])[usable])
        log.append(arrhenius["log"])
        if usable.sum() == 2:
            log.append("  ⚠ Two rates give a line but no test of it: add a temperature.")
    elif np.isfinite(T).any():
        log.append("Fewer than two determined rates: no Arrhenius line. The rates outside the "
                   "sensitive window are limits, not numbers.")
    overlay_items = []
    for p, fitted, sim in fits:
        res = sim["result"]
        resid = p["y"] - sim["sim"]
        overlay_items.append({
            "x_ppm": p["x"], "y": p["y"] * p["norm"], "sim": sim["sim"] * p["norm"],
            "residual": resid * p["norm"], "scale": sim["scale"] * p["norm"],
            "offset": sim["offset"] * p["norm"],
            "rms": float(np.sqrt(np.mean(resid ** 2))), "simulation": res, "log": res["log"],
            "window": p["window"]})
    overlay = {"overlays": overlay_items, "labels": [p["label"] for p, *_ in fits],
               "T_K": [p["T_K"] for p, *_ in fits],
               "log": "\n".join(f"{p['label']}: rms residual {100 * o['rms']:.2f} %"
                                for (p, *_), o in zip(fits, overlay_items))}
    return {"mode": i, "k_Hz": k, "log10_k": np.array([r["log10_k"] for r in rows]),
            "log10_k_err": np.array([r["err"] for r in rows]), "determined": det,
            "T_K": T, "rows": rows, "arrhenius": arrhenius, "overlay": overlay,
            "labels": overlay["labels"], "kwargs": dict(kw), "log": "\n".join(log)}


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def _arrhenius_axes(ax, text_size):
    ax.set_xlabel(r"1000/$T$ (K$^{-1}$)")
    ax.set_ylabel(r"log$_{10}$ $k$ (s$^{-1}$)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_motion_rates(rates, ax=None, text_size="small", colour=None, show_line=True):
    """The rates of :func:`fit_motion_rates` against 1000/T with the Arrhenius line through the
    determined ones; undetermined rates are drawn open (they are limits). Returns
    ``(fig, ax)``."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.375, 3.0))
    else:
        fig = ax.figure
    colour = colour or "C0"
    T = np.asarray(rates["T_K"], dtype=float)
    ok = np.isfinite(T)
    if not ok.any():
        raise ValueError("The spectra have no temperatures: there is nothing to put on 1000/T.")
    y, e, det = rates["log10_k"], rates["log10_k_err"], rates["determined"]
    for sel, filled in ((det & ok, True), (~det & ok, False)):
        if sel.any():
            ax.errorbar(1000.0 / T[sel], y[sel], yerr=np.maximum(e[sel], 0), fmt="o", ms=4.5,
                        color=colour, mfc=colour if filled else "none", capsize=2, lw=0.9)
    ar = rates["arrhenius"]
    if show_line and ar is not None:
        tt = np.linspace(T[ok].min() * 0.95, T[ok].max() * 1.05, 100)
        ax.plot(1000.0 / tt, ar["log10_nu0"] - ar["Ea_eV"] / (K_B_EV * tt * _LN10), "-",
                color="0.15", lw=1.0)
    notes = []
    if ar is not None:
        notes.append(f"$E_a$ (eV)\n{ar['Ea_eV']:.3g} ± {ar['Ea_err_eV']:.2g}")
    if (~det & ok).any():
        notes.append("open: not\ndetermined")
    if notes:
        ax.text(0.04, 0.04, "\n".join(notes), transform=ax.transAxes, ha="left", va="bottom",
                fontsize=text_size)
    _arrhenius_axes(ax, text_size)
    fig.tight_layout()
    return fig, ax


def plot_motion_fit(fit, fig=None, nucleus=None, text_size="small", offset=1.1):
    """Data and fit, stacked by temperature, beside the Arrhenius plot. ``fit`` is the result
    of :func:`fit_motion` (rate curves of the fitted model, drawn through the temperatures
    measured) or :func:`fit_motion_rates` (the rates with their line). With no temperatures
    only the stack is drawn. Returns ``(fig, (ax_stack, ax_arrhenius or None))``."""
    import matplotlib.pyplot as plt

    from .motion import plot_motion_overlay_stack

    overlay = fit["overlay"]
    T = np.array([t if t is not None else np.nan for t in overlay["T_K"]], dtype=float)
    with_T = bool(np.isfinite(T).any())
    if fig is None:
        fig = plt.figure(figsize=(6.5, 4.2))
    if with_T:
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.7, 1.0]})
    else:
        ax1, ax2 = fig.add_subplot(111), None
    plot_motion_overlay_stack(overlay, ax=ax1, nucleus=nucleus, offset=offset,
                              text_size=text_size)
    if ax2 is not None:
        if "arrhenius" in fit:                              # a fit_motion_rates result
            if fit.get("arrhenius") is not None or np.isfinite(fit["T_K"]).any():
                plot_motion_rates(fit, ax=ax2, text_size=text_size)
        else:
            _plot_fitted_rates(fit, ax2, T[np.isfinite(T)], text_size)
    fig.tight_layout()
    return fig, (ax1, ax2)


def _plot_fitted_rates(fit, ax, T, text_size):
    model = fit["model"]
    tt = np.linspace(T.min() * 0.95, T.max() * 1.05, 100)
    notes = []
    errors = fit.get("errors", {})
    for i, m in enumerate(model.modes, 1):
        if m.rate.get("Ea_eV") is None:
            continue
        k = m.rate.get("nu0_Hz", 1e12) * np.exp(-m.rate["Ea_eV"] / (K_B_EV * tt))
        line, = ax.plot(1000.0 / tt, np.log10(k), "-", lw=1.1)
        kk = m.rate.get("nu0_Hz", 1e12) * np.exp(-m.rate["Ea_eV"] / (K_B_EV * T))
        ax.plot(1000.0 / T, np.log10(kk), "o", ms=4, color=line.get_color())
        e = errors.get(f"mode{i}.Ea_eV")
        notes.append((line.get_color(), f"mode {i} $E_a$ (eV)\n{m.rate['Ea_eV']:.3g}"
                      + (f" ± {e:.2g}" if e is not None and np.isfinite(e) else "")))
    for h, hop in enumerate(model.hops, 1):
        if hop.get("Ea_eV") is None:
            continue
        k = hop.get("nu0_Hz", 1e12) * np.exp(-float(hop["Ea_eV"]) / (K_B_EV * tt))
        line, = ax.plot(1000.0 / tt, np.log10(k), "--", lw=1.1)
        e = errors.get(f"hop{h}.Ea_eV")
        notes.append((line.get_color(), f"hop {h} $E_a$ (eV)\n{float(hop['Ea_eV']):.3g}"
                      + (f" ± {e:.2g}" if e is not None and np.isfinite(e) else "")))
    if len(notes) == 1:
        notes = [(notes[0][0], "$E_a$ (eV)\n" + notes[0][1].split("\n", 1)[1])]
    for k, (colour, text) in enumerate(notes):
        ax.text(0.04, 0.04 + 0.17 * k, text, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=text_size, color=colour)
    _arrhenius_axes(ax, text_size)


# ----------------------------------------------------------------------
# Notebook code
# ----------------------------------------------------------------------

def motion_fit_code(model, parameters, route="global", mode=1, T_ref_K=None, kwargs=None,
                    n_starts=0, nucleus=None) -> str:
    """Python that repeats a GUI fit in a notebook. ``parameters``: ``{name: {'value', 'min',
    'max', 'vary'}}`` as in :func:`fit_motion`; ``route``: 'global' or 'rates'. The experimental
    spectra are a placeholder (``spectra = [{'ppm': ..., 'y': ..., 'T_K': ...}, ...]``)."""
    def fmt(obj):
        return repr(obj)
    kwargs = dict(kwargs or {})
    kw_text = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    lines = [
        "import numpy as np",
        "from opennmr.notebook import motion",
        "from opennmr.notebook import motion_fit",
        "",
        f"model = motion.MotionModel.from_dict({fmt(model.to_dict())})",
        "",
        "# One dict per experimental spectrum (e.g. from your loaded data):",
        "spectra = [{'ppm': ppm_array, 'y': intensity_array, 'T_K': 300.0}]  # ← fill in",
        "",
    ]
    if route == "rates":
        lines += [f"rates = motion_fit.fit_motion_rates(model, spectra, mode={int(mode)}"
                  + (f", {kw_text}" if kw_text else "") + ")",
                  "print(rates['log'])",
                  f"fig, ax = motion_fit.plot_motion_rates(rates)"]
    else:
        clean = {}
        for n, s in parameters.items():
            clean[n] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in s.items()}
        lines += [f"parameters = {fmt(clean)}",
                  "fit = motion_fit.fit_motion(model, spectra, parameters"
                  + (f", T_ref_K={float(T_ref_K)!r}" if T_ref_K else "")
                  + (f", n_starts={int(n_starts)}" if n_starts else "")
                  + (f", {kw_text}" if kw_text else "") + ")",
                  "print(fit['log'])",
                  f"fig, axes = motion_fit.plot_motion_fit(fit, nucleus={nucleus!r})"]
    return "\n".join(lines)
