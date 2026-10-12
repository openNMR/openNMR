"""
Fit the same set of sites to several spectra at once -- Combined-
Hamiltonian sites (:mod:`opennmr.notebook.combined_hamiltonian`), so
anything from an isotropic line to CSA + quadrupolar + J.

Three situations, one engine:

1. **A series** (variable temperature, variable time, ...): each spectrum
   is fitted in the order given, each seeded from the previous fit.
   Parameters that should not change along the series (a C_Q, a CSA, a
   site ratio) can be made *global* and are then fitted once, jointly.
2. **Different isotopes of one material** (e.g. 1H and 2H): sites are
   defined per nucleus, and **the same site label is the same chemical
   site**. Shielding (δiso, CSA, in ppm) is shared; the EFG is shared, so
   C_Q scales by the ratio of quadrupole moments and η_Q is common; J
   scales as γ_A·γ_X (the reduced coupling is isotope-independent). A
   per-isotope δ offset (primary isotope shift) can be freed.
3. **One isotope at several fields** (and spinning rates): everything
   chemical is shared -- which is what separates δiso from the second-
   order quadrupolar shift -- and the linewidth is split into a field-
   independent part (Hz) and a field-proportional part (ppm).

Every parameter has a **scope**: ``"experiment"`` (fitted separately in
each spectrum), ``"nucleus"`` (one value for every spectrum of that
isotope) or ``"global"`` (one value across all). Site amounts have one
scope for all sites (``amounts=``): per experiment, or a site ratio
shared within each nucleus / across everything, with a free overall
scale per spectrum.

How it runs: a **sequential pass** always comes first -- each spectrum
fitted on its own (:func:`~opennmr.notebook.combined_fitting.fit_combined`),
seeded from the previous one (``seed="previous"``) or from the starting
model (``seed="start"``). If every parameter is per-experiment, that is
the answer. Otherwise a **joint refinement** starts from it and fits
every spectrum at once, each weighted by its own residual noise from
the sequential pass, so no spectrum dominates by being more intense.

Linewidth split: FWHM(ppm) = width_Hz / ν0(MHz) (+) width_ppm, where
(+) adds in quadrature for a Gaussian and linearly otherwise (the widths
of convolved Gaussians add in quadrature; of Lorentzians, linearly).
With one field in a group the two cannot be told apart, so width_Hz is
held at 0 there (and the log says so).

Qt-free: the GUI's Fit > Multi-spectrum calls exactly these functions.
"""

from __future__ import annotations

import copy
import csv
import io as _io

import numpy as np
from scipy.optimize import least_squares

from .combined_fitting import (_default_bounds, _derived, _fmt, _get, _set, fit_combined,
                               site_parameter_names)
from .combined_hamiltonian import Site, _broaden_site, _site_sticks
from .fitting import build_region_mask, fit_quality
from .utils import _GAMMA_DF
from ..core import fit_progress as _fit_progress

SCOPES = ("experiment", "nucleus", "global")
PRESETS = ("series", "isotopes", "fields")

#: Site parameters that describe the chemistry (shielding, EFG) rather
#: than the spectrum -- shared by the "isotopes"/"fields" presets.
CHEMICAL = ("delta_iso", "aniso", "asym", "alpha", "beta", "gamma", "C_Q", "eta_Q")


class MultiFitCancelled(Exception):
    """Raised when a ``progress`` callback returns False."""


# ============================================================
# Experiments and isotopes
# ============================================================

def make_experiment(results, label=None, nucleus=None, larmor_Hz=None, mas_Hz=0.0,
                    order="second", transitions="all", rotor_angle="magic", n_zcw=None,
                    fit_regions=None, fit_baseline=False, x_value=None, fit_mas=False,
                    mas_window_Hz=500.0):
    """One spectrum to fit, with its own acquisition settings.

    ``results`` : ``x_ppm``, ``real_spectrum``, ``SFO1`` (MHz) and
    ``nuclide`` -- NMR1dimport output or the GUI's
    ``bridge.results_from_spectrum``. ``x_value``: where it sits in a
    series (temperature, time, ...), for the trend table and plots.
    ``fit_mas``: let this spectrum's MAS rate vary within ``mas_Hz`` ±
    ``mas_window_Hz`` in its own fit (see :func:`fit_combined`); a joint
    refinement then holds it at that fitted value."""
    nucleus = nucleus or results.get("nuclide")
    if not nucleus:
        raise ValueError(f"{label or 'An experiment'} has no nucleus -- pass nucleus=.")
    larmor = float(larmor_Hz) if larmor_Hz else float(results["SFO1"]) * 1e6
    return {"results": results, "label": str(label or f"#{id(results) % 10000}"),
            "nucleus": str(nucleus), "larmor_Hz": larmor, "mas_Hz": float(mas_Hz or 0.0),
            "order": order, "transitions": transitions, "rotor_angle": rotor_angle,
            "n_zcw": n_zcw, "fit_regions": fit_regions, "fit_baseline": bool(fit_baseline),
            "x_value": x_value, "fit_mas": bool(fit_mas) and float(mas_Hz or 0.0) > 0,
            "mas_window_Hz": float(mas_window_Hz)}


def _isotope(nucleus, column):
    rows = _GAMMA_DF[_GAMMA_DF["Nuclide"] == nucleus]
    if rows.empty:
        raise KeyError(f"{nucleus!r} is not in the isotope table.")
    try:
        return float(rows[column].iloc[0])
    except (TypeError, ValueError):
        return None


def quadrupole_moment(nucleus):
    """|Q| (fm²), or None for a spin-1/2 nucleus."""
    q = _isotope(nucleus, "Quadrupole Moment (fm^2)")
    return abs(q) if q is not None else None


def gyromagnetic_ratio(nucleus):
    return _isotope(nucleus, "Gyromagnetic Ratio (MHz/T)")


# ============================================================
# Scopes
# ============================================================

def default_scopes(preset="series"):
    """``(scopes, amounts)`` for a preset: ``scopes`` maps a parameter
    name (``"J"`` stands for every J group) to its scope.

    * ``"series"``: everything per experiment -- a sequential fit.
    * ``"isotopes"``: shielding, EFG and J shared (global); widths and
      site amounts per nucleus / per experiment.
    * ``"fields"``: everything chemical global; linewidths split into
      Hz + ppm per nucleus; site ratios shared per nucleus.
    """
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}.")
    if preset == "series":
        return {name: "experiment" for name in CHEMICAL + ("J", "fwhm", "eta")}, "experiment"
    scopes = {name: "global" for name in CHEMICAL + ("J",)}
    if preset == "isotopes":
        scopes.update(fwhm="experiment", eta="experiment")
        return scopes, "nucleus"
    scopes.update(fwhm="nucleus", eta="nucleus")
    return scopes, "nucleus"


def _scope_of(scopes, label, name):
    key = "J" if name.startswith("J") else name
    per_site = (scopes or {}).get(label)
    if isinstance(per_site, dict) and (name in per_site or key in per_site):
        return per_site.get(name, per_site.get(key))
    value = (scopes or {}).get(name, (scopes or {}).get(key, "experiment"))
    if value not in SCOPES:
        raise ValueError(f"{label}: scope of {name} must be one of {SCOPES}, got {value!r}.")
    return value


def _names_for(spec, label):
    """``vary``/``fixed`` in the engine's forms: a list (every site) or
    ``{label: [names]}``. None means "not given"."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec.get(label, spec.get("*"))
    return list(spec)


def _bounds_for(bounds, label, name):
    if not bounds:
        return None
    per_site = bounds.get(label)
    if isinstance(per_site, dict) and name in per_site:
        return tuple(per_site[name])
    value = bounds.get(name)
    return tuple(value) if value is not None and not isinstance(value, dict) else None


def _templates(sites_by_nucleus):
    """{nucleus: [site dicts with nucleus]} from Sites or dicts."""
    out = {}
    for nucleus, sites in sites_by_nucleus.items():
        dicts = []
        for s in (sites if isinstance(sites, (list, tuple)) else [sites]):
            d = s.to_dict() if isinstance(s, Site) else copy.deepcopy(dict(s))
            d["nucleus"] = nucleus
            dicts.append(Site.from_dict(d).to_dict())
        labels = [d["label"] for d in dicts]
        if len(set(labels)) != len(labels):
            raise ValueError(f"{nucleus}: site labels must be unique, got {labels}.")
        out[nucleus] = dicts
    return out


def _j_partner(d, k):
    groups = [i for i in d["interactions"] if i["type"] == "jcoupling"]
    return groups[k]["nuclide"] if k < len(groups) else None


class _Map:
    """How every experiment's site parameters come from the fitted
    parameters (keys), with the scaling each link carries."""

    def __init__(self, experiments, templates, scopes, amounts, vary, fixed, bounds):
        self.experiments = experiments
        self.templates = templates
        self.keys = {}
        self.links = []
        self.notes = []

        def group(scope, i, nucleus):
            return f"e{i}" if scope == "experiment" else (f"n:{nucleus}" if scope == "nucleus" else "all")

        def ref_nucleus(label, name, grp):
            # The first member experiment (in order) whose site has this
            # parameter: its nucleus is the one the key is written in.
            for i, e in enumerate(experiments):
                if grp.startswith("e") and grp != f"e{i}":
                    continue
                if grp.startswith("n:") and e["nucleus"] != grp[2:]:
                    continue
                site = self._site(e["nucleus"], label)
                if site is not None and name in site_parameter_names(site):
                    return e["nucleus"], site
            return None, None

        def add_key(key, value, free, lower, upper, scope, site_for_bounds=None):
            if key in self.keys:
                return
            label, name, grp = key
            user = _bounds_for(bounds, label, name)
            if user is not None:
                lower, upper = min(user), max(user)
            self.keys[key] = {"label": label, "name": name, "group": grp, "scope": scope,
                              "value": float(value), "free": bool(free),
                              "lower": float(lower), "upper": float(upper)}

        def is_free(label, name):
            # width_Hz / width_ppm follow "fwhm" in vary/fixed too.
            names = {name, "fwhm"} if name in ("width_Hz", "width_ppm") else {name}
            v, f = _names_for(vary, label), _names_for(fixed, label) or []
            base = (name not in ("alpha", "beta", "gamma", "delta_offset")) if v is None \
                else bool(names & set(v))
            return base and not (names & set(f))

        self._amount_groups = {}
        for i, e in enumerate(experiments):
            n = e["nucleus"]
            links = {}
            if n not in templates:
                raise ValueError(f"{e['label']}: no sites are defined for {n}.")
            uses_scale = False
            for d in templates[n]:
                label = d["label"]
                links[label] = {}
                for name in site_parameter_names(d):
                    if name == "amount":
                        if amounts == "experiment":
                            key = (label, "amount", f"e{i}")
                            add_key(key, d["amount"], is_free(label, "amount"), 0.0, np.inf,
                                    "experiment")
                            links[label][name] = ("id", key)
                        else:
                            grp = "all" if amounts == "global" else f"n:{n}"
                            key = (label, "amount", grp)
                            add_key(key, d["amount"], is_free(label, "amount"), 0.0, np.inf,
                                    amounts)
                            links[label][name] = ("scaled_amount", key, ("", "scale", f"e{i}"))
                            uses_scale = True
                            self._amount_groups.setdefault(grp, []).append(i)
                        continue
                    scope = _scope_of(scopes, label, name)
                    if name == "fwhm" and scope != "experiment":
                        grp = group(scope, i, n)
                        k_hz, k_ppm = (label, "width_Hz", grp), (label, "width_ppm", grp)
                        add_key(k_hz, 0.0, is_free(label, "width_Hz"), 0.0, np.inf, scope)
                        add_key(k_ppm, d["fwhm"], is_free(label, "width_ppm"), 0.0, np.inf, scope)
                        links[label][name] = ("width", k_hz, k_ppm, d.get("lineshape", "gaussian"))
                        continue
                    grp = group(scope, i, n)
                    key = (label, name, grp)
                    ref_n, ref_site = ref_nucleus(label, name, grp)
                    k = 1.0
                    if ref_n is not None and ref_n != n:
                        if name == "C_Q":
                            # One EFG: C_Q = e Q V_zz / h scales with Q.
                            q, q_ref = quadrupole_moment(n), quadrupole_moment(ref_n)
                            k = q / q_ref if q and q_ref else 1.0
                        elif name.startswith("J"):
                            # J = (reduced coupling) x gamma_A x gamma_X.
                            j = int(name[1:]) - 1
                            x, x_ref = _j_partner(d, j), _j_partner(ref_site, j)
                            k = (gyromagnetic_ratio(n) * gyromagnetic_ratio(x)
                                 / (gyromagnetic_ratio(ref_n) * gyromagnetic_ratio(x_ref)))
                    lo, hi = _default_bounds(ref_site if ref_site is not None else d, name)
                    start = _get(ref_site if ref_site is not None else d, name)
                    add_key(key, start, is_free(label, name), lo, hi, scope)
                    if name == "delta_iso" and scope == "global" and ref_n is not None and ref_n != n:
                        # A primary isotope shift, per nucleus; held at 0 unless freed.
                        off = (label, "delta_offset", f"n:{n}")
                        add_key(off, 0.0, is_free(label, "delta_offset"), -np.inf, np.inf, "nucleus")
                        links[label][name] = ("offset", key, off)
                        continue
                    links[label][name] = ("scaled", key, k)
            if uses_scale:
                add_key(("", "scale", f"e{i}"), 1.0, True, 0.0, np.inf, "experiment")
            self.links.append(links)

        # One spectrum per shared-ratio group anchors the scale.
        for grp, members in self._amount_groups.items():
            first = sorted(set(members), key=members.index)[0]
            self.keys[("", "scale", f"e{first}")]["free"] = False

        # A Hz/ppm linewidth split needs at least two fields in the group.
        for key, row in self.keys.items():
            if key[1] != "width_Hz":
                continue
            fields = {round(experiments[i]["larmor_Hz"]) for i in self._members(key)}
            if len(fields) < 2 and row["free"]:
                row["free"] = False
                self.notes.append(f"{key[0]}: one field only in {self._describe_group(key[2])}, "
                                  "so its linewidth is fitted in ppm alone (width_Hz held at 0)")

    def _site(self, nucleus, label):
        for d in self.templates.get(nucleus, []):
            if d["label"] == label:
                return d
        return None

    def _members(self, key):
        out = []
        for i, links in enumerate(self.links):
            for per in links.values():
                for link in per.values():
                    if key in link[1:] and i not in out:
                        out.append(i)
        return out

    def _describe_group(self, grp):
        if grp == "all":
            return "all experiments"
        if grp.startswith("n:"):
            return f"the {grp[2:]} experiments"
        return self.experiments[int(grp[1:])]["label"]

    def site_dicts(self, i, values):
        """Experiment ``i``'s site dicts for key values ``values``."""
        e = self.experiments[i]
        out = []
        for d in self.templates[e["nucleus"]]:
            d = copy.deepcopy(d)
            for name, link in self.links[i][d["label"]].items():
                kind = link[0]
                if kind == "id":
                    v = values[link[1]]
                elif kind == "scaled":
                    v = values[link[1]] * link[2]
                elif kind == "offset":
                    v = values[link[1]] + values[link[2]]
                elif kind == "scaled_amount":
                    v = values[link[1]] * values[link[2]]
                else:  # width: Hz part and ppm part
                    a = values[link[1]] / (e["larmor_Hz"] / 1e6)
                    b = values[link[2]]
                    v = float(np.hypot(a, b)) if link[3] == "gaussian" else a + b
                    v = max(v, 1e-4)
                _set(d, name, v)
            out.append(d)
        return out

    def seed_keys(self, per_experiment):
        """Start every key from per-experiment site dicts (the
        sequential pass): the mean of what each member experiment found,
        mapped back through its link."""
        plain, offsets, shares, totals, widths = {}, {}, {}, {}, {}
        for i, dicts in enumerate(per_experiment):
            if dicts is None:
                continue
            by_label = {d["label"]: d for d in dicts}
            total = sum(d["amount"] for d in dicts) or 1.0
            for label, per in self.links[i].items():
                d = by_label.get(label)
                if d is None:
                    continue
                for name, link in per.items():
                    kind, value = link[0], _get(d, name)
                    if kind == "id":
                        plain.setdefault(link[1], []).append(value)
                    elif kind == "scaled":
                        plain.setdefault(link[1], []).append(value / link[2] if link[2] else value)
                    elif kind == "offset":
                        offsets.setdefault((link[1], link[2]), []).append(value)
                        if self.keys[link[2]]["value"] == 0.0 and not self.keys[link[2]]["free"]:
                            plain.setdefault(link[1], []).append(value)
                    elif kind == "scaled_amount":
                        shares.setdefault(link[1], []).append(value / total)
                        totals.setdefault(link[2], []).append(total)
                    else:
                        widths.setdefault((link[1], link[2], link[3]), []).append(
                            (self.experiments[i]["larmor_Hz"] / 1e6, value))
        for key, got in plain.items():
            self.keys[key]["value"] = float(np.mean(got))
        # A free isotope offset starts from what that nucleus's spectra
        # found against the shared value.
        for (key, off), got in offsets.items():
            if self.keys[off]["free"]:
                self.keys[off]["value"] = float(np.mean(got)) - self.keys[key]["value"]
        # Shared site ratios: each spectrum's scale relative to the group's
        # first, and the shares in that first spectrum's units.
        mean_totals = {k: float(np.mean(v)) for k, v in totals.items()}
        for grp, members in self._amount_groups.items():
            first = members[0]
            ref_total = mean_totals.get(("", "scale", f"e{first}"), 1.0) or 1.0
            for i in dict.fromkeys(members):
                key = ("", "scale", f"e{i}")
                self.keys[key]["value"] = mean_totals.get(key, ref_total) / ref_total
            for key, got in shares.items():
                if key[2] == grp:
                    self.keys[key]["value"] = float(np.mean(got)) * ref_total
        # Linewidths across fields: FWHM(ppm) against 1/nu0 separates the
        # Hz part from the ppm part.
        for (k_hz, k_ppm, shape), got in widths.items():
            nu0 = np.array([g[0] for g in got])
            w = np.array([g[1] for g in got])
            if self.keys[k_hz]["free"] and len(set(np.round(nu0, 3))) > 1:
                if shape == "gaussian":
                    A = np.column_stack([1 / nu0 ** 2, np.ones_like(nu0)])
                    (a2, b2), *_ = np.linalg.lstsq(A, w ** 2, rcond=None)
                    a, b = np.sqrt(max(a2, 0.0)), np.sqrt(max(b2, 1e-8))
                else:
                    A = np.column_stack([1 / nu0, np.ones_like(nu0)])
                    (a, b), *_ = np.linalg.lstsq(A, w, rcond=None)
                    a, b = max(a, 0.0), max(b, 1e-4)
                self.keys[k_hz]["value"], self.keys[k_ppm]["value"] = float(a), float(b)
            else:
                self.keys[k_ppm]["value"] = float(np.mean(w))
        for row in self.keys.values():
            row["value"] = float(np.clip(row["value"], row["lower"], row["upper"]))

    def sequential_vary(self, i):
        """What a single-spectrum fit of experiment ``i`` may vary."""
        vary = {}
        for label, per in self.links[i].items():
            names = []
            for name, link in per.items():
                if link[0] == "width":
                    free = self.keys[link[1]]["free"] or self.keys[link[2]]["free"]
                elif link[0] == "scaled_amount":
                    free = self.keys[link[1]]["free"]
                else:
                    free = self.keys[link[1]]["free"]
                if free:
                    names.append(name)
            vary[label] = names
        return vary

    def sequential_bounds(self, i):
        out = {}
        for label, per in self.links[i].items():
            for name, link in per.items():
                if link[0] == "scaled":
                    row = self.keys[link[1]]
                    lo, hi = row["lower"] * link[2], row["upper"] * link[2]
                    out.setdefault(label, {})[name] = (min(lo, hi), max(lo, hi))
                elif link[0] in ("id",):
                    row = self.keys[link[1]]
                    out.setdefault(label, {})[name] = (row["lower"], row["upper"])
        return out


def _seed_from(template, same_nucleus_previous, other_previous):
    """A template site started from the last fit of the same site."""
    if same_nucleus_previous is not None:
        return copy.deepcopy(same_nucleus_previous)
    d = copy.deepcopy(template)
    if other_previous is None:
        return d
    names = set(site_parameter_names(d)) & set(site_parameter_names(other_previous))
    for name in names & {"delta_iso", "aniso", "asym", "alpha", "beta", "gamma", "eta_Q"}:
        _set(d, name, _get(other_previous, name))
    return d


# ============================================================
# The fit
# ============================================================

def fit_multi(experiments, sites_by_nucleus, scopes=None, amounts=None, vary=None, fixed=None,
              bounds=None, seed="previous", joint=None, preset=None, max_nfev=None,
              progress=None):
    """
    Fit CSA, Quad & J-coupling sites to several spectra, consistently.

    Parameters
    ----------
    experiments : list of :func:`make_experiment` dicts, **in the order
        to fit them** (any order: 5, 302, 12, 2, 49 is fine).
    sites_by_nucleus : ``{nucleus: [Site or dict, ...]}`` -- the starting
        model for each observed isotope. The same label is the same site.
    scopes : ``{name: scope}`` (every site) and/or ``{label: {name:
        scope}}``; ``"J"`` stands for every J group. Default: the preset,
        else every parameter per experiment.
    amounts : ``"experiment"`` (default) | ``"nucleus"`` | ``"global"`` --
        whether the site ratio is shared (with a free overall scale per
        spectrum).
    vary, fixed, bounds : as :func:`fit_combined`, applied to the shared
        parameters; also ``delta_offset`` (per-isotope δ offset, fixed
        unless varied), ``width_Hz``/``width_ppm`` (the linewidth split).
    seed : ``"previous"`` (each spectrum starts from the last fit of the
        same site) or ``"start"`` (each from the starting model).
    joint : refine every spectrum at once; default: whenever anything is
        shared (a non-experiment scope, or shared amounts).
    preset : ``"series"`` | ``"isotopes"`` | ``"fields"``: default scopes
        and amounts (see :func:`default_scopes`); explicit arguments win.
    progress : ``progress(done, total, message)``; return False to stop
        (raises :class:`MultiFitCancelled`).

    Returns
    -------
    dict with ``mode`` ('sequential' or 'joint'); ``experiments``: one
    fit per experiment, in order, shaped like :func:`fit_combined`'s
    (``parameters`` with value, error, scope; ``derived``; ``x``, ``y``,
    ``y_fit``, ``site_spectra``, ``baseline``, ``mask``; ``quality``)
    plus ``label``, ``nucleus``, ``x_value``, ``larmor_Hz``, ``mas_Hz``;
    ``shared``: the shared parameters (joint only); ``sequential``: the
    sequential pass's fits; ``warnings``; ``log``; ``settings``.
    """
    experiments = list(experiments)
    if not experiments:
        raise ValueError("Give at least one experiment.")
    if preset is not None:
        p_scopes, p_amounts = default_scopes(preset)
        scopes = scopes if scopes is not None else p_scopes
        amounts = amounts if amounts is not None else p_amounts
    amounts = amounts or "experiment"
    if amounts not in SCOPES:
        raise ValueError(f"amounts must be one of {SCOPES}.")
    if seed not in ("previous", "start"):
        raise ValueError("seed must be 'previous' or 'start'.")
    labels = [e["label"] for e in experiments]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Experiment labels must be unique, got {labels}.")
    templates = _templates(sites_by_nucleus)
    pmap = _Map(experiments, templates, scopes or {}, amounts, vary, fixed, bounds)
    shared = any(row["scope"] != "experiment" for row in pmap.keys.values())
    if joint is None:
        joint = shared and len(experiments) > 1
    n_steps = len(experiments) + (1 if joint else 0)
    warnings = list(pmap.notes)

    def tick(done, message):
        if progress is not None and progress(done, n_steps, message) is False:
            raise MultiFitCancelled(message)

    # ---- the sequential pass: each spectrum on its own, in order ----------------
    sequential, seq_dicts = [], []
    last_by_nucleus, last_any = {}, {}
    for i, e in enumerate(experiments):
        tick(i, f"Fitting {e['label']} ({i + 1} of {len(experiments)})")
        start_values = {key: row["value"] for key, row in pmap.keys.items()}
        start = pmap.site_dicts(i, start_values)
        if seed == "previous":
            start = [_seed_from(d, last_by_nucleus.get(e["nucleus"], {}).get(d["label"]),
                                last_any.get(d["label"]))
                     for d in start]
        try:
            fit = fit_combined(e["results"], start, vary=pmap.sequential_vary(i),
                               bounds=pmap.sequential_bounds(i) or None, larmor_Hz=e["larmor_Hz"],
                               mas_Hz=e["mas_Hz"], order=e["order"], transitions=e["transitions"],
                               rotor_angle=e["rotor_angle"], n_zcw=e["n_zcw"],
                               fit_regions=e["fit_regions"], fit_baseline=e["fit_baseline"],
                               fit_mas=e.get("fit_mas", False),
                               mas_window_Hz=e.get("mas_window_Hz", 500.0))
        except (RuntimeError, ValueError) as exc:
            warnings.append(f"{e['label']}: the single-spectrum fit failed ({exc}); its starting "
                            "values were kept")
            sequential.append(None)
            seq_dicts.append(None)
            continue
        fitted = [s.to_dict() for s in fit["sites"]]
        if e.get("fit_mas"):
            # This spectrum's own rate from here on: the joint refinement
            # (and the report) use it, held.
            mas_row = next(p for p in fit["parameters"] if p["name"] == "mas_Hz")
            experiments[i] = e = {**e, "mas_Hz": float(mas_row["value"]),
                                  "mas_fitted": (float(mas_row["value"]),
                                                 mas_row["error"], e["mas_Hz"])}
        sequential.append(fit)
        seq_dicts.append(fitted)
        last_by_nucleus[e["nucleus"]] = {d["label"]: d for d in fitted}
        last_any.update({d["label"]: d for d in fitted})

    r2 = [f["quality"]["r2"] if f else np.nan for f in sequential]
    for i in range(1, len(r2)):
        if np.isfinite(r2[i]) and np.isfinite(r2[i - 1]) and r2[i] < r2[i - 1] - 0.02:
            warnings.append(f"{experiments[i]['label']}: R² fell to {r2[i]:.4f} from "
                            f"{r2[i - 1]:.4f} -- if the spectra did not change that much, the "
                            "seeded start may have drifted; refit with seed='start' to check")
    for i, v in enumerate(r2):
        if np.isfinite(v) and v < 0.9:
            warnings.append(f"{experiments[i]['label']}: poor fit (R² {v:.3f})")
    settings = {"scopes": scopes or {}, "amounts": amounts, "seed": seed, "joint": bool(joint),
                "preset": preset}

    if not joint:
        tick(n_steps, "Done")
        out = []
        for e, fit in zip(experiments, sequential):
            if fit is None:
                continue
            fit = dict(fit)
            fit["parameters"] = [{**p, "scope": "experiment"} for p in fit["parameters"]]
            fit["warnings"] = [f"{e['label']}: {w}" for w in fit["warnings"]]
            out.append(_decorate(fit, e))
        for f in out:
            warnings += f["warnings"]
        return _finish("sequential", out, [], sequential, warnings, settings, experiments)

    # ---- the joint refinement ----------------------------------------------------------
    tick(len(experiments), "Joint refinement of every spectrum at once")
    pmap.seed_keys(seq_dicts)
    keys = list(pmap.keys)
    rows = [pmap.keys[k] for k in keys]
    free = np.array([r["free"] for r in rows])
    p0 = np.array([r["value"] for r in rows], dtype=float)
    lo = np.array([r["lower"] for r in rows])[free]
    hi = np.array([r["upper"] for r in rows])[free]

    data = []
    for i, e in enumerate(experiments):
        seq = sequential[i]
        x_full = np.asarray(e["results"]["x_ppm"], dtype=float)
        idx = np.argsort(x_full)
        x = x_full[idx]
        y = np.asarray(e["results"]["real_spectrum"], dtype=float)[idx]
        mask = build_region_mask(x, e["fit_regions"])
        # The sequential pass's baseline is held; its residual noise is
        # this spectrum's weight.
        baseline = np.asarray(seq["baseline"]) if seq is not None else np.zeros_like(x)
        rmse = float(seq["quality"]["rmse"]) if seq is not None else float(np.std(y[mask]))
        sigma = max(rmse, 1e-9 * (float(np.max(np.abs(y))) or 1.0))
        beta_r = (np.arccos(1 / np.sqrt(3)) if e["rotor_angle"] == "magic"
                  else np.deg2rad(float(e["rotor_angle"])))
        n_zcw = e["n_zcw"] or (2584 if e["mas_Hz"] else 28657)
        data.append({"x": x, "y": y, "mask": mask, "baseline": baseline, "sigma": sigma,
                     "beta_r": beta_r, "n_zcw": int(n_zcw), "cache": {}})

    def curves(i, dicts):
        e, dd = experiments[i], data[i]
        out = []
        for d in dicts:
            key = repr({k: v for k, v in d.items() if k not in ("amount", "fwhm", "eta")})
            if key not in dd["cache"]:
                if len(dd["cache"]) > 128:
                    dd["cache"].clear()
                dd["cache"][key] = _site_sticks(Site.from_dict(d), dd["x"], e["larmor_Hz"],
                                                e["mas_Hz"], e["order"], e["transitions"],
                                                dd["beta_r"], dd["n_zcw"])[0]
            out.append(np.sum(list(_broaden_site(Site.from_dict(d), dd["cache"][key],
                                                 dd["x"]).values()), axis=0))
        return out

    def values_of(theta):
        p = p0.copy()
        p[free] = theta
        return dict(zip(keys, p))

    calls = [0]

    def residuals(theta):
        calls[0] += 1
        if progress is not None and calls[0] % 20 == 0:
            tick(len(experiments), f"Joint refinement: {calls[0]} evaluations")
        values = values_of(theta)
        out = []
        for i in range(len(experiments)):
            dd = data[i]
            model = np.sum(curves(i, pmap.site_dicts(i, values)), axis=0) + dd["baseline"]
            out.append((dd["y"][dd["mask"]] - model[dd["mask"]]) / dd["sigma"])
        return np.concatenate(out)

    if free.any():
        # max_nfev: the iteration limit; None uses the app-wide one
        # (2000 unless changed). Reaching it keeps the result, flagged.
        solution = _fit_progress.limited_least_squares(
            _fit_progress.tracked_residuals(residuals, n_free=int(free.sum())),
            p0[free], bounds=(lo, hi), x_scale="jac", diff_step=1e-4,
            max_iterations=max_nfev, raise_on_limit=False, label="The joint refinement")
        theta = solution.x
        J = solution.jac
        dof = max(solution.fun.size - int(free.sum()), 1)
        s2 = float(np.sum(solution.fun ** 2)) / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        if solution.not_converged is not None:
            warnings.append(str(solution.not_converged))
        elif not solution.success:
            warnings.append(f"The joint refinement stopped without converging ({solution.message})")
        chi2_red = s2
    else:
        theta, cov, chi2_red = np.zeros(0), np.zeros((0, 0)), float("nan")
    values = values_of(theta)
    errors = dict.fromkeys(keys, None)
    free_keys = [k for k, f in zip(keys, free) if f]
    for j, key in enumerate(free_keys):
        errors[key] = float(np.sqrt(max(cov[j, j], 0.0)))

    shared_rows = []
    for key, row in zip(keys, rows):
        if row["scope"] == "experiment" and key[1] != "scale":
            continue
        shared_rows.append({**row, **{"value": float(values[key]), "error": errors[key]}})
    for r in shared_rows:
        if r["free"] and r["error"] is not None:
            e = r["error"] + 1e-9 * max(1.0, abs(r["value"]))
            if np.isfinite(r["upper"]) and r["value"] + e >= r["upper"] and r["name"] != "amount":
                warnings.append(f"{r['label'] or 'scale'} {r['name']} ({r['group']}) at its "
                                "upper bound")
            if np.isfinite(r["lower"]) and r["value"] - e <= r["lower"] and \
                    r["name"] not in ("C_Q", "width_Hz", "amount"):
                warnings.append(f"{r['label'] or 'scale'} {r['name']} ({r['group']}) at its "
                                "lower bound")

    # ---- each experiment's fit, written up like fit_combined's --------------------------
    out = []
    for i, e in enumerate(experiments):
        tick(len(experiments), f"Writing up {e['label']}")
        dicts = pmap.site_dicts(i, values)
        dd = data[i]
        site_curves = curves(i, dicts)
        y_fit = np.sum(site_curves, axis=0) + dd["baseline"]
        names = [(d["label"], name) for d in dicts for name in site_parameter_names(d)]

        def flat(th):
            ds = pmap.site_dicts(i, values_of(th))
            by = {d["label"]: d for d in ds}
            return np.array([_get(by[label], name) for label, name in names])

        # This experiment's parameters as functions of the free keys: their
        # covariance by propagation (a shared value carries its joint error).
        v0 = flat(theta)
        G = np.zeros((len(names), theta.size))
        for j in range(theta.size):
            step = 1e-6 * max(abs(theta[j]), 1.0)
            th = theta.copy()
            th[j] += step
            G[:, j] = (flat(th) - v0) / step
        C = G @ cov @ G.T if theta.size else np.zeros((len(names), len(names)))
        parameters = []
        for (label, name), value, k in zip(names, v0, range(len(names))):
            link = pmap.links[i][label][name]
            scope = pmap.keys[link[1]]["scope"]
            is_free = any(pmap.keys[kk]["free"] for kk in link[1:] if isinstance(kk, tuple))
            parameters.append({"label": label, "name": name, "value": float(value),
                               "error": float(np.sqrt(max(C[k, k], 0.0))) if is_free else None,
                               "free": is_free, "scope": scope,
                               "lower": pmap.keys[link[1]]["lower"],
                               "upper": pmap.keys[link[1]]["upper"]})
        free_names = [f"{label}: {name}" for label, name in names]
        fitted_sites = [Site.from_dict(d) for d in dicts]
        derived = _derived(parameters, C, free_names, fitted_sites, e["larmor_Hz"])
        n_par = sum(1 for p in parameters if p["free"])
        q = fit_quality(dd["y"][dd["mask"]], y_fit[dd["mask"]], n_par)
        tot = float(np.sum((dd["y"][dd["mask"]] - dd["y"][dd["mask"]].mean()) ** 2))
        q["r2"] = (1 - float(np.sum((dd["y"][dd["mask"]] - y_fit[dd["mask"]]) ** 2)) / tot
                   if tot else float("nan"))
        fit = {"sites": fitted_sites, "parameters": parameters, "derived": derived,
               "x": dd["x"], "y": dd["y"], "y_fit": y_fit,
               "site_spectra": {d["label"]: c for d, c in zip(dicts, site_curves)},
               "baseline": dd["baseline"], "mask": dd["mask"], "quality": q, "warnings": [],
               "settings": {"larmor_Hz": e["larmor_Hz"], "mas_Hz": e["mas_Hz"],
                            "order": e["order"], "transitions": e["transitions"],
                            "rotor_angle": e["rotor_angle"], "n_zcw": dd["n_zcw"],
                            "fit_regions": e["fit_regions"], "fit_baseline": e["fit_baseline"]},
               "nuclide": e["nucleus"], "log": ""}
        out.append(_decorate(fit, e))
    result = _finish("joint", out, shared_rows, sequential, warnings, settings, experiments)
    # The joint covariance over the free keys, for its correlation matrix
    # (notebook.fit_correlation.multi_fit_correlation): a shared parameter
    # and every spectrum's own ones in one matrix.
    def _tag(grp):
        if grp.startswith("e") and grp[1:].isdigit():
            return experiments[int(grp[1:])]["label"]
        return "shared" if grp == "all" else f"shared, {grp[2:]}" if grp.startswith("n:") else grp
    result["pcov"] = np.asarray(cov, dtype=float)
    result["free_names"] = [(f"{label}: {name}" if label else name) + f" [{_tag(grp)}]"
                            for label, name, grp in free_keys]
    result["joint_quality"] = {"reduced_chi2": chi2_red, "evaluations": calls[0]}
    result["log"] += (f"\nJoint refinement: {int(free.sum())} free parameters across "
                      f"{len(experiments)} spectra, {calls[0]} evaluations, reduced χ² "
                      f"{chi2_red:.3g} (each spectrum weighted by its own residual noise).")
    tick(n_steps, "Done")
    return result


def _decorate(fit, e):
    fit["label"] = e["label"]
    fit["nucleus"] = e["nucleus"]
    fit["x_value"] = e["x_value"]
    fit["larmor_Hz"] = e["larmor_Hz"]
    fit["mas_Hz"] = e["mas_Hz"]
    fit["mas_fitted"] = e.get("mas_fitted")
    # Computed while the curves are here: a saved result keeps every
    # experiment's numbers but only its own experiment's curves.
    fit["quantities"] = fit_quantities(fit)
    return fit


def _finish(mode, fits, shared_rows, sequential, warnings, settings, experiments):
    lines = [f"Multi-spectrum fit ({mode}): {len(experiments)} spectra, in this order: "
             + ", ".join(e["label"] for e in experiments) + "."]
    for e in experiments:
        lines.append(f"  {e['label']}: {e['nucleus']}, {e['larmor_Hz'] / 1e6:.4f} MHz, "
                     + ("static" if not e["mas_Hz"] else f"MAS {e['mas_Hz'] / 1e3:g} kHz")
                     + (f" (fitted in its own fit, from {e['mas_fitted'][2]:g} Hz"
                        + ("; held in the joint refinement)" if mode == "joint" else ")")
                        if e.get("mas_fitted") else "")
                     + (f", x = {e['x_value']}" if e["x_value"] is not None else ""))
    lines.append(f"Seeded from "
                 f"{'the previous fit' if settings['seed'] == 'previous' else 'the starting model'}"
                 f"; site amounts {settings['amounts']}.")
    return {"mode": mode, "experiments": fits, "shared": shared_rows, "sequential": sequential,
            "warnings": list(dict.fromkeys(warnings)), "log": "\n".join(lines),
            "settings": settings}


# ============================================================
# Tables, report, plots
# ============================================================

def multi_fit_table(result):
    """One row per (experiment, site parameter): experiment, x_value,
    label, name, value, error, scope -- long format, for pandas."""
    rows = []
    for fit in result["experiments"]:
        for p in fit["parameters"]:
            rows.append({"experiment": fit["label"], "x_value": fit["x_value"], "label": p["label"],
                         "name": p["name"], "value": p["value"], "error": p["error"],
                         "scope": p.get("scope", "experiment")})
        for label, d in fit["derived"].items():
            rows.append({"experiment": fit["label"], "x_value": fit["x_value"], "label": label,
                         "name": "share_%", "value": d["share_%"], "error": d["share_%_err"],
                         "scope": "derived"})
    return rows


def parameter_trend(result, label, name):
    """``(x_values, values, errors, experiment labels)`` of one site
    parameter along the experiments, in fitting order. ``name`` may be
    ``"share_%"``."""
    xs, vs, es, labs = [], [], [], []
    for row in multi_fit_table(result):
        if row["label"] == label and row["name"] == name:
            xs.append(row["x_value"])
            vs.append(row["value"])
            es.append(row["error"] if row["error"] is not None else np.nan)
            labs.append(row["experiment"])
    return xs, np.array(vs, dtype=float), np.array(es, dtype=float), labs


# ============================================================
# Fitted quantities along a series (explicit user request, 2026-10-06:
# "plot any fitted quantities -- linewidth, integral, intensity, shift,
# centre-of-mass, shift anisotropy -- as a function of another parameter
# associated with the data, e.g. temperature or time")
# ============================================================

#: The whole fit window, model-free: what a series' centre of mass or
#: total integral is usually wanted for (a VT shift, a growing signal).
WINDOW = "All (fit window)"

#: (key, title, unit), in the order offered. A peak has the ones its
#: interactions give it; J1, J2, ... follow.
QUANTITIES = (
    ("delta_iso", "Shift δiso", "ppm"),
    ("centre_of_mass", "Centre of mass", "ppm"),
    ("fwhm", "Linewidth (FWHM)", "ppm"),
    ("fwhm_Hz", "Linewidth (FWHM)", "Hz"),
    ("integral", "Integral", ""),
    ("height", "Intensity (height)", ""),
    ("amount", "Amount (model)", ""),
    ("share_%", "Share", "%"),
    ("aniso", "Anisotropy (Δ or span)", "ppm"),
    ("asym", "Asymmetry (η or skew)", ""),
    ("C_Q", "C_Q", "MHz"),
    ("eta_Q", "η_Q", ""),
    ("P_Q", "P_Q", "MHz"),
    ("delta_cg", "CT centre of gravity", "ppm"),
    ("eta", "Lorentzian fraction", ""),
    ("mas_Hz", "MAS rate (fitted)", "Hz"),
)
_QUANTITY_DEFINITIONS = {
    "centre_of_mass": "Σδ·I / ΣI of the peak's fitted curve (of the data minus baseline, for "
                      "the whole window), over the fit axis",
    "integral": "area under the peak's fitted curve (the data minus baseline, for the whole "
                "window) over the fit axis, in intensity × ppm; sidebands outside the axis "
                "are not counted",
    "height": "the curve's largest |value| (signed)",
    "amount": "the model's amount: the peak's total intensity, every transition and sideband",
    "fwhm_Hz": "FWHM (ppm) × ν0 (MHz)",
}


def quantity_title(key, with_unit=True):
    """'Linewidth (FWHM) (Hz)' for 'fwhm_Hz'; J1 -> 'J1 (Hz)'."""
    for k, title, unit in QUANTITIES:
        if k == key:
            return f"{title} ({unit})" if unit and with_unit else title
    if key.startswith("J"):
        return f"{key} (Hz)" if with_unit else key
    return key


def fit_quantities(fit):
    """Every quantity of one experiment's fit: ``{peak label or WINDOW:
    {key: (value, error)}}`` (error ``nan`` where none is defined).

    Parameters come with their fitted 1σ; integral and height carry the
    amount's relative error; the centre of mass has none. Computed from
    the fitted curves, so it needs ``x``, ``site_spectra``, ``baseline``
    and ``mask`` (a fit as :func:`fit_combined` or :func:`fit_multi`
    returns it). The window entry uses the data, not the model."""
    out = {}
    nan = float("nan")
    params = {}
    for p in fit.get("parameters") or []:
        if p.get("label"):
            err = p.get("error")
            params.setdefault(p["label"], {})[p["name"]] = (
                float(p["value"]), float(err) if err is not None else nan)
    larmor_MHz = float(fit.get("larmor_Hz") or (fit.get("settings") or {}).get("larmor_Hz")
                       or 0.0) / 1e6
    x = fit.get("x")
    curves = fit.get("site_spectra") or {}
    for label, values in params.items():
        q = dict(values)
        if "fwhm" in q and larmor_MHz:
            v, e = q["fwhm"]
            q["fwhm_Hz"] = (v * larmor_MHz, e * larmor_MHz)
        for key in ("share_%", "P_Q", "delta_cg"):
            d = (fit.get("derived") or {}).get(label) or {}
            if key in d:
                err = d.get(key + "_err")
                q[key] = (float(d[key]), float(err) if err is not None else nan)
        if x is not None and label in curves:
            xs = np.asarray(x, dtype=float)
            c = np.asarray(curves[label], dtype=float)
            amount, amount_err = q.get("amount", (nan, nan))
            rel = abs(amount_err / amount) if amount and np.isfinite(amount_err) else nan
            area = float(np.trapezoid(c, xs)) if hasattr(np, "trapezoid") else float(np.trapz(c, xs))
            peak = float(c[np.argmax(np.abs(c))]) if c.size else nan
            total = float(np.sum(c))
            q["integral"] = (area, abs(area) * rel)
            q["height"] = (peak, abs(peak) * rel)
            q["centre_of_mass"] = (float(np.sum(xs * c) / total) if total else nan, nan)
        out[label] = q
    if x is not None and fit.get("y") is not None:
        xs = np.asarray(x, dtype=float)
        y = np.asarray(fit["y"], dtype=float) - np.asarray(
            fit.get("baseline") if fit.get("baseline") is not None else 0.0, dtype=float)
        mask = np.asarray(fit["mask"], dtype=bool) if fit.get("mask") is not None \
            else np.ones(xs.size, dtype=bool)
        xm, ym = xs[mask], y[mask]
        if xm.size:
            area = float(np.trapezoid(ym, xm)) if hasattr(np, "trapezoid") else float(np.trapz(ym, xm))
            total = float(np.sum(ym))
            out[WINDOW] = {"integral": (area, nan),
                           "height": (float(ym[np.argmax(np.abs(ym))]), nan),
                           "centre_of_mass": (float(np.sum(xm * ym) / total) if total else nan,
                                              nan)}
    for p in fit.get("parameters") or []:
        if p.get("name") == "mas_Hz" and not p.get("label"):
            err = p.get("error")
            out.setdefault(WINDOW, {})["mas_Hz"] = (float(p["value"]),
                                                    float(err) if err is not None else nan)
    if fit.get("mas_fitted") and "mas_Hz" not in out.get(WINDOW, {}):
        value, err, _start = fit["mas_fitted"]
        out.setdefault(WINDOW, {})["mas_Hz"] = (float(value),
                                                float(err) if err is not None else nan)
    return out


def _quantities_of(fit):
    return fit.get("quantities") or fit_quantities(fit)


def available_quantities(result):
    """``(peak labels, quantity keys)`` present in at least one experiment
    of a multi-spectrum result, in a stable order: peaks as fitted, then
    the window; quantities as :data:`QUANTITIES`, then J1, J2, ..."""
    labels, keys = [], set()
    for fit in result["experiments"]:
        for label, q in _quantities_of(fit).items():
            if label not in labels:
                labels.append(label)
            keys |= set(q)
    labels = [l for l in labels if l != WINDOW] + ([WINDOW] if WINDOW in labels else [])
    order = [k for k, _t, _u in QUANTITIES if k in keys]
    order += sorted(k for k in keys if k.startswith("J") and k[1:].isdigit())
    return labels, order


def quantity_trend(result, label, key, x_values=None):
    """``(x, values, errors, experiment labels)`` of one quantity of one
    peak (or :data:`WINDOW`) along the experiments, in fitting order.
    ``x_values`` (one per experiment) replaces the fit's own x values --
    e.g. temperatures typed after the fit. Experiments where the peak or
    quantity is absent are left out."""
    xs, vs, es, labs = [], [], [], []
    for j, fit in enumerate(result["experiments"]):
        q = _quantities_of(fit).get(label, {})
        if key not in q:
            continue
        x = x_values[j] if x_values is not None else fit.get("x_value")
        xs.append(x)
        vs.append(q[key][0])
        es.append(q[key][1])
        labs.append(fit["label"])
    return xs, np.array(vs, dtype=float), np.array(es, dtype=float), labs


#: x and y transforms offered for a quantity plot: key -> (label format, function).
X_TRANSFORMS = {"x": ("{}", lambda v: v), "inverse": ("1/{}", lambda v: 1.0 / v),
                "1000_over": ("1000/{}", lambda v: 1000.0 / v),
                "log10": ("log10({})", np.log10)}
Y_TRANSFORMS = ("as is", "normalised to first", "normalised to max", "ln", "log10")


def transform_series(x, y, err, x_transform="x", y_transform="as is"):
    """Apply a plot's x and y transforms to one series, carrying the
    errors through (``ln``: err/|y|; ``log10``: err/(|y| ln 10); a
    normalisation divides them too). Points whose transform is undefined
    (1/0, log of <= 0) are dropped. Returns (x, y, err) as arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = X_TRANSFORMS[x_transform][1](x)
        if y_transform == "normalised to first":
            ref = y[0] if y.size else np.nan
            y, err = y / ref, err / abs(ref)
        elif y_transform == "normalised to max":
            ref = y[np.argmax(np.abs(y))] if y.size else np.nan
            y, err = y / ref, err / abs(ref)
        elif y_transform == "ln":
            y, err = np.log(y), err / np.abs(y)
        elif y_transform == "log10":
            y, err = np.log10(y), err / (np.abs(y) * np.log(10.0))
        elif y_transform != "as is":
            raise ValueError(f"y_transform must be one of {Y_TRANSFORMS}.")
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep], err[keep]


def plot_quantity_trend(result, key, labels=None, x_values=None, xlabel="x",
                        x_transform="x", y_transform="as is", ax=None, colours=None,
                        fontsize=None):
    """One fitted quantity against x for one or more peaks (default every
    peak; :data:`WINDOW` for the whole window), with 1σ error bars,
    points joined in x order. ``x_values`` as :func:`quantity_trend`.
    Returns the Axes.

        plot_quantity_trend(result, "fwhm_Hz", x_values=temperatures, xlabel="T (K)",
                            x_transform="1000_over")
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _fig, ax = plt.subplots()
    peaks, _keys = available_quantities(result)
    labels = list(labels) if labels else [p for p in peaks if p != WINDOW]
    if colours is None:
        from ..plotting.accessibility import ordered_qualitative_colours
        colours = ordered_qualitative_colours("okabe_ito")
    drawn = 0
    for k, label in enumerate(labels):
        xs, vs, es, _labs = quantity_trend(result, label, key, x_values=x_values)
        if not xs or any(x is None for x in xs):
            continue
        x, y, e = transform_series(np.asarray(xs, dtype=float), vs, es, x_transform, y_transform)
        order = np.argsort(x)
        ax.errorbar(x[order], y[order], yerr=np.where(np.isfinite(e), e, 0.0)[order], fmt="o-",
                    capsize=3, markersize=5, linewidth=1.2, label=label,
                    color=colours[k % len(colours)])
        drawn += 1
    if not drawn:
        raise ValueError("Nothing to plot: every experiment needs a numeric x value, and the "
                         "chosen peaks must have this quantity.")
    ax.set_xlabel(X_TRANSFORMS[x_transform][0].format(xlabel), fontsize=fontsize)
    ylabel = quantity_title(key)
    if y_transform != "as is":
        ylabel = {"ln": f"ln[{ylabel}]", "log10": f"log10[{ylabel}]"}.get(
            y_transform, f"{quantity_title(key, with_unit=False)} ({y_transform})")
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if drawn > 1:
        ax.legend(frameon=False)
    return ax


def quantity_table(result, key, labels, x_values=None, xlabel="x"):
    """CSV text: one row per experiment -- its label, x, and each peak's
    value and error of ``key``."""
    import io as _io
    buf = _io.StringIO()
    writer = csv.writer(buf)
    title = quantity_title(key)
    writer.writerow(["experiment", xlabel] + [h for l in labels
                                             for h in (f"{l}: {title}", f"{l}: error")])
    for j, fit in enumerate(result["experiments"]):
        q = _quantities_of(fit)
        x = x_values[j] if x_values is not None else fit.get("x_value")
        row = [fit["label"], "" if x is None else x]
        for l in labels:
            v = q.get(l, {}).get(key)
            row += ["", ""] if v is None else [f"{v[0]:.10g}",
                                               "" if not np.isfinite(v[1]) else f"{v[1]:.4g}"]
        writer.writerow(row)
    return buf.getvalue()


def multi_fit_to_csv(result, path=None):
    """Wide CSV: one row per experiment (in order), a value and an error
    column per site parameter. Returns the text; writes ``path`` if given."""
    rows = multi_fit_table(result)
    columns = list(dict.fromkeys(f"{r['label']}: {r['name']}" for r in rows))
    by_exp = {}
    for r in rows:
        by_exp.setdefault(r["experiment"], {"x_value": r["x_value"]})[f"{r['label']}: {r['name']}"] = r
    buffer = _io.StringIO()
    writer = csv.writer(buffer)
    header = ["experiment", "x_value"]
    for c in columns:
        header += [c, f"{c} error"]
    writer.writerow(header)
    for fit in result["experiments"]:
        got = by_exp.get(fit["label"], {})
        line = [fit["label"], "" if fit["x_value"] is None else fit["x_value"]]
        for c in columns:
            r = got.get(c)
            line += ["" if r is None else repr(float(r["value"])),
                     "" if r is None or r["error"] is None else repr(float(r["error"]))]
        writer.writerow(line)
    text = buffer.getvalue()
    if path:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    return text


_UNITS = {"delta_iso": "ppm", "fwhm": "ppm", "aniso": "ppm", "C_Q": "MHz", "alpha": "deg",
          "beta": "deg", "gamma": "deg", "delta_offset": "ppm", "width_Hz": "Hz",
          "width_ppm": "ppm"}


def _pname(name):
    unit = _UNITS.get(name, "Hz" if name.startswith("J") else "")
    return f"{name} ({unit})" if unit else name


def print_multi_fit_report(result, save_txt=None, advice=True):
    """The multi-spectrum fit written out: set-up, shared parameters,
    each experiment's parameters and quality, warnings, and (``advice``)
    what the fits tell you and what to try next, in plain words
    (:mod:`opennmr.notebook.fit_advice`). Returns the text."""
    lines = ["=" * 72, "MULTI-SPECTRUM FIT REPORT", "=" * 72, result["log"], ""]
    if result["shared"]:
        lines.append("Shared parameters")
        for r in result["shared"]:
            where = {"all": "all spectra"}.get(
                r["group"], r["group"].replace("n:", "every ") if r["group"].startswith("n:")
                else r["group"])
            name = f"{r['label']}: {_pname(r['name'])}" if r["label"] else _pname(r["name"])
            lines.append(f"  {name:<30}= "
                         f"{_fmt(r['value'], r['error'] if r['free'] else None):<24} [{where}]")
        lines.append("")
    for fit in result["experiments"]:
        q = fit["quality"]
        lines.append(f"{fit['label']}  ({fit['nucleus']}"
                     + (f", x = {fit['x_value']}" if fit["x_value"] is not None else "")
                     + f")   R² {q['r2']:.5f}")
        for p in fit["parameters"]:
            tag = "" if p.get("scope", "experiment") == "experiment" else f" [{p['scope']}]"
            lines.append(f"  {p['label'] + ': ' + _pname(p['name']):<30}= "
                         f"{_fmt(p['value'], p['error'] if p['free'] else None)}{tag}")
        for label, d in fit["derived"].items():
            lines.append(f"  {label + ': share (%)':<30}= "
                         f"{_fmt(d['share_%'], d['share_%_err'] or None)}")
        lines.append("")
    if advice:
        from .fit_advice import advice_text, advise_multi_fit
        items = advise_multi_fit(result)
        if items:
            lines += [advice_text(items), ""]
    if result["warnings"]:
        lines += ["Warnings:"] + [f"  - {w}" for w in result["warnings"]]
    lines.append("=" * 72)
    text = "\n".join(lines)
    print(text)
    if save_txt:
        with open(save_txt, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def plot_multi_fit(result, figure=None, show_sites=True, xlims="window"):
    """Every experiment's data and fit, one panel each, top to bottom in
    fitting order. Returns the figure.

    ``xlims`` : ``"window"`` (default) -- each panel shows its fit
    window (plus 5 % either side), not the whole spectrum, so a narrow
    line on a wide spectral width is not drawn as a spike; ``None`` --
    the whole axis; or ``(left, right)`` in ppm for every panel."""
    import matplotlib.pyplot as plt
    fits = result["experiments"]
    if figure is None:
        figure = plt.figure(figsize=(7, 2.2 * max(len(fits), 1)))
    figure.clear()
    axes = figure.subplots(len(fits), 1, squeeze=False)[:, 0]
    for ax, fit in zip(axes, fits):
        x, m = np.asarray(fit["x"], dtype=float), np.asarray(fit["mask"], dtype=bool)
        if isinstance(xlims, str) and xlims == "window" and m.any():
            lo, hi = float(x[m].min()), float(x[m].max())
            pad = 0.05 * (hi - lo)
            left, right = hi + pad, lo - pad
        elif xlims is None or isinstance(xlims, str):
            left, right = float(x.max()), float(x.min())
        else:
            left, right = max(xlims), min(xlims)
        shown = (x >= right) & (x <= left)
        ax.plot(x[shown], fit["y"][shown], color="black", lw=0.8, label="data")
        if show_sites and len(fit["site_spectra"]) > 1:
            for label, c in fit["site_spectra"].items():
                ax.fill_between(x[shown], (c + fit["baseline"])[shown], fit["baseline"][shown],
                                alpha=0.35, label=label)
        ax.plot(x[m & shown], fit["y_fit"][m & shown], "--", color="red", lw=1.2, label="fit")
        ax.set_xlim(left, right)
        ax.set_yticks([])
        ax.set_title(f"{fit['label']} ({fit['nucleus']})", fontsize=9, loc="left")
    try:
        from .utils import NUC_label
        axes[-1].set_xlabel(NUC_label(fits[-1]["nucleus"]))
    except Exception:  # noqa: BLE001 -- a label must not sink a plot
        axes[-1].set_xlabel("δ (ppm)")
    axes[0].legend(fontsize=7, loc="upper right")
    figure.tight_layout()
    return figure


def plot_parameter_trend(result, label, name, ax=None, xlabel="x"):
    """One parameter along the experiments, with 1σ error bars (against
    ``x_value`` when every experiment has one, else in fitting order)."""
    import matplotlib.pyplot as plt
    xs, vs, es, labs = parameter_trend(result, label, name)
    if ax is None:
        _fig, ax = plt.subplots()
    if all(x is not None for x in xs):
        ax.errorbar(xs, vs, yerr=np.nan_to_num(es), fmt="o-", color="black", capsize=3)
        ax.set_xlabel(xlabel)
    else:
        ax.errorbar(range(len(vs)), vs, yerr=np.nan_to_num(es), fmt="o-", color="black", capsize=3)
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_ylabel(f"{label}: {_pname(name)}")
    return ax
