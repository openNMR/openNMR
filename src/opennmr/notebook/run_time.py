"""
How long a Solid-state NMR simulation or fit will take -- worked out before
it starts, so a run of minutes is not a surprise (explicit user request,
2026-11-01; measured then: an exact-order 27Al MAS site took ~1 s per
1,000 orientations, and the Orientations box goes to 500,000).

Simulations
-----------
:func:`estimate_simulation_time` runs the *same* simulation twice at small
orientation counts and fits ``time = a + b * orientations``: ``a`` is
everything that does not scale with the powder average (the axis, the
broadening, the shared-pattern bookkeeping), ``b`` the cost of one
orientation with this spin, level of theory, MAS sampling and set of
interactions. Measured run times are close to linear in the orientation
count (2,000 / 8,000 / 32,000 orientations: 3.5 / 7.9 / 29 s), so the
extrapolation is good to tens of percent -- what the question "seconds or
minutes?" needs. The trial runs grow until one takes ~0.25 s, so they cost
about half a second when the full run is long, and much less when it is not.

Fits
----
A least-squares fit evaluates the model once per iteration plus once per
free parameter (the numerical Jacobian). In :func:`fit_combined` the
unbroadened powder pattern of a site is cached, so changing ``amount``,
``fwhm`` or ``eta`` only re-broadens, while a shape parameter (δiso, Δ, η,
C_Q, ...) re-simulates its own site. :func:`estimate_fit_time` therefore
costs an iteration as ``a`` per evaluation plus one site's share of ``b``
per free shape parameter, and gives a *range*: 4 to 25 iterations, typical
of these fits (trust-region reflective, 2-point Jacobian) -- the number a
fit needs cannot be known before it runs. The MAS-rate scan and multiple
starts are added when used.

Both work in a notebook::

    from opennmr.notebook.run_time import estimate_simulation_time
    est = estimate_simulation_time(sites, ppm_axis, B0_T=14.1, mas_Hz=12e3, n_zcw=20000)
    print(est["text"])          # 'about 40 s (…)'
"""

from __future__ import annotations

import time

import numpy as np

from .combined_hamiltonian import (as_sites, simulate_combined, _check_scheme, _powder,
                                   default_powder_scheme, resolve_links)
from .combined_fitting import site_parameter_names, _per_site, _BROADENING

#: Ask before a simulation estimated to take longer than this (seconds).
SIMULATION_WARN_S = 10.0
#: Ask before a fit whose slower estimate is longer than this (seconds).
FIT_WARN_S = 60.0
#: Iterations of a typical fit: the low and high ends of the estimate.
FIT_ITERATIONS = (4, 25)
#: Orientation density of the first trial run, and the least for the last.
PROBE_N = (89, 377)
#: A trial run is grown (x4 in density) until it takes this long, s.
PROBE_MIN_S = 0.25
#: An upper bound from one trial run must be this many times under the
#: threshold to stop there (run times grow a little faster than linearly).
BOUND_MARGIN = 2.0


def format_duration(seconds: float) -> str:
    """'0.4 s', '12 s', '3 min 20 s', '1 h 05 min'."""
    s = float(seconds)
    if s < 1:
        return f"{s:.1f} s"
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        m, r = divmod(int(round(s)), 60)
        return f"{m} min {r:02d} s"
    h, r = divmod(int(round(s)), 3600)
    return f"{h} h {r // 60:02d} min"


def _orientations(sites, n_zcw, mas_Hz, scheme, use_symmetry=True):
    """Orientations each site will be averaged over (as simulate_combined
    reports them), without simulating."""
    out = {}
    for s in sites:
        sch = default_powder_scheme(mas_Hz, s) if scheme == "auto" else scheme
        theta, _p, _w, _o = _powder(s, n_zcw, use_symmetry, static=not mas_Hz, scheme=sch)
        out[s.label] = int(theta.size)
    return out


def _timed(fn):
    t = time.perf_counter()
    result = fn()
    return time.perf_counter() - t, result


def _probe(sites, ppm_axis, kwargs, n_full, total_full=None, stop_below=None):
    """(a, b, points): seconds = a + b * total orientations, from trial runs.

    The first run is small (:data:`PROBE_N` [0]); the density is then
    multiplied by 4 until a run takes at least :data:`PROBE_MIN_S` or
    reaches a quarter of the full density, and the line goes through the
    last two runs. Extrapolating from two tiny runs, where the fixed cost
    dominates, under-estimated large exact-order MAS runs by ~30%
    (measured 2026-11-01)."""
    def run(n, repeats=1):
        best, result = None, None
        for _ in range(repeats):    # the first run may pay for imports: take the faster
            t, result = _timed(lambda: simulate_combined(sites, ppm_axis,
                                                         **{**kwargs, "n_zcw": n}))
            best = t if best is None else min(best, t)
        return sum(v["orientations"] for v in result["powder"].values()), best

    n = PROBE_N[0]
    points = [run(n, repeats=2)]
    if stop_below is not None and total_full is not None:
        # The first run's time per orientation includes the fixed cost, so it
        # over-states b: an upper bound. Clearly short? Stop here.
        n1, t1 = points[0]
        bound = t1 / max(n1, 1) * total_full
        if bound * BOUND_MARGIN < stop_below:
            return 0.0, t1 / max(n1, 1), points
    limit = max(int(n_full) // 4, PROBE_N[1])
    while True:
        # Aim the next run at ~PROBE_MIN_S from what has been seen so far,
        # at most 4x the density and never past a quarter of the full run.
        (n_last, t_last) = points[-1]
        per_n = t_last / max(n_last, 1)
        aim = int(n * PROBE_MIN_S * 1.2 / max(per_n * n_last, 1e-6)) if per_n else n * 4
        n = max(min(n * 4, limit, max(aim, n + 1)), n + 1)
        points.append(run(n))
        if points[-1][1] >= PROBE_MIN_S or n >= limit:
            break
    (n1, t1), (n2, t2) = points[-2], points[-1]
    b = max((t2 - t1) / (n2 - n1), 0.0) if n2 > n1 else t2 / max(n2, 1)
    a = max(t1 - b * n1, 0.0)
    return a, b, points


def estimate_simulation_time(sites, ppm_axis, larmor_Hz=None, B0_T=None, mas_Hz=0.0,
                             n_zcw=None, powder_scheme=None, stop_below=None, **kwargs) -> dict:
    """
    Seconds :func:`simulate_combined` will take with these arguments.

    Returns a dict: ``seconds``, ``orientations`` ({label: count} the run
    will use), ``total_orientations``, ``per_orientation_s`` (b),
    ``fixed_s`` (a), ``probe_s`` (what the trial runs cost), ``bound``
    and ``text``.

    ``stop_below`` (seconds): when one small trial run already shows the
    simulation is well under this, stop there and return that upper bound
    (``bound`` True) -- how the GUI checks a quick simulation for ~20 ms.
    """
    sites = as_sites(sites)
    mas_Hz = float(mas_Hz or 0.0)
    if n_zcw is None:
        n_zcw = 4181 if mas_Hz else 75025
    scheme = _check_scheme(powder_scheme, mas_Hz)
    resolved = resolve_links(sites)
    counts = _orientations(resolved, int(n_zcw), mas_Hz, scheme,
                           kwargs.get("use_symmetry", True))
    total = sum(counts.values())
    sim_kwargs = dict(kwargs, larmor_Hz=larmor_Hz, B0_T=B0_T, mas_Hz=mas_Hz,
                      powder_scheme=powder_scheme)
    t0 = time.perf_counter()
    a, b, points = _probe(sites, ppm_axis, sim_kwargs, int(n_zcw), total, stop_below)
    bound = len(points) == 1
    probe_s = time.perf_counter() - t0
    seconds = a + b * total
    text = (("at most " if bound else "about ") + format_duration(seconds)
            + f" ({total:,} orientations over "
            f"{len(counts)} site{'s' if len(counts) != 1 else ''}; "
            + (f"at most, from a trial run of {points[0][0]:,})" if bound else
               f"estimated from trial runs of {points[-2][0]:,} and {points[-1][0]:,})"))
    return {"seconds": float(seconds), "orientations": counts, "total_orientations": total,
            "per_orientation_s": float(b), "fixed_s": float(a), "probe_s": float(probe_s),
            "bound": bound, "text": text}


def free_shape_parameters(sites, vary=None, fixed=None) -> dict:
    """{site label: number of free parameters that re-simulate that site}
    -- as :func:`fit_combined` frees them (``vary``/``fixed`` the same
    forms), not counting amount, FWHM and η (broadening only)."""
    sites = as_sites(sites)
    labels = [s.label for s in sites]
    vary_m, fixed_m = _per_site(vary, labels, "vary"), _per_site(fixed, labels, "fixed")
    out = {}
    for s in sites:
        d = s.to_dict()
        names = site_parameter_names(d)
        if d.get("link"):
            names = [n for n in names if n in (d["link"].get("free") or [])]
        free = (set(vary_m[s.label]) if vary_m[s.label] is not None
                else set(names) - {"alpha", "beta", "gamma"})
        free -= set(fixed_m[s.label] or [])
        out[s.label] = len([n for n in free if n in names and n not in _BROADENING])
    return out


def estimate_fit_time(results, sites, vary=None, fixed=None, larmor_Hz=None, mas_Hz=0.0,
                      n_zcw=None, powder_scheme=None, fit_mas=False, mas_scan_points=41,
                      starts=1, screen_n_zcw=None, fit_baseline=False, stop_below=None,
                      **kwargs) -> dict:
    """
    A range of seconds for :func:`fit_combined` (or, with ``starts`` > 1,
    :func:`fit_combined_multistart`) with these arguments.

    ``results``/``sites``/``vary``/``fixed``/... as fit_combined; extra
    simulation keywords (order, transitions, rotor_angle, observe) pass
    through. Returns ``low``, ``high`` (seconds), ``simulation``
    (:func:`estimate_simulation_time` at the fit's orientations),
    ``free_shape`` ({label: count}), ``evaluations`` ((low, high)),
    ``bound`` and ``text``. ``stop_below``: as estimate_simulation_time,
    for the upper end of the fit.
    """
    sites = as_sites(sites)
    mas_Hz = float(mas_Hz or 0.0)
    if n_zcw is None:
        n_zcw = 2584 if mas_Hz else 28657
    if larmor_Hz is None:
        larmor_Hz = float(results["SFO1"]) * 1e6
    axis = np.sort(np.asarray(results["x_ppm"], dtype=float))
    sim_kw = {k: v for k, v in kwargs.items()
              if k in ("order", "transitions", "rotor_angle", "observe")}
    shape = free_shape_parameters(sites, vary, fixed)
    n_broad = sum(1 for s in sites for n in site_parameter_names(s) if n in _BROADENING)
    n_free = sum(shape.values()) + n_broad + (2 if fit_baseline else 0) + (1 if fit_mas else 0)
    # A fit costs at most ~ (n_free + 1) simulations per iteration (twice
    # that with a free MAS rate, plus the scan) and each start repeats it.
    worst = FIT_ITERATIONS[1] * (n_free + 2) * (2 if fit_mas else 1) * max(int(starts or 1), 1)
    sim = estimate_simulation_time(
        sites, axis, larmor_Hz=larmor_Hz, mas_Hz=mas_Hz, n_zcw=n_zcw,
        powder_scheme=powder_scheme,
        stop_below=(stop_below / (worst + (mas_scan_points if fit_mas else 0))
                    if stop_below is not None else None), **sim_kw)
    a, b = sim["fixed_s"], sim["per_orientation_s"]
    counts = sim["orientations"]
    # One iteration: the model at the current point, then one evaluation per
    # free parameter. Each re-broadens everything (a); a shape parameter also
    # re-simulates its own site (b * its orientations); a free MAS rate
    # re-simulates every site.
    per_iteration = (n_free + 1) * a + b * sum(shape[lab] * counts.get(lab, 0) for lab in shape)
    if fit_mas:
        per_iteration += b * sum(counts.values())
    first = sim["seconds"]
    low = first + FIT_ITERATIONS[0] * per_iteration
    high = first + FIT_ITERATIONS[1] * per_iteration
    extra = []
    if fit_mas and int(mas_scan_points) > 1:
        coarse = sum(_orientations(resolve_links(sites), min(int(n_zcw), 233), mas_Hz,
                                   _check_scheme(powder_scheme, mas_Hz)).values())
        scan = int(mas_scan_points) * (a + b * coarse)
        low, high = low + scan, high * 2 + scan        # two passes when the rate is free
        extra.append(f"a {int(mas_scan_points)}-point scan of the MAS rate")
    starts = max(int(starts or 1), 1)
    if starts > 1:
        if screen_n_zcw:
            ratio = min(float(screen_n_zcw) / float(n_zcw), 1.0)
            screened = (a * (n_free + 1) * FIT_ITERATIONS[0] + ratio * (low - a), ratio * high)
            low, high = starts * screened[0] + low, starts * screened[1] + high
            extra.append(f"{starts} starts screened at {int(screen_n_zcw):,} orientations, "
                         "then the best refined")
        else:
            low, high = starts * low, starts * high
            extra.append(f"{starts} starts")
    evaluations = (FIT_ITERATIONS[0] * (n_free + 1), FIT_ITERATIONS[1] * (n_free + 1))
    text = (f"about {format_duration(low)} to {format_duration(high)} "
            f"({n_free} free parameter{'s' if n_free != 1 else ''}, "
            f"{sum(shape.values())} of them re-simulating a site; one simulation "
            f"{format_duration(first)} at {sum(counts.values()):,} orientations"
            + "".join(f"; {e}" for e in extra) + ")")
    return {"low": float(low), "high": float(high), "simulation": sim, "free_shape": shape,
            "n_free": n_free, "evaluations": evaluations, "bound": sim["bound"], "text": text}
