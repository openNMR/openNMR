"""PFG diffusion attenuation, for planning an experiment.

Ported from ``leonmr.sim_diffusion`` (L. Gordon): the expected signal
attenuation ``I/I0 = exp(-D b)`` against gradient strength, for one
diffusion coefficient or a sweep of them, so the gradient strength and
timings can be chosen before running anything.

The original used the rectangular-gradient spin-echo b-value only,
``(2 pi gamma g delta)^2 (DELTA - delta/3)``. Here the b-value comes
from :func:`opennmr.processing.diffusion.b_value`, so the same plot can
be drawn for every gradient shape and sequence type that the fitting
side already models -- and :func:`compare_attenuation` draws them side
by side, which is the point: at the same nominal gradient and timings,
a half-sine gradient or a bipolar pair attenuates noticeably less than
the textbook rectangle. With ``shape="rect"`` and ``sequence="pgse"``
the result is exactly the original's.
"""

from __future__ import annotations

import numpy as np

from ..processing import diffusion as _diffusion

#: The original's default sweep when no D is given: 1e-8 .. 1e-15 m^2/s,
#: one curve per decade.
DEFAULT_D_SWEEP = tuple(np.logspace(-8, -15, 8))


def gradient_axis(max_gradient: float) -> np.ndarray:
    """The original's gradient axis: 0 to ``max_gradient`` in 100 steps
    (``np.arange(0, g + g/100, g/99)``), in T/m."""
    return np.arange(0, max_gradient + (max_gradient / 100.0), max_gradient / 99.0)


def pfg_attenuation(nucleus: str, delta_ms: float = 1.0, DELTA_ms: float = 20.0,
                    max_gradient: float = 17.0, D=None, shape: str = "rect",
                    sequence: str = "pgse", tau_ms: float = 0.0, ramp_ms: float = 0.0) -> dict:
    """Attenuation against gradient strength for one experiment.

    Defaults are the original's (delta 1 ms, DELTA 20 ms, 17 T/m -- a
    Bruker diffBB probe). ``D`` is a value or a sequence of values in
    m^2/s; ``None`` sweeps 1e-8 .. 1e-15 as the original did.

    Returns a dict: ``gradient`` (T/m), ``b`` (s/m^2), ``D`` (array),
    ``intensity`` (``(len(D), len(gradient))``), and the inputs.
    """
    D_values = np.atleast_1d(np.asarray(DEFAULT_D_SWEEP if D is None else D, dtype=float))
    gradient = gradient_axis(max_gradient)
    b = _diffusion.b_value(
        gradient, delta_ms / 1000.0, DELTA_ms / 1000.0, nucleus=nucleus,
        shape=shape, sequence=sequence, tau=tau_ms / 1000.0, ramp=ramp_ms / 1000.0,
    )
    intensity = np.exp(-np.outer(D_values, b))
    return {
        "gradient": gradient, "b": b, "D": D_values, "intensity": intensity,
        "nucleus": nucleus, "delta_ms": delta_ms, "DELTA_ms": DELTA_ms,
        "shape": shape, "sequence": sequence, "tau_ms": tau_ms, "ramp_ms": ramp_ms,
    }


def compare_attenuation(nucleus: str, D: float, vary: str, delta_ms: float = 1.0,
                        DELTA_ms: float = 20.0, max_gradient: float = 17.0,
                        shape: str = "rect", sequence: str = "pgse",
                        tau_ms: float = 0.0, ramp_ms: float = 0.0) -> list[dict]:
    """One :func:`pfg_attenuation` curve per gradient shape
    (``vary="shape"``) or per sequence type (``vary="sequence"``), at a
    single ``D``, with everything else held fixed.

    Each entry gains ``label``, ``key`` and ``error``. An option the
    timings cannot support (e.g. a bipolar recovery delay that leaves
    no diffusion time) is kept in the list with ``intensity=None`` and
    the reason in ``error``, rather than raising -- so a caller can
    still say which options were left out and why.
    """
    if vary == "shape":
        options = [(key, _diffusion.GRADIENT_SHAPE_LABELS[key]) for key in _diffusion.GRADIENT_SHAPE_LABELS]
    elif vary == "sequence":
        options = [(key, spec["label"]) for key, spec in _diffusion.DIFFUSION_SEQUENCES.items()]
    else:
        raise ValueError(f"vary must be 'shape' or 'sequence', got {vary!r}")

    curves = []
    for key, label in options:
        kwargs = dict(delta_ms=delta_ms, DELTA_ms=DELTA_ms, max_gradient=max_gradient, D=D,
                      shape=shape, sequence=sequence, tau_ms=tau_ms, ramp_ms=ramp_ms)
        kwargs[vary] = key
        try:
            curve = pfg_attenuation(nucleus, **kwargs)
        except ValueError as exc:
            curves.append({"label": label, "key": key, "intensity": None, "error": str(exc)})
            continue
        curve.update(label=label, key=key, error=None)
        curves.append(curve)
    return curves


def plot_attenuation(curves: list[dict], ax, labels: list[str] | None = None) -> None:
    """Draw attenuation curves onto ``ax``: the original's axes (gradient
    strength against I/I0, x running to 1.25x the maximum so the legend
    has room), one line per D for a single experiment, or one line per
    entry of a :func:`compare_attenuation` list."""
    import matplotlib.cm as cm

    lines = []
    for curve in curves:
        if curve.get("intensity") is None:
            continue
        for row, D in enumerate(curve["D"]):
            lines.append((curve, row, D))
    # Not the original's ``seismic``: it is white in the middle, so the
    # middle curves of a sweep (or of a five-way comparison) vanished
    # against the background. Distinct palette colours instead, and
    # alternating line styles so curves that coincide -- a trapezoid
    # with a short ramp is almost the rectangle -- both stay visible.
    if len(lines) <= 10:
        colours = [cm.tab10(i) for i in range(len(lines))]
    else:
        colours = cm.viridis(np.linspace(0, 0.9, len(lines)))
    styles = ["-", "--", "-.", ":"]
    for (curve, row, D), colour, i in zip(lines, colours, range(len(lines))):
        if labels is not None:
            label = labels[i]
        elif "label" in curve:
            label = curve["label"]
        else:
            label = f"{D:.0e} m$^2$ s$^{{-1}}$"
        ax.plot(curve["gradient"], curve["intensity"][row], color=colour, linewidth=2,
                linestyle=styles[i % len(styles)], label=label)
    if lines:
        max_gradient = max(c["gradient"][-1] for c, _row, _D in lines)
        ax.set_xlim(0, max_gradient * 1.25)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlabel("Gradient strength, $g$ / T m$^{-1}$")
    ax.set_ylabel("Intensity, $I/I_0$")
