"""Review of the tanh/tan adiabatic pulse (2026-10-13, user request).

Checks the shape against Hwang, van Zijl & Garwood, J. Magn. Reson. 133,
200 (1998) written as its two half-passages, and the Bloch propagation
against an independent integrator: the Bloch equations solved by
scipy's ODE solver in the frame that rotates with the RF's instantaneous
frequency, where the effective field is (nu1(t), 0, Omega - nu(t)) and
no phase ramp is involved. Before the review the phase was taken at the
end of each step rather than its midpoint, which put tanh/tan up to
0.04 out in Mz at the default 1 us step.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from opennmr.simulate import bloch

T_US, SW = 500.0, 100e3
XI, TAN_KAPPA = 10.0, 20.0


def _paper_shape(t, T):
    """Hwang et al. (1998): t in [0, T]; first half then second half."""
    kappa = np.arctan(TAN_KAPPA)
    A = SW / 2
    if t <= T / 2:
        return np.tanh(2 * XI * t / T), A * np.tan(kappa * (1 - 2 * t / T)) / TAN_KAPPA
    return np.tanh(2 * XI * (1 - t / T)), -A * np.tan(kappa * (2 * t / T - 1)) / TAN_KAPPA


def test_shape_is_the_published_tanh_tan_with_the_sweep_reversed():
    n = 1000
    amp, _ph, freq = bloch.adiabatic_shape("TANH_TAN", n, T_US, SW)
    t = (np.arange(n) + 0.5) / n * T_US
    paper = np.array([_paper_shape(x, T_US) for x in t])
    assert amp == pytest.approx(paper[:, 0] / paper[:, 0].max(), abs=1e-12)
    # The paper sweeps +A -> -A; openNMR sweeps -A -> +A (same pulse in
    # the opposite direction, which inverts identically -- checked below).
    assert freq == pytest.approx(-paper[:, 1], rel=1e-12, abs=1e-6)
    assert bloch.ADIABATIC_SHAPES["TANH_TAN"] == {"xi": XI, "tan_kappa": TAN_KAPPA}


def _ode_mz(b1, offsets):
    # RF frequency as openNMR sweeps it (-A -> +A): minus the paper's.
    T = T_US * 1e-6
    out = []
    for omega in offsets:
        def rhs(t, M):
            a, nu = _paper_shape(t / T * T_US, T_US)
            w = 2 * np.pi * np.array([b1 * a, 0.0, omega - (-nu)])
            return np.cross(w, M)
        sol = solve_ivp(rhs, (0, T), [0.0, 0.0, 1.0], rtol=1e-9, atol=1e-11, max_step=T / 4000)
        out.append(sol.y[2, -1])
    return np.array(out)


@pytest.mark.parametrize("b1", [10e3, 40e3])
def test_propagation_matches_an_independent_ode_solution(b1):
    offsets = np.array([-20e3, 0.0, 10e3, 30e3])
    sim = bloch.simulate_pulse("TANH_TAN", T_US, b1, offsets, sweep_width_hz=SW)
    assert sim["time_step_us"] == 1.0                     # the default, hardware step
    assert sim["Mz"][0] == pytest.approx(_ode_mz(b1, offsets), abs=0.01)


def test_default_step_is_converged_to_a_hundredth():
    # Was 0.04 out before the phase was taken at step midpoints.
    offsets = np.linspace(-60e3, 60e3, 61)
    for b1 in (20e3, 80e3):
        auto = bloch.simulate_pulse("TANH_TAN", T_US, b1, offsets, sweep_width_hz=SW)
        fine = bloch.simulate_pulse("TANH_TAN", T_US, b1, offsets, sweep_width_hz=SW,
                                    time_step_us=0.02)
        assert np.max(np.abs(auto["Mz"] - fine["Mz"])) < 0.01


def test_sweep_direction_does_not_change_the_inversion():
    offsets = np.array([-15e3, 0.0, 5e3, 25e3])
    up = bloch.simulate_pulse("TANH_TAN", T_US, 20e3, offsets, sweep_width_hz=SW)["Mz"][0]
    # The same pulse swept downwards: the profile mirrors in offset.
    down = bloch.simulate_pulse("TANH_TAN", T_US, 20e3, -offsets, sweep_width_hz=SW)["Mz"][0]
    assert up == pytest.approx(down, abs=1e-9)


def test_crossing_adiabaticity_is_the_on_resonance_formula():
    # Q = 2 pi nu1^2 / |d nu / dt| at the crossing (both in Hz).
    b1 = 20e3
    q = bloch.adiabaticity("TANH_TAN", T_US, SW, b1)
    kappa = np.arctan(TAN_KAPPA)
    rate_centre = (SW / 2) * kappa / TAN_KAPPA / (T_US * 1e-6 / 2)   # d nu/dt at tau = 0
    assert q["q0"] == pytest.approx(2 * np.pi * b1 ** 2 / rate_centre, rel=1e-3)
