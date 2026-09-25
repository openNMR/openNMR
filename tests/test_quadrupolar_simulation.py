"""Quadrupolar lineshapes (notebook/simulations.py, QUADRUPOLAR NMR).

The engine computes every frequency from closed-form expressions in the
B0 direction (first/second order) or from batched diagonalisation
(exact), and MAS by gamma-COMPUTE. Each piece is checked here against
something computed a different way:

* per-orientation frequencies against a dense, lab-frame reference
  (rotate the EFG tensor, build H_Z + H_Q as matrices, diagonalise /
  Rayleigh-Schroedinger perturbation theory with explicit sums) -- the
  same route as Euan's SCRIPT_Quad_Simulator notebook;
* textbook closed forms: the CT isotropic shift, the static eta = 0 CT
  edges, the 2H Pake doublet horns, transition intensity shares;
* MAS sideband intensities against an explicit average over starting
  rotor phases of the time signal, integrated step by step in time.
"""
import time

import numpy as np
import pytest

from opennmr.notebook import simulations as S

NU0 = 130.3e6          # 27Al at ~11.7 T


# --- independent dense reference ------------------------------------------------

def _ops(I):
    m = np.arange(I, -I - 1, -1.0)
    n = len(m)
    Ip = np.zeros((n, n))
    for j in range(1, n):
        Ip[j - 1, j] = np.sqrt(I * (I + 1) - m[j] * (m[j] + 1))
    Ix = (Ip + Ip.T) / 2
    Iy = (Ip - Ip.T) / 2j
    return Ix, Iy, np.diag(m)


def _reference(I, CQ, eta, nu0, alpha, beta):
    """Lab frame, B0 along z, EFG PAS rotated by Rz(alpha) Ry(beta).
    Returns exact, first- and second-order frequencies minus nu0 for
    transitions ordered m = I ... -I+1 (pair m-1 <-> m)."""
    Ix, Iy, Iz = _ops(I)
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]) @ np.array(
        [[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    V = R @ np.diag([-(1 - eta) / 2, -(1 + eta) / 2, 1.0]) @ R.T
    ops = [Ix, Iy, Iz]
    HQ = CQ / (2 * I * (2 * I - 1)) * sum(V[a, b] * ops[a] @ ops[b]
                                          for a in range(3) for b in range(3))
    HZ = -nu0 * Iz
    E = np.linalg.eigvalsh(HZ + HQ)                   # ascending: m = I first
    exact = np.diff(E) - nu0
    EZ = np.real(np.diag(HZ))
    E1 = np.real(np.diag(HQ))
    E2 = np.array([sum(abs(HQ[n, k]) ** 2 / (EZ[k] - EZ[n])
                       for n in range(len(EZ)) if n != k) for k in range(len(EZ))])
    first = np.diff(E1)             # basis order m = I..-I: E_{m-1} - E_m
    second = first + np.diff(E2)
    return exact, first, second, R.T @ np.array([0, 0, 1.0])


@pytest.mark.parametrize("I, CQ, eta", [(2.5, 3e6, 0.3), (1.5, 2.5e6, 0.8), (1.0, 0.2e6, 0.1),
                                        (3.5, 6e6, 0.0)])
def test_orientation_frequencies_match_a_dense_lab_frame_reference(I, CQ, eta):
    rng = np.random.default_rng(3)
    for _ in range(15):
        alpha, beta = rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi)
        exact, first, second, b = _reference(I, CQ, eta, NU0, alpha, beta)
        th, ph = np.arccos(np.clip(b[2], -1, 1)), np.arctan2(b[1], b[0])
        for order, ref in (("exact", exact), ("first", first), ("second", second)):
            got = S.quadrupolar_transition_frequencies(I, CQ, eta, NU0, th, ph,
                                                       order=order)["offset_Hz"][0]
            assert got == pytest.approx(ref, abs=1e-4), order


def test_second_order_is_within_third_order_of_exact():
    th, ph, _ = S.zcw_angles(2000)
    f = {o: S.quadrupolar_transition_frequencies(2.5, 3e6, 0.3, NU0, th, ph, order=o)["offset_Hz"]
         for o in S.QUADRUPOLAR_ORDERS}
    nuQ = S.quadrupolar_nu_Q(2.5, 3e6)
    assert np.abs(f["second"] - f["exact"]).max() < 10 * nuQ ** 3 / NU0 ** 2
    # ... while first order misses the (much larger) second-order part.
    assert np.abs(f["first"] - f["exact"]).max() > 100 * np.abs(f["second"] - f["exact"]).max()


@pytest.mark.parametrize("I, eta", [(1.5, 0.0), (2.5, 0.3), (3.5, 1.0)])
def test_ct_isotropic_shift_is_the_powder_mean(I, eta):
    th, ph, w = S.zcw_angles(10000)
    f = S.quadrupolar_transition_frequencies(I, 4e6, eta, NU0, th, ph, order="second",
                                             transitions="ct")["offset_Hz"][:, 0]
    assert np.sum(w * f) == pytest.approx(S.quadrupolar_ct_isotropic_shift_Hz(I, 4e6, eta, NU0),
                                          rel=2e-3)


def test_transition_table_and_intensity_shares():
    table = S.quadrupolar_transition_table(2.5)
    assert [r["group"] for r in table] == ["ST2", "ST1", "CT", "ST1", "ST2"]
    assert table[2]["label"] == "−1/2↔+1/2"
    ax = np.linspace(-20000, 20000, 8192)
    all_ = S.quadrupolar_static_spectrum(2.5, 3e6, 0.3, NU0, ax, order="second", fwhm=0,
                                         n_zcw=2000)
    assert all_["real_spec"].sum() == pytest.approx(1.0, abs=1e-9)
    assert all_["components"]["CT"].sum() == pytest.approx(9 / 35, abs=1e-9)
    assert set(all_["components"]) == {"CT", "ST1", "ST2"}
    ct = S.quadrupolar_static_spectrum(1.5, 2e6, 0.3, NU0, ax, transitions="ct", fwhm=0,
                                       n_zcw=2000)
    assert ct["real_spec"].sum() == pytest.approx(0.4, abs=1e-3)    # 4 / (3+4+3)
    assert set(ct["components"]) == {"CT"}


def test_integer_spin_has_no_ct():
    assert {r["group"] for r in S.quadrupolar_transition_table(1.0)} == {"ST1"}
    with pytest.raises(ValueError, match="no central"):
        S.quadrupolar_static_spectrum(1.0, 1e5, 0.0, NU0, np.linspace(-1, 1, 64),
                                      transitions="ct")
    with pytest.raises(ValueError, match="Unknown transition"):
        S.quadrupolar_static_spectrum(2.5, 1e6, 0.0, NU0, np.linspace(-1, 1, 64),
                                      transitions="ST7")


def test_static_ct_eta0_second_order_edges():
    """eta = 0: nu_CT = -A (1 - x)(9x - 1), x = cos^2 theta, A = nu_Q^2 [I(I+1) - 3/4] / (16 nu0),
    so the pattern runs from +A (x = 0) down to -16A/9 (x = 5/9)."""
    I, CQ = 2.5, 5e6
    A = S.quadrupolar_nu_Q(I, CQ) ** 2 * (I * (I + 1) - 0.75) / (16 * NU0) / NU0 * 1e6  # ppm
    ax = np.linspace(-60, 40, 20001)
    spec = S.quadrupolar_static_spectrum(I, CQ, 0.0, NU0, ax, order="second", transitions="ct",
                                         fwhm=0, n_zcw=50000)["real_spec"]
    support = ax[spec > 0]
    assert support.max() == pytest.approx(A, abs=0.02)
    assert support.min() == pytest.approx(-16 * A / 9, abs=0.02)


def test_deuterium_pake_horns():
    """2H, eta = 0: horns at +-3 C_Q / 8, feet at +-3 C_Q / 4."""
    CQ, nu0 = 160e3, 61.4e6
    ax = np.linspace(-3000, 3000, 12001)
    spec = S.quadrupolar_static_spectrum(1.0, CQ, 0.0, nu0, ax, order="exact", fwhm=0,
                                         n_zcw=50000)["real_spec"]
    half = ax > 0
    horn = ax[half][np.argmax(spec[half])] * nu0 / 1e6
    assert horn == pytest.approx(3 * CQ / 8, rel=0.01)
    assert (ax[spec > 0].max() * nu0 / 1e6) == pytest.approx(3 * CQ / 4, rel=0.01)


# --- MAS ------------------------------------------------------------------------

def _brute_force_sidebands(I, CQ, eta, nu0, vr, th, ph, m, order, nt=256, sub=32, n_g0=160):
    """Explicit gamma average of s(t) = exp(2 pi i int (nu - c0) dt), trapezoid in time."""
    beta = np.arccos(1 / np.sqrt(3))
    r, e1, e2 = S._rotor_frame(np.array([th]), np.array([ph]))
    r, e1, e2 = r[0], e1[0], e2[0]
    tf = np.arange(nt * sub + 1) / (vr * nt * sub)
    g = 2 * np.pi * np.arange(nt * sub) / (nt * sub)
    b = np.cos(beta) * r + np.sin(beta) * (np.cos(g)[:, None] * e1 + np.sin(g)[:, None] * e2)
    c0 = S._perturbative_offsets(I, CQ, eta, nu0, b, [m], order)[:, 0].mean()
    sbar = np.zeros(nt, complex)
    for g0 in 2 * np.pi * np.arange(n_g0) / n_g0:
        gg = g0 + 2 * np.pi * vr * tf
        bb = np.cos(beta) * r + np.sin(beta) * (np.cos(gg)[:, None] * e1 + np.sin(gg)[:, None] * e2)
        f = S._perturbative_offsets(I, CQ, eta, nu0, bb, [m], order)[:, 0] - c0
        phase = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(tf))])
        sbar += np.exp(2j * np.pi * phase[:-1:sub])
    return (np.fft.fft(sbar / n_g0) / nt).real, c0


@pytest.mark.parametrize("m, order", [(1.5, "first"), (0.5, "second"), (1.5, "second")])
def test_mas_sidebands_match_explicit_gamma_average(m, order):
    I, CQ, eta, vr = 2.5, 3e6, 0.3, 10e3
    th, ph = 0.9, 2.1
    brute, c0 = _brute_force_sidebands(I, CQ, eta, NU0, vr, th, ph, m, order)
    # the engine, for this one crystallite, via its internals
    beta = np.arccos(1 / np.sqrt(3))
    r, e1, e2 = S._rotor_frame(np.array([th]), np.array([ph]))
    gam = 2 * np.pi * np.arange(16) / 16
    b = np.cos(beta) * r[:, None] + np.sin(beta) * (np.cos(gam)[None, :, None] * e1[:, None]
                                                    + np.sin(gam)[None, :, None] * e2[:, None])
    nu = S._perturbative_offsets(I, CQ, eta, NU0, b, [m], order)[0, :, 0]
    co = np.fft.fft(nu) / 16
    assert co[0].real == pytest.approx(c0, abs=1e-6)
    k = np.fft.fftfreq(16, 1 / 16).round().astype(int)
    keep = (k != 0) & (abs(k) < 8)
    Phi = np.zeros(256, complex)
    Phi[np.mod(k[keep], 256)] = co[keep] / (2j * np.pi * k[keep] * vr)
    a = np.fft.fft(np.exp(2j * np.pi * np.fft.ifft(Phi).real * 256)) / 256
    assert np.abs(a) ** 2 == pytest.approx(brute, abs=1e-5 * brute.max())


def test_mas_ct_centre_of_gravity_and_first_order_limit():
    I, CQ, eta, dI = 2.5, 5e6, 0.4, 10.0
    ax = np.linspace(-2000, 2000, 65536)
    second = S.quadrupolar_mas_spectrum(I, CQ, eta, NU0, 20e3, ax, order="second",
                                        transitions="ct", delta_iso=dI, fwhm=0)
    spec = second["real_spec"]
    cog = np.sum(ax * spec) / spec.sum()
    iso = dI + S.quadrupolar_ct_isotropic_shift_Hz(I, CQ, eta, NU0) / NU0 * 1e6
    assert cog == pytest.approx(iso, abs=0.01)
    assert spec.sum() == pytest.approx(9 / 35, abs=1e-6)
    first = S.quadrupolar_mas_spectrum(I, CQ, eta, NU0, 20e3, ax, order="first",
                                       transitions="ct", delta_iso=dI, fwhm=0)["real_spec"]
    # at first order the CT is unshifted and unbroadened by C_Q: one line at delta_iso
    assert np.sum(first[np.abs(ax - dI) < 0.1]) == pytest.approx(9 / 35, abs=1e-6)


def test_mas_second_order_and_exact_agree_at_high_field():
    ax = np.linspace(-60, 20, 4096)
    kw = dict(transitions="ct", fwhm=0.5, n_zcw=987)
    s2 = S.quadrupolar_mas_spectrum(2.5, 3e6, 0.3, NU0, 12e3, ax, order="second", **kw)
    ex = S.quadrupolar_mas_spectrum(2.5, 3e6, 0.3, NU0, 12e3, ax, order="exact", **kw)
    assert np.max(np.abs(s2["real_spec"] - ex["real_spec"])) < 0.02 * s2["real_spec"].max()


def test_mas_satellites_give_a_sideband_manifold():
    ax = np.linspace(-8000, 8000, 16384)
    res = S.quadrupolar_mas_spectrum(2.5, 3e6, 0.0, NU0, 10e3, ax, order="second",
                                     transitions="st", fwhm=0.5, n_zcw=610)
    st = res["components"]["ST1"]
    spacing_ppm = 10e3 / NU0 * 1e6
    # sidebands every nu_r: sample the pattern at +-k nu_r vs half-way between
    k = np.arange(1, 20)
    on = np.interp(k * spacing_ppm, ax, st)
    off = np.interp((k + 0.5) * spacing_ppm, ax, st)
    assert np.all(on > 20 * off)
    assert res["real_spec"].sum() == pytest.approx(1 - 9 / 35, abs=0.01)
    assert "CT" not in res["components"]


def test_mas_is_fast():
    ax = np.linspace(-60, 20, 4096)
    t = time.perf_counter()
    S.quadrupolar_mas_spectrum(2.5, 3e6, 0.3, NU0, 10e3, ax, order="second", transitions="ct",
                               n_zcw=4181)
    assert time.perf_counter() - t < 3.0


def test_mas_refuses_an_aliasing_rotor_grid():
    with pytest.raises(ValueError, match="sidebands"):
        S.quadrupolar_mas_spectrum(2.5, 20e6, 0.0, 20e6, 500.0, np.linspace(-1, 1, 64),
                                   transitions="all", n_zcw=50, max_n_time=4096)
    with pytest.raises(ValueError, match="mas_Hz"):
        S.quadrupolar_mas_spectrum(2.5, 1e6, 0.0, NU0, 0.0, np.linspace(-1, 1, 64))


def test_dispatch_comparison_and_axis_limits():
    lo, hi = S.quadrupolar_axis_limits(2.5, 3e6, 0.3, NU0, transitions="ct", fwhm=0.5)
    ax = np.linspace(lo, hi, 2048)
    cmp = S.quadrupolar_order_comparison(2.5, 3e6, 0.3, NU0, ax, transitions="ct", n_zcw=987)
    assert {"first", "second", "exact", "second_minus_first", "exact_minus_second"} <= set(cmp)
    assert cmp["exact"]["regime"] == "static" and "exact diagonalisation" in cmp["exact"]["log"]
    # framed: nothing significant at the edges
    s = cmp["exact"]["real_spec"]
    assert max(s[0], s[-1]) < 1e-3 * s.max()
    mas = S.quadrupolar_spectrum(2.5, 3e6, 0.3, NU0, ax, mas_Hz=15e3, order="second",
                                 transitions="ct", n_zcw=377)
    assert mas["regime"] == "mas" and mas["mas_Hz"] == 15e3
    assert S.larmor_frequency_Hz("27Al", 11.7467) == pytest.approx(130.3e6, rel=2e-3)


def test_plot_quadrupolar_orders_draws_orders_and_differences(tmp_path):
    from opennmr.notebook.plotting import plot_quadrupolar_orders
    ax_ppm = np.linspace(-60, 30, 1024)
    cmp = S.quadrupolar_order_comparison(2.5, 5e6, 0.2, NU0, ax_ppm, transitions="ct", n_zcw=377)
    ax = plot_quadrupolar_orders(cmp, nuclide="27Al", fname=str(tmp_path / "q.png"), font=None)
    assert len(ax.get_lines()) == 3
    assert len(ax.difference_ax.get_lines()) >= 2
    assert (tmp_path / "q.png").exists()
