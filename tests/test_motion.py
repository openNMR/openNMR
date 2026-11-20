"""notebook/motion.py: motional lineshapes of anisotropic interactions.

Expected values are analytic or come from the engines already in openNMR:
* frequencies at k -> 0 equal Simulate's first-order transition frequencies;
* an isotropic two-site hop equals exchange.py (Bloch-McConnell) to round-off;
* a rotation about an axis tilted by theta from a tensor's unique axis
  reduces an axial tensor by P2(cos theta) = (3 cos^2 theta - 1)/2;
* the fast limit is the population-weighted tensor average, and the slow
  limit the population-weighted sum of static patterns;
* the Herzfeld-Berger skew follows Mason: kappa = 3(d22 - d_iso)/Omega.
"""
import math

import numpy as np
import pytest

from opennmr.notebook import motion as M
from opennmr.notebook import exchange as ex
from opennmr.notebook.combined_hamiltonian import (
    Site, _rotation_zyz, simulate_combined, site_transition_frequencies,
)

LARMOR_2H = 61.4e6
PPM_2H = np.linspace(-3500, 3500, 1024)


def _cdf(y):
    c = np.cumsum(y)
    return c / c[-1]


def _cdf_gap(a, b):
    return float(np.max(np.abs(_cdf(a) - _cdf(b))))


def _cd_site(label="D", fwhm=1.0, C_Q=0.17, eta_Q=0.0):
    return Site("2H", 0.0, label=label, fwhm=fwhm).add_quadrupolar(C_Q, eta_Q)


# ---------------------------------------------------------------- geometry

def test_euler_round_trip_including_edge_cases():
    rng = np.random.default_rng(3)
    cases = [_rotation_zyz(*rng.uniform(0, 360, 3)) for _ in range(40)]
    cases += [_rotation_zyz(*e) for e in [(0, 0, 30), (0, 180, 30), (10, 0, 0), (40, 180, 0)]]
    for R in cases:
        assert np.allclose(_rotation_zyz(*M.euler_from_rotation(R)), R, atol=1e-9)


def test_rotation_about_axis_and_axis_vector():
    R = M.rotation_about_axis("z", math.pi / 2)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(M.axis_vector((90, 0)), [1, 0, 0])
    assert np.allclose(M.axis_vector([0, 0, 5]), [0, 0, 1])
    with pytest.raises(ValueError):
        M.axis_vector("q")
    with pytest.raises(ValueError):
        M.axis_vector([0, 0, 0])


def test_dipolar_coupling_constant_of_a_ch_bond():
    assert M.dipolar_coupling_Hz("13C", "1H", 1.09) == pytest.approx(23.3e3, rel=0.01)
    assert M.dipolar_coupling_Hz("13C", "1H", 2.18) == pytest.approx(
        M.dipolar_coupling_Hz("13C", "1H", 1.09) / 8.0, rel=1e-9)


def test_dipolar_partner_validation():
    ms = M.MotionSite(Site("13C", 0.0))
    with pytest.raises(ValueError):
        ms.add_dipolar("2H", D_Hz=1000.0)          # spin 1 partner
    with pytest.raises(ValueError):
        ms.add_dipolar("13C", D_Hz=1000.0)         # homonuclear
    with pytest.raises(ValueError):
        ms.add_dipolar("1H")                       # neither D nor r
    with pytest.raises(ValueError):
        ms.add_dipolar("1H", D_Hz=1.0, r_A=1.0)    # both


# --------------------------------------------------------- frequencies

def test_state_frequencies_equal_simulate_first_order():
    site = (Site("2H", 5.0, fwhm=0.5).add_quadrupolar(0.17, 0.1)
            .add_csa(30.0, 0.4, "Haeberlen", euler=(20, 40, 10)))
    model = M.MotionModel([M.MotionSite(site)])
    data, lines = model.state_data(), model.lines("all")
    th = np.array([0.3, 1.0, 2.0, 2.9])
    ph = np.array([0.1, 2.0, 4.0, 5.5])
    b = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    mine = M._line_frequencies(data, M._forms(data, b), lines, LARMOR_2H, len(th))[:, 0, :]
    ref = site_transition_frequencies(site, LARMOR_2H, th, ph, order="first",
                                      transitions="all")["offset_Hz"]
    assert np.allclose(mine.T, ref, atol=1e-6)


def test_placement_euler_rotates_the_tensor():
    """A site placed by Euler angles in the molecular frame, seen from a
    rotated field direction, has the frequencies of the unplaced site."""
    site = _cd_site(fwhm=0.0)
    euler = (30.0, 50.0, 70.0)
    placed = M.MotionModel([M.MotionSite(site, euler=euler)])
    plain = M.MotionModel([M.MotionSite(site)])
    R = _rotation_zyz(*euler)
    b_mol = np.array([[0.3, -0.5, 0.81]])
    b_mol = b_mol / np.linalg.norm(b_mol)
    f1 = M._line_frequencies(placed.state_data(), M._forms(placed.state_data(), b_mol),
                             placed.lines(), LARMOR_2H, 1)
    f2 = M._line_frequencies(plain.state_data(), M._forms(plain.state_data(), b_mol @ R),
                             plain.lines(), LARMOR_2H, 1)
    assert np.allclose(f1, f2, atol=1e-6)


def test_dipolar_pattern_edges_are_plus_minus_D():
    ms = M.MotionSite(Site("13C", 0.0)).add_dipolar("1H", D_Hz=22000.0)
    model = M.MotionModel([ms])
    lines = model.lines()
    assert len(lines) == 2 and sum(l["weight"] for l in lines) == pytest.approx(1.0)
    data = model.state_data()
    b = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    nu = M._line_frequencies(data, M._forms(data, b), lines, 100e6, 2)[:, 0, :]
    assert sorted(nu[:, 0]) == pytest.approx([-22000.0, 22000.0])
    assert sorted(nu[:, 1]) == pytest.approx([-11000.0, 11000.0])


def test_quadrupolar_lines_weights_and_selection():
    model = M.MotionModel([Site("23Na", 0.0).add_quadrupolar(1.0, 0.0)])
    assert len(model.lines("all")) == 3
    assert sum(l["weight"] for l in model.lines("all")) == pytest.approx(1.0)
    ct = model.lines("ct")
    assert len(ct) == 1 and ct[0]["group"] == "CT" and ct[0]["qcoef"] == 0.0


# --------------------------------------------------------- rates / states

def test_rate_matrix_is_a_valid_generator_and_populations_stationary():
    flip_mode = M.flip(180.0, "z", k_Hz=500.0)
    ring = M.ring_jump(3, "x", k_Hz=200.0)
    sA, sB = _cd_site("A"), _cd_site("B")
    hops = [{"from": 0, "to": 1, "Ea_eV": 0.30, "Ea_rev_eV": 0.32, "nu0_Hz": 1e12}]
    model = M.MotionModel([sA, sB], hops=hops, modes=[ring, flip_mode])
    assert model.n_states == 2 * 3 * 2
    K = model.rate_matrix(T_K=300.0)
    assert K.shape == (12, 12)
    assert np.allclose(K.sum(axis=0), 0.0, atol=1e-6 * np.abs(K).max())
    p = model.populations(T_K=300.0)
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(K @ p, 0.0, atol=1e-9 * np.abs(K).max())
    kT = ex.K_B_EV * 300.0
    p_env = p.reshape(2, 6).sum(axis=1)
    assert p_env[1] / p_env[0] == pytest.approx(math.exp(0.02 / kT), rel=1e-6)


def test_arrhenius_hops_need_a_temperature():
    hops = [{"from": 0, "to": 1, "Ea_eV": 0.3, "nu0_Hz": 1e12}]
    model = M.MotionModel([_cd_site("A"), _cd_site("B")], hops=hops)
    with pytest.raises(ValueError):
        model.rate_matrix()
    assert model.rate_matrix(T_K=300.0).shape == (2, 2)


def test_arrhenius_mode_rate():
    mode = M.ring_jump(3, "z", Ea_eV=0.2, nu0_Hz=1e12)
    assert mode.rate_value(300.0) == pytest.approx(1e12 * math.exp(-0.2 / (ex.K_B_EV * 300.0)))
    with pytest.raises(ValueError):
        mode.rate_value()
    assert mode.rate_value(300.0, k_override=7.0) == 7.0


def test_ring_diffusion_decay_rates():
    D, n = 1000.0, 24
    K = M.ring_diffusion("z", D_Hz=D, n=n).rate_matrix()
    lam = np.sort(np.linalg.eigvals(K).real)[::-1]            # 0, then pairs m = 1, 2, ...
    assert lam[0] == pytest.approx(0.0, abs=1e-6)
    assert -lam[1] == pytest.approx(D, rel=1e-9)               # m = 1 decays at exactly D
    m2 = -2 * (D / (2 * (1 - math.cos(2 * math.pi / n)))) * (1 - math.cos(4 * math.pi / n))
    assert m2 == pytest.approx(-4 * D, rel=0.05)               # m = 2 within 5 % of 4D
    with pytest.raises(ValueError):
        M.ring_diffusion("z", D_Hz=1.0, n=4)


def test_groups_and_isotropic_averaging():
    assert M.random_reorientation("octahedral", k_Hz=1.0).n == 24
    assert M.random_reorientation("tetrahedral", k_Hz=1.0).n == 12
    for grp in ("octahedral", "tetrahedral"):
        model = M.MotionModel([M.MotionSite(_cd_site(), euler=(20, 60, 10))],
                              modes=[M.random_reorientation(grp, k_Hz=1.0)])
        eff = M.averaged_tensors(model)
        assert abs(eff["quad"]["C_Q"]) < 1e-3
    with pytest.raises(ValueError):
        M.random_reorientation("icosahedral", k_Hz=1.0)


def test_mode_order_is_innermost_first():
    m1 = M.ring_jump(3, "z", k_Hz=1.0)
    m2 = M.flip(180.0, "x", k_Hz=1.0)
    model = M.MotionModel([_cd_site()], modes=[m1, m2])
    R = model.state_rotations()
    for s, (_label, (g1, g2)) in enumerate(model.state_table()):
        assert np.allclose(R[s], m2.rotations[g2] @ m1.rotations[g1])
    # the rotations do not commute: with the modes swapped, the state that
    # is "ring site 1, flipped" is a different rotation
    swapped = M.MotionModel([_cd_site()], modes=[m2, m1])
    both = [i for i, (_l, idx) in enumerate(swapped.state_table()) if idx == (1, 1)][0]
    mine = [i for i, (_l, idx) in enumerate(model.state_table()) if idx == (1, 1)][0]
    assert not np.allclose(R[mine], swapped.state_rotations()[both])


def test_kron_sum_dimensions_and_independence():
    K1, K2 = M.ring_jump(3, "z", k_Hz=2.0).rate_matrix(), M.flip(180, "x", k_Hz=5.0).rate_matrix()
    K = M._kron_sum([np.zeros((1, 1)), K1, K2])
    assert np.allclose(K, np.kron(K1, np.eye(2)) + np.kron(np.eye(3), K2))


# --------------------------------------------------------- averaged tensors

@pytest.mark.parametrize("half_angle", [0.0, 20.0, 40.0, 70.0, 90.0])
@pytest.mark.parametrize("n_sites", [3, 4, 6])
def test_cone_jump_reduces_an_axial_tensor_by_P2(half_angle, n_sites):
    model = M.cone_jump(_cd_site(), half_angle, n_sites, k_Hz=1e6)
    eff = M.averaged_tensors(model)
    p2 = 0.5 * (3 * math.cos(math.radians(half_angle)) ** 2 - 1)
    assert eff["quad"]["C_Q"] == pytest.approx(p2 * 170e3, abs=1e-3)
    assert eff["quad"]["eta_Q"] == pytest.approx(0.0, abs=1e-6)


def test_cone_jump_with_two_sites_is_asymmetric():
    """Two sites do not average the perpendicular plane: eta_Q > 0."""
    model = M.cone_jump(_cd_site(), 60.0, 2, k_Hz=1e6)
    assert M.averaged_tensors(model)["quad"]["eta_Q"] > 0.05


def test_average_equals_group_average_of_the_tensor():
    rng = np.random.default_rng(5)
    ms = M.MotionSite(Site("13C", 3.0).add_csa(80.0, 0.3), euler=tuple(rng.uniform(0, 180, 3)))
    mode = M.ring_jump(5, M.axis_vector((35.0, 20.0)), k_Hz=10.0)
    model = M.MotionModel([ms], modes=[mode])
    T = ms.tensors()["csa"]
    manual = sum(R @ T @ R.T for R in mode.rotations) / mode.n
    avg = M._average_data(model.state_data(), model.populations())
    assert np.allclose(avg["csa"][0], manual, atol=1e-10)


def test_two_modes_average_is_the_nested_average():
    ms = M.MotionSite(_cd_site(), euler=(10.0, 55.0, 0.0))
    m1 = M.ring_jump(3, "z", k_Hz=1.0)
    m2 = M.flip(180.0, "x", k_Hz=1.0)
    model = M.MotionModel([ms], modes=[m1, m2])
    T = ms.tensors()["quad"]
    inner = sum(R @ T @ R.T for R in m1.rotations) / 3
    nested = sum(R @ inner @ R.T for R in m2.rotations) / 2
    avg = M._average_data(model.state_data(), model.populations())
    assert np.allclose(avg["quad"][0], nested, atol=1e-8)


def test_hop_average_shift_and_tensors_are_population_weighted():
    sA = Site("13C", 40.0, label="A", fwhm=1.0).add_csa(100.0, 0.0)
    sB = Site("13C", -10.0, label="B", fwhm=1.0).add_csa(50.0, 0.0)
    hops = [{"from": 0, "to": 1, "Ea_eV": 0.30, "Ea_rev_eV": 0.32, "nu0_Hz": 1e12}]
    model = M.MotionModel([sA, sB], hops=hops)
    eff = M.averaged_tensors(model, T_K=300.0)
    p = eff["populations"]
    assert eff["delta_iso"] == pytest.approx(p[0] * 40.0 + p[1] * (-10.0))
    assert eff["csa"]["delta_aniso"] == pytest.approx(p[0] * 100.0 + p[1] * 50.0)


def test_herzfeld_berger_skew_follows_mason():
    """Axial, unique d11 (Haeberlen Delta > 0): d22 = d33, kappa = -1.
    Axial, unique d33 (Delta < 0): d22 = d11, kappa = +1."""
    up = M.averaged_tensors(M.MotionModel([Site("13C", 0.0).add_csa(50.0, 0.0)]))["csa"]
    down = M.averaged_tensors(M.MotionModel([Site("13C", 0.0).add_csa(-50.0, 0.0)]))["csa"]
    assert up["skew"] == pytest.approx(-1.0, abs=1e-9) and up["span"] == pytest.approx(75.0)
    assert down["skew"] == pytest.approx(1.0, abs=1e-9)
    # a general tensor: Delta = 60, eta = 0.4 -> d = (60, -18, -42), d_iso = 0,
    # kappa = 3 (d22 - d_iso) / (d11 - d33) = 3 (-18) / 102
    gen = M.averaged_tensors(M.MotionModel([Site("13C", 0.0).add_csa(60.0, 0.4)]))["csa"]
    assert gen["span"] == pytest.approx(102.0) and gen["skew"] == pytest.approx(-54.0 / 102.0)


def test_csa_relative_euler_in_efg_frame_reconstructs_the_tensors():
    site = (Site("2H", 0.0, fwhm=1.0).add_quadrupolar(0.17, 0.2)
            .add_csa(40.0, 0.5, "Haeberlen", euler=(25.0, 35.0, 15.0)))
    model = M.cone_jump(site, 30.0, 3, k_Hz=1e5)
    eff = M.averaged_tensors(model)
    avg = M._average_data(model.state_data(), model.populations())
    (_vx, _vy, _vz), E_q = M._principal_axes(avg["quad"][0], "efg")
    D_pas = np.diag(M._principal_axes(avg["csa"][0], "csa")[0])
    R = _rotation_zyz(*eff["euler_csa_in_efg"])
    assert np.allclose(E_q @ (R @ D_pas @ R.T) @ E_q.T, avg["csa"][0], atol=1e-9)


def test_effective_site_is_a_simulate_site_with_the_averaged_values():
    model = M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1e6)
    site = M.effective_site(model)
    q = site.get("quadrupolar")[0]
    assert q["C_Q"] == pytest.approx(0.17 * 0.5 * (3 * math.cos(math.radians(40.0)) ** 2 - 1),
                                     abs=1e-6)
    spec = simulate_combined(site, PPM_2H, larmor_Hz=LARMOR_2H, order="first", n_zcw=1597)
    assert np.isfinite(spec["real_spec"]).all()


def test_effective_site_warns_about_dipolar():
    ms = M.MotionSite(Site("13C", 0.0)).add_dipolar("1H", D_Hz=20000.0)
    model = M.MotionModel([ms], modes=[M.flip(180.0, "x", k_Hz=1.0)])
    with pytest.warns(UserWarning):
        M.effective_site(model)


# ---------------------------------------------------------- spectra

def test_isotropic_two_site_exchange_equals_exchange_module():
    sA = Site("7Li", 20.0, label="A", fwhm=0.5, lineshape="lorentzian")
    sB = Site("7Li", 0.0, label="B", fwhm=0.5, lineshape="lorentzian")
    hops = [{"from": 0, "to": 1, "k_Hz": 3000.0}]
    x = np.linspace(40, -20, 2048)
    mine = M.motion_spectrum(M.MotionModel([sA, sB], hops=hops), x, larmor_Hz=155.5e6)["real_spec"]
    ref = ex.exchange_spectrum([{"shift_ppm": 20.0, "fwhm_ppm": 0.5},
                                {"shift_ppm": 0.0, "fwhm_ppm": 0.5}], hops, 300.0,
                               larmor_MHz=155.5, ppm_axis=x, normalise=None)["spectrum"]
    assert np.allclose(mine / mine.max(), ref / ref.max(), atol=1e-8)


def test_spectrum_area_is_the_amount():
    model = M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1e4)
    r = M.motion_spectrum(model, PPM_2H, larmor_Hz=LARMOR_2H, n_orient=1597, gauss_fwhm_ppm=40)
    assert r["real_spec"].sum() == pytest.approx(1.0, rel=0.03)
    two = M.MotionModel([_cd_site("A"), _cd_site("B")], hops=[{"from": 0, "to": 1, "k_Hz": 1e4}])
    r2 = M.motion_spectrum(two, PPM_2H, larmor_Hz=LARMOR_2H, n_orient=1597, gauss_fwhm_ppm=40)
    assert r2["real_spec"].sum() == pytest.approx(2.0, rel=0.03)


def test_slow_and_fast_rates_reach_the_two_limits():
    model = M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1.0)
    kw = dict(larmor_Hz=LARMOR_2H, gauss_fwhm_ppm=40)
    lim = M.limit_spectra(model, PPM_2H, **kw)
    slow = M.motion_spectrum(model, PPM_2H, k_modes={0: 1.0}, n_orient=4181, **kw)["real_spec"]
    fast = M.motion_spectrum(model, PPM_2H, k_modes={0: 1e10}, n_orient=4181, **kw)["real_spec"]
    assert _cdf_gap(slow, lim["static"]) < 0.03
    assert _cdf_gap(fast, lim["fast"]) < 0.03
    assert _cdf_gap(slow, fast) > 0.05            # the two ends genuinely differ


def _iqr(y, x):
    c = _cdf(y)
    return float(np.interp(0.75, c, x) - np.interp(0.25, c, x))


def test_motional_narrowing_of_the_interquartile_width():
    """A robust width (the interquartile range: Lorentzian tails of the
    exchange broadening do not enter it) is the static one at a very slow
    rate and the averaged tensor's at a very fast one -- here 8x narrower
    (P2(cos 60 deg) = -1/8)."""
    model = M.cone_jump(_cd_site(), 60.0, 3, k_Hz=1.0)
    series = M.motion_rate_series(model, [1e2, 1e9], which="mode:0", ppm_axis=PPM_2H,
                                  larmor_Hz=LARMOR_2H, n_orient=2584, gauss_fwhm_ppm=20,
                                  include_limits=False)
    slow, fast = (_iqr(s, PPM_2H) for s in series["spectra"])
    lim = M.limit_spectra(model, PPM_2H, larmor_Hz=LARMOR_2H, gauss_fwhm_ppm=20)
    assert slow == pytest.approx(_iqr(lim["static"], PPM_2H), rel=0.08)
    assert fast == pytest.approx(_iqr(lim["fast"], PPM_2H), rel=0.15)
    assert fast < 0.4 * slow


def test_hop_and_rotation_together_fast_limit_shift_and_state_count():
    sA = Site("13C", 40.0, label="A", fwhm=1.0).add_csa(100.0, 0.0)
    sB = Site("13C", -10.0, label="B", fwhm=1.0).add_csa(100.0, 0.0)
    model = M.MotionModel([sA, sB], hops=[{"from": 0, "to": 1, "k_Hz": 1e9}],
                          modes=[M.flip(180.0, "x", k_Hz=1e9)])
    assert model.n_states == 4
    x = np.linspace(-200, 200, 1024)
    avg = M.averaged_spectrum(model, x, larmor_Hz=100.6e6, gauss_fwhm_ppm=2.0)
    assert avg["effective"]["delta_iso"] == pytest.approx(15.0)


def test_dynamic_mas_runs_in_the_time_domain_engine_and_keeps_the_area():
    model = M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1e4)
    r = M.motion_spectrum(model, PPM_2H, larmor_Hz=LARMOR_2H, mas_Hz=5000.0)
    assert r["td"] is not None and np.all(np.isfinite(r["real_spec"]))
    assert r["real_spec"].sum() == pytest.approx(1.0, rel=0.02)


def test_single_state_mas_is_served_by_the_averaged_engine():
    site = Site("13C", 0.0, label="C", fwhm=1.5).add_csa(100.0, 0.3)
    x = np.linspace(-300, 300, 4096)
    r = M.motion_spectrum(M.MotionModel([site]), x, larmor_Hz=100.6e6, mas_Hz=3000.0)
    assert r["real_spec"].sum() == pytest.approx(1.0, rel=0.02)


def test_averaged_spectrum_matches_simulate_static_and_mas():
    site = Site("13C", 5.0, label="C", fwhm=1.5).add_csa(100.0, 0.3)
    x = np.linspace(-300, 300, 4096)
    kw = dict(larmor_Hz=100.6e6)
    model = M.MotionModel([site])
    static = M.averaged_spectrum(model, x, **kw)["real_spec"]
    ref = simulate_combined(site, x, order="first", n_zcw=10946, **kw)["real_spec"]
    assert _cdf_gap(static, ref) < 0.01
    mas = M.averaged_spectrum(model, x, mas_Hz=3000.0, **kw)["real_spec"]
    ref_mas = simulate_combined(site, x, order="first", mas_Hz=3000.0, **kw)["real_spec"]
    assert np.max(np.abs(mas / mas.max() - ref_mas / ref_mas.max())) < 0.03


def test_validity_notes_say_what_is_not_modelled():
    na = M.MotionModel([Site("23Na", 0.0, fwhm=1.0).add_quadrupolar(1.0, 0.0)],
                       modes=[M.ring_jump(3, "z", k_Hz=1e3)])
    r = M.averaged_spectrum(na, np.linspace(-50, 50, 512), larmor_Hz=105.8e6, transitions="ct")
    assert "central transition" in r["log"]
    fast = M.MotionModel([_cd_site()], modes=[M.ring_jump(3, "z", k_Hz=1e8)])
    out = M.motion_spectrum(fast, PPM_2H, larmor_Hz=LARMOR_2H, n_orient=987)
    assert "quadrupolar frequency scale" in out["log"]


def test_models_validate_their_inputs():
    with pytest.raises(ValueError):
        M.MotionModel([_cd_site("A"), _cd_site("A")])
    with pytest.raises(ValueError):
        M.MotionModel([_cd_site("A"), Site("13C", 0.0, label="B")])
    pv = Site("2H", 0.0, lineshape="pseudo_voigt").add_quadrupolar(0.17, 0.0)
    with pytest.raises(ValueError):
        M.MotionModel([pv])
    with pytest.raises(ValueError):
        M.ring_jump(1, "z", k_Hz=1.0)
    with pytest.raises(ValueError):
        M.ring_jump(3, "z")
    with pytest.raises(ValueError):
        M.ring_jump(3, "z", k_Hz=1.0, Ea_eV=0.1)
    with pytest.raises(ValueError):
        M.motion_rate_series(M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1.0), [1.0], which="nonsense",
                             ppm_axis=PPM_2H, larmor_Hz=LARMOR_2H)


def test_model_round_trips_through_a_dict():
    ms = M.MotionSite(Site("13C", 3.0, fwhm=1.0).add_csa(80.0, 0.3), euler=(10, 20, 30))
    ms.add_dipolar("1H", r_A=1.09, vector=(40.0, 10.0))
    model = M.MotionModel([ms], modes=[M.ring_jump(3, "x", k_Hz=100.0),
                                       M.random_reorientation("tetrahedral", k_Hz=5.0)])
    again = M.MotionModel.from_dict(model.to_dict())
    x = np.linspace(-200, 200, 512)
    a = M.averaged_spectrum(model, x, larmor_Hz=100.6e6)["real_spec"]
    b = M.averaged_spectrum(again, x, larmor_Hz=100.6e6)["real_spec"]
    assert np.allclose(a, b)
    assert again.n_states == model.n_states == 3 * 12


def test_rate_series_returns_limits_and_plots():
    import matplotlib
    matplotlib.use("Agg")
    model = M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1.0)
    series = M.motion_rate_series(model, [1e3, 1e5, 1e7], which="mode:0", ppm_axis=PPM_2H,
                                  larmor_Hz=LARMOR_2H, n_orient=987, gauss_fwhm_ppm=60)
    assert len(series["spectra"]) == 3 and set(series["limits"]) >= {"static", "fast"}
    fig, ax = M.plot_motion_series(series, nucleus="2H")
    assert fig is not None
    code = M.motion_notebook_code(model, rates=[1e3, 1e5], larmor_MHz=61.4)
    namespace = {}
    exec(compile(code.split("ppm = ")[0], "<code>", "exec"), namespace)
    assert namespace["model"].n_states == 3


# ====================================================================== stage 2
# second-order quadrupolar frequencies, the two fast limits, correlated couplings

LARMOR_23NA = 105.8e6
PPM_23NA = np.linspace(-150, 150, 2048)


def _na_site(label="Na", C_Q=2.0, eta_Q=0.3, fwhm=1.0):
    return Site("23Na", 0.0, label=label, fwhm=fwhm).add_quadrupolar(C_Q, eta_Q)


def _b_from_angles(th, ph):
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)


def test_euler_is_accurate_close_to_the_poles():
    for euler in [(30.0, 179.9999999, 40.0), (30.0, 1e-7, 40.0), (0.0, 90.0, 0.0)]:
        R = _rotation_zyz(*euler)
        assert np.allclose(_rotation_zyz(*M.euler_from_rotation(R)), R, atol=1e-9)


def test_second_order_state_frequencies_equal_simulate_also_after_a_rotation():
    site = _na_site().add_csa(25.0, 0.2, "Haeberlen", euler=(15, 30, 45))
    flip = M.flip(180.0, "x", k_Hz=1.0)
    model = M.MotionModel([M.MotionSite(site)], modes=[flip])
    data, lines = model.state_data(), model.lines("all")
    th = np.array([0.4, 1.1, 2.2, 2.9])
    ph = np.array([0.2, 2.5, 4.0, 5.9])
    b = _b_from_angles(th, ph)
    nu = M._frequencies(data, b, lines, LARMOR_23NA, "second", model.spin)       # (L, S, n)
    for s, R in enumerate(model.state_rotations()):
        b_site = b @ R                                  # the field direction in this state's frame
        th_s = np.arccos(np.clip(b_site[:, 2], -1, 1))
        ph_s = np.arctan2(b_site[:, 1], b_site[:, 0])
        ref = site_transition_frequencies(site, LARMOR_23NA, th_s, ph_s, order="second",
                                          transitions="all")["offset_Hz"]
        assert np.allclose(nu[:, s, :].T, ref, atol=1e-6)


def test_first_and_second_order_agree_without_a_quadrupolar_coupling():
    model = M.MotionModel([Site("13C", 0.0).add_csa(50.0, 0.0)])
    lines, data = model.lines(), model.state_data()
    b = _b_from_angles(np.array([0.5, 1.5]), np.array([0.3, 2.0]))
    assert np.allclose(M._frequencies(data, b, lines, 1e8, "first", 0.5),
                       M._frequencies(data, b, lines, 1e8, "second", 0.5))
    with pytest.raises(ValueError):
        M._frequencies(data, b, lines, 1e8, "exact", 0.5)


def test_second_order_central_transition_matches_simulate_static_and_mas():
    site = _na_site()
    model = M.MotionModel([site])
    for mas in (0.0, 15000.0):
        mine = M.averaged_spectrum(model, PPM_23NA, larmor_Hz=LARMOR_23NA, transitions="ct",
                                   order="second", mas_Hz=mas)["real_spec"]
        ref = simulate_combined(site, PPM_23NA, larmor_Hz=LARMOR_23NA, order="second",
                                transitions="ct", mas_Hz=mas, n_zcw=10946)["real_spec"]
        assert _cdf_gap(mine, ref) < 0.02, mas
        assert mine.sum() == pytest.approx(ref.sum(), rel=0.03)


def test_the_first_order_central_transition_is_one_line_and_says_so():
    model = M.MotionModel([_na_site()])
    r = M.averaged_spectrum(model, PPM_23NA, larmor_Hz=LARMOR_23NA, transitions="ct")
    assert "central transition" in r["log"]
    assert np.argmax(r["real_spec"]) == pytest.approx(len(PPM_23NA) // 2, abs=3)   # at delta_iso
    second = M.averaged_spectrum(model, PPM_23NA, larmor_Hz=LARMOR_23NA, transitions="ct",
                                 order="second")
    assert "central transition has no first-order" not in second["log"]
    assert _cdf_gap(r["real_spec"], second["real_spec"]) > 0.05    # the shift and shape are real


def test_dynamic_second_order_reaches_its_two_limits():
    model = M.cone_jump(_na_site(), 40.0, 3, k_Hz=1.0)
    kw = dict(larmor_Hz=LARMOR_23NA, transitions="ct", order="second", gauss_fwhm_ppm=3.0)
    lim = M.limit_spectra(model, PPM_23NA, **kw)
    slow = M.motion_spectrum(model, PPM_23NA, k_modes={0: 1.0}, n_orient=4181, **kw)
    fast = M.motion_spectrum(model, PPM_23NA, k_modes={0: 1e10}, n_orient=4181, **kw)
    assert "state by state" in slow["log"]
    assert _cdf_gap(slow["real_spec"], lim["static"]) < 0.03
    assert _cdf_gap(fast["real_spec"], lim["fast"]) < 0.03
    assert _cdf_gap(lim["static"], lim["fast"]) > 0.05


def test_second_order_has_two_different_fast_limits_first_order_one():
    na = M.cone_jump(_na_site(), 40.0, 3, k_Hz=1.0)
    lim = M.limit_spectra(na, PPM_23NA, larmor_Hz=LARMOR_23NA, transitions="ct", order="second",
                          gauss_fwhm_ppm=3.0)
    assert _cdf_gap(lim["fast"], lim["fast_tensor"]) > 0.01        # frequency average ≠ EFG average
    d = M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1.0)
    lim1 = M.limit_spectra(d, PPM_2H, larmor_Hz=LARMOR_2H, gauss_fwhm_ppm=40)
    assert _cdf_gap(lim1["fast"], lim1["fast_tensor"]) < 2e-3       # identical at first order


def test_second_order_validity_note_for_fast_motion():
    model = M.MotionModel([_na_site()], modes=[M.ring_jump(3, "x", k_Hz=1e6)])
    r = M.motion_spectrum(model, PPM_23NA, larmor_Hz=LARMOR_23NA, transitions="ct",
                          order="second", n_orient=987)
    assert "quadrupolar frequency scale" in r["log"]


# ---------------------------------------------------------------- couplings

def _two_li(k=3000.0):
    sA = Site("7Li", 20.0, label="A", fwhm=0.5, lineshape="lorentzian")
    sB = Site("7Li", 0.0, label="B", fwhm=0.5, lineshape="lorentzian")
    return sA, sB


def test_a_coupling_equals_the_equivalent_hop():
    sA, sB = _two_li()
    hop = M.MotionModel([sA, sB], hops=[{"from": 0, "to": 1, "k_Hz": 3000.0}])
    coupled = M.MotionModel([sA, sB], couplings=[{"from": 0, "to": 1, "k_Hz": 3000.0}])
    assert np.allclose(hop.rate_matrix(), coupled.rate_matrix())
    assert np.allclose(hop.populations(), coupled.populations())
    x = np.linspace(40, -20, 1024)
    a = M.motion_spectrum(hop, x, larmor_Hz=155.5e6)["real_spec"]
    b = M.motion_spectrum(coupled, x, larmor_Hz=155.5e6)["real_spec"]
    assert np.allclose(a, b, atol=1e-12)


def test_a_coupling_joins_environment_and_orientation_together():
    sA, sB = _two_li()
    model = M.MotionModel([sA, sB], modes=[M.flip(180.0, "x", k_Hz=0.0)],
                          couplings=[{"from": (0, 0), "to": (1, 1), "k_Hz": 500.0, "k_rev_Hz": 250.0}])
    K = model.rate_matrix()
    i, j = model._state_index((0, 0)), model._state_index((1, 1))
    assert K[j, i] == 500.0 and K[i, j] == 250.0
    assert np.count_nonzero(K - np.diag(np.diag(K))) == 2      # nothing else is connected
    assert np.allclose(K.sum(axis=0), 0.0)
    p = model.populations()
    assert p.sum() == pytest.approx(1.0) and np.allclose(K @ p, 0.0, atol=1e-9)
    assert p[j] / p[i] == pytest.approx(2.0)                   # detailed balance of the pair
    assert "coupled jump" in "\n".join(model.describe())


def test_coupling_arrhenius_populations_and_scaling():
    sA, sB = _two_li()
    model = M.MotionModel([sA, sB], couplings=[{"from": 0, "to": 1, "Ea_eV": 0.30,
                                                "Ea_rev_eV": 0.32, "nu0_Hz": 1e12}])
    with pytest.raises(ValueError):
        model.rate_matrix()
    p = model.populations(T_K=300.0)
    assert p[1] / p[0] == pytest.approx(math.exp(0.02 / (ex.K_B_EV * 300.0)), rel=1e-6)
    assert np.allclose(model.rate_matrix(T_K=300.0, hop_scale=2.0),
                       2.0 * model.rate_matrix(T_K=300.0))


def test_couplings_are_validated_and_round_trip():
    sA, sB = _two_li()
    with pytest.raises(ValueError):
        M.MotionModel([sA, sB], couplings=[{"from": 0, "to": 0, "k_Hz": 1.0}])
    with pytest.raises(ValueError):
        M.MotionModel([sA, sB], couplings=[{"from": 0, "to": 5, "k_Hz": 1.0}])
    with pytest.raises(ValueError):
        M.MotionModel([sA, sB], modes=[M.flip(180.0, "x", k_Hz=1.0)],
                      couplings=[{"from": (0,), "to": (1, 1), "k_Hz": 1.0}])
    ok = M.MotionModel([sA, sB], modes=[M.flip(180.0, "x", k_Hz=1.0)],
                       couplings=[{"from": (0, 0), "to": (1, 1), "k_Hz": 5.0}])
    again = M.MotionModel.from_dict(ok.to_dict())
    assert again.couplings == ok.couplings
    assert np.allclose(again.rate_matrix(), ok.rate_matrix())


# ====================================================================== stage 3
# time-domain engine: dynamics under MAS and echo sequences

LARMOR_13C = 100.6e6
PPM_13C = np.linspace(250, -150, 2048)
PPM_2H_FINE = np.linspace(-150e3, 150e3, 4096) / LARMOR_2H * 1e6     # +-150 kHz, 73 Hz steps


def _csa_pair():
    a = Site("13C", 150.0, label="A", fwhm=2.0, lineshape="lorentzian").add_csa(80.0, 0.4, "Haeberlen")
    b = Site("13C", 50.0, label="B", fwhm=2.0, lineshape="lorentzian").add_csa(-60.0, 0.2, "Haeberlen")
    return M.MotionModel([a, b], hops=[{"from": 0, "to": 1, "k_Hz": 1.0}])


def _cone_2h():
    d = Site("2H", 0.0, label="D", fwhm=3.0, lineshape="lorentzian").add_quadrupolar(0.17, 0.0)
    return M.cone_jump(d, 40.0, 3, k_Hz=1.0)


def test_batched_expm_matches_scipy_including_stiff_matrices():
    from scipy.linalg import expm
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 4, 4)) + 1j * rng.normal(size=(6, 4, 4))
    assert np.allclose(M._batched_expm(A), [expm(a) for a in A], atol=1e-10)
    K = np.array([[-1e4, 1e4], [1e4, -1e4]], dtype=complex)
    stiff = (K + np.diag([2j * np.pi * 3e3, -2j * np.pi * 3e3])) * 1e-3
    assert np.allclose(M._batched_expm(stiff[None])[0], expm(stiff), atol=1e-9)


def test_echo_arguments_are_validated():
    assert M._parse_echo(None) is None
    assert M._parse_echo(("hahn", 1e-4)) == {"kind": "hahn", "tau_s": 1e-4}
    assert M._parse_echo({"kind": "solid", "tau_us": 50})["tau_s"] == pytest.approx(5e-5)
    for bad in ("hahn", {"kind": "xx", "tau_s": 1e-4}, {"kind": "hahn"}, ("hahn", -1.0)):
        with pytest.raises(ValueError):
            M._parse_echo(bad)
    with pytest.raises(ValueError):                                  # solid echo needs I > 1/2
        M.motion_spectrum(M.MotionModel([Site("13C", 0.0).add_csa(50.0, 0.0)]),
                          np.linspace(100, -100, 512), larmor_Hz=1e8, echo=("solid", 1e-4))
    with pytest.raises(ValueError):
        M.motion_spectrum(_csa_pair(), PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=-1.0)


@pytest.mark.parametrize("kind", ["hahn", "solid"])
def test_static_echo_without_motion_is_the_static_spectrum_with_t2_loss(kind):
    model, tau = _cone_2h(), 30e-6
    static = M.motion_spectrum(model, PPM_2H_FINE, larmor_Hz=LARMOR_2H)["real_spec"]
    r = M.motion_spectrum(model, PPM_2H_FINE, larmor_Hz=LARMOR_2H, echo={"kind": kind, "tau_s": tau})
    assert _cdf_gap(r["real_spec"], static) < 2e-3
    expect = math.exp(-math.pi * 3.0 * LARMOR_2H * 1e-6 * 2.0 * tau)     # R2 = pi * FWHM, over 2 tau
    assert r["td"]["echo"]["intensity"] == pytest.approx(expect, rel=2e-3)
    assert r["td"]["echo"]["tau_s"] == pytest.approx(tau)
    assert "echo amplitude" in r["log"]


def test_echo_intensity_is_lost_at_intermediate_rates_and_recovers_when_fast():
    model, tau = _cone_2h(), 30e-6
    inten = [M.motion_spectrum(model, PPM_2H_FINE, larmor_Hz=LARMOR_2H, k_modes={0: k},
                               echo=("solid", tau))["td"]["echo"]["intensity"]
             for k in (1e2, 1e5, 1e8)]
    assert inten[1] < 0.2 * inten[0]
    assert inten[2] > 5 * inten[1]


def test_hahn_echo_refocuses_a_csa_and_an_offset_but_the_solid_echo_does_not():
    site = Site("2H", 20.0, label="D", fwhm=3.0, lineshape="lorentzian").add_quadrupolar(
        0.17, 0.0).add_csa(100.0, 0.0, "Haeberlen")
    model, tau = M.MotionModel([site]), 30e-6
    hahn = M.motion_spectrum(model, PPM_2H_FINE, larmor_Hz=LARMOR_2H, echo=("hahn", tau))
    assert hahn["td"]["echo"]["intensity"] == pytest.approx(
        math.exp(-math.pi * 3.0 * LARMOR_2H * 1e-6 * 2.0 * tau), rel=2e-3)
    solid = M.motion_spectrum(model, PPM_2H_FINE, larmor_Hz=LARMOR_2H, echo=("solid", tau))
    assert "does not refocus shifts" in solid["log"]
    assert _cdf_gap(hahn["real_spec"], solid["real_spec"]) > 1e-3


def test_dynamic_mas_slow_limit_is_the_sum_of_the_single_state_mas_spectra():
    model = _csa_pair()
    lim = M.limit_spectra(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    r = M.motion_spectrum(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    assert _cdf_gap(r["real_spec"], lim["static"]) < 3e-3
    assert r["real_spec"].sum() == pytest.approx(2.0, rel=0.03)
    assert r["td"]["tail"] < 5e-3 and "time domain, MAS 5 kHz" in r["log"]


def test_dynamic_mas_reaches_the_fast_limit_and_not_before():
    model = _csa_pair()
    lim = M.limit_spectra(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    gaps = [_cdf_gap(M.motion_spectrum(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0,
                                       hop_scale=hs)["real_spec"], lim["fast"])
            for hs in (1e5, 1e7, 1e9)]
    assert gaps[0] > 0.1 > 0.03 > gaps[1] > gaps[2]
    assert gaps[2] < 5e-3


def test_isotropic_exchange_is_the_same_static_and_under_mas():
    a = Site("13C", 30.0, label="A", fwhm=1.0, lineshape="lorentzian")
    b = Site("13C", 0.0, label="B", fwhm=1.0, lineshape="lorentzian")
    model = M.MotionModel([a, b], hops=[{"from": 0, "to": 1, "k_Hz": 1500.0}])
    x = np.linspace(60, -30, 1024)
    s = M.motion_spectrum(model, x, larmor_Hz=LARMOR_13C)["real_spec"]
    m = M.motion_spectrum(model, x, larmor_Hz=LARMOR_13C, mas_Hz=8000.0)["real_spec"]
    assert _cdf_gap(s, m) < 1e-3
    assert np.max(np.abs(s - m)) < 0.01 * s.max()


def test_mas_hahn_echo_is_the_slow_spectrum_when_rotor_synchronised():
    model = _csa_pair()
    lim = M.limit_spectra(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    r = M.motion_spectrum(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0,
                          echo={"kind": "hahn", "tau_s": 2.0 / 5000.0})
    assert _cdf_gap(r["real_spec"], lim["static"]) < 3e-3
    expect = math.exp(-math.pi * 2.0 * LARMOR_13C * 1e-6 * 2.0 * (2.0 / 5000.0))
    assert r["td"]["echo"]["intensity"] == pytest.approx(expect, rel=5e-3)
    assert "rotor-synchronised" not in r["log"]
    off = M.motion_spectrum(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0,
                            echo=("hahn", 1.5 / 5000.0))
    assert "rotor-synchronised" in off["log"]


def test_single_state_under_mas_keeps_the_exact_engine_unless_an_echo_is_asked():
    one = M.MotionModel([_csa_pair().sites[0]])
    assert "td" not in M.motion_spectrum(one, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    assert M.motion_spectrum(one, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0,
                             echo=("hahn", 2.0 / 5000.0))["td"] is not None


def test_time_domain_second_order_central_transition_matches_simulate_under_mas():
    site = Site("23Na", 0.0, label="Na", fwhm=1.0, lineshape="lorentzian").add_quadrupolar(2.0, 0.3)
    ref = simulate_combined(site, PPM_23NA, larmor_Hz=LARMOR_23NA, order="second",
                            transitions="ct", mas_Hz=15000.0, n_zcw=10946)["real_spec"]
    r = M.motion_spectrum(M.MotionModel([site]), PPM_23NA, larmor_Hz=LARMOR_23NA, mas_Hz=15000.0,
                          transitions="ct", order="second", echo=("hahn", 2.0 / 15000.0))
    assert _cdf_gap(r["real_spec"], ref) < 5e-3


def test_rate_series_and_notebook_code_carry_mas_and_echo():
    model = _csa_pair()
    ser = M.motion_rate_series(model, [1e3, 1e5], which="hops", ppm_axis=PPM_13C,
                               larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    assert len(ser["spectra"]) == 2 and ser["results"][0]["settings"]["mas_Hz"] == 5000.0
    assert {"static", "fast", "fast_tensor"} <= set(ser["limits"])
    code = M.motion_notebook_code(model, rates=[1e3], which="hops", larmor_MHz=100.6,
                                  mas_Hz=5000.0, echo={"kind": "hahn", "tau_s": 2e-4})
    compile(code, "notebook_code", "exec")
    assert "mas_Hz=5000" in code and "'hahn'" in code


# ====================================================================== stage 4
# building from plain data, temperature series, overlay on an experimental spectrum

def _arrhenius_2h():
    d = Site("2H", 0.0, label="D", fwhm=3.0, lineshape="lorentzian").add_quadrupolar(0.17, 0.0)
    return M.cone_jump(d, 40.0, 3, Ea_eV=0.30, nu0_Hz=1e12)


def test_mode_from_spec_builds_every_kind_and_rejects_nonsense():
    ring = M.mode_from_spec({"kind": "ring_jump", "n": "6", "axis": "z", "k_Hz": "1e4"})
    assert ring.n == 6 and ring.rate == {"k_Hz": 1e4}
    flip = M.mode_from_spec({"kind": "flip", "angle_deg": "180", "axis": "1,0,0", "Ea_eV": 0.2})
    assert flip.n == 2 and flip.rate["Ea_eV"] == 0.2 and flip.rate["nu0_Hz"] == 1e12
    diff = M.mode_from_spec({"kind": "ring_diffusion", "n": 12, "axis": "y", "k_Hz": 500.0})
    assert diff.n == 12 and diff.rate_is == "D"
    oct_ = M.mode_from_spec({"kind": "random_reorientation", "group": "octahedral", "k_Hz": 10})
    assert oct_.n == 24
    assert M.mode_from_spec(ring) is ring
    with pytest.raises(ValueError):
        M.mode_from_spec({"kind": "wobble", "k_Hz": 1.0})
    with pytest.raises(ValueError):
        M.mode_from_spec({"kind": "ring_jump", "n": 3, "k_Hz": 1.0, "Ea_eV": 0.3})


def test_model_from_spec_equals_the_direct_construction():
    site = _cd_site()
    direct = M.cone_jump(site, 40.0, 3, k_Hz=1e4)
    built = M.model_from_spec([M.MotionSite(site, euler=(0.0, 40.0, 0.0))],
                              [{"kind": "ring_jump", "n": 3, "axis": "z", "k_Hz": 1e4}])
    assert np.allclose(built.rate_matrix(), direct.rate_matrix())
    for a, b in zip(built.state_data()["quad"], direct.state_data()["quad"]):
        assert np.allclose(a, b)


def test_temperature_series_uses_arrhenius_rates_and_labels_in_kelvin():
    model = _arrhenius_2h()
    x = np.linspace(2500, -2500, 512)
    series = M.motion_rate_series(model, [200.0, 300.0, 400.0], which="T", ppm_axis=x,
                                  larmor_Hz=LARMOR_2H)
    assert series["which"] == "T" and series["limits_at_T_K"] == 200.0
    k = [model.rate_matrix(T_K=T)[1, 0] for T in (200.0, 300.0, 400.0)]
    assert k[0] < k[1] < k[2]
    assert _cdf_gap(series["spectra"][0], series["spectra"][2]) > 0.02
    fig, ax = M.plot_motion_series(series, nucleus="2H")
    assert any(t.get_text() == "300 K" for t in ax.texts)
    with pytest.raises(ValueError):
        M.motion_rate_series(model, [300.0], which="bogus", ppm_axis=x, larmor_Hz=LARMOR_2H)


def test_axis_limits_frame_the_pattern_and_grow_under_mas():
    model = M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1e4)
    hi, lo = M.motion_axis_limits(model, larmor_Hz=LARMOR_2H)
    assert hi > 0.75 * 0.17e6 / LARMOR_2H * 1e6 and lo == pytest.approx(-hi, rel=1e-6)
    hi_mas, _ = M.motion_axis_limits(model, larmor_Hz=LARMOR_2H, mas_Hz=10000.0)
    assert hi_mas > hi
    spec = M.motion_spectrum(model, np.linspace(hi, lo, 1024), larmor_Hz=LARMOR_2H)["real_spec"]
    assert spec.sum() == pytest.approx(1.0, rel=0.02)


def test_overlay_recovers_the_scale_and_offset_of_a_scaled_simulation():
    model = M.cone_jump(_cd_site(fwhm=3.0), 40.0, 3, k_Hz=1e5)
    x = np.linspace(2500, -2500, 1024)
    sim = M.motion_spectrum(model, x, larmor_Hz=LARMOR_2H)["real_spec"]
    ov = M.motion_overlay(model, x, 5.0 * sim + 0.01, larmor_Hz=LARMOR_2H, n_points=1024)
    assert ov["scale"] == pytest.approx(5.0, rel=1e-6) and ov["offset"] == pytest.approx(0.01, abs=1e-9)
    assert ov["rms"] < 1e-9 and "Scaled onto the experiment" in ov["log"]
    no_off = M.motion_overlay(model, x, 5.0 * sim, fit_offset=False, larmor_Hz=LARMOR_2H,
                              n_points=1024)
    assert no_off["offset"] == 0.0 and no_off["scale"] == pytest.approx(5.0, rel=1e-6)


def test_overlay_window_and_argument_checks():
    model = M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1e5)
    x = np.linspace(2500, -2500, 1024)
    y = np.ones_like(x)
    ov = M.motion_overlay(model, x, y, window=(-1000.0, 1000.0), larmor_Hz=LARMOR_2H, n_points=512)
    assert ov["x_ppm"].min() >= -1000.0 and ov["x_ppm"].max() <= 1000.0
    with pytest.raises(ValueError):
        M.motion_overlay(model, x, y[:-1], larmor_Hz=LARMOR_2H)
    with pytest.raises(ValueError):
        M.motion_overlay(model, x, y, window=(0.0, 1.0), larmor_Hz=LARMOR_2H)


def test_overlay_series_simulates_each_spectrum_at_its_own_temperature():
    model = _arrhenius_2h()
    x = np.linspace(2500, -2500, 512)
    spectra = []
    for T in (250.0, 350.0):
        sim = M.motion_spectrum(model, x, larmor_Hz=LARMOR_2H, T_K=T)["real_spec"]
        spectra.append({"ppm": x, "y": 2.0 * sim, "T_K": T})
    out = M.motion_overlay_series(model, spectra, larmor_Hz=LARMOR_2H, n_points=512)
    assert out["labels"] == ["250 K", "350 K"]
    assert [o["scale"] for o in out["overlays"]] == pytest.approx([2.0, 2.0], rel=1e-6)
    wrong = M.motion_overlay_series(model, spectra, per_spectrum_temperature=False,
                                    larmor_Hz=LARMOR_2H, T_K=250.0, n_points=512)
    assert wrong["overlays"][1]["rms"] > 10 * out["overlays"][1]["rms"] + 1e-6
    fig, ax = M.plot_motion_overlay_stack(out, nucleus="2H")
    assert len(ax.lines) == 4


def test_overlay_and_limit_plots_draw():
    model = M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1e5)
    x = np.linspace(2500, -2500, 512)
    sim = M.motion_spectrum(model, x, larmor_Hz=LARMOR_2H)["real_spec"]
    ov = M.motion_overlay(model, x, sim, larmor_Hz=LARMOR_2H, n_points=512)
    fig, ax = M.plot_motion_overlay(ov, nucleus="2H")
    assert len(ax.lines) == 3 and ax.get_xlim()[0] > ax.get_xlim()[1]
    fig2, ax2 = M.plot_motion_limits(M.limit_spectra(model, x, larmor_Hz=LARMOR_2H), nucleus="2H")
    assert len(ax2.lines) == 3


def test_notebook_code_carries_the_range_extras_and_overlay_block():
    model = M.cone_jump(_cd_site(), 40.0, 3, k_Hz=1e4)
    code = M.motion_notebook_code(model, rates=[1e3, 1e5], larmor_MHz=61.4,
                                  ppm_range=(2500.0, -2500.0), extra={"order": "first"},
                                  overlay={"k_modes": {0: 1e5}})
    compile(code, "notebook_code", "exec")
    assert "np.linspace(2500, -2500" in code and "order='first'" in code
    assert "motion_overlay(model, ppm_exp, y_exp" in code and "k_modes={0: 100000.0}" in code
