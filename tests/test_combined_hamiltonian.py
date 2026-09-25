"""Combined Hamiltonian (notebook/combined_hamiltonian.py): one site,
any mix of CSA, quadrupolar and J, static or MAS.

Each interaction alone must reproduce the dedicated simulator built and
tested separately (CSA static / finite MAS, quadrupolar, J multiplets);
combinations are checked against an independent per-orientation
calculation (explicit Euler rotation, dense Hamiltonian), and under MAS
against an explicit average over starting rotor phases.
"""
import numpy as np
import pytest

from opennmr.notebook import simulations as S
from opennmr.notebook import combined_hamiltonian as C
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

NU0_AL = 130.3e6
NU0_C = 100.6e6


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# --- the Site API -----------------------------------------------------------------

def test_adding_interactions_chains_and_round_trips():
    s = (Site("27Al", 12.0, label="Al1").add_quadrupolar(4.2, 0.35)
         .add("csa", aniso=40, asym=0.2, euler=(0, 90, 0))
         .add("j", nuclide="31P", n=2, J_Hz=15).add_jcoupling("1H", 1, 5))
    assert [i["type"] for i in s.interactions] == ["quadrupolar", "csa", "jcoupling", "jcoupling"]
    back = Site.from_dict(s.to_dict())
    assert back.to_dict() == s.to_dict()
    assert C.sites_from_json(C.sites_to_json([s]))[0].to_dict() == s.to_dict()
    with pytest.raises(ValueError, match="already has a CSA"):
        s.add_csa(10)
    with pytest.raises(ValueError, match="already has a quadrupolar"):
        s.add_quadrupolar(1)
    with pytest.raises(ValueError, match="no quadrupole moment"):
        Site("13C").add_quadrupolar(1)
    with pytest.raises(ValueError, match="Unknown interaction"):
        s.add("dipolar")
    s.remove("jcoupling")
    assert len(s.get("j")) == 0 and len(s.interactions) == 2
    assert "C_Q 4.2 MHz" in s.describe() and "Euler (0, 90, 0)" in s.describe()


# --- each interaction alone = its dedicated simulator -----------------------------

@pytest.mark.parametrize("mas", [0.0, 15e3])
def test_quadrupolar_alone_matches_the_quadrupolar_simulator(mas):
    ax = np.linspace(-80, 60, 4096)
    a = simulate_combined(Site("27Al", 12, fwhm=0.5).add_quadrupolar(4.2, 0.35), ax, NU0_AL,
                          mas_Hz=mas, transitions="ct")["real_spec"]
    b = S.quadrupolar_spectrum(2.5, 4.2e6, 0.35, NU0_AL, ax, mas_Hz=mas, transitions="ct",
                               delta_iso=12, fwhm=0.5, n_zcw=4181 if mas else 10946)["real_spec"]
    assert _rel(a, b) < 1e-3


def test_csa_alone_static_matches_the_csa_simulator():
    ax = np.linspace(-300, 300, 8192)
    a = simulate_combined(Site("13C", 20, fwhm=2).add_csa(150, 0.4), ax, NU0_C)["real_spec"]
    b = S.make_static_csa_spectrum(S.csa_definition(20, 150, 0.4), ax, fwhm=2,
                                   n_theta=600, n_phi=1200)["real_spec"]
    assert _rel(a / a.max(), b / b.max()) < 0.01


def test_csa_alone_mas_matches_finite_mas_csa_including_sideband_asymmetry():
    """Regression: the octant shortcut once averaged sideband k with -k,
    which made CSA sideband patterns symmetric. They are not."""
    ax, vr = np.linspace(-300, 300, 8192), 5e3
    a = simulate_combined(Site("13C", 20, fwhm=2).add_csa(150, 0.4), ax, NU0_C, mas_Hz=vr)["real_spec"]
    b = S.finite_mas_csa_spectrum(S.csa_definition(20, 150, 0.4), ax, NU0_C, vr, fwhm=2,
                                  powder_scheme="zcw", n_zcw=20000, n_sidebands=40)["real_spec"]
    assert _rel(a / a.max(), b / b.max()) < 0.01
    spacing = vr / NU0_C * 1e6
    band = lambda k: a[np.abs(ax - (20 + k * spacing)) < spacing / 2].sum()
    assert band(-1) > 2 * band(1)          # clearly asymmetric, as it should be


def test_jcoupling_alone_gives_the_first_order_multiplet():
    ax = np.linspace(-5, 5, 20001)
    s = Site("13C", 0.0, fwhm=0.002).add_jcoupling("1H", 3, 125.0)
    spec = simulate_combined(s, ax, NU0_C)["real_spec"]
    f, w = S.combine_jcouplings([{"nuclide": "1H", "n": 3, "J_Hz": 125.0}])
    for fi, wi in zip(f, w):
        near = np.abs(ax - fi / NU0_C * 1e6) < 0.05
        assert spec[near].sum() / spec.sum() == pytest.approx(wi, abs=1e-3)


# --- combinations ---------------------------------------------------------------------

def _dense_reference(I, CQ, eta_q, csa_aniso, csa_eta, euler, nu0, b):
    """Independent: EFG PAS as the frame, CSA rotated with an explicitly
    built ZYZ matrix, quadrupolar part from a lab-frame dense Hamiltonian
    (second order by explicit Rayleigh-Schroedinger sums)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("ZYZ", euler, degrees=True).as_matrix()   # intrinsic ZYZ = Rz Ry Rz
    Dp = np.diag(S.haeberlen_csa_tensor(0.0, csa_aniso, csa_eta).diagonal())
    D = R @ Dp @ R.T
    shift_Hz = nu0 * 1e-6 * b @ D @ b
    # quadrupolar: rotate the frame so B0 is along z
    z = b / np.linalg.norm(b)
    x = np.cross([0.37, 0.61, 0.7], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    L = np.array([x, y, z])                          # PAS -> lab
    V = L @ np.diag([-(1 - eta_q) / 2, -(1 + eta_q) / 2, 1.0]) @ L.T
    m = np.arange(I, -I - 1, -1.0)
    n = len(m)
    Ip = np.zeros((n, n))
    for j in range(1, n):
        Ip[j - 1, j] = np.sqrt(I * (I + 1) - m[j] * (m[j] + 1))
    ops = [(Ip + Ip.T) / 2, (Ip - Ip.T) / 2j, np.diag(m)]
    HQ = CQ / (2 * I * (2 * I - 1)) * sum(V[a, c] * ops[a] @ ops[c] for a in range(3) for c in range(3))
    EZ = -nu0 * m
    E1 = np.real(np.diag(HQ))
    E2 = np.array([sum(abs(HQ[q, k]) ** 2 / (EZ[k] - EZ[q]) for q in range(n) if q != k)
                   for k in range(n)])
    return np.diff(E1) + np.diff(E2) + shift_Hz


@pytest.mark.parametrize("euler", [(0, 0, 0), (0, 90, 0), (30, 50, 70), (120, 25, -40)])
def test_quadrupolar_plus_csa_frequencies_match_an_independent_calculation(euler):
    s = Site("27Al", 5.0).add_quadrupolar(4.0, 0.4).add_csa(60, 0.3, euler=euler)
    rng = np.random.default_rng(2)
    th, ph = rng.uniform(0, np.pi, 12), rng.uniform(0, 2 * np.pi, 12)
    got = C.site_transition_frequencies(s, NU0_AL, th, ph, order="second")["offset_Hz"]
    for k in range(12):
        b = np.array([np.sin(th[k]) * np.cos(ph[k]), np.sin(th[k]) * np.sin(ph[k]), np.cos(th[k])])
        ref = _dense_reference(2.5, 4e6, 0.4, 60, 0.3, euler, NU0_AL, b) + 5.0e-6 * NU0_AL
        assert got[k] == pytest.approx(ref, abs=1e-3)


def test_euler_angles_mean_what_the_docs_say():
    """beta = 90, alpha = 0: the CSA's unique axis lies along the EFG x axis,
    so B0 along EFG x sees d_zz of the CSA."""
    s = Site("27Al", 0.0).add_quadrupolar(1e-6, 0.0).add_csa(60, 0.0, euler=(0, 90, 0))
    f = C.site_transition_frequencies(s, NU0_AL, np.pi / 2, 0.0, transitions="ct")["offset_Hz"][0, 0]
    assert f / NU0_AL * 1e6 == pytest.approx(60.0, abs=1e-6)


def test_octant_is_used_only_when_exact():
    ax = np.linspace(-400, 400, 16384)
    coaxial = Site("27Al", 0, fwhm=1).add_quadrupolar(3, 0.3).add_csa(80, 0.5)
    tilted = Site("27Al", 0, fwhm=1).add_quadrupolar(3, 0.3).add_csa(80, 0.5, euler=(30, 50, 70))
    for site, octant in ((coaxial, True), (tilted, False)):
        for mas in (0.0, 5e3):
            fast = simulate_combined(site, ax, NU0_AL, mas_Hz=mas, n_zcw=4181, transitions="all")
            assert fast["powder"][site.label]["octant"] is octant
            full = simulate_combined(site, ax, NU0_AL, mas_Hz=mas,
                                     n_zcw=17711 if mas else 75025,
                                     use_symmetry=False, transitions="all")
            # static: powder ripple at 4181 orientations on sharp edges; MAS converges faster
            assert _rel(fast["real_spec"], full["real_spec"]) < (0.05 if not mas else 0.01)
    assert "full sphere" in simulate_combined(tilted, ax, NU0_AL)["log"]


def test_mas_sidebands_of_a_combined_site_match_an_explicit_rotor_average():
    """One crystallite with CSA + quadrupolar (tilted) under MAS: gamma-COMPUTE
    sideband intensities vs averaging exp(2 pi i int nu dt) over starting phases."""
    s = Site("27Al", 0.0).add_quadrupolar(3.0, 0.3).add_csa(80, 0.5, euler=(30, 50, 70))
    rows = S._select_transitions(2.5, "ST1")[:1]
    vr, nt, sub, n_g0 = 10e3, 256, 32, 160
    beta = np.arccos(1 / np.sqrt(3))
    r, e1, e2 = S._rotor_frame(np.array([0.8]), np.array([2.3]))
    r, e1, e2 = r[0], e1[0], e2[0]
    traj = lambda g: (np.cos(beta) * r + np.sin(beta) * (np.cos(g)[:, None] * e1
                                                        + np.sin(g)[:, None] * e2))
    nu = lambda g: C._site_frequencies(s, NU0_AL, traj(g), rows, "second")[0][:, 0]
    g16 = 2 * np.pi * np.arange(16) / 16
    prep = S._mas_prepare(nu(g16)[None, :, None], vr, 16, nt)
    (_a, _b, centre, _sb, weight), = S._mas_sticks(prep, np.ones((1, 1)), np.ones(1))
    engine = weight[0, :, 0]
    tf = np.arange(nt * sub + 1) / (vr * nt * sub)
    sbar = np.zeros(nt, complex)
    for g0 in 2 * np.pi * np.arange(n_g0) / n_g0:
        f = nu(g0 + 2 * np.pi * vr * tf) - centre[0, 0]
        phase = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(tf))])
        sbar += np.exp(2j * np.pi * phase[:-1:sub])
    brute = (np.fft.fft(sbar / n_g0) / nt).real
    assert engine == pytest.approx(brute, abs=1e-5 * brute.max())


def test_j_splitting_survives_mas_and_adds_to_the_quadrupolar_ct():
    ax = np.linspace(-60, 40, 16384)
    plain = Site("27Al", 10, fwhm=0.1).add_quadrupolar(2.0, 0.0)
    split = plain.copy().add_jcoupling("31P", 1, 400.0)
    a = simulate_combined(plain, ax, NU0_AL, mas_Hz=20e3, transitions="ct")["real_spec"]
    b = simulate_combined(split, ax, NU0_AL, mas_Hz=20e3, transitions="ct")["real_spec"]
    shift = 200.0 / NU0_AL * 1e6                        # +-J/2 in ppm
    expected = 0.5 * (np.interp(ax - shift, ax, a) + np.interp(ax + shift, ax, a))
    assert _rel(b, expected) < 0.02
    assert b.sum() == pytest.approx(a.sum(), rel=1e-6)


def test_several_sites_amounts_and_shares():
    ax = np.linspace(-100, 80, 8192)
    s1 = Site("27Al", 10, label="A", amount=2.0).add_quadrupolar(3, 0.2)
    s2 = Site("27Al", -20, label="B", amount=1.0).add_quadrupolar(5, 0.6).add_csa(30, 0.1)
    res = simulate_combined([s1, s2], ax, NU0_AL, mas_Hz=15e3, transitions="ct")
    assert res["sites"]["A"].sum() / res["sites"]["B"].sum() == pytest.approx(2.0, rel=0.02)
    assert res["real_spec"].sum() == pytest.approx(3 * 9 / 35, rel=0.02)   # CT share of each
    assert set(res["components"]["A"]) == {"CT"}
    with pytest.raises(ValueError, match="same nucleus"):
        simulate_combined([s1, Site("13C")], ax, NU0_AL)
    with pytest.raises(ValueError, match="unique"):
        simulate_combined([s1, s1.copy()], ax, NU0_AL)


def test_spin_half_and_transition_selection():
    ax = np.linspace(-300, 300, 4096)
    res = simulate_combined(Site("31P", 0).add_csa(100, 0.3), ax, 202.4e6, transitions="ct")
    assert res["real_spec"].sum() == pytest.approx(1.0, rel=1e-3)
    with pytest.raises(ValueError, match="no transitions"):
        simulate_combined(Site("31P", 0), ax, 202.4e6, transitions="st")


def test_python_export_reproduces_the_simulation_and_axis_limits_frame_it():
    s = (Site("27Al", 12.0, label="Al1", fwhm=0.5).add_quadrupolar(4.2, 0.35)
         .add_csa(40, 0.2, euler=(0, 90, 0)).add_jcoupling("31P", 2, 150))
    lo, hi = C.combined_axis_limits(s, NU0_AL, transitions="ct")
    ax = np.linspace(lo, hi, 4096)
    res = simulate_combined(s, ax, NU0_AL, mas_Hz=15e3, transitions="ct")
    assert max(res["real_spec"][0], res["real_spec"][-1]) < 1e-3 * res["real_spec"].max()
    code = C.combined_to_python([s], ax, larmor_Hz=NU0_AL, mas_Hz=15e3, transitions="ct")
    ns = {}
    exec(code, ns)
    assert np.allclose(ns["spec"]["real_spec"], res["real_spec"])


def test_plot_combined(tmp_path):
    ax = np.linspace(-100, 80, 2048)
    res = simulate_combined([Site("27Al", 10, label="A").add_quadrupolar(3, 0.2),
                             Site("27Al", -20, label="B").add_quadrupolar(5, 0.6)],
                            ax, NU0_AL, transitions="ct", n_zcw=987)
    a = C.plot_combined(res, nuclide="27Al", fname=str(tmp_path / "c.png"))
    assert [l.get_label() for l in a.get_lines()] == ["Total", "A", "B"]
    assert (tmp_path / "c.png").exists()


def test_mas_axis_limits_include_csa_sidebands_beyond_the_static_range():
    """A 60 ppm CSA spun at 15 kHz (115 ppm at 130 MHz): its first
    sidebands lie outside the static range, and framing on that range
    clipped them."""
    s = Site("27Al", 12.0, fwhm=0.4).add_quadrupolar(4.2, 0.35).add_csa(60, 0.2)
    lo_s, hi_s = C.combined_axis_limits(s, NU0_AL, transitions="ct")
    lo, hi = C.combined_axis_limits(s, NU0_AL, transitions="ct", mas_Hz=15e3)
    assert hi > 12 + 115 and lo < 12 - 115 - 20
    assert hi > hi_s and lo < lo_s
    ax = np.linspace(lo, hi, 8192)
    spec = simulate_combined(s, ax, NU0_AL, mas_Hz=15e3, transitions="ct")["real_spec"]
    assert max(spec[0], spec[-1]) < 2e-3 * spec.max()
