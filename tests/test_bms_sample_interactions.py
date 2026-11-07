"""The BMS sample and local interactions (explicit user request, 2026-11-07):
the demagnetising field of a rotor (cylinder) and a plate (cuboid) against
exact results and brute-force quadrature; the macroscopic shift; local
contact, spin-dipolar and quadrupolar interactions against openNMR's
combined simulator; the 3-D packing; the run-time estimates."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from opennmr.notebook.bms import fast as F
from opennmr.notebook.bms import interactions as I
from opennmr.notebook.bms import sample as S
from opennmr.notebook.bms import views as V
from opennmr.notebook.bms.io_utils import make_particle

CHI3 = (2.87e-3, 3.13e-3, 2.27e-3)


@pytest.fixture(autouse=True)
def _sideband_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENNMR_BMS_CACHE", str(tmp_path / "sidebands"))


def _packing(n=20, chi=None):
    from opennmr.notebook.bms.packing import grow_and_separate_pack
    parts, _ = grow_and_separate_pack(n, 200.0, (20, 20, 20), (3, 3, 3), seed=4, verbose=False)
    return parts


# ---- demagnetising fields ---------------------------------------------------------------

def _brute_cylinder(pt, R, L, n=300):
    g, w = leggauss(n)
    G = np.zeros((3, 3))
    ph = 2 * np.pi * (np.arange(2 * n) + 0.5) / (2 * n)
    dph = 2 * np.pi / (2 * n)
    PH, Z = np.meshgrid(ph, L / 2 * g, indexing="ij")
    W = dph * np.outer(np.ones(2 * n), L / 2 * w)
    rp = np.stack([R * np.cos(PH), R * np.sin(PH), Z], -1)
    nrm = np.stack([np.cos(PH), np.sin(PH), 0 * PH], -1)
    d = pt - rp
    G += np.einsum("abi,abj,ab->ij", d, nrm, R * W / np.linalg.norm(d, axis=-1) ** 3)
    PH, RR = np.meshgrid(ph, R / 2 * (g + 1), indexing="ij")
    W = dph * np.outer(np.ones(2 * n), R / 2 * w) * RR
    for zd, sg in ((L / 2, 1), (-L / 2, -1)):
        rp = np.stack([RR * np.cos(PH), RR * np.sin(PH), zd + 0 * RR], -1)
        d = pt - rp
        G[:, 2] += sg * np.einsum("abi,ab->i", d, W / np.linalg.norm(d, axis=-1) ** 3)
    return -G / (4 * np.pi)


def _brute_box(pt, sides, n=200):
    g, w = leggauss(n)
    h = np.array(sides) / 2
    G = np.zeros((3, 3))
    for k in range(3):
        p, q = [m for m in range(3) if m != k]
        P, Q = np.meshgrid(h[p] * g, h[q] * g, indexing="ij")
        W = np.outer(h[p] * w, h[q] * w)
        for f, sg in ((h[k], 1), (-h[k], -1)):
            rp = np.zeros(P.shape + (3,))
            rp[..., k], rp[..., p], rp[..., q] = f, P, Q
            d = pt - rp
            G[:, k] += sg * np.einsum("abi,ab->i", d, W / np.linalg.norm(d, axis=-1) ** 3)
    return -G / (4 * np.pi)


@pytest.mark.parametrize("rho, z", [(0.0, 0.0), (0.5, 0.7), (0.9, 1.2), (0.3, -1.4)])
def test_cylinder_field_against_brute_force(rho, z):
    N = S.cylinder_demag(rho, z, 1.0, 3.0)[0]
    assert np.trace(N) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(N, N.T, atol=1e-9)
    assert np.allclose(N, _brute_cylinder(np.array([rho, 0.0, z]), 1.0, 3.0), atol=1e-6)


def test_cylinder_exact_limits():
    R, L = 1.0, 3.0
    assert S.cylinder_demag(0, 0, R, L)[0][2, 2] == pytest.approx(
        1 - (L / 2) / np.sqrt((L / 2) ** 2 + R ** 2), rel=1e-10)
    assert np.allclose(np.diag(S.cylinder_demag(0.3, 0.2, 1.0, 400.0)[0]), [0.5, 0.5, 0.0],
                       atol=2e-5)


@pytest.mark.parametrize("pt", [(0, 0, 0), (1.2, -0.8, 0.2), (-1.7, 1.2, -0.4)])
def test_cuboid_field_against_brute_force(pt):
    sides = (4.0, 3.0, 1.0)
    N = S.cuboid_demag(np.array(pt, float), sides)
    assert np.trace(N) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(N, _brute_box(np.array(pt, float), sides), atol=1e-6)


def test_cuboid_exact_limits():
    a, b, c = 2.0, 1.5, 0.5
    assert S.cuboid_demag(np.zeros(3), (2 * a, 2 * b, 2 * c))[2, 2] == pytest.approx(
        2 / np.pi * np.arctan(a * b / (c * np.sqrt(a * a + b * b + c * c))), rel=1e-12)
    assert np.allclose(np.diag(S.cuboid_demag(np.zeros(3), (1, 1, 1))), 1 / 3, atol=1e-12)


def test_cylinder_cells_converge_at_the_wall():
    """Each ring's azimuth count is enough (checked against 16384 points)."""
    R, L = 1.1, 10.0
    N = S._cylinder_cells(R, L, 16, 24).reshape(16, 24, 3, 3)
    rho = R * np.sqrt((np.arange(16) + 0.5) / 16)
    z = L * ((np.arange(24) + 0.5) / 24 - 0.5)
    for i in (0, 15):
        ref = S.cylinder_demag(np.full(24, rho[i]), z, R, L, n_phi=16384)
        assert np.allclose(N[i], ref, atol=1e-7)


def test_rotor_sample_dimensions():
    s = S.rotor_sample(3.2)
    assert (s.diameter, s.length) == S.ROTORS[3.2]
    assert s.axis_angle == pytest.approx(S.MAGIC_ANGLE_DEG)
    assert all(inner < od for od, (inner, _L) in S.ROTORS.items())
    with pytest.raises(ValueError, match="smaller"):
        S.rotor_sample(1.3, inner_diameter=1.4)
    with pytest.raises(ValueError, match="inner_diameter"):
        S.rotor_sample(5.0)
    assert S.sample_from_dict(s.to_dict()).describe() == s.describe()


# ---- the macroscopic shift ---------------------------------------------------------------

def _field(chi=(2e-3,) * 3, n=48):
    return F.bms_field(_packing(), chi, box_size=200, gridpoints=n,
                       components=("tensor", "zz", "iso"), dtype="float64")


def test_static_tube_along_b0_shifts_by_its_mean_demagnetising_factor():
    """Isotropic χ: the BMS of the packing averages to its isotropic mean
    (0), so a static tube along B₀ moves the line by its volume-averaged
    −1e6 χ_v (N_zz − 1/3)."""
    field = _field()
    tube = S.Sample("cylinder", diameter=2.0, length=10.0, axis_angle=0.0)
    sp = I.sample_spectrum(field, mode="static", sample=tube, n_samples=200000)
    chi_v = np.trace(field["mean_chi"]) / 3
    N, _w = tube.cells()
    expect = float(np.mean(-1e6 * chi_v * (N[:, 2, 2] - 1 / 3)))
    assert sp["stats"]["chi_v"] == pytest.approx(chi_v)
    assert sp["stats"]["mean_ppm"] == pytest.approx(expect, abs=0.03 * sp["stats"]["std_ppm"])
    assert 0.6 < expect / (1e6 * chi_v / 3) < 1.0             # L/D = 5: most of +χ_v/3


def test_chi_v_is_the_particles_volume_fraction_times_chi():
    field = _field()
    phi = field["volume_fraction"]
    assert np.trace(field["mean_chi"]) / 3 == pytest.approx(2e-3 * phi, rel=0.02)


def test_mas_centrebands_do_not_see_the_sample_shape():
    """tr N = 1 everywhere, so the macroscopic term has no isotropic part:
    under MAS the rotor only adds sidebands."""
    field = _field(CHI3)
    sphere = I.sample_spectrum(field, mode="mas", spin_rate_hz=20000, n_samples=20000)
    rotor = I.sample_spectrum(field, mode="mas", spin_rate_hz=20000, n_samples=20000,
                              sample=S.rotor_sample(3.2))
    for k in ("mean_ppm", "std_ppm"):
        assert rotor["stats"][k] == pytest.approx(sphere["stats"][k], rel=1e-9, abs=1e-9)


def test_sampled_mas_matches_the_exact_grid_route():
    field = _field(CHI3)
    mc = I.sample_spectrum(field, mode="mas", spin_rate_hz=10000, n_samples=60000,
                           linewidth_hz=300, n_points=4096)
    x = mc["x_ppm"]
    ex = F.bms_spectrum(field, "7Li", 9.4, mode="mas", spin_rate_hz=10000, linewidth_hz=300,
                        max_order=40, centre_ppm=0.5 * (x[0] + x[-1]),
                        sw_ppm=x[-1] - x[0] + (x[1] - x[0]), n_points=4096)
    a = mc["real_spectrum"] / mc["real_spectrum"].sum()
    b = ex["real_spectrum"] / ex["real_spectrum"].sum()
    assert np.abs(a - b).max() < 0.04 * b.max()
    assert mc["stats"]["mean_ppm"] == pytest.approx(ex["stats"]["mean_ppm"], abs=0.5)


def test_monte_carlo_noise_falls_with_samples():
    field = _field(CHI3)
    few = I.sample_spectrum(field, mode="static", n_samples=20000)
    many = I.sample_spectrum(field, mode="static", n_samples=320000)
    assert many["stats"]["mc_noise"] < 0.5 * few["stats"]["mc_noise"]
    same = I.sample_spectrum(field, mode="static", n_samples=20000)
    assert np.array_equal(same["real_spectrum"], few["real_spectrum"])     # seeded


# ---- local interactions against the combined simulator -----------------------------------

@pytest.mark.parametrize("nuc, mode, rate, tr, order, CQ, n, tol", [
    ("7Li", "mas", 8000.0, "all", "second", 0.25, 60000, 0.01),
    ("23Na", "mas", 12000.0, "ct", "second", 2.5, 60000, 0.02),
    ("6Li", "mas", 5000.0, "all", "first", 0.0, 60000, 0.01),
    ("7Li", "static", 0.0, "all", "second", 0.25, 600000, 0.04),
])
def test_local_interactions_match_the_combined_simulator(nuc, mode, rate, tr, order, CQ, n, tol):
    from opennmr.notebook.combined_hamiltonian import Site, simulate_combined
    from opennmr.notebook.utils import compute_BF1_SFO1
    field = F.bms_field(_packing(), (1e-12,) * 3, box_size=200, gridpoints=32,
                        components=("tensor",), dtype="float64")      # BMS negligible
    B0 = 9.4
    BF1, _, _ = compute_BF1_SFO1(nuc, B0)
    loc = I.local_interactions(contact_ppm=50, aniso_ppm=300, eta=0.4, euler=(30, 50, 10),
                               C_Q_MHz=CQ, eta_Q=0.3)
    mc = I.sample_spectrum(field, nuclide=nuc, B0_tesla=B0, mode=mode, spin_rate_hz=rate,
                           local=loc, transitions=tr, quad_order=order, n_samples=n,
                           linewidth_hz=300, lineshape="gaussian")       # automatic axis
    site = Site(nuc, delta_iso=50, fwhm=300 / BF1, lineshape="gaussian")
    if CQ:
        site = site.add_quadrupolar(C_Q=CQ, eta_Q=0.3)
    site = site.add_csa(aniso=300, asym=0.4, euler=(30, 50, 10))
    ref = simulate_combined([site], mc["x_ppm"], larmor_Hz=BF1 * 1e6, mas_Hz=rate, order=order,
                            transitions=tr, n_zcw=30000)
    a = mc["real_spectrum"] / np.abs(mc["real_spectrum"]).sum()
    b = ref["real_spec"] / np.abs(ref["real_spec"]).sum()
    assert np.abs(a - b).max() < tol * b.max()


def test_local_interactions_validate():
    with pytest.raises(ValueError, match="η"):
        I.local_interactions(eta=1.5)
    assert not I.has_local(I.local_interactions(contact_ppm=10))
    assert I.has_local(I.local_interactions(C_Q_MHz=1.0))


def test_isotropic_local_shift_moves_the_grid_route_rigidly():
    parts = _packing()
    a, _ = F.fast_bms_spectrum(parts, CHI3, mode="mas", box_size=200, gridpoints=32)
    b, _ = F.fast_bms_spectrum(parts, CHI3, mode="mas", box_size=200, gridpoints=32,
                               local=I.local_interactions(contact_ppm=120.0))
    assert b["route"] == a["route"] == "grid"
    assert np.allclose(b["x_ppm"] - a["x_ppm"], 120.0)
    assert np.array_equal(b["real_spectrum"], a["real_spectrum"])


def test_routes():
    assert F.needs_sampling(None, None, "static")
    assert not F.needs_sampling(None, None, "mas")
    assert F.needs_sampling(S.rotor_sample(4.0).to_dict(), None, "mas")
    assert not F.needs_sampling(S.Sample("sphere"), None, "mas_infinite")
    assert F.needs_sampling(None, I.local_interactions(aniso_ppm=10), "mas")
    assert F.route("mas", None, None) == (False, ("tensor", "iso"))


def test_one_field_many_spectra():
    field = F.bms_field(_packing(), CHI3, box_size=200, gridpoints=32,
                        components=("tensor", "zz", "iso"))
    s1 = F.spectrum_from_field(field, mode="mas", spin_rate_hz=10000)
    s2 = F.spectrum_from_field(field, mode="static", n_samples=20000, sample=S.rotor_sample(4.0))
    assert s1["route"] == "grid" and s2["route"] == "sampled"
    bare = F.bms_field(_packing(), CHI3, box_size=200, gridpoints=32, components=("iso",))
    with pytest.raises(ValueError, match="lacks"):
        F.spectrum_from_field(bare, mode="static")


# ---- pictures and estimates --------------------------------------------------------------

@pytest.mark.parametrize("eq, r, t, dims", [("ellipsoid", 2, 2, (3.0, 2.0, 1.5)),
                                            ("ellipsoid", 4, 2, (3.0, 2.0, 1.5)),
                                            ("ellipsoid", 2, 6, (3.0, 2.0, 1.5)),
                                            ("hexagonal", 2, 2, (2.0, 0.0, 3.0))])
def test_particle_volume_by_monte_carlo(eq, r, t, dims):
    p = {"Equation": eq, "v1": dims[0], "v2": dims[1], "v3": dims[2], "r": r, "t": t,
         "Deformation": "C"}
    P = np.random.default_rng(0).uniform(-1, 1, (2_000_000, 3))
    if eq == "hexagonal":
        box = np.array([3.0, 3.0, 1.6])
        a, b, c = (P * box).T
        d = np.maximum.reduce([abs(a), abs(a / 2 + np.sqrt(3) / 2 * b),
                               abs(a / 2 - np.sqrt(3) / 2 * b)]) / dims[0]
        inside = (d <= 1) & (abs(c) <= dims[2] / 2)
    else:
        box = np.array(dims)
        Q = P * box
        inside = ((np.abs(Q[:, 0] / dims[0]) ** r + np.abs(Q[:, 1] / dims[1]) ** r) ** (t / r)
                  + np.abs(Q[:, 2] / dims[2]) ** t) <= 1
    assert V.particle_volume(p) == pytest.approx(inside.mean() * 8 * np.prod(box), rel=0.005)


@pytest.mark.parametrize("eq", ["ellipsoid", "hexagonal", "donut"])
def test_packing_3d(eq):
    import matplotlib
    matplotlib.use("Agg")
    parts = [make_particle(-30, 0, 0, (15, 12, 8), quaternion=(0.9, 0.3, 0.1, 0.2)),
             make_particle(30, 0, 0, (10, 6, 8))]
    fig = V.plot_packing_3d(parts, box_size=120, equation=eq, deformation="C", unit="µm")
    ax = fig.axes[0]
    assert ax.name == "3d" and "$" in ax.get_xlabel()
    assert len(fig.axes) == 2                                      # the colour bar


def test_run_time_estimates():
    parts = _packing()
    est = V.estimate_field_time(parts, CHI3, box_size=200, gridpoints=64,
                                components=("tensor", "zz", "iso"))
    assert est["seconds"] > 0 and est["memory_GB"] > 0 and est["text"].startswith("about")
    big = V.estimate_field_time(parts, CHI3, box_size=200, gridpoints=256,
                                components=("tensor", "zz", "iso"))
    assert big["seconds"] > 10 * est["seconds"] and big["memory_GB"] > 32 * est["memory_GB"]
    field = F.bms_field(parts, CHI3, box_size=200, gridpoints=32, components=("tensor",))
    s = V.estimate_sampling_time(field, 100000, mode="mas", spin_rate_hz=10000)
    assert s["seconds"] > 0 and s["per_sample_s"] > 0


# ---- the rotor's own field (explicit user request, 2026-11-07) ---------------------------

@pytest.mark.parametrize("pt", [(0.5, 0.0, 2.3), (0.2, 0.0, -3.0), (1.6, 0.0, 0.4)])
def test_cylinder_field_outside_against_brute_force(pt):
    N = S.cylinder_demag(pt[0], pt[2], 1.0, 4.0)[0]
    assert np.trace(N) == pytest.approx(0.0, abs=1e-12)          # outside: no source
    assert np.allclose(N, _brute_cylinder(np.array(pt, float), 1.0, 4.0), atol=1e-9)


def test_cylinder_far_field_is_a_dipole():
    V, z = np.pi * 4.0, 60.0                                      # R = 1, L = 4
    assert S.cylinder_demag(0.0, z, 1.0, 4.0)[0][2, 2] == pytest.approx(
        -2 * V / (4 * np.pi * z ** 3), rel=0.005)


def test_a_long_rotor_wall_has_no_field_in_its_bore():
    s = S.Sample("cylinder", diameter=2.2, length=10.0,
                 rotor={"outer_diameter": 3.2, "length": 400.0, "chi_wall": -8.5e-6,
                        "chi_caps": 0.0})
    # end effects of a 400-long tube ~ (R/L)²: N_wall < 1e-4 in the bore
    assert np.abs(s.rotor_cells()).max() < 1e-4 * 8.5e-6


def test_rotor_cells_equal_the_direct_sum():
    s = S.rotor_sample(3.2)
    M = s.rotor_cells().reshape(48, 96, 3, 3)
    R, L = 1.1, 10.0
    Ro, Lr = 1.6, s.rotor["length"]
    Lc = (Lr - L) / 2
    zc = (L + Lc) / 2
    rho = R * np.sqrt((np.arange(48) + 0.5) / 48)
    z = L * ((np.arange(96) + 0.5) / 96 - 0.5)
    for i in (0, 47):
        r = np.full(96, rho[i])
        ref = s.rotor["chi_wall"] * (S.cylinder_demag(r, z, Ro, Lr, n_phi=16384)
                                     - S.cylinder_demag(r, z, R, Lr, n_phi=16384))
        ref += s.rotor["chi_caps"] * (S.cylinder_demag(r, z - zc, R, Lc, n_phi=16384)
                                      + S.cylinder_demag(r, z + zc, R, Lc, n_phi=16384))
        assert np.allclose(M[i], ref, atol=1e-12 * 1e-5)
    assert np.abs(np.trace(M, axis1=2, axis2=3)).max() < 1e-15       # traceless


def test_rotor_shifts_the_static_line_and_not_the_mas_centrebands():
    field = _field()
    bare = S.Sample("cylinder", diameter=2.2, length=10.0, axis_angle=0.0)
    rotor = S.rotor_sample(3.2, axis_angle=0.0)
    a = I.sample_spectrum(field, mode="static", sample=bare, n_samples=200000)
    b = I.sample_spectrum(field, mode="static", sample=rotor, n_samples=200000)
    Mr = rotor.rotor_cells()
    expect = float(np.mean(-1e6 * Mr[:, 2, 2]))                    # B0 along the rotor axis
    assert b["stats"]["mean_ppm"] - a["stats"]["mean_ppm"] == pytest.approx(expect, abs=0.05)
    assert abs(expect) > 0.1
    m1 = I.sample_spectrum(field, mode="mas", spin_rate_hz=20000, n_samples=20000, sample=bare)
    m2 = I.sample_spectrum(field, mode="mas", spin_rate_hz=20000, n_samples=20000, sample=rotor)
    assert m2["stats"]["mean_ppm"] == pytest.approx(m1["stats"]["mean_ppm"], abs=1e-9)
    assert "rotor 3.2" in rotor.describe()
    assert S.sample_from_dict(rotor.to_dict()).rotor == rotor.rotor


def test_rotor_materials_and_checks():
    assert S.material_chi("zirconia") == pytest.approx(-8.5e-6)
    assert S.material_chi("vespel") == pytest.approx(-9.02e-6)
    assert S.material_chi(-1e-6) == -1e-6 and S.material_chi(None) == 0.0
    with pytest.raises(ValueError, match="Unknown material"):
        S.material_chi("unobtainium")
    assert S.rotor_sample(4.0, wall=None, caps=None).rotor is None
    with pytest.raises(ValueError, match="outer diameter"):
        S.Sample("cylinder", diameter=2.2, length=10, rotor={"outer_diameter": 2.0, "length": 20})
    with pytest.raises(ValueError, match="at least as long"):
        S.Sample("cylinder", diameter=2.2, length=10, rotor={"outer_diameter": 3.2, "length": 5})


# ---- the spectrum axis resolves the line width (found 2026-11-07) ------------------------

def test_automatic_axis_resolves_the_line():
    n, under = F.resolve_points(1000.0, 0.5)
    assert 1000.0 / n <= 0.5 / F.POINTS_PER_FWHM and not under
    assert F.resolve_points(2546.0, 1.93, 2048)[1]                  # 1.6 points per FWHM
    assert F.resolve_points(1e9, 1e-6)[0] == F.MAX_POINTS
    field = _field(CHI3)
    sp = I.sample_spectrum(field, mode="static", n_samples=20000, linewidth_hz=30.0)
    dx = sp["x_ppm"][1] - sp["x_ppm"][0]
    assert dx <= 30.0 / sp["BF1"] / F.POINTS_PER_FWHM * 1.0001 or sp["stats"]["n_points"] == F.MAX_POINTS
