"""Fast BMS shifts by superposition (explicit user request, 2026-11-07):
opennmr.notebook.bms.fast -- against analytic fields, the original
run_bms, and the sum rules of MAS sidebands."""

import numpy as np
import pytest
from scipy.integrate import quad

from opennmr.notebook.bms import fast as F
from opennmr.notebook.bms import run_bms
from opennmr.notebook.bms.geometry import shape_function
from opennmr.notebook.bms.io_utils import make_particle
from opennmr.notebook.bms.mas_sidebands import sideband_manifold

CHI3 = (2.87e-3, 3.13e-3, 2.27e-3)


@pytest.fixture(autouse=True)
def _sideband_cache(tmp_path, monkeypatch):
    """Never write into the user's own cache from a test."""
    monkeypatch.setenv("OPENNMR_BMS_CACHE", str(tmp_path / "sidebands"))
    F._ROWS.clear()
    yield
    F._ROWS.clear()


def _demag(a):
    """Demagnetising factors of an ellipsoid with semi-axes a."""
    a = np.asarray(a, float)

    def N(i):
        return np.prod(a) / 2 * quad(lambda s: 1 / ((a[i] ** 2 + s) * np.sqrt(np.prod(a ** 2 + s))),
                                     0, np.inf)[0]
    return np.array([N(i) for i in range(3)])


def _centre(f):
    return tuple(int(np.argmin(np.abs(f[ax]))) for ax in "xyz")


# ---- shapes -----------------------------------------------------------------------

@pytest.mark.parametrize("equation", ["ellipsoid", "hexagonal", "donut"])
@pytest.mark.parametrize("deformation", ["A", "C"])
@pytest.mark.parametrize("slope, r, t", [(2000, 2, 2), (8, 4, 2)])
def test_block_shape_equals_the_original_shape_function(equation, deformation, slope, r, t):
    p = F.prepare_particles([make_particle(0, 0, 0, (20, 12, 15), quaternion=(0.3, -0.5, 0.7, 0.2))],
                            (1e-3,) * 3, equation=equation, deformation=deformation, r=r, t=t,
                            slope_filter=slope)[0]
    xs, ys, zs = np.linspace(-30, 30, 23), np.linspace(-29, 31, 21), np.linspace(-28, 27, 19)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    assert np.allclose(F.block_shape(p, xs, ys, zs), shape_function(p, X, Y, Z), atol=1e-11)


def test_soft_edges_are_rasterised_beyond_the_bounding_radius():
    """A small slope filter reaches far outside the particle: the block
    must cover it (the occupancy matches the shape on the whole grid)."""
    L, n = 120.0, 32
    parts = [make_particle(0, 0, 0, (15, 15, 15))]
    f = F.bms_field(parts, (1e-3,) * 3, box_size=L, gridpoints=n, slope_filter=4,
                    components=("iso",), dtype="float64")
    p = F.prepare_particles(parts, (1e-3,) * 3, slope_filter=4)[0]
    X, Y, Z = np.meshgrid(f["x"], f["y"], f["z"], indexing="ij")
    assert np.allclose(f["occupancy"], np.clip(shape_function(p, X, Y, Z), 0, 1), atol=1e-6)


# ---- the field against analytic results ----------------------------------------------

def test_sphere_inside_zero_outside_point_dipole():
    chi, a, L, n = 1e-3, 20.0, 160.0, 96
    f = F.bms_field([make_particle(0, 0, 0, (a, a, a))], (chi,) * 3, box_size=L, gridpoints=n,
                    boundary="isolated", components=("zz", "iso"), dtype="float64")
    i = _centre(f)
    assert abs(f["delta_zz"][i]) < 1e-6 and abs(f["delta_iso"][i]) < 1e-6
    V = f["occupancy"].sum() * (L / n) ** 3            # the sphere as voxelised
    x = f["x"]
    for r in (50.0, 70.0):
        k = int(np.argmin(np.abs(x - r)))
        dip = 1e6 * chi * V / (4 * np.pi * x[k] ** 3)
        # B₀ along z: + 2× on the axis (above the pole), − 1× on the equator
        assert f["delta_zz"][i[0], i[1], k] == pytest.approx(2 * dip, rel=0.01)
        assert f["delta_zz"][k, i[1], i[2]] == pytest.approx(-dip, rel=0.01)


def test_spheroid_inside_is_its_demagnetising_factor():
    chi, a, c = 1e-3, 15.0, 40.0
    f = F.bms_field([make_particle(0, 0, 0, (a, a, c))], (chi,) * 3, box_size=200, gridpoints=96,
                    boundary="isolated", components=("zz",), dtype="float64")
    Nz = _demag((a, a, c))[2]
    assert f["delta_zz"][_centre(f)] == pytest.approx(1e6 * chi * (1 / 3 - Nz), rel=0.015)


def test_anisotropic_chi_inside_an_ellipsoid_and_translation():
    """δ_iso inside a uniformly magnetised ellipsoid with diagonal χ is
    −(1e6/3) Σ χ_i (N_i − 1/3), wherever the particle sits relative to the
    grid."""
    a = (12.0, 9.0, 7.0)
    expect = -1e6 / 3 * np.sum(np.array(CHI3) * (_demag(a) - 1 / 3))
    step = 100.0 / 96
    for dx in (0.0, step / 4, step / 2):
        f = F.bms_field([make_particle(dx, 0, 0, a)], CHI3, box_size=100, gridpoints=96,
                        components=("iso",), dtype="float64")
        occ = f["occupancy"]
        mean = (f["delta_iso"] * occ).sum() / occ.sum()
        assert mean == pytest.approx(expect, rel=0.04)


def test_isotropic_chi_has_no_isotropic_shift():
    """With an isotropic χ, tr(χN) = χ s, and the (proportional) Lorentz
    term removes it: MAS averages BMS shifts away completely."""
    parts = [make_particle(-20, 5, 0, (12, 9, 7), quaternion=(0.9, 0.3, 0.2, 0.1)),
             make_particle(25, -10, 8, (10, 10, 14), quaternion=(0.5, 0.5, 0.5, 0.5))]
    f = F.bms_field(parts, (2e-3,) * 3, box_size=100, gridpoints=48, components=("iso",),
                    dtype="float64")
    assert f["isotropic_chi"]
    assert np.abs(f["delta_iso"]).max() < 1e-4
    g = F.bms_field(parts, (2e-3,) * 3, box_size=100, gridpoints=48, components=("iso",),
                    dtype="float64", lorentz="threshold")
    assert np.abs(g["delta_iso"]).max() > 1.0          # the original's surface-voxel jumps


# ---- linearity, code paths, options ---------------------------------------------------

def _two():
    return [make_particle(-20, 5, 0, (12, 9, 7), quaternion=(0.9, 0.3, 0.2, 0.1)),
            make_particle(25, -10, 8, (10, 10, 14), quaternion=(0.5, 0.5, 0.5, 0.5))]


def test_superposition_is_exact():
    kw = dict(box_size=100, gridpoints=40, components=("zz", "iso", "tensor"), dtype="float64")
    p1, p2 = _two()
    fa, fb, fab = (F.bms_field(p, (2e-3, 3e-3, 1e-3), **kw) for p in ([p1], [p2], [p1, p2]))
    for key in ("delta_zz", "delta_iso"):
        assert np.allclose(fab[key], fa[key] + fb[key], atol=1e-8)
    for ij in fab["delta_tensor"]:
        assert np.allclose(fab["delta_tensor"][ij], fa["delta_tensor"][ij] + fb["delta_tensor"][ij],
                           atol=1e-8)


def test_isotropic_shortcut_equals_the_tensor_route():
    kw = dict(box_size=100, gridpoints=40, components=("zz", "iso", "tensor"), dtype="float64")
    f_iso = F.bms_field(_two(), (2e-3,) * 3, **kw)
    f_ten = F.bms_field(_two(), (2e-3, 2e-3, 2e-3 * (1 + 1e-9)), **kw)
    assert f_iso["isotropic_chi"] and not f_ten["isotropic_chi"]
    assert np.allclose(f_iso["delta_zz"], f_ten["delta_zz"], atol=1e-5)
    for ij in f_iso["delta_tensor"]:
        assert np.allclose(f_iso["delta_tensor"][ij], f_ten["delta_tensor"][ij], atol=1e-5)


def test_single_precision_is_ample():
    kw = dict(box_size=100, gridpoints=40, components=("zz",))
    f32 = F.bms_field(_two(), CHI3, dtype="float32", **kw)
    f64 = F.bms_field(_two(), CHI3, dtype="float64", **kw)
    assert np.abs(f32["delta_zz"] - f64["delta_zz"]).max() < 1e-5 * np.abs(f64["delta_zz"]).max()


def test_sample_shape_is_a_uniform_bulk_shift():
    """Periodic: a long cylinder along B₀ (D_zz = 0) shifts every voxel by
    +1e6 ⟨χ⟩ (1/3 − 0) relative to a sphere (D_zz = 1/3)."""
    kw = dict(box_size=100, gridpoints=40, components=("zz",), dtype="float64")
    fs = F.bms_field(_two(), (2e-3,) * 3, sample_shape="sphere", **kw)
    fc = F.bms_field(_two(), (2e-3,) * 3, sample_shape="cylinder_parallel", **kw)
    diff = fc["delta_zz"] - fs["delta_zz"]
    assert np.ptp(diff) < 1e-8
    assert diff.mean() == pytest.approx(1e6 * 2e-3 * fs["volume_fraction"] / 3, rel=1e-6)


def test_periodic_tends_to_isolated_in_a_big_box():
    chi = 1e-3
    out = []
    for L, n in ((100.0, 64), (200.0, 128)):
        f = F.bms_field([make_particle(0, 0, 0, (10, 10, 10))], (chi,) * 3, box_size=L,
                        gridpoints=n, components=("zz",), dtype="float64")
        i = _centre(f)
        k = int(np.argmin(np.abs(f["x"] - 25)))
        V = f["occupancy"].sum() * (L / n) ** 3
        out.append(f["delta_zz"][i[0], i[1], k] / (1e6 * chi * V / (4 * np.pi * f["x"][k] ** 3) * 2))
    assert abs(out[1] - 1) < abs(out[0] - 1) and abs(out[1] - 1) < 0.01


def test_matches_the_original_for_a_particle_on_a_grid_node():
    L, n = 100.0, 32
    parts = [make_particle(0, 0, 0, (18, 14, 10))]
    df = run_bms(parts, CHI3, box_size=L, gridpoints=n, equation="ellipsoid", n_jobs=1)
    old = df["d_iso"].to_numpy().reshape(n + 1, n + 1, n + 1)[:n, :n, :n]
    f = F.bms_field(parts, CHI3, box_size=L, gridpoints=n, boundary="isolated",
                    components=("iso",), dtype="float64", lorentz="threshold")
    i = _centre(f)
    assert f["delta_iso"][i] == pytest.approx(old[i], abs=0.2)
    assert np.corrcoef(f["delta_iso"].ravel(), old.ravel())[0, 1] > 0.99


def test_bad_inputs_say_what_is_wrong():
    with pytest.raises(ValueError, match="three principal"):
        F.bms_field(_two(), (1e-3, 2e-3), gridpoints=16)
    with pytest.raises(ValueError, match="boundary"):
        F.bms_field(_two(), CHI3, gridpoints=16, boundary="mirror")
    with pytest.raises(ValueError, match="trace 1"):
        F.bms_field(_two(), CHI3, gridpoints=16, sample_shape=np.eye(3))
    with pytest.raises(ValueError, match="components"):
        F.bms_field(_two(), CHI3, gridpoints=16, components=("xx",))


# ---- MAS sidebands --------------------------------------------------------------------

@pytest.mark.parametrize("mu, eta", [(0.5, 0.0), (2.0, 0.3), (5.0, 1.0)])
def test_sidebands_match_the_original_numerical_manifold(mu, eta):
    o, I = F.mas_sideband_intensities(mu, eta, n_beta=24, n_gamma=24, n_t=256)
    o2, I2 = sideband_manifold(mu, eta, 1.0, n_orient_beta=24, n_orient_gamma=24, n_t=512)
    ref = dict(zip(o2, I2))
    got = dict(zip(o, I))
    for k in range(-4, 5):
        assert got[k] == pytest.approx(ref[k], abs=1e-4)


@pytest.mark.parametrize("mu, eta", [(3.0, 0.0), (-3.0, 0.0), (3.0, 0.5), (12.0, 0.8)])
def test_sideband_moments_equal_the_static_powder(mu, eta):
    n = F._orientations(mu)
    o, I = F.mas_sideband_intensities(mu, eta, n_beta=n, n_gamma=n, n_t=256)
    assert I.sum() == pytest.approx(1.0)
    assert (o * I).sum() == pytest.approx(0.0, abs=1e-10)
    assert (o ** 2 * I).sum() == pytest.approx(mu ** 2 * (1 + eta ** 2 / 3) / 5, rel=1e-8)
    cb, wb = np.polynomial.legendre.leggauss(200)
    ph = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    B, P = np.meshgrid(np.arccos(cb), ph, indexing="ij")
    W = np.repeat(wb, 200).reshape(B.shape)
    w = mu * (3 * np.cos(B) ** 2 - 1 + eta * np.sin(B) ** 2 * np.cos(2 * P)) / 2
    assert (o ** 3 * I).sum() == pytest.approx((W * w ** 3).sum() / W.sum(), rel=1e-6)


def test_negative_anisotropy_mirrors_the_pattern():
    o, Ip = F.mas_sideband_intensities(2.0, 0.3)
    _, Im = F.mas_sideband_intensities(-2.0, 0.3)
    d = dict(zip(o, Im))
    for k, v in zip(o, Ip):
        if -k in d:                     # the FFT's −n_t/2 order has no +n_t/2 partner
            assert d[-k] == pytest.approx(v, abs=1e-12)


def test_the_table_covers_the_data_and_is_cached_on_disk(tmp_path):
    for need in (0.01, 0.5, 10.0):
        mu, eta, orders, table = F._manifold(8, need)
        assert mu.max() >= need and mu.min() <= -need
        assert np.array_equal(orders, np.arange(-8, 9))
    files = sorted((tmp_path / "sidebands").glob("row_*.npy"))
    assert files
    i = int(files[0].stem.split("_")[1])
    row = F._ROWS[i]
    F._ROWS.clear()
    assert np.array_equal(F._row(i), row)               # read back, not recomputed


# ---- spectra --------------------------------------------------------------------------

def _packing():
    from opennmr.notebook.bms.packing import grow_and_separate_pack
    parts, _ = grow_and_separate_pack(20, 200.0, (22, 22, 22), (3, 3, 3), seed=4, verbose=False)
    return parts


def test_static_spectrum_keeps_the_intensity_and_the_mean():
    spec, field = F.fast_bms_spectrum(_packing(), CHI3, "7Li", 9.4, mode="static", box_size=200,
                                      gridpoints=48, linewidth_hz=200.0, n_points=4096)
    st = spec["stats"]
    assert st["outside_window"] < 1e-3
    x, y = spec["x_ppm"], spec["real_spectrum"]
    assert y.sum() == pytest.approx(st["weight"] * (1 - st["outside_window"]), rel=1e-6)
    assert (x * y).sum() / y.sum() == pytest.approx(st["mean_ppm"], abs=0.02 * st["std_ppm"] + 0.5)
    assert set(("x_ppm", "real_spectrum", "BF1", "SFO1", "nuclide")) <= set(spec)
    assert "delta_zz" in field and "delta_tensor" not in field   # only what the mode needs


def test_mas_spectrum_has_sidebands_at_the_spinning_rate():
    rate = 5000.0
    chi = tuple(0.2 * np.array(CHI3))      # resolved sidebands (spacing ≫ isotropic width)
    spec, field = F.fast_bms_spectrum(_packing(), chi, "7Li", 9.4, mode="mas", box_size=200,
                                      gridpoints=48, spin_rate_hz=rate, linewidth_hz=20.0,
                                      n_points=16384, max_order=16)
    st = spec["stats"]
    assert st["sidebands_beyond_max_order"] < 1e-4 and st["beyond_mu_grid"] == 0.0
    x, y = spec["x_ppm"], spec["real_spectrum"]
    centre = st["mean_ppm"]
    step = rate / spec["BF1"]
    for k in (-1, 1):                               # a local maximum near each first sideband
        near = np.abs(x - (centre + k * step)) < 0.25 * step
        between = np.abs(x - (centre + k * step / 2)) < 0.05 * step
        assert y[near].max() > 10 * y[between].max()
    inf, _ = F.fast_bms_spectrum(_packing(), chi, "7Li", 9.4, mode="mas_infinite", box_size=200,
                                 gridpoints=48)
    # same isotropic shifts (one from the trace, one from the tensor's eigenvalues; float32)
    assert inf["stats"]["mean_ppm"] == pytest.approx(st["mean_ppm"], abs=1e-3)


@pytest.mark.parametrize("observe", ["inside", "outside", "all"])
def test_observed_weights(observe):
    _, field = F.fast_bms_spectrum(_packing(), CHI3, mode="static", box_size=200, gridpoints=32)
    w = F.observed_weights(field, observe)
    occ = field["occupancy"]
    expect = {"inside": occ, "outside": 1 - occ, "all": np.ones_like(occ)}[observe]
    assert np.allclose(w, expect)


def test_slice_plot_and_python():
    import matplotlib
    matplotlib.use("Agg")
    _, field = F.fast_bms_spectrum(_packing(), CHI3, mode="static", box_size=200, gridpoints=32)
    fig = F.plot_bms_slice(field)
    assert fig.axes and "$" in fig.axes[0].get_xlabel()          # italic axis symbols
