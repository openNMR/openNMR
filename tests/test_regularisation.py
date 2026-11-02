"""notebook.regularisation: the MATLAB_RELAX_ALGORITHM scripts in Python.

The reference numbers were produced by the original MATLAB files (csvd.m,
tikhonov.m, errestik.m, etanutik.m, nnls10.m), run in GNU Octave 8.4 on the
data below, with the scripts' own options (errestik(A, y, I, [], 3,
[1e-6 2000 1], U, s, V), then |x|-prior Tikhonov iterations).
"""
import numpy as np
import pytest

import opennmr.notebook as nmr
from opennmr.notebook import regularisation as reg


def _cpmg(n=120, noise=0.005, seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(1e-4, 0.5, n)
    y = 0.4 * np.exp(-t / 0.005) + 0.6 * np.exp(-t / 0.08) + rng.normal(0, noise, t.size)
    return t, y


def _octave_problem():
    t, y = _cpmg()
    # The file Octave read was written with %.17g: the same doubles.
    t = np.array([float(f"{v:.17g}") for v in t])
    y = np.array([float(f"{v:.17g}") for v in y])
    grid = np.logspace(-4, 1, 60)
    A = np.hstack([reg.kernel_matrix("t2", t, grid), np.ones((t.size, 1))])
    return t, y, A


def test_errestik_matches_the_matlab_code():
    t, y, A = _octave_problem()
    U, s, V = reg.csvd(A)
    scan = reg.errestik(A, y, U, s, V)
    assert scan["lambda"] == pytest.approx(2.3739214616367033, rel=1e-9)
    assert scan["evaluations"] == 49
    assert scan["x"].sum() == pytest.approx(0.9619286101856818, rel=1e-9)


def test_iterated_solution_and_nnls_match_the_matlab_code():
    t, y, _A = _octave_problem()
    result = reg.regularise(t, y, "t2", (1e-4, 10.0), 60, n_iter=50)
    x = np.concatenate([result["amplitudes"], [result["offset"]]])
    assert x.sum() == pytest.approx(1.005380222295235, rel=1e-9)
    assert x.max() == pytest.approx(0.10983891569716177, rel=1e-9)
    assert int(np.argmax(x)) == 34
    assert result["offset"] == pytest.approx(-0.0003251406687643951, rel=1e-6)
    assert result["nnls_amplitudes"].sum() + result["nnls_offset"] == pytest.approx(
        1.0025541517888295, rel=1e-6)


def test_tikhonov_with_and_without_a_prior():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 8))
    b = rng.normal(size=30)
    U, s, V = reg.csvd(A)
    lam = 0.7
    x, rho, eta = reg.tikhonov(U, s, V, b, lam)
    direct = np.linalg.solve(A.T @ A + lam ** 2 * np.eye(8), A.T @ b)
    np.testing.assert_allclose(x, direct, rtol=1e-10)
    assert rho == pytest.approx(np.linalg.norm(A @ x - b), rel=1e-10)
    assert eta == pytest.approx(np.linalg.norm(x), rel=1e-12)
    x0 = rng.normal(size=8)
    xp, _r, _e = reg.tikhonov(U, s, V, b, lam, x0)
    direct = np.linalg.solve(A.T @ A + lam ** 2 * np.eye(8), A.T @ b + lam ** 2 * x0)
    np.testing.assert_allclose(xp, direct, rtol=1e-10)


def test_two_t2_populations_are_recovered():
    t, y = _cpmg(n=300)
    result = reg.regularise(t, y, "t2", (1e-4, 10.0), 100)
    positions = sorted(p["position"] for p in result["peaks"][:2])
    assert positions[0] == pytest.approx(0.005, rel=0.15)
    assert positions[1] == pytest.approx(0.08, rel=0.15)
    shares = {round(p["position"], 3): p["share"] for p in result["peaks"][:2]}
    assert max(shares.values()) == pytest.approx(0.6, abs=0.05)
    assert result["log_mean"] == pytest.approx(np.exp(0.4 * np.log(0.005) + 0.6 * np.log(0.08)),
                                               rel=0.15)
    assert result["rms_residual"] < 0.006
    # the cumulative distribution ends at the total amplitude
    assert result["cumulative"][-1] == pytest.approx(result["amplitudes"].sum())


def test_nonnegative_solver_and_the_few_points_warning():
    rng = np.random.default_rng(1)
    t = np.logspace(-3, 1, 16)
    y = 0.7 * (1 - 2 * np.exp(-t / 0.05)) + 0.3 * (1 - 2 * np.exp(-t / 0.8)) \
        + rng.normal(0, 0.01, t.size)
    matlab = reg.regularise(t, y, "t1_ir", (1e-4, 10.0), 60)
    # 16 points, 61 unknowns: the error estimate's minimum is round-off.
    assert any("effectively zero" in n for n in matlab["notes"])
    nonneg = reg.regularise(t, y, "t1_ir", (1e-4, 10.0), 60, solver="nonnegative")
    assert np.all(nonneg["amplitudes"] >= 0)
    top = sorted(nonneg["peaks"], key=lambda p: p["position"])
    assert top[0]["position"] == pytest.approx(0.05, rel=0.3)
    assert top[-1]["position"] == pytest.approx(0.8, rel=0.4)
    assert top[0]["share"] == pytest.approx(0.7, abs=0.1)


def test_offset_may_be_negative_in_the_nonnegative_solver():
    t, y = _cpmg(n=200)
    result = reg.regularise(t, y - 0.05, "t2", (1e-4, 10.0), 60, solver="nonnegative")
    assert result["offset"] == pytest.approx(-0.05, abs=0.01)


def test_saturation_recovery_and_diffusion_kernels():
    t = np.logspace(-3, 1, 40)
    y = 1 - np.exp(-t / 0.3)
    result = reg.regularise(t, y, "t1_sr", (1e-3, 10.0), 60, solver="nonnegative", lam=1e-3)
    assert result["peaks"][0]["position"] == pytest.approx(0.3, rel=0.15)
    b = np.linspace(0, 2e9, 30)
    y = np.exp(-b * 2e-9)
    result = reg.regularise(b, y, "diffusion", (1e-11, 1e-8), 60, solver="nonnegative", lam=1e-3)
    assert result["peaks"][0]["position"] == pytest.approx(2e-9, rel=0.15)


def test_static_gradient_b_and_grid_range():
    gamma = 2 * np.pi * 42.577478e6
    assert reg.static_gradient_b(1e-3, 0.5) == pytest.approx(gamma ** 2 * 0.25 * 8e-9 / 12,
                                                              rel=1e-5)
    assert reg.suggest_range("t2", [0, 0.003, 0.3]) == pytest.approx((0.001, 0.9))
    lo, hi = reg.suggest_range("diffusion", [0, 1e8, 1e10])
    assert lo == pytest.approx(1 / 3e10) and hi == pytest.approx(3e-8)


def test_2d_t1_t2_map_finds_both_populations():
    rng = np.random.default_rng(2)
    te = np.linspace(2e-4, 0.3, 150)
    vd = np.logspace(-3, 1, 14)
    data = np.zeros((vd.size, te.size))
    for a, T1, T2 in ((0.5, 0.05, 0.01), (0.5, 1.0, 0.1)):
        data += a * np.outer(1 - 2 * np.exp(-vd / T1), np.exp(-te / T2))
    data += rng.normal(0, 0.002, data.shape)
    x1, thinned = reg.subsample(te, data, keep_first=20, step=3)
    assert thinned.shape == (14, 20 + len(range(20, 150, 3)))
    result = reg.regularise_2d(x1, vd, thinned, "t2", "t1_ir", (1e-4, 1.0), (1e-3, 10.0), 25, 25)
    assert result["map"].shape == (25, 25)
    assert result["log_mean1"] == pytest.approx(np.sqrt(0.01 * 0.1), rel=0.2)
    assert result["log_mean2"] == pytest.approx(np.sqrt(0.05 * 1.0), rel=0.25)
    # each population is a maximum near its (T2, T1)
    fmap = result["map"]
    g1, g2 = result["grid1"], result["grid2"]
    for T1, T2 in ((0.05, 0.01), (1.0, 0.1)):
        i = np.argmin(abs(np.log(g2) - np.log(T1)))
        j = np.argmin(abs(np.log(g1) - np.log(T2)))
        window = fmap[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4]
        assert window.max() > 0.3 * fmap.max()
    # projections restricted to a T1 window keep only that population
    p = reg.projections(result, within2=(0.01, 0.2))
    assert reg.log_mean(g1, p["projection1"]) == pytest.approx(0.01, rel=0.35)


def test_2d_size_guard():
    with pytest.raises(ValueError, match="too large"):
        reg.regularise_2d(np.linspace(0, 1, 4000), np.linspace(0.01, 1, 200),
                          np.zeros((200, 4000)), n1=60, n2=60)


def test_report_csv_and_plots(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    t, y = _cpmg()
    result = reg.regularise(t, y, "t2", (1e-4, 10.0), 50)
    text = reg.report(result, "ms")
    assert "Log mean T₂" in text and "λ =" in text
    path = tmp_path / "r.csv"
    reg.to_csv(result, path)
    body = path.read_text(encoding="utf-8")
    assert "amplitude" in body and "residual" in body
    fig = reg.plot_regularisation(result, show_nnls=True)
    assert len(fig.axes) == 4
    fig = reg.plot_regularisation(result, show_cumulative=False, show_residuals=False)
    assert len(fig.axes) == 2


def test_notebook_names():
    assert nmr.regularise is reg.regularise
    assert nmr.regularise_2d is reg.regularise_2d
    assert nmr.regularisation is reg
    assert nmr.regularisation_report is reg.report


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        reg.regularise([1, 2], [1, 2])
    with pytest.raises(ValueError):
        reg.regularise([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        reg.regularise([1, 2, 3], [1, np.nan, 2])
    with pytest.raises(ValueError):
        reg.log_grid(1.0, 0.5, 10)
    with pytest.raises(ValueError):
        reg.regularise([1, 2, 3], [3, 2, 1], kernel="nope")


@pytest.mark.parametrize("kernel", ["t1_ir", "t1_sr", "t2"])
def test_regularisation_figure_has_no_unicode_subscripts_in_any_text(kernel):
    """Arial has no glyph for the Unicode subscripts (U+2080-2089), so "T₁" must be
    drawn as mathtext: matplotlib warned "Glyph 8321 missing from font(s) Arial"
    at tight_layout (2026-11-02). Every text on the figure is checked; the layout
    call itself is left to do its job of keeping the export uncut."""
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.linspace(1e-3, 3.0, 60)
    y = (1 - np.exp(-t / 0.4)) if kernel == "t1_sr" else (
        1 - 2 * np.exp(-t / 0.4)) if kernel == "t1_ir" else np.exp(-t / 0.4)
    result = reg.regularise(t, y, kernel=kernel)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig = reg.plot_regularisation(result, show_nnls=True)
        fig.tight_layout()                  # still called, and still warning-free
        fig.canvas.draw()
    texts = [a.get_title() for a in fig.axes] + [a.get_xlabel() for a in fig.axes] \
        + [a.get_ylabel() for a in fig.axes]
    for a in fig.axes:
        leg = a.get_legend()
        if leg is not None:
            texts += [t_.get_text() for t_ in leg.get_texts()]
    bad = [x for x in texts if any("₀" <= c <= "₉" for c in x)]
    assert not bad, bad
    assert any("T_{" in x for x in texts)       # the symbol is there, as mathtext
    plt.close(fig)
