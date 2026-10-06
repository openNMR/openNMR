"""The correlation matrix for spectral fits (user request, 2026-10-06: "Can
we add the correlation matrix for normal spectral fits, too?") --
notebook.fit_correlation for isotropic peak fits, Combined-Hamiltonian and
multi-spectrum fits, and its buttons in Fit ▸ Isotropic, Fit ▸ Combined
Hamiltonian and Fit ▸ Multi-spectrum."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum  # noqa: E402
from opennmr.gui.main_window import MainWindow  # noqa: E402
from opennmr.notebook import combined_fitting as cf  # noqa: E402
from opennmr.notebook import fit_correlation as fc  # noqa: E402
from opennmr.notebook import multi_fitting as MF  # noqa: E402
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined  # noqa: E402
from opennmr.processing import fitting as pf  # noqa: E402

from test_gui import qapp, _load_spectrum_directly, _make_pseudo2d_relaxation_spectrum  # noqa: F401,E402

PPM = np.linspace(6, 4, 800)


def _lor(c, w, a):
    return a / (1 + ((PPM - c) / (w / 2)) ** 2)


def _peaks_spectrum(seed=0):
    rng = np.random.default_rng(seed)
    y = _lor(5.0, 0.05, 1) + _lor(5.06, 0.05, 0.6) + _lor(4.5, 0.04, 0.5) \
        + rng.normal(0, 0.005, PPM.size)
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=PPM, hz=PPM * 400)],
                       metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0,
                                            raw={"slice": {"index": 1, "shift": None}}))


DEFS = [{"type": "pseudo_voigt", "shift": 5.0, "fwhm": 0.05},
        {"type": "pseudo_voigt", "shift": 5.06, "fwhm": 0.05},
        {"type": "lorentzian", "shift": 4.5, "fwhm": 0.04}]


def _iso_fit(defs=DEFS):
    return pf.NMR2D_fit_slices({"rows": [_peaks_spectrum()]}, peak_defs=defs, mode="rows",
                               show_report=False, show_warnings=False)["rows"][0]


# ---- isotropic peak fits ---------------------------------------------------------------

def test_isotropic_fit_overlapping_peaks_correlate_and_a_separate_one_does_not():
    fit = _iso_fit()
    names, labels, R = fc.correlation_of(fit)
    assert names[:4] == ["Peak 1: shift", "Peak 1: fwhm", "Peak 1: amplitude", "Peak 1: eta"]
    assert len(names) == 11 and labels[2] == r"$\mathit{A}_{1}$"
    i = {n: k for k, n in enumerate(names)}
    # Peaks 1 and 2 overlap (0.06 ppm apart, 0.05 wide); peak 3 is alone.
    across = abs(R[i["Peak 1: shift"], i["Peak 2: shift"]])
    alone = max(abs(R[i["Peak 3: amplitude"], i[f"Peak 1: {p}"]])
                for p in ("shift", "fwhm", "amplitude", "eta"))
    assert across > 0.2 and alone < 0.05
    # The matrix is the fit's own covariance, normalised.
    full = fit["pcov"]
    a, b = 1, 6
    expected = full[a, b] / np.sqrt(full[a, a] * full[b, b])
    assert R[i["Peak 1: fwhm"], i["Peak 2: amplitude"]] == pytest.approx(expected)


def test_fixed_parameters_have_no_row_and_a_free_block_covariance_is_accepted():
    defs = [dict(DEFS[0], fixed={"shift": True}), DEFS[1], DEFS[2]]
    fit = _iso_fit(defs)
    names, _labels, R = fc.correlation_of(fit)
    assert "Peak 1: shift" not in names and len(names) == 10
    free = np.flatnonzero(fit["is_free"])
    block = dict(fit, pcov=np.asarray(fit["pcov"])[np.ix_(free, free)])
    assert np.allclose(fc.correlation_of(block)[2], R, equal_nan=True)
    with pytest.raises(ValueError, match="does not match"):
        fc.correlation_of(dict(fit, pcov=np.eye(3)))
    with pytest.raises(ValueError, match="no covariance"):
        fc.correlation_of({"x": [1, 2]})


def test_figure_separates_peaks_and_drops_cell_values_when_crowded():
    fit = _iso_fit()
    fig = fc.plot_correlation_matrix(fit, title="1D fit")
    ax = fig.correlation_axes["matrix"]
    assert ax.get_title() == "1D fit"
    assert len(ax.texts) == 11 * 12 // 2                    # 11 ≤ 12: annotated
    # One separator pair per peak boundary, inside the lower triangle only.
    separators = [ln for ln in ax.lines if ln.get_color() == "0.35"]
    assert len(separators) == 4
    for ln in separators:
        xs, ys = ln.get_data()
        assert all(x <= y + 1e-9 for x, y in zip(xs, ys)) or xs[0] == xs[1]
    # More than 12 parameters: no cell values ("auto").
    big = {"free_names": [f"p{k}" for k in range(14)], "pcov": np.eye(14)}
    assert not fc.plot_correlation_matrix(big).correlation_axes["matrix"].texts


# ---- Combined Hamiltonian ------------------------------------------------------------

NU0 = 100.6e6
X13 = np.linspace(-150, 350, 4096)


@pytest.fixture(scope="module")
def combined():
    y = simulate_combined([Site("13C", 120, label="C1", fwhm=0.6).add_csa(-150, 0.4),
                           Site("13C", 30, label="C2", fwhm=1.0)],
                          X13, NU0, mas_Hz=8000, n_zcw=987)["real_spec"]
    y = y / y.max() * 1000 + np.random.default_rng(0).normal(0, 1, X13.size)
    data = {"x_ppm": X13[::-1], "real_spectrum": y[::-1], "SFO1": NU0 / 1e6}
    return cf.fit_combined(data, [Site("13C", 120, label="C1", fwhm=0.6).add_csa(-130, 0.5),
                                  Site("13C", 30, label="C2", fwhm=1.0)],
                           mas_Hz=8000, n_zcw=987)


def test_combined_fit_labels_and_groups(combined):
    data = fc.correlation_input(combined)
    assert data["names"] == combined["free_names"]
    assert data["labels"][0] == r"$\delta_\mathrm{iso}$ (C1)"
    assert r"$\delta_\mathrm{aniso}$ (C1)" in data["labels"]
    assert r"$\eta_\mathrm{CSA}$ (C1)" in data["labels"]
    assert [g[1] for g in data["groups"]] == ["C1"] * 5 + ["C2"] * 3
    _n, _l, R = fc.correlation_of(combined)
    assert np.allclose(np.diag(R), 1)


def test_symbols_tell_a_euler_angle_from_a_stretch_exponent():
    assert fc._spectral_label("beta", "Al1") == r"$\beta$ (Al1)"
    assert fc._spectral_label("C_Q", "Al1") == r"$\mathit{C}_\mathrm{Q}$ (Al1)"
    assert fc._spectral_label("J2", "H1") == r"$\mathit{J}_{2}$ (H1)"
    assert fc._spectral_label("mas_Hz") == r"$\nu_\mathrm{r}$"
    assert fc.parameter_label("beta") == r"$\beta$"


# ---- multi-spectrum --------------------------------------------------------------------

def _al_spectrum(cq, seed):
    x = np.linspace(100, -150, 3000)
    site = Site("27Al", 12.0, label="Al1", fwhm=0.6).add_quadrupolar(cq, 0.35)
    y = simulate_combined([site], x[::-1], 130.3e6, mas_Hz=20e3, transitions="ct",
                          n_zcw=987)["real_spec"][::-1]
    y = y / y.max() * 1000 + np.random.default_rng(seed).standard_normal(x.size) * 2
    return {"x_ppm": x, "real_spectrum": y, "SFO1": 130.3, "nuclide": "27Al"}


@pytest.fixture(scope="module")
def joint():
    exps = [MF.make_experiment(_al_spectrum(4.2, k), lab, mas_Hz=20e3, transitions="ct",
                               n_zcw=987) for k, lab in enumerate(("a", "b"))]
    start = {"27Al": [Site("27Al", 10, label="Al1", fwhm=1.0).add_quadrupolar(3.9, 0.5)]}
    return MF.fit_multi(exps, start, preset="series", scopes={"C_Q": "global",
                                                               "eta_Q": "global"})


def test_joint_multi_fit_keeps_its_covariance_across_spectra(joint):
    assert joint["mode"] == "joint"
    names = joint["free_names"]
    assert np.asarray(joint["pcov"]).shape == (len(names), len(names))
    assert "Al1: C_Q [shared]" in names and "Al1: delta_iso [a]" in names
    data = fc.multi_fit_correlation(joint)
    labels = fc.correlation_input(data)["labels"]
    assert r"$\mathit{C}_\mathrm{Q}$ (Al1) [shared]" in labels
    # Sequential mode has no joint matrix, and says which to use instead.
    with pytest.raises(ValueError, match="refined\\s+jointly"):
        fc.multi_fit_correlation(joint, 0)
    seq = {"experiments": [{"label": "a", "pcov": np.eye(2), "free_names": ["x", "y"]}]}
    with pytest.raises(ValueError, match="one spectrum at a time"):
        fc.multi_fit_correlation(seq)
    assert fc.multi_fit_correlation(seq, "a")["title"] == "a"


# ---- GUI ---------------------------------------------------------------------------

@pytest.fixture
def said(monkeypatch):
    out = []
    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: out.append((_n, a[2] if len(a) > 2 else "")))
    return out


def test_isotropic_button_swaps_the_fit_for_its_matrix(qapp, said):
    window = MainWindow()
    _load_spectrum_directly(window, _peaks_spectrum())
    button = window.spectral_fit_correlation_button
    assert not button.isEnabled()
    for d in DEFS:
        window._add_peak_row(d["shift"], d["fwhm"], 0.5)
    window._on_fit_spectrum_clicked()
    assert said == [], said
    assert button.isEnabled()
    button.setChecked(True)
    canvas = window.spectral_fit_canvas
    names, R = canvas.figure.correlation
    assert names[0] == "Peak 1: shift" and R.shape[0] == len(names)
    assert window.canvas_tabs.currentWidget() is canvas
    assert window.spectral_fit_table.rowCount() == 3          # the table stays
    button.setChecked(False)
    assert canvas._external_plot is False                     # the fit is back


def test_isotropic_button_follows_the_slice_of_a_series(qapp, said):
    window = MainWindow()
    delays = np.geomspace(0.001, 0.3, 4)
    _load_spectrum_directly(window, _make_pseudo2d_relaxation_spectrum(delays, T2=0.05))
    window._add_peak_row(5.0, 0.05, 0.5)
    window._on_fit_spectrum_clicked()
    window.spectral_fit_correlation_button.setChecked(True)
    window.view_slice_fit_spin.setValue(3)
    fig = window.spectral_fit_canvas.figure
    assert window._last_spectral_fit["index"] == 3
    assert fig.correlation[0] == fc.correlation_of(window._last_spectral_fit)[0]
    assert np.allclose(fig.correlation[1], fc.correlation_of(window._last_spectral_fit)[2])


def _combined_window():
    window = MainWindow()
    y = simulate_combined([Site("13C", 120, label="C1", fwhm=0.6).add_csa(-150, 0.4)],
                          X13, NU0, mas_Hz=12000, n_zcw=987)["real_spec"]
    y = y / y.max() * 1000 + np.random.default_rng(1).normal(0, 1, X13.size)
    spectrum = NMRSpectrum(data=y[::-1], axes=[NMRAxis(kind="frequency", ppm=X13[::-1])],
                           metadata=NMRMetadata(vendor="simulated", nucleus="13C",
                                                sfo1=NU0 / 1e6,
                                                raw={"acqus": {"MASR": 12000}}))
    window._add_derived_spectrum(spectrum, "C")
    panel = window.combined_fit_panel
    panel.seed_from_spectrum(spectrum)
    panel.diso_spin.setValue(120.0)
    panel.fwhm_spin.setValue(0.6)
    panel.add_interaction("csa", aniso=-130.0, asym=0.5)
    panel.orientations_spin.setValue(987)
    return window, panel


def test_combined_button_draws_the_matrix_and_goes_back(qapp, said):
    window, panel = _combined_window()
    assert not panel.correlation_button.isEnabled()
    assert panel.fit(), said
    assert panel.correlation_button.isEnabled() and not panel.correlation_button.isChecked()
    panel.correlation_button.setChecked(True)
    names, _R = window.fit_canvas.figure.correlation
    assert names == panel.result["free_names"]
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    panel.correlation_button.setChecked(False)
    assert window.fit_canvas._external_plot is False
    # A new fit is drawn as its fit first.
    panel.correlation_button.setChecked(True)
    assert panel.fit()
    assert not panel.correlation_button.isChecked()


def test_multi_panel_offers_the_joint_matrix(qapp, said, joint):
    window = MainWindow()
    panel = window.multi_fit_panel
    panel.result = joint
    panel._report()
    combo = panel.show_combo
    items = [combo.itemText(i) for i in range(combo.count())]
    assert "Correlations: joint fit, all spectra" in items
    combo.setCurrentIndex(items.index("Correlations: joint fit, all spectra"))
    panel.show_selected()
    names, _R = window.fit_canvas.figure.correlation
    assert names == joint["free_names"]
