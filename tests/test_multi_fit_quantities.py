"""Fit ▸ Multi-spectrum: any fitted quantity of any peak against a
parameter of the data (explicit user request, 2026-10-06: "linewidth,
integral, intensity, shift, centre-of-mass, shift anisotropy ... as a
function of ... temperature for variable temperature series, or time ...
or anything else")."""

import os

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.multi_fit_panel import MultiFitPanel
from opennmr.notebook import multi_fitting as mf
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

from test_gui import qapp, _make_1d_dataset  # noqa: F401 -- fixture

NU0 = 400e6
TEMPS = (280.0, 300.0, 320.0, 340.0)


def _truth(k):
    """Peak A narrows and moves with temperature; B stays put."""
    return [Site("1H", 2.0 + 0.05 * k, label="A", amount=1.0, fwhm=0.20 - 0.04 * k,
                 lineshape="lorentzian"),
            Site("1H", 5.0, label="B", amount=2.0, fwhm=0.10, lineshape="lorentzian")]


def _spectrum(k, noise=0.5):
    x = np.linspace(10, -4, 8192)
    y = simulate_combined(_truth(k), x[::-1], NU0)["real_spec"][::-1]
    y = y / simulate_combined(_truth(0), x[::-1], NU0)["real_spec"].max() * 1000.0 \
        + np.random.default_rng(k).standard_normal(x.size) * noise
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x)],
                       metadata=NMRMetadata(vendor="simulated", nucleus="1H", sfo1=NU0 / 1e6,
                                            raw={"acqus": {"TE": TEMPS[k], "D": [0, 1 + k]}}))


def _start():
    return [Site("1H", 2.1, label="A", fwhm=0.15, lineshape="lorentzian"),
            Site("1H", 5.0, label="B", fwhm=0.12, lineshape="lorentzian")]


@pytest.fixture(scope="module")
def series():
    from opennmr.notebook import bridge
    experiments = []
    for k in range(len(TEMPS)):
        results = bridge.results_from_spectrum(_spectrum(k))
        experiments.append(mf.make_experiment(results, label=f"T{k}", nucleus="1H",
                                              larmor_Hz=NU0, x_value=TEMPS[k]))
    return mf.fit_multi(experiments, {"1H": _start()})


# ---- notebook ------------------------------------------------------------------

def test_quantities_follow_the_truth(series):
    labels, keys = mf.available_quantities(series)
    assert labels == ["A", "B", mf.WINDOW]
    for key in ("delta_iso", "centre_of_mass", "fwhm", "fwhm_Hz", "integral", "height",
                "amount", "share_%"):
        assert key in keys
    x, fwhm_hz, err, labs = mf.quantity_trend(series, "A", "fwhm_Hz")
    assert x == list(TEMPS) and labs == ["T0", "T1", "T2", "T3"]
    truth = [(0.20 - 0.04 * k) * 400.0 for k in range(4)]
    assert fwhm_hz == pytest.approx(truth, rel=0.03)
    assert np.all(err > 0)
    # Hz is the ppm value times ν0, error included.
    _x, fwhm, ferr, _l = mf.quantity_trend(series, "A", "fwhm")
    assert fwhm_hz == pytest.approx(fwhm * 400.0) and err == pytest.approx(ferr * 400.0)
    # A symmetric peak's centre of mass is its shift.
    _x, com, com_err, _l = mf.quantity_trend(series, "A", "centre_of_mass")
    assert com == pytest.approx([2.0 + 0.05 * k for k in range(4)], abs=0.01)
    assert np.all(np.isnan(com_err))
    # Integrals stand in the 1 : 2 ratio of the amounts, at every temperature.
    _x, ia, ia_err, _l = mf.quantity_trend(series, "A", "integral")
    _x, ib, _e, _l = mf.quantity_trend(series, "B", "integral")
    assert ib / ia == pytest.approx(2.0, rel=0.05)
    assert np.all(ia_err > 0)
    # Narrowing at constant area: the height grows as the width shrinks.
    _x, h, _e, _l = mf.quantity_trend(series, "A", "height")
    assert np.all(np.diff(h) > 0)
    assert h * fwhm == pytest.approx(np.full(4, h[0] * fwhm[0]), rel=0.05)
    # The window, model-free: the intensity-weighted mean shift of the data.
    _x, wcom, _e, _l = mf.quantity_trend(series, mf.WINDOW, "centre_of_mass")
    expected = [(1.0 * (2.0 + 0.05 * k) + 2.0 * 5.0) / 3.0 for k in range(4)]
    assert wcom == pytest.approx(expected, abs=0.05)


def test_quantities_survive_a_saved_result_without_curves(series):
    stripped = {**series, "experiments": [
        {k: (None if k in ("x", "y", "y_fit", "site_spectra", "baseline", "mask") else v)
         for k, v in fit.items()} for fit in series["experiments"]]}
    for key in ("integral", "centre_of_mass", "fwhm_Hz"):
        assert mf.quantity_trend(stripped, "A", key)[1] == pytest.approx(
            mf.quantity_trend(series, "A", key)[1])


def test_transforms_carry_the_errors():
    x, y, e = mf.transform_series([250.0, 500.0], [np.e, np.e ** 2], [0.1, 0.2],
                                  x_transform="1000_over", y_transform="ln")
    assert list(x) == [4.0, 2.0] and y == pytest.approx([1.0, 2.0])
    assert e == pytest.approx([0.1 / np.e, 0.2 / np.e ** 2])
    x, y, e = mf.transform_series([1, 2, 3], [2.0, 4.0, -1.0], [0.2, 0.4, 0.1],
                                  y_transform="normalised to first")
    assert list(y) == [1.0, 2.0, -0.5] and e == pytest.approx([0.1, 0.2, 0.05])
    # Undefined points are dropped, not plotted as nan.
    x, y, _e = mf.transform_series([0.0, 2.0], [1.0, -1.0], [0, 0], x_transform="inverse",
                                   y_transform="log10")
    assert x.size == 0


def test_plot_and_table(series):
    import matplotlib.figure
    ax = matplotlib.figure.Figure().add_subplot(111)
    mf.plot_quantity_trend(series, "fwhm_Hz", labels=["A", "B"], xlabel="T (K)",
                           x_transform="1000_over", ax=ax)
    assert ax.get_xlabel() == "1000/T (K)" and ax.get_ylabel() == "Linewidth (FWHM) (Hz)"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B"]
    xs = ax.lines[0].get_xdata()
    assert list(xs) == pytest.approx(sorted(1000.0 / np.array(TEMPS)))
    text = mf.quantity_table(series, "integral", ["A", mf.WINDOW], xlabel="T (K)")
    rows = text.strip().splitlines()
    assert rows[0] == f"experiment,T (K),A: Integral,A: error,{mf.WINDOW}: Integral,{mf.WINDOW}: error"
    assert len(rows) == 5 and rows[1].startswith("T0,280.0,")
    with pytest.raises(ValueError, match="numeric x"):
        mf.plot_quantity_trend(series, "fwhm", x_values=[None] * 4,
                               ax=matplotlib.figure.Figure().add_subplot(111))


# ---- GUI -----------------------------------------------------------------------------

@pytest.fixture
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _gui_fit(window):
    panel = window.multi_fit_panel
    indices = [window._add_derived_spectrum(_spectrum(k), f"T{k}") for k in range(len(TEMPS))]
    window.spectrum_list.clearSelection()
    for i in indices:
        window.spectrum_list.item(i).setSelected(True)
    panel.add_selected()
    panel.set_sites("1H", _start())
    assert not panel.quantity_box.isVisibleTo(panel)
    assert panel.fit(show_progress=False)
    return panel


def test_gui_plots_any_quantity_against_x(qapp, _dialogs):
    window = MainWindow()
    panel = _gui_fit(window)
    assert panel.quantity_box.isVisibleTo(panel)
    keys = [panel.quantity_combo.itemData(i) for i in range(panel.quantity_combo.count())]
    for key in ("delta_iso", "centre_of_mass", "fwhm_Hz", "integral", "height"):
        assert key in keys
    assert panel.quantity_labels() == ["A", "B"]          # the window is offered, unticked
    # No x yet: said so, nothing drawn.
    panel.quantity_combo.setCurrentIndex(panel.quantity_combo.findData("fwhm_Hz"))
    assert not panel.plot_quantity()
    assert "x value" in _dialogs[-1][1]
    # x after the fit: from the acquisition parameters (TE), then plotted.
    panel.fill_x_values("te")
    assert panel.plot_x_values() == list(TEMPS)
    assert panel.plot_quantity()
    ax = window.fit_canvas.figure.axes[0]
    assert window.canvas_tabs.currentWidget() is window.fit_canvas
    assert ax.get_xlabel() == "T (K)" and ax.get_ylabel() == "Linewidth (FWHM) (Hz)"
    a_line = next(c for c in ax.containers if c.get_label() == "A").lines[0]
    assert list(a_line.get_xdata()) == list(TEMPS)
    assert list(a_line.get_ydata()) == pytest.approx([80, 64, 48, 32], rel=0.03)
    # An Arrhenius-style axis and a log y.
    panel.quantity_x_combo.setCurrentIndex(panel.quantity_x_combo.findData("1000_over"))
    panel.quantity_y_combo.setCurrentIndex(panel.quantity_y_combo.findData("ln"))
    panel.set_quantity_labels(["A"])
    assert panel.plot_quantity()
    ax = window.fit_canvas.figure.axes[0]
    assert ax.get_xlabel() == "1000/T (K)" and ax.get_ylabel().startswith("ln[")
    assert "Centre" not in panel.quantity_note_label.text()
    csv = panel.quantity_csv().splitlines()
    assert csv[0].startswith("experiment,T (K),A: Linewidth")
    assert "plot_quantity_trend(result, 'fwhm_Hz'" in panel.quantity_python()


def test_gui_x_from_any_acquisition_parameter_or_number(qapp, _dialogs, tmp_path):
    window = MainWindow()
    panel = window.multi_fit_panel
    ds = tmp_path / "set"
    ds.mkdir()
    for expno, d1, te in (("12", 1.5, 300), ("7", 3.0, 300), ("30", 6.0, 300)):
        path = _make_1d_dataset(ds, n=16, name=expno)
        with open(os.path.join(path, "acqus"), "a") as f:
            f.write(f"##$TE= {te}\n##$D= (0..3)\n0 {d1} 0 0\n")
    panel.add_expnos(str(ds), ["12", "7", "30"])
    names = panel.acquisition_parameter_names()
    assert names[0] == "d1"                      # the one that varies comes first
    assert "te" in names
    panel.fill_x_values("param", "d1")
    assert [e["x"] for e in panel.experiments] == ["1.5", "3", "6"]
    assert panel.x_label_edit.text() == "d1"
    panel.fill_x_values("expno")
    assert [e["x"] for e in panel.experiments] == ["12", "7", "30"]
    panel.fill_x_values("order")
    assert [e["x"] for e in panel.experiments] == ["1", "2", "3"]
    panel.fill_x_values("param", "nonexistent")
    assert "No nonexistent in" in _dialogs[-1][1]


def test_expno_from_paths():
    assert MultiFitPanel._expno_of("/data/set/12") == "12"
    assert MultiFitPanel._expno_of(r"C:\data\set\302\pdata\1") == "302"
    assert MultiFitPanel._expno_of("derived:abc") is None
