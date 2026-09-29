"""2026-09-29 (fourth round), kept fits: each can be an Open experiment
(its data, carrying the fit), compared in Compare stacked (a row per fit:
data, fit, components, residual) or superimposed, with per-entry display
(right-click ▸ Fit Display…) and colours; the notebook draws the same.
Synthetic data only."""
import json

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.io import history as io_history
from opennmr.notebook import fit_models as fm


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    monkeypatch.setenv("OPENNMR_STATES_DIR", str(tmp_path / "states"))
    return MainWindow()


def _two_peaks():
    x = np.linspace(20, -20, 801)
    a = 100 * np.exp(-((x - 5) / 1.5) ** 2)
    b = 60 * np.exp(-((x + 4) / 2.0) ** 2)
    y = a + b + np.random.default_rng(0).normal(0, 1.0, x.size)
    return x, y, a, b


def _models():
    x, y, a, b = _two_peaks()
    one = 1.3 * (a + b) / 1.3
    m1 = fm.make_fit_model("1 site", "isotropic", x, y, a + 0.9 * b,
                           components=[a + 0.9 * b, 0 * x], labels=["A"], n_free=3)
    m2 = fm.make_fit_model("2 sites", "isotropic", x, y, a + b,
                           components=[a, b, 0 * x], labels=["A", "B"], n_free=6)
    return m1, m2


# ---- notebook -----------------------------------------------------------------

def test_fit_view_is_checked():
    view = fm.fit_view({"show_residual": 1, "style": {"show": "none"}})
    assert view["show_residual"] is True and view["style"]["show"] == "none"
    with pytest.raises(KeyError):
        fm.fit_view({"colour": "red"})


def test_draw_fit_model_scales_and_puts_the_residual_below():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _m1, m2 = _models()
    fig, ax = plt.subplots()
    out = fm.draw_fit_model(ax, m2, {"show_residual": True}, scale=0.5, fit_colour="#123456")
    assert out["fit"].get_color() == "#123456" and out["fit"].get_label() == "2 sites"
    np.testing.assert_allclose(np.max(out["fit"].get_ydata()), 0.5 * np.max(m2["y_fit"]))
    assert set(out["components"]) == {"A", "B"}
    res = out["residual"].get_ydata()
    assert np.nanmax(res) < min(0.0, np.min(out["data"].get_ydata()))   # entirely below
    # Hidden fit, no data, the view's own colour wins over the trace's.
    fig2, ax2 = plt.subplots()
    out = fm.draw_fit_model(ax2, m2, {"show_fit": False, "fit_colour": "green",
                                      "style": {"show": "none"}}, fit_colour="red",
                            draw_data=False)
    assert out["data"] is None and not out["fit"].get_visible()
    assert out["fit"].get_color() == "green" and not out["components"]
    plt.close("all")


def test_plot_fit_comparison_stacked_and_colours():
    import matplotlib
    matplotlib.use("Agg")
    m1, m2 = _models()
    fig = fm.plot_fit_comparison([m1, m2], layout="stacked", colours=["#aa0000", "#0000aa"])
    rows = fig.fit_comparison_axes["rows"]
    assert len(rows) == 2
    fits = [[l for l in ax.lines if l.get_label() == name][0]
            for ax, name in zip(rows, ("1 site", "2 sites"))]
    assert [f.get_color() for f in fits] == ["#aa0000", "#0000aa"]
    assert all(any(l.get_label() == "Data" for l in ax.lines) for ax in rows)   # data every row
    assert all(any(l.get_label().endswith("residual") for l in ax.lines) for ax in rows)
    fig = fm.plot_fit_comparison([m1, m2], colours=["#aa0000", "#0000aa"])
    labels = {l.get_label(): l.get_color() for l in fig.fit_comparison_axes["fit"].lines}
    assert labels["2 sites"] == "#0000aa"
    with pytest.raises(ValueError):
        fm.plot_fit_comparison([m1], layout="sideways")


# ---- GUI ----------------------------------------------------------------------

def _window_with_kept_fits(window):
    x, y, _a, _b = _two_peaks()
    spectrum = NMRSpectrum(y, [NMRAxis("frequency", ppm=x, hz=x * 400.0)],
                           NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))
    index = window._add_derived_spectrum(spectrum, "peaks")
    m1, m2 = _models()
    ids = []
    for m in (m1, m2):
        models = window._fit_models(index)
        m["id"] = len(models) + 1
        window._set_fit_models(index, models + [m])
        ids.append(m["id"])
    return index, ids


def _fit_lines(ax, name):
    return [l for l in ax.lines if l.get_label() == name]


def test_compare_ticked_stacked_makes_open_experiments_once(window):
    index, ids = _window_with_kept_fits(window)
    n = len(window.session)
    fp = window.fit_models_panel
    assert fp.layout_combo.currentData() == "stack"
    fp.set_ticked(ids)
    fp.residuals_check.setChecked(True)
    fp.compare()
    assert len(window.session) == n + 2
    rows = [n, n + 1]
    for row in rows:
        raw = window.session.spectra[row].metadata.raw
        assert raw["fit_model_source"]["spectrum"] == window.session.label(index)
        assert raw["fit_model_view"]["show_residual"] is True
    canvas = window.compare_canvas
    assert window.canvas_tabs.currentWidget() is canvas
    axes = canvas._compare_axes
    assert len(axes) == 2
    for ax, name in zip(axes, ("1 site", "2 sites")):
        label = f"{name} — peaks"
        assert _fit_lines(ax, label), [l.get_label() for l in ax.lines]
        assert any(l.get_label().endswith("residual") for l in ax.lines)
        assert any(l.get_label() == "Data" and l.get_visible() for l in ax.lines)
    # Components of the 2-site fit are drawn (filled) in its row only.
    assert {c.get_label() for c in axes[1].collections} >= {"A", "B"}
    # One legend entry for the repeated "Data".
    handles, labels = canvas._legend_entries()
    assert labels.count("Data") == 1
    # Again: the same experiments are reused.
    window._select_single_row(index)
    fp.set_ticked(ids)
    fp.compare()
    assert len(window.session) == n + 2
    # Saved state stays JSON-safe, and the history does not diff the fit.
    json.dumps(io_history.tracked_state(window.session.spectra[n]))
    for key in ("fit_model", "fit_model_view", "fit_model_source"):
        assert key in io_history._HIDDEN_KEYS


def test_compare_ticked_superimposed_draws_the_data_once(window):
    index, ids = _window_with_kept_fits(window)
    fp = window.fit_models_panel
    fp.set_ticked(ids)
    fp.layout_combo.setCurrentIndex(fp.layout_combo.findData("overlay"))
    fp.compare()
    canvas = window.compare_canvas
    assert canvas._compare_axes is None
    data = [l for l in canvas.ax.lines if l.get_visible() and l.get_label() == "Data"]
    assert len(data) == 1
    residuals = [l for l in canvas.ax.lines if l.get_label().endswith("residual")]
    assert len(residuals) == 2
    # One below the other, not on top of each other.
    assert abs(np.nanmean(residuals[0].get_ydata()) - np.nanmean(residuals[1].get_ydata())) > 1


def test_fit_display_dialog_and_colours(window):
    index, ids = _window_with_kept_fits(window)
    fp = window.fit_models_panel
    fp.set_ticked(ids)
    fp.compare()
    row = len(window.session) - 1                      # the 2-site entry
    label = window.session.label(row)
    dialog = window._open_fit_view_dialog(row, exec_dialog=False)
    assert dialog.components_panel.table.rowCount() == 2
    dialog.components_panel.show_combo.setCurrentIndex(
        dialog.components_panel.show_combo.findData("none"))
    dialog._set("fit_colour", "#00aa00")
    dialog.residual_check.setChecked(False)
    view = window.session.spectra[row].metadata.raw["fit_model_view"]
    assert view["style"]["show"] == "none" and view["fit_colour"] == "#00aa00"
    ax = window.compare_canvas._compare_axes[1]
    fit = _fit_lines(ax, label)[0]
    assert fit.get_color() == "#00aa00"
    assert not ax.collections                                  # components hidden
    assert not any(l.get_label().endswith("residual") for l in ax.lines)
    # A kept fit's colour (Kept fit models ▸ Colour…) reaches its entry too.
    window._select_single_row(index)
    window._recolour_fit_model(index, ids[0], "#aa00aa")
    first = len(window.session) - 2
    assert window.session.spectra[first].metadata.raw["fit_model_view"]["fit_colour"] == "#aa00aa"
    assert window._fit_models(index)[0]["colour"] == "#aa00aa"


def test_compare_list_right_click_offers_fit_display(window, monkeypatch):
    # QMenu.exec is modal and cannot be patched on the real class
    # headlessly: replace the QMenu main_window builds with a fake.
    import opennmr.gui.main_window as main_window_module
    index, ids = _window_with_kept_fits(window)
    fp = window.fit_models_panel
    fp.set_ticked(ids)
    fp.compare()
    added = []

    class _Action:
        def __init__(self, text):
            self._text = text

        def text(self):
            return self._text

        def __getattr__(self, name):          # setToolTip, setEnabled, ...
            return lambda *a, **k: None

    class _FakeMenu:
        def __init__(self, *a, **k):
            pass

        def addAction(self, text):
            added.append(text)
            return _Action(text)

        def addSeparator(self):
            pass

        def exec(self, *a, **k):
            return None

    monkeypatch.setattr(main_window_module, "QMenu", _FakeMenu)
    item = window.compare_list.item(0)
    window._on_compare_list_context_menu(window.compare_list.visualItemRect(item).center())
    assert "Fit Display…" in added
    added.clear()
    window._on_spectrum_list_context_menu(
        window.spectrum_list.visualItemRect(window.spectrum_list.item(index + 1)).center())
    assert "Show the Fit" in added and "Fit Display (in Compare)…" in added


def test_curve_fits_fall_back_to_the_summary(window):
    x = np.linspace(20, -20, 101)
    spectrum = NMRSpectrum(np.exp(-x ** 2), [NMRAxis("frequency", ppm=x, hz=x * 400.0)],
                           NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))
    index = window._add_derived_spectrum(spectrum, "s")
    t = np.geomspace(0.01, 1, 8)
    models = []
    for k, tau in enumerate((0.2, 0.3)):
        m = fm.make_fit_model(f"T1 {k}", "curve", t, 1 - 2 * np.exp(-t / 0.2),
                              1 - 2 * np.exp(-t / tau), n_free=3)
        m["id"] = k + 1
        models.append(m)
    window._set_fit_models(index, models)
    n = len(window.session)
    window._compare_fit_models(index, [1, 2], layout="stack")
    assert len(window.session) == n
    assert window.fit_canvas.figure.fit_comparison_axes["fit"] is not None
