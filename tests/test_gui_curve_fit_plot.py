"""Dynamics ▸ Fit Curve's plot (explicit user request, 2026-09-30): the fit
drawn as the fitted function (not its values joined point to point), the
fit line's colour/style/width, all peaks on one plot, keeping curve fits;
and a "Keep This Fit…" button in every Fit section."""
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.processing import relaxation as processing_relaxation

from test_gui import (  # noqa: F401 -- fixture
    _load_spectrum_directly, _make_pseudo2d_relaxation_spectrum, _set_delay_unit, qapp,
)

DELAYS = np.geomspace(0.001, 0.3, 10)


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


def _fitted(two_peaks=False):
    spectrum = _make_pseudo2d_relaxation_spectrum(DELAYS, x0=5.0, fwhm=0.05, T2=0.05)
    if two_peaks:
        other = _make_pseudo2d_relaxation_spectrum(DELAYS, x0=4.0, fwhm=0.05, T2=0.02, M0=0.5,
                                                   seed=1)
        spectrum.data = spectrum.data + other.data
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    window._add_peak_row(shift=5.0, fwhm=0.05, eta=0.0)
    if two_peaks:
        window._add_peak_row(shift=4.0, fwhm=0.05, eta=0.0)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in DELAYS))
    window.relaxation_model_combo.setCurrentIndex(2)          # exponential decay (T2)
    window._on_fit_curve_button_clicked()
    assert window._relaxation_fit_results
    return window


def _fit_line(window):
    return next(line for line in window.fit_canvas.ax.lines if line.get_label() == "Fit")


def test_the_fit_is_drawn_as_the_fitted_function(qapp):  # noqa: F811
    window = _fitted()
    line = _fit_line(window)
    x, y = line.get_data()
    result = window._relaxation_fit_results[0]
    assert len(x) == 400 > len(result["t"])                  # not the points joined up
    t_fine, y_fine = processing_relaxation.fit_curve(result)
    np.testing.assert_allclose(y, y_fine)
    np.testing.assert_allclose(y, result["params"]["M0"] * np.exp(-t_fine / result["params"]["T2"]),
                               rtol=1e-9)
    # Between two measured points the line follows the exponential, not a chord.
    t = np.sort(result["t"])
    mid = 0.5 * (t[-2] + t[-1])
    chord = np.interp(mid, t, np.asarray(result["y_fit"])[np.argsort(result["t"])])
    assert abs(np.interp(mid, x, y) - chord) > 1e-4


def test_fit_line_colour_style_and_width(qapp):  # noqa: F811
    window = _fitted()
    box = window.curve_fit_plot_box
    line = _fit_line(window)
    assert line.get_linestyle() == "--" and line.get_color() == "#d62728"
    box.set_colour("#0072b2")
    box.style_combo.setCurrentIndex(box.style_combo.findData("dotted"))
    box.width_spin.setValue(3.5)
    line = _fit_line(window)
    assert line.get_color() == "#0072b2" and line.get_linestyle() == ":"
    assert line.get_linewidth() == 3.5
    assert window._ui_settings["curve_fit_line"]["style"] == "dotted"
    box.style_combo.setCurrentIndex(box.style_combo.findData("dashdotdot"))
    assert len(_fit_line(window)._unscaled_dash_pattern[1]) == 6     # dash dot dot
    # Remembered: a new window starts from the saved line.
    other = MainWindow()
    assert other.curve_fit_plot_box.settings()["colour"] == "#0072b2"


def test_all_peaks_together_and_normalised(qapp):  # noqa: F811
    window = _fitted(two_peaks=True)
    box = window.curve_fit_plot_box
    box.view_combo.setCurrentIndex(box.view_combo.findData("all"))
    ax = window.fit_canvas.ax
    fits = [line for line in ax.lines if line.get_linestyle() != "None"]
    points = [line for line in ax.lines if line.get_linestyle() == "None"]
    assert len(fits) == 2 and len(points) == 2
    assert fits[0].get_color() != fits[1].get_color()
    assert fits[0].get_color() == points[0].get_color()
    assert len(ax.get_legend().get_texts()) == 2
    assert window.relaxation_fit_params_label.text().count("T2") >= 2
    box.normalise_check.setChecked(True)
    fits = [line for line in window.fit_canvas.ax.lines if line.get_linestyle() != "None"]
    assert max(abs(fits[0].get_ydata()).max(), abs(fits[1].get_ydata()).max()) == pytest.approx(1.0)
    assert "normalised" in window.fit_canvas.ax.get_ylabel()


def test_keeping_a_curve_fit_and_all_of_them(qapp):  # noqa: F811
    window = _fitted(two_peaks=True)
    kept = window._keep_curve_fits(name="T2 decay")
    assert len(kept) == 1 and kept[0]["kind"] == "curve"
    assert len(kept[0]["curve_x"]) == 400
    panel = window.curve_fit_models_panel
    panel.refresh()
    assert panel.table.rowCount() == 1
    box = window.curve_fit_plot_box
    box.view_combo.setCurrentIndex(box.view_combo.findData("all"))
    kept = window._keep_curve_fits(name="Both")
    assert [m["name"] for m in kept][0].startswith("Both: ")
    assert len(kept) == 2
    panel.refresh()
    assert panel.table.rowCount() == 3
    # A kept curve is shown smooth.
    window._show_fit_model(window._selected_index(), kept[0]["id"])
    assert len(_fit_line(window).get_xdata()) == 400


def test_every_fit_section_has_a_keep_button(qapp, _dialogs):  # noqa: F811
    window = MainWindow()
    assert set(window.keep_fit_buttons) == {"jcoupling", "csa", "quadrupolar", "czjzek",
                                            "combined", "isotropic"}
    assert all(b is not None for b in window.keep_fit_buttons.values())
    # Next to the section's own "Use fitted values as start".
    csa = window.keep_fit_buttons["csa"]
    assert csa.parentWidget() is window.csa_use_fitted_button.parentWidget()
    # Nothing fitted yet: says so, keeps nothing.
    assert window._keep_panel_fit("quadrupolar", name="x") is None
    assert _dialogs and "Run the fit first" in _dialogs[-1][1]


def test_a_sections_keep_button_keeps_that_sections_fit(qapp, _dialogs):  # noqa: F811
    from test_gui_quadrupolar_fit import _load
    window = MainWindow()
    panel = window.quad_fit_panel
    _load(window)
    panel.estimate()
    panel.fit()
    assert not _dialogs, _dialogs
    # Another plot on the Fit view since: the quadrupolar fit is redrawn to keep it.
    window.fit_canvas._curve_fit_shown = None
    window._last_fit_key = "csa"
    model = window._keep_panel_fit("quadrupolar", name="Quad 1")
    assert model is not None and model["kind"] == "quadrupolar"
    assert window._fit_models(window._selected_index())[-1]["name"] == "Quad 1"
