"""The GUI's CSA fit: fit window, broadening, fix/bounds, results table,
MAS rate per experiment (explicit user report and requests, 2026-09-23)."""
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import csa_fitting as C

from test_gui import qapp  # noqa: F401 -- fixture

LARMOR_MHZ = 100.0
MAS_HZ = 5000.0


def _mas_spectrum(scale=1.0e10, eta=0.0, lineshape="gaussian", masr=MAS_HZ):
    x = np.linspace(-300.0, 300.0, 2048)
    y = C.csa_model_full(x, 0.0, -100.0, 0.5, 2.0, eta=eta, convention="Haeberlen",
                         static=False, larmor_Hz=LARMOR_MHZ * 1e6, mas_Hz=MAS_HZ,
                         powder_scheme="zcw", n_zcw=400, lineshape=lineshape)
    ppm = x[::-1]
    return NMRSpectrum(
        data=(scale * y)[::-1], axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * LARMOR_MHZ)],
        metadata=NMRMetadata(vendor="bruker", nucleus="13C", sfo1=LARMOR_MHZ,
                             raw={"acqus": {"MASR": masr}}))


def _add(window, spectrum, name="csa"):
    window.session.add_derived(spectrum, name)
    window.spectrum_list.addItem(name)
    window.spectrum_list.setCurrentRow(window.spectrum_list.count() - 1)


@pytest.fixture
def mas_window(qapp):
    window = MainWindow()
    _add(window, _mas_spectrum())
    window.csa_static_combo.setCurrentIndex(window.csa_static_combo.findData(False))
    window.csa_orientations_spin.setValue(200)
    window.csa_render_orientations_spin.setValue(400)
    window._add_csa_site(("A", "2", "-80", "0.4", "3"))
    yield window
    window.close()


# ---- MAS rate follows the experiment ----------------------------------

def test_mas_rate_is_reseeded_when_the_experiment_changes(qapp):
    # The box kept the previous experiment's rate (it only filled itself
    # when empty), so a fit could silently run at another sample's rate.
    window = MainWindow()
    _add(window, _mas_spectrum(masr=5000), "a")
    window._refresh_csa_controls(window.session.spectra[0])
    assert window.csa_mas_spin.value() == pytest.approx(5000)
    _add(window, _mas_spectrum(masr=20000), "b")
    window._refresh_csa_controls(window.session.spectra[1])
    assert window.csa_mas_spin.value() == pytest.approx(20000)


# ---- fit window ---------------------------------------------------------

def test_the_integration_regions_are_not_used_unless_chosen(mas_window):
    # They were, silently: a region drawn around a centreband cut the
    # sidebands out of the CSA fit.
    window = mas_window
    window._add_region_row(-20.0, 20.0)
    assert window.csa_window_combo.currentData() == "all"
    assert window._csa_fit_regions() is None
    window.csa_window_combo.setCurrentIndex(window.csa_window_combo.findData("regions"))
    assert window._csa_fit_regions() == [(-20.0, 20.0)]
    window.csa_window_combo.setCurrentIndex(window.csa_window_combo.findData("range"))
    window.csa_window_lo_spin.setValue(150.0)
    window.csa_window_hi_spin.setValue(-150.0)
    assert window._csa_fit_regions() == [(-150.0, 150.0)]


def test_the_fit_recovers_the_tensor_and_reports_its_setup(mas_window):
    window = mas_window
    window._on_csa_fit_clicked()
    popt = window._csa_fit_result["popt"]
    assert popt[1] == pytest.approx(-100.0, abs=1.5)
    assert popt[2] == pytest.approx(0.5, abs=0.03)
    summary = window.csa_results_label.text()
    assert "CSA fit report" in summary
    assert "finite MAS at 5 kHz" in summary and "Larmor 100 MHz" in summary
    assert "window whole spectrum" in summary
    assert "⚠" not in summary


def test_a_window_without_sidebands_is_flagged_in_the_report(mas_window):
    window = mas_window
    window.csa_window_combo.setCurrentIndex(window.csa_window_combo.findData("range"))
    window.csa_window_lo_spin.setValue(-30.0)
    window.csa_window_hi_spin.setValue(30.0)
    window._on_csa_fit_clicked()
    assert "no spinning sideband" in window.csa_results_label.text()


# ---- broadening ---------------------------------------------------------

def test_broadening_choice_reaches_the_fit(mas_window):
    window = mas_window
    assert window.csa_sites_table.isColumnHidden(5)            # L fraction: PV only
    window.csa_broadening_combo.setCurrentIndex(window.csa_broadening_combo.findData("pseudo_voigt"))
    assert not window.csa_sites_table.isColumnHidden(5)
    site = window._read_csa_sites()[0]
    assert site["type"] == "pseudo_voigt" and site["eta"] == pytest.approx(0.5)
    window.csa_broadening_combo.setCurrentIndex(window.csa_broadening_combo.findData("lorentzian"))
    assert window._read_csa_sites()[0]["type"] == "lorentzian"


def test_pseudo_voigt_fits_the_lorentzian_fraction(qapp):
    window = MainWindow()
    _add(window, _mas_spectrum(eta=0.6, lineshape="pseudo_voigt"))
    window.csa_static_combo.setCurrentIndex(window.csa_static_combo.findData(False))
    window.csa_orientations_spin.setValue(200)
    window.csa_render_orientations_spin.setValue(400)
    window.csa_broadening_combo.setCurrentIndex(window.csa_broadening_combo.findData("pseudo_voigt"))
    window._add_csa_site(("A", "2", "-80", "0.4", "3", "0.5"))
    window._on_csa_fit_clicked()
    result = window._csa_fit_result
    assert result["info"]["sim_kwargs"]["lineshape"] == "pseudo_voigt"
    assert result["popt"][5] == pytest.approx(0.6, abs=0.03)
    headers = [window.csa_results_table.horizontalHeaderItem(c).text()
               for c in range(window.csa_results_table.columnCount())]
    assert "L fraction" in headers
    assert "pseudo-Voigt broadening" in window.csa_results_label.text()


# ---- fix / bounds -------------------------------------------------------

def test_fixing_a_value_by_right_click_holds_it(mas_window):
    window = mas_window
    window._set_csa_row_constraints(0, {}, {"skew_or_eta": True})
    assert window.csa_sites_table.item(0, 3).font().italic()
    assert window._read_csa_sites()[0]["fixed"] == {"skew_or_eta": True}
    window._on_csa_fit_clicked()
    result = window._csa_fit_result
    assert not result["is_free"][2]
    assert result["popt"][2] == pytest.approx(0.4)
    eta_cell = window.csa_results_table.item(0, 2).text()
    assert "(fixed)" in eta_cell


def test_bounds_set_on_a_site_constrain_the_fit(mas_window):
    window = mas_window
    window._set_csa_row_constraints(0, {"span_or_Delta": (-95.0, -85.0)}, {})
    assert window._read_csa_sites()[0]["bounds"] == {"span_or_Delta": (-95.0, -85.0)}
    window._on_csa_fit_clicked()
    assert -95.0 <= window._csa_fit_result["popt"][1] <= -85.0
    assert "bound" in window.csa_results_label.text()           # pressed against it: said so


def test_clearing_constraints(mas_window):
    window = mas_window
    window._set_csa_row_constraints(0, {"fwhm": (1.0, 4.0)}, {"d_iso": True})
    window._set_csa_row_constraints(0, {}, {})
    site = window._read_csa_sites()[0]
    assert "fixed" not in site and "bounds" not in site
    assert not window.csa_sites_table.item(0, 1).font().italic()


# ---- results table ------------------------------------------------------

def test_results_are_a_table_with_uncertainties(mas_window):
    window = mas_window
    window._on_csa_fit_clicked()
    table = window.csa_results_table
    assert table.isVisibleTo(window) or not table.isHidden()
    headers = [table.horizontalHeaderItem(c).text().replace("\n", " ") for c in range(table.columnCount())]
    assert headers == ["δiso (ppm)", "Δ (ppm)", "η", "FWHM (ppm)", "Amplitude", "Area (%)"]
    assert table.rowCount() == 1 and table.verticalHeaderItem(0).text() == "A"
    assert "\n± " in table.item(0, 1).text()
    assert table.item(0, 5).text() == "100.0"
    tsv = window._table_as_tsv(table, "Site").splitlines()
    assert tsv[0].split("\t")[0] == "Site" and len(tsv) == 2 and " ± " in tsv[1]


def test_herzfeld_berger_headers(mas_window):
    window = mas_window
    window.csa_convention_combo.setCurrentIndex(window.csa_convention_combo.findData("Herzfeld-Berger"))
    window._csa_fit_result = None
    window.csa_sites_table.setItem(0, 2, window.csa_sites_table.item(0, 2).clone())
    window.csa_sites_table.item(0, 2).setText("100")
    window.csa_sites_table.item(0, 3).setText("0.3")
    window._on_csa_fit_clicked()
    headers = [window.csa_results_table.horizontalHeaderItem(c).text().replace("\n", " ")
               for c in range(window.csa_results_table.columnCount())]
    assert headers[1:3] == ["Span (ppm)", "Skew"]


def test_two_sites_report_their_area_shares(qapp):
    window = MainWindow()
    x = np.linspace(-300.0, 300.0, 2048)
    sim = lambda d, amp: amp * C.csa_model_full(x, d, -100.0, 0.5, 2.0, static=False,
                                                 larmor_Hz=1e8, mas_Hz=MAS_HZ,
                                                 powder_scheme="zcw", n_zcw=400)
    y = sim(-20.0, 3.0) + sim(20.0, 1.0)
    ppm = x[::-1]
    _add(window, NMRSpectrum(data=y[::-1], axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 100)],
                             metadata=NMRMetadata(vendor="test", nucleus="13C", sfo1=100.0)))
    window.csa_static_combo.setCurrentIndex(window.csa_static_combo.findData(False))
    window.csa_mas_spin.setValue(MAS_HZ)
    window.csa_orientations_spin.setValue(200)
    window.csa_render_orientations_spin.setValue(400)
    window._add_csa_site(("A", "-18", "-90", "0.4", "3"))
    window._add_csa_site(("B", "18", "-90", "0.4", "3"))
    window._on_csa_fit_clicked()
    shares = [float(window.csa_results_table.item(r, 5).text()) for r in range(2)]
    assert shares[0] == pytest.approx(75.0, abs=2.0) and shares[1] == pytest.approx(25.0, abs=2.0)


# ---- the plot -----------------------------------------------------------

def test_the_fit_view_can_be_found_again_and_is_labelled(mas_window):
    window = mas_window
    window._on_csa_fit_clicked()
    tabs = window.canvas_tabs
    assert tabs.isTabVisible(tabs.indexOf(window.fit_canvas))
    assert "13" in window.fit_canvas.ax.get_xlabel() and "C" in window.fit_canvas.ax.get_xlabel()
