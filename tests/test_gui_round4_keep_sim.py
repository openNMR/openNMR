"""2026-09-29 (fourth round): simulations kept in Open experiments --
the buttons all say so, and any plot showing a simulation offers it on
right-click. Synthetic data only."""
import numpy as np
import pytest

from opennmr.gui.widgets.canvas import MplCanvas


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "_FIGURE_STYLES_PATH", str(tmp_path / "figure_styles.json"))
    return MainWindow()


def test_every_simulation_says_keep_in_open_experiments(window):
    from PySide6.QtWidgets import QPushButton
    texts = {b.text() for b in window.findChildren(QPushButton)}
    assert "Export to Open Experiments…" not in texts and "Add to Open Experiments" not in texts
    for button in (window.sim_csa_export_button, window.sim_quad_export_button,
                   window.czjzek_sim_panel.export_button,
                   window.jcoupling_sim_panel.export_button,
                   window.combined_sim_panel.export_button):
        assert button.text() == "Keep in Open Experiments"


def test_right_click_keeps_a_shown_simulation_once(window, monkeypatch):
    menus = []
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(MplCanvas, "_exec_context_menu", lambda self, m, p: menus.append(m))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(str(a[1:])))
    window.sim_csa_orientations_spin.setValue(200)
    window.sim_csa_points_spin.setValue(256)
    window.sim_csa_nucleus_edit.setText("13C")
    window._on_simulate_csa_clicked()
    spectrum = window.fit_canvas.simulation_spectrum
    assert spectrum is not None
    window.fit_canvas._on_context_menu(window.fit_canvas.rect().center())
    keep = [a for a in menus[-1].actions()
            if a.text() == "Keep This Simulation in Open Experiments"]
    assert keep
    n = len(window.session)
    keep[0].trigger()
    assert len(window.session) == n + 1
    assert window.session.label(n).startswith("Simulated")
    window._keep_simulation(spectrum)                 # again: selects, does not duplicate
    assert len(window.session) == n + 1 and window._selected_index() == n
    # Another view on the canvas: nothing to keep.
    window.fit_canvas.show_relaxation_fit(
        {"t": np.geomspace(0.01, 1, 5), "y": np.ones(5), "y_fit": np.ones(5),
         "peak_label": "p"}, xlabel="t", ylabel="I")
    assert window.fit_canvas.simulation_spectrum is None


def test_czjzek_simulation_plot_offers_keep(window):
    sim = window.czjzek_sim_panel
    sim.grid_cq_spin.setValue(16)
    sim.grid_eta_spin.setValue(6)
    sim.simulate()
    canvas = window.simulation_canvas
    assert canvas.simulation_spectrum is not None
    menu = canvas.context_menu()
    action = menu.actions()[0]
    assert action.text() == "Keep This Simulation in Open Experiments"
    n = len(window.session)
    action.trigger()
    assert len(window.session) == n + 1
    assert window.session.spectra[-1].metadata.raw["simulation"]["model"] == "Czjzek"
    # A different picture drawn there: nothing to keep.
    window._draw_tensor_picture(lambda figure: figure.add_subplot(111))
    assert canvas.context_menu() is None
