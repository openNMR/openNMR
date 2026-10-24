"""Strong J coupling (roofing) in Solution NMR -- Fit and Simulate -- as well
as in Solid-state NMR (explicit user request, 2026-10-24).

There is one strongly coupled spin-system panel per tab (one set of spins,
Js and fit results); it sits in whichever section chose it last.
"""

import numpy as np
import pytest
from scipy.signal import find_peaks

from test_gui import qapp  # noqa: F401 -- fixture
from opennmr.gui.jcoupling_panels import SOLUTION_STRONG_BLURB, STRONG_COUPLING_BLURB
from opennmr.gui.main_window import MainWindow


@pytest.fixture
def window(qapp):
    return MainWindow()


def test_solution_fit_offers_both_models(window):
    model = window.solution_fit_model
    assert [model.model_combo.itemData(i) for i in range(model.model_combo.count())] == [
        "peaks", "spin_system"]
    assert model.model() == "peaks"
    # The peak table is the first page, as before.
    assert model.peaks_widget.isAncestorOf(window.peak_table)
    # Solution NMR heads Fit, Solid-state NMR follows.
    assert window._solution_fit_section.isAncestorOf(window.solution_fit_model)


def test_fit_panel_moves_between_solution_and_solid(window):
    panel, host = window.jcoupling_fit_panel, window._fit_spin_host
    solid, solution = host.slots["solid"], host.slots["solution"]
    assert host.owner == "solid" and solid.holds_panel() and not solution.holds_panel()
    window.solution_fit_model.model_combo.setCurrentIndex(
        window.solution_fit_model.model_combo.findData("spin_system"))
    assert host.owner == "solution" and solution.holds_panel() and not solid.holds_panel()
    assert window._solution_fit_section.isAncestorOf(panel)
    assert "Solution NMR" in solid.away_label.text()     # the Solid slot says where it went
    assert not solid.away_label.isHidden() and not solid.bring_button.isHidden()
    # Solid-state NMR's strongly coupled model takes it back, state intact.
    panel.spin_editor.set_system(
        [{"label": "A", "nuclide": "1H", "shift_ppm": 2.0, "n": 1},
         {"label": "B", "nuclide": "1H", "shift_ppm": 2.05, "n": 1}], [])
    window.combined_fit_model.show_model("spin_system")
    assert host.owner == "solid" and solid.holds_panel()
    assert panel.spin_editor.labels() == ["A", "B"]
    # The away note offers the way back.
    solution.bring_button.click()
    assert host.owner == "solution" and solution.holds_panel()
    assert panel.spin_editor.labels() == ["A", "B"]
    # The panel keeps its own blurb for each home.
    assert panel.blurb_label.text() == SOLUTION_STRONG_BLURB
    window.combined_fit_model.show_model("spin_system")
    assert panel.blurb_label.text() == STRONG_COUPLING_BLURB


def test_kept_fit_restores_where_the_panel_is(window):
    window.solution_fit_model.show_model("spin_system")
    assert window._fit_spin_host.owner == "solution"
    window._show_spin_system_model("fit")
    assert window._fit_spin_host.owner == "solution"
    assert window.solution_fit_model.model() == "spin_system"


def test_solution_simulation_shows_roofing(window):
    section = window._solution_sim_section
    assert section.title() == "Solution NMR"
    sim = window.jcoupling_sim_panel
    host = window._sim_spin_host
    assert host.owner == "solid"
    section.setExpanded(True)                    # opening Solution NMR brings it there
    assert host.owner == "solution" and section.isAncestorOf(sim)
    sim.nucleus_combo.setCurrentText("1H")
    sim.field_spin.setValue(400.0 / 42.577478)
    sim.spin_editor.set_system(
        [{"label": "A", "nuclide": "1H", "shift_ppm": 2.00, "n": 1},
         {"label": "B", "nuclide": "1H", "shift_ppm": 2.05, "n": 1}],
        [{"a": "A", "b": "B", "J_Hz": 12.0}])
    assert sim.simulate()
    y = np.asarray(sim.spectrum.data)
    x = np.asarray(sim.spectrum.axes[0].ppm)
    peaks, _ = find_peaks(y, height=0.05 * y.max())
    assert len(peaks) == 4                       # an AB quartet, not two doublets
    heights = y[peaks[np.argsort(x[peaks])]]
    assert heights[1] > heights[0] and heights[2] > heights[3]    # roofing


def test_solid_state_keeps_strong_coupling(window):
    window.combined_sim_model.show_model("spin_system")
    assert window._sim_spin_host.owner == "solid"
    assert window.combined_sim_model.spin_widget.holds_panel()
    # Closing and reopening Solution NMR does not take it from a Solid-state
    # NMR that is using it.
    window._solution_sim_section.setExpanded(True)
    assert window._sim_spin_host.owner == "solid"


def test_simulation_to_fit_lands_in_the_matching_section(window):
    window._solution_sim_section.setExpanded(True)
    sim = window.jcoupling_sim_panel
    sim.nucleus_combo.setCurrentText("1H")
    sim.field_spin.setValue(400.0 / 42.577478)
    sim.spin_editor.set_system(
        [{"label": "A", "nuclide": "1H", "shift_ppm": 2.00, "n": 1},
         {"label": "B", "nuclide": "1H", "shift_ppm": 2.05, "n": 1}],
        [{"a": "A", "b": "B", "J_Hz": 12.0}])
    assert window._fit_spin_host.owner == "solid"
    sim.send_to_fit()
    assert window._fit_spin_host.owner == "solution"
    assert window.solution_fit_model.model() == "spin_system"
    assert window.jcoupling_fit_panel.spin_editor.labels() == ["A", "B"]
