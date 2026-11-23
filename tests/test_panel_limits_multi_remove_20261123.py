"""2026-11-23: (1) Plot Style > Axis limits / Reset to full view work on Compare
panels; (2) removing several selected experiments is one batched removal (a
per-row redraw with a stale list crashed on macOS with 4 selected)."""
import pytest
from PySide6.QtWidgets import QAbstractItemView, QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.canvas import MplCanvas

from test_compare_panels import _spec
from test_gui import qapp  # noqa: F401 -- fixture
from test_gui_fit_sim_state import _quad_spectrum


def _panels():
    canvas = MplCanvas()
    canvas.show_compare_panels([_spec(2), _spec(5), _spec(8)], [[0, 1], [2]], layout="row",
                               legend_labels=["a", "b", "c"])
    return canvas


def test_panels_take_typed_limits_and_reset(qapp):  # noqa: F811
    canvas = _panels()
    assert canvas.view_limit_support() == (True, False)
    a, b = canvas._spectrum_panels
    full = a.get_xlim()
    canvas.set_view_limits(x=(1, 4))
    assert sorted(a.get_xlim()) == pytest.approx([1, 4])
    assert (a.get_xlim()[0] > a.get_xlim()[1]) == (full[0] > full[1])      # direction kept
    canvas.clear_view_limits()
    assert a.get_xlim() == pytest.approx(full)


def test_reset_undoes_a_mouse_zoom_on_panels(qapp):  # noqa: F811
    canvas = _panels()
    a, _b = canvas._spectrum_panels
    full = a.get_xlim()
    canvas.note_natural_view()
    a.set_xlim(6, 4)
    canvas.reset_view()
    assert a.get_xlim() == pytest.approx(full)


def test_removing_several_selected_experiments(qapp, monkeypatch):  # noqa: F811
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)
    window = MainWindow()
    for name in ("a", "b", "c", "d", "e"):
        _quad_spectrum(window, name)
    window.spectrum_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    window.spectrum_list.clearSelection()
    for row in (0, 1, 2, 3):
        window.spectrum_list.item(row).setSelected(True)
    window._on_remove_clicked()
    assert len(window.session) == 1 and window.spectrum_list.count() == 1
    assert window.spectrum_list.signalsBlocked() is False
    assert window._compare_pinned_entries == [] or all(r < 1 for r, _k in window._compare_pinned_entries)
    window._remove_rows([0])
    assert len(window.session) == 0 and window.spectrum_list.count() == 0
    window.close()
