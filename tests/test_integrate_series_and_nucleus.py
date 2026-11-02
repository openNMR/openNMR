"""Two user reports, 2026-11-02, on a 7Li saturation-recovery pseudo-2D:

* After extracting a slice, keeping it and phasing it, Integrate could not
  give a T1: the slice integrates alone (one value, so "Export Integration to
  Pseudo-2D" stays greyed out) and its regions live on its own table.
  The Integration panel now offers the whole series with the same regions.
* "Suggest nucleus (500 MHz console / 100 MHz magnet)" / "Override nucleus"
  changed 1D spectra but not a pseudo-2D's plot axis label or its slice names.
"""
import numpy as np

from test_gui import (  # noqa: F401 -- fixtures and helpers
    MainWindow, _isolate_sidecars, _keep_a_slice, _make_raw_2d_dataset, qapp)


def _window(tmp_path, monkeypatch, n_f1=5):
    _isolate_sidecars(monkeypatch, tmp_path)
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=n_f1, n_f2=512), vendor="bruker")
    return window


def _whole_axis(window, index=0):
    """The middle half of the axis (a baseline needs points outside the region),
    to 3 decimals so the region table holds it exactly."""
    x = np.asarray(window.session.spectra[index].axes[-1].ppm, dtype=float)
    span = x.max() - x.min()
    return round(float(x.min() + span / 4), 3), round(float(x.max() - span / 4), 3)


def test_integrating_the_series_directly_enables_the_export(qapp, tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._select_single_row(0)
    window._add_region_row(*_whole_axis(window))
    window._on_integrate_regions_clicked()
    assert len(window.session.spectra[0].metadata.raw["slice_fit_results"]) == 5
    assert window.export_integration_pseudo2d_button.isEnabled()
    assert window.integrate_series_button.isHidden()          # nothing to offer on the series


def test_an_extracted_slice_integrates_alone_and_says_why(qapp, tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    kept = _keep_a_slice(window, index=2)
    assert window._selected_index() == 1
    window._add_region_row(*_whole_axis(window, 1))
    window._on_integrate_regions_clicked()
    assert len(kept.metadata.raw["slice_fit_results"]) == 1
    assert not window.export_integration_pseudo2d_button.isEnabled()      # one slice is no series
    assert not window.integrate_series_button.isHidden()
    assert "one slice of" in window.integration_series_hint_label.text()
    assert "5 slices" in window.integration_series_hint_label.text()


def test_integrate_the_whole_series_from_a_slice_carries_the_regions(qapp, tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    _keep_a_slice(window, index=2)
    lo, hi = _whole_axis(window, 1)
    window._add_region_row(lo, hi)
    window.integrate_series_button.click()
    assert window._selected_index() == 0                                   # the series is selected
    assert window._read_region_defs() == [(min(lo, hi), max(lo, hi))]      # with the slice's regions
    results = window.session.spectra[0].metadata.raw["slice_fit_results"]
    assert len(results) == 5
    assert window.export_integration_pseudo2d_button.isEnabled()
    assert "Integrated 5 slice(s)" in window.integration_results_label.text()
    assert window.integrate_series_button.isHidden()


def test_the_series_button_needs_regions_and_a_live_parent(qapp, tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    kept = _keep_a_slice(window)
    window.region_table.setRowCount(0)
    window.integrate_series_button.click()                                 # no regions: says so
    assert window._selected_index() == 1                                   # stays where it was
    window.session.remove(0)                                               # the series is gone
    window._refresh_integration_series_hint(kept)
    assert window.integrate_series_button.isHidden()


def test_changing_the_nucleus_of_a_pseudo_2d_updates_its_plot_and_slices(qapp, tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._select_single_row(0)
    assert window.compare_list.count() == 5
    assert all("1H" in window.compare_list.item(i).text() for i in range(5))
    window._set_nucleus(0, "7Li")
    assert window.session.spectra[0].metadata.nucleus == "7Li"
    assert "7Li" in window.session.label(0)
    # The slice names follow ...
    assert all("7Li" in window.compare_list.item(i).text()
               and "1H" not in window.compare_list.item(i).text() for i in range(5))
    # ... and so does the axis label of the plot already on screen.
    labels = " ".join(ax.get_xlabel() for ax in window.canvas.figure.axes)
    assert "Li" in labels and "{H}" not in labels


def test_suggesting_the_nucleus_from_the_menu_needs_no_message_box(qapp, tmp_path, monkeypatch):
    """The confirmation box crashed the Windows app (segmentation fault inside
    QMessageBox.question, 2026-11-02): the menu item names the nucleus and applies it."""
    import dataclasses

    from PySide6.QtWidgets import QMessageBox
    window = _window(tmp_path, monkeypatch)
    window.session.set_nucleus(0, "189Os")
    meta = dataclasses.replace(window.session.spectra[0].metadata, sfo1=16.54845 * 2.35)
    window.session.spectra[0] = dataclasses.replace(window.session.spectra[0], metadata=meta)

    def boom(*args, **kwargs):
        raise AssertionError("a message box was opened")
    for name in ("question", "information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, boom)
    text, enabled = window._suggest_nucleus_menu_item(0)
    assert enabled and "7Li" in text
    window._on_suggest_nucleus_action(0, confirm=False)
    assert window.session.spectra[0].metadata.nucleus == "7Li"
    assert "7Li" in window.statusBar().currentMessage()
    text, enabled = window._suggest_nucleus_menu_item(0)          # now it is already right
    assert not enabled and "already 7Li" in text


def test_the_menu_item_is_disabled_when_nothing_can_be_suggested(qapp, tmp_path, monkeypatch):
    import dataclasses

    from PySide6.QtWidgets import QMessageBox
    window = _window(tmp_path, monkeypatch)
    meta = dataclasses.replace(window.session.spectra[0].metadata, sfo1=None)
    window.session.spectra[0] = dataclasses.replace(window.session.spectra[0], metadata=meta)
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("box")))
    text, enabled = window._suggest_nucleus_menu_item(0)
    assert not enabled and "SFO1" in text
    window._on_suggest_nucleus_action(0, confirm=False)           # a status-bar note, no box
    assert "No nucleus suggestion" in window.statusBar().currentMessage()
