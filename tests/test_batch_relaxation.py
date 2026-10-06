"""Batch pseudo-2D relaxation: many series processed and fitted like one
set up by hand, grouped by field, into Relaxation vs Temperature (user
request, 2026-10-06) -- notebook.batch_relaxation and Dynamics ▸ Batch."""

import numpy as np
import pytest

from opennmr.gui.main_window import MainWindow
from opennmr.notebook import batch_relaxation as br

from test_gui import (qapp, _load_spectrum_directly, _make_pseudo2d_relaxation_spectrum,  # noqa: F401
                      _set_delay_unit, _write_params)

DELAYS = np.geomspace(0.001, 0.3, 8)


# ---- notebook ----------------------------------------------------------------------

def test_temperature_and_field_from_acqus(tmp_path):
    _write_params(tmp_path / "acqus", {"TE": 298.15, "SFO1": 400.1324, "NUC1": "1H"})
    assert br.read_temperature_and_field(tmp_path) == (pytest.approx(298.15),
                                                       pytest.approx(400.1324))
    assert br.read_temperature_and_field(tmp_path / "missing") == (None, None)


def _row(series, T, value, field, quantity="T1"):
    return {"experiment": f"e{T}", "series": series, "T_K": T, "value": value,
            "error": 0.01 * value, "larmor_MHz": field, "quantity": quantity}


def test_rows_group_by_peak_and_field_within_a_tolerance():
    rows = [_row("a", 250, 0.1, 400.1301), _row("a", 300, 0.2, 400.1324),   # same field
            _row("a", 250, 0.3, 600.13), _row("b", 250, 0.4, 400.13)]
    groups = br.group_rows(rows)
    assert sorted((s, round(f)) for s, f in groups) == [("a", 400), ("a", 600), ("b", 400)]
    assert len(next(v for (s, f), v in groups.items() if s == "a" and round(f) == 400)) == 2
    # Tighter than the O1 spread: split.
    assert len(br.group_rows(rows, field_tolerance_MHz=1e-4)) == 4


def test_vt_datasets_and_csv():
    rows = [_row("a", T, 0.1 * k, 400.13) for k, T in enumerate((250, 275, 300), 1)]
    rows.append(_row("a", None, 1.0, 400.13))          # no temperature: left out
    (ds,) = br.batch_vt_datasets(rows)
    assert list(ds["T_K"]) == [250, 275, 300]
    assert list(ds["value"]) == pytest.approx([0.1, 0.2, 0.3])
    text = br.batch_table({"rows": rows})
    lines = text.strip().splitlines()
    assert lines[0].startswith("experiment,series,T_K") and len(lines) == 5
    assert ",T1," in lines[1]


def test_rows_from_fits_reads_the_rate_parameter():
    fits = [{"rate_param": "T1", "params": {"T1": 0.5, "M0": 1}, "errors": {"T1": 0.01},
             "peak_label": "CH3"},
            {"rate_param": "D", "params": {"D": 1e-9}}]                 # not relaxation
    rows = br.rows_from_fits(fits, "e1", 300.0, 400.13)
    assert rows == [{"experiment": "e1", "series": "CH3", "T_K": 300.0, "value": 0.5,
                     "error": 0.01, "larmor_MHz": 400.13, "quantity": "T1"}]


def test_a_failing_path_is_reported_not_dropped(tmp_path):
    out = br.NMR2D_relaxation_batch([str(tmp_path / "nothing")], integration="regions",
                                    regions=[(4.5, 5.5)], model="T2")
    assert out["rows"] == [] and len(out["failed"]) == 1
    assert out["runs"][0]["error"]
    with pytest.raises(ValueError, match="1 datasets"):
        br.NMR2D_relaxation_batch([str(tmp_path)], temperatures=[1, 2])


# ---- GUI -----------------------------------------------------------------------------

def _series(T2, T_K, sfo1, seed):
    sp = _make_pseudo2d_relaxation_spectrum(DELAYS, T2=T2, seed=seed)
    sp.metadata.sfo1 = sfo1
    sp.metadata.raw = {**(sp.metadata.raw or {}), "acqus": {"TE": T_K}}
    return sp


def _set_up(qapp, n=3):
    window = MainWindow()
    truth = [(0.02, 250.0, 400.13), (0.05, 300.0, 400.1302), (0.1, 350.0, 600.2)][:n]
    for k, (T2, T, f) in enumerate(truth):
        _load_spectrum_directly(window, _series(T2, T, f, seed=k))
    # The reference, set up by hand: the first series.
    window._select_single_row(0)
    window._add_region_row(4.5, 5.5)
    window.delay_source_combo.setCurrentIndex(1)
    _set_delay_unit(window, "s")
    window.manual_delays_edit.setText(", ".join(f"{d:.6g}" for d in DELAYS))
    window.relaxation_model_combo.setCurrentIndex(2)
    window._save_current_analysis_state()
    panel = window.batch_relaxation_panel
    panel.processing_check.setChecked(False)        # in-memory series: nothing to re-read
    return window, panel, truth


def test_gui_batch_fits_every_series_and_sends_grouped_rows(qapp):
    window, panel, truth = _set_up(qapp)
    keys = [window._multi_key(i) for i in range(len(window.session))]
    assert panel.add_keys(keys) == []
    assert [e["T_K"] for e in panel.entries] == [250.0, 300.0, 350.0]
    assert panel.run(show_progress=False)
    assert [e["status"] for e in panel.entries] == ["fitted"] * 3, \
        [e["result"] for e in panel.entries]
    values = [r["value"] for r in panel.rows]
    assert values == pytest.approx([t[0] for t in truth], rel=0.05)
    assert "2 group(s)" in panel.summary_label.text()          # 400.13 and 600.2 MHz
    vt = window.vt_relaxation_panel
    assert vt.table.rowCount() == 3
    assert window._vt_relaxation_section.isExpanded()
    assert panel.csv_button.isEnabled()


def test_gui_reference_delays_and_a_typed_temperature(qapp):
    window, panel, truth = _set_up(qapp, n=2)
    panel.add_keys([window._multi_key(i) for i in range(2)])
    panel.delays_combo.setCurrentIndex(panel.delays_combo.findData("reference"))
    panel.send_check.setChecked(False)
    panel.table.item(1, 1).setText("310")               # T column
    assert panel.entries[1]["T_K"] == 310.0
    assert panel.run(show_progress=False)
    assert panel.rows[1]["T_K"] == 310.0
    assert panel.rows[1]["value"] == pytest.approx(truth[1][0], rel=0.05)


def test_gui_a_series_that_cannot_be_measured_fails_alone(qapp):
    window, panel, _truth = _set_up(qapp, n=2)
    panel.add_keys([window._multi_key(i) for i in range(2)])
    real = window._batch_run_one

    def flaky(key, options):
        if key == window._multi_key(1):
            return {"rows": [], "status": "failed", "error": "no signal"}
        return real(key, options)
    panel._run_one = flaky
    panel.send_check.setChecked(False)
    assert panel.run(show_progress=False)
    assert panel.entries[1]["status"] == "failed" and "no signal" in panel.entries[1]["result"]
    assert "1 failed" in panel.summary_label.text()


def test_gui_needs_a_pseudo2d_reference(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    said = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: said.append(a[2]))
    window = MainWindow()
    panel = window.batch_relaxation_panel
    assert panel.run(show_progress=False) is False      # nothing selected: says so
    assert "reference" in said[0]


def test_gui_batch_applies_the_reference_processing_to_a_read_series(qapp, tmp_path,
                                                                      monkeypatch):
    """Processing on: a series read from disk is re-processed with the
    reference's settings (here a zero-order phase and a baseline method)
    before it is measured -- even when its measurement then fails."""
    from test_gui import _isolate_sidecars, _make_2d_dataset
    (tmp_path / "sidecars").mkdir()
    _isolate_sidecars(monkeypatch, tmp_path / "sidecars")
    window = MainWindow()
    for name in ("vt250", "vt300"):
        window._load(_make_2d_dataset(tmp_path, n_f1=4, n_f2=64, name=name), vendor="bruker")
    for i in range(2):
        window.session.reload(i, pseudo_2d=True)
    ref = window.session.spectra[0]
    ref.metadata.raw.update({"phase0": 37.0, "baseline_method": "spline_auto",
                             "baseline_points": 7})
    window._select_single_row(0)
    window._add_region_row(window.session.spectra[0].axes[-1].ppm.min(),
                           window.session.spectra[0].axes[-1].ppm.max())
    window._save_current_analysis_state()
    out = window._batch_run_one(window._multi_key(1),
                                {"reference": window._multi_key(0), "apply_processing": True,
                                 "delays": "own"})
    target = window.session.spectra[1].metadata.raw
    assert target.get("phase0") == 37.0
    assert target.get("baseline_method") == "spline_auto" and target.get("baseline_points") == 7
    assert target["analysis_state"]["regions"]               # the reference's regions
    assert out["status"] in ("fitted", "failed") and "rows" in out
