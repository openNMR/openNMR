"""User requests of 2026-11-06 (afternoon), Simulate ▸ SIMPSON:

* a heatmap when two quantities are swept together -- from a table with one
  row per combination (``O1 O2 Re Im``, as the user's colleague's
  ``O1_O2_CSA_wide_10.txt`` / ``2D_plot_norm.py`` pivot), or from one FID per
  combination with the two values in each file name;
* the example REDOR file (``42c1_xy8_500_60kHz_aRFDR.fid``) is a SIMP FID --
  NP=200, SW=300000, imaginary column zero -- read here by a file of the same
  layout (the user's data are not copied into the tests).
"""

import os

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

import opennmr.notebook as nmr
from opennmr.gui.main_window import MainWindow
from opennmr.notebook import simpson_series as ss

import simpson_files as sf
from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


O1 = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
O2 = np.array([-5.0, 0.0, 5.0])


def _value(o1, o2):
    return 0.5 - 0.001 * o1 ** 2 + 0.01 * o2


def _table(tmp_path, header=True, drop=None, name="O1_O2_CSA_wide_10.txt"):
    lines = ["O1 O2 Re Im"] if header else []
    for o2 in O2:                                  # rows in any order
        for o1 in O1[::-1]:
            if drop == (o1, o2):
                continue
            lines.append(f"{o1:g}\t{o2:g}\t{_value(o1, o2):.8f}\t0")
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# ---- reading and gridding ---------------------------------------------------------

def test_table_with_header_pivots_like_pandas(tmp_path):
    table = nmr.SIMPSONread_scan(_table(tmp_path))
    assert table["names"] == ["O1", "O2", "Re", "Im"]
    grid = nmr.SIMPSONscan_from_table(_table(tmp_path), "O2", "O1", "Re")
    assert np.array_equal(grid["xs"], O2) and np.array_equal(grid["ys"], O1)
    expected = np.array([[_value(o1, o2) for o2 in O2] for o1 in O1])     # Z[j, i] = (O1_j, O2_i)
    assert np.allclose(grid["Z"], expected) and grid["missing"] == 0


def test_headerless_table_and_column_numbers(tmp_path):
    grid = nmr.SIMPSONscan_from_table(_table(tmp_path, header=False), 0, 1, 2)
    assert grid["xname"] == "c1" and grid["zname"] == "c3"
    assert grid["Z"].shape == (O2.size, O1.size)


def test_a_missing_combination_is_left_empty(tmp_path):
    grid = nmr.SIMPSONscan_from_table(_table(tmp_path, drop=(10.0, 5.0)), "O1", "O2", "Re")
    assert grid["missing"] == 1
    j, i = list(O2).index(5.0), list(O1).index(10.0)
    assert np.isnan(grid["Z"][j, i])


def test_a_combination_given_twice_is_refused():
    with pytest.raises(ValueError, match="appears twice"):
        nmr.scan_grid([1, 2, 1], [0, 0, 0], [1, 2, 3])


def test_rounded_scan_values_collapse():
    with pytest.raises(ValueError, match="appears twice"):
        nmr.scan_grid([0.1 + 0.2, 0.3], [0, 0], [1, 2])
    grid = nmr.scan_grid([0.1 + 0.2, 0.6, 0.3, 0.6], [0, 0, 1, 1], [1, 2, 3, 4])
    assert grid["xs"].size == 2 and grid["missing"] == 0


@pytest.mark.parametrize("bad, kwargs, match", [
    ("O1 O2 Re\n1 2 3 4\n", {"header": True}, "one name per column"),
    ("O1 O2 Re\n1 2 3 4\n", {"skip_rows": 0, "header": True}, "does not name 4 columns"),
    ("1 2 3\n1 2\n", {}, "same number of columns"),
    ("1 2 3\n# note\n1 2\n", {}, "line 1 has 3, line 3 has 2"),
    ("1 2 3\n1 2\n", {"skip_rows": 0}, "same number of columns"),
    ("O1 O2 Re\n1 2 x\n", {}, "line 2"),
])
def test_bad_tables_say_why(tmp_path, bad, kwargs, match):
    path = tmp_path / "bad.txt"
    path.write_text(bad)
    with pytest.raises(ValueError, match=match):
        nmr.SIMPSONread_scan(str(path), **kwargs)


def test_a_header_that_does_not_fit_is_passed_over_and_said(tmp_path):
    path = tmp_path / "scan.txt"
    path.write_text("O1 O2 Re\n1 2 3 4\n5 6 7 8\n")
    table = nmr.SIMPSONread_scan(str(path))
    assert table["names"] == ["c1", "c2", "c3", "c4"] and table["skipped"] == 1


def test_unknown_column_names_the_choices(tmp_path):
    with pytest.raises(ValueError, match="the columns are O1, O2, Re, Im"):
        nmr.SIMPSONscan_from_table(_table(tmp_path), "O1", "O3", "Re")


@pytest.mark.parametrize("name, numbers", [
    ("O1_10_O2_-5.fid", [1, 10, 2, -5]),
    ("off_-20_rf_50.fid", [-20, 50]),
    ("scan_10.5_-2.5e3.spe", [10.5, -2500]),
    ("rf50_off-20.fid", [50, -20]),
])
def test_numbers_in_names(name, numbers):
    assert ss.numbers_in_name(name) == pytest.approx(numbers)


def _scan_files(tmp_path):
    paths = []
    for o1 in O1:
        for o2 in O2:
            paths.append(sf.write_point_fid(tmp_path / f"off_{o1:g}_rf_{o2:g}.fid",
                                            [_value(o1, o2) + 0.2j, 0.1]))
    return paths


def test_grid_from_one_fid_per_combination(tmp_path):
    grid = nmr.SIMPSONscan_from_files(_scan_files(tmp_path), xname="offset", yname="rf")
    assert np.array_equal(grid["xs"], O1) and np.array_equal(grid["ys"], O2)
    expected = np.array([[_value(o1, o2) for o1 in O1] for o2 in O2])
    assert np.allclose(grid["Z"], expected)
    grid = nmr.SIMPSONscan_from_files(_scan_files(tmp_path), expression="A", part="imag")
    assert np.allclose(grid["Z"], 0.2)
    with pytest.raises(ValueError, match="not enough"):
        nmr.SIMPSONscan_from_files(_scan_files(tmp_path), x_number=0, y_number=5)


def test_heatmap_plot(tmp_path):
    grid = nmr.SIMPSONscan_from_table(_table(tmp_path, drop=(10.0, 5.0)), "O2", "O1", "Re")
    fig, ax = nmr.SIMPSONplot_heatmap(grid, zlabel="FID first point intensity",
                                      xlabel="Offset 2 (kHz)", vmin=-0.25, vmax=0.55)
    mesh = ax.collections[0]
    assert mesh.get_array().mask.sum() == 1                  # the missing cell is blank
    assert mesh.norm.vmin == -0.25 and mesh.norm.vmax == 0.55
    assert len(fig.axes) == 2                                 # with its colour bar
    assert ax.get_xlabel().startswith("Offset 2")
    with pytest.raises(ValueError, match="below the maximum"):
        nmr.SIMPSONplot_heatmap(grid, vmin=1, vmax=0)
    table = nmr.SIMPSONscan_table(grid)
    assert len(table) == O1.size * O2.size - 1


# ---- the REDOR example layout ---------------------------------------------------------

def test_redor_example_layout(tmp_path):
    y = np.r_[0, 0.0187344929, 0.0654078819, 0.121654772, 0.176390989, 0.22786907,
              np.full(194, 0.288)]
    path = sf._write(tmp_path / "42c1_xy8_500_60kHz_aRFDR.fid", "FID", y.astype(complex),
                     sw=300000)
    curve = nmr.SIMPSONcurve(path, x="file", time_unit="us")
    assert curve["n"] == 200 and curve["rate_Hz"] == 300000.0
    assert curve["x"][1] == pytest.approx(1e6 / 300000)       # 3.33 us per point (SW)
    assert np.allclose(curve["values"].real, y)
    # The user's own x: e.g. rotor periods at 60 kHz, 0.2 of one per point.
    own = nmr.SIMPSONcurve(path, x="steps", start=0, step=0.2)
    assert own["x"][-1] == pytest.approx(39.8)


# ---- the GUI section ----------------------------------------------------------------

def test_gui_heatmap_from_a_table(qapp, tmp_path):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([_table(tmp_path)])
    assert panel.heatmap_read_button.isEnabled()
    panel.heatmap_read_button.click()
    assert [panel.heatmap_x_combo.itemText(i) for i in range(4)] == ["O1", "O2", "Re", "Im"]
    panel.heatmap_x_combo.setCurrentText("O2")
    panel.heatmap_y_combo.setCurrentText("O1")
    panel.heatmap_vmin_edit.setText("-0.25")
    panel.heatmap_vmax_edit.setText("0.55")
    panel.plot_heatmap_button.click()
    fig = window.fit_canvas.figure
    assert len(fig.axes) == 2
    mesh = fig.axes[0].collections[0]
    assert mesh.norm.vmin == -0.25
    assert "3 O2 × 5 O1" in panel.heatmap_preview_label.text()


def test_gui_heatmap_from_files_and_its_code_runs(qapp, tmp_path):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths(_scan_files(tmp_path))
    panel.heatmap_source_combo.setCurrentIndex(panel.heatmap_source_combo.findData("files"))
    assert panel.heatmap_x_number_spin.isVisibleTo(panel)
    assert not panel.heatmap_x_combo.isVisibleTo(panel)
    panel.plot_heatmap_button.click()
    assert np.allclose(panel.grid["xs"], O1)
    namespace = {}
    exec(panel.heatmap_notebook_code(), namespace)
    assert np.allclose(namespace["grid"]["Z"], panel.grid["Z"], equal_nan=True)


def test_gui_heatmap_problems_are_reported(qapp, tmp_path, _no_modal_dialogs):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([_table(tmp_path)])
    panel.heatmap_vmin_edit.setText("abc")
    panel.plot_heatmap_button.click()
    assert _no_modal_dialogs and "not a number" in _no_modal_dialogs[-1][1]


def test_gui_heatmap_settings_are_kept(qapp, tmp_path):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([_table(tmp_path)])
    panel.heatmap_read_button.click()
    panel.heatmap_x_combo.setCurrentText("O2")
    panel.heatmap_cmap_combo.setCurrentText("magma")
    other = MainWindow().simpson_panel
    other.set_state(panel.get_state())
    assert other.heatmap_x_combo.currentText() == "O2"
    assert other.heatmap_cmap_combo.currentText() == "magma"


# ---- lines above the data: titles, comments, headers (2026-11-06, 16:04) -------------

_BODY = "".join(f"{o1:g} {o2:g} {_value(o1, o2):.8f} 0\n" for o2 in O2 for o1 in O1)


@pytest.mark.parametrize("top, names, skipped", [
    ("", ["c1", "c2", "c3", "c4"], 0),
    ("O1 O2 Re Im\n", ["O1", "O2", "Re", "Im"], 0),
    ("# O1 O2 Re Im\n", ["O1", "O2", "Re", "Im"], 0),
    ("Offset scan, CSA wide, 10 kHz steps\nO1 O2 Re Im\n", ["O1", "O2", "Re", "Im"], 1),
    ("Simulation run 2026-02-23 by Joe\n\n", ["c1", "c2", "c3", "c4"], 1),
    ("Offset 1 (kHz)\tOffset 2 (kHz)\tRe\tIm\n",
     ["Offset 1 (kHz)", "Offset 2 (kHz)", "Re", "Im"], 0),
    ("np 15\nsw 1e5\nO1,O2,Re,Im\n", ["O1", "O2", "Re", "Im"], 2),
])
def test_lines_above_the_data_are_found(tmp_path, top, names, skipped):
    path = tmp_path / "scan.txt"
    path.write_text(top + _BODY)
    table = nmr.SIMPSONread_scan(str(path))
    assert table["names"] == names and table["skipped"] == skipped
    assert len(table["columns"][names[0]]) == O1.size * O2.size


def test_skip_and_header_can_be_given(tmp_path):
    path = tmp_path / "scan.txt"
    path.write_text("a title that has four words\nO1 O2 Re Im\n" + _BODY)
    # Auto-detect takes the line above the numbers as names.
    assert nmr.SIMPSONread_scan(str(path))["names"] == ["O1", "O2", "Re", "Im"]
    table = nmr.SIMPSONread_scan(str(path), skip_rows=1, header=True)
    assert table["names"] == ["O1", "O2", "Re", "Im"] and table["header_line"] == 2
    table = nmr.SIMPSONread_scan(str(path), skip_rows=2, header=False)
    assert table["names"] == ["c1", "c2", "c3", "c4"] and table["first_data_line"] == 3
    table = nmr.SIMPSONread_scan(str(path), header=False)
    assert table["names"][0] == "c1" and table["skipped"] == 2
    with pytest.raises(ValueError, match="not a row of numbers"):
        nmr.SIMPSONread_scan(str(path), skip_rows=0, header=False)
    with pytest.raises(ValueError, match="not a row of numbers"):
        nmr.SIMPSONread_scan(str(path), skip_rows=0, header=True)   # line 2 is not data
    with pytest.raises(ValueError, match="cannot skip"):
        nmr.SIMPSONread_scan(str(path), skip_rows=999)


def test_header_true_needs_names(tmp_path):
    path = tmp_path / "scan.txt"
    path.write_text(_BODY)
    with pytest.raises(ValueError, match="one name per column"):
        nmr.SIMPSONread_scan(str(path), header=True)


def test_trailing_text_is_reported(tmp_path):
    path = tmp_path / "scan.txt"
    path.write_text("O1 O2 Re Im\n" + _BODY + "END\n")
    with pytest.raises(ValueError, match="run to the end"):
        nmr.SIMPSONread_scan(str(path))


def test_gui_skip_and_header_controls(qapp, tmp_path):  # noqa: F811
    path = tmp_path / "scan.txt"
    path.write_text("Offset scan\nO1 O2 Re Im\n" + _BODY)
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([str(path)])
    panel.heatmap_read_button.click()
    text = panel.heatmap_preview_label.text()
    assert "O1, O2, Re, Im" in text and "passed over 1 line" in text and "line 2" in text
    panel.heatmap_header_combo.setCurrentIndex(panel.heatmap_header_combo.findData("data"))
    assert panel.heatmap_x_combo.itemText(0) == "c1"            # re-read at once
    panel.heatmap_skip_spin.setValue(2)
    panel.heatmap_x_combo.setCurrentText("c2")
    panel.heatmap_y_combo.setCurrentText("c1")
    panel.plot_heatmap_button.click()
    assert np.array_equal(panel.grid["xs"], O2) and np.array_equal(panel.grid["ys"], O1)
    code = panel.heatmap_notebook_code()
    assert "skip_rows=2" in code and "header=False" in code
    namespace = {}
    exec(code, namespace)
    assert np.allclose(namespace["grid"]["Z"], panel.grid["Z"])
