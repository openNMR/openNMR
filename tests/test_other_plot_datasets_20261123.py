"""2026-11-23: several datasets (files, columns, Dynamics curve fits) on one 2-D / 3-D plot."""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure
from PySide6.QtWidgets import QInputDialog

from opennmr.notebook import other_plot as op
from opennmr.processing import relaxation

from test_gui import qapp  # noqa: F401 -- fixture


def _fit(tau=1.3, seed=0):
    t = np.linspace(0.1, 5, 10)
    y = 1 - np.exp(-t / tau) + np.random.default_rng(seed).normal(0, 0.005, t.size)
    return relaxation.fit_relaxation(t, y)


def test_two_datasets_each_with_its_own_kind_and_colour():
    sets = [op.dataset([0, 1, 2], [1, 2, 3], label="A", kind="scatter", colour="red"),
            op.dataset([0, 1, 2], [3, 2, 1], label="B", kind="line", colour="blue")]
    fig = op.plot_datasets(sets, xlabel="x", ylabel="y")
    ax = fig.axes[0]
    assert len(ax.collections) == 1 and len(ax.lines) == 1
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B"]
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "y"


def test_bars_of_several_datasets_sit_side_by_side():
    sets = [op.dataset([1, 2, 3], [1, 2, 3], kind="bar"), op.dataset([1, 2, 3], [3, 2, 1], kind="bar")]
    ax = op.plot_datasets(sets).axes[0]
    a, b = ax.patches[0], ax.patches[3]
    assert a.get_x() + a.get_width() / 2 < b.get_x() + b.get_width() / 2     # not on top of each other


def test_3d_datasets_and_mixing_is_refused():
    fig = op.plot_datasets([op.dataset([1, 2], [1, 2], [3, 4], label="a"),
                            op.dataset([1, 2], [2, 3], [1, 2], label="b", kind="line")])
    assert fig.axes[0].name == "3d"
    with pytest.raises(ValueError, match="cannot be mixed"):
        op.plot_datasets([op.dataset([1], [1]), op.dataset([1], [1], [1])])
    with pytest.raises(ValueError):
        op.plot_datasets([])
    with pytest.raises(ValueError):
        op.dataset([1, 2], [1])


def test_error_bars_and_spines():
    ax = op.plot_datasets([op.dataset([1, 2, 3], [1, 2, 3], yerr=[.1, .1, .1])],
                          top_spine=True, right_spine=True).axes[0]
    assert ax.containers                      # an errorbar container
    assert ax.spines["top"].get_visible() and ax.spines["right"].get_visible()


def test_curve_fit_datasets_points_and_curve_and_peak_choice():
    results = [_fit(1.3), _fit(2.0, seed=1)]
    sets = op.curve_fit_datasets(results)
    assert [d["kind"] for d in sets] == ["scatter", "line", "scatter", "line"]
    assert sets[0]["colour"] == sets[1]["colour"] != sets[2]["colour"]
    assert len(sets[1]["x"]) == 400 and sets[0]["x"].size == 10
    only = op.curve_fit_datasets(results, peak=1, curve=False)
    assert len(only) == 1 and only[0]["kind"] == "scatter"
    # the curve is the fitted function, not the points joined up
    tf, yf = relaxation.fit_curve(results[0])
    np.testing.assert_allclose(sets[1]["y"], yf)
    from_raw = op.curve_fit_datasets({"relaxation_fit_results": results})
    assert len(from_raw) == 4
    with pytest.raises(ValueError, match="no curve fit"):
        op.curve_fit_datasets({})
    with pytest.raises(ValueError, match="no peak"):
        op.curve_fit_datasets(results, peak=5)


def test_curve_fit_and_a_file_on_one_figure():
    sets = op.curve_fit_datasets([_fit()], label="T1") + [op.dataset([1, 2, 3], [.2, .4, .6], label="other")]
    ax = op.plot_datasets(sets, xlabel="Delay (s)").axes[0]
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts == ["T1 Peak 1 data", "T1 Peak 1 fit", "other"]


def _panel(**kw):
    from opennmr.gui.other_plot_panel import OtherPlotPanel
    drawn = []
    return OtherPlotPanel(draw=drawn.append, **kw), drawn


def _items(panel):
    tree = panel.dataset_tree
    return [[tree.topLevelItem(i).text(c) for c in range(4)] for i in range(tree.topLevelItemCount())]


def test_number_list():
    assert op.number_list("1, 2 3;4") == [1, 2, 3, 4]
    assert op.number_list("0:10:2") == [0, 2, 4, 6, 8, 10]
    assert op.number_list("1:3") == [1, 2, 3] and op.number_list("5:1:-2") == [5, 3, 1]
    assert op.number_list([1, 2]) == [1.0, 2.0] and op.number_list("") == []
    for bad in ("1 a", "1:2:0", "3:1"):
        with pytest.raises(ValueError):
            op.number_list(bad)


def test_two_files_and_a_custom_list_make_three_datasets(qapp):  # noqa: F811
    panel, drawn = _panel()
    panel.add_pasted("T\ta\tb\n0\t0\t3\n1\t2\t5\n2\t3\t8\n")
    panel.add_pasted("x\ty\n1\t1\n2\t4\n3\t9\n")
    assert [s["name"] for s in panel.sources] == ["pasted", "pasted 2"]
    assert len(panel.datasets) == 1                          # the first file brought the first dataset
    panel.add_dataset(source=1)                              # file 2: x against y
    panel.add_dataset(source=0, x="T", y="b")                # file 1: another pair of columns
    panel.add_dataset(source=None, x=[1, 2, 3], y=[2, 2, 2])  # typed numbers, no file
    rows = _items(panel)
    assert [r[1] for r in rows] == ["pasted: a vs T", "pasted 2: y vs x", "pasted: b vs T",
                                    "own numbers: custom list vs custom list"]
    panel.plot()
    assert "Plotted 4" in panel.report_label.text()
    figure = Figure()
    drawn[-1](figure)
    assert len(figure.axes[0].get_legend().get_texts()) == 4


def test_editing_a_dataset_changes_its_columns_list_name_and_look(qapp):  # noqa: F811
    panel, drawn = _panel()
    panel.add_pasted("T\ta\tb\n0\t0\t3\n1\t2\t5\n2\t3\t8\n")
    panel.y_combo.setCurrentIndex(panel.y_combo.findData("col:b"))
    assert panel.datasets[0]["y"] == {"col": "b"} and _items(panel)[0][0] == "b"       # automatic name follows
    panel.x_combo.setCurrentIndex(panel.x_combo.findData("@list"))
    panel.x_list_edit.setText("10:12")
    panel.x_list_edit.editingFinished.emit()
    assert panel.datasets[0]["x"] == {"list": [10.0, 11.0, 12.0]}
    panel.x_combo.setCurrentIndex(panel.x_combo.findData("@rows"))
    assert panel.datasets[0]["x"] == {"rows": True}
    panel.name_edit.setText("my data")
    panel.name_edit.editingFinished.emit()
    panel.kind_combo.setCurrentIndex(panel.kind_combo.findData("bar"))
    assert _items(panel)[0][0] == "my data" and _items(panel)[0][2] == "Bar chart"
    sets = panel.build_datasets()
    assert sets[0]["label"] == "my data" and sets[0]["kind"] == "bar"
    assert list(sets[0]["x"]) == [1, 2, 3] and list(sets[0]["y"]) == [3, 5, 8]
    code = panel.notebook_code()
    assert "list(range(1, len(table1) + 1))" in code and "label='my data'" in code and "kind='bar'" in code
    compile(code, "<code>", "exec")                      # valid Python


def test_mismatched_or_incomplete_datasets_say_which(qapp):  # noqa: F811
    panel, drawn = _panel()
    panel.add_pasted("T\ta\n0\t1\n1\t2\n")
    panel.add_dataset(source=None, x=[1, 2, 3], y=[1, 2])
    panel.plot()
    assert "Dataset 2" in panel.report_label.text() and len(drawn) == 0
    panel.remove_dataset()
    panel.dim_combo.setCurrentIndex(panel.dim_combo.findData("3d"))
    panel.datasets[0]["z"] = None
    panel.plot()
    assert "needs a z" in panel.report_label.text()


def test_removing_a_file_keeps_the_other_datasets_pointing_right(qapp):  # noqa: F811
    panel, _ = _panel()
    panel.add_pasted("T\ta\n0\t1\n1\t2\n")
    panel.add_pasted("x\ty\n1\t1\n2\t4\n")
    panel.add_dataset(source=1)
    panel.source_list.setCurrentRow(0)
    panel.remove_file()
    assert [s["name"] for s in panel.sources] == ["pasted 2"]
    assert panel.datasets[0]["source"] is None and panel.datasets[1]["source"] == 0
    with pytest.raises(ValueError, match="Dataset 1"):           # lost its columns with the file
        panel.build_datasets()


def test_skip_lines_rereads_the_selected_file(qapp, tmp_path):  # noqa: F811
    path = tmp_path / "d.txt"
    path.write_text("T\ta\n0\t1\n1\t2\n2\t3\n")
    panel, _ = _panel()
    assert panel.add_file(str(path))
    assert list(panel.sources[0]["table"].columns) == ["T", "a"]
    panel.skip_spin.setValue(1)                      # leave out the header line
    panel.skip_spin.editingFinished.emit()
    table = panel.sources[0]["table"]
    assert panel.sources[0]["skip"] == 1 and list(table.columns) == ["col1", "col2"] and table.shape == (3, 2)


def test_curve_fit_is_added_from_the_open_experiments(qapp, monkeypatch):  # noqa: F811
    panel, drawn = _panel(curve_fits=lambda: [("sat rec", [_fit()], 2)])
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: (a[3][0], True)))
    panel.add_pasted("T\ta\n0\t1\n1\t2\n")
    panel.add_curve_fit()
    assert len(panel.datasets) == 3 and _items(panel)[1][1] == "curve fit of sat rec"
    panel.plot()
    figure = Figure()
    drawn[-1](figure)
    assert len(figure.axes[0].get_legend().get_texts()) == 3
    code = panel.notebook_code()
    assert "op.curve_fit_datasets(spectra[2]" in code
    panel.dim_combo.setCurrentIndex(panel.dim_combo.findData("3d"))
    panel.datasets[0]["z"] = {"col": "a"}
    panel.plot()
    assert "curve fit" in panel.report_label.text() and "2-D" in panel.report_label.text()


def test_panel_without_curve_fits_says_so(qapp):  # noqa: F811
    panel, _ = _panel()
    panel.add_curve_fit()
    assert "No open experiment" in panel.report_label.text()


def test_plot_style_has_the_data_plot_settings_and_a_toolbar_button(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    assert "Data plots" in window.style_sections
    window.other_marker_size_spin.setValue(11.0)
    window.other_colormap_combo.setCurrentIndex(window.other_colormap_combo.findData("plasma"))
    style = window._other_data_style()
    assert style["marker_size"] == 11.0 and style["colormap"] == "plasma"
    window.other_plot_panel.add_pasted("T\ta\n0\t1\n1\t2\n")
    window.other_plot_panel.plot()
    ax = window.compare_canvas.figure.axes[0]
    assert ax.collections and ax.collections[0].get_sizes()[0] == pytest.approx(121.0)
    window.other_marker_size_spin.setValue(4.0)             # Plot Style redraws the plot on show
    ax = window.compare_canvas.figure.axes[0]
    assert ax.collections[0].get_sizes()[0] == pytest.approx(16.0)
    button = window.compare_canvas._style_button
    assert button.text() == "Plot style"
    window.close()
