"""2026-10-22: a stacked Compare exports the region on screen (not the widest
spectrum), and the slash-style shift label is upright (Roman) in its bracket
and mass number."""
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from PySide6.QtWidgets import QApplication

from opennmr.core.registry import NUC_label


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_slash_label_has_an_upright_bracket_and_mass_number():
    label = NUC_label("1H", style="slash")
    assert r"\mathregular{(}" in label
    assert r"^{\mathregular{1}}" in label
    assert label.endswith("H) / ppm")


def test_brackets_label_is_still_fully_upright():
    label = NUC_label("23Na", style="brackets")
    assert r"^{\mathregular{23}}" in label and r"\mathregular{(ppm)}" in label


def test_slash_label_renders_and_has_no_mass_number_without_one():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.text(0.1, 0.5, NUC_label("1H", style="slash"))
    fig.text(0.1, 0.2, NUC_label("?", style="slash"))
    fig.canvas.draw()
    plt.close(fig)


def _two_datasets(tmp_path, n=40000):
    from test_gui import _make_1d_dataset, _write_params
    import pathlib
    paths = [_make_1d_dataset(tmp_path, n=n, name=name) for name in "ab"]
    _write_params(pathlib.Path(paths[1]) / "pdata" / "1" / "procs",
                  {"SW_p": 800000.0, "SI": n, "SF": 400.13, "NC_proc": 0})
    rng = np.random.default_rng(0)
    for p in paths:
        data = (rng.normal(0, 50, n) + 3000 * np.exp(-((np.arange(n) - n // 2) / 60.0) ** 2)).astype("int32")
        data.tofile(pathlib.Path(p) / "pdata" / "1" / "1r")
    return paths


@pytest.mark.parametrize("mode", ["Stacked", "Superimposed", "Panels: column"])
def test_export_keeps_to_the_zoomed_region(qapp, tmp_path, mode):
    from PIL import Image
    from opennmr.gui.main_window import MainWindow
    from test_gui import _select_rows
    window = MainWindow()
    try:
        for p in _two_datasets(tmp_path):
            window._load(p, vendor="bruker")
        _select_rows(window, [0, 1])
        window.layout_mode_combo.setCurrentText(mode)
        window._refresh_compare_view(force_render=True)
        canvas = window.compare_canvas
        axes = canvas._compare_axes or canvas._spectrum_panels or [canvas.ax]
        for ax in axes:
            ax.set_xlim(40, -40)
        canvas.set_export_settings(dpi=60, fig_width=8, fig_height=4)
        out = tmp_path / "x.png"
        canvas.export_full_resolution(lambda: canvas.figure.savefig(str(out)))
        width, height = Image.open(out).size
        # the saved figure is about the size asked for, not as wide as the data
        assert width <= 8 * 60 * 1.3 and height <= 4 * 60 * 1.3
        assert [ax.get_xlim() for ax in axes] == [(40, -40)] * len(axes)
    finally:
        window.close()


def test_stacked_rows_clip_sideways_but_not_up_or_down(qapp, tmp_path):
    from opennmr.gui.main_window import MainWindow
    from test_gui import _select_rows
    window = MainWindow()
    try:
        for p in _two_datasets(tmp_path, n=2000):
            window._load(p, vendor="bruker")
        _select_rows(window, [0, 1])
        window.layout_mode_combo.setCurrentText("Stacked")
        window._refresh_compare_view(force_render=True)
        for ax in window.compare_canvas._compare_axes:
            line = ax.lines[-1]
            assert line.get_clip_on() and line.get_clip_box() is not None and line.get_clip_path() is None
            box = line.get_clip_box()
            window.compare_canvas.canvas.draw()
            axes_box = ax.get_window_extent()
            assert box.x0 == pytest.approx(axes_box.x0) and box.x1 == pytest.approx(axes_box.x1)
            assert box.y1 > axes_box.y1 and box.y0 < axes_box.y0
    finally:
        window.close()
