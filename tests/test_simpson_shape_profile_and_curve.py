"""User requests of 2026-11-06 for Simulate ▸ SIMPSON:

* a pulse shape (``save_shape``: amplitude in Hz, phase in degrees) read as
  the pulse it is -- nu1 against time (pulse length from the file name or
  typed, each of the N points lasting length / N as in SIMPSON's
  ``pulse_shaped``), the phase unwrapped, the frequency sweep the phase
  encodes, and the adiabaticity at the centre;
* a REDOR-like file -- real, imaginary (zeros) and the MAS rate -- read as
  one curve, against x values from the file, a typed list, a list file or
  start/step(/stop/points) with a multiplier (as the vdlist in Dynamics).

``tests/data/simpson/inv_pow.shape`` is the user's 10 us WURST
(inv_pow_10.0us_W10_1.0MHz_rot100kHz_rf400kHz.shape): 400 kHz, a 1 MHz sweep
low to high, 201 points written for 0.05 us design steps.
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

DATA = os.path.join(os.path.dirname(__file__), "data", "simpson")
WURST = os.path.join(DATA, "inv_pow.shape")


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


# ---- pulse shapes --------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("inv_pow_10.0us_W10_1.0MHz_rot100kHz_rf400kHz.shape", 10.0),
    ("OUT_inv_pow_shap_16.0us_W10_tan_atan(10)_4.0MHz_rot62kHz_rf500kHz_offset-900e3kHz.shape",
     16.0),
    ("pulse_2ms.shape", 2000.0),
    ("pulse_250ns.shape", 0.25),
    ("inv_pow.shape", None),
])
def test_pulse_length_is_read_from_the_file_name(name, expected):
    got = nmr.SIMPSONshape_duration_from_name(name)
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_wurst_profile_matches_its_design():
    p = nmr.SIMPSONshape_profile(WURST, duration_us=10.0)
    assert p["npoints"] == 201 and p["duration_source"] == "given"
    assert p["dt_us"] == pytest.approx(10.0 / 201)            # SIMPSON: length / N
    assert p["t_us"][-1] == pytest.approx(10.0 - 10.0 / 201)
    assert p["max_amplitude_Hz"] == pytest.approx(400e3, rel=1e-3)
    # Designed as 0.05 us steps sweeping -0.5 -> +0.5 MHz; played as 201 equal
    # steps of 10/201 us, the first and last offsets come out at +-0.5 MHz.
    assert p["offset_Hz"].size == 200
    assert p["offset_Hz"][0] == pytest.approx(-0.5e6, rel=1e-3)
    assert p["offset_Hz"][-1] == pytest.approx(0.5e6, rel=1e-3)
    assert p["sweep_Hz"] == pytest.approx(1.0e6, rel=1e-3)    # positive: low to high
    # The phase is written modulo 360; unwrapped, the sweep is smooth.
    assert np.all(np.diff(p["offset_Hz"]) > 0)
    # Q0 = 2 pi nu1^2 / |d nu/dt| with nu1 = 400 kHz and the played sweep rate
    # (1 MHz / 10 us, times (201/200)^2 for the shorter steps).
    rate = 1e6 / 10e-6 * (201 / 200) ** 2
    assert p["q0"] == pytest.approx(2 * np.pi * (400e3) ** 2 / rate, rel=0.01)


def test_profile_takes_the_length_from_the_name(tmp_path):
    named = tmp_path / "inv_pow_10.0us_W10_1.0MHz_rot100kHz_rf400kHz.shape"
    named.write_bytes(open(WURST, "rb").read())
    p = nmr.SIMPSONshape_profile(str(named))
    assert p["duration_us"] == 10.0 and p["duration_source"] == "file name"
    text = nmr.SIMPSONdescribe_shape(p)
    assert "Q₀" in text and "10 µs" in text


def test_without_a_length_there_is_no_time_axis_or_sweep():
    p = nmr.SIMPSONshape_profile(WURST)
    assert p["duration_us"] is None and p["offset_Hz"] is None and p["q0"] is None
    assert np.array_equal(p["t_us"], np.arange(201))
    assert "no pulse length" in nmr.SIMPSONdescribe_shape(p)


def test_a_wrapped_constant_offset_unwraps_to_one_frequency(tmp_path):
    n, duration_us, offset = 100, 10.0, 200e3
    t = np.arange(n) * duration_us / n * 1e-6
    phase = np.mod(360.0 * offset * t, 360.0)
    path = tmp_path / "cw.shape"
    np.savetxt(path, np.column_stack([np.full(n, 50e3), phase]))
    p = nmr.SIMPSONshape_profile(str(path), duration_us)
    assert np.allclose(p["offset_Hz"], offset, rtol=1e-9)
    assert p["sweep_Hz"] == pytest.approx(0.0, abs=1e-3)
    assert p["q0"] is None                                     # no sweep, no adiabaticity


def test_shape_plot_panels_and_labels():
    fig, axes = nmr.SIMPSONplot_shape(WURST)                   # unchanged default: two panels
    assert len(axes) == 2
    fig, axes = nmr.SIMPSONplot_shape(WURST, duration_us=10.0, show_offset=True)
    assert len(axes) == 3
    assert "mathit{t}" in axes[-1].get_xlabel()
    assert "mathit{\\nu}_1" in axes[0].get_ylabel() and "mathit{\\phi}" in axes[1].get_ylabel()
    assert axes[0].get_xlim() == pytest.approx((0.0, 10.0))
    amplitude = axes[0].lines[0].get_ydata()
    assert max(amplitude) == pytest.approx(400.0, rel=1e-3)   # kHz
    assert len(axes[0].lines[0].get_xdata()) == 202            # N steps, closed at the end


def test_gui_plots_a_shape_with_its_length(qapp):  # noqa: F811
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([WURST])
    panel.shape_duration_spin.setValue(10.0)
    panel.plot_button.click()
    assert len(window.fit_canvas.figure.axes) == 3
    text = panel.status_label.text()
    assert "pulse shape" in text and "Q₀" in text and "sweep" in text


# ---- one file as a curve (REDOR-like) ------------------------------------------------

REDOR = np.cos(np.linspace(0, 3, 12)) * np.exp(-np.linspace(0, 1, 12))


def _simp_curve(tmp_path, rate=10000.0):
    return sf._write(tmp_path / "redor.fid", "FID", REDOR.astype(complex), sw=rate)


def test_simp_file_gives_points_and_rate(tmp_path):
    curve = nmr.SIMPSONread_curve(_simp_curve(tmp_path))
    assert curve["n"] == 12 and curve["rate_Hz"] == 10000.0
    assert np.allclose(curve["values"].real, REDOR) and np.allclose(curve["values"].imag, 0)
    c = nmr.SIMPSONcurve(_simp_curve(tmp_path), x="file", time_unit="ms")
    assert np.allclose(c["x"], np.arange(12) * 0.1)            # k / 10 kHz, in ms
    assert c["xlabel"] == "t (ms)"
    c = nmr.SIMPSONcurve(_simp_curve(tmp_path), x="file", time_unit="us")
    assert np.allclose(c["x"], np.arange(12) * 100.0)


def test_columns_real_zeros_and_mas_rate(tmp_path):
    path = tmp_path / "redor.txt"
    np.savetxt(path, np.column_stack([REDOR, np.zeros(12), np.full(12, 12500.0)]))
    curve = nmr.SIMPSONread_curve(str(path))
    assert curve["rate_Hz"] == 12500.0 and curve["rate_source"] == "column 3"
    assert np.allclose(curve["values"].real, REDOR)


def test_a_rate_line_in_the_file(tmp_path):
    path = tmp_path / "redor.dat"
    path.write_text("# REDOR, 31P-13C\nmas = 12.5 kHz\n"
                    + "".join(f"{v:.8f}, 0\n" for v in REDOR))
    curve = nmr.SIMPSONread_curve(str(path))
    assert curve["rate_Hz"] == 12500.0 and curve["n"] == 12
    assert np.allclose(curve["x_file"], np.arange(12) / 12500.0)


def test_x_real_imaginary_columns_use_the_x_column(tmp_path):
    path = tmp_path / "xreim.dat"
    x = np.linspace(0, 5, 12)
    np.savetxt(path, np.column_stack([x, REDOR, np.zeros(12)]))
    c = nmr.SIMPSONcurve(str(path), x="file")
    assert np.allclose(c["x"], x) and np.allclose(c["values"].real, REDOR)


def test_no_rate_means_choosing_x_another_way(tmp_path):
    path = tmp_path / "y.txt"
    np.savetxt(path, REDOR)
    with pytest.raises(ValueError, match="gives no rate"):
        nmr.SIMPSONcurve(str(path), x="file")
    c = nmr.SIMPSONcurve(str(path), x="index")
    assert np.array_equal(c["x"], np.arange(12))


def test_bad_rows_are_named(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 0\n2 0\nthree 0\n")
    with pytest.raises(ValueError, match="line 3"):
        nmr.SIMPSONread_curve(str(path))


@pytest.mark.parametrize("kwargs, expected", [
    (dict(start=0, step=0.5, points=4), [0, 0.5, 1.0, 1.5]),
    (dict(start=0, step=0.5, stop=2.0), [0, 0.5, 1.0, 1.5, 2.0]),      # stop included
    (dict(start=0, stop=1.0, points=5), [0, 0.25, 0.5, 0.75, 1.0]),    # step worked out
    (dict(start=0.1, step=0.1, multiplier=2, points=4), [0.1, 0.2, 0.4, 0.8]),
    (dict(start=0, stop=7, points=4, multiplier=2), [0, 1, 3, 7]),
    (dict(start=0.1, step=0.1, multiplier=2, stop=1.0), [0.1, 0.2, 0.4, 0.8]),
    (dict(start=5, step=-1, stop=2), [5, 4, 3, 2]),
    (dict(start=0, step=2, n_default=3), [0, 2, 4]),
])
def test_stepped_values(kwargs, expected):
    assert np.allclose(nmr.stepped_values(**kwargs), expected)


def test_stepped_values_refuse_what_cannot_be_done():
    with pytest.raises(ValueError, match="never reaches"):
        nmr.stepped_values(0, step=1, stop=-3)
    with pytest.raises(ValueError, match="multiplier"):
        nmr.stepped_values(0, step=1, multiplier=0, points=3)
    with pytest.raises(ValueError, match="non-zero step"):
        nmr.stepped_values(0, step=0, points=3)


def test_typed_list_file_and_lengths(tmp_path):
    path = _simp_curve(tmp_path)
    typed = ", ".join(str(v) for v in range(14))                       # two more than needed
    c = nmr.SIMPSONcurve(path, x=typed)
    assert np.array_equal(c["x"], np.arange(12)) and c["x_unused"] == 2
    assert nmr.SIMPSONcurve(path, x=" ".join(["10m"] * 12))["x"][0] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="give one per point"):
        nmr.SIMPSONcurve(path, x="1, 2, 3")
    vd = tmp_path / "vdlist"
    vd.write_text("\n".join(str(v) for v in range(1, 13)) + "\n")
    c = nmr.SIMPSONcurve(path, x=str(vd), factor=0.08)
    assert np.allclose(c["x"], np.arange(1, 13) * 0.08)
    c = nmr.SIMPSONcurve(path, x="steps", start=0, step=0.08)
    assert np.allclose(c["x"], np.arange(12) * 0.08)


def test_scaling_and_plot(tmp_path):
    c = nmr.SIMPSONcurve(_simp_curve(tmp_path), x="index")
    assert ss.curve_y(c, normalise="first")[0] == 1.0
    assert np.max(np.abs(ss.curve_y(c, normalise="max"))) == pytest.approx(1.0)
    fig, ax = nmr.SIMPSONplot_curve(c, normalise="first")
    assert ax.get_ylabel() == r"$\mathit{S}/\mathit{S}_0$"
    assert np.allclose(ax.lines[0].get_ydata(), REDOR / REDOR[0])
    zero = dict(c, values=np.r_[0, c["values"][1:]])
    with pytest.raises(ValueError, match="first point is zero"):
        ss.curve_y(zero, normalise="first")
    table = nmr.SIMPSONcurve_table(c)
    assert list(table.columns) == ["x", "y"]
    out = nmr.SIMPSONsave_curve(c, str(tmp_path / "c.csv"))
    assert os.path.getsize(out) > 0


# ---- the GUI section -----------------------------------------------------------------

def _panel_with_curve(tmp_path):
    window = MainWindow()
    panel = window.simpson_panel
    panel.add_paths([_simp_curve(tmp_path)])
    return window, panel


def test_gui_curve_from_the_file(qapp, tmp_path):  # noqa: F811
    window, panel = _panel_with_curve(tmp_path)
    assert panel.curve_x_combo.currentData() == "file"
    panel.plot_curve_button.click()
    ax = window.fit_canvas.figure.axes[0]
    assert np.allclose(ax.lines[0].get_xdata(), np.arange(12) * 0.1)
    assert np.allclose(ax.lines[0].get_ydata(), REDOR)
    assert "12 points" in panel.curve_preview_label.text()
    assert "10000 Hz" in panel.curve_preview_label.text()


def test_gui_curve_start_step_and_scaling(qapp, tmp_path):  # noqa: F811
    window, panel = _panel_with_curve(tmp_path)
    panel.curve_x_combo.setCurrentIndex(panel.curve_x_combo.findData("steps"))
    assert panel.curve_start_spin.isVisibleTo(panel) and not panel.curve_typed_edit.isVisibleTo(panel)
    panel.curve_start_spin.setValue(1.0)
    panel.curve_step_spin.setValue(2.0)
    panel.curve_normalise_combo.setCurrentIndex(panel.curve_normalise_combo.findData("first"))
    panel.plot_curve_button.click()
    line = window.fit_canvas.figure.axes[0].lines[0]
    assert np.allclose(line.get_xdata(), 1.0 + 2.0 * np.arange(12))
    assert np.allclose(line.get_ydata(), REDOR / REDOR[0])


def test_gui_curve_too_few_values_is_reported(qapp, tmp_path, _no_modal_dialogs):  # noqa: F811
    window, panel = _panel_with_curve(tmp_path)
    panel.curve_x_combo.setCurrentIndex(panel.curve_x_combo.findData("typed"))
    panel.curve_typed_edit.setText("1 2 3")
    panel.plot_curve_button.click()
    assert _no_modal_dialogs and "one per point" in _no_modal_dialogs[-1][1]


def test_gui_curve_notebook_code_runs(qapp, tmp_path):  # noqa: F811
    window, panel = _panel_with_curve(tmp_path)
    panel.curve_x_combo.setCurrentIndex(panel.curve_x_combo.findData("steps"))
    panel.curve_step_spin.setValue(0.5)
    panel.curve_points_edit.setText("12")
    code = panel.curve_notebook_code()
    namespace = {}
    exec(code, namespace)
    assert np.allclose(namespace["curve"]["x"], 0.5 * np.arange(12))


def test_gui_curve_settings_are_kept(qapp, tmp_path):  # noqa: F811
    window, panel = _panel_with_curve(tmp_path)
    panel.curve_x_combo.setCurrentIndex(panel.curve_x_combo.findData("typed"))
    panel.curve_typed_edit.setText("0 1 2")
    state = panel.get_state()
    other = MainWindow().simpson_panel
    other.set_state(state)
    assert other.curve_x_combo.currentData() == "typed"
    assert other.curve_typed_edit.text() == "0 1 2"
    assert other.curve_typed_edit.isVisibleTo(other)
