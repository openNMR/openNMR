"""User reports, 2026-11-19: states keep the slice <-> 2-D link, vector text stays text,
FID shift / Align Echo predict the new phase, the display never drops the last points,
a y = 0 line while phasing, a multiply-by-a-factor tool (also per spectrum in Compare),
and the anisotropic-motion plots (superscript nucleus, 1 x 10^n labels, scrolling that
keeps the spacing)."""
import numpy as np
import pytest

from opennmr.notebook import motion as M
from opennmr.notebook import scaling
from opennmr.plotting import style, vector_export
from opennmr.processing import ft


# --- vector export: text stays text ------------------------------------------------

def test_svg_text_is_text_by_default_and_switchable():
    try:
        assert vector_export.set_svg_text("text") == "text"
        assert vector_export.text_rc()["svg.fonttype"] == "none"
        assert vector_export.text_rc()["pdf.fonttype"] == 42
        assert vector_export.set_svg_text("path") == "path"
        assert vector_export.text_rc()["svg.fonttype"] == "path"
        assert vector_export.get_svg_text() == "path"
    finally:
        vector_export.set_svg_text("text")


def test_saved_svg_contains_text_elements(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("shift label")
    out = tmp_path / "a.svg"
    vector_export.set_svg_text("text")
    vector_export.save_vector_safe(fig, out)
    plt.close(fig)
    assert "<text" in out.read_text(encoding="utf-8")


# --- FID shift predicts the phase --------------------------------------------------

@pytest.mark.parametrize("roll", [3, -5, 11])
def test_roll_phase_compensation_restores_the_spectrum(roll):
    n, dt, sfo = 2048, 1e-4, 100.0
    t = np.arange(n) * dt
    f0 = 40 * (1.0 / (n * dt))                      # on a bin: no leakage from the roll
    fid = np.exp(2j * np.pi * f0 * t) * np.exp(-t / 5e-3)
    freq, spec = ft.fft_spectrum(fid, dt)
    ppm = freq / sfo
    rolled = ft.fft_spectrum(np.roll(fid, roll), dt)[1]
    p0, p1 = ft.roll_phase_compensation(roll, dt, ppm, sfo, 0.5 * (ppm.max() + ppm.min()))
    fixed = ft.apply_phase(rolled, ppm, p0, p1)
    big = np.abs(spec) > 0.2 * np.abs(spec).max()
    assert np.allclose(fixed[big], spec[big], atol=0.05 * np.abs(spec).max())


def test_roll_phase_compensation_is_zero_for_no_roll():
    assert ft.roll_phase_compensation(0, 1e-4, np.linspace(10, -10, 64), 100.0, 0.0) == (0.0, 0.0)


def test_net_fid_roll_without_echo_is_the_shift():
    fid = np.exp(2j * np.pi * 0.05 * np.arange(256))
    assert ft.net_fid_roll(fid, 7.0, False, 0.0) == 7.0


# --- the display never drops the end of a line -------------------------------------

@pytest.mark.parametrize("n", [8192, 16384, 10001])
def test_downsample_keeps_the_last_sample(n):
    x = np.linspace(10, -10, n)
    y = np.zeros(n)
    y[-1] = 5.0                                     # a signal at the right-hand edge
    xs, ys = style.downsample_1d(x, y, 8000)
    assert ys.size <= 8000
    assert ys.max() == 5.0
    assert xs.min() < x[-100]                       # the trace reaches the right-hand end


def test_downsample_preserves_peak_height():
    x = np.linspace(0, 1, 9000)
    y = np.sin(40 * x)
    _xs, ys = style.downsample_1d(x, y, 2000)
    assert ys.max() == pytest.approx(y.max(), abs=1e-3)


# --- multiply by a factor ----------------------------------------------------------

@pytest.mark.parametrize("text,value", [("3", 3.0), ("1/4", 0.25), ("x2", 2.0), ("×2", 2.0),
                                         ("/4", 0.25), ("-1.5", -1.5), ("2e3", 2000.0),
                                         ("0,5", 0.5)])
def test_parse_factor(text, value):
    assert scaling.parse_factor(text) == pytest.approx(value)


@pytest.mark.parametrize("bad", ["", "abc", "0", "inf", "/0"])
def test_parse_factor_rejects(bad):
    with pytest.raises(ValueError):
        scaling.parse_factor(bad)


@pytest.mark.parametrize("value,text", [(3.0, "3"), (0.5, "1/2"), (1 / 3, "1/3"), (-1.5, "-3/2"),
                                         (0.125, "1/8"), (0.123456, "0.1235"), (2.0000000001, "2")])
def test_format_factor(value, text):
    assert scaling.format_factor(value) == text


# --- anisotropic motion: labels, senses --------------------------------------------

def test_rate_labels_use_times_ten_to_the_power():
    assert M.format_rate_label(1e5) == "$10^{5}$ s$^{-1}$"
    assert M.format_rate_label(3e4) == "$3\\times10^{4}$ s$^{-1}$"
    assert M.format_rate_label(250) == "250 s$^{-1}$"
    assert "e+" not in M.format_rate_label(2.5e7)


def test_delta_label_has_a_superscript_nucleus():
    assert M.delta_label("2H") == "$\\delta$($^{2}$H) (ppm)"
    assert M.delta_label(None) == "$\\delta$ (ppm)"


def test_one_way_ring_jump_rate_matrix():
    fwd = M.ring_jump(4, "z", k_Hz=10.0, model="forward").rate_matrix()
    back = M.ring_jump(4, "z", k_Hz=10.0, model="backward").rate_matrix()
    both = M.ring_jump(4, "z", k_Hz=10.0).rate_matrix()
    assert np.allclose(fwd.sum(axis=0), 0) and np.allclose(back.sum(axis=0), 0)
    assert fwd[1, 0] == 10.0 and fwd[3, 0] == 0.0       # 0 -> 1 only
    assert back[3, 0] == 10.0 and back[1, 0] == 0.0     # 0 -> 3 only
    assert both[1, 0] == both[3, 0] == 10.0


def test_mode_from_spec_passes_the_sense():
    m = M.mode_from_spec({"kind": "ring_jump", "n": 3, "axis": "z", "k_Hz": 1e4,
                          "model": "forward"})
    assert m.model == "forward"


def test_motion_series_traces_carry_their_baselines():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(100, -100, 50)
    series = {"rates": [1e3, 1e5], "x_ppm": x, "which": "mode:0",
              "spectra": [np.exp(-(x / 20) ** 2), np.exp(-(x / 10) ** 2)]}
    fig, ax = plt.subplots()
    M.plot_motion_series(series, ax=ax, offset=1.2, show_limits=False, nucleus="2H")
    bases = sorted(ln._opennmr_baseline for ln in ax.lines)
    assert bases == pytest.approx([0.0, 1.2])
    assert "^{2}" in ax.get_xlabel()
    labels = [t.get_text() for t in ax.texts]
    assert all("e+" not in t and "e0" not in t for t in labels)
    plt.close(fig)


# --- GUI -----------------------------------------------------------------------------

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from test_gui import qapp  # noqa: E402,F401 -- fixture


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)


def test_scroll_rescales_stacked_traces_about_their_own_baselines(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    c = MplCanvas()
    ax = c.figure.add_subplot(111)
    x = np.linspace(0, 1, 20)
    la, = ax.plot(x, np.ones(20) + 0.0)
    lb, = ax.plot(x, np.ones(20) + 1.1)
    la._opennmr_baseline, lb._opennmr_baseline = 0.0, 1.1
    ylim = ax.get_ylim()
    c._zoom_intensity_about_baseline(ax, 0.5)           # zoom in: traces twice as tall
    assert ax.get_ylim() == ylim                        # the spacing is untouched
    assert la.get_ydata()[0] == pytest.approx(2.0)
    assert lb.get_ydata()[0] == pytest.approx(1.1 + 2.0)


def test_quick_setup_writes_one_mode(qapp):
    from opennmr.gui.motion_panel import AnisotropicMotionPanel
    p = AnisotropicMotionPanel()
    p.quick_kind_combo.setCurrentIndex(p.quick_kind_combo.findData("ring_jump"))
    p.quick_n_spin.setValue(4)
    p.quick_axis_combo.setCurrentIndex(p.quick_axis_combo.findData("x"))
    p.quick_sense_combo.setCurrentIndex(p.quick_sense_combo.findData("forward"))
    p.quick_rate_edit.setText("2e4")
    assert p.apply_quick_motion()
    rows = p._mode_rows()
    assert len(rows) == 1 and rows[0]["param"] == "4" and rows[0]["axis"] == "x"
    assert rows[0]["sense"] == "forward"
    model = p.build_model()
    assert model.modes[0].model == "forward" and model.modes[0].n == 4


def test_quick_setup_arrhenius_needs_a_temperature(qapp):
    from opennmr.gui.motion_panel import AnisotropicMotionPanel
    p = AnisotropicMotionPanel()
    p.quick_rate_combo.setCurrentIndex(p.quick_rate_combo.findData("Ea"))
    p.quick_T_edit.setText("")
    assert not p.apply_quick_motion()
    p.quick_T_edit.setText("300")
    assert p.apply_quick_motion()
    assert p.temperature_edit.text() == "300"
    assert p._mode_rows()[0]["Ea"] == "0.3"


def test_sense_survives_a_state_round_trip(qapp):
    from opennmr.gui.motion_panel import AnisotropicMotionPanel
    p = AnisotropicMotionPanel()
    p._load_modes([{"kind": "ring_jump", "param": "3", "axis": "z", "k": "1e4", "sense": "backward"}])
    state = p.get_state()
    q = AnisotropicMotionPanel()
    q.set_state(state)
    assert q._mode_rows()[0]["sense"] == "backward"
