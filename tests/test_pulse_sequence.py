"""2026-10-21: the pulse-sequence viewer -- the parser for Bruker, Varian,
JEOL, X-Pulse and Magritek sources, the schematic layout, the notebook
drawing, and the GUI pop-up (right-click an experiment, or double-click the
pulse-programme name at the bottom right)."""
import os

import matplotlib
matplotlib.use("Agg")
import pytest
from matplotlib.patches import Polygon, Rectangle
from PySide6.QtWidgets import QApplication

from opennmr.io import pulse_sequence as io_ps
from opennmr.io.pulse_sequence import PulseSequenceError, compute_layout, parse_pulse_sequence
from opennmr.notebook import pulse_sequence as ps
from test_gui import _load_spectrum_directly, _make_1d_peak_spectrum_with_pulse  # noqa: F401


# ---------------------------------------------------------------- sources

BRUKER = """\
;hahn echo with options
#include <Avance.incl>
"p2=p1*2"
"d6=d20-p1"
1 ze
2 1m
#ifdef dec
  1m cpds2:f2
#else
  1m
#endif
  d1
  (p1 pl1 ph1):f1 (p3 ph3):f2
  d6
  (p6:sp6 ph3):f1
  30u
  (p2 ph2):f1
  d7
  go=2 ph31
  1m do:f2
  lo to 2 times l1
exit
ph1=0 2
"""

VARIAN = """\
#include <standard.h>
pulsesequence()
{
  char fsat[MAXSTR];
  getstr("fsat",fsat);
  status(A);
  if(fsat[0] == 'y')
  {
    rgpulse(d1,zero,2.0e-6,2.0e-6);   /* presat */
  }
  else
    delay(d1);
  status(C);
  rgpulse(pw,zero,1.0e-6,0.0);
  delay(taua);
  sim3pulse(2*pw,0.0e-6,2*pwn,zero,zero,zero,0.0,0.0);
  rgradient('z',gzlvl1);
  delay(gt1);
  rgradient('z',0.0);
  initval(4.0,v1);
  loop(v1,v2);
    dec2rgpulse(2*pwn,zero,0.0,0.0);
    delay(450.0e-6);
  endloop(v2);
  shaped_pulse(shape,pw_sl,two,2.0e-6,0.0);
  status(D);
  setreceiver(t7);
}
"""

JEOL = """\
header
	filename		=>	"echo_dec";
end header;
pulse
	obs_width_90		=>	10[us];
	obs_width_180		=>	20[us];
	total_echo			=> 1[ms];
	separate			=> True;
	pre_echo			=> 2[ms];
	post_echo			=> 3[ms];
	delay_pre_echo 	=? if separate then pre_echo
					else total_echo / 2;
	relaxation_delay	=>	5.0[s];
begin
	relaxation_delay;
	obs_width_90,		(obs.gate,	obs.phs.p);
	on (irr.gate, irr.phs.q);
	delay_pre_echo;
	off(irr.gate);
	obs_width_180,		(obs.gate,	obs.phs.e);
	acq( dead_time, delay, ph );
end pulse;
"""

XPULSE = """\
from firebird import *
@ParameterBlock
class Parameters:
    Sequence = Parameter("Sequence", "my_seq", ParameterTypes.String, "Sequence Name")
    RecycleDelay = Parameter("RD", 2000000, ParameterTypes.Int32, "RD")
    P90 = Parameter("P90", 9.0, ParameterTypes.Double, "p")
    G1 = Parameter("G1", 1.0, ParameterTypes.Double, "g")
    RampTime = Parameter("D70", 500.0, ParameterTypes.Double, "r")
    GradientOnTime = Parameter("D71", 2000.0, ParameterTypes.Double, "d")

def run(comms):
    P = Parameters
    with sequential:
        Delay(10000)
    for seqScans in range(4):
        with sequential_main(seqScans, 4):
            Delay(P.RecycleDelay)
            Transmit1(P.P90)
            Gradient3(P.RampTime, P.G1)
            Delay(P.GradientOnTime)
            Gradient3(P.RampTime, 0)
            with parallel:
                with sequential:
                    Transmit2(500.0)
                with sequential:
                    Channel2SetBasePhase(100.0, 0)
                    Channel2SetBasePhase(100.0, 2)
            Receiver1(1.0, 1024)
"""

XPULSE_LIBRARY = """\
# Hard RF Pulse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Parameters:

def run(comms):
    Transmit1(P.P90)

# Shaped RF Pulse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Parameters:

def run(comms):
    Transmit1ShapedPulse(P.W, TX_Phase, P.Shape)
"""

MAGRITEK = """\
procedure(pulse_program,dir,mode,pars)
   initpp(dir)
      pulse(1,a90,p1,d90)
      wait(wEvol)
      Spoiler:XYZ(nx, ny, nz, nxShim, nyShim, nzShim, dSpoil, dStab)
      pulse(1,a90,p2,dX)
      acquire("overwrite",n1)
   lst = endpp(0)
endproc(lst)
"""


def _kinds(seq, minor=True):
    return [e.kind for e in seq.events if minor or not e.minor]


# ----------------------------------------------------------------- Bruker

def test_bruker_pulses_delays_and_acquisition():
    seq, = parse_pulse_sequence(BRUKER, "bruker",
                                constants={"d1": 5.0, "p1": 10.0, "p2": 20.0, "d6": 1e-3, "l1": 8})
    kinds = _kinds(seq, minor=False)
    assert kinds == ["delay", "hard", "hard", "delay", "shaped", "hard", "delay", "acquire"]
    d1 = next(e for e in seq.events if e.label == "d1")
    assert d1.value == 5.0 and not d1.minor
    p1 = next(e for e in seq.events if e.label == "p1")
    assert p1.value == pytest.approx(10e-6) and p1.channel == "f1"
    p6 = next(e for e in seq.events if e.label == "p6")
    assert p6.kind == "shaped" and p6.shape == "sp6"


def test_bruker_two_groups_on_one_line_are_simultaneous():
    seq, = parse_pulse_sequence(BRUKER, "bruker")
    p1 = next(e for e in seq.events if e.label == "p1")
    p3 = next(e for e in seq.events if e.label == "p3")
    assert (p1.channel, p3.channel) == ("f1", "f2")
    assert p1.t0 == p3.t0


def test_bruker_housekeeping_delays_are_minor_and_merged():
    seq, = parse_pulse_sequence(BRUKER, "bruker")
    short = [e for e in seq.events if e.kind == "delay" and e.minor]
    assert short, "30u / 1m housekeeping delays should be flagged minor"
    # no two minor delays are left touching one another
    delays = [e for e in seq.events if e.kind == "delay"]
    for a, b in zip(delays, delays[1:]):
        assert not (a.minor and b.minor and abs(a.t1 - b.t0) < 1e-12)


def test_bruker_ifdef_blocks_follow_zgoptns_and_flags():
    off, = parse_pulse_sequence(BRUKER, "bruker")
    assert not off.of_kind("decoupling") and off.flags == {"dec": False}
    via_define, = parse_pulse_sequence(BRUKER, "bruker", defines={"dec"})
    via_flag, = parse_pulse_sequence(BRUKER, "bruker", flags={"dec": True})
    for seq in (via_define, via_flag):
        dec = seq.of_kind("decoupling")
        assert len(dec) == 1 and dec[0].channel == "f2" and dec[0].duration > 0
        assert seq.flags == {"dec": True}


def test_bruker_loop_records_its_count():
    seq, = parse_pulse_sequence(BRUKER, "bruker", constants={"l1": 8})
    assert len(seq.loops) == 1
    assert seq.loops[0].count == "l1" and seq.loops[0].n == 8
    assert seq.loops[0].t1 > seq.loops[0].t0


# ----------------------------------------------------------------- Varian

def test_varian_functions_and_defaults():
    seq, = parse_pulse_sequence(VARIAN, "varian")
    assert seq.flags == {"fsat": False}
    hard = [(e.channel, e.label) for e in seq.of_kind("hard")]
    assert ("f1", "pw") in hard and ("f1", "2pw") in hard
    assert ("f3", "2pwn") in hard and ("f2", "") not in hard   # 0.0 width skipped
    assert len(seq.of_kind("gradient")) == 1
    assert [e.kind for e in seq.events][-1] == "acquire"
    assert len(seq.of_kind("shaped")) == 1 and seq.of_kind("shaped")[0].label == "pw_sl"
    assert len(seq.loops) == 1 and seq.loops[0].n == 4


def test_varian_if_branch_follows_flag():
    off, = parse_pulse_sequence(VARIAN, "varian")
    on, = parse_pulse_sequence(VARIAN, "varian", flags={"fsat": True})
    assert any(e.label == "d1" and e.kind == "delay" for e in off.events)
    assert any(e.label == "d1" and e.kind == "hard" for e in on.events)


def test_varian_without_pulsesequence_function_says_so():
    with pytest.raises(PulseSequenceError, match="pulsesequence"):
        parse_pulse_sequence("int main() { return 0; }", "varian")


# ------------------------------------------------------------------- JEOL

def test_jeol_pulses_decoupling_delays_and_flag():
    seq, = parse_pulse_sequence(JEOL, "jeol")
    assert seq.name == "echo_dec"
    assert _kinds(seq) == ["delay", "hard", "decoupling", "delay", "hard", "acquire"] or \
        sorted(_kinds(seq)) == sorted(["delay", "hard", "decoupling", "delay", "hard", "acquire"])
    pre = next(e for e in seq.events if e.label == "delay_pre_echo")
    assert pre.value == pytest.approx(2e-3)                 # separate = True
    p90 = next(e for e in seq.events if e.label == "obs_width_90")
    assert p90.value == pytest.approx(10e-6) and p90.channel == "f1"
    dec = seq.of_kind("decoupling")[0]
    assert dec.channel == "f2" and dec.t1 >= pre.t1
    seq2, = parse_pulse_sequence(JEOL, "jeol", flags={"separate": False})
    assert next(e for e in seq2.events if e.label == "delay_pre_echo").value == pytest.approx(0.5e-3)


def test_jeol_file_by_extension(tmp_path):
    f = tmp_path / "sample.jxp"
    f.write_text(JEOL)
    seq = ps.read_pulse_sequences(str(f))[0]
    assert seq.vendor == "jeol" and seq.of_kind("hard")


# ---------------------------------------------------------------- X-Pulse

def test_xpulse_uses_only_the_scan_body_and_parameter_defaults():
    seq, = parse_pulse_sequence(XPULSE, "xpulse")
    assert seq.name == "my_seq"
    delays = seq.of_kind("delay")
    assert [e.label for e in delays] == ["RD", "D71"]          # the 10 ms set-up delay is excluded
    assert delays[0].value == pytest.approx(2.0)
    p90 = seq.of_kind("hard")[0]
    assert p90.label == "P90" and p90.value == pytest.approx(9e-6)


def test_xpulse_gradient_ramp_delay_ramp_is_one_event():
    seq, = parse_pulse_sequence(XPULSE, "xpulse")
    grad = seq.of_kind("gradient")
    assert len(grad) == 1 and grad[0].label == "G1"
    assert grad[0].duration == pytest.approx(500e-6 + 2000e-6 + 500e-6)


def test_xpulse_phase_cycled_transmit_beside_phase_only_branch_is_decoupling():
    seq, = parse_pulse_sequence(XPULSE, "xpulse")
    dec = seq.of_kind("decoupling")
    assert len(dec) == 1 and dec[0].channel == "f2"
    assert seq.of_kind("acquire")


def test_xpulse_library_file_gives_one_sequence_per_chunk():
    seqs = parse_pulse_sequence(XPULSE_LIBRARY, "xpulse")
    assert [s.name for s in seqs] == ["Hard RF Pulse", "Shaped RF Pulse"]
    assert seqs[0].of_kind("hard") and seqs[1].of_kind("shaped")


def test_xpulse_without_events_says_so():
    with pytest.raises(PulseSequenceError):
        parse_pulse_sequence("def run(comms):\n    x = 1\n", "xpulse")


# --------------------------------------------------------------- Magritek

def test_magritek_macro():
    seq, = parse_pulse_sequence(MAGRITEK, "magritek", name="COSY")
    assert _kinds(seq, minor=False) == ["hard", "delay", "gradient", "hard", "acquire"]
    assert seq.of_kind("hard")[0].label == "d90"


# -------------------------------------------------------------- detection

@pytest.mark.parametrize("text,path,vendor", [
    (VARIAN, None, "varian"), (XPULSE, None, "xpulse"), (MAGRITEK, "COSY_pp.mac", "magritek"),
    (JEOL, "x.jxp", "jeol"), (BRUKER, "hahn.enb", "bruker")])
def test_vendor_detection(text, path, vendor):
    assert io_ps.detect_vendor(text, path) == vendor


# ----------------------------------------------------------------- layout

def test_layout_is_ordered_and_collapses_housekeeping_delays():
    seq, = parse_pulse_sequence(BRUKER, "bruker")
    narrow = compute_layout(seq, show_minor=False)
    wide = compute_layout(seq, show_minor=True)
    assert narrow.width < wide.width
    for e in seq.events:
        assert narrow.x1[e.uid] >= narrow.x0[e.uid]
    hard = seq.of_kind("hard")
    assert all(a.t0 <= b.t0 for a, b in zip(hard, hard[1:]))


def test_overlapping_delays_get_separate_lanes():
    seq, = parse_pulse_sequence(XPULSE, "xpulse")
    lay = compute_layout(seq)
    assert lay.n_lanes >= 1
    seq2 = parse_pulse_sequence(
        "from firebird import *\ndef run(comms):\n    with parallel:\n"
        "        with sequential:\n            Delay(100.0)\n"
        "        with sequential:\n            Delay(200.0)\n", "xpulse")[0]
    lay2 = compute_layout(seq2)
    assert lay2.n_lanes == 2 and set(lay2.lane.values()) == {0, 1}


# --------------------------------------------------------------- notebook

def test_pulse_label_is_italic_with_subscript():
    assert ps.pulse_label("p1") == r"$\mathit{p}_{1}$"
    assert ps.pulse_label("2pw") == r"2$\mathit{pw}$"
    assert ps.pulse_label("tau1") == r"$\tau_{1}$"
    assert ps.pulse_label("") == ""


def test_drawing_shapes_and_colours():
    seq, = parse_pulse_sequence(BRUKER, "bruker", defines={"dec"})
    fig, ax = ps.plot_pulse_sequence(seq, colors={"hard": "#112233", "shaped": "#445566",
                                                  "decoupling": "#778899"})
    import matplotlib.colors as mc
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    polys = [p for p in ax.patches if isinstance(p, Polygon)]
    assert len(polys) == 1                                       # the shaped pulse
    assert mc.to_hex(polys[0].get_facecolor()) == "#445566"
    colours = {mc.to_hex(r.get_facecolor()) for r in rects}
    assert "#112233" in colours and "#778899" in colours
    heights = {mc.to_hex(r.get_facecolor()): r.get_height() for r in rects}
    assert heights["#778899"] < heights["#112233"]               # decoupling is the short one
    import matplotlib.pyplot as plt
    plt.close(fig)


def _label_texts(ax):
    return [t.get_text() for t in ax.texts]


def test_names_all_some_none():
    seq, = parse_pulse_sequence(BRUKER, "bruker")
    import matplotlib.pyplot as plt
    fig, ax = ps.plot_pulse_sequence(seq)
    all_texts = _label_texts(ax)
    assert ps.pulse_label("p1") in all_texts and ps.pulse_label("d6") in all_texts
    plt.close(fig)
    fig, ax = ps.plot_pulse_sequence(seq, show_pulse_names=False)
    texts = _label_texts(ax)
    assert ps.pulse_label("p1") not in texts and ps.pulse_label("d6") in texts
    plt.close(fig)
    fig, ax = ps.plot_pulse_sequence(seq, show_delay_names=False)
    texts = _label_texts(ax)
    assert ps.pulse_label("p1") in texts and ps.pulse_label("d6") not in texts
    plt.close(fig)
    only = next(e.uid for e in seq.events if e.label == "p6")
    fig, ax = ps.plot_pulse_sequence(seq, show_names={only})
    texts = _label_texts(ax)
    assert ps.pulse_label("p6") in texts and ps.pulse_label("p1") not in texts
    plt.close(fig)
    fig, ax = ps.plot_pulse_sequence(seq, show_names=set())
    assert ps.pulse_label("p1") not in _label_texts(ax)
    plt.close(fig)


def test_event_table_has_one_row_per_event():
    seq, = parse_pulse_sequence(BRUKER, "bruker", constants={"d1": 5.0})
    rows = ps.event_table(seq)
    assert len(rows) == len(seq.events)
    assert {"kind", "name", "expression", "duration"} <= set(rows[0])
    assert next(r for r in rows if r["name"] == "d1")["duration"] == "5 s"


def test_pulse_sequence_for_an_experiment_with_its_pulseprogram(tmp_path):
    (tmp_path / "pulseprogram").write_text(BRUKER)
    s = _make_1d_peak_spectrum_with_pulse(p1=7.5)
    s.metadata.source_path = str(tmp_path)
    s.metadata.raw["acqus"].update({"NUC1": "1H", "NUC2": "13C", "ZGOPTNS": "-Ddec"})
    seqs = ps.pulse_sequence_for(s)
    seq = seqs[0]
    assert seq.of_kind("decoupling")                              # ZGOPTNS -Ddec honoured
    assert next(e for e in seq.events if e.label == "p1").value == pytest.approx(7.5e-6)
    assert seq.channel_nuclei["f1"].endswith("H") and seq.channel_nuclei["f2"].endswith("C")


def test_pulse_sequence_for_an_experiment_without_the_source_says_what_to_do():
    s = _make_1d_peak_spectrum_with_pulse()
    s.metadata.pulse_program = "hahn"
    with pytest.raises(PulseSequenceError, match="hahn"):
        ps.pulse_sequence_for(s)


# -------------------------------------------------------------------- GUI

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1200, 800)
    yield w
    d = getattr(w, "_pulse_sequence_dialog", None)
    if d is not None:
        d.close()
    w.close()


def _experiment_with_programme(tmp_path, text=BRUKER):
    (tmp_path / "pulseprogram").write_text(text)
    s = _make_1d_peak_spectrum_with_pulse()
    s.metadata.source_path = str(tmp_path)
    s.metadata.pulse_program = "hahn"
    return s


def test_right_click_action_opens_the_viewer(window, tmp_path):
    _load_spectrum_directly(window, _experiment_with_programme(tmp_path))
    window._show_pulse_sequence(0)
    d = window._pulse_sequence_dialog
    assert d.isVisible() and d.sequence is not None
    assert d.sequence.of_kind("hard")


def test_double_clicking_the_programme_name_opens_the_viewer(window, tmp_path):
    _load_spectrum_directly(window, _experiment_with_programme(tmp_path))
    assert "pulse sequence" in window._status_experiment.toolTip().lower()
    window._status_experiment.doubleClicked.emit()
    d = window._pulse_sequence_dialog
    assert d.isVisible() and d.sequence is not None


def test_double_click_with_nothing_selected_does_not_crash(window):
    window._status_experiment.doubleClicked.emit()
    assert getattr(window, "_pulse_sequence_dialog", None) is None


def test_the_same_window_is_reused_for_another_experiment(window, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _load_spectrum_directly(window, _experiment_with_programme(a))
    _load_spectrum_directly(window, _experiment_with_programme(b, "1 ze\n  d1\n  (p1 ph1):f1\n  go=1 ph31\nexit\n"))
    window._show_pulse_sequence(0)
    first = window._pulse_sequence_dialog
    n_first = len(first.sequence.events)
    window._show_pulse_sequence(1)
    assert window._pulse_sequence_dialog is first
    assert len(first.sequence.events) < n_first


def test_no_source_shows_a_message_and_the_open_button(window):
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse())
    window.session.spectra[0].metadata.pulse_program = "zg30"
    window._show_pulse_sequence(0)
    d = window._pulse_sequence_dialog
    assert d.sequence is None and "zg30" in d.message.text()
    assert d.open_button.isEnabled()


def test_open_file_for_an_experiment_that_keeps_only_the_name(window, tmp_path):
    _load_spectrum_directly(window, _make_1d_peak_spectrum_with_pulse())
    f = tmp_path / "seq.c"
    f.write_text(VARIAN)
    window._show_pulse_sequence(0)
    d = window._pulse_sequence_dialog
    d.load(str(f))
    assert d.sequence is not None and d.sequence.vendor == "varian"
    assert d.message.text() == ""
    assert "fsat" in d.flag_checks


def test_colour_change_redraws_and_persists(window, tmp_path):
    _load_spectrum_directly(window, _experiment_with_programme(tmp_path))
    window._show_pulse_sequence(0)
    d = window._pulse_sequence_dialog
    d.colors["hard"] = "#ff0000"
    d._store_colours()
    assert window._ui_settings["pulse_sequence_colors"] == {"hard": "#ff0000"}
    import matplotlib.colors as mc
    fc = {mc.to_hex(p.get_facecolor()) for p in d.figure.axes[0].patches}
    assert "#ff0000" in fc
    d._reset_colours()
    assert window._ui_settings["pulse_sequence_colors"] == {}


def test_a_saved_colour_is_used_by_the_next_window(window, tmp_path):
    window._ui_settings["pulse_sequence_colors"] = {"shaped": "#123456"}
    _load_spectrum_directly(window, _experiment_with_programme(tmp_path))
    window._show_pulse_sequence(0)
    assert window._pulse_sequence_dialog.colors["shaped"] == "#123456"
    assert window._pulse_sequence_dialog.colors["hard"] == ps.DEFAULT_COLORS["hard"]


def test_names_all_some_none_in_the_window(window, tmp_path):
    from PySide6.QtCore import Qt
    _load_spectrum_directly(window, _experiment_with_programme(tmp_path))
    window._show_pulse_sequence(0)
    d = window._pulse_sequence_dialog
    everything = d._shown_uids()
    assert everything
    d._set_all_names(False)
    assert d._shown_uids() == set()
    assert not any(ps.pulse_label("p1") == t for t in _label_texts(d.figure.axes[0]))
    d._set_all_names(True)
    assert d._shown_uids() == everything
    # one group off (tri-state parent): only the pulses
    groups = {d.name_tree.topLevelItem(i).text(0): d.name_tree.topLevelItem(i)
              for i in range(d.name_tree.topLevelItemCount())}
    groups["Delays"].setCheckState(0, Qt.CheckState.Unchecked)
    shown = d._shown_uids()
    assert shown and shown < everything
    labels = {e.uid: e.label for e in d.sequence.events}
    assert all(labels[u].startswith("p") for u in shown)
    # a single name off -> the parent goes partially checked
    child = groups["Pulses"].child(0)
    child.setCheckState(0, Qt.CheckState.Unchecked)
    assert groups["Pulses"].checkState(0) == Qt.CheckState.PartiallyChecked


def test_conditional_block_checkbox_rebuilds_the_sequence(window, tmp_path):
    _load_spectrum_directly(window, _experiment_with_programme(tmp_path))
    window._show_pulse_sequence(0)
    d = window._pulse_sequence_dialog
    assert "dec" in d.flag_checks and not d.sequence.of_kind("decoupling")
    d.flag_checks["dec"].setChecked(True)
    assert d.sequence.of_kind("decoupling")


def test_display_options_persist(window, tmp_path):
    _load_spectrum_directly(window, _experiment_with_programme(tmp_path))
    window._show_pulse_sequence(0)
    d = window._pulse_sequence_dialog
    d.minor_check.setChecked(True)
    assert window._ui_settings["pulse_sequence_options"]["show_minor"] is True


def test_events_tab_lists_every_event(window, tmp_path):
    _load_spectrum_directly(window, _experiment_with_programme(tmp_path))
    window._show_pulse_sequence(0)
    d = window._pulse_sequence_dialog
    assert d.table.rowCount() == len(d.sequence.events)


@pytest.mark.parametrize("ext", ["pdf", "svg", "png"])
def test_save_figure(window, tmp_path, ext):
    _load_spectrum_directly(window, _experiment_with_programme(tmp_path))
    window._show_pulse_sequence(0)
    out = tmp_path / f"seq.{ext}"
    window._pulse_sequence_dialog.save_figure(str(out), dpi=80)
    assert out.exists() and out.stat().st_size > 500
