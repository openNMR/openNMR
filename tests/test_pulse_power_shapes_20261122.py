"""2026-11-22: the pulse-sequence viewer shows each pulse's power (W / dB) and draws adiabatic
and ramp pulses differently; a ramp is a quadrilateral whose top edge rises or falls."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from matplotlib.patches import Polygon

from opennmr.io.pulse_sequence import parse_pulse_sequence
from opennmr.notebook import pulse_sequence as ps
from test_gui import qapp  # noqa: F401 -- qapp is a fixture

PROGRAM = """\
;power test
1 ze
2 d1 pl1:f1
  (p1 pl1 ph1):f1
  d2
  (p13:sp1 ph2):f1
  d3
  (p14:sp2 ph2):f1
  d3
  (p15:sp3 ph2):f1
  d3
  (p16:sp4 ph2):f1
  go=2 ph31
exit
"""
CONSTS = {"d1": 1.0, "p1": 5e-6, "p13": 2e-3, "p14": 1e-3, "p15": 1e-3, "p16": 1e-3,
          "pl1": 3.0, "plw1": 0.5, "sp1": 10.0, "spw1": 0.1, "sp2": 8.0, "spw2": 0.16,
          "sp3": 6.0, "spw3": 0.25, "sp4": 6.0, "spw4": 0.25}
SPNAM = ["", "Gaus1.1000", "Crp60,0.5,20.1", "ramp70100.100", "RampDown.100"]


@pytest.fixture
def seq():
    s, = parse_pulse_sequence(PROGRAM, "bruker", constants=CONSTS)
    ps.resolve_pulse_details(s, CONSTS, {"SPNAM": SPNAM})
    return s


def _pulses(seq):
    return {e.label: e for e in seq.events if e.kind in ("hard", "shaped")}


def test_parser_attaches_the_power_a_pulse_names_or_inherits(seq):
    p = _pulses(seq)
    assert p["p1"].power == "pl1"                      # named in its own group
    assert p["p13"].power is None and p["p13"].shape == "sp1"      # a slot: its own sp1 / spw1


def test_a_power_set_on_the_channel_is_inherited():
    prog = "1 ze\n2 d1 pl2:f1\n  (p1 ph1):f1\n  go=2 ph31\nexit\n"
    s, = parse_pulse_sequence(prog, "bruker", constants={"d1": 1.0, "p1": 5e-6})
    assert _pulses(s)["p1"].power == "pl2"


def test_power_resolves_to_watts_and_db(seq):
    p = _pulses(seq)
    assert (p["p1"].power_w, p["p1"].power_db) == (0.5, 3.0)
    assert (p["p13"].power_w, p["p13"].power_db) == (0.1, 10.0)
    assert (p["p14"].power_w, p["p14"].power_db) == (0.16, 8.0)


def test_db_is_worked_out_from_watts_when_only_watts_are_recorded():
    s, = parse_pulse_sequence(PROGRAM, "bruker", constants={"plw1": 0.1, "p1": 5e-6, "d1": 1.0})
    ps.resolve_pulse_details(s, {"plw1": 0.1})
    p1 = _pulses(s)["p1"]
    assert p1.power_w == 0.1 and p1.power_db == pytest.approx(10.0)


def test_unrecorded_power_stays_none():
    s, = parse_pulse_sequence(PROGRAM, "bruker", constants={"p1": 5e-6, "d1": 1.0})
    ps.resolve_pulse_details(s, {})
    assert _pulses(s)["p1"].power_w is None and ps.power_text(_pulses(s)["p1"], "both") == ""


def test_power_text_modes(seq):
    p1 = _pulses(seq)["p1"]
    assert ps.power_text(p1, "none") == ""
    assert ps.power_text(p1, "W") == "0.5 W" and ps.power_text(p1, "dB") == "3 dB"
    assert ps.power_text(p1, "both") == "0.5 W\n3 dB"
    with pytest.raises(ValueError):
        ps.power_text(p1, "mW")


@pytest.mark.parametrize("name, kind", [
    ("Crp60,0.5,20.1", "adiabatic"), ("WURST-20", "adiabatic"), ("HS1", "adiabatic"),
    ("hypsec", "adiabatic"), ("BIR-4", "adiabatic"), ("ramp.100", "ramp"), ("RampDown.100", "ramp"),
    ("Gaus1.1000", "shaped"), ("Sinc1.1000", "shaped"), (None, "shaped")])
def test_shape_names_are_classified(name, kind):
    assert ps.classify_shape(name)[0] == kind


def test_ramp_levels_come_from_the_name_or_default_to_a_rising_ramp():
    assert ps.classify_shape("ramp70100.100") == ("ramp", (0.7, 1.0))
    assert ps.classify_shape("ramp.100") == ("ramp", (ps.RAMP_LOW, 1.0))
    assert ps.classify_shape("RampDown.100") == ("ramp", (1.0, ps.RAMP_LOW))


def test_pulse_kinds_resolved_from_the_shape_file_names(seq):
    p = _pulses(seq)
    assert [p[k].shape_kind for k in ("p13", "p14", "p15", "p16")] == ["shaped", "adiabatic", "ramp", "ramp"]
    assert p["p15"].ramp == (0.7, 1.0) and p["p16"].ramp == (1.0, ps.RAMP_LOW)


def _polys(ax):
    return [pt for pt in ax.patches if isinstance(pt, Polygon)]


def test_a_ramp_is_a_quadrilateral_with_three_right_angled_edges(seq):
    fig, ax = ps.plot_pulse_sequence(seq, pulse_types={"p15": "ramp_up"})
    quads = [pt.get_xy() for pt in _polys(ax) if len(pt.get_xy()) == 5]     # closed: 4 corners + repeat
    assert quads
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quads[0][:4]
    assert y0 == y1 and x1 == x2 and x0 == x3            # base, right edge and left edge are straight
    assert y3 < y2                                       # rising top edge: low on the left, high on the right


def test_ramp_down_falls(seq):
    fig, ax = ps.plot_pulse_sequence(seq, pulse_types={"p15": "ramp_down", "p16": "hard"})
    quads = [pt.get_xy() for pt in _polys(ax) if len(pt.get_xy()) == 5]
    (_, _), (_, _), (_, y2), (_, y3) = quads[0][:4]
    assert y3 > y2


def test_adiabatic_pulses_are_hatched_in_their_own_colour(seq):
    fig, ax = ps.plot_pulse_sequence(seq)
    hatched = [pt for pt in ax.patches if pt.get_hatch()]
    assert len(hatched) == 1
    assert matplotlib.colors.to_hex(hatched[0].get_edgecolor()) == ps.DEFAULT_COLORS["adiabatic"].lower()


def test_shapes_have_different_colours_and_the_legend_lists_them(seq):
    assert len({ps.DEFAULT_COLORS[k] for k in ("hard", "shaped", "adiabatic", "ramp")}) == 4
    fig, ax = ps.plot_pulse_sequence(seq)
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert {"Hard pulse", "Shaped pulse", "Adiabatic pulse", "Ramp pulse"} <= set(texts)


def test_power_is_written_under_each_pulse(seq):
    fig, ax = ps.plot_pulse_sequence(seq, show_power="both")
    texts = [t.get_text() for t in ax.texts]
    assert "0.5 W\n3 dB" in texts and "0.1 W\n10 dB" in texts
    fig, ax = ps.plot_pulse_sequence(seq, show_power="W")
    texts = [t.get_text() for t in ax.texts]
    assert "0.16 W" in texts and not any("dB" in t for t in texts)
    fig, ax = ps.plot_pulse_sequence(seq)                  # off by default
    assert not any(" W" in t.get_text() for t in ax.texts)


def test_bad_options_are_refused(seq):
    with pytest.raises(ValueError):
        ps.plot_pulse_sequence(seq, show_power="mW")
    with pytest.raises(ValueError):
        ps.plot_pulse_sequence(seq, pulse_types={"p1": "triangle"})


def test_event_table_has_power_and_shape_file(seq):
    rows = {r["name"]: r for r in ps.event_table(seq)}
    assert rows["p1"]["power"] == "0.5 W  3 dB" and rows["p14"]["shape"] == "Crp60,0.5,20.1"


def test_sequence_without_power_information_still_draws():
    s, = parse_pulse_sequence(PROGRAM, "bruker", constants=CONSTS)     # never resolved
    fig, ax = ps.plot_pulse_sequence(s, show_power="both")
    assert ax is not None


# ------------------------------------------------------------------ the viewer
def test_viewer_has_power_and_pulse_type_controls(qapp, seq):
    from opennmr.gui.pulse_sequence_dialog import PulseSequenceDialog, saved_draw_options
    saved = []
    settings = {}
    d = PulseSequenceDialog(None, settings, lambda: saved.append(1))
    d.sequences, d.sequence = [seq], seq
    d._rebuild_types()
    assert d.type_table.rowCount() == 5 and d.show_power_combo.findData("both") >= 0
    d.show_power_combo.setCurrentIndex(d.show_power_combo.findData("both"))
    assert d.draw_options()["show_power"] == "both"
    combo = d.type_table.cellWidget(2, 1)                  # p14: adiabatic in the file
    combo.setCurrentIndex(combo.findData("ramp_up"))
    assert d.draw_options()["pulse_types"] == {"p14": "ramp_up"} and saved
    assert settings["pulse_sequence_types"][seq.name] == {"p14": "ramp_up"}
    assert {"adiabatic", "ramp"} <= set(d.colour_buttons)
    kept = saved_draw_options(settings, seq.name)
    assert kept["show_power"] == "both" and kept["pulse_types"] == {"p14": "ramp_up"}
