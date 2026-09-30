"""CPMG echo co-addition (notebook CPMGprocess) and its Advanced
Processing panel.

The synthetic experiments are laid out the way CPMGprocess models a
real echo train: zeros before the first acquired point, then signal
plus a little noise, with echo k's top at the centre of window k.
Frequencies are checked against openNMR's own Bruker reader on a plain
FID with the same acqus, so the CPMG axis is held to the convention
that was validated against TopSpin."""
import os

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr import io
from opennmr import notebook as nb
from opennmr.core.acquisition import acquisition_constants
from opennmr.gui.cpmg_panel import cpmg_source_problem
from opennmr.gui.main_window import MainWindow
from opennmr.processing import vocs

from test_gui import qapp  # noqa: F401 -- fixture

SW = 100_000.0


def _write_params(path, params):
    lines = []
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"##${key}= (0..{len(value) - 1})")
            lines.append(" ".join(f"{v:g}" for v in value))
        elif isinstance(value, str):
            lines.append(f"##${key}= <{value}>")
        else:
            lines.append(f"##${key}= {value}")
    with open(path, "w") as f:
        f.write("##TITLE= synthetic\n" + "\n".join(lines) + "\n##END=\n")


def _acqus(expt, td, o1, d6, l22):
    delays = [0.0] * 32
    delays[3], delays[6] = 5e-6, d6
    cnst = [0.0] * 64
    cnst[20] = 20.0                     # blanking = 2*d3 + cnst20 = 30 us = 3 points
    loops = [0] * 32
    loops[22] = l22
    pulses = [0.0] * 64
    pulses[1] = 2.5
    _write_params(os.path.join(expt, "acqus"), {
        "NUC1": "13C", "BF1": 100.0, "SFO1": 100.0 + o1 / 1e6, "O1": o1, "SW_h": SW, "TD": 2 * td,
        "NS": 1, "BYTORDA": 0, "DTYPA": 0, "DECIM": 1, "DSPFVS": 20, "GRPDLY": 0,
        "D": delays, "CNST": cnst, "L": loops, "P": pulses, "PULPROG": "qcpmg"})


def _save_fid(expt, fid):
    raw = np.empty(2 * len(fid))
    raw[0::2], raw[1::2] = fid.real * 1e6, fid.imag * 1e6
    raw.astype("<i4").tofile(os.path.join(expt, "fid"))


def _procs(expt, sf=100.0):
    os.makedirs(os.path.join(expt, "pdata", "1"), exist_ok=True)
    _write_params(os.path.join(expt, "pdata", "1", "procs"),
                  {"SF": sf, "SW_p": SW, "SI": 4096, "WDW": 0, "PHC0": 0.0, "PHC1": 0.0, "LB": 0.0, "MC2": 0})


def _cpmg(tmp_path, name="cpmg", g=12000.0, o1=5000.0, d6=500e-6, l22=32, lead=100,
          noise=1e-3, decay=40.0, sf=None):
    expt = str(tmp_path / name)
    os.makedirs(expt)
    period = (d6 * 1e6 + 30.0) / (1e6 / SW)
    td = int(lead + 20 + period * (l22 + 1)) + 64
    rng = np.random.default_rng(0)
    fid = np.zeros(td, complex)
    fid[lead:] = noise * (1 + rng.standard_normal(td - lead) + 1j * rng.standard_normal(td - lead))
    idx = np.arange(td)
    for k in range(l22):
        centre = lead + 20 + k * period + period / 2
        tr = (idx - centre) / SW
        fid += np.where(idx >= lead, np.exp(-(tr / 60e-6) ** 2) * np.exp(2j * np.pi * g * tr)
                        * np.exp(-k / decay), 0)
    _acqus(expt, td, o1, d6, l22)
    _save_fid(expt, fid)
    if sf is not None:
        _procs(expt, sf)
    return expt


def _plain(tmp_path, g=12000.0, o1=5000.0, sf=100.0):
    expt = str(tmp_path / "plain")
    os.makedirs(expt)
    t = np.arange(2048) / SW
    _acqus(expt, 2048, o1, 500e-6, 1)
    _save_fid(expt, np.exp(2j * np.pi * g * t) * np.exp(-t / 2e-3))
    _procs(expt, sf)
    return expt


def _peak(x, y):
    return x[int(np.argmax(y))]


# --- notebook ------------------------------------------------------------
@pytest.mark.parametrize("o1, sf", [(5000.0, None), (100_000.0, None), (5000.0, 100.0002)])
def test_peak_lands_where_the_bruker_reader_puts_it(tmp_path, o1, sf):
    # 100 kHz offset: dividing by SFO1 instead of BF1 (the old axis) put
    # this 1120 ppm line 1.1 ppm low. sf != BF1: a TopSpin SR of 20 Hz.
    expt = _cpmg(tmp_path, o1=o1, sf=sf)
    result = nb.CPMGprocess(expt, zero_fill=16384)
    reader = io.load(_plain(tmp_path, o1=o1, sf=sf or 100.0), vendor="bruker")
    magnitude = np.abs(reader.data + 1j * reader.data_imag)
    bin_ppm = SW / 16384 / 100.0
    assert abs(_peak(result["x_ppm"], result["real_spectrum"])
               - _peak(reader.axis_values(0, "ppm"), magnitude)) < 3 * bin_ppm
    expected = (o1 + 12000.0 - ((sf or 100.0) - 100.0) * 1e6) / (sf or 100.0)
    assert abs(_peak(result["x_ppm"], result["real_spectrum"]) - expected) < 2 * bin_ppm


def test_fractional_period_does_not_drift(tmp_path):
    # 53.4 points per echo: cutting at multiples of 53 drifts 0.4 point
    # an echo, 80 points by echo 200, and the co-added echo decoheres.
    whole = nb.CPMGprocess(_cpmg(tmp_path, "whole", d6=500e-6, l22=200, decay=1e9), apodize=False)
    frac = nb.CPMGprocess(_cpmg(tmp_path, "frac", d6=504e-6, l22=200, decay=1e9), apodize=False)
    assert frac["period_exact"] == pytest.approx(53.4)
    assert np.abs(frac["fid_coadded"]).max() == pytest.approx(np.abs(whole["fid_coadded"]).max(), rel=0.05)
    layout = nb.cpmg_echo_layout(str(tmp_path / "frac"))
    offsets = layout["starts"] - layout["begin"]
    assert np.all(np.abs(offsets - 53.4 * np.arange(200)) <= 0.5)   # never more than half a point off


def test_whole_period_split_is_unchanged(tmp_path):
    layout = nb.cpmg_echo_layout(_cpmg(tmp_path))
    assert layout["period_exact"] == 53.0
    assert np.array_equal(layout["starts"], layout["begin"] + 53 * np.arange(32))
    assert layout["first_point"] == 120          # first acquired point (100) + offset 20


def test_echoes_added_counts_from_cutout(tmp_path):
    expt = _cpmg(tmp_path)
    assert nb.CPMGprocess(expt, cutout=2, num_echoes_added=10)["num_echoes_used"] == 10
    assert nb.CPMGprocess(expt, cutout=2)["num_echoes_used"] == 30
    assert nb.CPMGprocess(expt, cutout=30, num_echoes_added=10)["num_echoes_used"] == 2   # clamped
    with pytest.raises(ValueError, match="cutout"):
        nb.CPMGprocess(expt, cutout=32)


def test_layout_errors_are_explained(tmp_path):
    expt = _cpmg(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        nb.cpmg_echo_layout(expt, num_echoes_index=40)
    with pytest.raises(ValueError, match="too short"):
        nb.cpmg_echo_layout(expt, first_point=10_000)
    # A safety bigger than half the blanking no longer becomes a
    # negative slice index.
    assert nb.cpmg_echo_layout(expt, safety=500)["trim"] == 0


def test_result_as_a_spectrum_keeps_acqus(tmp_path):
    expt = _cpmg(tmp_path)
    magnitude = nb.cpmg_to_spectrum(nb.CPMGprocess(expt))
    assert magnitude.data_imag is None
    assert magnitude.metadata.nucleus == "13C"
    assert acquisition_constants(magnitude)["p1"] == 2.5
    assert magnitude.metadata.raw["cpmg"]["num_echoes_used"] == 32
    phased = nb.cpmg_to_spectrum(nb.CPMGprocess(expt, magnitude_mode=False))
    assert phased.data_imag is not None
    # VOCS resolves a pulse on it.
    vocs.apply_vocs(magnitude, pulse="P1")


# --- GUI -----------------------------------------------------------------
def test_advanced_processing_has_vocs_and_cpmg_sections(qapp):  # noqa: F811
    window = MainWindow()
    # Spelled out since the 2026-09-30 re-wording.
    assert window.vocs_section.title() == "Variable Offset Cumulative Spectroscopy (VOCS)"
    assert window.cpmg_section.title() == "CPMG"
    assert window.vocs_pulse_edit.window() is window
    assert window.cpmg_panel.parent() is not None


def test_panel_processes_the_selected_experiment(qapp, tmp_path):  # noqa: F811
    expt = _cpmg(tmp_path, d6=504e-6, sf=100.0)
    window = MainWindow()
    window._load(expt, vendor="bruker")
    window._select_single_row(0)
    panel = window.cpmg_panel
    assert panel.process_button.isEnabled()
    panel.echoes_added_spin.setValue(20)
    assert panel.check_split()
    assert "53.400 points" in panel.summary_label.text()
    assert "not a whole number" in panel.summary_label.text()
    assert "Co-adding echoes 1–20 (20)" in panel.summary_label.text()
    assert panel.first_point_spin.value() == 120

    assert panel.process()
    assert len(window.session) == 2
    result = window.session.spectra[1]
    assert result.metadata.vendor == "derived"
    assert result.metadata.raw["cpmg"]["num_echoes_used"] == 20
    assert "co-added 20 of 32 echoes" in panel.log_edit.toPlainText()
    # The result is selected; the panel still processes the experiment.
    assert window._selected_index() == 1
    assert panel.process_button.isEnabled()
    assert panel.process()
    assert len(window.session) == 3


def test_panel_refuses_what_it_cannot_process(qapp, tmp_path):  # noqa: F811
    assert "Select" in cpmg_source_problem(None)
    window = MainWindow()
    panel = window.cpmg_panel
    assert not panel.process_button.isEnabled()
    assert not panel.process()                  # no source: logged, not raised
    assert "Select" in panel.log_edit.toPlainText()


def test_bad_index_is_shown_not_raised(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = MainWindow()
    window._load(_cpmg(tmp_path, sf=100.0), vendor="bruker")
    window._select_single_row(0)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: pytest.fail("modal error"))
    window.cpmg_panel.num_echoes_spin.setValue(40)
    assert not window.cpmg_panel.process()
    assert "out of range" in window.cpmg_panel.summary_label.text()
    assert len(window.session) == 1
