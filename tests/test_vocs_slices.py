"""VOCS on a trace extracted from a 2D -- e.g. an F1 column of a
J-HMQC at a fixed F2 shift (user report: "This spectrum has no recorded
acquisition parameters, so P1 cannot be looked up").

The trace now carries the 2D's acquisition tables, so a named pulse
resolves on it; VOCS then weights along the trace's own (F1) Hz axis,
exactly as the notebook's ``heteronuclear_2d_slice`` workaround did
(axis from acqu2s, pulse lengths from acqus). A pulse length typed as
a number is the fallback for anything with no table at all."""
import os

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr import io
from opennmr.core.acquisition import acquisition_constants
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.processing import slicing, vocs

from test_gui import qapp, _make_2d_dataset, _write_params  # noqa: F401 -- fixture


def _jhmqc(tmp_path, p3=4.0):
    """A heteronuclear 2D (1H F2, 13C F1) that TopSpin transformed in F1,
    with a pulse table: P1 = 2.5 us (1H), P3 = ``p3`` us (13C)."""
    path = _make_2d_dataset(tmp_path, n_f1=64, n_f2=16, f1_nuc="13C")
    acqus = os.path.join(path, "acqus")
    with open(acqus, "a") as f:
        pulses = ["0"] * 64
        pulses[1], pulses[3] = "2.5", f"{p3:g}"
        f.write("##$P= (0..63)\n" + " ".join(pulses) + "\n")
    proc2s = os.path.join(path, "pdata", "1", "proc2s")
    params = io.bruker.parse_jcampdx_params(proc2s)
    params.update({"FT_mod": 4, "SW_p": 20000.0, "SF": 100.6})
    _write_params(proc2s, params)
    acqu2s = os.path.join(path, "acqu2s")
    params = io.bruker.parse_jcampdx_params(acqu2s)
    params.update({"BF1": 100.6, "SFO1": 100.6})
    _write_params(acqu2s, params)
    return path


def test_a_column_keeps_the_acquisition_parameters(tmp_path):
    spectrum = io.load(_jhmqc(tmp_path), vendor="bruker", pseudo_2d=False)
    column = slicing.get_trace(spectrum, axis="f2", index=5)
    assert column.metadata.raw["slice"]["dimension"] == "F1"
    assert column.metadata.nucleus == "13C"
    constants = acquisition_constants(column)
    assert constants["p1"] == 2.5 and constants["p3"] == 4.0
    # Processing tables of the 2D are not carried: they aren't the trace's.
    assert "procs" not in column.metadata.raw and "proc2s" not in column.metadata.raw
    row = slicing.get_trace(spectrum, axis="f1", index=3)
    assert row.metadata.raw["slice"]["dimension"] == "F2"
    assert acquisition_constants(row)["p3"] == 4.0


def test_vocs_on_an_f1_column_uses_the_f1_axis_and_the_named_pulse(tmp_path):
    spectrum = io.load(_jhmqc(tmp_path), vendor="bruker", pseudo_2d=False)
    column = slicing.get_trace(spectrum, axis="f2", index=5)
    corrected = vocs.apply_vocs(column, pulse="P3")
    expected = vocs.excitation_profile(column.axis_values(0, "hz"), vocs.pulse_to_w1(4.0))
    np.testing.assert_allclose(corrected.metadata.raw["vocs_profile"], expected)
    np.testing.assert_allclose(corrected.data, column.data * expected)
    # The profile genuinely varies across a 20 kHz F1 window at 62.5 kHz.
    assert expected.min() < 0.99 < expected.max() <= 1.0


def test_numeric_pulse_length_is_accepted_directly():
    axis = NMRAxis(kind="frequency", ppm=np.linspace(10, -10, 101), hz=np.linspace(4000, -4000, 101))
    bare = NMRSpectrum(data=np.ones(101), axes=[axis], metadata=NMRMetadata(vendor="derived"))
    for given in (2.5, "2.5", "2.5 us", "2.5 µs"):
        np.testing.assert_allclose(vocs.apply_vocs(bare, pulse=given).metadata.raw["vocs_profile"],
                                   vocs.excitation_profile(axis.hz, vocs.pulse_to_w1(2.5)))
    with pytest.raises(ValueError, match="positive"):
        vocs.apply_vocs(bare, pulse="-1")
    with pytest.raises(ValueError, match="microseconds"):
        vocs.apply_vocs(bare, pulse="P1")        # no table: says what to type instead


def test_gui_vocs_on_a_kept_column(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = MainWindow()
    window._load(_jhmqc(tmp_path), vendor="bruker")
    parent = window.session.spectra[0]
    assert not parent.is_pseudo_2d
    column = slicing.get_trace(parent, axis="f2", index=5)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    window._on_slice_keep_requested((column, "F1 column 5", parent, "f2", 5))
    kept_row = len(window.session) - 1
    assert window.session.spectra[kept_row].metadata.vendor == "derived"

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _p, _t, text: errors.append(text))
    window._select_single_row(kept_row)
    window.vocs_pulse_edit.setText("P3")
    window._on_apply_vocs_clicked()
    assert errors == []
    assert len(window.session) == kept_row + 2
    message = window.statusBar().currentMessage()
    assert "P3 = 4 µs" in message and "13C (F1 trace)" in message
