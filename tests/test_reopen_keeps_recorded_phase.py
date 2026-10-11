"""Reopening a raw-FID experiment with saved openNMR processing must keep
TopSpin's recorded phase underneath the user's own (2026-10-11).

The bug: a saved processing sidecar holding ``phase0``/``phase1`` (even
0) made the reopen pass ``autophase=False`` to the reader, which in its
default "recorded" mode switches the recorded PHC0/PHC1 off in *both*
dimensions. Found on a real 23Na MQMAS (Bruker expt 275): correlation
with TopSpin's 2rr +0.997 opened fresh, -0.30 reopened with phase 0
saved, +0.14 with the user's compensating 206.1 deg.
"""

import os

import numpy as np
import pytest

from opennmr.io import bruker

from test_bruker_raw_2d import _make_2d_experiment_dir, _mock_2d, _synthetic_2d_fid

ACQUS = {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1.0}


def _recorded_dir(tmp_path):
    path = _make_2d_experiment_dir(tmp_path, proc2s_overrides={"PHC0": 25.0, "PHC1": 40.0})
    bruker_procs = os.path.join(path, "pdata", "1", "procs")
    with open(bruker_procs, "w") as f:
        f.write("##$SW_p= 8000.0\n##$SI= 64\n##$SF= 400.13\n##$WDW= 0\n##$PHC0= 70.0\n##$PHC1= -90.0\n")
    return path


def test_reader_puts_the_user_phase_on_top_of_the_recorded_one(tmp_path, monkeypatch):
    """What the GUI relies on: with autophase on (phase_mode "recorded")
    phase0/phase1 are added to TopSpin's phase, not instead of it."""
    path = _recorded_dir(tmp_path)
    _mock_2d(monkeypatch, ACQUS, _synthetic_2d_fid())
    recorded = bruker.read_2d(path, homonuclear=True)
    on_top = bruker.read_2d(path, homonuclear=True, phase0=30.0)
    off = bruker.read_2d(path, homonuclear=True, autophase=False, phase0=30.0)
    z_rec = recorded.data + 1j * recorded.data_imag
    z_top = on_top.data + 1j * on_top.data_imag
    np.testing.assert_allclose(z_top, z_rec * np.exp(-1j * np.deg2rad(30.0)), atol=1e-9 * np.abs(z_rec).max())
    assert on_top.metadata.raw["recorded_phase0"] == 70.0
    assert on_top.metadata.raw["recorded_phase1_f1"] == 40.0
    # autophase=False drops the recorded phase in both dimensions -- the
    # behaviour the reopen must therefore not ask for
    assert off.metadata.raw["recorded_phase0"] == 0.0
    assert off.metadata.raw["recorded_phase0_f1"] == 0.0


def test_saved_processing_kwargs_keep_autophase_on():
    from opennmr.gui.main_window import MainWindow
    for phases in ({"phase0": 0.0, "phase1": 0.0}, {"phase0": 206.1, "phase1": 0.0}):
        kwargs = MainWindow._saved_processing_read_kwargs(dict(phases))
        assert kwargs["phase0"] == phases["phase0"]
        assert kwargs.get("autophase", True) is True
    # with echo alignment the reader replaces the recorded phase itself
    # (and would overwrite phase0), so there the user's phase goes in alone
    kwargs = MainWindow._saved_processing_read_kwargs({"phase0": 12.0, "live_align_echo": True})
    assert kwargs["autophase"] is False


def test_reload_phase_kwargs_keep_autophase_on(qapp_or_none=None):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    spec = NMRSpectrum(data=np.zeros((2, 4)), axes=[NMRAxis(kind="frequency", ppm=np.arange(2.0)),
                                                    NMRAxis(kind="frequency", ppm=np.arange(4.0))],
                       metadata=NMRMetadata(vendor="xpulse", raw={"phase0": 15.0, "phase1": 0.0}))
    kwargs = window._current_phase_kwargs(spec)
    assert kwargs["phase0"] == 15.0 and "autophase" not in kwargs
    spec.metadata.raw["align_echo"] = True
    assert window._current_phase_kwargs(spec)["autophase"] is False


_REAL = os.path.join(os.path.dirname(__file__), "..", "test_data", "23Na_MQMAS_275")


@pytest.mark.skipif(not os.path.isdir(_REAL), reason="real 23Na MQMAS (expt 275) not in test_data/")
def test_real_mqmas_reopened_with_saved_phase_matches_topspin(tmp_path):
    """Expt 275 (Na2CdFe(CN)6, mp3qzfil, States-TPPI) with a sidecar
    holding phase0 = 0 reopens as TopSpin shows it."""
    import shutil
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from opennmr.gui.main_window import MainWindow
    from opennmr.io import persistence
    path = str(tmp_path / "275")
    shutil.copytree(_REAL, path, ignore=shutil.ignore_patterns("opennmr_*"))
    window = MainWindow()
    window._load(path, vendor="bruker")
    spec = window.session.spectra[-1]
    spec.metadata.raw["phase0"] = 0.0
    spec.metadata.raw["phase1"] = 0.0
    persistence.save_processing_sidecar(spec)
    again = MainWindow()
    again._load(path, vendor="bruker")
    topspin = bruker.read_2d_processed(path, 1)
    a, b = again.session.spectra[-1].data.ravel(), topspin.data.ravel()
    assert np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b) > 0.99
