"""Autophase on a 2-D asks first.

A full 2-D autophase (every row of a true 2-D, or every slice of a
pseudo-2D with "Apply to every slice" ticked) is a grid search over the
whole block and can take a long time, so the GUI warns, points at the
quicker extract-a-slice route, and only goes ahead on *Yes*. A 1-D, or
one slice of a pseudo-2D, is not asked about.

These tests opt out of conftest's automatic *Yes*
(``full_2d_autophase_prompt``) and answer the real question themselves.
"""
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from opennmr.gui.main_window import MainWindow
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from test_gui import _load_spectrum_directly, _make_1d_dataset, _make_raw_2d_dataset

pytestmark = pytest.mark.full_2d_autophase_prompt


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _record_questions(monkeypatch, answer):
    asked = []

    def question(parent, title, text, buttons=None, default=None):
        asked.append({"title": title, "text": text, "buttons": buttons, "default": default})
        return answer

    monkeypatch.setattr(QMessageBox, "question", question)
    return asked


def _pseudo2d_window(tmp_path, monkeypatch, apply_all):
    window = MainWindow()
    window._load(_make_raw_2d_dataset(tmp_path, monkeypatch, n_f1=4), vendor="bruker")
    assert window._phase_mode == "pseudo2d"
    window.apply_all_slices_checkbox.setChecked(apply_all)
    return window


def test_warning_text_says_slow_suggests_a_slice_and_asks():
    text = MainWindow.FULL_2D_AUTOPHASE_WARNING
    assert text.startswith("Warning: auto-phasing a 2-D spectrum can take a long time.")
    assert "single slice" in text and "entire dataset" in text
    # names the real controls for the quicker route
    assert "Extract Slice" in text and "Apply Phase to Source 2D" in text
    assert text.rstrip().endswith("Would you still like to proceed with full 2-D autophasing?")


def test_pseudo2d_every_slice_asks_with_yes_no_and_no_as_default(qapp, tmp_path, monkeypatch):
    window = _pseudo2d_window(tmp_path, monkeypatch, apply_all=True)
    asked = _record_questions(monkeypatch, QMessageBox.StandardButton.No)

    window._on_autophase_clicked()

    assert len(asked) == 1
    assert asked[0]["text"] == MainWindow.FULL_2D_AUTOPHASE_WARNING
    assert asked[0]["buttons"] == QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    assert asked[0]["default"] == QMessageBox.StandardButton.No


def test_no_leaves_the_data_and_phase_untouched(qapp, tmp_path, monkeypatch):
    window = _pseudo2d_window(tmp_path, monkeypatch, apply_all=True)
    spectrum = window.session.spectra[window._selected_index()]
    before = spectrum.complex_data.copy()
    _record_questions(monkeypatch, QMessageBox.StandardButton.No)
    called = []
    monkeypatch.setattr(window, "_autophase_each_slice", lambda: called.append(True))

    window._on_autophase_clicked()

    assert called == []
    np.testing.assert_array_equal(spectrum.complex_data, before)
    assert not window.revert_autophase_button.isEnabled()
    assert "cancelled" in window.statusBar().currentMessage()


def test_yes_runs_the_full_autophase(qapp, tmp_path, monkeypatch):
    window = _pseudo2d_window(tmp_path, monkeypatch, apply_all=True)
    _record_questions(monkeypatch, QMessageBox.StandardButton.Yes)
    called = []
    monkeypatch.setattr(window, "_autophase_each_slice", lambda: called.append(True))

    window._on_autophase_clicked()

    assert called == [True]


def test_one_slice_of_a_pseudo2d_is_not_asked_about(qapp, tmp_path, monkeypatch):
    window = _pseudo2d_window(tmp_path, monkeypatch, apply_all=False)
    asked = _record_questions(monkeypatch, QMessageBox.StandardButton.No)

    window._on_autophase_clicked()

    assert asked == []
    assert window.revert_autophase_button.isEnabled()  # it did run


def test_a_1d_is_not_asked_about(qapp, tmp_path, monkeypatch):
    window = MainWindow()
    window._load(_make_1d_dataset(tmp_path, n=64), vendor="bruker")
    asked = _record_questions(monkeypatch, QMessageBox.StandardButton.No)

    window._on_autophase_clicked()

    assert asked == []
    assert window.revert_autophase_button.isEnabled()


def test_a_true_2d_is_asked_about(qapp, monkeypatch):
    ppm = np.linspace(-10.0, 10.0, 150)
    rows = np.array([np.exp(-((ppm - 1.0) ** 2) / 0.2) * (i + 1) for i in range(3)])
    axes = [NMRAxis(kind="frequency", ppm=np.linspace(0, 1, 3), hz=np.linspace(0, 100, 3)),
            NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)]
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=False)
    spectrum = NMRSpectrum(data=rows * np.cos(0.7), axes=axes, metadata=metadata,
                           data_imag=rows * np.sin(0.7))
    window = MainWindow()
    _load_spectrum_directly(window, spectrum)
    assert window._phase_mode == "2d"
    asked = _record_questions(monkeypatch, QMessageBox.StandardButton.No)
    phase0 = window.phase0_spin.value()

    window._on_autophase_clicked()
    assert len(asked) == 1
    assert window.phase0_spin.value() == phase0  # declined: unchanged

    _record_questions(monkeypatch, QMessageBox.StandardButton.Yes)
    window._on_autophase_clicked()
    assert window.phase0_spin.value() != phase0  # accepted: it ran


def test_edit_menu_autophase_asks_too(qapp, tmp_path, monkeypatch):
    window = _pseudo2d_window(tmp_path, monkeypatch, apply_all=True)
    asked = _record_questions(monkeypatch, QMessageBox.StandardButton.No)
    action = next(a for menu in window.menuBar().actions() if menu.menu()
                  for a in menu.menu().actions() if a.text().replace("&", "") == "Autophase")

    action.trigger()

    assert len(asked) == 1
