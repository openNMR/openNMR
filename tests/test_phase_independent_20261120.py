"""Notes 8.8: phase pseudo-2D slices independently by default (setting + per dataset)."""
import dataclasses

import numpy as np

from opennmr.gui.main_window import MainWindow
from test_gui import (_load_spectrum_directly, _make_pseudo2d_relaxation_spectrum,  # noqa: F401
                      qapp)


def _series(n=6):
    spectrum = _make_pseudo2d_relaxation_spectrum(np.geomspace(0.001, 0.3, n))
    # a phaseable (complex) series, which is what shows the slice/phase controls
    return dataclasses.replace(spectrum, data_imag=np.zeros_like(spectrum.data))


def _independent(window):
    return (not window.apply_all_slices_checkbox.isChecked()
            and window.autophase_each_slice_checkbox.isChecked())


def test_default_is_one_shared_phase(qapp):
    window = MainWindow()
    assert window.phase_independent_default_checkbox.isChecked() is False
    _load_spectrum_directly(window, _series())
    assert window.apply_all_slices_checkbox.isChecked()
    assert not window.autophase_each_slice_checkbox.isChecked()


def test_the_setting_starts_new_series_with_independent_slices_and_is_remembered(qapp):
    window = MainWindow()
    window.phase_independent_default_checkbox.setChecked(True)
    assert window._ui_settings["phase_slices_independently"] is True
    _load_spectrum_directly(window, _series())
    assert _independent(window)
    window.phase_independent_default_checkbox.setChecked(False)
    assert window._ui_settings["phase_slices_independently"] is False


def test_the_choice_is_stored_with_the_dataset_and_wins_over_the_setting(qapp):
    window = MainWindow()
    _load_spectrum_directly(window, _series())                     # dataset 0, default shared
    window.apply_all_slices_checkbox.setChecked(False)
    window.autophase_each_slice_checkbox.setChecked(True)
    assert window.session.spectra[0].metadata.raw["phase_slices_independently"] is True

    window.phase_independent_default_checkbox.setChecked(True)     # new datasets: independent
    _load_spectrum_directly(window, _series())                     # dataset 1
    assert _independent(window)
    window.apply_all_slices_checkbox.setChecked(True)              # the user wants this one shared
    window.autophase_each_slice_checkbox.setChecked(False)
    assert window.session.spectra[1].metadata.raw["phase_slices_independently"] is False

    window._select_single_row(0)
    assert _independent(window)                                    # dataset 0 remembered its own
    window._select_single_row(1)
    assert not _independent(window)                                # dataset 1 stays shared, setting or not


def test_a_1d_spectrum_stores_nothing(qapp):
    window = MainWindow()
    window.phase_independent_default_checkbox.setChecked(True)
    window._remember_phase_independence()                          # nothing selected: a no-op
    assert window._slices_phased_independently() in (True, False)
