"""Which way a Bruker 2D opens: genuine 2D when TopSpin itself
Fourier-transformed F1, pseudo-2D series otherwise; the user's own
choice always wins (explicit user request: stop having to switch every
genuine 2D by hand).

The synthetic experiments mirror the parameter patterns measured on a
real 43-experiment folder (see io.bruker.suggest_pseudo_2d): EXSY with
proc2s FT_mod 4, relaxation series with FT_mod 0 or NUC1 off."""
import json
import os

import pytest

from opennmr.gui.main_window import MainWindow
from opennmr.io import bruker, persistence

from test_gui import qapp, _make_2d_dataset, _write_params  # noqa: F401


def _set_ft_mod(path, ft_mod):
    proc2s = os.path.join(path, "pdata", "1", "proc2s")
    params = bruker.parse_jcampdx_params(proc2s)
    params["FT_mod"] = ft_mod
    _write_params(proc2s, params)


@pytest.mark.parametrize("nuc2, ft_mod, pseudo, why", [
    ("1H", 4, False, "Fourier-transformed F1"),
    ("1H", 0, True, "xf2 only"),
    ("off", 4, True, "no nucleus"),
    ("1H", None, True, "not processed as a 2D"),
])
def test_rule(tmp_path, nuc2, ft_mod, pseudo, why):
    path = _make_2d_dataset(tmp_path, f1_nuc=nuc2)
    if ft_mod is not None:
        _set_ft_mod(path, ft_mod)
    got, basis = bruker.suggest_pseudo_2d(path)
    assert got is pseudo and why in basis
    assert bruker.recorded_f1_transform(path) == ft_mod


def test_a_transformed_2d_opens_as_a_genuine_2d(qapp, tmp_path):  # noqa: F811
    path = _make_2d_dataset(tmp_path, n_f1=4, n_f2=8)
    _set_ft_mod(path, 4)
    window = MainWindow()
    window._load(path, vendor="bruker")
    spectrum = window.session.spectra[window._selected_index()]
    assert not spectrum.is_pseudo_2d
    assert "Fourier-transformed F1" in window._status_experiment.toolTip()
    assert window.process_indirect_button.isChecked()


def test_an_xf2_only_series_still_opens_as_pseudo_2d(qapp, tmp_path):  # noqa: F811
    path = _make_2d_dataset(tmp_path)
    _set_ft_mod(path, 0)
    window = MainWindow()
    window._load(path, vendor="bruker")
    assert window.session.spectra[window._selected_index()].is_pseudo_2d


def test_the_users_choice_wins_and_is_remembered(qapp, tmp_path):  # noqa: F811
    path = _make_2d_dataset(tmp_path)
    _set_ft_mod(path, 4)                       # would open as a genuine 2D...
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(False)          # ...but the user says series
    persistence.save_metadata_sidecar(window.session.spectra[window._selected_index()])
    saved = json.load(open(persistence.metadata_sidecar_path(path)))
    assert saved["pseudo_2d"] is True and saved["pseudo_2d_chosen"] is True

    again = MainWindow()
    again._load(path, vendor="bruker")
    assert again.session.spectra[again._selected_index()].is_pseudo_2d
    assert "your saved choice" in again._status_experiment.toolTip()


def test_a_saved_state_that_was_never_chosen_does_not_block_the_rule(qapp, tmp_path):  # noqa: F811
    # Before this change every 2D was forced to pseudo-2D and that state
    # was written to the sidecar on any save (phasing, mass...). Such a
    # "pseudo_2d": true is not a choice and must not keep a transformed
    # 2D as a series.
    path = _make_2d_dataset(tmp_path)
    _set_ft_mod(path, 4)
    with open(persistence.metadata_sidecar_path(path), "w") as fh:
        json.dump({"opennmr_metadata_version": 1, "vendor": "bruker", "pseudo_2d": True}, fh)
    window = MainWindow()
    window._load(path, vendor="bruker")
    assert not window.session.spectra[window._selected_index()].is_pseudo_2d


def test_choice_survives_later_saves(qapp, tmp_path):  # noqa: F811
    path = _make_2d_dataset(tmp_path)
    window = MainWindow()
    window._load(path, vendor="bruker")
    window.process_indirect_button.setChecked(True)            # choose genuine 2D
    spectrum = window.session.spectra[window._selected_index()]
    persistence.save_metadata_sidecar(spectrum)
    spectrum.metadata.raw.pop("pseudo_2d_chosen", None)        # as after a later reload
    persistence.save_metadata_sidecar(spectrum)
    saved = json.load(open(persistence.metadata_sidecar_path(path)))
    assert saved["pseudo_2d"] is False and saved["pseudo_2d_chosen"] is True
