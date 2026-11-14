"""A stable ID for every Open experiments entry, and stored rows that
follow their experiment (Phase 3, step 1 of the Open experiments tree;
agreed 2026-11-13: "a stable ID per entry, replacing row indices")."""

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.main_window import MainWindow
from opennmr.gui.session import Session, TrackedRow

from test_gui import qapp  # noqa: F401 -- fixture
from test_gui_fit_sim_state import _fit_quad, _quad_spectrum


def _spectrum(value=1.0):
    ppm = np.linspace(10, 0, 64)
    return NMRSpectrum(data=np.full(64, value), axes=[NMRAxis(kind="frequency", ppm=ppm)],
                       metadata=NMRMetadata(vendor="simulated", nucleus="1H", sfo1=400.0))


def _session(n=3):
    session = Session()
    for i in range(n):
        session.add_derived(_spectrum(i), f"s{i}")
    return session


# ---------------------------------------------------------------------------
# Session IDs
# ---------------------------------------------------------------------------

def test_ids_are_unique_and_follow_their_entry():
    session = _session(3)
    ids = [session.entry_id(i) for i in range(3)]
    assert len(set(ids)) == 3
    assert [session.entry_id(i) for i in range(3)] == ids          # the same on asking again
    session.swap(0, 2)
    assert [session.index_of_id(e) for e in ids] == [2, 1, 0]
    session.remove(0)                                               # the entry that was s2
    assert session.index_of_id(ids[2]) is None
    assert [session.index_of_id(ids[1]), session.index_of_id(ids[0])] == [0, 1]
    # Never reused.
    assert session.entry_id(session.add_derived(_spectrum(), "new")) not in ids


def test_an_id_survives_what_replaces_the_spectrum_object():
    session = _session(2)
    first = session.entry_id(0)
    session.set_nucleus(0, "13C")
    session.set_mass(0, 0.01)
    session.replace_data(0, np.zeros(64))
    assert session.entry_id(0) == first and session.index_of_id(first) == 0


def test_an_entry_added_behind_the_sessions_back_still_gets_an_id():
    # Tests (and any older route) append to the lists directly.
    session = _session(1)
    session.spectra.append(_spectrum())
    session._load_info.append({"path": "synthetic", "vendor": "bruker", "kwargs": {}})
    assert session.index_of_id(session.entry_id(1)) == 1


def test_another_sessions_id_is_not_this_sessions():
    a, b = _session(2), _session(2)
    theirs = a.entry_id(0)
    assert a.owns_id(theirs) and not b.owns_id(theirs)
    assert b.index_of_id(theirs) is None
    assert not b.owns_id(None) and b.index_of_id(None) is None


# ---------------------------------------------------------------------------
# A stored row follows its experiment
# ---------------------------------------------------------------------------

class _Holder:
    row = TrackedRow()

    def __init__(self, session):
        self.session = session


def test_a_tracked_row_follows_removal_and_reordering():
    session = _session(4)
    holder = _Holder(session)
    assert holder.row is None
    holder.row = 2
    session.remove(0)
    assert holder.row == 1                    # the same experiment, one row up
    session.swap(1, 2)
    assert holder.row == 2
    session.remove(2)
    assert holder.row is None                 # removed: never the row that took its place
    holder.row = 0
    holder.row = None
    assert holder.row is None
    holder.row = 99                           # not a row: kept as given
    assert holder.row == 99
    session.clear()
    holder.row = 0
    assert holder.row == 0


def test_a_pending_save_cannot_land_on_another_spectrum(qapp):  # noqa: F811
    window = MainWindow()
    first = _quad_spectrum(window, "first")
    second = _quad_spectrum(window, "second")
    window._sidecar_save_timer.stop()
    window._sidecar_save_pending_index = second
    window._phase_target_index = second
    window._remove_rows([first])
    # "second" is row 0 now (or the selection change it caused dealt with
    # it): never the stale row 1.
    assert window._sidecar_save_pending_index in (0, None)
    assert window._phase_target_index in (0, None)
    window._sidecar_save_pending_index = 0
    window._remove_rows([0])
    assert window._sidecar_save_pending_index is None
    assert window._phase_target_spectrum() is None
    window._run_debounced_sidecar_save()                # nothing to save: no error
    window.close()


# ---------------------------------------------------------------------------
# A fit result knows its experiment
# ---------------------------------------------------------------------------

@pytest.fixture
def quiet(monkeypatch):
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)


def test_a_fit_result_is_found_by_its_experiment_not_its_old_row(qapp, quiet):  # noqa: F811
    window = MainWindow()
    other = _quad_spectrum(window, "other")
    target = _quad_spectrum(window, "target")              # selected
    panel = _fit_quad(window)
    result = panel.result
    assert result is not None
    entry = window.session.entry_id(target)
    assert result[Session.ID_KEY] == entry
    assert window._result_row(result) == target
    # Reprocessing replaces the spectrum object; a row above is removed.
    window.session.set_mass(target, 0.02)
    assert window._row_of_spectrum(result["spectrum"]) == -1
    window._remove_rows([other])
    assert window._result_row(result) == 0                 # not the stale row 1
    assert window.session.index_of_id(entry) == 0
    # Its experiment removed: no row, never another spectrum's.
    extra = _quad_spectrum(window, "extra")
    window._remove_rows([0])
    assert extra == 1 and len(window.session) == 1
    assert window._result_row(result) is None
    window.close()


def test_results_without_an_id_fall_back_to_their_row(qapp):  # noqa: F811
    window = MainWindow()
    index = _quad_spectrum(window, "only")
    assert window._result_row({"spectrum_index": index}) == index
    assert window._result_row({"index": index}) == index
    assert window._result_row({"spectrum_index": 7}) is None
    # An ID saved in another session is ignored, not matched or trusted.
    assert window._result_row({Session.ID_KEY: "deadbeef-e1", "spectrum_index": index}) == index
    assert window._result_row(None) is None
    window.close()
