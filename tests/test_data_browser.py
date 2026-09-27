"""The Data-tab folder browser.

The behaviour that matters: experiments are the destinations, not the
files inside them. "People generally won't know to click the fid/ser."
"""
import os

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from opennmr import io
from opennmr.gui.widgets import data_browser as browser_module
from opennmr.gui.widgets.data_browser import _PATH_ROLE, _VENDOR_ROLE, DataBrowser


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _leaves(item, found=None):
    found = [] if found is None else found
    for i in range(item.childCount()):
        child = item.child(i)
        if child.data(0, _PATH_ROLE):
            found.append(child)
        _leaves(child, found)
    return found


def test_experiments_are_the_leaves_not_their_files(qapp):
    root = os.path.join("test_data", "Bruker Example Data", "Cyclosporine")
    if not os.path.isdir(root):
        pytest.skip(f"needs {root}")
    from opennmr.gui.main_window import MainWindow

    matches = MainWindow._scan_for_spectra(root)
    browser = DataBrowser()
    browser.set_matches(root, matches)

    leaves = _leaves(browser._tree.invisibleRootItem())
    assert leaves, "no experiments listed"
    for leaf in leaves:
        path = leaf.data(0, _PATH_ROLE)
        assert os.path.isdir(path)                  # an experiment folder...
        assert os.path.basename(path) not in ("fid", "ser", "pdata")
        assert leaf.data(0, _VENDOR_ROLE) == "bruker"


def test_leaves_are_annotated_from_peek_not_from_loading(qapp):
    # peek reads only the parameter files, so the tree can say what an
    # experiment is without opening anything large.
    root = os.path.join("test_data", "Bruker Example Data", "Cyclosporine")
    if not os.path.isdir(root):
        pytest.skip(f"needs {root}")
    from opennmr.gui.main_window import MainWindow

    browser = DataBrowser()
    browser.set_matches(root, MainWindow._scan_for_spectra(root))

    labels = [leaf.text(0) for leaf in _leaves(browser._tree.invisibleRootItem())]
    assert any("2D" in label for label in labels)
    assert any("1H" in label or "13C" in label for label in labels)


def test_nested_folders_group_and_describe_lazily(qapp):
    root = os.path.join("test_data", "Bruker Example Data")
    if not os.path.isdir(root):
        pytest.skip(f"needs {root}")
    from opennmr.gui.main_window import MainWindow

    browser = DataBrowser()
    browser.set_matches(root, MainWindow._scan_for_spectra(root))
    tree = browser._tree

    folders = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
               if not tree.topLevelItem(i).data(0, _PATH_ROLE)]
    assert folders, "expected experiments grouped under folders"

    folder = folders[0]
    # Not described until expanded -- pointing this at a big archive
    # must not peek hundreds of experiments up front.
    assert not folder.data(0, Qt.ItemDataRole.UserRole + 2)
    before = folder.child(0).text(0)

    browser._describe_children(folder)

    assert folder.data(0, Qt.ItemDataRole.UserRole + 2)
    assert folder.child(0).text(0) != before        # gained its summary


def test_activating_an_experiment_emits_it(qapp):
    root = os.path.join("test_data", "Bruker Example Data", "Cyclosporine")
    if not os.path.isdir(root):
        pytest.skip(f"needs {root}")
    from opennmr.gui.main_window import MainWindow

    browser = DataBrowser()
    browser.set_matches(root, MainWindow._scan_for_spectra(root))
    emitted = []
    browser.spectrumActivated.connect(lambda p, v: emitted.append((p, v)))

    leaf = _leaves(browser._tree.invisibleRootItem())[0]
    browser._on_item_activated(leaf)

    assert emitted == [(leaf.data(0, _PATH_ROLE), "bruker")]


def test_activating_a_folder_expands_instead_of_loading(qapp):
    root = os.path.join("test_data", "Bruker Example Data")
    if not os.path.isdir(root):
        pytest.skip(f"needs {root}")
    from opennmr.gui.main_window import MainWindow

    browser = DataBrowser()
    browser.set_matches(root, MainWindow._scan_for_spectra(root))
    emitted = []
    browser.spectrumActivated.connect(lambda p, v: emitted.append((p, v)))

    tree = browser._tree
    folder = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                  if not tree.topLevelItem(i).data(0, _PATH_ROLE))
    browser._on_item_activated(folder)

    assert emitted == []           # a folder is not loadable
    assert folder.isExpanded()


def test_empty_folder_says_so_rather_than_looking_broken(qapp, tmp_path):
    browser = DataBrowser()
    browser.set_matches(str(tmp_path), [])

    assert browser._tree.topLevelItemCount() == 0
    assert "No loadable experiments" in browser._status.text()


def test_undescribable_experiment_still_lists(qapp, tmp_path):
    # A folder peek cannot read is still loadable; refusing to list it
    # would be worse than listing it without a summary.
    expt = tmp_path / "expt"
    expt.mkdir()
    browser = DataBrowser()

    browser.set_matches(str(tmp_path), [(str(expt), "bruker")])

    leaves = _leaves(browser._tree.invisibleRootItem())
    assert len(leaves) == 1
    assert leaves[0].data(0, _PATH_ROLE) == str(expt)


def test_cloud_experiment_is_listed_without_being_peeked(qapp, tmp_path, monkeypatch):
    """The reason cloud detection exists: peeking reads the parameter
    files, and on a OneDrive/iCloud folder that *downloads* them -- so
    expanding a folder of a hundred cloud experiments would become a
    hundred sequential downloads. The experiment must still list, and
    must say plainly what it is."""
    from opennmr.io import cloud

    expt = tmp_path / "10"
    expt.mkdir()
    (expt / "acqus").write_text("##$NS= 16\n")
    monkeypatch.setattr(cloud, "parameters_are_local", lambda path: False)
    monkeypatch.setattr(cloud, "pending_download_bytes", lambda path: 633 * 1024**2)

    def fail(*args, **kwargs):
        raise AssertionError("peek() must not touch a cloud-backed experiment")

    monkeypatch.setattr(browser_module.io, "peek", fail)

    browser = DataBrowser()
    browser.set_matches(str(tmp_path), [(str(expt), "bruker")])

    leaf = _leaves(browser._tree.invisibleRootItem())[0]
    assert "in cloud" in leaf.text(0)
    assert "633 MB" in leaf.text(0)


def test_local_experiment_is_still_peeked(qapp, tmp_path, monkeypatch):
    """Cloud detection must not stop ordinary local data being
    described -- that is the common case."""
    from opennmr.io import cloud

    expt = tmp_path / "10"
    expt.mkdir()
    monkeypatch.setattr(cloud, "parameters_are_local", lambda path: True)
    monkeypatch.setattr(
        browser_module.io, "peek",
        lambda path, vendor=None: {"nucleus": "1H", "ndim": 2, "shape": (256, 1024)},
    )

    browser = DataBrowser()
    browser.set_matches(str(tmp_path), [(str(expt), "bruker")])

    leaf = _leaves(browser._tree.invisibleRootItem())[0]
    assert "1H" in leaf.text(0) and "256x1024" in leaf.text(0)
    assert "in cloud" not in leaf.text(0)


def test_double_click_loads_an_experiment_exactly_once(qapp, tmp_path):
    """One double-click is one load.

    `itemActivated` and `itemDoubleClicked` were both connected to the
    same handler, and a double-click emits *both* -- so every
    double-click in the browser loaded the same experiment twice: two
    identical rows in the list, and for a large 2D the whole
    multi-second read done twice. Real user report: "the folder browser
    shows each data set twice, I hope this isn't a double load."

    Both signals are emitted here, which is what Qt itself does for one
    double-click; the handler must respond to that gesture once."""
    expt = tmp_path / "10"
    expt.mkdir()
    browser = DataBrowser()
    activated = []
    browser.spectrumActivated.connect(lambda path, vendor: activated.append(path))
    browser.set_matches(str(tmp_path), [(str(expt), "bruker")])
    item = browser._tree.topLevelItem(0)

    browser._tree.itemDoubleClicked.emit(item, 0)
    browser._tree.itemActivated.emit(item, 0)

    assert activated == [str(expt)]


def test_return_key_also_activates(qapp, tmp_path):
    """`itemActivated` is kept rather than `itemDoubleClicked` because
    it is the platform's own activation signal -- Return counts, and so
    does a single click on a desktop configured for it."""
    expt = tmp_path / "10"
    expt.mkdir()
    browser = DataBrowser()
    activated = []
    browser.spectrumActivated.connect(lambda path, vendor: activated.append(path))
    browser.set_matches(str(tmp_path), [(str(expt), "bruker")])

    browser._tree.itemActivated.emit(browser._tree.topLevelItem(0), 0)

    assert activated == [str(expt)]


def test_leaf_summary_names_the_pulse_programme(qapp):
    """"PGSTE" or "hsqcetgp" identifies an experiment in a way no shape
    can -- a folder of 1H 1D 65536s is indistinguishable without it.
    Explicit user request, and all four vendors record one."""
    import os
    path = "test_data/bruker_T2/11"
    if not os.path.isdir(path):
        pytest.skip("test_data/bruker_T2 not present")
    summary = browser_module.DataBrowser._describe(path, "bruker")
    assert "hahn-echo" in summary


@pytest.mark.parametrize("vendor,path,expected", [
    ("bruker", "test_data/bruker_PFG/150", "diffSte"),
    ("varian", "test_data/varian_1d/Li7_8000ns_12s_june.fid", "onepul"),
    ("jeol", "test_data/jeol_1d/carbon.jdf", "carbon"),
    ("xpulse", "test_data/xpulse_PFG/4", "PGSTE"),
])
def test_every_vendors_peek_reports_a_pulse_programme(vendor, path, expected):
    import os
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    assert expected in str(io.peek(path, vendor=vendor).get("pulse_program"))


# -- multi-select: Ctrl-click and Shift-click ---------------------------
#
# User request: "select multiple experiments by doing either control and
# click (to select separate experiments that are not adjacent in the
# list of experiments in the folder) or shift and click (if the
# experiments are sequential)." These drive real mouse clicks with
# modifiers through the tree's viewport, rather than calling
# setSelected(), so they check the gesture and not just the bookkeeping.


def _browser_with_experiments(tmp_path, names, subfolder=None):
    matches = []
    for name in names:
        parent = tmp_path / subfolder if subfolder else tmp_path
        expt = parent / name
        expt.mkdir(parents=True)
        matches.append((str(expt), "bruker"))
    browser = DataBrowser()
    browser.resize(400, 600)
    browser.show()
    browser.set_matches(str(tmp_path), matches)
    return browser, [path for path, _vendor in matches]


def _click(browser, item, modifier=Qt.KeyboardModifier.NoModifier):
    from PySide6.QtTest import QTest

    tree = browser._tree
    tree.scrollToItem(item)
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, modifier,
                     tree.visualItemRect(item).center())


def _opened(browser):
    opened = []
    browser.spectrumActivated.connect(lambda path, vendor: opened.append(path))
    return opened


def test_ctrl_click_selects_experiments_that_are_not_adjacent(qapp, tmp_path):
    browser, paths = _browser_with_experiments(tmp_path, ["10", "11", "12", "13", "14"])
    tree = browser._tree
    opened = _opened(browser)

    _click(browser, tree.topLevelItem(0))
    _click(browser, tree.topLevelItem(2), Qt.KeyboardModifier.ControlModifier)
    _click(browser, tree.topLevelItem(4), Qt.KeyboardModifier.ControlModifier)

    assert [p for p, _v in browser.selected_paths()] == [paths[0], paths[2], paths[4]]
    assert browser._open_selected_button.isEnabled()
    assert browser._open_selected_button.text() == "Open 3 Selected"

    browser._open_selected_button.click()
    assert opened == [paths[0], paths[2], paths[4]]


def test_ctrl_click_on_a_selected_experiment_deselects_it(qapp, tmp_path):
    browser, paths = _browser_with_experiments(tmp_path, ["10", "11", "12"])
    tree = browser._tree

    _click(browser, tree.topLevelItem(0))
    _click(browser, tree.topLevelItem(2), Qt.KeyboardModifier.ControlModifier)
    _click(browser, tree.topLevelItem(0), Qt.KeyboardModifier.ControlModifier)

    assert [p for p, _v in browser.selected_paths()] == [paths[2]]
    assert browser._open_selected_button.text() == "Open Selected"


def test_shift_click_selects_a_run_of_experiments(qapp, tmp_path):
    browser, paths = _browser_with_experiments(tmp_path, ["10", "11", "12", "13", "14"])
    tree = browser._tree
    opened = _opened(browser)

    _click(browser, tree.topLevelItem(1))
    _click(browser, tree.topLevelItem(3), Qt.KeyboardModifier.ShiftModifier)

    assert [p for p, _v in browser.selected_paths()] == paths[1:4]
    browser._open_selected_button.click()
    assert opened == paths[1:4]


def test_a_backwards_shift_range_still_opens_top_to_bottom(qapp, tmp_path):
    # Opened in the order they are listed, not the order they were
    # clicked -- a range picked from the bottom up should not land in
    # the spectrum list reversed.
    browser, paths = _browser_with_experiments(tmp_path, ["10", "11", "12", "13"])
    tree = browser._tree
    opened = _opened(browser)

    _click(browser, tree.topLevelItem(3))
    _click(browser, tree.topLevelItem(0), Qt.KeyboardModifier.ShiftModifier)
    browser._open_selected()

    assert opened == paths


def test_return_on_a_multi_selection_opens_all_of_it(qapp, tmp_path):
    browser, paths = _browser_with_experiments(tmp_path, ["10", "11", "12"])
    tree = browser._tree
    opened = _opened(browser)

    _click(browser, tree.topLevelItem(0))
    _click(browser, tree.topLevelItem(2), Qt.KeyboardModifier.ControlModifier)
    # Return activates the item holding focus; Qt emits itemActivated.
    tree.itemActivated.emit(tree.currentItem(), 0)

    assert opened == [paths[0], paths[2]]


def test_plain_double_click_after_a_multi_selection_opens_just_that_one(qapp, tmp_path):
    # The first click of an unmodified double-click resets the selection
    # to the item under the cursor, so this must not open the earlier
    # multi-selection as a side effect.
    from PySide6.QtTest import QTest

    browser, paths = _browser_with_experiments(tmp_path, ["10", "11", "12"])
    tree = browser._tree
    opened = _opened(browser)

    _click(browser, tree.topLevelItem(0))
    _click(browser, tree.topLevelItem(1), Qt.KeyboardModifier.ControlModifier)
    item = tree.topLevelItem(2)
    center = tree.visualItemRect(item).center()
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, center)
    tree.itemActivated.emit(item, 0)   # what the second click of the pair does

    assert opened == [paths[2]]


def test_folders_in_a_selection_are_skipped_not_opened_wholesale(qapp, tmp_path):
    (tmp_path / "run2" / "20").mkdir(parents=True)
    matches = [(str(tmp_path / n), "bruker") for n in ("10", "11")]
    for path, _vendor in matches:
        os.mkdir(path)
    matches.append((str(tmp_path / "run2" / "20"), "bruker"))
    browser = DataBrowser()
    browser.resize(400, 600)
    browser.show()
    browser.set_matches(str(tmp_path), matches)
    tree = browser._tree
    items = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    folder = next(i for i in items if not i.data(0, _PATH_ROLE))
    leaves = [i for i in items if i.data(0, _PATH_ROLE)]

    for item in items:
        item.setSelected(True)

    assert folder.isSelected()
    assert [p for p, _v in browser.selected_paths()] == [
        leaf.data(0, _PATH_ROLE) for leaf in leaves]
    assert str(tmp_path / "run2" / "20") not in [p for p, _v in browser.selected_paths()]


def test_open_selected_is_disabled_with_nothing_or_only_a_folder_selected(qapp, tmp_path):
    browser, _paths = _browser_with_experiments(tmp_path, ["20"], subfolder="run2")
    assert not browser._open_selected_button.isEnabled()

    folder = browser._tree.topLevelItem(0)
    assert not folder.data(0, _PATH_ROLE)
    folder.setSelected(True)
    assert not browser._open_selected_button.isEnabled()


def test_choosing_a_new_folder_resets_open_selected(qapp, tmp_path):
    browser, _paths = _browser_with_experiments(tmp_path, ["10", "11"])
    browser._tree.topLevelItem(0).setSelected(True)
    browser._tree.topLevelItem(1).setSelected(True)
    assert browser._open_selected_button.text() == "Open 2 Selected"

    browser.set_matches(str(tmp_path), [])

    assert not browser._open_selected_button.isEnabled()
    assert browser._open_selected_button.text() == "Open Selected"


def test_main_window_opens_each_selected_experiment_once(qapp, tmp_path, monkeypatch):
    # End to end through MainWindow's own wiring: every selected
    # experiment reaches _open_one (and so its already-open check),
    # once each, in tree order.
    from opennmr.gui.main_window import MainWindow

    window = MainWindow()
    calls = []
    monkeypatch.setattr(window, "_open_one", lambda path, vendor: calls.append((path, vendor)))
    paths = []
    for name in ("10", "11", "12", "13"):
        (tmp_path / name).mkdir()
        paths.append(str(tmp_path / name))
    window.data_browser.set_matches(str(tmp_path), [(p, "bruker") for p in paths])
    tree = window.data_browser._tree
    tree.topLevelItem(1).setSelected(True)
    tree.topLevelItem(3).setSelected(True)

    window.data_browser._open_selected_button.click()

    assert calls == [(paths[1], "bruker"), (paths[3], "bruker")]
