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
