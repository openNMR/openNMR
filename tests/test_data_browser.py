"""The Data-tab folder browser.

The behaviour that matters: experiments are the destinations, not the
files inside them. "People generally won't know to click the fid/ser."
"""
import os

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

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
