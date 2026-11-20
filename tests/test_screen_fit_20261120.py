"""Notes 8.6: a long pop-up stays on the screen and scrolls."""
from PySide6.QtWidgets import QApplication, QMessageBox, QScrollArea

from opennmr.gui import screen_fit
from test_gui import qapp  # noqa: F401

LONG = "\n".join(f"line {i}: the release notes go on and on" for i in range(400))


def _screen_height():
    return QApplication.primaryScreen().availableGeometry().height()


def test_a_long_message_is_capped_to_the_screen_and_scrolls(qapp):
    box = QMessageBox()
    box.setText("Update installed")
    box.setInformativeText(LONG)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    assert screen_fit.fit_message_box_to_screen(box) is True
    box.show()
    qapp.processEvents()
    assert box.height() <= _screen_height()
    scroll = box.findChild(QScrollArea, "qt_msgbox_informativelabel_scroll")
    assert scroll is not None and scroll.maximumHeight() < _screen_height()
    # the buttons are still inside the box, so still on the screen
    ok = box.button(QMessageBox.StandardButton.Ok)
    assert ok.isVisible() and ok.geometry().bottom() <= box.height()
    box.close()


def test_a_short_message_is_left_alone(qapp):
    box = QMessageBox()
    box.setText("Done.")
    assert screen_fit.fit_message_box_to_screen(box) is False
    assert box.findChild(QScrollArea) is None


def test_fitting_twice_does_nothing_more(qapp):
    box = QMessageBox()
    box.setInformativeText(LONG)
    assert screen_fit.fit_message_box_to_screen(box) is True
    assert screen_fit.fit_message_box_to_screen(box) is False
    assert len(box.findChildren(QScrollArea)) == 1


def test_the_filter_fits_boxes_when_they_are_shown(qapp):
    assert screen_fit.install_screen_fit(qapp) is screen_fit.install_screen_fit(qapp)   # once only
    box = QMessageBox()
    box.setText(LONG)
    box.show()
    qapp.processEvents()
    assert box.height() <= _screen_height()
    assert box.findChild(QScrollArea, "qt_msgbox_label_scroll") is not None
    box.close()


def test_main_window_installs_the_filter(qapp):
    from opennmr.gui.main_window import MainWindow
    MainWindow()
    assert getattr(qapp, "_opennmr_screen_fit", None) is not None
