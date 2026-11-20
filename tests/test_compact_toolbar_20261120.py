"""Notes 8.5: a compact toolbar that shrinks the top buttons, not the font."""
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets.wrapping_tabs import WrappingTabWidget
from test_gui import qapp  # noqa: F401


def _icon():
    pixmap = QPixmap(64, 64)
    pixmap.fill()
    return QIcon(pixmap)


def _tabs(n=3):
    tabs = WrappingTabWidget()
    for i in range(n):
        tabs.addTab(QWidget(), f"Tab {i}", icon=_icon())
    return tabs


def test_compact_shrinks_icons_and_padding_but_not_the_font(qapp):
    tabs = _tabs()
    button = tabs._buttons[0]
    font = button.font().pointSizeF()
    big_icon, big_hint = button.iconSize(), button.sizeHint()
    tabs.setCompact(True)
    assert tabs.isCompact()
    assert button.iconSize().width() < big_icon.width()
    assert button.sizeHint().height() < big_hint.height()
    assert button.font().pointSizeF() == font                       # the text is untouched
    tabs.setCompact(False)
    assert button.iconSize() == big_icon and button.styleSheet() == ""


def test_buttons_added_after_compact_is_set_are_compact_too(qapp):
    tabs = _tabs(1)
    tabs.setCompact(True)
    tabs.addTab(QWidget(), "Late", icon=_icon())
    assert tabs._buttons[1].iconSize() == WrappingTabWidget._COMPACT_ICON_SIZE


def test_text_only_tabs_keep_their_icon_size_untouched(qapp):
    tabs = WrappingTabWidget()
    tabs.addTab(QWidget(), "Plain")
    default = tabs._buttons[0].iconSize()
    tabs.setCompact(True)
    assert tabs._buttons[0].iconSize() == default                    # no icon: nothing to shrink


def test_the_setting_shrinks_the_launcher_and_is_remembered(qapp):
    window = MainWindow()
    window.side_tabs.setCompact(False)
    normal = window.tile_bar.sizeHint().height()
    font = window.font().pointSizeF()
    window.compact_toolbar_checkbox.setChecked(True)
    assert window.side_tabs.isCompact()
    assert window._ui_settings["compact_toolbar"] is True
    assert window.tile_bar.sizeHint().height() < normal
    assert window.font().pointSizeF() == font                        # ...without touching the font
    window.compact_toolbar_checkbox.setChecked(False)
    assert not window.side_tabs.isCompact() and window._ui_settings["compact_toolbar"] is False


def test_a_saved_compact_setting_is_applied_at_start(qapp, monkeypatch):
    first = MainWindow()
    first._ui_settings["compact_toolbar"] = True
    first._apply_saved_ui_settings()
    assert first.side_tabs.isCompact() and first.compact_toolbar_checkbox.isChecked()
