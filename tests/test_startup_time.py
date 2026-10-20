"""Startup time: the stylesheet and wheel guard go in once the window
is built, with no change to what the user sees.

Measured 2026-10-13 (round in TODO.md): installing the application
stylesheet *before* building MainWindow cost ~3.5 s of a ~7.5 s build
-- with a sheet active, Qt re-polishes a widget's whole subtree on every
reparent, and the window is assembled bottom-up -- and the application-
wide wheel guard another ~2.3 s (every one of ~134,000 construction
events handed to Python). Both now go in at the end of __init__.

These tests pin the *order* (a wall-clock limit would flake on a busy CI
runner) and, above all, that nothing the user set is lost by it: the
first attempt at this change silently dropped a saved interface font
size, which the existing tests -- they check the spin box's value, not
the fonts -- did not catch.
"""

import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton, QWidget

from opennmr.gui import app as app_module
from opennmr.gui import main_window as main_window_module
from opennmr.gui import style as style_module
from opennmr.gui.main_window import MainWindow
from opennmr.gui.widgets import wheel_guard
from opennmr.gui.widgets.collapsible_section import HEADER_FONT_SCALE, CollapsibleSection


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def restore_system_theme(qapp):
    """A theme set by a test is process-wide; put "system" back."""
    yield
    style_module.apply_theme("system", qapp)


def _save_ui_settings(settings: dict) -> None:
    # conftest points _UI_SETTINGS_PATH at this test's tmp_path.
    with open(MainWindow._UI_SETTINGS_PATH, "w", encoding="utf-8") as handle:
        json.dump(settings, handle)


def _is_dark(widget: QWidget) -> bool:
    return widget.palette().color(QPalette.ColorRole.Window).lightness() < 128


# -- order: built first, then styled, then guarded ----------------------

def test_stylesheet_is_installed_once_after_the_widgets_exist(qapp, monkeypatch):
    calls = []
    real = style_module.apply_theme

    def recording(mode="system", app=None, radius=style_module.RADIUS, stylesheet=True):
        window = getattr(recording, "window", None)
        calls.append((stylesheet, len(window.findChildren(QWidget)) if window else 0))
        return real(mode, app, radius, stylesheet)

    monkeypatch.setattr(style_module, "apply_theme", recording)
    monkeypatch.setattr(main_window_module, "apply_theme", recording)
    real_init_ui = MainWindow._apply_startup_ui

    def note_window(self):
        recording.window = self
        real_init_ui(self)

    monkeypatch.setattr(MainWindow, "_apply_startup_ui", note_window)
    window = MainWindow()

    sheets = [count for with_sheet, count in calls if with_sheet]
    assert len(sheets) == 1, calls          # once, not once per restore step
    # ...and only when the window is (essentially) complete.
    assert sheets[0] > 0.9 * len(window.findChildren(QWidget))
    # The style and palette themselves go in first (cheap, and widgets
    # that cache their first size hint must measure in the right style).
    assert calls[0][0] is False
    assert style_module._STYLE_FORCED       # Fusion (its name reads "" under a sheet)
    assert style_module._MARKER in qapp.styleSheet()


def test_theme_goes_in_before_the_reflow_and_the_wheel_guard_after(qapp, monkeypatch):
    order = []
    real_reflow = MainWindow._make_side_panels_reflow
    real_ui = MainWindow._apply_startup_ui
    real_install = wheel_guard.install

    monkeypatch.setattr(MainWindow, "_apply_startup_ui",
                        lambda self: (order.append("theme"), real_ui(self)))
    monkeypatch.setattr(MainWindow, "_make_side_panels_reflow",
                        lambda self: (order.append("reflow"), real_reflow(self)))
    monkeypatch.setattr(wheel_guard, "install",
                        lambda app=None: (order.append("guard"), real_install(app))[1])
    MainWindow()
    # The reflow measures widgets, so it must see them in their final
    # style; nothing can be scrolled before the window is shown.
    assert order == ["theme", "reflow", "guard"]
    assert getattr(qapp, "_opennmr_wheel_guard", None) is not None


def test_app_main_leaves_the_theme_to_the_window(qapp, monkeypatch):
    """app.main() installing the sheet first would bring the slow start
    straight back (the window would then build under it)."""
    applied = []
    monkeypatch.setattr(style_module, "apply_theme",
                        lambda *a, **k: applied.append((a, k)) or "light")
    built = []

    class FakeWindow:
        def __init__(self):
            built.append(list(applied))

        def show(self):
            pass

        def maybe_show_welcome(self):
            pass

    monkeypatch.setattr(app_module, "MainWindow", FakeWindow)
    monkeypatch.setattr(app_module, "QApplication", lambda argv: qapp)
    monkeypatch.setattr(type(qapp), "exec", lambda self: 0)
    assert app_module.main() == 0
    assert built == [[]]                    # nothing themed before the window


# -- nothing the user set is lost -----------------------------------------

def test_saved_dark_theme_is_applied_not_overridden_by_system(qapp, restore_system_theme):
    _save_ui_settings({"ui_theme": "dark"})
    window = MainWindow()
    assert window.ui_theme_combo.currentData() == "dark"
    assert _is_dark(window)
    assert all(_is_dark(w) for w in window.findChildren(QComboBox)[:50])


def test_hidden_option_dialogs_take_the_saved_dark_theme(qapp, restore_system_theme):
    """The plot-option pop-ups are built hidden; their controls used to keep
    the light palette until first shown."""
    from opennmr.gui.plot_options_dialog import PlotOptionsDialog
    _save_ui_settings({"ui_theme": "dark"})
    window = MainWindow()
    dialogs = window.findChildren(PlotOptionsDialog)
    assert dialogs
    for dialog in dialogs:
        for combo in dialog.findChildren(QComboBox):
            assert _is_dark(combo)


def test_saved_light_theme_is_applied(qapp, restore_system_theme):
    _save_ui_settings({"ui_theme": "light"})
    window = MainWindow()
    assert window.ui_theme_combo.currentData() == "light"
    assert not _is_dark(window)


@pytest.mark.parametrize("size", [8, 14])
def test_saved_font_size_reaches_every_widget(qapp, size, restore_system_theme):
    """The real check, not the spin box: every label and button in every
    panel -- built long before the font is restored -- is drawn at the
    saved size, and section titles keep their scale."""
    _save_ui_settings({"ui_font_size": size, "ui_theme": "dark"})
    window = MainWindow()
    assert window.ui_font_size_spin.value() == size
    assert qapp.font().pointSizeF() == size
    # Not those with a font of their own: set in code, or a size in
    # their own sheet (the notes editor's 11 px hint, by design).
    plain = [w for w in window.findChildren(QLabel) + window.findChildren(QPushButton)
             if not w.testAttribute(Qt.WidgetAttribute.WA_SetFont)
             and "font-size" not in w.styleSheet()]
    assert len(plain) > 500
    wrong = [w for w in plain if w.font().pointSizeF() != size]
    assert not wrong, f"{len(wrong)} widgets not at {size} pt, e.g. {wrong[0]!r}"
    section = window.findChildren(CollapsibleSection)[0]
    assert section.headerFont().pointSizeF() == pytest.approx(size * HEADER_FONT_SCALE)


def test_no_saved_settings_follows_the_system(qapp):
    window = MainWindow()
    assert window.ui_theme_combo.currentData() == "system"
    resolved = style_module.resolve_mode("system", qapp)
    assert _is_dark(window) == (resolved == "dark")


def test_section_headers_get_the_sheets_padding(qapp):
    """Built before the sheet, measured after it: a collapsed section's
    header on a panel not on screen must still be its full styled height
    once shown (it came out 26 px instead of 32 when layouts kept the
    pre-sheet sizes)."""
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    heights = {}
    for index in range(window.side_tabs.count()):
        window.side_tabs.setCurrentIndex(index)
        qapp.processEvents()
        for section in window.findChildren(CollapsibleSection):
            if section.isVisible() and "\n" not in section._header.text():
                heights.setdefault(section._header.height(), set()).add(
                    (window.side_tabs.tabText(index), section.title()))
    # One height, whichever panel; on failure, which sections differ.
    assert len(heights) == 1, {h: sorted(s)[:6] for h, s in heights.items()
                               if len(s) < 10} or sorted(heights)


def test_wheel_still_scrolls_past_a_value_box(qapp):
    """The guard, installed at the end now, is still in force."""
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent

    window = MainWindow()
    combo = window.ui_theme_combo
    before = combo.currentIndex()
    event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(combo, event)
    assert combo.currentIndex() == before
