"""Notes 8.6: the Welcome wizard must fit a small screen (pages scroll)."""
from PySide6.QtWidgets import QScrollArea

from opennmr.gui.main_window import MainWindow
from opennmr.gui.welcome_wizard import WelcomeWizard

from test_gui import qapp  # noqa: F401


def test_every_page_scrolls_so_no_page_sets_the_window_height(qapp):  # noqa: F811
    wizard = WelcomeWizard(MainWindow())
    assert wizard.page_count == 7
    assert all(isinstance(wizard.stack.widget(i), QScrollArea) for i in range(wizard.page_count))
    # the buttons stay reachable: the window can be shrunk well below its tallest page
    assert wizard.minimumSizeHint().height() < 520


def test_the_size_is_capped_to_the_screen(qapp):  # noqa: F811
    wizard = WelcomeWizard(MainWindow())
    from PySide6.QtGui import QGuiApplication
    area = QGuiApplication.primaryScreen().availableGeometry()
    width, height = wizard._fitted_size(10_000, 10_000)
    assert width <= area.width() * 0.9 + 1 and height <= area.height() * 0.9 + 1
    assert wizard._fitted_size(100, 100) == (100, 100)
