"""The extra interface colour themes added for 0.8.3 stay legible.

Every theme keeps the property the stylesheet relies on (Button offset
from Window) and, new here, readable text: WCAG contrast of the text
colour on the panel and on text fields, of the highlighted-selection
text on the highlight, and a visible (if quieter) disabled text.
"""

from __future__ import annotations

import pytest

from opennmr.gui import style

try:                                                    # the shared Qt fixture
    from tests.test_welcome_wizard import qapp  # noqa: F401
except ImportError:                                     # pragma: no cover
    from test_welcome_wizard import qapp  # noqa: F401

NEW = ("gruvbox_light", "gruvbox_dark", "dracula", "forest")


def _luminance(colour: str) -> float:
    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
    r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_the_new_themes_are_listed_labelled_and_in_the_right_order():
    for name in NEW:
        assert name in style.THEME_MODES and name in style.THEME_LABELS and name in style._EXTRA_THEMES
    assert style.THEME_MODES[-1] == "high_contrast"            # the accessibility one stays last
    assert len(set(style.THEME_MODES)) == len(style.THEME_MODES)
    assert len(set(style.THEME_LABELS.values())) == len(style.THEME_LABELS)


@pytest.mark.parametrize("name", sorted(style._EXTRA_THEMES))
def test_every_extra_theme_is_legible(name):
    t = style._EXTRA_THEMES[name]
    assert contrast(t["text"], t["window"]) >= 7.0
    assert contrast(t["text"], t["base"]) >= 7.0
    assert contrast(t["text"], t["button"]) >= 4.5
    # Selected text: the older themes sit at 3.7-4.0 (white on a mid blue), which passes
    # WCAG for large text only; the ones added in 0.8.3 meet the 4.5 for ordinary text.
    assert contrast(t["highlight_text"], t["highlight"]) >= (4.5 if name in NEW else 3.0)
    assert contrast(t["disabled"], t["base"]) >= 2.4          # quieter on purpose (WCAG exempts disabled text)
    assert contrast(t["border"], t["window"]) >= 1.5          # the outline shows


def test_new_themes_resolve_to_the_right_light_or_dark(qapp):  # noqa: F811
    assert style.resolve_mode("gruvbox_light", qapp) == "light"
    for name in ("gruvbox_dark", "dracula", "forest"):
        assert style.resolve_mode(name, qapp) == "dark"


def test_new_themes_apply_with_buttons_offset_and_are_distinct(qapp):  # noqa: F811
    from PySide6.QtGui import QPalette
    seen = set()
    for name in NEW:
        style.apply_theme(name, qapp)
        pal = qapp.palette()
        window = pal.color(QPalette.ColorRole.Window).name()
        assert window != pal.color(QPalette.ColorRole.Button).name()
        seen.add(window)
    assert len(seen) == len(NEW)
    style.apply_theme("light", qapp)


def test_settings_and_wizard_offer_the_new_themes(qapp):  # noqa: F811
    from opennmr.gui.main_window import MainWindow
    from opennmr.gui.welcome_wizard import WelcomeWizard
    window = MainWindow()
    wizard = WelcomeWizard(window)
    for name in NEW:
        assert window.ui_theme_combo.findData(name) >= 0
        assert wizard.theme_combo.findData(name) >= 0
