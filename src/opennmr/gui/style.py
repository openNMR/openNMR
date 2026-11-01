"""
Theme and widget styling: light/dark palettes plus softened edges.

Two lessons are baked into the shape of this module, both from things
that went visibly wrong rather than from taste.

**1. Style few widget classes, not many.** A Qt stylesheet rule on a
widget class drops that class out of the platform's native renderer --
for *every* property, not just the one named. For a widget with
sub-controls that is destructive: styling ``QSpinBox`` leaves its
up/down arrows unpainted unless every sub-control is respecified with
its own ``image:``, which is why the first version of this module made
"the small buttons like the up/down arrows really messed up". It also
detaches the widget from ``QApplication.setFont``, so the interface
font-size control stopped reaching them. Nothing here touches
``QSpinBox``, ``QDoubleSpinBox``, ``QComboBox``, ``QSlider`` or
``QScrollBar``: they stay native, and take their softening from the
palette instead. See :data:`_NEVER_STYLED`.

**2. Set the palette; do not read ambiguous roles from it.** On this
platform the stock ``Button`` role is byte-identical to ``Window``
(measured: #efefef both, contrast 1.00:1), so a button painted
``background: palette(button)`` is invisible against the panel behind
it -- reported as "many buttons have no distinction from the
background colour". Rather than guess at a role that happens to
contrast, :func:`apply_theme` installs a palette whose ``Button`` is
deliberately offset from ``Window`` in both themes, and the sheet
below can then rely on it.

The user-facing setting is :func:`apply_theme`'s ``mode``: ``"system"``
(the default), ``"light"`` or ``"dark"``. "System" is not "leave the
palette alone" -- it means *follow* the system's light/dark choice and
then apply the matching palette, so buttons have the same guaranteed
contrast either way.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

#: Corner radius in px for buttons and text fields. 0 restores square
#: corners. :data:`CONTAINER_RADIUS` is used for group boxes and panes,
#: which read as softer when rounded slightly more than their contents.
RADIUS = 6
CONTAINER_RADIUS = 8

#: Selectable values for the Settings combo, in display order.
THEME_MODES = ("system", "light", "dark", "solarized_light", "solarized_dark", "nord",
               "midnight", "sepia", "gruvbox_light", "gruvbox_dark", "dracula", "forest",
               "high_contrast")
THEME_LABELS = {"system": "Follow system", "light": "Light", "dark": "Dark",
                "solarized_light": "Solarized light", "solarized_dark": "Solarized dark",
                "nord": "Nord (cool grey-blue)", "midnight": "Midnight (deep blue)",
                "sepia": "Sepia (warm paper)", "gruvbox_light": "Gruvbox light (warm cream)",
                "gruvbox_dark": "Gruvbox dark (warm charcoal)", "dracula": "Dracula (purple-grey)",
                "forest": "Forest (deep green)",
                "high_contrast": "High contrast (black, yellow)"}

#: The extra themes (explicit user requests, 2026-10-24 and 2026-10-25): name -> whether it
#: is a dark one, and its colours. Light and dark are built in
#: :func:`_palette`. Every theme keeps Button offset from Window and a
#: mid-tone border, the property the stylesheet relies on.
_EXTRA_THEMES = {
    "solarized_light": dict(dark=False, window="#eee8d5", base="#fdf6e3", alt="#f5eedb",
                            button="#fdf6e3", hover="#e1d9bf", text="#073642", disabled="#93a1a1",
                            border="#93a1a1", shadow="#657b83", highlight="#268bd2",
                            highlight_text="#ffffff"),
    "solarized_dark": dict(dark=True, window="#002b36", base="#00212b", alt="#073642",
                           button="#0b3d4a", hover="#14505f", text="#a7b5b6", disabled="#586e75",
                           border="#4f6a73", shadow="#001a21", highlight="#268bd2",
                           highlight_text="#ffffff"),
    "nord": dict(dark=True, window="#2e3440", base="#272c36", alt="#3b4252", button="#434c5e",
                 hover="#4c566a", text="#eceff4", disabled="#7b88a1", border="#667189",
                 shadow="#1f232b", highlight="#5e81ac", highlight_text="#ffffff"),
    "midnight": dict(dark=True, window="#0f1724", base="#0a101a", alt="#17223a", button="#1f2e4d",
                     hover="#2a3d66", text="#dbe4f5", disabled="#6b7a99", border="#4a5d86",
                     shadow="#05080e", highlight="#3d7de0", highlight_text="#ffffff"),
    "sepia": dict(dark=False, window="#efe6d6", base="#fbf6ea", alt="#f4ecdc", button="#fbf6ea",
                  hover="#e2d6bd", text="#3b3226", disabled="#9a8f7c", border="#a89878",
                  shadow="#7a6e58", highlight="#a8602a", highlight_text="#ffffff"),
    "gruvbox_light": dict(dark=False, window="#ebdbb2", base="#fbf1c7", alt="#f2e5bc",
                          button="#fbf1c7", hover="#d5c4a1", text="#3c3836", disabled="#928374",
                          border="#a89984", shadow="#7c6f64", highlight="#076678",
                          highlight_text="#ffffff"),
    "gruvbox_dark": dict(dark=True, window="#282828", base="#1d2021", alt="#32302f",
                         button="#3c3836", hover="#504945", text="#ebdbb2", disabled="#928374",
                         border="#665c54", shadow="#141617", highlight="#d79921",
                         highlight_text="#1d2021"),
    "dracula": dict(dark=True, window="#282a36", base="#21222c", alt="#343746", button="#44475a",
                    hover="#565a73", text="#f8f8f2", disabled="#8a92b8", border="#6272a4",
                    shadow="#191a21", highlight="#bd93f9", highlight_text="#21222c"),
    "forest": dict(dark=True, window="#1b2a22", base="#14201a", alt="#243a2e", button="#2c4638",
                   hover="#3a5c48", text="#e3efe6", disabled="#8aa594", border="#5a7f68",
                   shadow="#0b120e", highlight="#5fae7f", highlight_text="#0b120e"),
    "high_contrast": dict(dark=True, window="#000000", base="#000000", alt="#161616",
                          button="#1c1c1c", hover="#3a3a3a", text="#ffffff", disabled="#9a9a9a",
                          border="#ffffff", shadow="#000000", highlight="#ffd400",
                          highlight_text="#000000"),
}

#: Classes deliberately left native -- see this module's docstring.
#: Listed rather than merely omitted so that a later edit adding one of
#: them has to delete it from here first.
_NEVER_STYLED = ("QSpinBox", "QDoubleSpinBox", "QComboBox", "QSlider",
                 "QScrollBar", "QCheckBox", "QRadioButton", "QHeaderView")

#: Lets :func:`apply_theme` recognise its own sheet on a later call.
_MARKER = "/* opennmr: theme */"

#: Whether the Fusion style has been installed yet (see apply_theme).
_STYLE_FORCED = False

#: The system's own light/dark choice, sampled once before this module
#: overrides the palette and colour scheme. Without this, "follow
#: system" would read back whatever openNMR applied last and latch onto
#: it. None until :func:`remember_system_theme` has been called.
_SYSTEM_IS_DARK: bool | None = None


def remember_system_theme(app: QApplication | None = None) -> bool:
    """Sample the platform's light/dark choice and cache it.

    Call once, **before** the first :func:`apply_theme`. Sampling later
    reads openNMR's own palette back and always answers "light".
    """
    global _SYSTEM_IS_DARK
    app = app or QApplication.instance()
    if app is None:
        return False
    scheme = getattr(app.styleHints(), "colorScheme", None)
    if callable(scheme):
        try:
            value = scheme()
            if value != Qt.ColorScheme.Unknown:
                _SYSTEM_IS_DARK = value == Qt.ColorScheme.Dark
                return _SYSTEM_IS_DARK
        except (AttributeError, TypeError):
            pass
    colour = app.palette().color(QPalette.ColorRole.Window)
    luminance = (0.2126 * colour.redF() + 0.7152 * colour.greenF()
                 + 0.0722 * colour.blueF())
    _SYSTEM_IS_DARK = luminance < 0.5
    return _SYSTEM_IS_DARK


def _palette(dark) -> QPalette:
    """A palette with **Button deliberately offset from Window**, which
    is the whole reason this is built by hand rather than taken from
    the platform. Every other role is set too, so the two themes differ
    only in this table and a widget cannot pick up a stale colour from
    whichever theme was active first.
    """
    # "border" is the role the stylesheet draws every outline with, and
    # it is what actually makes a button read as a button: two adjacent
    # surfaces a few percent apart in luminance do not separate on their
    # own, however the numbers are arranged. It is deliberately a
    # mid-grey in both themes -- distinct from "shadow", which stays
    # near-black for the sunken/frame roles Qt uses it for.
    extra = _EXTRA_THEMES.get(dark) if isinstance(dark, str) else None
    if extra is not None:
        window, base, alt = extra["window"], extra["base"], extra["alt"]
        button, button_hover = extra["button"], extra["hover"]
        text, disabled_text = extra["text"], extra["disabled"]
        border, shadow = extra["border"], extra["shadow"]
        highlight, highlight_text = extra["highlight"], extra["highlight_text"]
    elif dark:
        window, base, alt = "#26262b", "#1b1b1f", "#33333a"
        button, button_hover = "#3f3f48", "#4c4c57"
        text, disabled_text = "#e6e6ea", "#787882"
        border, shadow = "#63636f", "#101013"
        highlight, highlight_text = "#4a8fd4", "#ffffff"
    else:
        window, base, alt = "#e9e9ed", "#ffffff", "#f5f5f8"
        button, button_hover = "#ffffff", "#dcdce4"
        text, disabled_text = "#1b1b1f", "#94949e"
        border, shadow = "#9a9aa6", "#6e6e78"
        highlight, highlight_text = "#3574c8", "#ffffff"

    palette = QPalette()
    role = QPalette.ColorRole
    group = QPalette.ColorGroup
    for name, colour in (
        ("Window", window), ("Base", base), ("AlternateBase", alt),
        ("Button", button), ("ToolTipBase", base), ("ToolTipText", text),
        ("WindowText", text), ("Text", text), ("ButtonText", text),
        ("BrightText", highlight_text), ("Light", button_hover),
        ("Midlight", border), ("Mid", border), ("Dark", shadow), ("Shadow", shadow),
        ("Highlight", highlight), ("HighlightedText", highlight_text),
        ("PlaceholderText", disabled_text),
    ):
        palette.setColor(getattr(role, name), QColor(colour))
    for name in ("WindowText", "Text", "ButtonText"):
        palette.setColor(group.Disabled, getattr(role, name), QColor(disabled_text))
    return palette


def system_prefers_dark(app: QApplication | None = None) -> bool:
    """Whether the platform is currently in dark mode.

    Qt 6.5+ answers this directly; older builds do not, so the fallback
    reads the luminance of the stock Window colour -- a dark theme is a
    dark window, whatever the API for asking happens to be.
    """
    app = app or QApplication.instance()
    if app is None:
        return False
    # Deliberately *not* styleHints().colorScheme(): apply_theme sets
    # that itself, so asking it back would return openNMR's own last
    # choice rather than the system's, and "follow system" would latch
    # onto whatever was applied first. The platform palette is the
    # honest source -- a dark theme is a dark window.
    if _SYSTEM_IS_DARK is not None:
        return _SYSTEM_IS_DARK
    colour = app.palette().color(QPalette.ColorRole.Window)
    luminance = (0.2126 * colour.redF() + 0.7152 * colour.greenF()
                 + 0.0722 * colour.blueF())
    return luminance < 0.5


def resolve_mode(mode: str, app: QApplication | None = None) -> str:
    """``"system"`` -> the concrete ``"light"``/``"dark"`` it means now."""
    if mode == "system":
        return "dark" if system_prefers_dark(app) else "light"
    if mode in _EXTRA_THEMES:
        return "dark" if _EXTRA_THEMES[mode]["dark"] else "light"
    return mode if mode in ("light", "dark") else "light"


def theme_qss(radius: int = RADIUS, container_radius: int = CONTAINER_RADIUS) -> str:
    """The stylesheet, as a string, so it can be asserted on without an
    application instance.

    Colours come from ``palette()`` roles only -- never a literal -- so
    one sheet serves both themes and the palette above is the single
    place a colour is decided.
    """
    return f"""
QPushButton, QToolButton {{
    border: 1px solid palette(mid);
    border-radius: {radius}px;
    padding: 4px 10px;
    background: palette(button);
    color: palette(button-text);
}}
QPushButton:hover, QToolButton:hover {{ background: palette(light); }}
QPushButton:pressed, QToolButton:pressed,
QPushButton:checked, QToolButton:checked {{
    background: palette(highlight);
    color: palette(highlighted-text);
    border-color: palette(highlight);
}}
QPushButton:disabled, QToolButton:disabled {{
    color: palette(placeholder-text);
    border-color: palette(midlight);
}}
/* A button that has asked for no border keeps none – the collapsible
   section headers and the ModeButton arrow both do, and boxing them in
   is not what "softer edges" meant. */
QToolButton[flat="true"] {{ border: none; background: transparent; }}
QToolButton::menu-indicator {{ border: none; }}

QLineEdit, QPlainTextEdit, QTextEdit {{
    border: 1px solid palette(mid);
    border-radius: {radius}px;
    padding: 2px 6px;
    background: palette(base);
    color: palette(text);
    selection-background-color: palette(highlight);
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: palette(highlight);
}}

QGroupBox {{
    border: 1px solid palette(midlight);
    border-radius: {container_radius}px;
    margin-top: 8px;
    padding-top: 6px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QTabWidget::pane {{
    border: 1px solid palette(midlight);
    border-radius: {container_radius}px;
}}
QTabBar::tab {{
    border-top-left-radius: {radius}px;
    border-top-right-radius: {radius}px;
    padding: 4px 10px;
}}
QMenu, QToolTip {{
    border: 1px solid palette(mid);
    border-radius: {radius}px;
}}
QProgressBar {{
    border: 1px solid palette(mid);
    border-radius: {radius}px;
    text-align: center;
}}
QProgressBar::chunk {{ border-radius: {max(radius - 2, 0)}px; }}
"""


def apply_theme(mode: str = "system", app: QApplication | None = None,
                radius: int = RADIUS, stylesheet: bool = True, fast: bool = False) -> str:
    """Install the palette and stylesheet for ``mode``; return the
    concrete theme (``"light"``/``"dark"``) actually applied.

    Safe to call repeatedly -- switching themes at runtime is the point
    -- and it replaces only its own sheet, leaving any an embedding
    host had set.

    ``stylesheet=False`` installs the style and palette only. MainWindow
    does that before building its widgets -- so every widget measures
    itself in the style it will be shown in -- and installs the sheet
    once at the end, because a sheet present while thousands of widgets
    are reparented into place re-polishes them over and over (startup
    time; see MainWindow._apply_startup_ui).

    ``fast`` (a change made in Settings while the window is on screen): when
    the sheet is already installed -- it is the same for every theme, the
    colours coming from palette() roles -- only re-polish what is on screen,
    and the rest as it is shown (:func:`restyle`), instead of re-setting the
    sheet over every widget.
    """
    app = app or QApplication.instance()
    if app is None:
        return resolve_mode(mode, None)
    resolved = resolve_mode(mode, app)

    # --- make every widget palette-driven, native ones included -----
    #
    # This is the step without which the theme is only half applied,
    # and it is worth being explicit about why. The widgets left out of
    # the stylesheet (see _NEVER_STYLED) are drawn by the platform
    # style, which on macOS follows the *operating system's* light/dark
    # appearance and ignores QPalette for its sub-controls. With the OS
    # in dark mode and openNMR set to light, that produced exactly the
    # reported symptoms: white spin-box and combo arrows, a dark
    # background behind the matplotlib toolbar and the phase up/down
    # buttons, a dark phase-pivot QSlider, and light-grey text in
    # tables and on buttons -- every one of them a natively-drawn
    # element still rendering in the system appearance.
    #
    # Fusion is the style that resolves everything, including those
    # sub-controls, from the palette below, so one palette now decides
    # the whole window. The cost is that the app no longer picks up
    # macOS's native widget look; that is the price of a theme toggle
    # that actually works, and it is what makes light-on-a-dark-system
    # (and dark-on-a-light-system) render correctly at all.
    # Tracked with a flag rather than by asking app.style(): once a
    # stylesheet is installed that returns the QStyleSheetStyle wrapper,
    # whose objectName is "", so a name check would re-run setStyle on
    # every theme change -- and setStyle resets the palette, so the
    # extra churn is not free.
    global _STYLE_FORCED
    if not _STYLE_FORCED:
        app.setStyle("Fusion")
        _STYLE_FORCED = True

    # Qt 6.8+ can also tell the platform theme which scheme to use,
    # which covers any chrome Fusion does not draw itself (native menu
    # bar, window frame). Older builds simply do not have it.
    hints = app.styleHints()
    setter = getattr(hints, "setColorScheme", None)
    if callable(setter):
        try:
            setter(Qt.ColorScheme.Dark if resolved == "dark" else Qt.ColorScheme.Light)
        except (AttributeError, TypeError):
            pass

    # After setStyle, which resets the palette to the style's own.
    app.setPalette(_palette(mode if mode in _EXTRA_THEMES else resolved == "dark"))
    if not stylesheet:
        return resolved

    existing = app.styleSheet() or ""
    wanted_tail = f"{_MARKER}\n{theme_qss(radius)}".strip()
    if _MARKER in existing:
        if fast and existing.rstrip().endswith(wanted_tail):
            # The sheet itself is the same for every theme (colours come from
            # palette() roles), so a theme change only needs the widgets'
            # cached colours refreshed -- on screen now, the rest when shown.
            restyle(app)
            return resolved
        existing = existing.split(_MARKER)[0]
    app.setStyleSheet(f"{existing}\n{wanted_tail}".strip())
    _reset_generation()
    return resolved


# ---- fast restyling: what is on screen now, the rest when it is shown ----------
#
# Re-setting the application stylesheet re-polishes *every* widget: Qt sends
# each a StyleChange and re-matches the sheet's rules against it. The main
# window holds ~7,900 widgets, of which ~75 are on screen, and that took
# 5-6 s per font-size step or theme change (measured 2026-11-01; most of the
# widgets are on other tiles, in collapsed sections or in hidden dialogs).
#
# A widget styled by the sheet caches its colours and font when polished, so
# the palette and font change themselves are instant (QApplication.setPalette
# / setFont, ~10 ms) but are not *seen* until the widget is polished again.
# restyle() re-polishes only the visible widgets, parents before children
# (a child's font is resolved from its parent's), and stamps each with the
# current generation; an application event filter re-polishes any other
# widget, and its stale ancestors, the moment it is shown. A widget built
# after the change is polished normally, with the new palette and font.
#
# tests/test_fast_interface_restyle.py checks the result against the slow,
# whole-sheet path: the same font and the same rendered colours for every
# widget once each page has been shown.

#: Dynamic property holding the restyle generation a widget was last polished in.
_GENERATION_PROP = "_opennmr_restyle_generation"
_restyle = {"generation": 0, "filter": None}


def _reset_generation() -> None:
    """After a whole-sheet re-set every widget is current again."""
    _restyle["generation"] = 0


def restyle_generation() -> int:
    """How many fast restyles have happened since the sheet was last set."""
    return _restyle["generation"]


def _stale(widget) -> bool:
    return (widget.property(_GENERATION_PROP) or 0) != _restyle["generation"]


def _repolish(widget, style) -> None:
    style.unpolish(widget)
    style.polish(widget)
    widget.setProperty(_GENERATION_PROP, _restyle["generation"])
    QApplication.sendEvent(widget, QEvent(QEvent.Type.StyleChange))
    widget.update()


def _repolish_with_ancestors(widget, style) -> None:
    chain = []
    w = widget
    while w is not None and _stale(w):
        chain.append(w)
        w = w.parentWidget()
    for w in reversed(chain):                   # parents first
        _repolish(w, style)


class _RepolishOnShow(QObject):
    """Application event filter: a widget polished before the last
    :func:`restyle` is re-polished when it is shown."""

    def eventFilter(self, obj, event):  # noqa: N802 -- Qt override
        if (event.type() == QEvent.Type.Show and _restyle["generation"]
                and isinstance(obj, QWidget)):
            try:
                if _stale(obj):
                    app = QApplication.instance()
                    if app is not None:
                        _repolish_with_ancestors(obj, app.style())
            except RuntimeError:                # deleted while being shown
                pass
        return False


def _depth(widget) -> int:
    d = 0
    while widget is not None:
        widget = widget.parentWidget()
        d += 1
    return d


def restyle(app: QApplication | None = None) -> int:
    """Make a palette or application-font change visible fast: re-polish the
    widgets on screen now (parents first) and the others when they are shown.
    Returns how many widgets were re-polished now."""
    app = app or QApplication.instance()
    if app is None:
        return 0
    _restyle["generation"] += 1
    if _restyle["filter"] is None:
        _restyle["filter"] = _RepolishOnShow(app)
        app.installEventFilter(_restyle["filter"])
    style = app.style()
    shown = [w for w in app.allWidgets() if w.isVisible()]
    shown.sort(key=_depth)
    for w in shown:
        _repolish(w, style)
    return len(shown)


def refresh_fonts(app: QApplication | None = None) -> None:
    """Re-apply the current stylesheet so widget fonts re-resolve.

    **Call this after every ``QApplication.setFont``.** Installing an
    application stylesheet -- *any* stylesheet, even one naming only
    ``QGroupBox`` -- marks the fonts of every widget that already
    exists as explicitly set, after which ``setFont`` no longer reaches
    them. Measured: with a sheet installed, a combo built beforehand
    stayed at 9pt while the app font went to 15pt, which is what broke
    the interface font-size control.

    Re-setting the same sheet makes Qt resolve every widget's font
    against the new application font again. Clearing the sheet first
    does **not** work (it leaves them at the stale size) and neither
    does a plain repolish, so this is the re-set and not something
    tidier.
    """
    app = app or QApplication.instance()
    if app is None:
        return
    sheet = app.styleSheet()
    if sheet:
        app.setStyleSheet(sheet)


#: Kept so the previous call site keeps working; "soft edges" is now
#: one part of the theme rather than the whole of it.
def apply_soft_edges(app: QApplication | None = None, radius: int = RADIUS) -> None:
    apply_theme("system", app, radius)
