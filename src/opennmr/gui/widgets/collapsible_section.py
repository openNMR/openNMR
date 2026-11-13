"""
A titled section whose body can be collapsed by clicking its own
header -- the "click the name, the box opens; click again, it closes"
building block behind the new side-panel launcher (Load/Save/Compare,
Processing, Advanced Processing, Fitting, Relaxation, Applications,
Simulations). Explicit user request, replacing what used to be one
long scrolling list of controls per tab: "Inside each should be a
series of dropdown menus ... so there aren't big lists and lots of
scrolling".

Each instance is independent -- opening one does **not** close any
other (an accordion, where only one section can be open at a time,
was considered and explicitly turned down: several of these panels
group controls a user may well want visible side by side, e.g.
Phase and Reference while working through Processing).

**Every section starts collapsed** (explicit user request, 2026-09-27:
"all collapsible boxes should, on opening a menu, be collapsed by
default -- it will look cleaner"). A panel then opens as a short list
of headings to choose from. It originally started expanded, to match
how the controls were all visible before this widget existed; that
made every panel a long scroll on first sight. Code that sends the
user to a particular section (e.g. "Open in Simulate") opens that
section itself with :meth:`setExpanded`. A section the user opened
stays open when they switch panels and come back -- collapsing their
working state on every switch would undo what they had just set up.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QSizePolicy, QStyle, QStyleOptionToolButton, QToolButton, QVBoxLayout, QWidget,
)

#: Prepended to the header text to show expanded/collapsed state.
#: Plain-text glyphs rather than QToolButton's own arrow-icon (via
#: setArrowType) -- that icon is drawn from the *style*, which varies
#: across platforms/themes and (as seen with WrappingTabWidget's own
#: buttons elsewhere in this project) is easy to end up with nothing
#: visibly different at all in a plain QToolButton. A text glyph is
#: unambiguous everywhere, headless test runs included.
_EXPANDED_GLYPH = "▾"    # ▾
_COLLAPSED_GLYPH = "▸"   # ▸

#: Section titles are drawn this much larger than the surrounding text,
#: and bold, so they read as headings rather than as one more control.
#: Explicit user request: section names "slightly bigger, to make them
#: more obvious section headers". A *ratio* rather than a point size,
#: so the titles keep their lead when the interface font is scaled up
#: or down (Settings ▸ Interface), instead of being overtaken by it.
HEADER_FONT_SCALE = 1.25


class _DoubleClickTogglesOnce(QObject):
    """Event filter: a double-click on a checkable toggle button flips it
    exactly once.

    Qt turns a double-click into press, release, *double-click*, release,
    and a ``QAbstractButton`` treats the double-click event as a second
    press -- so the second release clicks it again and a section opened
    by a double-click closed at once (net: nothing happened). Explicit
    user request, 2026-11-13: "enable any collapsible section to be
    opened or closed by double-clicking on it". The first click of the
    pair already toggled; swallowing the double-click event leaves the
    second release with no press to complete, so one double-click =
    one toggle, and a single click still toggles as before."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 -- Qt override
        if event.type() == QEvent.Type.MouseButtonDblClick:
            event.accept()
            return True
        return False


#: One filter serves every button (it holds no state).
_DOUBLE_CLICK_FILTER: _DoubleClickTogglesOnce | None = None


def toggle_once_on_double_click(button: QToolButton) -> QToolButton:
    """Make a double-click on ``button`` (a checkable disclosure toggle)
    open or close its section once, instead of twice. Returns ``button``."""
    global _DOUBLE_CLICK_FILTER
    if _DOUBLE_CLICK_FILTER is None:
        _DOUBLE_CLICK_FILTER = _DoubleClickTogglesOnce()
    button.installEventFilter(_DOUBLE_CLICK_FILTER)
    return button


class _HeaderButton(QToolButton):
    """The section's title button, whose height depends only on its font
    and number of lines -- never on which characters the title holds.

    A title with a character the interface font lacks (T₁, ₂, ...) is
    drawn with a fallback font whose line is taller, so that one header
    came out 32 px against everyone else's 31 on Windows (the
    ``test_section_headers_get_the_sheets_padding`` failure, 2026-10-17).
    The height is measured on a reference title instead."""

    _REFERENCE = f"{_COLLAPSED_GLYPH} Ag"

    def _reference_height(self) -> int:
        lines = self.text().count("\n") + 1
        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        text = "\n".join([self._REFERENCE] * lines)
        option.text = text
        contents = self.fontMetrics().size(Qt.TextFlag.TextShowMnemonic, text)
        return self.style().sizeFromContents(
            QStyle.ContentsType.CT_ToolButton, option, contents, self).height()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 -- Qt override
        # See _DoubleClickTogglesOnce: the pair's first click has already
        # opened/closed the section; do not let the second undo it.
        event.accept()

    def sizeHint(self) -> QSize:  # noqa: N802 -- Qt override
        hint = super().sizeHint()
        return QSize(hint.width(), self._reference_height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 -- Qt override
        hint = super().minimumSizeHint()
        return QSize(hint.width(), self._reference_height())


class CollapsibleSection(QWidget):
    """A header button (the section's own title) plus a body widget
    that shows/hides when the header is clicked.

    ``content`` -- if given, becomes the section's body immediately
    (the common case: a ``_build_*_subtab``-style method already
    returns one finished ``QWidget``, so wrapping it here needs no
    layout surgery). Pass nothing and call :meth:`setContentWidget`
    later if the content isn't built yet.

    ``expanded`` -- initial state, default ``False`` (see the module
    docstring).
    """

    #: Emitted with the new expanded state whenever it changes, by a
    #: click or by :meth:`setExpanded`.
    toggled = Signal(bool)

    def __init__(self, title: str, content: QWidget | None = None,
                 parent: QWidget | None = None, expanded: bool = False) -> None:
        super().__init__(parent)
        self._title = title
        self._content: QWidget | None = None

        self._header = _HeaderButton(self)
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._header.setAutoRaise(True)
        # Left-aligned, full width -- a raised/centered button here
        # would read as a push-button *action*, not a section title.
        self._header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Bold and size come from _apply_header_font, not the style
        # sheet: a style-sheet font size is fixed in points and would
        # stop following the interface font-size setting.
        self._header.setStyleSheet("QToolButton { text-align: left; }")
        # A long title must not set the side panel's minimum width (the
        # panel then cannot reflow when the column is narrowed -- explicit
        # user request, 2026-09-30): it wraps onto a second line instead
        # (_sync_header_text), so the header can be narrower than it.
        self._header.setMinimumWidth(60)
        self._apply_header_font()
        self._header.clicked.connect(self._on_header_clicked)
        self._sync_header_text()

        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(14, 2, 0, 4)
        self._body.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self._body)

        if content is not None:
            self.setContentWidget(content)

    # -- content ---------------------------------------------------

    def setContentWidget(self, widget: QWidget) -> None:  # noqa: N802 -- matches Qt's own naming
        """Replace the section's body with ``widget`` wholesale."""
        if self._content is not None:
            self._body_layout.removeWidget(self._content)
            self._content.setParent(None)
        self._content = widget
        self._body_layout.addWidget(widget)

    def contentWidget(self) -> QWidget | None:  # noqa: N802
        return self._content

    # -- expand/collapse ---------------------------------------------

    def isExpanded(self) -> bool:  # noqa: N802
        return self._header.isChecked()

    def setExpanded(self, expanded: bool) -> None:  # noqa: N802
        if self._header.isChecked() == expanded:
            return
        self._header.setChecked(expanded)
        self._apply_expanded(expanded)

    def title(self) -> str:
        return self._title

    def setTitle(self, title: str) -> None:  # noqa: N802
        self._title = title
        self._sync_header_text()

    def headerFont(self) -> QFont:  # noqa: N802
        """The font the title is drawn in (for tests and callers
        matching it)."""
        return self._header.font()

    # -- internals -----------------------------------------------------

    def _apply_header_font(self) -> None:
        """Title font: this section's own (inherited) font, bold and
        :data:`HEADER_FONT_SCALE` times the size."""
        font = QFont(self.font())
        font.setBold(True)
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * HEADER_FONT_SCALE)
        elif font.pixelSize() > 0:
            font.setPixelSize(round(font.pixelSize() * HEADER_FONT_SCALE))
        self._header.setFont(font)

    def changeEvent(self, event) -> None:  # noqa: N802 -- Qt override
        # The section's own font changes when the application font does
        # (the interface font-size setting) or a parent's does; the
        # header has an explicit font, so it must be recomputed rather
        # than inherited.
        if event.type() == QEvent.Type.FontChange:
            self._apply_header_font()
        super().changeEvent(event)

    def _on_header_clicked(self, checked: bool) -> None:
        self._apply_expanded(checked)

    def _apply_expanded(self, expanded: bool) -> None:
        self._body.setVisible(expanded)
        self._sync_header_text()
        self.toggled.emit(expanded)

    def resizeEvent(self, event) -> None:  # noqa: N802 -- Qt override
        super().resizeEvent(event)
        self._sync_header_text()

    def showEvent(self, event) -> None:  # noqa: N802 -- Qt override
        super().showEvent(event)
        self._sync_header_text()

    def _sync_header_text(self) -> None:
        glyph = _EXPANDED_GLYPH if self._header.isChecked() else _COLLAPSED_GLYPH
        # "&" would be a mnemonic in a QToolButton (and vanish): doubled to show.
        text = f"{glyph} {self._title.replace('&', '&&')}"
        width = self.width() if self.isVisible() else 0
        metrics = self._header.fontMetrics()
        room = width - 16
        if width > 0 and metrics.horizontalAdvance(text) > room:
            lines, line = [], ""
            for word in text.split(" "):
                trial = f"{line} {word}" if line else word
                if line and metrics.horizontalAdvance(trial) > room:
                    lines.append(line)
                    line = "    " + word           # under the title, past the glyph
                else:
                    line = trial
            lines.append(line)
            text = "\n".join(lines)
        if self._header.text() != text:
            self._header.setText(text)
