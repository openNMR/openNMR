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

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QSizePolicy, QToolButton, QVBoxLayout, QWidget

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

        self._header = QToolButton(self)
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

    def _sync_header_text(self) -> None:
        glyph = _EXPANDED_GLYPH if self._header.isChecked() else _COLLAPSED_GLYPH
        self._header.setText(f"{glyph} {self._title}")
