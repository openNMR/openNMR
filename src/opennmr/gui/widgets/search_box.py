"""The search box at the right-hand end of the tool row.

Explicit user request, 2026-11-12: the "Where is…?" search (Ctrl+H,
Help ▸ Search Commands…) should be *visible* -- a box beside the tool
buttons that can be shown or hidden -- since a search only reachable by
a shortcut is one people do not know is there.

It is a front end to the same search, not a second one: what you type
goes to a :class:`~opennmr.gui.command_palette.CommandPalette` opened in
its *anchored* mode, a results list just below the box that never takes
the keyboard focus, so typing stays here. ↑/↓ move through the results,
Enter goes to the highlighted one, Esc clears and closes, and clicking a
result goes to it. The index is rebuilt each time the box starts a new
search, so it describes the window as it is then.

**Narrow windows.** The tool row is full on a laptop screen. Below
:data:`COMPACT_BELOW` pixels of window width the box folds to a
magnifier button, which opens the ordinary pop-up palette, rather than
squeezing the tool buttons.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QSizePolicy, QToolButton, QWidget

#: Window width (px) below which the box folds to a button.
COMPACT_BELOW = 1240


class SearchBox(QWidget):
    """``SearchBox(window)``; ``window`` is the MainWindow (it provides
    ``show_command_palette`` and the palette's index)."""

    def __init__(self, window, parent=None) -> None:
        super().__init__(parent)
        self._window = window
        self._palette = None
        self._compact = False
        self.setProperty("palette_exclude", True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Search…  (Ctrl+H)")
        self.edit.setClearButtonEnabled(True)
        self.edit.setToolTip(
            "Where is…? Type part of what you want – baseline, export, T1, line "
            "broadening – and choose a result to go straight to it. ↑/↓ to choose, "
            "Enter to go, Esc to close. Show or hide this box with View ▸ Show Search Box.")
        # Marginally wider than first made (explicit user request, 2026-11-17).
        self.edit.setMinimumWidth(230)
        # Scales with the window (user report, 2026-11-19): expanding, with a generous cap.
        self.edit.setMaximumWidth(720)
        self.edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.edit.setProperty("palette_exclude", True)
        self.edit.textEdited.connect(self._on_text)
        self.edit.installEventFilter(self)
        layout.addWidget(self.edit, 1)

        self.button = QToolButton()
        self.button.setText("🔍")
        # Slightly larger magnifier (explicit user request, 2026-11-17).
        self.button.setMinimumSize(40, 34)
        self.button.setStyleSheet("QToolButton { font-size: 18px; }")
        self.button.setToolTip("Search (Ctrl+H): find any action, panel or control and go to it.")
        self.button.setProperty("palette_exclude", True)
        self.button.clicked.connect(lambda _c=False: self._window.show_command_palette())
        self.button.setVisible(False)
        layout.addWidget(self.button)

        # Closing on focus loss is deferred a moment, so a click on a
        # result (which takes the focus off the box) still lands.
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.setInterval(200)
        self._close_timer.timeout.connect(self._close_if_unfocused)
        window.installEventFilter(self)

    # -- state -------------------------------------------------------------

    def is_compact(self) -> bool:
        return self._compact

    def set_compact(self, compact: bool) -> None:
        """Fold to the magnifier button (``True``) or show the box."""
        compact = bool(compact)
        if compact == self._compact:
            return
        self._compact = compact
        if compact:
            self.close_results()
        self.edit.setVisible(not compact)
        self.button.setVisible(compact)

    def palette(self):
        """The results list on show (None when closed)."""
        return self._palette if self._palette is not None and self._palette.isVisible() else None

    # -- searching ---------------------------------------------------------

    def start(self) -> None:
        """Ctrl+H with the box shown: put the cursor in it, text selected,
        and show the recent choices."""
        if self._compact or not self.isVisible():
            self._window.show_command_palette()
            return
        self.edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.edit.selectAll()
        self._results().refresh(self.edit.text())

    def _results(self):
        """The results list, opened under the box. A list closed in the
        meantime (a result was clicked) is replaced, with a fresh index."""
        if self._palette is not None and not self._palette.isVisible():
            self._palette.deleteLater()
            self._palette = None
        if self._palette is None:
            from ..command_palette import CommandPalette
            self._palette = CommandPalette(self._window, anchored=True)
            self._palette.open_under(self)
        return self._palette

    def _on_text(self, text: str) -> None:
        self._results().refresh(text)

    def close_results(self) -> None:
        if self._palette is not None:
            self._palette.close()
            self._palette.deleteLater()
            self._palette = None

    def _go(self) -> None:
        palette = self.palette()
        if palette is None:
            return
        self._palette = None             # activate() closes it
        entry = palette.activate()
        palette.deleteLater()
        if entry is not None:
            self.edit.clear()

    def _close_if_unfocused(self) -> None:
        if not self.edit.hasFocus():
            self.close_results()

    def eventFilter(self, obj: QObject, event) -> bool:  # noqa: N802 -- Qt override
        kind = event.type()
        if obj is self._window:
            if kind == QEvent.Type.Resize:
                self.set_compact(self._window.width() < COMPACT_BELOW)
            elif kind in (QEvent.Type.Move, QEvent.Type.WindowDeactivate) and self.palette():
                self.close_results()
            return False
        if obj is self.edit:
            if kind == QEvent.Type.KeyPress:
                key = event.key()
                if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                    self._results().step(1 if key == Qt.Key.Key_Down else -1)
                    return True
                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    if self.palette() is None:
                        self._results().refresh(self.edit.text())
                    self._go()
                    return True
                if key == Qt.Key.Key_Escape:
                    self.edit.clear()
                    self.close_results()
                    self.edit.clearFocus()
                    return True
            elif kind == QEvent.Type.FocusIn:
                if self.edit.text():
                    self._results().refresh(self.edit.text())
            elif kind == QEvent.Type.FocusOut:
                self._close_timer.start()
        return super().eventFilter(obj, event)
