"""
The editable title/notes box under Open experiments.

Shows the selected experiment's title (for Bruker, TopSpin's own
``pdata/<procno>/title``) and saves edits back where they came from --
see :mod:`opennmr.notebook.notes`, which does all the reading and
writing, so a notebook user gets exactly the same behaviour.

Saving is automatic: shortly after typing stops, when another
experiment is selected, when the box loses focus, and when the window
closes. There is no Save button to forget.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QEvent, QTimer, Signal
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from ...notebook import notes as nb_notes


class ExperimentNotesEditor(QWidget):
    #: (spectrum, where it was written) after a successful save.
    saved = Signal(object, str)
    #: (spectrum, error message) when a save fails.
    failed = Signal(object, str)

    #: Idle time (ms) after the last keystroke before saving.
    SAVE_DELAY_MS = 800

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._spectrum = None
        self._location = ""
        self._dirty = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)
        header = QLabel("Title / notes")
        header.setStyleSheet("font-weight: 600;")
        layout.addWidget(header)

        self.edit = QPlainTextEdit()
        self.edit.setTabChangesFocus(True)
        self.edit.setPlaceholderText("Select an experiment to see or add its title and notes.")
        self.edit.setMinimumHeight(56)
        self.edit.textChanged.connect(self._on_text_changed)
        self.edit.installEventFilter(self)
        layout.addWidget(self.edit, 1)

        # One short line; the full path is its tooltip.
        self.location_label = QLabel("")
        self.location_label.setStyleSheet("color: palette(placeholder-text); font-size: 11px;")
        layout.addWidget(self.location_label)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.SAVE_DELAY_MS)
        self._timer.timeout.connect(self.flush)
        self.set_spectrum(None)

    # --- public -------------------------------------------------------
    @property
    def spectrum(self):
        return self._spectrum

    def text(self) -> str:
        return self.edit.toPlainText()

    def is_dirty(self) -> bool:
        return self._dirty

    def set_spectrum(self, spectrum) -> None:
        """Show ``spectrum``'s notes (``None`` empties and disables the
        box). Any unsaved edit to the previous experiment is saved first.
        Re-showing the same experiment re-reads it from disk (so an edit
        made in TopSpin appears) unless the box has focus -- replacing
        text under the cursor while someone types would lose keystrokes."""
        self.flush()
        location = nb_notes.notes_location(spectrum) if spectrum is not None else ""
        if spectrum is self._spectrum and location == self._location and self.edit.hasFocus():
            return
        self._spectrum = spectrum
        self._location = location
        text = ""
        if spectrum is not None:
            try:
                text = nb_notes.experiment_notes(spectrum)
            except OSError as exc:  # unreadable title file: show why, keep going
                self._show_location(f"Couldn't read the title: {exc}", error=True)
        if text != self.edit.toPlainText():
            self.edit.blockSignals(True)
            try:
                self.edit.setPlainText(text)
            finally:
                self.edit.blockSignals(False)
        self._dirty = False
        self.edit.setEnabled(spectrum is not None)
        self.edit.setPlaceholderText(
            "No title yet – type one, or any notes about this experiment."
            if spectrum is not None else
            "Select an experiment to see or add its title and notes.")
        if spectrum is not None:
            self._show_location(self._describe_location(spectrum, location))
        else:
            self._show_location("")

    def flush(self) -> bool:
        """Save a pending edit now. Returns False only if a save failed."""
        self._timer.stop()
        if not self._dirty or self._spectrum is None:
            return True
        self._dirty = False
        spectrum = self._spectrum
        try:
            where = nb_notes.set_experiment_notes(spectrum, self.edit.toPlainText())
        except OSError as exc:
            message = f"Couldn't save to {self._location}: {exc.strerror or exc}"
            self._show_location(f"Couldn't save: {exc.strerror or exc}", error=True)
            self.failed.emit(spectrum, message)
            return False
        self._show_location(self._describe_location(spectrum, where))
        self.saved.emit(spectrum, where)
        return True

    # --- internals ----------------------------------------------------
    def eventFilter(self, obj, event):  # noqa: N802 -- Qt override
        if obj is self.edit and event.type() == QEvent.Type.FocusOut:
            self.flush()
        return super().eventFilter(obj, event)

    def _on_text_changed(self) -> None:
        if self._spectrum is None:
            return
        self._dirty = True
        self._timer.start()

    @staticmethod
    def _describe_location(spectrum, location: str) -> str:
        meta = spectrum.metadata
        if not os.path.isabs(location):
            return "Session only (no file on disk)"
        if os.path.basename(location) == "title":
            rel = os.path.relpath(location, meta.source_path).replace(os.sep, "/")
            return f"Saves to the TopSpin title, {rel}"
        return f"Saves to {os.path.basename(location)}"

    def _show_location(self, text: str, error: bool = False) -> None:
        self.location_label.setText(text)
        tip = self._location
        if os.path.basename(tip) == "title":
            tip += ("\nShared with TopSpin. The first edit keeps the "
                    "title as acquired in title.orig beside it.")
        self.location_label.setToolTip(tip)
        self.location_label.setStyleSheet(
            "color: #c0392b; font-size: 11px;" if error
            else "color: palette(placeholder-text); font-size: 11px;")
