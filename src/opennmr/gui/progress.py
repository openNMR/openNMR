"""
Dialogs for File ▸ Save Progress…, Load Progress… and Processing Log….

The logic -- what a save point holds, how a restore is applied -- lives
in :mod:`opennmr.io.history` (Qt-free) and in ``MainWindow``'s
``_save_progress``/``_restore_progress``; these classes only ask the
user and show text, so tests can drive the logic without them.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout,
)

#: The "save point" id meaning "the spectrum as first loaded".
ORIGINAL = "original"


def _mono(widget) -> None:
    widget.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))


class SaveProgressDialog(QDialog):
    """Name (and optionally annotate) a save point. For a spectrum with
    no folder of its own (a simulation or derived spectrum) it also asks
    where to save it, prefilled with the suggested folder."""

    def __init__(self, parent=None, label: str = "", default_name: str = "",
                 folder: str | None = None, where: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Save progress")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        intro = QLabel(
            f"Save the current state of <b>{label}</b> as a named save point. "
            "Everything done since opening it is already in its log; a save "
            "point is a state you can return to with Load Progress.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.name_edit = QLineEdit(default_name)
        self.name_edit.setPlaceholderText("e.g. phased and baseline-corrected")
        self.name_edit.selectAll()
        form.addRow("Name:", self.name_edit)
        self.note_edit = QPlainTextEdit()
        self.note_edit.setPlaceholderText("Optional: why, what to check next…")
        self.note_edit.setFixedHeight(70)
        form.addRow("Note:", self.note_edit)

        self.folder_edit = None
        if folder is not None:
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            self.folder_edit = QLineEdit(folder)
            self.folder_edit.setToolTip(
                "This spectrum has no data folder of its own, so it is saved as an "
                "openNMR dataset folder: the data, what produced it, its processing "
                "and its log. Reopen it with File ▸ Open like any experiment.")
            browse = QPushButton("Browse…")
            browse.clicked.connect(self._on_browse)
            row_layout.addWidget(self.folder_edit, 1)
            row_layout.addWidget(browse)
            form.addRow("Save to:", row_layout)
        layout.addLayout(form)
        if where:
            note = QLabel(where)
            note.setWordWrap(True)
            note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            note.setStyleSheet("color: palette(placeholder-text);")
            layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_browse(self) -> None:
        start = os.path.dirname(self.folder_edit.text()) or os.path.expanduser("~")
        chosen = QFileDialog.getExistingDirectory(self, "Save the dataset folder inside…", start)
        if chosen:
            self.folder_edit.setText(os.path.join(chosen, os.path.basename(self.folder_edit.text())))

    def values(self) -> tuple[str, str, str | None]:
        folder = self.folder_edit.text().strip() if self.folder_edit is not None else None
        return self.name_edit.text().strip(), self.note_edit.toPlainText().strip(), folder or None


class LoadProgressDialog(QDialog):
    """Pick a save point (or the spectrum as first loaded) to go back
    to, with what each one holds shown before committing."""

    def __init__(self, parent=None, label: str = "", points: list[dict] | None = None,
                 original_note: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load progress")
        self.resize(760, 460)
        layout = QVBoxLayout(self)
        intro = QLabel(
            f"Return <b>{label}</b> to an earlier save point. Where the current state "
            "is not already saved, it is kept first as an automatic save point, so "
            "nothing is lost by restoring.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setSpacing(3)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        _mono(self.preview)
        splitter.addWidget(self.list)
        splitter.addWidget(self.preview)
        splitter.setSizes([280, 480])
        layout.addWidget(splitter, 1)

        self._details: dict[int, str] = {}
        for point in reversed(points or []):
            item = QListWidgetItem(f"{point['id']}. {point['name']}\n    {point['when']}")
            item.setData(Qt.ItemDataRole.UserRole, point["id"])
            self.list.addItem(item)
            self._details[point["id"]] = point["details"]
        if original_note is not None:
            item = QListWidgetItem("Original\n    as first loaded, before any openNMR processing")
            item.setData(Qt.ItemDataRole.UserRole, ORIGINAL)
            self.list.addItem(item)
            self._details[ORIGINAL] = original_note
        self.list.currentItemChanged.connect(self._on_current_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.restore_button = buttons.addButton("Restore", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.restore_button.setEnabled(False)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _on_current_changed(self, current, _previous) -> None:
        self.restore_button.setEnabled(current is not None)
        if current is not None:
            self.preview.setPlainText(self._details.get(current.data(Qt.ItemDataRole.UserRole), ""))

    def selected(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None


class ProcessingLogDialog(QDialog):
    """The spectrum's log, readable and copyable, with a way to open the
    folder holding the file."""

    def __init__(self, parent=None, text: str = "", file_path: str | None = None,
                 status: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing log")
        self.resize(780, 560)
        layout = QVBoxLayout(self)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlainText(text)
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        _mono(self.text_edit)
        layout.addWidget(self.text_edit, 1)
        self.status_label = QLabel(status)
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.copy_button = buttons.addButton("Copy", QDialogButtonBox.ButtonRole.ActionRole)
        self.copy_button.clicked.connect(
            lambda: QApplication.clipboard().setText(self.text_edit.toPlainText()))
        self.folder_button = buttons.addButton("Show Folder", QDialogButtonBox.ButtonRole.ActionRole)
        self.folder_button.setEnabled(bool(file_path) and os.path.exists(file_path or ""))
        self.folder_button.setToolTip(file_path or "Not written to disk yet.")
        self.folder_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(file_path))))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class SimulationHistoryDialog(QDialog):
    """Every logged simulation run, newest first, with its inputs and
    result shown before restoring one."""

    def __init__(self, parent=None, entries: list[dict] | None = None, log_path: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Simulation history")
        self.resize(820, 520)
        layout = QVBoxLayout(self)
        intro = QLabel("Restore a simulation's inputs into its Simulate panel and run it again "
                       "(a pNMR run is set up, ready to enumerate). The full log is "
                       f"<i>{log_path}</i>." if log_path else
                       "Restore a simulation's inputs into its Simulate panel and run it again.")
        intro.setWordWrap(True)
        intro.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setSpacing(3)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        _mono(self.preview)
        splitter.addWidget(self.list)
        splitter.addWidget(self.preview)
        splitter.setSizes([280, 540])
        layout.addWidget(splitter, 1)

        self._details: dict[int, str] = {}
        for entry in reversed(entries or []):
            when = (entry.get("updated") or entry.get("time") or "").replace("T", " ")[:16]
            runs = int(entry.get("runs", 1))
            item = QListWidgetItem(f"#{entry['id']} {entry.get('label', entry.get('panel'))}"
                                   f"{f' (×{runs})' if runs > 1 else ''}\n    {when}")
            item.setData(Qt.ItemDataRole.UserRole, entry["id"])
            self.list.addItem(item)
            text = ["Inputs:"] + ["  " + str(l) for l in entry.get("inputs_text") or []]
            if entry.get("summary"):
                text += ["", "Result:"] + ["  " + l for l in str(entry["summary"]).splitlines()]
            self._details[entry["id"]] = "\n".join(text)
        self.list.currentItemChanged.connect(self._on_current_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.restore_button = buttons.addButton("Restore", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.restore_button.setEnabled(False)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _on_current_changed(self, current, _previous) -> None:
        self.restore_button.setEnabled(current is not None)
        if current is not None:
            self.preview.setPlainText(self._details.get(current.data(Qt.ItemDataRole.UserRole), ""))

    def selected(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None
