"""
Proc numbers of a Bruker experiment, as a checklist (explicit user
request, 2026-10-14: "on right clicking a spectrum (either in Open
experiments or in the list of experiments that can be opened in the Open
tab), see and select the different proc numbers available -- perhaps as
a pop-up box with a checklist").

Each ``pdata/<n>`` is one row: its number, TopSpin title, what TopSpin
did there (window, SI, phase; F1 for a 2D) and which processed files it
holds. Tick the ones to open; each opens as its own experiment in Open
experiments. A proc number that is already open is shown ticked and
locked -- close it with Remove in Open experiments.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout,
)

PROCNO_ROLE = Qt.ItemDataRole.UserRole


def procno_text(entry: dict) -> str:
    """One row's text: ``procno 2 -- title  (QSINE SSB 3, SI 2048; 2rr)``."""
    title = f" — {entry['title']}" if entry.get("title") else ""
    details = [d for d in (entry.get("summary"),
                           ", ".join(entry.get("processed") or []) or "not processed")
               if d]
    return f"procno {entry['procno']}{title}   ({'; '.join(details)})"


class ProcnoDialog(QDialog):
    """``entries``: :func:`opennmr.io.bruker.available_procnos`;
    ``open_procnos``: those already open (ticked and locked);
    ``preselect``: also ticked to start with (e.g. the one right-clicked)."""

    def __init__(self, experiment: str, entries: list[dict], open_procnos=(), preselect=(),
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Proc numbers")
        self._open = {int(p) for p in open_procnos}
        layout = QVBoxLayout(self)
        intro = QLabel(
            f"<b>{experiment}</b>: tick the proc numbers to open. Each opens as its own "
            "experiment, read from the raw data with that proc number's processing "
            "(window, zero-fill, phase). Those already open are ticked and locked.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.list = QListWidget()
        for entry in entries:
            item = QListWidgetItem(procno_text(entry))
            item.setData(PROCNO_ROLE, int(entry["procno"]))
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            if entry["procno"] in self._open:
                flags &= ~Qt.ItemFlag.ItemIsEnabled
                item.setToolTip("Already open. Remove it from Open experiments to close it.")
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Checked if entry["procno"] in set(preselect)
                                   else Qt.CheckState.Unchecked)
            item.setFlags(flags)
            self.list.addItem(item)
        layout.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, state in (("Tick All", Qt.CheckState.Checked),
                            ("Untick All", Qt.CheckState.Unchecked)):
            button = QPushButton(text)
            button.clicked.connect(lambda _c=False, s=state: self._set_all(s))
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Open Ticked")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(560, 320)

    def _set_all(self, state) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(state)

    def set_checked(self, procnos) -> None:
        wanted = {int(p) for p in procnos}
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked if item.data(PROCNO_ROLE) in wanted
                                   else Qt.CheckState.Unchecked)

    def to_open(self) -> list[int]:
        """The ticked proc numbers not already open, in order."""
        out = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            procno = item.data(PROCNO_ROLE)
            if item.checkState() == Qt.CheckState.Checked and procno not in self._open:
                out.append(procno)
        return out
