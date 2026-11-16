"""Copy a fit to another experiment (explicit user request, 2026-11-16):
which experiment to copy it to, and what to do with it there.

A fit is copied as a *starting point* (see
:mod:`opennmr.notebook.fit_transfer`): the other experiment has its own
baseline and noise, so what is fitted there is its own fit, not the old
numbers pasted in.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QLabel, QRadioButton, QVBoxLayout,
)


class CopyFitDialog(QDialog):
    """``targets``: ``[(label, row)]``. ``modes``: ``[(label, key, tooltip)]``,
    the first being the default."""

    def __init__(self, title: str, intro: str, targets, modes, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._modes: list[tuple[QRadioButton, str]] = []
        layout = QVBoxLayout(self)
        note = QLabel(intro)
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(QLabel("Copy to:"))
        self.target_combo = QComboBox()
        for label, row in targets:
            self.target_combo.addItem(label, row)
        layout.addWidget(self.target_combo)
        layout.addWidget(QLabel("Then:"))
        for i, (label, key, tip) in enumerate(modes):
            button = QRadioButton(label)
            button.setToolTip(tip)
            button.setChecked(i == 0)
            layout.addWidget(button)
            self._modes.append((button, key))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setMinimumWidth(460)

    def target_row(self) -> int | None:
        data = self.target_combo.currentData()
        return None if data is None else int(data)

    def mode(self) -> str:
        return next((key for button, key in self._modes if button.isChecked()),
                    self._modes[0][1])
