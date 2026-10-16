"""A plot's own display options in a small window, opened from that plot's
right-click menu (explicit user request, 2026-10-16: "Fit plot: components
and colours" and the formatting of the Dynamics plots belong on the plot's
right-click menu, not in the side panel).

The dialog holds the options widget itself, so the widget keeps its
signals and attributes (``panel.y_axis_combo`` still works) and is only
shown somewhere else. Non-modal: the plot updates while it is open.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget


class PlotOptionsDialog(QDialog):
    """``content``: the options widget; ``parent``: whoever owns it (the
    dialog is deleted with them)."""

    def __init__(self, title: str, content: QWidget, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.content = content
        layout = QVBoxLayout(self)
        layout.addWidget(content)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.hide)
        buttons.accepted.connect(self.hide)
        layout.addWidget(buttons)

    def open_options(self) -> None:
        """Show (or bring forward) the window."""
        self.show()
        self.raise_()
        self.activateWindow()
