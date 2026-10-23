"""The "Export … Sites to Solid-state NMR" controls shared by
Simulate ▸ pNMR (CIF) and Fit ▸ pNMR (CIF bond pathways) (explicit user
request, 2026-10-12: pNMR computes isotropic peaks only; send them on so
CSA, quadrupolar and J interactions can be added to each).

The maths is :mod:`opennmr.notebook.pnmr_export`; the main window puts
the sites into the chosen panel."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

#: The panels sites can go to: (combo text, key).
DESTINATIONS = (("Simulate ▸ Solid-state NMR", "simulate"),
                ("Fit ▸ Solid-state NMR (as the starting model)", "fit"))


class SendSitesRow(QWidget):
    """Destination, replace/add, and Export All / Export Selected.
    Emits ``exportRequested("all" | "selected")``."""

    exportRequested = Signal(str)

    def __init__(self, default_destination: str = "simulate", selected_what: str = "peaks",
                 parent=None) -> None:
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Send to:"))
        self.destination_combo = QComboBox()
        for text, key in DESTINATIONS:
            self.destination_combo.addItem(text, key)
        self.destination_combo.setCurrentIndex(
            max(0, self.destination_combo.findData(default_destination)))
        self.destination_combo.setToolTip(
            "Where the sites go. There, select a site and add CSA, quadrupolar or J-coupling "
            "interactions to it.")
        row.addWidget(self.destination_combo, 1)
        v.addLayout(row)
        self.append_check = QCheckBox("Add to the sites already there (otherwise replace them)")
        v.addWidget(self.append_check)
        row = QHBoxLayout()
        self.export_all_button = QPushButton("Export All Sites to Solid-state NMR")
        self.export_all_button.setToolTip(
            "Every isotropic peak becomes a site (δiso, amount, width) in Solid-state NMR.")
        self.export_all_button.clicked.connect(lambda: self.exportRequested.emit("all"))
        self.export_selected_button = QPushButton("Export Selected")
        self.export_selected_button.setToolTip(
            f"Only the {selected_what} selected in the table above (Ctrl/Shift-click for several).")
        self.export_selected_button.clicked.connect(lambda: self.exportRequested.emit("selected"))
        row.addWidget(self.export_all_button, 1)
        row.addWidget(self.export_selected_button)
        v.addLayout(row)

    def destination(self) -> str:
        return self.destination_combo.currentData()

    def append(self) -> bool:
        return self.append_check.isChecked()

    def set_enabled(self, all_ok: bool, selected_ok: bool) -> None:
        self.export_all_button.setEnabled(all_ok)
        self.export_selected_button.setEnabled(selected_ok)
