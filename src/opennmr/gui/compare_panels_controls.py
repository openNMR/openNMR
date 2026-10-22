"""Compare ▸ Layout ▸ Panels: the controls for separate 1D panels in a row,
column or grid -- link the panels' zoom, choose the grid's columns, and say
which panel each checked spectrum goes in (one or many spectra per panel)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..notebook.compare_panels import panel_groups
from .widgets.typed_spinbox import TypedSpinBox


class ComparePanelsControls(QGroupBox):
    """``changed`` -- the assignment or grid shape changed (redraw);
    ``link_toggled(x_linked, y_linked)`` -- only the linking changed (no
    redraw needed)."""

    changed = Signal()
    link_toggled = Signal(bool, bool)

    def __init__(self, parent=None) -> None:
        super().__init__("Panels", parent)
        self._keys: list = []
        self._panel_of: dict = {}          # key -> panel number (kept when a row is unchecked)
        self._building = False
        layout = QVBoxLayout(self)

        links = QHBoxLayout()
        self.link_check = QCheckBox("Link panels")
        self.link_check.setChecked(True)
        self.link_check.setToolTip("On: zooming, panning or scrolling one panel does the same on all "
                                   "(the chemical-shift axis). Off: each panel is zoomed on its own.")
        self.link_y_check = QCheckBox("Link intensity")
        self.link_y_check.setToolTip("Also share the vertical scale. Only meaningful for spectra "
                                     "on a common intensity scale.")
        links.addWidget(self.link_check)
        links.addWidget(self.link_y_check)
        layout.addLayout(links)

        cols = QHBoxLayout()
        cols.addWidget(QLabel("Grid columns:"))
        self.columns_spin = TypedSpinBox()
        self.columns_spin.setRange(0, 12)
        self.columns_spin.setSpecialValueText("auto")
        self.columns_spin.setToolTip("Columns of a grid layout (auto: as square as possible).")
        cols.addWidget(self.columns_spin)
        cols.addStretch(1)
        layout.addLayout(cols)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Spectrum", "Panel"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(160)
        self.table.setToolTip("Which panel each checked spectrum is drawn in. Give several spectra "
                              "the same number to superimpose them in one panel.")
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        one = QPushButton("One per panel")
        one.clicked.connect(self.one_per_panel)
        every = QPushButton("All in one panel")
        every.clicked.connect(self.all_in_one)
        buttons.addWidget(one)
        buttons.addWidget(every)
        layout.addLayout(buttons)

        self.link_check.toggled.connect(self._emit_link)
        self.link_y_check.toggled.connect(self._emit_link)
        self.columns_spin.valueChanged.connect(lambda _v: self._emit_changed())

    # --- state -----------------------------------------------------------

    def linked(self) -> bool:
        return self.link_check.isChecked()

    def linked_y(self) -> bool:
        return self.link_y_check.isChecked()

    def columns(self):
        return self.columns_spin.value() or None

    def set_spectra(self, keys, labels) -> None:
        """The checked spectra (stable ``keys``, display ``labels``), in plot
        order. A spectrum keeps the panel it had; new ones get their own."""
        keys = [str(k) for k in keys]
        if keys == self._keys and self.table.rowCount() == len(keys):
            for row, label in enumerate(labels):
                self.table.item(row, 0).setText(label)
            return
        self._building = True
        try:
            used = {self._panel_of[k] for k in keys if k in self._panel_of}
            nxt = 1
            for k in keys:
                if k not in self._panel_of:
                    while nxt in used:
                        nxt += 1
                    self._panel_of[k] = nxt
                    used.add(nxt)
            self._keys = keys
            self.table.setRowCount(len(keys))
            for row, (k, label) in enumerate(zip(keys, labels)):
                item = QTableWidgetItem(label)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, 0, item)
                spin = TypedSpinBox()
                spin.setRange(1, 99)
                spin.setValue(self._panel_of[k])
                spin.valueChanged.connect(lambda v, key=k: self._on_panel(key, v))
                self.table.setCellWidget(row, 1, spin)
        finally:
            self._building = False

    def assignment(self) -> list:
        """The panel number of each spectrum, in plot order."""
        return [self._panel_of.get(k, i + 1) for i, k in enumerate(self._keys)]

    def groups(self) -> list:
        """Spectrum indices grouped by panel, panels in number order."""
        return panel_groups(self.assignment())

    def one_per_panel(self) -> None:
        self._assign(list(range(1, len(self._keys) + 1)))

    def all_in_one(self) -> None:
        self._assign([1] * len(self._keys))

    def _assign(self, numbers) -> None:
        self._building = True
        try:
            for row, (k, n) in enumerate(zip(self._keys, numbers)):
                self._panel_of[k] = n
                spin = self.table.cellWidget(row, 1)
                if spin is not None:
                    spin.setValue(n)
        finally:
            self._building = False
        self._emit_changed()

    def _on_panel(self, key, value) -> None:
        self._panel_of[key] = int(value)
        self._emit_changed()

    def _emit_changed(self) -> None:
        if not self._building:
            self.changed.emit()

    def _emit_link(self, _checked=None) -> None:
        if not self._building:
            self.link_toggled.emit(self.linked(), self.linked_y())
