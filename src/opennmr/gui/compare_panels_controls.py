"""Compare ▸ Layout ▸ Panels: the controls for separate panels in a row,
column or grid -- link the panels' zoom, choose the grid's columns, say which
panel each checked spectrum goes in (one or many spectra per panel) and whether
a panel shows the spectrum or its pulse sequence, and style each panel's axis.

1D spectra, 2D spectra and pulse sequences can share one figure. The widget
holds two collapsible sub-sections, *Panels* and *Axes*."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..notebook.compare_panels import TICK_DIRECTIONS, panel_groups
from .widgets.collapsible_section import CollapsibleSection
from .widgets.typed_spinbox import TypedDoubleSpinBox, TypedSpinBox

#: What a panel shows for a checked spectrum.
CONTENTS = (("Spectrum", "spectrum"), ("Pulse sequence", "pulse"))


class ComparePanelsControls(QWidget):
    """``changed`` -- the assignment or grid shape changed (redraw);
    ``link_toggled(x_linked, y_linked)`` -- only the linking changed (no
    redraw needed)."""

    changed = Signal()
    link_toggled = Signal(bool, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._keys: list = []
        self._panel_of: dict = {}          # key -> panel number (kept when a row is unchecked)
        self._content_of: dict = {}        # key -> "spectrum" | "pulse"
        self._building = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        panels_body = QWidget()
        layout = QVBoxLayout(panels_body)
        layout.setContentsMargins(0, 0, 0, 0)

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

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Spectrum", "Panel", "Show"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(180)
        self.table.setToolTip("Which panel each checked spectrum is drawn in. Give several spectra "
                              "the same number to put them in one panel (1D together are "
                              "superimposed; 2D together are overlaid); a different number gives a "
                              "spectrum its own axes. A 1D and a 2D cannot share a panel. "
                              "Show: the spectrum, or its pulse sequence.")
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        one = QPushButton("One per panel")
        one.clicked.connect(self.one_per_panel)
        every = QPushButton("All in one panel")
        every.clicked.connect(self.all_in_one)
        buttons.addWidget(one)
        buttons.addWidget(every)
        layout.addLayout(buttons)
        self.panels_section = CollapsibleSection("Panels", content=panels_body, expanded=False)
        outer.addWidget(self.panels_section)
        self.axes_section = CollapsibleSection("Axes", content=self._build_axis_group(), expanded=False)
        outer.addWidget(self.axes_section)

        self.link_check.toggled.connect(self._emit_link)
        self.link_y_check.toggled.connect(self._emit_link)
        self.columns_spin.valueChanged.connect(lambda _v: self._emit_changed())

    # --- axes ------------------------------------------------------------

    def _build_axis_group(self) -> QWidget:
        """Axis labels and the look of each panel's shift axis."""
        box = QWidget()
        g = QGridLayout(box)
        g.setContentsMargins(0, 0, 0, 0)
        self._axis_styles: dict = {}            # "all" or 0-based panel -> {option: value}
        g.addWidget(QLabel("Labels"), 0, 0)
        self.axis_labels_combo = QComboBox()
        self.axis_labels_combo.addItem("Automatic", "auto")
        self.axis_labels_combo.addItem("On every panel", "all")
        self.axis_labels_combo.addItem("Bottom panels only", "bottom")
        self.axis_labels_combo.setToolTip(
            "Automatic: every panel has its own label when the axes read differently (different "
            "nuclei) or when 2D spectra or pulse sequences are among the panels, only the bottom "
            "ones when 1D panels all read the same.")
        self.axis_labels_combo.currentIndexChanged.connect(lambda _i: self._emit_changed())
        g.addWidget(self.axis_labels_combo, 0, 1)
        g.addWidget(QLabel("Edit"), 1, 0)
        self.axis_target_combo = QComboBox()
        self.axis_target_combo.addItem("All panels", "all")
        self.axis_target_combo.currentIndexChanged.connect(self._load_axis_editor)
        g.addWidget(self.axis_target_combo, 1, 1)
        g.addWidget(QLabel("Label text"), 2, 0)
        self.axis_label_edit = QLineEdit()
        self.axis_label_edit.setPlaceholderText("(automatic)")
        self.axis_label_edit.setToolTip("Replace the axis label; LaTeX works, e.g. $\\delta$ ($^{23}$Na)")
        self.axis_label_edit.editingFinished.connect(lambda: self._set_axis("xlabel", self.axis_label_edit.text() or None))
        g.addWidget(self.axis_label_edit, 2, 1)
        self.axis_show_label = QCheckBox("Show the label")
        self.axis_show_label.setChecked(True)
        self.axis_show_label.toggled.connect(lambda v: self._set_axis("show_xlabel", bool(v)))
        g.addWidget(self.axis_show_label, 3, 0, 1, 2)
        g.addWidget(QLabel("Ticks point"), 4, 0)
        self.axis_direction_combo = QComboBox()
        self.axis_direction_combo.addItem("(as Plot Style)", None)
        for d in TICK_DIRECTIONS:
            self.axis_direction_combo.addItem({"in": "in", "out": "out", "inout": "both ways"}[d], d)
        self.axis_direction_combo.currentIndexChanged.connect(
            lambda _i: self._set_axis("tick_direction", self.axis_direction_combo.currentData()))
        g.addWidget(self.axis_direction_combo, 4, 1)
        self._axis_spins = {}
        for row, (key, text, hi) in enumerate((("tick_length", "Tick length", 30.0),
                                               ("tick_width", "Tick width", 10.0),
                                               ("tick_labelsize", "Tick label size", 60.0),
                                               ("xlabel_fontsize", "Label size", 60.0)), start=5):
            g.addWidget(QLabel(text), row, 0)
            spin = TypedDoubleSpinBox()
            spin.setRange(0.0, hi)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setSpecialValueText("(as Plot Style)")
            spin.valueChanged.connect(lambda v, k=key: self._set_axis(k, float(v) or None))
            g.addWidget(spin, row, 1)
            self._axis_spins[key] = spin
        self.axis_minor_check = QCheckBox("Minor ticks")
        self.axis_minor_check.setTristate(True)
        self.axis_minor_check.setCheckState(Qt.CheckState.PartiallyChecked)
        self.axis_minor_check.setToolTip("Partly filled: as Plot Style.")
        self.axis_minor_check.clicked.connect(self._on_minor_clicked)
        g.addWidget(self.axis_minor_check, 9, 0)
        self.axis_tick_labels_check = QCheckBox("Tick numbers")
        self.axis_tick_labels_check.setChecked(True)
        self.axis_tick_labels_check.toggled.connect(lambda v: self._set_axis("show_tick_labels", bool(v)))
        g.addWidget(self.axis_tick_labels_check, 9, 1)
        reset = QPushButton("Reset axes")
        reset.clicked.connect(self.reset_axes)
        g.addWidget(reset, 10, 0, 1, 2)
        return box

    def set_panel_count(self, n: int) -> None:
        """One entry per panel in the axis editor's target list (called with
        the number of panels currently drawn)."""
        have = self.axis_target_combo.count() - 1
        if have == n:
            return
        current = self.axis_target_combo.currentData()
        self._building = True
        try:
            self.axis_target_combo.clear()
            self.axis_target_combo.addItem("All panels", "all")
            for k in range(n):
                self.axis_target_combo.addItem(f"Panel {k + 1}", k)
            idx = self.axis_target_combo.findData(current)
            self.axis_target_combo.setCurrentIndex(max(0, idx))
        finally:
            self._building = False
        self._load_axis_editor()

    def _target(self):
        return self.axis_target_combo.currentData()

    def _set_axis(self, key: str, value) -> None:
        if self._building:
            return
        target = self._target()
        style = self._axis_styles.setdefault(target, {})
        if value is None:
            style.pop(key, None)
        else:
            style[key] = value
        if not style:
            self._axis_styles.pop(target, None)
        self._emit_changed()

    def _on_minor_clicked(self, _checked=None) -> None:
        # cycle: as Plot Style -> on -> off
        state = self.axis_minor_check.checkState()
        value = {Qt.CheckState.Checked: True, Qt.CheckState.Unchecked: False}.get(state)
        self._set_axis("minor_ticks", value)

    def _load_axis_editor(self, _i=None) -> None:
        """Show the chosen panel's stored options in the editor's widgets."""
        style = self._axis_styles.get(self._target(), {})
        self._building = True
        try:
            self.axis_label_edit.setText(style.get("xlabel") or "")
            self.axis_show_label.setChecked(style.get("show_xlabel", True))
            self.axis_direction_combo.setCurrentIndex(
                max(0, self.axis_direction_combo.findData(style.get("tick_direction"))))
            for key, spin in self._axis_spins.items():
                spin.setValue(float(style.get(key) or 0.0))
            minor = style.get("minor_ticks")
            self.axis_minor_check.setCheckState(Qt.CheckState.PartiallyChecked if minor is None else
                                                Qt.CheckState.Checked if minor else Qt.CheckState.Unchecked)
            self.axis_tick_labels_check.setChecked(style.get("show_tick_labels", True))
        finally:
            self._building = False

    def reset_axes(self) -> None:
        self._axis_styles = {}
        self._load_axis_editor()
        self._emit_changed()

    def axis_labels(self) -> str:
        return self.axis_labels_combo.currentData()

    def axis_styles(self) -> dict:
        """``{"all" or panel (0-based): {option: value}}``, as
        :func:`opennmr.notebook.compare_panels.plot_compare_panels` takes."""
        return {k: dict(v) for k, v in self._axis_styles.items() if v}

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
                show = QComboBox()
                for text, value in CONTENTS:
                    show.addItem(text, value)
                show.setCurrentIndex(max(0, show.findData(self._content_of.get(k, "spectrum"))))
                show.setToolTip("Spectrum: the data. Pulse sequence: this experiment's pulse programme, "
                                "drawn as in the Pulse sequence viewer (its colours, names and fonts), "
                                "in a panel of its own.")
                show.currentIndexChanged.connect(lambda _i, key=k, box=show: self._on_content(key, box.currentData()))
                self.table.setCellWidget(row, 2, show)
        finally:
            self._building = False

    def contents(self) -> list:
        """``"spectrum"`` or ``"pulse"`` for each spectrum, in plot order."""
        return [self._content_of.get(k, "spectrum") for k in self._keys]

    def contents_by_key(self) -> dict:
        """``{series key: "spectrum" | "pulse"}`` for the spectra in the table."""
        return {k: self._content_of.get(k, "spectrum") for k in self._keys}

    def has_pulse(self) -> bool:
        """Whether any spectrum is set to show its pulse sequence."""
        return any(v == "pulse" for v in self.contents_by_key().values())

    def _on_content(self, key, value) -> None:
        self._content_of[key] = value
        self._emit_changed()

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
