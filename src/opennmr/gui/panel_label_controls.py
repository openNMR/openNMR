"""Compare ▸ Panel labels: (a), (b), ... on the panels of a multi-panel figure.

The controls for :func:`opennmr.notebook.panel_labels.add_panel_labels`: which
style (``(a)``, ``A.``, ``(iv)`` ...), which label comes first, in which order
the panels are read, and where each label sits -- placed automatically (clear of
the tick labels, one row sharing a baseline, one column a left edge) or by hand,
by dragging a label on the plot or typing its offset.

The widget holds choices only; :meth:`PanelLabelControls.args` turns them into the
keyword dict the canvas and the notebook function take, and ``changed`` says the
labels should be redrawn.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView)

from ..notebook.panel_labels import LABEL_STYLES, describe_style, label_text, parse_start
from .widgets.typed_spinbox import TypedDoubleSpinBox, TypedSpinBox


class LabelStartSpin(TypedSpinBox):
    """A spin box that counts in labels: it shows ``(b)`` rather than ``2``, takes
    a typed ``b`` (or ``iv`` for a roman style), and steps through the alphabet."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._style = "(a)"
        self.setRange(1, 99)

    def set_style(self, style: str) -> None:
        self._style = style
        self.lineEdit().setText(self.textFromValue(self.value()))

    def textFromValue(self, value: int) -> str:  # noqa: N802 -- Qt override
        try:
            return label_text(value, self._style)
        except ValueError:
            return str(value)

    def valueFromText(self, text: str) -> int:  # noqa: N802 -- Qt override
        try:
            return parse_start(text, self._style)
        except ValueError:
            return self.value()

    def fixup(self, text: str) -> str:
        return self.textFromValue(self.valueFromText(text))

    def validate(self, text: str, pos: int):
        try:
            parse_start(text, self._style)
        except ValueError:
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Acceptable, text, pos


ORDERS = (("Left to right, then down", "row"),
          ("Top to bottom, then right", "column"),
          ("Custom order…", "custom"))
PLACEMENTS = (("Automatic: above the top-left corner", "auto"),
              ("Automatic: inside the top-left corner", "inside"),
              ("Manual", "manual"))


class PanelLabelControls(QWidget):
    """``changed`` -- redraw the labels (they are cheap: no replot)."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._building = False
        self._panel_count = 0
        self._custom: list = []                  # label number per panel, 0 = none
        self._positions: dict = {}               # "all" or panel -> (dx, dy) points

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.enable_check = QCheckBox("Label the panels")
        self.enable_check.setToolTip(
            "Write (a), (b)… on the panels. Works for panels of 1D, 2D and pulse-sequence "
            "plots (Layout ▸ Panels) and for 2D spectra side by side.")
        root.addWidget(self.enable_check)

        self.body = QWidget()
        form = QFormLayout(self.body)
        form.setContentsMargins(0, 0, 0, 0)

        self.style_combo = QComboBox()
        for style in LABEL_STYLES:
            self.style_combo.addItem(f"{describe_style(style)}  …", style)
        self.style_combo.setToolTip(
            "How the labels look. Roman numerals (I, II, III) and lower-case roman (i, ii, iii) "
            "are in the list too.")
        form.addRow("Style:", self.style_combo)

        self.start_spin = LabelStartSpin()
        self.start_spin.setToolTip(
            "The label the first panel gets: start at (b) or (c) if panel (a) is somewhere else. "
            "Type a letter, or use the arrows.")
        form.addRow("First label:", self.start_spin)

        self.order_combo = QComboBox()
        for text, key in ORDERS:
            self.order_combo.addItem(text, key)
        self.order_combo.setToolTip(
            "Which panel is read first. Custom: give each panel its own place in the sequence "
            "(0 leaves a panel unlabelled).")
        form.addRow("Order:", self.order_combo)

        self.order_table = QTableWidget(0, 2)
        self.order_table.setHorizontalHeaderLabels(["Panel", "Label no."])
        self.order_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.order_table.verticalHeader().setVisible(False)
        self.order_table.setMaximumHeight(150)
        self.order_table.setVisible(False)
        form.addRow(self.order_table)

        self.placement_combo = QComboBox()
        for text, key in PLACEMENTS:
            self.placement_combo.addItem(text, key)
        self.placement_combo.setToolTip(
            "Automatic keeps every label clear of the tick numbers, axis labels and titles, and "
            "lines them up: the labels of a row share a baseline, those of a column a left edge. "
            "Manual puts them where you drag them (drag any label on the plot) or type below.")
        form.addRow("Placement:", self.placement_combo)

        self.manual_box = QWidget()
        manual = QFormLayout(self.manual_box)
        manual.setContentsMargins(0, 0, 0, 0)
        self.target_combo = QComboBox()
        self.target_combo.addItem("All panels", "all")
        manual.addRow("Move:", self.target_combo)
        self.dx_spin = self._offset_spin("Points to the right of the panel's top-left corner "
                                         "(negative: to the left, outside the panel).")
        self.dy_spin = self._offset_spin("Points above the panel's top-left corner "
                                         "(negative: below it, inside the panel).")
        manual.addRow("Right (pt):", self.dx_spin)
        manual.addRow("Up (pt):", self.dy_spin)
        hint = QLabel("Or drag a label on the plot.")
        hint.setEnabled(False)
        manual.addRow(hint)
        self.together_check = QCheckBox("Dragging moves all labels together")
        self.together_check.setChecked(True)
        self.together_check.setToolTip(
            "On: dragging one label puts every label at that place relative to its own panel, so "
            "they stay aligned. Off: only the label you drag moves.")
        manual.addRow(self.together_check)
        self.manual_box.setVisible(False)
        form.addRow(self.manual_box)

        self.size_spin = TypedDoubleSpinBox()
        self.size_spin.setRange(0.0, 80.0)
        self.size_spin.setDecimals(1)
        self.size_spin.setSingleStep(1.0)
        self.size_spin.setSpecialValueText("Automatic")
        self.size_spin.setSuffix(" pt")
        self.size_spin.setToolTip("Automatic: a little larger than the axis labels.")
        form.addRow("Size:", self.size_spin)
        self.bold_check = QCheckBox("Bold")
        self.bold_check.setChecked(True)
        form.addRow("", self.bold_check)
        root.addWidget(self.body)
        self.body.setEnabled(False)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self.changed)

        self.enable_check.toggled.connect(self._on_enabled)
        self.style_combo.currentIndexChanged.connect(self._on_style)
        self.order_combo.currentIndexChanged.connect(self._on_order)
        self.placement_combo.currentIndexChanged.connect(self._on_placement)
        self.target_combo.currentIndexChanged.connect(self._load_offsets)
        for spin in (self.dx_spin, self.dy_spin):
            spin.valueChanged.connect(self._on_offset_edited)
        for widget in (self.start_spin, self.size_spin):
            widget.valueChanged.connect(self._schedule)
        self.bold_check.toggled.connect(self._schedule)

    # -- public ----------------------------------------------------------

    def is_enabled(self) -> bool:
        return self.enable_check.isChecked()

    def placement(self) -> str:
        return self.placement_combo.currentData()

    def positions(self) -> dict:
        return dict(self._positions)

    def set_panel_count(self, n: int) -> None:
        """How many panels the figure has (the custom order and the manual
        target list follow it)."""
        n = max(0, int(n))
        if n == self._panel_count:
            return
        self._panel_count = n
        self._custom = [(self._custom[i] if i < len(self._custom) else i + 1) for i in range(n)]
        self._building = True
        try:
            current = self.target_combo.currentData()
            self.target_combo.clear()
            self.target_combo.addItem("All panels", "all")
            for k in range(n):
                self.target_combo.addItem(f"Panel {k + 1}", k)
            self.target_combo.setCurrentIndex(max(0, self.target_combo.findData(current)))
            self._fill_order_table()
        finally:
            self._building = False
        self._load_offsets()

    def args(self) -> dict | None:
        """The keyword arguments of ``add_panel_labels``, or None when the
        labels are off."""
        if not self.is_enabled():
            return None
        order = self.order_combo.currentData()
        if order == "custom":
            order = [(v or None) for v in self._custom] or "row"
        out = {"style": self.style_combo.currentData(), "start": self.start_spin.value(),
               "order": order, "placement": self.placement(),
               "weight": "bold" if self.bold_check.isChecked() else "normal"}
        if self.placement() == "manual" or self._positions:
            out["positions"] = dict(self._positions)
        size = self.size_spin.value()
        if size > 0:
            out["fontsize"] = float(size)
        return out

    def seed_positions(self, offsets) -> None:
        """Freeze the labels where they are now (``offsets``: ``(dx, dy)`` per
        panel) -- what switching to Manual starts from, so nothing jumps."""
        self._positions = {i: (float(dx), float(dy)) for i, (dx, dy) in enumerate(offsets)}
        self._load_offsets()

    def apply_drag(self, panel: int, dx: float, dy: float, offsets=None) -> None:
        """A label was dragged on the plot: become Manual and keep it there.
        ``offsets``: where every label was just before, for the ones that stay."""
        if self.together_check.isChecked():
            self._positions = {"all": (float(dx), float(dy))}
        else:
            if offsets is not None and not self._positions:
                self.seed_positions(offsets)
            self._positions[int(panel)] = (float(dx), float(dy))
        self._building = True
        try:
            self.placement_combo.setCurrentIndex(self.placement_combo.findData("manual"))
            self.manual_box.setVisible(True)
        finally:
            self._building = False
        self._load_offsets()
        self._schedule()

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _offset_spin(tip: str) -> TypedDoubleSpinBox:
        spin = TypedDoubleSpinBox()
        spin.setRange(-400.0, 400.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setToolTip(tip)
        return spin

    def _schedule(self, *_args) -> None:
        if not self._building:
            self._debounce.start()

    def _on_enabled(self, on: bool) -> None:
        self.body.setEnabled(bool(on))
        self._schedule()

    def _on_style(self, _index=None) -> None:
        self.start_spin.set_style(self.style_combo.currentData())
        self._schedule()

    def _on_order(self, _index=None) -> None:
        self.order_table.setVisible(self.order_combo.currentData() == "custom")
        self._schedule()

    def _on_placement(self, _index=None) -> None:
        manual = self.placement() == "manual"
        self.manual_box.setVisible(manual)
        if manual:
            self._load_offsets()
        self._schedule()

    def _fill_order_table(self) -> None:
        table = self.order_table
        table.setRowCount(self._panel_count)
        for row in range(self._panel_count):
            item = QTableWidgetItem(f"Panel {row + 1}")
            table.setItem(row, 0, item)
            spin = TypedSpinBox()
            spin.setRange(0, 99)
            spin.setSpecialValueText("none")
            spin.setValue(self._custom[row])
            spin.valueChanged.connect(lambda v, r=row: self._on_custom(r, v))
            table.setCellWidget(row, 1, spin)

    def _on_custom(self, row: int, value: int) -> None:
        if 0 <= row < len(self._custom):
            self._custom[row] = int(value)
        self._schedule()

    def _target(self):
        return self.target_combo.currentData()

    def _load_offsets(self, _index=None) -> None:
        target = self._target()
        value = self._positions.get(target)
        if value is None:
            value = self._positions.get("all", (0.0, 4.0))
        self._building = True
        try:
            self.dx_spin.setValue(value[0])
            self.dy_spin.setValue(value[1])
        finally:
            self._building = False

    def _on_offset_edited(self, _value=None) -> None:
        if self._building:
            return
        target = self._target()
        pair = (self.dx_spin.value(), self.dy_spin.value())
        if target == "all":
            self._positions = {"all": pair}
        else:
            self._positions[int(target)] = pair
        self._schedule()
