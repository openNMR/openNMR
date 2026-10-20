"""Plot Style ▸ 2-D ▸ Inset…: a zoomed inset on the 2D spectrum on show,
without going through Compare.

Zoom the plot onto a region (the magnifier on the plot toolbar), press
*Add inset from the current zoom*, and the region is drawn as an inset
(box and connectors optional) in the corner farthest from it. Each inset's
region, corner, size and lowest contour can be edited here. The drawing is
:func:`opennmr.plotting.twod.draw_insets`, the same as Compare's.

Limits are in the units the plot is showing (ppm, or Hz when the axes are
in Hz).
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QListWidget,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..plotting.twod import INSET_CORNERS

_CORNERS = tuple(INSET_CORNERS)
_CENTRES = {"upper right": (0.8, 0.8), "upper left": (0.2, 0.8),
            "lower left": (0.2, 0.2), "lower right": (0.8, 0.2)}


def _spin() -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-1e9, 1e9)
    spin.setDecimals(3)
    spin.setSingleStep(0.1)
    return spin


class InsetDialog(QDialog):
    """Insets of the single 2D contour on ``canvas`` (an ``MplCanvas``)."""

    def __init__(self, canvas, parent=None) -> None:
        super().__init__(parent)
        self.canvas = canvas
        self.setWindowTitle("2D inset")
        self.resize(430, 520)
        self._specs: list[dict] = []
        self._loading = False

        v = QVBoxLayout(self)
        hint = QLabel("Zoom the 2D plot onto a region (magnifier on the plot toolbar), then press "
                      "Add inset from the current zoom. The plot returns to the full view with the "
                      "inset drawn on it.")
        hint.setWordWrap(True)
        v.addWidget(hint)

        row = QHBoxLayout()
        self.add_button = QPushButton("Add inset from the current zoom")
        self.add_button.clicked.connect(lambda: self.add_from_view())
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(lambda: self.remove_selected())
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: self.clear_insets())
        for button in (self.add_button, self.remove_button, self.clear_button):
            row.addWidget(button)
        v.addLayout(row)

        self.inset_list = QListWidget()
        self.inset_list.setMaximumHeight(100)
        self.inset_list.currentRowChanged.connect(self._load_editor)
        v.addWidget(self.inset_list)

        self.editor = QWidget()
        form = QFormLayout(self.editor)
        form.setContentsMargins(0, 0, 0, 0)
        self.f2_from, self.f2_to = _spin(), _spin()
        self.f1_from, self.f1_to = _spin(), _spin()
        form.addRow("F2 from / to:", self._pair(self.f2_from, self.f2_to))
        form.addRow("F1 from / to:", self._pair(self.f1_from, self.f1_to))
        self.corner_combo = QComboBox()
        self.corner_combo.addItems(_CORNERS)
        form.addRow("Corner:", self.corner_combo)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(10, 90)
        self.size_spin.setSuffix(" %")
        self.size_spin.setToolTip("Side of the (square) inset, as a percentage of the plot.")
        form.addRow("Size:", self.size_spin)
        self.floor_spin = QDoubleSpinBox()
        self.floor_spin.setRange(0.0, 90.0)
        self.floor_spin.setDecimals(2)
        self.floor_spin.setSuffix(" %")
        self.floor_spin.setSpecialValueText("Same as plot")
        self.floor_spin.setToolTip(
            "Lowest contour in this inset, as a percentage of the whole spectrum's maximum -- "
            "lower it to bring out weak peaks.")
        form.addRow("Lowest contour:", self.floor_spin)
        self.mark_check = QCheckBox("Box and connectors on the plot")
        form.addRow(self.mark_check)
        v.addWidget(self.editor)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        v.addWidget(self.status_label)
        v.addStretch(1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self.apply)
        for widget in (self.corner_combo,):
            widget.currentIndexChanged.connect(self._on_edit)
        self.mark_check.toggled.connect(self._on_edit)
        for widget in (self.f2_from, self.f2_to, self.f1_from, self.f1_to, self.size_spin,
                       self.floor_spin):
            widget.valueChanged.connect(self._on_edit)
        self.refresh()

    @staticmethod
    def _pair(a, b) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(a)
        row.addWidget(b)
        return widget

    # ---- public ---------------------------------------------------------
    def specs(self) -> list[dict]:
        return [dict(spec) for spec in self._specs]

    def refresh(self) -> None:
        """Re-read the insets from the canvas (it may have been redrawn
        for another spectrum, which drops them)."""
        self._specs = self.canvas.contour_insets()
        self._refresh_list()

    def add_from_view(self) -> dict | None:
        """An inset showing the 2D plot's current view. ``None`` (with a
        message) when no 2D is on show or the plot is not zoomed."""
        region = self.canvas.contour_view_region()
        if region is None:
            self.status_label.setText("Show a 2D spectrum first.")
            return None
        xlims, ylims = region
        full = self._full_limits()
        if full is not None and self._covers(xlims, full[0]) and self._covers(ylims, full[1]):
            self.status_label.setText("Zoom into a region of the 2D plot first (the magnifier on "
                                      "the plot toolbar), then Add inset.")
            return None
        spec = {"xlims": tuple(sorted(xlims, reverse=True)), "ylims": tuple(sorted(ylims)),
                "corner": self._free_corner(xlims, ylims, full), "size": 0.4, "factor": None,
                "mark_inset": True}
        self._specs.append(spec)
        self._refresh_list(select=len(self._specs) - 1)
        if self.apply():
            self.canvas.reset_view()      # back to the full spectrum, inset drawn on it
        return dict(spec)

    def remove_selected(self) -> None:
        row = self.inset_list.currentRow()
        if 0 <= row < len(self._specs):
            del self._specs[row]
            self._refresh_list()
            self.apply()

    def clear_insets(self) -> None:
        if self._specs:
            self._specs.clear()
            self._refresh_list()
            self.apply()

    def apply(self) -> bool:
        """Draw the insets on the canvas. False (with a message) if a
        region has no data in it; the previous insets stay."""
        try:
            self.canvas.set_contour_insets(self._specs)
        except ValueError as exc:
            self.status_label.setText(f"Inset not drawn: {exc}")
            return False
        self.status_label.setText("")
        return True

    # ---- internals ---------------------------------------------------------
    def _full_limits(self):
        try:
            display, _x, _y = self.canvas._contour_for_display()
            xlims, ylims = self.canvas._contour_display_limits(display)
            if ylims is None:
                ylims = (0.0, float(display.data.shape[0]))
            return tuple(xlims), tuple(ylims)
        except Exception:  # noqa: BLE001 -- only for placing the inset
            return None

    @staticmethod
    def _covers(view, full) -> bool:
        span = abs(full[1] - full[0]) or 1.0
        return (abs(abs(view[1] - view[0]) - span) < 1e-6 * span
                and abs(min(view) - min(full)) < 1e-6 * span)

    def _free_corner(self, xlims, ylims, full) -> str:
        """The corner farthest from the region (and not already used), so
        the inset never hides what it shows."""
        used = {spec["corner"] for spec in self._specs}
        if full is None:
            return next((c for c in _CORNERS if c not in used), _CORNERS[0])
        (hx0, hx1), (hy0, hy1) = full
        fx = (sum(xlims) / 2 - hx0) / ((hx1 - hx0) or 1.0)
        fy = (sum(ylims) / 2 - hy0) / ((hy1 - hy0) or 1.0)

        def score(name):
            cx, cy = _CENTRES[name]
            return ((name in used), -((cx - fx) ** 2 + (cy - fy) ** 2))
        return min(_CENTRES, key=score)

    @staticmethod
    def _describe(i: int, spec: dict) -> str:
        (x1, x2), (y1, y2) = spec["xlims"], spec["ylims"]
        return f"Inset {i + 1}: F2 {x1:.3g}–{x2:.3g}, F1 {y1:.3g}–{y2:.3g} ({spec['corner']})"

    def _refresh_list(self, select: int | None = None) -> None:
        self.inset_list.blockSignals(True)
        self.inset_list.clear()
        for i, spec in enumerate(self._specs):
            self.inset_list.addItem(self._describe(i, spec))
        self.inset_list.blockSignals(False)
        has = bool(self._specs)
        self.editor.setEnabled(has)
        self.remove_button.setEnabled(has)
        self.clear_button.setEnabled(has)
        if has:
            row = select if select is not None else min(max(self.inset_list.currentRow(), 0),
                                                         len(self._specs) - 1)
            self.inset_list.setCurrentRow(row)
            self._load_editor(row)

    def _load_editor(self, row: int) -> None:
        if not 0 <= row < len(self._specs):
            return
        spec = self._specs[row]
        self._loading = True
        try:
            self.f2_from.setValue(spec["xlims"][0])
            self.f2_to.setValue(spec["xlims"][1])
            self.f1_from.setValue(spec["ylims"][0])
            self.f1_to.setValue(spec["ylims"][1])
            self.corner_combo.setCurrentText(spec["corner"])
            self.size_spin.setValue(round(spec["size"] * 100))
            self.floor_spin.setValue((spec["factor"] or 0.0) * 100.0)
            self.mark_check.setChecked(bool(spec["mark_inset"]))
        finally:
            self._loading = False

    def _on_edit(self, *_args) -> None:
        row = self.inset_list.currentRow()
        if self._loading or not 0 <= row < len(self._specs):
            return
        spec = self._specs[row]
        spec["xlims"] = (self.f2_from.value(), self.f2_to.value())
        spec["ylims"] = (self.f1_from.value(), self.f1_to.value())
        spec["corner"] = self.corner_combo.currentText()
        spec["size"] = self.size_spin.value() / 100.0
        spec["factor"] = (self.floor_spin.value() / 100.0) or None
        spec["mark_inset"] = self.mark_check.isChecked()
        self.inset_list.item(row).setText(self._describe(row, spec))
        self._debounce.start()
