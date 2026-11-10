"""Insets on the plot on show: Plot Style ▸ Insets (right-click the plot ▸
Inset… opens Plot Style there), or right-click the toolbar's Inset button.

:class:`InsetPanel` is the controls themselves, shown as a collapsible
section of Plot Style (explicit user request, 2026-11-10: the separate
window was cut off and hard to see everything in; one place for a plot's
appearance is less confusing). :class:`InsetDialog` puts the same panel in
a window of its own, scrolling, for a notebook-style or scripted use.

An inset is a magnified copy of part of the plot -- a 1D spectrum, a stack
of spectra, a 2D contour, a fitted curve or a profile. Draw one with the
toolbar's Inset button (drag a box over the region), or zoom the plot onto
a region and press *Add from the current zoom*. Each inset's region,
position, axes, typeface and connecting lines are edited here and applied
as you change them.

Limits are in the units the plot is showing (ppm, Hz, seconds ...). The
drawing is :mod:`opennmr.plotting.inset` (a 2D contour:
:func:`opennmr.plotting.twod.draw_insets`).

On a figure of several plots (a Compare stack) the list holds **every**
plot's insets, each named by the spectrum it magnifies, and *Placed in*
puts an inset in another plot's area -- so an inset of one spectrum can
sit beside an inset of another (explicit user request, 2026-11-10: an
inset made on one spectrum could not be removed once another spectrum
had one).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QFont
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFontComboBox, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .widgets.typed_spinbox import TypedSpinBox
from ..plotting import inset as inset_module

_CORNERS = tuple(inset_module.CORNER_NAMES) + ("custom",)
_CENTRES = {"upper right": (0.8, 0.8), "upper left": (0.2, 0.8),
            "lower left": (0.2, 0.2), "lower right": (0.8, 0.2)}
_LINE_STYLES = (("Solid", "-"), ("Dashed", "--"), ("Dotted", ":"), ("Dash-dot", "-."))
SAME_AS_PLOT = "Same as the plot"


class _FloatEdit(QLineEdit):
    """A number box for a region limit: any size, scientific notation
    allowed. A spin box clamped the limits to ±1e9 -- an intensity axis of
    1.6e10 (a real Bruker spectrum, 2026-11-10) was then rewritten as ±1e9
    on the first edit, and the inset showed a flat line."""

    valueChanged = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self._value = 0.0
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.setValidator(validator)
        self.editingFinished.connect(self._on_edited)
        self._show()

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:  # noqa: N802 -- matches the spin boxes it replaces
        value = float(value)
        changed = value != self._value
        self._value = value
        self._show()
        if changed:
            self.valueChanged.emit(value)

    def _show(self) -> None:
        self.setText(f"{self._value:.6g}")

    def _on_edited(self) -> None:
        try:
            value = float(self.text().replace("−", "-"))
        except ValueError:
            self._show()
            return
        self.setValue(value)


def _spin() -> _FloatEdit:
    return _FloatEdit()


class _ColourButton(QPushButton):
    """A button showing a colour; clicking opens the colour chooser."""

    def __init__(self, colour: str, on_change) -> None:
        super().__init__()
        self._on_change = on_change
        self.setFixedWidth(60)
        self.set_colour(colour)
        self.clicked.connect(self._choose)

    def colour(self) -> str:
        return self._colour

    def set_colour(self, colour: str) -> None:
        try:
            parsed = QColor(colour) if colour.startswith("#") else None
            if parsed is None:
                import matplotlib.colors as mcolors
                parsed = QColor(mcolors.to_hex(colour))
        except (ValueError, TypeError):
            parsed = QColor("#555555")
        self._colour = parsed.name()
        self.setStyleSheet(f"background-color: {self._colour};")

    def _choose(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._colour), self)
        if chosen.isValid():
            self.set_colour(chosen.name())
            self._on_change()


class InsetPanel(QWidget):
    """The inset controls for the plot on ``canvas`` (an ``MplCanvas``; may
    be None and set later with :meth:`set_canvas`)."""

    def __init__(self, canvas=None, parent=None) -> None:
        super().__init__(parent)
        self.canvas = None
        self._specs: list[dict] = []
        self._loading = False
        self._rows: list[str] = []

        v = QVBoxLayout(self)
        hint = QLabel(
            "Press Draw an inset, then drag a box over the region to magnify – on a spectrum, a "
            "2D contour, a curve or a profile. Or zoom the plot onto a region and press Add from "
            "the current zoom. Everything below is applied as you change it.")
        hint.setWordWrap(True)
        v.addWidget(hint)

        row = QGridLayout()
        self.draw_button = QPushButton("Draw an inset")
        self.draw_button.setToolTip("Switch on the Inset tool: drag a box over the part of the "
                                    "plot to magnify.")
        self.draw_button.clicked.connect(lambda: self._start_drawing())
        self.add_button = QPushButton("Add from the current zoom")
        self.add_button.clicked.connect(lambda: self.add_from_view())
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(lambda: self.remove_selected())
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: self.clear_insets())
        # Two by two: four in a row did not fit a narrow window.
        for k, button in enumerate((self.draw_button, self.add_button,
                                    self.remove_button, self.clear_button)):
            row.addWidget(button, k // 2, k % 2)
        v.addLayout(row)

        self.inset_list = QListWidget()
        self.inset_list.setMaximumHeight(130)
        # Wrapped, not cut off at the edge (2026-11-10 screenshot): a long
        # spectrum name and its region read in full, with no sideways scroll.
        self.inset_list.setWordWrap(True)
        self.inset_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inset_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.inset_list.setToolTip(
            "Every inset on this figure – on a Compare stack, every spectrum's, each "
            "named by the spectrum it magnifies.")
        self.inset_list.currentRowChanged.connect(self._load_editor)
        v.addWidget(self.inset_list)

        self.editor = QWidget()
        editor_layout = QVBoxLayout(self.editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)

        # -- what is magnified
        region_box = QGroupBox("Region")
        region = QFormLayout(region_box)
        self.x_from, self.x_to, self.y_from, self.y_to = _spin(), _spin(), _spin(), _spin()
        region.addRow("Horizontal, from / to:", self._pair(self.x_from, self.x_to))
        region.addRow("Vertical, from / to:", self._pair(self.y_from, self.y_to))
        self.auto_y_check = QCheckBox("Fit the vertical range to the data")
        self.auto_y_check.setToolTip(
            "Ignore the vertical limits above and scale the inset to the signal inside the "
            "horizontal range – the usual choice for magnifying a small peak.")
        region.addRow(self.auto_y_check)
        self.floor_spin = QDoubleSpinBox()
        self.floor_spin.setRange(0.0, 90.0)
        self.floor_spin.setDecimals(2)
        self.floor_spin.setSuffix(" %")
        self.floor_spin.setSpecialValueText("Same as plot")
        self.floor_spin.setToolTip(
            "2D only: the lowest contour in this inset, as a percentage of the whole "
            "spectrum's maximum – lower it to bring out weak peaks.")
        self.floor_label = QLabel("Lowest contour:")
        region.addRow(self.floor_label, self.floor_spin)
        editor_layout.addWidget(region_box)

        # -- where it sits
        position_box = QGroupBox("Position and size")
        position = QFormLayout(position_box)
        # Which plot's area the inset sits in (a Compare stack only).
        self.host_combo = QComboBox()
        self.host_combo.setToolTip(
            "Which spectrum's area the inset sits in. Its own (the default), or another "
            "spectrum's – to put insets of two spectra side by side. Position and size "
            "below are then fractions of that spectrum's area; the lines still run to the "
            "region it magnifies.")
        self.host_label = QLabel("Placed in:")
        position.addRow(self.host_label, self.host_combo)
        self.corner_combo = QComboBox()
        self.corner_combo.addItems(_CORNERS)
        self.corner_combo.setToolTip("A corner of the plot, or Custom to place it by hand.")
        position.addRow("Position:", self.corner_combo)
        self.pos_x_spin = TypedSpinBox()
        self.pos_x_spin.setRange(0, 95)
        self.pos_x_spin.setSuffix(" %")
        self.pos_y_spin = TypedSpinBox()
        self.pos_y_spin.setRange(0, 95)
        self.pos_y_spin.setSuffix(" %")
        self.custom_label = QLabel("Custom, left / bottom:")
        self.custom_pair = self._pair(self.pos_x_spin, self.pos_y_spin)
        position.addRow(self.custom_label, self.custom_pair)
        self.width_spin = TypedSpinBox()
        self.width_spin.setRange(5, 95)
        self.width_spin.setSuffix(" %")
        self.height_spin = TypedSpinBox()
        self.height_spin.setRange(5, 95)
        self.height_spin.setSuffix(" %")
        position.addRow("Width / height:", self._pair(self.width_spin, self.height_spin))
        editor_layout.addWidget(position_box)

        # -- axes and frame
        axes_box = QGroupBox("Axes and frame")
        axes = QFormLayout(axes_box)
        self.show_axes_check = QCheckBox("Show the axes (ticks and numbers)")
        self.show_axes_check.setToolTip(
            "Off shows only the peaks – no ticks, no numbers. A spectrum, curve or profile "
            "inset shows the horizontal axis only (its vertical values are never shown); a 2D "
            "contour inset shows both.")
        axes.addRow(self.show_axes_check)
        # Axis labels, off unless asked for (explicit user request, 2026-10-24).
        self.show_xlabel_check = QCheckBox("x-axis label")
        self.xlabel_edit = QLineEdit()
        self.xlabel_edit.setPlaceholderText("Same as the plot")
        self.xlabel_edit.setToolTip("The text under the inset; empty copies the plot's own label.")
        axes.addRow(self.show_xlabel_check, self.xlabel_edit)
        self.show_ylabel_check = QCheckBox("y-axis label")
        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.setPlaceholderText("Same as the plot")
        self.ylabel_edit.setToolTip("The text beside the inset; empty copies the plot's own label.")
        axes.addRow(self.show_ylabel_check, self.ylabel_edit)
        self.frame_check = QCheckBox("Frame around the inset")
        axes.addRow(self.frame_check)
        self.frame_colour = _ColourButton("0.2", self._on_edit)
        self.frame_width = QDoubleSpinBox()
        self.frame_width.setRange(0.1, 5.0)
        self.frame_width.setSingleStep(0.1)
        axes.addRow("Frame colour / width:", self._pair(self.frame_colour, self.frame_width))
        editor_layout.addWidget(axes_box)

        # -- text
        text_box = QGroupBox("Typeface")
        text = QFormLayout(text_box)
        self.font_combo = QFontComboBox()
        self.same_font_check = QCheckBox(SAME_AS_PLOT)
        self.same_font_check.setChecked(True)
        text.addRow("Typeface:", self._pair(self.same_font_check, self.font_combo))
        self.font_size_spin = QDoubleSpinBox()
        self.font_size_spin.setRange(0.0, 48.0)
        self.font_size_spin.setDecimals(1)
        self.font_size_spin.setSpecialValueText("Automatic")
        self.font_size_spin.setSuffix(" pt")
        text.addRow("Size:", self.font_size_spin)
        self.bold_check = QCheckBox("Bold")
        self.italic_check = QCheckBox("Italic")
        text.addRow(self._pair(self.bold_check, self.italic_check))
        editor_layout.addWidget(text_box)

        # -- connectors
        connector_box = QGroupBox("Lines to the region")
        connector = QFormLayout(connector_box)
        self.connectors_check = QCheckBox("Box on the plot and lines to the inset")
        connector.addRow(self.connectors_check)
        self.connector_colour = _ColourButton("0.35", self._on_edit)
        self.connector_width = QDoubleSpinBox()
        self.connector_width.setRange(0.1, 5.0)
        self.connector_width.setSingleStep(0.1)
        connector.addRow("Colour / width:", self._pair(self.connector_colour, self.connector_width))
        self.connector_style_combo = QComboBox()
        for label, value in _LINE_STYLES:
            self.connector_style_combo.addItem(label, value)
        connector.addRow("Style:", self.connector_style_combo)
        editor_layout.addWidget(connector_box)
        v.addWidget(self.editor)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        v.addWidget(self.status_label)
        v.addStretch(1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self.apply)
        for widget in (self.corner_combo, self.connector_style_combo, self.host_combo):
            widget.currentIndexChanged.connect(self._on_edit)
        for check in (self.auto_y_check, self.show_axes_check, self.frame_check, self.bold_check,
                      self.italic_check, self.connectors_check, self.same_font_check,
                      self.show_xlabel_check, self.show_ylabel_check):
            check.toggled.connect(self._on_edit)
        self.xlabel_edit.textEdited.connect(self._on_edit)
        self.ylabel_edit.textEdited.connect(self._on_edit)
        self.font_combo.currentFontChanged.connect(self._on_edit)
        for widget in (self.x_from, self.x_to, self.y_from, self.y_to, self.floor_spin,
                       self.pos_x_spin, self.pos_y_spin, self.width_spin, self.height_spin,
                       self.frame_width, self.font_size_spin, self.connector_width):
            widget.valueChanged.connect(self._on_edit)
        self.set_canvas(canvas)

    def set_canvas(self, canvas) -> None:
        """Show (and edit) the insets of ``canvas`` -- Plot Style calls this
        whenever the plot it styles changes."""
        if canvas is self.canvas:
            self.refresh()
            return
        if self.canvas is not None:
            try:
                self.canvas.insetsChanged.disconnect(self.refresh)
            except (RuntimeError, TypeError):
                pass
        self.canvas = canvas
        if canvas is not None:
            canvas.insetsChanged.connect(self.refresh)
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
        for another spectrum) -- every plot's, each with its ``row``."""
        if self.canvas is None:
            self._specs, self._rows = [], []
        else:
            self._specs = self.canvas.all_inset_specs()
            self._rows = self.canvas.inset_rows()
        self._refresh_list()

    def _start_drawing(self) -> None:
        if self.canvas is None:
            return
        self.canvas.set_inset_mode(True)
        self.status_label.setText("Drag a box over the region on the plot.")

    def add_from_view(self) -> dict | None:
        """An inset showing the plot's current view. ``None`` (with a
        message) when there is no plot or it is not zoomed in."""
        region = self.canvas.inset_view_region() if self.canvas is not None else None
        if region is None:
            self.status_label.setText("There is no plot to put an inset on.")
            return None
        xlim, ylim = region
        if self._is_unzoomed(xlim, ylim):
            self.status_label.setText(
                "Zoom into a region of the plot first (the magnifier on the plot toolbar), then "
                "Add from the current zoom – or press Draw an inset and drag a box.")
            return None
        try:
            spec = self.canvas.add_inset_region(xlim, ylim)
        except ValueError as exc:
            self.status_label.setText(f"Inset not drawn: {exc}")
            return None
        self.refresh()
        self._select_last_added()
        self.canvas.reset_view()          # back to the full view, inset drawn on it
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
        region has nothing in it; the previous insets stay."""
        if self.canvas is None:
            return False
        try:
            self.canvas.set_all_inset_specs(self._specs)
        except ValueError as exc:
            self.status_label.setText(f"Inset not drawn: {exc}")
            return False
        self.status_label.setText("")
        return True

    # ---- internals ---------------------------------------------------------
    def _is_contour(self) -> bool:
        return self.canvas is not None and self.canvas.contour_view_region() is not None

    def _is_unzoomed(self, xlim, ylim) -> bool:
        """Whether the view still shows the whole plot, so there is nothing
        to magnify."""
        full = self._full_limits()
        if full is None:
            return False
        if self._is_contour():
            return self._covers(xlim, full[0]) and self._covers(ylim, full[1])
        span = abs(full[0][1] - full[0][0]) or 1.0
        return abs(xlim[1] - xlim[0]) >= 0.9 * span

    def _full_limits(self):
        try:
            if self._is_contour():
                display, _x, _y = self.canvas._contour_for_display()
                xlims, ylims = self.canvas._contour_display_limits(display)
                if ylims is None:
                    ylims = (0.0, float(display.data.shape[0]))
                return tuple(xlims), tuple(ylims)
            ax, _index = self.canvas._inset_target_axes()
            data = ax.dataLim
            return (data.x0, data.x1), (data.y0, data.y1)
        except Exception:  # noqa: BLE001 -- only for deciding whether the plot is zoomed
            return None

    @staticmethod
    def _covers(view, full) -> bool:
        span = abs(full[1] - full[0]) or 1.0
        return (abs(abs(view[1] - view[0]) - span) < 1e-3 * span
                and abs(min(view) - min(full)) < 1e-3 * span)

    def _select_last_added(self) -> None:
        """Select the inset just added: the last one of the plot it was
        drawn on (the list is ordered by plot)."""
        _ax, row = self.canvas._inset_target_axes() if self.canvas.contour_view_region() is None \
            else (None, 0)
        indices = [i for i, spec in enumerate(self._specs) if spec.get("row", 0) == row]
        self.inset_list.setCurrentRow(indices[-1] if indices else len(self._specs) - 1)

    def _row_name(self, row) -> str:
        rows = getattr(self, "_rows", None) or []
        if row is None or not 0 <= int(row) < len(rows):
            return ""
        return rows[int(row)]

    def _describe(self, i: int, spec: dict) -> str:
        """The list's line: the inset, the plot it magnifies (on a stack)
        and its horizontal range; the rest is in the editor below."""
        (x1, x2), _y = spec["xlim"], spec["ylim"]
        text = f"Inset {i + 1}: {x1:.4g} to {x2:.4g}"
        if len(getattr(self, "_rows", None) or []) > 1:
            text = f"{self._row_name(spec.get('row', 0))} – " + text
            host = spec.get("host")
            if host is not None and host != spec.get("row", 0):
                text += f", placed in {self._row_name(host)}"
        return text

    def _tooltip(self, i: int, spec: dict) -> str:
        (x1, x2), (y1, y2) = spec["xlim"], spec["ylim"]
        return (f"{self._describe(i, spec)}\nHorizontal {x1:.6g} to {x2:.6g}, vertical "
                f"{y1:.6g} to {y2:.6g}; {spec.get('corner', '')}")

    def _fill_host_combo(self, spec: dict) -> None:
        """The plots an inset can sit in: its own first, then every other."""
        rows = getattr(self, "_rows", None) or []
        several = len(rows) > 1
        self.host_label.setVisible(several)
        self.host_combo.setVisible(several)
        self.host_combo.blockSignals(True)
        try:
            self.host_combo.clear()
            own = spec.get("row", 0)
            self.host_combo.addItem(f"Its own ({self._row_name(own)})", None)
            for index, name in enumerate(rows):
                if index != own:
                    self.host_combo.addItem(name, index)
            host = spec.get("host")
            found = self.host_combo.findData(host) if host is not None and host != own else 0
            self.host_combo.setCurrentIndex(max(found, 0))
        finally:
            self.host_combo.blockSignals(False)

    def _refresh_list(self, select: int | None = None) -> None:
        self.inset_list.blockSignals(True)
        self.inset_list.clear()
        for i, spec in enumerate(self._specs):
            item = QListWidgetItem(self._describe(i, spec))
            item.setToolTip(self._tooltip(i, spec))
            self.inset_list.addItem(item)
        self.inset_list.blockSignals(False)
        has = bool(self._specs)
        self.editor.setEnabled(has)
        self.remove_button.setEnabled(has)
        self.clear_button.setEnabled(has)
        several = len(getattr(self, "_rows", None) or []) > 1
        self.host_label.setVisible(several)
        self.host_combo.setVisible(several)
        contour = self._is_contour()
        self.floor_spin.setVisible(contour)
        self.floor_label.setVisible(contour)
        self.auto_y_check.setVisible(not contour)
        if has:
            row = select if select is not None else min(max(self.inset_list.currentRow(), 0),
                                                         len(self._specs) - 1)
            self.inset_list.setCurrentRow(row)
            self._load_editor(row)

    def _load_editor(self, row: int) -> None:
        if not 0 <= row < len(self._specs):
            return
        spec = {**inset_module.INSET_DEFAULTS, **self._specs[row]}
        self._loading = True
        try:
            self._fill_host_combo(spec)
            self.x_from.setValue(spec["xlim"][0])
            self.x_to.setValue(spec["xlim"][1])
            self.y_from.setValue(spec["ylim"][0])
            self.y_to.setValue(spec["ylim"][1])
            self.auto_y_check.setChecked(bool(spec["auto_ylim"]))
            self.floor_spin.setValue((spec.get("factor") or 0.0) * 100.0)
            self.corner_combo.setCurrentText(spec["corner"])
            self.pos_x_spin.setValue(round(float(spec["x"]) * 100))
            self.pos_y_spin.setValue(round(float(spec["y"]) * 100))
            self.width_spin.setValue(round(float(spec["width"]) * 100))
            self.height_spin.setValue(round(float(spec["height"]) * 100))
            self.show_axes_check.setChecked(bool(spec["show_axes"]))
            self.show_xlabel_check.setChecked(bool(spec["show_xlabel"]))
            self.show_ylabel_check.setChecked(bool(spec["show_ylabel"]))
            self.xlabel_edit.setText(spec["xlabel"] or "")
            self.ylabel_edit.setText(spec["ylabel"] or "")
            self.frame_check.setChecked(bool(spec["frame"]))
            self.frame_colour.set_colour(str(spec["frame_colour"]))
            self.frame_width.setValue(float(spec["frame_width"]))
            self.same_font_check.setChecked(not spec["family"])
            if spec["family"]:
                self.font_combo.setCurrentFont(QFont(spec["family"]))
            self.font_size_spin.setValue(float(spec["fontsize"] or 0.0))
            self.bold_check.setChecked(bool(spec["bold"]))
            self.italic_check.setChecked(bool(spec["italic"]))
            self.connectors_check.setChecked(bool(spec["connectors"]))
            self.connector_colour.set_colour(str(spec["connector_colour"]))
            self.connector_width.setValue(float(spec["connector_width"]))
            index = self.connector_style_combo.findData(spec["connector_style"])
            self.connector_style_combo.setCurrentIndex(max(index, 0))
        finally:
            self._loading = False
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        custom = self.corner_combo.currentText() == "custom"
        self.custom_label.setEnabled(custom)
        self.custom_pair.setEnabled(custom)
        self.xlabel_edit.setEnabled(self.show_xlabel_check.isChecked())
        self.ylabel_edit.setEnabled(self.show_ylabel_check.isChecked())
        # Text is drawn when the axes or either label is on.
        axes_on = (self.show_axes_check.isChecked() or self.show_xlabel_check.isChecked()
                   or self.show_ylabel_check.isChecked())
        for widget in (self.font_combo, self.same_font_check, self.font_size_spin,
                       self.bold_check, self.italic_check):
            widget.setEnabled(axes_on)
        self.font_combo.setEnabled(axes_on and not self.same_font_check.isChecked())
        lines_on = self.connectors_check.isChecked()
        for widget in (self.connector_colour, self.connector_width, self.connector_style_combo):
            widget.setEnabled(lines_on)

    def _on_edit(self, *_args) -> None:
        row = self.inset_list.currentRow()
        if self._loading or not 0 <= row < len(self._specs):
            return
        spec = self._specs[row]
        spec["xlim"] = (self.x_from.value(), self.x_to.value())
        spec["ylim"] = (self.y_from.value(), self.y_to.value())
        spec["auto_ylim"] = self.auto_y_check.isChecked()
        spec["factor"] = (self.floor_spin.value() / 100.0) or None
        spec["corner"] = self.corner_combo.currentText()
        spec["x"] = self.pos_x_spin.value() / 100.0
        spec["y"] = self.pos_y_spin.value() / 100.0
        spec["width"] = self.width_spin.value() / 100.0
        spec["height"] = self.height_spin.value() / 100.0
        spec["show_axes"] = self.show_axes_check.isChecked()
        spec["show_xlabel"] = self.show_xlabel_check.isChecked()
        spec["show_ylabel"] = self.show_ylabel_check.isChecked()
        spec["xlabel"] = self.xlabel_edit.text() or None
        spec["ylabel"] = self.ylabel_edit.text() or None
        spec["frame"] = self.frame_check.isChecked()
        spec["frame_colour"] = self.frame_colour.colour()
        spec["frame_width"] = self.frame_width.value()
        spec["family"] = (None if self.same_font_check.isChecked()
                          else self.font_combo.currentFont().family())
        spec["fontsize"] = self.font_size_spin.value() or None
        spec["bold"] = self.bold_check.isChecked()
        spec["italic"] = self.italic_check.isChecked()
        spec["connectors"] = self.connectors_check.isChecked()
        spec["connector_colour"] = self.connector_colour.colour()
        spec["connector_width"] = self.connector_width.value()
        spec["connector_style"] = self.connector_style_combo.currentData()
        if self.host_combo.isVisible() or self.host_combo.count() > 1:
            spec["host"] = self.host_combo.currentData()
        self.inset_list.item(row).setText(self._describe(row, spec))
        self._sync_enabled()
        self._debounce.start()


class InsetDialog(QDialog):
    """:class:`InsetPanel` in a window of its own, scrolling (so nothing is
    cut off however short the screen). Attribute access is passed to the
    panel, so ``dialog.inset_list``, ``dialog.apply()``... work as before."""

    def __init__(self, canvas, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Insets")
        self.panel = InsetPanel(canvas, self)
        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.resize(480, 640)

    def __getattr__(self, name):
        # Only reached for names QDialog does not have.
        panel = self.__dict__.get("panel")
        if panel is None:
            raise AttributeError(name)
        return getattr(panel, name)
