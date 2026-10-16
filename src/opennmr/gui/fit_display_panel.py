"""Fit plot: how fitted components are drawn -- right-click a fit ▸ Fit Plot:
Components and Colours… (a section of the Fit tile until 2026-10-16).

Show every component, none, or only chosen ones (the others as faint
outlines); fill them or draw lines only; set each one's line and fill
colour, and the data and total-fit colours. The drawing itself is
:func:`opennmr.notebook.fit_display.draw_fit_components`, so a notebook
figure given the same style (``canvas.fit_component_style()``) matches.

One style for every fit view (the Fit view and the Spectral Fit view),
kept across refits: a colour chosen for a site stays with that site's
label.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from matplotlib.colors import to_hex

from ..notebook import fit_display

_SHOW_ITEMS = (("Automatic (every component of a multi-site fit)", "auto"),
               ("All components", "all"), ("None (data and fit only)", "none"),
               ("Highlight the ticked components only", "highlight"))


def _swatch(colour) -> QIcon:
    pixmap = QPixmap(14, 14)
    pixmap.fill(QColor(to_hex(colour)))
    return QIcon(pixmap)


class FitDisplayPanel(QWidget):
    """``canvases``: the MplCanvas instances whose fit views this styles.
    Or, with no canvases, a style of its own (``style``) reported to
    ``on_change(style)`` -- one kept fit's display in Compare; its
    components come from :meth:`set_components`."""

    def __init__(self, canvases, parent=None, style=None, on_change=None, intro=None) -> None:
        super().__init__(parent)
        self._canvases = list(canvases)
        self._on_change = on_change
        self._labels: list[str] = []
        self._defaults: dict = {}
        self._style = (fit_display.component_style(style) if style is not None else
                       self._canvases[0].fit_component_style() if self._canvases else
                       fit_display.component_style())
        for canvas in self._canvases:
            canvas.fitComponentsShown.connect(
                lambda labels, c=canvas: self._on_components_shown(c, labels))
            canvas.styleEditedFromMenu.connect(lambda c=canvas: self._adopt_from(c))

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(intro or
                       "How fitted components are drawn on the Fit and Spectral Fit views, and "
                       "in exported figures. Quick choices are also on the plot's right-click "
                       "menu ▸ Components.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        form = QFormLayout()
        self.show_combo = QComboBox()
        for text, data in _SHOW_ITEMS:
            self.show_combo.addItem(text, data)
        self.show_combo.setToolTip(
            "Automatic: every component of a fit with several; a single-site fit shows only "
            "its total. Highlight: only the components ticked below are drawn in colour, the "
            "rest as faint outlines.")
        self.show_combo.currentIndexChanged.connect(lambda _i: self._push())
        form.addRow("Components:", self.show_combo)

        self.fill_check = QCheckBox("Fill components")
        self.fill_check.setToolTip("Shade each component; off draws each as a line only.")
        self.fill_check.toggled.connect(lambda _on: self._push())
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.05, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setToolTip("Fill opacity.")
        self.alpha_spin.valueChanged.connect(lambda _v: self._push())
        fill_row = QHBoxLayout()
        fill_row.addWidget(self.fill_check)
        fill_row.addWidget(QLabel("opacity"))
        fill_row.addWidget(self.alpha_spin)
        fill_row.addStretch(1)
        form.addRow("", fill_row)

        self.baseline_check = QCheckBox("Show baseline")
        self.baseline_check.toggled.connect(lambda _on: self._push())
        self.on_baseline_check = QCheckBox("Components sit on the baseline")
        self.on_baseline_check.setToolTip(
            "Draw each component on top of the fitted baseline (what it adds to the total) "
            "rather than from zero.")
        self.on_baseline_check.toggled.connect(lambda _on: self._push())
        base_row = QHBoxLayout()
        base_row.addWidget(self.baseline_check)
        base_row.addWidget(self.on_baseline_check)
        base_row.addStretch(1)
        form.addRow("", base_row)

        self.data_colour_button = QPushButton("Data")
        self.data_colour_button.clicked.connect(lambda: self._pick_line("data_colour"))
        self.fit_colour_button = QPushButton("Total fit")
        self.fit_colour_button.clicked.connect(lambda: self._pick_line("fit_colour"))
        colour_row = QHBoxLayout()
        colour_row.addWidget(self.data_colour_button)
        colour_row.addWidget(self.fit_colour_button)
        colour_row.addStretch(1)
        form.addRow("Line colours:", colour_row)
        v.addLayout(form)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Highlight", "Component", "Line", "Fill"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c in (0, 2, 3):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setToolTip("The components of the fit on show. Tick to highlight (with "
                              "Components set to Highlight); click a swatch to change a colour.")
        v.addWidget(self.table)
        self.empty_label = QLabel("<i>Run a fit to list its components here.</i>")
        v.addWidget(self.empty_label)

        self.reset_button = QPushButton("Reset colours")
        self.reset_button.setToolTip("Back to the default palette and black data / red fit.")
        self.reset_button.clicked.connect(self.reset_colours)
        v.addWidget(self.reset_button)
        self._load_controls()
        self._rebuild_rows()

    # ------------------------------------------------------------------

    def style(self) -> dict:
        return fit_display.component_style(self._style)

    def _load_controls(self) -> None:
        s = self._style
        for widget, setter, value in (
                (self.show_combo, "setCurrentIndex", self.show_combo.findData(s["show"])),
                (self.fill_check, "setChecked", bool(s["fill"])),
                (self.alpha_spin, "setValue", float(s["fill_alpha"])),
                (self.baseline_check, "setChecked", bool(s["show_baseline"])),
                (self.on_baseline_check, "setChecked", bool(s["on_baseline"]))):
            blocked = widget.blockSignals(True)
            getattr(widget, setter)(value)
            widget.blockSignals(blocked)
        self.alpha_spin.setEnabled(bool(s["fill"]))
        self.data_colour_button.setIcon(_swatch(s["data_colour"]))
        self.fit_colour_button.setIcon(_swatch(s["fit_colour"]))

    def _push(self) -> None:
        self._style.update(show=self.show_combo.currentData(), fill=self.fill_check.isChecked(),
                           fill_alpha=self.alpha_spin.value(),
                           show_baseline=self.baseline_check.isChecked(),
                           on_baseline=self.on_baseline_check.isChecked())
        self._apply()

    def _apply(self) -> None:
        self._style = fit_display.component_style(self._style)
        self.alpha_spin.setEnabled(bool(self._style["fill"]))
        for canvas in self._canvases:
            canvas.set_fit_component_style(self._style)
        if self._on_change is not None:
            self._on_change(fit_display.component_style(self._style))
        self._rebuild_rows()

    def set_components(self, labels, curves=None) -> None:
        """The components to list (and their default colours) when this
        panel styles something other than a canvas's fit view."""
        self._labels = [str(label) for label in labels]
        curves = list(curves or [])
        self._defaults = (fit_display.default_component_colours(curves, self._labels)
                          if len(curves) == len(self._labels) else {})
        self._rebuild_rows()

    def _adopt_from(self, canvas) -> None:
        """A change made from one plot's right-click menu: take it, and
        give it to the other fit views too."""
        self._style = canvas.fit_component_style()
        self._load_controls()
        for other in self._canvases:
            if other is not canvas:
                other.set_fit_component_style(self._style)
        self._rebuild_rows()

    def _on_components_shown(self, canvas, labels) -> None:
        self._labels = list(labels)
        args = canvas._slice_fit_args or {}
        comps = args.get("components") or []
        curves = [c for c in comps[:-1]]
        self._defaults = (fit_display.default_component_colours(curves, self._labels)
                          if len(curves) == len(self._labels) else {})
        self._rebuild_rows()

    def _colour(self, key: str, label: str):
        return self._style[key].get(label, self._defaults.get(label, (0.5, 0.5, 0.5)))

    def _rebuild_rows(self) -> None:
        self.table.setRowCount(len(self._labels))
        self.empty_label.setVisible(not self._labels)
        self.table.setVisible(bool(self._labels))
        highlight = set(self._style["highlight"])
        for row, label in enumerate(self._labels):
            check = QCheckBox()
            check.setChecked(label in highlight)
            check.setToolTip(f"Highlight {label}.")
            check.setAccessibleName(f"Highlight {label}")
            check.toggled.connect(lambda on, lab=label: self.set_highlighted(lab, on))
            self.table.setCellWidget(row, 0, check)
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, item)
            for column, key in ((2, "line_colours"), (3, "fill_colours")):
                button = QPushButton()
                button.setIcon(_swatch(self._colour(key, label)))
                button.setToolTip(f"{'Line' if column == 2 else 'Fill'} colour of {label}.")
                button.setAccessibleName(f"{'Line' if column == 2 else 'Fill'} colour of {label}")
                button.clicked.connect(lambda _c=False, k=key, lab=label: self._pick(k, lab))
                if column == 3:
                    button.setEnabled(bool(self._style["fill"]))
                self.table.setCellWidget(row, column, button)
        self.table.setFixedHeight(self.table.horizontalHeader().height() + 4
                                  + sum(self.table.rowHeight(r) for r in range(min(len(self._labels), 6))))

    # ------------------------------------------------------------------

    def set_highlighted(self, label: str, on: bool) -> None:
        wanted = [h for h in self._style["highlight"] if h != label]
        if on:
            wanted.append(label)
        self._style["highlight"] = wanted
        if on and self._style["show"] != "highlight":
            # Ticking a component is asking to highlight it.
            self._style["show"] = "highlight"
            self._load_controls()
        self._apply()

    def set_component_colour(self, key: str, label: str, colour) -> None:
        """``key`` 'line_colours' or 'fill_colours'."""
        self._style[key] = {**self._style[key], label: to_hex(colour)}
        self._apply()

    def _pick(self, key: str, label: str) -> None:
        colour = QColorDialog.getColor(QColor(to_hex(self._colour(key, label))), self,
                                       f"Colour of {label}")
        if colour.isValid():
            self.set_component_colour(key, label, colour.name())

    def _pick_line(self, key: str) -> None:
        colour = QColorDialog.getColor(QColor(to_hex(self._style[key])), self, "Colour")
        if colour.isValid():
            self._style[key] = colour.name()
            self._load_controls()
            self._apply()

    def reset_colours(self) -> None:
        default = fit_display.DEFAULT_COMPONENT_STYLE
        self._style.update(line_colours={}, fill_colours={}, data_colour=default["data_colour"],
                           fit_colour=default["fit_colour"])
        self._load_controls()
        self._apply()
