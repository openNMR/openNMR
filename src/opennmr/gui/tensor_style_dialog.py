"""Right-click ▸ Tensor Plot Style… on Simulate's tensor pictures.

Explicit user request (2026-10-01): "a specific tensor plot style" -- the
colour map (Plot Style's text, ticks and axis-label settings apply as
for every plot), the colour bar, principal axes, the ellipsoid's colours
and opacity, and the view. Edits apply live to the picture on show; they
are :data:`opennmr.notebook.tensor_views.TENSOR_STYLE_DEFAULTS`' keys,
so a notebook passes the same dict as ``tensor_style=``.

2026-10-27 (explicit user requests): the 3D box's text -- tick labels
and axis labels on or off, the whole box on or off, and their padding --
and, for the structure picture, the atoms and bonds (size, sizing,
opacity, outline, depth cueing, element colours, bond width and
colour). Three tabs: Tensors, Axes & Labels, Atoms & Bonds.

``canvas`` is the MplCanvas showing the picture; ``save_default(style)``
(optional) keeps the settings as the default for tensor pictures.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

from ..notebook import dft_structure as ds
from ..notebook import tensor_views as tv
from ..plotting import accessibility

#: Colour scales offered for a surface: (label, matplotlib name, diverging).
#: "" is each picture's own default (RdBu_r for a surface of both signs,
#: viridis otherwise).
TENSOR_COLORMAPS = (
    ("Picture's own (RdBu / viridis)", "", None),
    ("Viridis", "viridis", False), ("Cividis", "cividis", False),
    ("Plasma", "plasma", False), ("Inferno", "inferno", False), ("Magma", "magma", False),
    ("Blue–red (diverging)", "RdBu_r", True), ("Purple–orange (diverging)", "PuOr_r", True),
    ("Purple–green (diverging)", "PRGn", True), ("Brown–teal (diverging)", "BrBG", True),
    ("Cool–warm (diverging)", "coolwarm", True), ("Spectral", "Spectral_r", True),
    ("Greys", "Greys", False),
)


def colourblind_friendly_maps() -> set:
    """The offered maps that pass the package's measured colour-blind
    check (for at least three distinguishable levels)."""
    names = {name for name, _n in accessibility.friendly_palettes("colourblind", 3)}
    friendly = set()
    for _label, cmap, _div in TENSOR_COLORMAPS:
        if not cmap:
            continue
        base = cmap[:-2] if cmap.endswith("_r") else cmap
        if base in names or cmap in names:
            friendly.add(cmap)
    return friendly


class TensorStyleDialog(QDialog):
    def __init__(self, canvas, save_default=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tensor Plot Style")
        self._canvas = canvas
        self._save_default = save_default
        self._loading = False
        v = QVBoxLayout(self)
        intro = QLabel("How tensor pictures are drawn. Text size, font, ticks and axis labels "
                       "follow Plot Style, as for every plot. Changing Plot Style's series or map "
                       "colour scale, or its grid, sets the ones here too: the last change wins.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        self.tabs = QTabWidget()
        tensors_page = QWidget()
        form = QFormLayout(tensors_page)
        self.tabs.addTab(tensors_page, "Tensors")

        friendly = colourblind_friendly_maps()
        self.colormap_combo = QComboBox()
        for label, name, _div in TENSOR_COLORMAPS:
            text = label + ("  ✓ colour-blind friendly" if name in friendly else "")
            self.colormap_combo.addItem(text, name)
        self.colormap_combo.setToolTip(
            "The colour scale of a frequency surface. ✓ marks scales that pass openNMR's "
            "measured colour-blind check. A surface of both signs is centred on zero, so a "
            "diverging scale shows the sign.")
        form.addRow("Colour map:", self.colormap_combo)
        self.reverse_check = QCheckBox("Reverse the colour map")
        form.addRow(self.reverse_check)
        self.colorbar_check = QCheckBox("Colour bar")
        form.addRow(self.colorbar_check)
        self.axes_check = QCheckBox("Principal axes")
        form.addRow(self.axes_check)
        self.grid_check = QCheckBox("Gridlines")
        self.grid_check.setToolTip("The 3D box's gridlines. Plot Style's \"Show grid\" box sets "
                                   "this too while a tensor picture is on show.")
        form.addRow(self.grid_check)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.05, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(2)
        form.addRow("Surface opacity:", self.alpha_spin)

        self._colour_buttons = {}
        for key, label in (("positive_colour", "Positive axes:"), ("negative_colour", "Negative axes:"),
                           ("ellipsoid_colour", "Ellipsoid:")):
            button = QPushButton()
            button.clicked.connect(lambda _c=False, k=key: self._pick_colour(k))
            self._colour_buttons[key] = button
            form.addRow(label, button)
        self.ellipsoid_alpha_spin = QDoubleSpinBox()
        self.ellipsoid_alpha_spin.setRange(0.05, 1.0)
        self.ellipsoid_alpha_spin.setSingleStep(0.05)
        self.ellipsoid_alpha_spin.setDecimals(2)
        form.addRow("Ellipsoid opacity:", self.ellipsoid_alpha_spin)

        view = QHBoxLayout()
        self.elev_spin = QDoubleSpinBox()
        self.elev_spin.setRange(-90.0, 90.0)
        self.elev_spin.setSuffix("°")
        self.azim_spin = QDoubleSpinBox()
        self.azim_spin.setRange(-360.0, 360.0)
        self.azim_spin.setSuffix("°")
        view.addWidget(QLabel("elevation"))
        view.addWidget(self.elev_spin)
        view.addWidget(QLabel("azimuth"))
        view.addWidget(self.azim_spin)
        form.addRow("View:", view)
        self.current_view_button = QPushButton("Use the Current View")
        self.current_view_button.setToolTip("Take the angles the picture has been turned to.")
        self.current_view_button.clicked.connect(self._use_current_view)
        form.addRow(self.current_view_button)
        self.tabs.addTab(self._build_axes_page(), "Axes && Labels")
        self.tabs.addTab(self._build_atoms_page(), "Atoms && Bonds")
        v.addWidget(self.tabs)

        buttons = QHBoxLayout()
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset)
        self.save_button = QPushButton("Save as Default for Tensor Pictures")
        self.save_button.setEnabled(save_default is not None)
        self.save_button.clicked.connect(lambda: self._save_default and self._save_default(self.style()))
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons.addWidget(self.reset_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(close)
        v.addLayout(buttons)

        self.load(canvas.style_settings().get("tensor_style") or {})
        # Plot Style's colour maps / grid reach the picture through the
        # canvas; show them here rather than overwrite them on the next edit.
        if hasattr(canvas, "tensorStyleChanged"):
            canvas.tensorStyleChanged.connect(self._reload_from_canvas)
        self.colormap_combo.currentIndexChanged.connect(self._changed)
        for check in (self.reverse_check, self.colorbar_check, self.axes_check, self.grid_check,
                      self.tick_labels_check, self.axis_labels_check, self.axes_box_check,
                      self.depthshade_check, self.bond_own_colour_check):
            check.toggled.connect(self._changed)
        for spin in (self.alpha_spin, self.ellipsoid_alpha_spin, self.elev_spin, self.azim_spin,
                     self.tick_pad_spin, self.label_pad_spin, self.atom_scale_spin,
                     self.atom_alpha_spin, self.atom_edge_width_spin, self.bond_width_spin,
                     self.cell_width_spin):
            spin.valueChanged.connect(self._changed)
        self.atom_size_combo.currentIndexChanged.connect(self._changed)
        self.element_combo.currentIndexChanged.connect(lambda _i: self._paint_buttons())
        self.bond_own_colour_check.toggled.connect(lambda _on: self._paint_buttons())

    # ------------------------------------------------------------------
    def _build_axes_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        note = QLabel("The x, y, z axes of every tensor picture (the ellipsoid, the frequency "
                      "surface and the structure).")
        note.setWordWrap(True)
        form.addRow(note)
        self.tick_labels_check = QCheckBox("Tick labels (the numbers)")
        form.addRow(self.tick_labels_check)
        self.axis_labels_check = QCheckBox("Axis labels (x, y, z)")
        form.addRow(self.axis_labels_check)
        self.axes_box_check = QCheckBox("Box and axes")
        self.axes_box_check.setToolTip("Off: no box, panes, ticks or labels, only the "
                                       "picture.")
        form.addRow(self.axes_box_check)
        self.tick_pad_spin = QDoubleSpinBox()
        self.tick_pad_spin.setRange(-10.0, 40.0)
        self.tick_pad_spin.setSingleStep(1.0)
        self.tick_pad_spin.setDecimals(1)
        self.tick_pad_spin.setSuffix(" pt")
        self.tick_pad_spin.setToolTip("Extra space between the axes and their tick labels.")
        form.addRow("Tick label padding:", self.tick_pad_spin)
        self.label_pad_spin = QDoubleSpinBox()
        self.label_pad_spin.setRange(-10.0, 40.0)
        self.label_pad_spin.setSingleStep(1.0)
        self.label_pad_spin.setDecimals(1)
        self.label_pad_spin.setSuffix(" pt")
        self.label_pad_spin.setToolTip("Extra space between the tick labels and the axis label "
                                       "(the box shrinks if a label would leave the figure).")
        form.addRow("Axis label padding:", self.label_pad_spin)
        return page

    def _build_atoms_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        note = QLabel("The structure picture (DFT tensors ▸ Show: On the structure).")
        note.setWordWrap(True)
        form.addRow(note)
        self.atom_scale_spin = QDoubleSpinBox()
        self.atom_scale_spin.setRange(0.0, 5.0)
        self.atom_scale_spin.setSingleStep(0.1)
        self.atom_scale_spin.setDecimals(2)
        self.atom_scale_spin.setSuffix(" ×")
        self.atom_scale_spin.setToolTip("Size of the atoms (0: bonds and tensors only).")
        form.addRow("Atom size:", self.atom_scale_spin)
        self.atom_size_combo = QComboBox()
        self.atom_size_combo.addItem("By covalent radius", "radius")
        self.atom_size_combo.addItem("All the same", "uniform")
        form.addRow("Sized:", self.atom_size_combo)
        self.atom_alpha_spin = QDoubleSpinBox()
        self.atom_alpha_spin.setRange(0.05, 1.0)
        self.atom_alpha_spin.setSingleStep(0.05)
        self.atom_alpha_spin.setDecimals(2)
        form.addRow("Atom opacity:", self.atom_alpha_spin)
        self.atom_edge_width_spin = QDoubleSpinBox()
        self.atom_edge_width_spin.setRange(0.0, 5.0)
        self.atom_edge_width_spin.setSingleStep(0.1)
        self.atom_edge_width_spin.setDecimals(2)
        self.atom_edge_width_spin.setSuffix(" pt")
        self.atom_edge_width_spin.setToolTip("Outline of each atom (the atoms whose tensors are "
                                             "drawn get a heavier one).")
        form.addRow("Outline width:", self.atom_edge_width_spin)
        self.depthshade_check = QCheckBox("Depth cueing (fade atoms further away)")
        form.addRow(self.depthshade_check)

        element_row = QHBoxLayout()
        self.element_combo = QComboBox()
        self.element_combo.setToolTip("An element of the structure on show.")
        self.element_colour_button = QPushButton()
        self.element_colour_button.clicked.connect(self._pick_element_colour)
        self.element_reset_button = QPushButton("Jmol Colours")
        self.element_reset_button.setToolTip("Every element back to its Jmol colour.")
        self.element_reset_button.clicked.connect(self._reset_element_colours)
        element_row.addWidget(self.element_combo, 1)
        element_row.addWidget(self.element_colour_button, 1)
        element_row.addWidget(self.element_reset_button)
        form.addRow("Element colour:", element_row)

        self.bond_width_spin = QDoubleSpinBox()
        self.bond_width_spin.setRange(0.0, 12.0)
        self.bond_width_spin.setSingleStep(0.2)
        self.bond_width_spin.setDecimals(1)
        self.bond_width_spin.setSuffix(" pt")
        form.addRow("Bond width:", self.bond_width_spin)
        bond_row = QHBoxLayout()
        self.bond_own_colour_check = QCheckBox("One colour")
        self.bond_own_colour_check.setToolTip("Off: each half of a bond in its atom's colour.")
        bond_row.addWidget(self.bond_own_colour_check)
        self._colour_buttons_extra = {}
        for key, label in (("bond_colour", None), ("atom_edge_colour", "Outline colour:"),
                           ("cell_colour", "Cell edges colour:")):
            button = QPushButton()
            button.clicked.connect(lambda _c=False, k=key: self._pick_colour(k))
            self._colour_buttons_extra[key] = button
            if label is None:
                bond_row.addWidget(button, 1)
                form.addRow("Bond colour:", bond_row)
            else:
                form.addRow(label, button)
        self.cell_width_spin = QDoubleSpinBox()
        self.cell_width_spin.setRange(0.0, 6.0)
        self.cell_width_spin.setSingleStep(0.2)
        self.cell_width_spin.setDecimals(1)
        self.cell_width_spin.setSuffix(" pt")
        form.addRow("Cell edges width:", self.cell_width_spin)
        return page

    def _elements_on_show(self) -> list:
        """The elements in the picture on show (its legend), then any whose
        colour has been set."""
        found = []
        for ax in self._canvas.figure.axes:
            legend = ax.get_legend()
            if legend is None:
                continue
            for text in legend.get_texts():
                name = text.get_text()
                if name in ds.ELEMENT_COLOURS and name not in found:
                    found.append(name)
        for name in self._element_colours:
            if name not in found:
                found.append(name)
        return found

    def _fill_elements(self) -> None:
        current = self.element_combo.currentText()
        self.element_combo.blockSignals(True)
        self.element_combo.clear()
        for name in self._elements_on_show():
            self.element_combo.addItem(name)
        index = self.element_combo.findText(current)
        self.element_combo.setCurrentIndex(max(index, 0))
        self.element_combo.blockSignals(False)
        has = self.element_combo.count() > 0
        self.element_colour_button.setEnabled(has)

    def _pick_element_colour(self) -> None:
        name = self.element_combo.currentText()
        if not name:
            return
        colour = QColorDialog.getColor(QColor(ds.colour_of(name, self._element_colours)), self,
                                       f"Colour of {name}")
        if colour.isValid():
            self.set_element_colour(name, colour.name())

    def set_element_colour(self, element: str, colour: str | None) -> None:
        """``element`` drawn in ``colour`` (``None``: its Jmol colour)."""
        if colour:
            self._element_colours[element] = colour
        else:
            self._element_colours.pop(element, None)
        self._fill_elements()
        index = self.element_combo.findText(element)
        if index >= 0:
            self.element_combo.setCurrentIndex(index)
        self._paint_buttons()
        self._changed()

    def _reset_element_colours(self) -> None:
        self._element_colours = {}
        self._fill_elements()
        self._paint_buttons()
        self._changed()

    # ------------------------------------------------------------------
    def load(self, overrides: dict) -> None:
        look = tv.tensor_style(overrides)
        self._loading = True
        try:
            name = look["colormap"] or ""
            index = self.colormap_combo.findData(name)
            if index < 0:
                # A scale chosen in Plot Style that this list does not
                # offer: listed, so the next edit here does not drop it.
                self.colormap_combo.addItem(name, name)
                index = self.colormap_combo.count() - 1
            self.colormap_combo.setCurrentIndex(index)
            self.reverse_check.setChecked(bool(look["reverse"]))
            self.colorbar_check.setChecked(bool(look["colorbar"]))
            self.axes_check.setChecked(bool(look["show_axes"]))
            self.grid_check.setChecked(bool(look["grid"]))
            self.alpha_spin.setValue(float(look["surface_alpha"]))
            self.ellipsoid_alpha_spin.setValue(float(look["ellipsoid_alpha"]))
            self.elev_spin.setValue(float(look["elev"]))
            self.azim_spin.setValue(float(look["azim"]))
            self.tick_labels_check.setChecked(bool(look["tick_labels"]))
            self.axis_labels_check.setChecked(bool(look["axis_labels"]))
            self.axes_box_check.setChecked(bool(look["axes_box"]))
            self.tick_pad_spin.setValue(float(look["tick_pad"]))
            self.label_pad_spin.setValue(float(look["label_pad"]))
            self.atom_scale_spin.setValue(float(look["atom_scale"]))
            self.atom_size_combo.setCurrentIndex(
                max(self.atom_size_combo.findData(look["atom_size_by"]), 0))
            self.atom_alpha_spin.setValue(float(look["atom_alpha"]))
            self.atom_edge_width_spin.setValue(float(look["atom_edge_width"]))
            self.depthshade_check.setChecked(bool(look["atom_depthshade"]))
            self.bond_width_spin.setValue(float(look["bond_width"]))
            self.bond_own_colour_check.setChecked(bool(look["bond_colour"]))
            self.cell_width_spin.setValue(float(look["cell_width"]))
            self._colours = {k: look[k] for k in self._colour_buttons}
            self._colours.update({k: look[k] or "#808080" for k in self._colour_buttons_extra})
            self._element_colours = dict(look["atom_colours"] or {})
            self._fill_elements()
            self._paint_buttons()
        finally:
            self._loading = False

    def style(self) -> dict:
        return {
            "colormap": self.colormap_combo.currentData() or None,
            "reverse": self.reverse_check.isChecked(),
            "colorbar": self.colorbar_check.isChecked(),
            "show_axes": self.axes_check.isChecked(),
            "grid": self.grid_check.isChecked(),
            "surface_alpha": self.alpha_spin.value(),
            "ellipsoid_alpha": self.ellipsoid_alpha_spin.value(),
            "elev": self.elev_spin.value(),
            "azim": self.azim_spin.value(),
            **self._colours,
            "tick_labels": self.tick_labels_check.isChecked(),
            "axis_labels": self.axis_labels_check.isChecked(),
            "axes_box": self.axes_box_check.isChecked(),
            "tick_pad": self.tick_pad_spin.value(),
            "label_pad": self.label_pad_spin.value(),
            "atom_scale": self.atom_scale_spin.value(),
            "atom_size_by": self.atom_size_combo.currentData(),
            "atom_alpha": self.atom_alpha_spin.value(),
            "atom_edge_width": self.atom_edge_width_spin.value(),
            "atom_depthshade": self.depthshade_check.isChecked(),
            "atom_colours": dict(self._element_colours),
            "bond_width": self.bond_width_spin.value(),
            # "" = each half in its atom's colour.
            "bond_colour": self._colours["bond_colour"] if self.bond_own_colour_check.isChecked()
            else "",
            "cell_width": self.cell_width_spin.value(),
        }

    def _reload_from_canvas(self) -> None:
        if not self._loading:
            self.load(self._canvas.style_settings().get("tensor_style") or {})

    def _paint_buttons(self) -> None:
        buttons = {**self._colour_buttons, **getattr(self, "_colour_buttons_extra", {})}
        for key, button in buttons.items():
            _paint(button, self._colours[key])
        self._colour_buttons_extra["bond_colour"].setEnabled(
            self.bond_own_colour_check.isChecked())
        name = self.element_combo.currentText()
        if name:
            _paint(self.element_colour_button, ds.colour_of(name, self._element_colours))
        else:
            self.element_colour_button.setText("(no structure on show)")
            self.element_colour_button.setStyleSheet("")

    def showEvent(self, event) -> None:  # noqa: N802 -- Qt's name
        # The structure on show may have changed since the last time.
        if hasattr(self, "_element_colours"):
            self._fill_elements()
            self._paint_buttons()
        super().showEvent(event)

    def _pick_colour(self, key: str) -> None:
        colour = QColorDialog.getColor(QColor(self._colours[key]), self, "Choose a colour")
        if colour.isValid():
            self.set_colour(key, colour.name())

    def set_colour(self, key: str, colour: str) -> None:
        self._colours[key] = colour
        self._paint_buttons()
        self._changed()

    def _use_current_view(self) -> None:
        threes = [ax for ax in self._canvas.figure.axes if getattr(ax, "name", "") == "3d"]
        if not threes:
            return
        self._loading = True
        self.elev_spin.setValue(float(threes[0].elev))
        self.azim_spin.setValue(float(threes[0].azim))
        self._loading = False
        self._changed()

    def _reset(self) -> None:
        self.load({})
        self._changed()

    def _changed(self, *_args) -> None:
        if not self._loading:
            self._canvas.set_style(tensor_style=self.style())


def _paint(button, colour: str) -> None:
    button.setText(colour)
    text = "#ffffff" if QColor(colour).lightnessF() < 0.55 else "#000000"
    button.setStyleSheet(f"background: {colour}; color: {text};")
