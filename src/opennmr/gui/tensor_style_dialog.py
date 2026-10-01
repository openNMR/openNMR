"""Right-click ▸ Tensor Plot Style… on Simulate's tensor pictures.

Explicit user request (2026-10-01): "a specific tensor plot style" -- the
colour map (Plot Style's text, ticks and axis-label settings apply as
for every plot), the colour bar, principal axes, the ellipsoid's colours
and opacity, and the view. Edits apply live to the picture on show; they
are :data:`opennmr.notebook.tensor_views.TENSOR_STYLE_DEFAULTS`' keys,
so a notebook passes the same dict as ``tensor_style=``.

``canvas`` is the MplCanvas showing the picture; ``save_default(style)``
(optional) keeps the settings as the default for tensor pictures.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QPushButton, QVBoxLayout,
)

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
                       "follow Plot Style, as for every plot.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        form = QFormLayout()

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
        v.addLayout(form)

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
        self.colormap_combo.currentIndexChanged.connect(self._changed)
        for check in (self.reverse_check, self.colorbar_check, self.axes_check):
            check.toggled.connect(self._changed)
        for spin in (self.alpha_spin, self.ellipsoid_alpha_spin, self.elev_spin, self.azim_spin):
            spin.valueChanged.connect(self._changed)

    # ------------------------------------------------------------------
    def load(self, overrides: dict) -> None:
        look = tv.tensor_style(overrides)
        self._loading = True
        try:
            index = self.colormap_combo.findData(look["colormap"] or "")
            self.colormap_combo.setCurrentIndex(max(index, 0))
            self.reverse_check.setChecked(bool(look["reverse"]))
            self.colorbar_check.setChecked(bool(look["colorbar"]))
            self.axes_check.setChecked(bool(look["show_axes"]))
            self.alpha_spin.setValue(float(look["surface_alpha"]))
            self.ellipsoid_alpha_spin.setValue(float(look["ellipsoid_alpha"]))
            self.elev_spin.setValue(float(look["elev"]))
            self.azim_spin.setValue(float(look["azim"]))
            self._colours = {k: look[k] for k in self._colour_buttons}
            self._paint_buttons()
        finally:
            self._loading = False

    def style(self) -> dict:
        return {
            "colormap": self.colormap_combo.currentData() or None,
            "reverse": self.reverse_check.isChecked(),
            "colorbar": self.colorbar_check.isChecked(),
            "show_axes": self.axes_check.isChecked(),
            "surface_alpha": self.alpha_spin.value(),
            "ellipsoid_alpha": self.ellipsoid_alpha_spin.value(),
            "elev": self.elev_spin.value(),
            "azim": self.azim_spin.value(),
            **self._colours,
        }

    def _paint_buttons(self) -> None:
        for key, button in self._colour_buttons.items():
            colour = self._colours[key]
            button.setText(colour)
            text = "#ffffff" if QColor(colour).lightnessF() < 0.55 else "#000000"
            button.setStyleSheet(f"background: {colour}; color: {text};")

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
