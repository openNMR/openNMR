"""
"Tensor picture" controls for the Simulate panels (CSA, Quadrupolar,
Combined Hamiltonian): draw the simulated site's tensors in 3D on the
Simulation plot, as its frequency surface or as tensor glyphs.

The pictures are :mod:`opennmr.notebook.tensor_views`' own figures,
drawn onto the Simulation plot's figure, so the GUI and a notebook show
the same thing. The Simulation plot is a plain matplotlib canvas, so the
3D view can be turned with the mouse.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout,
)

from ..notebook import tensor_views as tv


class TensorViewControls(QGroupBox):
    """``site(): (Site, larmor_Hz or None, order)`` -- the site the panel
    would simulate now (raises ValueError with a message if its inputs
    are incomplete); ``draw(draw_fn)`` -- show ``draw_fn(figure)`` on the
    Simulation plot; ``style()`` -- that plot's Plot Style settings, read
    as the picture is drawn (text size; ``map_colormap``, the surface's
    colour scale), so a style change redraws it in the new style."""

    def __init__(self, site, draw, parent=None, style=None) -> None:
        super().__init__("Tensor picture (3D)", parent)
        self._site = site
        self._draw = draw
        self._style = style or (lambda: {})
        self.notes = []

        v = QVBoxLayout(self)
        form = QFormLayout()
        self.style_combo = QComboBox()
        self.style_combo.addItem("Frequency surface", "surface")
        self.style_combo.addItem("Ellipsoid and axes", "ellipsoid")
        self.style_combo.setToolTip(
            "Frequency surface: for each direction of B0 in the site frame, the line's "
            "frequency drawn as the distance from the centre -- what the powder pattern is "
            "made of (an isotropic shift is a sphere of radius |δiso|).\n"
            "Ellipsoid and axes: the CSA ellipsoid (semi-axes |δii|) and the EFG's principal "
            "axes in one frame, showing their relative orientation.")
        self.style_combo.currentIndexChanged.connect(lambda _i: self._sync())
        form.addRow("Show:", self.style_combo)
        self.relative_combo = QComboBox()
        self.relative_combo.addItem("Whole frequency (δ)", "zero")
        self.relative_combo.addItem("Anisotropic part (δ − δiso)", "iso")
        self.relative_combo.setToolTip(
            "δ − δiso leaves out the isotropic shift, so the anisotropy is visible even "
            "when δiso is much larger than it.")
        form.addRow("Frequency:", self.relative_combo)
        self.transition_combo = QComboBox()
        self.transition_combo.setToolTip("Which transition's surface to draw.")
        form.addRow("Transition:", self.transition_combo)
        self._form = form
        v.addLayout(form)
        self.show_button = QPushButton("Show Tensors")
        self.show_button.setToolTip("Draw on the Simulation plot; drag it to turn the view.")
        self.show_button.clicked.connect(self.show_tensors)
        v.addWidget(self.show_button)
        self.notes_label = QLabel("")
        self.notes_label.setWordWrap(True)
        v.addWidget(self.notes_label)
        self._sync()

    def _sync(self) -> None:
        surface = self.style_combo.currentData() == "surface"
        self._form.setRowVisible(self.relative_combo, surface)
        self._form.setRowVisible(self.transition_combo, surface)

    def refresh_transitions(self, site) -> None:
        """Offer the site's transitions, keeping the choice if it exists."""
        keep = self.transition_combo.currentText()
        labels = tv.transition_labels(site)
        blocked = self.transition_combo.blockSignals(True)
        self.transition_combo.clear()
        self.transition_combo.addItems(labels)
        self.transition_combo.setCurrentText(keep if keep in labels else tv.default_transition(site))
        self.transition_combo.blockSignals(blocked)

    def show_tensors(self) -> bool:
        try:
            site, larmor_hz, order = self._site()
            self.refresh_transitions(site)
            style = self.style_combo.currentData()
            kwargs = {}
            if style == "surface":
                kwargs = {"transition": self.transition_combo.currentText() or None,
                          "order": order or "second",
                          "relative_to": self.relative_combo.currentData()}
            holder = {}

            def draw(figure):
                look = self._style() or {}
                extra = dict(kwargs)
                if style == "surface" and look.get("map_colormap"):
                    extra["cmap"] = look["map_colormap"]
                holder["notes"] = tv.plot_tensors(site, larmor_hz, style=style, figure=figure,
                                                  fontsize=float(look.get("labelsize") or 11.0),
                                                  **extra)[1]
            self._draw(draw)
        except (ValueError, KeyError) as exc:
            self.notes_label.setText(f"Cannot draw: {exc}")
            return False
        self.notes = holder.get("notes", [])
        self.notes_label.setText("\n".join(self.notes))
        return True
