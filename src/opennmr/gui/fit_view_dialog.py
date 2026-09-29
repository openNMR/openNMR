"""How one kept fit is drawn in Compare (right-click its entry ▸ Fit
Display…): data, total fit and residual on or off, the fit's colour, and
its components -- shown, hidden or highlighted, filled or not, in which
colours (the Fit plot controls, for this one entry).

The view is :func:`opennmr.notebook.fit_models.fit_view`, saved with the
entry as ``metadata.raw["fit_model_view"]``; the drawing is
:func:`opennmr.notebook.fit_models.draw_fit_model`, so a notebook figure
given the same view matches. Changes apply as they are made.
"""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QPushButton, QVBoxLayout,
)
from matplotlib.colors import to_hex

from ..notebook import fit_models as nb_fit_models
from .fit_display_panel import FitDisplayPanel, _swatch


class FitViewDialog(QDialog):
    """``model``: the decoded kept fit; ``view``: its current view;
    ``on_change(view)``: called with the complete view after each edit."""

    def __init__(self, model, view, on_change, title="Fit display", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._model = model
        self._view = nb_fit_models.fit_view(view)
        self._on_change = on_change
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.data_check = QCheckBox("Data")
        self.fit_check = QCheckBox("Total fit")
        self.residual_check = QCheckBox("Residual (data − fit), just below")
        for check, key in ((self.data_check, "show_data"), (self.fit_check, "show_fit"),
                           (self.residual_check, "show_residual")):
            check.setChecked(self._view[key])
            check.toggled.connect(lambda on, k=key: self._set(k, on))
        row = QHBoxLayout()
        for w in (self.data_check, self.fit_check, self.residual_check):
            row.addWidget(w)
        row.addStretch(1)
        form.addRow("Show:", row)
        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.0, 2.0)
        self.gap_spin.setSingleStep(0.02)
        self.gap_spin.setDecimals(2)
        self.gap_spin.setValue(self._view["residual_gap"])
        self.gap_spin.setToolTip("Space between the spectrum and its residual, as a fraction of "
                                 "the spectrum's height.")
        self.gap_spin.valueChanged.connect(lambda value: self._set("residual_gap", value))
        form.addRow("Residual gap:", self.gap_spin)
        self.fit_colour_button = QPushButton("Choose…")
        self.fit_colour_button.clicked.connect(self._pick_fit_colour)
        self.trace_colour_button = QPushButton("Use the trace's colour")
        self.trace_colour_button.setToolTip("The colour Compare gives this entry (Plot Style).")
        self.trace_colour_button.clicked.connect(lambda: self._set("fit_colour", None))
        colour_row = QHBoxLayout()
        colour_row.addWidget(self.fit_colour_button)
        colour_row.addWidget(self.trace_colour_button)
        colour_row.addStretch(1)
        form.addRow("Fit colour:", colour_row)
        v.addLayout(form)
        self.components_panel = FitDisplayPanel(
            [], style=self._view["style"], on_change=lambda style: self._set("style", style),
            intro="Components of this fit, and the data's colour (the total fit's colour is set "
                  "above).")
        self.components_panel.fit_colour_button.setVisible(False)
        comps = list(model.get("components") or [])[:-1]
        self.components_panel.set_components(model.get("labels") or [], comps)
        v.addWidget(self.components_panel)
        v.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.accept)
        box.accepted.connect(self.accept)
        v.addWidget(box)
        self._sync()

    def view(self) -> dict:
        return nb_fit_models.fit_view(self._view)

    def _set(self, key, value) -> None:
        self._view[key] = value
        self._view = nb_fit_models.fit_view(self._view)
        self._sync()
        self._on_change(self.view())

    def _sync(self) -> None:
        self.gap_spin.setEnabled(self._view["show_residual"])
        colour = self._view["fit_colour"]
        if colour:
            self.fit_colour_button.setIcon(_swatch(colour))
            self.fit_colour_button.setText(to_hex(colour))
        else:
            self.fit_colour_button.setIcon(_swatch("#ffffff"))
            self.fit_colour_button.setText("trace colour")

    def _pick_fit_colour(self) -> None:
        start = self._view["fit_colour"] or "#d62728"
        colour = QColorDialog.getColor(QColor(to_hex(start)), self, "Fit colour")
        if colour.isValid():
            self._set("fit_colour", colour.name())
