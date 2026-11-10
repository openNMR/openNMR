"""How a relaxation/diffusion curve fit is drawn, and keeping it (explicit
user request, 2026-09-30) -- the Fit view's right-click ▸ Curve Fit Plot…
since 2026-10-16 (formatting a plot is a right-click on it).

* **Show**: the selected peak (the spin box above browses peaks) or every
  peak together, each in its own colour-blind-safe colour, optionally
  normalised to its own fitted maximum to compare shapes.
* **y axis**: the intensity as measured, I/I0, or ln(I/I0) -- the
  Stejskal-Tanner plot of a PFG series (explicit user request,
  2026-11-10). I0 is the fitted curve at x = 0
  (``processing.relaxation.attenuation``).
* **Fit line**: its colour, line style (solid, dashed, dotted, dash-dot,
  long dash, dash-dot-dot) and width. Remembered between sessions
  (``~/.opennmr/ui_settings.json``, key ``curve_fit_line``).
* **Keep**: the fit on the plot joins the spectrum's kept fits, as a CSA
  or quadrupolar fit does -- one kept fit per peak when all are shown.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QPushButton,
)

#: (label, key, matplotlib linestyle).
LINE_STYLES = (
    ("Solid", "solid", "-"),
    ("Dashed", "dashed", "--"),
    ("Dotted", "dotted", ":"),
    ("Dash-dot", "dashdot", "-."),
    ("Long dash", "longdash", (0, (10, 4))),
    ("Dash-dot-dot", "dashdotdot", (0, (6, 2, 1.5, 2, 1.5, 2))),
)
_MPL_STYLE = {key: mpl for _label, key, mpl in LINE_STYLES}

DEFAULTS = {"colour": "#d62728", "style": "dashed", "width": 2.0, "view": "one",
            "normalise": False, "y_mode": "intensity"}

#: (label, key) of the y-axis choices; keys are
#: processing.relaxation.ATTENUATION_MODES.
Y_MODES = (
    ("Intensity (as measured)", "intensity"),
    ("I/I₀ (normalised to the fit at x = 0)", "relative"),
    ("ln(I/I₀) – Stejskal–Tanner / log plot", "ln"),
)


class CurveFitPlotBox(QGroupBox):
    """Signals: ``changed`` (redraw), ``keepRequested``."""

    changed = Signal()
    keepRequested = Signal()

    def __init__(self, parent=None, keep_in_box: bool = True) -> None:
        """``keep_in_box=False``: the Keep button is made but left for the
        caller to place (the box itself lives in the plot's right-click
        Curve Fit Plot… window, the Keep button beside the fit)."""
        super().__init__("Fit plot", parent)
        self._colour = DEFAULTS["colour"]
        form = QFormLayout(self)

        self.view_combo = QComboBox()
        self.view_combo.addItem("Selected peak", "one")
        self.view_combo.addItem("All peaks together", "all")
        self.view_combo.setToolTip(
            "Selected peak: the peak chosen above, with its report. All peaks together: every "
            "peak's points and fit on one plot, each in its own colour.")
        self.view_combo.currentIndexChanged.connect(self._on_changed)
        form.addRow("Show:", self.view_combo)

        self.normalise_check = QCheckBox("Normalise each peak to its fitted maximum")
        self.normalise_check.setToolTip(
            "With all peaks together: divide each peak's points and fit by the fit's own "
            "maximum, to compare curves of very different intensity by shape.")
        self.normalise_check.toggled.connect(self._on_changed)
        form.addRow("", self.normalise_check)

        self.y_mode_combo = QComboBox()
        for label, key in Y_MODES:
            self.y_mode_combo.addItem(label, key)
        self.y_mode_combo.setToolTip(
            "What the y axis shows.\n\n"
            "I/I₀: every point divided by I₀, the fitted curve at x = 0 (so a fit "
            "of the whole series, not one noisy point, sets the scale).\n\n"
            "ln(I/I₀): the log plot. For a PFG series this is the Stejskal–Tanner "
            "plot – one freely diffusing species is a straight line of slope −D "
            "against b, and curvature shows more than one. Points at or below zero "
            "have no logarithm and are left out (the status bar says how many).")
        self.y_mode_combo.currentIndexChanged.connect(self._on_changed)
        form.addRow("y axis:", self.y_mode_combo)

        self.colour_button = QPushButton()
        self.colour_button.setToolTip("Colour of the fitted line (the selected-peak view; with "
                                      "all peaks, each fit takes its peak's colour).")
        self.colour_button.clicked.connect(self._pick_colour)
        form.addRow("Fit line colour:", self.colour_button)

        self.style_combo = QComboBox()
        for label, key, _mpl in LINE_STYLES:
            self.style_combo.addItem(label, key)
        self.style_combo.setCurrentIndex(self.style_combo.findData(DEFAULTS["style"]))
        self.style_combo.currentIndexChanged.connect(self._on_changed)
        form.addRow("Fit line style:", self.style_combo)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.25, 8.0)
        self.width_spin.setSingleStep(0.25)
        self.width_spin.setDecimals(2)
        self.width_spin.setSuffix(" pt")
        self.width_spin.setValue(DEFAULTS["width"])
        self.width_spin.valueChanged.connect(self._on_changed)
        form.addRow("Fit line width:", self.width_spin)

        self.keep_button = QPushButton("Keep This Curve Fit…")
        self.keep_button.setToolTip(
            "Keep the fit on the plot with this spectrum, under a name – its points, the fitted "
            "curve and its report – as CSA or quadrupolar fits are kept. With all peaks shown, "
            "each peak's fit is kept. Listed under Kept curve fits below (and in Fit ▸ Kept fit "
            "models).")
        self.keep_button.setEnabled(False)
        self.keep_button.clicked.connect(self.keepRequested.emit)
        if keep_in_box:
            form.addRow(self.keep_button)
        self._sync()

    # ---- values -------------------------------------------------------------
    def view(self) -> str:
        return self.view_combo.currentData()

    def normalise(self) -> bool:
        """Normalise each peak to its fitted maximum -- only with all peaks
        shown on the intensity axis (I/I0 is already normalised)."""
        return self.normalise_check.isChecked() and self.y_mode() == "intensity"

    def y_mode(self) -> str:
        return self.y_mode_combo.currentData() or "intensity"

    def fit_style(self) -> dict:
        """The fit line, as MplCanvas.show_relaxation_fit takes it."""
        return {"color": self._colour, "linestyle": _MPL_STYLE[self.style_combo.currentData()],
                "linewidth": float(self.width_spin.value())}

    def settings(self) -> dict:
        return {"colour": self._colour, "style": self.style_combo.currentData(),
                "width": float(self.width_spin.value()), "view": self.view(),
                "normalise": self.normalise_check.isChecked(), "y_mode": self.y_mode()}

    def apply_settings(self, settings: dict | None) -> None:
        settings = {**DEFAULTS, **(settings or {})}
        widgets = (self.view_combo, self.normalise_check, self.style_combo, self.width_spin,
                   self.y_mode_combo)
        for w in widgets:
            w.blockSignals(True)
        try:
            if QColor(str(settings["colour"])).isValid():
                self._colour = QColor(str(settings["colour"])).name()
            index = self.style_combo.findData(settings["style"])
            if index >= 0:
                self.style_combo.setCurrentIndex(index)
            self.width_spin.setValue(float(settings["width"]))
            index = self.view_combo.findData(settings["view"])
            if index >= 0:
                self.view_combo.setCurrentIndex(index)
            self.normalise_check.setChecked(bool(settings["normalise"]))
            index = self.y_mode_combo.findData(settings.get("y_mode", "intensity"))
            self.y_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._sync()

    def set_colour(self, colour: str) -> None:
        if QColor(colour).isValid():
            self._colour = QColor(colour).name()
            self._on_changed()

    # ---- internals ------------------------------------------------------------
    def _pick_colour(self) -> None:
        colour = QColorDialog.getColor(QColor(self._colour), self, "Fit line colour")
        if colour.isValid():
            self.set_colour(colour.name())

    def _on_changed(self, *_args) -> None:
        self._sync()
        self.changed.emit()

    def _sync(self) -> None:
        self.colour_button.setText(self._colour)
        self.colour_button.setStyleSheet(
            f"QPushButton {{ background-color: {self._colour}; "
            f"color: {'black' if QColor(self._colour).lightness() > 140 else 'white'}; }}")
        self.normalise_check.setEnabled(self.view() == "all" and self.y_mode() == "intensity")
        self.colour_button.setEnabled(self.view() == "one")
