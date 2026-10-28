"""A draggable bar for the spacing between stacked spectra (Compare ▸ Stacked).

The old control was a spin box, "Gap between stacked spectra", in percent --
cryptic, and it stopped at zero, so spectra could not be brought closer than
"not quite touching". This one is a slider that runs from *overlapping* to
*spread out* and says in words what the current position means.

The value is the same number the canvas keeps (``stack_gap``): the space below
each stacked spectrum as a percentage of its own height, negative when the
spectra overlap. :meth:`SpacingSlider.value` / :meth:`setValue` speak that
percentage, so the slider stands in for the spin box it replaces
(``valueChanged(float)``, ``blockSignals`` and all).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QToolButton, QVBoxLayout, QWidget

#: Overlap by up to 90 % of a spectrum's height ... a gap of two heights.
MINIMUM_PERCENT = -90
MAXIMUM_PERCENT = 200


def describe_spacing(percent: float, top_gap_percent: float = 5.0) -> str:
    """What the slider position means, in words. ``percent`` is the stored gap;
    each spectrum also keeps ``top_gap_percent`` clear above its highest point,
    so the real clear space between one spectrum's lowest point and the next
    one's highest is the sum -- that is what is described."""
    clear = float(percent) + float(top_gap_percent)
    if abs(clear) < 0.5:
        return "Spectra just touch"
    if clear < 0:
        return f"Spectra overlap by {abs(clear):.0f} % of their height"
    return f"{clear:.0f} % of a spectrum's height between spectra"


class SpacingSlider(QWidget):
    """``valueChanged(float)`` -- the new gap in percent, while dragging."""

    valueChanged = Signal(float)

    def __init__(self, default_percent: float = 5.0, top_gap_percent: float = 5.0,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._default = float(default_percent)
        self._top_gap = float(top_gap_percent)
        self._value = self._default

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(MINIMUM_PERCENT, MAXIMUM_PERCENT)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.setTracking(True)           # live: the plot follows the drag
        self.slider.setValue(round(self._default))
        self.slider.setToolTip(
            "Drag to move the stacked spectra closer together (left, until they overlap) or "
            "further apart (right). Double-click the bar, or press Reset, for the default.")
        self.reset_button = QToolButton()
        self.reset_button.setText("Reset")
        self.reset_button.setToolTip("Back to the default spacing.")

        self.readout = QLabel()
        self.readout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.readout.setWordWrap(True)

        ends = QHBoxLayout()
        closer = QLabel("closer")
        apart = QLabel("further apart")
        for label in (closer, apart):
            label.setEnabled(False)               # greyed: a caption, not a control
        ends.addWidget(closer)
        ends.addStretch(1)
        ends.addWidget(apart)

        row = QHBoxLayout()
        row.addWidget(self.slider, 1)
        row.addWidget(self.reset_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addLayout(row)
        layout.addLayout(ends)
        layout.addWidget(self.readout)

        self.slider.valueChanged.connect(self._on_slider)
        self.slider.mouseDoubleClickEvent = lambda _event: self.reset()  # noqa: E731
        self.reset_button.clicked.connect(lambda _checked=False: self.reset())
        self._refresh_readout()

    # -- the QDoubleSpinBox-compatible face -----------------------------

    def value(self) -> float:
        """The gap in percent of a spectrum's height (negative: overlap)."""
        return float(self._value)

    def setValue(self, percent: float) -> None:  # noqa: N802 -- Qt naming
        percent = min(max(float(percent), MINIMUM_PERCENT), MAXIMUM_PERCENT)
        if percent == self._value:
            return
        self._value = percent
        blocked = self.slider.blockSignals(True)
        try:
            self.slider.setValue(round(percent))
        finally:
            self.slider.blockSignals(blocked)
        self._refresh_readout()
        self.valueChanged.emit(self._value)

    def reset(self) -> None:
        """Back to the default spacing."""
        self.setValue(self._default)

    def defaultValue(self) -> float:  # noqa: N802
        return self._default

    def text(self) -> str:
        """The sentence under the bar."""
        return self.readout.text()

    # -- internals ------------------------------------------------------

    def _on_slider(self, position: int) -> None:
        self._value = float(position)          # a drag lands on whole percents
        self._refresh_readout()
        self.valueChanged.emit(self._value)

    def _refresh_readout(self) -> None:
        self.readout.setText(describe_spacing(self._value, self._top_gap))
