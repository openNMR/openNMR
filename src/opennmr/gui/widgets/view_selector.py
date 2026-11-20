"""One slice selector, in one place, for every view that browses slices (notes 8.2).

The slice a view shows used to be chosen in three different places: a spin box
in the Process panel, another in the Fitting panel for spectral fits, and a
third beside the plot for curve fits. :class:`MirroredSpinBar` is the one
control that now sits at the top right of the plot area and **mirrors** whichever
spin box belongs to the view on show, in both directions. The panel spin boxes
stay (they carry the saved state and the existing behaviour); the bar only
re-points at the right one when the view changes, so the three views behave alike.
"""
from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSpinBox, QWidget

from .typed_spinbox import TypedSpinBox


class MirroredSpinBar(QWidget):
    """``label  ◀ [n] ▶`` that follows another spin box and drives it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._target: QSpinBox | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.label = QLabel("Slice:")
        layout.addWidget(self.label)
        self.prev_button = QPushButton("◀")
        self.prev_button.setFixedWidth(28)
        self.prev_button.setToolTip("Previous")
        self.spin = TypedSpinBox()
        self.spin.setMinimum(1)
        self.next_button = QPushButton("▶")
        self.next_button.setFixedWidth(28)
        self.next_button.setToolTip("Next")
        for widget in (self.prev_button, self.spin, self.next_button):
            layout.addWidget(widget)
        self.prev_button.clicked.connect(lambda: self.spin.setValue(self.spin.value() - 1))
        self.next_button.clicked.connect(lambda: self.spin.setValue(self.spin.value() + 1))
        self.spin.valueChanged.connect(self._push)

    @property
    def target(self) -> QSpinBox | None:
        return self._target

    def bind(self, target: QSpinBox | None, label: str = "Slice:", tooltip: str = "") -> None:
        """Follow ``target`` (``None`` to follow nothing). Safe to call repeatedly."""
        if target is self._target:
            self.label.setText(label)
            self.spin.setToolTip(tooltip)
            self.sync()
            return
        if self._target is not None:
            try:
                self._target.valueChanged.disconnect(self._pull)
            except (RuntimeError, TypeError):
                pass
        self._target = target
        self.label.setText(label)
        self.spin.setToolTip(tooltip)
        if target is not None:
            target.valueChanged.connect(self._pull)
            self.sync()

    def sync(self) -> None:
        """Copy the target's range, enabled state and value (without echoing back)."""
        target = self._target
        if target is None:
            return
        blocked = self.spin.blockSignals(True)
        self.spin.setRange(target.minimum(), target.maximum())
        self.spin.setValue(target.value())
        self.spin.blockSignals(blocked)
        self.setEnabled(target.isEnabled())

    def _pull(self, value: int) -> None:
        blocked = self.spin.blockSignals(True)
        self.spin.setValue(int(value))
        self.spin.blockSignals(blocked)

    def _push(self, value: int) -> None:
        if self._target is not None and self._target.value() != value:
            self._target.setValue(int(value))
