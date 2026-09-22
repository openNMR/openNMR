"""A plain matplotlib canvas for the Simulate panel's plots.

Deliberately not :class:`MplCanvas`: that one is built around a
spectrum -- a reversed ppm axis, spectral scroll-zoom, contours,
peak markers -- none of which means anything for an attenuation curve
or a pulse's excitation profile. This is a figure, its Qt canvas and
the standard navigation toolbar, and the simulation code draws onto
``figure`` directly.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class SimulationCanvas(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    def draw(self) -> None:
        self.canvas.draw_idle()
        # A fresh figure invalidates the toolbar's zoom history.
        self.toolbar.update()
