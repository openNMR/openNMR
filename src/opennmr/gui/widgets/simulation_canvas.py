"""A plain matplotlib canvas for the Simulate panel's plots.

Deliberately not :class:`MplCanvas`: that one is built around a
spectrum -- a reversed ppm axis, spectral scroll-zoom, contours,
peak markers -- none of which means anything for an attenuation curve
or a pulse's excitation profile. This is a figure, its Qt canvas and
the standard navigation toolbar, and the simulation code draws onto
``figure`` directly.

When the plot is a simulated spectrum (a Czjzek simulation beside its
distribution map, say), whoever drew it sets :attr:`simulation_spectrum`
and the right-click menu offers to keep it in Open experiments.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QMenu, QVBoxLayout, QWidget


class SimulationCanvas(QWidget):
    #: "Keep This Simulation in Open Experiments", with the spectrum.
    keepSimulationRequested = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.simulation_spectrum = None
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._on_context_menu)

    def draw(self) -> None:
        self.canvas.draw_idle()
        # A fresh figure invalidates the toolbar's zoom history.
        self.toolbar.update()

    def context_menu(self) -> QMenu | None:
        """The right-click menu (None when there is nothing to offer)."""
        if self.simulation_spectrum is None:
            return None
        menu = QMenu(self)
        action = menu.addAction("Keep This Simulation in Open Experiments")
        action.setToolTip("List the simulated spectrum under Open experiments, to compare it "
                          "with data or other simulations (Compare), process or save it.")
        action.triggered.connect(
            lambda _c=False: self.keepSimulationRequested.emit(self.simulation_spectrum))
        return menu

    def _on_context_menu(self, pos) -> None:
        menu = self.context_menu()
        if menu is not None:
            self._exec_context_menu(menu, self.canvas.mapToGlobal(pos))

    def _exec_context_menu(self, menu, global_pos) -> None:
        """A seam: tests replace it (QMenu.exec is modal)."""
        menu.exec(global_pos)
