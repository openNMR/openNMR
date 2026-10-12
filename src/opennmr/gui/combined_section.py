"""Fit ▸ CSA, Quad & J-coupling and Simulate ▸ CSA, Quad & J-coupling: the one
entry point for every spectral interaction (explicit user request,
2026-10-06: "condense the models into CSA, Quad & J-coupling, so that people
can just add peaks and then, to each peak, add an interaction").

Two models, one section:

* **Peaks with interactions** -- :class:`~.combined_panels.CombinedHamiltonianPanel`
  (or its fit subclass): add peaks, then add a CSA, a quadrupolar
  coupling (first order, second order or exact) or first-order
  J-couplings to each, static or MAS.
* **Strongly coupled peaks** -- the J-coupling panel locked to its exact
  spin-system mode (:func:`~.jcoupling_panels.lock_to_spin_system`). Strong
  coupling is J *between observed peaks*, solved by diagonalising the
  whole spin system; a per-peak interaction cannot express it, and the
  spin-system engine has no CSA or quadrupolar term, so it stays its own
  model. A J interaction's "Strong coupling" treatment, or "J-coupling
  between peaks" in Add interaction, switches here with the peaks as spins.
"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget

MODELS = (("peaks", "Peaks with interactions (CSA, quadrupolar, J to other nuclei)"),
          ("spin_system", "Strongly coupled peaks (exact J between them)"))


class CombinedModelSection(QWidget):
    """A model chooser over two pages: ``peaks`` and ``spin_system``."""

    def __init__(self, peaks_widget: QWidget, spin_widget: QWidget, parent=None) -> None:
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.model_combo = QComboBox()
        for key, text in MODELS:
            self.model_combo.addItem(text, key)
        self.model_combo.setToolTip(
            "Peaks with interactions: any mix of CSA, quadrupolar (first order, second order "
            "or exact) and first-order J on each peak, static or MAS.\n\nStrongly coupled "
            "peaks: J between observed peaks comparable to their shift difference, solved "
            "exactly (solution-state spin systems).")
        form.addRow("Model:", self.model_combo)
        v.addLayout(form)
        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        self.note_label.setVisible(False)
        v.addWidget(self.note_label)
        self.stack = QStackedWidget()
        self.peaks_widget = peaks_widget
        self.spin_widget = spin_widget
        self.stack.addWidget(peaks_widget)
        self.stack.addWidget(spin_widget)
        v.addWidget(self.stack)
        self.model_combo.currentIndexChanged.connect(self._on_model)

    def model(self) -> str:
        return self.model_combo.currentData()

    def show_model(self, key: str, note: str = "") -> None:
        self.model_combo.setCurrentIndex(self.model_combo.findData(key))
        self.note_label.setText(note)
        self.note_label.setVisible(bool(note))

    def _on_model(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.note_label.setVisible(False)
