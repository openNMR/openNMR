"""Fit ▸ Solid-state NMR and Simulate ▸ Solid-state NMR: the one
entry point for every spectral interaction (explicit user request,
2026-10-06: "condense the models into Solid-state NMR, so that people
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

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (QComboBox, QFormLayout, QLabel, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

MODELS = (("peaks", "Peaks with interactions (CSA, quadrupolar, J to other nuclei)"),
          ("spin_system", "Strongly coupled peaks (exact J between them)"))

MODELS_TIP = (
    "Peaks with interactions: any mix of CSA, quadrupolar (first order, second order "
    "or exact) and first-order J on each peak, static or MAS.\n\nStrongly coupled "
    "peaks: J between observed peaks comparable to their shift difference, solved "
    "exactly (solution-state spin systems).")

#: Solution NMR's two models (explicit user request, 2026-10-24: the
#: strong-J / roofing option back in Solution NMR, and kept in Solid-state NMR).
SOLUTION_MODELS = (
    ("peaks", "Peaks and multiplets (first order: d, t, q …)"),
    ("spin_system", "Strongly coupled spin system (exact: roofing, AB, ABX …)"))

SOLUTION_TIP = (
    "Peaks and multiplets: pseudo-Voigt peaks, each optionally a first-order multiplet "
    "(d, t, q …) – right when every shift difference is much larger than J.\n\n"
    "Strongly coupled spin system: every spin's shift and the J between them, solved "
    "exactly – roofing, AB quartets, ABX and AA'BB' patterns – for when J is comparable "
    "to the shift difference.")


class SpinSystemSlot(QWidget):
    """One place the shared strongly-coupled spin-system panel can sit.

    There is one panel (and so one set of spins, Js and fit results) however
    many sections offer it -- Solution NMR and Solid-state NMR both do -- so
    it moves to whichever section asks (:class:`SpinSystemHost`). A slot the
    panel has left says where it went, with a button to bring it back."""

    def __init__(self, host: "SpinSystemHost", key: str, title: str, blurb: str,
                 parent=None) -> None:
        super().__init__(parent)
        self.host, self.key, self.title, self.blurb = host, key, title, blurb
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.away_label = QLabel("")
        self.away_label.setWordWrap(True)
        self.bring_button = QPushButton("Show it here")
        self.bring_button.setToolTip("Move the spin-system editor here; its spins, J values "
                                     "and fit result come with it.")
        self.bring_button.clicked.connect(lambda: host.show_in(key))
        v.addWidget(self.away_label)
        v.addWidget(self.bring_button)
        self._layout = v
        self.set_away(None)

    @property
    def panel(self) -> QWidget:
        return self.host.panel

    def holds_panel(self) -> bool:
        return self.host.panel.parentWidget() is self

    def hold(self) -> None:
        self.away_label.setVisible(False)
        self.bring_button.setVisible(False)
        self._layout.addWidget(self.host.panel)
        self.host.panel.show()

    def set_away(self, where: str | None) -> None:
        shown = where is not None
        self.away_label.setText(f"The strongly coupled spin system is open under {where}."
                                if shown else "")
        self.away_label.setVisible(shown)
        self.bring_button.setVisible(shown)


class SpinSystemHost(QObject):
    """Owns the one strongly-coupled spin-system panel and the slots it can sit in."""

    def __init__(self, panel: QWidget, parent=None) -> None:
        super().__init__(parent)
        self.panel = panel
        self.slots: dict[str, SpinSystemSlot] = {}
        self.owner: str | None = None

    def add_slot(self, key: str, title: str, blurb: str) -> SpinSystemSlot:
        slot = SpinSystemSlot(self, key, title, blurb)
        self.slots[key] = slot
        if self.owner is None:
            self.show_in(key)
        else:
            slot.set_away(self.slots[self.owner].title)
        return slot

    def show_in(self, key: str) -> None:
        """Move the panel to slot ``key`` (a no-op when it is already there)."""
        target = self.slots[key]
        if self.owner == key and target.holds_panel():
            return
        self.owner = key
        label = getattr(self.panel, "blurb_label", None)
        if label is not None:
            label.setText(target.blurb)
        hook = getattr(self.panel, "on_hosted_in", None)
        if callable(hook):
            hook(key)
        target.hold()
        for other in self.slots.values():
            if other is not target:
                other.set_away(target.title)


class CombinedModelSection(QWidget):
    """A model chooser over two pages: ``peaks`` and ``spin_system``.

    ``models`` and ``tooltip`` retitle the choices (Solution NMR words them
    differently); ``claim`` is called whenever the spin-system page is
    chosen, so a shared panel (:class:`SpinSystemHost`) moves here."""

    def __init__(self, peaks_widget: QWidget, spin_widget: QWidget, parent=None,
                 models=MODELS, tooltip: str = MODELS_TIP, claim=None) -> None:
        super().__init__(parent)
        self._claim = claim
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.model_combo = QComboBox()
        for key, text in models:
            self.model_combo.addItem(text, key)
        self.model_combo.setToolTip(tooltip)
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
        self._claim_if_spin_system()
        self.note_label.setText(note)
        self.note_label.setVisible(bool(note))

    def _claim_if_spin_system(self) -> None:
        if self._claim is not None and self.model() == "spin_system":
            self._claim()

    def _on_model(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self._claim_if_spin_system()
        self.note_label.setVisible(False)
