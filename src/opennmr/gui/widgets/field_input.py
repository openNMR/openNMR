"""
Enter the magnet's field (T); the Larmor frequency follows from the
nucleus.

Simulate panels asked for the observed nucleus's Larmor frequency in MHz
-- a number that changes with every nucleus at the same magnet, and one
people had to work out first. Explicit user request (2026-09-27): enter
the nucleus and the field in tesla instead.

:func:`link_field` adds a *Field* box in front of a panel's existing
Larmor box and ties them together:

* editing the field (or changing the nucleus) sets the Larmor frequency,
  nu0 = (gamma / 2 pi) B0, from openNMR's gyromagnetic-ratio table --
  the magnet stays put when the nucleus changes, as in reality;
* the Larmor box becomes a read-only display of the result, and stays
  the value every simulation reads, so no simulation code changes;
* anything that *sets* the Larmor frequency -- seeding from a selected
  spectrum's SFO1, restoring a saved panel -- moves the field to match,
  so the two never disagree.

Beside the field is the same magnet as its 1H frequency ("400.000 MHz
1H"), which is how spectrometers are usually named -- an editable box
since 2026-10-13 (explicit user request: "allow a user to choose the 1H
frequency in MHz, as in the NMR Periodic Table"): type either, the other
follows. Both accept typing freely (:class:`TypedDoubleSpinBox`).
"""

from __future__ import annotations

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QAbstractSpinBox, QHBoxLayout, QLabel, QWidget

from ...notebook.utils import find_gamma
from .typed_spinbox import TypedDoubleSpinBox

#: A "400 MHz" magnet: the default when nothing better is known.
DEFAULT_FIELD_T = 9.3947
#: Highest field a field box takes (T): above today's 45 T hybrid magnets.
MAX_FIELD_T = 50.0

_GAMMA_1H = None


def gamma_hz_per_t(nucleus: str) -> float | None:
    """gamma / 2 pi of ``nucleus`` in Hz/T, or None if it is not a
    recognised nucleus."""
    text = (nucleus or "").strip()
    if not text:
        return None
    try:
        gamma = (find_gamma(text, quiet=True) or {}).get("gamma")
    except Exception:  # noqa: BLE001 -- an unrecognised name is simply unknown
        return None
    return abs(float(gamma)) if gamma else None


def _gamma_1h() -> float:
    global _GAMMA_1H
    if _GAMMA_1H is None:
        _GAMMA_1H = gamma_hz_per_t("1H")
    return _GAMMA_1H


class FieldSpinBox(TypedDoubleSpinBox):
    """A field box: typed freely, and says (``userEdited``) when the
    *user* changed it -- typing, the arrow keys or its buttons -- as
    opposed to code setting it."""


def proton_frequency_box(field_spin, parent=None) -> TypedDoubleSpinBox:
    """A box for the same magnet as its 1H frequency (MHz), tied both ways
    to ``field_spin`` (T): typing either sets the other. Editing it counts
    as the user setting the field (``field_spin.userEdited``)."""
    box = TypedDoubleSpinBox(parent)
    box.setDecimals(3)
    box.setRange(0.0, _gamma_1h() * field_spin.maximum() / 1e6)
    box.setSingleStep(1.0)
    box.setSuffix(" MHz ¹H")
    box.setToolTip("The same magnet as its ¹H frequency -- how spectrometers are usually "
                   "named (400 MHz = 9.3946 T). Type either this or the field.")
    state = {"busy": False}

    def from_field(_v=None):
        if state["busy"]:
            return
        box.blockSignals(True)
        box.setValue(_gamma_1h() * field_spin.value() / 1e6)
        box.blockSignals(False)

    def to_field(_v=None):
        state["busy"] = True
        try:
            field_spin.setValue(box.value() * 1e6 / _gamma_1h())
        finally:
            state["busy"] = False
        if hasattr(field_spin, "userEdited"):
            field_spin.userEdited.emit()

    field_spin.valueChanged.connect(from_field)
    box.valueChanged.connect(to_field)
    from_field()
    return box


class FieldLink(QObject):
    """Keeps ``field_spin`` (T) and ``larmor_spin`` (MHz) consistent for
    the nucleus ``nucleus()`` returns. See the module docstring."""

    def __init__(self, field_spin, larmor_spin, nucleus, proton_label=None, parent=None):
        super().__init__(parent or field_spin)
        self.field_spin = field_spin
        self.larmor_spin = larmor_spin
        self._nucleus = nucleus
        self._proton_label = proton_label
        self._busy = False
        #: True once the user has set the field themselves: a selected
        #: spectrum then no longer re-seeds it.
        self.user_set = False
        field_spin.valueChanged.connect(lambda _v: self.field_changed())
        if hasattr(field_spin, "userEdited"):
            field_spin.userEdited.connect(self._mark_user_set)
        larmor_spin.valueChanged.connect(lambda _v: self.larmor_changed())

    def gamma(self):
        return gamma_hz_per_t(self._nucleus())

    def _set_quietly(self, spin, value):
        self._busy = True
        try:
            spin.setValue(value)
        finally:
            self._busy = False

    def field_changed(self) -> None:
        """Field or nucleus changed: recompute the Larmor frequency."""
        if self._busy:
            return
        self._recompute()

    def _mark_user_set(self) -> None:
        self.user_set = True

    def nucleus_changed(self) -> None:
        self._recompute()

    def _recompute(self) -> None:
        gamma = self.gamma()
        if gamma is not None:
            self._set_quietly(self.larmor_spin, gamma * self.field_spin.value() / 1e6)
        self._update_label()

    def seed_from_spectrum(self, sfo1_mhz, nucleus) -> bool:
        """Set the field from a spectrum -- B0 = SFO1 / (gamma/2pi) of the
        spectrum's *own* nucleus, so it is right whatever nucleus the panel
        simulates -- unless the user has set the field. Returns whether it
        did."""
        gamma = gamma_hz_per_t(nucleus)
        if self.user_set or not sfo1_mhz or gamma is None:
            return False
        self._set_quietly(self.field_spin, float(sfo1_mhz) * 1e6 / gamma)
        self._recompute()
        return True

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 -- mirrors QWidget
        self.field_spin.setEnabled(enabled)
        self.larmor_spin.setEnabled(enabled)

    def larmor_changed(self) -> None:
        """The Larmor frequency was set from outside (a spectrum's SFO1, a
        restored panel): move the field to match."""
        if self._busy:
            return
        gamma = self.gamma()
        if gamma is not None and self.larmor_spin.value() > 0:
            self._set_quietly(self.field_spin, self.larmor_spin.value() * 1e6 / gamma)
        self._update_label()

    def _update_label(self) -> None:
        if self._proton_label is None or not hasattr(self._proton_label, "setText"):
            return
        field = self.field_spin.value()
        self._proton_label.setText(f"{_gamma_1h() * field / 1e6:.2f} MHz ¹H" if field > 0 else "")


def link_field(form, larmor_spin, nucleus, nucleus_changed_signal=None, default_t=None):
    """Put a *Field (T)* row before ``larmor_spin``'s row in QFormLayout
    ``form``, make the Larmor box a read-only result, and link them.
    ``nucleus()`` returns the observed nucleus ("27Al"); pass the signal
    that fires when it changes. Returns the :class:`FieldLink` (its
    ``field_spin`` is the new box)."""
    field_spin = FieldSpinBox()
    field_spin.setRange(0.0, MAX_FIELD_T)
    # Six decimals (1 µT), as the Periodic Table's: at four, a "400 MHz"
    # magnet typed as ¹H would read back as 399.998 MHz.
    field_spin.setDecimals(6)
    field_spin.setSingleStep(0.1)
    field_spin.setSuffix(" T")
    field_spin.setToolTip(
        "The magnet's field, B0 -- or type its ¹H frequency in the box beside it. The Larmor "
        "frequency below follows from it and the nucleus (9.3946 T is a '400 MHz' "
        "spectrometer).")
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(field_spin, 1)
    h.addWidget(QLabel("or"))
    proton = proton_frequency_box(field_spin, parent=row)
    h.addWidget(proton, 1)

    position, _role = form.getWidgetPosition(larmor_spin)
    form.insertRow(position, "Field B₀:", row)
    label = form.labelForField(larmor_spin)
    if label is not None and hasattr(label, "setText"):
        label.setText("Larmor frequency:")
    larmor_spin.setReadOnly(True)
    larmor_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    larmor_spin.setToolTip(
        "The observed nucleus's Larmor frequency, calculated from the nucleus and the field "
        "above (γ/2π × B0). Set by a selected spectrum's SFO1 where there is one -- the field "
        "then follows.")

    link = FieldLink(field_spin, larmor_spin, nucleus, parent=row)
    link.field_spin = field_spin
    link.proton_spin = proton
    if nucleus_changed_signal is not None:
        nucleus_changed_signal.connect(lambda *_a: link.nucleus_changed())
    gamma = link.gamma()
    if larmor_spin.value() > 0 and gamma is not None:
        link.larmor_changed()                     # keep the panel's existing default
    else:
        field_spin.setValue(default_t if default_t is not None else DEFAULT_FIELD_T)
    link._update_label()
    return link
