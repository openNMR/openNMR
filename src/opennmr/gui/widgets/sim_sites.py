"""
The list of sites for a one-interaction simulator (Simulate ▸ CSA,
Simulate ▸ Quadrupolar): explicit user request, so that a spectrum fitted
with several sites in Fit can be simulated whole in Simulate, adjusted,
and sent back to Fit as starting values with every site in place.

The simulator's own per-site controls (δiso, span/Δ, FWHM, ...) stay
where they were -- plain spin boxes in the panel, which is what the rest
of the window, the logs and the tests address -- and become the editor
of the *selected* site: :meth:`SimSiteList.bind` ties each one to a key
of the site dicts held here. Selecting a site loads its values into
them; editing one writes back into that site. The label and amount rows
belong to this widget and head the "Selected site" form.

State: the sites live in :attr:`SimSiteList.site_dicts` (plain dicts,
keys as the notebook's ``*_multisite_spectrum`` functions take them), so
a panel's saved state is ``get_state()`` next to its widgets. While a
panel's widgets are being restored, :meth:`loading` stops the bound
editors from writing into whichever site happens to be selected.
"""

from __future__ import annotations

import contextlib
import copy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLineEdit, QListView, QListWidget, QPushButton, QVBoxLayout, QWidget,
)


class SimSiteList(QWidget):
    """Sites box (list + Add / Duplicate / Remove) and a "Selected site"
    box whose form starts with Label and Amount; the panel adds its own
    per-site rows to :attr:`site_form` and :meth:`bind`\\ s them."""

    #: Any change to the sites: a value, a label, adding or removing one.
    changed = Signal()
    #: The selected site changed (its index).
    selected = Signal(int)

    def __init__(self, defaults: dict, describe=None, amount_tip: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._defaults = dict(defaults)
        self._describe = describe or (lambda site: site.get("label", ""))
        self._editors: dict[str, QWidget] = {}
        self._loading = 0
        self.site_dicts: list[dict] = []
        self._current = -1

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        sites_box = QGroupBox("Sites")
        sites_layout = QHBoxLayout(sites_box)
        self.site_list = QListWidget()
        self.site_list.setMaximumHeight(110)
        # Entries wrap to the column's width rather than scroll sideways.
        self.site_list.setWordWrap(True)
        self.site_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.site_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.site_list.setToolTip(
            "Every site is simulated with the settings above and summed, weighted by its "
            "amount. Select one to edit it below.")
        self.site_list.currentRowChanged.connect(self._on_row_changed)
        sites_layout.addWidget(self.site_list, 1)
        buttons = QVBoxLayout()
        self.add_button = QPushButton("Add")
        self.add_button.setToolTip("Add a site with default values.")
        self.add_button.clicked.connect(lambda: self.add_site())
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.setToolTip("Add a copy of the selected site, to adjust.")
        self.duplicate_button.clicked.connect(self.duplicate_site)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove the selected site (one always stays).")
        self.remove_button.clicked.connect(self.remove_site)
        for button in (self.add_button, self.duplicate_button, self.remove_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        sites_layout.addLayout(buttons)
        v.addWidget(sites_box)

        self.site_box = QGroupBox("Selected site")
        self.site_form = QFormLayout(self.site_box)
        self.label_edit = QLineEdit()
        self.label_edit.setToolTip("The site's name, used in the legend, the log and Fit.")
        self.label_edit.textEdited.connect(self._on_label_edited)
        self.site_form.addRow("Label:", self.label_edit)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(0.0, 1.0e6)
        self.amount_spin.setDecimals(3)
        self.amount_spin.setSingleStep(0.1)
        self.amount_spin.setValue(1.0)
        self.amount_spin.setToolTip(amount_tip or (
            "Relative number of nuclei in this site: its integrated intensity relative to "
            "the other sites'. Sent from a fit, it is the fitted share (largest = 1)."))
        self.amount_spin.valueChanged.connect(lambda value: self._write("amount", float(value)))
        self.site_form.addRow("Amount:", self.amount_spin)
        v.addWidget(self.site_box)

        self.add_site()

    # ---- editors ------------------------------------------------------------

    def bind(self, key: str, widget: QWidget) -> None:
        """``widget`` (a spin box or combo) edits ``key`` of the selected
        site from now on."""
        self._editors[key] = widget
        if isinstance(widget, QAbstractSpinBox):
            widget.valueChanged.connect(lambda value, k=key: self._write(k, float(value)))
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(
                lambda _i, k=key, w=widget: self._write(k, w.currentData()))
        else:
            raise TypeError(f"Cannot bind a {type(widget).__name__}.")
        current = self.current_site()
        if current is not None and key in current:
            self._load_editor(key, current[key])

    @contextlib.contextmanager
    def loading(self):
        """Editors change without writing into a site (e.g. while the
        panel's widgets are restored)."""
        self._loading += 1
        try:
            yield
        finally:
            self._loading -= 1

    def _write(self, key: str, value) -> None:
        if self._loading or not (0 <= self._current < len(self.site_dicts)):
            return
        self.site_dicts[self._current][key] = value
        self._refresh_item(self._current)
        self.changed.emit()

    def _on_label_edited(self, text: str) -> None:
        if self._loading or not (0 <= self._current < len(self.site_dicts)):
            return
        self.site_dicts[self._current]["label"] = text.strip() or f"Site {self._current + 1}"
        self._refresh_item(self._current)
        self.changed.emit()

    def _load_editor(self, key: str, value) -> None:
        widget = self._editors[key]
        blocked = widget.blockSignals(True)
        try:
            if isinstance(widget, QAbstractSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QComboBox):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
        finally:
            widget.blockSignals(blocked)

    def _load_current(self) -> None:
        site = self.current_site()
        if site is None:
            return
        # Signals stay connected for editors (their own dependents --
        # e.g. an auto-framed axis -- must follow the new site); only the
        # write-back is suppressed.
        with self.loading():
            self.label_edit.setText(str(site.get("label", "")))
            self.amount_spin.setValue(float(site.get("amount", 1.0)))
            for key, widget in self._editors.items():
                if key not in site:
                    continue
                if isinstance(widget, QAbstractSpinBox):
                    widget.setValue(float(site[key]))
                elif isinstance(widget, QComboBox):
                    index = widget.findData(site[key])
                    if index >= 0:
                        widget.setCurrentIndex(index)

    def sync_from_editors(self) -> None:
        """Copy the editors' current values into the selected site (for a
        state saved before sites existed: its spin boxes *are* the site)."""
        site = self.current_site()
        if site is None:
            return
        for key, widget in self._editors.items():
            if isinstance(widget, QAbstractSpinBox):
                site[key] = float(widget.value())
            elif isinstance(widget, QComboBox):
                site[key] = widget.currentData()
        self._refresh_item(self._current)

    def reset_from_editors(self) -> None:
        """One site, holding what the editors show now (a state saved
        before sites existed restores only the editors)."""
        values = {}
        for key, widget in self._editors.items():
            if isinstance(widget, QAbstractSpinBox):
                values[key] = float(widget.value())
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentData()
        with self.loading():
            self.set_sites([values])

    def editor_widgets(self) -> list[QWidget]:
        """The bound per-site editors (not kept as panel widgets: they
        are the selected site's, and the sites are kept whole)."""
        return list(self._editors.values())

    # ---- sites ---------------------------------------------------------------

    def current(self) -> int:
        return self._current

    def current_site(self) -> dict | None:
        return self.site_dicts[self._current] if 0 <= self._current < len(self.site_dicts) \
            else None

    def sites(self) -> list[dict]:
        return copy.deepcopy(self.site_dicts)

    def _new_label(self) -> str:
        taken = {s.get("label") for s in self.site_dicts}
        k = len(self.site_dicts) + 1
        while f"Site {k}" in taken:
            k += 1
        return f"Site {k}"

    def add_site(self, values: dict | None = None) -> None:
        site = {**self._defaults, "amount": 1.0, **(values or {})}
        site.setdefault("label", None)
        if not site["label"]:
            site["label"] = self._new_label()
        self.site_dicts.append(site)
        self.site_list.addItem(self._describe(site))
        self.site_list.setCurrentRow(len(self.site_dicts) - 1)
        self._sync_buttons()
        self.changed.emit()

    def duplicate_site(self) -> None:
        site = self.current_site()
        if site is None:
            return
        self.add_site({**copy.deepcopy(site), "label": None})

    def remove_site(self) -> None:
        if len(self.site_dicts) <= 1 or not (0 <= self._current < len(self.site_dicts)):
            return
        row = self._current
        del self.site_dicts[row]
        # The list's own current-row change re-selects; keep _current in
        # range first so _on_row_changed reads a valid site.
        self._current = min(row, len(self.site_dicts) - 1)
        self.site_list.blockSignals(True)
        self.site_list.takeItem(row)
        self.site_list.setCurrentRow(self._current)
        self.site_list.blockSignals(False)
        self._load_current()
        self._sync_buttons()
        self.selected.emit(self._current)
        self.changed.emit()

    def set_sites(self, sites: list[dict], current: int = 0) -> None:
        """Replace every site (at least one is kept: an empty list gives
        one default site)."""
        sites = [{**self._defaults, "amount": 1.0, **dict(s)} for s in (sites or [])] or \
            [{**self._defaults, "amount": 1.0}]
        for k, site in enumerate(sites):
            if not site.get("label"):
                site["label"] = f"Site {k + 1}"
        self.site_dicts = sites
        self.site_list.blockSignals(True)
        self.site_list.clear()
        for site in sites:
            self.site_list.addItem(self._describe(site))
        self._current = max(0, min(int(current), len(sites) - 1))
        self.site_list.setCurrentRow(self._current)
        self.site_list.blockSignals(False)
        self._load_current()
        self._sync_buttons()
        self.selected.emit(self._current)
        self.changed.emit()

    def select(self, index: int) -> None:
        if 0 <= index < len(self.site_dicts):
            self.site_list.setCurrentRow(index)

    def _on_row_changed(self, row: int) -> None:
        if not (0 <= row < len(self.site_dicts)):
            return
        self._current = row
        self._load_current()
        self.selected.emit(row)

    def _refresh_item(self, row: int) -> None:
        item = self.site_list.item(row)
        if item is not None:
            item.setText(self._describe(self.site_dicts[row]))

    def refresh(self) -> None:
        """Re-describe every site (e.g. after a shared setting that the
        description mentions, such as the CSA convention, changed)."""
        for row in range(len(self.site_dicts)):
            self._refresh_item(row)

    def _sync_buttons(self) -> None:
        self.remove_button.setEnabled(len(self.site_dicts) > 1)

    # ---- state -----------------------------------------------------------------

    def get_state(self) -> dict:
        return {"sites": self.sites(), "current": self._current}

    def set_state(self, state: dict | None) -> None:
        state = state or {}
        if state.get("sites"):
            self.set_sites(state["sites"], state.get("current", 0))
