"""
Compare > 2D spectra: how 2D spectra in a comparison are drawn --
overlaid, or side by side in a row/column/grid -- and zoomed insets on
either.

The GUI side of the notebook's ``NMR2D_overlay_plot`` /
``NMR2D_multi_plot`` / ``inset=``. Drawing is
:mod:`opennmr.plotting.twod` (``overlay_plot``, ``multi_plot``,
``draw_insets``); this widget only holds the choices and the list of
insets, and says when the plot needs redrawing.

An inset is made by zooming the Compare plot onto a region and pressing
*Add inset from zoom*; its region, corner, size, lowest contour and
connector box can then be edited here.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QPushButton, QVBoxLayout, QWidget,
)

from .widgets.typed_spinbox import TypedSpinBox
from ..plotting.twod import INSET_CORNERS, MAX_OVERLAY_SPECTRA

LAYOUTS = (("Overlay", "overlay"), ("Side by side: row", "row"),
           ("Side by side: column", "column"), ("Side by side: grid", "grid"))
_CORNERS = tuple(INSET_CORNERS)


class Compare2DControls(QGroupBox):
    """Layout, contour levels and insets for 2D spectra in Compare.

    ``changed`` : the plot should be redrawn.
    ``add_inset_requested`` : the owner should read the zoomed region
    off the canvas and call :meth:`add_inset`.
    """

    changed = Signal()
    add_inset_requested = Signal()
    #: The "Link panels" box was ticked or cleared: applied to the plot at
    #: once, without redrawing it (a redraw would put back the full view).
    link_toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__("2D spectra", parent)
        self._insets: list[dict] = []
        self._panel_count = 1
        self._loading = False
        self._copies_seen = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.layout_combo = QComboBox()
        for text, key in LAYOUTS:
            self.layout_combo.addItem(text, key)
        self.layout_combo.setToolTip(
            f"Overlay: every checked 2D on one set of axes, one colour each "
            f"(2-{MAX_OVERLAY_SPECTRA} spectra with the same nuclei). Side by side: one "
            f"panel each. An overlay that isn't possible is shown side by side, with the reason.")
        form.addRow("Layout:", self.layout_combo)
        self.link_check = QCheckBox("Link panels (zoom and scroll together)")
        self.link_check.setChecked(True)
        self.link_check.setToolTip(
            "Side by side: linked panels show the same region, and zooming, panning or scrolling "
            "an axis on one does the same on all. Unlinked, only the panel you act on moves. "
            "Takes effect at once.")
        self.link_check.toggled.connect(self.link_toggled)
        form.addRow(self.link_check)
        self.copies_spin = TypedSpinBox()
        self.copies_spin.setRange(1, 4)
        self.copies_spin.setValue(1)
        self.copies_spin.setToolTip(
            "Show each checked spectrum this many times, side by side, to compare different "
            "regions of the same spectrum: with 2 or more the panels are unlinked so each can be "
            "zoomed on its own (tick Link panels to move them together again). An overlay of a "
            "spectrum on itself would show nothing, so these are drawn side by side.")
        self.copies_spin.valueChanged.connect(self._on_copies_changed)
        form.addRow("Panels per spectrum:", self.copies_spin)
        self.projections_check = QCheckBox("Projections")
        self.projections_check.setToolTip(
            "F1/F2 (positive) projections: one beside each panel, or, in an overlay, one trace "
            "per spectrum in its colour. Scroll over the contours to raise or lower the lowest contour.")
        form.addRow(self.projections_check)
        self.floor_spin = QDoubleSpinBox()
        self.floor_spin.setRange(0.01, 90.0)
        self.floor_spin.setDecimals(2)
        self.floor_spin.setValue(7.0)
        self.floor_spin.setSuffix(" %")
        self.floor_spin.setToolTip("Lowest contour, as a percentage of each spectrum's own maximum.")
        form.addRow("Lowest contour:", self.floor_spin)
        self.levels_spin = TypedSpinBox()
        self.levels_spin.setRange(2, 40)
        self.levels_spin.setValue(7)
        form.addRow("Contour levels:", self.levels_spin)
        layout.addLayout(form)

        # --- insets -----------------------------------------------------
        insets_box = QGroupBox("Insets")
        insets_layout = QVBoxLayout(insets_box)
        hint = QLabel("Zoom the plot onto a region (magnifier on the plot toolbar), then Add inset.")
        hint.setWordWrap(True)
        insets_layout.addWidget(hint)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add inset from zoom")
        self.add_button.clicked.connect(self.add_inset_requested)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_insets)
        for button in (self.add_button, self.remove_button, self.clear_button):
            buttons.addWidget(button)
        insets_layout.addLayout(buttons)
        self.inset_list = QListWidget()
        self.inset_list.setMaximumHeight(90)
        self.inset_list.currentRowChanged.connect(self._load_editor)
        insets_layout.addWidget(self.inset_list)

        self.editor = QWidget()
        edit_form = QFormLayout(self.editor)
        edit_form.setContentsMargins(0, 0, 0, 0)
        self.panel_combo = QComboBox()
        edit_form.addRow("On panel:", self.panel_combo)
        self.f2_from, self.f2_to = self._ppm_spin(), self._ppm_spin()
        self.f1_from, self.f1_to = self._ppm_spin(), self._ppm_spin()
        edit_form.addRow("F2 from/to:", self._pair(self.f2_from, self.f2_to))
        edit_form.addRow("F1 from/to:", self._pair(self.f1_from, self.f1_to))
        self.corner_combo = QComboBox()
        self.corner_combo.addItems(_CORNERS)
        edit_form.addRow("Corner:", self.corner_combo)
        self.size_spin = TypedSpinBox()
        self.size_spin.setRange(10, 90)
        self.size_spin.setSuffix(" %")
        self.size_spin.setToolTip("Side of the (square) inset, as a percentage of the panel.")
        edit_form.addRow("Size:", self.size_spin)
        self.inset_floor_spin = QDoubleSpinBox()
        self.inset_floor_spin.setRange(0.0, 90.0)
        self.inset_floor_spin.setDecimals(2)
        self.inset_floor_spin.setSuffix(" %")
        self.inset_floor_spin.setSpecialValueText("Same as plot")
        self.inset_floor_spin.setToolTip(
            "Lowest contour in this inset, as a percentage of the whole spectrum's maximum – "
            "lower it to bring out weak peaks. Levels are always relative to the whole spectrum, "
            "so a contour means the same intensity in the inset and the plot.")
        edit_form.addRow("Lowest contour:", self.inset_floor_spin)
        self.mark_check = QCheckBox("Box and connectors on the plot")
        edit_form.addRow(self.mark_check)
        insets_layout.addWidget(self.editor)
        layout.addWidget(insets_box)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self.changed)

        for widget in (self.layout_combo, self.corner_combo, self.panel_combo):
            widget.currentIndexChanged.connect(self._on_edit)
        for widget in (self.projections_check, self.mark_check):
            widget.toggled.connect(self._on_edit)
        for widget in (self.floor_spin, self.levels_spin, self.copies_spin, self.f2_from,
                       self.f2_to, self.f1_from, self.f1_to, self.size_spin, self.inset_floor_spin):
            widget.valueChanged.connect(self._on_edit)
        self.set_panel_count(1)
        self._refresh_list()

    # --- public -------------------------------------------------------
    def layout_key(self) -> str:
        return self.layout_combo.currentData()

    def copies(self) -> int:
        """Panels each checked spectrum is drawn in (to compare regions of
        one spectrum with itself)."""
        return self.copies_spin.value()

    def _on_copies_changed(self, value: int) -> None:
        # Comparing two regions of one spectrum needs panels that zoom
        # separately: the first time copies are asked for, unlink.
        if value > 1 and self.link_check.isChecked() and not self._copies_seen:
            self.link_check.setChecked(False)
        self._copies_seen = value > 1

    def plot_options(self) -> dict:
        return {"factor": self.floor_spin.value() / 100.0, "clevels": self.levels_spin.value()}

    def insets(self) -> list[dict]:
        return [dict(inset) for inset in self._insets]

    def set_panel_count(self, n: int) -> None:
        """How many panels the current comparison has (for the
        per-inset panel choice)."""
        self._panel_count = max(1, int(n))
        self._loading = True
        try:
            current = self.panel_combo.currentData()
            self.panel_combo.clear()
            self.panel_combo.addItem("All panels", None)
            for i in range(self._panel_count):
                self.panel_combo.addItem(f"Panel {i + 1}", i)
            index = self.panel_combo.findData(current)
            self.panel_combo.setCurrentIndex(max(index, 0))
        finally:
            self._loading = False

    def add_inset(self, panel, xlims, ylims, corner: str | None = None,
                  factor: float | None = None) -> dict:
        """Add an inset showing ``xlims`` (F2) x ``ylims`` (F1) on
        ``panel`` (0-based, or ``None`` for every panel), in ``corner``
        (default: the next free one)."""
        if corner is None:
            used = {inset["corner"] for inset in self._insets}
            corner = next((c for c in _CORNERS if c not in used), _CORNERS[0])
        inset = {"panel": panel, "xlims": tuple(sorted(xlims, reverse=True)),
                 "ylims": tuple(sorted(ylims)), "corner": corner, "size": 0.4,
                 "factor": factor, "mark_inset": True}
        self._insets.append(inset)
        self._refresh_list()
        self.inset_list.setCurrentRow(len(self._insets) - 1)
        self.changed.emit()
        return inset

    def inset_specs(self, panel: int | None = None) -> list[dict]:
        """The plotting specs (:func:`opennmr.plotting.twod.draw_insets`)
        for ``panel``, or every inset when ``panel`` is ``None`` (an
        overlay has one panel)."""
        out = []
        for inset in self._insets:
            if panel is not None and inset["panel"] not in (None, panel):
                continue
            out.append({"xlims": inset["xlims"], "ylims": inset["ylims"], "corner": inset["corner"],
                        "size": inset["size"], "factor": inset["factor"],
                        "mark_inset": inset["mark_inset"]})
        return out

    def remove_selected(self) -> None:
        row = self.inset_list.currentRow()
        if 0 <= row < len(self._insets):
            del self._insets[row]
            self._refresh_list()
            self.changed.emit()

    def clear_insets(self) -> None:
        if self._insets:
            self._insets.clear()
            self._refresh_list()
            self.changed.emit()

    # --- internals ----------------------------------------------------
    @staticmethod
    def _ppm_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        return spin

    @staticmethod
    def _pair(a, b) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(a)
        row.addWidget(b)
        return widget

    @staticmethod
    def _describe(i: int, inset: dict) -> str:
        where = "all panels" if inset["panel"] is None else f"panel {inset['panel'] + 1}"
        (x1, x2), (y1, y2) = inset["xlims"], inset["ylims"]
        return f"Inset {i + 1}: F2 {x1:.3g}–{x2:.3g}, F1 {y1:.3g}–{y2:.3g} ({where})"

    def _refresh_list(self) -> None:
        row = self.inset_list.currentRow()
        self.inset_list.blockSignals(True)
        self.inset_list.clear()
        for i, inset in enumerate(self._insets):
            self.inset_list.addItem(self._describe(i, inset))
        self.inset_list.blockSignals(False)
        has = bool(self._insets)
        self.editor.setEnabled(has)
        self.remove_button.setEnabled(has)
        self.clear_button.setEnabled(has)
        if has:
            self.inset_list.setCurrentRow(min(max(row, 0), len(self._insets) - 1))
            self._load_editor(self.inset_list.currentRow())

    def _load_editor(self, row: int) -> None:
        if not 0 <= row < len(self._insets):
            return
        inset = self._insets[row]
        self._loading = True
        try:
            self.panel_combo.setCurrentIndex(max(self.panel_combo.findData(inset["panel"]), 0))
            self.f2_from.setValue(inset["xlims"][0])
            self.f2_to.setValue(inset["xlims"][1])
            self.f1_from.setValue(inset["ylims"][0])
            self.f1_to.setValue(inset["ylims"][1])
            self.corner_combo.setCurrentText(inset["corner"])
            self.size_spin.setValue(round(inset["size"] * 100))
            self.inset_floor_spin.setValue((inset["factor"] or 0.0) * 100.0)
            self.mark_check.setChecked(inset["mark_inset"])
        finally:
            self._loading = False

    def _on_edit(self, *_args) -> None:
        if self._loading:
            return
        row = self.inset_list.currentRow()
        sender = self.sender()
        inset_widgets = (self.panel_combo, self.f2_from, self.f2_to, self.f1_from, self.f1_to,
                         self.corner_combo, self.size_spin, self.inset_floor_spin, self.mark_check)
        if sender in inset_widgets and 0 <= row < len(self._insets):
            inset = self._insets[row]
            inset["panel"] = self.panel_combo.currentData()
            inset["xlims"] = (self.f2_from.value(), self.f2_to.value())
            inset["ylims"] = (self.f1_from.value(), self.f1_to.value())
            inset["corner"] = self.corner_combo.currentText()
            inset["size"] = self.size_spin.value() / 100.0
            inset["factor"] = (self.inset_floor_spin.value() / 100.0) or None
            inset["mark_inset"] = self.mark_check.isChecked()
            self.inset_list.item(row).setText(self._describe(row, inset))
        self._debounce.start()
