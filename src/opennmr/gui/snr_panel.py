"""
Integrate ▸ Signal-to-Noise: the S/N of one or more 1D or 2D spectra, with
the statistics stated (explicit user request, 2026-10-12; 2D and the
conventions side by side, 2026-10-14).

1. **Spectra** -- tick the spectra (1D and genuine 2D spectra in Open
   experiments; a 2D's boxes take their F1 side from *Signal F1* /
   *Noise F1*).
2. **Regions** -- a signal region and a noise region, typed (``12, 8``;
   several noise windows separated by ``;``) or dragged on the plot.
   By default every spectrum uses the same regions. Tick *Different
   regions for some spectra* and choose a spectrum in *Set regions for*
   to give that one its own; the table marks it †, the log records it,
   and the result says, spectrum by spectrum, which regions each used.
3. **Calculate** -- :func:`opennmr.notebook.snr.signal_to_noise_many`,
   as a notebook calls it: S/N ± its uncertainty and a 95 % interval,
   TopSpin's and the Pharmacopoeia's conventions, the integral S/N, a
   detection test, checks on the noise region, and each spectrum against
   a reference (ratio, significance, √NS). Kept with each spectrum (its
   log and sidecar) like a fit; exportable as CSV and as notebook code.
"""

from __future__ import annotations

import math
import re
import time

from .widgets.report_label import ReportLabel
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..notebook import snr

#: What an armed drag on the plot sets.
DRAG_SIGNAL, DRAG_NOISE = "signal", "noise"
#: The "Set regions for" entry meaning every spectrum.
SHARED = "__shared__"

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_window(text: str) -> tuple[float, float]:
    """'12, 8', '12 8', '12 to 8', '12–8' or '12 - 8' -> (12.0, 8.0);
    '5 -5' and '5, -5' -> (5.0, -5.0). A hyphen written between two
    numbers ('12-8', '12 - 8') is a range; one written against the number
    after a space ('5 -5') is a minus sign."""
    t = str(text).strip()
    t = t.replace("−", "-").replace("–", " to ").replace("—", " to ")
    t = re.sub(r"(?<=[\d.])-(?=[\d.])", " to ", t)              # 12-8
    t = re.sub(r"(?<=[\d.])\s+-\s+(?=[\d.])", " to ", t)        # 12 - 8
    numbers = re.findall(_NUMBER, t.replace(" to ", " "))
    if len(numbers) != 2:
        raise ValueError(f"'{text}' is not two numbers (ppm), e.g. '12, 8'.")
    a, b = float(numbers[0]), float(numbers[1])
    if a == b:
        raise ValueError(f"'{text}': the two limits are the same.")
    return (max(a, b), min(a, b))


def parse_windows(text: str) -> list[tuple[float, float]]:
    """Several windows separated by ';' (noise only)."""
    parts = [p for p in str(text).split(";") if p.strip()]
    if not parts:
        raise ValueError("No region given.")
    return [parse_window(p) for p in parts]


def format_window(w) -> str:
    return f"{w[0]:g} to {w[1]:g}"


def format_windows(ws) -> str:
    return "; ".join(format_window(w) for w in ws)


def _item(text, align_right=False) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if align_right:
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def _elide(text: str, width: int = 16) -> str:
    """A long spectrum name shortened in the middle (the full name is the
    column's tooltip)."""
    text = str(text)
    if len(text) <= width:
        return text
    keep = width - 1
    return text[: (keep + 1) // 2] + "…" + text[-(keep // 2):]


class FitTable(QTableWidget):
    """A table exactly as tall as its rows: no inner vertical scroll bar,
    so the whole result shows at once (explicit user report, 2026-10-15:
    the result "is really hard to see"). Wider than the panel, it scrolls
    sideways, and its height allows for that scroll bar."""

    def __init__(self, parent=None) -> None:
        super().__init__(0, 0, parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft
                                                  | Qt.AlignmentFlag.AlignVCenter)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def content_height(self) -> int:
        rows = sum(self.rowHeight(i) for i in range(self.rowCount()))
        height = rows + 2 * self.frameWidth()
        if not self.horizontalHeader().isHidden():
            height += self.horizontalHeader().sizeHint().height()
        wide = self.verticalHeader().sizeHint().width() + self.horizontalHeader().length()
        if wide + 2 * self.frameWidth() > self.width():
            height += self.horizontalScrollBar().sizeHint().height()
        return height

    def sizeHint(self) -> QSize:
        return QSize(super().sizeHint().width(), self.content_height())

    def minimumSizeHint(self) -> QSize:
        return QSize(super().minimumSizeHint().width(), self.content_height())

    def refit(self) -> None:
        self.resizeRowsToContents()
        self.setFixedHeight(self.content_height())
        self.updateGeometry()

    def resizeEvent(self, event) -> None:                  # noqa: N802 -- Qt's name
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            height = self.content_height()
            if height != self.height():
                self.setFixedHeight(height)


class SnrReportDialog(QDialog):
    """The full S/N report -- the result table again (wide enough here for
    many spectra), then every number in sentences, the checks and the
    regions each spectrum used -- in its own resizable window rather than
    a long paragraph squeezed into the side panel."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Signal-to-Noise: Full Report")
        v = QVBoxLayout(self)
        self.table = FitTable()
        v.addWidget(self.table)
        self.label = ReportLabel("")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        inner = QVBoxLayout(holder)
        inner.addWidget(self.label)
        inner.addStretch(1)
        scroll.setWidget(holder)
        v.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.copy_button = buttons.addButton("Copy", QDialogButtonBox.ButtonRole.ActionRole)
        self.copy_button.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.label.text()))
        buttons.rejected.connect(self.close)
        v.addWidget(buttons)
        self.resize(680, 620)


def _g(value, digits=3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}g}"


class SnrPanel(QWidget):
    """Integrate ▸ Signal-to-Noise (see the module docstring).

    Callbacks (from MainWindow): ``spectra()`` -> ``[{"label", "spectrum",
    "index"}]`` (1D and 2D spectra in Open experiments); ``selected()`` -> the
    session rows selected there; ``draw(draw, view)``: show
    ``draw(figure)`` on the Compare view (``"spectrum"`` one,
    ``"spectra"`` several); ``arm(mode)``: the Compare view's ``"region"``
    drag, or None; ``store(kind, record, indices)``: keep a result with
    those spectra; ``style(view)``: the view's plot style."""

    COLUMNS = ("Spectrum", "Signal (ppm)", "Noise (ppm)", "S/N", "95 % interval",
               "S/N per √scan", "Note")

    #: A result was calculated (for tests and the main window).
    calculated = Signal()

    def __init__(self, spectra, selected, draw, arm, store, style=None, parent=None,
                 view_limits=None, colours=None) -> None:
        super().__init__(parent)
        #: ``colours(entries)`` -> one colour (or None) per ticked
        #: spectrum: the colour each has in Compare, so the S/N plot
        #: draws it the same (explicit user report, 2026-10-15: "the
        #: colour of the spectrum line changes"). None: the palette.
        self._colours = colours
        #: ``view_limits()`` -> the (left, right) ppm the plot shows now,
        #: or None: the S/N plot is drawn in that view rather than zoomed
        #: to the regions (explicit user report, 2026-10-15: picking a
        #: region zoomed in and the noise region was then off screen).
        self._view_limits = view_limits
        self._spectra, self._selected = spectra, selected
        self._draw, self._arm, self._store = draw, arm, store
        self._style = style or (lambda _view: {})
        self._rows: list[dict] = []
        self._checked_ids: set[int] | None = None
        self._entries: list[dict] = []
        #: Shared regions: (high, low) and a list of such.
        self.shared_signal: tuple[float, float] | None = None
        self.shared_noise: list[tuple[float, float]] | None = None
        #: id(spectrum) -> {"signal": window or None, "noise": [windows] or None}
        self.own_regions: dict[int, dict] = {}
        self._armed: str | None = None
        self.result: dict | None = None
        self._result_entries: list[dict] = []
        self._syncing = False

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Signal-to-noise of one or more spectra: a signal region (the line) and a noise "
            "region (no signal), typed or dragged on the plot. S/N = height above the noise "
            "level / the noise standard deviation, with its uncertainty – and the same "
            "measurement by the other conventions (TopSpin's ×½, ×⅓, Pharmacopoeia, integral "
            "or volume), side by side. 2D spectra: give the boxes' F1 sides too.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        v.addWidget(self._build_spectra())
        v.addWidget(self._build_regions())
        v.addWidget(self._build_results())
        v.addWidget(self._build_log())
        self.refresh()
        self._sync()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_spectra(self) -> QGroupBox:
        box = QGroupBox("1. Spectra")
        v = QVBoxLayout(box)
        self.spectra_table = QTableWidget(0, len(self.COLUMNS))
        self.spectra_table.setHorizontalHeaderLabels(self.COLUMNS)
        header = self.spectra_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.spectra_table.verticalHeader().setVisible(False)
        self.spectra_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.spectra_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.spectra_table.setMinimumHeight(140)
        self.spectra_table.setToolTip(
            "Tick the spectra to measure. † marks a spectrum with its own regions.")
        self.spectra_table.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.spectra_table)
        row = QHBoxLayout()
        self.use_selected_button = QPushButton("Use Selected")
        self.use_selected_button.setToolTip("Tick exactly the spectra selected in Open "
                                            "experiments (Ctrl/Shift-click to select several).")
        self.use_selected_button.clicked.connect(self.use_selected)
        self.all_button = QPushButton("All")
        self.all_button.clicked.connect(lambda: self._tick_all(True))
        self.none_button = QPushButton("None")
        self.none_button.clicked.connect(lambda: self._tick_all(False))
        for b in (self.use_selected_button, self.all_button, self.none_button):
            row.addWidget(b)
        v.addLayout(row)
        self.spectra_status = QLabel("")
        self.spectra_status.setWordWrap(True)
        v.addWidget(self.spectra_status)
        return box

    def _build_regions(self) -> QGroupBox:
        box = QGroupBox("2. Regions")
        v = QVBoxLayout(box)
        self.different_check = QCheckBox("Different regions for some spectra")
        self.different_check.setToolTip(
            "Off: every spectrum uses the same regions (recommended – S/N over different "
            "regions are not strictly comparable). On: choose a spectrum below and set its own; "
            "it is marked † and logged.")
        self.different_check.toggled.connect(self._on_different_toggled)
        v.addWidget(self.different_check)
        form = QFormLayout()
        self.target_combo = QComboBox()
        self.target_combo.setToolTip("Whose regions the boxes below and a drag on the plot set.")
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        form.addRow("Set regions for:", self.target_combo)
        self.signal_edit = QLineEdit()
        self.signal_edit.setPlaceholderText("e.g. 12, 8")
        self.signal_edit.setToolTip("The line: two limits in ppm, either order ('12, 8', "
                                    "'12 to 8').")
        self.signal_edit.editingFinished.connect(self._on_signal_typed)
        self.signal_drag_button = QPushButton("Drag…")
        self.signal_drag_button.setCheckable(True)
        self.signal_drag_button.setToolTip("Then drag across the line on the Compare view.")
        self.signal_drag_button.toggled.connect(lambda on: self._arm_for(DRAG_SIGNAL, on))
        row = QHBoxLayout()
        row.addWidget(self.signal_edit, 1)
        row.addWidget(self.signal_drag_button)
        form.addRow("Signal (ppm):", row)
        self.noise_edit = QLineEdit()
        self.noise_edit.setPlaceholderText("e.g. 60, 40  (several: 60, 40; -40, -60)")
        self.noise_edit.setToolTip(
            "Somewhere with no signal, no sidebands and no spikes: two limits in ppm; several "
            "windows separated by ';'. Wider is better – σ is known to ±1/√(2·independent "
            "points).")
        self.noise_edit.editingFinished.connect(self._on_noise_typed)
        self.noise_drag_button = QPushButton("Drag…")
        self.noise_drag_button.setCheckable(True)
        self.noise_drag_button.setToolTip("Then drag across a signal-free region on the Compare "
                                          "view (a drag replaces the noise windows; type ';' to "
                                          "add more).")
        self.noise_drag_button.toggled.connect(lambda on: self._arm_for(DRAG_NOISE, on))
        row = QHBoxLayout()
        row.addWidget(self.noise_edit, 1)
        row.addWidget(self.noise_drag_button)
        form.addRow("Noise (ppm):", row)
        # 2D spectra: the boxes' F1 sides (explicit user request,
        # 2026-10-14). Shared by every 2D; blank = all of F1.
        self.signal_f1_edit = QLineEdit()
        self.signal_f1_edit.setPlaceholderText("2D only – blank: all of F1")
        self.signal_f1_edit.setToolTip(
            "For a 2D spectrum, the signal box is the F2 range above by this F1 range (ppm). "
            "Blank: the whole of F1. Used by every 2D ticked.")
        self.signal_f1_edit.editingFinished.connect(lambda: self._on_f1_typed("signal"))
        form.addRow("Signal F1 (ppm):", self.signal_f1_edit)
        self.noise_f1_edit = QLineEdit()
        self.noise_f1_edit.setPlaceholderText("2D only – blank: all of F1")
        self.noise_f1_edit.setToolTip(
            "For a 2D spectrum, the noise box(es) are the F2 windows above by this F1 range "
            "(several: separate with ';'). Blank: the whole of F1. Pick a part with no peaks "
            "and no t1 noise ridges.")
        self.noise_f1_edit.editingFinished.connect(lambda: self._on_f1_typed("noise"))
        form.addRow("Noise F1 (ppm):", self.noise_f1_edit)
        self.reset_own_button = QPushButton("Use the Shared Regions")
        self.reset_own_button.setToolTip("Drop this spectrum's own regions.")
        self.reset_own_button.clicked.connect(self.reset_own)
        form.addRow("", self.reset_own_button)

        self.detrend_combo = QComboBox()
        for text, order, tip in (
                ("Linear (as TopSpin sino)", 1, "A straight line is fitted to the noise region "
                 "and removed before σ."),
                ("Offset only", 0, "Only the noise region's mean is removed."),
                ("Quadratic", 2, "A parabola is removed: for a curved baseline under the noise.")):
            self.detrend_combo.addItem(text, order)
            self.detrend_combo.setItemData(self.detrend_combo.count() - 1, tip,
                                           Qt.ItemDataRole.ToolTipRole)
        form.addRow("Noise trend removed:", self.detrend_combo)
        self.sign_combo = QComboBox()
        self.sign_combo.addItem("Positive", "positive")
        self.sign_combo.addItem("Negative (inverted)", "negative")
        self.sign_combo.addItem("Largest either way", "absolute")
        form.addRow("Line:", self.sign_combo)
        self.offset_combo = QComboBox()
        self.offset_combo.addItem("The noise region's mean", "mean")
        self.offset_combo.addItem("The noise region's trend, extended", "trend")
        self.offset_combo.addItem("Zero", "none")
        self.offset_combo.setToolTip("The level the line's height is measured from.")
        form.addRow("Height measured from:", self.offset_combo)
        v.addLayout(form)
        row = QHBoxLayout()
        self.show_button = QPushButton("Show Spectra")
        self.show_button.setToolTip("The ticked spectra on the Compare view, regions shaded "
                                    "(signal green, noise grey).")
        self.show_button.clicked.connect(self.show_spectra)
        self.calculate_button = QPushButton("Calculate S/N")
        self.calculate_button.clicked.connect(lambda: self.calculate(explicit=True))
        row.addWidget(self.show_button)
        row.addWidget(self.calculate_button)
        v.addLayout(row)
        self.regions_status = QLabel("")
        self.regions_status.setWordWrap(True)
        v.addWidget(self.regions_status)
        return box

    def _build_results(self) -> QGroupBox:
        box = QGroupBox("3. Result")
        v = QVBoxLayout(box)
        form = QFormLayout()
        self.reference_combo = QComboBox()
        self.reference_combo.setToolTip("Every other spectrum's S/N is compared with this one's.")
        self.reference_combo.currentIndexChanged.connect(
            lambda _i: self.result is not None and self.calculate())
        form.addRow("Compare against:", self.reference_combo)
        v.addLayout(form)
        # One compact table, a row per quantity and a column per spectrum
        # (snr.snr_summary): the main S/N, every convention side by side
        # (explicit user request, 2026-10-14: "evaluate these values and
        # present them to people to compare"), the noise, detection,
        # √scan and the comparison. It replaced a conventions table with
        # sentence-long row names and the whole report as a paragraph, which
        # needed a shrunk window and a widened panel to read (explicit
        # user report, 2026-10-15). The sentences are under Full Report….
        self.result_status = QLabel("")
        self.result_status.setWordWrap(True)
        v.addWidget(self.result_status)
        self.result_table = FitTable()
        self.result_table.setVisible(False)
        v.addWidget(self.result_table)
        self.result_hint = QLabel(
            "Hover a row's name for its formula and meaning, a value for its notes. Quote the "
            "convention with the number: they differ by fixed factors or by what is measured.")
        self.result_hint.setWordWrap(True)
        self.result_hint.setStyleSheet("color: palette(mid);")
        self.result_hint.setVisible(False)
        v.addWidget(self.result_hint)
        self.report_dialog = SnrReportDialog(self)
        #: The full report (plain text in ``text()``), shown by Full Report….
        self.result_label = self.report_dialog.label
        row = QHBoxLayout()
        self.report_button = QPushButton("Full Report…")
        self.report_button.setToolTip("Every number in sentences, with the checks and the "
                                      "regions each spectrum used, in its own window.")
        self.report_button.clicked.connect(self.show_report)
        row.addWidget(self.report_button)
        self.export_button = QPushButton("Export CSV…")
        self.export_button.clicked.connect(self._on_export)
        self.copy_report_button = QPushButton("Copy Report")
        self.copy_report_button.clicked.connect(self.copy_report)
        self.copy_code_button = QPushButton("Copy Notebook Code")
        self.copy_code_button.setToolTip("The notebook call that gives this result.")
        self.copy_code_button.clicked.connect(self.copy_code)
        for b in (self.export_button, self.copy_report_button, self.copy_code_button):
            row.addWidget(b)
        v.addLayout(row)
        return box

    def show_report(self) -> None:
        if self.result is None:
            return
        self.report_dialog.show()
        self.report_dialog.raise_()
        self.report_dialog.activateWindow()

    def _build_log(self) -> QGroupBox:
        box = QGroupBox("Log")
        v = QVBoxLayout(box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        v.addWidget(self.log_view)
        return box

    def _log(self, text: str) -> None:
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    # ------------------------------------------------------------------
    # spectra
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read Open experiments' 1D spectra, keeping what was ticked."""
        rows = self._spectra()
        table = self.spectra_table
        self._syncing = True
        table.setRowCount(len(rows))
        self._rows = rows
        live = {id(e["spectrum"]) for e in rows}
        self.own_regions = {k: v for k, v in self.own_regions.items() if k in live}
        for r, entry in enumerate(rows):
            name = QTableWidgetItem(entry["label"])
            name.setFlags((name.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                          & ~Qt.ItemFlag.ItemIsEditable)
            on = self._checked_ids is not None and id(entry["spectrum"]) in self._checked_ids
            name.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            table.setItem(r, 0, name)
        self._syncing = False
        self._update_entries()

    def _on_item_changed(self, item) -> None:
        if not self._syncing and item.column() == 0:
            self._update_entries()

    def _update_entries(self) -> None:
        table = self.spectra_table
        ticked = [self._rows[r] for r in range(table.rowCount())
                  if table.item(r, 0) is not None
                  and table.item(r, 0).checkState() == Qt.CheckState.Checked]
        self._entries = ticked
        self._checked_ids = {id(e["spectrum"]) for e in ticked}
        for combo, keep_first in ((self.target_combo, True), (self.reference_combo, False)):
            keep = combo.currentData()
            blocked = combo.blockSignals(True)
            combo.clear()
            if keep_first:
                combo.addItem("All spectra (shared)", SHARED)
            for e in ticked:
                combo.addItem(e["label"], id(e["spectrum"]))
            index = combo.findData(keep)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(blocked)
        n = len(ticked)
        self.spectra_status.setText("Tick at least one spectrum." if not n else
                                    f"{n} ticked.")
        self._fill_region_columns()
        self._show_target_regions()
        self._sync()

    def _tick_all(self, on: bool) -> None:
        self._syncing = True
        for r in range(self.spectra_table.rowCount()):
            self.spectra_table.item(r, 0).setCheckState(
                Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._syncing = False
        self._update_entries()

    def use_selected(self) -> None:
        selected = set(self._selected())
        self._syncing = True
        for r, entry in enumerate(self._rows):
            self.spectra_table.item(r, 0).setCheckState(
                Qt.CheckState.Checked if entry.get("index") in selected else Qt.CheckState.Unchecked)
        self._syncing = False
        self._update_entries()

    def set_checked(self, labels) -> None:
        """Tick exactly the spectra named ``labels`` (for code and tests)."""
        wanted = set(labels)
        self._syncing = True
        for r, entry in enumerate(self._rows):
            self.spectra_table.item(r, 0).setCheckState(
                Qt.CheckState.Checked if entry["label"] in wanted else Qt.CheckState.Unchecked)
        self._syncing = False
        self._update_entries()

    def labels(self) -> list[str]:
        return [e["label"] for e in self._entries]

    # ------------------------------------------------------------------
    # regions
    # ------------------------------------------------------------------

    def _target(self):
        """SHARED, or id() of the spectrum whose regions are being set."""
        if not self.different_check.isChecked():
            return SHARED
        return self.target_combo.currentData() or SHARED

    def _label_of(self, key) -> str:
        for e in self._rows:
            if id(e["spectrum"]) == key:
                return e["label"]
        return "?"

    def regions_for(self, entry) -> tuple:
        """(signal window, noise windows, own?) a spectrum is measured with."""
        own = self.own_regions.get(id(entry["spectrum"])) or {}
        signal = own.get("signal") if own.get("signal") is not None else self.shared_signal
        noise = own.get("noise") if own.get("noise") is not None else self.shared_noise
        is_own = ((own.get("signal") is not None and own["signal"] != self.shared_signal)
                  or (own.get("noise") is not None and own["noise"] != self.shared_noise))
        return signal, noise, is_own

    def set_signal_region(self, a: float, b: float, target=None) -> None:
        a, b = float(a), float(b)
        self._set_region("signal", (max(a, b), min(a, b)), target)

    def set_noise_region(self, windows, target=None) -> None:
        if windows and isinstance(windows[0], (int, float)):
            windows = [windows]
        self._set_region("noise", [(max(float(a), float(b)), min(float(a), float(b)))
                                   for a, b in windows], target)

    def _set_region(self, kind: str, value, target=None) -> None:
        target = self._target() if target is None else target
        text = format_window(value) if kind == "signal" else format_windows(value)
        if target == SHARED:
            if kind == "signal":
                self.shared_signal = value
            else:
                self.shared_noise = value
            self._log(f"{kind.capitalize()} region (all spectra): {text} ppm.")
        else:
            own = self.own_regions.setdefault(target, {"signal": None, "noise": None})
            own[kind] = value
            shared = self.shared_signal if kind == "signal" else self.shared_noise
            if shared is None:
                shared_text = "none set"
            else:
                shared_text = format_window(shared) if kind == "signal" else format_windows(shared)
            if shared is None or value != shared:
                tail = f" – DIFFERS from the shared region ({shared_text})."
            else:
                tail = " (the same as the shared region)."
            self._log(f"{kind.capitalize()} region for {self._label_of(target)}: {text} ppm"
                      + tail)
        self._invalidate()
        self._fill_region_columns()
        self._show_target_regions()
        self._sync()

    def reset_own(self, target=None) -> None:
        target = self._target() if target is None else target
        if target != SHARED and target in self.own_regions:
            del self.own_regions[target]
            self._log(f"{self._label_of(target)} uses the shared regions again.")
            self._invalidate()
            self._fill_region_columns()
            self._show_target_regions()
            self._sync()

    def _on_signal_typed(self) -> None:
        text = self.signal_edit.text().strip()
        if not text:
            return
        try:
            w = parse_window(text)
        except ValueError as exc:
            self.regions_status.setText(str(exc))
            return
        if self._current_region("signal") != w:
            self._set_region("signal", w)

    def _on_noise_typed(self) -> None:
        text = self.noise_edit.text().strip()
        if not text:
            return
        try:
            ws = parse_windows(text)
        except ValueError as exc:
            self.regions_status.setText(str(exc))
            return
        if self._current_region("noise") != ws:
            self._set_region("noise", ws)

    #: Shared F1 windows for 2D spectra: None = all of F1.
    shared_signal_f1: tuple[float, float] | None = None
    shared_noise_f1: list[tuple[float, float]] | None = None

    def set_f1_region(self, kind: str, value) -> None:
        """The 2D boxes' F1 side, shared by every 2D: ``(a, b)`` (signal),
        a list of such (noise), or None for all of F1."""
        if value is not None:
            if kind == "signal":
                a, b = value
                value = (max(float(a), float(b)), min(float(a), float(b)))
            else:
                if value and isinstance(value[0], (int, float)):
                    value = [value]
                value = [(max(float(a), float(b)), min(float(a), float(b))) for a, b in value]
        if kind == "signal":
            self.shared_signal_f1 = value
            text = format_window(value) if value else ""
        else:
            self.shared_noise_f1 = value
            text = format_windows(value) if value else ""
        edit = self.signal_f1_edit if kind == "signal" else self.noise_f1_edit
        blocked = edit.blockSignals(True)
        edit.setText(text)
        edit.blockSignals(blocked)
        self._log(f"2D {kind} box, F1: " + (f"{text} ppm." if text else "all of F1."))
        self._invalidate()
        self._sync()

    def _on_f1_typed(self, kind: str) -> None:
        edit = self.signal_f1_edit if kind == "signal" else self.noise_f1_edit
        text = edit.text().strip()
        try:
            value = (None if not text else
                     parse_window(text) if kind == "signal" else parse_windows(text))
        except ValueError as exc:
            self.regions_status.setText(str(exc))
            return
        current = self.shared_signal_f1 if kind == "signal" else self.shared_noise_f1
        if value != current:
            self.set_f1_region(kind, value)

    def _current_region(self, kind):
        target = self._target()
        if target == SHARED:
            return self.shared_signal if kind == "signal" else self.shared_noise
        own = self.own_regions.get(target) or {}
        if own.get(kind) is not None:
            return own[kind]
        return self.shared_signal if kind == "signal" else self.shared_noise

    def _show_target_regions(self) -> None:
        sig, noise = self._current_region("signal"), self._current_region("noise")
        for edit, text in ((self.signal_edit, format_window(sig) if sig else ""),
                           (self.noise_edit, format_windows(noise) if noise else "")):
            blocked = edit.blockSignals(True)
            edit.setText(text)
            edit.blockSignals(blocked)
        target = self._target()
        self.reset_own_button.setVisible(self.different_check.isChecked())
        self.reset_own_button.setEnabled(target != SHARED and target in self.own_regions)
        own = [e["label"] for e in self._entries if self.regions_for(e)[2]]
        self.regions_status.setText(
            ("Own regions (†): " + ", ".join(own) + ". Their S/N are compared with care: the "
             "result says so.") if own else "")

    def _fill_region_columns(self) -> None:
        self._syncing = True
        ticked = {id(e["spectrum"]) for e in self._entries}
        for r, entry in enumerate(self._rows):
            if id(entry["spectrum"]) not in ticked:
                for c in (1, 2):
                    self.spectra_table.setItem(r, c, _item(""))
                continue
            sig, noise, own = self.regions_for(entry)
            mark = " †" if own else ""
            self.spectra_table.setItem(r, 1, _item((format_window(sig) if sig else "—") + mark))
            self.spectra_table.setItem(r, 2, _item((format_windows(noise) if noise else "—")
                                                   + mark))
        self._syncing = False

    def _on_different_toggled(self, on: bool) -> None:
        self.target_combo.setEnabled(on)
        if not on:
            if self.own_regions:
                self._log("Different regions switched off: every spectrum uses the shared "
                          "regions (own regions kept, unused, until switched on again).")
            self.target_combo.setCurrentIndex(0)
        self._invalidate()
        self._fill_region_columns()
        self._show_target_regions()
        self._sync()

    def _on_target_changed(self, _i) -> None:
        self._show_target_regions()
        self._sync()

    def own_regions_in_use(self) -> dict[int, dict]:
        return self.own_regions if self.different_check.isChecked() else {}

    def regions_for_entries(self) -> list:
        """(signal, noise, own) for each ticked spectrum, in order --
        honouring the 'Different regions' switch."""
        out = []
        use_own = self.different_check.isChecked()
        for e in self._entries:
            if use_own:
                out.append(self.regions_for(e))
            else:
                out.append((self.shared_signal, self.shared_noise, False))
        return out

    # ------------------------------------------------------------------
    # canvas
    # ------------------------------------------------------------------

    def _buttons(self):
        return {DRAG_SIGNAL: self.signal_drag_button, DRAG_NOISE: self.noise_drag_button}

    def _arm_for(self, mode: str, on: bool) -> None:
        if on:
            for other, button in self._buttons().items():
                if other != mode and button.isChecked():
                    blocked = button.blockSignals(True)
                    button.setChecked(False)
                    button.blockSignals(blocked)
            if not self.show_spectra():
                self._disarm()
                return
            self._armed = mode
            self._arm("region")
        elif self._armed == mode:
            self._armed = None
            self._arm(None)

    def _disarm(self) -> None:
        self._armed = None
        for button in self._buttons().values():
            blocked = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(blocked)
        self._arm(None)

    @property
    def armed(self) -> str | None:
        return self._armed

    def canvas_mode_released(self) -> None:
        if self._armed is not None:
            self._armed = None
            for button in self._buttons().values():
                blocked = button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(blocked)

    def on_canvas_region(self, lo: float, hi: float) -> None:
        mode = self._armed
        if mode is None:
            return
        self._disarm()
        if mode == DRAG_SIGNAL:
            self.set_signal_region(float(lo), float(hi))
        else:
            self.set_noise_region([(float(lo), float(hi))])
        self.show_spectra()

    def _plot_kwargs(self) -> dict:
        """What the plot takes from the view's plot style: colours, and the
        names and S/N values at the legend's text size, named by a legend
        or on the traces as Plot Style says -- they were fixed fractions
        of 12 pt, so smaller than every other plot's legend (explicit user
        report, 2026-10-15). Read at draw time, so a Plot Style change
        (which redraws the figure) reaches them too."""
        many = len(self._entries) > 1
        style = self._style("spectra" if many else "spectrum")
        cmap = (style.get("compare_colormap") if many else style.get("trace_color")) or "black"
        kwargs = {"cmap": cmap}
        if self._colours is not None and self._entries:
            colours = list(self._colours(self._entries))
            if any(c is not None for c in colours):
                kwargs["colours"] = colours
        size = style.get("legend_fontsize") or style.get("labelsize")
        if size:
            kwargs.update(fontsize=float(style.get("labelsize") or size),
                          legend_fontsize=float(size), annotation_fontsize=float(size))
        if style:
            kwargs["names"] = "labels" if style.get("legend_style") == "labels" else "legend"
        return kwargs

    def show_spectra(self) -> bool:
        if not self._entries:
            QMessageBox.information(self, "Signal-to-Noise", "Tick the spectra first (step 1).")
            return False
        spectra = [e["spectrum"] for e in self._entries]
        regions = [{"signal": s, "noise": n, "signal_f1": self.shared_signal_f1,
                    "noise_f1": self.shared_noise_f1}
                   for s, n, _own in self.regions_for_entries()]
        labels = self.labels()
        result = self.result if self._result_entries == self._entries else None
        kwargs = {}
        view = self._view_limits() if self._view_limits is not None else None
        if view is not None:
            lo, hi = min(view), max(view)
            edges = [v for s, n, _o in self.regions_for_entries()
                     for w in ([s] if s else []) + list(n or []) for v in w]
            # Kept unless a region lies wholly outside it (then the plot
            # widens to show the regions, as before).
            if not edges or all(lo <= min(w) and max(w) <= hi for s, n, _o in
                                self.regions_for_entries()
                                for w in ([s] if s else []) + list(n or [])):
                kwargs["xlims"] = view

        def draw(figure):
            style = {**self._plot_kwargs(), **kwargs}
            if result is not None:
                snr.plot_snr(spectra, result, labels=labels, figure=figure, **style)
            else:
                snr.plot_snr(spectra, regions=regions, labels=labels, figure=figure, **style)
        self._draw(draw, "spectra" if len(spectra) > 1 else "spectrum")
        return True

    # ------------------------------------------------------------------
    # calculation
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        n = len(self._entries)
        have = all(s is not None and noise for s, noise, _o in self.regions_for_entries()) and n
        self.calculate_button.setEnabled(bool(have))
        self.show_button.setEnabled(n > 0)
        self.signal_drag_button.setEnabled(n > 0)
        self.noise_drag_button.setEnabled(n > 0)
        self.target_combo.setEnabled(self.different_check.isChecked())
        self.reference_combo.setEnabled(n > 1)
        done = self.result is not None
        for b in (self.report_button, self.export_button, self.copy_report_button,
                  self.copy_code_button):
            b.setEnabled(done)

    def _invalidate(self) -> None:
        if self.result is not None:
            self.result = None
            self._result_entries = []
            self.result_label.setText("Regions changed: calculate again.")
            self.result_status.setText("Regions changed: calculate again.")
            self.result_table.setVisible(False)
            self.result_hint.setVisible(False)
            for r in range(self.spectra_table.rowCount()):
                for c in range(3, len(self.COLUMNS)):
                    self.spectra_table.setItem(r, c, _item(""))

    def settings(self) -> dict:
        return {"detrend": self.detrend_combo.currentData(), "sign": self.sign_combo.currentData(),
                "offset": self.offset_combo.currentData()}

    def _arguments(self):
        regions_list = self.regions_for_entries()
        if not self._entries:
            raise ValueError("Tick at least one spectrum (step 1).")
        for e, (sig, noise, _own) in zip(self._entries, regions_list):
            if sig is None or not noise:
                raise ValueError(f"{e['label']} has no {'signal' if sig is None else 'noise'} "
                                 "region: type it or drag it on the plot (step 2).")
        shared_sig, shared_noise = self.shared_signal, self.shared_noise
        regions = {}
        for k, (sig, noise, own) in enumerate(regions_list):
            if shared_sig is None or shared_noise is None or own:
                regions[k] = {"signal": sig, "noise": noise}
        ref = max(0, self.reference_combo.currentIndex())
        return shared_sig, shared_noise, regions, ref

    def calculate(self, explicit: bool = False) -> dict | None:
        try:
            shared_sig, shared_noise, regions, ref = self._arguments()
            result = snr.signal_to_noise_many(
                [e["spectrum"] for e in self._entries], shared_sig, shared_noise,
                labels=self.labels(), regions=regions or None, reference=ref,
                **self._f1_kwargs(), **self.settings())
        except ValueError as exc:
            if explicit:
                QMessageBox.warning(self, "Signal-to-Noise", str(exc))
            self._log(f"Not calculated: {exc}")
            return None
        self.result = result
        self._result_entries = list(self._entries)
        self._fill_results()
        self._fill_summary()
        report = snr.snr_report(result)
        self.result_label.setText(report)
        self._log("Calculated S/N for " + ", ".join(self.labels()) + ".\n"
                  + ("Regions DIFFERED: " + ", ".join(result["differing"]) + " used their own."
                     if result["regions_differ"] else
                     "Same regions for all: " + snr.describe_regions(
                         result["rows"][0]["signal_region"], result["rows"][0]["noise_region"])
                     + ".")
                  + "\n" + "\n".join(f"  {r['label']}: S/N {r['snr']:.4g} ± "
                                     f"{r['snr_error']:.2g}" for r in result["rows"]))
        self._keep(result, report)
        self.show_spectra()
        self.calculated.emit()
        self._sync()
        return result

    def _f1_kwargs(self) -> dict:
        """The 2D F1 ranges, only when a 2D is ticked."""
        if not any(getattr(e["spectrum"], "ndim", 1) == 2 for e in self._entries):
            return {}
        return {"signal_region_f1": self.shared_signal_f1,
                "noise_region_f1": self.shared_noise_f1}

    def conventions_rows(self) -> list[tuple[str, list[str]]]:
        """[(convention "name: formula", [value per spectrum])] of the
        last result -- what the conventions table shows."""
        if self.result is None:
            return []
        rows = self.result["rows"]
        names: dict[str, list] = {}
        for j, r in enumerate(rows):
            for c in snr.snr_conventions(r):
                key = f"{c['name']}: {c['formula']}"
                names.setdefault(key, ["—"] * len(rows))[j] = f"{c['value']:.4g}"
        return list(names.items())

    #: Summary values drawn in a warning colour (readable on light and dark).
    _ALERT = QColor("#d35400")

    def _fill_summary(self) -> None:
        """The result table: :func:`snr.snr_summary`, a row per quantity
        and a column per spectrum, as tall as its rows -- in the panel and
        in Full Report…."""
        summary = snr.snr_summary(self.result)
        self._fill_summary_table(self.report_dialog.table, summary, elide=False)
        self._fill_summary_table(self.result_table, summary)
        self.result_table.setVisible(True)
        self.result_hint.setVisible(True)
        self.result_status.setText(self._status_line())

    def _fill_summary_table(self, table, summary, elide: bool = True) -> None:
        labels = [r["label"] for r in self.result["rows"]]
        table.clear()
        table.setColumnCount(len(labels))
        table.setRowCount(len(summary))
        for j, label in enumerate(labels):
            head = QTableWidgetItem(_elide(label) if elide else label)
            head.setToolTip(label)
            table.setHorizontalHeaderItem(j, head)
        for i, row in enumerate(summary):
            head = QTableWidgetItem(row["short"])
            head.setToolTip(f"{row['name']}: {row['detail']}\n{row['about']}")
            table.setVerticalHeaderItem(i, head)
            for j, (text, note) in enumerate(zip(row["values"], row["notes"])):
                item = _item(text, align_right=True)
                if note:
                    item.setToolTip(note)
                if i == 0:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                if text == "NO" or (row["name"] == "Checks" and text != "OK"):
                    item.setForeground(self._ALERT)
                table.setItem(i, j, item)
        head_font = table.verticalHeaderItem(0).font()
        head_font.setBold(True)
        table.verticalHeaderItem(0).setFont(head_font)
        table.refit()

    def _status_line(self) -> str:
        """One line above the table: which regions, and any warnings."""
        result = self.result
        rows = result["rows"]
        if result.get("regions_differ"):
            text = ("Regions DIFFERED: " + ", ".join(result["differing"])
                    + " used their own (†). See Full Report… for each one's.")
        else:
            text = "Same regions for every spectrum: " + snr.describe_regions(
                rows[0]["signal_region"], rows[0]["noise_region"]) + "."
        warned = sum(bool(r["warnings"]) for r in rows)
        if warned:
            text += f" {warned} spectrum{'s' if warned > 1 else ''} with warnings (Checks row)."
        return text

    def _fill_results(self) -> None:
        rows = {id(e["spectrum"]): r for e, r in zip(self._result_entries, self.result["rows"])}
        differing = set(self.result["differing"])
        self._syncing = True
        for i, entry in enumerate(self._rows):
            r = rows.get(id(entry["spectrum"]))
            if r is None:
                for c in range(3, len(self.COLUMNS)):
                    self.spectra_table.setItem(i, c, _item(""))
                continue
            lo, hi = r["snr_interval"]
            notes = []
            if not r["detected"]:
                notes.append("not detected")
            if r["label"] in differing:
                notes.append("own regions †")
            if r["warnings"]:
                notes.append(f"{len(r['warnings'])} warning(s)")
            cells = (f"{r['snr']:.4g} ± {r['snr_error']:.2g}", f"{_g(lo)} – {_g(hi)}",
                     _g(r["snr_per_sqrt_scan"], 4), "; ".join(notes))
            for c, text in enumerate(cells, start=3):
                item = _item(text, align_right=c < 6)
                if c == 6 and r["warnings"]:
                    item.setToolTip("\n".join(r["warnings"]))
                self.spectra_table.setItem(i, c, item)
        self._syncing = False

    def _keep(self, result, report) -> None:
        indices = [e["index"] for e in self._result_entries if e.get("index") is not None]
        if not indices:
            return
        inputs = [f"Spectra: {', '.join(self.labels())}",
                  f"Settings: {self.settings()}"]
        for e, (sig, noise, own) in zip(self._result_entries, self.regions_for_entries()):
            inputs.append(f"{e['label']}: {snr.describe_regions(sig, noise)}"
                          + (" (OWN regions)" if own else ""))
        f1 = self._f1_kwargs()
        if f1:
            inputs.append("2D boxes, F1: signal "
                          + (format_window(f1["signal_region_f1"]) if f1["signal_region_f1"]
                             else "all of F1") + "; noise "
                          + (format_windows(f1["noise_region_f1"]) if f1["noise_region_f1"]
                             else "all of F1"))
        record = {"label": "Signal-to-noise", "target": ", ".join(self.labels()),
                  "inputs_text": inputs, "report": report,
                  "result": {"table": snr.snr_table(result),
                             "regions_differ": result["regions_differ"],
                             "differing": result["differing"]}}
        try:
            self._store("snr", record, indices)
        except Exception:  # noqa: BLE001 -- keeping is best-effort; the result is on screen
            pass

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def notebook_code(self) -> str:
        shared_sig, shared_noise, regions, ref = self._arguments()
        kwargs = dict(self.settings())
        kwargs.update({k: v for k, v in self._f1_kwargs().items() if v is not None})
        if ref:
            kwargs["reference"] = ref
        return snr.snr_python(shared_sig, shared_noise, regions=regions or None,
                              labels=self.labels(), **kwargs)

    def copy_code(self) -> None:
        try:
            QGuiApplication.clipboard().setText(self.notebook_code())
        except ValueError as exc:
            QMessageBox.warning(self, "Signal-to-Noise", str(exc))

    def copy_report(self) -> None:
        if self.result is not None:
            QGuiApplication.clipboard().setText(snr.snr_report(self.result))

    def _on_export(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export signal-to-noise", "snr.csv",
                                              "CSV files (*.csv)")
        if path:
            snr.snr_to_csv(self.result, path)
            self._log(f"Exported to {path}")
