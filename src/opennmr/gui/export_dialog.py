"""File > Export Data… (Ctrl/Cmd+Shift+X): write spectra, from any vendor, as
.csv, .xy or JCAMP-DX (.jdx).

The writing is :func:`opennmr.io.export.export_spectrum` -- this dialog only
chooses what to write and where, and :func:`export_many` (no widgets) does
the loop, so the notebook and the tests use the same code path.
"""

from __future__ import annotations

import os
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QVBoxLayout,
)

from ..io import export as io_export
from .widgets.typed_spinbox import TypedSpinBox

FORMAT_CHOICES = (("csv", "CSV table (.csv)"), ("xy", "Plain x y text (.xy)"),
                  ("jdx", "JCAMP-DX (.jdx)"))
X_UNITS = ((None, "Format default (ppm; Hz for JCAMP-DX)"), ("ppm", "ppm"), ("hz", "Hz"))
IMAG = ((None, "Format default (CSV only)"), (True, "Include"), (False, "Real part only"))
DOMAINS = (("spectrum", "The spectrum"), ("fid", "The FID, as acquired"))


def safe_name(text: str) -> str:
    """A file-name stem from an Open-experiments label."""
    stem = re.sub(r"[^\w.\-+ ()]+", "_", str(text)).strip(" ._") or "spectrum"
    return stem[:100]


def export_many(items, destination: str, fmt: str, *, x_unit=None, include_imag=None,
                domain: str = "spectrum", row: int | None = None, ascending: bool = False,
                metadata_comments: bool | None = None) -> list[str]:
    """Write each ``(label, spectrum)`` of ``items``.

    ``destination`` is the file for one spectrum, or a folder for several
    (each named from its label, never overwriting another of this batch).
    ``row`` (0-based) picks the trace of a 2D / pseudo-2D spectrum for the
    formats that hold one; a 1D spectrum ignores it. Returns the paths."""
    items = list(items)
    if not items:
        raise ValueError("Choose at least one spectrum to export.")
    ext = io_export.FORMATS[fmt]
    written, used = [], set()
    for label, spectrum in items:
        if len(items) == 1 and not os.path.isdir(destination):
            path = destination
            if not os.path.splitext(path)[1]:
                path += ext
        else:
            os.makedirs(destination, exist_ok=True)
            stem, k = safe_name(label), 2
            candidate = stem
            while candidate.lower() in used:
                candidate, k = f"{stem}_{k}", k + 1
            used.add(candidate.lower())
            path = os.path.join(destination, candidate + ext)
        trace = None
        if spectrum.ndim == 2 and (fmt != "csv" or row is not None):
            trace = 0 if row is None else min(max(int(row), 0), spectrum.shape[0] - 1)
        written.append(io_export.export_spectrum(
            spectrum, path, fmt, x_unit=x_unit, include_imag=include_imag, domain=domain,
            row=trace, ascending=ascending, metadata_comments=metadata_comments, title=label))
    return written


class ExportDialog(QDialog):
    """``items`` is ``[(label, spectrum), ...]`` (everything in Open
    experiments); ``checked`` the indices ticked at the start."""

    def __init__(self, items, checked=(), row: int | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export data")
        self.items = list(items)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Spectra to export (any vendor):"))
        self.list = QListWidget()
        self.list.setMinimumHeight(120)
        for i, (label, _s) in enumerate(self.items):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i in set(checked) else Qt.Unchecked)
            self.list.addItem(item)
        layout.addWidget(self.list)

        form = QFormLayout()
        self.format_combo = QComboBox()
        for key, text in FORMAT_CHOICES:
            self.format_combo.addItem(text, key)
        form.addRow("Format:", self.format_combo)
        self.domain_combo = QComboBox()
        for key, text in DOMAINS:
            self.domain_combo.addItem(text, key)
        self.domain_combo.setToolTip("The FID is written against time (s), exactly as read, "
                                     "before any window or zero-fill. Only for spectra that came "
                                     "from an FID.")
        form.addRow("Write:", self.domain_combo)
        self.x_unit_combo = QComboBox()
        for key, text in X_UNITS:
            self.x_unit_combo.addItem(text, key)
        form.addRow("x axis:", self.x_unit_combo)
        self.imag_combo = QComboBox()
        for key, text in IMAG:
            self.imag_combo.addItem(text, key)
        form.addRow("Imaginary part:", self.imag_combo)
        self.row_spin = TypedSpinBox()
        self.row_spin.setRange(1, 1_000_000)
        self.row_spin.setValue((row or 0) + 1)
        self.row_spin.setToolTip("For a 2D or pseudo-2D spectrum: which row (1 = first). "
                                 "CSV writes the whole matrix instead if this is unticked.")
        self.whole_check = QCheckBox("2D: write the whole matrix (CSV only)")
        self.whole_check.setChecked(True)
        form.addRow("2D row:", self.row_spin)
        form.addRow("", self.whole_check)
        self.ascending_check = QCheckBox("Lowest x first")
        form.addRow("Order:", self.ascending_check)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export…")
        buttons.accepted.connect(self._on_export)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.written: list[str] = []

    def chosen(self):
        return [self.items[i] for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def options(self) -> dict:
        fmt = self.format_combo.currentData()
        return dict(fmt=fmt, x_unit=self.x_unit_combo.currentData(),
                    include_imag=self.imag_combo.currentData(),
                    domain=self.domain_combo.currentData(),
                    row=(None if fmt == "csv" and self.whole_check.isChecked()
                         else self.row_spin.value() - 1),
                    ascending=self.ascending_check.isChecked())

    def _on_export(self) -> None:
        chosen = self.chosen()
        if not chosen:
            QMessageBox.information(self, "Export data", "Tick at least one spectrum.")
            return
        options = self.options()
        ext = io_export.FORMATS[options["fmt"]]
        if len(chosen) == 1:
            start = safe_name(chosen[0][0]) + ext
            destination, _ = QFileDialog.getSaveFileName(
                self, "Export data", start, f"*{ext};;All files (*)")
        else:
            destination = QFileDialog.getExistingDirectory(
                self, f"Folder for {len(chosen)} files")
        if not destination:
            return
        try:
            self.written = export_many(chosen, destination, **{
                k: v for k, v in options.items() if k != "fmt"}, fmt=options["fmt"])
        except Exception as exc:  # noqa: BLE001 -- shown
            QMessageBox.warning(self, "Export data", str(exc))
            return
        self.accept()
