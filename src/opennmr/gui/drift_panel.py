"""Process ▸ Field Drift Correction: the GUI over
:mod:`opennmr.notebook.drift_correction` (Najbauer & Andreas, J. Magn.
Reson. 305 (2019) 1-4).

Three steps, each usable on its own:

1. **Measure** (optional): the tallest peak in a ppm region of a
   reference 1D, and of 1D spectra recorded before and after the nD
   experiment, gives the start and end shifts (reference − observed,
   Hz). With the reference left as "the 'before' spectrum", the start
   shift is 0 and the end shift is the drift across the experiment.
2. **Shifts**: start/end in Hz of the detected nucleus, typed or
   measured; positive moves peaks to higher frequency.
3. **Apply**: to the selected Bruker 2D/pseudo-2D in place (reprocessed
   from its raw FIDs, removable), or written to a new Bruker EXPNO --
   which also covers 3D/4D and NUS data this program cannot display.
   NUS blocks corrected that way can then be concatenated.

The panel holds no maths; MainWindow supplies the callbacks:
``spectra()`` -> ``[{"label", "spectrum", "index"}]`` (every open
experiment), ``selected()`` -> the selected session index or None,
``apply(request)`` -> apply ``{"start_hz", "end_hz"}`` (or ``None`` to
remove) to the selected spectrum, returning an error message or None,
and ``log(index, text)`` -> add an event to that spectrum's processing log.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QVBoxLayout,
    QWidget,
)

from ..notebook import drift_correction as dc
from . import data_directory

_SAME_AS_BEFORE = "(the 'before' spectrum)"


def _wrapped(text: str, rich: bool = False) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    if rich:
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setOpenExternalLinks(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    return label


def _hz_spin() -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-1.0e6, 1.0e6)
    spin.setDecimals(4)
    spin.setSuffix(" Hz")
    return spin


def _ppm_spin(value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-1.0e5, 1.0e5)
    spin.setDecimals(3)
    spin.setSuffix(" ppm")
    spin.setValue(value)
    return spin


class DriftCorrectionPanel(QWidget):
    """See the module docstring."""

    #: Something was measured, applied or written (for tests and MainWindow).
    changed = Signal()

    def __init__(self, spectra, selected, apply, log=None, parent=None) -> None:
        super().__init__(parent)
        self._spectra, self._selected, self._apply, self._log = spectra, selected, apply, log
        self.last_result: dc.DriftCorrectionResult | None = None
        self.last_measurement: dict | None = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(_wrapped(
            "Undo a magnet's field drift across a long 2D/3D/4D experiment, assuming it is "
            "linear in time: each FID is shifted back by a drift interpolated between a start "
            "and an end value, in the direct and (scaled by the frequency ratio) every indirect "
            "dimension."))
        v.addWidget(self._build_measure())
        v.addWidget(self._build_shifts())
        v.addWidget(self._build_apply())
        self.citation_label = _wrapped(
            f"Method and original TopSpin AU programs (lindriftcomp, concatenate_ser_nuslist): "
            f"E. E. Najbauer and L. B. Andreas, <i>J. Magn. Reson.</i> <b>305</b> (2019) 1–4, "
            f"<a href=\"{dc.DOI_URL}\">doi:10.1016/j.jmr.2019.05.005</a>. Please cite it.", rich=True)
        v.addWidget(self.citation_label)
        self.refresh()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_measure(self) -> QGroupBox:
        box = QGroupBox("1. Measure the drift (optional)")
        form = QFormLayout(box)
        form.addRow(_wrapped("From 1D spectra of the detected nucleus recorded just before and "
                             "just after the nD experiment, on the same peak."))
        self.reference_combo = QComboBox()
        self.before_combo = QComboBox()
        self.after_combo = QComboBox()
        for combo in (self.reference_combo, self.before_combo, self.after_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        self.reference_combo.setToolTip(
            "Where the peak should be. Leave it as the 'before' spectrum to correct the drift "
            "during the experiment only (start = 0); pick another spectrum to put several "
            "experiments on one common reference before adding them up.")
        form.addRow("Reference:", self.reference_combo)
        form.addRow("Before:", self.before_combo)
        form.addRow("After:", self.after_combo)
        region = QWidget()
        h = QHBoxLayout(region)
        h.setContentsMargins(0, 0, 0, 0)
        self.region_hi_spin = _ppm_spin(5.5)
        self.region_lo_spin = _ppm_spin(4.0)
        h.addWidget(self.region_hi_spin)
        h.addWidget(QLabel("to"))
        h.addWidget(self.region_lo_spin)
        form.addRow("Peak region:", region)
        self.measure_button = QPushButton("Measure Start and End Shifts")
        self.measure_button.clicked.connect(self._on_measure_clicked)
        form.addRow(self.measure_button)
        self.measure_label = _wrapped("")
        form.addRow(self.measure_label)
        return box

    def _build_shifts(self) -> QGroupBox:
        box = QGroupBox("2. Shifts")
        form = QFormLayout(box)
        self.start_spin = _hz_spin()
        self.end_spin = _hz_spin()
        self.start_spin.setToolTip("Applied to the first FID acquired (reference − observed, Hz).")
        self.end_spin.setToolTip("Applied to the last FID acquired (reference − observed, Hz).")
        form.addRow("Start shift:", self.start_spin)
        form.addRow("End shift:", self.end_spin)
        form.addRow(_wrapped("Hz of the detected nucleus; positive moves peaks to higher "
                             "frequency (left). Enter where the peak should be minus where it is."))
        return box

    def _build_apply(self) -> QGroupBox:
        box = QGroupBox("3. Apply")
        v = QVBoxLayout(box)
        self.target_label = _wrapped("")
        v.addWidget(self.target_label)
        self.apply_button = QPushButton("Apply to Selected Spectrum")
        self.apply_button.setToolTip("Reprocess the selected Bruker 2D or pseudo-2D from its raw "
                                     "FIDs with the correction. Undo with Remove Correction.")
        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.remove_button = QPushButton("Remove Correction")
        self.remove_button.clicked.connect(self._on_remove_clicked)
        row = QHBoxLayout()
        row.addWidget(self.apply_button)
        row.addWidget(self.remove_button)
        v.addLayout(row)
        self.write_button = QPushButton("Write Corrected Bruker Dataset…")
        self.write_button.setToolTip(
            "Copy a Bruker 2D/3D/4D experiment (planes, points or NUS) to a new EXPNO with its "
            "ser corrected -- like lindriftcomp. The original is not touched; reprocess the copy "
            "here or in TopSpin.")
        self.write_button.clicked.connect(self._on_write_clicked)
        self.concat_button = QPushButton("Concatenate NUS Blocks…")
        self.concat_button.setToolTip(
            "Join NUS experiments recorded as separate blocks of one nuslist into a new EXPNO "
            "(concatenate_ser_nuslist). Correct each block first, against one reference.")
        self.concat_button.clicked.connect(self._on_concat_clicked)
        row2 = QHBoxLayout()
        row2.addWidget(self.write_button)
        row2.addWidget(self.concat_button)
        v.addLayout(row2)
        self.status_label = _wrapped("")
        v.addWidget(self.status_label)
        return box

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def _entries(self) -> list[dict]:
        try:
            return list(self._spectra())
        except Exception:  # noqa: BLE001 -- the panel must never take the window down
            return []

    @staticmethod
    def _fill(combo: QComboBox, entries: list[dict], extra: str | None = None) -> None:
        keep = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if extra:
            combo.addItem(extra, None)
        for entry in entries:
            combo.addItem(entry["label"], entry["index"])
        position = combo.findData(keep) if keep is not None else -1
        combo.setCurrentIndex(position if position >= 0 else 0)
        combo.blockSignals(False)

    def refresh(self) -> None:
        """Re-read Open experiments and the selection."""
        one_d = [e for e in self._entries() if e["spectrum"].ndim == 1]
        self._fill(self.reference_combo, one_d, extra=_SAME_AS_BEFORE)
        self._fill(self.before_combo, one_d)
        self._fill(self.after_combo, one_d)
        if self.after_combo.count() > 1 and self.after_combo.currentIndex() == self.before_combo.currentIndex():
            self.after_combo.setCurrentIndex(1)
        self.measure_button.setEnabled(len(one_d) >= 1)
        self._sync_target()

    def _selected_spectrum(self):
        try:
            index = self._selected()
        except RuntimeError:  # the window is being torn down
            return None, None
        if index is None:
            return None, None
        for entry in self._entries():
            if entry["index"] == index:
                return index, entry["spectrum"]
        return index, None

    def _sync_target(self) -> None:
        index, spectrum = self._selected_spectrum()
        can_apply = False
        if spectrum is None:
            text = "Select a Bruker 2D or pseudo-2D in Open experiments to correct it here."
        elif spectrum.metadata.vendor != "bruker" or spectrum.ndim != 2 \
                or spectrum.metadata.raw.get("fid_raw") is None:
            text = ("The selected spectrum cannot be corrected here: that needs a Bruker 2D or "
                    "pseudo-2D read from its raw ser. For 3D/4D or NUS data, write a corrected "
                    "dataset instead.")
        else:
            record = spectrum.metadata.raw.get("drift_correction")
            text = (f"Corrected: {record['start_hz']:+.3f} → {record['end_hz']:+.3f} Hz over "
                    f"{record.get('n_points', '?')} points. {record.get('description', '')}"
                    if record else "Selected spectrum: not drift-corrected.")
            can_apply = True
        self.target_label.setText(text)
        self.apply_button.setEnabled(can_apply)
        self.remove_button.setEnabled(bool(
            can_apply and spectrum is not None and spectrum.metadata.raw.get("drift_correction")))

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def _spectrum_by_index(self, index):
        for entry in self._entries():
            if entry["index"] == index:
                return entry["spectrum"]
        return None

    def _on_measure_clicked(self) -> None:
        before = self._spectrum_by_index(self.before_combo.currentData())
        after = self._spectrum_by_index(self.after_combo.currentData())
        ref_index = self.reference_combo.currentData()
        reference = before if ref_index is None else self._spectrum_by_index(ref_index)
        if before is None or after is None or reference is None:
            self.measure_label.setText("Choose the 'before' and 'after' 1D spectra first.")
            return
        region = (self.region_lo_spin.value(), self.region_hi_spin.value())
        try:
            start = dc.measure_drift_hz(reference, before, region)
            end = dc.measure_drift_hz(reference, after, region)
            positions = {name: dc.peak_frequency_hz(s, region)
                         for name, s in (("reference", reference), ("before", before), ("after", after))}
        except ValueError as exc:
            self.measure_label.setText(f"Could not measure: {exc}")
            return
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)
        self.last_measurement = {"start_hz": start, "end_hz": end, "region_ppm": region,
                                 "positions_hz": positions,
                                 "labels": {"reference": self.reference_combo.currentText(),
                                            "before": self.before_combo.currentText(),
                                            "after": self.after_combo.currentText()}}
        self.measure_label.setText(
            f"Start {start:+.3f} Hz, end {end:+.3f} Hz (drift of {end - start:+.3f} Hz across the "
            f"experiment). Check the peak is the same one in all three.")
        self.changed.emit()

    def request(self) -> dict:
        return {"start_hz": float(self.start_spin.value()), "end_hz": float(self.end_spin.value())}

    def _on_apply_clicked(self) -> None:
        error = self._apply(self.request())
        if error:
            self.status_label.setText(error)
            return
        index, spectrum = self._selected_spectrum()
        record = spectrum.metadata.raw.get("drift_correction") if spectrum is not None else None
        # The correction itself is logged as a change of state (the
        # "drift_correction" record); what only the panel knows is where
        # the numbers came from.
        m = self.last_measurement
        if record and self._log and index is not None and m and \
                (m["start_hz"], m["end_hz"]) == (record["start_hz"], record["end_hz"]):
            self._log(index, (
                f"Field drift shifts measured on {m['labels']['before']} (start) and "
                f"{m['labels']['after']} (end), reference {m['labels']['reference']}, tallest peak "
                f"in {min(m['region_ppm']):.3f}–{max(m['region_ppm']):.3f} ppm: "
                f"{m['start_hz']:+.4f} / {m['end_hz']:+.4f} Hz."))
        self.status_label.setText("Applied. " + (record.get("description", "") if record else ""))
        self._sync_target()
        self.changed.emit()

    def _on_remove_clicked(self) -> None:
        error = self._apply(None)
        if error:
            self.status_label.setText(error)
            return
        self.status_label.setText("Correction removed.")
        self._sync_target()
        self.changed.emit()

    @staticmethod
    def _describe(record: dict) -> str:
        return (f"{record['start_hz']:+.4f} → {record['end_hz']:+.4f} Hz over "
                f"{record.get('n_points', '?')} points ({record.get('description', '')}); "
                f"method: {dc.CITATION}")

    def _default_folder(self) -> str:
        _index, spectrum = self._selected_spectrum()
        path = getattr(getattr(spectrum, "metadata", None), "source_path", None) if spectrum else None
        # The selected experiment itself is the likeliest one to correct;
        # else the default data folder (Settings ▸ Data).
        return path if path and os.path.isdir(path) else data_directory.start_directory()

    def write_dataset(self, source: str, destination: str | None = None) -> dc.DriftCorrectionResult:
        """Correct ``source`` into ``destination`` (``None``: next free
        EXPNO) with the panel's shifts. Also the seam the tests use."""
        result = dc.correct_bruker_dataset(source, destination, self.start_spin.value(),
                                           self.end_spin.value())
        self.last_result = result
        self.status_label.setText(f"Written to {result.destination}. {result.description}. "
                                  f"Its pdata is a stale copy: reprocess it.")
        self.changed.emit()
        return result

    def _on_write_clicked(self) -> None:
        source = QFileDialog.getExistingDirectory(
            self, "Bruker experiment to correct (EXPNO folder)", self._default_folder())
        if not source:
            return
        try:
            setup = dc.bruker_drift_setup(source)
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            QMessageBox.warning(self, "Cannot correct this experiment", str(exc))
            return
        destination = dc.next_free_expno(source)
        answer = QMessageBox.question(
            self, "Write corrected dataset",
            f"{setup.describe()}\n\nShift {self.start_spin.value():+.4f} → "
            f"{self.end_spin.value():+.4f} Hz.\n\nWrite the corrected copy to\n{destination}?\n\n"
            f"The original experiment is not changed.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.write_dataset(source, destination)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Drift correction failed", str(exc))

    def _on_concat_clicked(self) -> None:
        dialog = NusConcatDialog(self._default_folder(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            out = dc.concatenate_nus_datasets(dialog.sources(), dialog.destination())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Concatenation failed", str(exc))
            return
        self.status_label.setText(
            f"Concatenated {len(out['sources'])} blocks ({out['nus_points']} NUS points) into "
            f"{out['destination']}. Check its NUS status parameters (e.g. NusTD) before processing.")
        self.changed.emit()

    def copy_report(self) -> None:
        if self.last_result is not None:
            QApplication.clipboard().setText(self.last_result.report())


class NusConcatDialog(QDialog):
    """Pick the NUS blocks (in order) and where the joined copy goes."""

    def __init__(self, start_folder: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Concatenate NUS Blocks")
        self._start = start_folder
        v = QVBoxLayout(self)
        v.addWidget(_wrapped("Blocks in acquisition order. The joined experiment is a copy of the "
                             "first with every block's ser and nuslist appended."))
        self.list = QListWidget()
        v.addWidget(self.list)
        row = QHBoxLayout()
        add = QPushButton("Add Block…")
        add.clicked.connect(self._add)
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda: self.list.takeItem(self.list.currentRow()))
        row.addWidget(add)
        row.addWidget(remove)
        v.addLayout(row)
        form = QFormLayout()
        self.destination_edit = QLineEdit()
        form.addRow("New experiment:", self.destination_edit)
        v.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def add_source(self, path: str) -> None:
        self.list.addItem(path)
        if not self.destination_edit.text():
            self.destination_edit.setText(dc.next_free_expno(path))

    def _add(self) -> None:
        start = self.list.item(self.list.count() - 1).text() if self.list.count() else self._start
        start = start or data_directory.start_directory()
        path = QFileDialog.getExistingDirectory(self, "NUS block (EXPNO folder)", start)
        if path:
            self.add_source(path)

    def sources(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def destination(self) -> str:
        return self.destination_edit.text().strip()

    def _accept(self) -> None:
        if len(self.sources()) < 2 or not self.destination():
            QMessageBox.information(self, "Concatenate NUS Blocks",
                                    "Add at least two blocks and name the new experiment.")
            return
        self.accept()
