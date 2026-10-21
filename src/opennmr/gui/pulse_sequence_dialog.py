"""
The pulse-sequence viewer pop-up (explicit user request, 2026-10-21): read an
experiment's pulse programme -- Bruker, Varian, JEOL, Magritek, X-Pulse -- and
show it, with hard pulses as rectangles, shaped pulses as a rounded
half-ellipse and decoupling as a short rectangle, each kind in its own
(customisable) colour, and the pulse / delay names switchable all / some /
none.

Opened by right-clicking an open experiment ("Show Pulse Sequence…") or by
double-clicking the pulse-programme name at the bottom right of the window.
The same drawing, from a notebook::

    from opennmr.notebook import pulse_sequence as ps
    fig, ax = ps.plot_pulse_sequence(ps.pulse_sequence_for(spectrum)[0])

Only Qt is imported here at module level; matplotlib and the parser load when
the window is first opened, so this adds nothing to start-up time.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)


class ClickableLabel(QLabel):
    """A label that says when it is double-clicked (the pulse-programme
    name in the status bar)."""

    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt name)
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


_UID = Qt.ItemDataRole.UserRole
_GROUPS = (("Pulses", ("hard", "shaped", "decoupling")), ("Gradients", ("gradient",)),
           ("Delays", ("delay",)))


class PulseSequenceDialog(QDialog):
    """``spectrum``: the open experiment (its own parameter values fill in
    ``d1``, ``p1``...). ``settings``: the window's ``_ui_settings`` dict,
    where the colours and options persist; ``save_settings`` writes it."""

    def __init__(self, spectrum=None, settings: dict | None = None, save_settings=None,
                 parent=None) -> None:
        super().__init__(parent)
        from ..notebook import pulse_sequence as ps   # lazy: matplotlib
        self._ps = ps
        self.setWindowTitle("Pulse sequence")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(1100, 700)
        self.spectrum = spectrum
        self._settings = settings if settings is not None else {}
        self._save_settings = save_settings or (lambda: None)
        self.sequences: list = []
        self.sequence = None
        self._flags: dict = {}
        self._path: str | None = None
        self._building = False
        self.colors = {**ps.DEFAULT_COLORS, **(self._settings.get("pulse_sequence_colors") or {})}
        opts = self._settings.get("pulse_sequence_options") or {}
        self._opts = {"show_minor": bool(opts.get("show_minor", False)),
                      "show_values": bool(opts.get("show_values", False)),
                      "show_loops": bool(opts.get("show_loops", True))}

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.source_label = QLabel("")
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.sequence_combo = QComboBox()
        self.sequence_combo.setToolTip("A file can hold several sequences (an X-Pulse library).")
        self.sequence_combo.currentIndexChanged.connect(self._on_sequence_chosen)
        self.open_button = QPushButton("Open pulse sequence file…")
        self.open_button.setToolTip(
            "Bruker keeps the programme with the data; Varian, JEOL, Magritek and X-Pulse "
            "keep only its name, so point at the .c / .jxp / _pp.mac / .py file.")
        self.open_button.clicked.connect(self._choose_file)
        top.addWidget(self.source_label, 1)
        top.addWidget(self.sequence_combo)
        top.addWidget(self.open_button)
        root.addLayout(top)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color:#8a4b00;")
        root.addWidget(self.message)

        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        self.tabs = QTabWidget()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        self._Figure = Figure
        self.figure = Figure(figsize=(8, 4.5), facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        diagram = QWidget()
        dl = QVBoxLayout(diagram)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(NavigationToolbar2QT(self.canvas, diagram))
        dl.addWidget(self.canvas, 1)
        self.tabs.addTab(diagram, "Diagram")
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["#", "Kind", "Channel", "Name", "Expression", "Duration", "Shape", "Phase"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.table, "Events")
        split.addWidget(self.tabs)

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        # colours
        cbox = QGroupBox("Colours")
        cg = QGridLayout(cbox)
        self.colour_buttons: dict = {}
        for row, kind in enumerate(("hard", "shaped", "decoupling", "gradient", "acquire", "delay")):
            cg.addWidget(QLabel(ps.KIND_NAMES[kind]), row, 0)
            b = QPushButton()
            b.setFixedWidth(60)
            b.setToolTip(f"Colour of the {ps.KIND_NAMES[kind].lower()}s.")
            b.clicked.connect(lambda _c=False, k=kind: self._pick_colour(k))
            cg.addWidget(b, row, 1)
            self.colour_buttons[kind] = b
        reset = QPushButton("Reset colours")
        reset.clicked.connect(self._reset_colours)
        cg.addWidget(reset, 6, 0, 1, 2)
        sl.addWidget(cbox)
        # names
        nbox = QGroupBox("Names shown")
        nl = QVBoxLayout(nbox)
        row = QHBoxLayout()
        all_b, none_b = QPushButton("All"), QPushButton("None")
        all_b.clicked.connect(lambda: self._set_all_names(True))
        none_b.clicked.connect(lambda: self._set_all_names(False))
        row.addWidget(all_b)
        row.addWidget(none_b)
        nl.addLayout(row)
        self.name_tree = QTreeWidget()
        self.name_tree.setHeaderHidden(True)
        self.name_tree.itemChanged.connect(self._on_names_changed)
        nl.addWidget(self.name_tree, 1)
        sl.addWidget(nbox, 1)
        # options
        obox = QGroupBox("Display")
        ol = QVBoxLayout(obox)
        self.minor_check = QCheckBox("Show short housekeeping delays")
        self.minor_check.setToolTip("Delays such as 30u, 1m or the dead time 'de' that only "
                                    "set the hardware up are hidden by default.")
        self.values_check = QCheckBox("Show durations under the names")
        self.values_check.setToolTip("Only where the data record the value (e.g. d1, p1 of a "
                                     "Bruker experiment).")
        self.loops_check = QCheckBox("Show loops")
        for chk, key in ((self.minor_check, "show_minor"), (self.values_check, "show_values"),
                         (self.loops_check, "show_loops")):
            chk.setChecked(self._opts[key])
            chk.toggled.connect(lambda v, k=key: self._set_option(k, v))
            ol.addWidget(chk)
        sl.addWidget(obox)
        # conditional blocks
        self.flag_box = QGroupBox("Conditional blocks (#ifdef / flags)")
        self.flag_layout = QVBoxLayout(self.flag_box)
        sl.addWidget(self.flag_box)
        split.addWidget(side)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)

        bottom = QHBoxLayout()
        self.note = QLabel("Schematic – not to scale: long delays are compressed so every "
                           "pulse stays visible.")
        self.note.setStyleSheet("color:#666;")
        bottom.addWidget(self.note, 1)
        save = QPushButton("Save figure…")
        save.clicked.connect(self._save_figure)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        bottom.addWidget(save)
        bottom.addWidget(close)
        root.addLayout(bottom)

        self._refresh_colour_buttons()
        self.load()

    # -- loading ---------------------------------------------------------
    def set_spectrum(self, spectrum) -> None:
        self.spectrum = spectrum
        self._flags = {}
        self._path = None
        self.load()

    def load(self, path: str | None = None) -> None:
        """(Re)read the sequence -- from ``path``, else the experiment's own."""
        ps = self._ps
        self.message.setText("")
        if path:
            self._path = path
        try:
            if self.spectrum is not None:
                self.sequences = ps.pulse_sequence_for(self.spectrum, path=self._path,
                                                       flags=self._flags or None)
                self._path = ps.find_sequence_file(self.spectrum) if not self._path else self._path
            elif self._path:
                self.sequences = ps.read_pulse_sequences(self._path, flags=self._flags or None)
            else:
                raise ps.PulseSequenceError("No pulse sequence file chosen.")
        except ps.PulseSequenceError as exc:
            self.sequences = []
            self.sequence = None
            self.message.setText(str(exc))
            self._fill_combo()
            self._redraw()
            return
        except Exception as exc:  # a malformed file must not take the window down
            self.sequences = []
            self.sequence = None
            self.message.setText(f"Could not read this pulse sequence: {exc}")
            self._fill_combo()
            self._redraw()
            return
        name = getattr(getattr(self.spectrum, "metadata", None), "pulse_program", None)
        self.source_label.setText(
            f"<b>{name or ''}</b>  {os.path.basename(self._path) if self._path else ''}")
        self.setWindowTitle(f"Pulse sequence — {name or os.path.basename(self._path or '')}")
        self._fill_combo()

    def _fill_combo(self) -> None:
        self._building = True
        self.sequence_combo.clear()
        for s in self.sequences:
            self.sequence_combo.addItem(s.name)
        self.sequence_combo.setVisible(len(self.sequences) > 1)
        self._building = False
        if self.sequences:
            self.sequence_combo.setCurrentIndex(0)
            self._select_sequence(0)
        else:
            self._rebuild_flags()
            self._rebuild_names()

    def _on_sequence_chosen(self, i: int) -> None:
        if not self._building and i >= 0:
            self._select_sequence(i)

    def _select_sequence(self, i: int) -> None:
        self.sequence = self.sequences[i]
        if self.spectrum is not None and not self.sequence.channel_nuclei:
            self.sequence.channel_nuclei = self._ps.channel_nuclei(self.spectrum)
        self._rebuild_flags()
        self._rebuild_names()
        self._fill_table()
        self._redraw()

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open pulse sequence file", os.path.dirname(self._path or "") or "",
            "Pulse sequences (*.c *.enb *.jxp *.mac *.py pulseprogram *);;All files (*)")
        if not path:
            return
        self._flags = {}
        self.load(path)
        if self.sequences and self.spectrum is not None:
            try:
                self.spectrum.metadata.raw["pulse_sequence_file"] = path
            except Exception:
                pass

    # -- controls --------------------------------------------------------
    def _rebuild_flags(self) -> None:
        while self.flag_layout.count():
            w = self.flag_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        flags = self.sequence.flags if self.sequence else {}
        self.flag_box.setVisible(bool(flags))
        self.flag_checks = {}
        for name, state in flags.items():
            c = QCheckBox(name)
            c.setChecked(bool(state))
            c.toggled.connect(lambda v, n=name: self._on_flag(n, v))
            self.flag_layout.addWidget(c)
            self.flag_checks[name] = c

    def _on_flag(self, name: str, value: bool) -> None:
        self._flags = {n: c.isChecked() for n, c in self.flag_checks.items()}
        keep = self.sequence_combo.currentIndex()
        self.load()
        if 0 < keep < len(self.sequences):
            self.sequence_combo.setCurrentIndex(keep)

    def _rebuild_names(self) -> None:
        self._building = True
        self.name_tree.clear()
        seq = self.sequence
        if seq:
            for title, kinds in _GROUPS:
                events = [e for e in seq.events if e.kind in kinds and e.label]
                if not events:
                    continue
                parent = QTreeWidgetItem([title])
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable
                                | Qt.ItemFlag.ItemIsAutoTristate)
                self.name_tree.addTopLevelItem(parent)
                for e in events:
                    txt = f"{e.label}"
                    if e.kind in ("shaped", "decoupling"):
                        txt += f"  ({self._ps.KIND_NAMES[e.kind].split(' /')[0].lower()})"
                    if e.value is not None:
                        txt += f"   {e.value_text()}"
                    if e.minor:
                        txt += "   (short)"
                    child = QTreeWidgetItem([txt])
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.CheckState.Checked)
                    child.setData(0, _UID, e.uid)
                    parent.addChild(child)
                parent.setExpanded(title != "Delays" or len(events) < 12)
        self._building = False

    def _shown_uids(self) -> set:
        out = set()
        for i in range(self.name_tree.topLevelItemCount()):
            parent = self.name_tree.topLevelItem(i)
            for j in range(parent.childCount()):
                ch = parent.child(j)
                if ch.checkState(0) == Qt.CheckState.Checked:
                    out.add(ch.data(0, _UID))
        return out

    def _on_names_changed(self, *_a) -> None:
        if not self._building:
            self._redraw()

    def _set_all_names(self, on: bool) -> None:
        self._building = True
        for i in range(self.name_tree.topLevelItemCount()):
            self.name_tree.topLevelItem(i).setCheckState(
                0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._building = False
        self._redraw()

    def _set_option(self, key: str, value: bool) -> None:
        self._opts[key] = bool(value)
        self._settings["pulse_sequence_options"] = dict(self._opts)
        self._save_settings()
        self._redraw()

    def _pick_colour(self, kind: str) -> None:
        c = QColorDialog.getColor(QColor(self.colors[kind]), self,
                                  f"Colour of the {self._ps.KIND_NAMES[kind].lower()}s")
        if c.isValid():
            self.colors[kind] = c.name()
            self._store_colours()

    def _reset_colours(self) -> None:
        self.colors = dict(self._ps.DEFAULT_COLORS)
        self._store_colours()

    def _store_colours(self) -> None:
        changed = {k: v for k, v in self.colors.items() if v != self._ps.DEFAULT_COLORS.get(k)}
        self._settings["pulse_sequence_colors"] = changed
        self._save_settings()
        self._refresh_colour_buttons()
        self._redraw()

    def _refresh_colour_buttons(self) -> None:
        for kind, b in self.colour_buttons.items():
            b.setStyleSheet(f"background-color:{self.colors[kind]}; border:1px solid #888;")

    # -- drawing ---------------------------------------------------------
    def draw_options(self) -> dict:
        return dict(colors=dict(self.colors), show_names=self._shown_uids(),
                    show_values=self._opts["show_values"],
                    show_minor_delays=self._opts["show_minor"],
                    show_loops=self._opts["show_loops"])

    def _redraw(self) -> None:
        self.figure.clear()
        if self.sequence is None:
            ax = self.figure.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No pulse sequence to show", ha="center", va="center",
                    color="#777", transform=ax.transAxes)
        else:
            ax = self.figure.add_subplot(111)
            self._ps.plot_pulse_sequence(self.sequence, ax, **self.draw_options())
            self.figure.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.04)
        self.canvas.draw_idle()

    def _fill_table(self) -> None:
        rows = self._ps.event_table(self.sequence) if self.sequence else []
        self.table.setRowCount(len(rows))
        keys = ["#", "kind", "channel", "name", "expression", "duration", "shape", "phase"]
        for r, row in enumerate(rows):
            for c, k in enumerate(keys):
                self.table.setItem(r, c, QTableWidgetItem(str(row[k])))

    def _save_figure(self) -> None:
        if self.sequence is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save pulse sequence figure", f"{self.sequence.name}.pdf",
            "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)")
        if path:
            self.save_figure(path)

    def save_figure(self, path: str, dpi: int = 300) -> str:
        """Write the diagram with the current colours, names and options,
        at its own proportions (not the window's)."""
        opts = self.draw_options()
        fig = self._Figure(figsize=self._ps.figure_size(
            self.sequence, opts["show_minor_delays"], opts["show_loops"]), facecolor="white")
        ax = fig.add_subplot(111)
        self._ps.plot_pulse_sequence(self.sequence, ax, **opts)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.04)
        fig.savefig(path, dpi=dpi, facecolor="white")
        return path
