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
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QLineEdit, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from .widgets.typed_spinbox import TypedDoubleSpinBox, TypedSpinBox


class ClickableLabel(QLabel):
    """A label that says when it is double-clicked (the pulse-programme
    name in the status bar)."""

    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt name)
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class _ShowAllCheck(QCheckBox):
    """"Show all": a click on the half-ticked state (some, not all, shown)
    ticks everything rather than clearing it."""

    def nextCheckState(self) -> None:  # noqa: N802 (Qt name)
        self.setCheckState(Qt.CheckState.Unchecked if self.checkState() == Qt.CheckState.Checked
                           else Qt.CheckState.Checked)


_UID = Qt.ItemDataRole.UserRole
_EXPORT_FORMATS = ("pdf", "svg", "png", "tiff", "eps")
_GROUPS = (("Pulses", ("hard", "shaped", "decoupling")), ("Gradients", ("gradient",)),
           ("Delays", ("delay",)))


def saved_draw_options(settings: dict | None, sequence_name: str = "") -> dict:
    """The look the pulse-sequence viewer has saved -- its colours, options and
    the renames kept for ``sequence_name`` -- as keywords for
    :func:`opennmr.notebook.pulse_sequence.plot_pulse_sequence`, without opening
    the viewer. What Compare uses to draw a pulse sequence in a panel in the
    style the user has set up (every name shown: which names are ticked in the
    viewer's tree is not kept). ``settings`` is the window's ``_ui_settings``."""
    from ..notebook import pulse_sequence as ps
    settings = settings or {}
    saved = settings.get("pulse_sequence_options") or {}
    style = saved.get("delay_style")
    if style not in ps.DELAY_STYLES:
        style = "line"
    opts = {"show_values": bool(saved.get("show_values", False)),
            "show_loops": bool(saved.get("show_loops", True)), "delay_style": style}
    for key, default in PulseSequenceDialog._STYLE_DEFAULTS.items():
        value = saved.get(key, default)
        opts[key] = value if isinstance(value, type(default)) or (
            isinstance(default, float) and isinstance(value, int)) else default
    for key, allowed in (("legend_position", ps.LEGEND_POSITIONS), ("title_position", ps.TITLE_POSITIONS)):
        if opts[key] not in allowed:
            opts[key] = PulseSequenceDialog._STYLE_DEFAULTS[key]
    renames = (settings.get("pulse_sequence_renames") or {}).get(sequence_name) or {}
    return dict(colors={**ps.DEFAULT_COLORS, **(settings.get("pulse_sequence_colors") or {})},
                show_values=opts["show_values"], show_loops=opts["show_loops"],
                delay_style=opts["delay_style"],
                renames=dict(renames.get("names") or {}),
                channel_labels=dict(renames.get("channels") or {}),
                legend_labels=dict(renames.get("legend") or {}),
                acquire_separate=opts["acquire_separate"], fontfamily=opts["fontfamily"] or None,
                fontsize=float(opts["fontsize"]),
                channel_fontsize=float(opts["channel_fontsize"]) or None,
                legend_fontsize=float(opts["legend_fontsize"]) or None,
                title_fontsize=float(opts["title_fontsize"]) or None,
                linewidth=float(opts["linewidth"]), pulse_height=float(opts["pulse_height"]),
                title=opts["title"], title_text=opts["title_text"] or None,
                title_position=opts["title_position"], legend=opts["legend"],
                legend_position=opts["legend_position"], note=opts["note"])


class PulseSequenceDialog(QDialog):
    """``spectrum``: the open experiment (its own parameter values fill in
    ``d1``, ``p1``...). ``settings``: the window's ``_ui_settings`` dict,
    where the colours and options persist; ``save_settings`` writes it."""

    #: Look-and-feel options (and their defaults) kept with the window's settings.
    _STYLE_DEFAULTS = {
        "acquire_separate": False, "fontfamily": "", "fontsize": 10.0, "channel_fontsize": 0.0,
        "legend_fontsize": 0.0, "title_fontsize": 0.0, "linewidth": 1.0, "pulse_height": 1.0,
        "title": True, "title_text": "", "title_position": "left", "legend": True,
        "legend_position": "below", "note": True,
        "export_width": 0.0, "export_height": 0.0, "export_dpi": 300, "export_format": "pdf",
    }

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
        style = opts.get("delay_style")
        if style not in ps.DELAY_STYLES:
            style = "line"            # (an old saved "delays_inline" choice is superseded)
        self._opts = {"show_values": bool(opts.get("show_values", False)),
                      "show_loops": bool(opts.get("show_loops", True)),
                      "delay_style": style}
        for key, default in self._STYLE_DEFAULTS.items():
            value = opts.get(key, default)
            self._opts[key] = value if isinstance(value, type(default)) or (
                isinstance(default, float) and isinstance(value, int)) else default
        for key, allowed in (("legend_position", ps.LEGEND_POSITIONS),
                             ("title_position", ps.TITLE_POSITIONS)):
            if self._opts[key] not in allowed:
                self._opts[key] = self._STYLE_DEFAULTS[key]
        self._renames = {"names": {}, "channels": {}, "legend": {}}

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
        dialog = self

        class _Toolbar(NavigationToolbar2QT):
            def home(self, *args):      # the house icon: the whole diagram, even after scroll / drag
                dialog.reset_view()

        self.toolbar = _Toolbar(self.canvas, diagram)
        dl.addWidget(self.toolbar)
        dl.addWidget(self.canvas, 1)
        # Scroll to zoom along time (Ctrl: vertical, Shift: pan), drag to pan,
        # double-click for the full view -- unless the toolbar's own pan /
        # zoom tool is active and owns the mouse.
        self._disable_zoom = self._ps.enable_zoom_pan(
            self.figure, busy=lambda: bool(getattr(self.toolbar, "mode", "")),
            drag_mode=lambda: "pan" if self._settings.get("drag_mode") == "pan" else "zoom")
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
        self.values_check = QCheckBox("Show durations under the names")
        self.values_check.setToolTip("Only where the data record the value (e.g. d1, p1 of a "
                                     "Bruker experiment).")
        self.loops_check = QCheckBox("Show loops")
        for chk, key in ((self.values_check, "show_values"), (self.loops_check, "show_loops")):
            chk.setChecked(self._opts[key])
            chk.toggled.connect(lambda v, k=key: self._set_option(k, v))
            ol.addWidget(chk)
        self.acquire_check = QCheckBox("Acquisition on the pulse line")
        self.acquire_check.setToolTip(
            "On (default): the acquisition is drawn on the first channel's line after the "
            "pulses and delays, as in TopSpin. Off: on a line of its own.")
        self.acquire_check.setChecked(not self._opts["acquire_separate"])
        self.acquire_check.toggled.connect(lambda v: self._set_option("acquire_separate", not v))
        ol.addWidget(self.acquire_check)
        ol.addWidget(QLabel("Delays:"))
        self.delay_style_combo = QComboBox()
        for key, text in (("line", "Named on the pulse line"),
                          ("brackets", "Brackets under the pulses"),
                          ("block", "Brackets in a separate block")):
            self.delay_style_combo.addItem(text, key)
        self.delay_style_combo.setCurrentIndex(
            max(self.delay_style_combo.findData(self._opts["delay_style"]), 0))
        self.delay_style_combo.setToolTip(
            "Named on the pulse line (default): each delay's name is written in its gap, just "
            "above the first channel's line.\nBrackets: a bracket per delay, directly under that "
            "line or in a block of their own.\nUntick a delay under Names shown to leave it out; "
            "with none ticked there is no delay row at all.")
        self.delay_style_combo.currentIndexChanged.connect(
            lambda _i: self._set_delay_style(self.delay_style_combo.currentData()))
        ol.addWidget(self.delay_style_combo)
        sl.addWidget(obox)
        # conditional blocks
        self.flag_box = QGroupBox("Conditional blocks (#ifdef / flags)")
        self.flag_layout = QVBoxLayout(self.flag_box)
        sl.addWidget(self.flag_box)
        self._build_style_groups(sl)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(side)
        scroll.setMinimumWidth(300)
        split.addWidget(scroll)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)

        bottom = QHBoxLayout()
        self.note = QLabel("Schematic – not to scale: long delays are compressed so every "
                           "pulse stays visible. Scroll to zoom; drag zooms into a box or pans (Settings ▸ Click and drag); double-click to reset.")
        self.note.setStyleSheet("color:#666;")
        bottom.addWidget(self.note, 1)
        reset_view = QPushButton("Reset view")
        reset_view.setToolTip("Back to the whole sequence after scrolling / dragging "
                              "(double-clicking the diagram does the same).")
        reset_view.clicked.connect(self.reset_view)
        bottom.addWidget(reset_view)
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
            self._load_renames()

    def _on_sequence_chosen(self, i: int) -> None:
        if not self._building and i >= 0:
            self._select_sequence(i)

    def _select_sequence(self, i: int) -> None:
        self.sequence = self.sequences[i]
        if self.spectrum is not None and not self.sequence.channel_nuclei:
            self.sequence.channel_nuclei = self._ps.channel_nuclei(self.spectrum)
        self._rebuild_flags()
        self._rebuild_names()
        self._load_renames()
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
                events = [e for e in seq.events if e.kind in kinds and e.name]
                if not events:
                    continue
                parent = QTreeWidgetItem([title])
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable
                                | Qt.ItemFlag.ItemIsAutoTristate)
                self.name_tree.addTopLevelItem(parent)
                for e in events:
                    txt = f"{e.name}"
                    if e.kind in ("shaped", "decoupling"):
                        txt += f"  ({self._ps.KIND_NAMES[e.kind].split(' /')[0].lower()})"
                    if e.value is not None and e.name != e.value_text():
                        txt += f"   {e.value_text()}"
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
            self._redraw(keep_view=True)

    def _set_all_names(self, on: bool) -> None:
        self._building = True
        for i in range(self.name_tree.topLevelItemCount()):
            self.name_tree.topLevelItem(i).setCheckState(
                0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._building = False
        self._redraw(keep_view=True)

    def _set_delay_style(self, style) -> None:
        if style in self._ps.DELAY_STYLES:
            self._opts["delay_style"] = style
            self._settings["pulse_sequence_options"] = dict(self._opts)
            self._save_settings()
            self._redraw()

    #: Options that change the layout (the view goes back to the whole diagram).
    _LAYOUT_KEYS = ("show_values", "show_loops", "acquire_separate", "pulse_height", "fontsize",
                    "channel_fontsize", "title", "legend", "legend_position", "note")

    def _set_option(self, key: str, value) -> None:
        self._opts[key] = value if not isinstance(value, (bool, int)) or key == "export_dpi" else bool(value)
        self._settings["pulse_sequence_options"] = dict(self._opts)
        self._save_settings()
        self._redraw(keep_view=key not in self._LAYOUT_KEYS)

    # -- look and feel ---------------------------------------------------
    def _spin(self, key: str, lo: float, hi: float, step: float, decimals: int = 2, special: str = ""):
        box = TypedDoubleSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        if special:
            box.setSpecialValueText(special)
        box.setValue(float(self._opts[key]))
        box.valueChanged.connect(lambda v, k=key: self._set_option(k, float(v)))
        return box

    def _build_style_groups(self, sl) -> None:
        from PySide6.QtGui import QFontDatabase
        ps = self._ps
        # rename
        rbox = QGroupBox("Rename (drawing only)")
        rl = QVBoxLayout(rbox)
        hint = QLabel("Double-click a name in the right-hand column to change what the diagram "
                      "shows; the file and the order stay as they are. LaTeX works: "
                      "<code>\\tau_{c}</code>, <code>\\pi/2</code>, <code>$t_1$</code>. "
                      "Empty = the original.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666;")
        rl.addWidget(hint)
        self.rename_table = QTableWidget(0, 2)
        self.rename_table.setHorizontalHeaderLabels(["Name in the file", "Shown as"])
        self.rename_table.verticalHeader().setVisible(False)
        self.rename_table.horizontalHeader().setStretchLastSection(True)
        self.rename_table.setMinimumHeight(150)
        self.rename_table.itemChanged.connect(self._on_rename_changed)
        rl.addWidget(self.rename_table)
        reset = QPushButton("Reset names")
        reset.clicked.connect(self._reset_renames)
        rl.addWidget(reset)
        sl.addWidget(rbox)
        # text
        tbox = QGroupBox("Text")
        tg = QGridLayout(tbox)
        tg.addWidget(QLabel("Typeface"), 0, 0)
        self.font_combo = QComboBox()
        self.font_combo.setEditable(True)
        self.font_combo.addItem("(default)", "")
        self.font_combo.addItems(sorted(set(QFontDatabase.families())))
        current = self._opts["fontfamily"]
        self.font_combo.setCurrentText(current if current else "(default)")
        self.font_combo.setToolTip("One typeface for every label, the legend, the title and the "
                                   "maths in $...$ names.")
        self.font_combo.currentTextChanged.connect(self._on_font_chosen)
        tg.addWidget(self.font_combo, 0, 1)
        self.size_spins = {}
        for row, (key, text, tip) in enumerate((
                ("fontsize", "Names", "Size of the pulse and delay names (pt)."),
                ("channel_fontsize", "Channels", "Size of the channel labels (0 = as Names)."),
                ("legend_fontsize", "Legend", "Size of the legend (0 = automatic)."),
                ("title_fontsize", "Title", "Size of the title (0 = automatic)."))):
            tg.addWidget(QLabel(text), row + 1, 0)
            box = self._spin(key, 0.0 if key != "fontsize" else 4.0, 60.0, 0.5, 1,
                             special="auto" if key != "fontsize" else "")
            box.setToolTip(tip)
            tg.addWidget(box, row + 1, 1)
            self.size_spins[key] = box
        sl.addWidget(tbox)
        # lines and pulses
        lbox = QGroupBox("Lines and pulses")
        lg = QGridLayout(lbox)
        lg.addWidget(QLabel("Line thickness"), 0, 0)
        self.linewidth_spin = self._spin("linewidth", 0.2, 8.0, 0.25, 2)
        self.linewidth_spin.setToolTip("Multiplies every line: baselines, brackets, pulse outlines, loops.")
        lg.addWidget(self.linewidth_spin, 0, 1)
        lg.addWidget(QLabel("Pulse height"), 1, 0)
        self.pulse_height_spin = self._spin("pulse_height", 0.3, 4.0, 0.1, 2)
        self.pulse_height_spin.setToolTip("Multiplies the height of every pulse, gradient and the "
                                          "acquisition by the same factor.")
        lg.addWidget(self.pulse_height_spin, 1, 1)
        sl.addWidget(lbox)
        # title and legend
        gbox = QGroupBox("Title and legend")
        gg = QGridLayout(gbox)
        self.title_check = QCheckBox("Title")
        self.title_check.setChecked(self._opts["title"])
        self.title_check.toggled.connect(lambda v: self._set_option("title", bool(v)))
        gg.addWidget(self.title_check, 0, 0)
        self.title_edit = QLineEdit(self._opts["title_text"])
        self.title_edit.setPlaceholderText("(the sequence name)")
        self.title_edit.setToolTip("Text of the title; LaTeX works. Empty = the sequence's name.")
        self.title_edit.editingFinished.connect(
            lambda: self._set_option("title_text", self.title_edit.text()))
        gg.addWidget(self.title_edit, 0, 1)
        gg.addWidget(QLabel("Title position"), 1, 0)
        self.title_position_combo = QComboBox()
        self.title_position_combo.addItems(list(ps.TITLE_POSITIONS))
        self.title_position_combo.setCurrentText(self._opts["title_position"])
        self.title_position_combo.currentTextChanged.connect(
            lambda t: self._set_option("title_position", t))
        gg.addWidget(self.title_position_combo, 1, 1)
        self.legend_check = QCheckBox("Legend")
        self.legend_check.setChecked(self._opts["legend"])
        self.legend_check.toggled.connect(lambda v: self._set_option("legend", bool(v)))
        gg.addWidget(self.legend_check, 2, 0)
        self.legend_position_combo = QComboBox()
        self.legend_position_combo.addItems([p for p in ps.LEGEND_POSITIONS if p != "none"])
        self.legend_position_combo.setCurrentText(
            self._opts["legend_position"] if self._opts["legend_position"] != "none" else "below")
        self.legend_position_combo.currentTextChanged.connect(
            lambda t: self._set_option("legend_position", t))
        gg.addWidget(self.legend_position_combo, 2, 1)
        self.note_check = QCheckBox("“schematic – not to scale” note")
        self.note_check.setChecked(self._opts["note"])
        self.note_check.toggled.connect(lambda v: self._set_option("note", bool(v)))
        gg.addWidget(self.note_check, 3, 0, 1, 2)
        sl.addWidget(gbox)
        # export
        ebox = QGroupBox("Export")
        eg = QGridLayout(ebox)
        eg.addWidget(QLabel("Width (in)"), 0, 0)
        self.export_width_spin = self._spin("export_width", 0.0, 60.0, 0.5, 2, special="auto")
        eg.addWidget(self.export_width_spin, 0, 1)
        eg.addWidget(QLabel("Height (in)"), 1, 0)
        self.export_height_spin = self._spin("export_height", 0.0, 60.0, 0.5, 2, special="auto")
        eg.addWidget(self.export_height_spin, 1, 1)
        eg.addWidget(QLabel("DPI"), 2, 0)
        self.export_dpi_spin = TypedSpinBox()
        self.export_dpi_spin.setRange(50, 4800)
        self.export_dpi_spin.setValue(int(self._opts["export_dpi"]))
        self.export_dpi_spin.valueChanged.connect(lambda v: self._set_option("export_dpi", int(v)))
        eg.addWidget(self.export_dpi_spin, 2, 1)
        eg.addWidget(QLabel("Format"), 3, 0)
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(list(_EXPORT_FORMATS))
        self.export_format_combo.setCurrentText(self._opts["export_format"])
        self.export_format_combo.currentTextChanged.connect(
            lambda t: self._set_option("export_format", t))
        eg.addWidget(self.export_format_combo, 3, 1)
        note = QLabel("Width / height “auto” keep the diagram's own size; give one and the other "
                      "follows. DPI matters for PNG and TIFF; PDF, SVG and EPS stay vector.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666;")
        eg.addWidget(note, 4, 0, 1, 2)
        sl.addWidget(ebox)

    def _on_font_chosen(self, text: str) -> None:
        family = "" if text in ("", "(default)") else text
        self._set_option("fontfamily", family)

    # -- renaming --------------------------------------------------------
    def _renames_key(self) -> str:
        return self.sequence.name if self.sequence else ""

    def _load_renames(self) -> None:
        stored = (self._settings.get("pulse_sequence_renames") or {}).get(self._renames_key()) or {}
        self._renames = {g: dict(stored.get(g) or {}) for g in ("names", "channels", "legend")}
        self._rebuild_renames()

    def _save_renames(self) -> None:
        book = self._settings.setdefault("pulse_sequence_renames", {})
        mine = {g: v for g, v in self._renames.items() if v}
        if mine:
            book[self._renames_key()] = mine
        else:
            book.pop(self._renames_key(), None)
        self._save_settings()

    def _rebuild_renames(self) -> None:
        self._building = True
        t = self.rename_table
        t.setRowCount(0)
        rows = []
        seq = self.sequence
        if seq:
            seen = set()
            for e in seq.events:
                name = e.label if e.kind != "delay" else e.name
                if e.kind == "acquire" or not name or name in seen:
                    continue
                seen.add(name)
                rows.append(("names", name, f"{name}  ({self._ps.KIND_NAMES.get(e.kind, e.kind).lower()})"))
            nuc = getattr(seq, "channel_nuclei", {}) or {}
            chans = seq.channels or ["f1"]
            for c in chans:
                rows.append(("channels", c, f"channel {c}" + (f"  ({nuc[c]})" if c in nuc else "")))
            for k in ("hard", "shaped", "decoupling", "gradient", "acquire"):
                if any(e.kind == k for e in seq.events):
                    rows.append(("legend", k, f"legend: {self._ps.KIND_NAMES[k]}"))
        for group, key, text in rows:
            r = t.rowCount()
            t.insertRow(r)
            left = QTableWidgetItem(text)
            left.setFlags(left.flags() & ~Qt.ItemFlag.ItemIsEditable)
            left.setData(_UID, (group, key))
            t.setItem(r, 0, left)
            t.setItem(r, 1, QTableWidgetItem(self._renames[group].get(key, "")))
        t.resizeColumnToContents(0)
        self._building = False

    def _on_rename_changed(self, item) -> None:
        if self._building or item.column() != 1:
            return
        group, key = self.rename_table.item(item.row(), 0).data(_UID)
        text = item.text().strip()
        if text:
            self._renames[group][key] = text
        else:
            self._renames[group].pop(key, None)
        self._save_renames()
        self._redraw(keep_view=True)

    def _reset_renames(self) -> None:
        self._renames = {"names": {}, "channels": {}, "legend": {}}
        self._save_renames()
        self._rebuild_renames()
        self._redraw(keep_view=True)

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
        self._redraw(keep_view=True)

    def _refresh_colour_buttons(self) -> None:
        for kind, b in self.colour_buttons.items():
            b.setStyleSheet(f"background-color:{self.colors[kind]}; border:1px solid #888;")

    # -- drawing ---------------------------------------------------------
    def draw_options(self) -> dict:
        return dict(colors=dict(self.colors), show_names=self._shown_uids(),
                    show_values=self._opts["show_values"],
                    show_loops=self._opts["show_loops"],
                    delay_style=self._opts["delay_style"],
                    renames=dict(self._renames["names"]),
                    channel_labels=dict(self._renames["channels"]),
                    legend_labels=dict(self._renames["legend"]),
                    acquire_separate=self._opts["acquire_separate"],
                    fontfamily=self._opts["fontfamily"] or None,
                    fontsize=float(self._opts["fontsize"]),
                    channel_fontsize=float(self._opts["channel_fontsize"]) or None,
                    legend_fontsize=float(self._opts["legend_fontsize"]) or None,
                    title_fontsize=float(self._opts["title_fontsize"]) or None,
                    linewidth=float(self._opts["linewidth"]),
                    pulse_height=float(self._opts["pulse_height"]),
                    title=self._opts["title"],
                    title_text=self._opts["title_text"] or None,
                    title_position=self._opts["title_position"],
                    legend=self._opts["legend"], legend_position=self._opts["legend_position"],
                    note=self._opts["note"])

    def _redraw(self, keep_view: bool = False) -> None:
        """Draw the sequence again. ``keep_view``: stay at the zoomed /
        panned region (for a change that does not move anything, such as a
        colour or which names are shown)."""
        view = None
        if keep_view and self.figure.axes:
            ax0 = self.figure.axes[0]
            if hasattr(ax0, "_opennmr_home_view"):
                view = (ax0.get_xlim(), ax0.get_ylim())
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
            if view is not None:
                ax.set_xlim(view[0])
                ax.set_ylim(view[1])
        self.canvas.draw_idle()

    def reset_view(self) -> None:
        """Back to the whole sequence (after scrolling / dragging)."""
        if self.figure.axes and hasattr(self.figure.axes[0], "_opennmr_home_view"):
            ax = self.figure.axes[0]
            ax.set_xlim(ax._opennmr_home_view[0])
            ax.set_ylim(ax._opennmr_home_view[1])
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
        fmt = self._opts["export_format"]
        filters = ";;".join(f"{f.upper()} (*.{f})" for f in _EXPORT_FORMATS)
        path, chosen = QFileDialog.getSaveFileName(
            self, "Save pulse sequence figure", f"{self.sequence.name}.{fmt}", filters)
        if path:
            if not os.path.splitext(path)[1]:
                path += f".{fmt}"
            self.save_figure(path)

    def save_figure(self, path: str, dpi: int | None = None, width: float | None = None,
                    height: float | None = None) -> str:
        """Write the diagram with the current colours, names and options, at
        the Export size and DPI (inches; a missing size keeps the diagram's
        own proportions). The format follows the file name's extension."""
        import matplotlib.pyplot as plt
        dpi = int(dpi or self._opts["export_dpi"])
        width = width if width is not None else float(self._opts["export_width"])
        height = height if height is not None else float(self._opts["export_height"])
        fig, ax = self._ps.plot_pulse_sequence(self.sequence, None, **self.draw_options())
        try:
            w0, h0 = fig.get_size_inches()
            if width and not height:
                height = width * h0 / w0
            elif height and not width:
                width = height * w0 / h0
            if width and height:
                fig.set_size_inches(width, height)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.93 if self._opts["title"] else 0.99,
                                bottom=0.04)
            fig.savefig(path, dpi=dpi, facecolor="white")
        finally:
            plt.close(fig)
        return path
