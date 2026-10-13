"""
The NMR Periodic Table view and its Larmor-frequency calculator.

Explicit user request (2026-09-28): a launcher button, just before
Settings, that shows an "NMR Periodic Table" in place of the spectrum --
searchable, highlighting what matches, and click an element to see its
nuclides -- modelled on https://euannbassey.github.io/periodic_table.html,
with a Larmor calculator in the left-hand panel.

Two widgets, which the main window wires together:

* :class:`PeriodicTableView` -- the table (replaces the plot while the
  tile is open). A search box highlights matches (symbol, name, Z, or a
  nuclide such as "27Al"; Enter opens the best match); the spin legend
  dims every element without an isotope of that spin; "Tiles show"
  switches each tile's bottom line between atomic weight, Larmor
  frequency at the calculator's field, receptivity and principal
  isotope. Clicking a tile opens a card over the table with the
  element's electron configuration and every NMR isotope -- spin,
  abundance, gamma, Q, receptivity and Larmor frequency at the current
  field. Clicking an isotope row sends it to the calculator.
* :class:`LarmorCalculatorPanel` -- the left panel: the field (tesla, or
  the magnet's 1H frequency, or a common magnet), the nucleus, and the
  result with its working written out, plus every nucleus at that field
  sorted by frequency.

Neither holds any nuclear data: both read
:mod:`opennmr.notebook.periodic_table`, which reads openNMR's one gamma
table, so every number here is the one the notebook functions (and the
rest of openNMR's referencing) give.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDoubleSpinBox, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from ..notebook import periodic_table as pt
from .widgets.collapsible_section import CollapsibleSection
from .widgets.typed_spinbox import TypedDoubleSpinBox

#: Accent (the website's teal) for search highlights and the selection.
ACCENT = "#0f6b72"

#: What a tile's bottom line can show: (key, combo label).
TILE_MODES = (
    ("mass", "Atomic weight"),
    ("larmor", "Larmor frequency (MHz)"),
    ("receptivity", "Receptivity (rel. ¹³C)"),
    ("isotope", "Principal isotope and spin"),
)

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def nuclide_label(iso) -> str:
    """ "²⁷Al", with a dagger for the radioactive ones."""
    return f"{str(iso.A).translate(_SUP)}{iso.symbol}" + (" †" if iso.radioactive else "")


def _fmt_abundance(value: float) -> str:
    if value <= 0:
        return "0 (synthetic)"
    if value < 0.01:
        return f"{value:.4g} %"
    return f"{value:.4g} %"


def _fmt_receptivity(value: float) -> str:
    if value <= 0:
        return "0"
    return f"{value:.3g}" if value >= 0.01 else f"{value:.2e}"


def _is_dark(widget: QWidget) -> bool:
    return widget.palette().color(QPalette.ColorRole.Window).lightness() < 128


# ============================================================
# Tiles
# ============================================================

class ElementTile(QWidget):
    """One element's tile, painted (not a stack of labels), so it scales
    with the window and can dim, highlight and show selection cheaply."""

    clicked = Signal(str)

    def __init__(self, el, parent=None) -> None:
        super().__init__(parent)
        self.element = el
        self.dimmed = False
        self.highlighted = False
        self.selected = False
        self.bottom_text = el.mass
        self._hover = False
        self.setMinimumSize(34, 38)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setAccessibleName(f"{el.name}, Z {el.Z}")
        lines = [f"<b>{el.name}</b> ({el.symbol}), Z = {el.Z}"]
        if el.isotopes:
            for iso in el.isotopes:
                lines.append(f"{nuclide_label(iso)}: I = {iso.spin_label}, "
                             f"{_fmt_abundance(iso.abundance)}")
        else:
            lines.append("No NMR-active isotope")
        lines.append("<i>Click for details</i>")
        self.setToolTip("<br>".join(lines))

    def sizeHint(self) -> QSize:  # noqa: N802 -- Qt override
        return QSize(58, 62)

    def spin_colour(self) -> QColor:
        iso = self.element.principal_isotope
        return QColor(pt.SPIN_COLOURS.get(iso.spin, "#999999")) if iso else QColor(0, 0, 0, 0)

    def event(self, ev) -> bool:
        if ev.type() == QEvent.Type.HoverEnter:
            self._hover = True
            self.update()
        elif ev.type() == QEvent.Type.HoverLeave:
            self._hover = False
            self.update()
        return super().event(ev)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802 -- Qt override
        if ev.button() == Qt.MouseButton.LeftButton and self.rect().contains(ev.position().toPoint()):
            self.clicked.emit(self.element.symbol)
        super().mouseReleaseEvent(ev)

    def paintEvent(self, _ev) -> None:  # noqa: N802 -- Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        dark = _is_dark(self)
        if self.dimmed:
            p.setOpacity(0.22)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if self.element.has_nmr:
            bg = pal.color(QPalette.ColorRole.Base)
        else:
            bg = QColor("#3a3d40") if dark else QColor(pt.NO_NMR_COLOUR)
        if self.selected:
            bg = QColor("#1d4f53") if dark else QColor("#dcecec")
        border = QColor(ACCENT) if (self.highlighted or self.selected or self._hover) \
            else pal.color(QPalette.ColorRole.Mid)
        pen = QPen(border, 2.0 if (self.highlighted or self.selected) else 1.0)
        p.setPen(pen)
        p.setBrush(bg)
        p.drawRoundedRect(rect, 3, 3)
        if self.element.has_nmr:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self.spin_colour())
            p.drawRoundedRect(QRectF(rect.left() + 1, rect.top() + 1, 4, rect.height() - 2), 2, 2)

        ink = pal.color(QPalette.ColorRole.Text)
        soft = QColor(ink)
        soft.setAlpha(165)
        h = rect.height()
        w = rect.width()
        small = QFont(self.font())
        small.setPointSizeF(max(5.5, min(9.0, h * 0.12)))
        p.setFont(small)
        p.setPen(soft)
        top = QRectF(rect.left() + 7, rect.top() + 2, w - 9, h * 0.24)
        p.drawText(top, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   str(self.element.Z))
        n = len(self.element.isotopes)
        if n:
            p.drawText(top.adjusted(0, 0, -2, 0),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), str(n))

        big = QFont(self.font())
        big.setBold(True)
        big.setPointSizeF(max(8.0, min(20.0, min(h * 0.28, w * 0.34))))
        p.setFont(big)
        p.setPen(ink)
        p.drawText(QRectF(rect.left() + 4, rect.top() + h * 0.2, w - 4, h * 0.45),
                   int(Qt.AlignmentFlag.AlignCenter), self.element.symbol)

        p.setFont(small)
        p.setPen(soft)
        bottom = QRectF(rect.left() + 5, rect.top() + h * 0.66, w - 6, h * 0.3)
        text = p.fontMetrics().elidedText(self.bottom_text, Qt.TextElideMode.ElideRight,
                                          int(bottom.width()))
        p.drawText(bottom, int(Qt.AlignmentFlag.AlignCenter), text)
        p.end()


class _Placeholder(QWidget):
    """The dashed "57–71" / "89–103" slot where the f-block belongs."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self._text = text
        self.setMinimumSize(34, 38)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.palette().color(QPalette.ColorRole.Mid), 1.0, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 3, 3)
        f = QFont(self.font())
        f.setPointSizeF(max(5.5, min(9.0, self.height() * 0.12)))
        p.setFont(f)
        p.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), self._text)
        p.end()


# ============================================================
# Element card (the click-through)
# ============================================================

class ElementCard(QWidget):
    """Covers the table with a dimmed backdrop and shows one element's
    card. Esc, the × or a click on the backdrop closes it."""

    nuclideChosen = Signal(str)
    closed = Signal()

    COLUMNS = ("Isotope", "I", "Abundance", "γ/2π (MHz/T)", "Q (fm²)",
               "Receptivity (rel. ¹³C)", "ν₀ (MHz)")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setVisible(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._element = None
        self._B0 = None

        self.card = QFrame(self)
        self.card.setObjectName("ptCard")
        self.card.setFrameShape(QFrame.Shape.StyledPanel)
        self.card.setAutoFillBackground(True)
        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(22, 16, 22, 16)
        lay.setSpacing(8)

        head = QHBoxLayout()
        self.symbol_label = QLabel()
        f = QFont(self.font())
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() * 3.0)
        self.symbol_label.setFont(f)
        head.addWidget(self.symbol_label)
        names = QVBoxLayout()
        self.name_label = QLabel()
        nf = QFont(self.font())
        nf.setPointSizeF(nf.pointSizeF() * 1.35)
        self.name_label.setFont(nf)
        self.sub_label = QLabel()
        self.sub_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        names.addWidget(self.name_label)
        names.addWidget(self.sub_label)
        head.addLayout(names, 1)
        self.close_button = QToolButton()
        self.close_button.setText("×")
        self.close_button.setAutoRaise(True)
        self.close_button.setToolTip("Close (Esc)")
        self.close_button.clicked.connect(self.close_card)
        head.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(head)

        lay.addWidget(QLabel("<small>Ground-state electron configuration</small>"))
        self.econfig_label = QLabel()
        self.econfig_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.econfig_label.setFont(mono)
        lay.addWidget(self.econfig_label)

        self.iso_heading = QLabel("<small>NMR-active isotopes</small>")
        lay.addWidget(self.iso_heading)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setToolTip("Click an isotope to put it in the Larmor calculator.")
        self.table.cellClicked.connect(self._on_row_clicked)
        lay.addWidget(self.table)
        self.none_label = QLabel()
        self.none_label.setWordWrap(True)
        lay.addWidget(self.none_label)
        self.note_label = QLabel()
        self.note_label.setWordWrap(True)
        lay.addWidget(self.note_label)

    # -- content ---------------------------------------------------------
    def show_element(self, el, B0_T: float | None) -> None:
        self._element = el
        self._B0 = B0_T
        iso0 = el.principal_isotope
        colour = pt.SPIN_COLOURS.get(iso0.spin) if iso0 else None
        self.symbol_label.setText(el.symbol)
        if colour:
            # The pale spin colours (1/2, 3/2, 7/2...) are darkened for
            # contrast against the card; the hue is what identifies them.
            c = QColor(colour)
            if c.lightness() > 150:
                c = c.darker(150)
            self.symbol_label.setStyleSheet(f"color: {c.name()};")
        else:
            self.symbol_label.setStyleSheet("")
        self.name_label.setText(el.name)
        self.sub_label.setText(f"Z = {el.Z}  ·  standard atomic weight {el.mass}")
        self.econfig_label.setText(el.econfig)
        self._fill_table()
        self.setVisible(True)
        self.raise_()
        self._layout_card()
        self.setFocus()

    def set_field(self, B0_T: float | None) -> None:
        self._B0 = B0_T
        if self.isVisible() and self._element is not None:
            self._fill_table()

    def _fill_table(self) -> None:
        el = self._element
        isos = el.isotopes
        self.table.setVisible(bool(isos))
        self.none_label.setVisible(not isos)
        if not isos:
            self.none_label.setText(
                f"None — {el.name} has no NMR-active isotope in openNMR's table.")
        field_txt = (f"{self._B0:.4f} T ({pt.field_to_proton_frequency(self._B0):.3f} MHz ¹H)"
                     if self._B0 else "—")
        # The field itself is in the note under the table: in the header
        # it made the last column too wide for the card.
        self.table.setRowCount(len(isos))
        for r, iso in enumerate(isos):
            cells = (
                nuclide_label(iso), iso.spin_label, _fmt_abundance(iso.abundance),
                f"{iso.gamma_MHz_per_T:+.4f}",
                ("n/a (I = 1/2)" if iso.spin == 0.5 else
                 (f"{iso.quadrupole_fm2:+.3f}" if iso.quadrupole_fm2 is not None else "—")),
                _fmt_receptivity(iso.receptivity_13C),
                f"{iso.larmor_MHz(self._B0):.4f}" if self._B0 else "—",
            )
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, iso.nuclide)
                if c == 0:
                    item.setBackground(QColor(pt.SPIN_COLOURS.get(iso.spin, "#999999")).lighter(115))
                    item.setForeground(QColor("#1b1e22"))
                self.table.setItem(r, c, item)
        rows_h = self.table.horizontalHeader().height() + sum(
            self.table.rowHeight(r) for r in range(len(isos))) + 6
        self.table.setFixedHeight(min(rows_h, 260))
        notes = [f"Larmor frequencies at {field_txt} — |γ/2π| × B₀, bare nucleus."]
        if any(i.radioactive for i in isos):
            notes.append("† radioactive; tabulated abundance is that of natural samples.")
        self.note_label.setText("<small>" + "<br>".join(notes) + "</small>")
        self._layout_card()

    def _on_row_clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.nuclideChosen.emit(item.data(Qt.ItemDataRole.UserRole))

    # -- overlay mechanics -----------------------------------------------
    def close_card(self) -> None:
        if self.isVisible():
            self.setVisible(False)
            self.closed.emit()

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self._layout_card()

    def _layout_card(self) -> None:
        if not self.parentWidget():
            return
        self.setGeometry(self.parentWidget().rect())
        width = min(640, max(320, self.width() - 40))
        self.card.setFixedWidth(width)
        self.card.adjustSize()
        height = min(self.card.sizeHint().height(), self.height() - 30)
        self.card.setFixedHeight(max(120, height))
        self.card.move((self.width() - width) // 2, max(10, (self.height() - self.card.height()) // 2))

    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 22, 24, 140))
        p.end()

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if not self.card.geometry().contains(ev.position().toPoint()):
            self.close_card()
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev) -> None:  # noqa: N802
        if ev.key() == Qt.Key.Key_Escape:
            self.close_card()
            return
        super().keyPressEvent(ev)


# ============================================================
# The table view
# ============================================================

class PeriodicTableView(QWidget):
    """The whole right-hand view: controls, legend, grid, footnote, card."""

    #: The user clicked an element (symbol).
    elementClicked = Signal(str)
    #: An isotope row in the card was clicked (nuclide, e.g. "27Al").
    nuclideChosen = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("periodicTableView")
        self._B0: float | None = None
        self._spin_filter: float | None = None
        self._no_nmr_filter = False
        self._mode = "mass"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 6)
        outer.setSpacing(6)

        title = QLabel("NMR Periodic Table")
        tf = QFont(self.font())
        tf.setBold(True)
        tf.setPointSizeF(tf.pointSizeF() * 1.4)
        title.setFont(tf)
        outer.addWidget(title)

        controls = QHBoxLayout()
        controls.setSpacing(12)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Symbol, name, Z or nuclide — e.g. Pt, platinum, 78, 195Pt")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(240)
        self.search_edit.setToolTip(
            "Highlights every element that matches and dims the rest. Enter opens the best "
            "match; a nuclide (\"27Al\") also puts that nuclide in the Larmor calculator.")
        self.search_edit.textChanged.connect(self._apply_search)
        self.search_edit.returnPressed.connect(self._open_best_match)
        controls.addWidget(QLabel("Find:"))
        controls.addWidget(self.search_edit, 1)
        controls.addWidget(QLabel("Tiles show:"))
        self.mode_combo = QComboBox()
        for key, label in TILE_MODES:
            self.mode_combo.addItem(label, key)
        self.mode_combo.setToolTip(
            "The bottom line of every tile. Larmor frequency and receptivity are for the "
            "element's most abundant NMR isotope; the frequency is at the calculator's field.")
        self.mode_combo.currentIndexChanged.connect(
            lambda _i: self.set_tile_mode(self.mode_combo.currentData()))
        controls.addWidget(self.mode_combo)
        outer.addLayout(controls)

        # Spin legend, doubling as a filter.
        legend = QHBoxLayout()
        legend.setSpacing(4)
        cap = QLabel("Spin I:")
        cap.setToolTip("The coloured bar on each tile is the spin of the element's most "
                       "abundant NMR isotope.")
        legend.addWidget(cap)
        self.legend_group = QButtonGroup(self)
        self.legend_group.setExclusive(False)
        self.legend_buttons: dict = {}
        for s in pt.spins_present():
            b = self._legend_button(f"{pt.spin_label(s)}", pt.SPIN_COLOURS.get(s, "#999"))
            b.setToolTip(f"Show only elements with an NMR isotope of spin {pt.spin_label(s)} "
                         f"(click again to show all).")
            b.clicked.connect(lambda checked, spin=s: self.set_spin_filter(spin if checked else None))
            self.legend_buttons[s] = b
            legend.addWidget(b)
        self.no_nmr_button = self._legend_button("none", pt.NO_NMR_COLOUR)
        self.no_nmr_button.setToolTip("Show only elements with no NMR-active isotope.")
        self.no_nmr_button.clicked.connect(self._on_no_nmr_clicked)
        legend.addWidget(self.no_nmr_button)
        legend.addStretch(1)
        outer.addLayout(legend)

        # The grid.
        self.grid_host = QWidget()
        grid = QGridLayout(self.grid_host)
        grid.setSpacing(3)
        grid.setContentsMargins(0, 0, 0, 0)
        self.tiles: dict = {}
        for el in pt.elements():
            row, col = pt.grid_position(el)
            tile = ElementTile(el)
            tile.clicked.connect(self.open_element)
            grid.addWidget(tile, row - 1, col - 1)
            self.tiles[el.symbol] = tile
        grid.addWidget(_Placeholder("57–71"), 5, 2)
        grid.addWidget(_Placeholder("89–103"), 6, 2)
        spacer = QWidget()
        spacer.setFixedHeight(10)
        grid.addWidget(spacer, 7, 0)
        for r in range(10):
            grid.setRowStretch(r, 0 if r == 7 else 1)
        for c in range(18):
            grid.setColumnStretch(c, 1)
        outer.addWidget(self.grid_host, 1)

        self.footnote = QLabel(
            "<small>Bar colour: spin of the most abundant NMR isotope; number top-right: how "
            "many NMR isotopes. † radioactive (³H and ⁹⁹Tc are not naturally present; tabulated "
            "at 0 %). Q is not defined for I = 1/2. Nuclear data are openNMR's own table, as used "
            "for referencing all spectra in openNMR. Larmor frequencies are |γ/2π| × B₀ for the "
            "'bare' nucleus.</small>")
        self.footnote.setWordWrap(True)
        outer.addWidget(self.footnote)

        self.card = ElementCard(self)
        self.card.nuclideChosen.connect(self.nuclideChosen)
        self._selected: str | None = None
        self._refresh_bottom_text()

    def _legend_button(self, text: str, colour: str) -> QToolButton:
        b = QToolButton()
        b.setCheckable(True)
        b.setText(text)
        fg = "#ffffff" if QColor(colour).lightness() < 110 else "#1b1e22"
        b.setStyleSheet(
            f"QToolButton {{ background: {colour}; color: {fg}; border: 1px solid #9aa3a5; "
            f"border-radius: 3px; padding: 1px 6px; }}"
            f"QToolButton:checked {{ border: 2px solid {ACCENT}; font-weight: bold; }}")
        self.legend_group.addButton(b)
        return b

    # -- state setters (public: the main window and tests drive these) ----
    def set_field(self, B0_T: float | None) -> None:
        self._B0 = float(B0_T) if B0_T else None
        if self._mode == "larmor":
            self._refresh_bottom_text()
        self.card.set_field(self._B0)

    def set_tile_mode(self, mode: str) -> None:
        if mode not in dict(TILE_MODES):
            raise ValueError(mode)
        self._mode = mode
        i = self.mode_combo.findData(mode)
        if i != self.mode_combo.currentIndex():
            self.mode_combo.setCurrentIndex(i)
        self._refresh_bottom_text()

    def set_spin_filter(self, spin: float | None) -> None:
        self._spin_filter = spin
        self._no_nmr_filter = False
        for s, b in self.legend_buttons.items():
            b.setChecked(spin is not None and abs(s - spin) < 1e-9)
        self.no_nmr_button.setChecked(False)
        self._apply_dimming()

    def _on_no_nmr_clicked(self, checked: bool) -> None:
        self.set_spin_filter(None)
        self._no_nmr_filter = checked
        self.no_nmr_button.setChecked(checked)
        self._apply_dimming()

    def select_element(self, symbol: str | None) -> None:
        """Mark the calculator's element on the table (not a search hit)."""
        if self._selected in self.tiles:
            self.tiles[self._selected].selected = False
            self.tiles[self._selected].update()
        self._selected = symbol if symbol in self.tiles else None
        if self._selected:
            self.tiles[self._selected].selected = True
            self.tiles[self._selected].update()

    def open_element(self, symbol: str) -> None:
        el = pt.element(symbol)
        self.card.show_element(el, self._B0)
        self.elementClicked.emit(el.symbol)

    # -- search / filters --------------------------------------------------
    def matches(self) -> list:
        return pt.search(self.search_edit.text())

    def _apply_search(self, _text: str = "") -> None:
        self._apply_dimming()

    def _apply_dimming(self) -> None:
        text = self.search_edit.text().strip()
        hits = {e.symbol for e in pt.search(text)} if text else None
        for sym, tile in self.tiles.items():
            el = tile.element
            dim = False
            if hits is not None:
                dim = sym not in hits
            if self._spin_filter is not None and not any(
                    abs(i.spin - self._spin_filter) < 1e-9 for i in el.isotopes):
                dim = True
            if self._no_nmr_filter and el.has_nmr:
                dim = True
            tile.highlighted = hits is not None and sym in hits
            tile.dimmed = dim
            tile.update()

    def _open_best_match(self) -> None:
        text = self.search_edit.text()
        found = pt.search(text)
        if not found:
            return
        self.open_element(found[0].symbol)
        nuc = pt.normalise_nuclide(text)
        if nuc is not None:
            try:
                self.nuclideChosen.emit(pt.isotope(nuc).nuclide)
            except KeyError:
                pass

    def _refresh_bottom_text(self) -> None:
        for tile in self.tiles.values():
            el = tile.element
            iso = el.principal_isotope
            if self._mode == "mass":
                text = el.mass
            elif iso is None:
                text = "—"
            elif self._mode == "larmor":
                text = f"{iso.larmor_MHz(self._B0):.2f}" if self._B0 else "—"
            elif self._mode == "receptivity":
                text = _fmt_receptivity(iso.receptivity_13C)
            else:
                text = f"{nuclide_label(iso).rstrip(' †')} {iso.spin_label}"
            tile.bottom_text = text
            tile.update()

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        if self.card.isVisible():
            self.card._layout_card()

    def keyPressEvent(self, ev) -> None:  # noqa: N802
        if ev.key() == Qt.Key.Key_Escape and self.card.isVisible():
            self.card.close_card()
            return
        super().keyPressEvent(ev)


# ============================================================
# The left panel: Larmor calculator
# ============================================================

class LarmorCalculatorPanel(QWidget):
    """Field + nucleus -> Larmor frequency, with the working shown."""

    fieldChanged = Signal(float)
    nucleusChanged = Signal(str)

    def __init__(self, spectrum_field=None, parent=None) -> None:
        """``spectrum_field()`` returns ``(B0_T, description)`` for the
        selected spectrum, or None; it drives the "From selected
        spectrum" button."""
        super().__init__(parent)
        self._spectrum_field = spectrum_field
        self._busy = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        intro = QLabel("Pick an element on the table, or type a nucleus, and set the magnet.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        magnet = QGroupBox("Magnet")
        form = QFormLayout(magnet)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom", None)
        for mhz, label in pt.COMMON_MAGNETS:
            self.preset_combo.addItem(label, mhz)
        self.preset_combo.setToolTip("Common spectrometers, named by their ¹H frequency.")
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        form.addRow("Spectrometer:", self.preset_combo)

        # Typed freely (explicit user request, 2026-10-13: 30 T could only be
        # reached with the up arrow -- Qt refused keystrokes on the way).
        self.field_spin = TypedDoubleSpinBox()
        self.field_spin.setRange(0.0, 50.0)
        # Six decimals (1 µT): at four, a "400 MHz" magnet rounds to
        # 9.3946 T and reads back as 399.998 MHz.
        self.field_spin.setDecimals(6)
        self.field_spin.setSingleStep(0.1)
        self.field_spin.setSuffix(" T")
        self.field_spin.setToolTip("B₀, the magnet's field (up to 50 T). Type a value and "
                                   "press Enter, or use the arrows.")
        form.addRow("Field B₀:", self.field_spin)

        self.proton_spin = TypedDoubleSpinBox()
        self.proton_spin.setRange(0.0, 2200.0)
        self.proton_spin.setDecimals(3)
        self.proton_spin.setSingleStep(1.0)
        self.proton_spin.setSuffix(" MHz")
        self.proton_spin.setToolTip("The same field, given as its ¹H Larmor frequency.")
        form.addRow("¹H frequency:", self.proton_spin)

        self.from_spectrum_button = QPushButton("From selected spectrum")
        self.from_spectrum_button.setToolTip(
            "Set the field from the selected spectrum: B₀ = BF1 / (γ/2π) of that spectrum's "
            "own nucleus.")
        self.from_spectrum_button.clicked.connect(self.use_spectrum_field)
        form.addRow("", self.from_spectrum_button)
        layout.addWidget(magnet)

        nucleus_box = QGroupBox("Nucleus")
        nform = QFormLayout(nucleus_box)
        self.nucleus_combo = QComboBox()
        self.nucleus_combo.setEditable(True)
        self.nucleus_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for iso in pt.isotopes():
            self.nucleus_combo.addItem(
                f"{iso.nuclide}  (I = {iso.spin_label}, {_fmt_abundance(iso.abundance)})",
                iso.nuclide)
        self.nucleus_combo.setToolTip("Type a nucleus (27Al, Al-27, ²⁷Al) or pick one.")
        self.nucleus_combo.currentIndexChanged.connect(lambda _i: self._on_nucleus_edited())
        self.nucleus_combo.lineEdit().editingFinished.connect(self._on_nucleus_edited)
        nform.addRow("Nucleus:", self.nucleus_combo)
        self.properties_label = QLabel()
        self.properties_label.setWordWrap(True)
        self.properties_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        nform.addRow(self.properties_label)
        layout.addWidget(nucleus_box)

        result_box = QGroupBox("Larmor frequency")
        rl = QVBoxLayout(result_box)
        self.result_label = QLabel()
        rf = QFont(self.font())
        rf.setBold(True)
        rf.setPointSizeF(rf.pointSizeF() * 1.6)
        self.result_label.setFont(rf)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rl.addWidget(self.result_label)
        self.working_label = QLabel()
        self.working_label.setWordWrap(True)
        self.working_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.working_label.setToolTip(
            "How the number was worked out. γ is openNMR's tabulated value for the bare "
            "nucleus; the 0 ppm reference of a real sample sits off this by the reference "
            "compound's shielding (ppm for ¹H and ¹³C; up to ~1 % for heavy nuclei).")
        rl.addWidget(self.working_label)
        copy = QPushButton("Copy")
        copy.setToolTip("Copy the result and its working to the clipboard.")
        copy.clicked.connect(self.copy_result)
        rl.addWidget(copy, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(result_box)

        # Every nucleus at this field, by frequency (collapsed: a reference).
        self.all_table = QTableWidget(0, 4)
        self.all_table.setHorizontalHeaderLabels(["Nucleus", "I", "Abundance", "ν₀ (MHz)"])
        self.all_table.verticalHeader().setVisible(False)
        self.all_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.all_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.all_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.all_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.all_table.setMinimumHeight(260)
        self.all_table.cellClicked.connect(self._on_all_table_clicked)
        self.all_table.setToolTip("Every tabulated nucleus at this field, highest frequency "
                                  "first -- the nucleus above is selected, so its neighbours "
                                  "(what else a probe tuned here might see) are beside it.")
        self.all_section = CollapsibleSection("All nuclei at this field", content=self.all_table,
                                              expanded=False)
        layout.addWidget(self.all_section)
        layout.addStretch(1)

        self.field_spin.valueChanged.connect(self._on_field_edited)
        self.proton_spin.valueChanged.connect(self._on_proton_edited)
        self._nuclide = "1H"
        self.proton_spin.setValue(400.0)
        self.set_nucleus("1H")
        self.refresh_spectrum_button()

    # -- public ---------------------------------------------------------
    def field(self) -> float:
        return self.field_spin.value()

    def nuclide(self) -> str:
        return self._nuclide

    def larmor_MHz(self) -> float:
        return pt.larmor_frequency(self._nuclide, self.field())

    def set_field(self, B0_T: float) -> None:
        self.field_spin.setValue(float(B0_T))    # -> _on_field_edited

    def set_nucleus(self, nuclide: str) -> bool:
        """Select ``nuclide`` (any spelling). Returns False, changing
        nothing, if it is not an NMR-active nuclide in the table."""
        try:
            iso = pt.isotope(nuclide)
        except KeyError:
            # An element ("Al", "aluminium") means its most abundant NMR isotope.
            try:
                iso = pt.element(nuclide).principal_isotope
            except KeyError:
                iso = None
            if iso is None or pt.normalise_nuclide(str(nuclide)) is not None:
                return False
        i = self.nucleus_combo.findData(iso.nuclide)
        self._busy = True
        try:
            self.nucleus_combo.setCurrentIndex(i)
        finally:
            self._busy = False
        self._nuclide = iso.nuclide
        self._update_result()
        self.nucleusChanged.emit(iso.nuclide)
        return True

    def set_element(self, symbol: str) -> None:
        """An element was picked: use its most abundant NMR isotope, unless
        the current nucleus already belongs to it."""
        el = pt.element(symbol)
        if not el.isotopes:
            return
        if pt.isotope(self._nuclide).symbol == el.symbol:
            self.nucleusChanged.emit(self._nuclide)
            return
        self.set_nucleus(el.principal_isotope.nuclide)

    def refresh_spectrum_button(self) -> None:
        info = self._spectrum_field() if self._spectrum_field else None
        self.from_spectrum_button.setEnabled(info is not None)
        if info is not None:
            self.from_spectrum_button.setToolTip(
                f"Set the field from the selected spectrum ({info[1]}): {info[0]:.4f} T.")

    def use_spectrum_field(self) -> bool:
        info = self._spectrum_field() if self._spectrum_field else None
        if info is None:
            return False
        self.set_field(info[0])
        return True

    def result_text(self) -> str:
        return f"{self.result_label.text()}\n{pt.larmor_working(self._nuclide, self.field())}"

    def copy_result(self) -> None:
        QApplication.clipboard().setText(self.result_text())

    # -- internals ------------------------------------------------------
    def _on_preset(self, _i: int) -> None:
        mhz = self.preset_combo.currentData()
        if mhz is not None and not self._busy:
            # Through the 1H box, so the magnet reads exactly "400.000 MHz".
            self.proton_spin.setValue(float(mhz))

    def _on_field_edited(self, value: float) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            self.proton_spin.setValue(pt.field_to_proton_frequency(value))
            self._sync_preset()
        finally:
            self._busy = False
        self._update_result()
        self._fill_all_table()
        self.fieldChanged.emit(float(value))

    def _on_proton_edited(self, value: float) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            self.field_spin.setValue(pt.proton_frequency_to_field(value))
            self._sync_preset()
        finally:
            self._busy = False
        self._update_result()
        self._fill_all_table()
        self.fieldChanged.emit(float(self.field_spin.value()))

    def _sync_preset(self) -> None:
        mhz = self.proton_spin.value()
        index = 0
        for i in range(1, self.preset_combo.count()):
            if abs(self.preset_combo.itemData(i) - mhz) < 0.5e-3:
                index = i
        self.preset_combo.setCurrentIndex(index)

    def _on_nucleus_edited(self) -> None:
        if self._busy:
            return
        # A picked item reads "27Al  (I = 5/2, 100 %)"; anything else was typed.
        text = self.nucleus_combo.currentText().split("(")[0].strip()
        if not self.set_nucleus(text):
            # Not a tabulated NMR nuclide: say so, keep the last good one.
            self.properties_label.setText(
                f"<span style='color:#b3261e'>'{text}' is not an NMR-active nuclide in "
                f"openNMR's table.</span>")

    def _update_result(self) -> None:
        iso = pt.isotope(self._nuclide)
        el = pt.element(iso.symbol)
        B0 = self.field()
        q = ("n/a" if iso.spin == 0.5 else
             f"{iso.quadrupole_fm2:+.3f} fm²" if iso.quadrupole_fm2 is not None else "—")
        self.properties_label.setText(
            f"{el.name}, I = {iso.spin_label}, abundance {_fmt_abundance(iso.abundance)}<br>"
            f"γ/2π = {iso.gamma_MHz_per_T:+.6f} MHz/T, Q = {q}<br>"
            f"Receptivity {_fmt_receptivity(iso.receptivity_13C)} × ¹³C "
            f"({_fmt_receptivity(iso.receptivity_1H)} × ¹H)"
            + ("<br>† radioactive" if iso.radioactive else ""))
        self.result_label.setText(f"ν₀({iso.nuclide}) = {iso.larmor_MHz(B0):.4f} MHz")
        ratio = 100.0 * abs(iso.gamma_MHz_per_T) / abs(pt.gamma_1H_MHz_per_T())
        self.working_label.setText(
            f"= |γ/2π| × B₀ = {abs(iso.gamma_MHz_per_T):.6f} MHz/T × {B0:.6f} T<br>"
            f"= {ratio:.6f} % of ¹H ({pt.field_to_proton_frequency(B0):.3f} MHz)"
            + ("<br>γ < 0: precesses in the opposite sense to ¹H." if iso.gamma_MHz_per_T < 0 else ""))
        self._select_in_all_table()

    def _fill_all_table(self) -> None:
        df = pt.larmor_table(self.field())
        self.all_table.setRowCount(len(df))
        for r, row in enumerate(df.itertuples(index=False)):
            for c, text in enumerate((row.nuclide, row.spin_label,
                                      _fmt_abundance(row.abundance_pct), f"{row.larmor_MHz:.4f}")):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, row.nuclide)
                self.all_table.setItem(r, c, item)
        self._select_in_all_table()

    def _select_in_all_table(self) -> None:
        for r in range(self.all_table.rowCount()):
            item = self.all_table.item(r, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == self._nuclide:
                self.all_table.blockSignals(True)
                self.all_table.selectRow(r)
                self.all_table.blockSignals(False)
                self.all_table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                return

    def _on_all_table_clicked(self, row: int, _col: int) -> None:
        item = self.all_table.item(row, 0)
        if item is not None:
            self.set_nucleus(item.data(Qt.ItemDataRole.UserRole))
