"""
Command palette / "Where is…?": Ctrl+H (⌃H on a Mac), or Help ▸ Search commands….

Type a few letters of what you want -- "baseline", "export png", "T1",
"line broadening" -- and pick from every menu action, panel, section,
tab and labelled control in the window. Choosing one **takes you
there**: it switches to the right panel, opens the sections on the way,
scrolls the control into view and outlines it for a moment. A menu
action is simply run. Notebook functions are listed too; choosing one
copies its ``opennmr.notebook.<name>`` so the GUI and the notebook can be
searched together.

**The index is built from the window itself** each time the palette
opens (menus, ``CollapsibleSection`` titles, button and label text,
tooltips), never from a hand-kept list that would go stale. The test in
``tests/test_command_palette.py`` checks that every section and every
menu action is reachable, so a new feature that the palette cannot see
fails the suite.

**Matching** (:func:`rank`, no Qt): every word you type must match;
word starts beat substrings beat tooltip text beat letters-in-order;
British and American spellings are treated alike ("apodisation" finds
"Apodization"); a small alias table maps what people call things to what
the program calls them ("T1" -> relaxation).

**Ctrl+H on a Mac:** ⌘H belongs to the system ("Hide"), which Qt cannot
override, so the palette is on the Control key there.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import sys
import unicodedata
from typing import Callable

from PySide6.QtCore import QEvent, QObject, QRect, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QScrollArea, QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from .shortcut_sheet import clean_text, iter_menu_actions
from .widgets.collapsible_section import CollapsibleSection

#: Ctrl+H, except on a Mac where ⌘H is the system's "Hide".
PALETTE_KEY = "Meta+H" if sys.platform == "darwin" else "Ctrl+H"

MAX_RESULTS = 40
RECENT_KEY = "palette_recent"
MAX_RECENT = 8

# ---------------------------------------------------------------------------
# Matching (no Qt objects)
# ---------------------------------------------------------------------------

_SPELLING_SUFFIXES = (("isation", "ization"), ("ising", "izing"), ("ised", "ized"),
                      ("ise", "ize"), ("yse", "yze"), ("ysing", "yzing"))
_SPELLING_ANYWHERE = (("colour", "color"), ("centre", "center"), ("grey", "gray"),
                      ("behaviour", "behavior"))

#: What people call things -> what the program calls them. Keys and values
#: are normalised before use; a multi-word value needs all its words.
ALIASES: dict[str, tuple[str, ...]] = {
    "t1": ("relaxation", "inversion recovery", "saturation recovery"),
    "t2": ("relaxation", "echo"), "t1rho": ("relaxation", "spin lock"),
    "lb": ("line broadening", "exponential"), "ft": ("fourier",), "fft": ("fourier",),
    "fid": ("time domain",), "apodization": ("window", "line broadening"),
    "ph0": ("phase",), "ph1": ("phase",), "phasing": ("phase",), "autophase": ("phase",),
    "sr": ("reference",), "referencing": ("reference",), "calibrate": ("reference",),
    "calibration": ("reference",), "zoom": ("limits", "range"), "range": ("limits",),
    "save": ("export",), "png": ("figure", "export"), "pdf": ("figure", "export"),
    "svg": ("figure", "export"), "tiff": ("figure", "export"), "image": ("figure", "export"),
    "picture": ("figure",), "font": ("typeface",), "typeface": ("font",),
    "undo": ("history", "revert"), "revert": ("history", "reset"),
    "peak": ("fit", "pick"), "peaks": ("fit",), "integral": ("integrate", "area"),
    "area": ("integrate",), "csa": ("shielding", "anisotropy"), "cq": ("quadrupolar",),
    "dark": ("theme",), "light": ("theme",), "theme": ("dark",),
    "shortcut": ("keyboard",), "shortcuts": ("keyboard",), "hotkey": ("keyboard",),
    "notebook": ("jupyter", "script"), "jupyter": ("notebook",), "script": ("notebook",),
    "snr": ("noise", "signal"), "noise": ("snr",), "legend": ("labels",),
    "dosy": ("diffusion",), "diffusion": ("dosy",),
}


def _respell(word: str) -> str:
    for old, new in _SPELLING_ANYWHERE:
        word = word.replace(old, new)
    for old, new in _SPELLING_SUFFIXES:
        if word.endswith(old):
            return word[: -len(old)] + new
    return word


def normalise(text: str) -> str:
    """Lower case, accents and punctuation gone, subscripts as digits,
    British spellings as American -- applied to the index and the query
    alike, so they meet in the middle."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text.replace("&", ""))
    return " ".join(_respell(word) for word in text.split())


# Keys normalised like the query ("noise" is "noize" once respelt).
ALIASES = {normalise(key): values for key, values in ALIASES.items()}


def _subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


@dataclasses.dataclass
class Entry:
    """One thing the palette can take you to or run."""
    kind: str                       # Action | Panel | Section | Tab | Control | Notebook
    title: str
    path: str = ""                  # where it lives: "Process ▸ Baseline"
    keywords: str = ""              # tooltip / status text, searched last
    shortcut: str = ""
    action: Callable[[], object] | None = None
    enabled: bool = True
    # filled in by prepare()
    _title: str = ""
    _words: tuple = ()
    _path: str = ""
    _kw: str = ""

    def prepare(self) -> "Entry":
        self._title = normalise(self.title)
        self._words = tuple(self._title.split())
        self._path = normalise(self.path)
        self._kw = normalise(self.keywords)
        return self

    @property
    def key(self) -> str:
        return f"{self.kind}|{self.path}|{self.title}"


_KIND_WEIGHT = {"Action": 8, "Panel": 8, "Section": 5, "Tab": 5, "Control": 0, "Notebook": -6}


def _token_score(token: str, entry: Entry) -> float:
    # A word in a long sentence-like label is a weaker hit than the same
    # word in a short name ("apodisation" should find "Window" before a
    # 14-word explanation that happens to mention it).
    n = len(entry._words)
    brevity = 1.0 if n <= 3 else 1.0 / (1.0 + 0.06 * (n - 3))
    if token in entry._words:
        return 100.0 * brevity
    if any(word.startswith(token) for word in entry._words):
        return 85.0 * brevity
    if token in entry._title:
        return 65.0 * brevity
    if any(word.startswith(token) for word in entry._path.split()):
        return 40.0
    if token in entry._path:
        return 32.0
    if token in entry._kw:
        return 25.0
    if len(token) >= 3 and _subsequence(token, entry._title.replace(" ", "")):
        return 12.0
    return 0.0


def _phrase_score(phrase: str, entry: Entry) -> float:
    scores = [_token_score(token, entry) for token in phrase.split()]
    return min(scores) if scores and all(scores) else 0.0


def score(query: str, entry: Entry) -> float:
    """How well ``entry`` answers ``query`` (0 = does not)."""
    q = normalise(query)
    if not q:
        return 0.0
    total = 0.0
    for token in q.split():
        best = _token_score(token, entry)
        for alias in ALIASES.get(token, ()):
            best = max(best, 0.8 * _phrase_score(normalise(alias), entry))
        if best <= 0:
            return 0.0
        total += best
    if entry._title.startswith(q):
        total += 30.0
    elif q in entry._title:
        total += 12.0
    return total + _KIND_WEIGHT.get(entry.kind, 0) - 0.05 * len(entry._title)


def rank(query: str, entries, limit: int = MAX_RESULTS) -> list[Entry]:
    """The best ``limit`` entries for ``query``, best first."""
    scored = []
    for entry in entries:
        value = score(query, entry)
        if value > 0:
            scored.append((-value, entry.title.lower(), entry))
    scored.sort(key=lambda item: item[:2])
    return [entry for _, _, entry in scored[:limit]]


# ---------------------------------------------------------------------------
# Building the index from the window
# ---------------------------------------------------------------------------

def _tidy(text: str) -> str:
    """Single-line label text: mnemonics, HTML tags and line breaks out."""
    text = re.sub(r"<[^>]+>", "", text or "")
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def _is_section_header(widget) -> bool:
    return type(widget).__name__ == "_HeaderButton"


def _breadcrumb(widget: QWidget, tile: str) -> str:
    """``Tile ▸ Section ▸ Group`` for the containers around ``widget``."""
    names = []
    node = widget.parentWidget()
    child = widget
    while node is not None:
        if isinstance(node, CollapsibleSection):
            names.append(_tidy(node.title()))
        elif isinstance(node, QGroupBox) and node.title():
            names.append(_tidy(node.title()))
        elif isinstance(node, QStackedWidget) and isinstance(node.parentWidget(), QTabWidget):
            tabs = node.parentWidget()
            index = tabs.indexOf(child)
            if index >= 0 and tabs is not widget:
                names.append(_tidy(tabs.tabText(index)))
        child, node = node, node.parentWidget()
    names.reverse()
    return " ▸ ".join([tile] + [n for n in names if n and n != tile])


def build_index(window) -> list[Entry]:
    """Everything the palette can offer for ``window`` (a ``MainWindow``)."""
    entries: list[Entry] = []
    seen: set[str] = set()

    def add(entry: Entry) -> None:
        if not entry.title or entry.key in seen:
            return
        seen.add(entry.key)
        entries.append(entry.prepare())

    # Menu actions: run them.
    for path, action in iter_menu_actions(window.menuBar().actions()):
        if action.property("palette_exclude"):
            continue
        tip = action.toolTip() if action.toolTip() != action.text() else ""
        add(Entry("Action", clean_text(action.text()), " ▸ ".join(path),
                  keywords=" ".join(filter(None, (tip, action.statusTip()))),
                  shortcut=action.shortcut().toString(QKeySequence.SequenceFormat.NativeText),
                  action=action.trigger, enabled=action.isEnabled()))

    # Panels (the tiles) and what is inside them: go there.
    tabs = window.side_tabs
    for index in range(tabs.count()):
        tile = _tidy(tabs.tabText(index))
        if not tile or not tabs.isTabVisible(index):
            continue
        add(Entry("Panel", f"{tile} panel", "Panels",
                  action=lambda i=index: tabs.setCurrentIndex(i)))
        page = tabs.widget(index)
        for widget in page.findChildren(QWidget):
            _index_widget(window, widget, tile, add)
    _index_notebook(window, add)
    return entries


def _index_widget(window, widget: QWidget, tile: str, add) -> None:
    def go(target=widget):
        return reveal(window, target)

    if isinstance(widget, CollapsibleSection):
        add(Entry("Section", _tidy(widget.title()), _breadcrumb(widget, tile), action=go))
    elif isinstance(widget, QTabWidget):
        for i in range(widget.count()):
            page = widget.widget(i)
            add(Entry("Tab", _tidy(widget.tabText(i)), _breadcrumb(widget, tile),
                      action=lambda p=page: reveal(window, p)))
    elif isinstance(widget, QGroupBox):
        if widget.title():
            add(Entry("Section", _tidy(widget.title()), _breadcrumb(widget, tile),
                      keywords=widget.toolTip(), action=go))
    elif isinstance(widget, QAbstractButton):
        text = _tidy(widget.text())
        if text and not _is_section_header(widget):
            add(Entry("Control", text, _breadcrumb(widget, tile), keywords=_tidy(widget.toolTip()),
                      action=go))
    elif isinstance(widget, QLabel):
        text = _tidy(widget.text())
        if text and len(text) <= 60 and (text.endswith(":") or widget.buddy() is not None):
            buddy = widget.buddy()
            target = buddy if buddy is not None else widget
            add(Entry("Control", text.rstrip(":").strip(), _breadcrumb(widget, tile),
                      keywords=_tidy(target.toolTip()), action=lambda t=target: reveal(window, t)))


def _index_notebook(window, add) -> None:
    try:
        from .. import notebook
    except Exception:           # the palette must still open without it
        return
    for name in dir(notebook):
        if name.startswith("_"):
            continue
        obj = getattr(notebook, name, None)
        if not inspect.isfunction(obj) or not getattr(obj, "__module__", "").startswith("opennmr."):
            continue
        summary = (inspect.getdoc(obj) or "").strip().split("\n")[0]
        text = f"opennmr.notebook.{name}"

        def copy(text=text):
            QGuiApplication.clipboard().setText(text)
            _say(window, f"Copied {text}")
        add(Entry("Notebook", name, "Notebook function", keywords=summary, action=copy))


# ---------------------------------------------------------------------------
# Going there
# ---------------------------------------------------------------------------

def _say(window, message: str, ms: int = 5000) -> None:
    bar = getattr(window, "statusBar", None)
    if callable(bar):
        bar().showMessage(message, ms)


def reveal(window, target: QWidget, flash: bool = True, defer: bool = True) -> bool:
    """Bring ``target`` into view: its panel, the tabs and sections around
    it open, scrolled to, and briefly outlined. Returns whether it is
    actually visible afterwards (False for a control the current
    spectrum does not show, which also says so in the status bar)."""
    tabs = window.side_tabs
    for index in range(tabs.count()):
        page = tabs.widget(index)
        if page is target or page.isAncestorOf(target):
            tabs.setCurrentIndex(index)
            break
    chain = []
    node = target
    while node is not None:
        chain.append(node)
        node = node.parentWidget()
    for node in reversed(chain):                 # outermost first
        parent = node.parentWidget()
        if isinstance(parent, QStackedWidget):
            owner = parent.parentWidget()
            if isinstance(owner, QTabWidget) and owner.indexOf(node) >= 0:
                owner.setCurrentWidget(node)
            else:
                parent.setCurrentWidget(node)
        if isinstance(node, CollapsibleSection):
            node.setExpanded(True)
    QApplication.sendPostedEvents(None, int(QEvent.Type.LayoutRequest))

    def settle() -> bool:
        for node in chain:
            if isinstance(node, QScrollArea):
                node.ensureWidgetVisible(target, 20, 60)
        visible = target.isVisibleTo(window)
        if not visible:
            _say(window, "That control is hidden for the current selection "
                         "(it appears when the spectrum or option it belongs to is chosen).")
        elif flash:
            _flash(window, target)
        return visible

    if defer:
        QTimer.singleShot(60, settle)
        return target.isVisibleTo(window)
    return settle()


def _flash(window, target: QWidget, ms: int = 1500) -> None:
    top = window.window()
    box = QFrame(top)
    box.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    box.setStyleSheet("QFrame { border: 3px solid #f5a623; border-radius: 5px; background: transparent; }")
    corner = target.mapTo(top, target.rect().topLeft())
    box.setGeometry(QRect(corner, target.size()).adjusted(-4, -4, 4, 4))
    box.show()
    box.raise_()
    QTimer.singleShot(ms, box.deleteLater)


# ---------------------------------------------------------------------------
# The palette window
# ---------------------------------------------------------------------------

class _Row(QWidget):
    def __init__(self, entry: Entry) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        text = QVBoxLayout()
        text.setSpacing(0)
        title = QLabel(f"<b>{_escape(entry.title)}</b>" + ("" if entry.enabled else " (unavailable now)"))
        where = QLabel(f"{entry.kind} · {entry.path}" if entry.path else entry.kind)
        where.setStyleSheet("color: palette(mid);")
        title.setEnabled(entry.enabled)
        text.addWidget(title)
        text.addWidget(where)
        layout.addLayout(text, 1)
        if entry.shortcut:
            keys = QLabel(entry.shortcut)
            keys.setStyleSheet("color: palette(mid);")
            layout.addWidget(keys)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class CommandPalette(QFrame):
    """The pop-up. ``CommandPalette(window).open_palette()``."""

    def __init__(self, window, entries=None, anchored: bool = False) -> None:
        """``anchored``: the results list for the search box in the tool
        row (widgets/search_box.py) -- no field of its own (the box is
        the field) and a window that never takes the keyboard focus, so
        typing stays in the box while the results follow it."""
        flags = (Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                 | Qt.WindowType.WindowDoesNotAcceptFocus) if anchored else Qt.WindowType.Popup
        super().__init__(window, flags)
        self._anchored = anchored
        if anchored:
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._window = window
        self._entries = list(entries) if entries is not None else build_index(window)
        self._shown: list[Entry] = []
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Where is…?  Search actions, panels and controls")
        self.edit.setClearButtonEnabled(True)
        self.edit.textChanged.connect(self.refresh)
        self.edit.installEventFilter(self)
        layout.addWidget(self.edit)
        self.results = QListWidget()
        self.results.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.results.itemActivated.connect(lambda _item: self.activate())
        self.results.itemClicked.connect(lambda _item: self.activate())
        layout.addWidget(self.results, 1)
        self.hint = QLabel()
        self.hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.hint)
        QShortcut(QKeySequence(PALETTE_KEY), self, activated=self.close)
        if anchored:
            self.edit.setVisible(False)
            self.results.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.refresh("")

    # -- content ---------------------------------------------------------

    def entries(self) -> list[Entry]:
        return list(self._entries)

    def shown(self) -> list[Entry]:
        return list(self._shown)

    def _recent_keys(self) -> list[str]:
        settings = getattr(self._window, "_ui_settings", None) or {}
        keys = settings.get(RECENT_KEY) or []
        return [k for k in keys if isinstance(k, str)]

    def _remember(self, entry: Entry) -> None:
        settings = getattr(self._window, "_ui_settings", None)
        if not isinstance(settings, dict):
            return
        keys = [entry.key] + [k for k in self._recent_keys() if k != entry.key]
        settings[RECENT_KEY] = keys[:MAX_RECENT]
        save = getattr(self._window, "_save_ui_settings", None)
        if callable(save):
            save()

    def refresh(self, query: str = "") -> None:
        if query.strip():
            self._shown = rank(query, self._entries)
            self.hint.setText("↑ ↓ to choose · Enter to go · Esc to close" if self._shown
                              else "Nothing matches. Try fewer or different words.")
        else:
            by_key = {e.key: e for e in self._entries}
            self._shown = [by_key[k] for k in self._recent_keys() if k in by_key]
            self.hint.setText("Recent choices. Type to search, e.g. baseline, export, T1, line broadening."
                              if self._shown else
                              "Type to search, e.g. baseline, export, T1, line broadening.")
        self.results.clear()
        for entry in self._shown:
            item = QListWidgetItem()
            row = _Row(entry)
            item.setSizeHint(row.sizeHint())
            self.results.addItem(item)
            self.results.setItemWidget(item, row)
        if self._shown:
            self.results.setCurrentRow(0)

    # -- acting ----------------------------------------------------------

    def activate(self, row: int | None = None) -> Entry | None:
        """Run the highlighted result (or result ``row``); closes the
        palette first so the focus goes back to the window."""
        row = self.results.currentRow() if row is None else row
        if not 0 <= row < len(self._shown):
            return None
        entry = self._shown[row]
        self.close()
        if not entry.enabled:
            _say(self._window, f"'{entry.title}' is not available right now.")
            return entry
        self._remember(entry)
        if entry.action is not None:
            try:
                entry.action()
            except Exception as exc:                 # never take the window down
                _say(self._window, f"Could not open '{entry.title}': {exc}", 10000)
        return entry

    def step(self, step: int) -> None:
        """Move the highlight ``step`` rows (wrapping)."""
        count = self.results.count()
        if count:
            self.results.setCurrentRow((self.results.currentRow() + step) % count)

    def eventFilter(self, obj: QObject, event) -> bool:  # noqa: N802 -- Qt override
        if obj is self.edit and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                self.step(1 if key == Qt.Key.Key_Down else -1)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.activate()
                return True
        return super().eventFilter(obj, event)

    def open_under(self, anchor) -> "CommandPalette":
        """Show the results just below ``anchor`` (the search box),
        right-aligned with it, without taking the focus from it."""
        host = self._window
        width = max(360, min(560, host.width() - 40))
        self.resize(width, min(420, max(220, host.height() // 2)))
        corner = anchor.mapToGlobal(anchor.rect().bottomRight())
        left = max(host.mapToGlobal(host.rect().topLeft()).x() + 8, corner.x() - width)
        self.move(left, corner.y() + 2)
        self.show()
        self.raise_()
        return self

    def open_palette(self) -> "CommandPalette":
        host = self._window
        width = max(360, min(640, host.width() - 40))
        self.resize(width, min(460, max(240, host.height() - 120)))
        centre = host.mapToGlobal(host.rect().center())
        self.move(centre.x() - width // 2, host.mapToGlobal(host.rect().topLeft()).y() + host.height() // 8)
        self.show()
        self.edit.setFocus()
        return self
