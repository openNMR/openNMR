"""
Capture and restore a panel's inputs -- the plain Qt controls it holds
(combo boxes, spin boxes, check boxes, line edits, tables with their
per-item data) -- as JSON-able data, with a readable rendering for the
processing/simulation logs.

A panel opts in by giving the widgets it wants kept as attributes and,
where its state is more than its widgets (a model held in Python lists,
styling derived from item data), by overriding :func:`capture_state` /
:func:`restore_state` with its own ``get_state``/``set_state``.

What is captured per widget: its value (to restore it), what it shows
(``display``: the spin box's text with its unit, the combo's text) and
the label next to it in its form layout, so the log can say "Larmor
frequency: 100.6 MHz" rather than "larmor_spin: 100.6".
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton, QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPlainTextEdit, QRadioButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QTextEdit, QWidget,
)

from ..io.serialize import decode, encode

#: Item-data roles kept for table cells (besides text and check state):
#: the panels keep constraints, couplings and bounds there.
_TABLE_ROLES = tuple(int(Qt.ItemDataRole.UserRole) + k for k in range(8))

#: Attribute-name fragments that mark a widget as output, not input.
_OUTPUT_FRAGMENTS = ("result", "report", "summary", "status", "solution")


def _is_input_name(name: str) -> bool:
    return not name.startswith("_") and not any(f in name for f in _OUTPUT_FRAGMENTS)


def _form_label(widget: QWidget) -> str:
    """The text of the label a form layout shows beside ``widget``
    (looked for in the nearest two ancestors)."""
    parent = widget.parentWidget()
    for _depth in range(2):
        if parent is None:
            break
        forms = parent.findChildren(QFormLayout)
        if isinstance(parent.layout(), QFormLayout):
            forms.insert(0, parent.layout())
        for form in forms:
            label = form.labelForField(widget)
            if isinstance(label, QLabel) and label.text().strip():
                return label.text().strip().rstrip(":")
        parent = parent.parentWidget()
    if isinstance(widget, QAbstractButton):
        return widget.text().replace("&", "").strip()
    if isinstance(widget, QGroupBox):
        return widget.title()
    return ""


def _is_supported(widget) -> bool:
    return isinstance(widget, (QComboBox, QAbstractSpinBox, QCheckBox, QRadioButton, QLineEdit,
                               QPlainTextEdit, QTextEdit, QTableWidget)) or (
        isinstance(widget, QGroupBox) and widget.isCheckable()) or (
        isinstance(widget, QAbstractButton) and widget.isCheckable())


def input_widgets(owner, names=None, prefixes=None, exclude=()) -> dict[str, QWidget]:
    """``owner``'s input widgets by attribute name: the given ``names``,
    or every supported widget attribute (starting with one of
    ``prefixes``, if given) whose name does not mark it as output."""
    found = {}
    items = ((n, getattr(owner, n, None)) for n in names) if names is not None else vars(owner).items()
    for name, widget in items:
        if widget is None or name in exclude:
            continue
        if names is None:
            if prefixes is not None and not name.startswith(tuple(prefixes)):
                continue
            if not _is_input_name(name):
                continue
        if isinstance(widget, QWidget) and _is_supported(widget):
            if isinstance(widget, (QLineEdit, QPlainTextEdit, QTextEdit)) and widget.isReadOnly():
                continue
            found[name] = widget
    return found


def _table_state(table: QTableWidget) -> dict:
    rows = []
    for r in range(table.rowCount()):
        row = []
        for c in range(table.columnCount()):
            item = table.item(r, c)
            if item is None:
                row.append(None)
                continue
            cell = {"text": item.text()}
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                cell["check"] = int(item.checkState().value if hasattr(item.checkState(), "value")
                                    else item.checkState())
            roles = {}
            for role in _TABLE_ROLES:
                data = item.data(role)
                if data is not None:
                    roles[str(role)] = encode(data)
            if roles:
                cell["roles"] = roles
            row.append(cell)
        rows.append(row)
    headers = [table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else str(c + 1)
               for c in range(table.columnCount())]
    return {"rows": rows, "columns": table.columnCount(), "headers": headers}


def _set_table_state(table: QTableWidget, state: dict) -> None:
    table.blockSignals(True)
    try:
        table.setRowCount(0)
        if table.columnCount() < state.get("columns", 0):
            table.setColumnCount(state["columns"])
        table.setRowCount(len(state.get("rows", [])))
        for r, row in enumerate(state.get("rows", [])):
            for c, cell in enumerate(row):
                if cell is None or c >= table.columnCount():
                    continue
                item = QTableWidgetItem(cell.get("text", ""))
                if "check" in cell:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState(cell["check"]))
                for role, data in (cell.get("roles") or {}).items():
                    item.setData(int(role), decode(data))
                table.setItem(r, c, item)
    finally:
        table.blockSignals(False)


_NAME_PREFIXES = ("sim_csa_", "sim_quad_", "sim_", "csa_", "bloch_")
_NAME_SUFFIXES = ("_spin", "_combo", "_check", "_checkbox", "_edit", "_table", "_button")


def pretty_name(name: str) -> str:
    """An attribute name as a label, for a widget with none of its own:
    ``window_lo_spin`` -> "Window lo"."""
    for prefix in _NAME_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in _NAME_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_", " ").strip().capitalize() or name


def _widget_state(widget) -> dict:
    label = _form_label(widget)
    if isinstance(widget, QTableWidget):
        return {"kind": "table", "value": _table_state(widget), "label": label}
    if isinstance(widget, QComboBox):
        data = widget.currentData()
        return {"kind": "combo", "value": encode(data) if data is not None else None,
                "text": widget.currentText(), "display": widget.currentText(), "label": label}
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return {"kind": "spin", "value": widget.value(), "display": widget.text().strip(), "label": label}
    if isinstance(widget, QAbstractSpinBox):
        return {"kind": "text", "value": widget.text(), "display": widget.text(), "label": label}
    if isinstance(widget, (QAbstractButton, QGroupBox)):
        return {"kind": "check", "value": widget.isChecked(),
                "display": "on" if widget.isChecked() else "off",
                "label": label or (widget.title() if isinstance(widget, QGroupBox) else "")}
    if isinstance(widget, QLineEdit):
        return {"kind": "line", "value": widget.text(), "display": widget.text(), "label": label}
    return {"kind": "plain", "value": widget.toPlainText(), "display": widget.toPlainText(), "label": label}


def _apply(widget, entry: dict) -> None:
    kind, value = entry.get("kind"), entry.get("value")
    if kind == "table":
        _set_table_state(widget, value or {})
    elif kind == "combo":
        index = widget.findData(decode(value)) if value is not None else -1
        if index < 0:
            index = widget.findText(entry.get("text", ""))
        if index >= 0:
            widget.setCurrentIndex(index)
        elif widget.isEditable():
            widget.setCurrentText(entry.get("text", ""))
    elif kind == "spin":
        widget.setValue(value)
    elif kind == "check":
        widget.setChecked(bool(value))
    elif kind == "line":
        widget.setText(value or "")
    elif kind == "plain":
        widget.setPlainText(value or "")


_ORDER = {"combo": 0, "check": 1, "spin": 2, "text": 2, "line": 3, "plain": 3, "table": 4}


def capture_widgets(owner, names=None, prefixes=None, exclude=()) -> dict:
    """The state of ``owner``'s input widgets (see :func:`input_widgets`)."""
    return {name: _widget_state(w) for name, w in input_widgets(owner, names, prefixes, exclude).items()}


def restore_widgets(owner, state: dict) -> list[str]:
    """Put captured widget values back. Combo boxes and check boxes go
    first (they show, hide and re-range the others), then values, then
    tables; values are applied twice so one a combo box changed on the
    way is set again. Returns the names that no longer exist."""
    missing = []
    entries = []
    for name, entry in (state or {}).items():
        widget = getattr(owner, name, None)
        if widget is None or not isinstance(entry, dict):
            missing.append(name)
            continue
        entries.append((name, widget, entry))
    entries.sort(key=lambda e: _ORDER.get(e[2].get("kind"), 9))
    for _pass in range(2):
        for _name, widget, entry in entries:
            if _pass == 1 and entry.get("kind") == "table":
                continue
            _apply(widget, entry)
    return missing


def capture_state(owner) -> dict:
    """``owner.get_state()`` if it has one, else its input widgets,
    recursing into child panels that have their own state."""
    getter = getattr(owner, "get_state", None)
    if callable(getter):
        return getter()
    return default_state(owner)


def default_state(owner) -> dict:
    state = {"widgets": capture_widgets(owner)}
    children = {}
    for name, child in vars(owner).items():
        if (_is_input_name(name) and isinstance(child, QWidget) and child is not owner
                and callable(getattr(child, "get_state", None))):
            children[name] = child.get_state()
    if children:
        state["children"] = children
    return state


def restore_state(owner, state: dict) -> None:
    setter = getattr(owner, "set_state", None)
    if callable(setter):
        setter(state)
        return
    restore_default_state(owner, state)


def restore_default_state(owner, state: dict) -> None:
    restore_widgets(owner, (state or {}).get("widgets") or {})
    for name, child_state in ((state or {}).get("children") or {}).items():
        child = getattr(owner, name, None)
        if child is not None and callable(getattr(child, "set_state", None)):
            child.set_state(child_state)


# ------------------------------------------------------------------
# For people
# ------------------------------------------------------------------

def describe_state(state: dict, indent: str = "") -> list[str]:
    """One line per input -- "Larmor frequency: 100.6000 MHz" -- and a
    table as its header row plus one line per row."""
    lines = []
    widgets = (state or {}).get("widgets") or {}
    for name, entry in widgets.items():
        if not isinstance(entry, dict):
            continue
        label = entry.get("label") or pretty_name(name)
        if entry.get("kind") == "table":
            table = entry.get("value") or {}
            rows = table.get("rows") or []
            if not rows:
                continue
            lines.append(f"{indent}{label}:")
            headers = [h.replace("\n", " ") for h in table.get("headers") or []]
            lines.append(f"{indent}  " + " | ".join(headers))
            for row in rows:
                lines.append(f"{indent}  " + " | ".join((c or {}).get("text", "") for c in row))
            continue
        display = str(entry.get("display", entry.get("value", "")))
        if entry.get("kind") in ("line", "plain") and not display.strip():
            continue
        lines.append(f"{indent}{label}: {display}")
    for name, child in ((state or {}).get("children") or {}).items():
        lines += describe_state(child, indent)
    for key, value in ((state or {}).get("extra") or {}).items():
        lines.append(f"{indent}{key}: {value}")
    return lines


def table_text(table: QTableWidget, sep: str = " | ") -> str:
    """A results table as text, one row per line, row headers included
    (``sep="\t"`` for a spreadsheet)."""
    headers = [(table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else "")
               .replace("\n", " ") for c in range(table.columnCount())]
    lines = [sep.join([""] + headers).strip()] if any(headers) else []
    for r in range(table.rowCount()):
        head = table.verticalHeaderItem(r)
        cells = [(table.item(r, c).text().replace("\n", " ") if table.item(r, c) else "")
                 for c in range(table.columnCount())]
        lines.append(sep.join([head.text() if head else str(r + 1)] + cells))
    return "\n".join(lines)
