"""Every text file openNMR writes is opened with an explicit encoding.

Without one, Python uses the system's locale encoding -- cp1252 on most
Windows machines -- and a report holding σ, ρ, χ² or a subscript cannot be
written: Fit Curve ▸ Export failed on Windows with "'charmap' codec can't
encode character 'σ'" (2026-10-16), while passing on Linux and macOS,
whose default is UTF-8."""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "opennmr"


def _text_writes_without_encoding():
    found = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open"):
                continue
            mode = None
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if not isinstance(mode, str) or "b" in mode or not set(mode) & set("wax"):
                continue
            if not any(kw.arg == "encoding" for kw in node.keywords):
                found.append(f"{path.relative_to(SRC)}:{node.lineno}")
    return found


def test_every_text_write_names_its_encoding():
    assert _text_writes_without_encoding() == []

