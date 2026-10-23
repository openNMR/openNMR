"""Text a user can read (labels, tooltips, messages) holds no ' -- ' stand-in
for a dash and no mention of TODO."""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "opennmr"
PATTERN = re.compile(r" -- |TODO")


def _docstring_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(getattr(node.body[0], "value", None), ast.Constant):
            ids.add(id(node.body[0].value))
    return ids


def test_no_double_hyphen_or_todo_in_strings():
    found = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = _docstring_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docs and PATTERN.search(node.value):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.value[:60]!r}")
    assert not found, "\n".join(found)
