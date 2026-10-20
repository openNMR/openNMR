"""tools/rehearse_update.py (2026-10-20): the pieces that need no network."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "rehearse_update.py"


def _load():
    spec = importlib.util.spec_from_file_location("rehearse_update", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_patch_and_current_version():
    m = _load()
    assert m.next_patch("0.1.0") == "0.1.1"
    assert m.next_patch("1.9.9") == "1.9.10"
    assert m.next_patch("0.2.0.dev1") == "0.2.1"
    assert m.current_version()[0].isdigit()
