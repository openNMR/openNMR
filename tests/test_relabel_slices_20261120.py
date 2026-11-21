"""Re-labelling the slices extracted from a 2D spectrum, after the fact."""
import numpy as np
import pytest

from test_gui import qapp  # noqa: F401
from opennmr import notebook as nb


class _Meta:
    def __init__(self):
        self.raw = {}


class _Spec:
    def __init__(self):
        self.metadata = _Meta()


def _spec_with_slices():
    spec = _Spec()
    spec.metadata.raw["contour_slices"] = [
        {"axis": "f1", "index": 2, "label": "Impurity", "color": "#111111"},
        {"axis": "f1", "index": 5, "label": "", "color": "#222222"},
    ]
    return spec


def test_slice_labels_show_defaults_for_unlabelled():
    assert nb.slice_labels(_spec_with_slices()) == ["Impurity", "Slice 2"]
    assert nb.slice_labels(_Spec()) == []


def test_relabel_by_position_and_by_name_keeps_the_rest():
    spec = _spec_with_slices()
    assert nb.relabel_slice(spec, 1, "Bulk") == "Bulk"
    assert nb.relabel_slice(spec, "Impurity", "Surface") == "Surface"
    stored = spec.metadata.raw["contour_slices"]
    assert [s["label"] for s in stored] == ["Surface", "Bulk"]
    assert [s["color"] for s in stored] == ["#111111", "#222222"]
    assert [s["index"] for s in stored] == [2, 5]


def test_relabel_blank_restores_the_default():
    spec = _spec_with_slices()
    assert nb.relabel_slice(spec, 0, "  ") == "Slice 1"


def test_relabel_refuses_unknown_or_ambiguous():
    spec = _spec_with_slices()
    with pytest.raises(IndexError):
        nb.relabel_slice(spec, 7, "x")
    with pytest.raises(KeyError):
        nb.relabel_slice(spec, "nope", "x")
    nb.relabel_slice(spec, 1, "Impurity")
    with pytest.raises(KeyError):
        nb.relabel_slice(spec, "Impurity", "x")


def test_canvas_rename_updates_spec_series_and_emits(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    canvas._contour_slices = [
        {"axis": "f1", "index": 2, "label": "Impurity", "color": "#111111",
         "color_override": True},
        {"axis": "f2", "index": 9, "label": "", "color": "#222222"},
    ]
    canvas._sync_slice_series()
    colours_before = [spec["color"] for spec in canvas._contour_slices]
    # A deliberately set colour is kept; the other comes from the palette.
    assert colours_before[0] == "#111111"
    heard = []
    canvas.contourSlicesChanged.connect(heard.append)

    assert canvas.contour_slice_label(1) == "Slice 2"
    assert canvas.rename_contour_slice(1, "  Bulk ") is True
    assert canvas.contour_slice_label(1) == "Bulk"
    assert heard and heard[-1][1]["label"] == "Bulk"
    # A rename made in the series list first must not shadow this one.
    canvas.set_series_style("0", label="Typed")
    assert canvas.rename_contour_slice(0, "Final")
    canvas._sync_slice_series()
    assert [e.label for e in canvas._series["slices"]] == ["Final", "Bulk"]
    # Blank restores the default; colours are untouched.
    canvas.rename_contour_slice(0, "")
    canvas._sync_slice_series()
    assert canvas._series["slices"][0].label == "Slice 1"
    assert [spec["color"] for spec in canvas._contour_slices] == colours_before
    assert canvas.rename_contour_slice(5, "x") is False


def test_prompt_rename_uses_the_dialog(qapp, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    canvas._contour_slices = [{"axis": "f1", "index": 2, "label": "A", "color": "#111111"}]
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("B", True)))
    canvas._prompt_rename_slice(0)
    assert canvas.contour_slice_label(0) == "B"
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("C", False)))
    canvas._prompt_rename_slice(0)
    assert canvas.contour_slice_label(0) == "B"
