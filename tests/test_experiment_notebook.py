"""Export one experiment as a Jupyter notebook (opennmr.notebook.experiment_notebook).

Qt-free. The notebook is plain JSON (nbformat 4.5), written without nbformat.
What matters, and is tested here:

* the code cells are valid Python and, run, re-read the data with exactly the
  recorded reader arguments (checked by running them against a saved dataset
  folder, and against a recording stub for the exact arguments);
* steps the notebook does NOT re-run (baseline, referencing, ...) are said so,
  in their own section, not folded silently into the processing list;
* a spectrum with no file on disk is refused with a message that says what to do.
"""
import json
import os

import numpy as np
import pytest

from opennmr import io
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.io import history
from opennmr.notebook import (
    experiment_notebook, export_experiment_notebook, write_experiment_notebook,
)
from opennmr.notebook.experiment_notebook import literal_kwargs, split_state

import datetime as dt

NOW = dt.datetime(2026, 10, 31, 15, 0, tzinfo=dt.timezone.utc)


def _text(nb, kind):
    return "\n\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == kind)


def _make(path="/data/exp/1", state=None, kwargs=None, **options):
    return experiment_notebook(
        path, vendor="bruker", load_kwargs=kwargs, state=state, version="9.9.9", now=NOW, **options)


def _spectrum(folder=None, **raw):
    ppm = np.linspace(10.0, 0.0, 32)
    data = np.exp(-((ppm - 4.0) ** 2))
    return NMRSpectrum(
        data=data, data_imag=data * 0.1,
        axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="13C", sfo1=100.6,
                             source_path=folder, raw=dict(raw)))


# ---------------------------------------------------------------- structure

def test_the_notebook_is_valid_nbformat_json_with_unique_cell_ids():
    nb = _make(state={"live_lb": 5.0}, kwargs={"procno": 1})
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] == 5
    ids = [cell["id"] for cell in nb["cells"]]
    assert len(set(ids)) == len(ids)
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    json.dumps(nb)                                    # serialisable


def test_it_validates_with_nbformat_when_installed():
    nbformat = pytest.importorskip("nbformat")
    nb = _make(state={"live_lb": 5.0, "meta.nucleus": "23Na"}, kwargs={"procno": 1})
    nbformat.validate(nbformat.from_dict(nb))


def test_every_code_cell_compiles():
    nb = _make(state={"meta.nucleus": "23Na", "meta.mass_g": 0.02}, kwargs={"procno": 2})
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "<cell>", "exec")


def test_the_header_names_the_version_the_source_and_the_vendor():
    text = _text(_make(), "markdown")
    assert "openNMR 9.9.9" in text and "2026-10-31" in text
    assert "/data/exp/1" in text and "(bruker)" in text


def test_an_experiment_needs_a_path():
    with pytest.raises(ValueError):
        experiment_notebook("")


# ------------------------------------------------------------ reader arguments

def test_the_load_cell_passes_exactly_the_recorded_reader_arguments(monkeypatch):
    kwargs = {"procno": 3, "window": "exponential", "window_kwargs": {"lb": 5.0},
              "zero_fill": 16384, "phase0": 12.5, "truncate": None, "pseudo_2d": False}
    nb = _make(kwargs=kwargs)
    calls = []

    def recording_load(path, vendor=None, **kw):
        calls.append((path, vendor, kw))
        raise RuntimeError("stop")                    # only the call matters
    monkeypatch.setattr(io, "load", recording_load)
    code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"][0]
    with pytest.raises(RuntimeError):
        exec(code, {})
    assert calls == [("/data/exp/1", "bruker", kwargs)]


def test_numpy_values_become_plain_and_internal_or_unwritable_ones_are_reported():
    kept, dropped = literal_kwargs({
        "procno": np.int64(2), "pivot": np.float64(0.5), "points": np.array([1, 2, 3]),
        "_raw_fid_cache": np.zeros(4),               # internal: silently left out
        "callback": object(),                        # cannot be source code
    })
    assert kept == {"procno": 2, "pivot": 0.5, "points": [1, 2, 3]}
    assert type(kept["procno"]) is int
    assert dropped == ["callback"]
    nb = _make(kwargs={"procno": 1, "callback": object()})
    assert "could not be written as code" in _text(nb, "markdown")
    assert "`callback`" in _text(nb, "markdown")


# ---------------------------------------------- what is and is not re-run

STATE = {
    "live_lb": 5.0, "live_window": "exponential", "phase0": 12.0, "phase1": 0.5,
    "baseline_correct": True, "baseline_method": "polynomial", "baseline_degree": 3,
    "sr_shift_hz": 120.0, "meta.nucleus": "23Na", "meta.mass_g": 0.02,
    "analysis_state": {"regions": [[1.0, 2.0]], "integration_method": "simpson"},
}


def test_state_is_sorted_into_what_is_re_run_and_what_is_not():
    groups = split_state(STATE, reprocessable=True)
    assert set(groups["reader"]) == {"live_lb", "live_window", "phase0", "phase1"}
    assert set(groups["metadata"]) == {"meta.nucleus", "meta.mass_g"}
    assert set(groups["after"]) == {"baseline_correct", "baseline_method", "baseline_degree",
                                    "sr_shift_hz"}
    assert set(groups["results"]) == {"analysis_state"}


def test_without_a_raw_fid_the_phase_and_fid_settings_are_not_re_run_either():
    groups = split_state(STATE, reprocessable=False)
    assert groups["reader"] == {}
    assert {"live_lb", "phase0", "baseline_correct", "sr_shift_hz"} <= set(groups["after"])


def test_the_markdown_says_plainly_which_steps_are_not_re_run():
    text = _text(_make(state=STATE), "markdown")
    rerun, rest = text.split("## Done in the GUI and NOT re-run here")
    assert "Line broadening" in rerun and "Phase PH0" in rerun
    assert "Baseline" not in rerun and "Referencing" not in rerun
    assert "Baseline" in rest and "Referencing" in rest
    assert "## Results held by the GUI (not recomputed)" in text


def test_nothing_is_said_to_be_re_run_when_nothing_was_recorded():
    text = _text(_make(state={}), "markdown")
    assert "reader's defaults are used" in text
    assert "Nothing recorded" in text


def test_metadata_overrides_are_re_applied_with_dataclasses_replace():
    nb = _make(state={"meta.nucleus": "23Na", "meta.mass_g": 0.02})
    cells = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    override = [c for c in cells if "dataclasses.replace(" in c][0]
    meta = NMRMetadata(vendor="bruker", nucleus="1H")
    namespace = {"dataclasses": __import__("dataclasses"),
                 "spectrum": type("S", (), {"metadata": meta})()}
    exec(override, namespace)
    assert namespace["spectrum"].metadata.nucleus == "23Na"
    assert namespace["spectrum"].metadata.mass == 0.02


# ------------------------------------------------------------------- writing

def test_write_adds_the_extension_and_leaves_no_temporary_file(tmp_path):
    out = write_experiment_notebook(str(tmp_path / "exp"), "/data/exp/1", vendor="bruker")
    assert out.endswith("exp.ipynb") and os.path.exists(out)
    assert sorted(os.listdir(tmp_path)) == ["exp.ipynb"]
    with open(out, encoding="utf-8") as handle:
        assert json.load(handle)["nbformat"] == 4


def test_write_will_not_replace_a_file_unless_told_to(tmp_path):
    out = write_experiment_notebook(str(tmp_path / "a.ipynb"), "/p")
    with pytest.raises(FileExistsError):
        write_experiment_notebook(out, "/p")
    write_experiment_notebook(out, "/p", overwrite=True)


def test_unicode_survives_the_file(tmp_path):
    out = write_experiment_notebook(str(tmp_path / "u.ipynb"), "/p", title="Zeolite 27Al δ-scale")
    with open(out, encoding="utf-8") as handle:
        assert "δ-scale" in handle.read()


# ------------------------------------------------- from a loaded spectrum

def test_a_spectrum_with_no_file_is_refused_with_a_way_forward(tmp_path):
    with pytest.raises(ValueError, match="save_spectrum_folder"):
        export_experiment_notebook(_spectrum(folder=None), str(tmp_path / "x.ipynb"))
    with pytest.raises(ValueError):
        export_experiment_notebook(_spectrum(folder=str(tmp_path / "missing")),
                                   str(tmp_path / "x.ipynb"))


def test_running_the_exported_notebook_reproduces_the_spectrum(tmp_path):
    """End to end: save a dataset, export it, run every code cell, compare."""
    import matplotlib.pyplot as plt

    folder = history.save_spectrum_folder(_spectrum(), str(tmp_path / "dataset"), label="test")
    original = io.load(folder)
    original.metadata.mass = 0.05
    out = export_experiment_notebook(original, str(tmp_path / "run.ipynb"), load_kwargs={})
    with open(out, encoding="utf-8") as handle:
        nb = json.load(handle)
    namespace = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec(cell["source"], namespace)
    plt.close("all")
    replayed = namespace["spectrum"]
    np.testing.assert_array_equal(replayed.data, original.data)
    assert replayed.metadata.mass == 0.05
    assert replayed.metadata.nucleus == original.metadata.nucleus


def test_the_functions_are_exported_from_the_notebook_package():
    import opennmr.notebook as nbk
    for name in ("experiment_notebook", "write_experiment_notebook", "export_experiment_notebook"):
        assert name in nbk.__all__ and callable(getattr(nbk, name))
