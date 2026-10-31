"""
Export one experiment as a Jupyter notebook (``.ipynb``).

File ▸ Export experiment to notebook… in the GUI writes the notebook for
the selected experiment; from a notebook, :func:`export_experiment_notebook`
does the same for a loaded spectrum. Qt-free, and it needs neither
``nbformat`` nor Jupyter to *write* the file.

**What the notebook does, and what it does not.** It re-reads the
experiment with exactly the arguments the GUI's reader was last called
with (``io.load(path, vendor=..., **load_kwargs)``) -- FID window,
zero-fill, truncation, linear prediction, phase and the pseudo-2D choice
all go through those arguments -- re-applies the nucleus and mass
overrides, and plots the result. Anything the GUI does *after* the
reader (baseline correction, referencing (SR), field-drift correction,
per-slice phase) is **not** re-run: re-implementing it here without
being able to prove it gives the GUI's numbers would risk a notebook
that silently differs from the figure it came from. Those steps, and the
fit/analysis results the GUI holds, are listed in the notebook in plain
words with their recorded values, under their own headings, so nobody
mistakes the notebook's spectrum for the GUI's.

When the spectrum cannot be re-read from raw data by the GUI (a Bruker
experiment with only processed data, say), the phase and FID settings
are not applied by the reader either, so they move to the "not re-run"
list as well (``reprocessable=False``).
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import os
import pprint

from ..io import history

NBFORMAT = 4
NBFORMAT_MINOR = 5

#: Metadata overrides re-applied with ``dataclasses.replace`` -- the way
#: the GUI's session applies them: state key -> NMRMetadata field.
_METADATA_FIELDS = {
    "meta.nucleus": "nucleus",
    "meta.nucleus_indirect": "nucleus_indirect",
    "meta.mass_g": "mass",
}

#: Done by the GUI after the reader; not re-run here.
_AFTER_READER_PREFIXES = ("baseline_", "sr_", "drift_correction", "slice_phase")

#: Settings that travel to the reader as arguments (only when the GUI
#: could really re-read the raw data).
_READER_PREFIXES = ("live_", "phase", "magnitude", "meta.pseudo_2d")

#: Results the GUI holds; shown, never recomputed.
_RESULT_PREFIXES = ("analysis_state", "fits", "simulation", "relaxation_fit_results",
                    "slice_fit_results", "score_result", "dosy_result", "enmr_result")


def _starts(key: str, prefixes) -> bool:
    return any(key == p or key.startswith(p) for p in prefixes)


def _plain(value):
    """``value`` as plain Python (numpy scalars/arrays -> numbers/lists)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def literal_kwargs(kwargs: dict | None) -> tuple[dict, list[str]]:
    """``(kept, dropped)``: the reader arguments that can be written as
    Python source and read back unchanged, and the names of those that
    cannot (internal ``_``-prefixed ones are silently omitted)."""
    kept, dropped = {}, []
    for key, value in (kwargs or {}).items():
        if str(key).startswith("_"):
            continue
        value = _plain(value)
        try:
            ok = ast.literal_eval(repr(value)) == value
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            ok = False
        if ok:
            kept[str(key)] = value
        else:
            dropped.append(str(key))
    return kept, dropped


def split_state(state: dict | None, reprocessable: bool = True) -> dict[str, dict]:
    """Sort a tracked state into what the notebook re-runs
    (``"reader"``, ``"metadata"``), what the GUI did that it does not
    (``"after"``), recorded results (``"results"``) and everything else
    recorded (``"other"``)."""
    groups = {"reader": {}, "metadata": {}, "after": {}, "results": {}, "other": {}}
    for key, value in (state or {}).items():
        if key in _METADATA_FIELDS:
            groups["metadata"][key] = value
        elif _starts(key, _AFTER_READER_PREFIXES):
            groups["after"][key] = value
        elif _starts(key, _RESULT_PREFIXES):
            groups["results"][key] = value
        elif _starts(key, _READER_PREFIXES):
            groups["reader" if reprocessable else "after"][key] = value
        else:
            groups["other"][key] = value
    return groups


def _lines(group: dict, source: dict | None) -> list[str]:
    return history.state_summary(group, source) if group else []


def _bullets(lines: list[str], empty: str) -> str:
    return "\n".join(f"- {line}" for line in lines) if lines else f"_{empty}_"


def _cell_id(n: int) -> str:
    return f"opennmr-cell-{n}"


def _markdown(n: int, text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(n), "metadata": {}, "source": text}


def _code(n: int, text: str) -> dict:
    return {"cell_type": "code", "id": _cell_id(n), "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


def experiment_notebook(path: str, *, vendor: str | None = None, load_kwargs: dict | None = None,
                        state: dict | None = None, source: dict | None = None,
                        title: str | None = None, reprocessable: bool = True,
                        version: str | None = None, now: _dt.datetime | None = None) -> dict:
    """The notebook (an ``nbformat`` 4.5 dictionary) for the experiment at
    ``path``.

    ``load_kwargs`` -- what the reader was last called with (the GUI's
    ``Session.get_load_kwargs``). ``state`` -- :func:`opennmr.io.history.tracked_state`
    of the spectrum. ``source`` -- :func:`opennmr.io.history.describe_source`
    (lets a nucleus that is simply the file's own be left out of the log).
    """
    if not path:
        raise ValueError("An experiment needs a path on disk to be exported to a notebook.")
    if version is None:
        from .. import _version
        version = _version.__version__
    now = now or _dt.datetime.now().astimezone()
    title = title or os.path.basename(os.path.normpath(str(path))) or str(path)
    kept, dropped = literal_kwargs(load_kwargs)
    groups = split_state(state, reprocessable)

    reader_lines = _lines({**groups["reader"], **groups["metadata"]}, source)
    after_lines = _lines(groups["after"], source)
    other_lines = _lines(groups["other"], source)
    result_lines = _lines(groups["results"], source)

    header = (
        f"# {title}\n\n"
        f"Written by openNMR {version} on {now:%Y-%m-%d %H:%M}.  \n"
        f"Source: `{path}`" + (f" ({vendor})" if vendor else "") + "\n\n"
        "This notebook re-reads the experiment and re-applies the settings listed "
        "under **Re-run by this notebook**. Read the other two lists before comparing "
        "it with a figure from the GUI: they are things the GUI did that this notebook "
        "does **not** repeat."
    )
    log = (
        "## Re-run by this notebook\n\n"
        + _bullets(reader_lines, "No processing settings recorded: the reader's defaults are used.")
        + "\n\n## Done in the GUI and NOT re-run here\n\n"
        + _bullets(after_lines, "Nothing recorded (no baseline correction, referencing, drift "
                                "correction or per-slice phase).")
    )
    if other_lines:
        log += "\n\n## Other settings recorded (not applied by code)\n\n" + _bullets(other_lines, "")
    if result_lines:
        log += ("\n\n## Results held by the GUI (not recomputed)\n\n" + _bullets(result_lines, "")
                + "\n\nOpen the experiment in openNMR (Open ▸ Load progress) to see these in full.")
    if dropped:
        log += ("\n\n## Reader arguments that could not be written as code\n\n"
                + _bullets([f"`{name}`" for name in dropped], "")
                + "\n\nThe load below uses the reader's default for each of them.")

    load_cell = (
        "import dataclasses\n\n"
        "import matplotlib.pyplot as plt\n\n"
        "from opennmr import io\n\n"
        f"path = {str(path)!r}   # edit this if the data has moved\n"
        f"vendor = {vendor!r}\n\n"
        "load_kwargs = " + pprint.pformat(kept, sort_dicts=False, width=88) + "\n\n"
        "spectrum = io.load(path, vendor=vendor, **load_kwargs)\n"
    )
    overrides = {}
    for key, field in _METADATA_FIELDS.items():
        if key in groups["metadata"]:
            value = _plain(groups["metadata"][key])
            if value is not None:
                overrides[field] = value
    cells = [_markdown(0, header), _markdown(1, log), _code(2, load_cell)]
    if overrides:
        body = "".join(f"    {field}={value!r},\n" for field, value in overrides.items())
        cells.append(_code(3, "spectrum.metadata = dataclasses.replace(\n"
                              "    spectrum.metadata,\n" + body + ")\n"))
    plot_cell = (
        "if spectrum.ndim == 1:\n"
        "    ppm = spectrum.axis_values(0, \"ppm\")\n"
        "    fig, ax = plt.subplots(figsize=(6, 3.5))\n"
        "    ax.plot(ppm, spectrum.data, lw=1, color=\"black\")\n"
        "    ax.invert_xaxis()\n"
        "    ax.set_xlabel(r\"$\\delta$ / ppm\")\n"
        "    ax.set_ylabel(\"Intensity (arb. units)\")\n"
        "    fig.tight_layout()\n"
        "else:\n"
        "    print(f\"{spectrum.ndim}D spectrum, shape {spectrum.data.shape}\")\n"
    )
    cells.append(_code(len(cells) + 1, plot_cell))
    cells = [{**cell, "id": _cell_id(i)} for i, cell in enumerate(cells)]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "opennmr": {"version": version, "source_path": str(path), "vendor": vendor},
        },
        "nbformat": NBFORMAT,
        "nbformat_minor": NBFORMAT_MINOR,
    }


def write_experiment_notebook(out_path: str, path: str, *, overwrite: bool = False, **kwargs) -> str:
    """Write :func:`experiment_notebook` to ``out_path`` (``.ipynb`` is
    added if missing) and return the path written. Refuses to replace an
    existing file unless ``overwrite``."""
    out_path = os.fspath(out_path)
    if not out_path.lower().endswith(".ipynb"):
        out_path += ".ipynb"
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"{out_path} already exists.")
    notebook = experiment_notebook(path, **kwargs)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, out_path)
    return out_path


def export_experiment_notebook(spectrum, out_path: str, *, load_kwargs: dict | None = None,
                               reprocessable: bool | None = None, overwrite: bool = False,
                               title: str | None = None) -> str:
    """Write the notebook for a loaded ``spectrum`` (from
    :func:`opennmr.io.load`, or the GUI) to ``out_path``.

    ``load_kwargs`` -- the arguments it was loaded with (default none:
    the reader's defaults, plus whatever the experiment's saved
    processing the reader restores itself). ``reprocessable`` defaults to
    whether the spectrum still holds its raw FID.
    """
    meta = spectrum.metadata
    path = meta.source_path
    if not path or not os.path.exists(path):
        raise ValueError(
            "This spectrum is not read from a file on disk (a simulation or derived spectrum?), "
            "so a notebook cannot re-read it. Save it as an openNMR dataset folder first "
            "(opennmr.io.history.save_spectrum_folder), then export that.")
    if reprocessable is None:
        reprocessable = (meta.raw or {}).get("fid_raw") is not None
    return write_experiment_notebook(
        out_path, path, vendor=meta.vendor, load_kwargs=load_kwargs,
        state=history.tracked_state(spectrum), source=history.describe_source(spectrum),
        title=title, reprocessable=reprocessable, overwrite=overwrite)
