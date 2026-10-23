"""
Processing logs and save points from a notebook -- the same files the
GUI's File ▸ Save Progress / Load Progress / Processing Log use
(``opennmr_history.json`` and the human-readable ``opennmr_log.txt``
next to the data; see :mod:`opennmr.io.history`).

::

    from opennmr import notebook as nb

    print(nb.processing_log("/data/sample3/10"))        # what was done, when
    nb.list_save_points("/data/sample3/10")             # [{'id': 1, 'name': ...}, ...]
    nb.save_point_state("/data/sample3/10", "phased")   # the settings it holds

    # A simulation (or anything computed) saved as an openNMR dataset
    # folder, with its parameters logged -- the GUI opens it like data:
    nb.save_progress(sim_spectrum, "EM 5 Hz", folder="~/openNMR/simulations/csa1")
    nb.log_event(sim_spectrum, "fitted with fit_csa: delta_iso = 12.4(1) ppm")
    nb.load_progress("~/openNMR/simulations/csa1", "As saved")

What a notebook cannot do is re-run the GUI's processing chain on raw
vendor data from a save point's *settings*: for an experiment folder,
:func:`load_progress` returns the spectrum only where a save point kept
its data (derived, simulated and processed-only spectra), and
:func:`save_point_state` gives the settings otherwise.
"""

from __future__ import annotations

import copy
import dataclasses
import os

from .. import _version
from ..io import history as io_history
from ..io import persistence as io_persistence
from ..io import serialize as io_serialize

__all__ = [
    "processing_log", "processing_history", "list_save_points", "save_point_state",
    "save_progress", "load_progress", "log_event", "load_saved_spectrum",
    "saved_fits", "simulation_log", "simulation_history", "log_simulation",
]


def _path_of(target) -> str | None:
    if isinstance(target, (str, os.PathLike)):
        return os.path.expanduser(os.fspath(target))
    meta = getattr(target, "metadata", None)
    return getattr(meta, "source_path", None) if meta is not None else None


def _on_disk(spectrum) -> bool:
    meta = spectrum.metadata
    return bool(meta.source_path) and meta.vendor != "derived" and os.path.exists(meta.source_path)


def processing_history(target) -> dict | None:
    """The history of an experiment (a path, or a spectrum read from
    one): its saved record, or for a spectrum with nothing on disk the
    one kept on it by :func:`save_progress`/:func:`log_event` this
    session. ``None`` if there is none."""
    if not isinstance(target, (str, os.PathLike)):
        in_memory = (getattr(target.metadata, "raw", None) or {}).get("_history")
        if in_memory is not None:
            return in_memory
    return io_history.load_history(_path_of(target))


def processing_log(target) -> str:
    """The human-readable log -- exactly what ``opennmr_log.txt`` holds."""
    history = processing_history(target)
    if history is None:
        return f"No openNMR processing log for {_path_of(target) or 'this spectrum'} yet.\n"
    return io_history.render_log(history)


def list_save_points(target) -> list[dict]:
    """Every save point, oldest first: ``id``, ``name``, ``time``,
    ``note`` and whether it kept the data (``has_data``)."""
    return [{"id": e["id"], "name": e["name"], "time": e["time"], "note": e.get("note", ""),
             "has_data": bool(e.get("snapshot"))}
            for e in io_history.savepoints(processing_history(target))]


def save_point_state(target, which) -> dict:
    """The settings save point ``which`` (number or name) holds:
    ``{"processing": {...}, "metadata": {...}, "summary": [lines]}``."""
    entry = io_history.find_savepoint(processing_history(target), which)
    processing, metadata = io_history.split_state(entry.get("state") or {})
    return {"processing": processing, "metadata": metadata,
            "summary": io_history.state_summary(entry.get("state") or {})}


def _history_for(spectrum) -> dict:
    raw = spectrum.metadata.raw
    if _on_disk(spectrum):
        history = io_history.load_history(spectrum.metadata.source_path)
        if history is None:
            history = io_history.new_history(spectrum)
            io_history.add_event(history, f"Opened in openNMR {_version.__version__} (notebook)",
                                 kind="opened")
        raw["_history"] = history
        return history
    history = raw.get("_history")
    if history is None:
        history = io_history.new_history(spectrum)
        text = f"Created in openNMR {_version.__version__} (notebook)"
        if isinstance(raw.get("simulation"), dict):
            text += "\nSimulation parameters:\n" + "\n".join(
                "  " + line for line in io_history._param_lines(raw["simulation"]))
        io_history.add_event(history, text, kind="created")
        raw["_history"] = history
    return history


def log_event(spectrum, text: str) -> dict:
    """Add your own line to ``spectrum``'s log ("fitted with ...",
    "compared with sample 2"). Written straight away if it has a folder;
    kept on the spectrum otherwise, until :func:`save_progress`."""
    history = _history_for(spectrum)
    io_history.record_state(history, io_history.tracked_state(spectrum))
    entry = io_history.add_event(history, str(text), kind="note")
    if _on_disk(spectrum):
        io_history.write_history(spectrum.metadata.source_path, history)
    return entry


def save_progress(spectrum, name: str = "", note: str = "", folder: str | None = None):
    """Record a save point for ``spectrum`` and write its log.

    A spectrum with a folder of its own gets it next to its data; the
    processing sidecar is updated too, so the GUI reopens it in this
    state. One without (a simulation, anything computed) needs
    ``folder`` and is saved there as an openNMR dataset folder, which
    the GUI opens like any experiment. Its data are kept with the save
    point, so :func:`load_progress` returns them exactly.

    Returns ``(spectrum, entry)``: the spectrum as now saved (vendor
    ``"opennmr"`` if it was just given a folder) and the save point.
    """
    history = _history_for(spectrum)
    if not _on_disk(spectrum):
        if not folder:
            raise ValueError("This spectrum has no folder of its own -- pass folder=.")
        folder = os.path.expanduser(folder)
        if os.path.exists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
            folder = io_history.unique_folder(folder)
        io_history.save_spectrum_folder(spectrum, folder, label=os.path.basename(folder.rstrip("\\/")))
        raw = {k: v for k, v in spectrum.metadata.raw.items() if k != "_history"}
        raw["origin_vendor"] = spectrum.metadata.vendor
        spectrum = dataclasses.replace(spectrum, metadata=dataclasses.replace(
            spectrum.metadata, vendor="opennmr", source_path=folder, raw=raw))
        spectrum.metadata.raw["_history"] = history
        history["source"] = io_history.describe_source(spectrum)
        io_history.add_event(history, f"Saved as an openNMR dataset folder: {folder}", kind="saved")
    io_persistence.save_metadata_sidecar(spectrum)
    io_persistence.save_processing_sidecar(spectrum)
    snapshot = None
    if spectrum.metadata.vendor == "opennmr" or not _raw_fid_behind(spectrum):
        number = int(history.get("next_savepoint", 1))
        folder_ = io_history.snapshot_dir(spectrum.metadata.source_path)
        path = os.path.join(folder_, f"savepoint_{number}.npz")
        io_history.write_snapshot(path, spectrum)
        snapshot = os.path.relpath(path, os.path.dirname(io_history.history_path(spectrum.metadata.source_path)))
    entry = io_history.add_savepoint(history, io_history.tracked_state(spectrum), name=name, note=note,
                                     snapshot=snapshot)
    io_history.write_history(spectrum.metadata.source_path, history)
    return spectrum, entry


def _raw_fid_behind(spectrum) -> bool:
    meta = spectrum.metadata
    if meta.vendor in ("xpulse", "varian", "jeol", "spinsolve", "simpson"):
        return True
    if meta.vendor == "bruker" and meta.source_path:
        return any(os.path.exists(os.path.join(meta.source_path, name)) for name in ("fid", "ser"))
    return False


def load_saved_spectrum(folder: str):
    """An openNMR dataset folder (a saved simulation or derived
    spectrum) as a spectrum, with its saved processing state -- its
    data were saved already processed, so nothing is recomputed."""
    folder = os.path.expanduser(folder)
    spectrum = io_history.read_spectrum_folder(folder)
    spectrum.metadata.raw.update(io_persistence.load_processing_sidecar(folder) or {})
    saved = io_persistence.load_metadata_sidecar(folder) or {}
    if saved.get("mass_g") is not None:
        spectrum.metadata.mass = saved["mass_g"]
    return spectrum


def load_progress(target, which):
    """The spectrum as it was at save point ``which`` (number or name).

    Only possible where the save point kept its data -- anything saved
    from a notebook, and in the GUI any spectrum that cannot be
    recomputed from a raw FID. A save point of a raw-FID experiment
    holds its *settings*; re-applying them means re-running the GUI's
    processing, so this raises ``NotImplementedError`` and points to
    :func:`save_point_state`. Open the experiment in the GUI and use
    File ▸ Load Progress for those."""
    path = _path_of(target)
    history = processing_history(target)
    entry = io_history.find_savepoint(history, which)
    if not entry.get("snapshot"):
        raise NotImplementedError(
            f"Save point {entry['id']} “{entry['name']}” holds processing settings, not data: "
            "re-applying them to raw data needs the GUI's processing (File ▸ Load Progress). "
            "nb.save_point_state(...) returns the settings.")
    data, data_imag, axes = io_history.read_snapshot(
        os.path.join(os.path.dirname(io_history.history_path(path)), entry["snapshot"]))
    if io_history.is_dataset_folder(path):
        base = io_history.read_spectrum_folder(path)
    else:
        from .. import io
        base = io.load(path)
    processing, metadata = io_history.split_state(entry.get("state") or {})
    raw = {k: v for k, v in base.metadata.raw.items() if k not in io_persistence.processing_state(base)}
    raw.update(copy.deepcopy(processing))
    meta = dataclasses.replace(base.metadata, raw=raw, mass=metadata.get("mass_g", base.metadata.mass),
                               nucleus=metadata.get("nucleus", base.metadata.nucleus))
    return dataclasses.replace(base, data=data, data_imag=data_imag, axes=axes, metadata=meta)


# ------------------------------------------------------------------
# Fits and simulations
# ------------------------------------------------------------------

def saved_fits(target) -> dict:
    """The fits saved with an experiment (GUI Fit panels: CSA,
    J-coupling, quadrupolar), by panel: ``label``, ``time``, ``target``,
    ``report`` (text), ``settings`` (lines) and ``result`` -- the fit's
    full result, decoded exactly (``popt``, ``pcov``, ``x``, ``y_fit``,
    ...), ready to use in a notebook."""
    if isinstance(target, (str, os.PathLike)):
        state = io_persistence.load_processing_sidecar(_path_of(target)) or {}
    else:
        state = target.metadata.raw or {}
    out = {}
    for key, fit in (state.get("fits") or {}).items():
        if not isinstance(fit, dict):
            continue
        out[key] = {"label": fit.get("label"), "time": fit.get("time"), "target": fit.get("target"),
                    "report": fit.get("report", ""), "settings": list(fit.get("inputs_text") or []),
                    "result": io_serialize.decode(fit.get("result"))}
    return out


def simulation_log(folder: str | None = None) -> str:
    """The simulation journal written out (``opennmr_simulation_log.txt``)."""
    return io_history.render_simulation_log(io_history.load_simulation_journal(folder))


def simulation_history(folder: str | None = None) -> list[dict]:
    """Every journalled simulation run, oldest first."""
    return list(io_history.load_simulation_journal(folder).get("entries") or [])


def log_simulation(label: str, parameters: dict, summary: str = "", folder: str | None = None) -> dict:
    """Add a notebook simulation to the same journal the GUI's Simulate
    panels write to, so one log holds every simulation. ``parameters``
    are written one per line (they are for reading; a notebook run is
    restored by re-running its cell, not from the GUI)."""
    return io_history.record_simulation(
        "notebook", f"{label} (notebook)", {"parameters": io_serialize.encode(parameters)},
        io_history._param_lines(parameters), summary, folder=folder, merge_window_s=-1)
