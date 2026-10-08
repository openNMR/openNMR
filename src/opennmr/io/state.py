"""
openNMR states: a named, self-describing folder holding a chosen group of
spectra -- experiments with their processing, slices of 2Ds, derived
spectra (VOCS, arithmetic, kept slices), simulations -- together with
every fit and kept fit model on them, so the whole piece of work can be
reopened later exactly as it was.

Layout of ``<name>.opennmr/``::

    opennmr_state.json        what is in it (read back)
    opennmr_state_log.txt     the same, for people (never read back)
    spectra/01_<label>/       one openNMR dataset folder per saved copy:
        opennmr_spectrum.npz  the data as processed, exactly
        opennmr_spectrum.json what it is, where it came from
        opennmr_processing.json   its processing state, fits and kept fit models

Each entry in ``opennmr_state.json``: ``label``, ``kind`` ("experiment",
"slice", "derived", "simulation"), ``source`` (an experiment's own data
folder, reader and load settings -- reopened from there and reprocessed
with ``state``, the openNMR way), ``folder`` (the saved copy, when there
is one: always for anything without raw data of its own, and for a 1D
experiment; for a 2D experiment when asked, since it can be large),
``state`` (the tracked processing state, as a save point records it),
``prefer_copy`` (open from the copy even when the data are still there:
a 2D read from already-processed data, whose saved processing cannot be
re-applied by re-reading), ``origin`` and a readable ``summary``.

Every copy folder is also an ordinary openNMR dataset folder, so File ▸
Open can open any one spectrum from a state on its own.

Qt-free; used by the GUI (File ▸ Save / Open openNMR State) and by
:mod:`opennmr.notebook.state`.
"""

from __future__ import annotations

import datetime as _dt
import json
import os

from . import history as io_history
from . import persistence

STATE_FILE = "opennmr_state.json"
STATE_LOG_FILE = "opennmr_state_log.txt"
STATE_SUFFIX = ".opennmr"
STATE_VERSION = 1
KINDS = ("experiment", "slice", "derived", "simulation")
KIND_NAMES = {"experiment": "Experiment", "slice": "Slice", "derived": "Derived spectrum",
              "simulation": "Simulation"}


#: The folder chosen in Settings ▸ Saving (explicit user request,
#: 2026-10-08: "allow people to define the default save location for
#: saving openNMR states ... Sometimes it is not obvious where it got
#: saved to"). ``None``: the built-in default.
_USER_STATE_ROOT: list = [None]


def set_default_state_root(folder: str | None) -> str:
    """Make ``folder`` where new states are saved by default (``None`` or
    ``""`` goes back to the built-in default). Returns the folder now in
    use."""
    _USER_STATE_ROOT[0] = os.path.expanduser(str(folder)) if folder else None
    return default_state_root()


def builtin_state_root() -> str:
    """The default with no folder chosen in Settings: ``OPENNMR_STATES_DIR``
    if set, else ``~/Documents/openNMR/states``."""
    override = os.environ.get("OPENNMR_STATES_DIR")
    if override:
        return os.path.expanduser(override)
    home = os.path.expanduser("~")
    documents = os.path.join(home, "Documents")
    return os.path.join(documents if os.path.isdir(documents) else home, "openNMR", "states")


def default_state_root() -> str:
    """The folder chosen with :func:`set_default_state_root` (Settings ▸
    Saving), else :func:`builtin_state_root`."""
    if _USER_STATE_ROOT[0]:
        return _USER_STATE_ROOT[0]
    return builtin_state_root()


def default_state_folder(name: str) -> str:
    return os.path.join(default_state_root(), io_history.safe_folder_name(name) + STATE_SUFFIX)


def is_state_folder(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.isfile(os.path.join(path, STATE_FILE))


def _json_safe(value):
    return json.loads(json.dumps(value, default=io_history._json_default))


def _fit_lines(state: dict) -> list[str]:
    """Readable lines for the fits and kept fit models in a state."""
    lines = []
    for key, fit in sorted((state.get("fits") or {}).items()):
        if isinstance(fit, dict):
            lines.append(f"Fit ({fit.get('label', key)}) on {fit.get('target', '')}, "
                         f"{fit.get('time', '')}:")
            report = (fit.get("report") or "").strip()
            lines += ["    " + line for line in report.splitlines()[:40]]
    models = state.get("fit_models") or []
    if models:
        from ..notebook.fit_models import decode_fit_model, fit_statistics
        lines.append(f"Kept fit models ({len(models)}):")
        for entry in models:
            try:
                model = decode_fit_model(entry)
            except Exception:  # noqa: BLE001 -- describe what can be read
                lines.append("    (unreadable record)")
                continue
            stats = fit_statistics(model)
            lines.append(f"    “{model['name']}” -- {model.get('kind_label', model['kind'])}, "
                         f"R² {stats['r2']:.5f}"
                         + (f", {stats['k']} free parameters" if stats["k"] is not None else "")
                         + f", kept {model.get('time', '')}")
    return lines


def entry_summary(entry: dict) -> list[str]:
    """Readable lines describing one entry (without its fits)."""
    lines = []
    source = entry.get("source")
    if source:
        lines.append(f"Data: {source.get('path')} ({source.get('vendor')})")
    if entry.get("origin"):
        lines.append(f"From: {entry['origin']}")
    if entry.get("folder"):
        lines.append(f"Saved copy: {entry['folder']}")
    state = entry.get("state") or {}
    try:
        lines += io_history.state_summary(state, source)
    except Exception:  # noqa: BLE001 -- a summary must never stop a save
        pass
    return lines


def render_state_log(info: dict) -> str:
    lines = [f"openNMR state: {info['name']}", f"Saved {str(info['created']).replace('T', ' ')}"
             + (f" by openNMR {info['app_version']}" if info.get("app_version") else ""), ""]
    if info.get("notes"):
        lines += ["Notes:", *("  " + line for line in info["notes"].splitlines()), ""]
    lines.append(f"{len(info['entries'])} spectra:")
    for number, entry in enumerate(info["entries"], 1):
        lines.append("")
        lines.append(f"{number}. {entry['label']} -- {KIND_NAMES.get(entry['kind'], entry['kind'])}")
        lines += ["   " + line for line in entry.get("summary") or []]
        lines += ["   " + line for line in _fit_lines(entry.get("state") or {})]
    return "\n".join(lines) + "\n"


def save_state(folder: str, name: str, entries: list[dict], notes: str = "",
               app_version: str = "", overwrite: bool = False) -> str:
    """Write an openNMR state to ``folder``.

    ``entries``: dicts with ``label``, ``kind`` (:data:`KINDS`),
    ``spectrum`` (the :class:`NMRSpectrum` as it is now), ``state`` (its
    tracked processing state; default: read from the spectrum),
    ``source`` (``{"path", "vendor", "load_kwargs"}`` for an experiment,
    else None), ``origin`` (optional) and ``copy`` (keep a copy of the
    data; forced on for anything without a ``source``).

    Refuses to write into an existing folder unless ``overwrite`` (then
    only into one that is already an openNMR state). Returns ``folder``.
    """
    if not entries:
        raise ValueError("Choose at least one spectrum to save.")
    if os.path.exists(folder) and os.listdir(folder):
        if not overwrite:
            raise FileExistsError(f"{folder} already exists -- choose another name or folder.")
        if not is_state_folder(folder):
            raise FileExistsError(f"{folder} exists and is not an openNMR state; not overwriting it.")
    os.makedirs(os.path.join(folder, "spectra"), exist_ok=True)
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    out = []
    for number, entry in enumerate(entries, 1):
        kind = entry.get("kind", "derived")
        if kind not in KINDS:
            raise ValueError(f"Unknown kind {kind!r}; use one of {KINDS}.")
        spectrum = entry["spectrum"]
        state = entry.get("state")
        if state is None:
            state = io_history.tracked_state(spectrum)
        state = _json_safe(state)
        source = entry.get("source")
        if source and not source.get("path"):
            source = None
        copy = bool(entry.get("copy", True)) or source is None
        record = {"label": str(entry["label"]), "kind": kind, "source": _json_safe(source),
                  "origin": entry.get("origin"), "state": state, "folder": None,
                  "prefer_copy": bool(entry.get("prefer_copy", False)) and copy,
                  "shape": list(getattr(spectrum.data, "shape", ())),
                  "nucleus": spectrum.metadata.nucleus}
        if copy:
            rel = os.path.join("spectra", f"{number:02d}_{io_history.safe_folder_name(entry['label'])}")
            sub = os.path.join(folder, rel)
            io_history.save_spectrum_folder(
                spectrum, sub, label=str(entry["label"]), origin=entry.get("origin")
                or (source or {}).get("path"),
                origin_vendor=(source or {}).get("vendor") or spectrum.metadata.vendor)
            processing, _meta = io_history.split_state(state)
            persistence._write_json(persistence.processing_sidecar_path(sub),
                                    {"opennmr_processing_version": persistence._PROCESSING_VERSION,
                                     "state": processing})
            record["folder"] = rel.replace(os.sep, "/")
        record["summary"] = entry_summary(record)
        out.append(record)
    info = {"opennmr_state_version": STATE_VERSION, "name": str(name), "created": stamp,
            "notes": str(notes or ""), "app_version": app_version, "entries": out}
    tmp = os.path.join(folder, STATE_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(folder, STATE_FILE))
    with open(os.path.join(folder, STATE_LOG_FILE), "w", encoding="utf-8") as handle:
        handle.write(render_state_log(info))
    return folder


def read_state_info(folder: str) -> dict:
    """``opennmr_state.json`` as a dict (no data read)."""
    path = folder if os.path.isfile(folder) else os.path.join(folder, STATE_FILE)
    with open(path, encoding="utf-8") as handle:
        info = json.load(handle)
    if info.get("opennmr_state_version", 0) > STATE_VERSION:
        raise ValueError("This state was saved by a newer openNMR; update to open it.")
    return info


def load_state(folder: str) -> dict:
    """The state's info with, for each entry, ``spectrum`` (its saved
    copy read back, with ``state`` merged into ``metadata.raw``, or None
    when there is no copy) and ``source_available`` (whether an
    experiment's own data are still where they were)."""
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    info = read_state_info(folder)
    for entry in info["entries"]:
        entry["spectrum"] = None
        if entry.get("folder"):
            path = os.path.join(folder, *entry["folder"].split("/"))
            spectrum = io_history.read_spectrum_folder(path)
            processing, _meta = io_history.split_state(entry.get("state") or {})
            spectrum.metadata.raw.update(processing)
            entry["spectrum"] = spectrum
        source = entry.get("source") or {}
        entry["source_available"] = bool(source.get("path")) and os.path.exists(source["path"])
    info["folder"] = folder
    return info


def describe_state(folder: str) -> str:
    """The readable log of a saved state."""
    path = os.path.join(folder, STATE_LOG_FILE)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    return render_state_log(read_state_info(folder))
