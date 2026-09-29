"""
openNMR states from a notebook: save a group of spectra (with their
fits and kept fit models) under a name, open one the GUI saved, and read
what is in it. The format is :mod:`opennmr.io.state` -- the GUI's File ▸
Save openNMR State / Open openNMR State -- so either side opens the
other's.

    from opennmr.notebook.state import save_state, load_state, describe_state

    save_state("~/Documents/openNMR/states/glass fits.opennmr",
               [("27Al MAS", spectrum), ("slice 3", slice3)], name="glass fits")
    print(describe_state(folder))
    for item in load_state(folder):
        item["label"], item["spectrum"], item["fits"], item["fit_models"]

A notebook opens every entry from its saved copy of the data. An
experiment saved *without* a copy (a large 2D) is re-read from its own
folder, but the GUI's processing chain is not re-applied here -- the
entry says so (``note``) -- as for :func:`opennmr.notebook.load_progress`.
"""

from __future__ import annotations

import os

from ..io import history as io_history
from ..io import serialize
from ..io import state as io_state
from .fit_models import decode_fit_model

__all__ = ["save_state", "load_state", "describe_state", "saved_fit_models"]


def _kind(spectrum) -> str:
    raw = spectrum.metadata.raw or {}
    if raw.get("simulation"):
        return "simulation"
    if spectrum.metadata.vendor in ("derived", "opennmr", "simulated"):
        return "derived"
    return "experiment"


def save_state(folder, spectra, name=None, notes="", overwrite=False) -> str:
    """Save ``spectra`` -- NMRSpectrum objects, or ``(label, spectrum)``
    pairs -- as an openNMR state in ``folder``. Each is saved as a copy of
    its data with its processing state (``metadata.raw``: fits, kept fit
    models...). Returns the folder."""
    folder = os.path.expanduser(str(folder))
    entries = []
    for k, item in enumerate(spectra, 1):
        label, spectrum = item if isinstance(item, tuple) else (None, item)
        label = label or (spectrum.metadata.raw or {}).get("dataset_label") \
            or os.path.basename(str(spectrum.metadata.source_path or "")) or f"Spectrum {k}"
        kind = _kind(spectrum)
        path = spectrum.metadata.source_path
        source = ({"path": path, "vendor": spectrum.metadata.vendor, "load_kwargs": {}}
                  if kind == "experiment" and path and os.path.exists(str(path)) else None)
        entries.append({"label": label, "kind": kind, "spectrum": spectrum, "source": source,
                        "copy": True})
    from .. import _version
    return io_state.save_state(folder, name or os.path.splitext(os.path.basename(folder))[0],
                               entries, notes=notes, app_version=_version.__version__,
                               overwrite=overwrite)


def saved_fit_models(spectrum) -> list[dict]:
    """The kept fit models on ``spectrum`` (``metadata.raw["fit_models"]``),
    decoded -- ready for :func:`opennmr.notebook.fit_models.compare_fit_models`
    and ``plot_fit_comparison``."""
    return [decode_fit_model(e) for e in (spectrum.metadata.raw or {}).get("fit_models") or []]


def load_state(folder) -> list[dict]:
    """Every entry of a saved state: ``label``, ``kind``, ``spectrum``,
    ``fits`` (``{panel: record}`` with each full result decoded),
    ``fit_models`` (decoded), ``source``, ``summary`` and ``note``."""
    from .. import io
    info = io_state.load_state(os.path.expanduser(str(folder)))
    out = []
    for entry in info["entries"]:
        spectrum, note = entry["spectrum"], ""
        if spectrum is None:
            source = entry.get("source") or {}
            if not entry["source_available"]:
                note = f"No saved copy, and {source.get('path')} is not there any more."
            else:
                spectrum = io.load(source["path"], vendor=source.get("vendor"))
                processing, _meta = io_history.split_state(entry.get("state") or {})
                for key in ("fits", "fit_models"):
                    if key in processing:
                        spectrum.metadata.raw[key] = processing[key]
                note = ("Re-read from its own folder; the saved processing settings are "
                        "recorded in 'state' but not re-applied here (the GUI does that).")
        fits = {}
        if spectrum is not None:
            for key, fit in ((spectrum.metadata.raw or {}).get("fits") or {}).items():
                fits[key] = {**fit, "result": serialize.decode(fit.get("result"))}
        out.append({"label": entry["label"], "kind": entry["kind"], "spectrum": spectrum,
                    "fits": fits,
                    "fit_models": saved_fit_models(spectrum) if spectrum is not None else [],
                    "source": entry.get("source"), "state": entry.get("state"),
                    "summary": entry.get("summary"), "note": note})
    return out


def describe_state(folder) -> str:
    """The saved state's readable log."""
    return io_state.describe_state(os.path.expanduser(str(folder)))
