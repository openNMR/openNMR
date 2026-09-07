"""
Sidecar-file persistence: two small JSON files written next to each
experiment on disk, so a spectrum's own sample-level facts (mass,
probe, nucleus overrides -- anything no vendor file ever records,
plus a few vendor-recorded fields worth "taking note of" for
at-a-glance browsing) and its own live reprocessing recipe (window/
zero-fill/phase/truncate/baseline/analysis-state -- everything
already tracked in ``metadata.raw`` for in-session persistence, see
``opennmr.gui.main_window``'s own ``_persist_fid_processing_controls_
to_metadata`` and its siblings) both survive an app restart, not just
a reselection within one running GUI session.

Explicit user request: "the best thing to do for file io and storing
metadata is to actually write a file to the same experiment folder...
I think there should be 2: one for data and metadata, and another for
processing parameters." Two files, not one, so a quick look at the
metadata file alone (small, essentially static once a sample is
mounted) doesn't require wading through the processing file (larger,
rewritten on nearly every live control tweak).

Naming: ``opennmr_metadata.json``/``opennmr_processing.json``, written
directly inside ``source_path`` when it's a directory (Bruker, Varian,
and a folder-per-experiment X-Pulse dataset all use one directory per
experiment); as ``<basename>.opennmr_metadata.json``/
``<basename>.opennmr_processing.json`` alongside ``source_path`` when
it's a single file instead (JEOL, and a bare-file X-Pulse dataset) --
several JEOL experiments in this project's own test_data share one
directory with no per-experiment subfolder at all, so a directory-
scoped name there would silently collide across sibling experiments.
"""

from __future__ import annotations

import json
import os

#: Bumped only if a future format change needs old sidecar files to
#: be recognizably distinguishable from new ones -- nothing currently
#: reads this back to gate behaviour, but it costs nothing to have in
#: place before it's actually needed.
_METADATA_VERSION = 1
_PROCESSING_VERSION = 1

#: Keys in ``NMRSpectrum.metadata.raw`` deliberately never written to
#: the processing-parameters sidecar: full vendor parameter dumps
#: (large, and always freely re-derivable by re-reading the vendor's
#: own files on the next load -- persisting them would just be a
#: stale, bloated copy of something already on disk elsewhere), raw
#: FID arrays (not JSON-serializable at all, and reconstructed by a
#: fresh load regardless), and load-time diagnostic notes (re-derived
#: fresh every load -- meaningless, possibly misleading, once stale).
_PROCESSING_RAW_BLOCKLIST = frozenset({
    "acqus", "procs", "proc2s", "params", "exp_params",
    "fid", "fid_raw",
    "window_unsupported", "read_2d_shape_guess_failed", "loaded_from_processed",
    # Re-derived from the vendor's own PH_mod on every load, so it
    # would be a stale copy -- and, worse, an actively wrong one for
    # the case that matters: "requested" only describes *this*
    # session's explicit toggle. `magnitude_mode` itself is persisted
    # (it's a real user choice worth restoring); only the provenance
    # note is dropped.
    "magnitude_mode_source",
    # An extracted slice's link back to the 2D it came from. A live
    # spectrum object and a within-session relationship -- neither
    # survives a reload, and the object isn't JSON-serialisable
    # anyway. See MainWindow._on_slice_keep_requested.
    "slice_parent", "slice_axis", "slice_index",
    # Re-derived from SF/BF1 on every load, so persisting it would be a
    # stale copy of something the vendor files already say. The user's
    # own SR (`sr_shift_hz`) *is* persisted -- it is a real choice.
    "sr_recorded_hz", "sr_recorded_hz_f1",
})


def _sidecar_base(source_path: str) -> str:
    if os.path.isdir(source_path):
        return os.path.join(source_path, "opennmr")
    return source_path + ".opennmr"


def metadata_sidecar_path(source_path: str) -> str:
    """Where :func:`save_metadata_sidecar`/:func:`load_metadata_sidecar`
    read/write for a spectrum whose ``metadata.source_path`` is
    ``source_path`` -- see this module's own docstring for the
    directory-vs-file naming rule."""
    return _sidecar_base(source_path) + "_metadata.json"


def processing_sidecar_path(source_path: str) -> str:
    """:func:`metadata_sidecar_path`'s own sibling for the processing-
    parameters file."""
    return _sidecar_base(source_path) + "_processing.json"


def _write_json(path: str, payload: dict) -> None:
    # Write-to-temp-then-rename is atomic on both POSIX and Windows --
    # a crash or concurrent read mid-write never observes a truncated/
    # half-written sidecar file.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        # Fails soft -- a corrupted or foreign (e.g. hand-edited into
        # invalid JSON) sidecar file should never block a spectrum
        # from loading; it just means nothing gets restored this time.
        return None


def save_metadata_sidecar(spectrum) -> None:
    """Write ``spectrum``'s own sample-level facts to its
    ``opennmr_metadata.json``. A no-op if ``metadata.source_path``
    isn't set or doesn't exist on disk (a synthetic/derived spectrum
    with nothing real to write next to)."""
    source_path = spectrum.metadata.source_path
    if not source_path or not os.path.exists(source_path):
        return
    meta = spectrum.metadata
    payload = {
        "opennmr_metadata_version": _METADATA_VERSION,
        "vendor": meta.vendor,
        "nucleus": meta.nucleus,
        "nucleus_indirect": meta.nucleus_indirect,
        "probe": meta.probe,
        "sfo1_mhz": meta.sfo1,
        "procno": meta.procno,
        "mass_g": meta.mass,
        # Only meaningful (and only ever restored on the next load) for
        # a genuine 2D spectrum -- lets a deliberate "Process Indirect
        # Dimension" opt-in survive an app restart, not just get reset
        # back to the pseudo-2D default every time (see MainWindow._load's
        # own "Pseudo-2D is the default treatment for any 2D data..."
        # comment for why that default exists in the first place).
        "pseudo_2d": meta.pseudo_2d if spectrum.ndim == 2 else None,
    }
    _write_json(metadata_sidecar_path(source_path), payload)


def load_metadata_sidecar(source_path: str) -> dict | None:
    """The parsed dict from ``opennmr_metadata.json``, or ``None`` if
    there isn't one yet, ``source_path`` is falsy, or it can't be
    parsed (see :func:`_read_json`)."""
    if not source_path:
        return None
    return _read_json(metadata_sidecar_path(source_path))


def save_processing_sidecar(spectrum) -> None:
    """Write ``spectrum``'s current live reprocessing recipe --
    everything already tracked in ``metadata.raw`` for in-session
    persistence, minus the large/non-serializable/redundant entries
    in :data:`_PROCESSING_RAW_BLOCKLIST` (and anything else that
    doesn't survive a plain ``json.dumps`` -- e.g. a stashed numpy
    axis array -- rather than needing every such key named here up
    front) -- to ``opennmr_processing.json``. Same no-op guard as
    :func:`save_metadata_sidecar`."""
    source_path = spectrum.metadata.source_path
    if not source_path or not os.path.exists(source_path):
        return
    state = {}
    for key, value in spectrum.metadata.raw.items():
        if key in _PROCESSING_RAW_BLOCKLIST:
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        state[key] = value
    payload = {"opennmr_processing_version": _PROCESSING_VERSION, "state": state}
    _write_json(processing_sidecar_path(source_path), payload)


def load_processing_sidecar(source_path: str) -> dict | None:
    """The saved ``state`` dict from ``opennmr_processing.json`` (see
    :func:`save_processing_sidecar`) -- ready to merge straight into a
    freshly-loaded spectrum's own ``metadata.raw``. ``None`` under the
    same conditions as :func:`load_metadata_sidecar`."""
    if not source_path:
        return None
    payload = _read_json(processing_sidecar_path(source_path))
    if payload is None:
        return None
    state = payload.get("state")
    return state if isinstance(state, dict) else None
