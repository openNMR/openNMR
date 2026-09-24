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
_PROCESSING_VERSION = 2

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
    # How a 2D came to open as it did: re-derived on every load, and the
    # choice itself lives in the metadata sidecar (pseudo_2d_chosen).
    "pseudo_2d_basis", "pseudo_2d_chosen",
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
    # Whether that pseudo_2d is the user's own choice (Process Indirect
    # Dimension) or just the state it opened in -- only a choice
    # overrides the automatic default on the next load. Kept across saves
    # while the state is unchanged, since a reload rebuilds metadata.raw.
    previous = load_metadata_sidecar(source_path) or {}
    chosen = bool(meta.raw.get("pseudo_2d_chosen")) or (
        bool(previous.get("pseudo_2d_chosen")) and previous.get("pseudo_2d") == payload["pseudo_2d"])
    if spectrum.ndim == 2 and chosen:
        payload["pseudo_2d_chosen"] = True
    # The user's own title/notes (non-Bruker formats -- a Bruker title
    # lives in pdata/<procno>/title) are written by save_notes_sidecar,
    # not rebuilt from the spectrum, so carry them across.
    if isinstance(previous.get("notes"), str):
        payload["notes"] = previous["notes"]
    _write_json(metadata_sidecar_path(source_path), payload)


def load_notes_sidecar(source_path: str) -> str | None:
    """The title/notes saved for ``source_path`` in its
    ``opennmr_metadata.json``, or ``None`` when none were ever saved."""
    notes = (load_metadata_sidecar(source_path) or {}).get("notes")
    return notes if isinstance(notes, str) else None


def save_notes_sidecar(source_path: str, text: str) -> str:
    """Store ``text`` as the title/notes of ``source_path`` in its
    ``opennmr_metadata.json``, leaving every other key as it was (a
    sidecar is created holding just the notes if there is none yet).
    Returns the sidecar path. Raises ``FileNotFoundError`` if
    ``source_path`` does not exist."""
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError(f"No such experiment: {source_path!r}")
    path = metadata_sidecar_path(source_path)
    payload = _read_json(path) or {"opennmr_metadata_version": _METADATA_VERSION}
    payload["notes"] = text
    _write_json(path, payload)
    return path


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


def load_processing_sidecar(source_path: str, vendor: str | None = None) -> dict | None:
    """The saved ``state`` dict from ``opennmr_processing.json`` (see
    :func:`save_processing_sidecar`) -- ready to merge straight into a
    freshly-loaded spectrum's own ``metadata.raw``. ``None`` under the
    same conditions as :func:`load_metadata_sidecar`.

    ``vendor`` lets an older file be migrated correctly where the fix
    depends on which reader wrote it -- see
    :func:`_migrate_v1_phase`. Optional, so a caller that only wants
    the settings (and not to reason about phase provenance) need not
    supply it; omitting it is the conservative choice, keeping
    whatever a version-1 file recorded.
    """
    if not source_path:
        return None
    payload = _read_json(processing_sidecar_path(source_path))
    if payload is None:
        return None
    state = payload.get("state")
    if not isinstance(state, dict):
        return None
    if vendor is not None and payload.get("opennmr_processing_version", 1) < 2:
        state = _migrate_v1_phase(state, vendor)
    return state


#: The recorded-phase keys dropped when reading a version-1 processing
#: sidecar, and the user-phase keys dropped alongside them for the one
#: vendor whose base was mis-modelled -- see :func:`_migrate_v1_phase`.
_V1_STALE_RECORDED_PHASE_KEYS = (
    "recorded_phase0", "recorded_phase1", "recorded_phase0_f1", "recorded_phase1_f1",
)
_V1_STALE_USER_PHASE_KEYS = ("phase0", "phase1")

#: Vendors whose recorded phase is not read from the file at all but
#: computed by the reader (autophase), and so was written in the wrong
#: model by version-1 code.
_V1_COMPUTED_BASE_VENDORS = frozenset({"xpulse"})


def _migrate_v1_phase(state: dict, vendor: str) -> dict:
    """Drop phase that a version-1 sidecar cannot express correctly.

    **The recorded phase, for every vendor.** It is re-derived on every
    read regardless (Bruker from ``procs``'s ``PHC0``, X-Pulse from
    autophase), so a saved copy is at best redundant -- and a
    version-1 X-Pulse copy is actively wrong, written into a slot that
    means TopSpin's index ramp (``exp(+i*phi)``) while autophase
    returns the pivot model (``exp(-i*phi)``). See
    :func:`opennmr.io.bruker.recorded_phase_from_pivot_model`. Dropped
    rather than converted: the conversion needs the pivot the phase was
    found at, which a version-1 file does not record.

    **The user's own phase, for X-Pulse only.** With the base applied
    backwards, any phase a user dialled in was dialled in against a
    spectrum that was already rotated the wrong way -- so it is a
    correction to a bug, not to the data, and carrying it forward
    lands further from the right answer than starting clean does.
    Measured on ``test_data/xpulse_PFG/4``: reloading with the saved
    +38.33 deg on top of a correctly-composed base leaves row 0 needing
    another -132 deg, while the same read with no saved user phase at
    all lands on it exactly. Bruker's recorded phase was always applied
    correctly, so a user phase saved against it still means what it
    said and is kept.

    What is lost, once, on the first load after this upgrade: a saved
    "Zero Phase 0/1" on the recorded half, and an X-Pulse user phase
    that happened to be right anyway. Both are one re-autophase away.
    """
    drop = set(_V1_STALE_RECORDED_PHASE_KEYS)
    if vendor in _V1_COMPUTED_BASE_VENDORS:
        drop |= set(_V1_STALE_USER_PHASE_KEYS)
    return {key: value for key, value in state.items() if key not in drop}
