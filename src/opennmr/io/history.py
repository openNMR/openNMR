"""
Per-spectrum processing history: a human-readable log, and named save
points that can be restored.

Explicit user request: *"have a user 'save progress' on a spectrum, and
... create a log file for each spectrum handled ... used to 'save' and
'load' progress ... stored in the same location as the data itself ...
human readable so that it's easy to know what was done to the spectrum
and how. The same is true for simulated data."*

Two files per experiment, next to the data (same directory-vs-file rule
as :mod:`opennmr.io.persistence`'s sidecars):

``opennmr_history.json``
    The record: every change, save point and restore, with the full
    state at each save point. **This is what Load Progress reads.**

``opennmr_log.txt``
    The same record written out for people -- dates, times, what
    changed from what to what, and a full summary at each save point.
    It is *generated* from the JSON on every write, never parsed back:
    a number reformatted or hand-edited in a text file must not be able
    to silently change what a restore produces. So the two can never
    disagree, and the text can be edited or deleted without harm (it
    is rewritten on the next change).

What is tracked is exactly what :func:`opennmr.io.persistence.processing_state`
returns (the recipe the processing sidecar persists and a reopen
restores) plus the sample-level metadata the metadata sidecar holds
(mass, nucleus overrides, pseudo-2D choice). A save point therefore
records the same thing an app restart would restore, by construction --
there is no second definition of "the state" to drift out of step.

Changes are found by *diffing* that state, not by instrumenting each
control: whatever a control writes into ``metadata.raw`` is logged, so
a control added later is logged without anyone remembering to add a
log call. A burst of edits to the same setting (dragging a phase,
typing a number) is folded into one entry while it stays within
:data:`MERGE_WINDOW_S` of the previous one, so the log reads "LB 0 → 5
Hz", not every intermediate value.

Spectra with nothing on disk (simulations, derived spectra) keep their
history in memory until saved as an openNMR dataset folder --
:func:`save_spectrum_folder` -- after which they behave like any other
experiment. For a spectrum that cannot be recomputed from a raw FID
(derived, simulated, processed-only), a save point also stores the
data itself (:func:`write_snapshot`), because the recipe alone could
not reproduce it.

Qt-free: used identically by the GUI and from a notebook.
"""

from __future__ import annotations

import copy
import datetime as _dt
import html
import json
import os
import re

import numpy as np

from . import persistence

HISTORY_VERSION = 1

#: Consecutive edits to settings already in the newest change entry are
#: merged into it while they arrive within this many seconds of its
#: last update.
MERGE_WINDOW_S = 90.0

#: Folder (next to the history file) holding save-point data snapshots.
SNAPSHOT_DIRNAME = "opennmr_savepoints"

#: File names inside an openNMR dataset folder (a saved simulation or
#: derived spectrum) -- see :func:`save_spectrum_folder`.
DATASET_DATA_FILE = "opennmr_spectrum.npz"
DATASET_INFO_FILE = "opennmr_spectrum.json"


# ============================================================
# Paths
# ============================================================

def _base(source_path: str) -> str:
    if os.path.isdir(source_path):
        return os.path.join(source_path, "opennmr")
    return source_path + ".opennmr"


def history_path(source_path: str) -> str:
    """Where the machine-readable history of ``source_path`` lives."""
    return _base(source_path) + "_history.json"


def log_path(source_path: str) -> str:
    """Where the human-readable log of ``source_path`` lives."""
    return _base(source_path) + "_log.txt"


def snapshot_dir(source_path: str) -> str:
    folder = source_path if os.path.isdir(source_path) else os.path.dirname(source_path)
    prefix = "" if os.path.isdir(source_path) else os.path.basename(source_path) + "."
    return os.path.join(folder, prefix + SNAPSHOT_DIRNAME)


def experiment_files(source_path: str) -> list[str]:
    """Every file openNMR keeps beside ``source_path`` that exists now:
    the history (``opennmr_history.json``), its text log
    (``opennmr_log.txt``), the metadata and processing sidecars
    (``opennmr_metadata.json``, ``opennmr_processing.json``) and the
    save points' data folder (``opennmr_savepoints``)."""
    candidates = [history_path(source_path), log_path(source_path),
                  persistence.metadata_sidecar_path(source_path),
                  persistence.processing_sidecar_path(source_path),
                  snapshot_dir(source_path)]
    found = [path for path in candidates if os.path.exists(path)]
    found += [path + ".tmp" for path in candidates[:4] if os.path.exists(path + ".tmp")]
    return found


def clear_experiment_files(source_path: str, keep_notes: bool = True) -> list[str]:
    """Delete openNMR's own files beside ``source_path`` (see
    :func:`experiment_files`), so the experiment opens next time as the
    vendor's files alone say -- no earlier processing, referencing,
    analysis, log or save points. The vendor's own files are never
    touched. Returns what was deleted.

    ``keep_notes`` : the title/notes typed for the experiment live in
    ``opennmr_metadata.json`` too; kept (that file rewritten holding
    only them) unless this is False.

    Works from a notebook as from the GUI (File ▸ Clear All History…)::

        from opennmr.io import history
        history.clear_experiment_files(r"C:/data/my_expt/3")
    """
    import shutil

    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError(f"No such experiment: {source_path!r}")
    notes = None
    if keep_notes:
        notes = (persistence._read_json(persistence.metadata_sidecar_path(source_path)) or {}).get("notes")
    deleted = []
    for path in experiment_files(source_path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        deleted.append(path)
    if isinstance(notes, str) and notes.strip():
        persistence.save_notes_sidecar(source_path, notes)
    return deleted


# ============================================================
# The tracked state
# ============================================================

#: Metadata fields tracked alongside the processing state, under
#: ``meta.``-prefixed keys so they cannot collide with a raw key.
_META_FIELDS = (("mass", "meta.mass_g"), ("nucleus", "meta.nucleus"),
                ("nucleus_indirect", "meta.nucleus_indirect"))


def _normalise(value):
    """A JSON round trip: tuples become lists, numpy scalars plain
    numbers -- so equality means "would be saved the same"."""
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")


def tracked_state(spectrum) -> dict:
    """The state the history follows: the persisted processing recipe
    plus the sample-level metadata overrides."""
    state = _normalise(persistence.processing_state(spectrum))
    meta = spectrum.metadata
    for field, key in _META_FIELDS:
        value = getattr(meta, field, None)
        if value is not None:
            state[key] = _normalise(value)
    if spectrum.ndim == 2:
        state["meta.pseudo_2d"] = bool(meta.pseudo_2d)
    return state


def split_state(state: dict) -> tuple[dict, dict]:
    """``(processing_state, metadata)`` from a tracked state -- the two
    halves a restore applies differently."""
    processing = {k: v for k, v in state.items() if not k.startswith("meta.")}
    metadata = {k[len("meta."):]: v for k, v in state.items() if k.startswith("meta.")}
    return processing, metadata


# ============================================================
# Describing values and changes, for people
# ============================================================

from ..processing.report import WINDOW_NAMES  # noqa: E402 -- after the helpers it does not need

_BASELINE_METHODS = {
    "linear": "linear", "polynomial": "polynomial", "sinusoidal": "sinusoidal",
    "spline": "smoothing spline", "whittaker": "Whittaker (arPLS)",
    "spline_auto": "cubic spline (automatic points)",
    "spline_points": "cubic spline (points chosen on the plot)",
    "manual_polynomial": "manual Ax⁴+Bx³+Cx²+Dx+E", "manual_sine": "manual A·sin(Bx+C)+D",
}

#: Keys never shown in the log: bookkeeping that follows from a logged
#: setting (the SR originals are the axis before the logged SR shift),
#: or display plumbing. Still part of the state, so still restored.
_HIDDEN_KEYS = frozenset({
    "sr_original_hz", "sr_original_ppm", "sr_original_hz_f1", "sr_original_ppm_f1",
    "fit_view", "analysis_state.option_names", "analysis_state.peak_bounds",
    "analysis_state.model_key", "analysis_state.auto_regions_note",
    "analysis_state.peak_fixed_params", "relaxation_fit_context",
    # Facts the readers record about the file itself (re-read on every
    # load, never a user choice) -- listing them would read as if
    # something had been done.
    "fid_dt", "fid_dt_f1", "fid_original_length",
    # Zero-fill's own box value; the FFT size used is live_zero_fill.
    "live_zero_fill_choice", "fn_mode", "is_diffusion",
    "gradient_relative", "recorded_phase0", "recorded_phase1", "recorded_phase0_f1",
    "recorded_phase1_f1", "window_unsupported_f1", "xdim_f1", "xdim_f2",
    "f1_quadrature_sign_flip", "origin", "origin_vendor", "dataset_label",
    # Kept fit models: logged by their own events (kept / renamed /
    # deleted, with the fit's report), not as a diff of encoded curves.
    "fit_models",
    # An Open experiment made from a kept fit: the fit it carries, and
    # how Compare draws it (logged by their own events).
    "fit_model", "fit_model_view", "fit_model_source",
})


def _num(value, decimals: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value == int(value) and abs(value) < 1e12:
        return str(int(value))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _onoff(value) -> str:
    return "on" if value else "off"


def _points(value) -> str:
    return f"{int(value)} points" if value not in (None, "") else "—"


def _window(value) -> str:
    if value is None:
        return "—"
    name, abbreviation = WINDOW_NAMES.get(value, (str(value), None))
    return f"{name} ({abbreviation})" if abbreviation else name


def _truncate(value) -> str:
    return "full FID" if not value else f"first {int(value)} points"


def _degrees(value) -> str:
    return f"{_num(value, 2)}°"


def _hz(value) -> str:
    return f"{_num(value, 3)} Hz"


def _grams(value) -> str:
    return f"{_num(value, 6)} g"


def _pseudo(value) -> str:
    return "a series (pseudo-2D)" if value else "a genuine 2D"


def _text(value) -> str:
    return f"“{value}”" if value not in (None, "") else "—"


def _list_numbers(value, unit: str = "") -> str:
    if not value:
        return "none"
    items = [_num(v, 3) for v in value]
    shown = ", ".join(items[:12]) + (f", … ({len(items)} in all)" if len(items) > 12 else "")
    return shown + unit


#: key -> (label, formatter). Order here is the order changes are
#: listed within one entry (processing order, then analysis).
KEY_INFO: dict[str, tuple[str, object]] = {
    "meta.pseudo_2d": ("Treated as", _pseudo),
    "magnitude_override": ("Magnitude mode", _onoff),
    "magnitude_mode": ("Magnitude mode", _onoff),
    "live_fid_shift": ("FID left-shift", _points),
    "live_align_echo": ("Echo alignment", _onoff),
    "live_echo_offset": ("Echo offset", _points),
    "live_truncate": ("Truncation (TDeff)", _truncate),
    "live_truncate_f1": ("Truncation F1", _truncate),
    "live_linear_predict_points": ("Linear prediction", lambda v: f"{int(v)} points" if v else "off"),
    "live_linear_predict_direction": ("Linear prediction direction", str),
    "live_linear_predict_order": ("Linear prediction order", lambda v: str(int(v))),
    "live_window": ("Apodization", _window),
    "live_lb": ("Line broadening (LB)", _hz),
    "live_gb": ("Gaussian factor (GB)", lambda v: _num(v, 4)),
    "live_ssb": ("Sine-bell shift (SSB)", lambda v: _num(v, 2)),
    "live_window_f1": ("Apodization F1", _window),
    "live_lb_f1": ("Line broadening F1", _hz),
    "live_ssb_f1": ("Sine-bell shift F1", lambda v: _num(v, 2)),
    "live_zero_fill": ("Zero-fill", _points),
    "live_zero_fill_f1": ("Zero-fill F1", _points),
    "phase0": ("Phase PH0", _degrees),
    "phase1": ("Phase PH1", _degrees),
    "phase0_f1": ("Phase PH0 (F1)", _degrees),
    "phase1_f1": ("Phase PH1 (F1)", _degrees),
    "slice_phase0": ("Per-slice PH0", None),
    "slice_phase1": ("Per-slice PH1", None),
    "baseline_correct": ("Baseline correction", _onoff),
    "baseline_method": ("Baseline method", lambda v: _BASELINE_METHODS.get(v, str(v))),
    "baseline_degree": ("Baseline polynomial degree", lambda v: str(int(v))),
    "baseline_spline_x": ("Baseline spline points", lambda v: (
        ", ".join(f"{float(p):.4g}" for p in v) + " ppm") if v else "none"),
    "baseline_lam": ("Baseline smoothing (λ)", lambda v: _num(v, 4)),
    "baseline_smoothing": ("Baseline smoothing", lambda v: _num(v, 4)),
    "baseline_period": ("Baseline period", lambda v: _num(v, 4)),
    "baseline_harmonics": ("Baseline harmonics", lambda v: str(v)),
    "baseline_points": ("Baseline spline points", lambda v: "auto" if not v else str(int(v))),
    "baseline_manual_regions": ("Baseline regions", lambda v: "; ".join(
        f"{_num(min(r), 4)} – {_num(max(r), 4)} ppm" for r in v or [])),
    "baseline_manual_polynomial_params": ("Baseline A, B, C, D, E", lambda v: ", ".join(
        f"{float(x):.6g}" for x in v or [])),
    "baseline_manual_sine_params": ("Baseline A, B, C, D", lambda v: ", ".join(
        f"{float(x):.6g}" for x in v or [])),
    "baseline_manual_polynomial_fixed": ("Baseline fixed coefficients", lambda v: ", ".join(
        n for n, f in zip("ABCDE", v or []) if f) or "none"),
    "baseline_manual_sine_fixed": ("Baseline fixed coefficients", lambda v: ", ".join(
        n for n, f in zip("ABCD", v or []) if f) or "none"),
    "baseline_manual_refit": ("Baseline refitted per trace", _onoff),
    "drift_correction": ("Field drift correction", lambda v: (
        "off" if not v else
        f"{float(v.get('start_hz', 0)):+.4f} → {float(v.get('end_hz', 0)):+.4f} Hz over "
        f"{v.get('n_points', '?')} points (linear; Najbauer & Andreas, J. Magn. Reson. 305 (2019) 1–4)")),
    "sr_shift_hz": ("Referencing (SR shift)", _hz),
    "sr_shift_hz_f1": ("Referencing F1 (SR shift)", _hz),
    "meta.nucleus": ("Nucleus", str),
    "meta.nucleus_indirect": ("Nucleus F1", str),
    "meta.mass_g": ("Sample mass", _grams),
    "display_label": ("Name", _text),
    "notes": ("Title/notes", _text),
    "excluded_slices": ("Excluded slices", lambda v: _list_numbers([int(x) + 1 for x in v or []])),
    "imaging_gradient_percent": ("Imaging gradient", lambda v: f"{_num(v, 3)} %"),
    "contour_slices": ("Marked contour slices", lambda v: f"{len(v or [])}"),
    "nq_line": ("NQ line", str),
    "nq_value": ("NQ value", lambda v: _num(v, 4)),
    # Analysis (the Fit / Curve-fit panels' per-spectrum state).
    "analysis_state.peaks": ("Peaks", None),
    "analysis_state.regions": ("Integration regions", None),
    "analysis_state.integration_method": ("Integration rule", lambda v: {
        "trapezoid": "trapezoid", "simpson": "Simpson"}.get(v, str(v))),
    "analysis_state.auto_region_threshold": ("Automatic-region threshold", lambda v: _num(v, 4)),
    "analysis_state.auto_region_mode": ("Automatic-region threshold in", lambda v: {
        "noise": "× noise σ", "fraction": "% of maximum"}.get(v, str(v))),
    "analysis_state.fit_within_regions": ("Fit within regions", _onoff),
    "analysis_state.amplitude_bound_regions": ("Bound amplitudes to regions", _onoff),
    "analysis_state.fit_type": ("Curve-fit type", None),
    "analysis_state.model": ("Curve-fit model", None),
    "analysis_state.custom_equation": ("Custom equation", _text),
    "analysis_state.custom_p0": ("Custom equation start values", _text),
    "analysis_state.value_source": ("Curve-fit values from", None),
    "analysis_state.delay_source": ("Delays from", None),
    "analysis_state.seed_mode": ("Slice seeding", None),
    "analysis_state.manual_text": ("Manual delays", _text),
    "analysis_state.list_file": ("Delay list file", _text),
    "analysis_state.list_factor": ("Delay list factor", _text),
    "analysis_state.n_components": ("Diffusion components", str),
    "analysis_state.stretched": ("Stretched exponential", _onoff),
    "analysis_state.global_fit": ("Global fit", _onoff),
    "analysis_state.shared_params": ("Shared parameters", _text),
    "analysis_state.results_text": ("Fit results", "block"),
    "analysis_state.integration_results_text": ("Integration results", "block"),
    "analysis_state.score_results_text": ("SCORE results", "block"),
    "analysis_state.curve_fit_summary_text": ("Curve-fit results", "block"),
    "analysis_state.dosy_results_text": ("DOSY results", "block"),
    "analysis_state.gradient_calibration_text": ("Gradient calibration", "block"),
    "simulation": ("Simulation parameters", "params"),
    "relaxation_fit_results": ("Relaxation/diffusion fit", "result"),
    "slice_fit_results": ("Per-slice spectral fit", "result"),
    "score_result": ("SCORE decomposition", "result"),
    "dosy_result": ("DOSY transform", "result"),
    "enmr_result": ("eNMR analysis", "result"),
}

#: analysis_state keys that are combo indices; their readable text is
#: captured alongside in ``analysis_state.option_names``.
_OPTION_KEYS = ("fit_type", "model", "value_source", "delay_source", "seed_mode")

_ORDER = {key: i for i, key in enumerate(KEY_INFO)}

#: The Curve Fitting panel's settings: only worth summarising once a
#: curve has actually been fitted with them.
_CURVE_FIT_KEYS = frozenset({
    "fit_type", "model", "custom_equation", "custom_p0", "value_source", "delay_source",
    "seed_mode", "manual_text", "list_file", "list_factor", "n_components", "stretched",
    "global_fit", "shared_params",
})


def _strip_markup(text: str) -> str:
    """Rich-text label contents as plain text."""
    text = re.sub(r"<br\s*/?>|</p>|</tr>|</div>", "\n", str(text), flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _flatten(state: dict) -> dict:
    """``analysis_state`` (and its ``pfg`` block) flattened to dotted
    keys, so a change to one fit setting is one line, not "analysis
    state changed"."""
    flat = {}
    for key, value in state.items():
        if key == "analysis_state" and isinstance(value, dict):
            for sub, sub_value in value.items():
                if sub == "pfg" and isinstance(sub_value, dict):
                    for pfg_key, pfg_value in sub_value.items():
                        flat[f"analysis_state.pfg.{pfg_key}"] = pfg_value
                else:
                    flat[f"analysis_state.{sub}"] = sub_value
        elif key == "fits" and isinstance(value, dict):
            for panel, fit in value.items():
                flat[f"fits.{panel}"] = fit
        else:
            flat[key] = value
    return flat


def _fit_lines(fit, with_settings: bool = True) -> list[str]:
    """A fit record (see the GUI's ``_store_fit``) written out: what was
    fitted, the result, and the settings it was run with."""
    if not isinstance(fit, dict):
        return []
    head = f"{fit.get('label', 'Fit')} fit"
    if fit.get("target"):
        head += f" of {fit['target']}"
    lines = [head + ":"]
    report = [line.rstrip() for line in _strip_markup(fit.get("report") or "").splitlines() if line.strip()]
    if report:
        lines.append("    Result:")
        lines += ["      " + line for line in report[:60]]
        if len(report) > 60:
            lines.append(f"      … ({len(report) - 60} more lines)")
    settings = fit.get("inputs_text") or []
    if with_settings and settings:
        lines.append("    Settings (sites are the starting values):")
        lines += ["      " + str(line) for line in settings[:80]]
    return lines


def _label(key: str) -> str:
    if key in KEY_INFO:
        return KEY_INFO[key][0]
    if key.startswith("analysis_state.pfg."):
        return f"PFG {key.rsplit('.', 1)[-1]}"
    if key.startswith("analysis_state."):
        return key.split(".", 1)[1].replace("_", " ").capitalize()
    if key.startswith("fits."):
        return f"{key.split('.', 1)[1].replace('_', ' ').capitalize()} fit"
    return key


def _generic(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return _onoff(value)
    if isinstance(value, (int, float)):
        return _num(value, 4)
    if isinstance(value, str):
        return _text(value) if len(value) <= 80 else _text(value[:77] + "…")
    if isinstance(value, (list, dict)):
        text = json.dumps(value)
        return text if len(text) <= 80 else f"{type(value).__name__} of {len(value)} items"
    return str(value)


def format_value(key: str, value, state: dict | None = None) -> str:
    """One setting's value as a person would write it."""
    info = KEY_INFO.get(key)
    formatter = info[1] if info else None
    if value is None:
        return "—"
    if key.startswith("analysis_state.") and key.split(".", 1)[1] in _OPTION_KEYS:
        names = (state or {}).get("analysis_state.option_names") or {}
        name = names.get(key.split(".", 1)[1]) if isinstance(names, dict) else None
        return _text(name) if name else f"option {int(value) + 1}"
    if key == "analysis_state.peaks":
        shifts = [row[0] for row in value or [] if row and str(row[0]).strip()]
        return f"{len(value or [])} ({', '.join(shifts[:10])}{'…' if len(shifts) > 10 else ''} ppm)" if shifts else f"{len(value or [])}"
    if key == "analysis_state.regions":
        spans = [f"{row[0]}–{row[1]}" for row in value or [] if len(row) >= 2]
        return f"{len(spans)} ({', '.join(spans[:8])}{'…' if len(spans) > 8 else ''} ppm)" if spans else "0"
    if callable(formatter):
        try:
            return formatter(value)
        except (TypeError, ValueError):
            return _generic(value)
    return _generic(value)


def _is_blank(value) -> bool:
    return value in (None, "", [], {}, False, 0, 0.0)


def _slice_phase_lines(key: str, old, new) -> list[str]:
    old = list(old or [])
    new = list(new or [])
    length = max(len(old), len(new))
    changed = []
    for row in range(length):
        a = float(old[row]) if row < len(old) and old[row] is not None else 0.0
        b = float(new[row]) if row < len(new) and new[row] is not None else 0.0
        if a != b:
            changed.append(f"slice {row + 1}: {_degrees(a)} → {_degrees(b)}")
    label = _label(key)
    if not changed:
        return []
    if len(changed) > 8:
        return [f"{label}: {len(changed)} slices changed ({'; '.join(changed[:8])}; …)"]
    return [f"{label}: {'; '.join(changed)}"]


def _param_lines(params, limit: int = 160) -> list[str]:
    """A parameter dict (a simulation's inputs) one entry per line."""
    if not isinstance(params, dict):
        return [_generic(params)]
    lines = []
    for key, value in params.items():
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, default=_json_default)
            if len(text) > limit:
                text = text[: limit - 1] + "…"
        else:
            text = _num(value, 6) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
        lines.append(f"{key}: {text}")
    return lines


def describe_change(key: str, old, new, old_state: dict | None = None,
                    new_state: dict | None = None) -> list[str]:
    """Human-readable line(s) for one setting changing ``old`` → ``new``.
    Multi-line results come back as a header line then indented text."""
    info = KEY_INFO.get(key)
    kind = info[1] if info else None
    label = _label(key)
    if key in ("slice_phase0", "slice_phase1"):
        return _slice_phase_lines(key, old, new)
    if key.startswith("fits."):
        if not isinstance(new, dict):
            return [f"{label}: removed"]
        return _fit_lines(new)
    if kind == "block":
        text = _strip_markup(new or "")
        if not text:
            return [f"{label}: cleared"]
        body = [line.rstrip() for line in text.splitlines() if line.strip()]
        if len(body) > 40:
            body = body[:40] + [f"… ({len(body) - 40} more lines)"]
        return [f"{label}:"] + ["    " + line for line in body]
    if kind == "params":
        return [f"{label}:"] + ["    " + line for line in _param_lines(new)]
    if kind == "result":
        if _is_blank(new):
            return [f"{label}: cleared"]
        return [f"{label}: {'updated' if not _is_blank(old) else 'computed'}"]
    old_text = format_value(key, old, old_state)
    new_text = format_value(key, new, new_state)
    if old is None:
        return [f"{label}: {new_text}"]
    if new is None:
        return [f"{label}: {old_text} → (removed)"]
    return [f"{label}: {old_text} → {new_text}"]


def diff_states(old: dict, new: dict) -> list[dict]:
    """Every setting that differs between two tracked states, as
    ``{"key", "old", "new"}`` dicts in processing order. A setting that
    appears for the first time at a blank value (``False``, ``0``,
    empty) is not a change anyone made and is skipped."""
    old_flat = _flatten(old or {})
    new_flat = _flatten(new or {})
    changes = []
    for key in sorted(set(old_flat) | set(new_flat), key=lambda k: (_ORDER.get(k, len(_ORDER)), k)):
        a = old_flat.get(key)
        b = new_flat.get(key)
        if a == b:
            continue
        if key not in old_flat and _is_blank(b):
            continue
        if key not in new_flat and _is_blank(a):
            continue
        changes.append({"key": key, "old": copy.deepcopy(a), "new": copy.deepcopy(b)})
    return changes


def _visible(change: dict) -> bool:
    return change["key"] not in _HIDDEN_KEYS


def change_lines(changes: list[dict], old_state: dict | None = None,
                 new_state: dict | None = None) -> list[str]:
    old_flat = _flatten(old_state or {})
    new_flat = _flatten(new_state or {})
    lines = []
    for change in changes:
        if not _visible(change):
            continue
        lines += describe_change(change["key"], change["old"], change["new"], old_flat, new_flat)
    return lines


def _relevant(key: str, flat: dict, source: dict | None) -> bool:
    """Whether a setting means anything given the others -- LB with no
    exponential/Gaussian window, a linear-prediction order with none
    applied, a baseline method with correction off are all left out of
    a summary (they are still in the state, and still restored)."""
    window = flat.get("live_window")
    window_f1 = flat.get("live_window_f1")
    rules = {
        "live_lb": window in ("exponential", "gaussian"),
        "live_gb": window == "gaussian",
        "live_ssb": window in ("sine", "qsine"),
        "live_lb_f1": window_f1 in ("exponential", "gaussian"),
        "live_ssb_f1": window_f1 in ("sine", "qsine"),
        "live_linear_predict_direction": bool(flat.get("live_linear_predict_points")),
        "live_linear_predict_order": bool(flat.get("live_linear_predict_points")),
        "live_echo_offset": bool(flat.get("live_align_echo")),
        "live_window_f1": window_f1 not in (None, "none"),
    }
    if key in rules:
        return rules[key]
    if key in ("analysis_state.fit_within_regions", "analysis_state.amplitude_bound_regions",
               "analysis_state.integration_method"):
        return bool(flat.get("analysis_state.regions"))
    if key in ("analysis_state.auto_region_threshold", "analysis_state.auto_region_mode"):
        return bool(flat.get("analysis_state.auto_regions_note"))
    if key.startswith("analysis_state.") and key.split(".", 1)[1] in _CURVE_FIT_KEYS:
        return bool(flat.get("relaxation_fit_results") or flat.get("analysis_state.curve_fit_summary_text"))
    if key.startswith("baseline_") and key != "baseline_correct":
        method = flat.get("baseline_method", "linear")
        needs = {"baseline_degree": "polynomial", "baseline_lam": "whittaker",
                 "baseline_smoothing": "spline", "baseline_period": "sinusoidal",
                 "baseline_harmonics": "sinusoidal",
                 "baseline_points": "spline_auto",
                 "baseline_manual_polynomial_params": "manual_polynomial",
                 "baseline_manual_sine_params": "manual_sine",
                 "baseline_manual_polynomial_fixed": "manual_polynomial",
                 "baseline_manual_sine_fixed": "manual_sine"}
        if key in ("baseline_manual_regions", "baseline_manual_refit"):
            return bool(flat.get("baseline_correct")) and str(method).startswith("manual_")
        return bool(flat.get("baseline_correct")) and (key == "baseline_method" or needs.get(key) == method)
    if key == "live_truncate":
        full = flat.get("fid_original_length")
        return bool(flat.get(key)) and not (full and int(flat[key]) >= int(full))
    if key in ("meta.nucleus", "meta.nucleus_indirect"):
        recorded = (source or {}).get("nucleus") if key == "meta.nucleus" else None
        return key == "meta.nucleus_indirect" or flat.get(key) != recorded
    return True


def state_summary(state: dict, source: dict | None = None) -> list[str]:
    """The settings in ``state`` that mean something, one line each --
    what a save point records, written out. Blank settings, settings
    that do nothing given the others (see :func:`_relevant`) and
    settings openNMR does not name are omitted here; they are still
    part of the saved state and are restored with it. ``source`` (the
    history's own record of the file) lets a nucleus that is simply the
    file's own be left out."""
    flat = _flatten(state or {})
    lines = []
    for key in sorted(flat, key=lambda k: (_ORDER.get(k, len(_ORDER)), k)):
        value = flat[key]
        if key.startswith("fits.") and isinstance(value, dict):
            lines += _fit_lines(value, with_settings=False)
            continue
        if key in _HIDDEN_KEYS or key not in KEY_INFO:
            continue
        if _is_blank(value) or not _relevant(key, flat, source):
            continue
        info = KEY_INFO.get(key)
        kind = info[1] if info else None
        if key in ("slice_phase0", "slice_phase1"):
            nonzero = [f"{i + 1}: {_degrees(v)}" for i, v in enumerate(value or []) if v]
            if nonzero:
                lines.append(f"{_label(key)}: " + "; ".join(nonzero[:12]) + ("; …" if len(nonzero) > 12 else ""))
            continue
        if kind in ("block", "params"):
            lines += describe_change(key, None, value)
            continue
        if kind == "result":
            lines.append(f"{_label(key)}: present")
            continue
        lines.append(f"{_label(key)}: {format_value(key, value, flat)}")
    return lines


# ============================================================
# The history record
# ============================================================

def _now() -> _dt.datetime:
    return _dt.datetime.now().astimezone()


def _stamp(when: _dt.datetime | None) -> str:
    return (when or _now()).isoformat(timespec="seconds")


def _parse(stamp: str) -> _dt.datetime | None:
    try:
        return _dt.datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None


def describe_source(spectrum) -> dict:
    meta = spectrum.metadata
    return {
        "source_path": meta.source_path,
        "vendor": meta.vendor,
        "nucleus": meta.nucleus,
        "procno": meta.procno,
        "ndim": int(spectrum.ndim),
        "shape": list(np.shape(spectrum.data)),
    }


def new_history(spectrum, state: dict | None = None) -> dict:
    """An empty history for ``spectrum`` whose starting point is
    ``state`` (default: its current tracked state)."""
    return {
        "opennmr_history_version": HISTORY_VERSION,
        "source": describe_source(spectrum),
        "state": _normalise(state if state is not None else tracked_state(spectrum)),
        "entries": [],
        "next_savepoint": 1,
    }


def load_history(source_path: str | None) -> dict | None:
    """The saved history of ``source_path``, or ``None`` (none yet,
    unreadable, or not a history file)."""
    if not source_path:
        return None
    path = history_path(source_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        return None
    payload.setdefault("state", {})
    payload.setdefault("next_savepoint", 1 + max(
        [e.get("id", 0) for e in payload["entries"] if e.get("kind") == "savepoint"] or [0]))
    return payload


def add_event(history: dict, text: str, kind: str = "event", when=None, **extra) -> dict:
    """Append a free-text entry (opened, reset, restored, created...)."""
    entry = {"kind": kind, "time": _stamp(when), "text": str(text), **extra}
    history["entries"].append(entry)
    return entry


#: A changed value bigger than this (as JSON) is recorded in a change
#: entry by its fingerprint, not in full: a DOSY map or a per-slice fit
#: table would otherwise be copied into the history on every edit. The
#: full value is still in the current state and in every save point.
_LARGE_VALUE_CHARS = 20000


def _compact(key: str, value):
    if key.startswith("fits.") and isinstance(value, dict):
        # What the log shows of a fit; its inputs and full result stay
        # in the state and the save points.
        return {k: copy.deepcopy(value[k]) for k in ("label", "target", "time", "report", "inputs_text")
                if k in value}
    if value is None or (KEY_INFO.get(key, (None, None))[1] == "block"):
        return copy.deepcopy(value)
    text = json.dumps(value, default=_json_default, sort_keys=True)
    if len(text) <= _LARGE_VALUE_CHARS:
        return copy.deepcopy(value)
    import hashlib
    return {"_digest": hashlib.sha1(text.encode()).hexdigest()[:16], "_size": len(text)}


def _option_names(state: dict | None) -> dict:
    names = ((state or {}).get("analysis_state") or {}).get("option_names")
    return copy.deepcopy(names) if isinstance(names, dict) else {}


def _names_state(names: dict | None) -> dict:
    return {"analysis_state": {"option_names": names or {}}}


def record_state(history: dict, state: dict, when=None,
                 merge_window_s: float = MERGE_WINDOW_S) -> list[dict]:
    """Compare ``state`` with the last recorded one, append what changed
    (merged into the newest change entry where it continues a burst of
    edits), and make ``state`` the new reference. Returns the changes
    found (``[]`` when nothing changed)."""
    state = _normalise(state)
    previous = history.get("state") or {}
    changes = diff_states(previous, state)
    history["state"] = state
    if not changes:
        return []
    when = when or _now()
    last = history["entries"][-1] if history["entries"] else None
    last_time = _parse(last.get("updated") or last.get("time")) if last else None
    compact = [{"key": c["key"], "old": _compact(c["key"], c["old"]), "new": _compact(c["key"], c["new"])}
               for c in changes]
    if (last is not None and last.get("kind") == "change" and last_time is not None
            and 0 <= (when - last_time).total_seconds() <= merge_window_s):
        by_key = {c["key"]: c for c in last["changes"]}
        for change in compact:
            if change["key"] in by_key:
                by_key[change["key"]]["new"] = change["new"]
            else:
                last["changes"].append(change)
                by_key[change["key"]] = change
        last["changes"] = [c for c in last["changes"] if c["old"] != c["new"]]
        last["changes"].sort(key=lambda c: (_ORDER.get(c["key"], len(_ORDER)), c["key"]))
        last["updated"] = _stamp(when)
        last["names_after"] = _option_names(state)
        if not last["changes"]:
            history["entries"].pop()
        return changes
    history["entries"].append({
        "kind": "change", "time": _stamp(when), "changes": compact,
        "names_before": _option_names(previous), "names_after": _option_names(state),
    })
    return changes


def add_savepoint(history: dict, state: dict, name: str = "", note: str = "",
                  when=None, snapshot: str | None = None) -> dict:
    """Record a named save point holding the full ``state``. Records any
    pending change first, so the log shows how the state got here."""
    record_state(history, state, when=when)
    number = int(history.get("next_savepoint", 1))
    history["next_savepoint"] = number + 1
    entry = {
        "kind": "savepoint", "id": number, "time": _stamp(when),
        "name": str(name or "").strip() or f"Save point {number}",
        "note": str(note or "").strip(), "state": _normalise(state),
    }
    if snapshot:
        entry["snapshot"] = snapshot
    history["entries"].append(entry)
    return entry


def savepoints(history: dict | None) -> list[dict]:
    """Every save point, oldest first."""
    return [e for e in (history or {}).get("entries", []) if e.get("kind") == "savepoint"]


def find_savepoint(history: dict | None, which) -> dict:
    """A save point by number, or by name (the newest one with that
    name). Raises ``KeyError`` if there is none."""
    points = savepoints(history)
    for entry in reversed(points):
        if (isinstance(which, int) and entry.get("id") == which) or (
                isinstance(which, str) and entry.get("name") == which):
            return entry
    raise KeyError(f"No save point {which!r}. Available: "
                   f"{[(e['id'], e['name']) for e in points] or 'none'}")


# ============================================================
# The human-readable log
# ============================================================

_RULE = "-" * 72


def _day_heading(when: _dt.datetime) -> str:
    offset = when.strftime("%z")
    zone = f" (UTC{offset[:3]}:{offset[3:]})" if offset else ""
    return when.strftime("%A %d %B %Y") + zone


def render_log(history: dict, title: str | None = None) -> str:
    """The whole history written out for people."""
    source = history.get("source") or {}
    lines = ["openNMR processing log", "=" * 22, ""]
    if title:
        lines.append(f"Spectrum    : {title}")
    if source.get("source_path"):
        lines.append(f"Data        : {source['source_path']}")
    bits = [str(source[k]) for k in ("vendor", "nucleus") if source.get(k)]
    if source.get("procno") is not None:
        bits.append(f"procno {source['procno']}")
    if source.get("shape"):
        bits.append(" × ".join(str(n) for n in source["shape"]) + " points")
    if bits:
        lines.append(f"Format      : {', '.join(bits)}")
    lines += [
        "",
        "Written by openNMR from opennmr_history.json after every change. Edits",
        "made to this text file are not read back – use Load Progress in",
        "openNMR (or opennmr.notebook.load_progress) to return to a save point.",
        "Times are local; each day's heading gives the offset from UTC. Phase",
        "values are openNMR's own correction, applied on top of any phase the",
        "data files record (e.g. TopSpin's PHC0/PHC1).",
        "",
    ]
    current_day = None
    for entry in history.get("entries", []):
        when = _parse(entry.get("time"))
        day = _day_heading(when) if when else "(no date)"
        if day != current_day:
            lines += [_RULE, day, _RULE]
            current_day = day
        clock = when.strftime("%H:%M:%S") if when else "--:--:--"
        updated = _parse(entry.get("updated")) if entry.get("updated") else None
        if updated and when and updated.strftime("%H:%M:%S") != clock:
            clock = f"{clock}–{updated.strftime('%H:%M:%S')}"
        kind = entry.get("kind")
        if kind == "change":
            body = change_lines(entry.get("changes", []), _names_state(entry.get("names_before")),
                                _names_state(entry.get("names_after")))
            if not body:
                continue
            lines.append(f"{clock}  Changed")
            lines += [f"    {line}" for line in body]
        elif kind == "savepoint":
            lines.append(f"{clock}  ★ SAVE POINT {entry.get('id')}: {entry.get('name')}")
            if entry.get("note"):
                lines += [f"    Note: {line}" for line in entry["note"].splitlines()]
            summary = state_summary(entry.get("state") or {}, source)
            lines.append("    State at this point:" if summary else "    State at this point: as loaded (nothing changed)")
            lines += [f"      {line}" for line in summary]
            if entry.get("snapshot"):
                lines.append(f"    Data snapshot: {entry['snapshot']}")
        else:
            text_lines = str(entry.get("text", "")).splitlines() or [""]
            lines.append(f"{clock}  {text_lines[0]}")
            lines += [f"    {line}" for line in text_lines[1:]]
        lines.append("")
    if not history.get("entries"):
        lines.append("(Nothing recorded yet.)")
    lines += [_RULE, "Current state:"]
    summary = state_summary(history.get("state") or {}, source)
    lines += [f"  {line}" for line in summary] or ["  as loaded (nothing changed)"]
    return "\n".join(lines).rstrip() + "\n"


# ============================================================
# Writing
# ============================================================

def write_history(source_path: str, history: dict, title: str | None = None) -> tuple[str, str]:
    """Write ``history`` as ``opennmr_history.json`` and regenerate
    ``opennmr_log.txt`` next to ``source_path``. Returns both paths.
    Raises ``OSError`` if the folder cannot be written (read-only data,
    an offline cloud folder); the caller decides how to tell the user.
    Atomic, like the other sidecars."""
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError(f"No such experiment: {source_path!r}")
    json_path = history_path(source_path)
    text_path = log_path(source_path)
    history["source"] = {**(history.get("source") or {}), "source_path": source_path}
    # Keys starting with "_" are session-only (the GUI's control
    # defaults, say) and never written.
    record = {key: value for key, value in history.items() if not key.startswith("_")}
    for path, content in ((json_path, json.dumps(record, indent=1, default=_json_default, ensure_ascii=False)),
                          (text_path, render_log(history, title=title))):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp, path)
    return json_path, text_path


# ============================================================
# Data snapshots and openNMR dataset folders
# ============================================================

def _arrays(spectrum) -> dict:
    arrays = {"data": np.asarray(spectrum.data)}
    if spectrum.data_imag is not None:
        arrays["data_imag"] = np.asarray(spectrum.data_imag)
    for i, axis in enumerate(spectrum.axes):
        for part in ("ppm", "hz", "index"):
            values = getattr(axis, part, None)
            if values is not None:
                arrays[f"axis{i}_{part}"] = np.asarray(values)
    return arrays


def _axes_info(spectrum) -> list[dict]:
    return [{"kind": axis.kind, "label": axis.label} for axis in spectrum.axes]


def write_snapshot(path: str, spectrum) -> str:
    """Save ``spectrum``'s data and axes exactly (``.npz``, full
    precision) to ``path``. Returns ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, axes_json=np.array(json.dumps(_axes_info(spectrum))), **_arrays(spectrum))
    os.replace(tmp, path)
    return path


def read_snapshot(path: str):
    """``(data, data_imag, axes)`` from :func:`write_snapshot`."""
    from ..core.spectrum import NMRAxis

    with np.load(path, allow_pickle=False) as npz:
        axes_info = json.loads(str(npz["axes_json"]))
        data = np.array(npz["data"])
        data_imag = np.array(npz["data_imag"]) if "data_imag" in npz.files else None
        axes = []
        for i, info in enumerate(axes_info):
            parts = {part: np.array(npz[f"axis{i}_{part}"]) for part in ("ppm", "hz", "index")
                     if f"axis{i}_{part}" in npz.files}
            axes.append(NMRAxis(kind=info["kind"], label=info.get("label"), **parts))
    return data, data_imag, axes


def is_dataset_folder(path: str) -> bool:
    """Whether ``path`` is an openNMR dataset folder."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, DATASET_DATA_FILE)) \
        and os.path.exists(os.path.join(path, DATASET_INFO_FILE))


_SAFE = re.compile(r"[^A-Za-z0-9._ ()+-]+")


def safe_folder_name(label: str) -> str:
    """``label`` made safe as a folder name on Windows, macOS and Linux."""
    name = _SAFE.sub("_", str(label)).strip(" ._") or "spectrum"
    return name[:80]


_METADATA_FIELDS = ("nucleus", "nucleus_indirect", "sfo1", "sfo1_indirect", "bf1", "o1", "ns",
                    "procno", "pseudo_2d", "homonuclear", "mass", "probe", "pulse_program")


def save_spectrum_folder(spectrum, folder: str, label: str | None = None,
                         origin: str | None = None, origin_vendor: str | None = None) -> str:
    """Write ``spectrum`` as a self-contained openNMR dataset folder --
    for a simulation or derived spectrum, which has no data folder of
    its own. Holds the data exactly (``opennmr_spectrum.npz``), what it
    is (``opennmr_spectrum.json``: nucleus, frequencies, where it came
    from) and its processing state (the usual sidecars); the history
    and log are written by the caller. Reopened with :func:`opennmr.io.load`
    like any experiment. Returns ``folder``."""
    os.makedirs(folder, exist_ok=True)
    meta = spectrum.metadata
    info = {
        "opennmr_dataset_version": 1,
        "label": label,
        "origin_vendor": origin_vendor or (meta.raw.get("origin_vendor") if meta.vendor == "opennmr" else meta.vendor),
        "origin": origin or meta.raw.get("origin"),
        "saved": _stamp(None),
        "metadata": {field: _normalise(getattr(meta, field)) for field in _METADATA_FIELDS},
    }
    tmp = os.path.join(folder, DATASET_INFO_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(folder, DATASET_INFO_FILE))
    write_snapshot(os.path.join(folder, DATASET_DATA_FILE), spectrum)
    return folder


def read_spectrum_folder(path: str, **_ignored):
    """Read an openNMR dataset folder (see :func:`save_spectrum_folder`)
    as an :class:`~opennmr.core.spectrum.NMRSpectrum` with vendor
    ``"opennmr"``. Its saved processing state is *not* merged here --
    the data were saved already processed; the GUI merges the state as
    for any experiment, and :func:`opennmr.notebook.load_saved_spectrum`
    does so for a notebook."""
    from ..core.spectrum import NMRMetadata, NMRSpectrum

    with open(os.path.join(path, DATASET_INFO_FILE), encoding="utf-8") as handle:
        info = json.load(handle)
    data, data_imag, axes = read_snapshot(os.path.join(path, DATASET_DATA_FILE))
    fields = {k: v for k, v in (info.get("metadata") or {}).items() if k in _METADATA_FIELDS}
    raw = {"origin_vendor": info.get("origin_vendor"), "origin": info.get("origin")}
    if info.get("label"):
        raw["dataset_label"] = info["label"]
    metadata = NMRMetadata(vendor="opennmr", source_path=path, raw=raw, **fields)
    return NMRSpectrum(data=data, axes=axes, metadata=metadata, data_imag=data_imag)


def _peek_spectrum_folder(path: str, **_ignored) -> dict:
    with open(os.path.join(path, DATASET_INFO_FILE), encoding="utf-8") as handle:
        info = json.load(handle)
    meta = info.get("metadata") or {}
    with np.load(os.path.join(path, DATASET_DATA_FILE), allow_pickle=False) as npz:
        shape = tuple(npz["data"].shape)
    return {"vendor": "opennmr", "ndim": len(shape), "shape": shape,
            "nucleus": meta.get("nucleus"), "nucleus_indirect": meta.get("nucleus_indirect"),
            "pseudo_2d": bool(meta.get("pseudo_2d"))}


def default_simulation_root() -> str:
    """Where simulations are saved unless the user picks elsewhere:
    ``~/Documents/openNMR/simulations``, or the folder named by the
    ``OPENNMR_SIMULATIONS_DIR`` environment variable."""
    override = os.environ.get("OPENNMR_SIMULATIONS_DIR")
    if override:
        return os.path.expanduser(override)
    home = os.path.expanduser("~")
    documents = os.path.join(home, "Documents")
    return os.path.join(documents if os.path.isdir(documents) else home, "openNMR", "simulations")


def default_dataset_folder(spectrum, label: str, parent_path: str | None = None,
                           when: _dt.datetime | None = None) -> str:
    """The suggested folder for saving a spectrum with nothing on disk.

    * Derived from a real experiment (a slice, a CPMG sum, a VOCS
      correction, a fit): inside that experiment's folder, under
      ``opennmr_derived/`` -- it stays with the data it came from.
    * Anything else (a simulation): a dated folder under
      :func:`default_simulation_root`.
    """
    stamp = (when or _now()).strftime("%Y-%m-%d_%H%M%S")
    name = safe_folder_name(label)
    if parent_path and os.path.isdir(parent_path):
        return os.path.join(parent_path, "opennmr_derived", name)
    if parent_path and os.path.isfile(parent_path):
        return os.path.join(os.path.dirname(parent_path), "opennmr_derived", name)
    return os.path.join(default_simulation_root(), f"{stamp}_{name}")


def unique_folder(folder: str) -> str:
    """``folder``, or ``folder (2)``, ``(3)``... -- the first that does
    not exist yet, so saving never overwrites another dataset."""
    if not os.path.exists(folder):
        return folder
    n = 2
    while os.path.exists(f"{folder} ({n})"):
        n += 1
    return f"{folder} ({n})"


# ============================================================
# The simulation journal
# ============================================================
#
# Simulations belong to no experiment, so their record is one journal in
# the simulations folder: every run of every Simulate panel, with the
# inputs it was run with (restorable) and what it produced (readable).

SIMULATION_JOURNAL_FILE = "opennmr_simulation_history.json"
SIMULATION_LOG_FILE = "opennmr_simulation_log.txt"


def simulation_journal_paths(folder: str | None = None) -> tuple[str, str]:
    folder = folder or default_simulation_root()
    return os.path.join(folder, SIMULATION_JOURNAL_FILE), os.path.join(folder, SIMULATION_LOG_FILE)


def load_simulation_journal(folder: str | None = None) -> dict:
    """The journal (``{"entries": [...], "next_id": n}``); empty if none
    has been written yet or it cannot be read."""
    path, _text = simulation_journal_paths(folder)
    try:
        with open(path, encoding="utf-8") as handle:
            journal = json.load(handle)
        if isinstance(journal, dict) and isinstance(journal.get("entries"), list):
            journal.setdefault("next_id", 1 + max([e.get("id", 0) for e in journal["entries"]] or [0]))
            return journal
    except (OSError, ValueError):
        pass
    return {"opennmr_simulation_journal_version": 1, "entries": [], "next_id": 1}


def record_simulation(panel: str, label: str, inputs: dict, inputs_text: list[str],
                      summary: str = "", folder: str | None = None, when=None,
                      merge_window_s: float = MERGE_WINDOW_S) -> dict:
    """Add a run to the journal and rewrite its text log. A re-run of
    the same panel within ``merge_window_s`` of the last entry replaces
    that entry's inputs and result (counted in ``runs``), so trying
    values one after another reads as one line of work, not dozens.
    Raises ``OSError`` if the folder cannot be written."""
    folder = folder or default_simulation_root()
    os.makedirs(folder, exist_ok=True)
    journal = load_simulation_journal(folder)
    when = when or _now()
    last = journal["entries"][-1] if journal["entries"] else None
    last_time = _parse(last.get("updated") or last.get("time")) if last else None
    body = {"panel": panel, "label": label, "inputs": inputs, "inputs_text": list(inputs_text),
            "summary": str(summary or "")}
    if (last is not None and last.get("panel") == panel and last_time is not None
            and 0 <= (when - last_time).total_seconds() <= merge_window_s):
        last.update(body)
        last["runs"] = int(last.get("runs", 1)) + 1
        last["updated"] = _stamp(when)
        entry = last
    else:
        entry = {"id": int(journal["next_id"]), "time": _stamp(when), "runs": 1, **body}
        journal["next_id"] = entry["id"] + 1
        journal["entries"].append(entry)
    write_simulation_journal(journal, folder)
    return entry


def write_simulation_journal(journal: dict, folder: str | None = None) -> tuple[str, str]:
    path, text_path = simulation_journal_paths(folder)
    for target, content in ((path, json.dumps(journal, indent=1, ensure_ascii=False, default=_json_default)),
                            (text_path, render_simulation_log(journal))):
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp, target)
    return path, text_path


def render_simulation_log(journal: dict) -> str:
    lines = ["openNMR simulation log", "=" * 22, "",
             "Every simulation run in openNMR, with its inputs. Written from",
             f"{SIMULATION_JOURNAL_FILE} after every run; edits here are not read back.",
             "use File ▸ Simulation History in openNMR to restore a run's inputs.", ""]
    current_day = None
    for entry in journal.get("entries", []):
        when = _parse(entry.get("time"))
        day = _day_heading(when) if when else "(no date)"
        if day != current_day:
            lines += [_RULE, day, _RULE]
            current_day = day
        clock = when.strftime("%H:%M:%S") if when else "--:--:--"
        updated = _parse(entry.get("updated")) if entry.get("updated") else None
        if updated and when and updated.strftime("%H:%M:%S") != clock:
            clock = f"{clock}–{updated.strftime('%H:%M:%S')}"
        runs = int(entry.get("runs", 1))
        lines.append(f"{clock}  #{entry.get('id')} {entry.get('label', entry.get('panel'))}"
                     + (f" (run {runs} times; the last run is shown)" if runs > 1 else ""))
        if entry.get("inputs_text"):
            lines.append("    Inputs:")
            lines += ["      " + str(line) for line in entry["inputs_text"]]
        summary = [line for line in _strip_markup(entry.get("summary") or "").splitlines() if line.strip()]
        if summary:
            lines.append("    Result:")
            lines += ["      " + line for line in summary]
        lines.append("")
    if not journal.get("entries"):
        lines.append("(No simulations yet.)")
    return "\n".join(lines).rstrip() + "\n"
