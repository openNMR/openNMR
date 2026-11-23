"""
Saved fit results: the per-slice spectral fits (and the "model applied to
every slice" fits) and the Dynamics curve fits (relaxation / diffusion /
any profile fit of a fitted peak area against delay or b-value).

Why this module exists (user report, 2026-11-23): a pseudo-2D series was
fitted slice by slice, the fit exported to every slice, and the peak
areas fitted against the delay list (Dynamics ▸ Fit Curve). After saving
an openNMR state, closing and reopening openNMR, none of that came back:
the entries in the experiment tree said "could not be shown again ... no
longer stored with this spectrum". The cause: these results live in
``metadata.raw["slice_fit_results"]`` / ``["relaxation_fit_results"]``
as dicts holding numpy arrays (and, per slice, an ``NMRSpectrum``), which
``json.dumps`` cannot write, and :func:`opennmr.io.persistence.processing_state`
silently skips any value it cannot write. So they were never saved -- not
in the processing sidecar, not in a state.

They are kept in a file of their own (gzip-compressed JSON, using the
exact encoding of :mod:`opennmr.io.serialize`) rather than inside the
processing JSON, because they are large (every slice's data, fit and
components) and the processing JSON is rewritten on nearly every control
tweak and compared on every history step.

What is written for each slice's ``"slice"`` trace: its axes and
metadata, *not* its data (nothing that shows or exports a fit reads the
data of that trace, only its axis, nucleus and SFO1), and the axes
shared by all slices are written once. On reading it comes back with a
zero-filled data array and ``metadata.raw["slice_data_not_stored"]``.

Qt-free; used by the GUI, :mod:`opennmr.io.persistence`,
:mod:`opennmr.io.state` and the notebook.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
import tempfile

import numpy as np

from . import serialize

#: The ``metadata.raw`` keys whose values are saved here.
RESULT_KEYS = ("slice_fit_results", "relaxation_fit_results")

FORMAT_VERSION = 1


# ----------------------------------------------------------------------
# The per-slice trace
# ----------------------------------------------------------------------

def _is_spectrum(value) -> bool:
    from ..core.spectrum import NMRSpectrum
    return isinstance(value, NMRSpectrum)


def _axis_record(axis) -> dict:
    return {f.name: serialize.encode(getattr(axis, f.name)) for f in dataclasses.fields(axis)}


def _metadata_record(metadata) -> dict:
    return {f.name: serialize.encode(getattr(metadata, f.name))
            for f in dataclasses.fields(metadata) if f.name != "raw"}


def _base_key(record: dict) -> str:
    return hashlib.sha1(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()


def _encode_slice(spectrum, bases: list, base_index: dict) -> dict:
    """``spectrum`` (one slice's own 1D trace) as a small record pointing
    at a shared base (its axes and metadata)."""
    base = {"axes": [_axis_record(a) for a in spectrum.axes],
            "metadata": _metadata_record(spectrum.metadata),
            "shape": list(spectrum.data.shape)}
    key = _base_key(base)
    if key not in base_index:
        base_index[key] = len(bases)
        bases.append(base)
    raw = spectrum.metadata.raw or {}
    return {"__slice__": base_index[key],
            "slice": serialize.encode(raw.get("slice")),
            "complex": spectrum.data_imag is not None}


def _decode_slice(record: dict, bases: list):
    from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    base = bases[record["__slice__"]]
    axes = [NMRAxis(**{k: serialize.decode(v) for k, v in a.items()}) for a in base["axes"]]
    fields = {k: serialize.decode(v) for k, v in base["metadata"].items()}
    raw = {"slice": serialize.decode(record.get("slice")), "slice_data_not_stored": True}
    metadata = NMRMetadata(**fields, raw=raw)
    shape = tuple(base["shape"])
    data = np.zeros(shape)
    return NMRSpectrum(data=data, axes=axes, metadata=metadata,
                       data_imag=np.zeros(shape) if record.get("complex") else None)


# ----------------------------------------------------------------------
# A curve fit's model (``result["spec"]``: holds functions)
# ----------------------------------------------------------------------

def _encode_spec(spec: dict):
    """A model spec as a recipe: the name of the registry entry it is, or
    how to build it again (``spec["rebuild"]``, which a typed equation and
    a generalised diffusion model carry). Anything else comes back
    without its functions (the fit's data and parameters are intact)."""
    from ..processing import relaxation
    for name, entry in relaxation.RELAXATION_MODELS.items():
        if entry is spec:
            return {"__spec__": "registry", "name": name}
    rebuild = spec.get("rebuild")
    if isinstance(rebuild, dict):
        return {"__spec__": "rebuild", "rebuild": serialize.encode(rebuild)}
    return {"__spec__": "unavailable",
            "params": serialize.encode(spec.get("params")),
            "rate_param": serialize.encode(spec.get("rate_param"))}


def _decode_spec(record: dict):
    from ..processing import diffusion, relaxation
    kind = record.get("__spec__")
    try:
        if kind == "registry":
            return relaxation.RELAXATION_MODELS[record["name"]]
        if kind == "rebuild":
            rebuild = serialize.decode(record["rebuild"])
            if rebuild.get("kind") == "equation":
                return relaxation.build_custom_model(
                    rebuild["equation"], p0_overrides=rebuild.get("p0_overrides"),
                    constants=rebuild.get("constants"))
            if rebuild.get("kind") == "diffusion":
                return diffusion.build_diffusion_model(
                    rebuild["n_components"], stretched=rebuild["stretched"],
                    d_bounds_factor=rebuild.get("d_bounds_factor", 100.0),
                    offset=rebuild.get("offset", False))
    except Exception:  # noqa: BLE001 -- the fit's numbers are still intact
        pass
    return {"params": serialize.decode(record.get("params")),
            "rate_param": serialize.decode(record.get("rate_param"))}


# ----------------------------------------------------------------------
# Encoding a list of results
# ----------------------------------------------------------------------

def encode_results(results) -> dict | None:
    """``results`` (a list of result dicts) as JSON-able data, or None
    when there is nothing to keep."""
    if not results:
        return None
    bases: list = []
    base_index: dict = {}
    out = []
    for result in results:
        if not isinstance(result, dict):
            out.append(serialize.encode(result))
            continue
        record = {}
        for key, value in result.items():
            if _is_spectrum(value):
                record[key] = _encode_slice(value, bases, base_index)
            elif key == "spec" and isinstance(value, dict) and callable(value.get("func")):
                record[key] = _encode_spec(value)
            else:
                record[key] = serialize.encode(value)
        out.append(record)
    return {"bases": bases, "results": out}


def decode_results(payload) -> list | None:
    """The inverse of :func:`encode_results`."""
    if not payload:
        return None
    bases = payload.get("bases") or []
    out = []
    for record in payload.get("results") or []:
        if not isinstance(record, dict):
            out.append(serialize.decode(record))
            continue
        result = {}
        for key, value in record.items():
            if isinstance(value, dict) and "__slice__" in value:
                result[key] = _decode_slice(value, bases)
            elif isinstance(value, dict) and "__spec__" in value:
                result[key] = _decode_spec(value)
            else:
                result[key] = serialize.decode(value)
        out.append(result)
    return out


def encode_spectrum_results(spectrum) -> dict:
    """``{key: encoded}`` for every saved-result key ``spectrum`` holds."""
    raw = spectrum.metadata.raw or {}
    out = {}
    for key in RESULT_KEYS:
        payload = encode_results(raw.get(key))
        if payload is not None:
            out[key] = payload
    return out


def decode_spectrum_results(payload: dict | None) -> dict:
    """``{key: results}`` back from :func:`encode_spectrum_results`."""
    out = {}
    for key in RESULT_KEYS:
        value = decode_results((payload or {}).get(key))
        if value:
            out[key] = value
    return out


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------

def write_results_file(path: str, spectrum) -> bool:
    """Write ``spectrum``'s saved results to ``path`` (gzip JSON,
    atomically). With none to keep, an old file at ``path`` is removed
    (so a fit that has been replaced or cleared does not come back).
    Returns whether a file now exists."""
    payload = encode_spectrum_results(spectrum)
    if not payload:
        try:
            os.remove(path)
        except OSError:
            pass
        return False
    document = {"opennmr_results_version": FORMAT_VERSION, "results": payload}
    data = gzip.compress(json.dumps(document, separators=(",", ":")).encode("utf-8"),
                         compresslevel=5)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=".opennmr_results_", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(handle, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return True


def read_results_file(path: str) -> dict:
    """``{key: results}`` saved at ``path`` -- empty if there is no file
    or it cannot be read (a damaged file never stops a spectrum opening)."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with gzip.open(path, "rb") as f:
            document = json.loads(f.read().decode("utf-8"))
        return decode_spectrum_results(document.get("results"))
    except Exception:  # noqa: BLE001 -- fail soft, like the other sidecars
        return {}
