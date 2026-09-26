"""
Exact, JSON-safe encoding of the Python values a fit or simulation
produces -- nested dicts/lists/tuples of numbers, strings, numpy arrays
and scalars -- so a fit's full result can be saved with a spectrum's
processing state and brought back bit-for-bit.

* numpy arrays -> ``{"__nd__": {"dtype", "shape", "b64"}}`` (raw bytes,
  base64): exact, and ~5x smaller than a JSON list of floats.
* tuples -> ``{"__tuple__": [...]}``; complex numbers ->
  ``{"__complex__": [re, im]}``; dicts with non-string keys ->
  ``{"__dict__": [[k, v], ...]}``.
* Anything else (a spectrum object, a function) ->
  ``{"__unsupported__": "<type name>"}``, decoded as ``None``. Callers
  drop such keys before encoding when they matter; the marker says what
  was left out rather than failing the whole save.

Qt-free; used by the GUI and :mod:`opennmr.io.history`.
"""

from __future__ import annotations

import base64
import math

import numpy as np

_MARKERS = ("__nd__", "__tuple__", "__complex__", "__dict__", "__unsupported__", "__float__")


def encode(value):
    """``value`` as plain JSON-able data (see the module docstring)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isfinite(value):
            return value
        return {"__float__": repr(value)}  # nan/inf are not JSON
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (complex, np.complexfloating)):
        return {"__complex__": [encode(float(value.real)), encode(float(value.imag))]}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return {"__tuple__": [encode(v) for v in value.tolist()]} if value.ndim else encode(value.item())
        array = np.ascontiguousarray(value)
        return {"__nd__": {"dtype": array.dtype.str, "shape": list(array.shape),
                           "b64": base64.b64encode(array.tobytes()).decode("ascii")}}
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value) and not any(k in _MARKERS for k in value):
            return {k: encode(v) for k, v in value.items()}
        return {"__dict__": [[encode(k), encode(v)] for k, v in value.items()]}
    return {"__unsupported__": type(value).__name__}


def decode(value):
    """The inverse of :func:`encode`."""
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        (key, inner), = value.items()
        if key == "__nd__":
            raw = base64.b64decode(inner["b64"])
            return np.frombuffer(raw, dtype=np.dtype(inner["dtype"])).reshape(inner["shape"]).copy()
        if key == "__tuple__":
            return tuple(decode(v) for v in inner)
        if key == "__complex__":
            return complex(decode(inner[0]), decode(inner[1]))
        if key == "__float__":
            return float(inner)
        if key == "__dict__":
            return {_hashable(decode(k)): decode(v) for k, v in inner}
        if key == "__unsupported__":
            return None
    return {k: decode(v) for k, v in value.items()}


def _hashable(key):
    return tuple(key) if isinstance(key, list) else key


def unsupported_keys(value, path: str = "") -> list[str]:
    """Where in ``value`` :func:`encode` would have to leave something
    out -- for tests and for saying what a restore cannot bring back."""
    found = []
    if isinstance(value, dict):
        for k, v in value.items():
            found += unsupported_keys(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            found += unsupported_keys(v, f"{path}[{i}]")
    elif encode(value) == {"__unsupported__": type(value).__name__}:
        found.append(path)
    return found
