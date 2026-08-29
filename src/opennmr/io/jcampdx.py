"""
Generic parser for Bruker-style JCAMP-DX parameter files (``acqus``,
``acqu2s``, ``acqu3s``, ``procs``, ``proc2s``, ``proc3s``, ...).

This is the single place ``opennmr.io.bruker`` goes to read a
parameter file. Previously (in ``nmrtools2``) this same
``##$KEY=value`` format was hand-parsed inline, separately, in five
different places (once per file/field combination needed), each
re-implementing "read lines until every field of interest is filled or
EOF" with its own NaN-sentinel bookkeeping. Parsing the whole file once
into a flat dict removes both problems: one parser, and no sentinel
loop -- a field is simply present in the dict or it isn't.
"""

from __future__ import annotations

import re

_KEY_RE = re.compile(r"^##\$([^=]+)=\s*(.*?)\s*$")
_ARRAY_RE = re.compile(r"^\(\s*(\d+)\s*\.\.\s*(\d+)\s*\)$")

# A `<...>` bracketed token (kept whole, even if it contains spaces) or
# a run of non-whitespace -- used to tokenize array continuation lines,
# which are whitespace-separated EXCEPT some arrays (e.g. GPNAM/SPNAM/
# CPDPRG) hold `<name>` strings rather than plain numbers.
_TOKEN_RE = re.compile(r"<[^>]*>|\S+")


def _coerce_scalar(value: str):
    if len(value) >= 2 and value[0] == "<" and value[-1] == ">":
        return value[1:-1]
    try:
        return float(value)
    except ValueError:
        return value


def parse_jcampdx_params(path: str) -> dict:
    """Parse every ``##$KEY=value`` line in a Bruker-style JCAMP-DX
    parameter file, including multi-line array parameters such as::

        ##$P= (0..63)
        10 5.5 20 ...

    Returns a flat dict keyed by the bare parameter name (``$`` and
    ``##`` stripped, e.g. ``"NUC1"``, ``"SW_h"``, ``"P"``). Scalar
    values are coerced to ``float`` where possible, with ``<...>``
    string values un-bracketed; anything else is left as a string.
    Array parameters become a ``list``, elementwise-coerced the same
    way -- some (e.g. Bruker's ``GPNAM``/``SPNAM``/``CPDPRG``) hold
    ``<name>`` strings rather than numbers.
    """

    params: dict = {}

    with open(path, "r", errors="ignore") as f:
        lines = iter(f)

        for raw_line in lines:
            line = raw_line.strip()

            match = _KEY_RE.match(line)
            if not match:
                continue

            key, value = match.group(1).strip(), match.group(2).strip()

            array_match = _ARRAY_RE.match(value)
            if array_match:
                lo, hi = int(array_match.group(1)), int(array_match.group(2))
                n = hi - lo + 1
                values: list[str] = []
                while len(values) < n:
                    values.extend(_TOKEN_RE.findall(next(lines)))
                params[key] = [_coerce_scalar(v) for v in values[:n]]
            else:
                params[key] = _coerce_scalar(value)

    return params
