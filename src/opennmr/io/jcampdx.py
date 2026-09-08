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

import os
import re

_KEY_RE = re.compile(r"^##\$([^=]+)=\s*(.*?)\s*$")
_ARRAY_RE = re.compile(r"^\(\s*(\d+)\s*\.\.\s*(\d+)\s*\)$")
#: The same size marker, but with the values on the *same* line
#: (``##$QS= (0..7)83 83 ...``) rather than the next.
_INLINE_ARRAY_RE = re.compile(r"^\(\s*(\d+)\s*\.\.\s*(\d+)\s*\)\s*(.+)$")

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
                continue

            inline_match = _INLINE_ARRAY_RE.match(value)
            if inline_match:
                # `##$QS= (0..7)83 83 83 ...` -- the size marker and the
                # values share one line, unlike the commoner form where
                # the values start on the next. Missing this returned
                # the whole thing as one meaningless string.
                lo, hi = int(inline_match.group(1)), int(inline_match.group(2))
                n = hi - lo + 1
                values = _TOKEN_RE.findall(inline_match.group(3))
                while len(values) < n:
                    values.extend(_TOKEN_RE.findall(next(lines)))
                params[key] = [_coerce_scalar(v) for v in values[:n]]
                continue

            if value.startswith("<") and not value.endswith(">"):
                # A `<...>` string whose closing bracket is on a later
                # line -- Bruker wraps long ones, and `PROBHD` really
                # does wrap on real data. Without this the value came
                # back with its opening bracket still attached and its
                # tail missing ('<2.5 mm MAS 1H/BB H9558/0002'), which
                # matters because `metadata.probe` is what the probe
                # gradient-default config is keyed by.
                parts = [value]
                for continuation in lines:
                    parts.append(continuation.strip())
                    if parts[-1].endswith(">"):
                        break
                value = " ".join(part for part in parts if part)
                coerced = _coerce_scalar(value)
                # The wrap itself is formatting, not content, so the
                # whitespace it introduces is dropped. nmrglue keeps
                # the newline here ('...0002\n'); a trailing newline in
                # a probe name is an artifact, not data, and this value
                # is used as a config key.
                params[key] = coerced.strip() if isinstance(coerced, str) else coerced
                continue

            params[key] = _coerce_scalar(value)

    return params


def read_acqus_files(path: str) -> dict:
    """Every acquisition-parameter file in a Bruker experiment
    directory, keyed by filename -- ``{"acqus": {...}, "acqu2s": {...}}``.

    Replaces ``nmrglue.bruker.read_acqus_file``, whose shape this
    matches so it can be swapped in directly. Reads ``acqus`` through
    ``acqu4s``, skipping any that are absent.

    **Three deliberate differences from nmrglue**, all checked across
    every dataset in ``test_data``:

    * ``yes``/``no`` parameters (``LOCKED``, ``LOCSHFT``, ``NusFPNZ``,
      ``PROSOL``, ``scaledByNS``, ``scaledByRG``) stay as strings here
      rather than becoming ``True``/``False``. Nothing in this project
      reads them.
    * A ``<...>`` value wrapped across lines comes back stripped
      (``'2.5 mm MAS 1H/BB H9558/0002'``) rather than carrying
      nmrglue's trailing newline. ``PROBHD`` is the real case, and it
      keys the probe gradient-default config, so the artifact is worth
      losing.
    * ``SPNAM``-style arrays containing ``< >`` (a bracketed *space*)
      parse as one value here. nmrglue splits it into ``'<'`` and
      ``'>'``, yielding 65 entries for a 64-entry array -- this one is
      simply a bug there, not a convention to match.
    """
    params = {}
    for name in ("acqus", "acqu2s", "acqu3s", "acqu4s"):
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            params[name] = parse_jcampdx_params(candidate)
    return params
