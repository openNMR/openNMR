"""A QDoubleSpinBox that also accepts a typed arithmetic delta.

Typing e.g. ``45+90`` (appending ``+90`` after the value already
shown, ``45``) commits ``135``, not a parse error -- same for
subtraction (``45-90`` commits ``-45``). Useful for nudging a value by
a known amount (a phase correction, a reference shift, ...) without
doing the arithmetic yourself first.

Deliberately requires *both* operands spelled out in the typed text
(``45+90``, not a bare ``+90``) rather than tracking "the value before
this edit" as an implicit left operand: a bare leading ``-`` (e.g.
selecting all and typing ``-45`` to set a literal negative value, a
completely normal and expected thing to do) would otherwise be
ambiguous between "set to -45" and "subtract 45" -- there's no way to
tell those apart from the committed text alone. A bare leading ``+``
has no such ambiguity (nobody types "+45" to mean "set to positive
45"), but supporting only one of +/- for the bare form and not the
other would be a confusing half-measure, so neither is supported --
only the fully self-contained expression form, which has no ambiguity
either way and matches how the feature was actually asked for.
"""

from __future__ import annotations

import re

from PySide6.QtWidgets import QDoubleSpinBox

_EXPR_RE = re.compile(
    r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([+\-])\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$"
)


class RelativeDoubleSpinBox(QDoubleSpinBox):
    """See module docstring."""

    def valueFromText(self, text: str) -> float:  # noqa: N802 -- Qt override
        stripped = self._strip_affixes(text)
        match = _EXPR_RE.match(stripped)
        if match:
            left, op, right = match.groups()
            try:
                left_val, right_val = float(left), float(right)
            except ValueError:
                pass
            else:
                return left_val + right_val if op == "+" else left_val - right_val
        return super().valueFromText(text)

    def _strip_affixes(self, text: str) -> str:
        stripped = text.strip()
        prefix, suffix = self.prefix(), self.suffix()
        if prefix and stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
        if suffix and stripped.endswith(suffix):
            stripped = stripped[: len(stripped) - len(suffix)]
        return stripped.strip()
