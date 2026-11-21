"""Re-label the slices extracted from a 2D spectrum.

An extracted slice is stored on its spectrum as a small dict in
``spectrum.metadata.raw["contour_slices"]`` (axis, index, label, colour).
The label used to be settable only when the slice was extracted. These
helpers read and change it afterwards, on exactly the store the GUI uses,
so a relabel made here shows in the GUI and the other way round.
"""
from __future__ import annotations


def _slices(spectrum) -> list[dict]:
    return spectrum.metadata.raw.get("contour_slices") or []


def slice_labels(spectrum) -> list[str]:
    """The name each extracted slice is shown under, in order: its own
    label, or the default ``Slice n`` (n from 1) when it has none."""
    return [spec.get("label") or f"Slice {i + 1}" for i, spec in enumerate(_slices(spectrum))]


def relabel_slice(spectrum, which, label: str) -> str:
    """Give one extracted slice a new label.

    ``which`` is the slice's position (0-based) or its current label.
    An empty ``label`` restores the default ``Slice n``. Returns the
    label now shown. Raises ``KeyError`` for an unknown or ambiguous
    name and ``IndexError`` for a position out of range, so a typo
    cannot silently relabel the wrong trace.
    """
    slices = _slices(spectrum)
    if isinstance(which, str):
        matches = [i for i, name in enumerate(slice_labels(spectrum)) if name == which]
        if len(matches) != 1:
            raise KeyError(
                f"{which!r} matches {len(matches)} slices; use the position instead "
                f"(labels are {slice_labels(spectrum)})")
        position = matches[0]
    else:
        position = int(which)
        if not (0 <= position < len(slices)):
            raise IndexError(f"slice {position} does not exist; there are {len(slices)}")
    slices[position]["label"] = (label or "").strip()
    # A fresh list, so anything holding the old one sees a change.
    spectrum.metadata.raw["contour_slices"] = list(slices)
    return slice_labels(spectrum)[position]
