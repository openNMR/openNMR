"""The "at least 3-5 spinning sidebands" reminder shown beside the MAS
rate of every fit section that fits an anisotropic interaction (Fit ▸
CSA, CSA, Quad & J-coupling, Quadrupolar).

Explicit user request (2026-10-08): "Indicate that, for fitting any
anisotropic interactions, it is sensible to have at least 3-5 sidebands
for accuracy". The wording is :data:`opennmr.notebook.fit_advice.
SIDEBAND_GUIDANCE`, so the notebook and the GUI say the same thing; the
fit's own advice then counts the sidebands of the fitted pattern
(:func:`~opennmr.notebook.fit_advice.sideband_count`).
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel

from ...notebook.fit_advice import GOOD_SIDEBANDS, MIN_SIDEBANDS, SIDEBAND_GUIDANCE

#: The short line shown under the MAS rate; the full guidance is its tooltip.
SHORT_TEXT = (f"Tip: fit anisotropic interactions with at least {MIN_SIDEBANDS}–{GOOD_SIDEBANDS} "
              "spinning sidebands for accuracy.")


def sideband_hint_label(parent=None) -> QLabel:
    label = QLabel(SHORT_TEXT, parent)
    label.setObjectName("sidebandHint")
    label.setWordWrap(True)
    label.setToolTip(SIDEBAND_GUIDANCE)
    font = label.font()
    font.setItalic(True)
    label.setFont(font)
    return label
