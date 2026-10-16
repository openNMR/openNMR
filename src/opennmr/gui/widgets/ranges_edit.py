"""A line edit for several ppm ranges ("5 to 10, 12 to 30") -- the
"Several ranges" fit window of the Fit panels (explicit user request,
2026-10-16). Parsing is :func:`opennmr.notebook.fit_ranges.parse_ppm_ranges`,
which a notebook uses too."""

from __future__ import annotations

from PySide6.QtWidgets import QLineEdit

from ...notebook import fit_ranges

_TIP = ("The fit uses only these ranges (ppm), e.g. '5 to 10, 12 to 30' fits 5-10 and "
        "12-30 ppm but not 10-12. Pairs may be written 5-10, 5 to 10 or (5, 10), separated "
        "by commas or semicolons; overlapping ranges are joined.")


class PpmRangesEdit(QLineEdit):
    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setPlaceholderText("e.g. 5 to 10, 12 to 30")
        self.setToolTip(_TIP)
        self.textChanged.connect(lambda _t: self._check())

    def ranges(self) -> list[tuple[float, float]]:
        """The ranges, sorted and merged; ValueError when unreadable."""
        return fit_ranges.parse_ppm_ranges(self.text())

    def set_ranges(self, ranges) -> None:
        self.setText(fit_ranges.format_ppm_ranges(fit_ranges.merge_ranges(ranges)))

    def _check(self) -> None:
        if not self.text().strip():
            self.setStyleSheet("")
            self.setToolTip(_TIP)
            return
        try:
            ranges = self.ranges()
        except ValueError as exc:
            self.setStyleSheet("QLineEdit { border: 1px solid #c0392b; }")
            self.setToolTip(f"{exc}\n\n{_TIP}")
            return
        self.setStyleSheet("")
        self.setToolTip(f"Fitting {fit_ranges.format_ppm_ranges(ranges)} ppm.\n\n{_TIP}")
