"""File ▸ Save Figure As… (Ctrl/Cmd+S): a preview of the figure **exactly
as it will be written**, before choosing where to write it.

The live plot is sized by Qt's window layout, but an export is laid out
again at the export size (Plot Style ▸ Export, 8 × 8 in by default). Text
keeps its point size, so at a different figure size the legend, labels
and margins sit differently from the live view. The preview goes through
the same code path as the save itself --
:meth:`MplCanvas.export_full_resolution` (export size, re-layout, full
resolution, crosshair hidden, crop-to-content when something crosses the
edge) -- and only renders at a lower resolution, which changes nothing
about where anything sits: text sizes and line widths are in points, so
the layout depends on the size in inches, not on the dpi.

:func:`render_export_preview` and :func:`save_with_overrides` hold no
widgets beyond the canvas, so tests (and anything scripted) use the same
path the dialog does.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os

import matplotlib
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QMessageBox, QScrollArea, QVBoxLayout,
)

from .widgets.typed_spinbox import LengthSpinBox, TypedSpinBox

#: Longest side of the preview image (fit-to-window mode), in pixels.
PREVIEW_MAX_SIDE_PX = 900
#: Formats offered in the save dialog, first is the default.
SAVE_FORMATS = (("png", "PNG image"), ("pdf", "PDF (vector)"), ("svg", "SVG (vector)"),
                ("eps", "EPS (vector)"), ("tif", "TIFF image"), ("jpg", "JPEG image"))
MAX_DPI = 4800


@dataclasses.dataclass
class ExportPreview:
    """One rendered preview. ``width_in``/``height_in`` are the size of
    the file that will be written (after any crop to content), ``dpi`` the
    export dpi, ``png`` the preview image (rendered at ``preview_dpi``)."""

    png: bytes
    width_in: float
    height_in: float
    dpi: float
    preview_dpi: float
    cropped: bool

    @property
    def pixels(self) -> tuple[int, int]:
        """The written raster's size in pixels (PNG/TIFF/JPEG)."""
        return round(self.width_in * self.dpi), round(self.height_in * self.dpi)


def export_settings(canvas) -> tuple[float, float, float]:
    """``(width_in, height_in, dpi)`` the canvas would export at now."""
    default_w, default_h = canvas.default_export_size()
    width = canvas._export_fig_width if canvas._export_fig_width is not None else default_w
    height = canvas._export_fig_height if canvas._export_fig_height is not None else default_h
    dpi = canvas._export_dpi if canvas._export_dpi is not None else canvas.figure.get_dpi()
    return float(width), float(height), float(dpi)


@contextlib.contextmanager
def _overrides(canvas, width=None, height=None, dpi=None):
    """Export width/height/dpi set for the duration only."""
    saved = (canvas._export_fig_width, canvas._export_fig_height, canvas._export_dpi)
    if width is not None:
        canvas._export_fig_width = float(width)
    if height is not None:
        canvas._export_fig_height = float(height)
    if dpi is not None:
        canvas._export_dpi = float(dpi)
    try:
        yield
    finally:
        canvas._export_fig_width, canvas._export_fig_height, canvas._export_dpi = saved


def render_export_preview(canvas, *, width=None, height=None, dpi=None,
                          preview_dpi: float | None = None,
                          max_side_px: int = PREVIEW_MAX_SIDE_PX) -> ExportPreview:
    """Render ``canvas``'s figure as File ▸ Save Figure As… would write it,
    at ``width`` × ``height`` in and ``dpi`` (each defaulting to the
    canvas's export settings), into a PNG at ``preview_dpi`` (default:
    whatever puts the longest side at ``max_side_px``)."""
    with _overrides(canvas, width, height, dpi):
        w, h, export_dpi = export_settings(canvas)
        if preview_dpi is None:
            preview_dpi = min(export_dpi, max_side_px / max(w, h))
        preview_dpi = max(float(preview_dpi), 10.0)
        buffer = io.BytesIO()
        seen = {}

        def _save():
            seen["bbox"] = matplotlib.rcParams["savefig.bbox"]
            canvas.figure.savefig(buffer, format="png", dpi=preview_dpi)

        canvas.export_full_resolution(_save)
    png = buffer.getvalue()
    image = QImage.fromData(png, "PNG")
    width_in = image.width() / preview_dpi if not image.isNull() else w
    height_in = image.height() / preview_dpi if not image.isNull() else h
    return ExportPreview(png=png, width_in=width_in, height_in=height_in, dpi=export_dpi,
                         preview_dpi=preview_dpi, cropped=seen.get("bbox") == "tight")


def save_with_overrides(canvas, path: str, *, width=None, height=None, dpi=None) -> str:
    """Write the figure to ``path`` (format from its extension) exactly as
    the preview showed it."""
    with _overrides(canvas, width, height, dpi):
        canvas.export_full_resolution(lambda: canvas.figure.savefig(path))
    return path


def describe(preview: ExportPreview, unit: str = "in") -> str:
    """One line: the written size, in inches or cm, and in pixels."""
    factor = 2.54 if unit == "cm" else 1.0
    px_w, px_h = preview.pixels
    text = (f"{preview.width_in * factor:.2f} × {preview.height_in * factor:.2f} {unit}"
            f" at {preview.dpi:g} dpi → {px_w} × {px_h} px for PNG/TIFF/JPEG")
    if preview.cropped:
        text += " (cropped to the content: something reached the edge)"
    return text


class FigurePreviewDialog(QDialog):
    """Preview, adjust size/dpi, then Save… Emits :attr:`settingsKept`
    ``(dpi, width_in, height_in)`` when the user ticks "Keep for later
    exports" and saves, so the Plot Style ▸ Export boxes follow."""

    settingsKept = Signal(float, float, float)

    def __init__(self, canvas, parent=None, unit: str = "in") -> None:
        super().__init__(parent)
        self.setWindowTitle("Save figure – preview")
        self.canvas = canvas
        self.unit = unit if unit in LengthSpinBox.UNITS else "in"
        self.saved_path: str | None = None
        self.preview: ExportPreview | None = None
        width, height, dpi = export_settings(canvas)

        layout = QVBoxLayout(self)
        intro = QLabel("This is the figure as it will be written: same size, layout, legend "
                       "position and text size relative to the figure. Change the size here "
                       "to see how the layout responds.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QHBoxLayout()
        form = QFormLayout()
        self.width_spin = LengthSpinBox()
        self.width_spin.setRange(1.0, 40.0)
        self.width_spin.set_unit(self.unit)
        self.width_spin.setValue(width)
        form.addRow("Width:", self.width_spin)
        self.height_spin = LengthSpinBox()
        self.height_spin.setRange(1.0, 40.0)
        self.height_spin.set_unit(self.unit)
        self.height_spin.setValue(height)
        form.addRow("Height:", self.height_spin)
        controls.addLayout(form)
        form2 = QFormLayout()
        self.dpi_spin = TypedSpinBox()
        self.dpi_spin.setRange(50, MAX_DPI)
        self.dpi_spin.setValue(int(round(dpi)))
        self.dpi_spin.setToolTip("Resolution of a raster file (PNG, TIFF, JPEG). The layout "
                                 "does not depend on it; vector formats ignore it.")
        form2.addRow("DPI:", self.dpi_spin)
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItem("Fit to window", "fit")
        self.zoom_combo.addItem("Actual size on screen", "actual")
        self.zoom_combo.setToolTip("Actual size draws the figure at its printed size on this "
                                   "screen, to judge whether the text will be legible.")
        form2.addRow("View:", self.zoom_combo)
        controls.addLayout(form2)
        layout.addLayout(controls)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background: #8a8a8a; padding: 12px; }")
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumSize(560, 460)
        layout.addWidget(self.scroll, 1)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.keep_check = QCheckBox("Keep this size and DPI for later exports (Plot Style ▸ Export)")
        self.keep_check.setChecked(True)
        layout.addWidget(self.keep_check)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save…")
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.refresh)
        for spin in (self.width_spin, self.height_spin, self.dpi_spin):
            spin.valueChanged.connect(lambda *_: self._timer.start())
        self.zoom_combo.currentIndexChanged.connect(lambda *_: self.refresh())
        self.resize(820, 760)
        self.refresh()

    # -- values --------------------------------------------------------
    def values(self) -> dict:
        return dict(width=self.width_spin.value(), height=self.height_spin.value(),
                    dpi=float(self.dpi_spin.value()))

    def _screen_dpi(self) -> float:
        screen = self.screen() or QGuiApplication.primaryScreen()
        return float(screen.logicalDotsPerInch()) if screen is not None else 96.0

    def refresh(self) -> None:
        """Render the preview again for the current boxes."""
        actual = self.zoom_combo.currentData() == "actual"
        preview_dpi = self._screen_dpi() if actual else None
        try:
            self.preview = render_export_preview(self.canvas, preview_dpi=preview_dpi,
                                                 **self.values())
        except Exception as exc:  # noqa: BLE001 -- shown, never fatal
            self.preview = None
            self.image_label.setText(f"Could not render a preview:\n{exc}")
            self.info_label.setText("")
            return
        pixmap = QPixmap()
        pixmap.loadFromData(self.preview.png, "PNG")
        self.image_label.setPixmap(pixmap)
        self.scroll.setWidgetResizable(not actual)
        if actual:
            self.image_label.adjustSize()
        self.info_label.setText("Written size: " + describe(self.preview, self.unit))

    # -- save ----------------------------------------------------------
    def _filters(self) -> list[str]:
        return [f"{label} (*.{ext})" for ext, label in SAVE_FORMATS] + ["All files (*)"]

    def _on_save(self) -> None:
        start_dir = matplotlib.rcParams.get("savefig.directory", "") or ""
        start = os.path.join(os.path.expanduser(start_dir), "figure.png") if start_dir else "figure.png"
        filters = self._filters()
        path, chosen = QFileDialog.getSaveFileName(self, "Save figure", start, ";;".join(filters))
        if not path:
            return
        self.save_to(path, chosen_filter=chosen)

    def save_to(self, path: str, chosen_filter: str = "") -> str | None:
        """Write to ``path`` (an extension is added from the chosen filter,
        else .png, if it has none); closes the dialog on success."""
        if not os.path.splitext(path)[1]:
            ext = next((e for e, label in SAVE_FORMATS if chosen_filter.startswith(label)), "png")
            path += "." + ext
        values = self.values()
        try:
            save_with_overrides(self.canvas, path, **values)
        except Exception as exc:  # noqa: BLE001 -- shown
            QMessageBox.warning(self, "Save figure", f"Could not save {path}:\n{exc}")
            return None
        with contextlib.suppress(Exception):
            matplotlib.rcParams["savefig.directory"] = os.path.dirname(path)
        if self.keep_check.isChecked():
            self.canvas.set_export_settings(dpi=values["dpi"], fig_width=values["width"],
                                            fig_height=values["height"])
            self.settingsKept.emit(values["dpi"], values["width"], values["height"])
        self.saved_path = path
        self.accept()
        return path
