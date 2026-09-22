"""Turn the toolbar icon artwork into the PNGs the GUI loads.

The artwork in ``assets/tab_icons/`` is exported from Affinity Designer
at whatever size and aspect the drawing has. Every icon comes out as
the same 128 px transparent square, the drawing scaled to fit and
centred, so the five sit at one visual size in the toolbar however
tightly each was cropped.

Exports with an alpha channel are used as they are. An export on an
opaque white page (the first round of artwork) has its background
removed first, since a white square reads as a sticker on a tinted
macOS toolbar and as a hole in dark mode. Only the white *around* the drawing is removed: the region of near-white
pixels connected to the image border, plus small enclosed holes such as
the counter of the delta. Large interior white, like the sheet of paper
in the Load folder, is part of the drawing and stays opaque. The
antialiased fringe between that background and the linework is
un-composited from white ("colour to alpha"), so edges stay smooth
instead of keeping a pale halo.

Run from the repo root after re-exporting any icon::

    python packaging/make_toolbar_icons.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter
from scipy import ndimage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(REPO_ROOT, "assets", "tab_icons")
OUT_DIR = os.path.join(REPO_ROOT, "src", "opennmr", "gui", "resources", "toolbar")

# 128 px covers a 32 pt icon at 2x (Retina) with room to spare; Qt
# scales down from here for the actual toolbar size.
OUT_SIZE = 128
# Transparent border inside the square, so a tightly cropped drawing
# does not touch the edge of its button.
MARGIN_PX = 6
# Pixels at least this white on every channel count as background.
WHITE_THRESHOLD = 245
# How many pixels into the drawing the colour-to-alpha fringe reaches.
FRINGE_PX = 4
# Enclosed white smaller than this fraction of the image is a counter
# (the hole in a glyph like the delta) and goes transparent too; larger
# enclosed white is a deliberate fill and stays.
HOLE_FRACTION = 0.005

ICONS = ("load", "processing", "spectrum_fitting", "curve_fitting", "compare")


def _read_rgb(path: str) -> np.ndarray:
    image = QImage(path).convertToFormat(QImage.Format.Format_RGB888)
    width, height = image.width(), image.height()
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8)
    rows = buffer.reshape(height, image.bytesPerLine())
    return rows[:, : width * 3].reshape(height, width, 3).astype(np.float64)


def remove_background(rgb: np.ndarray) -> np.ndarray:
    """RGBA float array (0..255) with the border-connected white removed."""
    near_white = rgb.min(axis=2) >= WHITE_THRESHOLD
    labels, _count = ndimage.label(near_white)
    border = np.unique(np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    sizes = np.bincount(labels.ravel())
    small = np.flatnonzero(sizes < HOLE_FRACTION * near_white.size)
    background = np.isin(labels, border[border != 0]) | (
        np.isin(labels, small) & near_white)

    fringe = ndimage.binary_dilation(background, iterations=FRINGE_PX) & ~background

    alpha = np.ones(rgb.shape[:2])
    alpha[background] = 0.0
    # Colour to alpha against white: the smallest alpha that explains
    # the pixel as some colour composited over white.
    fringe_alpha = (1.0 - rgb[fringe] / 255.0).max(axis=1)
    alpha[fringe] = fringe_alpha

    out = rgb.copy()
    safe = np.where(fringe_alpha > 0, fringe_alpha, 1.0)[:, None]
    out[fringe] = np.clip((rgb[fringe] - 255.0 * (1.0 - safe)) / safe, 0, 255)
    return np.dstack([out, alpha * 255.0])


def _rgba_image(rgba: np.ndarray) -> QImage:
    data = np.ascontiguousarray(np.round(rgba).astype(np.uint8))
    height, width = data.shape[:2]
    # .copy(): the QImage would otherwise point into ``data``'s buffer.
    return QImage(data.data, width, height, width * 4, QImage.Format.Format_RGBA8888).copy()


def load_icon(path: str) -> QImage:
    image = QImage(path)
    if image.isNull():
        raise OSError(f"could not read {path}")
    if not image.hasAlphaChannel():
        image = _rgba_image(remove_background(_read_rgb(path)))
    # Scale premultiplied, or transparent (black-behind) pixels bleed
    # into the edges as a dark fringe.
    return image.convertToFormat(QImage.Format.Format_RGBA8888_Premultiplied)


def fit_square(image: QImage) -> QImage:
    """``image`` scaled to fit inside the margin and centred on a
    transparent OUT_SIZE square."""
    inner = OUT_SIZE - 2 * MARGIN_PX
    scaled = image.scaled(inner, inner, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
    square = QImage(OUT_SIZE, OUT_SIZE, QImage.Format.Format_RGBA8888_Premultiplied)
    square.fill(QColor(0, 0, 0, 0))
    painter = QPainter(square)
    painter.drawImage((OUT_SIZE - scaled.width()) // 2, (OUT_SIZE - scaled.height()) // 2, scaled)
    painter.end()
    return square


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ICONS:
        source = os.path.join(SOURCE_DIR, f"icon_{name}.png")
        if not os.path.exists(source):
            print(f"missing: {source}", file=sys.stderr)
            return 1
        target = os.path.join(OUT_DIR, f"{name}.png")
        if not fit_square(load_icon(source)).save(target):
            raise OSError(f"could not write {target}")
        print(f"{source} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
