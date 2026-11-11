"""A message box whose body scrolls.

``QMessageBox`` grows with its text and never scrolls, so a long message
-- opening an openNMR state of 41 spectra lists 41 lines, each with a
full data path -- ran off the bottom of the screen with its OK button
(explicit user request, 2026-11-11). This shows the same kind of message
as rich text in a read-only, scrollable browser no taller than most of
the screen, with Copy (the plain text) and OK.

It is shown with :meth:`QDialog.show`, not ``exec``: an information
message has nothing to answer, so it need not stop the event loop, and a
test that opens a state is not left waiting on it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QStyle, QTextBrowser,
    QVBoxLayout,
)


class ScrollableMessageDialog(QDialog):
    """An information message whose rich-text body scrolls."""

    def __init__(self, parent, title: str, html: str, plain: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._plain = plain

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        icon = QLabel()
        size = self.style().pixelMetric(QStyle.PixelMetric.PM_MessageBoxIconSize, None, self)
        icon.setPixmap(self.style().standardIcon(
            QStyle.StandardPixmap.SP_MessageBoxInformation).pixmap(size, size))
        icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        row.addWidget(icon)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setFrameShape(QTextBrowser.Shape.NoFrame)
        # The dialog's own background, so it reads as a message rather
        # than as an editor.
        self.browser.viewport().setAutoFillBackground(False)
        self.browser.setStyleSheet("QTextBrowser { background: transparent; }")
        self.browser.setHtml(html)
        row.addWidget(self.browser, 1)
        layout.addLayout(row, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        self.copy_button = buttons.addButton("Copy", QDialogButtonBox.ButtonRole.ActionRole)
        self.copy_button.setToolTip("Copy this message as plain text.")
        self.copy_button.clicked.connect(self.copy)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self._fit_to_screen()

    def plain_text(self) -> str:
        return self._plain or self.browser.toPlainText()

    def copy(self) -> None:
        QApplication.clipboard().setText(self.plain_text())

    def _fit_to_screen(self) -> None:
        """As wide as the text wants up to 60% of the screen, as tall
        as it needs up to 70%; beyond that it scrolls."""
        screen = (self.parentWidget().screen() if self.parentWidget() is not None
                  else QGuiApplication.primaryScreen())
        avail = screen.availableGeometry() if screen is not None else None
        max_w = int(avail.width() * 0.6) if avail is not None else 900
        max_h = int(avail.height() * 0.7) if avail is not None else 700
        width = max(480, min(max_w, 820))
        doc = self.browser.document()
        doc.setTextWidth(width - 80)
        height = int(doc.size().height()) + 110
        self.resize(width, max(220, min(max_h, height)))


def show_message(parent, title: str, html: str, plain: str = "") -> ScrollableMessageDialog:
    """Show a scrollable information message (non-blocking); returns it."""
    dialog = ScrollableMessageDialog(parent, title, html, plain)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.show()
    return dialog
