"""One button of the main toolbar: an icon, with its dropdown arrow
centred underneath rather than beside it.

``QToolButton``'s own split-button mode (``MenuButtonPopup``) always
puts the arrow on the right, which made the buttons that have a menu
wider than the ones that do not and pushed their icons off-centre. Real
user request: "the dropdown arrows for the icons should be below the
icon and centred".

So this is two buttons stacked: the icon on top (checkable, raises the
panel) and a small arrow below it (opens the menu). A button with no
menu keeps an empty, disabled arrow slot, so every icon in the row
sits at the same height.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QMenu, QToolButton, QVBoxLayout, QWidget

ARROW_HEIGHT = 12


class ModeButton(QWidget):
    """``button`` is the icon; ``arrow`` is the dropdown beneath it."""

    def __init__(self, label: str, icon, icon_size: QSize, menu: QMenu | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.label = label

        self.button = QToolButton(self)
        # The text stays set even though only the icon shows: it is the
        # accessible name and what tests and code look buttons up by.
        self.button.setText(label)
        self.button.setIcon(icon)
        self.button.setIconSize(icon_size)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.button.setCheckable(True)
        self.button.setAutoRaise(True)
        self.button.setAccessibleName(label)
        self.button.setToolTip(label)

        self.arrow = QToolButton(self)
        self.arrow.setAutoRaise(True)
        self.arrow.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if menu is not None:
            self.arrow.setArrowType(Qt.ArrowType.DownArrow)
            self.arrow.setMenu(menu)
            self.arrow.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            # InstantPopup draws its own small indicator as well as the
            # arrow; one is enough.
            self.arrow.setStyleSheet("QToolButton::menu-indicator { image: none; }")
            self.arrow.setToolTip(f"{label} options")
            self.arrow.setAccessibleName(f"{label} options")
        else:
            self.arrow.setEnabled(False)
            self.arrow.setStyleSheet("QToolButton { border: none; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.arrow, 0, Qt.AlignmentFlag.AlignHCenter)

    def menu(self) -> QMenu | None:
        return self.arrow.menu()

    def setButtonWidth(self, width: int) -> None:  # noqa: N802 -- Qt casing
        """Fix the whole column to ``width``, so every button in the row
        is the same size whether or not it has a menu."""
        self.button.setFixedSize(width, self.button.sizeHint().height())
        self.arrow.setFixedSize(width, ARROW_HEIGHT)
        self.setFixedWidth(width)
