"""Welcome to openNMR: a short, skippable tour of the Settings.

Shown by ``MainWindow.maybe_show_welcome()`` the first time a given
version of openNMR is opened (``welcome_seen_version`` in
``~/.opennmr/ui_settings.json``), and again whenever the user chooses
Help ▸ Welcome to openNMR…. Nothing here is a second copy of a setting:
each page reads the Settings panel's own widgets when it opens and, on
*Apply and start*, writes the chosen values back into them, so the
existing handlers apply and remember them exactly as if they had been
set in Settings. *Skip* closes the wizard and changes nothing.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from .widgets.typed_spinbox import TypedDoubleSpinBox, TypedSpinBox

WEBSITE_URL = "https://opennmr.github.io/"


def matplotlib_to_hex(color) -> str:
    import matplotlib.colors
    return matplotlib.colors.to_hex(color)


def _page(title: str, intro: str) -> tuple[QWidget, QVBoxLayout]:
    page = QWidget()
    layout = QVBoxLayout(page)
    heading = QLabel(title)
    font = heading.font()
    font.setPointSizeF(font.pointSizeF() * 1.4)
    font.setBold(True)
    heading.setFont(font)
    layout.addWidget(heading)
    text = QLabel(intro)
    text.setWordWrap(True)
    layout.addWidget(text)
    return page, layout


def _copy_combo(source: QComboBox) -> QComboBox:
    combo = QComboBox()
    for i in range(source.count()):
        combo.addItem(source.itemText(i), source.itemData(i))
    combo.setCurrentIndex(max(0, combo.findData(source.currentData())))
    combo.setToolTip(source.toolTip())
    return combo


class WelcomeWizard(QDialog):
    """The first-run / Help ▸ Welcome tour of the settings of ``window``."""

    def __init__(self, window, parent=None) -> None:
        super().__init__(parent or window)
        self.window_ = window
        self.applied = False
        self.setWindowTitle("Welcome to openNMR")
        self.resize(620, 520)

        outer = QVBoxLayout(self)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        outer.addWidget(rule)

        buttons = QHBoxLayout()
        self.skip_button = QPushButton("Skip and get started")
        self.skip_button.setToolTip("Close this and change nothing. Help ▸ Welcome to openNMR… "
                                    "opens it again.")
        self.skip_button.clicked.connect(self.skip)
        self.step_label = QLabel("")
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.go_back)
        self.next_button = QPushButton("Next")
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self.go_next)
        buttons.addWidget(self.skip_button)
        buttons.addStretch(1)
        buttons.addWidget(self.step_label)
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.next_button)
        outer.addLayout(buttons)

        for build in (self._welcome_page, self._appearance_page, self._plots_page,
                      self._layout_page, self._data_page, self._done_page):
            self.stack.addWidget(build())
        self._update_buttons()

    # ---- pages -----------------------------------------------------------
    def _welcome_page(self) -> QWidget:
        from .._version import __version__
        page, layout = _page(
            "Welcome to openNMR",
            f"Version {__version__}. Reading, processing, plotting, simulating and fitting NMR "
            "spectra, with every step logged so you can see how you arrived at an answer.\n\n"
            "The next few pages let you set up how openNMR looks and behaves: appearance, mouse "
            "and plots, which panels are shown, and where your data lives. Everything can be "
            "changed later in Settings, and this tour is always available from the Help menu.\n\n"
            "Short on time? Press Skip and start working; nothing will have been changed.")
        link = QLabel(f'Project website (in progress): <a href="{WEBSITE_URL}">{WEBSITE_URL}</a>')
        link.setTextFormat(Qt.TextFormat.RichText)
        link.setOpenExternalLinks(True)
        layout.addWidget(link)
        layout.addStretch(1)
        return page

    def _appearance_page(self) -> QWidget:
        w = self.window_
        page, layout = _page("Appearance", "Light or dark, and the size of menu and panel text. "
                             "Plot text is set separately, in Plot Style, because it belongs to "
                             "the figure you export.")
        form = QFormLayout()
        self.theme_combo = _copy_combo(w.ui_theme_combo)
        form.addRow("Appearance:", self.theme_combo)
        self.font_spin = TypedSpinBox()
        self.font_spin.setRange(w.ui_font_size_spin.minimum(), w.ui_font_size_spin.maximum())
        self.font_spin.setSuffix(" pt")
        self.font_spin.setValue(w.ui_font_size_spin.value())
        form.addRow("Menu font size:", self.font_spin)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _plots_page(self) -> QWidget:
        w = self.window_
        page, layout = _page("Mouse and plots", "How the plots respond to the mouse, and the unit "
                             "used for the size of exported figures.")
        form = QFormLayout()
        self.drag_combo = _copy_combo(w.drag_mode_combo)
        form.addRow("Click and drag:", self.drag_combo)
        self.double_click_check = QCheckBox("Double-click a plot to reset its view")
        self.double_click_check.setChecked(w.double_click_reset_checkbox.isChecked())
        form.addRow("Double-click:", self.double_click_check)
        self.contour_scroll_check = QCheckBox("Scroll up shows more contours")
        self.contour_scroll_check.setToolTip(w.contour_scroll_up_more_checkbox.toolTip())
        self.contour_scroll_check.setChecked(w.contour_scroll_up_more_checkbox.isChecked())
        form.addRow("2D contours:", self.contour_scroll_check)
        self.export_unit_combo = _copy_combo(w.export_unit_combo)
        form.addRow("Export size in:", self.export_unit_combo)
        # More of Settings ▸ Interface and Default plot styles (explicit
        # user request, 2026-10-23): the same settings, set here once.
        color, width = w._all_canvases()[0].crosshair_style()
        self._crosshair_color = matplotlib_to_hex(color)
        self.crosshair_colour_button = QPushButton()
        self.crosshair_colour_button.setToolTip(w.crosshair_colour_button.toolTip())
        self.crosshair_colour_button.clicked.connect(self._pick_crosshair_colour)
        self.crosshair_width_spin = TypedDoubleSpinBox()
        self.crosshair_width_spin.setRange(w.crosshair_width_spin.minimum(),
                                           w.crosshair_width_spin.maximum())
        self.crosshair_width_spin.setDecimals(2)
        self.crosshair_width_spin.setSingleStep(0.1)
        self.crosshair_width_spin.setSuffix(" pt")
        self.crosshair_width_spin.setValue(width)
        crosshair_row = QHBoxLayout()
        crosshair_row.addWidget(self.crosshair_colour_button)
        crosshair_row.addWidget(self.crosshair_width_spin)
        form.addRow("Crosshair:", crosshair_row)
        self._show_crosshair_colour()
        self.zoom_box_width_spin = TypedDoubleSpinBox()
        self.zoom_box_width_spin.setRange(w.zoom_box_width_spin.minimum(),
                                          w.zoom_box_width_spin.maximum())
        self.zoom_box_width_spin.setDecimals(1)
        self.zoom_box_width_spin.setSingleStep(0.5)
        self.zoom_box_width_spin.setSuffix(" px")
        self.zoom_box_width_spin.setToolTip(w.zoom_box_width_spin.toolTip())
        self.zoom_box_width_spin.setValue(w.zoom_box_width_spin.value())
        form.addRow("Zoom box:", self.zoom_box_width_spin)
        self.auto_styles_check = QCheckBox("Use saved default plot styles automatically")
        self.auto_styles_check.setToolTip(w.auto_apply_type_styles_checkbox.toolTip())
        self.auto_styles_check.setChecked(w.auto_apply_type_styles_checkbox.isChecked())
        form.addRow("Plot styles:", self.auto_styles_check)
        top = getattr(w, "settings_curve_top_spine_checkbox", None)
        right = getattr(w, "settings_curve_right_spine_checkbox", None)
        self.curve_top_check = QCheckBox("Top")
        self.curve_right_check = QCheckBox("Right")
        self.curve_top_check.setChecked(bool(top.isChecked()) if top is not None else False)
        self.curve_right_check.setChecked(bool(right.isChecked()) if right is not None else False)
        spine_row = QHBoxLayout()
        spine_row.addWidget(self.curve_top_check)
        spine_row.addWidget(self.curve_right_check)
        spine_row.addStretch(1)
        form.addRow("Curve plots, extra axis lines:", spine_row)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _show_crosshair_colour(self) -> None:
        swatch = QColor(self._crosshair_color)
        self.crosshair_colour_button.setText(swatch.name())
        text = "#000000" if swatch.lightness() > 128 else "#ffffff"
        self.crosshair_colour_button.setStyleSheet(
            f"background-color: {swatch.name()}; color: {text};")

    def _pick_crosshair_colour(self) -> None:
        picked = QColorDialog.getColor(QColor(self._crosshair_color), self, "Crosshair colour")
        if picked.isValid():
            self._crosshair_color = picked.name()
            self._show_crosshair_colour()

    def _layout_page(self) -> QWidget:
        w = self.window_
        page, layout = _page("Layout: which sections to show",
                             "Untick anything you will never use and it is hidden from the "
                             "panels. Nothing is lost: tick it again (here or in Settings) and "
                             "it is back as it was.")
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        self.section_checks: dict[str, QCheckBox] = {}
        for key, (label, _attrs) in w.OPTIONAL_SECTIONS.items():
            box = QCheckBox(label)
            current = w.section_shown_checkboxes.get(key)
            box.setChecked(True if current is None else current.isChecked())
            inner_layout.addWidget(box)
            self.section_checks[key] = box
        inner_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)
        row = QHBoxLayout()
        show_all = QPushButton("Show all")
        show_all.clicked.connect(lambda: [b.setChecked(True) for b in self.section_checks.values()])
        row.addWidget(show_all)
        row.addStretch(1)
        layout.addLayout(row)
        return page

    def _data_page(self) -> QWidget:
        w = self.window_
        page, layout = _page("Data and fitting", "Where openNMR looks for your data and saves "
                             "your work, and how hard a fit may try before it stops.")
        form = QFormLayout()
        self.data_folder_edit = QLineEdit(w.data_folder_edit.text())
        self.data_folder_edit.setPlaceholderText("None -- dialogs start where they always did")
        form.addRow("Default data folder:", self._with_browse(self.data_folder_edit,
                                                              "Default data folder"))
        self.browse_at_start_check = QCheckBox("List its experiments in Open's browser at start-up")
        self.browse_at_start_check.setChecked(w.data_folder_browse_at_start_checkbox.isChecked())
        form.addRow("", self.browse_at_start_check)
        self.topspin_check = QCheckBox("Open Bruker data with TopSpin's processing "
                                       "(phase, window, TDeff)")
        self.topspin_check.setToolTip(w.topspin_processing_checkbox.toolTip())
        self.topspin_check.setChecked(w.topspin_processing_checkbox.isChecked())
        form.addRow("", self.topspin_check)
        self.state_folder_edit = QLineEdit(w.state_folder_edit.text())
        form.addRow("Save states in:", self._with_browse(self.state_folder_edit,
                                                         "Folder for saved openNMR states"))
        self.iterations_spin = TypedSpinBox()
        self.iterations_spin.setRange(w.max_iterations_spin.minimum(), w.max_iterations_spin.maximum())
        self.iterations_spin.setSingleStep(500)
        self.iterations_spin.setGroupSeparatorShown(True)
        self.iterations_spin.setValue(w.max_iterations_spin.value())
        self.iterations_spin.setToolTip(w.max_iterations_spin.toolTip())
        form.addRow("Maximum fit iterations:", self.iterations_spin)
        updates = getattr(w, "updates_panel", None)
        self.update_check = QCheckBox("Check for a newer openNMR when it starts "
                                      "(at most once a day)")
        self.update_check.setChecked(bool(updates.startup_checkbox.isChecked())
                                     if updates is not None else False)
        self.update_check.setEnabled(updates is not None)
        form.addRow("Updates:", self.update_check)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _with_browse(self, edit: QLineEdit, caption: str) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(edit, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._browse(edit, caption))
        row_layout.addWidget(button)
        return row

    def _browse(self, edit: QLineEdit, caption: str) -> None:
        folder = QFileDialog.getExistingDirectory(self, caption, edit.text())
        if folder:
            edit.setText(folder)

    def _done_page(self) -> QWidget:
        page, layout = _page(
            "All set",
            "Press Apply and start to use these choices. They are remembered, and can be "
            "changed any time in Settings. Help ▸ Welcome to openNMR… brings this tour back, "
            "and Help ▸ openNMR website opens the project page.")
        layout.addStretch(1)
        return page

    # ---- navigation ------------------------------------------------------
    @property
    def page_count(self) -> int:
        return self.stack.count()

    def current_page(self) -> int:
        return self.stack.currentIndex()

    def _update_buttons(self) -> None:
        index, last = self.current_page(), self.page_count - 1
        self.back_button.setEnabled(index > 0)
        self.next_button.setText("Apply and start" if index == last else "Next")
        self.step_label.setText(f"{index + 1} of {self.page_count}")

    def go_back(self) -> None:
        if self.current_page() > 0:
            self.stack.setCurrentIndex(self.current_page() - 1)
        self._update_buttons()

    def go_next(self) -> None:
        if self.current_page() >= self.page_count - 1:
            self.apply()
            self.accept()
            self.list_data_folder()
            return
        self.stack.setCurrentIndex(self.current_page() + 1)
        self._update_buttons()

    def skip(self) -> None:
        """Close without changing anything."""
        self.reject()

    def list_data_folder(self) -> bool:
        """Once the wizard has closed, list the chosen default data folder
        in Open's browser straight away -- previously it only appeared the
        next time openNMR started (explicit user request, 2026-10-20).
        Honours the "List its experiments in Open's browser" tick, as
        start-up does; loads nothing. Returns whether a listing was
        scheduled."""
        from PySide6.QtCore import QTimer
        from . import data_directory
        w = self.window_
        if not (data_directory.default_data_directory() and self.browse_at_start_check.isChecked()):
            return False
        # After the dialog is gone: a big folder's scan must not hold the
        # wizard on screen.
        QTimer.singleShot(0, w, w._browse_default_data_folder)
        return True

    # ---- applying --------------------------------------------------------
    def apply(self) -> None:
        """Write the chosen values into the Settings panel's widgets, whose
        own handlers apply and remember them."""
        w = self.window_

        def pick(combo: QComboBox, mine: QComboBox) -> None:
            index = combo.findData(mine.currentData())
            if index >= 0 and index != combo.currentIndex():
                combo.setCurrentIndex(index)

        pick(w.ui_theme_combo, self.theme_combo)
        w.ui_font_size_spin.setValue(self.font_spin.value())
        pick(w.drag_mode_combo, self.drag_combo)
        w.double_click_reset_checkbox.setChecked(self.double_click_check.isChecked())
        w.contour_scroll_up_more_checkbox.setChecked(self.contour_scroll_check.isChecked())
        pick(w.export_unit_combo, self.export_unit_combo)
        color, width = w._all_canvases()[0].crosshair_style()
        if (matplotlib_to_hex(color) != self._crosshair_color
                or abs(width - self.crosshair_width_spin.value()) > 1e-9):
            w._set_crosshair_style(self._crosshair_color, self.crosshair_width_spin.value())
        if abs(w.zoom_box_width_spin.value() - self.zoom_box_width_spin.value()) > 1e-9:
            w.zoom_box_width_spin.setValue(self.zoom_box_width_spin.value())
        w.auto_apply_type_styles_checkbox.setChecked(self.auto_styles_check.isChecked())
        for mine, name in ((self.curve_top_check, "settings_curve_top_spine_checkbox"),
                           (self.curve_right_check, "settings_curve_right_spine_checkbox")):
            target = getattr(w, name, None)
            if target is not None:
                target.setChecked(mine.isChecked())
        updates = getattr(w, "updates_panel", None)
        if updates is not None:
            updates.startup_checkbox.setChecked(self.update_check.isChecked())
        for key, box in self.section_checks.items():
            target = w.section_shown_checkboxes.get(key)
            if target is not None:
                target.setChecked(box.isChecked())
        w.data_folder_browse_at_start_checkbox.setChecked(self.browse_at_start_check.isChecked())
        w.topspin_processing_checkbox.setChecked(self.topspin_check.isChecked())
        if self.data_folder_edit.text().strip() != w.data_folder_edit.text().strip():
            w._set_data_folder(self.data_folder_edit.text())
        if self.state_folder_edit.text().strip() != w.state_folder_edit.text().strip():
            w._set_state_folder(self.state_folder_edit.text())
        w.max_iterations_spin.setValue(self.iterations_spin.value())
        self.applied = True
