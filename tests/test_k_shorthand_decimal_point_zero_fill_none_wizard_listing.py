"""2026-10-20 round: items 1, 2, 3 and 7.

1. The Welcome wizard lists the chosen default data folder in Open's
   browser as soon as it closes (not only at the next start-up).
2. Zero-fill has a "None (do not zero-fill)" choice.
3. Whole-number boxes read ``2k`` as 2048, ``8k`` as 8192, ``16k`` as
   16384 (the power of two nearest the thousands typed), like TopSpin.
7. A decimal point is a full stop in every number box, whatever the
   computer's regional settings.
"""

import numpy as np
import pytest
from PySide6.QtCore import QLocale
from PySide6.QtGui import QValidator
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox

from opennmr.gui import data_directory
from opennmr.gui import main_window as mw
from opennmr.gui.main_window import MainWindow
from opennmr.gui.numeric_locale import LOCALE, use_decimal_point
from opennmr.gui.welcome_wizard import WelcomeWizard
from opennmr.gui.widgets.typed_spinbox import TypedDoubleSpinBox, TypedSpinBox, expand_k

from test_gui import _make_raw_1d_dataset, qapp  # noqa: F401 -- qapp is a fixture


@pytest.fixture(autouse=True)
def _restore_locale():
    before = QLocale().name()
    yield
    QLocale.setDefault(QLocale(before))


def _type(spin, text):
    """What typing ``text`` over the current entry and pressing Enter does."""
    spin.show()
    spin.lineEdit().setFocus()
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin.lineEdit(), text)
    spin.interpretText()


# ---- 3. the k shorthand -----------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("1k", 1024), ("2k", 2048), ("4k", 4096), ("8k", 8192), ("16k", 16384), ("32k", 32768),
    ("64K", 65536), ("128 k", 131072), (" 2k ", 2048),
    ("3k", 4096),            # 3000 is nearer 4096 than 2048 on a logarithmic scale
    ("2.5k", 2048), ("0.5k", 512), ("1.5k", 2048),
])
def test_expand_k_gives_the_nearest_power_of_two(text, expected):
    assert expand_k(text) == expected


@pytest.mark.parametrize("text", ["", "k", "2", "2kk", "k2", "0k", "-2k", "2 k pts", "abc", "2e3k"])
def test_expand_k_ignores_anything_else(text):
    assert expand_k(text) is None


def test_a_whole_number_box_reads_k(qapp):  # noqa: F811
    spin = TypedSpinBox()
    spin.setRange(1, 1 << 22)
    for text, expected in (("2k", 2048), ("4k", 4096), ("16k", 16384), ("8K", 8192)):
        _type(spin, text)
        assert spin.value() == expected
        assert spin.text() == str(expected)


def test_k_is_clamped_to_the_box_range_and_plain_numbers_still_work(qapp):  # noqa: F811
    spin = TypedSpinBox()
    spin.setRange(1, 5000)
    _type(spin, "16k")
    assert spin.value() == 5000
    _type(spin, "1234")
    assert spin.value() == 1234
    _type(spin, "12.6")                     # not k: rounded, not refused
    assert spin.value() == 13


def test_k_is_accepted_while_typing_and_a_suffix_box_works(qapp):  # noqa: F811
    spin = TypedSpinBox()
    spin.setRange(1, 1 << 20)
    spin.setSuffix(" pts")
    state, _text, _pos = spin.validate("2k pts", 0)
    assert state == QValidator.State.Acceptable
    assert spin.validate("2.", 0)[0] == QValidator.State.Intermediate      # on the way to 2.5k
    assert spin.validate("k", 0)[0] == QValidator.State.Invalid
    _type(spin, "8k pts")
    assert spin.value() == 8192


def test_special_value_text_still_works_in_the_typed_box(qapp):  # noqa: F811
    spin = TypedSpinBox()
    spin.setRange(0, 100)
    spin.setSpecialValueText("auto")
    spin.setValue(0)
    assert spin.text() == "auto"
    assert spin.validate("auto", 0)[0] == QValidator.State.Acceptable
    assert spin.validate("au", 0)[0] == QValidator.State.Intermediate
    _type(spin, "7")
    assert spin.value() == 7
    _type(spin, "auto")
    assert spin.value() == 0


def test_every_whole_number_box_in_the_gui_reads_k(qapp):  # noqa: F811
    window = MainWindow()
    plain = [w for w in window.findChildren(QSpinBox) if not isinstance(w, TypedSpinBox)]
    # Only boxes Qt itself builds (a QDoubleSpinBox's inner line edit, the
    # table editors) may be left over -- none of openNMR's own.
    assert plain == []
    assert window.output_dpi_spin.maximum() >= 1200
    window.zero_fill_custom_spin.setRange(1, 1 << 22)
    _type(window.zero_fill_custom_spin, "16k")
    assert window.zero_fill_custom_spin.value() == 16384


def test_decimal_boxes_do_not_read_k(qapp):  # noqa: F811
    spin = TypedDoubleSpinBox()
    spin.setRange(0, 1e6)
    _type(spin, "2k")
    assert spin.value() != 2048


# ---- 2. no zero-fill -------------------------------------------------------------------

def _raw_window(tmp_path, monkeypatch, n=64):
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=n)
    window = MainWindow()
    window._load(path, vendor="bruker")
    return window


def test_zero_fill_offers_none_and_it_means_the_points_transformed(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = _raw_window(tmp_path, monkeypatch)
    combo = window.zero_fill_combo
    index = combo.findData(mw._NO_ZERO_FILL)
    assert index >= 0 and "do not zero-fill" in combo.itemText(index).lower()

    acquired = window._truncate_acquired_points
    assert acquired
    combo.setCurrentIndex(index)
    assert window._zero_fill_value() == acquired
    spectrum = window.session.spectra[window._selected_index()]
    assert spectrum.data.shape[-1] == acquired            # the FFT is exactly as long as the FID

    # Truncating shortens it; linear prediction lengthens it -- "none" follows both.
    window.tdeff_spin.setValue(acquired // 2)
    assert window._zero_fill_value() == acquired // 2
    window.linear_predict_points_spin.setValue(4)
    assert window._zero_fill_value() == acquired // 2 + 4

    # ...and a real size is a real size again.
    combo.setCurrentIndex(combo.findData(1024))
    window.linear_predict_points_spin.setValue(0)
    window.tdeff_spin.setValue(acquired)
    assert window._zero_fill_value() == 1024


def test_none_is_remembered_as_a_choice_not_a_size(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = _raw_window(tmp_path, monkeypatch)
    window._set_zero_fill_value(mw._NO_ZERO_FILL)
    assert window.zero_fill_combo.currentData() == mw._NO_ZERO_FILL
    assert window._zero_fill_choice() == mw._NO_ZERO_FILL
    assert window._zero_fill_value() > 0


def test_ft_zero_fill_zero_is_a_no_op():
    from opennmr.processing import ft
    fid = np.arange(10, dtype=complex)
    assert ft.zero_fill(fid, 0) is fid
    assert ft.zero_fill(fid, None) is fid
    assert len(ft.zero_fill(fid, 16)) == 16 and len(ft.zero_fill(fid, 4)) == 4


def test_bruker_read_1d_zero_fill_zero_does_not_zero_fill(tmp_path, monkeypatch):
    from opennmr.io import bruker
    path = _make_raw_1d_dataset(tmp_path, monkeypatch, n=256)
    padded = bruker.read_1d(path, autophase=False, zero_fill=1024)
    unpadded = bruker.read_1d(path, autophase=False, zero_fill=0)
    assert len(padded.data) == 1024
    assert len(unpadded.data) == len(padded.metadata.raw["fid_raw"])
    assert len(unpadded.axes[0].ppm) == len(unpadded.data)


# ---- 7. the decimal point ---------------------------------------------------------------

@pytest.mark.parametrize("language, country", [
    (QLocale.Language.French, QLocale.Country.France),
    (QLocale.Language.German, QLocale.Country.Germany),
    (QLocale.Language.Spanish, QLocale.Country.Spain),
])
def test_number_boxes_always_use_a_full_stop(qapp, language, country):  # noqa: F811
    QLocale.setDefault(QLocale(language, country))
    assert QLocale().decimalPoint() == ","           # the situation on a continental computer
    window = MainWindow()                             # sets the locale itself
    assert QLocale().decimalPoint() == "."
    spin = window.output_linewidth_spin
    spin.setValue(1.5)
    assert "," not in spin.text() and "." in spin.text()
    boxes = [w for w in window.findChildren(QDoubleSpinBox) if w.decimals() > 0 and w.isEnabled()][:40]
    assert boxes
    for box in boxes:
        box.setValue(min(max(1.5, box.minimum()), box.maximum()))
        assert "," not in box.text(), f"{box.objectName() or box}: {box.text()!r}"


def test_use_decimal_point_is_idempotent_and_returns_the_locale(qapp):  # noqa: F811
    QLocale.setDefault(QLocale(QLocale.Language.French, QLocale.Country.France))
    first = use_decimal_point()
    second = use_decimal_point()
    assert first.name() == second.name() == LOCALE.name()
    assert QLocale().toString(1.5) == "1.5"


def test_a_comma_typed_into_a_decimal_box_is_still_read_as_a_point(qapp):  # noqa: F811
    spin = TypedDoubleSpinBox()
    spin.setRange(0, 100)
    spin.setDecimals(2)
    _type(spin, "2,5")
    assert spin.value() == pytest.approx(2.5)
    assert "," not in spin.text()


# ---- 1. the wizard lists the folder when it closes ----------------------------------------------

def _finish_wizard(window, folder, browse=True):
    wizard = WelcomeWizard(window)
    wizard.data_folder_edit.setText(str(folder))
    wizard.browse_at_start_check.setChecked(browse)
    while wizard.current_page() < wizard.page_count - 1:
        wizard.go_next()
    wizard.go_next()                                       # Apply and start
    QApplication.processEvents()
    return wizard


def test_the_chosen_folder_is_listed_straight_after_the_wizard(qapp, tmp_path, monkeypatch):  # noqa: F811
    data = tmp_path / "data"
    data.mkdir()
    path = _make_raw_1d_dataset(data, monkeypatch, n=32)
    window = MainWindow()
    listed = []
    monkeypatch.setattr(window.data_browser, "set_matches",
                        lambda directory, matches, **k: listed.append((directory, matches)))
    wizard = _finish_wizard(window, data)
    assert wizard.applied
    assert data_directory.default_data_directory() == str(data)
    assert listed, "the folder was not listed when the wizard closed"
    assert listed[-1][0] == str(data)
    assert len(listed[-1][1]) == 1 and "rawexpt1" in str(listed[-1][1][0])


def test_the_wizard_does_not_list_if_the_start_up_listing_is_off_or_no_folder(qapp, tmp_path, monkeypatch):  # noqa: F811
    data = tmp_path / "data"
    data.mkdir()
    window = MainWindow()
    listed = []
    monkeypatch.setattr(window, "_browse_default_data_folder", lambda: listed.append(1))
    _finish_wizard(window, data, browse=False)
    assert listed == []
    other = MainWindow()
    monkeypatch.setattr(other, "_browse_default_data_folder", lambda: listed.append(2))
    wizard = WelcomeWizard(other)
    wizard.data_folder_edit.setText("")
    while wizard.current_page() < wizard.page_count - 1:
        wizard.go_next()
    wizard.go_next()
    QApplication.processEvents()
    assert listed == []


def test_skip_lists_nothing(qapp, tmp_path, monkeypatch):  # noqa: F811
    window = MainWindow()
    listed = []
    monkeypatch.setattr(window, "_browse_default_data_folder", lambda: listed.append(1))
    wizard = WelcomeWizard(window)
    wizard.data_folder_edit.setText(str(tmp_path))
    wizard.skip()
    QApplication.processEvents()
    assert listed == []
