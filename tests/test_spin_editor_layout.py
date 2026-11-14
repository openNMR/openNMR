"""Simulate ▸ Solution NMR: no gap between the spin-system editor's tables
(user report, 2026-11-14: "a huge gap between different tables"). The
model pages share a stack, which took its tallest page's height and
stretched the shorter ones."""

from opennmr.gui.jcoupling_panels import SpinSystemEditor
from opennmr.gui.main_window import MainWindow

from test_gui import qapp  # noqa: F401 -- fixture


def test_spare_height_goes_below_the_tables_not_between_them(qapp):  # noqa: F811
    editor = SpinSystemEditor("simulate")
    editor.add_spin()
    editor.add_spin()
    editor.resize(420, 2000)                       # far taller than it needs
    editor.layout().activate()
    spins, j = editor.spins_table, editor.j_table
    # Buttons and the "J (Hz) between spins:" label sit between them: no more.
    between = j.geometry().top() - spins.geometry().bottom()
    assert 0 < between < 120, between
    assert editor.info_label.geometry().top() - j.geometry().bottom() < 40
    assert j.geometry().bottom() < 700             # the rest is below


def test_the_model_stack_is_as_tall_as_the_page_on_show(qapp):  # noqa: F811
    window = MainWindow()
    for panel in (window.jcoupling_sim_panel, window.jcoupling_fit_panel):
        stack = panel.model_stack
        heights = []
        for index in range(stack.count()):
            panel.model_combo.setCurrentIndex(index)
            assert stack.currentIndex() == index
            # The stack asks for the page on show, not its tallest page --
            # also by height-for-width, which wrapped text brings in.
            page = stack.currentWidget()
            assert stack.sizeHint().height() == page.sizeHint().height()
            assert stack.minimumSizeHint().height() == page.minimumSizeHint().height()
            assert stack.hasHeightForWidth() == page.hasHeightForWidth()
            expected = (page.heightForWidth(300) if page.hasHeightForWidth()
                        else page.sizeHint().height())
            assert stack.heightForWidth(300) == expected
            heights.append(stack.sizeHint().height())
        # The pages really do differ: the tallest no longer sets them all.
        assert len(set(heights)) > 1
    window.close()
