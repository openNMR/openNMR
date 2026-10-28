"""Panel labels (a), (b)... : styles, start, reading order, placement and alignment.
Synthetic data only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from opennmr.notebook import panel_labels as pl


@pytest.mark.parametrize("style,expected", [
    ("(a)", ["(a)", "(b)", "(c)"]), ("a)", ["a)", "b)", "c)"]), ("a.", ["a.", "b.", "c."]),
    ("a", ["a", "b", "c"]), ("A", ["A", "B", "C"]), ("A.", ["A.", "B.", "C."]),
    ("A)", ["A)", "B)", "C)"]), ("(A)", ["(A)", "(B)", "(C)"]),
    ("I", ["I", "II", "III"]), ("I.", ["I.", "II.", "III."]),
    ("I)", ["I)", "II)", "III)"]), ("(I)", ["(I)", "(II)", "(III)"]),
])
def test_every_requested_style(style, expected):
    assert [pl.label_text(k, style) for k in (1, 2, 3)] == expected


def test_unknown_style_and_beyond_z():
    with pytest.raises(ValueError):
        pl.label_text(1, "[a]")
    assert pl.label_text(27, "(a)") == "(aa)"
    assert pl.label_text(4, "I") == "IV"


def test_start_can_be_a_letter_a_label_or_a_number():
    assert pl.parse_start("b") == 2 and pl.parse_start("(b)") == 2 and pl.parse_start(3) == 3
    assert pl.parse_start("iv", "I") == 4 and pl.parse_start("(C)", "(A)") == 3
    with pytest.raises(ValueError):
        pl.parse_start(0)
    with pytest.raises(ValueError):
        pl.parse_start("iiii", "I")


def test_reading_orders():
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]            # a 2 x 2 grid in row order
    assert pl.label_numbers(cells, "row") == [1, 2, 3, 4]
    assert pl.label_numbers(cells, "column") == [1, 3, 2, 4]   # down the first column first
    assert pl.label_numbers(cells, "row", start=2) == [2, 3, 4, 5]
    assert pl.label_numbers(cells, [2, 1, 4, 3]) == [2, 1, 4, 3]
    assert pl.label_numbers(cells, [1, None, 2, 3]) == [1, None, 2, 3]
    with pytest.raises(ValueError):
        pl.label_numbers(cells, [1, 2])


def _grid(nrows=2, ncols=3):
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 5))
    for ax in axes.flat:
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("δ (ppm)")
        ax.set_ylabel("I")
    return fig, list(axes.flat)


def test_labels_are_added_in_the_chosen_order_from_the_chosen_start():
    fig, axes = _grid()
    labels = pl.add_panel_labels(fig, axes, style="(a)", start="b", order="column", layout=fig.tight_layout)
    assert [t.get_text() for t in labels.texts] == ["(b)", "(d)", "(f)", "(c)", "(e)", "(g)"]
    plt.close(fig)


def _boxes(fig, labels):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    return [t.get_window_extent(r) for t in labels.texts]


def test_automatic_labels_line_up_and_clear_the_tick_labels():
    fig, axes = _grid()
    labels = pl.add_panel_labels(fig, axes, layout=fig.tight_layout)
    boxes = _boxes(fig, labels)
    for row in (boxes[:3], boxes[3:]):                  # one baseline per row
        assert max(b.y0 for b in row) - min(b.y0 for b in row) < 0.75
    for col in range(3):                                # one left edge per column
        assert abs(boxes[col].x0 - boxes[col + 3].x0) < 0.75
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for box, ax in zip(boxes, axes):
        for tick in ax.get_yticklabels() + ax.get_xticklabels():
            if tick.get_visible() and tick.get_text():
                assert not box.overlaps(tick.get_window_extent(r))
    plt.close(fig)


def test_manual_placement_is_exact_and_survives_a_relayout():
    fig, axes = _grid(1, 2)
    labels = pl.add_panel_labels(fig, axes, placement="manual", positions={0: (10, 6), 1: (-4, 2)},
                                 layout=fig.tight_layout)
    assert labels.offsets[0] == (10.0, 6.0) and labels.offsets[1] == (-4.0, 2.0)
    fig.tight_layout()
    labels.place()
    assert labels.offsets[0] == (10.0, 6.0)
    plt.close(fig)


def test_all_key_moves_every_label_together():
    fig, axes = _grid(1, 3)
    labels = pl.add_panel_labels(fig, axes, placement="manual", positions={"all": (5, 5)})
    labels.place()
    assert all(o == (5.0, 5.0) for o in labels.offsets)
    plt.close(fig)


def test_a_second_call_replaces_the_first_set_and_remove_clears():
    fig, axes = _grid(1, 2)
    first = pl.add_panel_labels(fig, axes)
    second = pl.add_panel_labels(fig, axes, style="A")
    assert fig._opennmr_panel_labels is second
    assert all(t not in axes[0].texts for t in first.texts if t is not None)
    second.remove()
    assert not [t for ax in axes for t in ax.texts if getattr(t, "_opennmr_panel_label", False)]
    plt.close(fig)


def test_bad_arguments_are_reported():
    fig, axes = _grid(1, 2)
    with pytest.raises(ValueError):
        pl.add_panel_labels(fig, axes, placement="sideways")
    with pytest.raises(ValueError):
        pl.add_panel_labels(fig, [])
    plt.close(fig)
