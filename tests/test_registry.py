"""Nucleus label formatting -- see opennmr.core.registry.NUC_label's
own docstring. This is the single implementation shared by the GUI
(via MplCanvas._chemical_shift_label) and every notebook-side
plotting function (via opennmr.notebook.utils.NUC_label, a thin
wrapper around this) -- one function to keep correct, not one to
keep two implementations in sync with by hand.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from opennmr.core.registry import NUC_label


def _renders_as_valid_mathtext(label: str) -> None:
    """NUC_label's whole point is to be handed straight to
    ax.set_xlabel -- a string matplotlib's mathtext parser rejects is
    not a usable label, however plausible it looks as text."""
    fig, ax = plt.subplots()
    ax.set_xlabel(label)
    fig.canvas.draw()
    plt.close(fig)


def test_brackets_is_the_default_style():
    # Explicit user request: "the axis labels should also be...
    # brackets... not a slash ppm".
    assert NUC_label("1H") == NUC_label("1H", style="brackets")


def test_brackets_style_is_fully_mathregular_and_bracketed():
    label = NUC_label("1H", x_unit="ppm", style="brackets")
    _renders_as_valid_mathtext(label)
    assert label == r"$\delta\ ^{\mathregular{1}}\mathregular{H}\ \mathregular{(ppm)}$"
    assert " / " not in label


def test_slash_style_matches_the_older_manuscript_form():
    label = NUC_label("1H", x_unit="ppm", style="slash")
    _renders_as_valid_mathtext(label)
    assert label == r"$\mathregular{\delta} \,\, ({}^{1}$H) / ppm"
    assert "(ppm)" not in label


@pytest.mark.parametrize("nuclide,mass,element", [
    ("1H", "1", "H"), ("13C", "13", "C"), ("17O", "17", "O"),
    ("27Al", "27", "Al"), ("119Sn", "119", "Sn"),
])
def test_both_styles_carry_the_mass_number_and_element(nuclide, mass, element):
    for style in ("brackets", "slash"):
        label = NUC_label(nuclide, style=style)
        _renders_as_valid_mathtext(label)
        assert mass in label
        assert element in label


def test_unrecorded_nucleus_gets_no_empty_superscript():
    """A dataset whose indirect-dimension nucleus was never populated
    labels as '?' rather than crashing -- and, since it has no mass
    number, gets no superscript at all rather than a stray empty
    ^{} mathtext renders as a raised nothing."""
    for style in ("brackets", "slash"):
        label = NUC_label(None, style=style)
        _renders_as_valid_mathtext(label)
        assert "?" in label
        assert "^{}" not in label
        assert "^{\\mathregular{}}" not in label


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        NUC_label("1H", style="parentheses")


def test_template_overrides_style_entirely():
    label = NUC_label("13C", x_unit="ppm", template="{element}-{mass} shift ({unit})")
    assert label == "C-13 shift (ppm)"


def test_template_takes_precedence_over_an_explicit_style():
    label = NUC_label("1H", style="slash", template="{nuc} ({unit})")
    assert label == "1H (ppm)"


def test_hz_unit_is_substituted_in_both_styles():
    assert "Hz" in NUC_label("1H", x_unit="Hz", style="brackets")
    assert "Hz" in NUC_label("1H", x_unit="Hz", style="slash")


def test_notebook_utils_delegates_to_the_same_implementation():
    """opennmr.notebook.utils.NUC_label used to be a second, drifted
    implementation (no style/slash option, and different mathtext
    wrapping around the element symbol) -- now a thin wrapper around
    this module's own NUC_label, so notebook and GUI labels can never
    disagree again."""
    from opennmr.notebook.utils import NUC_label as notebook_NUC_label

    for style in ("brackets", "slash"):
        assert notebook_NUC_label("1H", style=style) == NUC_label("1H", style=style)
    assert (notebook_NUC_label("13C", template="{element}-{mass}")
            == NUC_label("13C", template="{element}-{mass}"))
