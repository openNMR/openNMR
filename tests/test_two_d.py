"""
Tests for opennmr.notebook.two_d's inset support: the `inset` keyword
on NMR2Dplotter/NMR2D_overlay_plot (and, by composition, on
NMR2D_multi_plot's shared/per-panel kwargs and NMR2D_summary_plot's
`contour_kwargs`).

Built on synthetic in-memory dataset dicts that match NMR2dimport's
own return shape (same keys/array shapes every plotting function
actually reads) rather than real Bruker files -- see
test_bruker_raw_2d.py's own module docstring for the same rationale
(real code, synthetic data; no Bruker files are available in this
environment).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from opennmr.notebook import two_d as td


# ============================================================
# Synthetic dataset helper
# ============================================================

def _gaussian_peak_2d(n_f1, n_f2, f1c, f2c, sigma=2.5, amplitude=1000.0):
    F1, F2 = np.meshgrid(np.arange(n_f1), np.arange(n_f2), indexing="ij")
    return amplitude * np.exp(
        -(((F1 - f1c) ** 2 + (F2 - f2c) ** 2) / (2 * sigma ** 2))
    )


def _make_dataset(n_f1=32, n_f2=64, nuc1="13C", nuc2="1H",
                   homonuclear=False, pseudo_2d=False):
    """A minimal but structurally-real NMR2dimport-shaped dict -- the
    same keys/shapes _prepare_2d_contour/NMR2Dplotter/NMR2D_overlay_plot
    actually read, synthesised rather than read off disk."""

    x_ppm = np.linspace(10, 0, n_f2)
    y_ppm = np.zeros(n_f1) if pseudo_2d else np.linspace(200, 0, n_f1)
    y_index = np.arange(1, n_f1 + 1)
    nuc2_field = "off" if pseudo_2d else (nuc1 if homonuclear else nuc2)

    spectrum = (
        _gaussian_peak_2d(n_f1, n_f2, n_f1 // 2, n_f2 // 2, amplitude=1000.0)
        + _gaussian_peak_2d(n_f1, n_f2, n_f1 // 4, 3 * n_f2 // 4, amplitude=500.0)
    )

    return {
        "spectrum": spectrum,
        "x_ppm": x_ppm, "y_ppm": y_ppm, "y_index": y_index,
        "x_Hz": x_ppm * 400.13, "y_Hz": y_ppm * 100.6,
        "NUC1": nuc1, "NUC2": nuc2_field,
        "SF1": 400.13, "SF2": 100.6,
        "OBS1": 400.13, "OBS2": 100.6,
        "O1": 0.0, "O1_2": 0.0,
        "SW1": 8000.0, "SW2": 8000.0,
        "SI1": n_f2, "SI2": n_f1,
        "pseudo_2d": pseudo_2d,
        "procno": 1, "datapath": "<synthetic>", "homonuclear": homonuclear,
    }


def _assert_valid_insets(ax, n=1):
    """The inset(s) on `ax` exist and have a sane (positive-area) bbox
    once the figure has actually been laid out -- the regression this
    guards against is a silently negative-size inset (from asking
    matplotlib/AxesDivider to place or subdivide an inset before its
    parent's own layout has settled), not just "an axes got created"."""
    ax.figure.canvas.draw()
    assert len(ax.child_axes) == n
    for child in ax.child_axes:
        pos = child.get_position()
        assert pos.width > 0.01 and pos.height > 0.01, pos


@pytest.fixture
def dataset():
    return _make_dataset()


# ============================================================
# NMR2Dplotter
# ============================================================

def test_inset_basic_bounds_only(dataset):
    fig, ax = td.NMR2Dplotter(dataset, inset={"bounds": (0.55, 0.55, 0.4, 0.4)})
    _assert_valid_insets(ax, 1)


def test_inset_with_own_kwargs_and_mark_inset(dataset):
    fig, ax = td.NMR2Dplotter(
        dataset,
        inset={
            "bounds": (0.6, 0.05, 0.35, 0.35),
            "xlims": (6, 4), "ylims": (150, 100),
            "clevels": 5, "fontsize": 6,
            "mark_inset": True, "mark_inset_kwargs": {"edgecolor": "red"},
        },
    )
    _assert_valid_insets(ax, 1)
    inset_ax = ax.child_axes[0]
    assert inset_ax.get_xlim() == (6.0, 4.0)
    assert inset_ax.get_ylim() == (150.0, 100.0)


def test_inset_list_creates_multiple(dataset):
    fig, ax = td.NMR2Dplotter(
        dataset,
        inset=[{"bounds": (0.6, 0.6, 0.3, 0.3)}, {"bounds": (0.05, 0.05, 0.3, 0.3)}],
    )
    _assert_valid_insets(ax, 2)


def test_inset_defaults_to_parent_dataset(dataset):
    fig, ax = td.NMR2Dplotter(dataset, inset={"bounds": (0.6, 0.6, 0.3, 0.3)})
    _assert_valid_insets(ax, 1)
    inset_ax = ax.child_axes[0]
    # Same nuclide labels as the parent -> same axis label text.
    assert inset_ax.get_xlabel() == ax.get_xlabel()


def test_inset_can_show_a_different_dataset(dataset):
    other = _make_dataset(nuc1="15N", nuc2="1H")
    fig, ax = td.NMR2Dplotter(dataset, inset={"bounds": (0.6, 0.6, 0.3, 0.3), "dataset": other})
    _assert_valid_insets(ax, 1)
    inset_ax = ax.child_axes[0]
    assert inset_ax.get_xlabel() != ax.get_xlabel()


def test_inset_missing_bounds_raises(dataset):
    with pytest.raises(ValueError, match="bounds"):
        td.NMR2Dplotter(dataset, inset={"clevels": 5})


def test_inset_bad_bounds_length_raises(dataset):
    with pytest.raises(ValueError, match="bounds"):
        td.NMR2Dplotter(dataset, inset={"bounds": (0.1, 0.1, 0.3)})


@pytest.mark.parametrize("key", ["font", "output_format", "pl_width", "pl_height",
                                  "save_fig", "fig_name", "save_format", "ax", "fig"])
def test_inset_rejects_figure_level_keys(dataset, key):
    with pytest.raises(ValueError, match="figure-level"):
        td.NMR2Dplotter(dataset, inset={"bounds": (0.1, 0.1, 0.3, 0.3), key: object()})


def test_inset_defaults_show_projections_off(dataset):
    """An inset's own show_projections defaults to False -- the
    parent's projection_size (inches) is normally far larger than the
    inset's physical footprint, and asking the same AxesDivider
    machinery for more room than the inset has silently produces a
    negative-size axes rather than an error. Regression test for that:
    with no override, the inset should carry no projection strips of
    its own (no grandchild axes)."""
    fig, ax = td.NMR2Dplotter(dataset, inset={"bounds": (0.55, 0.55, 0.4, 0.4)})
    _assert_valid_insets(ax, 1)
    inset_ax = ax.child_axes[0]
    assert len(inset_ax.child_axes) == 0


@pytest.mark.parametrize("value", [True, "f1", "f2"])
def test_inset_show_projections_is_rejected(dataset, value):
    """show_projections inside an inset is refused outright (rather
    than silently mislaying the figure) -- see _draw_2d_insets's own
    docstring/comment for why: the AxesDivider mechanism projections
    rely on is not compatible with an inset's own axes-fraction-
    positioned axes."""
    with pytest.raises(ValueError, match="show_projections"):
        td.NMR2Dplotter(dataset, inset={"bounds": (0.55, 0.55, 0.4, 0.4),
                                         "show_projections": value})


def test_inset_survives_outer_projections_and_layout(dataset):
    """The common case: outer plot has its own projections (default
    show_projections=True), which resize the main axes via an
    AxesDivider -- the inset must still land with a sane, positive
    bbox once everything is laid out."""
    fig, ax = td.NMR2Dplotter(dataset, inset={"bounds": (0.03, 0.6, 0.35, 0.35)})
    _assert_valid_insets(ax, 1)


def test_inset_on_pseudo_2d(dataset):
    ds = _make_dataset(pseudo_2d=True)
    fig, ax = td.NMR2Dplotter(ds, inset={"bounds": (0.6, 0.6, 0.3, 0.3)})
    _assert_valid_insets(ax, 1)


def test_inset_included_in_saved_figure(dataset, tmp_path):
    out = tmp_path / "contour_with_inset"
    fig, ax = td.NMR2Dplotter(
        dataset, inset={"bounds": (0.55, 0.55, 0.35, 0.35)},
        save_fig=True, fig_name=str(out), save_format="png",
    )
    assert out.with_suffix(".png").exists()
    _assert_valid_insets(ax, 1)


def test_inset_none_by_default_no_child_axes(dataset):
    fig, ax = td.NMR2Dplotter(dataset)
    assert ax.child_axes == []


def test_default_fontsize_no_longer_crashes_standalone(dataset):
    """Regression test for a pre-existing bug: NMR2Dplotter's own
    default was fontsize=None, which it forwarded straight into
    setup_plotting's rcParams update, and matplotlib's fontsize
    validator rejects None outright -- so *any* standalone call with
    default arguments raised TypeError. Every sibling function in this
    module (NMR2D_overlay_plot, NMR2D_multi_plot, NMR2D_summary_plot)
    already defaults fontsize=12; NMR2Dplotter now matches them."""
    fig, ax = td.NMR2Dplotter(dataset)
    assert fig is not None and ax is not None


# ============================================================
# NMR2D_overlay_plot
# ============================================================

def test_overlay_inset_basic(dataset):
    other = _make_dataset()
    fig, ax = td.NMR2D_overlay_plot(
        [dataset, other], labels=["A", "B"], inset={"bounds": (0.6, 0.6, 0.3, 0.3)}
    )
    _assert_valid_insets(ax, 1)


def test_overlay_inset_defaults_to_parent_datasets(dataset):
    other = _make_dataset()
    fig, ax = td.NMR2D_overlay_plot([dataset, other], inset={"bounds": (0.6, 0.6, 0.3, 0.3)})
    _assert_valid_insets(ax, 1)


def test_overlay_inset_missing_bounds_raises(dataset):
    other = _make_dataset()
    with pytest.raises(ValueError, match="bounds"):
        td.NMR2D_overlay_plot([dataset, other], inset={"colours": ["blue", "red"]})


def test_overlay_inset_rejects_figure_level_keys(dataset):
    other = _make_dataset()
    with pytest.raises(ValueError, match="figure-level"):
        td.NMR2D_overlay_plot([dataset, other],
                               inset={"bounds": (0.1, 0.1, 0.3, 0.3), "font": "Times"})


# ============================================================
# Composition: NMR2D_multi_plot / NMR2D_summary_plot forward `inset`
# ============================================================

def test_multi_plot_forwards_inset_via_shared_kwargs(dataset):
    other = _make_dataset()
    fig, axes = td.NMR2D_multi_plot([dataset, other], inset={"bounds": (0.6, 0.6, 0.3, 0.3)})
    fig.canvas.draw()
    for ax in axes:
        assert len(ax.child_axes) == 1
        pos = ax.child_axes[0].get_position()
        assert pos.width > 0.01 and pos.height > 0.01


def test_multi_plot_forwards_inset_via_panel_kwargs(dataset):
    other = _make_dataset()
    fig, axes = td.NMR2D_multi_plot(
        [dataset, other],
        panel_kwargs=[{"inset": {"bounds": (0.6, 0.6, 0.3, 0.3)}}, {}],
    )
    fig.canvas.draw()
    assert len(axes[0].child_axes) == 1
    assert len(axes[1].child_axes) == 0


def test_summary_plot_forwards_inset_via_contour_kwargs(dataset):
    fig, axes = td.NMR2D_summary_plot(
        dataset, contour_kwargs={"inset": {"bounds": (0.6, 0.6, 0.3, 0.3)}},
    )
    fig.canvas.draw()
    contour_ax = axes["contour"]
    assert len(contour_ax.child_axes) == 1
    pos = contour_ax.child_axes[0].get_position()
    assert pos.width > 0.01 and pos.height > 0.01
