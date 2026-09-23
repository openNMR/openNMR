"""
Tests for opennmr.notebook.one_d's inset support: the `inset` keyword
on NMR1dplot/NMR1dheatmap, and the ax/fig embedding these two
functions gained specifically to support it (previously they always
created their own standalone figure, unlike the notebook/two_d.py
functions).

Built on a synthetic, on-disk Bruker 1D experiment directory --
NMR1dimport (unlike NMR2Dplotter, which accepts a plain dict) always
reads real files (acqus, pdata/<procno>/procs, .../1r, .../1i), so
there is no dict-bypass available here; see test_two_d.py's module
docstring for the same "real code, synthetic data" rationale (no
Bruker files are available in this environment).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from opennmr.notebook import one_d as od


# ============================================================
# Synthetic Bruker 1D experiment helper
# ============================================================

def _make_1d_experiment(root, nuc="1H", bf1=400.13, sfo1=400.132, o1=2000.0,
                         sw=8000.0, si=4096, sf=400.130, ns=16,
                         peak_idx_frac=0.5, amplitude=1.0e6):
    """Write a minimal but structurally-real Bruker experiment
    directory (acqus + pdata/1/procs + pdata/1/1r + pdata/1/1i) with a
    single Gaussian peak -- the same files/keys read_bruker_metadata
    and NMR1dimport actually parse, synthesised rather than read off
    disk (see test_bruker_raw_2d.py's own module docstring for the
    same "real code, synthetic data" rationale)."""

    root = str(root)
    os.makedirs(root, exist_ok=True)

    acqus = os.path.join(root, "acqus")
    with open(acqus, "w") as f:
        f.write(f"##$NUC1= <{nuc}>\n")
        f.write(f"##$BF1= {bf1}\n")
        f.write(f"##$SFO1= {sfo1}\n")
        f.write(f"##$O1= {o1}\n")
        f.write(f"##$NS= {ns}\n")
        f.write("##$P= (0..1)\n")
        f.write("10.0 5.0\n")

    pdata = os.path.join(root, "pdata", "1")
    os.makedirs(pdata, exist_ok=True)
    procs = os.path.join(pdata, "procs")
    with open(procs, "w") as f:
        f.write(f"##$SW_p= {sw}\n")
        f.write(f"##$SI= {si}\n")
        f.write(f"##$SF= {sf}\n")
        f.write("##$NC_proc= 0\n")

    x = np.arange(si)
    peak = amplitude * np.exp(-((x - int(si * peak_idx_frac)) ** 2) / (2 * 20.0 ** 2))
    peak.astype(np.int32).tofile(os.path.join(pdata, "1r"))
    (peak * 0.1).astype(np.int32).tofile(os.path.join(pdata, "1i"))

    return root


def _assert_valid_insets(ax, n=1):
    """The inset(s) on `ax` exist and have a sane (positive-area) bbox
    once the figure has actually been laid out (see test_two_d.py's
    identically-named helper for the regression this guards against)."""
    ax.figure.canvas.draw()
    assert len(ax.child_axes) == n
    for child in ax.child_axes:
        pos = child.get_position()
        assert pos.width > 0.01 and pos.height > 0.01, pos


def _bboxes_overlap(b1, b2):
    x0a, y0a, wa, ha = b1
    x0b, y0b, wb, hb = b2
    return not (x0a + wa <= x0b or x0b + wb <= x0a or y0a + ha <= y0b or y0b + hb <= y0a)


@pytest.fixture
def expt1(tmp_path):
    return _make_1d_experiment(tmp_path / "expt1", peak_idx_frac=0.5)


@pytest.fixture
def expt2(tmp_path):
    return _make_1d_experiment(tmp_path / "expt2", peak_idx_frac=0.3)


@pytest.fixture
def paths(expt1, expt2):
    return [expt1, expt2]


# ============================================================
# Baseline sanity (notebook/one_d.py has no prior test coverage at
# all -- these establish a working baseline before trusting the
# ax/fig refactor's own regression tests below).
# ============================================================

def test_import_reads_the_synthetic_experiment(expt1):
    results = od.NMR1dimport(expt1)
    assert results["nuclide"] == "1H"
    assert results["x_ppm"].shape == results["real_spectrum"].shape
    assert np.argmax(results["real_spectrum"]) == pytest.approx(4096 * 0.5, abs=2)


def test_plot_standalone_still_returns_a_new_figure_with_no_insets(paths):
    fig, ax = od.NMR1dplot(paths)
    assert fig is not None and ax is not None
    assert ax.child_axes == []


def test_heatmap_standalone_still_returns_a_new_figure(paths):
    fig, ax, im = od.NMR1dheatmap(paths)
    assert fig is not None and ax is not None and im is not None
    assert ax.child_axes == []


# ============================================================
# ax/fig embedding (the prerequisite refactor for insets -- NMR1dplot
# and NMR1dheatmap previously had no ax/fig parameters at all and
# always created their own figure unconditionally).
# ============================================================

def test_plot_can_be_embedded_in_a_caller_supplied_ax(paths):
    fig, axes = plt.subplots(1, 2)
    fig_out, ax_out = od.NMR1dplot(paths, ax=axes[1], fig=fig)
    assert fig_out is fig
    assert ax_out is axes[1]
    assert axes[1].lines  # something was actually drawn there
    assert not axes[0].lines  # and nothing leaked onto the other axes


def test_embedding_does_not_resize_or_save_the_shared_figure(paths, tmp_path):
    fig, ax = plt.subplots()
    fig.set_figheight(7.0)
    fig.set_figwidth(9.0)
    out = tmp_path / "should_not_exist"
    od.NMR1dplot(paths, ax=ax, fig=fig, pl_width=1.0, pl_height=1.0,
                 save_fig=True, fig_name=str(out))
    # pl_width/pl_height and save_fig are standalone-only concerns;
    # embedding must leave the caller's own figure alone.
    assert fig.get_figheight() == 7.0
    assert fig.get_figwidth() == 9.0
    assert not out.with_suffix(".png").exists()


def test_heatmap_can_be_embedded_in_a_caller_supplied_ax(paths):
    fig, axes = plt.subplots(1, 2)
    fig_out, ax_out, im = od.NMR1dheatmap(paths, ax=axes[0], fig=fig)
    assert fig_out is fig
    assert ax_out is axes[0]
    assert im is not None


def test_fig_is_inferred_from_ax_when_not_given(paths):
    fig, ax = plt.subplots()
    fig_out, ax_out = od.NMR1dplot(paths, ax=ax)
    assert fig_out is fig


# ============================================================
# NMR1dplot insets
# ============================================================

def test_inset_basic_bounds_only(paths):
    fig, ax = od.NMR1dplot(paths, inset={"bounds": (0.55, 0.55, 0.4, 0.4)})
    _assert_valid_insets(ax, 1)


def test_inset_none_by_default_no_child_axes(paths):
    fig, ax = od.NMR1dplot(paths)
    assert ax.child_axes == []


def test_inset_with_own_kwargs_and_mark_inset(paths):
    fig, ax = od.NMR1dplot(
        paths,
        inset={
            "bounds": (0.6, 0.05, 0.35, 0.35),
            "xlims": (12, -2), "colour": "diverging", "fontsize": 6,
            "mark_inset": True, "mark_inset_kwargs": {"edgecolor": "red"},
        },
    )
    _assert_valid_insets(ax, 1)
    inset_ax = ax.child_axes[0]
    assert inset_ax.get_xlim() == (12.0, -2.0)


def test_inset_list_creates_multiple(paths):
    fig, ax = od.NMR1dplot(
        paths,
        inset=[{"bounds": (0.6, 0.6, 0.3, 0.3)}, {"bounds": (0.05, 0.6, 0.3, 0.3)}],
    )
    _assert_valid_insets(ax, 2)


def test_inset_defaults_to_parent_data_paths(paths, expt1):
    fig, ax = od.NMR1dplot(paths, inset={"bounds": (0.6, 0.6, 0.3, 0.3)})
    _assert_valid_insets(ax, 1)
    inset_ax = ax.child_axes[0]
    # Same data_paths (both experiments) -> same number of lines as the parent.
    assert len(inset_ax.lines) == len(ax.lines) == 2


def test_inset_can_show_different_data_paths(paths, expt1):
    fig, ax = od.NMR1dplot(paths, inset={"bounds": (0.6, 0.6, 0.3, 0.3), "data_paths": expt1})
    _assert_valid_insets(ax, 1)
    inset_ax = ax.child_axes[0]
    assert len(ax.lines) == 2
    assert len(inset_ax.lines) == 1


def test_inset_settings_are_independent_of_the_parents(paths):
    """The user's explicit requirement: main and inset must be able to
    take *different* settings, not just placement/size -- fontsize is
    the sharpest test of this, since NMR1dplot previously relied
    entirely on global rcParams (which cannot differ between two axes
    on the same figure) rather than a per-axes override."""
    fig, ax = od.NMR1dplot(paths, fontsize=20,
                            inset={"bounds": (0.55, 0.55, 0.4, 0.4), "fontsize": 6})
    fig.canvas.draw()
    inset_ax = ax.child_axes[0]
    assert ax.xaxis.label.get_fontsize() == 20
    assert inset_ax.xaxis.label.get_fontsize() == 6
    assert ax.xaxis.get_ticklabels()[0].get_fontsize() == 20
    assert inset_ax.xaxis.get_ticklabels()[0].get_fontsize() == 6


def test_inset_missing_bounds_raises(paths):
    with pytest.raises(ValueError, match="bounds"):
        od.NMR1dplot(paths, inset={"colour": "diverging"})


def test_inset_bad_bounds_length_raises(paths):
    with pytest.raises(ValueError, match="bounds"):
        od.NMR1dplot(paths, inset={"bounds": (0.1, 0.1, 0.3)})


@pytest.mark.parametrize("key", ["font", "output_format", "pl_width", "pl_height",
                                  "save_fig", "fig_name", "save_format", "ax", "fig"])
def test_inset_rejects_figure_level_keys(paths, key):
    with pytest.raises(ValueError, match="figure-level"):
        od.NMR1dplot(paths, inset={"bounds": (0.1, 0.1, 0.3, 0.3), key: object()})


def test_inset_included_in_saved_figure(paths, tmp_path):
    out = tmp_path / "oned_with_inset"
    fig, ax = od.NMR1dplot(
        paths, inset={"bounds": (0.55, 0.55, 0.35, 0.35)},
        save_fig=True, fig_name=str(out), save_format="png",
    )
    assert out.with_suffix(".png").exists()
    _assert_valid_insets(ax, 1)


def test_inset_survives_vocs_sum_mode(paths):
    """The VOCS-summed branch takes a different code path up to the
    point where the axis is finalised; the inset-drawing block sits
    after both branches converge, so it must work here too."""
    fig, ax = od.NMR1dplot(paths, vocs_sum=True,
                            inset={"bounds": (0.55, 0.55, 0.35, 0.35)})
    _assert_valid_insets(ax, 1)


def test_inset_can_itself_be_a_heatmap(paths):
    """`inset` forwards every non-reserved key straight into a
    recursive NMR1dplot call, so `heatmap=True` inside an inset spec
    must dispatch to NMR1dheatmap exactly like a top-level call would."""
    fig, ax = od.NMR1dplot(paths, inset={"bounds": (0.55, 0.55, 0.4, 0.4), "heatmap": True})
    _assert_valid_insets(ax, 1)
    heatmap_inset = ax.child_axes[0]
    assert len(heatmap_inset.collections) >= 1  # the pcolormesh


def test_top_level_heatmap_forwards_inset_to_nmr1dheatmap(paths):
    """NMR1dplot(heatmap=True, inset=...) must hand `inset` down to
    NMR1dheatmap rather than silently dropping it on dispatch."""
    fig, ax, im = od.NMR1dplot(paths, heatmap=True, inset={"bounds": (0.6, 0.6, 0.3, 0.3)})
    _assert_valid_insets(ax, 1)


# ============================================================
# NMR1dheatmap insets, including the colorbar/locator regression
# ============================================================

def test_heatmap_inset_basic(paths):
    fig, ax, im = od.NMR1dheatmap(paths, inset={"bounds": (0.55, 0.55, 0.4, 0.4)})
    _assert_valid_insets(ax, 1)


def test_heatmap_with_colorbar_and_inset(paths):
    """The outer plot's own colorbar (attached to the *standalone*,
    non-locator-controlled ax) was never at risk -- this just confirms
    it keeps working once an inset is added alongside it."""
    fig, ax, im = od.NMR1dheatmap(paths, colorbar=True,
                                   inset={"bounds": (0.55, 0.05, 0.35, 0.35)})
    _assert_valid_insets(ax, 1)
    # inset_axes() deliberately does not register itself on fig.axes
    # (only ever on its parent's child_axes) -- the colorbar's own new
    # axes is the only thing that should have been added to fig.axes.
    assert len(fig.axes) == 2  # main heatmap ax + its colorbar ax


def test_heatmap_inset_with_its_own_colorbar_does_not_overlap(paths):
    """Regression test for a real bug found while building this: an
    inset axes is positioned via Axes.inset_axes' own transAxes-based
    locator, and fig.colorbar(im, ax=<that axes>) tries to shrink the
    axes in place via set_position() to make room -- a shrink the
    axes' own locator silently discards on every redraw, leaving the
    colorbar drawn on top of the right-hand edge of the inset instead
    of beside it. Exercised here via NMR1dplot's `inset=` dict (the
    common path: heatmap=True + heatmap_colorbar=True inside an inset
    spec)."""
    fig, ax = od.NMR1dplot(
        paths,
        inset={"bounds": (0.55, 0.55, 0.4, 0.4), "heatmap": True, "heatmap_colorbar": True},
    )
    fig.canvas.draw()

    heatmap_inset = next(c for c in ax.child_axes if c.collections)
    cax = heatmap_inset.child_axes[-1]

    pos_inset = heatmap_inset.get_position()
    pos_cax = cax.get_position()
    assert pos_inset.width > 0.01 and pos_inset.height > 0.01
    assert pos_cax.width > 0.001 and pos_cax.height > 0.001
    assert not _bboxes_overlap(pos_inset.bounds, pos_cax.bounds)
    # The colorbar should sit to the right of the inset it belongs to,
    # not somewhere unrelated on the figure.
    assert pos_cax.x0 >= pos_inset.x0 + pos_inset.width - 1e-9


def test_heatmap_colorbar_direct_call_on_an_inset_axes_does_not_overlap(paths):
    """Same regression as above, but via the other public entry point
    the ax/fig docstring explicitly advertises: building the inset
    axes by hand and calling NMR1dheatmap(ax=..., colorbar=True)
    directly, without going through NMR1dplot's `inset=` dict at all."""
    fig, ax = od.NMR1dplot(paths)
    inset_ax = ax.inset_axes((0.1, 0.1, 0.35, 0.35))
    fig2, ax2, im = od.NMR1dheatmap(paths, ax=inset_ax, fig=fig, colorbar=True)
    fig.canvas.draw()

    assert ax2 is inset_ax
    cax = inset_ax.child_axes[-1]
    pos_inset = inset_ax.get_position()
    pos_cax = cax.get_position()
    assert not _bboxes_overlap(pos_inset.bounds, pos_cax.bounds)


def test_standalone_heatmap_colorbar_placement_is_unaffected(paths):
    """The fix branches on whether `ax` has a locator at all
    (`ax.get_axes_locator() is not None`); a plain standalone/subplot
    axes has none, so it must keep using fig.colorbar's ordinary
    ax= path rather than the inset-only cax workaround."""
    fig, ax, im = od.NMR1dheatmap(paths, colorbar=True)
    assert ax.get_axes_locator() is None
    # Exactly one extra axes (the colorbar) was added to the figure.
    assert len(fig.axes) == 2


def test_heatmap_inset_missing_bounds_raises(paths):
    with pytest.raises(ValueError, match="bounds"):
        od.NMR1dheatmap(paths, inset={"colorbar": True})


def test_heatmap_inset_rejects_figure_level_keys(paths):
    with pytest.raises(ValueError, match="figure-level"):
        od.NMR1dheatmap(paths, inset={"bounds": (0.1, 0.1, 0.3, 0.3), "font": "Times"})


def test_heatmap_inset_can_show_different_data_paths(paths, expt1):
    fig, ax, im = od.NMR1dheatmap(
        paths, inset={"bounds": (0.6, 0.6, 0.3, 0.3), "data_paths": expt1},
    )
    _assert_valid_insets(ax, 1)


def test_heatmap_inset_included_in_saved_figure(paths, tmp_path):
    out = tmp_path / "heatmap_with_inset"
    fig, ax, im = od.NMR1dheatmap(
        paths, inset={"bounds": (0.55, 0.55, 0.35, 0.35)},
        save_fig=True, fig_name=str(out), save_format="png",
    )
    assert out.with_suffix(".png").exists()
    _assert_valid_insets(ax, 1)
