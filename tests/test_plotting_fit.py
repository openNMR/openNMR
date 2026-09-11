import matplotlib.pyplot as plt
import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.plotting import fit as fitplot
from opennmr.processing import fitting, relaxation, slicing
from opennmr.processing.fitting import gaussian


def _make_single_gaussian_slice(x0=5.0, fwhm=0.05, amplitude=1.0, n_pts=800):
    """A one-slice pseudo-2D wrapping a single Gaussian -- same shape
    as test_fitting.py's own helper of the same name, duplicated
    locally per this project's established cross-file-fixture
    convention."""
    ppm = np.linspace(3.0, 7.0, n_pts)
    y = gaussian(ppm, x0, fwhm, amplitude)
    f1_axis = NMRAxis(kind="index", index=np.array([1]))
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    return NMRSpectrum(data=y[np.newaxis, :], axes=[f1_axis, f2_axis], metadata=metadata)


def _fit_one_slice(x0=5.0, fwhm=0.05, amplitude=1.0, **kwargs):
    spectrum = _make_single_gaussian_slice(x0=x0, fwhm=fwhm, amplitude=amplitude)
    slices = slicing.get_slices(spectrum, rows=[1])
    peak_defs = [{"type": "gaussian", "shift": x0, "fwhm": fwhm}]
    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows", show_report=False, show_warnings=False, **kwargs)
    return fitted["rows"][0]


def _make_pseudo2d_relaxation(delays, M0=1.0, T1=0.8, C=2.0, x0=5.0, fwhm=0.05, b0=0.005, n_pts=1000):
    """Same fixture as test_relaxation.py's own -- a real T1-inversion-
    recovery pseudo-2D series, duplicated locally per this project's
    own established cross-file-fixture convention."""
    ppm = np.linspace(3.0, 7.0, n_pts)
    n_rows = len(delays)
    data = np.zeros((n_rows, n_pts))
    for i, t in enumerate(delays):
        A = relaxation.inversion_recovery(t, M0, T1, C)
        data[i] = gaussian(ppm, x0, fwhm, A) + b0
    f1_axis = NMRAxis(kind="index", index=np.arange(1, n_rows + 1))
    f2_axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True)
    return NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)


# ============================================================
# plot_1d_with_fit
# ============================================================

def test_plot_1d_with_fit_draws_data_and_fit_tracking_each_other():
    result = _fit_one_slice()
    fig, ax = fitplot.plot_1d_with_fit(result)
    data_line, fit_line = ax.lines[0], ax.lines[1]
    # The fit should closely track the data everywhere, not just
    # nominally share an axes -- a real correctness check, not just
    # "did it render something".
    np.testing.assert_allclose(fit_line.get_ydata(), data_line.get_ydata(), atol=0.02)
    assert data_line.get_label() == "Data"
    assert fit_line.get_label() == "Fit"
    plt.close(fig)


def test_plot_1d_with_fit_show_components_toggle():
    result = _fit_one_slice()

    fig_on, ax_on = fitplot.plot_1d_with_fit(result, show_components=True)
    fig_off, ax_off = fitplot.plot_1d_with_fit(result, show_components=False)
    # One peak -> one component line, on top of Data+Fit.
    assert len(ax_on.lines) == 3
    assert len(ax_off.lines) == 2
    plt.close(fig_on)
    plt.close(fig_off)


def test_plot_1d_with_fit_uses_nucleus_based_xlabel_by_default():
    result = _fit_one_slice()
    fig, ax = fitplot.plot_1d_with_fit(result)
    assert "1" in ax.get_xlabel() and "H" in ax.get_xlabel()
    plt.close(fig)

    fig2, ax2 = fitplot.plot_1d_with_fit(result, xlabel="Custom label")
    assert ax2.get_xlabel() == "Custom label"
    plt.close(fig2)


def test_plot_1d_with_fit_embeds_into_a_given_axis():
    result = _fit_one_slice()
    fig, ax = plt.subplots()
    returned_fig, returned_ax = fitplot.plot_1d_with_fit(result, ax=ax)
    assert returned_fig is fig
    assert returned_ax is ax
    plt.close(fig)


# ============================================================
# plot_2d_integrals
# ============================================================

def test_plot_2d_integrals_pseudo2d_uses_slice_number_axis():
    delays = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.0])
    spec = _make_pseudo2d_relaxation(delays)
    slices = slicing.get_slices(spec, rows=list(range(1, len(delays) + 1)))
    peak_defs = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.05}]
    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows", show_report=False, show_warnings=False,
                                       positive_intensity=False, allow_negative_intensity=True)

    fig, ax = fitplot.plot_2d_integrals(fitted["rows"], mode="rows")
    line = ax.lines[0]
    np.testing.assert_allclose(line.get_xdata(), np.arange(1, len(delays) + 1))
    assert ax.get_xlabel() == "Slice number"
    plt.close(fig)


def test_plot_2d_integrals_matches_the_true_inversion_recovery_shape():
    # Real correctness check, not just "a line got drawn": the plotted
    # y-values should trace the known inversion-recovery curve shape
    # (negative -> crossing zero -> a positive plateau), in the same
    # order as the (sorted) x-axis.
    delays = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.0])
    M0, T1, C = 1.0, 0.8, 2.0
    spec = _make_pseudo2d_relaxation(delays, M0=M0, T1=T1, C=C)
    slices = slicing.get_slices(spec, rows=list(range(1, len(delays) + 1)))
    peak_defs = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.05}]
    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows", show_report=False, show_warnings=False,
                                       positive_intensity=False, allow_negative_intensity=True)

    fig, ax = fitplot.plot_2d_integrals(fitted["rows"], mode="rows")
    y = ax.lines[0].get_ydata()
    assert y[0] < 0 < y[-1]
    assert np.all(np.diff(y) > 0)  # monotonically recovering
    plt.close(fig)


def test_plot_2d_integrals_normalised_uses_integrals_norm():
    result = _fit_one_slice()
    fig, ax = fitplot.plot_2d_integrals([result], mode="rows", normalised=True)
    y = ax.lines[0].get_ydata()
    # A single peak's own integral, normalised against itself, is 100%.
    assert y[0] == pytest.approx(100.0, rel=1e-6)
    assert ax.get_ylabel() == "Peak integral (%)"
    plt.close(fig)


def test_plot_2d_integrals_rejects_empty_fits():
    with pytest.raises(ValueError):
        fitplot.plot_2d_integrals([], mode="rows")


def test_plot_2d_integrals_rejects_invalid_x_axis():
    result = _fit_one_slice()
    with pytest.raises(ValueError):
        fitplot.plot_2d_integrals([result], x_axis="vdlist")


# ============================================================
# plot_relaxation
# ============================================================

def _fit_t1_series():
    delays = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.0])
    M0, T1, C = 1.0, 0.8, 2.0
    spec = _make_pseudo2d_relaxation(delays, M0=M0, T1=T1, C=C)
    slices = slicing.get_slices(spec, rows=list(range(1, len(delays) + 1)))
    peak_defs = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.05}]
    fitted = fitting.NMR2D_fit_slices(slices, peak_defs, mode="rows", show_report=False, show_warnings=False,
                                       positive_intensity=False, allow_negative_intensity=True)
    vdlist = {"name": "vdlist", "raw": [str(d) for d in delays], "seconds": delays}
    return relaxation.NMR2D_fit_relaxation(fitted["rows"], model="inversion_recovery",
                                            vdlist=vdlist, show_report=False), T1


def _peak_label(fits, index=0):
    """The label a fit actually carries. Peak labels now include the
    fitted shift ("Peak 1 (5.02 ppm)") rather than a bare number, so
    tests ask the result rather than hard-coding it."""
    return fits[index]["peak_label"]


def test_peak_labels_carry_the_fitted_shift():
    # Explicit user request: the fitting canvases should note the peak
    # ppm. Without it a curve fit reports a number with nothing tying
    # it back to the spectrum it came from.
    fits, _ = _fit_t1_series()
    label = _peak_label(fits)
    assert label.startswith("Peak 1 (")
    assert label.endswith("ppm)")


def test_plot_relaxation_draws_data_points_and_matching_fit_curve():
    fits, T1_true = _fit_t1_series()
    fig, ax = fitplot.plot_relaxation(fits)
    # data (marker, no line) + fit curve (line, no marker) for 1 series.
    # An unlabelled artist (label=None) reports matplotlib's own
    # auto-generated "_child0"-style name from get_label(), not "None".
    data_artist = next(a for a in ax.lines if a.get_label().startswith("_"))
    fit_artist = next(a for a in ax.lines if a.get_label() == _peak_label(fits))
    assert data_artist.get_linestyle() == "None"
    assert fit_artist.get_marker() in ("None", "")
    # The plotted curve should match evaluating the model function
    # directly with the fit's own fitted parameters -- confirms the
    # plotting genuinely uses fit["params"], not just draws something
    # curve-shaped. (Not compared against the *true* synthetic M0/C
    # directly: NMR2D_fit_relaxation fits the peak's integrated area,
    # not the raw amplitude, so the fitted M0 is scaled by the peak's
    # own area-per-unit-amplitude factor, not equal to the raw
    # amplitude=1.0 used to build the fixture.)
    fit = fits[0]
    y_at_t1 = np.interp(fit["params"]["T1"], fit_artist.get_xdata(), fit_artist.get_ydata())
    expected = relaxation.inversion_recovery(
        fit["params"]["T1"], fit["params"]["M0"], fit["params"]["T1"], fit["params"]["C"],
    )
    assert y_at_t1 == pytest.approx(expected, rel=0.02)
    plt.close(fig)


def test_plot_relaxation_annotate_shows_the_correct_fitted_t1():
    fits, T1_true = _fit_t1_series()
    fig, ax = fitplot.plot_relaxation(fits, annotate=True)
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 1
    assert "T_{1}" in texts[0]
    assert f"{T1_true * 1000:.0f}" in texts[0]  # formatted in ms, per format_time_value
    plt.close(fig)


def test_plot_relaxation_show_series_filters_by_label_and_index():
    fits, _ = _fit_t1_series()
    fig1, ax1 = fitplot.plot_relaxation(fits, show_series=[_peak_label(fits)])
    fig2, ax2 = fitplot.plot_relaxation(fits, show_series=[0])
    # Selecting the same single series by label vs. by index must
    # produce the identical plotted fit curve.
    curve1 = next(a for a in ax1.lines if a.get_label() == _peak_label(fits))
    curve2 = next(a for a in ax2.lines if a.get_label() == _peak_label(fits))
    np.testing.assert_allclose(curve1.get_ydata(), curve2.get_ydata())
    plt.close(fig1)
    plt.close(fig2)


def test_plot_relaxation_rejects_unknown_series_label():
    fits, _ = _fit_t1_series()
    with pytest.raises(ValueError):
        fitplot.plot_relaxation(fits, show_series=["Not a real label"])


def test_plot_relaxation_rejects_empty_results():
    with pytest.raises(ValueError):
        fitplot.plot_relaxation([])


def test_plot_relaxation_logplot_x_only():
    fits, _ = _fit_t1_series()
    fig, ax = fitplot.plot_relaxation(fits, logplot="x")
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "linear"
    plt.close(fig)


def test_plot_relaxation_fit_curve_defaults_to_a_paler_tint_of_the_data_colour():
    fits, _ = _fit_t1_series()
    fig, ax = fitplot.plot_relaxation(fits, colors=["red"])
    data_artist = next(a for a in ax.lines if a.get_label().startswith("_"))
    fit_artist = next(a for a in ax.lines if a.get_label() == _peak_label(fits))
    import matplotlib.colors as mcolors
    data_rgb = mcolors.to_rgb(data_artist.get_color())
    fit_rgb = mcolors.to_rgb(fit_artist.get_color())
    assert data_rgb == pytest.approx((1.0, 0.0, 0.0))
    assert fit_rgb != data_rgb
    assert all(f >= d for f, d in zip(fit_rgb, data_rgb))  # lighter, not darker/different hue
    plt.close(fig)


# ============================================================
# The DOSY map's own styling
# ============================================================

def _dosy_fixture():
    from opennmr.processing import ilt

    b = np.linspace(0, 4e9, 24)
    ppm = np.linspace(0, 10, 128)
    data = np.zeros((b.size, ppm.size))
    for shift, d, amplitude in [(2.0, 1.5e-9, 1.0), (7.5, 1.2e-10, 1.2)]:
        data += np.outer(np.exp(-d * b), amplitude / (1.0 + ((ppm - shift) / 0.06) ** 2))
    return ilt.dosy_map(b, data, ppm, alpha=1e-3, threshold=0.05)


def test_plot_dosy_draws_line_contours_not_filled_ones():
    # Reported directly: it "should be plotted as a normal contour plot
    # (not filled) and match the styles accordingly" -- the same kind
    # of plot as plot_2d_contour, which has always drawn lines.
    from matplotlib.collections import PathCollection

    fig, (contour_ax, _projection_ax) = fitplot.plot_dosy(_dosy_fixture())
    assert contour_ax.collections
    assert not [c for c in contour_ax.collections if isinstance(c, PathCollection)]
    plt.close(fig)


def test_plot_dosy_uses_the_font_size_it_is_given():
    fig, (contour_ax, _projection_ax) = fitplot.plot_dosy(_dosy_fixture(), fontsize=17)
    assert contour_ax.xaxis.label.get_fontsize() == pytest.approx(17)
    assert contour_ax.yaxis.label.get_fontsize() == pytest.approx(17)
    plt.close(fig)


def test_plot_dosy_closes_the_frame_around_the_map():
    # A contour is read as a box: with the top and right spines hidden,
    # a peak near either edge trails off into nothing.
    fig, (contour_ax, projection_ax) = fitplot.plot_dosy(_dosy_fixture())
    assert all(contour_ax.spines[side].get_visible()
               for side in ("top", "right", "bottom", "left"))
    # ...and the strip above it has no frame, so it must have no ticks
    # either -- they would float above the map belonging to nothing.
    assert not any(spine.get_visible() for spine in projection_ax.spines.values())
    assert all(not tick.tick1line.get_visible() and not tick.tick2line.get_visible()
               for tick in projection_ax.xaxis.get_major_ticks())
    plt.close(fig)


def test_plot_dosy_d_limits_fit_the_populated_range_but_stay_inside_the_grid():
    dosy = _dosy_fixture()
    low, high = fitplot._dosy_d_limits(dosy["Z"], dosy["D"], factor=0.02)
    grid_low, grid_high = float(np.min(dosy["D"])), float(np.max(dosy["D"]))
    assert grid_low <= low < high <= grid_high
    assert (high / low) < (grid_high / grid_low)


def test_plot_dosy_d_limits_fall_back_to_the_whole_grid_when_empty():
    # An empty map has no populated range to fit; collapsing to a
    # degenerate one would be worse than showing everything.
    d_grid = np.logspace(-13, -8, 32)
    assert fitplot._dosy_d_limits(np.zeros((32, 8)), d_grid, factor=0.02) == (
        pytest.approx(float(d_grid.min())), pytest.approx(float(d_grid.max())),
    )
