"""DNP polarisation against a delay (opennmr.notebook.dnp_polarisation)."""
import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import dnp_polarisation as dp
from opennmr.processing import slicing

PPM = np.linspace(-20.0, 20.0, 4001)
SFO1 = 400.0
TAU_ON, TAU_OFF = 4.0, 6.0
DELAYS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])


def _trace(height, ns=8, rg=64.0, seed=0, centre=3.0, width=0.5, noise=0.002):
    rng = np.random.default_rng(seed)
    y = height / (1.0 + ((PPM - centre) / (0.5 * width)) ** 2) + noise * rng.standard_normal(PPM.size)
    # JEOL data are summed over their scans, so the per-scan scaling has something to do.
    md = NMRMetadata(vendor="jeol", nucleus="1H", sfo1=SFO1, bf1=SFO1, ns=ns,
                     raw={"acqus": {"SFO1": SFO1, "RG": rg, "NS": ns}})
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=PPM, hz=PPM * SFO1)], metadata=md)


def _series(amplitude, tau, ns=8, rg=64.0, seed=0):
    traces = [_trace(amplitude * (1 - math.exp(-t / tau)), ns=ns, rg=rg, seed=seed + i)
              for i, t in enumerate(DELAYS)]
    stacked = slicing.build_pseudo2d(traces)
    stacked.metadata.vendor, stacked.metadata.ns = "jeol", ns
    stacked.metadata.raw = {"acqus": {"SFO1": SFO1, "RG": rg, "NS": ns}}
    return stacked


@pytest.fixture(scope="module")
def pair():
    return _series(10.0, TAU_ON), _series(2.0, TAU_OFF, seed=100)


def test_delay_values_from_a_list_start_stop_step_and_start_step():
    assert dp.delay_values(3, values=[1, 2, 3, 4]).tolist() == [1, 2, 3]
    assert dp.delay_values(3, values=[10, 20, 30], scale=1e-3) == pytest.approx([0.01, 0.02, 0.03])
    assert dp.delay_values(5, start=0, stop=4, step=1).tolist() == [0, 1, 2, 3, 4]
    assert dp.delay_values(4, start=1, step=1, multiplier=2).tolist() == [1, 2, 4, 8]
    assert dp.delay_values(3, start=0, stop=1) == pytest.approx([0, 0.5, 1])
    with pytest.raises(ValueError, match="shorter list"):
        dp.delay_values(5, values=[1, 2])
    with pytest.raises(ValueError, match="5 slices"):
        dp.delay_values(5, start=0, stop=3, step=1)
    with pytest.raises(ValueError):
        dp.delay_values(3)


def test_integral_series_recovers_the_build_up(pair):
    on, _ = pair
    s = dp.measure_series(on, DELAYS, "integral", region=(0.0, 6.0), per_scan=False, divide_rg=False)
    assert s["x"].tolist() == DELAYS.tolist()
    ratio = s["y"] / s["y"][-1]
    expected = (1 - np.exp(-DELAYS / TAU_ON)) / (1 - math.exp(-DELAYS[-1] / TAU_ON))
    np.testing.assert_allclose(ratio, expected, rtol=0.03)
    assert np.all(s["y_err"] > 0)


def test_height_series_and_scan_and_gain_scaling():
    one = _series(10.0, TAU_ON, ns=8, rg=64.0)
    doubled = _series(10.0, TAU_ON, ns=16, rg=128.0)
    a = dp.measure_series(one, DELAYS, "height", peak=3.0, per_scan=False, divide_rg=False)
    b = dp.measure_series(doubled, DELAYS, "height", peak=3.0, per_scan=False, divide_rg=False)
    np.testing.assert_allclose(a["y"], b["y"], rtol=0.02)
    pa = dp.measure_series(one, DELAYS, "height", peak=3.0, per_scan=True, divide_rg=True)
    assert pa["y"][-1] == pytest.approx(a["y"][-1] / 8 / 64, rel=1e-9)
    gain_only = dp.measure_series(one, DELAYS, "height", peak=3.0, per_scan=False, divide_rg=True)
    assert gain_only["y"][-1] == pytest.approx(a["y"][-1] / 64, rel=1e-9)


def test_measure_refuses_bad_input(pair):
    on, _ = pair
    with pytest.raises(ValueError, match="region"):
        dp.measure_series(on, DELAYS, "integral")
    with pytest.raises(ValueError, match="peak"):
        dp.measure_series(on, DELAYS, "height")
    with pytest.raises(ValueError, match="delays"):
        dp.measure_series(on, DELAYS[:3], "integral", region=(0, 6))
    with pytest.raises(ValueError, match="does not exist"):
        dp.measure_series(on, DELAYS, "integral", region=(0, 6), rows=[99])
    with pytest.raises(ValueError, match="pseudo-2D"):
        dp.measure_series(_trace(1.0), DELAYS, "integral", region=(0, 6))
    with pytest.raises(ValueError, match="series_from_fits"):
        dp.measure_series(on, DELAYS, "fit_area", region=(0, 6))


def test_chosen_rows_take_their_own_delays(pair):
    on, _ = pair
    s = dp.measure_series(on, DELAYS, "integral", region=(0, 6), rows=[2, 4, 6],
                          per_scan=False, divide_rg=False)
    assert s["x"].tolist() == DELAYS[[1, 3, 5]].tolist()
    full = dp.measure_series(on, DELAYS, "integral", region=(0, 6), per_scan=False, divide_rg=False)
    np.testing.assert_allclose(s["y"], full["y"][[1, 3, 5]])


def test_difference_ratio_enhancement_sum_and_errors(pair):
    on, off = pair
    kw = dict(quantity="integral", region=(0.0, 6.0), per_scan=False, divide_rg=False)
    a, b = dp.measure_series(on, DELAYS, **kw), dp.measure_series(off, DELAYS, **kw)
    d = dp.polarisation(a, b, "difference")
    np.testing.assert_allclose(d["y"], a["y"] - b["y"])
    np.testing.assert_allclose(d["y_err"], np.hypot(a["y_err"], b["y_err"]))
    r = dp.polarisation(a, b, "ratio")
    np.testing.assert_allclose(r["y"], a["y"] / b["y"])
    expected_err = np.abs(r["y"]) * np.sqrt((a["y_err"] / a["y"]) ** 2 + (b["y_err"] / b["y"]) ** 2)
    np.testing.assert_allclose(r["y_err"], expected_err)
    e = dp.polarisation(a, b, "enhancement")
    np.testing.assert_allclose(e["y"], (a["y"] - b["y"]) / b["y"])
    np.testing.assert_allclose(dp.polarisation(a, b, "sum")["y"], a["y"] + b["y"])
    with pytest.raises(ValueError, match="mode"):
        dp.polarisation(a, b, "product")
    # The late-time difference tends to the difference of the plateaus: 10 - 2 ≈ 8 × the line's area.
    assert d["y"][-1] / (a["y"][-1] / (1 - math.exp(-DELAYS[-1] / TAU_ON)) * 0.2) > 0.5


def test_mismatched_delays_are_refused_unless_interpolated(pair):
    on, off = pair
    kw = dict(quantity="integral", region=(0.0, 6.0), per_scan=False, divide_rg=False)
    a = dp.measure_series(on, DELAYS, **kw)
    b = dp.measure_series(off, DELAYS * 1.5, **kw)
    with pytest.raises(ValueError, match="do not share"):
        dp.polarisation(a, b)
    d = dp.polarisation(a, b, interpolate=True)
    assert any("interpolated" in n for n in d["notes"])
    assert any("outside" in n for n in d["notes"])
    assert len(d["x"]) == len(DELAYS)


def test_unsorted_delays_are_sorted_and_zero_in_b_is_flagged(pair):
    on, off = pair
    kw = dict(quantity="integral", region=(0.0, 6.0), per_scan=False, divide_rg=False)
    order = np.array([3, 0, 2, 1, 6, 5, 4])
    a = dp.measure_series(on, DELAYS[order], **kw)
    b = dp.measure_series(off, DELAYS[order], **kw)
    d = dp.polarisation(a, b, "difference")
    assert np.all(np.diff(d["x"]) > 0)
    unsorted = dp.polarisation(a, b, "difference", sort=False)
    assert unsorted["x"].tolist() == DELAYS[order].tolist()
    z = dict(b, y=np.zeros_like(b["y"]))
    assert any("zero" in n for n in dp.polarisation(a, z, "ratio")["notes"])


def _fits(amplitudes, with_popt=True):
    fits = []
    for i, amp in enumerate(amplitudes, start=1):
        fit = {"index": i, "integrals": [amp * 2.0, 5.0], "integral_errors": [0.1, 0.2]}
        if with_popt:
            fit.update({
                "popt": np.array([3.0, 0.5, amp, 1.0, 0.5, 5.0]),
                "pcov": np.diag([0.01, 0.01, 0.04, 0.01, 0.01, 0.09]),
                "peak_defs": [{"type": "lorentzian"}, {"type": "lorentzian"}],
                "is_free": np.ones(6, dtype=bool),
            })
        fits.append(fit)
    return fits


def test_series_from_fits_area_and_amplitude_and_peak_choice():
    fits = _fits([1.0, 2.0, 3.0])
    area = dp.series_from_fits(fits, [0.1, 0.2, 0.3], "area")
    assert area["y"].tolist() == [2.0, 4.0, 6.0] and area["quantity"] == "fit_area"
    amp = dp.series_from_fits(fits, [0.1, 0.2, 0.3], "amplitude")
    assert amp["y"].tolist() == [1.0, 2.0, 3.0]
    assert amp["y_err"] == pytest.approx([0.2, 0.2, 0.2])
    second = dp.series_from_fits(fits, [0.1, 0.2, 0.3], "area", peak_index=1)
    assert second["y"].tolist() == [5.0, 5.0, 5.0]
    out_of_range = dp.series_from_fits(fits, [0.1, 0.2, 0.3], "area", peak_index=5)
    assert np.all(np.isnan(out_of_range["y"]))
    with pytest.raises(ValueError, match="integrated, not fitted"):
        dp.series_from_fits(_fits([1, 2], with_popt=False), [0.1, 0.2], "amplitude")
    with pytest.raises(ValueError, match="value_source"):
        dp.series_from_fits(fits, [0.1, 0.2, 0.3], "volume")


def test_series_from_fits_uses_each_slices_own_number():
    fits = _fits([1.0, 2.0, 3.0])
    for fit, index in zip(fits, (2, 4, 6)):
        fit["index"] = index
    s = dp.series_from_fits(fits, [10, 20, 30, 40, 50, 60], "area")
    assert s["x"].tolist() == [20, 40, 60]
    s = dp.series_from_fits(fits, [7, 8, 9], "area")      # one per fit
    assert s["x"].tolist() == [7, 8, 9]
    with pytest.raises(ValueError, match="covers slice"):
        dp.series_from_fits(fits, [1, 2], "area")


def test_fitted_and_measured_series_can_be_combined():
    fits_a, fits_b = _fits([1.0, 2.0, 3.0]), _fits([0.5, 0.5, 0.5])
    a = dp.series_from_fits(fits_a, [0.1, 0.2, 0.3], "amplitude")
    b = dp.series_from_fits(fits_b, [0.1, 0.2, 0.3], "amplitude")
    d = dp.polarisation(a, b)
    assert d["y"].tolist() == [0.5, 1.5, 2.5]


def test_csv_and_report(pair, tmp_path):
    on, off = pair
    kw = dict(quantity="integral", region=(0.0, 6.0), per_scan=False, divide_rg=False)
    d = dp.polarisation(dp.measure_series(on, DELAYS, label="on", **kw),
                        dp.measure_series(off, DELAYS, label="off", **kw))
    path = tmp_path / "p.csv"
    text = dp.polarisation_to_csv(d, path)
    assert path.read_text(encoding="utf-8-sig") == text
    rows = [line for line in text.splitlines() if not line.startswith("#")]
    assert rows[0] == "delay_s,result,result_error,A,A_error,B,B_error"
    assert len(rows) == len(DELAYS) + 1
    assert float(rows[1].split(",")[0]) == DELAYS[0]
    report = dp.polarisation_report(d)
    assert "A: on" in report and "B: off" in report and "region 0 to 6 ppm" in report


@pytest.mark.parametrize("show_individual", [False, True])
@pytest.mark.parametrize("connect", [False, True])
def test_plot_options(pair, show_individual, connect):
    import matplotlib.pyplot as plt
    on, off = pair
    kw = dict(quantity="integral", region=(0.0, 6.0), per_scan=False, divide_rg=False)
    d = dp.polarisation(dp.measure_series(on, DELAYS, **kw), dp.measure_series(off, DELAYS, **kw))
    fig = dp.plot_polarisation(d, show_individual=show_individual, connect=connect, time_unit="s")
    assert len(fig.axes) == (2 if show_individual else 1)
    result_ax = fig.axes[-1]
    joined = [ln for ln in result_ax.lines if ln.get_linestyle() == "-"]
    assert bool(joined) is connect
    assert result_ax.get_xlabel() == "delay (s)"
    if show_individual:
        assert len(fig.axes[0].get_legend().get_texts()) == 2
    fig2 = dp.plot_polarisation(d, time_unit="ms", xlabel="t", ylabel="P")
    assert fig2.axes[0].get_xlabel() == "t" and fig2.axes[0].get_ylabel() == "P"
    with pytest.raises(ValueError):
        dp.plot_polarisation(d, time_unit="years")
    plt.close("all")


def test_time_unit_auto():
    assert dp._time_unit([1e-6, 5e-4])[1] == "µs"
    assert dp._time_unit([1e-3, 0.5])[1] == "ms"
    assert dp._time_unit([1, 100])[1] == "s"


def test_experiment_delays_from_recorded_values_or_nothing(pair):
    on, _ = pair
    assert dp.experiment_delays(on) == (None, None)
    on.metadata.raw["indirect_values"] = {"unit": "s", "values": list(DELAYS) + [99.0],
                                          "source": "the test"}
    values, source = dp.experiment_delays(on)
    assert values.tolist() == DELAYS.tolist() and source == "the test"
    on.metadata.raw["indirect_values"] = {"unit": "s", "values": [1.0, 2.0]}
    assert dp.experiment_delays(on) == (None, None)
    del on.metadata.raw["indirect_values"]
