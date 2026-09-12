"""Electrophoretic NMR -- opennmr.processing.enmr.

The headline test recovers an **accepted literature mobility** from a
real calibration standard, which is the only kind of check that can
catch a wrong factor of 2*pi, a wrong drift time, or a unit slip: every
one of those still produces a beautiful straight line.
"""

import os

import numpy as np
import pytest

from opennmr import io
from opennmr.io import bruker
from opennmr.io import lists as io_lists
from opennmr.processing import enmr

#: TMABr calibration standard. mu(TMA+) = 3.79e-8 m^2/Vs is the
#: accepted literature value, supplied by the user along with the data.
TMA_LITERATURE_MOBILITY = 3.79e-8

#: The cell's electrode separation, known independently of this
#: measurement. No vendor file records it.
ELECTRODE_GAP_M = 0.0385

#: This cell's wiring: which way its positive electrode faces relative
#: to the gradient's +z. No vendor file records it either, so it is
#: *measured* here from the one external value available. The vdlist
#: puts the positive polarity first (confirmed by the user) and the
#: calibration's own diff.xml records gmax = +0.3 T/m, and with those
#: two the raw formula gives TMA+ -- a cation -- a mobility of the
#: right magnitude and a negative sign. So this cell is wired -1.
CELL_POLARITY = -1

CALIBRATION = os.path.join(
    "test_data", "eNMR", "Diff50_1H_eNMR_Calibration_2024-05-13", "21")


def _synthetic(slope_rad_per_volt=0.008, voltages=(0, 0, 8, 8, 16, 16, 24, 24, 32, 32),
                noise=0.0, seed=0):
    """Phases for a species drifting at a known rate, acquired as
    +V/-V pairs -- **positive first**, which is the order the
    instrument writes the vdlist in."""
    voltages = np.asarray(voltages, dtype=float)
    sign = np.where(np.arange(voltages.size) % 2, -1.0, 1.0)
    phases = sign * slope_rad_per_volt * voltages
    if noise:
        phases = phases + np.random.default_rng(seed).normal(0, noise, voltages.size)
    return voltages, phases


# ============================================================
# Voltage-polarity pairs, within one experiment
# ============================================================

def test_polarity_pairs_collapses_plus_minus_into_one_difference():
    voltages, phases = _synthetic(slope_rad_per_volt=0.01)
    paired = enmr.polarity_pairs(voltages, phases)
    assert paired["paired"] is True
    np.testing.assert_allclose(paired["voltages"], [0, 8, 16, 24, 32])
    np.testing.assert_allclose(paired["dphi"], [0, 0.08, 0.16, 0.24, 0.32])


def test_polarity_pairs_cancels_a_phase_error_common_to_both():
    # The reason for pairing: a receiver-phase or drift offset is the
    # same at both polarities and differences away, while the
    # field-driven phase doubles.
    voltages, phases = _synthetic(slope_rad_per_volt=0.01)
    clean = enmr.polarity_pairs(voltages, phases)["dphi"]
    offset = enmr.polarity_pairs(voltages, phases + 0.35)["dphi"]
    np.testing.assert_allclose(offset, clean, atol=1e-12)


def test_polarity_pairs_detects_an_unpaired_series_rather_than_assuming():
    voltages = np.array([0.0, 8.0, 16.0, 24.0, 32.0, 40.0])
    result = enmr.polarity_pairs(voltages, 0.01 * voltages)
    assert result["paired"] is False
    np.testing.assert_allclose(result["voltages"], voltages)


def test_polarity_pairs_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="voltages against"):
        enmr.polarity_pairs([0, 1, 2], [0.0, 0.1])


# ============================================================
# Unwrapping an interleaved series
# ============================================================

def test_unwrapping_across_polarities_invents_a_jump():
    # Consecutive points of a +V/-V series are opposite polarity, so
    # they legitimately sit far apart. Unwrapping across them reads a
    # polarity jump as a wrap and inserts a 2*pi step that was never
    # there -- silently, and the fit then looks like a sample with no
    # mobility rather than like a broken calculation.
    ppm = np.linspace(10.0, 0.0, 1024)
    voltages = np.array([0.0, 0.0, 20.0, 20.0, 40.0, 40.0, 60.0, 60.0, 80.0, 80.0])
    sign = np.where(np.arange(voltages.size) % 2, 1.0, -1.0)
    truth = sign * 0.03 * voltages          # reaches +/-2.4 rad, past pi apart
    rows = _fake_rows(truth, ppm)

    interleaved = enmr.phase_series(rows, ppm, (3.0, 3.6))            # stride 1
    per_polarity = enmr.phase_series(rows, ppm, (3.0, 3.6), stride=2)

    np.testing.assert_allclose(per_polarity, truth, atol=1e-6)
    assert not np.allclose(interleaved, truth, atol=1e-6)


def test_phase_ramp_picks_the_stride_from_the_voltages():
    # The one-call path: it has the voltages, so it can see the
    # interleaving that phase_series cannot.
    ppm = np.linspace(10.0, 0.0, 1024)
    voltages = np.array([0.0, 0.0, 20.0, 20.0, 40.0, 40.0, 60.0, 60.0, 80.0, 80.0])
    sign = np.where(np.arange(voltages.size) % 2, 1.0, -1.0)
    truth = sign * 0.03 * voltages
    rows = _fake_rows(truth, ppm)
    np.testing.assert_allclose(
        enmr.phase_ramp(rows, ppm, (3.0, 3.6), voltages), truth, atol=1e-6)


def test_phase_ramp_falls_back_to_a_plain_unwrap_when_unpaired():
    ppm = np.linspace(10.0, 0.0, 1024)
    voltages = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    truth = 0.05 * voltages
    rows = _fake_rows(truth, ppm)
    np.testing.assert_allclose(
        enmr.phase_ramp(rows, ppm, (3.0, 3.6), voltages), truth, atol=1e-6)


# ============================================================
# Combining the gradient-reversed repeat experiments
# ============================================================

def test_combine_experiments_averages_repeats():
    combined = enmr.combine_experiments([0.0124389, 0.0124389])
    assert combined["slope"] == pytest.approx(0.0124389)
    assert combined["spread"] == pytest.approx(0.0, abs=1e-12)
    assert combined["n"] == 2


def test_combine_experiments_aligns_a_reversed_repeat():
    # Reversing the gradient flips the sign of the fitted slope --
    # measured on the real pair (LiAc 20/21: +1.244e-2 and -1.396e-2).
    # Averaging them as given would cancel the measurement away.
    combined = enmr.combine_experiments([0.0124389, -0.0139647])
    assert combined["slope"] == pytest.approx(0.0132018, rel=1e-4)
    assert combined["sign_flipped"].tolist() == [False, True]

    as_given = enmr.combine_experiments([0.0124389, -0.0139647], align_sign=False)
    assert abs(as_given["slope"]) < 0.1 * abs(combined["slope"])


def test_combine_experiments_spread_flags_repeats_that_disagree():
    # Two runs of the same sample should agree to well within their own
    # errors; a spread comparable to the slope means they do not, and
    # the average of them is not a measurement.
    agree = enmr.combine_experiments([0.0124, -0.0132])
    disagree = enmr.combine_experiments([0.0019, -0.0095])
    assert agree["spread"] < 0.1 * abs(agree["slope"])
    assert disagree["spread"] > 0.5 * abs(disagree["slope"])


def test_combine_experiments_accepts_fit_results_and_propagates_error():
    first = enmr.fit_enmr(*_synthetic(slope_rad_per_volt=0.008, noise=0.01))
    second = enmr.fit_enmr(*_synthetic(slope_rad_per_volt=0.008, noise=0.01, seed=1))
    combined = enmr.combine_experiments([first, second])
    assert combined["slope"] == pytest.approx(
        (first["slope"] + second["slope"]) / 2, rel=1e-9)
    assert combined["slope_error"] < max(first["slope_error"], second["slope_error"])


def test_combine_experiments_needs_at_least_two():
    with pytest.raises(ValueError, match="at least 2 experiments"):
        enmr.combine_experiments([0.008])


# ============================================================
# The mobility conversion
# ============================================================

def test_mobility_from_slope_is_the_stated_formula():
    # mu = slope * d / (gamma * g * delta * DELTA)
    mu = enmr.mobility_from_slope(
        slope=0.0079015, electrode_gap=0.0385, gamma=2 * np.pi * 42577464.6,
        gradient=0.3, delta=1e-3, big_delta=0.1)
    assert mu == pytest.approx(TMA_LITERATURE_MOBILITY, rel=0.01)


def test_mobility_from_slope_requires_an_electrode_gap():
    # It scales the answer directly and no vendor file records it, so
    # there is deliberately no default.
    with pytest.raises(ValueError, match="electrode_gap is required"):
        enmr.mobility_from_slope(0.008, None, 2.675e8, 0.3, 1e-3, 0.1)
    with pytest.raises(ValueError, match="must be positive"):
        enmr.mobility_from_slope(0.008, -0.01, 2.675e8, 0.3, 1e-3, 0.1)


def test_mobility_from_slope_rejects_a_zero_denominator():
    with pytest.raises(ValueError, match="is zero"):
        enmr.mobility_from_slope(0.008, 0.0385, 2.675e8, 0.0, 1e-3, 0.1)


def test_mobility_scales_with_the_electrode_gap():
    kwargs = dict(gamma=2.675e8, gradient=0.3, delta=1e-3, big_delta=0.1)
    near = enmr.mobility_from_slope(0.008, 0.02, **kwargs)
    far = enmr.mobility_from_slope(0.008, 0.04, **kwargs)
    assert far == pytest.approx(2 * near)


# ============================================================
# Phase extraction
# ============================================================

def _fake_rows(phases, ppm, centre=3.3, width=0.05):
    """A single Lorentzian-ish line carrying a given phase per row."""
    profile = width ** 2 / ((ppm - centre) ** 2 + width ** 2)
    return np.array([profile * np.exp(1j * p) for p in phases])


def test_phase_series_reads_the_phase_of_each_slice():
    ppm = np.linspace(10.0, 0.0, 2048)
    truth = np.array([0.0, 0.2, 0.4, 0.6])
    rows = _fake_rows(truth, ppm)
    np.testing.assert_allclose(enmr.phase_series(rows, ppm, (3.0, 3.6)), truth, atol=1e-9)


def test_phase_series_unwraps_a_ramp_past_pi():
    # A fit against wrapped phase is meaningless the moment the ramp
    # passes pi.
    ppm = np.linspace(10.0, 0.0, 1024)
    truth = np.linspace(0.0, 8.0, 12)
    rows = _fake_rows(truth, ppm)
    np.testing.assert_allclose(enmr.phase_series(rows, ppm, (3.0, 3.6)), truth, atol=1e-6)
    wrapped = enmr.phase_series(rows, ppm, (3.0, 3.6), unwrap=False)
    assert np.abs(wrapped).max() <= np.pi + 1e-9


def test_phase_series_needs_the_imaginary_part():
    ppm = np.linspace(10.0, 0.0, 256)
    with pytest.raises(ValueError, match="no imaginary part"):
        enmr.phase_series(np.ones((3, 256)), ppm, (3.0, 3.6))


def test_phase_series_rejects_an_empty_region():
    ppm = np.linspace(10.0, 0.0, 256)
    rows = _fake_rows([0.0, 0.1], ppm)
    with pytest.raises(ValueError, match="No spectral points"):
        enmr.phase_series(rows, ppm, (50.0, 60.0))


# ============================================================
# The fit
# ============================================================

def _fit_kwargs():
    return dict(electrode_gap=ELECTRODE_GAP_M, gamma=2 * np.pi * 42577464.6,
                gradient=0.3, delta=1e-3, big_delta=0.1)


def test_fit_enmr_gives_the_slope_without_any_constants():
    # The slope is what the data determines; it needs nothing beyond
    # the data. Requiring a cell gap to get it would force a number
    # into the calculation whether or not the user has a good one.
    voltages, phases = _synthetic(slope_rad_per_volt=0.0079015)
    result = enmr.fit_enmr(voltages, phases)
    assert result["slope"] == pytest.approx(0.0079015, rel=1e-6)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert result["mobility"] is None


def test_fit_enmr_adds_a_mobility_when_every_constant_is_given():
    voltages, phases = _synthetic(slope_rad_per_volt=0.0079015)
    result = enmr.fit_enmr(voltages, phases, **_fit_kwargs())
    assert result["mobility"] == pytest.approx(TMA_LITERATURE_MOBILITY, rel=0.01)


def test_fit_enmr_rejects_a_half_supplied_conversion():
    # Silently skipping the mobility because one constant was missed
    # would look like the slope-only path and hide a typo.
    voltages, phases = _synthetic()
    with pytest.raises(ValueError, match="missing big_delta"):
        enmr.fit_enmr(voltages, phases, electrode_gap=0.0385, gamma=2.675e8,
                       gradient=0.3, delta=1e-3)


def test_fit_enmr_reports_zero_for_a_species_that_does_not_drift():
    voltages, phases = _synthetic(slope_rad_per_volt=0.0, noise=0.004)
    result = enmr.fit_enmr(voltages, phases)
    assert abs(result["slope"]) < 3 * result["slope_error"]
    assert result["r_squared"] < 0.5


def test_fit_enmr_needs_enough_points_to_judge_a_line():
    with pytest.raises(ValueError, match="at least 3 points"):
        enmr.fit_enmr([0.0, 8.0], [0.0, 0.06])


def test_fit_enmr_handles_an_unpaired_series():
    voltages = np.array([0.0, 8.0, 16.0, 24.0, 32.0, 40.0])
    result = enmr.fit_enmr(voltages, 0.008 * voltages)
    assert result["paired"] is False
    assert result["slope"] == pytest.approx(0.008, rel=1e-6)


def test_mobility_error_is_the_slope_error_scaled():
    voltages, phases = _synthetic(slope_rad_per_volt=0.008, noise=0.01)
    result = enmr.fit_enmr(voltages, phases, **_fit_kwargs())
    assert (result["mobility_error"] / abs(result["mobility"])) == pytest.approx(
        result["slope_error"] / abs(result["slope"]), rel=1e-9)


def test_report_leads_with_the_slope():
    voltages, phases = _synthetic(slope_rad_per_volt=0.008)
    report = enmr.format_enmr_report(enmr.fit_enmr(voltages, phases), label="TMA+")
    assert "TMA+" in report
    assert "rad/V" in report and "R²" in report
    assert "+V/−V pairs" in report
    # No constants supplied, so no mobility invented.
    assert "Mobility not computed" in report
    # And it says this is one experiment of a reversed pair, pointing
    # at the function that averages the ramps rather than the slopes.
    assert "average_gradient_pair" in report
    # The slope in degrees as well, since that is the unit phase is
    # set in at the spectrometer.
    assert "°/V" in report


def test_combined_report_puts_the_spread_beside_the_slope():
    combined = enmr.combine_experiments([0.0124389, -0.0139647])
    report = enmr.format_combined_report(combined)
    assert "spread across 2 repeats" in report
    assert "sign flipped" in report


def test_report_names_the_electrode_gap_beside_the_mobility():
    voltages, phases = _synthetic(slope_rad_per_volt=0.008)
    report = enmr.format_enmr_report(enmr.fit_enmr(voltages, phases, **_fit_kwargs()))
    assert "m²/Vs" in report
    # The assumption the whole number rests on, stated with it.
    assert "electrode gap" in report and "38.5 mm" in report


# ============================================================
# The real calibration standard
# ============================================================

def _calibration_series():
    spectrum = io.load(CALIBRATION, pseudo_2d=True, autophase=False)
    ppm = spectrum.axes[1].values("ppm")
    rows = spectrum.data + 1j * spectrum.data_imag
    vdlist = io_lists.read_value_list(os.path.join(CALIBRATION, "vdlist"))
    # The user's own conversion: V = (vdlist[us] - 0.5u) * 2.
    voltages = ((vdlist["values"] * 1e6 - 0.5) * 2)[: rows.shape[0]]
    diff = bruker.read_diff_xml(CALIBRATION, nucleus="1H")
    return ppm, rows, voltages, diff


@pytest.mark.skipif(not os.path.isdir(CALIBRATION), reason="needs the real eNMR dataset")
def test_recovers_the_literature_tma_mobility_from_real_data():
    # The check that matters. A wrong 2*pi, a wrong drift time or a
    # unit slip all still give a straight line; only an accepted
    # external value catches them.
    ppm, rows, voltages, diff = _calibration_series()
    phases = enmr.phase_ramp(rows, ppm, (3.18, 3.42), voltages)
    result = enmr.fit_enmr(
        voltages, phases, electrode_gap=ELECTRODE_GAP_M,
        gamma=2 * np.pi * diff["gamma"], gradient=float(diff["gradient"][0]),
        delta=diff["delta"], big_delta=diff["DELTA"],
        cell_polarity=CELL_POLARITY)

    assert result["paired"] is True
    assert result["r_squared"] > 0.99
    assert result["mobility"] == pytest.approx(TMA_LITERATURE_MOBILITY, rel=0.05)
    # Read with the interleaving-aware unwrap, which is what
    # phase_ramp exists to get right. Negative because the positive
    # polarity comes first and this cell's + electrode opposes +z --
    # the magnitude is the part the literature value checks.
    assert result["slope"] == pytest.approx(-7.909e-3, rel=0.01)
    # And the sign really is only the cell: flipping it flips the
    # answer and nothing else.
    flipped = enmr.fit_enmr(
        voltages, phases, electrode_gap=ELECTRODE_GAP_M,
        gamma=2 * np.pi * diff["gamma"], gradient=float(diff["gradient"][0]),
        delta=diff["delta"], big_delta=diff["DELTA"], cell_polarity=1)
    assert flipped["mobility"] == pytest.approx(-result["mobility"])
    assert flipped["slope"] == pytest.approx(result["slope"])


@pytest.mark.skipif(not os.path.isdir(CALIBRATION), reason="needs the real eNMR dataset")
def test_the_neutral_species_in_the_same_spectrum_does_not_drift():
    # Water is in the same tube, the same field and the same shim, so
    # it controls for everything except charge. If it showed a ramp,
    # the TMA+ result would be measuring the cell, not the ion.
    ppm, rows, voltages, diff = _calibration_series()
    phases = enmr.phase_ramp(rows, ppm, (4.80, 5.04), voltages)
    result = enmr.fit_enmr(
        voltages, phases, electrode_gap=ELECTRODE_GAP_M,
        gamma=2 * np.pi * diff["gamma"], gradient=float(diff["gradient"][0]),
        delta=diff["delta"], big_delta=diff["DELTA"])

    assert abs(result["slope"]) < 3 * result["slope_error"]
    assert result["r_squared"] < 0.5
    # Two orders of magnitude below the ion measured in the same rows.
    assert abs(result["mobility"]) < TMA_LITERATURE_MOBILITY / 50.0


#: A gradient-reversed repeat pair, identified by the user: the same
#: lithium acetate sample run twice with the gradients switched.
LIAC = os.path.join("test_data", "Diff50_eNMR_Li_Ac_6p25M_2024-09-03")


def _liac_slope(expt):
    path = os.path.join(LIAC, expt)
    spectrum = io.load(path, pseudo_2d=True, autophase=False)
    ppm = spectrum.axes[1].values("ppm")
    rows = spectrum.data + 1j * spectrum.data_imag
    vdlist = io_lists.read_value_list(os.path.join(path, "vdlist"))
    voltages = ((vdlist["values"] * 1e6 - 0.5) * 2)[: rows.shape[0]]
    # The acetate methyl, ~1.47 ppm.
    phases = enmr.phase_ramp(rows, ppm, (1.34, 1.60), voltages)
    return enmr.fit_enmr(voltages, phases)


@pytest.mark.skipif(not os.path.isdir(LIAC), reason="needs the real eNMR dataset")
def test_gradient_reversed_repeats_oppose_and_agree_in_magnitude():
    first, second = _liac_slope("20"), _liac_slope("21")
    # Reversing the gradient flips the sign of the encoded phase, so
    # the two slopes oppose before anything corrects for it.
    assert first["slope"] < 0 < second["slope"]
    # And the magnitudes agree, which is what makes averaging them
    # meaningful rather than wishful.
    assert abs(second["slope"]) == pytest.approx(abs(first["slope"]), rel=0.15)

    combined = enmr.combine_experiments([first, second])
    assert combined["sign_flipped"].tolist() == [False, True]
    assert combined["spread"] < 0.15 * abs(combined["slope"])


@pytest.mark.skipif(not os.path.isdir(LIAC), reason="needs the real eNMR dataset")
def test_interleaved_unwrap_is_what_makes_that_pair_work():
    # Regression for the real bug: on expt 21 a stride-1 unwrap invents
    # a 2*pi step in the last polarity pair, which turns a clean ramp
    # (R^2 ~ 0.88) into no correlation at all (R^2 ~ 0.01) -- and the
    # pair then looks like one good run and one dead one rather than a
    # reversed repeat.
    path = os.path.join(LIAC, "21")
    spectrum = io.load(path, pseudo_2d=True, autophase=False)
    ppm = spectrum.axes[1].values("ppm")
    rows = spectrum.data + 1j * spectrum.data_imag
    vdlist = io_lists.read_value_list(os.path.join(path, "vdlist"))
    voltages = ((vdlist["values"] * 1e6 - 0.5) * 2)[: rows.shape[0]]

    correct = enmr.fit_enmr(voltages, enmr.phase_ramp(rows, ppm, (1.34, 1.60), voltages))
    naive = enmr.fit_enmr(voltages, enmr.phase_series(rows, ppm, (1.34, 1.60)))

    assert correct["r_squared"] > 0.85
    assert naive["r_squared"] < 0.1


# ============================================================
# Reading the PFG constants from the experiment
# ============================================================

@pytest.mark.skipif(not os.path.isdir(CALIBRATION), reason="needs the real eNMR dataset")
def test_pfg_params_reads_the_constants_from_the_experiment():
    spectrum = io.load(CALIBRATION, pseudo_2d=True)
    params = enmr.pfg_params(spectrum)
    assert params["gradient"] == pytest.approx(0.3)
    assert params["delta"] == pytest.approx(1e-3)
    assert params["big_delta"] == pytest.approx(0.1)
    # gamma comes back in rad/s/T, not the Hz/T the file records --
    # converting once here is what stops a stray 2*pi at a call site.
    assert params["gamma"] == pytest.approx(2 * np.pi * 42577464.6, rel=1e-6)
    assert all(source.startswith("diff.xml") for source in params["sources"].values())


@pytest.mark.skipif(not os.path.isdir(CALIBRATION), reason="needs the real eNMR dataset")
def test_pfg_params_cross_checks_big_delta_against_the_pulse_programme():
    # Delta = d20 reproduces diff.xml exactly on every real dataset
    # here, so agreement is recorded and disagreement would be flagged.
    spectrum = io.load(CALIBRATION, pseudo_2d=True)
    assert "acqus d20" in enmr.pfg_params(spectrum)["sources"]["big_delta"]


@pytest.mark.skipif(not os.path.isdir(CALIBRATION), reason="needs the real eNMR dataset")
def test_fit_enmr_can_read_its_own_constants():
    # Only the electrode gap should need typing: it is the one number
    # no file records.
    ppm, rows, voltages, _diff = _calibration_series()
    spectrum = io.load(CALIBRATION, pseudo_2d=True)
    phases = enmr.phase_ramp(rows, ppm, (3.18, 3.42), voltages)

    result = enmr.fit_enmr(voltages, phases, spectrum=spectrum,
                            electrode_gap=ELECTRODE_GAP_M, cell_polarity=CELL_POLARITY)
    assert result["mobility"] == pytest.approx(TMA_LITERATURE_MOBILITY, rel=0.05)


@pytest.mark.skipif(not os.path.isdir(CALIBRATION), reason="needs the real eNMR dataset")
def test_an_explicit_constant_overrides_what_was_read():
    ppm, rows, voltages, _diff = _calibration_series()
    spectrum = io.load(CALIBRATION, pseudo_2d=True)
    phases = enmr.phase_ramp(rows, ppm, (3.18, 3.42), voltages)

    doubled = enmr.fit_enmr(voltages, phases, spectrum=spectrum,
                             electrode_gap=ELECTRODE_GAP_M, gradient=0.6,
                             cell_polarity=CELL_POLARITY)
    assert doubled["mobility"] == pytest.approx(TMA_LITERATURE_MOBILITY / 2, rel=0.05)


def test_pfg_params_without_a_diff_xml_says_why(tmp_path):
    # The gradient in T/m needs the probe calibration, which acqus does
    # not carry -- guessing one from a percentage would silently scale
    # every mobility.
    (tmp_path / "acqus").write_text("##$TD= 1024\n")
    with pytest.raises(ValueError, match="No diff.xml"):
        enmr.pfg_params(str(tmp_path))


# ============================================================
# Unfolding the pairs onto a signed axis, and the two arms
# ============================================================

def _signed_pairs(slope=0.008, artefact=0.0, offset=0.0,
                   magnitudes=(0, 8, 16, 24, 32, 40), noise=0.0, seed=0):
    """Phases for a series acquired **positive polarity first**, which
    is the real acquisition order.

    ``slope`` is the field-driven phase, which reverses with the
    polarity. ``artefact`` does not reverse -- it is the convection
    term bubble formation produces, and it is what the two arms exist
    to separate. ``offset`` is a constant common phase.
    """
    magnitudes = np.repeat(np.asarray(magnitudes, dtype=float), 2)
    signs = np.where(np.arange(magnitudes.size) % 2 == 0, 1.0, -1.0)
    phases = offset + signs * slope * magnitudes + artefact * magnitudes
    if noise:
        phases = phases + np.random.default_rng(seed).normal(0, noise, magnitudes.size)
    return magnitudes, phases


def test_unfold_gives_the_first_of_each_pair_the_positive_voltage():
    voltages, phases = _signed_pairs(magnitudes=(0, 8, 16))
    unfolded = enmr.unfold_polarity(voltages, phases)
    assert unfolded["paired"] is True
    np.testing.assert_allclose(unfolded["voltages"], [0, -0, 8, -8, 16, -16])
    # The ramp runs from -Vmax to +Vmax once sorted, rather than
    # folding every point onto its partner.
    np.testing.assert_allclose(
        unfolded["voltages"][unfolded["order"]], [-16, -8, 0, -0, 8, 16])


def test_unfold_can_be_told_the_acquisition_took_negative_first():
    voltages, _ = _signed_pairs(magnitudes=(0, 8, 16))
    flipped = enmr.unfold_polarity(voltages, positive_first=False)
    np.testing.assert_allclose(flipped["voltages"], [-0, 0, -8, 8, -16, 16])


def test_the_zero_pair_is_split_by_acquisition_order_not_by_sign():
    # -0.0 >= 0 is True, so a comparison would put both halves of the
    # V = 0 pair in the positive arm and leave the negative arm a
    # point short. The masks come from the acquisition order instead.
    voltages, phases = _signed_pairs(magnitudes=(0, 8, 16, 24))
    unfolded = enmr.unfold_polarity(voltages, phases)
    assert int(unfolded["positive"].sum()) == 4
    assert int(unfolded["negative"].sum()) == 4
    assert unfolded["positive"][0] and unfolded["negative"][1]


def test_unfold_assumes_nothing_when_the_series_is_not_paired():
    voltages = np.array([0.0, 8.0, 16.0, 24.0, 32.0])
    unfolded = enmr.unfold_polarity(voltages, np.zeros(5))
    assert unfolded["paired"] is False
    np.testing.assert_allclose(unfolded["voltages"], voltages)


def test_both_arms_carry_the_same_slope_when_nothing_else_drifts():
    voltages, phases = _signed_pairs(slope=0.008, offset=1.3)
    branches = enmr.branch_slopes(voltages, phases)
    assert branches["positive_branch"]["slope"] == pytest.approx(0.008)
    assert branches["negative_branch"]["slope"] == pytest.approx(0.008)
    assert branches["slope"] == pytest.approx(0.008)
    assert branches["asymmetry"] == pytest.approx(0.0, abs=1e-12)
    assert branches["agree"] is True
    # Both arms extrapolate to the same phase where the field is off --
    # and that phase is zero, because the shift is quoted against the
    # field-off measurement. The 1.3 rad the spectrum carried is the
    # reference, not part of the answer.
    assert branches["reference_phase"] == pytest.approx(1.3)
    assert branches["referenced_to"] == "V = 0"
    assert branches["zero_crossing"] == pytest.approx(0.0, abs=1e-12)
    assert branches["phases"][0] == pytest.approx(0.0, abs=1e-12)
    assert branches["branch_gap"] == pytest.approx(0.0, abs=1e-12)


def test_the_arms_separate_the_measurement_from_what_did_not_reverse():
    # The whole point of running both polarities. A phase that does
    # not follow the field -- convection -- lands in the asymmetry and
    # leaves the mean alone.
    voltages, phases = _signed_pairs(slope=0.008, artefact=0.003)
    branches = enmr.branch_slopes(voltages, phases)
    assert branches["slope"] == pytest.approx(0.008)
    assert branches["asymmetry"] == pytest.approx(0.003)
    assert branches["agree"] is False


def test_the_asymmetry_is_the_pairwise_half_sum_read_the_other_way():
    # Two views of one number, not two pieces of evidence: the arms'
    # half-difference is identically the slope of the half-sum against
    # |V|. Asserted so nobody reports them as independent.
    voltages, phases = _signed_pairs(slope=0.008, artefact=0.003, noise=0.01, seed=3)
    branches = enmr.branch_slopes(voltages, phases)
    half_sum = (phases[0::2] + phases[1::2]) / 2.0
    common = np.polyfit(np.abs(voltages[0::2]), half_sum, 1)[0]
    assert branches["asymmetry"] == pytest.approx(common)


def test_the_arm_mean_and_the_paired_fit_are_the_same_measurement():
    # fit_enmr differences plus-minus-minus against |V|; the arms read
    # the same data on a signed axis. With the acquisition order known
    # -- positive first -- the two agree outright, sign included.
    voltages, phases = _signed_pairs(slope=0.008, artefact=0.002, noise=0.01, seed=5)
    branches = enmr.branch_slopes(voltages, phases)
    assert branches["slope"] == pytest.approx(enmr.fit_enmr(voltages, phases)["slope"])


def test_branch_slopes_refuses_an_unpaired_series():
    with pytest.raises(ValueError, match="not in .V/-V pairs"):
        enmr.branch_slopes(np.arange(5.0), np.zeros(5))


def test_branch_slopes_refuses_too_few_points_to_judge_an_arm():
    voltages, phases = _signed_pairs(magnitudes=(0, 8))
    with pytest.raises(ValueError, match="at least 3 points"):
        enmr.branch_slopes(voltages, phases)


def test_the_per_arm_fit_line_kinks_at_the_origin_and_the_combined_one_does_not():
    # What makes a disagreement between the polarities visible on the
    # plot rather than only in a report.
    voltages, phases = _signed_pairs(slope=0.008, artefact=0.004)
    branches = enmr.branch_slopes(voltages, phases)
    signed = branches["voltages"]
    # The combined line is straight: one slope across the whole axis.
    straight = np.polyfit(signed, branches["fit_combined"], 1)
    np.testing.assert_allclose(
        branches["fit_combined"], straight[0] * signed + straight[1], atol=1e-12)
    # The per-arm line is not -- the two halves have different slopes.
    positive, negative = branches["positive"], branches["negative"]
    assert (np.polyfit(signed[positive], branches["fit"][positive], 1)[0]
            != pytest.approx(np.polyfit(signed[negative], branches["fit"][negative], 1)[0]))


def test_fit_enmr_carries_the_arm_comparison_with_the_fit():
    # Attached rather than left to a second call, so the check travels
    # with the number it qualifies.
    voltages, phases = _signed_pairs(slope=0.008, artefact=0.003)
    result = enmr.fit_enmr(voltages, phases)
    assert result["branches"] is not None
    assert result["branches"]["agree"] is False
    # Nothing to compare on an unpaired series, and that is not an error.
    assert enmr.fit_enmr(np.arange(5.0), np.arange(5.0) * 0.01)["branches"] is None


def test_the_fit_report_warns_when_the_arms_disagree():
    _, clean = _signed_pairs(slope=0.008)
    voltages, dirty = _signed_pairs(slope=0.008, artefact=0.004)
    assert "WARNING" not in enmr.format_enmr_report(enmr.fit_enmr(voltages, clean))
    warned = enmr.format_enmr_report(enmr.fit_enmr(voltages, dirty))
    assert "WARNING" in warned and "did not reverse" in warned


# ============================================================
# The two arms on real data
# ============================================================

@pytest.mark.skipif(not os.path.isdir(CALIBRATION), reason="needs the real eNMR dataset")
def test_the_calibration_standards_two_arms_agree():
    ppm, rows, voltages, _diff = _calibration_series()
    phases = enmr.phase_ramp(rows, ppm, (3.18, 3.42), voltages)
    branches = enmr.branch_slopes(voltages, phases)

    # Both arms are good straight lines with the same slope, which is
    # what a measurement with nothing else going on looks like.
    assert branches["positive_branch"]["slope"] == pytest.approx(-7.642e-3, rel=0.01)
    assert branches["negative_branch"]["slope"] == pytest.approx(-8.161e-3, rel=0.01)
    assert branches["positive_branch"]["r_squared"] > 0.99
    assert branches["negative_branch"]["r_squared"] > 0.99
    assert branches["agree"] is True
    # What did not reverse is 3% of what did.
    assert branches["asymmetry_fraction"] < 0.05
    # And the arms meet where the field is off.
    assert abs(branches["branch_gap"]) < 0.05


@pytest.mark.skipif(not os.path.isdir(LIAC), reason="needs the real eNMR dataset")
def test_one_direction_convecting_is_caught_where_the_averaged_fit_cannot_see_it():
    # The failure the user described: bubble formation drives
    # convection one way only, so one polarity is wrecked and the
    # other is not. Pair-differencing cancels whatever reversed and
    # says nothing about what did not, so its own R^2 misses this
    # entirely -- only the arms show it.
    path = os.path.join(LIAC, "560")
    spectrum = io.load(path, pseudo_2d=True, autophase=False)
    ppm = spectrum.axes[1].values("ppm")
    rows = spectrum.data + 1j * spectrum.data_imag
    vdlist = io_lists.read_value_list(os.path.join(path, "vdlist"))
    voltages = ((vdlist["values"] * 1e6 - 0.5) * 2)[: rows.shape[0]]
    phases = enmr.phase_ramp(rows, ppm, (1.34, 1.60), voltages)
    branches = enmr.branch_slopes(voltages, phases)

    # One arm is a clean line, the other is wreckage.
    assert branches["positive_branch"]["r_squared"] > 0.98
    assert branches["negative_branch"]["r_squared"] < 0.70
    assert branches["agree"] is False
    # The part that did not reverse is several times the part that did,
    # so their average is not a measurement of anything.
    assert branches["asymmetry_fraction"] > 4.0
    # The arms miss each other badly at the zero crossing too.
    assert abs(branches["branch_gap"]) > 0.4
    assert "DISAGREE" in enmr.format_branch_report(branches)


@pytest.mark.skipif(not os.path.isdir(LIAC), reason="needs the real eNMR dataset")
def test_two_clean_arms_can_still_disagree():
    # Not every contaminated run has an obviously bad arm: on 572 both
    # arms fit well (R^2 > 0.98) and simply have different slopes.
    # Judging a run by fit quality alone would pass this.
    path = os.path.join(LIAC, "572")
    spectrum = io.load(path, pseudo_2d=True, autophase=False)
    ppm = spectrum.axes[1].values("ppm")
    rows = spectrum.data + 1j * spectrum.data_imag
    vdlist = io_lists.read_value_list(os.path.join(path, "vdlist"))
    voltages = ((vdlist["values"] * 1e6 - 0.5) * 2)[: rows.shape[0]]
    phases = enmr.phase_ramp(rows, ppm, (1.34, 1.60), voltages)
    branches = enmr.branch_slopes(voltages, phases)

    assert branches["positive_branch"]["r_squared"] > 0.98
    assert branches["negative_branch"]["r_squared"] > 0.98
    assert branches["asymmetry_fraction"] == pytest.approx(0.28, abs=0.05)
    assert branches["agree"] is False


# ============================================================
# The second pairing: the gradient-reversed repeat
# ============================================================

def _gradient_pair(slope=0.008, artefact=0.0, offsets=(1.3, -0.4),
                    gradients=(0.9, -0.9), magnitudes=(0, 8, 16, 24, 32, 40)):
    """Two whole experiments, one at ``+g`` and one at ``-g``.

    Reversing the gradient reverses everything it encoded, so the
    second experiment's ramp is the first's negated -- on top of its
    own, unrelated receiver phase.
    """
    voltages, first = _signed_pairs(slope=slope, artefact=artefact,
                                     offset=offsets[0], magnitudes=magnitudes)
    _, ramp = _signed_pairs(slope=slope, artefact=artefact, offset=0.0,
                             magnitudes=magnitudes)
    second = offsets[1] - ramp
    return [{"voltages": voltages, "phases": first, "gradient": gradients[0],
             "label": "plus"},
            {"voltages": voltages, "phases": second, "gradient": gradients[1],
             "label": "minus"}]


def test_the_reversed_repeat_is_corrected_by_its_own_recorded_gradient_sign():
    pair = enmr.average_gradient_pair(_gradient_pair(slope=0.008))
    assert pair["gradients_reversed"] is True
    # Both land on the +g frame, so they agree and their average is
    # the same slope.
    np.testing.assert_allclose(pair["slopes"], [0.008, 0.008])
    assert pair["slope"] == pytest.approx(0.008)
    assert pair["agree"] is True
    assert pair["spread"] == pytest.approx(0.0, abs=1e-12)


def test_averaging_removes_the_receiver_phase_each_run_brought_with_it():
    # The constant part of a phase differs between two acquisitions for
    # no physical reason, so each ramp is centred on its own common
    # offset before averaging. A large offset must not tilt anything.
    wide = enmr.average_gradient_pair(_gradient_pair(offsets=(12.0, -30.0)))
    narrow = enmr.average_gradient_pair(_gradient_pair(offsets=(0.0, 0.0)))
    assert wide["slope"] == pytest.approx(narrow["slope"])
    np.testing.assert_allclose(wide["average"], narrow["average"], atol=1e-12)


def test_a_disagreeing_pair_is_not_averaged_silently():
    pair = _gradient_pair(slope=0.008)
    # One direction measures something quite different.
    pair[1]["phases"] = pair[1]["phases"] * 0.3
    combined = enmr.average_gradient_pair(pair)
    assert combined["agree"] is False
    assert "DISAGREE" in enmr.format_gradient_pair_report(combined)


def test_two_experiments_run_the_same_way_round_are_flagged_as_not_a_pair():
    same = _gradient_pair(gradients=(0.9, 0.9))
    # Both at +g: this is a repeat of one direction, and nothing that
    # follows the gradient cancels between them.
    combined = enmr.average_gradient_pair(same)
    assert combined["gradients_reversed"] is False
    # Collapsed, because the report word-wraps its prose.
    flat = " ".join(enmr.format_gradient_pair_report(combined).split())
    assert "gradient the same way round" in flat


def test_ramps_measured_at_different_voltages_are_refused():
    pair = _gradient_pair()
    other, phases = _signed_pairs(magnitudes=(0, 5, 10, 15, 20, 25))
    pair[1] = {**pair[1], "voltages": other, "phases": phases}
    with pytest.raises(ValueError, match="different voltages"):
        enmr.average_gradient_pair(pair)


def test_averaging_needs_two_experiments_and_a_gradient_for_each():
    with pytest.raises(ValueError, match="at least 2 experiments"):
        enmr.average_gradient_pair(_gradient_pair()[:1])
    missing = _gradient_pair()
    del missing[1]["gradient"]
    with pytest.raises(ValueError, match="signed 'gradient'"):
        enmr.average_gradient_pair(missing)
    zeroed = _gradient_pair(gradients=(0.9, 0.0))
    with pytest.raises(ValueError, match="zero gradient"):
        enmr.average_gradient_pair(zeroed)


def test_each_experiment_is_measured_through_its_arms_not_one_line():
    # The estimator matters: when the arms disagree they also have
    # different intercepts, and one shared intercept then drags the
    # slope. Both levels use the arms so the numbers are comparable.
    experiments = _gradient_pair(slope=0.008, artefact=0.004)
    combined = enmr.average_gradient_pair(experiments)
    arms = enmr.branch_slopes(experiments[0]["voltages"], experiments[0]["phases"])
    assert combined["experiments"][0]["slope"] == pytest.approx(arms["slope"])
    assert combined["slope"] == pytest.approx(0.008)


def test_the_drawn_arm_line_is_continuous_at_the_origin_and_still_kinks():
    voltages, phases = _signed_pairs(slope=0.008, artefact=0.004)
    branches = enmr.branch_slopes(voltages, phases)
    signed = branches["voltages"]
    # Both arms meet at the shared zero crossing -- no vertical jump to
    # read as a spike on the plot.
    at_zero = branches["fit_joined"][np.abs(signed) == 0]
    np.testing.assert_allclose(at_zero, branches["centre"])
    # But the two slopes are still their own, so a disagreement shows.
    assert (np.polyfit(signed[branches["positive"]],
                       branches["fit_joined"][branches["positive"]], 1)[0]
            == pytest.approx(branches["positive_branch"]["slope"]))


@pytest.mark.skipif(not os.path.isdir(LIAC), reason="needs the real eNMR dataset")
def test_the_gradient_sign_is_read_from_diff_xml_not_guessed():
    # Every reversed repeat in this dataset records the sign; the
    # earlier align-by-slope guesswork was never necessary.
    signs = {expt: np.sign(
        enmr.pfg_params(os.path.join(LIAC, expt), nucleus="1H")["gradient"])
             for expt in ("20", "21", "560", "561", "572", "573")}
    assert signs == {"20": 1.0, "21": -1.0, "560": 1.0,
                     "561": -1.0, "572": 1.0, "573": -1.0}


def _liac_series(expt, region=(1.34, 1.60)):
    path = os.path.join(LIAC, expt)
    spectrum = io.load(path, pseudo_2d=True, autophase=False)
    ppm = spectrum.axes[1].values("ppm")
    rows = spectrum.data + 1j * spectrum.data_imag
    vdlist = io_lists.read_value_list(os.path.join(path, "vdlist"))
    voltages = ((vdlist["values"] * 1e6 - 0.5) * 2)[: rows.shape[0]]
    return {"voltages": voltages,
            "phases": enmr.phase_ramp(rows, ppm, region, voltages),
            "gradient": enmr.pfg_params(spectrum)["gradient"], "label": expt}


@pytest.mark.skipif(not os.path.isdir(LIAC), reason="needs the real eNMR dataset")
def test_a_real_gradient_pair_that_agrees_and_one_that_does_not():
    agreeing = enmr.average_gradient_pair([_liac_series("20"), _liac_series("21")])
    assert agreeing["gradients_reversed"] is True
    np.testing.assert_allclose(agreeing["slopes"], [-1.244e-2, -1.396e-2], rtol=0.01)
    assert agreeing["spread_fraction"] < 0.10
    assert agreeing["agree"] is True

    # 560/561 is the same sample and the same pairing, and does not.
    disagreeing = enmr.average_gradient_pair([_liac_series("560"), _liac_series("561")])
    assert disagreeing["spread_fraction"] > 0.5
    assert disagreeing["agree"] is False


@pytest.mark.skipif(not os.path.isdir(LIAC), reason="needs the real eNMR dataset")
def test_the_single_line_estimator_would_have_been_wrong_on_real_data():
    # Why _measure_ramp goes through the arms. On experiment 20, whose
    # arms differ by 93%, one shared intercept moves the slope by a
    # quarter -- an error from the estimator, not from the data.
    series = _liac_series("20")
    arms = enmr.branch_slopes(series["voltages"], series["phases"])
    unfolded = enmr.unfold_polarity(series["voltages"], series["phases"])
    one_line = np.polyfit(unfolded["voltages"], series["phases"], 1)[0]
    assert arms["slope"] == pytest.approx(-1.244e-2, rel=0.01)
    assert one_line == pytest.approx(-9.37e-3, rel=0.01)
    assert abs(one_line - arms["slope"]) / abs(arms["slope"]) > 0.2


# ============================================================
# Trimming the top of the voltage range
# ============================================================

def test_a_voltage_cutoff_drops_both_polarities_of_each_pair():
    voltages, phases = _signed_pairs(magnitudes=(0, 8, 16, 24, 32, 40))
    branches = enmr.branch_slopes(voltages, phases, max_voltage=24)
    used = branches["used"]
    # 0, 8, 16 and 24 survive at both polarities; 32 and 40 do not.
    assert int(used.sum()) == 8
    assert np.all(np.abs(branches["voltages"][used]) <= 24)
    # Balanced: an unbalanced pair would make the arms measure
    # different ranges and read that as an asymmetry.
    assert int((branches["positive"] & used).sum()) == 4
    assert int((branches["negative"] & used).sum()) == 4


def test_an_explicit_exclusion_removes_one_point_and_keeps_its_partner():
    # One polarised electrode bubbles and shifts one polarity only, so
    # the bad points do not come in pairs. Dropping the partner too
    # would discard good data to preserve a symmetry already broken.
    voltages, phases = _signed_pairs(magnitudes=(0, 8, 16, 24, 32, 40))
    branches = enmr.branch_slopes(voltages, phases, exclude=[4])
    assert not branches["used"][4]              # +16 V, removed
    assert branches["used"][5]                  # -16 V, kept
    assert int(branches["used"].sum()) == 11
    # The arms now cover different voltages, and the result says so --
    # part of the asymmetry is that difference, not the sample.
    assert branches["balanced"] is False


def test_exclusions_can_be_widened_to_the_partner_on_request():
    voltages, phases = _signed_pairs(magnitudes=(0, 8, 16, 24, 32, 40))
    branches = enmr.branch_slopes(voltages, phases, exclude=[4], balance_pairs=True)
    assert not branches["used"][4] and not branches["used"][5]
    assert int(branches["used"].sum()) == 10
    assert branches["balanced"] is True


def test_a_one_sided_range_trims_only_the_polarity_that_deviates():
    # The distortion runs one direction, so the range worth keeping is
    # not centred on zero.
    voltages, phases = _signed_pairs(magnitudes=(0, 8, 16, 24, 32, 40))
    branches = enmr.branch_slopes(voltages, phases, max_voltage=(-16, 40))
    signed = branches["voltages"][branches["used"]]
    assert signed.min() == pytest.approx(-16) and signed.max() == pytest.approx(40)
    assert branches["balanced"] is False
    assert "-16 ≤ V ≤ 40 V" in enmr.format_branch_report(branches)


def test_the_high_voltage_points_can_be_kept_off_a_fit_they_would_pull():
    # A ramp that goes nonlinear at the top: the straight part measures
    # 0.008 rad/V, and the last pair is off it.
    voltages, phases = _signed_pairs(slope=0.008, magnitudes=(0, 8, 16, 24, 32, 40))
    phases[-2:] += np.array([0.9, -0.9])       # the +-40 V pair, badly off

    assert enmr.branch_slopes(voltages, phases)["slope"] != pytest.approx(0.008, rel=0.05)
    trimmed = enmr.branch_slopes(voltages, phases, max_voltage=32)
    assert trimmed["slope"] == pytest.approx(0.008)
    # Excluded, not deleted: still returned, so the plot can show them.
    assert int(trimmed["excluded"].sum()) == 2
    assert trimmed["phases"].size == voltages.size
    # And the fitted line is extrapolated across them, so the deviation
    # is visible rather than hidden.
    assert trimmed["fit_joined"].size == voltages.size


def test_the_cutoff_applies_to_the_headline_slope_and_the_arms_alike():
    # A report whose slope and arm comparison came from different
    # points would contradict itself.
    voltages, phases = _signed_pairs(slope=0.008, magnitudes=(0, 8, 16, 24, 32, 40))
    phases[-2:] += np.array([0.9, -0.9])
    result = enmr.fit_enmr(voltages, phases, max_voltage=32)
    assert result["slope"] == pytest.approx(0.008)
    assert result["branches"]["slope"] == pytest.approx(0.008)
    assert int(result["used"].sum()) == 5          # pairs, not points
    report = enmr.format_enmr_report(result)
    assert "|V| ≤ 32 V" in report


def test_too_hard_a_cutoff_says_it_was_the_cutoff():
    voltages, phases = _signed_pairs(magnitudes=(0, 8, 16, 24, 32, 40))
    with pytest.raises(ValueError, match="[Rr]aise the voltage cutoff"):
        enmr.branch_slopes(voltages, phases, max_voltage=8)


def test_the_same_cutoff_is_applied_to_every_repeat():
    pair = _gradient_pair(slope=0.008, magnitudes=(0, 8, 16, 24, 32, 40))
    combined = enmr.average_gradient_pair(pair, max_voltage=24)
    assert int(combined["used"].sum()) == 8
    assert combined["max_voltage"] == 24
    assert "|V| ≤ 24 V" in enmr.format_gradient_pair_report(combined)
    assert "the same range" in " ".join(
        enmr.format_gradient_pair_report(combined).split())


def test_the_cutoff_sweep_shows_what_the_choice_costs():
    voltages, phases = _signed_pairs(slope=0.008, magnitudes=(0, 8, 16, 24, 32, 40))
    swept = enmr.cutoff_sweep(voltages, phases)
    # Largest first, and it stops when an arm runs out of points
    # rather than raising.
    assert [entry["cutoff"] for entry in swept] == [40, 32, 24, 16]
    # A clean ramp gives the same slope at every cutoff -- which is
    # what a slope the range is not deciding looks like.
    assert all(entry["slope"] == pytest.approx(0.008) for entry in swept)


@pytest.mark.skipif(not os.path.isdir(LIAC), reason="needs the real eNMR dataset")
def test_on_real_data_the_cutoff_moves_the_answer_a_long_way():
    # The reason cutoff_sweep exists. On 20/21 the slope changes
    # sevenfold across the range while R^2 *improves*, so a cutoff
    # picked by fit quality alone picks the answer.
    pair = [_liac_series("20"), _liac_series("21")]
    full = enmr.average_gradient_pair(pair)
    trimmed = enmr.average_gradient_pair(pair, max_voltage=40)
    assert full["slope"] == pytest.approx(-1.32e-2, rel=0.02)
    assert trimmed["slope"] == pytest.approx(-2.66e-3, rel=0.05)
    assert trimmed["r_squared"] > full["r_squared"]
    assert abs(full["slope"] / trimmed["slope"]) > 4


def test_cell_polarity_is_plus_or_minus_one_and_nothing_else():
    with pytest.raises(ValueError, match="cell_polarity is \\+1 or -1"):
        enmr.mobility_from_slope(0.008, 0.0385, 2.67e8, 0.3, 1e-3, 0.1,
                                  cell_polarity=0)


def test_cell_polarity_moves_the_mobility_and_leaves_the_slope_alone():
    voltages, phases = _synthetic(slope_rad_per_volt=0.008)
    forward = enmr.fit_enmr(voltages, phases, **_fit_kwargs())
    reversed_cell = enmr.fit_enmr(voltages, phases, cell_polarity=-1, **_fit_kwargs())
    assert reversed_cell["slope"] == pytest.approx(forward["slope"])
    assert reversed_cell["mobility"] == pytest.approx(-forward["mobility"])


def test_removing_one_point_loses_that_pairs_half_difference_but_not_the_arm():
    # The honest consequence of point-level removal: a half-difference
    # needs both polarities and simply loses the pair, while the arm
    # fit uses each point on its own and keeps the survivor.
    voltages, phases = _signed_pairs(slope=0.008, magnitudes=(0, 8, 16, 24, 32, 40))
    result = enmr.fit_enmr(voltages, phases, exclude=[8])   # the +32 V point

    assert result["broken_pairs"] == 1
    assert int(result["used"].sum()) == 5                   # of 6 pairs
    # The arm fit kept 11 of 12 points, not 10.
    assert int(result["branches"]["used"].sum()) == 11
    assert result["branches"]["balanced"] is False
    report = enmr.format_enmr_report(result)
    assert "could not" in report and "arm fit" in report


@pytest.mark.skipif(not os.path.isdir(LIAC), reason="needs the real eNMR dataset")
def test_removing_points_beats_removing_pairs_on_a_one_sided_failure():
    # Experiment 560's negative arm is clean to -60 V and then breaks
    # upward: -2.649, then +3.081, +3.867, +4.483 at -80, -100, -120.
    # The positive arm is a good line across the whole range. That is a
    # single polarised electrode bubbling, and it is the case where
    # dropping *pairs* throws away good data to preserve a symmetry the
    # artefact has already broken.
    series = _liac_series("560")
    voltages, phases = series["voltages"], series["phases"]

    everything = enmr.branch_slopes(voltages, phases)
    assert everything["asymmetry_fraction"] > 4.0          # 465%

    # Three points -- not three pairs.
    points = enmr.branch_slopes(voltages, phases, exclude=[11, 13, 15])
    assert points["balanced"] is False
    assert int(points["used"].sum()) == 13
    assert points["positive_branch"]["r_squared"] > 0.98
    assert points["negative_branch"]["r_squared"] > 0.99
    assert points["asymmetry_fraction"] < 0.05             # 465% -> 3%
    # The positive arm is untouched: nothing was removed from it.
    assert (points["positive_branch"]["slope"]
            == pytest.approx(everything["positive_branch"]["slope"]))

    # The symmetric cutoff that removes the same bad points also
    # removes the three good ones opposite them, and agrees less well
    # from fewer points.
    pairs = enmr.branch_slopes(voltages, phases, max_voltage=60)
    assert pairs["balanced"] is True
    assert int(pairs["used"].sum()) == 10
    assert pairs["asymmetry_fraction"] > points["asymmetry_fraction"]


# ============================================================
# Quoting the shift against the field-off measurement
# ============================================================

def test_the_shift_is_quoted_against_the_zero_volt_measurement():
    # 0 V is the acquisition's own record of "no shift": the same
    # receiver phase, shim and lineshape, with no electrophoresis.
    voltages, phases = _signed_pairs(slope=0.008, offset=2.75)
    branches = enmr.branch_slopes(voltages, phases)
    assert branches["reference_phase"] == pytest.approx(2.75)
    # Both 0 V points read zero, which is the whole request.
    at_zero = branches["voltages"] == 0
    np.testing.assert_allclose(branches["phases"][at_zero], 0.0, atol=1e-12)


def test_referencing_is_a_change_of_origin_and_nothing_else():
    # It must not be able to move the measurement -- only what the
    # numbers are quoted against.
    voltages, phases = _signed_pairs(slope=0.008, artefact=0.003,
                                      offset=2.75, noise=0.01, seed=11)
    referenced = enmr.branch_slopes(voltages, phases)
    raw = enmr.branch_slopes(voltages, phases, reference=None)
    for key in ("slope", "asymmetry", "r_squared", "branch_gap",
                 "positive_branch", "negative_branch"):
        left, right = referenced[key], raw[key]
        if isinstance(left, dict):
            assert left["slope"] == pytest.approx(right["slope"])
            assert left["r_squared"] == pytest.approx(right["r_squared"])
        else:
            assert left == pytest.approx(right)
    assert raw["reference_phase"] == 0.0
    # The reference is the *measured* zero, so with noise it is near
    # 2.75 rather than exactly it -- which is the point of measuring it.
    assert referenced["reference_phase"] == pytest.approx(2.75, abs=0.05)
    np.testing.assert_allclose(referenced["phases"] + referenced["reference_phase"],
                                raw["phases"], atol=1e-12)


def test_a_series_with_no_zero_volt_point_says_so_rather_than_inventing_one():
    voltages, phases = _signed_pairs(slope=0.008, offset=2.75,
                                      magnitudes=(8, 16, 24, 32, 40))
    branches = enmr.branch_slopes(voltages, phases)
    assert branches["reference_phase"] == 0.0
    assert "no V = 0 point" in branches["referenced_to"]


def test_an_unknown_reference_is_refused():
    voltages, phases = _signed_pairs()
    with pytest.raises(ValueError, match="reference is 'zero'"):
        enmr.branch_slopes(voltages, phases, reference="centre")


def test_each_repeat_is_referenced_to_its_own_field_off_measurement():
    # Two acquisitions carry unrelated receiver phases; each is quoted
    # against its own zero before they are averaged.
    pair = _gradient_pair(slope=0.008, offsets=(2.75, -1.4))
    combined = enmr.average_gradient_pair(pair)
    at_zero = combined["voltages"] == 0
    np.testing.assert_allclose(combined["corrected"][:, at_zero], 0.0, atol=1e-12)
    np.testing.assert_allclose(combined["average"][at_zero], 0.0, atol=1e-12)
    assert combined["slope"] == pytest.approx(0.008)


def test_the_average_is_measured_through_its_arms_not_a_single_line():
    # Regression: `reference` was both a parameter and a local holding
    # the shared voltage axis, so `reference == "zero"` compared an
    # array and raised -- and a bare `except ValueError` turned that
    # into a silent fall back to the single-line estimator. It moved
    # the real 20/21 answer from -1.32e-2 to -1.05e-2 and said nothing.
    pair = _gradient_pair(slope=0.008, artefact=0.004)
    combined = enmr.average_gradient_pair(pair)
    assert combined["branches"] is not None
    assert combined["slope"] == pytest.approx(0.008)
