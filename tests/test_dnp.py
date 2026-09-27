"""DNP: enhancement factors and field-sweep profiles (opennmr.notebook.dnp).

The synthetic spectra here are built the way the Bruker reader builds
real ones -- an Hz axis relative to the reference frequency SF, SFO1 and
RG in ``metadata.raw["acqus"]`` -- with each line placed at the absolute
frequency gamma * B0 * (1 + delta) for a *known* B0, so every field
the module reports can be checked against the truth.
"""
import math
import os

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import dnp
from opennmr.notebook.utils import find_gamma

GAMMA_NA = find_gamma("23Na")["gamma"]           # Hz/T
SF = 105.892246                                  # MHz, the ppm reference (= BF1)
DELTA = 250.0                                    # ppm, the line's chemical shift
FWHM = 1000.0                                    # Hz


def _lorentz(x, centre, fwhm, height):
    return height / (1.0 + ((x - centre) / (0.5 * fwhm)) ** 2)


def _spectrum(field_t, carrier_mhz, height=1000.0, fwhm=FWHM, rg=100.0, ns=16, sw=200e3,
              n=16384, noise=1.0, nucleus="23Na", delta=DELTA, pulse_program="hahnecho",
              seed=0, extra_lines=()):
    """A 1D 23Na spectrum acquired at ``field_t`` with the carrier at
    ``carrier_mhz``: the line at gamma*B0*(1+delta)."""
    rng = np.random.default_rng(seed)
    gamma = find_gamma(nucleus)["gamma"]
    line_abs = gamma * field_t * (1 + delta * 1e-6)
    offset = np.linspace(sw / 2, -sw / 2, n)             # descending, Bruker order
    absolute = carrier_mhz * 1e6 + offset
    hz = absolute - SF * 1e6
    y = _lorentz(absolute, line_abs, fwhm, height) + noise * rng.standard_normal(n)
    for rel_offset, h in extra_lines:
        y += _lorentz(absolute, line_abs + rel_offset, fwhm, h)
    md = NMRMetadata(vendor="bruker", nucleus=nucleus, sfo1=SF, bf1=SF,
                     o1=(carrier_mhz - SF) * 1e6, ns=ns, pulse_program=pulse_program,
                     raw={"acqus": {"SFO1": carrier_mhz, "RG": rg, "NS": ns}})
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=hz / SF, hz=hz)], metadata=md)


def _line_offset(field_t, carrier_mhz, delta=DELTA):
    return GAMMA_NA * field_t * (1 + delta * 1e-6) - carrier_mhz * 1e6


# -- frequencies and fields ----------------------------------------------

def test_field_from_frequency_inverts_gamma():
    assert dnp.field_from_frequency(GAMMA_NA * 9.4, "23Na") == pytest.approx(9.4, rel=1e-12)
    # A shift scales the field by 1/(1 + delta): 1 ppm -> 1e-6 relative.
    b = dnp.field_from_frequency(GAMMA_NA * 9.4 * (1 + 100e-6), "23Na", shift_ppm=100.0)
    assert b == pytest.approx(9.4, rel=1e-12)


def test_carrier_is_sfo1_not_the_reference_frequency():
    # openNMR's metadata.sfo1 holds SF (the ppm reference); the carrier
    # is acqus SFO1. They differ by O1 -- here 50 kHz.
    s = _spectrum(9.4, SF + 0.05)
    carrier, source = dnp.carrier_frequency_mhz(s)
    assert source == "SFO1" and carrier == pytest.approx(SF + 0.05)
    assert dnp.reference_frequency_mhz(s) == pytest.approx(SF)


def test_carrier_falls_back_to_bf1_plus_o1():
    s = _spectrum(9.4, SF + 0.05)
    s.metadata.raw = {}
    carrier, source = dnp.carrier_frequency_mhz(s)
    assert source == "BF1 + O1" and carrier == pytest.approx(SF + 0.05, abs=1e-12)


def test_carrier_offset_axis_puts_the_line_where_it_is_relative_to_the_carrier():
    field = 9.4
    carrier = GAMMA_NA * field * (1 + DELTA * 1e-6) / 1e6 - 0.003     # line 3 kHz above carrier
    s = _spectrum(field, carrier)
    off = dnp.carrier_offset_hz(s)
    assert off[np.argmax(s.data)] == pytest.approx(3000.0, abs=15)


def test_field_from_sfo1_assumes_the_line_on_the_carrier():
    s = _spectrum(9.4, 105.9)
    assert dnp.field_from_sfo1(s) == pytest.approx(105.9e6 / GAMMA_NA, rel=1e-12)


# -- enhancement -----------------------------------------------------------

def _pair(h_on=12500.0, h_off=1000.0, fwhm_on=FWHM, rg_on=100.0, rg_off=100.0, carrier_off=None):
    field = 9.4
    carrier = GAMMA_NA * field * (1 + DELTA * 1e-6) / 1e6
    on = _spectrum(field, carrier, height=h_on, fwhm=fwhm_on, rg=rg_on, seed=1)
    off = _spectrum(field, carrier if carrier_off is None else carrier_off, height=h_off,
                    rg=rg_off, seed=2)
    peak_ppm = on.axes[0].ppm[np.argmax(np.abs(on.data))]
    return on, off, peak_ppm


def test_enhancement_by_height_and_area_agree_for_an_unchanged_lineshape():
    on, off, pk = _pair()
    r = dnp.enhancement(on, off, peak=pk + 1.0, region=(pk - 60, pk + 60))
    assert r["intensity"]["ratio"] == pytest.approx(12.5, rel=0.01)
    assert r["area"]["ratio"] == pytest.approx(12.5, rel=0.01)
    assert r["agreement"]["consistent"]
    assert r["intensity"]["position_1"] == pytest.approx(pk, abs=0.3)   # snapped to the max


def test_enhancement_is_signed():
    on, off, pk = _pair(h_on=-4000.0)
    r = dnp.enhancement(on, off, peak=pk, region=(pk - 60, pk + 60))
    assert r["intensity"]["ratio"] == pytest.approx(-4.0, rel=0.01)
    assert r["area"]["ratio"] == pytest.approx(-4.0, rel=0.02)


def test_broadening_makes_height_and_area_disagree_and_says_so():
    # Same area, twice the width: height halves, area does not.
    on, off, pk = _pair(h_on=5000.0, fwhm_on=2 * FWHM)
    r = dnp.enhancement(on, off, peak=pk, region=(pk - 150, pk + 150))
    assert r["intensity"]["ratio"] == pytest.approx(5.0, rel=0.02)
    assert r["area"]["ratio"] == pytest.approx(10.0, rel=0.05)
    assert not r["agreement"]["consistent"]
    assert any("line shape differs" in n for n in r["notes"])


def test_enhancement_encourages_the_other_measurement():
    on, off, pk = _pair()
    by_height = dnp.enhancement(on, off, peak=pk)
    assert by_height["area"] is None
    assert any("integrate a region too" in n for n in by_height["notes"])
    by_area = dnp.enhancement(on, off, region=(pk - 60, pk + 60))
    assert by_area["intensity"] is None
    assert any("pick the peak too" in n for n in by_area["notes"])
    with pytest.raises(ValueError):
        dnp.enhancement(on, off)


def test_enhancement_corrects_receiver_gain():
    # Same signal per unit gain: acquired at 4x the RG, the on-spectrum
    # is 4x bigger on disk, which is not enhancement.
    on, off, pk = _pair(h_on=4 * 12500.0, rg_on=400.0, rg_off=100.0)
    r = dnp.enhancement(on, off, peak=pk, region=(pk - 60, pk + 60))
    assert r["intensity"]["ratio"] == pytest.approx(12.5, rel=0.01)
    assert r["area"]["ratio"] == pytest.approx(12.5, rel=0.01)
    assert any("Receiver gains differ" in n for n in r["notes"])
    raw = dnp.enhancement(on, off, peak=pk, correct_rg=False)
    assert raw["intensity"]["ratio"] == pytest.approx(50.0, rel=0.01)


def test_enhancement_warns_when_carriers_differ():
    on, off, pk = _pair(carrier_off=GAMMA_NA * 9.4 * (1 + DELTA * 1e-6) / 1e6 + 0.001)
    r = dnp.enhancement(on, off, peak=pk)
    assert any("carriers differ" in n for n in r["notes"])


def test_enhancement_report_names_both_ratios():
    on, off, pk = _pair()
    text = dnp.enhancement_report(dnp.enhancement(on, off, peak=pk, region=(pk - 60, pk + 60)))
    assert "By peak height" in text and "By area" in text and "agree" in text


def test_enhancement_refuses_2d():
    on, off, pk = _pair()
    two_d = NMRSpectrum(data=np.zeros((2, 3)), axes=[NMRAxis("index", index=np.arange(2)),
                                                      NMRAxis("frequency", ppm=np.arange(3.0))],
                        metadata=NMRMetadata(vendor="bruker", nucleus="23Na"))
    with pytest.raises(ValueError, match="1D"):
        dnp.enhancement(two_d, off, peak=pk)


# -- field sweep -------------------------------------------------------------

def _sweep(fields, carrier_errors_hz, heights=None, **kw):
    """Spectra at the given fields, each carrier set ``carrier_error``
    Hz *below* the line -- SFO1 following the line imperfectly, as it
    does by hand."""
    heights = heights if heights is not None else [1000.0] * len(fields)
    spectra = []
    for k, (b, err, h) in enumerate(zip(fields, carrier_errors_hz, heights)):
        line = GAMMA_NA * b * (1 + DELTA * 1e-6)
        spectra.append(_spectrum(b, (line - err) / 1e6, height=h, seed=10 + k, **kw))
    return spectra


FIELDS = [9.3962, 9.3966, 9.3970, 9.3974, 9.3978]
ERRORS = [0.0, 8000.0, -6000.0, 12000.0, -3000.0]      # line above/below its carrier, Hz


def test_field_sweep_finds_the_true_field_from_the_line():
    spectra = _sweep(FIELDS, ERRORS)
    r = dnp.field_sweep(spectra, peak=0.0, shift_ppm=DELTA)
    for row, b in zip(r["rows"], FIELDS):
        assert row["found"]
        # One point is 12 Hz here: 1 uT for 23Na. The line's own
        # frequency gives the field to that.
        assert row["field_T"] == pytest.approx(b, abs=2e-6)
        assert row["field_source"] == "peak"


def test_field_from_sfo1_is_wrong_by_the_carrier_error_and_is_reported():
    spectra = _sweep(FIELDS, ERRORS)
    r = dnp.field_sweep(spectra, peak=0.0, shift_ppm=DELTA)
    for row, err in zip(r["rows"], ERRORS):
        expected_mT = -err / (GAMMA_NA * (1 + DELTA * 1e-6)) * 1e3
        assert row["sfo1_error_mT"] == pytest.approx(expected_mT, abs=2e-3)
    assert any("SFO1 did not follow the line" in n for n in r["notes"])
    by_sfo1 = dnp.field_sweep(spectra, peak=0.0, shift_ppm=DELTA, field_from="sfo1")
    assert [row["field_source"] for row in by_sfo1["rows"]] == ["SFO1"] * len(FIELDS)


def test_sfo1_errors_can_misorder_the_sweep_but_the_line_does_not():
    # Two fields 0.4 mT apart; the carrier of the higher one set 12 kHz
    # (1.06 mT) low. From SFO1 the order flips, from the line it does not
    # -- exactly what happened to 100287/100288 of the first real sweep.
    fields = [9.3800, 9.3804]
    spectra = _sweep(fields, [0.0, 12000.0])
    r = dnp.field_sweep(spectra, peak=0.0, shift_ppm=DELTA)
    assert r["rows"][0]["field_T"] < r["rows"][1]["field_T"]
    assert r["rows"][0]["field_sfo1_T"] > r["rows"][1]["field_sfo1_T"]


def test_default_shift_makes_the_reference_spectrum_sit_at_its_nominal_field():
    # With delta read off the reference's own ppm axis, the reference's
    # field is SF / gamma: the field its ppm axis already implies.
    spectra = _sweep(FIELDS, ERRORS)
    r = dnp.field_sweep(spectra, peak=0.0, reference=2)
    assert r["rows"][2]["field_T"] == pytest.approx(SF * 1e6 / GAMMA_NA, rel=1e-9)
    assert r["rows"][2]["delta_B_mT"] == 0.0
    # ... and differences between fields are the same as with the true delta.
    exact = dnp.field_sweep(spectra, peak=0.0, reference=2, shift_ppm=DELTA)
    for a, b in zip(r["rows"], exact["rows"]):
        assert a["delta_B_mT"] == pytest.approx(b["delta_B_mT"], abs=2e-3)


def test_intensity_and_area_are_tracked_with_the_line_and_signed():
    heights = [1000.0, 3000.0, -2000.0, 500.0, 1500.0]
    spectra = _sweep(FIELDS, ERRORS, heights=heights)
    r = dnp.field_sweep(spectra, peak=0.0, region=(-3000, 3000), shift_ppm=DELTA)
    area_1 = r["rows"][0]["area"]
    for row, h in zip(r["rows"], heights):
        assert row["intensity"] == pytest.approx(h, rel=0.015, abs=5)
        # The region moved with the line, so it holds the same fraction
        # of every line: area proportional to height.
        assert row["area"] / area_1 == pytest.approx(h / heights[0], rel=0.01)
        assert row["region_Hz"][1] - row["region_Hz"][0] == pytest.approx(6000.0)


def test_without_tracking_a_fixed_region_misses_lines_off_the_carrier():
    spectra = _sweep(FIELDS, ERRORS)
    tracked = dnp.field_sweep(spectra, peak=0.0, region=(-3000, 3000), shift_ppm=DELTA)
    fixed = dnp.field_sweep(spectra, peak=0.0, region=(-3000, 3000), shift_ppm=DELTA, track=False)
    assert fixed["rows"][3]["area"] < 0.2 * tracked["rows"][3]["area"]      # 12 kHz off
    assert [r["field_source"] for r in fixed["rows"]] == ["SFO1 (tracking off)"] * len(FIELDS)


def test_a_vanished_line_is_not_replaced_by_a_noise_maximum():
    # At a zero-crossing of the profile there is no line: the maximum
    # within the window is noise. It must be flagged, read at the pick,
    # and its field taken from SFO1 -- not reported as a peak.
    heights = [1000.0, 0.0, 1000.0]
    spectra = _sweep(FIELDS[:3], ERRORS[:3], heights=heights)
    r = dnp.field_sweep(spectra, peak=0.0, shift_ppm=DELTA)
    lost = r["rows"][1]
    assert not lost["found"]
    assert lost["field_source"] == "SFO1 (line not found)"
    assert math.isnan(lost["field_peak_T"])
    assert abs(lost["intensity"]) < 6 * lost["intensity_error"]
    assert any("No line found" in n for n in r["notes"])


def test_a_window_too_small_for_the_carrier_error_is_flagged_not_trusted():
    spectra = _sweep(FIELDS, ERRORS)
    r = dnp.field_sweep(spectra, peak=0.0, window=5000.0, shift_ppm=DELTA)
    assert not r["rows"][3]["found"]          # 12 kHz off: outside +-5 kHz
    assert r["rows"][0]["found"]


def test_sweep_scales_receiver_gain_to_the_reference():
    spectra = _sweep(FIELDS[:2], ERRORS[:2], heights=[1000.0, 2000.0])
    spectra[1].metadata.raw["acqus"]["RG"] = 200.0            # recorded at twice the gain
    r = dnp.field_sweep(spectra, peak=0.0, shift_ppm=DELTA)
    assert r["rows"][1]["intensity"] == pytest.approx(1000.0, rel=0.015)
    assert any("Receiver gains differ" in n for n in r["notes"])


def test_sweep_warns_about_mixed_pulse_programmes():
    spectra = _sweep(FIELDS[:2], ERRORS[:2])
    spectra[1].metadata.pulse_program = "zg"
    r = dnp.field_sweep(spectra, peak=0.0)
    assert any("mixes pulse programmes" in n for n in r["notes"])


def test_sweep_refuses_mixed_nuclei_and_needs_a_pick():
    spectra = _sweep(FIELDS[:2], ERRORS[:2])
    with pytest.raises(ValueError):
        dnp.field_sweep(spectra)
    spectra[1].metadata.nucleus = "1H"
    with pytest.raises(ValueError, match="one nucleus"):
        dnp.field_sweep(spectra, peak=0.0)


def test_profile_values_sort_by_field_and_normalise():
    heights = [1000.0, 3000.0, -2000.0, 500.0, 1500.0]
    order = [3, 0, 4, 1, 2]
    spectra = _sweep([FIELDS[i] for i in order], [ERRORS[i] for i in order],
                     heights=[heights[i] for i in order])
    r = dnp.field_sweep(spectra, peak=0.0, shift_ppm=DELTA)
    field, value, error, labels, found = dnp.profile_values(r)
    assert np.all(np.diff(field) > 0)
    assert value == pytest.approx(heights, rel=0.015)
    _f, norm, _e, _l, _fd = dnp.profile_values(r, normalise="max")
    assert np.max(np.abs(norm)) == pytest.approx(1.0)
    # Divided by one spectrum (e.g. microwaves off): that one becomes 1.
    _f, ratio, _e, lab, _fd = dnp.profile_values(r, normalise=1)
    assert ratio[lab.index(r["rows"][1]["label"])] == pytest.approx(1.0)


def test_csv_and_report(tmp_path):
    spectra = _sweep(FIELDS, ERRORS)
    r = dnp.field_sweep(spectra, peak=0.0, region=(-3000, 3000), labels=list("abcde"))
    path = tmp_path / "sweep.csv"
    dnp.field_sweep_to_csv(r, path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# DNP field sweep")
    import pandas as pd
    table = pd.read_csv(path, comment="#")
    assert list(table["Spectrum"]) == list("abcde")
    assert table["B0 (T)"].to_numpy() == pytest.approx([row["field_T"] for row in r["rows"]])
    assert "γ/2π" in dnp.field_sweep_report(r)


def test_plots_draw_with_labelled_axes():
    import matplotlib
    matplotlib.use("Agg")
    spectra = _sweep(FIELDS, ERRORS)
    r = dnp.field_sweep(spectra, peak=0.0, region=(-3000, 3000))
    fig = dnp.plot_field_profile(r, quantity="both")
    assert len(fig.axes) == 2
    assert "B$_0$ (mT)" in fig.axes[-1].get_xlabel()
    assert "$^{23}$Na" in fig.axes[0].get_ylabel()
    fig = dnp.plot_sweep_spectra(spectra, r)
    assert "carrier" in fig.axes[0].get_xlabel()


# -- real data ------------------------------------------------------------------

REAL_SWEEP = os.path.join("test_data", "DNP", "20260511_Na2CdMnFeCN6_171025A_DR_3p2mm")


@pytest.mark.skipif(not os.path.isdir(REAL_SWEEP), reason=f"needs {REAL_SWEEP}")
def test_real_sweep_repeats_agree_and_ordering_follows_the_current():
    from opennmr import io
    expnos = [100286, 100287, 100288, 100289, 100300, 100304, 100305]
    spectra = [io.load(os.path.join(REAL_SWEEP, str(e)), vendor="bruker") for e in expnos]
    ref = spectra[0]
    off = dnp.carrier_offset_hz(ref)
    r = dnp.field_sweep(spectra, peak=float(off[np.argmax(ref.data)]), labels=[str(e) for e in expnos])
    f = {row["label"]: row["field_T"] for row in r["rows"]}
    # -2 A / -8000, -1.5 A / 0, -1.5 A / -4000: increasing field. SFO1
    # put 100288 below 100287; the line does not.
    assert f["100287"] < f["100288"] < f["100289"]
    # Three acquisitions at identical settings.
    assert max(f["100300"], f["100304"], f["100305"]) - min(
        f["100300"], f["100304"], f["100305"]) < 50e-6


# -- baseline and noise (added after the first real data) --------------------

def _rolled(spectrum, amplitude):
    """Add a smooth rolling baseline, as a badly phased / distorted
    spectrum has (the first real microwave-on/off pair did)."""
    x = spectrum.axes[0].hz
    t = (x - x.min()) / np.ptp(x)
    spectrum.data = spectrum.data + amplitude * np.sin(2 * np.pi * 1.3 * t + 0.4)
    return spectrum


def test_local_baseline_follows_a_rolling_baseline_the_global_one_cannot():
    on, off, pk = _pair(h_on=3000.0, h_off=1000.0)
    _rolled(on, 4000.0)                      # larger than the line itself
    local = dnp.enhancement(on, off, peak=pk, region=(pk - 60, pk + 60))
    assert local["intensity"]["ratio"] == pytest.approx(3.0, rel=0.03)
    assert local["area"]["ratio"] == pytest.approx(3.0, rel=0.03)
    glob = dnp.enhancement(on, off, peak=pk, region=(pk - 60, pk + 60), baseline="global")
    assert abs(glob["intensity"]["ratio"] - 3.0) > 0.3
    with pytest.raises(ValueError):
        dnp.enhancement(on, off, peak=pk, baseline="quadratic")


def test_noise_estimate_ignores_a_rolling_baseline():
    s = _spectrum(9.4, 105.9, height=0.0, noise=10.0)
    _rolled(s, 5000.0)
    assert dnp._noise_sigma(s.data) == pytest.approx(10.0, rel=0.1)


def test_noise_estimate_is_not_fooled_by_correlated_points():
    # Zero-filling / apodisation correlate neighbouring points; the
    # point-to-point difference estimate then reads low (it did, by 18%,
    # on the real data). This one must not.
    rng = np.random.default_rng(5)
    white = rng.standard_normal(20000)
    correlated = np.convolve(white, [0.5, 1.0, 0.5], mode="same")
    true_sigma = np.std(correlated)
    diff_estimate = 1.4826 * np.median(np.abs(np.diff(correlated))) / np.sqrt(2)
    assert diff_estimate < 0.85 * true_sigma            # the trap being avoided
    assert dnp._noise_sigma(correlated) == pytest.approx(true_sigma, rel=0.07)


def test_region_and_baseline_window_sensitivity_is_reported():
    on, off, pk = _pair()
    r = dnp.enhancement(on, off, peak=pk, region=(pk - 60, pk + 60))
    assert r["intensity"]["ratio_systematic"] >= 0.0
    assert r["area"]["ratio_systematic"] >= 0.0
    assert "sys" in dnp.enhancement_report(r)
