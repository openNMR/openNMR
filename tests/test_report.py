"""The processing report: what was actually done, written out."""

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.processing import report as processing_report


def _spectrum(**raw):
    ppm = np.linspace(0.0, 10.0, 64)
    return NMRSpectrum(
        data=np.zeros(64),
        axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.13, raw=dict(raw)),
    )


def test_prose_reports_the_window_and_its_line_broadening():
    # The user's own example: "spectra are reported using a windowed EM
    # function and 5 Hz of line broadening".
    spectrum = _spectrum(live_window="exponential", live_lb=5.0)
    prose = processing_report.processing_prose(spectrum)
    assert "exponential (EM)" in prose
    assert "5 Hz of line broadening" in prose


def test_prose_reports_backwards_linear_prediction():
    # The user's other example: "8 points of backwards linear
    # prediction are used".
    spectrum = _spectrum(live_linear_predict_points=8,
                          live_linear_predict_direction="backward",
                          live_linear_predict_order=16)
    prose = processing_report.processing_prose(spectrum)
    assert "8 points of backwards linear prediction" in prose
    assert "order 16" in prose


def test_prose_names_the_steps_in_the_order_they_are_applied():
    spectrum = _spectrum(live_fid_shift=4, live_linear_predict_points=8,
                          live_window="exponential", live_lb=2.0, live_zero_fill=32768)
    prose = processing_report.processing_prose(spectrum)
    order = [prose.index(bit) for bit in
             ("left-shifted", "linear prediction", "apodized", "zero-filled")]
    assert order == sorted(order)


def test_untouched_settings_are_not_mentioned():
    # A methods paragraph does not say "no linear prediction was used",
    # and a table of defaults is noise in something meant to be pasted.
    spectrum = _spectrum(live_window="exponential", live_lb=1.0)
    prose = processing_report.processing_prose(spectrum)
    assert "linear prediction" not in prose
    assert "left-shift" not in prose
    names = [name for name, _value in processing_report.processing_summary(spectrum)]
    assert "Linear prediction" not in names
    assert "Truncated to" not in names


def test_nothing_applied_gives_no_sentence_at_all():
    # An empty paragraph is easier to notice and fix than a
    # confidently wrong one.
    assert processing_report.processing_prose(_spectrum()) == ""
    assert "No processing has been applied" in processing_report.format_processing_report(_spectrum())


def test_phase_and_baseline_are_reported_after_the_transform():
    spectrum = _spectrum(live_window="none", phase0=-37.5, phase1=12.0,
                          baseline_correct=True, baseline_method="whittaker")
    prose = processing_report.processing_prose(spectrum)
    assert "-37.5" in prose and "12" in prose
    assert "Whittaker" in prose


def test_per_slice_phase_is_reported_as_such():
    # One shared phase and a per-slice phase are different claims about
    # the data, and a methods section that says the wrong one is wrong.
    spectrum = _spectrum(slice_phase0=[10.0, 20.0, 30.0], phase0=0.0)
    prose = processing_report.processing_prose(spectrum)
    # Sentence-cased, since it opens its own sentence.
    assert "each slice was phased independently" in prose.lower()


def test_sine_bell_shift_is_spelled_out_not_left_as_ssb():
    # "SSB = 2" means nothing to a reader who does not use TopSpin.
    spectrum = _spectrum(live_window="qsine", live_ssb=2.0)
    prose = processing_report.processing_prose(spectrum)
    assert "pi/2" in prose
    assert "squared sine-bell" in prose


def test_summary_is_label_value_pairs_in_processing_order():
    spectrum = _spectrum(live_fid_shift=2, live_window="gaussian",
                          live_lb=1.0, live_gb=0.3, live_zero_fill=16384)
    rows = processing_report.processing_summary(spectrum)
    assert all(isinstance(row, tuple) and len(row) == 2 for row in rows)
    names = [name for name, _value in rows]
    assert names.index("FID left-shift") < names.index("Apodization") < names.index("Zero-filled to")
    values = dict(rows)
    assert "Gaussian" in values["Apodization"]
    assert values["Zero-filled to"] == "16384 points"


def test_full_report_carries_both_the_prose_and_the_table():
    spectrum = _spectrum(live_window="exponential", live_lb=5.0, live_zero_fill=65536)
    text = processing_report.format_processing_report(spectrum, label="expt 11")
    assert "expt 11" in text
    assert "5 Hz of line broadening" in text      # the paragraph
    assert "Zero-filled to" in text               # the table it was written from


def test_compare_marks_only_the_settings_that_differ():
    # The "for consistency" half: an inconsistency between datasets
    # about to be compared is worth seeing before the comparison.
    a = _spectrum(live_window="exponential", live_lb=5.0, live_zero_fill=65536)
    b = _spectrum(live_window="exponential", live_lb=1.0, live_zero_fill=65536)
    text = processing_report.compare_processing([a, b], labels=["A", "B"])

    lines = {line.split("  ")[0].strip(): line for line in text.splitlines() if line.strip()}
    assert "differs" in lines["Apodization"]       # 5 Hz vs 1 Hz
    assert "differs" not in lines["Zero-filled to"]  # both 65536


def test_compare_shows_a_dash_where_a_spectrum_did_not_use_a_setting():
    a = _spectrum(live_window="exponential", live_lb=5.0, live_linear_predict_points=8)
    b = _spectrum(live_window="exponential", live_lb=5.0)
    text = processing_report.compare_processing([a, b], labels=["A", "B"])
    row = next(line for line in text.splitlines() if line.startswith("Linear prediction"))
    assert "—" in row
    assert "differs" in row
