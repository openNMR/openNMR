"""2026-11-09 (explicit user request): Dynamics' curve-fit report laid out
like Fit's read-outs -- a title, the fitted rate as a large bold headline,
a parameter table with italic symbols, a quality row of muted labels over
bold values -- instead of the monospaced "====" text block. The plain
report (text(): logged, copied, exported, kept) is unchanged."""

import numpy as np
import pytest

from opennmr.gui import report_format as rf
from opennmr.processing.relaxation import (
    fit_relaxation, fit_relaxation_global, format_relaxation_report,
)

from test_gui import qapp  # noqa: F401 -- fixture
from test_gui import _fit_spectrum_then_curve, _pfg_series_window


def _t1_fit():
    t = np.linspace(0.0, 0.5, 16)
    y = 3.0 * (1.0 - np.exp(-t / 0.07)) + np.random.default_rng(3).normal(0, 0.01, t.size)
    return fit_relaxation(t, y, model="saturation_recovery")


def _biexp_fit():
    t = np.linspace(0.0, 0.5, 24)
    y = 2.0 * np.exp(-t / 0.01) + 1.0 * np.exp(-t / 0.2)
    return fit_relaxation(t, y, model="biexponential_decay")


@pytest.mark.parametrize("name, expected", [
    ("T1", "<i>T</i><sub>1</sub>"), ("T2a", "<i>T</i><sub>2a</sub>"),
    ("M0", "<i>M</i><sub>0</sub>"), ("I0b", "<i>I</i><sub>0b</sub>"),
    ("D", "<i>D</i>"), ("Da", "<i>D</i><sub>a</sub>"), ("C", "<i>C</i>"),
    ("beta", "<i>β</i>"), ("tau_c", "<i>τ</i><sub>c</sub>"),
    ("T1rho", "<i>T</i><sub>1ρ</sub>"),
    ("amp", "amp"), ("k<1", "k&lt;1"),   # not a symbol: as written, escaped
])
def test_parameter_names_become_italic_symbols(name, expected):
    assert rf.param_html(name) == expected


def test_model_names_read_as_words():
    assert rf.model_name("saturation_recovery") == "Saturation recovery"
    assert rf.model_name("stejskal_tanner_biexponential") == "Stejskal–Tanner, biexponential"


def test_single_fit_has_a_headline_table_and_quality_row():
    fit = _t1_fit()
    out = rf.relaxation_report_html(fit, title="Region 1 (0.46-7.75 ppm)",
                                    muted="#888888", warn="#aa6600")
    assert "font-size:large" in out and "Region 1 (0.46-7.75 ppm)" in out
    assert "Saturation&nbsp;recovery" in out          # kept whole on its line
    # The rate is the headline, in the same words as the plain report.
    plain = format_relaxation_report(fit)
    rate_line = next(line for line in plain.splitlines() if line.strip().startswith("T1 ="))
    value = rate_line.split("=", 1)[1].strip()
    assert "font-size:x-large" in out and f"<i>T</i><sub>1</sub> = {value}" in out
    # Every parameter is in the table; quality labels are muted over bold values.
    assert "<i>M</i><sub>0</sub>" in out
    assert '<span style="color:#888888;"><i>R</i>²</span><br><span style="font-weight:600;">' in out
    assert "reduced χ²" in out and "points / d.o.f." in out and "final <i>I</i>/<i>I</i>₀" in out
    assert "± is 1σ (from residual scatter)." in out
    assert "====" not in out


def test_multi_component_fit_shows_shares_and_their_rates():
    fit = _biexp_fit()
    out = rf.relaxation_report_html(fit)
    assert "Relative amplitudes" in out
    assert "% of total" in out
    assert "<i>A</i><sub>1</sub>" in out and "<i>T</i><sub>2a</sub> =" in out


def test_fixed_parameter_is_marked_fixed():
    fit = _t1_fit()
    fit = {**fit, "errors": {**fit["errors"], "M0": None}}
    assert "fixed" in rf.relaxation_report_html(fit)


def test_global_fit_headlines_the_shared_parameter():
    t = np.linspace(0.0, 0.3, 16)
    data = [(t, a * np.exp(-t / 0.05)) for a in (10.0, 4.0)]
    result = fit_relaxation_global(data, model="exponential_decay", shared=("T2",),
                                   labels=["A", "B"])
    out = rf.global_report_html(result)
    assert "Global curve fit" in out and "2&nbsp;curves" in out
    assert "<i>T</i><sub>2</sub> =" in out and "shared across every curve" in out
    assert "<b>A</b>" in out and "<b>B</b>" in out          # a row per curve
    assert "Fit quality (joint, over every curve)" in out


def test_several_peaks_are_one_table():
    a, b = _t1_fit(), _t1_fit()
    a["peak_label"], b["peak_label"] = "Peak 1", "Peak 2"
    out = rf.peaks_summary_html([a, b])
    assert "2 peaks" in out and "<b>Peak 1</b>" in out and "<b>Peak 2</b>" in out
    assert out.count("<i>T</i><sub>1</sub> =") == 2


def test_summary_puts_the_values_warning_first_and_the_provenance_last():
    out = rf.curve_fit_summary_html([_t1_fit()], note="Delays were invented",
                                    provenance="Delays from: vdlist", warn="#aa6600")
    assert out.index("Delays were invented") < out.index("Parameters") < out.index("vdlist")
    assert "color:#aa6600" in out


def test_values_are_escaped():
    fit = _t1_fit()
    out = rf.relaxation_report_html(fit, title="<b>x</b> & y")
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in out


# ---- in the window ------------------------------------------------------------------------

def test_dynamics_report_is_laid_out_and_text_stays_plain(qapp):
    window, b_values, _spectrum = _pfg_series_window(qapp)
    window.delay_source_combo.setCurrentIndex(1)
    window.manual_delays_edit.setText(", ".join(f"{b:.8g}" for b in b_values))
    window.relaxation_model_combo.setCurrentIndex(0)
    _fit_spectrum_then_curve(window)

    for label in (window.curve_fit_summary_label, window.relaxation_fit_params_label):
        assert "====" in label.text() and "Uncertainty" in label.text()   # plain, unchanged
        shown = label.html()
        assert "====" not in shown and "font-size:x-large" in shown
        assert "<i>D</i> =" in shown

    # The per-spectrum state keeps the laid-out summary, so reselecting the
    # spectrum brings it back rather than the plain text.
    state = window._capture_analysis_state()
    assert state["curve_fit_summary_html"] == window.curve_fit_summary_label.html()
