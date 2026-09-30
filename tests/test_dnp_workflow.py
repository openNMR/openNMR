"""The step-by-step DNP workflow (opennmr.notebook.dnp_workflow): fields,
alignment on the line's shape, hand corrections, aligned ppm axes with a
reference frequency each, height/integral measurement, field profiles and
enhancements (explicit user request, 2026-09-30).

Spectra are tests/test_dnp.py's synthetic 23Na ones: the line at
gamma*B0*(1+delta) for a *known* B0, the carrier set by hand to within
several kHz of it, and each spectrum's ppm axis referenced to one fixed SF
-- so across a sweep the line wanders by thousands of ppm, as it does in
real data.
"""
import numpy as np
import pytest

from opennmr.notebook import dnp_workflow as dw
from test_dnp import DELTA, ERRORS, FIELDS, FWHM, GAMMA_NA, _spectrum, _sweep


def _true_offset(b, err):
    """The line's offset from its carrier in _sweep: +err Hz."""
    return err


def _peak_ppm(spectrum):
    return float(spectrum.axes[0].ppm[np.argmax(np.abs(spectrum.data))])


# -- 1. fields -------------------------------------------------------------------

def test_fields_from_sfo1():
    spectra = _sweep(FIELDS, ERRORS)
    info = dw.spectrum_fields(spectra)
    assert info["nucleus"] == "23Na"
    for b, err, got, sfo1 in zip(FIELDS, ERRORS, info["field_T"], info["sfo1_MHz"]):
        line = GAMMA_NA * b * (1 + DELTA * 1e-6)
        assert sfo1 == pytest.approx((line - err) / 1e6, rel=1e-12)
        assert got == pytest.approx(sfo1 * 1e6 / GAMMA_NA, rel=1e-12)


def test_mixed_nuclei_are_refused():
    a = _sweep(FIELDS[:1], ERRORS[:1])[0]
    b = _spectrum(9.4, 105.9, nucleus="7Li")
    with pytest.raises(ValueError, match="different nuclei"):
        dw.spectrum_fields([a, b])


# -- 2. alignment ------------------------------------------------------------------

def test_every_line_is_found_to_a_fraction_of_a_point():
    spectra = _sweep(FIELDS, ERRORS)
    al = dw.align_spectra(spectra, reference=0, peak_ppm=_peak_ppm(spectra[0]))
    ref_offset = ERRORS[0]
    assert al["ref_offset_hz"] == pytest.approx(ref_offset, abs=15)
    assert al["ref_fwhm_hz"] == pytest.approx(FWHM, rel=0.1)
    for e, err in zip(al["entries"], ERRORS):
        assert e["found"], e
        # The offset between lines, from the carriers: exact to well
        # under one 12 Hz point.
        assert e["shift_hz"] == pytest.approx(err - ref_offset, abs=3.0)
        assert e["score"] > 0.99
        assert e["width_ratio"] == pytest.approx(1.0, abs=0.1)


def test_the_line_is_matched_by_shape_not_by_being_largest():
    # A taller, much broader hump 7 kHz from the line, inside the search
    # window: the largest signal there, but not the line's shape.
    spectra = _sweep(FIELDS[:3], ERRORS[:3])
    spectra[2] = _sweep([FIELDS[2]], [ERRORS[2]], extra_lines=())[0]
    b, err = FIELDS[2], ERRORS[2]
    line = GAMMA_NA * b * (1 + DELTA * 1e-6)
    x = np.asarray(spectra[2].axes[0].hz) + 105.892246e6
    hump = 3000.0 / (1 + ((x - (line + 7000.0)) / 4000.0) ** 2)
    spectra[2].data = spectra[2].data + hump
    al = dw.align_spectra(spectra, reference=0, peak_ppm=_peak_ppm(spectra[0]))
    # Found, to within about one 12 Hz point: the hump's curvature under
    # the line pulls the match a little.
    assert al["entries"][2]["shift_hz"] == pytest.approx(err - ERRORS[0], abs=20.0)
    assert al["entries"][2]["width_ratio"] == pytest.approx(1.0, abs=0.25)


def test_an_inverted_line_is_found_and_flagged():
    spectra = _sweep(FIELDS, ERRORS, heights=[1000, 800, -600, 400, 900])
    al = dw.align_spectra(spectra, reference=0, peak_ppm=_peak_ppm(spectra[0]))
    e = al["entries"][2]
    assert e["found"] and e["sign"] == -1 and "inverted" in e["note"]
    assert e["shift_hz"] == pytest.approx(ERRORS[2] - ERRORS[0], abs=3.0)


def test_no_line_at_a_zero_crossing_is_flagged_not_invented():
    spectra = _sweep(FIELDS, ERRORS, heights=[1000, 800, 1.0, 400, 900])
    al = dw.align_spectra(spectra, reference=0, peak_ppm=_peak_ppm(spectra[0]))
    e = al["entries"][2]
    assert not e["found"] and "no line matched" in e["note"]
    assert e["auto_shift_hz"] == 0.0
    assert any("No line matched" in n for n in al["notes"])
    assert "NOT FOUND" in dw.alignment_report(al)


# -- aligned axes, fields, manual correction -------------------------------------------

def test_aligned_axes_put_every_line_at_the_reference_shift():
    spectra = _sweep(FIELDS, ERRORS)
    al = dw.align_spectra(spectra, reference=0, peak_ppm=_peak_ppm(spectra[0]))
    aligned = dw.aligned_spectra(spectra, al)
    step = abs(float(np.median(np.diff(aligned[0].axes[0].ppm))))
    for k, s in enumerate(aligned):
        assert _peak_ppm(s) == pytest.approx(al["peak_ppm"], abs=1.5 * step)
        record = s.metadata.raw["dnp_alignment"]
        assert s.metadata.sfo1 == pytest.approx(record["own_reference_MHz"])
        # The Hz axis is kept: integrals are unchanged by the relabelling.
        np.testing.assert_array_equal(s.axes[0].hz, spectra[k].axes[0].hz)


def test_field_at_the_line_is_the_true_field():
    spectra = _sweep(FIELDS, ERRORS)
    al = dw.align_spectra(spectra, reference=0, peak_ppm=_peak_ppm(spectra[0]))
    # The reference's ppm axis puts its line at DELTA only if its SF is the
    # true zero; give the true shift for an absolute check.
    al["peak_ppm"] = DELTA
    fields = dw.alignment_fields(spectra, al, source="line")
    for got, b in zip(fields["field_T"], FIELDS):
        assert got == pytest.approx(b, abs=2e-7)
    sfo1 = dw.alignment_fields(spectra, al, source="sfo1")
    # SFO1 misses by the carrier error: 12 kHz is ~1 mT for 23Na.
    assert abs(sfo1["field_T"][3] - FIELDS[3]) > 5e-4
    assert fields["difference_mT"][3] == pytest.approx(
        (fields["field_line_T"][3] - fields["field_sfo1_T"][3]) * 1e3)
    with pytest.raises(ValueError, match="aligned first"):
        dw.alignment_fields(spectra, None, source="line")


def test_a_hand_shift_and_a_drag_on_the_ppm_axis():
    spectra = _sweep(FIELDS, ERRORS)
    al = dw.align_spectra(spectra, reference=0, peak_ppm=_peak_ppm(spectra[0]))
    al["confirmed"] = True
    before = dw.aligned_spectrum(spectra[3], al, 3)
    # Drag spectrum 3 by +5 ppm on the aligned axis.
    dw.set_manual_shift(al, 3, dw.shift_for_display_ppm(al, 3, 5.0))
    assert not al["confirmed"]
    after = dw.aligned_spectrum(spectra[3], al, 3)
    assert _peak_ppm(after) - _peak_ppm(before) == pytest.approx(5.0, abs=0.2)
    assert al["entries"][3]["shift_hz"] == pytest.approx(
        al["entries"][3]["auto_shift_hz"] + al["entries"][3]["manual_shift_hz"])
    assert "by hand" in dw.alignment_report(al)


# -- 4./5. measurement, profile, enhancement --------------------------------------------

def _measured(heights, rgs=None):
    spectra = _sweep(FIELDS, ERRORS, heights=heights)
    if rgs is not None:
        # A spectrum taken at twice the receiver gain reads twice as high.
        for s, rg in zip(spectra, rgs):
            s.metadata.raw["acqus"]["RG"] = rg
            s.data = s.data * rg / 100.0
    al = dw.align_spectra(spectra, reference=0, peak_ppm=_peak_ppm(spectra[0]))
    aligned = dw.aligned_spectra(spectra, al)
    pk = al["peak_ppm"]
    width = FWHM / (spectra[0].metadata.sfo1)          # 1 kHz in ppm
    m = dw.measure(aligned, peak_ppm=pk, region=(pk - 8 * width, pk + 8 * width))
    return spectra, al, aligned, m


def test_heights_and_integrals_follow_the_true_heights():
    heights = [1000.0, 2000.0, -500.0, 4000.0, 3000.0]
    _s, _al, _a, m = _measured(heights)
    h = m["height"]["value"]
    np.testing.assert_allclose(h / h[0], np.array(heights) / heights[0], rtol=0.01, atol=0.01)
    a = m["area"]["value"]
    np.testing.assert_allclose(a / a[0], np.array(heights) / heights[0], rtol=0.03, atol=0.02)
    assert np.all(m["height"]["error"] > 0) and np.all(m["area"]["systematic"] >= 0)
    assert "per scan" in dw.measurement_report(m)


def test_receiver_gain_is_scaled_to_the_reference():
    heights = [1000.0] * 5
    _s, _al, _a, m = _measured(heights, rgs=[100, 200, 100, 50, 100])
    # A spectrum at twice the gain reads twice as high: scaled back.
    h = m["height"]["value"]
    np.testing.assert_allclose(h / h[0], 1.0, rtol=0.01)
    assert any("Receiver gains differ" in n for n in m["notes"])


def test_profile_sorted_by_field_in_every_mode():
    heights = [1000.0, 2000.0, -500.0, 4000.0, 3000.0]
    spectra, al, _a, m = _measured(heights)
    fields = dw.alignment_fields(spectra, al)
    shuffled = {k: v for k, v in fields.items()}
    absolute = dw.field_profile(m, shuffled, mode="absolute")
    assert np.all(np.diff(absolute["field_T"]) > 0)
    ratio = dw.field_profile(m, fields, mode="ratio", off=0)
    np.testing.assert_allclose(ratio["height"]["value"], np.array(heights) / 1000.0,
                               rtol=0.01, atol=0.01)
    one_minus = dw.field_profile(m, fields, mode="one_minus", off=0)
    np.testing.assert_allclose(one_minus["area"]["value"], 1 - ratio["area"]["value"])
    without_off = dw.field_profile(m, fields, mode="ratio", off=0, exclude=[0])
    assert len(without_off["labels"]) == 4
    with pytest.raises(ValueError, match="OFF"):
        dw.field_profile(m, fields, mode="ratio")
    assert "ON / OFF" in dw.profile_report(ratio)


def test_enhancement_table_by_height_and_integral():
    heights = [1000.0, 12500.0, 1000.0, 1000.0, 1000.0]
    _s, _al, _a, m = _measured(heights)
    rows = dw.enhancement_table(m, off=0, on=[1])
    row = rows[0]
    assert row["height"]["ratio"] == pytest.approx(12.5, rel=0.02)
    assert row["area"]["ratio"] == pytest.approx(12.5, rel=0.03)
    assert row["height"]["one_minus"] == pytest.approx(1 - row["height"]["ratio"])
    assert row["agreement"]["consistent"]
    text = dw.enhancement_report(rows)
    assert "ε = ON/OFF" in text and "1 − ON/OFF" in text and "agree" in text


def test_plots_and_csv(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    spectra, al, aligned, m = _measured([1000.0, 2000.0, -500.0, 4000.0, 3000.0])
    fig, lines = dw.plot_aligned_spectra(aligned, mode="stack", peak_ppm=al["peak_ppm"],
                                         region=m["settings"]["region"])
    assert len(lines) == 5 and [line.dnp_index for line in lines] == list(range(5))
    fig, lines = dw.plot_aligned_spectra(aligned, mode="overlay", xlims=(200, 300))
    prof = dw.field_profile(m, dw.alignment_fields(spectra, al), mode="ratio", off=0)
    fig = dw.plot_field_profile(prof, quantity="both", nucleus="23Na")
    assert len(fig.axes) == 2 and "ON/OFF" in fig.axes[0].get_ylabel()
    fig = dw.plot_field_profile(prof, quantity="height", relative=True)
    assert "mT" in fig.axes[0].get_xlabel()
    path = tmp_path / "profile.csv"
    dw.profile_to_csv(prof, path)
    text = path.read_text()
    assert text.splitlines()[0].startswith("label,field_T,height,height_error")
    assert len(text.splitlines()) == 6


def test_spectra_with_different_widths_and_points_align():
    # Receiver settings changed between experiments: another SW, another
    # size, the carrier moved by hand -- nothing positional is shared.
    b = FIELDS
    spectra = []
    for k, (field, err) in enumerate(zip(b, ERRORS)):
        line = GAMMA_NA * field * (1 + DELTA * 1e-6)
        sw, n = (200e3, 16384) if k % 2 == 0 else (150e3, 8192)
        spectra.append(_spectrum(field, (line - err) / 1e6, sw=sw, n=n, seed=30 + k))
    al = dw.align_spectra(spectra, reference=2, peak_ppm=_peak_ppm(spectra[2]))
    for e, err in zip(al["entries"], ERRORS):
        assert e["found"]
        assert e["shift_hz"] == pytest.approx(err - ERRORS[2], abs=5.0)
