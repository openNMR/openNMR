import os

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.processing import vocs

_VOCS_DATA = os.path.join(
    "test_data", "VOCS_example_Data", "15_02_26_Li2MnO3_O17_1_3mm",
)


def test_pulse_to_w1_matches_docstring_example():
    assert vocs.pulse_to_w1(5) == 50000.0


def test_vocs_power_windowed_zero_outside_bandwidth():
    freq_hz = np.linspace(-100000, 100000, 2001)
    power = vocs.vocs_power(freq_hz, w1=1000.0, window_filter=True)
    x = (freq_hz - freq_hz[len(freq_hz) // 2]) / 1000.0
    assert np.all(power[np.abs(x) >= 1.73] == 0)


def test_vocs_power_peaks_near_centre():
    freq_hz = np.linspace(-5000, 5000, 501)
    power = vocs.vocs_power(freq_hz, w1=1000.0, window_filter=False)
    assert np.argmax(power) == np.argmin(np.abs(freq_hz))


def _make_1d(n=64, center_ppm=5.0, sfo1=400.0, p1=5.0):
    ppm = np.linspace(center_ppm - 2, center_ppm + 2, n)
    hz = ppm * sfo1
    data = np.exp(-((ppm - center_ppm) ** 2) / 0.05)
    # `P` as a real indexed ARRAY, the way every Bruker acqus actually
    # records it -- there is no "P1" key in a real file. This fixture
    # used to write {"P1": p1}, which is why the whole VOCS path could
    # be green in tests while failing on every real dataset with "P1
    # not found in this spectrum's acquisition parameters" (real user
    # report). A fixture that doesn't match the file format tests
    # nothing but itself.
    acqus = {"P": [0.0, p1, 2 * p1] + [0.0] * 61}
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=sfo1, raw={"acqus": acqus})
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=hz)
    return NMRSpectrum(data=data, axes=[axis], metadata=metadata)


def test_apply_vocs_weights_and_never_amplifies():
    # The original methodology: multiply by the excitation profile, so
    # each offset counts where its own pulse actually excited. The
    # weight is <= 1 everywhere, so this can only ever attenuate --
    # which is what makes the scale safe with no threshold anywhere.
    #
    # A divide-by-the-profile deconvolution was tried and reverted: it
    # amplified the sweep edges until they dominated the real peak
    # ("signal is large at the edges and suddenly drops to zero and
    # has low intensity at the centres").
    spec = _make_1d_wide()
    weighted = vocs.apply_vocs(spec, pulse="P1")

    assert "vocs_profile" in weighted.metadata.raw
    assert len(weighted.data) == len(spec.data)
    nonzero = spec.data != 0
    gain = weighted.data[nonzero] / spec.data[nonzero]
    assert gain.max() <= 1.0 + 1e-12      # never amplifies, anywhere
    assert gain.min() >= 0.0
    assert not np.allclose(weighted.data, spec.data)   # did do something


def test_apply_vocs_leaves_the_on_resonance_intensity_alone():
    # The profile is 1.0 on resonance, so the peak a VOCS experiment
    # is centred on comes back unchanged -- a correction must not
    # rescale the data it corrects. Before the fix this point was
    # multiplied by 2.0.
    spec = _make_1d_wide()
    corrected = vocs.apply_vocs(spec, pulse="P1")

    centre = len(spec.data) // 2
    assert corrected.data[centre] == pytest.approx(spec.data[centre], rel=1e-6)


def test_excitation_profile_is_one_on_resonance():
    # vocs_power itself peaks at 2.0 (faithful to nmrtools2 and left
    # that way); excitation_profile is the normalised, physical one.
    hz = np.linspace(-50000, 50000, 1001)
    w1 = 50000.0
    assert vocs.vocs_power(hz, w1)[500] == pytest.approx(2.0)
    assert vocs.excitation_profile(hz, w1)[500] == pytest.approx(1.0)
    assert vocs.excitation_profile(hz, w1).max() == pytest.approx(1.0)


def _make_1d_wide(n=512, center_hz=0.0, sfo1=67.8, p1=1.0, amplitude=1000.0):
    """A spectrum genuinely wide relative to its own excitation
    bandwidth -- the regime VOCS exists for. Over a narrow sweep the
    profile is 1.0 to several decimals and every correction looks like
    a no-op, which tests nothing."""
    hz = np.linspace(center_hz - 300000, center_hz + 300000, n)
    data = np.full(n, amplitude) * vocs.excitation_profile(hz, vocs.pulse_to_w1(p1))
    metadata = NMRMetadata(vendor="bruker", nucleus="17O", sfo1=sfo1,
                            raw={"acqus": {"P": [0.0, p1, 2 * p1] + [0.0] * 61}})
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=hz / sfo1, hz=hz)],
                        metadata=metadata)


def test_vocs_sum_reads_true_amplitude_where_coverage_is_best():
    # The scalar plateau divide is what puts the bare sum back on the
    # spectra's own scale: sum(S*w) responds to a flat true spectrum as
    # sum(w**2), so dividing by that quantity's own maximum makes the
    # best-covered shift read the true amplitude -- without touching
    # the lineshape, which a per-point denominator would.
    amplitude = 1000.0
    specs = [_make_1d_wide(center_hz=c, amplitude=amplitude)
             for c in (-150000.0, 0.0, 150000.0)]

    result = vocs.vocs_sum(specs, pulse="P1")

    assert result["plateau"] > 0
    assert np.max(result["vocs_sum"]) == pytest.approx(amplitude, rel=1e-6)
    # Never scaled *up* past the truth -- coverage only ever falls off
    # from the plateau.
    assert np.max(result["vocs_sum"]) <= amplitude + 1e-6


def test_vocs_sum_scalar_divide_leaves_the_lineshape_alone():
    # The reason a scalar beats a per-point denominator: dividing by
    # sum(w) point-by-point imposes its own structure on a response
    # that the offsets had already made flat. Measured on this
    # project's own 17O geometry with the coverage gap filled, the
    # weighted mean rippled 1.17x (3.5% rms) where the bare sum
    # rippled 1.06x (1.3% rms) -- i.e. it made a flat response worse.
    # Here: the composite must stay exactly proportional to the bare
    # sum, differing only by one constant.
    specs = [_make_1d_wide(center_hz=c) for c in (-150000.0, 0.0, 150000.0)]

    result = vocs.vocs_sum(specs, pulse="P1")

    bare = result["vocs_sum"] * result["plateau"]
    nonzero = np.abs(bare) > 1e-9
    ratio = result["vocs_sum"][nonzero] / bare[nonzero]
    assert np.allclose(ratio, ratio[0])   # one constant, not a per-point curve


def test_vocs_sum_never_exceeds_the_spectra_that_went_into_it():
    # A weighted mean is bounded by its inputs, which is what "the
    # scale of each individual spectrum is preserved across the
    # addition step" means in practice -- overlapping offsets improve
    # the average rather than stacking up amplitude.
    specs = [_make_1d_wide(center_hz=c, amplitude=1000.0)
             for c in (-200000.0, 0.0, 200000.0)]

    result = vocs.vocs_sum(specs, pulse="P1")

    biggest_input = max(np.abs(s.data).max() for s in specs)
    assert np.abs(result["vocs_sum"]).max() <= biggest_input + 1e-9
    assert np.isfinite(result["vocs_sum"]).all()


def test_vocs_sum_peaks_where_coverage_is_good_not_at_the_edges():
    # The failure this guards against: "signal is large at the edges
    # and suddenly drops to zero and has low intensity at the centres".
    # A deconvolution put the composite's own maximum out where almost
    # nothing was excited (measured on real data: max at a coverage
    # weight of 0.0028, four times higher than the well-covered
    # regions). A weighted mean cannot do that -- it is bounded by its
    # inputs everywhere.
    specs = [_make_1d_wide(center_hz=c) for c in (-200000.0, 200000.0)]

    result = vocs.vocs_sum(specs, pulse="P1")
    y, weight = result["vocs_sum"], result["weight"]

    assert np.isfinite(y).all()
    # Nothing is reported where nothing was excited...
    assert np.all(y[weight == 0.0] == 0.0)
    # ...and the composite's own maximum sits in a well-covered region,
    # not out at a poorly-covered edge.
    assert weight[np.argmax(np.abs(y))] > 0.5


def test_vocs_weighting_rolls_off_smoothly_from_resonance():
    # The weight is the excitation profile itself: 1.0 on resonance,
    # falling away monotonically with offset, no step anywhere. The
    # hard-floored deconvolution this replaced stepped straight to zero
    # at its threshold, which is the "suddenly drops to zero" half of
    # the reported bug.
    spec = _make_1d_wide()
    weighted = vocs.apply_vocs(spec, pulse="P1")

    nonzero = spec.data != 0
    gain = weighted.data[nonzero] / spec.data[nonzero]
    centre = len(gain) // 2
    assert gain[centre] == pytest.approx(1.0, rel=1e-6)
    # Falls away from resonance, in small steps rather than a cliff.
    assert np.all(np.diff(gain[centre:]) <= 1e-9)
    assert np.abs(np.diff(gain)).max() < 0.05


def test_apply_vocs_missing_pulse_raises():
    # P9 exists in a real acqus but is 0 (unused) -- it must raise a
    # ValueError naming the problem, not reach pulse_to_w1 and come
    # back as a bare ZeroDivisionError.
    spec = _make_1d()
    try:
        vocs.apply_vocs(spec, pulse="P9")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_pulse_length_resolves_an_indexed_p_array():
    # The actual bug: a real acqus records P as an array, so a direct
    # `"P1" in acqus` lookup is always False and VOCS raised on every
    # real dataset. Resolution goes through acquisition_constants,
    # which expands P/D/CNST/L/GPZ into p1/d1/cnst2/l4/gpz1 names.
    spec = _make_1d(p1=4.25)
    assert "P1" not in spec.metadata.raw["acqus"]   # the real file's shape
    assert vocs._pulse_length_us(spec, "P1") == 4.25
    assert vocs._pulse_length_us(spec, "p1") == 4.25  # case-insensitive
    assert vocs._pulse_length_us(spec, "P2") == 8.50


def test_pulse_length_error_names_the_pulses_that_are_set():
    # A Bruker acqus always carries all 64 P slots and most are 0, so
    # the message must name the two or three actually in use rather
    # than listing every key.
    spec = _make_1d(p1=4.25)
    try:
        vocs._pulse_length_us(spec, "P9")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert "P1 (4.25 us)" in message
    assert "P2 (8.5 us)" in message
    assert "P3" not in message  # zero/unset, so not offered
    assert "recorded as 0" in message  # says *why* P9 is unusable


def test_pulse_length_on_a_spectrum_with_no_acqus_says_so():
    spec = _make_1d()
    spec.metadata.raw.pop("acqus")
    try:
        vocs._pulse_length_us(spec, "P1")
    except ValueError as exc:
        assert "no recorded acquisition parameters" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_vocs_sum_combines_multiple_spectra():
    specs = [_make_1d(center_ppm=c) for c in (4.0, 5.0, 6.0)]
    result = vocs.vocs_sum(specs, npoints=512)
    assert len(result["ppm"]) == 512
    assert result["interpolated_slices"].shape == (3, 512)
    assert len(result["vocs_sum"]) == 512


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_VOCS_DATA, "12")),
    reason="needs the real VOCS_example_Data dataset",
)
def test_vocs_on_the_real_offset_series():
    # The dataset the bug was reported on: 17O on Li2MnO3, experiments
    # 12-15, four offset acquisitions covering overlapping excitation
    # windows (O1 = +671, +400, -136, -407 kHz) -- exactly what VOCS
    # exists to recombine. Every one of these raised "P1 not found in
    # this spectrum's acquisition parameters" before the fix.
    from opennmr.io import bruker

    specs = [bruker.read_1d(os.path.join(_VOCS_DATA, e)) for e in ("12", "13", "14", "15")]
    assert {s.metadata.nucleus for s in specs} == {"17O"}
    # Genuinely different offsets -- otherwise this is not a VOCS set.
    assert len({round(s.metadata.o1) for s in specs}) == 4
    # The real file's shape, which the old direct lookup could not read.
    assert "P1" not in specs[0].metadata.raw["acqus"]

    for spec in specs:
        corrected = vocs.apply_vocs(spec, pulse="P1")
        assert corrected.data.shape == spec.data.shape
        assert np.isfinite(corrected.data).all()

    result = vocs.vocs_sum(specs, pulse="P1")
    assert np.isfinite(result["vocs_sum"]).all()
    assert result["interpolated_slices"].shape[0] == 4
    # The combined axis has to span every contributing offset, not just
    # one window -- that recombination is the whole point of VOCS.
    for spec in specs:
        assert result["ppm"].min() <= spec.axes[0].ppm.max()
        assert result["ppm"].max() >= spec.axes[0].ppm.min()
