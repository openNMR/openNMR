"""Spinsolve PGSTE reader, pinned against the instrument's own output.

A format reader that guesses wrong about byte order, interleave or
frequency direction still returns a plausible-looking spectrum and no
error. Spinsolve writes its own answer next to the data -- the fitted D
in ``fit.csv``, the gradient array in ``data.2d.txt``, the integral
region in ``display.par`` -- so every one of those becomes a check that
the whole chain (binary -> FT -> ppm axis -> integral -> fit) is right,
not just self-consistent.
"""
import os

import numpy as np
import pytest

from opennmr.io import spinsolve

_PATH = os.path.join(
    "test_data", "Spinsolve_PFG", "2026", "08", "AL-70A-3", "PGSTE CDEC_1"
)

needs_data = pytest.mark.skipif(
    not os.path.isdir(_PATH), reason=f"needs the real dataset at {_PATH}"
)


@pytest.fixture(scope="module")
def dataset():
    return spinsolve.read_spinsolve_pfg(_PATH)


@needs_data
def test_detects_the_folder():
    assert spinsolve.is_spinsolve_pfg(_PATH)
    assert not spinsolve.is_spinsolve_pfg(os.path.dirname(_PATH))
    assert spinsolve.find_spinsolve_experiments(
        os.path.join("test_data", "Spinsolve_PFG")
    ) == [_PATH]


@needs_data
def test_par_files_are_typed_not_left_as_strings(dataset):
    assert dataset.acqu["Sample"] == "AL-70A-3"
    assert dataset.acqu["nrSteps"] == 64
    assert dataset.acqu["b1Freq"] == pytest.approx(61.4457393530756)
    # Only the outer quotes come off: graphTitle has unescaped inner ones.
    assert dataset.acqu["graphTitle"].startswith("1D-1H-16")
    assert dataset.acqu["graphTitle"].endswith("-1")
    assert dataset.acqu["90PulseLength"] == pytest.approx(9.3)
    # display.par carries a stray key-less "= 0" line; parsing survives it.
    assert dataset.display["PGSTEFit"] == "SinglePGSTE"


@needs_data
def test_matrix_shape_and_axes(dataset):
    assert dataset.fid.shape == (64, 8192)
    assert dataset.fid.dtype == complex
    assert dataset.dwell == pytest.approx(200e-6)
    assert dataset.sweep_width == pytest.approx(5000.0)
    # data.2d.txt states the same axis in ppm; the reader derives it in
    # Hz from acqu.par. The DOSY export rounds its copy to 7 figures and
    # is a further ~0.1 Hz off, so this only confirms they are the same
    # edge of the same window -- acqu.par is the value to trust.
    assert dataset.dosy.params["Lowest Frequency"] == pytest.approx(
        dataset.lowest_frequency / dataset.b1_freq, rel=1e-4
    )
    assert dataset.dosy.params["Spectral Width"] == pytest.approx(
        dataset.sweep_width / dataset.b1_freq, rel=1e-6
    )


@needs_data
def test_the_three_copies_of_the_matrix_agree(dataset):
    # data.2d.bin is the same numbers as big-endian int64 x 1e4, so it
    # agrees to one quantum -- a wrong endianness or block order would
    # miss by orders of magnitude, not 1e-4.
    from_bin = spinsolve.read_spinsolve_pfg(_PATH, source="data.2d.bin").fid
    assert np.abs(from_bin - dataset.fid).max() < 1e-4

    # data.1d is one row of the 2D (whichever step the UI had selected).
    one_d = spinsolve.read_prospa(os.path.join(_PATH, "data.1d"))
    assert one_d.data_type == 504
    assert one_d.x[1] == pytest.approx(dataset.dwell)
    step = int(dataset.display["StackIndex"])
    assert np.allclose(one_d.data.real, dataset.fid[step].real, rtol=1e-5)

    # data_enhanced.2d is Spinsolve's own apodized copy: same shape,
    # deliberately different values, so it must not be mistaken for raw.
    enhanced = spinsolve.read_spinsolve_pfg(_PATH, source="data_enhanced.2d").fid
    assert enhanced.shape == dataset.fid.shape
    assert not np.allclose(enhanced, dataset.fid)


@needs_data
def test_delay_times_are_b_values_matching_the_gradient_array(dataset):
    # delayTimes.txt is a misnomer for PGSTE and is written with a comma
    # decimal separator; recomputing b from data.2d.txt's gradients
    # confirms both the locale parsing and delta/DELTA.
    assert dataset.b.shape == (64,)
    assert dataset.delta == pytest.approx(0.0031)    # 3 ms pulse + 0.1 ms ramp
    assert dataset.big_delta == pytest.approx(0.05)
    assert np.allclose(dataset.b_from_gradients(), dataset.b, rtol=1e-4)
    assert dataset.gradients.max() == pytest.approx(
        0.01 * dataset.acqu["gradientPercentage"] * dataset.acqu["maxGradient"] * 1e-3,
        rel=1e-3,
    )


@needs_data
def test_processing_defaults_come_from_the_instrument(dataset):
    assert dataset.display_line_broadening() == pytest.approx(1.0)
    assert spinsolve.read_integral_regions(dataset.display) == [
        pytest.approx((-0.36134783804285, 0.837444649062855))
    ]


@needs_data
def test_spectrum_lands_on_the_right_side_of_the_ppm_axis(dataset):
    # The one real check on the FT direction: the single peak must fall
    # inside the region Spinsolve integrated. Transforming without the
    # conjugate mirrors it to ~9.3 ppm, which still looks like a spectrum.
    ppm = dataset.ppm_axis()
    peak = ppm[np.abs(dataset.spectra()[0]).argmax()]
    low, high = spinsolve.read_integral_regions(dataset.display)[0]
    assert low < peak < high
    assert peak == pytest.approx(0.22, abs=0.02)


@needs_data
def test_fit_reproduces_the_instruments_own_diffusion_coefficient(dataset):
    reference = dataset.reference_fit[0]
    assert reference["Name"] == "I0"
    fit = dataset.fit()
    assert fit["params"]["D"] == pytest.approx(reference["D"], rel=2e-3)
    assert fit["params"]["I0"] == pytest.approx(reference["A"], rel=5e-3)
    # The fit comes from processing.diffusion, so it carries that
    # engine's typed uncertainties and multi-component machinery.
    assert fit["errors"]["D"] / fit["params"]["D"] < 0.01
    assert fit["ppm_range"] == pytest.approx(
        (-0.36134783804285, 0.837444649062855)
    )


@needs_data
def test_phase_shifts_are_one_per_step(dataset):
    assert dataset.ph0.shape == (64,)
    assert dataset.ph1.shape == (64,)
    assert np.abs(dataset.ph0).max() < 5.0


def test_localised_float_handles_both_decimal_separators():
    assert spinsolve._parse_localised_float("2,600749E+07") == pytest.approx(2.600749e7)
    assert spinsolve._parse_localised_float("2.600749E+07") == pytest.approx(2.600749e7)
    assert spinsolve._parse_localised_float("1,234.5") == pytest.approx(1234.5)


def test_rejects_a_file_that_is_not_prospa(tmp_path):
    bogus = tmp_path / "data.2d"
    bogus.write_bytes(b"not a prospa file" + b"\0" * 64)
    with pytest.raises(ValueError, match="not a Prospa"):
        spinsolve.read_prospa(str(bogus))
