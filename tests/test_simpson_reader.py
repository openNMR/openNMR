"""io/simpson.py: SIMPSON spectra and FIDs as ordinary 1D NMRSpectrum objects."""
import numpy as np
import pytest

from opennmr.io import base, simpson
from opennmr.notebook import SIMPSONload

import simpson_files as sf


def test_registered_and_detected_by_content_not_name(tmp_path):
    path = sf.write_fid(tmp_path / "result.out")          # an unhelpful extension
    assert base.detect_vendor(path) == "simpson"
    assert simpson.sniff_format(path) == "simp"
    assert simpson.sniff_format(sf.write_xreim(tmp_path / "b.dat")) != "simp"


def test_spectrum_lines_at_the_simulated_frequencies(tmp_path):
    spectrum = simpson.read_1d(sf.write_spe(tmp_path / "a.spe"))
    assert spectrum.ndim == 1 and spectrum.metadata.vendor == "simpson"
    hz = spectrum.axes[0].hz
    assert np.all(np.diff(hz) < 0)                          # high frequency first, as everywhere
    low, high = sf.line_hz(spectrum)
    step = sf.SW / sf.N
    assert low == pytest.approx(-1500.0, abs=step) and high == pytest.approx(3000.0, abs=step)


def test_fid_transformed_here_matches_simpsons_own_spectrum(tmp_path):
    from_fid = simpson.read_1d(sf.write_fid(tmp_path / "a.fid"))
    from_spe = simpson.read_1d(sf.write_spe(tmp_path / "a.spe"))
    assert sf.line_hz(from_fid) == pytest.approx(sf.line_hz(from_spe), abs=sf.SW / sf.N)
    # same shape, not just the same places (no mirror image, no phase twist). The two
    # frequency grids differ by one bin (openNMR's FFT convention), so compare at equal Hz.
    hz_f, hz_s = from_fid.axes[0].hz, from_spe.axes[0].hz
    a = np.abs(from_fid.data + 1j * from_fid.data_imag)
    b = np.interp(hz_f[::-1], hz_s[::-1], np.abs(from_spe.data + 1j * from_spe.data_imag)[::-1])[::-1]
    assert np.corrcoef(a[1:-1], b[1:-1])[0, 1] > 0.99


def test_fid_keeps_the_raw_points_for_live_processing(tmp_path):
    spectrum = simpson.read_1d(sf.write_fid(tmp_path / "a.fid"))
    raw = spectrum.metadata.raw
    assert len(raw["fid_raw"]) == sf.N and raw["fid_dt"] == pytest.approx(1 / sf.SW)
    zf = simpson.read_1d(sf.write_fid(tmp_path / "b.fid"), zero_fill=2048)
    assert zf.data.size == 2048
    lb = simpson.read_1d(sf.write_fid(tmp_path / "c.fid"), window="exponential",
                         window_kwargs={"lb": 200.0})
    assert lb.data.max() < spectrum.data.max()              # broader lines are lower


def test_ppm_axis_needs_nucleus_and_frequency(tmp_path):
    path = sf.write_spe(tmp_path / "a.spe")
    plain = simpson.read_1d(path)
    assert plain.metadata.raw.get("axis_is_hz")
    ppm = simpson.read_1d(path, nucleus="31P", field=9.4, reference_ppm=10.0)
    assert ppm.axes[0].ppm is not None
    # the 3000 Hz line is 3000/sfo1 ppm above the carrier (10 ppm)
    peak = ppm.axes[0].ppm[int(np.argmax(np.abs(ppm.data)))]
    assert peak == pytest.approx(10.0 + 3000.0 / _sfo(ppm), abs=0.05)


def _sfo(spectrum):
    ppm, hz = spectrum.axes[0].ppm, spectrum.axes[0].hz
    return (hz[0] - hz[-1]) / (ppm[0] - ppm[-1])


def test_xreim_files_are_detected_and_kind_can_override(tmp_path):
    path = sf.write_xreim(tmp_path / "a.dat", "spe")
    guessed = simpson.read_1d(path)                  # x runs through zero: a spectrum
    assert guessed.metadata.raw[simpson.RAW_KEY]["kind"] == "SPE"
    spectrum = simpson.read_1d(path, kind="spe")
    low, high = sf.line_hz(spectrum)
    assert high == pytest.approx(3000.0, abs=sf.SW / sf.N)


def test_binary_files_are_refused_clearly(tmp_path):
    path = tmp_path / "b.fid"
    path.write_bytes(b"\x00\x01\x02\xff" * 64)
    with pytest.raises(Exception) as err:
        simpson.read_1d(str(path), kind="fid")
    assert "binary" in str(err.value).lower() or "text" in str(err.value).lower()


def test_notebook_wrapper_matches_the_reader(tmp_path):
    path = sf.write_fid(tmp_path / "a.fid")
    a = SIMPSONload(path, nuclide="13C", field=11.7)
    b = simpson.read_1d(path, nucleus="13C", field=11.7)
    assert np.allclose(a.data, b.data) and np.allclose(a.axes[0].ppm, b.axes[0].ppm)
