"""io/export.py: .csv, .xy and JCAMP-DX from any vendor."""
import numpy as np
import pytest

from opennmr.io import export, simpson

import simpson_files as sf


@pytest.fixture
def spectrum(tmp_path):
    return simpson.read_1d(sf.write_fid(tmp_path / "a.fid"), nucleus="31P", field=9.4,
                           reference_ppm=5.0)


def test_csv_round_trip(spectrum, tmp_path):
    path = export.export_spectrum(spectrum, str(tmp_path / "o.csv"))
    rows = np.genfromtxt(path, delimiter=",", names=True)
    assert np.allclose(rows["ppm"], spectrum.axes[0].ppm, rtol=1e-9)
    assert np.allclose(rows["real"], spectrum.data, rtol=1e-8, atol=1e-12)
    assert "imag" in rows.dtype.names


def test_xy_has_comments_and_two_columns(spectrum, tmp_path):
    path = export.export_spectrum(spectrum, str(tmp_path / "o.xy"))
    text = open(path).read().splitlines()
    assert text[0].startswith("#")
    data = np.loadtxt(path)
    assert data.shape == (spectrum.data.size, 2)
    assert np.allclose(data[:, 1], spectrum.data, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("ext", [".jdx", ".dx"])
def test_jcamp_round_trip(spectrum, tmp_path, ext):
    path = export.export_spectrum(spectrum, str(tmp_path / f"o{ext}"))
    head = open(path).read()
    assert head.startswith("##TITLE") and "##JCAMP-DX=5.01" in head and "##END=" in head
    assert "##.OBSERVE FREQUENCY" in head
    back = export.read_jcamp_spectrum(path)
    x, y = np.asarray(back["x"]), np.asarray(back["real"])
    assert y.size == spectrum.data.size
    assert np.allclose(y, spectrum.data, rtol=1e-6, atol=1e-9 * np.abs(spectrum.data).max())
    assert np.allclose(x, spectrum.axes[0].hz, rtol=1e-9, atol=1e-6)


def test_jcamp_complex_writes_both_parts(spectrum, tmp_path):
    path = export.export_spectrum(spectrum, str(tmp_path / "c.jdx"), include_imag=True)
    assert "NTUPLES" in open(path).read()
    back = export.read_jcamp_spectrum(path)
    assert np.allclose(np.asarray(back["imag"]), spectrum.data_imag, rtol=1e-6,
                       atol=1e-9 * np.abs(spectrum.data_imag).max())


def test_fid_domain_writes_the_acquired_points(spectrum, tmp_path):
    path = export.export_spectrum(spectrum, str(tmp_path / "f.csv"), domain="fid")
    rows = np.genfromtxt(path, delimiter=",", names=True)
    assert rows.shape[0] == sf.N
    assert rows[rows.dtype.names[0]][1] == pytest.approx(1 / sf.SW)


def test_ascending_and_hz(spectrum, tmp_path):
    path = export.export_spectrum(spectrum, str(tmp_path / "h.csv"), x_unit="hz", ascending=True)
    rows = np.genfromtxt(path, delimiter=",", names=True)
    x = rows[rows.dtype.names[0]]
    assert np.all(np.diff(x) > 0)


def test_unknown_extension_and_format_are_errors(spectrum, tmp_path):
    with pytest.raises(ValueError):
        export.export_spectrum(spectrum, str(tmp_path / "o.bin"))
    with pytest.raises(ValueError):
        export.export_spectrum(spectrum, str(tmp_path / "o.csv"), "pdf")


def test_two_dimensional_needs_a_row_for_xy(tmp_path):
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    ppm = np.linspace(10, -10, 64)
    data = np.random.default_rng(0).normal(size=(4, 64))
    s = NMRSpectrum(data=data, axes=[NMRAxis(kind="index", index=np.arange(1, 5)),
                                     NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 100)],
                    metadata=NMRMetadata(vendor="derived", source_path="", nucleus="1H"))
    with pytest.raises(ValueError):
        export.export_spectrum(s, str(tmp_path / "m.xy"))
    path = export.export_spectrum(s, str(tmp_path / "m.xy"), row=2)
    assert np.allclose(np.loadtxt(path)[:, 1], data[2], rtol=1e-8)
    whole = export.export_spectrum(s, str(tmp_path / "m.csv"))
    assert np.genfromtxt(whole, delimiter=",", skip_header=1).shape[0] == 4
