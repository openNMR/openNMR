"""Importing the CSV / XY / JCAMP-DX files that export_spectrum writes."""
import numpy as np
import pytest

from opennmr import io
from opennmr.io.export import export_spectrum
from opennmr.io import simpson


def _spectrum():
    ppm = np.linspace(10, -2, 512)
    real = np.exp(-((ppm - 4.0) / 0.1) ** 2) + 0.5 * np.exp(-((ppm - 1.0) / 0.1) ** 2)
    imag = 0.2 * np.exp(-((ppm - 4.0) / 0.2) ** 2)
    sp = simpson.read_1d.__globals__["NMRSpectrum"](
        data=real,
        axes=[simpson.NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=simpson.NMRMetadata(vendor="x", nucleus="1H", sfo1=400.0, bf1=400.0, o1=0.0,
                                     source_path="a", ns=None, pulse_program=None,
                                     raw={"title": "demo"}),
        data_imag=imag)
    return sp


@pytest.mark.parametrize("name,imag", [("a.csv", True), ("a.xy", False), ("a.jdx", False),
                                       ("a.jdx", True), ("a.dx", False)])
def test_round_trip_spectrum(tmp_path, name, imag):
    sp = _spectrum()
    path = str(tmp_path / name)
    export_spectrum(sp, path, include_imag=imag, metadata_comments=True)
    assert io.detect_vendor(path) == "text"
    back = io.load(path)
    assert back.metadata.nucleus in ("1H", None) or name.endswith(".csv")
    ppm = back.axes[-1].ppm
    assert ppm[0] > ppm[-1]
    np.testing.assert_allclose(ppm, sp.axes[0].ppm, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(back.data, sp.data, atol=1e-4)
    if imag:
        np.testing.assert_allclose(back.data_imag, sp.data_imag, atol=1e-4)


def test_ascending_and_hz(tmp_path):
    sp = _spectrum()
    path = str(tmp_path / "b.xy")
    export_spectrum(sp, path, x_unit="hz", ascending=True)
    back = io.load(path, sfo1=400.0)
    np.testing.assert_allclose(back.axes[-1].ppm, sp.axes[0].ppm, atol=1e-3)


def test_hz_without_frequency_stays_hz(tmp_path):
    sp = _spectrum()
    path = str(tmp_path / "c.csv")
    export_spectrum(sp, path, x_unit="hz", metadata_comments=False)
    back = io.load(path)
    assert back.metadata.raw["axis_is_hz"] is True
    assert abs(back.axes[-1].ppm[0] - 4000.0) < 1.0 or back.axes[-1].ppm[0] > 1000


def test_fid_round_trip(tmp_path):
    t = np.arange(1024) * 1e-4
    fid = np.exp(-t / 0.02) * np.exp(2j * np.pi * 500 * t)
    raw = {"NP": 1024, "SW": 1e4, "real": fid.real, "imag": fid.imag, "NELEM": 1,
           "element": 0, "kind": "FID", "format": "x"}
    sp = simpson.read_1d("f.dat", raw=raw, title="fid", nucleus="1H", sfo1=400.0, conjugate=False)
    path = str(tmp_path / "f.csv")
    export_spectrum(sp, path, domain="fid", metadata_comments=True)
    back = io.load(path)
    assert back.metadata.raw["fid_dt"] == pytest.approx(1e-4, rel=1e-6)
    np.testing.assert_allclose(back.metadata.raw["fid_raw"], fid, atol=1e-5)


def test_2d_csv(tmp_path):
    sp = _spectrum()
    sp2 = type(sp)(data=np.vstack([sp.data, 2 * sp.data]),
                   axes=[simpson.NMRAxis(kind="index", index=np.array([1.0, 2.0])), sp.axes[0]],
                   metadata=sp.metadata)
    path = str(tmp_path / "d.csv")
    export_spectrum(sp2, path)
    back = io.load(path)
    assert back.data.shape == (2, 512)
    np.testing.assert_allclose(back.data[1], 2 * sp.data, atol=1e-4)


def test_plain_two_column_file(tmp_path):
    path = tmp_path / "plain.xy"
    path.write_text("\n".join(f"{x} {x * x}" for x in range(5)))
    back = io.load(str(path))
    assert back.data.shape == (5,)


def test_bad_file(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\nx,y\n")
    with pytest.raises(ValueError):
        io.load(str(path))
