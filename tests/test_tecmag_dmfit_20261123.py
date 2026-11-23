"""2026-11-23: TecMag (.tnt and Data Point Listing .txt) and DMFit (.asc / .fxml)
import. Synthetic files written in the real layouts (no instrument data needed)."""
import struct

import numpy as np
import pytest

from opennmr import io
from opennmr.io import dmfit, tecmag

DT = 1e-6
N = 256


def _fid(n=N, offset_hz=20e3, t2=60e-6, seed=0):
    t = np.arange(n) * DT
    rng = np.random.default_rng(seed)
    return np.exp(2j * np.pi * offset_hz * t - t / t2) * 1000.0 + rng.normal(0, 1, n) \
        + 1j * rng.normal(0, 1, n)


def _write_tnt(path, rows, nucleus=b"H1", ob_freq=14.77):
    """A TNT1.008 file: magic, TMAG, DATA, TMG2 and a PSEQ section whose length
    field is not a length (as in real files)."""
    rows = np.atleast_2d(rows)
    records, points = rows.shape
    tmag = np.zeros(1, tecmag._TMAG)[0]
    tmag["npts"] = [points, records, 1, 1]
    tmag["actual_npts"] = [points, records, 1, 1]
    tmag["acq_points"] = points
    tmag["scans"] = tmag["actual_scans"] = 1
    tmag["ob_freq"] = [ob_freq, 0, 0, 0]
    tmag["base_freq"] = [ob_freq, 0, 0, 0]
    tmag["sw"] = [1.0 / DT, 5000, 5000, 5000]
    tmag["dwell"] = [DT, 1e-4, 1e-4, 1e-4]
    tmag["obs_channel"] = 1
    tmag["nuclei"] = [nucleus, b"", b"", b""]
    tmag["actual_temperature"] = 293.0
    tmag["magnet_field"] = 2.11
    data = np.asfortranarray(rows.T).astype("<c8").tobytes(order="F")
    blob = b"TNT1.008"
    for tag, payload in ((b"TMAG", tmag.tobytes()), (b"DATA", data), (b"TMG2", bytes(2048))):
        blob += tag + struct.pack("<II", 1, len(payload)) + payload
    blob += b"PSEQ" + struct.pack("<II", 1, 942747185) + b"junk"
    path.write_bytes(blob)
    return path


def _write_listing(path, rows, extra_header=""):
    rows = np.atleast_2d(rows)
    records, points = rows.shape
    lines = [r"Data Point Listing of File C:\x\y.tnt", f"Starting Point:\t1\tEnding Point:\t{points}"]
    if records > 1:
        lines.append(f"Starting 2D Record:\t1\tEnding Record:\t{records}")
    lines.append("Real\tImag\tusec")
    for row in rows:
        for k, value in enumerate(row):
            lines.append(f"{value.real:.3f}\t{value.imag:.3f}\t{k * DT * 1e6:.3f}")
    path.write_text("\r\n".join(lines) + "\r\n")
    return path


# ------------------------------------------------------------------ TecMag
def test_tnt_1d_is_read_with_its_header(tmp_path):
    fid = _fid()
    path = _write_tnt(tmp_path / "a.tnt", fid)
    assert io.detect_vendor(str(path)) == "tecmag"
    spectrum = io.load(str(path))
    assert spectrum.data.shape == (N,) and spectrum.metadata.nucleus == "1H"
    assert spectrum.metadata.sfo1 == pytest.approx(14.77)
    assert not spectrum.metadata.pseudo_2d
    np.testing.assert_allclose(spectrum.metadata.raw["fid_raw"], fid.astype("<c8"), rtol=1e-6)
    assert spectrum.metadata.raw["fid_dt"] == pytest.approx(DT)
    # The line is at +20 kHz = +20e3 / 14.77 ppm.
    peak = spectrum.axes[0].ppm[int(np.argmax(np.abs(spectrum.data + 1j * spectrum.data_imag)))]
    assert peak == pytest.approx(20e3 / 14.77, abs=1e6 / (N * DT) / 14.77)


def test_tnt_series_is_a_pseudo_2d_with_records_as_rows(tmp_path):
    rows = np.array([_fid(offset_hz=10e3 * (k + 1), seed=k) for k in range(4)])
    spectrum = io.load(str(_write_tnt(tmp_path / "s.tnt", rows)))
    assert spectrum.data.shape == (4, N) and spectrum.metadata.pseudo_2d
    np.testing.assert_allclose(spectrum.metadata.raw["fid_raw"], rows.astype("<c8"), rtol=1e-6)
    assert list(spectrum.axes[0].index) == [1, 2, 3, 4]


def test_listing_matches_the_tnt_and_needs_the_frequency_for_ppm(tmp_path):
    rows = np.array([_fid(seed=k) for k in range(3)]).astype("<c8")
    tnt = _write_tnt(tmp_path / "s.tnt", rows)
    txt = _write_listing(tmp_path / "s.txt", np.round(rows.real) + 1j * np.round(rows.imag))
    assert io.detect_vendor(str(txt)) == "tecmag"
    bare = io.load(str(txt))
    assert bare.metadata.raw["axis_is_hz"] and bare.axes[-1].label == "Frequency (Hz)"
    assert bare.data.shape == (3, N) and bare.metadata.raw["fid_dt"] == pytest.approx(DT)
    with_frequency = io.load(str(txt), nucleus="1H", sfo1=14.77)
    assert not with_frequency.metadata.raw["axis_is_hz"]
    np.testing.assert_allclose(with_frequency.axes[-1].ppm, io.load(str(tnt)).axes[-1].ppm)


def test_conjugate_mirrors_the_spectrum(tmp_path):
    path = _write_tnt(tmp_path / "a.tnt", _fid(offset_hz=30e3))
    a = io.load(str(path), autophase=False)
    b = io.load(str(path), autophase=False, conjugate=True)
    pa = a.axes[0].ppm[int(np.argmax(np.abs(a.data + 1j * a.data_imag)))]
    pb = b.axes[0].ppm[int(np.argmax(np.abs(b.data + 1j * b.data_imag)))]
    assert pa == pytest.approx(-pb, abs=0.1)


def test_bad_files_say_what_is_wrong(tmp_path):
    bad = tmp_path / "bad.tnt"
    bad.write_bytes(b"TNT1.008" + b"\0" * 20)
    with pytest.raises(ValueError, match="TMAG|DATA|section"):
        io.load(str(bad), vendor="tecmag")
    listing = tmp_path / "bad.txt"
    listing.write_text("Data Point Listing of File x\nStarting Point:\t1\tEnding Point:\t5\n"
                       "Real\tImag\tusec\n1\t2\t0\n1\t2\t1\n")
    with pytest.raises(ValueError, match="data lines"):
        io.load(str(listing))


def test_a_tecmag_fid_is_reprocessable_in_the_gui(qapp_free=None):
    from opennmr.gui.main_window import MainWindow
    import inspect
    assert '"tecmag"' in inspect.getsource(MainWindow._fid_reprocessable)


# ------------------------------------------------------------------ DMFit
FXML = """<?xml version="1.0" encoding="utf-8" ?>
<NMRFit>
<FitParameters>
\t<timestamp>2025-12-16T09:01:45Z</timestamp>
\t<DMFitVersion>20230120</DMFitVersion>
\t<FitModeAsc>Fit 1D</FitModeAsc>
\t<Sdev>588188.0035</Sdev>
\t<ComputeParam><ShapeSize>4096</ShapeSize></ComputeParam>
\t<Dimension>F2
\t\t<nucleus>7Li</nucleus>
\t\t<frequency>194.3796699</frequency>
\t\t<BaseLine><BLmode>Polynomial</BLmode><Baseline0>-100</Baseline0></BaseLine>
\t\t<spinrate>40000</spinrate>
\t\t<line>
\t\t\t<ModelName>CSA mas</ModelName>
\t\t\t<amp Fit="*" >1000</amp>
\t\t\t<pos Fit="*"  Unit="ppm" Error="0.005">-5.48</pos>
\t\t\t<gl>0</gl>
\t\t</line>
\t</Dimension>
</FitParameters>
<DataFile><FileName>1r</FileName><FileTitle>7Li NMR</FileTitle></DataFile>
</NMRFit>
"""


def _write_asc(path, n=512, centre_hz=-1065.0, freq=194.38):
    hz = np.linspace(20000.0, -20000.0, n)
    line = 1000.0 * np.exp(-((hz - centre_hz) / 300.0) ** 2)
    spectrum = line + 5.0
    rows = ["ti:7Li NMR", f"##freq {freq}", "##col_ Hz\tSpectrum\tModel\tLine#1"]
    rows += [f"{h:g}\t{s:g}\t{s:g}\t{l:g}" for h, s, l in zip(hz, spectrum, line)]
    path.write_text("\n".join(rows) + "\n")
    return hz, spectrum, line


def test_asc_and_fxml_open_as_a_spectrum_with_the_fit(tmp_path):
    (tmp_path / "1r.fxml").write_text(FXML)
    hz, spectrum, line = _write_asc(tmp_path / "1r spec and model all lines.asc")
    asc = str(tmp_path / "1r spec and model all lines.asc")
    assert io.detect_vendor(asc) == "dmfit" and io.detect_vendor(str(tmp_path / "1r.fxml")) == "dmfit"
    s = io.load(asc)
    assert s.metadata.nucleus == "7Li" and s.metadata.sfo1 == pytest.approx(194.3796699)
    assert s.metadata.raw["dmfit"]["frequency_source"].startswith("fit file")
    # ppm = Hz / MHz with the fit file's frequency, not the rounded ##freq; descending.
    np.testing.assert_allclose(s.axes[0].ppm, hz / 194.3796699, rtol=1e-4)
    assert s.axes[0].ppm[0] > s.axes[0].ppm[-1]
    np.testing.assert_allclose(s.data, spectrum, rtol=1e-5)
    curves = dmfit.fit_curves(s)
    assert list(curves["lines"]) == ["Line#1"]
    assert s.axes[0].ppm[int(np.argmax(curves["lines"]["Line#1"]))] == pytest.approx(-1065.0 / 194.3796699,
                                                                                 abs=0.1)
    assert curves["fit"]["dimensions"][0]["lines"][0]["parameters"]["pos"]["error"] == 0.005
    # Opening the .fxml opens the .asc beside it.
    assert io.load(str(tmp_path / "1r.fxml")).data.shape == s.data.shape


def test_fxml_parameters(tmp_path):
    (tmp_path / "f.fxml").write_text(FXML)
    fit = dmfit.read_fxml(str(tmp_path / "f.fxml"))
    dimension = fit["dimensions"][0]
    assert dimension["nucleus"] == "7Li" and dimension["spinrate_Hz"] == 40000
    line = dimension["lines"][0]
    assert line["model"] == "CSA mas"
    assert line["parameters"]["amp"]["fitted"] and not line["parameters"]["gl"]["fitted"]
    assert line["parameters"]["pos"] == {"value": -5.48, "fitted": True, "fit_flag": "*",
                                         "unit": "ppm", "error": 0.005}
    assert fit["sdev"] == pytest.approx(588188.0035)
    rows = dmfit.fit_table(fit)
    assert {r["parameter"] for r in rows} == {"amp", "pos", "gl"}


def test_asc_without_a_fit_file_uses_the_freq_line_or_stays_in_hz(tmp_path):
    _write_asc(tmp_path / "x.asc")
    s = io.load(str(tmp_path / "x.asc"))
    assert s.metadata.raw["dmfit"]["frequency_source"] == "##freq line"
    assert s.metadata.sfo1 == pytest.approx(194.38)
    t = io.load(str(tmp_path / "x.asc"), sfo1=100.0)
    assert t.metadata.sfo1 == 100.0


def test_a_lone_fxml_says_what_to_export(tmp_path):
    (tmp_path / "f.fxml").write_text(FXML)
    with pytest.raises(ValueError, match="spectrum and model all lines"):
        io.load(str(tmp_path / "f.fxml"))
    (tmp_path / "bad.fxml").write_text("<NMRFit><broken")
    with pytest.raises(ValueError):
        dmfit.read_fxml(str(tmp_path / "bad.fxml"))
