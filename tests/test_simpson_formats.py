"""SIMPSON file shapes beyond the plain real FID / spectrum (user request,
2026-10-23): complex data, NELEM, headerless x/re/im .dat, scan tables, pulse
shapes and input scripts. The small files in tests/data/simpson are the user's
own examples; the large ones are regenerated here in the same layout."""
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

import opennmr.notebook as nmr
from opennmr.io import simpson as sio

import simpson_files as sf

DATA = os.path.join(os.path.dirname(__file__), "data", "simpson")


def sample(name):
    return os.path.join(DATA, name)


def _six_figures(values):
    """What SIMPSON writes: x to about six significant figures."""
    return np.array([float(f"{v:.5e}") for v in values])


def _write_rounded_xreim(path, n=4096, sw=6.0e6):
    fid = sf.make_fid(n=n, sw=sw, lines=((250e3, 20e3, 1.0), (-700e3, 40e3, 0.6)))
    x = np.fft.fftshift(np.fft.fftfreq(n, 1 / sw))
    values = np.fft.fftshift(np.fft.fft(fid))
    with open(path, "w") as handle:
        for xi, v in zip(_six_figures(x), values):
            handle.write(f"{xi:.5e} {v.real:.8e} {v.imag:.8e}\n")
    return str(path), x, values


# ---- NELEM -------------------------------------------------------------------

def test_the_example_fid_holds_two_elements():
    path = sample("example.fid")
    assert sio.n_elements(path) == 2
    first = sio.read_1d(path, element=0)
    second = sio.read_1d(path, element=1)
    assert first.metadata.raw["title"] == "example.fid [1/2]"
    assert second.metadata.raw["title"] == "example.fid [2/2]"
    assert len(first.data) == len(second.data) == 128
    assert not np.allclose(first.data, second.data)


def test_elements_are_consecutive_blocks(tmp_path):
    a, b = sf.make_fid(n=64), 0.5 * sf.make_fid(n=64, lines=((1000.0, 100.0, 1.0),))
    path = tmp_path / "two.fid"
    with open(path, "w") as handle:
        handle.write("SIMP\nNP=64\nSW=20000\nNELEM=2\nTYPE=FID\nDATA\n")
        for v in np.concatenate([a, b]):
            handle.write(f"{v.real:.10e} {v.imag:.10e}\n")
        handle.write("END\n")
    for k, expected in enumerate((a, b)):
        raw = sio.load_raw(str(path), element=k)
        assert np.allclose(raw["real"], expected.real) and np.allclose(raw["imag"], expected.imag)
    with pytest.raises(ValueError, match="out of range"):
        sio.load_raw(str(path), element=2)


def test_a_wrong_line_count_for_nelem_is_reported(tmp_path):
    path = tmp_path / "short.fid"
    path.write_text("SIMP\nNP=4\nSW=1000\nNELEM=2\nTYPE=FID\nDATA\n" + "1 0\n" * 4 + "END\n")
    with pytest.raises(ValueError, match="NELEM=2"):
        sio.load_raw(str(path))


# ---- real and imaginary parts ------------------------------------------------

def test_a_fid_with_trailing_zero_rows_keeps_both_parts():
    spectrum = nmr.SIMPSONload(sample("example.fid"))
    assert spectrum.data_imag is not None
    assert len(spectrum.data) == 128


def test_both_parts_of_a_fid_and_spectrum_survive(tmp_path):
    fid = sf.write_fid(tmp_path / "a.fid")
    spe = sf.write_spe(tmp_path / "a.csdf")             # .csdf is a spectrum too
    for path in (fid, spe):
        spectrum = nmr.SIMPSONload(path)
        assert np.max(np.abs(spectrum.data_imag)) > 0.1 * np.max(np.abs(spectrum.data))


# ---- headerless x, re, im ----------------------------------------------------

def test_xreim_spectrum_is_recognised_without_kind(tmp_path):
    path, x, values = _write_rounded_xreim(tmp_path / "s.dat")
    info = sio.describe(path)
    assert info["role"] == "data"
    spectrum = nmr.SIMPSONload(path)
    assert spectrum.metadata.raw[sio.RAW_KEY]["kind"] == "SPE"
    assert spectrum.metadata.raw[sio.RAW_KEY]["kind_guessed"] is True
    hz = spectrum.axes[0].hz
    assert np.all(np.diff(hz) < 0)                       # descending, like every spectrum
    # rounded x is replaced by the evenly spaced axis it stands for
    assert np.allclose(np.diff(hz), np.diff(hz)[0], rtol=1e-9)
    assert np.max(np.abs(hz[::-1] - x)) < 1e-6 * 3e6
    complex_data = (spectrum.data + 1j * spectrum.data_imag)[::-1]
    assert np.allclose(complex_data, values, atol=1e-6 * np.max(np.abs(values)))


def test_xreim_matches_the_spe_it_was_saved_with(tmp_path):
    fid = sf.make_fid()
    spe = sf.write_spe(tmp_path / "x.spe", fid)
    dat = sf.write_xreim(tmp_path / "x.dat", kind="spe")
    a, b = sio.read_1d(spe), sio.read_1d(dat)
    assert np.allclose(a.axes[0].hz, b.axes[0].hz, atol=1e-6)
    assert np.allclose(a.data, b.data, atol=1e-6) and np.allclose(a.data_imag, b.data_imag, atol=1e-6)


def test_xreim_fid_starting_at_zero_is_a_fid(tmp_path):
    path = sf.write_xreim(tmp_path / "t.dat", kind="fid")
    raw = sio.load_raw(path, time_unit="s")
    assert raw["kind"] == "FID" and raw["kind_guessed"] is True
    assert raw["SW"] == pytest.approx(sf.SW, rel=1e-9)


def test_kind_overrides_the_guess(tmp_path):
    path = sf.write_xreim(tmp_path / "t.dat", kind="spe")
    assert sio.load_raw(path, kind="spe")["kind_guessed"] is False


# ---- scan tables -------------------------------------------------------------

def test_a_scan_table_is_not_read_as_a_spectrum():
    path = sample("quad_invert_output.dat")
    assert sio.describe(path)["role"] == "table"
    with pytest.raises(ValueError, match="table of x, y, z"):
        sio.read_1d(path)
    with pytest.raises(ValueError, match="table of x, y, z"):
        sio.read_1d(path, kind="spe")


def test_scan_table_grid():
    table = nmr.SIMPSONread_table(sample("quad_invert_output.dat"))
    assert table["grid"] and table["Z"].shape == (15, 16)
    assert table["xs"][0] == 500 and table["xs"][-1] == 8000
    assert table["ys"][0] == 2 and table["ys"][-1] == 30
    j, i = 3, 5                                           # a spot value matches the raw column
    row = np.flatnonzero((table["x"] == table["xs"][i]) & (table["y"] == table["ys"][j]))[0]
    assert table["Z"][j, i] == table["z"][row]


def test_table_plots_as_map_and_as_lines():
    table = nmr.SIMPSONread_table(sample("quad_invert_output.dat"))
    fig, ax = nmr.SIMPSONplot_table(table, xlabel="rf (kHz)", ylabel="offset", zlabel="efficiency")
    assert ax.get_xlabel() and len(fig.axes) == 2           # map + colour bar
    fig, ax = nmr.SIMPSONplot_table(sample("quad_invert_output.dat"), mode="lines")
    assert len(ax.lines) == 15
    with pytest.raises(ValueError):
        nmr.SIMPSONplot_table(table, mode="bogus")


def test_ragged_table_has_no_grid(tmp_path):
    path = tmp_path / "r.dat"
    path.write_text("1 1 5\n1 2 6\n2 1 7\n")
    table = nmr.SIMPSONread_table(str(path))
    assert not table["grid"]
    fig, ax = nmr.SIMPSONplot_table(table)
    assert len(ax.lines) == 1
    with pytest.raises(ValueError, match="complete grid"):
        nmr.SIMPSONplot_table(table, mode="map")


# ---- pulse shapes and scripts ------------------------------------------------

def test_pulse_shape_is_recognised_and_plotted():
    path = sample("inv_pow.shape")
    assert sio.describe(path)["role"] == "shape"
    with pytest.raises(ValueError, match="pulse-shape"):
        sio.read_1d(path)
    shape = nmr.SIMPSONread_shape(path)
    assert shape["npoints"] > 100
    fig, (ax_amp, ax_phase) = nmr.SIMPSONplot_shape(path)
    assert len(ax_amp.lines) == 1 and len(ax_phase.lines) == 1


def test_scripts_give_nucleus_and_field():
    wurst = nmr.SIMPSONread_script(sample("inv_pow_wurst.tcl"))
    assert wurst["nucleus"] == "31P" and wurst["proton_mhz"] == pytest.approx(500.0)
    assert wurst["spin_rate"] == 100000 and wurst["sw"] == 6.0e6 and wurst["np"] == 8192 * 6
    quad = nmr.SIMPSONread_script(sample("quadrupolar_87Rb.tcl"))
    assert quad["nucleus"] == "87Rb" and quad["proton_mhz"] == pytest.approx(400.0)
    assert sio.describe(sample("inv_pow_wurst.tcl"))["role"] == "script"
    with pytest.raises(ValueError, match="input script"):
        sio.read_1d(sample("inv_pow_wurst.tcl"))


def test_a_script_is_read_not_run(tmp_path):
    path = tmp_path / "evil.tcl"
    path.write_text("spinsys {\n channels 13C\n}\npar {\n proton_frequency 600e6\n"
                    " np [exec touch SHOULD_NOT_EXIST]\n sw __import__('os')\n}\n")
    settings = nmr.SIMPSONread_script(str(path))
    assert settings["nucleus"] == "13C" and settings["proton_mhz"] == pytest.approx(600.0)
    assert settings["np"] is None and settings["sw"] is None
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists() and not os.path.exists("SHOULD_NOT_EXIST")


# ---- the big examples, regenerated ---------------------------------------------

def test_a_131072_point_dat_and_spe_agree(tmp_path):
    n, sw = 131072, 6.0e6
    fid = sf.make_fid(n=n, sw=sw, lines=((250e3, 30e3, 1.0), (-1.2e6, 60e3, 0.4)))
    spe_values = np.fft.fftshift(np.fft.fft(fid))
    spe = sf._write(tmp_path / "big.spe", "SPE", spe_values, sw)
    x = np.fft.fftshift(np.fft.fftfreq(n, 1 / sw))
    dat = tmp_path / "big.dat"
    np.savetxt(dat, np.column_stack([_six_figures(x), spe_values.real, spe_values.imag]),
               fmt=["%.5e", "%.8e", "%.8e"])
    a, b = sio.read_1d(spe), sio.read_1d(str(dat))
    scale = np.max(np.abs(a.data + 1j * a.data_imag))
    assert np.max(np.abs((a.data + 1j * a.data_imag) - (b.data + 1j * b.data_imag))) < 1e-5 * scale
    assert b.axes[0].hz[0] == pytest.approx(a.axes[0].hz[0], abs=1.0)


def test_unknown_files_are_called_unknown(tmp_path):
    path = tmp_path / "junk.dat"
    path.write_text("a b c d e\n1 2 3 4 5\n")
    assert sio.describe(str(path))["role"] == "unknown"
