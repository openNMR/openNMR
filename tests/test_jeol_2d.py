"""2D JEOL Delta ``.jdf`` files (explicit user request, 2026-11-09):
genuine 2D (States, complex indirect axis) and pseudo-2D series (a real
indirect axis with its own list of delays), read through
``opennmr.io.load`` like every other vendor.

Synthetic files (built below, field for field what ``io.jeol`` reads)
cover the layout -- submatrix tiling, section order, the used-row trim,
the delay list and its unit -- and the processing options. The two real
files the reader was written against (a 14N-1H HMQC and a 1H saturation
recovery, JEOL ECZ 600, Delta 5.2.1) are checked at the end when they
are present in ``test_data/jeol_2d/`` (local, not committed).
"""

import os
import struct

import numpy as np
import pytest

from opennmr import io
from opennmr.io import jeol

from test_gui import qapp  # noqa: F401 -- fixture
from test_jeol import BE, _pack_param

_TWO_D, _SMALL_TWO_D = 2, 12
_REAL, _COMPLEX = 1, 3
_SECOND, _HERTZ, _PPM = 28, 13, 26


def _tile(array: np.ndarray, edge: int) -> np.ndarray:
    """The inverse of ``jeol._untile``: an (n_y, n_x) section in JEOL's
    submatrix order."""
    n_y, n_x = array.shape
    return array.reshape(n_y // edge, edge, n_x // edge, edge).transpose(0, 2, 1, 3).ravel()


def _build_jdf_2d(rows_cos, rows_sin=None, *, data_format=_TWO_D, stored_rows=None,
                  x_sweep_hz=1000.0, x_freq_hz=400e6, x_offset_ppm=5.0, x_domain="Proton",
                  y_domain="Carbon13", y_sweep_hz=500.0, y_freq_hz=100e6, y_offset_ppm=50.0,
                  axis_list=None, list_milli=True, little_endian_body=True,
                  experiment="synthetic_2d"):
    """A synthetic 2D ``.jdf``: ``rows_cos`` (n_y x n_x complex) alone for a
    real indirect axis, with ``rows_sin`` for a complex one. ``stored_rows``
    pads the indirect axis with zero rows past the acquired ones, as the
    real file does to fill its submatrices. ``axis_list``: the indirect
    axis's values (in ms when ``list_milli``)."""
    edge = 32 if data_format == _TWO_D else 4
    rows_cos = np.asarray(rows_cos, dtype=complex)
    n_used, n_x = rows_cos.shape
    n_y = stored_rows or n_used
    complex_y = rows_sin is not None

    def padded(rows):
        out = np.zeros((n_y, n_x), dtype=complex)
        out[:n_used] = rows
        return out

    parts = [padded(rows_cos)] + ([padded(rows_sin)] if complex_y else [])
    sections = []
    for part in parts:                     # x part changes fastest: R, I for each y part
        sections += [part.real, part.imag]

    header = bytearray()
    header += b"JEOL.NMR"
    header += struct.pack(BE + "b", 1 if little_endian_body else 0)
    header += struct.pack(BE + "B", 6) + struct.pack(BE + "H", 0)
    header += struct.pack(BE + "B", 2)                      # dataDimensionNumber
    header += struct.pack(BE + "B", 0b11000000)             # dataDimensionExist
    header += struct.pack(BE + "B", (0 << 6) | data_format)  # 64-bit float, format
    header += struct.pack(BE + "b", 3) + b"\x00" * 8
    header += bytes([_COMPLEX, _COMPLEX if complex_y else _REAL] + [0] * 6)
    y_unit_prefix = (1 << 4) | 1 if (axis_list is not None and list_milli) else 1
    header += bytes([1, _SECOND, y_unit_prefix, _SECOND]) + bytes([0, 0]) * 6
    header += b"synthetic".ljust(124, b"\x00")
    header += bytes([0x02 if axis_list is not None else 0, 0, 0, 0])   # dataAxisRanged
    header += b"".join(struct.pack(BE + "I", v) for v in [n_x, n_y] + [1] * 6)
    header += b"".join(struct.pack(BE + "I", v) for v in [0] * 8)
    header += b"".join(struct.pack(BE + "I", v) for v in [n_x - 1, n_used - 1] + [0] * 6)
    y_start = float(axis_list[0]) if axis_list is not None else 0.0
    y_stop = float(axis_list[-1]) if axis_list is not None else (n_used - 1) / y_sweep_hz
    header += b"".join(struct.pack(BE + "d", v) for v in [0.0, y_start] + [0.0] * 6)
    header += b"".join(struct.pack(BE + "d", v)
                       for v in [(n_x - 1) / x_sweep_hz, y_stop] + [0.0] * 6)
    header += b"\x00" * (4 + 4 + 16 + 128 + 128)
    header += b"test fixture".ljust(128, b"\x00")
    titles = ["Proton", "interval" if axis_list is not None else y_domain]
    header += b"".join(t.encode("ascii").ljust(32, b"\x00") for t in titles) + b"\x00" * (32 * 6)
    header += b"\x00" * (8 * 8 + 8 * 8 + 8 + 3 + 1 + 4 + 4)

    endian = "<" if little_endian_body else BE
    params = [
        _pack_param("X_DOMAIN", x_domain, "String", endian=endian),
        _pack_param("X_SWEEP", x_sweep_hz, "Float", unit_base=_HERTZ, endian=endian),
        _pack_param("X_FREQ", x_freq_hz, "Float", unit_base=_HERTZ, endian=endian),
        _pack_param("X_OFFSET", x_offset_ppm, "Float", unit_base=_PPM, endian=endian),
        _pack_param("Y_DOMAIN", y_domain, "String", endian=endian),
        _pack_param("Y_SWEEP", y_sweep_hz, "Float", unit_base=_HERTZ, endian=endian),
        _pack_param("Y_FREQ", y_freq_hz, "Float", unit_base=_HERTZ, endian=endian),
        _pack_param("Y_OFFSET", y_offset_ppm, "Float", unit_base=_PPM, endian=endian),
        _pack_param("SCANS", 8, "Integer", endian=endian),
        _pack_param("experiment", experiment, "String", endian=endian),
    ]
    param_section = bytearray(struct.pack(endian + "I", 0) + struct.pack(endian + "I", 0)
                              + struct.pack(endian + "I", len(params) - 1)
                              + struct.pack(endian + "I", 0))
    for p in params:
        param_section += p

    param_start = len(header) + 4 + 4 + 32 + 32 + 4
    list_bytes = b""
    if axis_list is not None:
        values = list(axis_list) + [0.0] * (n_y - len(axis_list))
        list_bytes = b"".join(struct.pack(">d", v) for v in values)   # big-endian, as real
    list_start = param_start + len(param_section)
    data_start = list_start + len(list_bytes)
    header += struct.pack(BE + "I", param_start) + struct.pack(BE + "I", len(param_section))
    header += b"".join(struct.pack(BE + "I", v) for v in [list_start] * 8)
    header += b"".join(struct.pack(BE + "I", v)
                       for v in [0, len(list_bytes)] + [0] * 6)
    header += struct.pack(BE + "I", data_start)
    assert len(header) == param_start

    data = b"".join(_tile(sec, edge).astype(endian + "f8").tobytes() for sec in sections)
    return bytes(header) + bytes(param_section) + list_bytes + data


def _write(tmp_path, name="two_d.jdf", **kwargs):
    path = tmp_path / name
    path.write_bytes(_build_jdf_2d(**kwargs))
    return str(path)


# ---- synthetic signals -------------------------------------------------------------------

_N2, _N1 = 64, 32
_SW2, _SW1 = 1000.0, 500.0
_F2, _F1 = 125.0, 62.5          # Hz from each carrier: exact bins of the transforms


def _hmqc_like(n1=_N1, n2=_N2):
    """One cross peak at (+_F1, +_F2) Hz from the carriers, as cosine and
    sine rows. The shared FFT puts e^(-iωt) at +ω; JEOL stores both axes
    alike (real, imaginary), so a positive F1 offset is cos + 1j*(-sin),
    as a positive F2 offset is e^(-iωt) (see io.jeol._F1_SIGN_FLIP)."""
    t2 = np.arange(n2) / _SW2
    t1 = np.arange(n1) / _SW1
    f2 = np.exp(-2j * np.pi * _F2 * t2) * np.exp(-t2 * 20)
    decay1 = np.exp(-t1 * 10)
    cos = np.outer(np.cos(2 * np.pi * _F1 * t1) * decay1, f2)
    sin = np.outer(-np.sin(2 * np.pi * _F1 * t1) * decay1, f2)
    return cos, sin


_DELAYS_MS = [15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0]


def _recovery_rows(t1_s=0.2, n2=_N2):
    t2 = np.arange(n2) / _SW2
    fid = np.exp(-2j * np.pi * _F2 * t2) * np.exp(-t2 * 20)
    amplitude = 1.0 - np.exp(-np.asarray(_DELAYS_MS) / 1e3 / t1_s)
    return np.outer(amplitude, fid)


# ---- layout ---------------------------------------------------------------------------------

@pytest.mark.parametrize("edge", [4, 32])
def test_untile_inverts_the_submatrix_order(edge):
    a = np.arange(64 * 32, dtype=float).reshape(32, 64)
    assert np.array_equal(jeol._untile(_tile(a, edge), 64, 32, edge), a)


def test_genuine_2d_peak_lands_where_it_was_put(tmp_path):
    cos, sin = _hmqc_like()
    path = _write(tmp_path, rows_cos=cos, rows_sin=sin)
    spectrum = io.load(path, autophase=False, window_kwargs={"lb": 0.0})
    assert spectrum.ndim == 2 and not spectrum.is_pseudo_2d
    meta = spectrum.metadata
    assert (meta.vendor, meta.nucleus, meta.nucleus_indirect) == ("jeol", "1H", "13C")
    assert meta.sfo1_indirect == pytest.approx(100.0)
    assert meta.raw["fn_mode"] == 4 and meta.raw["f1_quadrature_sign_flip"] is False
    i, j = np.unravel_index(np.argmax(spectrum.data), spectrum.data.shape)
    assert spectrum.axes[1].ppm[j] == pytest.approx(5.0 + _F2 / 400.0, abs=0.02)
    assert spectrum.axes[0].ppm[i] == pytest.approx(50.0 + _F1 / 100.0, abs=0.1)
    # Absorptive in both dimensions: the peak is positive and real.
    assert spectrum.data[i, j] > 0
    assert abs(spectrum.data_imag[i, j]) < 0.2 * spectrum.data[i, j]


def test_small_two_d_tiles_and_big_endian_body(tmp_path):
    cos, sin = _hmqc_like(n1=8, n2=16)
    path = _write(tmp_path, rows_cos=cos, rows_sin=sin, data_format=_SMALL_TWO_D,
                  little_endian_body=False)
    spectrum = io.load(path, autophase=False, window_kwargs={"lb": 0.0}, zero_fill_f1=0)
    assert spectrum.data.shape == (8, 16)


def test_raw_rows_are_cosine_then_sine_per_increment(tmp_path):
    cos, sin = _hmqc_like()
    path = _write(tmp_path, rows_cos=cos, rows_sin=sin)
    raw = io.load(path, autophase=False).metadata.raw["fid_raw"]
    assert raw.shape == (2 * _N1, _N2)
    assert np.allclose(raw[0::2], cos) and np.allclose(raw[1::2], sin)


def test_unused_padding_rows_are_dropped(tmp_path):
    rows = _recovery_rows()
    path = _write(tmp_path, rows_cos=rows, data_format=_SMALL_TWO_D, stored_rows=8,
                  axis_list=_DELAYS_MS, y_domain="")
    spectrum = io.load(path, autophase=False)
    assert spectrum.data.shape[0] == len(_DELAYS_MS)      # 7 acquired of 8 stored


# ---- pseudo-2D --------------------------------------------------------------------------------

def test_series_carries_its_delays_in_seconds(tmp_path):
    path = _write(tmp_path, rows_cos=_recovery_rows(), data_format=_SMALL_TWO_D,
                  stored_rows=8, axis_list=_DELAYS_MS, y_domain="")
    spectrum = io.load(path, autophase=False)
    assert spectrum.is_pseudo_2d
    assert spectrum.axes[0].kind == "index"
    recorded = spectrum.metadata.raw["indirect_values"]
    assert recorded["unit"] == "s"
    assert np.allclose(recorded["values"], np.asarray(_DELAYS_MS) / 1e3)
    assert "interval" in recorded["source"] and "list" in recorded["source"]
    # The rows grow with the delay, in the list's order.
    heights = spectrum.data.max(axis=1)
    assert np.all(np.diff(heights) > 0)


def test_series_without_a_list_spaces_the_axis_range_evenly(tmp_path):
    path = _write(tmp_path, rows_cos=_recovery_rows(), data_format=_SMALL_TWO_D,
                  stored_rows=8, axis_list=None, y_domain="")
    recorded = io.load(path, autophase=False).metadata.raw["indirect_values"]
    assert recorded["values"][0] == pytest.approx(0.0)
    assert len(recorded["values"]) == len(_DELAYS_MS)
    assert "evenly spaced" in recorded["source"]


def test_peek_says_what_each_file_holds(tmp_path):
    cos, sin = _hmqc_like()
    genuine = io.peek(_write(tmp_path, "g.jdf", rows_cos=cos, rows_sin=sin))
    assert genuine["ndim"] == 2 and genuine["shape"] == (_N1, _N2)
    assert genuine["nucleus_indirect"] == "13C" and genuine["pseudo_2d"] is False
    series = io.peek(_write(tmp_path, "s.jdf", rows_cos=_recovery_rows(),
                            data_format=_SMALL_TWO_D, stored_rows=8, axis_list=_DELAYS_MS,
                            y_domain=""))
    assert series["pseudo_2d"] is True and series["shape"] == (len(_DELAYS_MS), _N2)


# ---- options -------------------------------------------------------------------------------------

def test_f1_zero_fill_and_truncation(tmp_path):
    cos, sin = _hmqc_like()
    path = _write(tmp_path, rows_cos=cos, rows_sin=sin)
    assert io.load(path, autophase=False).data.shape[0] == 64          # default: 2 x 32
    assert io.load(path, autophase=False, zero_fill_f1=128).data.shape[0] == 128
    assert io.load(path, autophase=False, zero_fill_f1=0).data.shape[0] == _N1
    assert io.load(path, autophase=False, zero_fill_f1=0, truncate_f1=16).data.shape[0] == 16


def test_genuine_2d_can_be_shown_as_its_rows(tmp_path):
    cos, sin = _hmqc_like()
    spectrum = io.load(_write(tmp_path, rows_cos=cos, rows_sin=sin),
                       autophase=False, pseudo_2d=True)
    assert spectrum.is_pseudo_2d and spectrum.data.shape[0] == 2 * _N1
    assert "your choice" in spectrum.metadata.raw["pseudo_2d_basis"]


def test_real_indirect_axis_cannot_be_transformed(tmp_path):
    path = _write(tmp_path, rows_cos=_recovery_rows(), data_format=_SMALL_TWO_D,
                  stored_rows=8, axis_list=_DELAYS_MS, y_domain="")
    with pytest.raises(ValueError, match="no F1 quadrature"):
        io.load(path, pseudo_2d=False)


def test_drift_correction_series_only(tmp_path):
    series = _write(tmp_path, "s.jdf", rows_cos=_recovery_rows(), data_format=_SMALL_TWO_D,
                    stored_rows=8, axis_list=_DELAYS_MS, y_domain="")
    corrected = io.load(series, autophase=False, drift_correction={"start_hz": 0.0, "end_hz": 5.0})
    assert "drift_correction" in corrected.metadata.raw
    cos, sin = _hmqc_like()
    with pytest.raises(ValueError, match="Drift correction"):
        io.load(_write(tmp_path, "g.jdf", rows_cos=cos, rows_sin=sin),
                drift_correction={"start_hz": 0.0, "end_hz": 5.0})


def test_recorded_phase_mode_falls_back_and_says_so(tmp_path):
    cos, sin = _hmqc_like()
    spectrum = io.load(_write(tmp_path, rows_cos=cos, rows_sin=sin), phase_mode="recorded")
    assert "records no phase" in spectrum.metadata.raw["phase_note"]


def test_unknown_kwargs_from_the_gui_are_ignored(tmp_path):
    cos, sin = _hmqc_like()
    spectrum = io.load(_write(tmp_path, rows_cos=cos, rows_sin=sin), autophase=False,
                       baseline_correct=True, normalize=False, procno=1)
    assert spectrum.ndim == 2


def test_read_1d_points_a_2d_file_to_load(tmp_path):
    cos, sin = _hmqc_like()
    with pytest.raises(ValueError, match="is a 2D JEOL file"):
        jeol.read_1d(_write(tmp_path, rows_cos=cos, rows_sin=sin))


def test_the_readers_f1_sense_is_not_saved_as_processing(tmp_path):
    """A saved copy would override a later correction to the reader on
    every reopen of the file (found while writing this reader)."""
    from opennmr.io import persistence
    cos, sin = _hmqc_like()
    spectrum = io.load(_write(tmp_path, rows_cos=cos, rows_sin=sin), phase_mode="recorded")
    state = persistence.processing_state(spectrum)
    assert "f1_quadrature_sign_flip" not in state and "phase_note" not in state


def test_not_divided_by_scans(tmp_path):
    cos, sin = _hmqc_like()
    meta = io.load(_write(tmp_path, rows_cos=cos, rows_sin=sin), autophase=False).metadata
    assert meta.raw["per_scan"] is False and meta.ns is None and meta.raw["scans"] == 8


# ---- in the window -------------------------------------------------------------------------

def test_dynamics_auto_values_are_the_files_delays(qapp, tmp_path):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    path = _write(tmp_path, rows_cos=_recovery_rows(), data_format=_SMALL_TWO_D,
                  stored_rows=8, axis_list=_DELAYS_MS, y_domain="")
    window._load(path, vendor="jeol")
    spectrum = window.session.spectra[-1]
    assert spectrum.is_pseudo_2d
    window.delay_source_combo.setCurrentIndex(0)     # Auto
    values = window._resolve_independent_values(spectrum, is_diffusion=False)
    assert np.allclose(values, np.asarray(_DELAYS_MS) / 1e3)
    assert "JEOL" in window._values_provenance
    assert window._values_note is None


def test_window_opens_a_genuine_2d_as_one_and_reprocesses_it_alike(qapp, tmp_path):
    """The window opens other vendors' 2D as a series by default (Bruker's
    own detection is unreliable); a JEOL file says which it is, so its
    reader's choice stands. The in-memory recompute (F2 controls, then
    F1 again) must put the peak where the reader did -- same F1 sign."""
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    cos, sin = _hmqc_like()
    window._load(_write(tmp_path, rows_cos=cos, rows_sin=sin), vendor="jeol")
    spectrum = window.session.spectra[-1]
    assert not spectrum.is_pseudo_2d
    assert window.process_indirect_button.isChecked()

    def peak(s):
        magnitude = np.hypot(s.data, s.data_imag)
        i, j = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        return s.axes[0].ppm[i], s.axes[1].ppm[j]

    f1, f2 = peak(spectrum)
    assert f1 == pytest.approx(50.0 + _F1 / 100.0, abs=0.1)
    assert window._recompute_multi_row_data(spectrum)
    assert peak(spectrum) == pytest.approx((f1, f2), abs=0.1)


# ---- the real files (local only) ---------------------------------------------------------

_REAL_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "test_data", "jeol_2d")
_HMQC = os.path.join(_REAL_DIR, "hmqc_14N_1H.jdf")
_SATREC = os.path.join(_REAL_DIR, "satrec_1H.jdf")


@pytest.mark.skipif(not os.path.exists(_SATREC), reason="real JEOL data not present")
def test_real_saturation_recovery():
    spectrum = io.load(_SATREC)
    assert spectrum.is_pseudo_2d and spectrum.data.shape == (21, 1014)
    delays = spectrum.metadata.raw["indirect_values"]["values"]
    assert delays[0] == pytest.approx(0.015) and delays[-1] == pytest.approx(150.0)
    assert np.all(np.diff(delays) > 0)
    # The magnitude recovers monotonically along the list.
    heights = np.abs(spectrum.data + 1j * spectrum.data_imag).max(axis=1)
    assert np.all(np.diff(heights) > -0.01 * heights.max())


@pytest.mark.skipif(not os.path.exists(_HMQC), reason="real JEOL data not present")
def test_real_hmqc_cross_peaks():
    spectrum = io.load(_HMQC, window_kwargs={"lb": 300.0})
    assert not spectrum.is_pseudo_2d
    assert (spectrum.metadata.nucleus, spectrum.metadata.nucleus_indirect) == ("1H", "14N")
    magnitude = np.hypot(spectrum.data, spectrum.data_imag)
    h, n = spectrum.axes[1].ppm, spectrum.axes[0].ppm

    def f1_of_peak(lo, hi):
        columns = (h > lo) & (h < hi)
        i, _j = np.unravel_index(np.argmax(magnitude[:, columns]), magnitude[:, columns].shape)
        return n[i]

    nh2 = f1_of_peak(7.8, 9.0)          # thioamide NH2
    acid = f1_of_peak(16.5, 18.5)       # COOH...N(pyridine)
    # The H-bonded pyridine N sits ~430 ppm above the NH2 in 14N --
    # independent of the 14N reference (see io.jeol._F1_SIGN_FLIP).
    assert 300 < acid - nh2 < 550
