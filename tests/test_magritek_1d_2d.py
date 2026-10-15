"""Magritek Spinsolve plain 1D and 2D data (explicit user request,
2026-10-15: "enable Magritek data to be read ... do not use nmrglue").

Two kinds of check, so a reader that is merely plausible cannot pass:

* **Against Spinsolve itself** (``test_data/Magritek_COSY``, gitignored,
  skipped without it): Spinsolve Expert writes its own processed
  spectra beside the raw FIDs (``PilotScan/spectrum.pt1``,
  ``cosySpectrum.pt2``). The reader, working only from ``data.1d`` /
  ``data.2d``, must reproduce them -- the 1D to float32 rounding, the
  2D magnitude COSY to correlation 0.9999 -- and the COSY's diagonal
  must lie on F1 = F2.
* **Synthetic folders** in the same on-disk format, with lines at known
  shifts, for CI: right ppm, right side, phase and first-point
  correction as recorded, diagonal peaks on the diagonal, and the GUI's
  live rebuild agreeing with the read.
"""

import os

import numpy as np
import pytest

from opennmr import io
from opennmr.io import base, spinsolve

from test_gui import qapp  # noqa: F401 -- qapp is a fixture

REAL = os.path.join(os.path.dirname(__file__), "..", "test_data", "Magritek_COSY")
needs_real = pytest.mark.skipif(not os.path.isdir(REAL),
                                reason="test_data/Magritek_COSY not present")


# ---- synthetic Spinsolve folders ------------------------------------------------------

B1, CENTRE, DWELL_US = 80.1244625, 5.0, 200.0


def _write_prospa(path, data, data_type, x=None):
    data = np.atleast_2d(data) if data.ndim == 2 else data
    ny, nx = (data.shape if data.ndim == 2 else (1, data.size))
    header = b"SORPATAD1.1V" + np.array([data_type, nx, ny, 1, 1], "<i4").tobytes()
    pairs = np.empty((data.size, 2), "<f4")
    pairs[:, 0], pairs[:, 1] = data.real.ravel(), data.imag.ravel()
    body = (np.asarray(x, "<f4").tobytes() if x is not None else b"") + pairs.tobytes()
    with open(path, "wb") as handle:
        handle.write(header + body)


def _write_par(path, values):
    with open(path, "w") as handle:
        for key, value in values:
            handle.write(f"{key} = {value}\n")


def _acqu(npts, **extra):
    base_values = {"b1Freq": B1, "centerFreqPPM": CENTRE, "dwellTime": DWELL_US,
                   "nrPnts": npts, "nrScans": 4, "rxChannel": '"1H"', "filter": '"no"',
                   "filterType": '"exp:0.5"', "experiment": '"Proton"', "zf": 1}
    base_values.update(extra)
    return list(base_values.items())


def _line(ppm, t, t2=0.05):
    """A line at ``ppm`` as the stored FID holds it (exp(-2 pi i f t))."""
    offset = (ppm - CENTRE) * B1
    return np.exp(-2j * np.pi * offset * t) * np.exp(-t / t2)


def make_1d(folder, peaks=(1.2, 3.6), p0=25.0, npts=2048, filter_on=False):
    os.makedirs(folder, exist_ok=True)
    t = np.arange(npts) * DWELL_US * 1e-6
    fid = sum(_line(ppm, t) for ppm in peaks) * np.exp(-1j * np.deg2rad(p0))
    _write_prospa(os.path.join(folder, "data.1d"), fid, 504, x=t * 1e3)
    _write_par(os.path.join(folder, "acqu.par"),
               _acqu(npts, filter='"yes"' if filter_on else '"no"'))
    _write_par(os.path.join(folder, "proc.par"), [
        ("apodizationFunction", '"exp:2"'), ("p0Phase", p0), ("p1Phase", 0),
        ("phaseMethod", '"p0, p1 fixed phase"'), ("zeroFill", 1)])
    return folder


def make_cosy(folder, peaks=(1.5, 3.5), n1=64, n2=1024, sw1_khz=0.8012446, f1_type="Complex"):
    os.makedirs(folder, exist_ok=True)
    t2 = np.arange(n2) * DWELL_US * 1e-6
    t1 = np.arange(n1) / (sw1_khz * 1e3)
    data = np.zeros((n1, n2), complex)
    for ppm in peaks:
        offset = (ppm - CENTRE) * B1
        # F1 is stored with the opposite sense (Spinsolve inverts it).
        data += (np.exp(2j * np.pi * offset * t1) * np.exp(-t1 / 0.05))[:, None] * _line(ppm, t2)[None, :]
    _write_prospa(os.path.join(folder, "data.2d"), data, 501)
    _write_par(os.path.join(folder, "acqu.par"), _acqu(
        n2, nrSteps=n1, bandwidth2=sw1_khz, experiment='"COSY"', nucleus='"1H-1H"',
        rxChannel2='"1H"'))
    _write_par(os.path.join(folder, "proc.par"), [
        ("f1FTInvert", '"yes"'), ("f1FTType", f'"{f1_type}"'), ("f1PhaseMethod", '"Magnitude"'),
        ("f1PPMOffset", CENTRE), ("f1WindowFunction", '"sinebellsquared"'),
        ("f1WindowFunction", '"sinebellsquared"'), ("f1ZeroFill", 1),
        ("f2FTInvert", '"no"'), ("f2PhaseMethod", '"None"'), ("f2ZeroFill", 1)])
    return folder


def test_detection_and_peek(tmp_path):
    one = make_1d(str(tmp_path / "Proton"))
    two = make_cosy(str(tmp_path / "COSY"))
    assert base.detect_vendor(one) == base.detect_vendor(two) == "spinsolve"
    assert base.peek(one)["ndim"] == 1 and base.peek(two)["ndim"] == 2
    assert base.peek(two)["shape"] == (64, 1024)
    # A T1/T2 array (delayTimes.txt, no diffusion keys) is still left alone.
    open(os.path.join(two, "delayTimes.txt"), "w").write("1\n")
    assert not spinsolve.is_spinsolve_standard(two)


def test_1d_line_positions_phase_and_first_point(tmp_path):
    spec = io.load(make_1d(str(tmp_path / "Proton")))
    ppm, y = spec.axes[0].ppm, spec.data
    assert ppm[0] > ppm[-1]                                     # descending
    found = sorted(ppm[i] for i in np.argsort(y)[-2:])
    assert found == pytest.approx([1.2, 3.6], abs=0.01)         # right side of the axis
    # p0 applied with the right sense. The whole spectrum sums to N x
    # the first FID point, which was written as 2 exp(-i p0): real and
    # positive once p0 is undone (a wrong sign leaves -50 deg).
    total = (spec.data + 1j * spec.data_imag).sum()
    assert total.real > 0 and abs(np.angle(total, deg=True)) < 0.01
    raw = spec.metadata.raw
    assert raw["recorded_phase0"] == 25.0 and raw["phase0"] == 0.0
    assert raw["live_window"] == "none"                         # filter = "no"
    assert raw["fid_raw"][0] == pytest.approx(0.5 * 2 * np.exp(-1j * np.deg2rad(25.0)), rel=1e-5)
    assert spec.metadata.sfo1 == pytest.approx(B1) and spec.metadata.ns == 4


def test_1d_filter_on_uses_the_recorded_window(tmp_path):
    spec = io.load(make_1d(str(tmp_path / "Proton"), filter_on=True))
    assert spec.metadata.raw["live_window"] == "exponential"
    assert spec.metadata.raw["live_lb"] == 2.0


def test_cosy_peaks_on_the_diagonal(tmp_path):
    spec = io.load(make_cosy(str(tmp_path / "COSY")))
    assert spec.ndim == 2 and not spec.is_pseudo_2d
    assert spec.data_imag is None and spec.data.min() >= 0       # magnitude
    ppm1, ppm2 = spec.axes[0].ppm, spec.axes[1].ppm
    for shift in (1.5, 3.5):
        column = int(np.argmin(np.abs(ppm2 - shift)))
        f1_peak = ppm1[int(np.argmax(spec.data[:, column]))]
        assert f1_peak == pytest.approx(shift, abs=abs(ppm1[1] - ppm1[0]))
    raw = spec.metadata.raw
    assert raw["live_window"] == raw["live_window_f1"] == "qsine"
    assert raw["live_ssb"] == raw["live_ssb_f1"] == 0.0
    assert raw["fn_mode"] == 1 and raw["magnitude_mode"] == "magnitude"


def test_an_unchecked_f1_transform_is_refused(tmp_path):
    folder = make_cosy(str(tmp_path / "HSQC"), f1_type="States")
    with pytest.raises(ValueError, match="States"):
        io.load(folder)


def test_window_names():
    assert spinsolve._spinsolve_window("exp:0.5")[:2] == ("exponential", {"lb": 0.5})
    assert spinsolve._spinsolve_window("sinebellsquared")[:2] == ("qsine", {"ssb": 0.0})
    key, _kwargs, note = spinsolve._spinsolve_window("kaiser")
    assert key == "none" and "not recognised" in note


# ---- the GUI rebuilds what the reader read ---------------------------------------------

def test_gui_opens_both_and_a_live_rebuild_matches(qapp, tmp_path):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    one = make_1d(str(tmp_path / "Proton"))
    two = make_cosy(str(tmp_path / "COSY"))
    window._load(one, vendor="spinsolve")
    window._load(two, vendor="spinsolve")
    assert [s.ndim for s in window.session.spectra] == [1, 2]
    cosy = window.session.spectra[1]
    window._select_single_row(1)
    assert window.window_combo.currentData() == "qsine"
    assert window.window_f1_combo.currentData() == "qsine"
    assert window.ssb_spin.value() == 0.0
    before = cosy.data.copy()
    assert window._recompute_multi_row_data(cosy)
    np.testing.assert_allclose(cosy.data, before, atol=1e-9 * before.max())
    np.testing.assert_allclose(cosy.axes[0].ppm, io.load(two).axes[0].ppm)
    window._select_single_row(0)
    proton = window.session.spectra[0]
    assert window.window_combo.currentData() == "none"
    preview = window._compute_processing_preview(proton)
    assert preview is not None
    _fid, _dt, rebuilt, _title = preview
    np.testing.assert_allclose(rebuilt.data, proton.data, atol=1e-9 * np.abs(proton.data).max())


# ---- against Spinsolve's own processed spectra -----------------------------------------

def _pt1_spectrum(path, n):
    raw = open(path, "rb").read()
    offset = len(raw) - n * 12 - 8          # x (n float32), then n complex float32, then 8 bytes
    x = np.frombuffer(raw[offset:offset + n * 4], "<f4").astype(float)
    pairs = np.frombuffer(raw[offset + n * 4:offset + n * 12], "<f4").reshape(n, 2)
    return x, pairs[:, 0] + 1j * pairs[:, 1]


@needs_real
def test_real_1d_matches_spinsolves_own_spectrum():
    spec = io.load(os.path.join(REAL, "PilotScan"))
    n = spec.data.size
    x, ref = _pt1_spectrum(os.path.join(REAL, "PilotScan", "spectrum.pt1"), n)
    ours = spec.data + 1j * spec.data_imag
    # Same points, opposite order and one grid point apart (Spinsolve's
    # grid starts one bin lower); compare at equal ppm.
    ours_on_ref = np.roll(ours[::-1], 1)
    np.testing.assert_allclose(x[1:], np.roll(spec.axes[0].ppm[::-1], 1)[1:], atol=1e-4)
    scale = np.vdot(ours_on_ref, ref) / np.vdot(ours_on_ref, ours_on_ref)
    assert abs(np.angle(scale, deg=True)) < 0.01                 # p0 exactly Spinsolve's
    residual = np.linalg.norm(ref - scale * ours_on_ref) / np.linalg.norm(ref)
    assert residual < 1e-5
    assert spec.axes[0].ppm[int(np.argmax(spec.data))] == pytest.approx(4.7393, abs=0.002)


@needs_real
def test_real_cosy_matches_spinsolve_and_its_diagonal():
    spec = io.load(REAL)
    ref = np.frombuffer(open(os.path.join(REAL, "cosySpectrum.pt2"), "rb").read()[3386:],
                        "<f4").reshape(128, 8192).astype(float)
    # Spinsolve stores both axes ascending, each one grid point from
    # ours (F1: its rows start one point lower than the reversed grid's
    # true frequencies -- the diagonal below shows ours is the right one).
    ours = np.roll(np.roll(spec.data[::-1, ::-1], -1, axis=0), 1, axis=1)
    assert np.corrcoef(ours.ravel(), ref.ravel())[0, 1] > 0.9999
    ppm1, ppm2 = spec.axes[0].ppm, spec.axes[1].ppm
    bin1 = abs(ppm1[1] - ppm1[0])
    for shift in (1.153, 4.741):                                 # isopropanol CH3, water
        column = int(np.argmin(np.abs(ppm2 - shift)))
        assert ppm1[int(np.argmax(spec.data[:, column]))] == pytest.approx(shift, abs=bin1)
    assert spec.metadata.raw["live_window"] == spec.metadata.raw["live_window_f1"] == "qsine"


def test_open_tab_scan_lists_a_2d_and_the_1d_inside_it(qapp, tmp_path):
    """Spinsolve Expert keeps a 2D's 1D pilot scan inside the 2D's folder."""
    from opennmr.gui.main_window import MainWindow
    cosy = make_cosy(str(tmp_path / "data" / "COSY"))
    make_1d(os.path.join(cosy, "PilotScan"))
    found = MainWindow._scan_for_spectra(str(tmp_path / "data"))
    assert [(os.path.relpath(p, tmp_path), v) for p, v in found] == [
        (os.path.join("data", "COSY"), "spinsolve"),
        (os.path.join("data", "COSY", "PilotScan"), "spinsolve")]
