"""Leo's two PFG readers wired into io.load, the notebook helpers and
the GUI (TODO.md "What's next" section 0).

Real datasets for both readers live in the gitignored test_data/, so
these tests build synthetic ones in the documented on-disk formats and
check the results against values computed here by hand -- Stejskal-
Tanner written out, not via processing.diffusion -- so a convention
error in the library cannot also hide in the expected answer.
"""
import os

import numpy as np
import pytest

from opennmr import io
from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.io import spinsolve
from opennmr.notebook import diffusion as nb_diffusion

GAMMA_1H_HZ_PER_T = 42.577478518e6   # CODATA 2018
#: core.registry's 1H gamma is 2.675221e8 rad/s/T cut to 7 significant
#: figures (42.5774646 MHz/T), 3.3e-7 below CODATA -- so b agrees with
#: the hand formula to ~7e-7, not to machine precision. A convention
#: error (bipolar tau, p30 not doubled, G/cm vs T/m) is >= 1e-2 off.
B_RTOL = 2e-6


def _stejskal_tanner(g, delta, big_delta, tau=0.0, bipolar=False):
    """b = (2 pi gamma g delta)^2 (Delta - delta/3 [- tau/2]), s/m^2."""
    correction = big_delta - delta / 3.0 - (tau / 2.0 if bipolar else 0.0)
    return (2 * np.pi * GAMMA_1H_HZ_PER_T * np.asarray(g) * delta) ** 2 * correction


# ============================================================
# Synthetic Spinsolve PGSTE folder
# ============================================================

SPINSOLVE = dict(b1=43.5, npts=512, dwell_us=200.0, zero_fill=2, nsteps=8,
                 centre_ppm=4.7, peak_ppm=2.0, T2=0.1, D=1.5e-9,
                 delta=3.1e-3, big_delta=50e-3)


def _write_spinsolve_pgste(folder, with_dosy_header=True, diffusion=True, p=SPINSOLVE):
    os.makedirs(folder, exist_ok=True)
    sw = 1e6 / p["dwell_us"]
    lowest = p["centre_ppm"] * p["b1"] - sw / 2
    g = np.linspace(0.01, 0.4, p["nsteps"])                    # T/m
    b = _stejskal_tanner(g, p["delta"], p["big_delta"])
    t = np.arange(p["npts"]) * p["dwell_us"] * 1e-6
    # The format's FT is fftshift(fft(conj(fid))) onto an ascending axis
    # starting at lowestFrequency, so a peak at offset f from that
    # axis's centre is exp(-i 2 pi f t) in the stored FID.
    offset = p["peak_ppm"] * p["b1"] - lowest - sw / 2
    fid = (np.exp(-b * p["D"])[:, None]
           * np.exp(-2j * np.pi * offset * t)[None, :] * np.exp(-t / p["T2"])[None, :])

    header = b"SORPATAD1.1V" + np.array([501, p["npts"], p["nsteps"], 1, 1], "<i4").tobytes()
    pairs = np.empty((fid.size, 2), "<f4")
    pairs[:, 0], pairs[:, 1] = fid.real.ravel(), fid.imag.ravel()
    with open(os.path.join(folder, "data.2d"), "wb") as handle:
        handle.write(header + pairs.tobytes())

    acqu = {"b1Freq": p["b1"], "dwellTime": p["dwell_us"], "lowestFrequency": lowest,
            "nrPnts": p["npts"], "nrSteps": p["nsteps"], "zeroFill": p["zero_fill"],
            "rxChannel": '"1H"', "nrScans": 4}
    if diffusion:
        acqu.update(littleDelta=3.0, bigDelta=50.0, gradRamp=0.1)
    with open(os.path.join(folder, "acqu.par"), "w") as handle:
        handle.writelines(f"{k} = {v}\n" for k, v in acqu.items())
    with open(os.path.join(folder, "display.par"), "w") as handle:
        handle.write('SNR = [{"ApodizationType":"Exponential","Value":1.0,"IsApplied":true}]\n'
                     "Phase0 = 0\n")
    # Written in a comma-decimal locale, as the instrument does.
    with open(os.path.join(folder, "delayTimes.txt"), "w") as handle:
        handle.writelines(f"{value:.6E}".replace(".", ",") + "\n" for value in b)
    if with_dosy_header:
        with open(os.path.join(folder, "data.2d.txt"), "w") as handle:
            handle.write(f"#Diffusion Encoding Time (double ; s) {p['delta']}\n")
            handle.write(f"#Diffusion Delay (double ; s) {p['big_delta']}\n")
            handle.write(f"#Dosygamma (double ; rad/s/T) {2 * np.pi * GAMMA_1H_HZ_PER_T}\n")
            handle.write(f"#Gradient Amplitude [{g.size}] (double ; T/m)\n")
            handle.writelines(f"{value}\n" for value in g)
    return b, g


@pytest.fixture
def pgste(tmp_path):
    folder = str(tmp_path / "PGSTE_1")
    b, g = _write_spinsolve_pgste(folder)
    return folder, b, g


def test_spinsolve_diffusion_folder_is_detected(pgste):
    folder, _b, _g = pgste
    assert io.detect_vendor(folder) == "spinsolve"


def test_spinsolve_t1_t2_folder_is_not_claimed_as_diffusion(tmp_path):
    # Same file set, but delayTimes.txt would hold delays: opening it as
    # diffusion would fit delays as b-values. Better unrecognised.
    folder = str(tmp_path / "T1_1")
    _write_spinsolve_pgste(folder, with_dosy_header=False, diffusion=False)
    assert spinsolve.is_spinsolve_pfg(folder)          # Leo's broader detector
    assert not spinsolve.is_spinsolve_diffusion(folder)
    with pytest.raises(ValueError):
        io.detect_vendor(folder)


def test_spinsolve_without_dosy_header_still_detected_from_acqu_par(tmp_path):
    folder = str(tmp_path / "PGSTE_2")
    _write_spinsolve_pgste(folder, with_dosy_header=False)
    assert io.detect_vendor(folder) == "spinsolve"


def test_spinsolve_loads_as_a_pseudo_2d_with_the_peak_on_the_right_side(pgste):
    folder, _b, _g = pgste
    spectrum = io.load(folder)
    p = SPINSOLVE
    assert spectrum.metadata.vendor == "spinsolve"
    assert spectrum.metadata.pseudo_2d
    assert spectrum.data.shape == (p["nsteps"], p["npts"] * p["zero_fill"])
    assert spectrum.data_imag is not None
    ppm = spectrum.axes[1].ppm
    assert ppm[0] > ppm[-1]                         # descending, like every vendor
    peak = ppm[int(np.argmax(spectrum.data[0]))]
    resolution = abs(ppm[1] - ppm[0])
    # Without the format's conjugate the peak lands at the mirror image
    # about the centre, 2*4.7 - 2.0 = 7.4 ppm -- and still looks fine.
    assert peak == pytest.approx(p["peak_ppm"], abs=2 * resolution)
    # Not stored: the GUI's generic FID path would reprocess it
    # without the conjugate (see read_spectrum's docstring).
    for key in ("fid", "fid_raw", "fid_dt"):
        assert key not in spectrum.metadata.raw


def test_spinsolve_b_values_survive_the_comma_locale(pgste):
    folder, b, _g = pgste
    spectrum = io.load(folder)
    assert np.allclose(spectrum.metadata.raw["b_values"], b, rtol=1e-6)
    assert "delayTimes.txt" in spectrum.metadata.raw["b_source"]


def test_spinsolve_peek_agrees_with_load(pgste):
    folder, _b, _g = pgste
    assert io.peek(folder)["shape"] == io.load(folder).data.shape


def test_spinsolve_intensities_decay_with_the_true_D(pgste):
    folder, b, _g = pgste
    spectrum = io.load(folder)
    heights = spectrum.data.max(axis=1)
    fitted_D = -np.polyfit(b, np.log(heights), 1)[0]
    assert fitted_D == pytest.approx(SPINSOLVE["D"], rel=1e-3)


def test_notebook_helpers_read_spinsolve(pgste):
    folder, b, _g = pgste
    spectrum = io.load(folder)
    found = nb_diffusion.recorded_b_values(spectrum)
    assert np.allclose(found["b"], b, rtol=1e-6) and found["warnings"] == []
    params = nb_diffusion.recorded_pfg_params(spectrum)
    assert params["delta"] == pytest.approx(SPINSOLVE["delta"])
    assert params["DELTA"] == pytest.approx(SPINSOLVE["big_delta"])
    assert params["delta_is_plateau_only"] is False
    assert "data.2d.txt" in params["sources"]["delta"]


# ============================================================
# Synthetic Bruker dosy-AU experiment (no diff.xml)
# ============================================================

_PP_BIPOLAR = """;ledbpgp2s
  p1 ph1
  p30:gp6*diff
  d16
  p2 ph1
  p30:gp6*-1*diff
  d16
;p1  : f1 channel -  90 degree high power pulse
;p30: gradient pulse (little DELTA * 0.5)
;d16: delay for gradient recovery
;d20: diffusion time (big DELTA)
"""

BRUKER = dict(p30_us=1100.0, d20=0.1, d16=2e-4)


def _jcamp_array(name, values):
    body = " ".join(f"{v:g}" for v in values)
    return f"##${name}= (0..{len(values) - 1})\n{body}\n"


def _write_bruker_dosy(folder, difflist=True, pp=_PP_BIPOLAR):
    os.makedirs(folder, exist_ok=True)
    P = [0.0] * 64
    P[1], P[30] = 10.0, BRUKER["p30_us"]
    D = [0.0] * 64
    D[16], D[20] = BRUKER["d16"], BRUKER["d20"]
    gpz = [0.0] * 32
    gpz[6] = 100.0
    with open(os.path.join(folder, "acqus"), "w") as handle:
        handle.write("##TITLE= Parameter file\n##$PULPROG= <ledbpgp2s>\n##$NUC1= <1H>\n")
        handle.write(_jcamp_array("P", P) + _jcamp_array("D", D) + _jcamp_array("GPZ", gpz))
        handle.write("##END=\n")
    if pp is not None:
        with open(os.path.join(folder, "pulseprogram"), "w") as handle:
            handle.write(pp)
    g_per_cm = np.linspace(1.0, 50.0, 8)
    if difflist:
        with open(os.path.join(folder, "difflist"), "w") as handle:
            handle.writelines(f"{v:.6f}\n" for v in g_per_cm)
    return g_per_cm


def _bruker_stub(folder, n=8):
    ppm = np.linspace(10.0, 0.0, 64)
    return NMRSpectrum(
        data=np.ones((n, ppm.size)),
        axes=[NMRAxis(kind="index", index=np.arange(1, n + 1)),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * 400.0)],
        metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, pseudo_2d=True,
                             source_path=folder),
    )


def _expected_bruker_b(g_per_cm):
    delta = 2 * BRUKER["p30_us"] * 1e-6          # bipolar: p30 is half of delta
    return _stejskal_tanner(np.asarray(g_per_cm) / 100.0, delta, BRUKER["d20"],
                            tau=BRUKER["d16"], bipolar=True)


def test_bruker_without_diff_xml_gets_b_from_difflist(tmp_path):
    folder = str(tmp_path / "10")
    g = _write_bruker_dosy(folder)
    found = nb_diffusion.recorded_b_values(_bruker_stub(folder))
    assert found["b"] == pytest.approx(_expected_bruker_b(g), rel=B_RTOL)
    # Says where it came from, including the doubling of p30.
    assert "difflist" in found["source"] and "x 2" in found["source"]


def test_bruker_pfg_params_double_p30_for_a_bipolar_sequence(tmp_path):
    folder = str(tmp_path / "10")
    _write_bruker_dosy(folder)
    params = nb_diffusion.recorded_pfg_params(_bruker_stub(folder))
    assert params["delta"] == pytest.approx(2 * BRUKER["p30_us"] * 1e-6)
    assert params["DELTA"] == pytest.approx(BRUKER["d20"])
    assert params["tau"] == pytest.approx(BRUKER["d16"])
    assert params["sequence"] == "bipolar"
    assert params["shape"] == "rect"               # difflist already includes the shape factor
    assert "P30" in params["sources"]["delta"]


def test_bruker_without_difflist_reports_why_instead_of_guessing(tmp_path):
    folder = str(tmp_path / "10")
    _write_bruker_dosy(folder, difflist=False)
    found = nb_diffusion.recorded_b_values(_bruker_stub(folder))
    assert found["b"] is None
    assert any("difflist" in w for w in found["warnings"])


def test_non_pfg_vendor_returns_none():
    spectrum = _bruker_stub("/nonexistent/path")
    spectrum.metadata.vendor = "jeol"
    assert nb_diffusion.recorded_b_values(spectrum) is None


# ============================================================
# The GUI
# ============================================================

def _add_to_session(window, spectrum):
    """As test_gui's _load_spectrum_directly: a synthetic spectrum into
    the session and selected, bypassing file I/O."""
    window.session.spectra.append(spectrum)
    window.session._load_info.append({"path": spectrum.metadata.source_path,
                                      "vendor": spectrum.metadata.vendor, "kwargs": {}})
    window.spectrum_list.addItem(window.session.label(len(window.session) - 1))
    window._select_single_row(len(window.session) - 1)


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_gui_opens_spinsolve_and_fits_the_true_D(qapp, pgste):
    from opennmr.gui.main_window import MainWindow
    folder, b, _g = pgste
    window = MainWindow()
    window._load(folder, vendor="spinsolve")
    assert window.session.spectra[0].metadata.vendor == "spinsolve"
    # A Spinsolve PGSTE series is diffusion-weighted: default to it.
    assert window._is_diffusion_fit()

    window.delay_source_combo.setCurrentIndex(0)   # Auto
    resolved = window._resolve_independent_values(window.session.spectra[0], is_diffusion=True)
    assert np.allclose(resolved, b, rtol=1e-6)
    assert window._values_note is None
    assert "delayTimes.txt" in window._values_provenance

    fwhm_ppm = (1 / (np.pi * SPINSOLVE["T2"]) + 1.0) / SPINSOLVE["b1"]
    window._add_peak_row(shift=SPINSOLVE["peak_ppm"], fwhm=fwhm_ppm, eta=0.5)
    window._on_fit_curve_button_clicked()
    assert window._relaxation_fit_results
    summary = window.curve_fit_summary_label.text()
    assert "from: delayTimes.txt" in summary
    result = window._relaxation_fit_results[0]
    fitted_D = float(dict(zip(result["param_names"], result["popt"]))["D"])
    assert fitted_D == pytest.approx(SPINSOLVE["D"], rel=0.02)


def test_gui_auto_values_use_difflist_for_a_bruker_dosy_experiment(qapp, tmp_path):
    from opennmr.gui.main_window import MainWindow
    folder = str(tmp_path / "10")
    g = _write_bruker_dosy(folder)
    window = MainWindow()
    spectrum = _bruker_stub(folder)
    window.fit_type_combo.setCurrentIndex(1)
    window.delay_source_combo.setCurrentIndex(0)
    resolved = window._resolve_independent_values(spectrum, is_diffusion=True)
    assert resolved == pytest.approx(_expected_bruker_b(g), rel=B_RTOL)
    assert "difflist" in window._values_provenance


def test_gui_read_from_file_fills_every_recorded_timing_with_sources(qapp, tmp_path):
    from opennmr.gui.main_window import MainWindow
    folder = str(tmp_path / "10")
    _write_bruker_dosy(folder)
    window = MainWindow()
    spectrum = _bruker_stub(folder)
    _add_to_session(window, spectrum)
    window.pfg_delta_spin.setValue(0.5)
    window._on_read_pfg_params_clicked()
    assert window.pfg_delta_spin.value() == pytest.approx(2 * BRUKER["p30_us"] / 1000.0)
    assert window.pfg_big_delta_spin.value() == pytest.approx(BRUKER["d20"] * 1000.0)
    assert window.pfg_tau_spin.value() == pytest.approx(BRUKER["d16"] * 1000.0)
    assert window.pfg_sequence_combo.currentData() == "bipolar"
    record = window.pfg_params_source_label.text()
    assert "P30" in record and "D20" in record


def test_gui_read_from_file_explains_when_nothing_is_recorded(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from opennmr.gui.main_window import MainWindow
    folder = str(tmp_path / "10")
    _write_bruker_dosy(folder, pp=None)            # no pulse programme to read δ/Δ from
    window = MainWindow()
    spectrum = _bruker_stub(folder)
    _add_to_session(window, spectrum)
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    window._on_read_pfg_params_clicked()
    assert shown and "diff.xml instead" not in shown[0]
