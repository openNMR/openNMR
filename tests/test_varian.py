import os

import numpy as np
import nmrglue as ng
import pytest

from opennmr import io
from opennmr.io import varian


# ============================================================
# Synthetic Varian fixture writer
#
# Varian's binary FID format (block headers, hypercomplex-2D
# interleaving) is intricate enough that hand-rolling a fixture byte
# layout directly risks getting the very thing this module leans on
# nmrglue to avoid getting wrong -- instead, build a `dic`/`data` pair
# nmrglue's own `ng.varian.write` accepts, then read it back with the
# same `ng.varian.read` this module itself uses, so tests exercise the
# real binary round-trip nmrglue guarantees rather than a fixture that
# might not resemble real Varian output.
# ============================================================

def _procpar_entry(name, values, basictype="1"):
    return {
        "name": name, "subtype": "1", "basictype": basictype, "maxvalue": "0", "minvalue": "0",
        "stepsize": "0", "Ggroup": "0", "Dgroup": "0", "protection": "0", "active": "1",
        "intptr": "0", "values": [str(v) for v in values], "enumerable": "0", "enumerables": [],
    }


def _write_varian_1d(tmp_path, n=8, sw=1000.0, sfrq=100.0, reffrq=100.0, tof=0.0,
                      tn="Li7", nt=1.0, name="synthetic.fid", probe=None, seqfil=None):
    data = (np.arange(1, n + 1) + 1j * np.arange(1, n + 1)).astype(np.complex64)
    dic = {
        "nblocks": 1, "ntraces": 1, "np": 2 * n, "ebytes": 4, "tbytes": 4 * 2 * n, "bbytes": 4 * 2 * n + 28,
        "vers_id": 0, "status": 201, "nbheaders": 1,
        "S_DATA": 1, "S_SPEC": 0, "S_32": 0, "S_FLOAT": 1, "S_COMPLEX": 0, "S_HYPERCOMPLEX": 0,
        "S_ACQPAR": 1, "S_SECND": 0, "S_TRANSF": 0, "S_NP": 0, "S_NF": 0, "S_NI": 0, "S_NI2": 0,
        "procpar": {
            "sw": _procpar_entry("sw", [sw]),
            "sfrq": _procpar_entry("sfrq", [sfrq]),
            "reffrq": _procpar_entry("reffrq", [reffrq]),
            "tof": _procpar_entry("tof", [tof]),
            "tn": _procpar_entry("tn", [tn], basictype="2"),
            "nt": _procpar_entry("nt", [nt]),
            "np": _procpar_entry("np", [2 * n]),
            "ni": _procpar_entry("ni", [1]),
        },
    }
    if probe is not None:
        dic["procpar"]["probe"] = _procpar_entry("probe", [probe], basictype="2")
    if seqfil is not None:
        dic["procpar"]["seqfil"] = _procpar_entry("seqfil", [seqfil], basictype="2")
    path = str(tmp_path / name)
    ng.varian.write(path, dic, data, overwrite=True)
    return path


def _write_varian_2d(tmp_path, ni=3, n_direct=4, sw=1000.0, sfrq=100.0, reffrq=100.0, tof=0.0,
                      tn="H1", nt=1.0, name="synthetic2d.fid"):
    n_traces = 2 * ni
    data = np.zeros((n_traces, n_direct), dtype=np.complex64)
    for i in range(n_traces):
        # Real/imaginary phase pair per t1 increment differ by a
        # constant offset, so a test can tell them apart after
        # read_2d's own real+1j*imag reassembly.
        data[i] = (i + 1) + 1j * np.arange(n_direct)

    dic = {
        "nblocks": n_traces, "ntraces": 1, "np": 2 * n_direct, "ebytes": 4,
        "tbytes": 4 * 2 * n_direct, "bbytes": 4 * 2 * n_direct + 28,
        "vers_id": 0, "status": 201, "nbheaders": 1,
        "S_DATA": 1, "S_SPEC": 0, "S_32": 0, "S_FLOAT": 1, "S_COMPLEX": 0, "S_HYPERCOMPLEX": 0,
        "S_ACQPAR": 1, "S_SECND": 0, "S_TRANSF": 0, "S_NP": 0, "S_NF": 0, "S_NI": 0, "S_NI2": 0,
        "procpar": {
            "sw": _procpar_entry("sw", [sw]),
            "sfrq": _procpar_entry("sfrq", [sfrq]),
            "reffrq": _procpar_entry("reffrq", [reffrq]),
            "tof": _procpar_entry("tof", [tof]),
            "tn": _procpar_entry("tn", [tn], basictype="2"),
            "nt": _procpar_entry("nt", [nt]),
            "np": _procpar_entry("np", [2 * n_direct]),
            "ni": _procpar_entry("ni", [ni]),
        },
    }
    path = str(tmp_path / name)
    ng.varian.write(path, dic, data, overwrite=True)
    return path


# ============================================================
# Pure-Python helpers
# ============================================================

def test_normalize_nucleus_swaps_symbol_and_number():
    assert varian._normalize_nucleus("Li7") == "7Li"
    assert varian._normalize_nucleus("H1") == "1H"
    assert varian._normalize_nucleus("C13") == "13C"


def test_normalize_nucleus_passes_through_unrecognised_format():
    assert varian._normalize_nucleus("already7Li") == "already7Li"


def test_normalize_nucleus_none_for_missing():
    assert varian._normalize_nucleus(None) is None
    assert varian._normalize_nucleus("") is None


def test_procpar_value_casts_and_defaults():
    procpar = {"sw": _procpar_entry("sw", [12345.0])}
    assert varian._procpar_value(procpar, "sw") == pytest.approx(12345.0)
    assert varian._procpar_value(procpar, "missing", default=42.0) == 42.0


# ============================================================
# read_1d
# ============================================================

def test_read_1d_produces_nmrspectrum(tmp_path):
    path = _write_varian_1d(tmp_path, n=8)
    spectrum = varian.read_1d(path, autophase=False)

    assert spectrum.ndim == 1
    assert len(spectrum.data) == 8
    assert spectrum.metadata.vendor == "varian"
    assert spectrum.metadata.nucleus == "7Li"  # normalized from procpar's "Li7"
    assert spectrum.metadata.sfo1 == pytest.approx(100.0)
    assert len(spectrum.axes[0].ppm) == 8


def test_read_1d_records_probe(tmp_path):
    path = _write_varian_1d(tmp_path, n=8, probe="3.2mm HXY probe")
    spectrum = varian.read_1d(path, autophase=False)
    assert spectrum.metadata.probe == "3.2mm HXY probe"


def test_read_1d_probe_is_none_when_not_recorded(tmp_path):
    path = _write_varian_1d(tmp_path, n=8)
    spectrum = varian.read_1d(path, autophase=False)
    assert spectrum.metadata.probe is None


def test_read_1d_records_pulse_program_from_seqfil(tmp_path):
    path = _write_varian_1d(tmp_path, n=8, seqfil="onepul")
    spectrum = varian.read_1d(path, autophase=False)
    assert spectrum.metadata.pulse_program == "onepul"


def test_read_1d_pulse_program_is_none_when_not_recorded(tmp_path):
    path = _write_varian_1d(tmp_path, n=8)
    spectrum = varian.read_1d(path, autophase=False)
    assert spectrum.metadata.pulse_program is None


def test_read_1d_zero_fill(tmp_path):
    path = _write_varian_1d(tmp_path, n=8)
    spectrum = varian.read_1d(path, autophase=False, zero_fill=32)
    assert len(spectrum.data) == 32


def test_read_1d_caches_raw_fid(tmp_path):
    # metadata.raw["fid_raw"] -- the GUI's Processing-preview live view
    # needs this cached (see _compute_processing_preview); zero-filled
    # here specifically so the raw cache (pre-zero-fill) is provably
    # distinct in length from the processed "fid"/final data.
    path = _write_varian_1d(tmp_path, n=8)
    spectrum = varian.read_1d(path, autophase=False, zero_fill=32)
    raw_fid = spectrum.metadata.raw["fid_raw"]
    assert len(raw_fid) == 8
    assert len(spectrum.data) == 32


def test_read_1d_divides_by_ns_and_mass(tmp_path):
    path = _write_varian_1d(tmp_path, n=8, nt=4.0)
    baseline = varian.read_1d(path, autophase=False)
    halved = varian.read_1d(path, autophase=False, mass=2.0)
    np.testing.assert_allclose(halved.data, baseline.data / 2.0)


def test_read_1d_fid_shift_changes_output(tmp_path):
    path = _write_varian_1d(tmp_path, n=8)
    baseline = varian.read_1d(path, autophase=False)
    shifted = varian.read_1d(path, autophase=False, fid_shift=1.0)
    assert not np.allclose(baseline.data, shifted.data)


def test_read_1d_raises_without_sw(tmp_path):
    path = _write_varian_1d(tmp_path, n=8, sw=0.0)
    with pytest.raises(ValueError):
        varian.read_1d(path, autophase=False)


# ============================================================
# Shift referencing
#
# VnmrJ centres the spectrum on sfrq and puts 0 ppm at reffrq; sfrq
# already includes tof. Until 2026-10-22 tof was subtracted a second
# time, moving every spectrum by -tof/sfrq ppm.
# ============================================================

#: A real 27Al acquisition's frequencies (EMIM chloroaluminate,
#: test_data/varian_27Al): sfrq, reffrq (MHz), tof, sw (Hz). VnmrJ puts
#: this spectrum's centre at 150.0 ppm.
_AL27_FREQS = dict(sfrq=130.2444175, reffrq=130.22488377, tof=20266.1, sw=65093.5720098)
_AL27_CENTRE_PPM = 150.0


def _expected_ppm(hz, sfrq, reffrq):
    """VnmrJ's axis, written out independently of the reader."""
    return (np.asarray(hz) + (sfrq - reffrq) * 1e6) / reffrq


def test_read_1d_ppm_axis_is_centred_on_sfrq_and_measured_from_reffrq(tmp_path):
    path = _write_varian_1d(tmp_path, n=64, tn="Al27", **_AL27_FREQS)
    spectrum = varian.read_1d(path, autophase=False)
    axis = spectrum.axes[0]

    np.testing.assert_allclose(
        axis.ppm, _expected_ppm(axis.hz, _AL27_FREQS["sfrq"], _AL27_FREQS["reffrq"]), rtol=0, atol=1e-9)
    assert axis.ppm[np.argmin(np.abs(axis.hz))] == pytest.approx(_AL27_CENTRE_PPM, abs=0.01)
    # sfo1 = bf1 + o1, as TopSpin's own vconv conversion stores it.
    assert spectrum.metadata.o1 == pytest.approx((_AL27_FREQS["sfrq"] - _AL27_FREQS["reffrq"]) * 1e6)
    assert spectrum.metadata.bf1 + spectrum.metadata.o1 / 1e6 == pytest.approx(spectrum.metadata.sfo1)


def test_read_1d_ppm_axis_does_not_depend_on_tof(tmp_path):
    """The regression itself: the same sfrq/reffrq must give the same
    axis whatever tof says, because sfrq already includes it."""
    no_offset = varian.read_1d(
        _write_varian_1d(tmp_path, sfrq=100.001, reffrq=100.0, tof=0.0, name="a.fid"), autophase=False)
    offset = varian.read_1d(
        _write_varian_1d(tmp_path, sfrq=100.001, reffrq=100.0, tof=20000.0, name="b.fid"), autophase=False)
    np.testing.assert_allclose(no_offset.axes[0].ppm, offset.axes[0].ppm, rtol=0, atol=1e-12)


def test_read_2d_ppm_axis_matches_the_1d_definition(tmp_path):
    path = _write_varian_2d(tmp_path, ni=3, n_direct=8, tn="Al27", **_AL27_FREQS)
    spectrum = varian.read_2d(path, autophase=False)
    axis = spectrum.axes[1]

    np.testing.assert_allclose(
        axis.ppm, _expected_ppm(axis.hz, _AL27_FREQS["sfrq"], _AL27_FREQS["reffrq"]), rtol=0, atol=1e-9)
    assert axis.ppm[np.argmin(np.abs(axis.hz))] == pytest.approx(_AL27_CENTRE_PPM, abs=0.01)
    assert spectrum.metadata.o1 == pytest.approx((_AL27_FREQS["sfrq"] - _AL27_FREQS["reffrq"]) * 1e6)


_AL27_EMIM = "test_data/varian_27Al/EMIM_Leo_1p1_27Al_071119.fid"


@pytest.mark.skipif(not os.path.isdir(_AL27_EMIM), reason=f"{_AL27_EMIM} not present")
def test_real_27al_peaks_land_at_their_known_shifts():
    """EMIM chloroaluminate ionic liquid, 27Al at 11.7 T, lock-referenced
    in VnmrJ to Al(NO3)3/D2O (``tnref``). Its two peaks are known to sit
    at about 103 and 97 ppm (AlCl4- and Al2Cl7-). With tof (20.3 kHz)
    subtracted twice they came out at -52.6 and -58.5 ppm."""
    spectrum = io.load(_AL27_EMIM, vendor="varian")
    ppm = np.asarray(spectrum.axes[0].ppm)
    magnitude = np.abs(np.asarray(spectrum.data))

    assert ppm[np.argmax(magnitude)] == pytest.approx(103.0, abs=0.3)
    window = (ppm > 95.0) & (ppm < 99.0)
    assert ppm[window][np.argmax(magnitude[window])] == pytest.approx(97.1, abs=0.3)


# ============================================================
# read_2d
# ============================================================

def test_read_2d_produces_pseudo_2d_spectrum(tmp_path):
    path = _write_varian_2d(tmp_path, ni=3, n_direct=4)
    spectrum = varian.read_2d(path, autophase=False)

    assert spectrum.ndim == 2
    assert spectrum.is_pseudo_2d is True
    assert spectrum.data.shape[0] == 3  # ni
    assert spectrum.data.shape[1] == 4  # n_direct (post-FFT, no zero-fill)


def test_read_2d_assembles_states_hypercomplex_pairs(tmp_path):
    # Row i's real phase-1 trace is (i+1)+0j..(i+1)+3j and its
    # phase-2 trace is (i+2)+0j..(i+2)+3j (consecutive traces in the
    # synthetic fixture) -- read_2d's own real+1j*imag combination
    # should produce a complex F1 value with both real parts present,
    # not silently drop one phase.
    path = _write_varian_2d(tmp_path, ni=2, n_direct=2)
    spectrum = varian.read_2d(path, autophase=False)
    assert spectrum.data.shape == (2, 2)
    # Both rows must differ (not collapsed/duplicated) -- a real
    # regression this shape would catch if the reshape ever silently
    # picked the wrong axis order.
    assert not np.allclose(spectrum.data[0], spectrum.data[1])


def test_read_2d_zero_fill(tmp_path):
    path = _write_varian_2d(tmp_path, ni=2, n_direct=4)
    spectrum = varian.read_2d(path, autophase=False, zero_fill=16)
    assert spectrum.data.shape == (2, 16)


def test_read_2d_fid_shift_changes_output(tmp_path):
    path = _write_varian_2d(tmp_path, ni=2, n_direct=4)
    baseline = varian.read_2d(path, autophase=False)
    shifted = varian.read_2d(path, autophase=False, fid_shift=1.0)
    assert not np.allclose(baseline.data, shifted.data)


def test_read_2d_align_echo_runs(tmp_path):
    path = _write_varian_2d(tmp_path, ni=2, n_direct=4)
    spectrum = varian.read_2d(path, autophase=False, align_echo=True)
    assert spectrum.data.shape == (2, 4)


def test_read_2d_linear_predict_extends_points(tmp_path):
    path = _write_varian_2d(tmp_path, ni=2, n_direct=4)
    spectrum = varian.read_2d(path, autophase=False, linear_predict_points=2, linear_predict_order=1)
    assert spectrum.data.shape == (2, 6)


# ============================================================
# Vendor detection / registration
# ============================================================

def test_detect_accepts_varian_directory(tmp_path):
    path = _write_varian_1d(tmp_path)
    assert varian._detect(path) is True


def test_detect_rejects_non_varian_directory(tmp_path):
    empty = tmp_path / "not_varian"
    empty.mkdir()
    assert varian._detect(str(empty)) is False


def test_read_auto_dispatches_1d(tmp_path):
    path = _write_varian_1d(tmp_path, n=8)
    spectrum = varian._read_auto(path, autophase=False)
    assert spectrum.ndim == 1


def test_read_auto_dispatches_2d(tmp_path):
    path = _write_varian_2d(tmp_path, ni=3, n_direct=4)
    spectrum = varian._read_auto(path, autophase=False)
    assert spectrum.ndim == 2


def test_io_load_autodetects_varian(tmp_path):
    path = _write_varian_1d(tmp_path, n=8)
    assert io.detect_vendor(path) == "varian"
    spectrum = io.load(path, autophase=False)
    assert spectrum.metadata.vendor == "varian"


# ============================================================
# peek(): procpar only, never the binary fid
# ============================================================

def test_peek_reports_shape_without_ever_reading_the_fid(tmp_path):
    """The reason this exists: without a registered ``peek``,
    :func:`opennmr.io.base.peek` falls back to a full ``load()`` --
    so the Data-tab browser was reading *and processing* every
    experiment in a folder just to label it. Proven the same way the
    Bruker one is: with the big data file deleted, peek still works.
    """
    import os
    path = _write_varian_1d(tmp_path, n=8, tn="Li7")
    os.remove(os.path.join(path, "fid"))

    assert varian.peek(path) == {
        "vendor": "varian", "ndim": 1, "shape": (8,),
        "nucleus": "7Li", "nucleus_indirect": None, "pseudo_2d": False,
        # peek reports the pulse programme too; these fixtures record none.
        "pulse_program": None,
    }
    # And the fallback would have needed it.
    with pytest.raises(Exception):
        varian.read_1d(path)


@pytest.mark.parametrize("path", [
    "test_data/varian_1d/Li7_8000ns_12s_june.fid",
    "test_data/varian_2d/HLi_CP_2D_8p5kHz_june2024.fid",
])
def test_peek_agrees_with_a_real_load(path):
    """``np`` counts real+imaginary points and ``ni`` is the indirect
    increment count -- this pins that derivation against what the
    readers actually produce, so a browser label cannot drift from the
    spectrum it describes."""
    import os
    if not os.path.isdir(path):
        pytest.skip(f"{path} not present")
    info = varian.peek(path)
    spectrum = io.load(path, vendor="varian")
    assert tuple(info["shape"]) == tuple(spectrum.data.shape)
    assert info["nucleus"] == spectrum.metadata.nucleus
    assert info["ndim"] == spectrum.data.ndim
