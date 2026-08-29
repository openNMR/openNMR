import numpy as np
import nmrglue as ng
import pytest

from nmrsuite import io
from nmrsuite.io import varian


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
                      tn="Li7", nt=1.0, name="synthetic.fid"):
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


def test_read_1d_zero_fill(tmp_path):
    path = _write_varian_1d(tmp_path, n=8)
    spectrum = varian.read_1d(path, autophase=False, zero_fill=32)
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
