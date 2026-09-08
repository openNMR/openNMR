"""Byte-comparability of the apodization windows against nmrglue.

These replaced ``nmrglue.process.proc_base``'s ``em``/``gm``/``sp``.
They are short closed-form expressions, so the bar is exact equality
rather than approximate -- including where the cast to the data's own
dtype happens, which is enough on its own to shift the last bits.
"""
import numpy as np
import pytest

from opennmr.processing import windows

proc_base = pytest.importorskip("nmrglue.process.proc_base")


def _data(n, dtype):
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)


@pytest.mark.parametrize("n", [64, 1000, 4096])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("lb", [0.0, 0.001, 0.05, 0.3])
def test_em_matches_nmrglue_byte_for_byte(n, dtype, lb):
    data = _data(n, dtype)
    ours = windows.em(data.copy(), lb=lb)
    theirs = proc_base.em(data.copy(), lb=lb)
    assert ours.dtype == theirs.dtype
    assert ours.tobytes() == theirs.tobytes()


@pytest.mark.parametrize("n", [64, 1000, 4096])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("power", [1.0, 2.0])          # Bruker SINE and QSINE
@pytest.mark.parametrize("off", [0.0, 0.5, 1 / 3])
def test_sp_matches_nmrglue_byte_for_byte(n, dtype, power, off):
    data = _data(n, dtype)
    ours = windows.sp(data.copy(), off=off, end=1.0, pow=power)
    theirs = proc_base.sp(data.copy(), off=off, end=1.0, pow=power)
    assert ours.dtype == theirs.dtype
    assert ours.tobytes() == theirs.tobytes()


@pytest.mark.parametrize("g1,g2", [(0.0, 0.0), (0.01, 0.0), (0.0, 0.02), (0.01, 0.02)])
def test_gm_matches_nmrglue_byte_for_byte(g1, g2):
    data = _data(1000, np.complex128)
    ours = windows.gm(data.copy(), g1=g1, g2=g2, g3=0.5)
    theirs = proc_base.gm(data.copy(), g1=g1, g2=g2, g3=0.5)
    assert ours.tobytes() == theirs.tobytes()


@pytest.mark.parametrize("rev", [False, True])
@pytest.mark.parametrize("inv", [False, True])
def test_flags_match_including_the_degenerate_inverse(rev, inv):
    # inv=True divides by an apodization that decays to ~0, so it
    # overflows to inf/NaN. Those patterns have to match too -- an
    # implementation that merely avoided the warning would not be a
    # drop-in replacement.
    data = _data(256, np.complex128)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        ours = windows.em(data.copy(), lb=0.05, inv=inv, rev=rev)
        theirs = proc_base.em(data.copy(), lb=0.05, inv=inv, rev=rev)
    assert ours.tobytes() == theirs.tobytes()


def test_windows_are_what_the_bruker_reader_now_calls():
    # The swap is only meaningful if the reader actually routes here.
    from opennmr.io import bruker

    fid = np.ones(64, dtype=complex)
    em_result = bruker._apply_bruker_window(fid, {"WDW": 1, "LB": 10.0}, sw_hz=1000.0)
    np.testing.assert_array_equal(em_result, windows.em(fid, lb=10.0 / 1000.0))

    qsine = bruker._apply_bruker_window(fid, {"WDW": 4, "SSB": 2}, sw_hz=1000.0)
    np.testing.assert_array_equal(qsine, windows.sp(fid, off=0.5, end=1.0, pow=2.0))
