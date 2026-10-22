"""Agreement of the digital filter with nmrglue.

Exact on synthetic data; 1 ULP on real FIDs, from the composed
FFT/multiply/FFT chain. See the module docstring for why that residue
is acceptable and the earlier "24 broken tests" was a red herring.
"""
import os

import numpy as np
import pytest

from opennmr.io import digital_filter as df
from opennmr.io.jcampdx import parse_jcampdx_params

nmrglue = pytest.importorskip("nmrglue")


def _real_parameter_combinations():
    combos = set()
    for root, _subdirs, files in os.walk("test_data"):
        if "acqus" in files and ("fid" in files or "ser" in files):
            try:
                acqus = parse_jcampdx_params(os.path.join(root, "acqus"))
            except Exception:
                continue
            grpdly = acqus.get("GRPDLY")
            combos.add((acqus.get("DECIM"), acqus.get("DSPFVS"),
                        0 if grpdly is None else grpdly))
    return sorted(combos, key=str)


def test_matches_nmrglue_on_every_real_parameter_combination():
    combos = _real_parameter_combinations()
    if not combos:
        pytest.skip("needs real Bruker datasets in test_data")

    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 4096)) + 1j * rng.standard_normal((3, 4096))

    compared = 0
    for decim, dspfvs, grpdly in combos:
        dic = {"acqus": {"DECIM": decim, "DSPFVS": dspfvs, "GRPDLY": grpdly}}
        for kwargs in ({"truncate": True}, {"truncate": False}, {"post_proc": True}):
            try:
                theirs = nmrglue.bruker.remove_digital_filter(dic, data.copy(), **kwargs)
            except Exception:
                continue          # a combination nmrglue itself rejects
            ours = df.remove_digital_filter(dic, data.copy(), **kwargs)
            assert ours.shape == theirs.shape
            assert ours.tobytes() == theirs.tobytes(), (decim, dspfvs, grpdly, kwargs)
            compared += 1
    assert compared > 50          # the sweep really did run


@pytest.mark.skipif(
    not os.path.isdir(os.path.join("test_data", "bruker_PFG", "150")),
    reason="needs a real Bruker dataset",
)
def test_matches_nmrglue_on_a_real_fid_to_within_one_ulp():
    from opennmr.io import bruker

    path = os.path.join("test_data", "bruker_PFG", "150")
    dic = bruker.read_acqus_files(path)
    raw = bruker._read_bruker_binary(os.path.join(path, "ser"), dic)

    ours = df.remove_digital_filter(dic, raw.copy())
    theirs = nmrglue.bruker.remove_digital_filter(dic, raw.copy())

    assert ours.shape == theirs.shape
    # 1 ULP of a float64, i.e. the last bit -- orders of magnitude
    # below anything an NMR intensity carries.
    assert np.max(np.abs(ours - theirs)) / np.max(np.abs(theirs)) < 1e-15


def test_group_delay_prefers_grpdly_then_the_table():
    assert df.group_delay(decim=20, dspfvs=20, grpdly=67.98) == pytest.approx(67.98)
    assert df.group_delay(decim=8, dspfvs=22, grpdly=0) == 0.0        # DSPFVS >= 14
    assert df.group_delay(decim=24, dspfvs=12, grpdly=0) == pytest.approx(70.16666666666667)
    with pytest.raises(ValueError, match="DECIM"):
        df.group_delay(decim=7, dspfvs=12, grpdly=0)
    # Below 14 but absent from the table -- above 14 means "no
    # correction needed" and returns 0.0 before any lookup happens.
    with pytest.raises(ValueError, match="DSPFVS"):
        df.group_delay(decim=24, dspfvs=9, grpdly=0)
    assert df.group_delay(decim=24, dspfvs=99, grpdly=0) == 0.0


def test_dspfvs_zero_without_a_group_delay_means_no_filter_and_warns():
    data = (np.arange(64) + 1j * np.arange(64)).astype(np.complex128)
    dic = {"acqus": {"DECIM": 12, "DSPFVS": 0.0, "GRPDLY": 0}}
    with pytest.warns(UserWarning, match="no digital filter"):
        out = df.remove_digital_filter(dic, data)
    assert np.array_equal(out, data) and out is not data


def test_dspfvs_zero_with_a_recorded_group_delay_is_still_corrected():
    data = np.exp(2j * np.pi * 0.05 * np.arange(256))
    dic = {"acqus": {"DECIM": 12, "DSPFVS": 0.0, "GRPDLY": 20.0}}
    out = df.remove_digital_filter(dic, data)
    assert len(out) == 256 - 22


def test_other_unknown_dspfvs_values_still_raise():
    with pytest.raises(ValueError, match="DSPFVS"):
        df.remove_digital_filter({"acqus": {"DECIM": 24, "DSPFVS": 9, "GRPDLY": 0}}, np.zeros(64, complex))


def test_group_delay_is_zero_without_bruker_dsp():
    """DSPFVS <= 0: no DSP firmware, as TopSpin's vconv writes for
    converted Varian data (DSPFVS 0, GRPDLY -1, a nominal DECIM). Used
    to raise "DSPFVS 0.0 is not in the group-delay lookup table"."""
    assert df.group_delay(decim=1680.0, dspfvs=0.0, grpdly=-1.0) == 0.0
    assert df.group_delay(decim=1680.0, dspfvs=-1.0, grpdly=0.0) == 0.0
    # A recorded GRPDLY still wins, as for every other DSPFVS (real
    # DIGMOD=0 data in test_data records DSPFVS -1 with GRPDLY 67.97).
    assert df.group_delay(decim=20.0, dspfvs=-1.0, grpdly=67.96875) == pytest.approx(67.96875)
