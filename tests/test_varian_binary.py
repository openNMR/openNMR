"""Byte-comparability of the Varian reader against nmrglue.

The point of this file: replacing a format reader is only safe if the
replacement produces *the same bytes*, not merely a similar-looking
spectrum. A wrong header offset or a flipped real/imaginary
interleave yields a plausible result and no error at all.
"""
import os

import numpy as np
import pytest

from opennmr.io import varian_binary

_DATA = (
    ("varian_1d", os.path.join("test_data", "varian_1d", "Li7_8000ns_12s_june.fid"), False),
    ("varian_2d", os.path.join("test_data", "varian_2d", "HLi_CP_2D_8p5kHz_june2024.fid"), True),
)


def _cases():
    return [pytest.param(path, as_2d, id=name) for name, path, as_2d in _DATA]


@pytest.mark.parametrize("path,as_2d", _cases())
def test_fid_is_byte_identical_to_nmrglue(path, as_2d):
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")
    nmrglue = pytest.importorskip("nmrglue")

    theirs_dic, theirs = nmrglue.varian.read(path, as_2d=as_2d)
    ours_dic, ours = varian_binary.read(path, as_2d=as_2d)

    assert ours.shape == theirs.shape
    assert ours.dtype == theirs.dtype          # complex64 vs 128 must match too
    assert ours.tobytes() == theirs.tobytes()  # not allclose -- identical
    for key in ("nblocks", "ntraces", "np", "ebytes", "tbytes", "bbytes",
                "vers_id", "status", "nbheaders"):
        assert ours_dic[key] == theirs_dic[key], key


@pytest.mark.parametrize("path,as_2d", _cases())
def test_procpar_matches_nmrglue_field_for_field(path, as_2d):
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")
    nmrglue = pytest.importorskip("nmrglue")

    theirs = nmrglue.varian.read_procpar(os.path.join(path, "procpar"))
    ours = varian_binary.read_procpar(os.path.join(path, "procpar"))

    assert set(ours) == set(theirs)
    assert ours == theirs          # every field, not just "values"
    assert len(ours) > 100         # the files really do carry ~790 parameters


def test_values_stay_strings():
    # nmrglue leaves procpar values as strings and the importer's own
    # _procpar_value does the casting. Converting here would silently
    # change what every caller receives.
    path = os.path.join("test_data", "varian_1d", "Li7_8000ns_12s_june.fid")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")

    procpar = varian_binary.read_procpar(os.path.join(path, "procpar"))

    assert all(isinstance(v, str) for v in procpar["np"]["values"])


def test_multi_trace_blocks_raise_rather_than_mis_read(tmp_path):
    # ntraces != 1 needs nmrglue's own read_fid_ntraces layout, which
    # this module deliberately does not reimplement -- no data here
    # exercises it, and guessing is how wrong readers get written. It
    # must say so rather than return a mis-shaped array.
    import struct

    fid = tmp_path / "fid"
    # nblocks, ntraces, np, ebytes, tbytes, bbytes, vers_id, status, nbheaders
    fid.write_bytes(struct.pack(">6lhhl", 1, 4, 8, 4, 32, 60, 0, 0x4 | 0x10, 1)
                    + b"\0" * 256)

    with pytest.raises(NotImplementedError, match="traces per block"):
        varian_binary.read_fid(str(fid))
