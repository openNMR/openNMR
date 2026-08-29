import os

import numpy as np
import pytest

from nmrsuite.io import bruker


def _write_params(path, params):
    lines = []
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"##${key}= (0..{len(value) - 1})")
            lines.append(" ".join(str(v) for v in value))
        elif isinstance(value, str):
            lines.append(f"##${key}= <{value}>")
        else:
            lines.append(f"##${key}= {value}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _make_1d_dataset(tmp_path, procno=1, n=8):
    expt = tmp_path / "expt1"
    pdata = expt / "pdata" / str(procno)
    pdata.mkdir(parents=True)

    _write_params(expt / "acqus", {
        "NUC1": "1H", "BF1": 400.0, "SFO1": 400.13, "O1": 2000.0, "NS": 4,
        "P": [10.0, 5.0] + [0.0] * 6,
    })
    _write_params(pdata / "procs", {
        "SW_p": 8000.0, "SI": n, "SF": 400.13, "NC_proc": 0,
    })

    real = np.arange(n, dtype="int32") - n // 2
    imag = np.zeros(n, dtype="int32")
    real.tofile(pdata / "1r")
    imag.tofile(pdata / "1i")
    return str(expt)


def test_read_1d_processed_axis_and_scaling(tmp_path):
    path = _make_1d_dataset(tmp_path, n=8)
    spectrum = bruker.read_1d_processed(path, mass=2.0)

    assert spectrum.ndim == 1
    assert len(spectrum.data) == 8
    assert spectrum.metadata.vendor == "bruker"
    assert spectrum.metadata.nucleus == "1H"
    # NC_proc=0 -> unscaled; mass=2, NS=4 -> divide by 8.
    expected = (np.arange(8) - 4) / 2.0 / 4.0
    np.testing.assert_allclose(spectrum.data, expected)
    # Descending ppm axis (NMR convention).
    assert spectrum.axes[0].ppm[0] > spectrum.axes[0].ppm[-1]


def test_read_1d_processed_crop(tmp_path):
    path = _make_1d_dataset(tmp_path, n=8)
    full = bruker.read_1d_processed(path)
    lo, hi = full.axes[0].ppm[-1], full.axes[0].ppm[0]
    mid = (lo + hi) / 2
    cropped = bruker.read_1d_processed(path, f1p=lo, f2p=mid)
    assert len(cropped.data) < len(full.data)
    assert cropped.axes[0].ppm.max() <= mid + 1e-9


def test_detect_requires_acqus_and_data(tmp_path):
    expt = tmp_path / "not_bruker"
    expt.mkdir()
    assert bruker._detect(str(expt)) is False

    (expt / "acqus").write_text("##$NUC1= <1H>\n")
    assert bruker._detect(str(expt)) is False  # no fid/ser/pdata yet

    (expt / "fid").write_bytes(b"\x00" * 16)
    assert bruker._detect(str(expt)) is True


def test_apply_bruker_window_none_is_noop():
    fid = np.ones(16, dtype=complex)
    out = bruker._apply_bruker_window(fid, {"WDW": 0}, sw_hz=8000.0)
    np.testing.assert_allclose(out, fid)


def test_apply_bruker_window_em_decays():
    fid = np.ones(16, dtype=complex)
    out = bruker._apply_bruker_window(fid, {"WDW": 1, "LB": 50.0}, sw_hz=8000.0)
    assert out[0] == pytest.approx(1.0)
    assert np.all(np.diff(np.abs(out)) <= 0)


def test_apply_bruker_window_unsupported_wdw_raises():
    fid = np.ones(16, dtype=complex)
    with pytest.raises(NotImplementedError):
        bruker._apply_bruker_window(fid, {"WDW": 2}, sw_hz=8000.0)


# ============================================================
# vdlist / vclist
# ============================================================

def test_read_vdlist_parses_suffixed_and_plain_values(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    (expt / "vdlist").write_text("10m\n1s\n500u\n2\n")

    result = bruker.read_vdlist(str(expt))
    np.testing.assert_allclose(result["seconds"], [0.01, 1.0, 0.0005, 2.0])
    assert result["raw"] == ["10m", "1s", "500u", "2"]
    assert result["name"] == "vdlist"


def test_read_vdlist_falls_back_to_lists_vd_subdir(tmp_path):
    expt = tmp_path / "expt1"
    (expt / "lists" / "vd").mkdir(parents=True)
    (expt / "lists" / "vd" / "vdlist").write_text("1m\n2m\n")

    result = bruker.read_vdlist(str(expt))
    np.testing.assert_allclose(result["seconds"], [0.001, 0.002])


def test_read_vdlist_missing_raises(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    with pytest.raises(FileNotFoundError):
        bruker.read_vdlist(str(expt))


def test_read_vclist_reads_plain_integer_counters(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    (expt / "vclist").write_text("1\n2\n4\n8\n16\n")

    result = bruker.read_vclist(str(expt))
    np.testing.assert_allclose(result["seconds"], [1, 2, 4, 8, 16])
    assert result["raw"] == ["1", "2", "4", "8", "16"]


# ============================================================
# diff.xml (PFG diffusion parameters)
# ============================================================

_DIFF_XML_TEMPLATE = """<?xml version="1.0" encoding="ISO-8859-1"?>
<diff>
    <delta>1.00</delta>
    <DELTA>20.0</DELTA>
    <exDiffCoff>6e-11</exDiffCoff>
    <xValues>16<List>0 27.84 11153.2 95950068.7 1 55.68 22306.4 383800000.0</List>
    </xValues>
</diff>
"""


def test_read_diff_xml_parses_gradient_and_b_values(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    (expt / "diff.xml").write_text(_DIFF_XML_TEMPLATE)

    result = bruker.read_diff_xml(str(expt))

    assert result["delta"] == pytest.approx(0.001)  # 1.00 ms -> s
    assert result["DELTA"] == pytest.approx(0.02)  # 20.0 ms -> s
    assert result["expD"] == pytest.approx(6e-11)
    np.testing.assert_allclose(result["gradient"], [0.2784, 0.5568])  # G/cm -> T/m
    np.testing.assert_allclose(result["B"], [95950068.7, 383800000.0])
    assert result["gamma"] is None


def test_read_diff_xml_includes_gamma_when_nucleus_given(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    (expt / "diff.xml").write_text(_DIFF_XML_TEMPLATE)

    result = bruker.read_diff_xml(str(expt), nucleus="1H")
    assert result["gamma"] is not None
    assert result["gamma"] > 0
