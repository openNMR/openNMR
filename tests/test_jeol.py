import struct

import numpy as np
import pytest

from opennmr import io
from opennmr.io import jeol

BE = ">"


def _pack_param(name, value, value_type, unit_base=0, endian=BE):
    """One JEOL parameter record: 4 unused bytes, a 2-byte scaler, 5
    unit slots (2 bytes each -- axis 0 carries ``unit_base``, the rest
    None/None), a 16-byte value payload (shape depends on
    ``value_type``), a 4-byte value-type code, then a 28-byte
    space-padded name -- see ``opennmr.io.jeol._parse_parameters``,
    which this mirrors field-for-field."""
    buf = bytearray()
    buf += b"\x00" * 4
    buf += struct.pack(endian + "h", 0)
    buf += bytes([0, unit_base]) + bytes([0, 0]) * 4
    if value_type == "String":
        buf += value.encode("ascii").ljust(16, b" ")[:16]
    elif value_type == "Integer":
        buf += struct.pack(endian + "i", value) + b"\x00" * 12
    elif value_type == "Float":
        buf += struct.pack(endian + "d", value) + b"\x00" * 8
    else:
        raise ValueError(value_type)
    buf += struct.pack(endian + "i", {"String": 0, "Integer": 1, "Float": 2}[value_type])
    buf += name.encode("ascii").ljust(28, b" ")[:28]
    return bytes(buf)


def _build_jdf(n_points=8, x_sweep_hz=1000.0, x_freq_hz=400_000_000.0, x_offset_ppm=5.0,
               x_domain="1H", solvent="CDCl3", title="synthetic", comment="test fixture",
               little_endian_body=False, data_format=1, fid=None):
    """A minimal, structurally-valid synthetic ``.jdf`` byte buffer --
    real field values only where ``opennmr.io.jeol`` actually reads
    them (nucleus, sweep width, frequency, offset, solvent, the data
    section itself); everything else is zeroed. Field layout/order
    verified by round-tripping through ``jeol.read_1d`` directly
    (both endiannesses) before being ported into this test file --
    not just assumed correct from the reference source."""
    header = bytearray()
    header += b"JEOL.NMR"
    header += struct.pack(BE + "b", 1 if little_endian_body else 0)  # endian
    header += struct.pack(BE + "B", 6)   # majorVersion
    header += struct.pack(BE + "H", 0)   # minorVersion
    header += struct.pack(BE + "B", 1)   # dataDimensionNumber
    header += struct.pack(BE + "B", 0b10000000)  # dataDimensionExist
    header += struct.pack(BE + "B", (0 << 6) | data_format)  # dataType=64bit float, dataFormat
    header += struct.pack(BE + "b", 3)   # dataInstrument
    header += b"\x00" * 8                 # translate
    header += bytes([3] + [0] * 7)        # dataAxisType: Complex on axis 0
    header += bytes([0, 28]) + bytes([0, 0]) * 7  # dataUnits: axis 0 = Second
    header += title.encode("ascii").ljust(124, b"\x00")[:124]
    header += b"\x00" * 4                 # dataAxisRanged
    header += b"".join(struct.pack(BE + "I", v) for v in ([n_points] + [0] * 7))    # dataPoints
    header += b"".join(struct.pack(BE + "I", v) for v in ([0] * 8))                  # dataOffsetStart
    header += b"".join(struct.pack(BE + "I", v) for v in ([n_points - 1] + [0] * 7))  # dataOffsetStop
    header += b"".join(struct.pack(BE + "d", v) for v in ([0.0] * 8))                # dataAxisStart
    header += b"".join(struct.pack(BE + "d", v) for v in ([(n_points - 1) / x_sweep_hz] + [0.0] * 7))  # dataAxisStop
    header += b"\x00" * 4   # creationTime
    header += b"\x00" * 4   # revisionTime
    header += b"\x00" * 16  # nodeName
    header += b"\x00" * 128  # site
    header += b"\x00" * 128  # author
    header += comment.encode("ascii").ljust(128, b"\x00")[:128]
    header += b"\x00" * (32 * 8)  # dataAxisTitles
    header += b"\x00" * (8 * 8)   # baseFreq
    header += b"\x00" * (8 * 8)   # zeroPoint
    header += b"\x00" * 8         # reversed
    header += b"\x00" * 3
    header += b"\x00" * 1  # annotationOK
    header += b"\x00" * 4  # historyUsed
    header += b"\x00" * 4  # historyLength

    param_endian = "<" if little_endian_body else BE
    params = [
        _pack_param("X_DOMAIN", x_domain, "String", endian=param_endian),
        _pack_param("X_SWEEP", x_sweep_hz, "Float", unit_base=13, endian=param_endian),
        _pack_param("X_FREQ", x_freq_hz, "Float", unit_base=13, endian=param_endian),
        _pack_param("X_OFFSET", x_offset_ppm, "Float", unit_base=26, endian=param_endian),
        _pack_param("solvent", solvent, "String", endian=param_endian),
        _pack_param("field_strength", 9.4, "Float", unit_base=31, endian=param_endian),
    ]
    param_section = bytearray()
    param_section += struct.pack(param_endian + "I", 0)  # parameterSize
    param_section += struct.pack(param_endian + "I", 0)  # lowIndex
    param_section += struct.pack(param_endian + "I", len(params) - 1)  # highIndex
    param_section += struct.pack(param_endian + "I", 0)  # totalSize
    for p in params:
        param_section += p

    # paramStart/paramLength/ListStart/ListLength/dataStart, in order,
    # right after everything already appended to `header` above.
    param_start = len(header) + 4 + 4 + 32 + 32 + 4
    data_start = param_start + len(param_section)
    header += struct.pack(BE + "I", param_start)
    header += struct.pack(BE + "I", len(param_section))
    header += b"\x00" * 32  # ListStart
    header += b"\x00" * 32  # ListLength
    header += struct.pack(BE + "I", data_start)
    assert len(header) == param_start

    if fid is None:
        fid = [complex(1.0 / (i + 1), 0.5 / (i + 1)) for i in range(n_points)]
    data_endian = "<" if little_endian_body else ">"
    data = bytearray()
    data += b"".join(struct.pack(data_endian + "d", v.real) for v in fid)
    data += b"".join(struct.pack(data_endian + "d", v.imag) for v in fid)

    return bytes(header) + bytes(param_section) + bytes(data)


def _write_jdf(tmp_path, name="spectrum.jdf", **kwargs):
    path = tmp_path / name
    path.write_bytes(_build_jdf(**kwargs))
    return str(path)


# ============================================================
# Nucleus-name normalization
# ============================================================

def test_normalize_nucleus_passes_through_standard_notation():
    assert jeol._normalize_nucleus("1H") == "1H"
    assert jeol._normalize_nucleus("13C") == "13C"


def test_normalize_nucleus_translates_special_aliases():
    assert jeol._normalize_nucleus("Proton") == "1H"
    assert jeol._normalize_nucleus("Deuteron") == "2H"


def test_normalize_nucleus_splits_spelled_out_element_and_mass_number():
    assert jeol._normalize_nucleus("Fluorine19") == "19F"
    assert jeol._normalize_nucleus("Phosphorus31") == "31P"
    assert jeol._normalize_nucleus("Nitrogen15") == "15N"


def test_normalize_nucleus_unrecognised_passes_through():
    assert jeol._normalize_nucleus("Bogonium42") == "Bogonium42"


def test_normalize_nucleus_none_input():
    assert jeol._normalize_nucleus(None) is None
    assert jeol._normalize_nucleus("") is None


# ============================================================
# Vendor detection
# ============================================================

def test_detect_recognises_jdf_with_magic_header(tmp_path):
    path = _write_jdf(tmp_path)
    assert io.detect_vendor(path) == "jeol"


def test_detect_rejects_wrong_extension(tmp_path):
    path = tmp_path / "spectrum.dat"
    path.write_bytes(_build_jdf())
    assert jeol._detect(str(path)) is False


def test_detect_rejects_missing_magic_header(tmp_path):
    path = tmp_path / "spectrum.jdf"
    path.write_bytes(b"NOT.A.JEOL.FILE." + b"\x00" * 100)
    assert jeol._detect(str(path)) is False


# ============================================================
# read_1d
# ============================================================

def test_read_1d_produces_nmrspectrum(tmp_path):
    path = _write_jdf(tmp_path, n_points=16, x_freq_hz=400_130_000.0, x_offset_ppm=5.0, x_domain="1H")
    spectrum = jeol.read_1d(path, autophase=False, baseline_correct=False)

    assert spectrum.ndim == 1
    assert len(spectrum.data) == 16
    assert spectrum.metadata.vendor == "jeol"
    assert spectrum.metadata.nucleus == "1H"
    assert spectrum.metadata.sfo1 == pytest.approx(400.13)
    assert spectrum.data_imag is not None
    # NMR convention: frequency axis decreasing left-to-right.
    assert spectrum.axes[0].hz[0] > spectrum.axes[0].hz[-1]


def test_read_1d_normalizes_spelled_out_nucleus(tmp_path):
    path = _write_jdf(tmp_path, x_domain="Fluorine19")
    spectrum = jeol.read_1d(path, autophase=False, baseline_correct=False)
    assert spectrum.metadata.nucleus == "19F"


def test_read_1d_extracts_solvent_title_comment(tmp_path):
    path = _write_jdf(tmp_path, solvent="DMSO-D6", title="my_experiment", comment="qHNMR test")
    spectrum = jeol.read_1d(path, autophase=False, baseline_correct=False)
    assert spectrum.metadata.raw["solvent"] == "DMSO-D6"
    assert spectrum.metadata.raw["title"] == "my_experiment"
    assert spectrum.metadata.raw["comment"] == "qHNMR test"


def test_read_1d_little_endian_body(tmp_path):
    path = _write_jdf(tmp_path, little_endian_body=True, x_domain="13C")
    spectrum = jeol.read_1d(path, autophase=False, baseline_correct=False)
    assert spectrum.metadata.nucleus == "13C"
    assert len(spectrum.data) == 8


def test_read_1d_ppm_axis_centred_on_x_offset(tmp_path):
    # A wide-enough sweep that the offset (the centre) sits roughly in
    # the middle of the resulting ppm axis, not at either extreme.
    path = _write_jdf(tmp_path, n_points=64, x_sweep_hz=2000.0, x_freq_hz=400_000_000.0, x_offset_ppm=5.0)
    spectrum = jeol.read_1d(path, autophase=False, baseline_correct=False)
    ppm = spectrum.axes[0].ppm
    assert ppm.min() < 5.0 < ppm.max()


def test_read_1d_rejects_non_1d_data_format(tmp_path):
    path = _write_jdf(tmp_path, data_format=2)  # Two_D
    with pytest.raises(ValueError, match="only supports 1D"):
        jeol.read_1d(path)


def test_read_1d_fid_shift_changes_output(tmp_path):
    path = _write_jdf(tmp_path, n_points=16)
    baseline = jeol.read_1d(path, autophase=False, baseline_correct=False)
    shifted = jeol.read_1d(path, autophase=False, baseline_correct=False, fid_shift=1.0)
    assert not np.allclose(baseline.data, shifted.data)


def test_read_1d_zero_fill(tmp_path):
    path = _write_jdf(tmp_path, n_points=8)
    spectrum = jeol.read_1d(path, autophase=False, baseline_correct=False, zero_fill=32)
    assert len(spectrum.data) == 32
