"""
JEOL (Delta software, ``.jdf``) importer.

Like X-Pulse, a ``.jdf`` file hands back a raw time-domain FID (at
least for every real file checked so far -- the format *can* also
store an already-Fourier-transformed spectrum, see ``dataUnits``
below, but none of this project's real/reference data does) --
apodization, zero-filling, FFT, phasing and baseline correction all
happen here in Python, via :mod:`opennmr.processing.ft`, exactly the
same pipeline X-Pulse/Varian already go through.

There's no official public specification for this binary format.
``_parse_header``/``_parse_parameters`` below are a Python port of
`jeolconverter <https://github.com/bjonnh/jeolconverter>`_'s
``parseJEOL.js`` (MIT licensed) -- the field layout, lookup tables, and
even the two-endianness (header always big-endian; the parameter and
data sections switch to little-endian when the header's own
``endian`` byte says so) quirk all come from reading that
implementation directly, not guessed at. Verified against it, not just
self-consistent: parsed real files (see ``test_data/jeol_1d/``, from
`cheminfo/jeol-data-test <https://github.com/cheminfo/jeol-data-test>`_,
MIT licensed) reproduce the exact numeric values ``jeolconverter``'s
own test suite asserts for the same files (nucleus, frequency,
spectral offset, point count).

**2D** (2026-11-09): :func:`read_2d` reads both a genuine 2D (complex
indirect axis -- States, F1 transformed) and a pseudo-2D series (real
indirect axis, with its own list of delays), and :func:`_read_auto`,
the registered reader, picks 1D or 2D from the header. The 2D layout
(submatrix tiles, section order, used-row trim, the big-endian axis
list) and the F1 sign were established on two real files and are
documented where they are used (``_SUBMATRIX_EDGE``,
:func:`_read_data_2d`, :func:`_axis_values`, ``_F1_SIGN_FLIP``).

Real files carry a ``FILTER_FACTOR`` parameter (an integer point
count) that's a digital-filter group delay -- analogous to Bruker's
``GRPDLY``. Corrected for via ``nmrglue.bruker.rm_dig_filter`` (the
same validated algorithm Bruker's own digital-filter removal uses,
called directly with ``grpdly=FILTER_FACTOR`` rather than through its
Bruker-dict-specific wrapper), immediately after reading the raw FID
and before it's cached as ``metadata.raw["fid_raw"]`` -- matches
:mod:`opennmr.io.bruker`'s own "post digital-filter removal only"
convention for that cache.
"""

from __future__ import annotations

import inspect
import os
import re
import struct

import numpy as np

from . import base
from . import digital_filter
from ..processing import ft as processing_ft
from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum

# ============================================================
# Lookup tables (ported from jeolconverter's conversionTables.js)
# ============================================================

_ENDIANNESS = {0: "big", 1: "little"}
_DATA_TYPE_TABLE = {0: "64Bit Float", 1: "32Bit Float", 2: "Reserved", 3: "Reserved"}
_DATA_FORMAT_TABLE = {
    1: "One_D", 2: "Two_D", 3: "Three_D", 4: "Four_D", 5: "Five_D", 6: "Six_D",
    7: "Seven_D", 8: "Eight_D", 12: "Small_Two_D", 13: "Small_Three_D", 14: "Small_Four_D",
}
_DATA_AXIS_TYPE_TABLE = {0: "None", 1: "Real", 2: "TPPI", 3: "Complex", 4: "Real_Complex", 5: "Envelope"}
_VALUE_TYPE_TABLE = {0: "String", 1: "Integer", 2: "Float", 3: "Complex", 4: "Infinity"}
#: byte >> 4 only ever ranges 0-15 (an unsigned byte) -- the reference
#: table also has entries for -8..-1 that this lookup can never reach;
#: ported as-is (a dict.get(..., default) below) rather than "fixed",
#: since it matches the reference implementation's own (validated)
#: behaviour on every real file checked.
_PREFIX_TABLE = {0: "None", 1: "Milli", 2: "Micro", 3: "Nano", 4: "Pico", 5: "Femto", 6: "Atto", 7: "Zepto"}
_UNIT_PREFIX_MAGNITUDE = {
    "None": 0, "Milli": -3, "Micro": -6, "Nano": -9, "Pico": -12, "Femto": -15, "Atto": -18, "Zepto": -21,
}
_BASE_TABLE = {
    0: "None", 1: "Abundance", 2: "Ampere", 3: "Candela", 4: "Celsius", 5: "Coulomb", 6: "Degree",
    7: "Electronvolt", 8: "Farad", 9: "Sievert", 10: "Gram", 11: "Gray", 12: "Henry", 13: "Hertz",
    14: "Kelvin", 15: "Joule", 16: "Liter", 17: "Lumen", 18: "Lux", 19: "Meter", 20: "Mole",
    21: "Newton", 22: "Ohm", 23: "Pascal", 24: "Percent", 25: "Point", 26: "Ppm", 27: "Radian",
    28: "Second", 29: "Siemens", 30: "Steradian", 31: "Tesla", 32: "Volt", 33: "Watt", 34: "Weber",
    35: "Decibel", 36: "Dalton", 37: "Thompson", 38: "Ugeneric", 39: "LPercent", 40: "PPT",
    41: "PPB", 42: "Index",
}

#: JEOL spells some nuclides out ("Fluorine19", "Phosphorus31") rather
#: than using standard notation -- found by comparing X_DOMAIN across
#: several real files, not all of which needed this (1H/13C already
#: come through as-is). "Proton"/"Deuteron" don't fit the regex
#: fallback below (element name + mass number run together) at all.
_NUCLEUS_ALIASES = {"Proton": "1H", "Deuteron": "2H", "Deuterium": "2H", "Triton": "3H"}
_ELEMENT_NAME_TO_SYMBOL = {
    "Hydrogen": "H", "Helium": "He", "Lithium": "Li", "Beryllium": "Be", "Boron": "B",
    "Carbon": "C", "Nitrogen": "N", "Oxygen": "O", "Fluorine": "F", "Neon": "Ne",
    "Sodium": "Na", "Magnesium": "Mg", "Aluminum": "Al", "Silicon": "Si", "Phosphorus": "P",
    "Sulfur": "S", "Chlorine": "Cl", "Potassium": "K", "Calcium": "Ca", "Scandium": "Sc",
    "Titanium": "Ti", "Vanadium": "V", "Chromium": "Cr", "Manganese": "Mn", "Iron": "Fe",
    "Cobalt": "Co", "Nickel": "Ni", "Copper": "Cu", "Zinc": "Zn", "Gallium": "Ga",
    "Germanium": "Ge", "Arsenic": "As", "Selenium": "Se", "Bromine": "Br", "Krypton": "Kr",
    "Rubidium": "Rb", "Strontium": "Sr", "Yttrium": "Y", "Zirconium": "Zr", "Niobium": "Nb",
    "Molybdenium": "Mo", "Molybdenum": "Mo", "Ruthenium": "Ru", "Rhodium": "Rh", "Palladium": "Pd",
    "Silver": "Ag", "Cadmium": "Cd", "Indium": "In", "Tin": "Sn", "Antimony": "Sb",
    "Tellurium": "Te", "Iodine": "I", "Xenon": "Xe", "Cesium": "Cs", "Barium": "Ba",
    "Lanthanum": "La", "Tungsten": "W", "Platinum": "Pt", "Gold": "Au", "Mercury": "Hg",
    "Thallium": "Tl", "Lead": "Pb", "Bismuth": "Bi", "Uranium": "U",
}


def _normalize_nucleus(x_domain: str | None) -> str | None:
    """``"1H"``/``"13C"`` pass through unchanged; ``"Fluorine19"``,
    ``"Phosphorus31"`` etc. get split into element name + mass number
    and looked up; ``"Proton"``/``"Deuteron"`` are special-cased (see
    ``_NUCLEUS_ALIASES``). Anything unrecognised is returned as-is
    rather than dropped -- better a slightly odd label than losing the
    information."""
    if not x_domain:
        return None
    if x_domain in _NUCLEUS_ALIASES:
        return _NUCLEUS_ALIASES[x_domain]
    if re.match(r"^\d+[A-Za-z]+$", x_domain):
        return x_domain
    match = re.match(r"^([A-Za-z]+?)(\d+)$", x_domain)
    if match:
        name, mass = match.groups()
        symbol = _ELEMENT_NAME_TO_SYMBOL.get(name)
        if symbol:
            return f"{mass}{symbol}"
    return x_domain


# ============================================================
# Binary reader
# ============================================================

class _Reader:
    """A ``struct``-based binary reader with a switchable byte order --
    JEOL headers are always big-endian, but the parameter and data
    sections that follow switch to little-endian when the header's own
    ``endian`` field says so (see :func:`_parse_header`)."""

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.offset = 0
        self.big_endian = True

    def _fmt(self, code: str) -> str:
        return (">" if self.big_endian else "<") + code

    def _read(self, code: str, size: int):
        value = struct.unpack_from(self._fmt(code), self.buf, self.offset)[0]
        self.offset += size
        return value

    def u8(self) -> int: return self._read("B", 1)
    def i8(self) -> int: return self._read("b", 1)
    def u16(self) -> int: return self._read("H", 2)
    def i16(self) -> int: return self._read("h", 2)
    def u32(self) -> int: return self._read("I", 4)
    def i32(self) -> int: return self._read("i", 4)
    def f64(self) -> float: return self._read("d", 8)

    def chars(self, n: int) -> str:
        s = self.buf[self.offset:self.offset + n]
        self.offset += n
        return s.decode("ascii", errors="replace")

    def skip(self, n: int) -> None:
        self.offset += n

    def seek(self, offset: int) -> None:
        self.offset = offset

    def array(self, n: int, reader) -> list:
        return [reader() for _ in range(n)]

    def string(self, size: int) -> str:
        """A fixed-size, NUL-padded string field (header title/site/
        author/comment/axis-title fields) -- NUL bytes are dropped."""
        chars = [c for c in self.chars(size) if c != "\x00"]
        return "".join(chars)

    def param_name(self, size: int) -> str:
        """A fixed-size, space-padded name field (parameter names) --
        spaces are dropped, unlike :meth:`string`'s NUL-stripping."""
        chars = [c for c in self.chars(size) if c != " "]
        return "".join(chars)


def _get_unit(r: _Reader, size: int) -> list[dict]:
    units = []
    for _ in range(size):
        byte = r.u8()
        units.append({"prefix": _PREFIX_TABLE.get(byte >> 4), "power": byte & 0b1111, "base": _BASE_TABLE[r.i8()]})
    return units


# ============================================================
# Header + parameter parsing
# ============================================================

def _parse_header(r: _Reader) -> dict:
    h: dict = {"fileIdentifier": r.chars(8)}
    h["endian"] = _ENDIANNESS[r.i8()]
    h["majorVersion"] = r.u8()
    h["minorVersion"] = r.u16()
    h["dataDimensionNumber"] = r.u8()

    r.skip(1)  # dataDimensionExist -- a per-axis-in-use bitmask, unused here
    byte = r.u8()
    h["dataType"] = _DATA_TYPE_TABLE[byte >> 6]
    h["dataFormat"] = _DATA_FORMAT_TABLE.get(byte & 0b0011_1111, "Unknown")
    r.skip(1)  # dataInstrument
    r.skip(8)  # translate
    h["dataAxisType"] = [_DATA_AXIS_TYPE_TABLE[x] for x in r.array(8, r.u8)]
    units = _get_unit(r, 8)
    h["dataUnits"] = [u["base"] for u in units]
    # The prefix too (2026-11-09): a pseudo-2D's indirect axis and its
    # list of delays are in *milli*seconds on the real saturation-recovery
    # file, which the base unit alone ("Second") does not say.
    h["dataUnitPrefix"] = [u["prefix"] for u in units]
    h["title"] = r.string(124)

    # dataAxisRanged: two axes per byte, axis 0 in the high nibble. 0 for
    # an evenly spaced axis; non-zero (2 on the real saturation-recovery
    # file) for one whose values are listed (see "ListStart" below).
    ranged = []
    for byte in r.array(4, r.u8):
        ranged += [byte >> 4, byte & 0b1111]
    h["dataAxisRanged"] = ranged
    h["dataPoints"] = r.array(8, r.u32)
    h["dataOffsetStart"] = r.array(8, r.u32)
    h["dataOffsetStop"] = r.array(8, r.u32)
    h["dataAxisStart"] = r.array(8, r.f64)
    h["dataAxisStop"] = r.array(8, r.f64)

    r.skip(4)  # creationTime
    r.skip(4)  # revisionTime
    r.skip(16)  # nodeName
    r.skip(128)  # site
    r.skip(128)  # author
    h["comment"] = r.string(128)
    h["dataAxisTitles"] = [r.string(32) for _ in range(8)]

    h["baseFreq"] = r.array(8, r.f64)  # MHz, per axis
    h["zeroPoint"] = r.array(8, r.f64)
    h["reversed"] = [bool(x) for x in r.array(8, r.u8)]
    r.skip(3)
    r.skip(1)  # annotationOK
    r.skip(4)  # historyUsed
    r.skip(4)  # historyLength
    h["paramStart"] = r.u32()
    r.skip(4)  # paramLength
    # Where each axis's list of values sits, when it has one -- the
    # delays of a pseudo-2D (see _axis_values).
    h["listStart"] = r.array(8, r.u32)
    h["listLength"] = r.array(8, r.u32)
    h["dataStart"] = r.u32()
    # dataLength/contextStart/contextLength/annoteStart/annoteLength/
    # totalSize/unitLocation/compoundUnit follow -- not needed to
    # locate/read the parameter or data sections, so not parsed.

    if h["endian"] == "little":
        r.big_endian = False
    return h


def _parse_parameters(r: _Reader, header: dict) -> list[dict]:
    r.seek(header["paramStart"])
    r.u32()  # parameterSize
    r.u32()  # lowIndex
    high_index = r.u32()
    r.u32()  # totalSize

    params = []
    for _ in range(high_index + 1):
        r.skip(4)
        r.i16()  # scaler -- not used downstream, but must still be consumed to stay aligned
        unit = _get_unit(r, 5)
        value_start = r.offset
        r.skip(16)
        value_type = _VALUE_TYPE_TABLE.get(r._read("i", 4))
        r.seek(value_start)
        if value_type == "String":
            value = r.param_name(16)
        elif value_type == "Integer":
            value = r.i32()
            r.skip(12)
        elif value_type == "Float":
            value = r.f64()
            r.skip(8)
        elif value_type == "Infinity":
            value = r.i32()
            r.skip(12)
        else:
            value = None
            r.skip(16)
        r.skip(4)
        name = r.param_name(28)
        params.append({"name": name, "unit": unit, "value": value, "valueType": value_type})
    return params


def _get_param(params: list[dict], name: str) -> dict | None:
    return next((p for p in params if p["name"] == name), None)


def _get_magnitude(params: list[dict], name: str) -> float:
    """A parameter's value converted to base SI units (its recorded
    unit prefix applied) -- e.g. a ``X_SWEEP`` of 10 with prefix
    ``"Kilo"`` returns ``10000.0`` (Hz)."""
    param = _get_param(params, name)
    if param is None:
        raise ValueError(f"Parameter {name!r} not found in this JEOL file.")
    exponent = _UNIT_PREFIX_MAGNITUDE[param["unit"][0]["prefix"]]
    return param["value"] * 10.0 ** exponent


def _read_data_1d(r: _Reader, header: dict) -> np.ndarray:
    """The FID (or, for a file whose ``dataUnits`` says ``"Ppm"``, an
    already-transformed spectrum -- not handled here, see this
    module's own docstring) as one complex array, from the real (and,
    for a Complex/Real_Complex/TPPI axis, imaginary) data section(s)."""
    r.seek(header["dataStart"])
    n = header["dataPoints"][0]
    code = "d" if header["dataType"] == "64Bit Float" else "f"
    size = 8 if code == "d" else 4
    axis_type = header["dataAxisType"][0]
    n_sections = 2 if axis_type in ("Complex", "Real_Complex", "TPPI") else 1

    sections = []
    for _ in range(n_sections):
        section = np.frombuffer(r.buf, dtype=r._fmt(code), count=n, offset=r.offset)
        r.offset += n * size
        sections.append(np.asarray(section, dtype=np.float64))

    if n_sections == 1:
        return sections[0].astype(np.complex128)
    return sections[0] + 1j * sections[1]


def _open_jdf(path: str) -> tuple[_Reader, dict, list[dict]]:
    """Read ``path`` and parse its header and parameters (no data yet)."""
    with open(path, "rb") as f:
        buf = f.read()
    r = _Reader(buf)
    header = _parse_header(r)
    if header["fileIdentifier"] != "JEOL.NMR":
        raise ValueError(f"{path!r} is not a JEOL .jdf file (unexpected file identifier).")
    params = _parse_parameters(r, header)
    return r, header, params


def _is_2d_header(header: dict) -> bool:
    return header["dataFormat"] in _SUBMATRIX_EDGE


def _read_jeol_jdf(path: str) -> dict:
    """Parse a ``.jdf`` file's header, parameters, and 1D FID/spectrum
    data -- the JEOL-format-specific piece; :func:`read_1d` handles
    everything vendor-agnostic (windowing, FFT, phasing, ...) from
    here on, same split as :func:`opennmr.io.xpulse._read_xpulse_jcamp`."""
    r, header, params = _open_jdf(path)
    if header["dataFormat"] != "One_D":
        if _is_2d_header(header):
            raise ValueError(
                f"{os.path.basename(path)!r} is a 2D JEOL file ({header['dataFormat']}); "
                "open it with opennmr.io.load (or jeol.read_2d), not read_1d.")
        raise ValueError(
            f"JEOL dataFormat={header['dataFormat']!r} is not supported (1D and 2D only).")
    data = _read_data_1d(r, header)
    return {"header": header, "params": params, "data": data}


# ============================================================
# 2D data: submatrices, sections, axis value lists
# ============================================================

#: Edge of the square submatrices ("tiles") a 2D ``.jdf`` stores its
#: data in -- the data are not row after row but tile after tile, rows
#: of tiles in turn, each tile row-major inside. Established on the two
#: real files this reader was written against (2026-11-09; a JEOL ECZ
#: 600, Delta 5.2.1): read with any other tiling, the FIDs are not FIDs
#: -- ``Two_D`` (a 14N-1H HMQC, 1024 x 32) only reads as 32 x 32 tiles,
#: every row then starting at the same point and decaying, and
#: ``Small_Two_D`` (a 1H saturation recovery, 1024 x 24) only as 4 x 4,
#: its rows then growing monotonically with the recovery delay.
_SUBMATRIX_EDGE = {"Two_D": 32, "Small_Two_D": 4}

#: Real and imaginary parts per point, by axis type. TPPI, Real_Complex
#: and Envelope axes are not read: no real file to check them against.
_PARTS = {"Real": 1, "Complex": 2}


def _untile(flat: np.ndarray, n_x: int, n_y: int, edge: int) -> np.ndarray:
    """One section's ``n_y`` x ``n_x`` array from its submatrix order."""
    tiles = flat.reshape(n_y // edge, n_x // edge, edge, edge)
    return tiles.transpose(0, 2, 1, 3).reshape(n_y, n_x)


def _read_data_2d(r: _Reader, header: dict) -> np.ndarray:
    """The 2D data as ``(parts_y, n_y, n_x)`` complex FIDs: for each part of
    the indirect axis (one if it is real, cosine and sine if it is
    complex) the rows of complex direct-dimension FIDs.

    A section per combination of parts, the direct axis's part changing
    fastest -- RR, IR, RI, II for complex x complex. Checked on the real
    HMQC, not assumed: sections 0 and 1, and 2 and 3, are the pairs that
    give a one-sided (quadrature) 1H spectrum; any other pairing gives a
    near-symmetric one.

    Only the rows that were acquired are returned (``dataOffsetStop``):
    the real saturation recovery stores 24 rows of which 21 hold data,
    the last three being the padding the 4 x 4 tiles need."""
    edge = _SUBMATRIX_EDGE[header["dataFormat"]]
    axis_x, axis_y = header["dataAxisType"][0], header["dataAxisType"][1]
    if axis_x not in _PARTS or axis_y not in _PARTS:
        raise ValueError(
            f"JEOL 2D data with {axis_x} x {axis_y} axes is not supported yet (Real or "
            "Complex axes only).")
    n_x, n_y = header["dataPoints"][0], header["dataPoints"][1]
    if n_x % edge or n_y % edge:
        raise ValueError(
            f"JEOL {header['dataFormat']} data of {n_x} x {n_y} points does not divide into "
            f"its {edge} x {edge} submatrices.")
    parts_x, parts_y = _PARTS[axis_x], _PARTS[axis_y]
    code = "d" if header["dataType"] == "64Bit Float" else "f"
    count = n_x * n_y
    sections = np.frombuffer(r.buf, dtype=r._fmt(code), count=count * parts_x * parts_y,
                             offset=header["dataStart"]).reshape(parts_x * parts_y, count)
    sections = [_untile(np.asarray(sec, dtype=np.float64), n_x, n_y, edge) for sec in sections]

    used_x = min(n_x, header["dataOffsetStop"][0] + 1)
    used_y = min(n_y, header["dataOffsetStop"][1] + 1)
    out = np.empty((parts_y, used_y, used_x), dtype=np.complex128)
    for iy in range(parts_y):
        real = sections[parts_x * iy][:used_y, :used_x]
        imag = sections[parts_x * iy + 1][:used_y, :used_x] if parts_x == 2 else 0.0
        out[iy] = real + 1j * imag
    return out


_SI_PREFIX_SCALE = {"None": 1.0, "Milli": 1e-3, "Micro": 1e-6, "Nano": 1e-9, "Pico": 1e-12}


def _axis_values(r: _Reader, header: dict, axis: int, n_used: int) -> dict | None:
    """The values of a non-frequency axis (a pseudo-2D's delays), as
    ``{"values": ndarray, "unit": "s" | <JEOL unit>, "source": str}``.

    The file's own list when it has one -- big-endian float64s, one per
    stored row, in the axis's unit (the real saturation recovery: 15,
    23.77, 37.68, ... 150 000 *ms*, log-spaced, exactly
    ``dataAxisStart``..``dataAxisStop``). Otherwise, for an axis in
    seconds, evenly spaced from ``dataAxisStart`` to ``dataAxisStop``.
    None when neither gives ``n_used`` finite values."""
    unit = header["dataUnits"][axis]
    scale = _SI_PREFIX_SCALE.get(header["dataUnitPrefix"][axis], None)
    name = header.get("dataAxisTitles", [""] * 8)[axis]
    values, how = None, ""
    start, length = header["listStart"][axis], header["listLength"][axis]
    if length >= 8 * n_used and start + length <= len(r.buf):
        for dtype in (">f8", "<f8"):
            candidate = np.frombuffer(r.buf, dtype=dtype, count=n_used, offset=start).astype(float)
            if np.all(np.isfinite(candidate)) and np.all(np.abs(candidate) < 1e12):
                values, how = candidate, "the file's own list"
                break
    if values is None and unit == "Second":
        first, last = header["dataAxisStart"][axis], header["dataAxisStop"][axis]
        if last > first:
            values, how = np.linspace(first, last, n_used), "evenly spaced over the file's axis range"
    if values is None:
        return None
    label = f"'{name}' " if name else ""
    if unit == "Second" and scale is not None:
        return {"values": values * scale, "unit": "s", "name": name,
                "source": f"the JEOL .jdf {label}axis ({how})"}
    return {"values": values, "unit": unit, "name": name,
            "source": f"the JEOL .jdf {label}axis ({how}, in {unit})"}


# ============================================================
# NMRSpectrum-producing reader
# ============================================================

def read_1d(path: str, *, window: str = "exponential", window_kwargs: dict | None = None,
            zero_fill: int | bool | None = None, fid_shift: float = 0.0, align_echo: bool = False,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward",
            autophase: bool = True, phase_mode: str = "apc0",
            phase0: float = 0.0, phase1: float = 0.0, pivot_frac: float = 0.5, phase0_inc: float = 0.1,
            phase1_max: float = 20.0, phase1_improvement: float = 0.15,
            phase_penalty: str = "magnitude",
            baseline_correct: bool = True, normalize: bool = False) -> NMRSpectrum:
    """Import a JEOL Delta ``.jdf`` FID and process it (apodize,
    zero-fill, FFT, phase, baseline-correct) into an
    :class:`NMRSpectrum` -- see this module's own docstring for the
    format itself and the ``FILTER_FACTOR`` digital-filter correction
    applied before any of the steps here run.

    ``window``/``window_kwargs``/``zero_fill``/``fid_shift``/
    ``align_echo``/``linear_predict_*``/``autophase``/``phase_mode``/
    ``phase0``/``phase1``/``baseline_correct``/``normalize`` :
    identical meaning to :func:`opennmr.io.xpulse.read_1d`'s own
    same-named parameters -- both go through the same
    :mod:`opennmr.processing.ft` pipeline. No ``truncate`` here
    (unlike X-Pulse): nothing in the real 1D data checked so far
    suggested it's needed -- every real file looks like an ordinary
    one-sided decaying acquisition, not a full echo. ``align_echo`` is
    offered anyway, purely
    for interface consistency with every other vendor's ``read_1d``
    (the GUI's Processing tab sends it unconditionally) -- unlike
    X-Pulse's own ``align_echo``, it does *not* also cap a wrap-point
    truncation (there is no ``truncate`` to cap here), so if a genuine
    full-echo JEOL acquisition ever needs this, check for the same
    "looks like a time-reversal" wrapped-tail artifact X-Pulse had
    before trusting it.
    """
    parsed = _read_jeol_jdf(path)
    header, params = parsed["header"], parsed["params"]
    fid = parsed["data"]

    # FILTER_FACTOR is JEOL's digital-filter group-delay count -- the
    # same physical quantity as Bruker's GRPDLY (an FID whose true
    # start has been shifted a few points into the array by the
    # digital filter, producing a phase error that grows with
    # frequency if left uncorrected). Reused directly via nmrglue's
    # own Bruker digital-filter-removal algorithm (rm_dig_filter,
    # its generic grpdly-based entry point, not the Bruker-dict-
    # specific remove_digital_filter wrapper) rather than
    # reimplementing the same DSP by hand -- decim/dspfvs are Bruker's
    # own table-lookup fallback for when grpdly is 0, unused here
    # since FILTER_FACTOR is always already a known point count.
    filter_factor_param = _get_param(params, "FILTER_FACTOR")
    if filter_factor_param is not None and filter_factor_param["value"]:
        fid = digital_filter.rm_dig_filter(fid, decim=0, dspfvs=0, grpdly=float(filter_factor_param["value"]))

    fid_raw = fid  # the truly raw FID (post digital-filter removal only) -- see metadata.raw["fid_raw"]'s own comment below
    original_length = len(fid)

    dt = 1.0 / _get_magnitude(params, "X_SWEEP")
    freq_hz = _get_magnitude(params, "X_FREQ")  # the observe/base frequency, in Hz
    offset_ppm = _get_param(params, "X_OFFSET")["value"]  # centre of the acquired window, in ppm
    np_points = header["dataPoints"][0]

    if fid_shift:
        fid = processing_ft.roll_fid(fid, fid_shift)
    if align_echo:
        fid = processing_ft.align_echo(fid)

    if linear_predict_points:
        fid = processing_ft.linear_predict(
            fid, order=linear_predict_order, n_predict=linear_predict_points,
            direction=linear_predict_direction,
        )
        np_points = len(fid)

    fid = processing_ft.apodize(fid, dt, window=window, **(window_kwargs or {}))

    if zero_fill is True:
        si_points = int(_get_param(params, "X_POINTS")["value"]) if _get_param(params, "X_POINTS") else np_points
    elif isinstance(zero_fill, int):
        si_points = zero_fill
    else:
        si_points = None
    if si_points is not None and si_points > len(fid):
        fid = processing_ft.zero_fill(fid, si_points)
    elif np_points is not None:
        fid = fid[:np_points]

    freq_hz_axis, spectrum = processing_ft.fft_spectrum(fid, dt)
    # X_OFFSET is already the acquired window's own centre, in ppm --
    # simpler than the SF/SFO1/O1 combination other vendors need,
    # since JEOL records the centre directly rather than an absolute
    # transmitter-vs-reference offset.
    freq_ppm = offset_ppm + freq_hz_axis / (freq_hz / 1e6)

    if autophase:
        if phase_mode == "joint":
            phase0, phase1 = processing_ft.autophase01(
                spectrum, freq_ppm, phase0_range=(-180, 180), phase0_inc=phase0_inc,
                phase1_max=phase1_max, phase1_improvement=phase1_improvement,
                penalty=phase_penalty, pivot_frac=0.5,
            )
        elif phase_mode == "apc0":
            phase0 = processing_ft.apc0(spectrum, freq_ppm, phase0_inc)
        else:
            raise ValueError("phase_mode must be 'apc0' or 'joint'")

    spectrum = processing_ft.apply_phase(spectrum, freq_ppm, phase0, phase1, pivot_frac=pivot_frac)

    if baseline_correct:
        spectrum = processing_ft.baseline_correct(freq_ppm, spectrum)

    if normalize:
        maxval = np.max(np.abs(spectrum))
        if maxval > 0:
            spectrum = spectrum / maxval

    nucleus = _normalize_nucleus(_get_param(params, "X_DOMAIN")["value"] if _get_param(params, "X_DOMAIN") else None)
    solvent_param = _get_param(params, "solvent")
    # "experiment" is JEOL's own closest equivalent to Bruker's PULPROG/
    # X-Pulse's .PULSESEQUENCE/Varian's seqfil -- confirmed present and
    # giving a sensible pulse-sequence-like name (e.g. "single_pulse_dec",
    # "carbon.jxp") on all four real JEOL test files available.
    experiment_param = _get_param(params, "experiment")

    metadata = NMRMetadata(
        vendor="jeol",
        nucleus=nucleus,
        sfo1=freq_hz / 1e6,
        bf1=freq_hz / 1e6,
        o1=offset_ppm * freq_hz / 1e6,
        source_path=path,
        pulse_program=experiment_param["value"] if experiment_param else None,
        raw={
            "header": header, "params": params, "phase0": phase0, "phase1": phase1,
            "solvent": solvent_param["value"] if solvent_param else None,
            "title": header["title"], "comment": header["comment"],
            # Unlike the other readers, JEOL data are NOT divided by the
            # number of scans here (intensities match nmrglue's); Compare's
            # per-scan modes divide by total_scans/scans instead
            # (notebook.intensity_scaling).
            "per_scan": False,
            # The fully processed (pre-FFT) FID -- an optional GUI
            # preview of what FID-domain processing actually did.
            "fid": fid, "fid_dt": dt, "fid_original_length": original_length,
            # The *truly* raw FID -- straight off disk, before even
            # shift/align/LP, let alone windowing/zero-fill. See
            # opennmr.io.bruker's matching "fid_raw" comment: lets the
            # GUI recompute a live preview of a not-yet-applied
            # processing choice entirely in memory.
            "fid_raw": fid_raw,
        },
    )
    axis = NMRAxis(kind="frequency", ppm=freq_ppm, hz=freq_hz_axis + offset_ppm * freq_hz / 1e6)

    return NMRSpectrum(
        data=np.real(spectrum),
        axes=[axis],
        metadata=metadata,
        data_imag=np.imag(spectrum),
    )


# ============================================================
# 2D
# ============================================================

#: F1 quadrature, for a 2D whose indirect axis is Complex: the cosine and
#: sine halves of a States acquisition, combined as ``Re(cos) +
#: 1j*Re(sin)`` after the F2 transform -- **not** negated, unlike
#: Bruker's (``metadata.raw["f1_quadrature_sign_flip"] = False``, which
#: the GUI's in-memory reprocessing honours).
#:
#: Why, on the one real dataset (a 14N-1H HMQC of an ethionamide-
#: salicylic acid cocrystal, 70 kHz MAS) -- not against JEOL's own
#: processing, which was not available:
#:
#: * **Consistency with F2.** JEOL stores both axes the same way
#:   ("Complex": real section, imaginary section), and F2 is read as
#:   real + 1j*imag with no conjugation -- verified on the 1H spectrum
#:   (alkyl strongest at ~1.6 ppm, the H-bonded acid protons at 14-18).
#:   Bruker's F1 negation goes with its reader conjugating F2; JEOL has
#:   no conjugation to compensate.
#: * **The two cross peaks' order, which needs no 14N reference.** The
#:   acid proton H-bonded to the pyridine N (1H 17.5 ppm) correlates to a
#:   14N ~430 ppm *above* the thioamide NH2 (1H 8.4 ppm): a hydrogen-
#:   bonded pyridine N has both the higher shift and the larger C_Q
#:   (larger positive quadrupolar-induced shift), so it must be above.
#:   Negating the sine rows puts it ~430 ppm below instead.
#:
#: First checked the other way (2026-11-09) from the NH2 peak alone and
#: an assumed reference scale -- the argument above replaced that. Check
#: against Delta's own processed 2D when one is available.
_F1_SIGN_FLIP = False


def read_2d(path: str, *, pseudo_2d: bool | None = None,
            window: str = "exponential", window_kwargs: dict | None = None,
            zero_fill: int | bool | None = None,
            truncate: int | None = None,
            fid_shift: float = 0.0, align_echo: bool = False, echo_offset: float = 0.0,
            linear_predict_points: int = 0, linear_predict_order: int = 20,
            linear_predict_direction: str = "forward",
            window_f1: str | None = None, window_kwargs_f1: dict | None = None,
            zero_fill_f1: int | bool | None = None, truncate_f1: int | None = None,
            autophase: bool = True, phase_mode: str = "apc0",
            phase0: float = 0.0, phase1: float = 0.0, pivot_frac: float = 0.5,
            phase0_inc: float = 0.1, phase1_max: float = 20.0, phase1_improvement: float = 0.15,
            phase_penalty: str = "magnitude",
            drift_correction: dict | None = None) -> NMRSpectrum:
    """Read a 2D JEOL Delta ``.jdf`` -- a genuine 2D (HSQC, HMQC, ...) or a
    pseudo-2D series (saturation/inversion recovery, ...) -- and process
    it: F2 (digital filter, shift/echo/truncate/LP, window, zero-fill,
    FFT, phase) row by row, then, for a genuine 2D, F1 (States decoding,
    window, zero-fill, FFT).

    ``pseudo_2d`` : ``None`` (default) decides from the file: an indirect
    axis that is Real, or has no nucleus (``Y_DOMAIN`` empty), is a
    series -- one spectrum per row, no F1 transform -- and anything else
    a genuine 2D. ``True`` shows a genuine 2D's rows as a series (the
    cosine and sine row of each increment in turn, as Bruker's reader
    does); ``False`` on a Real indirect axis is refused, as there is no
    F1 quadrature to transform.

    A series carries the indirect axis's own values (the delays) in
    ``metadata.raw["indirect_values"]`` -- in seconds, from the file's
    list (see :func:`_axis_values`) -- which Dynamics uses as its Auto
    x-values.

    F2 options (``window`` ... ``linear_predict_*``, ``truncate``) mean
    what they mean for :func:`read_1d`, applied to every row.
    ``zero_fill``: ``True`` to ``X_POINTS``, an int to that many points,
    ``None``/``0`` none.

    F1 quadrature: see ``_F1_SIGN_FLIP``.

    F1 options (genuine 2D): ``window_f1``/``window_kwargs_f1`` (None:
    no window -- JEOL records no processing to reproduce);
    ``zero_fill_f1``: an int, ``True`` for the next power of two, ``0``/
    ``False`` none, ``None`` (default) the next power of two of twice
    the increments -- a smooth contour from a short F1 without inventing
    anything; ``truncate_f1``: the first that many increments.

    ``autophase``/``phase_mode`` (``"apc0"`` or ``"joint"``): F2 phase
    from the strongest row, applied to all; ``"recorded"`` (the Bruker
    default the GUI may send) has nothing recorded to use in a ``.jdf``
    and falls back to ``"apc0"``, saying so in
    ``metadata.raw["phase_note"]``. F1 is not phased here (no recorded
    phase); the GUI's 2D phasing applies to it.

    ``drift_correction`` : as for the other vendors' pseudo-2D readers
    (see :mod:`opennmr.notebook.drift_correction`); refused for a
    genuine 2D, whose indirect dimension would need correcting too.

    Intensities are not divided by the number of scans, as for
    :func:`read_1d` (``metadata.raw["per_scan"] = False``).
    """
    r, header, params = _open_jdf(path)
    if not _is_2d_header(header):
        raise ValueError(f"{os.path.basename(path)!r} is not a 2D JEOL file "
                         f"(dataFormat {header['dataFormat']!r}); use read_1d.")
    parts = _read_data_2d(r, header)                     # (parts_y, n1, n2)
    axis_y_complex = parts.shape[0] == 2
    y_domain = _get_param(params, "Y_DOMAIN")
    nucleus_indirect = _normalize_nucleus(y_domain["value"] if y_domain else None) or None
    auto_pseudo = (not axis_y_complex) or not nucleus_indirect
    if pseudo_2d is None:
        pseudo_2d = auto_pseudo
    if not pseudo_2d and not axis_y_complex:
        raise ValueError(
            "This JEOL 2D has a real (not complex) indirect axis, so it has no F1 quadrature "
            "to transform – it is read as a series (pseudo-2D).")
    if drift_correction and not pseudo_2d:
        from ..notebook import drift_correction as _dc
        raise ValueError(_dc.NOT_SUPPORTED_MESSAGE)

    # Rows in acquisition order: for a complex indirect axis, increment k's
    # cosine row then its sine row (Bruker's FnMODE 4 layout, so the
    # GUI's own 2D code reads it the same way, with the sign flag below).
    n1, n2 = parts.shape[1], parts.shape[2]
    rows = np.empty((parts.shape[0] * n1, n2), dtype=np.complex128)
    for k in range(parts.shape[0]):
        rows[k::parts.shape[0]] = parts[k]
    fn_mode = 4 if (axis_y_complex and not pseudo_2d) else 1

    filter_factor = _get_param(params, "FILTER_FACTOR")
    if filter_factor is not None and filter_factor["value"]:
        rows = digital_filter.rm_dig_filter(rows, decim=0, dspfvs=0, grpdly=float(filter_factor["value"]))
    rows = np.asarray(rows, dtype=np.complex128)
    original_length = rows.shape[1]

    dt = 1.0 / _get_magnitude(params, "X_SWEEP")
    freq_hz = _get_magnitude(params, "X_FREQ")
    offset_ppm = _get_param(params, "X_OFFSET")["value"]

    indirect_values = None
    if pseudo_2d and not axis_y_complex:
        indirect_values = _axis_values(r, header, 1, n1)

    drift_record = None
    if drift_correction:
        from ..notebook import drift_correction as _dc
        rows = _dc.correct_pseudo_2d_fids(
            rows, dt, start_hz=drift_correction.get("start_hz", 0.0),
            end_hz=drift_correction.get("end_hz"), shifts_hz=drift_correction.get("shifts_hz"),
            endpoint=bool(drift_correction.get("endpoint", True)))
        drift_record = _dc.pseudo_2d_record(rows.shape[0], drift_correction, "JEOL")
    fid_raw = rows

    if fid_shift or align_echo or truncate or linear_predict_points:
        rows = np.array([processing_ft.prepare_fid(
            row, fid_shift=fid_shift, align_echo_enabled=align_echo, echo_offset=echo_offset,
            truncate=truncate, linear_predict_points=linear_predict_points,
            linear_predict_order=linear_predict_order,
            linear_predict_direction=linear_predict_direction) for row in rows])
    rows = processing_ft.apodize(rows, dt, window=window, **(window_kwargs or {}))
    if zero_fill is True:
        x_points = _get_param(params, "X_POINTS")
        si2 = int(x_points["value"]) if x_points else rows.shape[1]
    elif isinstance(zero_fill, int) and zero_fill > 0:
        si2 = int(zero_fill)
    else:
        si2 = rows.shape[1]
    if si2 > rows.shape[1]:
        rows = processing_ft.zero_fill(rows, si2)
    hz2, spectrum = processing_ft.fft_spectrum(rows, dt)
    ppm2 = offset_ppm + hz2 / (freq_hz / 1e6)

    phase_note = None
    if autophase:
        if phase_mode == "recorded":
            phase_note = ("A JEOL .jdf records no phase to reuse, so the F2 phase was found "
                          "automatically (APC0) instead.")
            phase_mode = "apc0"
        reference = spectrum[processing_ft.reference_row(spectrum)]
        if phase_mode == "joint":
            phase0, phase1 = processing_ft.autophase01(
                reference, ppm2, phase0_range=(-180, 180), phase0_inc=phase0_inc,
                phase1_max=phase1_max, phase1_improvement=phase1_improvement,
                penalty=phase_penalty, pivot_frac=0.5)
        elif phase_mode == "apc0":
            phase0 = processing_ft.apc0(reference, ppm2, phase0_inc)
        else:
            raise ValueError("phase_mode must be 'apc0', 'joint' or 'recorded'")
    spectrum = processing_ft.apply_phase(spectrum, ppm2, phase0, phase1, pivot_frac=pivot_frac)

    dt1 = None
    y_sweep = _get_param(params, "Y_SWEEP")
    if pseudo_2d:
        y_axis = NMRAxis(kind="index", index=np.arange(1, spectrum.shape[0] + 1))
    else:
        dt1 = 1.0 / _get_magnitude(params, "Y_SWEEP")
        y_freq = _get_magnitude(params, "Y_FREQ")
        y_offset = _get_param(params, "Y_OFFSET")["value"]
        if isinstance(truncate_f1, int) and not isinstance(truncate_f1, bool) and 0 < truncate_f1 < n1:
            spectrum = spectrum[:2 * truncate_f1]
        interferogram = spectrum.copy()
        if _F1_SIGN_FLIP:
            interferogram[1::2] *= -1
        interferogram = processing_ft.decode_hypercomplex_f1(interferogram, fn_mode)
        if window_f1:
            interferogram = processing_ft.apodize(
                interferogram.T, dt1, window=window_f1, **(window_kwargs_f1 or {})).T
        n_inc = interferogram.shape[0]
        if zero_fill_f1 is None:
            si1 = int(2 ** np.ceil(np.log2(max(2 * n_inc, 1))))
        elif zero_fill_f1 is True:
            si1 = int(2 ** np.ceil(np.log2(max(n_inc, 1))))
        elif isinstance(zero_fill_f1, int) and zero_fill_f1 > 0:
            si1 = int(zero_fill_f1)
        else:
            si1 = n_inc
        filled = processing_ft.zero_fill(interferogram.T, max(si1, n_inc))
        hz1, transformed = processing_ft.fft_spectrum(filled, dt1)
        spectrum = transformed.T
        ppm1 = y_offset + hz1 / (y_freq / 1e6)
        y_axis = NMRAxis(kind="frequency", ppm=ppm1, hz=hz1 + y_offset * y_freq / 1e6)

    x_domain = _get_param(params, "X_DOMAIN")
    nucleus = _normalize_nucleus(x_domain["value"] if x_domain else None)
    experiment = _get_param(params, "experiment")
    solvent = _get_param(params, "solvent")
    scans = _get_param(params, "SCANS")
    y_freq_param = _get_param(params, "Y_FREQ")
    metadata = NMRMetadata(
        vendor="jeol",
        nucleus=nucleus,
        nucleus_indirect=None if pseudo_2d else nucleus_indirect,
        sfo1=freq_hz / 1e6,
        sfo1_indirect=(None if pseudo_2d or y_freq_param is None
                       else _get_magnitude(params, "Y_FREQ") / 1e6),
        bf1=freq_hz / 1e6,
        o1=offset_ppm * freq_hz / 1e6,
        source_path=path,
        pseudo_2d=bool(pseudo_2d),
        homonuclear=bool(not pseudo_2d and nucleus_indirect == nucleus),
        pulse_program=experiment["value"] if experiment else None,
        raw={
            "header": header, "params": params, "phase0": phase0, "phase1": phase1,
            "solvent": solvent["value"] if solvent else None,
            "title": header["title"], "comment": header["comment"],
            "per_scan": False,      # as read_1d: JEOL data are not divided by scans
            "scans": scans["value"] if scans else None,
            "fn_mode": fn_mode,
            "f1_quadrature_sign_flip": _F1_SIGN_FLIP,
            "fid": rows, "fid_dt": dt, "fid_original_length": original_length,
            "fid_raw": fid_raw,
            "fid_dt_f1": dt1,
            "pseudo_2d_basis": _pseudo_2d_basis(bool(pseudo_2d), auto_pseudo, axis_y_complex,
                                                nucleus_indirect),
            **({"indirect_values": indirect_values} if indirect_values else {}),
            **({"phase_note": phase_note} if phase_note else {}),
            **({"drift_correction": drift_record} if drift_record is not None else {}),
            **({"y_sweep_hz": _get_magnitude(params, "Y_SWEEP")} if y_sweep else {}),
        },
    )
    x_axis = NMRAxis(kind="frequency", ppm=ppm2, hz=hz2 + offset_ppm * freq_hz / 1e6)
    return NMRSpectrum(data=np.real(spectrum), axes=[y_axis, x_axis], metadata=metadata,
                       data_imag=np.imag(spectrum))


def _pseudo_2d_basis(pseudo_2d: bool, auto: bool, complex_y: bool, nucleus) -> str:
    """Why this 2D was read as a series or a genuine 2D (shown by the GUI)."""
    if pseudo_2d != auto:
        return ("a series" if pseudo_2d else "a genuine 2D") + ": your choice"
    if pseudo_2d:
        return "a series: " + ("the indirect axis is real (no F1 quadrature)" if not complex_y
                               else "no indirect nucleus recorded")
    return f"a genuine 2D: complex indirect axis of {nucleus}"


def _accepted(func, kwargs: dict) -> dict:
    """``kwargs`` that ``func`` takes: the GUI sends every vendor one set
    of processing options, and a 1D read has no F1 to apply F1's to."""
    names = set(inspect.signature(func).parameters)
    return {k: v for k, v in kwargs.items() if k in names}


def _read_auto(path: str, **kwargs) -> NMRSpectrum:
    """Vendor-registry entry point: :func:`read_2d` for a 2D file,
    :func:`read_1d` otherwise."""
    _r, header, _params = _open_jdf(path)
    if _is_2d_header(header):
        return read_2d(path, **_accepted(read_2d, kwargs))
    if kwargs.get("drift_correction"):
        from ..notebook import drift_correction as _dc
        raise ValueError(_dc.NOT_SUPPORTED_MESSAGE)
    return read_1d(path, **_accepted(read_1d, kwargs))


# ============================================================
# Vendor registration
# ============================================================

def _detect(path: str) -> bool:
    """A single ``.jdf`` file starting with the ``JEOL.NMR`` magic
    string -- JEOL Delta doesn't use a directory-of-files layout the
    way Bruker/Varian do."""
    if not os.path.isfile(path):
        return False
    if os.path.splitext(path)[1].lower() != ".jdf":
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return False
    return head == b"JEOL.NMR"


def peek(path: str) -> dict:
    """Metadata-only probe: shape/nucleus from the ``.jdf`` header and
    parameter block, skipping the FT pipeline entirely.

    Without this, :func:`opennmr.io.base.peek` fell back to a full
    ``load()``, and for JEOL that is almost *all* pipeline: measured at
    1.4 ms to parse the file against 522 ms to read it, the difference
    being apodisation, zero-fill, FFT, autophase and baseline
    correction -- none of which the Data-tab browser needs to print
    "13C, 1D, 32750".

    **The digital-filter trim is applied, not re-derived.** The final
    point count is whatever ``rm_dig_filter`` leaves (32768 -> 32750 on
    the real files here, from a ``FILTER_FACTOR`` of 16 -- not a
    subtraction of 16), so this calls the same function
    :func:`read_1d` does on the same array rather than reimplementing
    its arithmetic, which would be a second place for the two to
    disagree. It costs a single array slice on data already in memory.
    """
    r, header, params = _open_jdf(path)
    if _is_2d_header(header):
        return _peek_2d(r, header, params)
    fid = _read_data_1d(r, header)
    filter_factor_param = _get_param(params, "FILTER_FACTOR")
    if filter_factor_param is not None and filter_factor_param["value"]:
        fid = digital_filter.rm_dig_filter(
            fid, decim=0, dspfvs=0, grpdly=float(filter_factor_param["value"]))
    x_domain = _get_param(params, "X_DOMAIN")
    experiment = _get_param(params, "experiment")
    return {
        "vendor": "jeol",
        "pulse_program": experiment["value"] if experiment else None,
        "ndim": int(header.get("dataDimensionNumber", 1)),
        "shape": (int(fid.shape[0]),),
        "nucleus": _normalize_nucleus(x_domain["value"] if x_domain else None),
        "nucleus_indirect": None,
        "pseudo_2d": False,
    }


def _peek_2d(r: _Reader, header: dict, params: list[dict]) -> dict:
    """:func:`peek` for a 2D file: the acquired shape (increments or rows
    x direct points after the digital filter), from the header and one
    row, without the transforms."""
    complex_y = header["dataAxisType"][1] == "Complex"
    y_domain = _get_param(params, "Y_DOMAIN")
    nucleus_indirect = _normalize_nucleus(y_domain["value"] if y_domain else None) or None
    pseudo = (not complex_y) or not nucleus_indirect
    n1 = min(header["dataPoints"][1], header["dataOffsetStop"][1] + 1)
    n2 = min(header["dataPoints"][0], header["dataOffsetStop"][0] + 1)
    filter_factor = _get_param(params, "FILTER_FACTOR")
    if filter_factor is not None and filter_factor["value"]:
        n2 = digital_filter.rm_dig_filter(
            np.zeros(n2, dtype=np.complex128), decim=0, dspfvs=0,
            grpdly=float(filter_factor["value"])).shape[-1]
    x_domain = _get_param(params, "X_DOMAIN")
    experiment = _get_param(params, "experiment")
    return {
        "vendor": "jeol",
        "pulse_program": experiment["value"] if experiment else None,
        "ndim": 2,
        "shape": (n1, n2),
        "nucleus": _normalize_nucleus(x_domain["value"] if x_domain else None),
        "nucleus_indirect": None if pseudo else nucleus_indirect,
        "pseudo_2d": bool(pseudo),
    }


base.register("jeol", _detect, _read_auto, peek=peek)
