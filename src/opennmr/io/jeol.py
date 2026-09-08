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
    h["dataUnits"] = [u["base"] for u in _get_unit(r, 8)]
    h["title"] = r.string(124)

    r.skip(4)  # dataAxisRanged (2 axes packed per byte) -- unused here
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
    r.skip(32 * 8)  # dataAxisTitles

    r.skip(8 * 8)  # baseFreq
    r.skip(8 * 8)  # zeroPoint
    r.skip(8)  # reversed
    r.skip(3)
    r.skip(1)  # annotationOK
    r.skip(4)  # historyUsed
    r.skip(4)  # historyLength
    h["paramStart"] = r.u32()
    r.skip(4)  # paramLength
    r.skip(8 * 4)  # ListStart
    r.skip(8 * 4)  # ListLength
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


def _read_jeol_jdf(path: str) -> dict:
    """Parse a ``.jdf`` file's header, parameters, and 1D FID/spectrum
    data -- the JEOL-format-specific piece; :func:`read_1d` handles
    everything vendor-agnostic (windowing, FFT, phasing, ...) from
    here on, same split as :func:`opennmr.io.xpulse._read_xpulse_jcamp`."""
    with open(path, "rb") as f:
        buf = f.read()
    r = _Reader(buf)
    header = _parse_header(r)
    if header["fileIdentifier"] != "JEOL.NMR":
        raise ValueError(f"{path!r} is not a JEOL .jdf file (unexpected file identifier).")
    if header["dataFormat"] != "One_D":
        raise ValueError(
            f"read_1d only supports 1D JEOL data so far, got dataFormat={header['dataFormat']!r} "
            f"-- 2D (COSY/HSQC/HMBC/...) JEOL support isn't built yet, see TODO.md."
        )
    params = _parse_parameters(r, header)
    data = _read_data_1d(r, header)
    return {"header": header, "params": params, "data": data}


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


base.register("jeol", _detect, read_1d)
