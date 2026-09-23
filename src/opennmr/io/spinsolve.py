"""
Spinsolve (Magritek benchtop) importer -- PGSTE/PFG diffusion datasets.

Standalone for now: nothing here is registered with :mod:`opennmr.io.base`
and nothing in the GUI calls it, so it can be merged with Euan's branch
without conflicting. Wiring it in later is one ``base.register()`` call
plus an :class:`~opennmr.core.spectrum.NMRSpectrum` adapter.

A Spinsolve PGSTE experiment folder (``<root>/<year>/<month>/<sample>/
<protocol>_<n>/``) holds the same matrix four times over, plus the
diffusion axis in two equivalent forms:

===========================  ================================================
``acqu.par``                 acquisition parameters, ``key = value`` text
``display.par``              what the Spinsolve UI last showed: phase,
                             integral regions, apodization
``data.2d``                  **the raw FID** -- Prospa V1.1 binary,
                             nrSteps x nrPnts complex float32
``data.1d``                  one row of it (whichever step the UI had
                             selected), Prospa V1.1 with a time axis
``data.2d.txt``              DOSY Toolbox ASCII header: diffusion
                             parameters and the **gradient amplitude
                             array** in T/m
``data.2d.bin``              the same matrix as big-endian int64 in
                             units of 1e-4 (companion to the .txt)
``data_enhanced.2d[.txt]``   the same matrix after Spinsolve's own
                             apodization + drift correction -- derived,
                             not raw
``delayTimes.txt``           the **b-values** (s/m^2), one per step,
                             written with a comma decimal separator
``phaseShifts.txt``          per-step ``ph0;ph1;autophase;<n>``
``fit.csv``                  Spinsolve's own D from its integral region
===========================  ================================================

The name ``delayTimes.txt`` is a misnomer inherited from the T1/T2
protocols: for PGSTE it holds b = gamma^2 g^2 delta^2 (DELTA - delta/3),
which this reader checks against the gradient array in ``data.2d.txt``.

Verified against ``test_data/Spinsolve_PFG/2026/08/AL-70A-3/PGSTE CDEC_1``:
reading ``data.2d``, apodizing by the 1 Hz exponential in ``display.par``,
zero-filling 4x, FTing and integrating ``display.par``'s own region
reproduces ``fit.csv`` (D = 1.339e-10 vs 1.338e-10 m^2/s, I0 = 82.2 vs
82.1) -- see :func:`selftest`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..processing import diffusion as _diffusion
from ..processing import ft as _ft
from ..processing import relaxation as _relaxation


# ============================================================
# ``key = value`` parameter files (acqu.par, display.par)
# ============================================================

#: An acqu.par value is a quoted string, an int, or a float. Spinsolve
#: does not escape quotes inside a quoted string -- graphTitle is
#: literally ``"1D-1H-...-"Off"-...-"Linear q2"-1"`` -- so only the
#: outermost pair is stripped, never an inner one.
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _coerce(text: str) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def read_par(path: str) -> dict[str, Any]:
    """Parse a Spinsolve ``*.par`` file into a dict, typing each value.

    Handles both files: ``acqu.par`` keys can start with a digit
    (``90PulseLength``), and ``display.par`` can carry blank values and
    even a stray key-less ``= 0`` line, which is skipped rather than
    raising.
    """
    params: dict[str, Any] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            params[key] = _coerce(value)
    return params


# ============================================================
# Prospa V1.1 binary (data.1d, data.2d)
# ============================================================

#: The 32-byte header is 12 magic bytes then five int32. The magic is
#: three 4-char tags -- "PROS", "DATA", "V1.1" -- each stored
#: byte-reversed, so on disk it reads ``SORPATAD1.1V``.
_PROSPA_MAGIC = b"SORPATAD1.1V"

#: Prospa ``dataType`` codes, as (numpy dtype, floats per point,
#: has a leading real x-axis column).
_PROSPA_TYPES = {
    500: ("<f4", 1, False),   # real float32
    501: ("<f4", 2, False),   # complex float32          <- data.2d
    502: ("<f8", 1, False),   # real float64
    503: ("<f8", 2, False),   # complex float64
    504: ("<f4", 2, True),    # x-axis + complex float32 <- data.1d
}


@dataclass
class ProspaFile:
    """One Prospa V1.1 binary: its declared shape and its data."""

    data_type: int
    #: (nx, ny, nz, nq) exactly as the header declares it.
    dims: tuple[int, int, int, int]
    #: Complex (or real) data, shaped (ny, nx) for a 2D file and (nx,)
    #: for a 1D one.
    data: np.ndarray
    #: The x-axis column, for dataType 504 only (seconds, for a FID).
    x: np.ndarray | None = None


def read_prospa(path: str) -> ProspaFile:
    """Read a Prospa V1.1 binary (``data.1d``, ``data.2d``, ...)."""
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw[:12] != _PROSPA_MAGIC:
        raise ValueError(f"{path!r} is not a Prospa V1.1 file (magic {raw[:12]!r})")

    data_type, nx, ny, nz, nq = np.frombuffer(raw[12:32], dtype="<i4")
    data_type = int(data_type)
    if data_type not in _PROSPA_TYPES:
        raise ValueError(f"{path!r}: unsupported Prospa dataType {data_type}")
    dtype, per_point, has_x = _PROSPA_TYPES[data_type]

    flat = np.frombuffer(raw[32:], dtype=dtype)
    npoints = int(nx) * int(ny) * int(nz) * int(nq)
    expected = npoints * (per_point + (1 if has_x else 0))
    if flat.size != expected:
        raise ValueError(
            f"{path!r}: expected {expected} values for dataType {data_type} "
            f"and dims {(nx, ny, nz, nq)}, found {flat.size}"
        )

    x = None
    if has_x:
        x = np.array(flat[:npoints], dtype=float)
        flat = flat[npoints:]
    if per_point == 2:
        pairs = flat.reshape(npoints, 2)
        data = (pairs[:, 0] + 1j * pairs[:, 1]).astype(complex)
    else:
        data = np.array(flat, dtype=float)

    # nz/nq are 1 in every Spinsolve file seen; keep the matrix 2D when
    # ny > 1 and 1D otherwise rather than carrying trailing 1-axes.
    if int(ny) > 1:
        data = data.reshape(int(ny), int(nx))
        if x is not None:
            x = x.reshape(int(ny), int(nx))[0]
    return ProspaFile(data_type, (int(nx), int(ny), int(nz), int(nq)), data, x)


# ============================================================
# DOSY Toolbox pair (data.2d.txt header + data.2d.bin)
# ============================================================

#: ``data.2d.bin`` stores the matrix as big-endian int64 in units of
#: 1e-4. Verified against ``data.2d``: the two agree to exactly half a
#: quantum (5e-5), i.e. the .bin is round(value * 1e4) of the same
#: underlying numbers the .2d holds as float32.
DOSY_BIN_SCALE = 1e4

#: ``#Name (type ; units) value`` -- and, for an array,
#: ``#Name [64] (...)`` followed by that many numbers on their own lines.
_DOSY_KEY_RE = re.compile(r"^#(?P<name>[^(\[]+?)\s*(?:\[(?P<count>\d+)\])?\s*\((?P<spec>[^)]*)\)\s*(?P<value>.*?)\s*$")


@dataclass
class DosyHeader:
    """The ASCII half of a DOSY Toolbox pair (``data.2d.txt``)."""

    #: Scalar entries, keyed by their name as written ("Diffusion Delay").
    params: dict[str, Any] = field(default_factory=dict)
    #: Array entries ("Gradient Amplitude", "Left Phase Array", ...).
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


def read_dosy_header(path: str) -> DosyHeader:
    """Parse ``data.2d.txt`` / ``data_enhanced.2d.txt``.

    Only the header is parsed; an "Actual Data Points" section, if the
    file carries one instead of pointing at a ``.bin``, is left alone.
    """
    header = DosyHeader()
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.startswith("#") or line.startswith("##"):
            continue
        match = _DOSY_KEY_RE.match(line)
        if match is None:
            continue
        name = match.group("name").strip()
        count = match.group("count")
        if count is not None:
            values, wanted = [], int(count)
            while i < len(lines) and len(values) < wanted:
                row = lines[i].strip()
                i += 1
                if not row or row.startswith("#"):
                    break
                values.extend(float(tok) for tok in row.split())
            header.arrays[name] = np.array(values, dtype=float)
        else:
            value = match.group("value").strip()
            header.params[name] = _coerce(value) if value else None
    return header


def read_dosy_bin(path: str, nrows: int, npoints: int,
                  scale: float = DOSY_BIN_SCALE) -> np.ndarray:
    """Read ``data.2d.bin`` as a complex ``(nrows, npoints)`` matrix.

    Unlike the interleaved Prospa file, this one is blocked: every real
    part first, then every imaginary part, big-endian int64, scaled by
    ``scale``.
    """
    flat = np.fromfile(path, dtype=">i8").astype(float) / scale
    expected = 2 * nrows * npoints
    if flat.size != expected:
        raise ValueError(f"{path!r}: expected {expected} int64s, found {flat.size}")
    real, imag = flat[: flat.size // 2], flat[flat.size // 2:]
    return (real + 1j * imag).reshape(nrows, npoints)


# ============================================================
# The small sidecar lists
# ============================================================

def _parse_localised_float(text: str) -> float:
    """Spinsolve writes ``delayTimes.txt`` in the acquiring PC's locale,
    so a value can arrive as ``2,600749E+07``. Treat a lone comma as a
    decimal point; treat commas alongside a dot as thousands separators.
    """
    text = text.strip()
    if "," in text:
        text = text.replace(",", "") if "." in text else text.replace(",", ".")
    return float(text)


def read_delay_times(path: str) -> np.ndarray:
    """Read ``delayTimes.txt``: for PGSTE, the b-values in s/m^2."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return np.array([_parse_localised_float(line) for line in handle if line.strip()])


def read_phase_shifts(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read ``phaseShifts.txt`` -> (ph0 in degrees, ph1 in degrees), one
    per step. The trailing ``True;5`` fields (autophase flag and its
    mode) are not used here."""
    ph0, ph1 = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.split(";")
            ph0.append(float(fields[0]))
            ph1.append(float(fields[1]) if len(fields) > 1 and fields[1].strip() else 0.0)
    return np.array(ph0), np.array(ph1)


def read_fit_csv(path: str) -> list[dict[str, Any]]:
    """Read Spinsolve's own ``fit.csv`` (``Name,Start,End,A,D``) -- one
    dict per integral region, for cross-checking :meth:`SpinsolvePFG.fit`.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        rows = [line.strip() for line in handle if line.strip()]
    if not rows:
        return []
    keys = [k.strip() for k in rows[0].split(",")]
    return [dict(zip(keys, (_coerce(v) for v in row.split(",")))) for row in rows[1:]]


def read_integral_regions(display: dict[str, Any]) -> list[tuple[float, float]]:
    """Pull the integral regions out of ``display.par``'s ``Integrals``
    field -- ``{start|end|...}{start|end|...}`` in ppm, high ppm first.
    """
    text = str(display.get("Integrals", "") or "")
    regions = []
    for block in re.findall(r"\{([^}]*)\}", text):
        fields = block.split("|")
        if len(fields) >= 2:
            try:
                start, end = float(fields[0]), float(fields[1])
            except ValueError:
                continue
            regions.append((min(start, end), max(start, end)))
    return regions


# ============================================================
# The dataset
# ============================================================

@dataclass
class SpinsolvePFG:
    """One Spinsolve PGSTE/PFG experiment folder."""

    path: str
    #: Raw FID, (nsteps, npoints) complex, from ``data.2d``.
    fid: np.ndarray
    #: Diffusion weighting b = gamma^2 g^2 delta^2 (DELTA - delta/3), s/m^2.
    b: np.ndarray
    #: Gradient amplitude per step, T/m (None if ``data.2d.txt`` is absent).
    gradients: np.ndarray | None
    dwell: float            #: s
    b1_freq: float          #: MHz
    lowest_frequency: float #: Hz, the low-ppm edge of the spectrum
    nucleus: str
    #: Hz/T -- the convention :mod:`opennmr.processing.diffusion` uses.
    #: ``data.2d.txt`` states it as rad s^-1 T^-1 and is converted on read.
    gamma: float
    delta: float            #: s, the gradient pulse (incl. ramp)
    big_delta: float        #: s
    acqu: dict[str, Any] = field(default_factory=dict)
    display: dict[str, Any] = field(default_factory=dict)
    dosy: DosyHeader | None = None
    ph0: np.ndarray | None = None
    ph1: np.ndarray | None = None
    #: Spinsolve's own fitted D, for cross-checking.
    reference_fit: list[dict[str, Any]] = field(default_factory=list)

    # -- axes ------------------------------------------------

    @property
    def nsteps(self) -> int:
        return self.fid.shape[0]

    @property
    def npoints(self) -> int:
        return self.fid.shape[1]

    @property
    def sweep_width(self) -> float:
        """Hz."""
        return 1.0 / self.dwell

    @property
    def time(self) -> np.ndarray:
        """Acquisition time axis, s."""
        return np.arange(self.npoints) * self.dwell

    def ppm_axis(self, zero_fill: int | None = None) -> np.ndarray:
        """Chemical shift axis for :meth:`spectra`, **descending** --
        the convention every other vendor in this package uses, and the
        one :func:`opennmr.processing.ft.fft_spectrum` builds.

        ``acqu.par``'s ``lowestFrequency`` is the low-ppm edge of the
        window, so the centre is that plus half the sweep width; the
        rest is whatever ``fft_spectrum`` says, rather than a second
        hand-rolled axis that could drift from it.
        """
        n = self._ft_size(zero_fill)
        centre_hz = self.lowest_frequency + self.sweep_width / 2.0
        freq_hz, _ = _ft.fft_spectrum(np.zeros(n), self.dwell)
        return (freq_hz + centre_hz) / self.b1_freq

    def _ft_size(self, zero_fill: int | None) -> int:
        if zero_fill is None:
            zero_fill = int(self.acqu.get("zeroFill", 1) or 1)
        return self.npoints * max(int(zero_fill), 1)

    # -- processing ------------------------------------------

    def spectra(self, line_broadening: float | None = None,
                zero_fill: int | None = None,
                phase: bool = True) -> np.ndarray:
        """FT every step -> ``(nsteps, n)`` complex, on :meth:`ppm_axis`.

        ``line_broadening`` (Hz) defaults to the exponential apodization
        recorded in ``display.par``; ``zero_fill`` to ``acqu.par``'s
        ``zeroFill`` factor. Both reproduce what the instrument showed.

        **The transform is the shared one**, :func:`opennmr.processing.
        ft.fft_spectrum`, not a private one. That matters beyond reuse:
        its frequency axis descends, which is the same mirroring this
        format needs and which a bare forward transform on an ascending
        axis gets backwards (the peak lands at 9.25 ppm instead of
        0.22). Going through it means the GUI's live processing preview,
        which reprocesses the cached raw FID through exactly these
        functions, reproduces this output rather than its complex
        conjugate one bin over -- which is what an earlier hand-rolled
        ``fft(conj(fid))`` here did, and why Spinsolve data had no
        preview at all.

        Scaling by ``1/n`` matches Spinsolve's intensities, so an
        integral here is directly comparable with ``fit.csv``'s A.
        """
        if line_broadening is None:
            line_broadening = self.display_line_broadening()
        n = self._ft_size(zero_fill)

        fid = self.fid
        if line_broadening:
            fid = _ft.apodize(fid.astype(complex), self.dwell,
                              window="exponential", lb=line_broadening)
        _, spec = _ft.fft_spectrum(_ft.zero_fill(fid.astype(complex), n), self.dwell)
        spec = spec / n

        if phase:
            ph0 = self.ph0 if self.ph0 is not None else np.zeros(self.nsteps)
            ph0 = ph0 + float(self.display.get("Phase0", 0.0) or 0.0)
            spec = spec * np.exp(1j * np.deg2rad(ph0))[:, None]
        return spec

    def display_line_broadening(self) -> float:
        """The exponential apodization ``display.par`` last had applied,
        in Hz (0.0 if none).

        ``display.par``'s ``SNR`` field is a JSON list of apodization
        entries, one per type, each flagged applied or not.
        """
        text = str(self.display.get("SNR", "") or "").strip()
        if not text:
            return 0.0
        try:
            entries = json.loads(text)
        except ValueError:
            return 0.0
        for entry in entries:
            if entry.get("ApodizationType") == "Exponential" and entry.get("IsApplied"):
                return float(entry.get("Value", 0.0))
        return 0.0

    def integrate(self, ppm_range: tuple[float, float], **kwargs) -> np.ndarray:
        """Real-part integral of ``ppm_range`` for every step."""
        low, high = min(ppm_range), max(ppm_range)
        ppm = self.ppm_axis(kwargs.get("zero_fill"))
        window = (ppm >= low) & (ppm <= high)
        if not window.any():
            raise ValueError(f"no points between {low} and {high} ppm")
        return self.spectra(**kwargs)[:, window].real.sum(axis=1)

    # -- the fit ---------------------------------------------

    def fit(self, ppm_range: tuple[float, float] | None = None,
            n_components: int = 1, stretched: bool = False,
            expected_D: float | None = None, **kwargs) -> dict:
        """Fit I(b) = sum_i I_i exp[-(D_i b)^beta_i] over ``ppm_range``.

        The fit itself is :mod:`opennmr.processing.diffusion`'s -- the
        same engine the Bruker and X-Pulse PFG paths use -- so a
        Spinsolve dataset gets multi-component and stretched fits,
        log-space D searching and typed uncertainties for free. Only
        the b-axis and the integrals are this reader's own work.

        ``ppm_range`` defaults to the first integral region in
        ``display.par``, which is the one Spinsolve fitted in
        ``fit.csv``. Returns :func:`~opennmr.processing.relaxation.
        fit_relaxation`'s result dict, with D in m^2/s.
        """
        if ppm_range is None:
            regions = read_integral_regions(self.display)
            if not regions:
                raise ValueError("no integral region in display.par; pass ppm_range=")
            ppm_range = regions[0]

        intensity = self.integrate(ppm_range, **kwargs)
        model = _diffusion.build_diffusion_model(
            n_components=n_components, stretched=stretched, expected_D=expected_D,
        )
        result = _relaxation.fit_relaxation(
            self.b, intensity, model="diffusion", custom_model=model,
        )
        result["ppm_range"] = (float(min(ppm_range)), float(max(ppm_range)))
        return result

    def b_from_gradients(self) -> np.ndarray:
        """Recompute b from the gradient array, as a check on
        ``delayTimes.txt``. Raises if the gradients are unavailable.

        PGSTE's weighting is the plain Stejskal-Tanner one, and
        ``data.2d.txt`` reports the gradient shape as "Square", so this
        is ``sequence="pgse"``/``shape="rect"`` -- no bipolar or
        smoothed-square correction applies.
        """
        if self.gradients is None:
            raise ValueError("no gradient array (data.2d.txt missing)")
        return _diffusion.b_value(
            self.gradients, self.delta, self.big_delta,
            gamma=self.gamma, shape="rect", sequence="pgse",
        )


# ============================================================
# Reading a folder
# ============================================================

def is_spinsolve_pfg(path: str) -> bool:
    """True if ``path`` is a Spinsolve folder holding an arrayed
    (PGSTE/T1/T2) experiment -- a 2D matrix plus a delay list."""
    if not os.path.isdir(path):
        return False
    return all(os.path.isfile(os.path.join(path, name))
               for name in ("acqu.par", "data.2d", "delayTimes.txt"))


def read_spinsolve_pfg(path: str, source: str = "data.2d") -> SpinsolvePFG:
    """Read a Spinsolve PGSTE experiment folder.

    ``source`` picks which copy of the matrix to load: ``"data.2d"``
    (the raw FID, the default), ``"data.2d.bin"`` (the DOSY Toolbox
    copy of the same numbers), or ``"data_enhanced.2d"`` (Spinsolve's
    already-apodized version -- derived, so don't apodize it again).
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(path)

    def _maybe(name: str) -> str | None:
        full = os.path.join(path, name)
        return full if os.path.isfile(full) else None

    acqu_path = _maybe("acqu.par")
    if acqu_path is None:
        raise FileNotFoundError(f"{path!r} has no acqu.par")
    acqu = read_par(acqu_path)
    display_path = _maybe("display.par")
    display = read_par(display_path) if display_path else {}

    # -- the diffusion axis, from the DOSY Toolbox header where it
    # exists (it carries gamma, delta and the gradient list, none of
    # which acqu.par states directly -- acqu.par's littleDelta omits
    # the gradient ramp that the sequence actually plays).
    dosy_path = _maybe("data.2d.txt")
    dosy = read_dosy_header(dosy_path) if dosy_path else None
    gradients = dosy.arrays.get("Gradient Amplitude") if dosy else None

    nucleus = str(acqu.get("rxChannel", "1H"))
    # data.2d.txt states gamma in rad s^-1 T^-1; everything downstream
    # of here works in Hz/T. Prefer the file's own value over the table
    # so a reprocessed dataset stays self-consistent.
    if dosy and "Dosygamma" in dosy.params:
        gamma = float(dosy.params["Dosygamma"]) / (2 * np.pi)
    else:
        gamma = _diffusion.gamma_for(nucleus)
    if dosy and "Diffusion Encoding Time" in dosy.params:
        delta = float(dosy.params["Diffusion Encoding Time"])
        big_delta = float(dosy.params["Diffusion Delay"])
    else:
        # ms in acqu.par, and littleDelta there is the plateau only --
        # the ramp (gradRamp) is played as part of the encoding time.
        delta = (float(acqu.get("littleDelta", 0)) + float(acqu.get("gradRamp", 0))) * 1e-3
        big_delta = float(acqu.get("bigDelta", 0)) * 1e-3

    # -- the matrix
    if source == "data.2d.bin":
        nsteps = int(acqu.get("nrSteps", 1))
        npoints = int(acqu.get("nrPnts", 0))
        fid = read_dosy_bin(os.path.join(path, "data.2d.bin"), nsteps, npoints)
    else:
        prospa = read_prospa(os.path.join(path, source))
        fid = np.atleast_2d(prospa.data)

    # -- b-values: prefer the instrument's own list, fall back to the
    # gradient array. When both exist, disagreement means one of the two
    # was edited by hand, which is worth surfacing rather than averaging.
    delay_path = _maybe("delayTimes.txt")
    b = read_delay_times(delay_path) if delay_path else None
    if b is None:
        if gradients is None:
            raise FileNotFoundError(f"{path!r} has neither delayTimes.txt nor a gradient array")
        b = _diffusion.b_value(gradients, delta, big_delta, gamma=gamma,
                               shape="rect", sequence="pgse")
    if b.size != fid.shape[0]:
        raise ValueError(f"{b.size} b-values for {fid.shape[0]} rows in {path!r}")

    phase_path = _maybe("phaseShifts.txt")
    ph0, ph1 = read_phase_shifts(phase_path) if phase_path else (None, None)
    fit_path = _maybe("fit.csv")

    return SpinsolvePFG(
        path=path,
        fid=fid,
        b=b,
        gradients=gradients,
        dwell=float(acqu["dwellTime"]) * 1e-6,
        b1_freq=float(acqu["b1Freq"]),
        lowest_frequency=float(acqu["lowestFrequency"]),
        nucleus=nucleus,
        gamma=gamma,
        delta=delta,
        big_delta=big_delta,
        acqu=acqu,
        display=display,
        dosy=dosy,
        ph0=ph0,
        ph1=ph1,
        reference_fit=read_fit_csv(fit_path) if fit_path else [],
    )


def find_spinsolve_experiments(root: str) -> list[str]:
    """Walk a Spinsolve data tree (``<root>/<year>/<month>/<sample>/...``)
    and return every arrayed experiment folder in it."""
    found = []
    for dirpath, _dirnames, _filenames in os.walk(root):
        if is_spinsolve_pfg(dirpath):
            found.append(dirpath)
    return sorted(found)


# ============================================================
# Self-check against the instrument's own numbers
# ============================================================

def selftest(path: str) -> dict[str, Any]:
    """Read ``path`` and compare against what Spinsolve itself wrote:
    the b-values against the gradient array, and D/I0 against
    ``fit.csv``. Returns the comparison; raises nothing, so it can be
    pointed at a new folder to see whether this reader still holds.
    """
    data = read_spinsolve_pfg(path)
    result: dict[str, Any] = {
        "path": path,
        "shape": data.fid.shape,
        "nucleus": data.nucleus,
        "delta_s": data.delta,
        "big_delta_s": data.big_delta,
        "b_range": (float(data.b[0]), float(data.b[-1])),
        "line_broadening_hz": data.display_line_broadening(),
    }
    if data.gradients is not None:
        recomputed = data.b_from_gradients()
        result["b_vs_gradients_max_rel_diff"] = float(
            np.max(np.abs(recomputed - data.b) / np.abs(data.b))
        )
    fit = data.fit()
    result["fit"] = {
        "D": fit["params"]["D"], "I0": fit["params"]["I0"],
        "D_stderr": fit["errors"]["D"], "ppm_range": fit["ppm_range"],
        "rmse": fit["fit_quality"]["rmse"],
    }
    if data.reference_fit:
        reference = data.reference_fit[0]
        result["spinsolve_fit"] = reference
        result["D_rel_diff"] = abs(fit["params"]["D"] - float(reference["D"])) / float(reference["D"])
        result["I0_rel_diff"] = abs(fit["params"]["I0"] - float(reference["A"])) / float(reference["A"])
    return result


if __name__ == "__main__":  # pragma: no cover -- a look at one folder
    import sys
    from pprint import pprint

    for folder in (sys.argv[1:] or find_spinsolve_experiments(".")):
        pprint(selftest(folder))


# ============================================================
# NMRSpectrum adapter and registration (wired in 2026-09-23)
# ============================================================
#
# TODO.md section 0, seam 3: join ``io.load()``'s auto-detection. The
# reader above is unchanged; this only presents its result the way every
# other vendor does, so the GUI and ``opennmr.io.load`` open it with no
# Spinsolve-specific code of their own.

def is_spinsolve_diffusion(path: str) -> bool:
    """True only for a Spinsolve **diffusion** (PGSTE/PFG) folder.

    :func:`is_spinsolve_pfg` also matches Spinsolve's T1/T2 arrays --
    same files, but there ``delayTimes.txt`` holds *delays*, not
    b-values. Registering on that detector alone would open a T1 series
    labelled as diffusion, with its delays read as b-values. So a folder
    is claimed only when it carries something only a diffusion protocol
    writes: the DOSY Toolbox header (``data.2d.txt``), or ``acqu.par``'s
    ``bigDelta``/``littleDelta``. A T1/T2 folder is left unrecognised
    (openable by nobody) rather than opened wrong.

    The acqu.par criterion is inferred from the PGSTE dataset's keys;
    no Spinsolve T1/T2 folder has been available to confirm that those
    protocols lack them.
    """
    if not is_spinsolve_pfg(path):
        return False
    if os.path.isfile(os.path.join(path, "data.2d.txt")):
        return True
    try:
        acqu = read_par(os.path.join(path, "acqu.par"))
    except OSError:
        return False
    return "bigDelta" in acqu and "littleDelta" in acqu


def read_spectrum(path: str, source: str = "data.2d",
                  line_broadening: float | None = None,
                  zero_fill: int | None = None, **_ignored):
    """Read a Spinsolve PGSTE folder as a pseudo-2D
    :class:`~opennmr.core.spectrum.NMRSpectrum` (one row per gradient
    step), processed exactly as :meth:`SpinsolvePFG.spectra` does --
    the instrument's own apodization and zero-fill by default, the
    conjugate FT, per-step phase -- so it is the spectrum the validated
    diffusion fit was checked on.

    The ppm axis is **descending**, as every other vendor's is --
    :meth:`SpinsolvePFG.spectra` now produces it that way through the
    shared :func:`opennmr.processing.ft.fft_spectrum`, so no reversal
    happens here.

    What the GUI and notebook helpers need for a diffusion fit travels
    in ``metadata.raw``:

    ``b_values``       s/m^2, one per row -- ``delayTimes.txt``
    ``b_source``       where those came from, as a sentence
    ``pfg_params``     ``delta``/``DELTA`` (s), ``sequence``, ``shape``,
                       ``gamma`` (Hz/T), ``gradient`` (T/m, or None)
    ``processing``     the apodization/zero-fill actually applied
    ``reference_fit``  Spinsolve's own ``fit.csv`` rows, to compare with

    ``fid`` / ``fid_raw`` / ``fid_dt``  the raw time-domain rows, so the
                       Processing tab's live preview works

    Those three were once withheld, on the reasoning that the generic
    reprocessing path could not apply this format's conjugate and would
    mirror the peak. Measured rather than assumed, that turned out to be
    wrong in one direction and right in another: ``fft_spectrum``'s
    frequency axis already descends, so the shift is **not** mirrored --
    but the old hand-rolled ``fft(conj(fid))`` here produced the complex
    *conjugate* of it, one bin over, so a phase drag in the preview
    would have rotated opposite to the Spectrum tab. Both paths now go
    through ``fft_spectrum``, so they agree, and the FID can be cached.
    Other keyword arguments (e.g. processing a caller passes for every
    vendor) are ignored.
    """
    from ..core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum

    pfg = read_spinsolve_pfg(path, source=source)
    applied_lb = pfg.display_line_broadening() if line_broadening is None else float(line_broadening)
    applied_zf = int(pfg.acqu.get("zeroFill", 1) or 1) if zero_fill is None else int(zero_fill)
    spec = pfg.spectra(line_broadening=applied_lb, zero_fill=applied_zf)
    ppm = pfg.ppm_axis(applied_zf)

    b_source = ("delayTimes.txt (Spinsolve's own b-values; despite the name they are "
                "b in s/m^2 for PGSTE)")
    if pfg.gradients is not None:
        try:
            recomputed = pfg.b_from_gradients()
            rel = float(np.max(np.abs(recomputed - pfg.b)) / np.max(np.abs(pfg.b)))
            b_source += f"; agrees with data.2d.txt's gradient list to {rel:.1e} relative"
        except (ValueError, ZeroDivisionError):
            pass

    metadata = NMRMetadata(
        vendor="spinsolve",
        nucleus=pfg.nucleus,
        sfo1=pfg.b1_freq,
        bf1=pfg.b1_freq,
        ns=pfg.acqu.get("nrScans"),
        source_path=path,
        pseudo_2d=True,
        pulse_program=os.path.basename(os.path.normpath(path)) or None,
        raw={
            "b_values": np.asarray(pfg.b, dtype=float),
            "b_source": b_source,
            "is_diffusion": True,
            "pfg_params": {
                "delta": pfg.delta, "DELTA": pfg.big_delta,
                "sequence": "pgse", "shape": "rect",
                "gamma": pfg.gamma, "gradient": pfg.gradients,
                "sources": {
                    "delta": ("data.2d.txt 'Diffusion Encoding Time'" if pfg.dosy is not None
                              else "acqu.par littleDelta + gradRamp"),
                    "DELTA": ("data.2d.txt 'Diffusion Delay'" if pfg.dosy is not None
                              else "acqu.par bigDelta"),
                },
            },
            "processing": {"line_broadening_hz": applied_lb, "zero_fill": applied_zf,
                           "source": source},
            # Both keys, as every other vendor stores them: "fid_raw"
            # is what the Processing tab reprocesses from, "fid" what
            # the FID preview draws. Untouched rows -- apodization and
            # zero-filling are the live controls' job, not the reader's.
            "fid": np.ascontiguousarray(pfg.fid),
            "fid_raw": np.ascontiguousarray(pfg.fid),
            "fid_dt": pfg.dwell,
            "reference_fit": pfg.reference_fit,
            "acqu": pfg.acqu,
        },
    )
    return NMRSpectrum(
        data=np.ascontiguousarray(spec.real),
        axes=[NMRAxis(kind="index", index=np.arange(1, pfg.nsteps + 1), label="gradient step"),
              NMRAxis(kind="frequency", ppm=ppm, hz=ppm * pfg.b1_freq)],
        metadata=metadata,
        data_imag=np.ascontiguousarray(spec.imag),
    )


def peek(path: str, **_ignored) -> dict[str, Any]:
    """Shape and nucleus from ``acqu.par`` alone, without reading the
    matrix -- what the folder browser needs to list a dataset."""
    acqu = read_par(os.path.join(path, "acqu.par"))
    zero_fill = int(acqu.get("zeroFill", 1) or 1)
    return {
        "vendor": "spinsolve", "ndim": 2,
        "shape": (int(acqu.get("nrSteps", 0)), int(acqu.get("nrPnts", 0)) * max(zero_fill, 1)),
        "nucleus": str(acqu.get("rxChannel", "1H")), "nucleus_indirect": None,
        "pseudo_2d": True,
        "pulse_program": os.path.basename(os.path.normpath(path)) or None,
    }


from . import base as _base  # noqa: E402 -- registration needs the functions above

_base.register("spinsolve", is_spinsolve_diffusion, read_spectrum, peek=peek)
