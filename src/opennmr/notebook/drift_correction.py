"""
Linear magnetic-field drift correction for multidimensional datasets.

A port of the two TopSpin AU programs published with

    E. E. Najbauer and L. B. Andreas, "Correcting for magnetic field
    drift in magic-angle spinning NMR datasets", J. Magn. Reson. 305
    (2019) 1-4. https://doi.org/10.1016/j.jmr.2019.05.005

(``lindriftcomp``, 30.01.2019, and ``concatenate_ser_nuslist``,
10.04.2019). Cite that paper when you use this. Explicit user request
(2026-10-01): field drift correction in the Process tab, translated
from those programs so it can be applied here.

The idea
--------
A superconducting magnet's field drifts, and without a lock (as in most
MAS experiments) every peak moves with it. Over the ~12-36 h of a
multidimensional experiment the drift is close to linear, so: measure a
reference peak in a 1D spectrum before and after the nD experiment;
the difference from a reference position gives a *start* and an *end*
shift in Hz of the detected nucleus; interpolate linearly over the FIDs
in the order they were acquired; and undo each FID's shift in the time
domain.

For an FID acquired when the field had moved the detected nucleus by
``Δ`` Hz:

* **direct dimension**: multiply the FID by ``exp(i 2π Δ t2)``, with t2
  measured from the true start of the FID (after the digital filter's
  group delay);
* **each indirect dimension** ``d``: the same field change moves that
  nucleus by ``Δ · BF_d / BF_direct`` (the field shifts every nucleus by
  the same fraction -- the same physics as transferring an SR, see
  :mod:`opennmr.notebook.referencing`), which at evolution time ``t_d``
  is a phase ``θ = 2π Δ (BF_d/BF_dir) t_d`` -- applied as a rotation
  between the cosine- and sine-modulated FIDs (States, States-TPPI),
  as opposite phases on the echo and antiecho FIDs (Echo-Antiecho), or
  as a phase on the FID (QF). A pseudo-2D's indirect axis is not an
  evolution time, and gets the direct-dimension correction only.

The sign convention is the AU program's: **a positive shift moves
peaks to the left (higher frequency)**. Enter ``reference − observed``,
in Hz of the detected nucleus: a peak that has drifted 3 Hz below where
it belongs needs +3 Hz. :func:`measure_drift_hz` returns exactly that.

What differs from the AU programs, and why
------------------------------------------
* **The last FID gets the end shift exactly.** The AU program applies
  ``start + (end − start)·p/P`` to FID ``p`` of ``P``, so its last FID
  gets one increment's worth less than the end value it was given --
  despite its own description ("the end shift is applied to the last
  acquired FID"). Here it is ``p/(P−1)``. ``endpoint=False`` reproduces
  the AU numbers exactly.
* **One elapsed-time index everywhere.** In the AU's 3D *points* mode
  the direct-dimension phase uses only the slow loop counter while the
  indirect dimensions use the full elapsed index -- inconsistent within
  one FID. Here every dimension uses the elapsed index.
* **QF pseudo-2D**: the AU's 2D branch writes ``arg2 == 0`` (a
  comparison, a no-op) where its 3D/4D branches write ``arg2 = 0``, so
  on a QF 2D with a non-zero IN_F its indirect rotation is *not*
  switched off and mixes row pairs -- on a pseudo-2D, neighbouring slices. Here a
  pseudo-2D's rows are never mixed.
* **Echo-Antiecho stays Echo-Antiecho.** The AU converts EA rows to
  States and rewrites FnMODE; the identical correction is a phase of
  ``exp(−iθ)`` on the echo and ``exp(+iθ)`` on the antiecho (derived
  from the AU's own conversion, and checked against openNMR's reader
  in the tests), so the dataset and its parameters are left as they
  were. Also allowed for 3D/4D, which the AU refuses.
* **Genuine QF 2D** (e.g. a magnitude COSY) gets its F1 correction too;
  the AU skips it, treating every QF dimension as a pseudo dimension.
* **Planes, points and NUS** are one code path: the acquisition order
  is worked out once (:func:`acquisition_layout`) and the correction
  applied to every hypercomplex point in that order.
* **Nothing is overwritten.** Corrected Bruker data are written to a
  new EXPNO (as ``lindriftcomp`` does); NUS blocks are concatenated
  into a new EXPNO rather than appended onto the first dataset (which
  ``concatenate_ser_nuslist`` does, irreversibly).

Not handled: TPPI/QSEQ indirect dimensions (FnMODE 2/3; one real
interferogram per t1 point, which no per-row rotation can shift --
neither does the AU program, which silently treats them as States),
and drift within a single 1D FID's accumulation, which is summed before
it is stored and cannot be undone afterwards. The initial t1 delay
(d0) is taken as zero, as in the AU program; a non-zero one leaves a
small constant F1 phase that varies with the drift.

Usage, notebook and GUI alike (the GUI's Process ▸ Field Drift
Correction calls these)::

    from opennmr.notebook import drift_correction as dc

    # 1. how far has the peak moved? (Hz, reference − observed)
    start = dc.measure_drift_hz(ref_1d, before_1d, region_ppm=(5.2, 4.4))
    end = dc.measure_drift_hz(ref_1d, after_1d, region_ppm=(5.2, 4.4))

    # 2a. correct a Bruker 2D/3D/4D on disk into a new EXPNO (for TopSpin,
    #     or to reload here)
    dc.correct_bruker_dataset("data/hnh/12", "data/hnh/112", start, end)

    # 2b. or load a 2D here with the correction applied
    spec = io.load("data/hnh/12", drift_correction={"start_hz": start, "end_hz": end})

    # NUS acquired in blocks: correct each block, then concatenate
    dc.concatenate_nus_datasets(["data/hnca/112", "data/hnca/113"], "data/hnca/120")
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

CITATION = ("E. E. Najbauer and L. B. Andreas, \"Correcting for magnetic field drift in "
            "magic-angle spinning NMR datasets\", J. Magn. Reson. 305 (2019) 1-4. "
            "doi:10.1016/j.jmr.2019.05.005")
DOI_URL = "https://doi.org/10.1016/j.jmr.2019.05.005"

#: Bruker FnMODE -> how an indirect dimension is corrected.
_FNMODE_KIND = {0: "qf", 1: "qf", 4: "states", 5: "states", 6: "echo_antiecho"}
_FNMODE_NAMES = {0: "undefined", 1: "QF", 2: "QSEQ", 3: "TPPI", 4: "States",
                 5: "States-TPPI", 6: "Echo-Antiecho"}
#: Kinds whose hypercomplex point is two FIDs (a cos/sin or echo/antiecho pair).
_PAIRED = ("states", "echo_antiecho")


@dataclass
class IndirectDim:
    """One indirect dimension, as far as drift correction needs it.

    ``name`` : ``"F1"``, ``"F2"``, ``"F3"``.
    ``n`` : number of *complex* (hypercomplex) increments.
    ``kind`` : ``"states"``, ``"echo_antiecho"``, ``"qf"``, or
    ``"none"`` (a pseudo dimension: no evolution, no correction).
    ``dwell_s`` : evolution-time increment per complex point (s).
    ``freq_ratio`` : BF of this dimension's nucleus / BF of the
    detected nucleus.
    """

    name: str
    n: int
    kind: str
    dwell_s: float = 0.0
    freq_ratio: float = 1.0

    @property
    def paired(self) -> bool:
        return self.kind in _PAIRED


def drift_profile(n_points: int, start_hz: float, end_hz: float, endpoint: bool = True) -> np.ndarray:
    """The shift (Hz) for each of ``n_points`` hypercomplex points, in
    acquisition order: ``start_hz`` for the first, ``end_hz`` for the
    last (``endpoint=False``: the AU program's ``start + (end −
    start)·p/P``)."""
    n_points = int(n_points)
    if n_points <= 0:
        return np.zeros(0)
    if n_points == 1:
        return np.array([float(start_hz)])
    p = np.arange(n_points, dtype=float)
    frac = p / (n_points - 1) if endpoint else p / n_points
    return float(start_hz) + (float(end_hz) - float(start_hz)) * frac


def acquisition_layout(dims: Sequence[IndirectDim], loop_order: Sequence[int] | None = None,
                       fntype: int = 0, nus_indices: np.ndarray | None = None):
    """Where each hypercomplex point's FIDs are in the file, in the
    order they were acquired.

    ``dims`` : the indirect dimensions, ordered F1, F2, F3.
    ``loop_order`` : indices into ``dims``, outermost (slowest) loop
    first. Default F1 outermost (Bruker AQSEQ 321 / 4321).
    ``fntype`` : 0 planes, 1 points, 2 NUS (Bruker ``FnTYPE``).
    ``nus_indices`` : for NUS, an ``(n_points, len(dims))`` int array of
    each sampled point's increment index per dimension (F1, F2, ...).

    Returns ``(rows, k)``: ``rows`` an ``(P, 2**m)`` int array of FID
    (row) numbers, columns ordered by the hypercomplex bit pattern
    (bit ``b`` of the column number belongs to ``dims[paired[b]]``,
    where ``paired`` lists the paired dimensions innermost loop first),
    and ``k`` a ``(P, len(dims))`` int array of increment indices.

    Bruker's layouts, read off the AU program's own index arithmetic:

    * **planes**: nested loops, each dimension's cos/sin bit directly
      inside its own increment -- ``[k_out][b_out][k_in][b_in][t2]``;
    * **points**: all the increment loops first, then all the bits,
      outer dimension's bit most significant -- ``[k_out][k_in][b_out][b_in]``;
    * **NUS**: one block of ``2**m`` FIDs per nuslist entry, bits as
      for points.
    """
    dims = list(dims)
    nd = len(dims)
    order = list(range(nd)) if loop_order is None else [int(i) for i in loop_order]
    if sorted(order) != list(range(nd)):
        raise ValueError(f"loop_order {loop_order} is not a permutation of the {nd} indirect dimensions.")
    paired_inner_first = [d for d in reversed(order) if dims[d].paired]
    m = len(paired_inner_first)
    bit_of = {d: b for b, d in enumerate(paired_inner_first)}
    n_bits = 2 ** m
    patterns = np.arange(n_bits)

    if fntype == 2:
        if nus_indices is None:
            raise ValueError("NUS data need the nuslist's increment indices.")
        k = np.asarray(nus_indices, dtype=int).reshape(-1, nd)
        rows = np.arange(k.shape[0])[:, None] * n_bits + patterns[None, :]
        return rows, k

    # Every increment combination, outer loop slowest.
    sizes = [dims[d].n for d in order]
    grids = np.meshgrid(*[np.arange(s) for s in sizes], indexing="ij")
    k_in_order = np.stack([g.ravel() for g in grids], axis=1) if nd else np.zeros((1, 0), int)
    k = np.empty_like(k_in_order)
    for column, d in enumerate(order):
        k[:, d] = k_in_order[:, column]
    n_points = k.shape[0]

    if fntype == 1:
        rows = np.arange(n_points)[:, None] * n_bits + patterns[None, :]
        return rows, k

    # Planes: row = Σ_d (factor_d*k_d + bit_d) * stride_d over the loop
    # order, with stride the number of FIDs one step of that loop spans.
    rows = np.zeros((n_points, n_bits), dtype=np.int64)
    stride = 1
    for d in reversed(order):  # innermost first
        factor = 2 if dims[d].paired else 1
        bits = ((patterns >> bit_of[d]) & 1) if dims[d].paired else np.zeros(n_bits, int)
        rows += (factor * k[:, d][:, None] + bits[None, :]) * stride
        stride *= factor * dims[d].n
    return rows, k


def acquired_point_mask(rows: np.ndarray, n_fids: int) -> np.ndarray:
    """Which hypercomplex points (rows of :func:`acquisition_layout`)
    are completely present in a file holding ``n_fids`` FIDs."""
    if rows.size == 0:
        return np.zeros(rows.shape[0], dtype=bool)
    return rows.max(axis=1) < int(n_fids)


def apply_drift_correction(fids: np.ndarray, shifts_hz: np.ndarray, sw_hz: float,
                           dims: Sequence[IndirectDim] = (), loop_order: Sequence[int] | None = None,
                           fntype: int = 0, nus_indices: np.ndarray | None = None,
                           group_delay: float = 0.0, convention: str = "bruker") -> np.ndarray:
    """Correct every FID in ``fids`` for the per-point shift ``shifts_hz``.

    ``fids`` : complex ``(n_fids, n_t2)`` array, in file order.
    ``shifts_hz`` : one shift (Hz of the detected nucleus, positive =
    move left) per hypercomplex point in acquisition order -- see
    :func:`drift_profile`.
    ``sw_hz`` : direct-dimension spectral width (``SW_h``).
    ``group_delay`` : the digital filter's group delay in complex points
    (0 once it has been removed, as for openNMR's own ``fid_raw``).
    ``convention`` : ``"bruker"`` for FIDs as stored in a ``ser``
    file; ``"opennmr"`` for openNMR's ``metadata.raw["fid_raw"]``, which
    is the complex conjugate of that (see ``io.bruker.read_2d``).

    Returns a new array; ``fids`` is not modified. FIDs the layout does
    not reach (trailing rows past TD) are passed through untouched.
    """
    if convention not in ("bruker", "opennmr"):
        raise ValueError("convention must be 'bruker' or 'opennmr'.")
    data = np.array(fids, dtype=np.complex128, copy=True)
    if convention == "opennmr":
        data = np.conj(data)
    dims = list(dims)
    rows, k = acquisition_layout(dims, loop_order, fntype, nus_indices)
    # An acquisition stopped early holds fewer FIDs than its parameters
    # describe: correct the points that were completely acquired.
    complete = acquired_point_mask(rows, data.shape[0])
    rows, k = rows[complete], k[complete]
    shifts = np.asarray(shifts_hz, dtype=float).ravel()
    if shifts.size == complete.size and shifts.size != rows.shape[0]:
        shifts = shifts[complete]
    if shifts.size != rows.shape[0]:
        raise ValueError(f"{shifts.size} shifts given for {rows.shape[0]} acquired points.")

    order = list(range(len(dims))) if loop_order is None else list(loop_order)
    paired_inner_first = [d for d in reversed(order) if dims[d].paired]
    bit_of = {d: b for b, d in enumerate(paired_inner_first)}
    n_bits = rows.shape[1]
    patterns = np.arange(n_bits)

    t2 = (np.arange(data.shape[1]) - float(group_delay)) / float(sw_hz)
    block = data[rows]  # (P, 2**m, n_t2) -- a copy
    # Direct dimension: one complex phase ramp per point.
    block *= np.exp(2j * np.pi * shifts[:, None, None] * t2[None, None, :])

    for d, dim in enumerate(dims):
        if dim.kind == "none" or not dim.dwell_s:
            continue
        theta = 2 * np.pi * shifts * dim.freq_ratio * dim.dwell_s * k[:, d]  # (P,)
        if dim.kind == "qf":
            # Bruker's QF F1 runs the opposite way to its F2 (measured
            # through openNMR's reader, which reproduces TopSpin): a
            # row phase of exp(-iθ) moves the F1 peak to higher frequency.
            block *= np.exp(-1j * theta)[:, None, None]
        elif dim.kind == "states":
            bit = bit_of[d]
            lo = patterns[((patterns >> bit) & 1) == 0]
            hi = lo + (1 << bit)
            c = np.cos(theta)[:, None, None]
            s = np.sin(theta)[:, None, None]
            a, b = block[:, lo, :].copy(), block[:, hi, :].copy()
            block[:, lo, :] = c * a - s * b
            block[:, hi, :] = s * a + c * b
        elif dim.kind == "echo_antiecho":
            bit = bit_of[d]
            echo = ((patterns >> bit) & 1) == 0
            phase = np.where(echo[None, :], np.exp(-1j * theta)[:, None], np.exp(1j * theta)[:, None])
            block *= phase[:, :, None]
        else:
            raise ValueError(f"Unknown indirect-dimension kind {dim.kind!r}.")

    data[rows] = block
    if convention == "opennmr":
        data = np.conj(data)
    return data


# ----------------------------------------------------------------------
# Bruker parameters
# ----------------------------------------------------------------------

_ACQU_FILES = ("acqus", "acqu2s", "acqu3s", "acqu4s")


def _read_params(path: str) -> dict:
    from ..io.jcampdx import read_acqus_files
    return read_acqus_files(path)


def _num(params: dict, key: str, default=None):
    value = params.get(key, default)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class BrukerDriftSetup:
    """Everything :func:`apply_drift_correction` needs, read from a
    Bruker experiment's status parameter files."""

    ndim: int
    sw_hz: float
    bf_direct: float
    dims: list[IndirectDim]
    loop_order: list[int]
    fntype: int
    nus_indices: np.ndarray | None
    group_delay: float
    td_direct: int
    dtypa: int
    big_endian: bool
    notes: list[str] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        if self.fntype == 2:
            return int(self.nus_indices.shape[0])
        n = 1
        for dim in self.dims:
            n *= dim.n
        return n

    def acquired_points(self, n_fids: int) -> int:
        """How many hypercomplex points a file of ``n_fids`` FIDs holds
        completely -- fewer than :attr:`n_points` when the acquisition
        was stopped early. The drift ramp runs over these, so the end
        shift lands on the last FID actually recorded."""
        rows, _ = acquisition_layout(self.dims, self.loop_order, self.fntype, self.nus_indices)
        return int(acquired_point_mask(rows, n_fids).sum())

    @property
    def fids_per_point(self) -> int:
        return 2 ** sum(1 for d in self.dims if d.paired)

    def describe(self) -> str:
        mode = {0: "planes", 1: "points", 2: "NUS"}.get(self.fntype, str(self.fntype))
        parts = [f"{self.ndim}D, {mode}, {self.n_points} acquired points x {self.fids_per_point} FID(s)"]
        for dim in self.dims:
            if dim.kind == "none":
                parts.append(f"{dim.name}: pseudo (no evolution) – direct correction only")
            else:
                parts.append(f"{dim.name}: {dim.kind}, {dim.n} increments of {dim.dwell_s * 1e6:.3f} µs, "
                             f"shift x {dim.freq_ratio:.6f}")
        return "; ".join(parts)


def read_nuslist(path: str, ndim_indirect: int) -> np.ndarray:
    """A Bruker ``nuslist`` as an ``(n, ndim_indirect)`` int array in
    F1, F2, ... column order. The file itself lists the highest-numbered
    indirect dimension first (``F2 F1`` for a 3D, ``F3 F2 F1`` for a
    4D -- the order ``lindriftcomp`` reads it in)."""
    with open(path) as handle:
        values = [int(v) for v in re.findall(r"-?\d+", handle.read())]
    arr = np.array(values, dtype=int)
    if arr.size % ndim_indirect:
        raise ValueError(f"{path}: {arr.size} numbers do not divide into rows of {ndim_indirect}.")
    return arr.reshape(-1, ndim_indirect)[:, ::-1].copy()


def bruker_drift_setup(path: str, pseudo: bool | None = None, params: dict | None = None,
                       n_fids: int | None = None) -> BrukerDriftSetup:
    """Read a Bruker 2D/3D/4D experiment's acquisition layout.

    ``pseudo`` : force (``True``) or forbid (``False``) treating every
    indirect dimension as a pseudo dimension; ``None`` decides per
    dimension from its ``NUC1`` (``off`` = pseudo, Bruker's own
    convention, as ``io.bruker.read_2d`` does).
    ``n_fids`` : the number of FIDs actually read -- used for a 2D whose
    ``acqu2s`` records no ``TD``.
    """
    from ..io import digital_filter

    params = params if params is not None else _read_params(path)
    acqus = params.get("acqus")
    if not acqus:
        raise FileNotFoundError(f"No acqus in {path}.")
    present = [name for name in _ACQU_FILES[1:] if name in params]
    ndim = 1 + len(present)
    if ndim < 2:
        raise ValueError("Drift correction needs a 2D, 3D or 4D experiment: a 1D FID's scans are "
                         "summed before they are stored, so drift within it cannot be undone.")

    notes: list[str] = []
    sw_hz = _num(acqus, "SW_h")
    bf_dir = _num(acqus, "BF1")
    if not sw_hz or not bf_dir:
        raise ValueError("acqus lacks SW_h or BF1.")
    fntype = int(_num(acqus, "FnTYPE", 0) or 0)
    aqseq = int(_num(acqus, "AQSEQ", 0) or 0)

    # acqu2s describes the indirect dimension nearest the direct one:
    # F1 of a 2D, F2 of a 3D, F3 of a 4D. So F1 is always the last file.
    files_f_order = list(reversed(present))  # F1 first
    dims: list[IndirectDim] = []
    for number, name in enumerate(files_f_order, start=1):
        p = params[name]
        fn_mode = int(_num(p, "FnMODE", 1) or 0)
        nuc = str(p.get("NUC1", "")).strip("<> ").lower()
        is_pseudo = (nuc == "off") if pseudo is None else bool(pseudo)
        td = int(_num(p, "TD", 0) or 0)
        if td <= 0 and ndim == 2 and n_fids:
            td = int(n_fids)
            notes.append(f"acqu2s records no TD; taken as the {td} FIDs in the file.")
        if is_pseudo:
            kind = "none"
        elif fn_mode in _FNMODE_KIND:
            kind = _FNMODE_KIND[fn_mode]
        else:
            raise NotImplementedError(
                f"F{number} is acquired as {_FNMODE_NAMES.get(fn_mode, fn_mode)} (FnMODE {fn_mode}); drift "
                f"correction supports QF, States, States-TPPI and Echo-Antiecho indirect dimensions.")
        if kind in _PAIRED:
            if td % 2:
                notes.append(f"F{number}: odd TD ({td}) for a paired mode; the last FID is ignored.")
            n = td // 2
        else:
            n = td
        sw_ind = _num(p, "SW_h")
        bf_ind = _num(p, "BF1", bf_dir)
        dwell = (1.0 / sw_ind) if (sw_ind and kind != "none") else 0.0
        dims.append(IndirectDim(name=f"F{number}", n=max(n, 0), kind=kind, dwell_s=dwell,
                                freq_ratio=(bf_ind / bf_dir) if bf_ind else 1.0))

    if ndim == 2:
        loop_order = [0]
    elif ndim == 3:
        loop_order = [1, 0] if aqseq == 1 else [0, 1]  # AQSEQ 312: F2 outermost
    else:
        if aqseq not in (0,):
            notes.append(f"4D AQSEQ {aqseq} read as 4321, the only order lindriftcomp supports.")
        loop_order = [0, 1, 2]

    nus = None
    if fntype == 2:
        nus_path = os.path.join(path, "nuslist")
        if not os.path.isfile(nus_path):
            raise FileNotFoundError(f"NUS data (FnTYPE 2) but no nuslist in {path}.")
        nus = read_nuslist(nus_path, ndim - 1)
        npoints = int(_num(acqus, "NusPOINTS", 0) or 0)
        if npoints and npoints < nus.shape[0]:
            nus = nus[:npoints]
        if ndim == 2:
            notes.append("2D NUS is not one of lindriftcomp's modes; handled the same way as 3D/4D NUS.")
    elif fntype not in (0, 1):
        raise ValueError(f"FnTYPE {fntype} is not planes (0), points (1) or NUS (2).")

    decim = _num(acqus, "DECIM", 1.0) or 1.0
    dspfvs = _num(acqus, "DSPFVS", 0.0) or 0.0
    grpdly = _num(acqus, "GRPDLY", 0.0) or 0.0
    if decim == 1.0:
        gd = 0.0
    else:
        try:
            gd = digital_filter.group_delay(decim, dspfvs, grpdly)
        except ValueError as exc:
            gd = 0.0
            notes.append(f"Group delay unknown ({exc}); taken as 0, which adds a small constant phase.")

    return BrukerDriftSetup(ndim=ndim, sw_hz=sw_hz, bf_direct=bf_dir, dims=dims, loop_order=loop_order,
                            fntype=fntype, nus_indices=nus, group_delay=gd,
                            td_direct=int(_num(acqus, "TD", 0) or 0),
                            dtypa=int(_num(acqus, "DTYPA", 0) or 0),
                            big_endian=int(_num(acqus, "BYTORDA", 0) or 0) == 1, notes=notes)


def _ser_dtype(setup: BrukerDriftSetup) -> np.dtype:
    base = "f8" if setup.dtypa == 2 else "i4"
    return np.dtype(base).newbyteorder(">" if setup.big_endian else "<")


def _row_words(setup: BrukerDriftSetup) -> int:
    block = 128 if setup.dtypa == 2 else 256
    return -(-setup.td_direct // block) * block


@dataclass
class DriftCorrectionResult:
    """What a correction did -- enough to log and to reproduce it."""

    start_hz: float
    end_hz: float
    n_points: int
    description: str
    source: str | None = None
    destination: str | None = None
    endpoint: bool = True
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            "Linear field drift correction",
            f"  start shift {self.start_hz:+.4f} Hz -> end shift {self.end_hz:+.4f} Hz "
            f"over {self.n_points} acquired points (positive = moved to higher frequency)",
            f"  {self.description}",
        ]
        if self.source:
            lines.append(f"  from {self.source}")
        if self.destination:
            lines.append(f"  written to {self.destination}")
        lines.extend("  Note: " + n for n in self.notes)
        lines.append("  Method: " + CITATION)
        return "\n".join(lines)


def _copy_experiment(source: str, destination: str, overwrite: bool) -> None:
    source = os.path.abspath(source)
    destination = os.path.abspath(destination)
    if source == destination:
        raise ValueError("The corrected dataset must go to a new experiment folder, not over the original.")
    if os.path.exists(destination):
        if not overwrite:
            raise FileExistsError(f"{destination} already exists.")
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def next_free_expno(path: str, start: int | None = None) -> str:
    """The first unused numbered experiment folder beside ``path``
    (``.../12`` -> ``.../13`` or later)."""
    parent, name = os.path.split(os.path.abspath(path.rstrip("/\\")))
    number = int(name) + 1 if name.isdigit() else (start or 1)
    if start is not None:
        number = start
    while os.path.exists(os.path.join(parent, str(number))):
        number += 1
    return os.path.join(parent, str(number))


def correct_bruker_dataset(source: str, destination: str | None, start_hz: float, end_hz: float,
                           shifts_hz: Sequence[float] | None = None, endpoint: bool = True,
                           pseudo: bool | None = None, overwrite: bool = False) -> DriftCorrectionResult:
    """``lindriftcomp``: copy a Bruker 2D/3D/4D experiment to
    ``destination`` (``None``: the next free EXPNO) with its ``ser``
    drift-corrected. Planes, points and NUS; QF, States, States-TPPI
    and Echo-Antiecho indirect dimensions; int32 and float64 data.

    ``shifts_hz`` : an explicit per-point shift (acquisition order)
    instead of the linear ``start_hz`` -> ``end_hz`` ramp -- e.g. a
    drift measured more often than at the two ends.

    The original experiment is never modified. Processed data in
    ``pdata`` are copied as they were and are therefore *stale*: process
    the new EXPNO again (here or in TopSpin).
    """
    setup = bruker_drift_setup(source, pseudo=pseudo)
    if destination is None:
        destination = next_free_expno(source)

    ser_path = os.path.join(source, "ser")
    if not os.path.isfile(ser_path):
        raise FileNotFoundError(f"No ser file in {source}.")
    dtype = _ser_dtype(setup)
    words = _row_words(setup)
    raw = np.fromfile(ser_path, dtype=dtype)
    n_rows = raw.size // words
    tail = raw[n_rows * words:]
    fids = raw[: n_rows * words].astype(np.float64).reshape(n_rows, words)
    fids = fids[:, 0::2] + 1j * fids[:, 1::2]
    n_acquired = setup.acquired_points(n_rows)
    if n_acquired == 0:
        raise ValueError(f"{ser_path} does not hold one complete acquired point.")
    shifts = (np.asarray(shifts_hz, dtype=float) if shifts_hz is not None
              else drift_profile(n_acquired, start_hz, end_hz, endpoint=endpoint))

    corrected = apply_drift_correction(fids, shifts, setup.sw_hz, setup.dims, setup.loop_order,
                                       setup.fntype, setup.nus_indices, setup.group_delay, "bruker")

    out = np.empty((n_rows, words), dtype=np.float64)
    out[:, 0::2] = corrected.real
    out[:, 1::2] = corrected.imag
    notes = list(setup.notes)
    if setup.dtypa != 2:
        limit = np.iinfo(np.int32)
        if np.abs(out).max(initial=0) > limit.max:
            notes.append("Values exceeded int32 after correction and were clipped.")
        out = np.clip(np.rint(out), limit.min, limit.max)
    out = np.concatenate([out.ravel().astype(dtype), tail])

    _copy_experiment(source, destination, overwrite)
    out.tofile(os.path.join(destination, "ser"))
    _append_audit(destination, start_hz, end_hz, shifts_hz is not None, source)
    if n_acquired < setup.n_points:
        notes.append(f"Only {n_acquired} of {setup.n_points} points were acquired (stopped early?); "
                     f"the ramp runs over those.")
    notes.append("pdata was copied unchanged and is stale: reprocess the new experiment.")
    return DriftCorrectionResult(start_hz=float(shifts[0]) if shifts.size else start_hz,
                                 end_hz=float(shifts[-1]) if shifts.size else end_hz,
                                 n_points=int(shifts.size), description=setup.describe(),
                                 source=source, destination=destination, endpoint=endpoint, notes=notes)


def _append_audit(destination: str, start_hz: float, end_hz: float, custom: bool, source: str) -> None:
    """Say in the new experiment's own folder what was done to it, so a
    TopSpin user who opens it later can find out."""
    text = (f"openNMR linear field drift correction of {source}\n"
            f"  start {start_hz:+.4f} Hz, end {end_hz:+.4f} Hz"
            f"{' (explicit per-point shifts)' if custom else ''}\n"
            f"  method: {CITATION}\n")
    with open(os.path.join(destination, "opennmr_drift_correction.txt"), "a", encoding="utf-8") as handle:
        handle.write(text)


def correct_spectrum_fids(fid_raw: np.ndarray, setup: BrukerDriftSetup, start_hz: float, end_hz: float,
                          shifts_hz: Sequence[float] | None = None, endpoint: bool = True) -> np.ndarray:
    """Correct openNMR's own ``metadata.raw["fid_raw"]`` (conjugated,
    digital filter already removed) for a 2D. Used by
    ``io.bruker.read_2d(..., drift_correction=...)``."""
    n_acquired = setup.acquired_points(np.asarray(fid_raw).shape[0])
    shifts = (np.asarray(shifts_hz, dtype=float) if shifts_hz is not None
              else drift_profile(n_acquired, start_hz, end_hz, endpoint=endpoint))
    return apply_drift_correction(fid_raw, shifts, setup.sw_hz, setup.dims, setup.loop_order,
                                  setup.fntype, setup.nus_indices, group_delay=0.0, convention="opennmr")


# ----------------------------------------------------------------------
# Any vendor's pseudo-2D (one spectrum per row, acquired in row order)
# ----------------------------------------------------------------------

#: Vendors whose readers take ``drift_correction=`` for a pseudo-2D series
#: (X-Pulse arrays, Varian arrayed 2D, Spinsolve PGSTE). Bruker is separate:
#: it also does genuine 2D/3D/4D, from its own acquisition parameters.
PSEUDO_2D_VENDORS = ("xpulse", "varian", "spinsolve", "jeol")

NOT_SUPPORTED_MESSAGE = (
    "Drift correction needs a Bruker 2D/pseudo-2D read from its raw ser, or a pseudo-2D series "
    "(one spectrum per row, acquired in order) from X-Pulse, Varian, Spinsolve or JEOL. A genuine 2D "
    "from another vendor also has an indirect dimension to correct, which needs that vendor's "
    "acquisition order – not available yet. For Bruker data, use Write Corrected Bruker Dataset…"
)


def drift_support(spectrum) -> tuple[bool, str]:
    """Whether :func:`correct_pseudo_2d_fids`/the readers' ``drift_correction``
    can correct ``spectrum``, and if not, why (the message the GUI shows)."""
    vendor = getattr(spectrum.metadata, "vendor", None)
    if spectrum.ndim != 2:
        return False, ("Drift correction needs a 2D or pseudo-2D spectrum. For 3D/4D or NUS data, "
                       "write a corrected dataset instead.")
    if vendor == "bruker":
        if spectrum.metadata.raw.get("fid_raw") is None:
            return False, ("The selected spectrum cannot be corrected here: that needs a Bruker 2D or "
                           "pseudo-2D read from its raw ser. For 3D/4D or NUS data, write a corrected "
                           "dataset instead.")
        return True, ""
    if vendor in PSEUDO_2D_VENDORS and spectrum.is_pseudo_2d:
        return True, ""
    return False, NOT_SUPPORTED_MESSAGE


def pseudo_2d_shifts(n_rows: int, request: dict) -> np.ndarray:
    """The per-row shifts (Hz) a ``drift_correction`` request describes:
    its explicit ``shifts_hz``, else the linear ``start_hz`` -> ``end_hz``
    ramp over ``n_rows`` rows (see :func:`drift_profile`)."""
    shifts = request.get("shifts_hz")
    if shifts is not None:
        shifts = np.asarray(shifts, dtype=float).ravel()
        if shifts.size != n_rows:
            raise ValueError(f"{shifts.size} shifts given for {n_rows} rows.")
        return shifts
    start = float(request.get("start_hz", 0.0))
    end = float(request.get("end_hz", start))
    return drift_profile(n_rows, start, end, endpoint=bool(request.get("endpoint", True)))


def pseudo_2d_record(n_rows: int, request: dict, vendor: str = "") -> dict:
    """The ``metadata.raw["drift_correction"]`` record for a pseudo-2D
    correction (same keys as Bruker's)."""
    start = float(request.get("start_hz", 0.0))
    end = float(request.get("end_hz", start))
    shifts = request.get("shifts_hz")
    return {"start_hz": start, "end_hz": end, "endpoint": bool(request.get("endpoint", True)),
            "n_points": int(n_rows), "custom_shifts": shifts is not None,
            "description": (f"pseudo-2D{f' ({vendor})' if vendor else ''}, {n_rows} rows – "
                            "direct-dimension correction only (no evolution dimension)"),
            "notes": [], "citation": CITATION}


def correct_pseudo_2d_fids(fid_rows, dt_s: float, start_hz: float = 0.0, end_hz: float | None = None,
                           shifts_hz: Sequence[float] | None = None, endpoint: bool = True) -> np.ndarray:
    """Correct the FIDs of a pseudo-2D from **any vendor** for linear field
    drift, row by row in acquisition order.

    ``fid_rows`` : complex ``(n_rows, n_points)`` array of FIDs as openNMR
    transforms them (``metadata.raw["fid_raw"]`` -- for Bruker that is the
    conjugate of the ``ser``, and this is the convention used here).
    ``dt_s`` : dwell time per complex point (``metadata.raw["fid_dt"]``).
    ``start_hz``/``end_hz`` : the shift of the first/last row, Hz of the
    detected nucleus, ``reference - observed`` (positive moves peaks
    left); or ``shifts_hz``, one per row.

    A pseudo-2D's rows are separate experiments, not an evolution time, so
    only the direct-dimension phase ramp ``exp(i 2 pi shift t)`` applies.
    Returns a new array. In the readers this is what
    ``drift_correction={"start_hz": ..., "end_hz": ...}`` does for
    X-Pulse, Varian and Spinsolve series."""
    rows = np.atleast_2d(np.asarray(fid_rows, dtype=np.complex128))
    request = {"start_hz": start_hz, "end_hz": start_hz if end_hz is None else end_hz,
               "endpoint": endpoint, "shifts_hz": shifts_hz}
    shifts = pseudo_2d_shifts(rows.shape[0], request)
    dims = [IndirectDim(name="F1", n=rows.shape[0], kind="none")]
    return apply_drift_correction(rows, shifts, 1.0 / float(dt_s), dims, None, 0, None,
                                  group_delay=0.0, convention="opennmr")


# ----------------------------------------------------------------------
# NUS block concatenation (concatenate_ser_nuslist)
# ----------------------------------------------------------------------

def _set_param(path: str, key: str, value) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, encoding="latin-1") as handle:
        text = handle.read()
    pattern = re.compile(rf"^(##\${re.escape(key)}=\s*)(\S+)", re.MULTILINE)
    if not pattern.search(text):
        return False
    text = pattern.sub(lambda m: m.group(1) + str(value), text, count=1)
    with open(path, "w", encoding="latin-1") as handle:
        handle.write(text)
    return True


def concatenate_nus_datasets(sources: Sequence[str], destination: str, overwrite: bool = False) -> dict:
    """``concatenate_ser_nuslist``: join NUS experiments recorded as
    separate blocks (each sampling a different part of one nuslist)
    into one experiment at ``destination``, a copy of the first source
    with every ``ser`` and ``nuslist`` appended in turn.

    Correct each block for drift first (:func:`correct_bruker_dataset`),
    against one common reference, then concatenate the corrected ones.

    Unlike the AU program this writes a new experiment instead of
    appending onto the first one, joins any number of blocks in one
    call, checks that the blocks are compatible (same dimensionality,
    direct TD and data type), and updates ``NusPOINTS`` in
    ``acqus``/``acqu`` to the new total. As the AU program's notes say,
    check the remaining NUS status parameters (e.g. ``NusTD``) match how
    the complete measurement was set up before processing.
    """
    sources = list(sources)
    if len(sources) < 2:
        raise ValueError("Give at least two NUS experiments to concatenate.")
    setups = [bruker_drift_setup(s) for s in sources]
    first = setups[0]
    for s, path in zip(setups, sources):
        if s.fntype != 2:
            raise ValueError(f"{path} is not NUS data (FnTYPE {s.fntype}).")
        if (s.ndim, s.td_direct, s.dtypa, s.big_endian) != (first.ndim, first.td_direct, first.dtypa,
                                                            first.big_endian):
            raise ValueError(f"{path} does not match {sources[0]} (dimensions, TD or data type differ).")
    dtype = _ser_dtype(first)
    words = _row_words(first)
    blocks, lists = [], []
    for s, path in zip(setups, sources):
        n_fids = s.n_points * s.fids_per_point
        raw = np.fromfile(os.path.join(path, "ser"), dtype=dtype)
        need = n_fids * words
        if raw.size < need:
            raise ValueError(f"{path}/ser holds fewer FIDs than its nuslist/NusPOINTS describe.")
        blocks.append(raw[:need])
        lists.append(s.nus_indices[:, ::-1])  # back to the file's own column order
    _copy_experiment(sources[0], destination, overwrite)
    np.concatenate(blocks).tofile(os.path.join(destination, "ser"))
    nus = np.concatenate(lists)
    with open(os.path.join(destination, "nuslist"), "w", encoding="ascii") as handle:
        for row in nus:
            handle.write(" ".join(str(int(v)) for v in row) + "\n")
    total = int(nus.shape[0])
    updated = [name for name in ("acqus", "acqu")
               if _set_param(os.path.join(destination, name), "NusPOINTS", total)]
    with open(os.path.join(destination, "opennmr_drift_correction.txt"), "a", encoding="utf-8") as handle:
        handle.write("openNMR NUS concatenation of " + ", ".join(sources) + f"\n  {total} points\n"
                     f"  method: {CITATION}\n")
    return {"destination": destination, "nus_points": total, "sources": sources,
            "updated_parameters": updated}


# ----------------------------------------------------------------------
# Measuring the drift
# ----------------------------------------------------------------------

def _absolute_hz_axis(spectrum) -> np.ndarray:
    """A 1D spectrum's points as frequencies relative to its nucleus's
    base frequency, independent of any referencing: ``δ·SF + SR``
    (Hz), with ``SF = SFO1``-ish and SR whatever is in effect -- so two
    spectra referenced differently still compare by frequency."""
    from .referencing import spectrum_sr_hz

    ppm = np.asarray(spectrum.axes[-1].ppm, dtype=float)
    sfo1 = spectrum.metadata.sfo1
    if not sfo1:
        raise ValueError("The spectrum has no SFO1, so its ppm axis cannot be turned into frequency.")
    sr_hz = spectrum_sr_hz(spectrum, "f2")
    sf = (spectrum.metadata.bf1 or sfo1) + sr_hz * 1e-6
    return ppm * sf + sr_hz


def peak_frequency_hz(spectrum, region_ppm: tuple[float, float]) -> float:
    """The tallest peak's frequency inside ``region_ppm``, in Hz on the
    referencing-independent axis of :func:`_absolute_hz_axis`, refined
    to a fraction of a point by a parabola through its top three
    points -- a drift is often well under one point."""
    hz = _absolute_hz_axis(spectrum)
    ppm = np.asarray(spectrum.axes[-1].ppm, dtype=float)
    lo, hi = sorted(region_ppm)
    mask = (ppm >= lo) & (ppm <= hi)
    if mask.sum() < 3:
        raise ValueError(f"Fewer than three points between {lo} and {hi} ppm.")
    y = np.asarray(spectrum.data, dtype=float)
    idx = np.flatnonzero(mask)
    i = idx[np.argmax(y[idx])]
    if i <= idx[0] or i >= idx[-1]:
        return float(hz[i])  # on the region's edge: no parabola to fit
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom else 0.0
    step = hz[i + 1] - hz[i]
    return float(hz[i] + offset * step)


def measure_drift_hz(reference, spectrum, region_ppm: tuple[float, float]) -> float:
    """The shift to enter for ``spectrum`` (Hz, positive = move left):
    ``ν_reference − ν_spectrum`` of the tallest peak in ``region_ppm``.
    Both must be 1D spectra of the same nucleus; they are compared by
    frequency, so an SR applied to either does not enter."""
    for s in (reference, spectrum):
        if s.ndim != 1:
            raise ValueError("Measure the drift on 1D spectra (e.g. recorded before and after the nD).")
    if reference.metadata.nucleus and spectrum.metadata.nucleus and \
            str(reference.metadata.nucleus).strip("<> ") != str(spectrum.metadata.nucleus).strip("<> "):
        raise ValueError("The reference and the spectrum are of different nuclei.")
    return peak_frequency_hz(reference, region_ppm) - peak_frequency_hz(spectrum, region_ppm)
