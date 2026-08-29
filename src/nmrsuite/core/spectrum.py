"""
Vendor-agnostic NMR data model.

Every importer in :mod:`nmrsuite.io`, regardless of vendor, produces an
:class:`NMRSpectrum`. A 1D spectrum, a 2D spectrum, a pseudo-2D
relaxation/diffusion series, and a single row/column extracted from a 2D
dataset are all the same class -- a pseudo-2D dataset's indirect axis
just has ``kind="index"`` instead of ``kind="frequency"``, and an
extracted trace is simply an ``NMRSpectrum`` with one axis instead of
two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

AxisKind = Literal["frequency", "index", "time"]
NormalizePart = Literal["real", "imag", "magnitude"]


@dataclass
class NMRAxis:
    """One axis of an :class:`NMRSpectrum`.

    ``kind`` says what the axis fundamentally *is*:

    - ``"frequency"`` : a chemical-shift axis. ``ppm`` and/or ``hz``
      should be set (both, where possible -- see ``NMRSpectrum``).
    - ``"index"``      : a non-frequency indirect dimension, e.g. the
      slice/experiment number of a pseudo-2D relaxation series. ``index``
      should be set (1-based, matching TopSpin's own convention).
    - ``"time"``       : a time-domain axis (e.g. an FID before FFT).

    ``label`` overrides the axis label a plotting function would
    otherwise generate (e.g. nucleus-based) -- used for axes with no
    physical/chemical-shift meaning, such as "Temperature (C)".
    """

    kind: AxisKind
    ppm: np.ndarray | None = None
    hz: np.ndarray | None = None
    index: np.ndarray | None = None
    label: str | None = None

    def values(self, unit: str = "ppm") -> np.ndarray:
        arr = getattr(self, unit, None)
        if arr is None:
            raise ValueError(
                f"This axis has no '{unit}' values (kind={self.kind!r})."
            )
        return arr

    def __len__(self) -> int:
        for arr in (self.ppm, self.hz, self.index):
            if arr is not None:
                return len(arr)
        raise ValueError("NMRAxis has no values set.")


@dataclass
class NMRMetadata:
    """Acquisition/processing metadata for one :class:`NMRSpectrum`.

    Only the fields used across vendors/processing/plotting are named
    explicitly. Everything else the importer read (the full parameter
    dump) lives in ``raw``, so nothing is lost even though it isn't
    promoted to a named field.
    """

    vendor: str
    nucleus: str | None = None
    nucleus_indirect: str | None = None
    sfo1: float | None = None
    sfo1_indirect: float | None = None
    bf1: float | None = None
    o1: float | None = None
    ns: float | None = None
    source_path: str | None = None
    procno: int | None = None
    pseudo_2d: bool = False
    homonuclear: bool = False
    #: Sample mass (g) -- never recorded by any vendor file, so unset
    #: (None) until a caller supplies it (e.g. the GUI's "Set sample
    #: mass..." action). Used for mass-normalized multi-spectrum
    #: comparison; unrelated to the ``mass=`` load-time parameter some
    #: Bruker readers accept (which divides the data immediately,
    #: rather than recording the value for later).
    mass: float | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class NMRSpectrum:
    """A vendor-agnostic NMR spectrum (1D, 2D, or a trace extracted from
    a 2D dataset -- which is just a 1D spectrum in this model).

    ``data`` : shape ``(n,)`` for a 1D spectrum/trace, ``(n_f1, n_f2)``
    for 2D. Always the real part.

    ``axes`` : one :class:`NMRAxis` per dimension of ``data``, outer to
    inner (i.e. ``axes[0]`` indexes ``data``'s axis 0).
    """

    data: np.ndarray
    axes: list[NMRAxis]
    metadata: NMRMetadata
    data_imag: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.axes) != self.data.ndim:
            raise ValueError(
                f"NMRSpectrum has {self.data.ndim} data dimension(s) but "
                f"{len(self.axes)} axes were given."
            )

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def is_pseudo_2d(self) -> bool:
        return self.metadata.pseudo_2d

    @property
    def complex_data(self) -> np.ndarray | None:
        if self.data_imag is None:
            return None
        return self.data + 1j * self.data_imag

    def axis_values(self, dim: int, unit: str = "ppm") -> np.ndarray:
        """Values of the ``dim``-th axis (supports negative indexing,
        e.g. ``dim=-1`` for the direct/F2 dimension)."""
        return self.axes[dim].values(unit)

    def normalized(self, part: NormalizePart = "real") -> np.ndarray:
        """The requested part of the spectrum, scaled to [-1, 1] (or
        [0, 1] for ``"magnitude"``) by its own maximum absolute value.

        Computed on demand rather than cached, so there is exactly one
        source of truth for ``data``/``data_imag`` -- no risk of a
        stored "normalised" copy going stale after in-place processing.
        """

        if part == "real":
            values = self.data
        elif part == "imag":
            if self.data_imag is None:
                raise ValueError("This spectrum has no imaginary part.")
            values = self.data_imag
        elif part == "magnitude":
            complex_data = self.complex_data
            values = np.abs(complex_data) if complex_data is not None else np.abs(self.data)
        else:
            raise ValueError(f"Unknown part {part!r}; expected 'real', 'imag', or 'magnitude'.")

        peak = np.max(np.abs(values))
        return values / peak if peak != 0 else values
