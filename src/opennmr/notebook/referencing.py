"""
Transfer a spectral reference (SR) from one nucleus to another.

Explicit user request (2026-10-01): a calculator in Process ▸ Reference
that works out the SR of one nucleus from the SR calibrated on another,
through the ratio of their frequencies (equivalently, of their
gyromagnetic ratios).

**Why it works.** TopSpin's SR is ``SF - BF1``: how far the frequency
that is called 0 ppm sits from the spectrometer's nominal base
frequency for that nucleus. Every nucleus's BF1 is derived from the
same field, so a field that is off by a fraction ``ΔB/B`` moves every
nucleus's reference frequency by that same fraction. Dividing the SR
calibrated on one nucleus by its transmitter frequency gives that
fraction, in ppm (Hz / MHz = ppm)::

    correction_ppm = SR_cal / SFO1_cal
    SR_new         = correction_ppm * SFO1_new

``SR_new`` is in Hz of the new nucleus. Using ``SFO1`` rather than
``BF1`` changes the result by the carrier offset's share of the
frequency (O1/BF1, ~1e-4 of SR): a few mHz, far below anything
measurable.

**When it is valid.** Only for the same magnet, at a similar time
(the field drifts: after a re-shim, a quench, a probe change, a lock or
a sweep-coil adjustment the calibration is gone), and with the new
nucleus's BF1 taken from the same spectrometer nucleus table that
TopSpin used (that table's frequency ratios -- IUPAC Ξ values -- are
what make the transferred SR land on that nucleus's recommended 0 ppm
reference). It transfers the *field*, not the sample: a calibration on
a different sample (external referencing) still carries any
susceptibility difference between the two samples.

Works identically in a notebook and in the GUI, which calls
:func:`transfer_sr`::

    from opennmr.notebook.referencing import transfer_sr

    result = transfer_sr(sr_cal_hz=-35.2, sfo1_cal_mhz=400.1326,
                         sfo1_new_mhz=100.6228)
    result.sr_new_hz          # -8.852 Hz
    result.correction_ppm     # -0.08797 ppm
    print(result.report())

When the new nucleus's SFO1 is not to hand, give both nuclei instead
and their gyromagnetic ratios are used (``SFO1_new = SFO1_cal *
|γ_new / γ_cal|``) -- an approximation, see :func:`transfer_sr`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Shown wherever the result is (GUI and report), so nobody reads the
#: number without the condition it depends on.
VALIDITY_WARNING = (
    "Only valid for spectra recorded on the same magnet at a similar time -- no re-shim, field "
    "or lock change, probe change or quench in between. The new nucleus is referenced through "
    "the spectrometer's own nucleus table; susceptibility differences between two different "
    "samples are not corrected."
)


@dataclass
class SRTransfer:
    """The result of :func:`transfer_sr`. All frequencies as given:
    SR in Hz, SFO1 in MHz."""

    sr_cal_hz: float
    sfo1_cal_mhz: float
    sfo1_new_mhz: float
    correction_ppm: float
    sr_new_hz: float
    nucleus_cal: str | None = None
    nucleus_new: str | None = None
    #: ``"sfo1"`` when SFO1_new was given, ``"gamma"`` when it was
    #: estimated from the two nuclei's gyromagnetic ratios.
    sfo1_new_source: str = "sfo1"
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        cal = f" ({self.nucleus_cal})" if self.nucleus_cal else ""
        new = f" ({self.nucleus_new})" if self.nucleus_new else ""
        lines = [
            "SR transfer between nuclei",
            f"  SR_cal{cal}   = {self.sr_cal_hz:.4f} Hz",
            f"  SFO1_cal{cal} = {self.sfo1_cal_mhz:.7f} MHz",
            f"  SR_cal / SFO1_cal = {self.correction_ppm:.6f} ppm",
            f"  SFO1_new{new} = {self.sfo1_new_mhz:.7f} MHz"
            + ("  (from the gyromagnetic-ratio ratio)" if self.sfo1_new_source == "gamma" else ""),
            f"  SR_new{new}   = (SR_cal / SFO1_cal) x SFO1_new = {self.sr_new_hz:.4f} Hz",
            "",
            "  " + VALIDITY_WARNING,
        ]
        for warning in self.warnings:
            lines.append("  Warning: " + warning)
        return "\n".join(lines)


def gamma_mhz_per_t(nucleus: str) -> float:
    """|γ/2π| in MHz/T for ``nucleus`` (any spelling openNMR accepts,
    e.g. ``"13C"``, ``"C13"``), from the one nuclear-data table openNMR
    uses everywhere (``notebook.utils._GAMMA_DF``)."""
    from .two_d import canonical_nucleus
    from .utils import _GAMMA_DF

    label = canonical_nucleus(nucleus) or str(nucleus).strip()
    rows = _GAMMA_DF.loc[_GAMMA_DF["Nuclide"] == label, "Gyromagnetic Ratio (MHz/T)"]
    if rows.empty:
        raise ValueError(f"Nucleus {nucleus!r} is not in openNMR's nuclear-data table.")
    return abs(float(rows.iloc[0]))


def transfer_sr(sr_cal_hz: float, sfo1_cal_mhz: float, sfo1_new_mhz: float | None = None,
                nucleus_cal: str | None = None, nucleus_new: str | None = None) -> SRTransfer:
    """SR for a new nucleus from the SR calibrated on another.

    ``sr_cal_hz`` : the SR (Hz) calibrated on the reference nucleus.
    ``sfo1_cal_mhz`` : that experiment's SFO1 (MHz).
    ``sfo1_new_mhz`` : the new experiment's SFO1 (MHz). When ``None``,
    estimated as ``sfo1_cal_mhz * |γ_new / γ_cal|`` from
    ``nucleus_cal``/``nucleus_new`` -- which then must both be given.
    That estimate ignores the reference compounds' shielding (the
    difference between bare-nucleus γ ratios and IUPAC Ξ ratios), which
    is parts per million for light nuclei but up to ~1 % for heavy ones
    -- 1 % of the SR, so typically well under 1 Hz, but use the real
    SFO1 whenever it is known.

    Returns an :class:`SRTransfer` with ``correction_ppm = SR_cal /
    SFO1_cal`` and ``sr_new_hz = correction_ppm * SFO1_new``.
    """
    sr_cal_hz = float(sr_cal_hz)
    sfo1_cal_mhz = float(sfo1_cal_mhz)
    if not sfo1_cal_mhz > 0:
        raise ValueError("SFO1 of the calibrated experiment must be a positive frequency (MHz).")
    warnings: list[str] = []
    source = "sfo1"
    if sfo1_new_mhz is None:
        if not (nucleus_cal and nucleus_new):
            raise ValueError("Give SFO1 of the new experiment, or both nuclei so it can be "
                             "estimated from their gyromagnetic ratios.")
        sfo1_new_mhz = sfo1_cal_mhz * gamma_mhz_per_t(nucleus_new) / gamma_mhz_per_t(nucleus_cal)
        source = "gamma"
        warnings.append("SFO1_new was estimated from the gyromagnetic ratios, not read from the "
                        "new experiment; enter the real SFO1 when you have it.")
    sfo1_new_mhz = float(sfo1_new_mhz)
    if not sfo1_new_mhz > 0:
        raise ValueError("SFO1 of the new experiment must be a positive frequency (MHz).")

    correction_ppm = sr_cal_hz / sfo1_cal_mhz
    sr_new_hz = correction_ppm * sfo1_new_mhz

    # A transfer between two experiments whose SFO1s do not match their
    # stated nuclei is a sign one of the numbers came from the wrong
    # experiment -- worth saying, not worth refusing.
    if nucleus_cal and nucleus_new and source == "sfo1":
        try:
            expected = gamma_mhz_per_t(nucleus_new) / gamma_mhz_per_t(nucleus_cal)
        except ValueError:
            expected = None
        if expected:
            measured = sfo1_new_mhz / sfo1_cal_mhz
            if abs(measured / expected - 1.0) > 0.02:
                warnings.append(
                    f"SFO1_new / SFO1_cal = {measured:.5f}, but {nucleus_new}/{nucleus_cal} "
                    f"should be about {expected:.5f}: the two experiments may not be on the same "
                    f"magnet, or a nucleus is mislabelled.")
    if abs(correction_ppm) > 5.0:
        warnings.append(f"A {correction_ppm:.2f} ppm field correction is unusually large -- "
                        f"check SR_cal is in Hz and SFO1 in MHz.")

    return SRTransfer(sr_cal_hz=sr_cal_hz, sfo1_cal_mhz=sfo1_cal_mhz, sfo1_new_mhz=sfo1_new_mhz,
                      correction_ppm=correction_ppm, sr_new_hz=sr_new_hz,
                      nucleus_cal=nucleus_cal, nucleus_new=nucleus_new,
                      sfo1_new_source=source, warnings=warnings)


def spectrum_sr_hz(spectrum, dim: str = "f2") -> float:
    """The SR (Hz) currently in effect on ``spectrum`` for ``dim``
    (``"f2"`` direct, ``"f1"`` indirect): the value the user set in
    Process ▸ Reference if any, otherwise the one the file recorded.
    Same rule as the GUI's SR box, so a notebook reads the same number."""
    raw = spectrum.metadata.raw
    key, recorded_key = (("sr_shift_hz", "sr_recorded_hz") if dim == "f2"
                         else ("sr_shift_hz_f1", "sr_recorded_hz_f1"))
    value = raw.get(key)
    if value is None:
        value = raw.get(recorded_key)
    return float(value) if value else 0.0


def transfer_sr_between(calibrated, target, dim_cal: str = "f2", dim_target: str = "f2") -> SRTransfer:
    """:func:`transfer_sr` with every number read from two loaded
    spectra: the SR in effect on ``calibrated`` and both SFO1s."""
    meta_c, meta_t = calibrated.metadata, target.metadata
    sfo1_c = meta_c.sfo1 if dim_cal == "f2" else meta_c.sfo1_indirect
    sfo1_t = meta_t.sfo1 if dim_target == "f2" else meta_t.sfo1_indirect
    nuc_c = meta_c.nucleus if dim_cal == "f2" else meta_c.nucleus_indirect
    nuc_t = meta_t.nucleus if dim_target == "f2" else meta_t.nucleus_indirect
    if not sfo1_c or not sfo1_t:
        raise ValueError("Both spectra need an SFO1 for this dimension.")
    return transfer_sr(spectrum_sr_hz(calibrated, dim_cal), sfo1_c, sfo1_t,
                       nucleus_cal=nuc_c, nucleus_new=nuc_t)
