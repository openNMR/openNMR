"""SR transfer between nuclei (notebook.referencing)."""

import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import referencing as ref


def test_transfer_follows_the_stated_formula():
    result = ref.transfer_sr(-35.2, 400.1326, 100.6228)
    assert result.correction_ppm == pytest.approx(-35.2 / 400.1326)
    assert result.sr_new_hz == pytest.approx(-35.2 / 400.1326 * 100.6228)
    assert result.sfo1_new_source == "sfo1"


def test_report_carries_the_validity_warning():
    text = ref.transfer_sr(12.0, 600.0, 150.9, "1H", "13C").report()
    assert "same magnet" in text and "similar time" in text
    assert "SR_new" in text


def test_gamma_estimate_when_sfo1_new_is_unknown():
    result = ref.transfer_sr(-35.2, 400.1326, None, nucleus_cal="1H", nucleus_new="13C")
    # gamma(13C)/gamma(1H) = 0.25150; the real SFO1 would be 100.6228
    assert result.sfo1_new_mhz == pytest.approx(100.63, abs=0.02)
    assert result.sfo1_new_source == "gamma"
    assert any("estimated" in w for w in result.warnings)
    with pytest.raises(ValueError):
        ref.transfer_sr(1.0, 400.0)


def test_mismatched_sfo1_and_nuclei_is_flagged():
    result = ref.transfer_sr(5.0, 400.0, 161.9, nucleus_cal="1H", nucleus_new="13C")  # 161.9 is 31P
    assert any("same magnet" in w or "mislabelled" in w for w in result.warnings)
    assert not ref.transfer_sr(5.0, 400.0, 100.6, "1H", "13C").warnings


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        ref.transfer_sr(1.0, 0.0, 100.0)
    with pytest.raises(ValueError):
        ref.transfer_sr(1.0, 400.0, -1.0)


def _spectrum(nucleus, sfo1, **raw):
    meta = NMRMetadata(vendor="bruker", nucleus=nucleus, sfo1=sfo1, raw=dict(raw))
    import numpy as np
    return NMRSpectrum(data=np.zeros(4), axes=[NMRAxis(kind="frequency", ppm=np.arange(4.0))], metadata=meta)


def test_between_spectra_uses_the_sr_in_effect():
    cal = _spectrum("1H", 400.0, sr_recorded_hz=8.0, sr_shift_hz=-20.0)  # the user's value wins
    target = _spectrum("13C", 100.6)
    result = ref.transfer_sr_between(cal, target)
    assert result.sr_cal_hz == -20.0
    assert result.sr_new_hz == pytest.approx(-20.0 / 400.0 * 100.6)
    assert ref.spectrum_sr_hz(_spectrum("1H", 400.0, sr_recorded_hz=8.0)) == 8.0
