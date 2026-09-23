"""Bruker PFG gradient info without a ``diff.xml``.

The failure this file exists to prevent: ``difflist`` means two
different things depending on how the experiment was set up, and both
readings produce a plausible-looking decay curve. A diff-module
``difflist`` holds b-values (1e6 .. 1e11 s/m^2); a ``dosy``-AU one
holds gradient amplitudes (1 .. 3000 G/cm). Confusing them is wrong by
twenty orders of magnitude and still fits.
"""
import os

import numpy as np
import pytest

from opennmr.io import bruker_pfg
from opennmr.processing import diffusion

_DOSY = os.path.join("test_data", "NMR400_KT_DPC_ALSK_AL_70B_3", "10")
_DIFF_MODULE = os.path.join("test_data", "bruker_PFG", "150")
_DIFF_MODULE_7LI = os.path.join(
    "test_data", "Diff50_eNMR_Li_Ac_6p25M_2024-09-03", "11")

needs_dosy = pytest.mark.skipif(
    not os.path.isdir(_DOSY), reason=f"needs the real dataset at {_DOSY}")
needs_diff_module = pytest.mark.skipif(
    not os.path.isdir(_DIFF_MODULE), reason=f"needs the real dataset at {_DIFF_MODULE}")


# ============================================================
# Pulse-programme parsing, on text alone
# ============================================================

_LEDBPGP2S_TAIL = """
  p1 ph1
  p30:gp6*diff
  d16
  p2 ph1
  p30:gp6*-1*diff
  d16
;p1  : f1 channel -  90 degree high power pulse
;p19: gradient pulse 2 (spoil gradient)
;p30: gradient pulse (little DELTA * 0.5)
;d16: delay for gradient recovery
;d20: diffusion time (big DELTA)
;d21: eddy current delay (Te)			[5 ms]
"""

_MONOPOLAR_TAIL = """
  p30:gp6*diff
  d16
;p30: gradient pulse (little DELTA)
;d16: delay for gradient recovery
;d20: diffusion time (big DELTA)
"""


def test_reads_the_programmes_own_parameter_documentation():
    doc = bruker_pfg.parse_pp_parameter_doc(_LEDBPGP2S_TAIL)
    assert doc["p30"] == "gradient pulse (little DELTA * 0.5)"
    assert doc["d20"] == "diffusion time (big DELTA)"


def test_bipolar_programme_takes_delta_as_twice_p30():
    found = bruker_pfg.describe_diffusion_pulseprogram(_LEDBPGP2S_TAIL, "ledbpgp2s")
    assert found.resolved
    assert (found.delta_param, found.delta_factor) == ("p30", 2.0)
    assert found.big_delta_param == "d20"
    assert found.tau_param == "d16"
    # Read from the body's "*-1*", not from the sequence's name.
    assert found.bipolar and found.sequence == "bipolar"
    assert found.gradient_channel == 6


def test_monopolar_programme_takes_delta_as_p30_itself():
    found = bruker_pfg.describe_diffusion_pulseprogram(_MONOPOLAR_TAIL, "ledgp2s")
    assert (found.delta_param, found.delta_factor) == ("p30", 1.0)
    assert not found.bipolar and found.sequence == "pgste"


def test_an_undocumented_programme_reports_unresolved_rather_than_guessing():
    found = bruker_pfg.describe_diffusion_pulseprogram(
        ";d18: gradient duration\n;d17: gradient ramp up time\n", "diffSte")
    assert not found.resolved
    assert found.delta_param is None
    assert any("does not document" in note for note in found.notes)


# ============================================================
# The two meanings of difflist
# ============================================================

@needs_dosy
def test_dosy_au_difflist_is_read_as_gradients():
    listed = bruker_pfg.read_difflist(_DOSY)
    assert listed["kind"] == "gradient"
    assert listed["gradient_g_per_cm"][0] == pytest.approx(1.046)
    assert listed["gradient_g_per_cm"][-1] == pytest.approx(51.244)
    assert listed["b"] is None
    assert not bruker_pfg.is_diff_module_experiment(_DOSY)


@needs_diff_module
def test_diff_module_difflist_is_read_as_b_values():
    from opennmr.io import bruker

    listed = bruker_pfg.read_difflist(_DIFF_MODULE)
    assert listed["kind"] == "b_value"
    assert listed["gradient_g_per_cm"] is None
    # Not a heuristic guess: it is diff.xml's own B, digit for digit.
    assert np.allclose(listed["b"], bruker.read_diff_xml(_DIFF_MODULE)["B"])
    assert bruker_pfg.is_diff_module_experiment(_DIFF_MODULE)


@needs_diff_module
def test_magnitude_alone_would_also_have_caught_it():
    # The diff-module files are the primary signal, but the magnitude
    # guard has to stand on its own for a dataset stripped of them.
    listed = bruker_pfg.read_difflist(_DIFF_MODULE)
    assert listed["b"].min() > bruker_pfg._MAX_PLAUSIBLE_G_PER_CM


# ============================================================
# The whole description
# ============================================================

@needs_dosy
def test_dosy_dataset_resolves_every_value_and_says_where_from():
    found = bruker_pfg.read_pfg_params(_DOSY)
    assert found.pulseprogram == "ledbpgp2s"
    assert found.nucleus == "1H"
    assert found.sequence == "bipolar"
    assert found.delta == pytest.approx(4.42e-3)     # 2 x p30
    assert found.big_delta == pytest.approx(60e-3)   # d20
    assert found.tau == pytest.approx(0.2e-3)        # d16
    assert found.shape == "rect"
    assert found.complete
    assert found.warnings == []
    # Every number is attributed, so a wrong one can be traced.
    assert "P30 x 2" in found.sources["delta"]
    assert "D20" in found.sources["big_delta"]
    assert found.sources["gradient"] == "difflist (G/cm)"


@needs_dosy
def test_difflist_is_reproduced_by_the_four_files_behind_it():
    # This is what pins the convention: difflist already carries the
    # gradient shape's integral factor, so nothing downstream may apply
    # a shape factor again. If this identity ever fails, read_pfg_params
    # warns instead of silently returning a b-axis that is off by the
    # square of the factor.
    from opennmr.io.jcampdx import parse_jcampdx_params

    calib = bruker_pfg.read_gradient_calib(_DOSY)
    integ = bruker_pfg.read_shape_integral_factor(_DOSY, 6)
    ramp = bruker_pfg.read_difframp(_DOSY)
    gpz = parse_jcampdx_params(os.path.join(_DOSY, "acqus"))["GPZ"][6]
    assert (calib, integ, gpz) == (pytest.approx(53.5), pytest.approx(0.97737556561086),
                                   pytest.approx(100.0))

    rebuilt = calib * integ * (gpz / 100.0) * ramp
    listed = bruker_pfg.read_difflist(_DOSY)["gradient_g_per_cm"]
    # difflist is written to three decimals, so the comparison is on
    # absolute error against that rounding, not a per-element relative
    # one (which the smallest entry could never pass).
    assert np.max(np.abs(rebuilt - listed)) < 1e-3
    assert np.max(np.abs(rebuilt - listed)) / np.max(listed) < 2e-5


@needs_dosy
def test_b_axis_fits_one_consistent_D_across_the_solutes_peaks():
    from opennmr import io
    from opennmr.processing import relaxation

    found = bruker_pfg.read_pfg_params(_DOSY)
    spectrum = io.load(_DOSY, pseudo_2d=True)
    ppm = spectrum.axis_values(-1)
    data = np.real(spectrum.data)
    assert data.shape[0] == found.b.size

    fitted = []
    for centre in (0.09, 0.24, 0.34):          # three resolved solute peaks
        window = (ppm > centre - 0.05) & (ppm < centre + 0.05)
        result = relaxation.fit_relaxation(
            found.b, data[:, window].sum(axis=1), model="diffusion",
            custom_model=diffusion.build_diffusion_model(1))
        assert result["fit_quality"]["rmse"] / result["params"]["I0"] < 0.03
        fitted.append(result["params"]["D"])

    # Same molecule, so one D -- peaks agreeing to 1% is a check on the
    # b-axis, not on the fitter.
    assert np.std(fitted) / np.mean(fitted) < 0.01
    assert fitted[0] == pytest.approx(1.78e-10, rel=0.05)


@needs_diff_module
def test_diff_xml_still_wins_where_it_exists():
    from opennmr.io import bruker

    found = bruker_pfg.read_pfg_params(_DIFF_MODULE)
    assert found.sources["b"] == "diff.xml"
    assert np.allclose(found.b, bruker.read_diff_xml(_DIFF_MODULE)["B"])
    # The diff-module sequences document no "little DELTA", so the
    # pulse-programme path must not be what answered here.
    assert found.delta == pytest.approx(1e-3)


@pytest.mark.skipif(not os.path.isdir(_DIFF_MODULE_7LI),
                    reason=f"needs the real dataset at {_DIFF_MODULE_7LI}")
def test_non_proton_diff_module_dataset():
    found = bruker_pfg.read_pfg_params(_DIFF_MODULE_7LI)
    assert found.nucleus == "7Li"
    assert found.gamma == pytest.approx(diffusion.gamma_for("7Li"))
    assert found.complete


# ============================================================
# The GUI seam
# ============================================================

@needs_dosy
def test_as_pfg_params_matches_the_shape_the_gui_already_speaks():
    found = bruker_pfg.read_pfg_params(_DOSY).as_pfg_params()
    assert found["delta"] == pytest.approx(4.42e-3)
    assert found["DELTA"] == pytest.approx(60e-3)
    assert found["sequence"] == "bipolar"
    # delta is already the total encoding time, so the GUI's
    # plateau-plus-ramp arithmetic must be a no-op on it.
    assert diffusion.effective_delta(
        found["delta"], 0.0, found["delta_is_plateau_only"]) == pytest.approx(found["delta"])


@needs_dosy
def test_applying_a_shape_factor_on_top_is_flagged_not_silent():
    found = bruker_pfg.read_pfg_params(_DOSY, shape="smsq10")
    assert any("double-counts" in warning for warning in found.warnings)


def test_missing_dataset_raises_rather_than_returning_empty(tmp_path):
    with pytest.raises(NotADirectoryError):
        bruker_pfg.read_pfg_params(str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="acqus"):
        bruker_pfg.read_pfg_params(str(tmp_path))
