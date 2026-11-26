"""Dynamics ▸ Anisotropic motion, rotation set-up of 2026-11-26: tensors typed in directly,
several rotation axes each with its own rate, open arcs of positions (a step angle), the sense
of rotation, and the notebook builder (``rotation_model``) that gives the same model."""

import math

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.gui.motion_panel import AnisotropicMotionPanel, InteractionDialog
from opennmr.notebook import motion as M
from opennmr.notebook.combined_hamiltonian import Site

from test_gui import qapp  # noqa: F401 -- fixture


@pytest.fixture
def said(monkeypatch):
    messages = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: messages.append(a[2] if len(a) > 2 else ""))
    return messages


def _cd_site(beta=70.5):
    return M.interaction_site("2H", quad=(0.17, 0.0), euler=(0.0, beta, 0.0), label="C-D")


# ------------------------------------------------------------------ arcs of positions

def test_arc_is_an_open_chain_and_a_full_step_is_a_ring():
    arc = M.ring_jump(3, "z", k_Hz=10.0, step_deg=20.0)
    assert arc.periodic is False and "arc" in arc.label
    K = arc.rate_matrix()
    assert np.allclose(K.sum(axis=0), 0.0)
    assert K[1, 0] == K[0, 1] == K[2, 1] == K[1, 2] == 10.0
    assert K[2, 0] == 0.0 and K[0, 2] == 0.0                 # no hop round the back
    angles = [math.degrees(math.atan2(R[1, 0], R[0, 0])) for R in arc.rotations]
    assert angles == pytest.approx([0.0, 20.0, 40.0])
    ring = M.ring_jump(3, "z", k_Hz=10.0, step_deg=120.0)    # n x step = 360: closed
    assert ring.periodic is True and ring.rate_matrix()[2, 0] == 10.0


def test_arc_steps_must_fit_and_one_way_arcs_are_refused():
    with pytest.raises(ValueError):
        M.ring_jump(4, "z", k_Hz=1.0, step_deg=100.0)       # 400 degrees
    with pytest.raises(ValueError):
        M.ring_jump(3, "z", k_Hz=1.0, step_deg=0.0)
    with pytest.raises(ValueError, match="closed ring"):
        M.ring_jump(3, "z", k_Hz=1.0, step_deg=20.0, model="forward")


def test_open_arc_populations_are_equal():
    model = M.rotation_model(_cd_site(), [{"n": 3, "step_deg": 20.0, "k_Hz": 1e5}])
    assert model.populations() == pytest.approx(np.full(3, 1 / 3))


def test_periodic_flag_survives_serialisation_and_old_dicts_stay_rings():
    arc = M.ring_jump(3, "z", k_Hz=10.0, step_deg=20.0)
    assert M.Mode.from_dict(arc.to_dict()).periodic is False
    old = arc.to_dict()
    old.pop("periodic")
    assert M.Mode.from_dict(old).periodic is True
    model = M.MotionModel.from_dict(M.rotation_model(_cd_site(), [{"n": 3, "k_Hz": 1.0}]).to_dict())
    assert model.modes[0].periodic is True


def test_mode_from_spec_takes_a_step():
    m = M.mode_from_spec({"kind": "ring_jump", "n": 3, "axis": "z", "k_Hz": 1e4, "step_deg": "20"})
    assert m.periodic is False and m.n == 3


# ------------------------------------------------------------------ several axes

def test_two_axes_compose_inner_first_and_keep_their_own_rates():
    model = M.rotation_model(
        _cd_site(),
        [{"n": 3, "k_Hz": 1e8},
         {"n": 2, "polar": 55.0, "azimuth": 30.0, "kind": "ring_jump", "k_Hz": 1e4}])
    assert model.n_states == 6
    # state (environment 0, position 1 about axis 1, position 1 about axis 2)
    R = model.state_rotations()[1 * 2 + 1]
    expect = (M.rotation_about_axis((55.0, 30.0), math.pi)
              @ M.rotation_about_axis("z", 2.0 * math.pi / 3.0))
    assert np.allclose(R, expect)
    factors = model.rate_factors()                           # [hops, axis 1, axis 2]
    assert factors[1][1, 0] == pytest.approx(1e8)
    assert factors[2][1, 0] == pytest.approx(1e4)


def test_each_axis_can_be_arrhenius_while_another_has_a_fixed_rate():
    model = M.rotation_model(_cd_site(), [
        {"n": 3, "k_Hz": 1e6},
        {"n": 2, "polar": 90.0, "Ea_eV": 0.3, "nu0_Hz": 1e12}])
    k2 = model.rate_factors(T_K=300.0)[2][1, 0]
    assert k2 == pytest.approx(1e12 * math.exp(-0.3 / (8.617333262e-5 * 300.0)), rel=1e-6)
    assert model.rate_factors(T_K=300.0)[1][1, 0] == pytest.approx(1e6)
    with pytest.raises(ValueError):
        model.rate_factors()                                 # Ea needs a temperature


def test_sense_words_map_to_the_hop_networks():
    for word, expected in (("both", "nearest"), ("anticlockwise", "forward"),
                           ("clockwise", "backward"), ("any", "random")):
        m = M.rotation_model(_cd_site(), [{"n": 4, "sense": word, "k_Hz": 10.0}]).modes[0]
        assert m.model == expected
    with pytest.raises(ValueError):
        M.rotation_model(_cd_site(), [{"n": 4, "sense": "sideways", "k_Hz": 1.0}])


def test_axis_one_is_the_reference_and_axes_are_required():
    with pytest.raises(ValueError, match="reference"):
        M.rotation_model(_cd_site(), [{"n": 3, "polar": 20.0, "k_Hz": 1.0}])
    with pytest.raises(ValueError):
        M.rotation_model(_cd_site(), [])


def test_plain_site_can_be_placed_with_euler():
    site = Site("2H", 0.0, label="D", fwhm=2.0, lineshape="lorentzian").add_quadrupolar(0.17, 0.0)
    model = M.rotation_model(site, [{"n": 3, "k_Hz": 1e5}], euler=(0.0, 70.5, 0.0))
    assert model.sites[0].euler == (0.0, 70.5, 0.0)


def test_methyl_fast_limit_reduces_the_coupling_by_the_geometric_factor():
    model = M.rotation_model(_cd_site(70.5), [{"n": 3, "k_Hz": 1e9}])
    quad = M.averaged_tensors(model)["quad"]
    factor = abs(3.0 * math.cos(math.radians(70.5)) ** 2 - 1.0) / 2.0
    assert abs(quad["C_Q"]) == pytest.approx(0.17e6 * factor, rel=1e-6)
    assert quad["eta_Q"] == pytest.approx(0.0, abs=1e-6)


def test_a_second_axis_changes_the_averaged_tensor():
    one = M.averaged_tensors(M.rotation_model(_cd_site(), [{"n": 3, "k_Hz": 1e9}]))
    two = M.averaged_tensors(M.rotation_model(_cd_site(), [
        {"n": 3, "k_Hz": 1e9}, {"n": 3, "polar": 90.0, "k_Hz": 1e9}]))
    # the unique axis of the averaged tensor is swept round a cone of 90 degrees: P2 = -1/2
    assert abs(two["quad"]["C_Q"]) == pytest.approx(0.5 * abs(one["quad"]["C_Q"]), rel=1e-6)


def test_interaction_site_builds_tensors_directly():
    ms = M.interaction_site("13C", 10.0, csa=(50.0, 0.2), dipolar=[("1H", 1.09)],
                            euler=(0.0, 60.0, 0.0))
    assert ms.euler == (0.0, 60.0, 0.0)
    assert ms.dipolar[0]["D_Hz"] == pytest.approx(M.dipolar_coupling_Hz("13C", "1H", 1.09))
    assert ms.site.get("csa")[0]["aniso"] == 50.0
    explicit = M.interaction_site("13C", dipolar=[("1H", {"D_Hz": 21000.0})])
    assert explicit.dipolar[0]["D_Hz"] == 21000.0


# ------------------------------------------------------------------ the panel

def test_tensors_can_be_typed_in_directly(qapp):
    p = AnisotropicMotionPanel()
    p.sites_table.setRowCount(0)
    row = p.add_interaction("2H", 1.5, label="CD", quad=(0.17, 0.05), euler=(0, 70.5, 0))
    assert row == 0 and p.sites_table.rowCount() == 1
    assert "C_Q 0.17 MHz" in p.sites_table.item(0, 2).text()
    assert p.sites_table.item(0, 4).text() == "70.5"
    model = p.build_model()
    assert model.sites[0].site.delta_iso == 1.5 and model.sites[0].euler[1] == 70.5
    assert p.nucleus() == "2H"


def test_csa_and_dipolar_typed_in_directly(qapp):
    p = AnisotropicMotionPanel()
    p.sites_table.setRowCount(0)
    p.add_interaction("13C", 20.0, csa=(60.0, 0.3), dipolar="1H@1.09", euler=(0, 90, 0))
    model = p.build_model()
    assert model.sites[0].site.get("csa")[0]["aniso"] == 60.0
    assert model.sites[0].dipolar[0]["partner"] == "1H"


def test_an_interaction_can_be_edited_in_place(qapp):
    p = AnisotropicMotionPanel()
    n = p.sites_table.rowCount()
    p.add_interaction("2H", 7.0, label="C-D", quad=(0.2, 0.0), euler=(0, 60, 0), row=0)
    assert p.sites_table.rowCount() == n
    assert p.build_model().sites[0].site.delta_iso == 7.0
    assert p.sites_table.item(0, 4).text() == "60.0"


def test_bad_interactions_are_refused_without_changing_the_table(qapp):
    p = AnisotropicMotionPanel()
    n = p.sites_table.rowCount()
    with pytest.raises(ValueError):
        p.add_interaction("2H", 0.0, dipolar="nonsense")
    with pytest.raises(KeyError):
        p.add_interaction("Xx", 0.0)
    with pytest.raises(ValueError):
        p.add_interaction("13C", 0.0, quad=(1.0, 0.0))        # spin-1/2: no quadrupole
    assert p.sites_table.rowCount() == n


def test_dialog_values_round_trip_a_site(qapp):
    site = Site("2H", 3.0, label="D1", fwhm=4.0, lineshape="gaussian").add_quadrupolar(0.17, 0.1)
    dlg = InteractionDialog(site=site, euler=(10.0, 70.5, 20.0), dipolar="1H@1.09")
    v = dlg.values()
    assert v["nucleus"] == "2H" and v["label"] == "D1" and v["lineshape"] == "gaussian"
    assert v["quad"] == pytest.approx((0.17, 0.1)) and v["csa"] is None
    assert v["euler"] == pytest.approx((10.0, 70.5, 20.0)) and v["dipolar"] == "1H@1.09"
    assert v["delta_iso"] == 3.0 and v["fwhm"] == 4.0
    dlg.csa_box.setChecked(True)
    dlg.aniso_spin.setValue(45.0)
    assert dlg.values()["csa"] == pytest.approx((45.0, 0.0))


def test_dialog_refuses_an_unknown_nucleus(qapp, said):
    dlg = InteractionDialog(nucleus="Xx")
    dlg._check()
    assert said and dlg.result() == 0


def _quick(p, kind="ring_jump", n=3, axis="z", axis_text="", sense="nearest", rate="1e5", step=""):
    p.quick_kind_combo.setCurrentIndex(p.quick_kind_combo.findData(kind))
    if kind in ("ring_jump", "flip"):
        p.quick_n_spin.setValue(n)
    p.quick_axis_combo.setCurrentIndex(p.quick_axis_combo.findData(axis))
    p.quick_axis_edit.setText(axis_text)
    if kind == "ring_jump":
        p.quick_sense_combo.setCurrentIndex(p.quick_sense_combo.findData(sense))
        p.quick_step_edit.setText(step)
    p.quick_rate_edit.setText(rate)


def test_axes_are_added_one_by_one_each_with_its_own_rate(qapp):
    p = AnisotropicMotionPanel()
    _quick(p, n=3, rate="1e8")
    assert p.apply_quick_motion()
    _quick(p, n=2, axis="custom", axis_text="55, 30", rate="2e3")
    assert p.apply_quick_motion(replace=False)
    rows = p._mode_rows()
    assert [float(r["k"]) for r in rows] == [1e8, 2e3] and rows[1]["axis"] == "55, 30"
    model = p.build_model()
    assert [m.n for m in model.modes] == [3, 2] and model.n_states == 6
    assert [m.rate["k_Hz"] for m in model.modes] == [1e8, 2e3]
    assert p.sweep_combo.findData("mode:0") >= 0 and p.sweep_combo.findData("mode:1") >= 0
    assert p.sweep_combo.findData("all") >= 0
    assert "6 states" in p.summary_label.text()


def test_replace_keeps_a_single_axis(qapp):
    p = AnisotropicMotionPanel()
    _quick(p, n=3)
    p.apply_quick_motion()
    _quick(p, n=2, axis="x")
    p.apply_quick_motion(replace=False)
    assert p.modes_table.rowCount() == 2
    _quick(p, n=4)
    p.apply_quick_motion()
    assert p.modes_table.rowCount() == 1 and p._mode_rows()[0]["param"] == "4"


def test_step_and_sense_reach_the_model_and_the_state(qapp, said):
    p = AnisotropicMotionPanel()
    _quick(p, n=3, step="20")
    assert p.apply_quick_motion()
    model = p.build_model()
    assert model.modes[0].periodic is False and model.modes[0].n == 3
    _quick(p, n=3, sense="forward", step="20")
    assert not p.apply_quick_motion() and said               # a one-way arc is refused
    _quick(p, n=4, step="100")
    assert not p.apply_quick_motion()                         # 400 degrees do not fit
    _quick(p, n=3, sense="backward", step="")
    assert p.apply_quick_motion()
    state = p.get_state()
    other = AnisotropicMotionPanel()
    other.set_state(state)
    row = other._mode_rows()[0]
    assert row["sense"] == "backward" and row["step"] == ""
    p._load_modes([{"kind": "ring_jump", "param": "3", "axis": "z", "k": "1e4", "step": "30"}])
    other.set_state(p.get_state())
    assert other._mode_rows()[0]["step"] == "30"
    assert other.build_model().modes[0].periodic is False


def test_moving_an_axis_keeps_its_sense_and_step(qapp):
    p = AnisotropicMotionPanel()
    p._load_modes([
        {"kind": "ring_jump", "param": "3", "axis": "z", "k": "1e6", "sense": "forward"},
        {"kind": "ring_jump", "param": "3", "axis": "50, 0", "k": "1e3", "step": "30"}])
    p.modes_table.setCurrentCell(1, 1)
    p.move_mode(-1)
    rows = p._mode_rows()
    assert rows[0]["step"] == "30" and rows[1]["sense"] == "forward"
    assert [m.rate["k_Hz"] for m in p.build_model().modes] == [1e3, 1e6]


def test_sense_and_step_only_apply_to_ring_jumps(qapp):
    p = AnisotropicMotionPanel()
    p._load_modes([{"kind": "flip", "param": "180", "axis": "x", "k": "1e3"}])
    assert not p.modes_table.cellWidget(0, 6).isEnabled()
    combo = p.modes_table.cellWidget(0, 0)
    combo.setCurrentIndex(combo.findData("ring_jump"))
    assert p.modes_table.cellWidget(0, 6).isEnabled()


def test_summary_says_what_is_missing(qapp):
    p = AnisotropicMotionPanel()
    p.sites_table.setRowCount(0)
    p._refresh_summary()                                     # removing rows sends no cell signal
    assert "Not complete" in p.summary_label.text()
    p.add_interaction("2H", 0.0, quad=(0.17, 0.0))
    assert "states" in p.summary_label.text() and "Not complete" not in p.summary_label.text()


def test_panel_model_matches_the_notebook_builder(qapp):
    p = AnisotropicMotionPanel()
    p.sites_table.setRowCount(0)
    p.add_interaction("2H", 0.0, label="C-D", quad=(0.17, 0.0), euler=(0, 70.5, 0))
    p._load_modes([
        {"kind": "ring_jump", "param": "3", "axis": "z", "k": "1e8"},
        {"kind": "ring_jump", "param": "2", "axis": "55, 30", "k": "1e4"}])
    gui = p.build_model()
    nb = M.rotation_model(
        M.interaction_site("2H", label="C-D", quad=(0.17, 0.0), euler=(0, 70.5, 0), fwhm=2.0),
        [{"n": 3, "k_Hz": 1e8}, {"n": 2, "polar": 55.0, "azimuth": 30.0, "k_Hz": 1e4}])
    assert np.allclose(gui.state_rotations(), nb.state_rotations())
    a, b = M.averaged_tensors(gui), M.averaged_tensors(nb)
    assert a["quad"]["C_Q"] == pytest.approx(b["quad"]["C_Q"])
