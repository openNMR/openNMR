"""Tensor pictures (opennmr.notebook.tensor_views): frequency surfaces
and tensor glyphs for CSA, quadrupolar, both, and J.

The surfaces are checked against textbook values -- principal shifts
along the principal axes, the first-order quadrupolar satellite formula
-- not just against the engine they are drawn from.
"""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from opennmr.notebook import combined_hamiltonian as ch  # noqa: E402
from opennmr.notebook import tensor_views as tv  # noqa: E402

LARMOR_27AL = 104.3e6


def _value_along(surf, theta, phi):
    i = np.argmin(np.abs(surf["theta"][:, 0] - theta))
    j = np.argmin(np.abs(surf["phi"][0, :] - phi))
    return surf["value"][i, j]


def test_isotropic_shift_is_a_sphere_of_radius_delta_iso():
    surf = tv.frequency_surface(ch.Site("13C", delta_iso=120.0))
    assert surf["value"] == pytest.approx(np.full_like(surf["value"], 120.0))


def test_csa_surface_passes_through_the_principal_shifts_along_the_principal_axes():
    site = ch.Site("13C", delta_iso=120.0).add_csa(aniso=-80.0, asym=0.4)   # Haeberlen
    comps = tv.tensor_components(site)
    d_xx, d_yy, d_zz = comps["csa"]["principal"]
    assert d_zz == pytest.approx(120.0 - 80.0)                      # delta_iso + Delta
    assert d_xx + d_yy + d_zz == pytest.approx(3 * 120.0)            # trace = 3 delta_iso
    surf = tv.frequency_surface(site)
    assert _value_along(surf, 0.0, 0.0) == pytest.approx(d_zz)
    assert _value_along(surf, np.pi / 2, 0.0) == pytest.approx(d_xx, abs=1e-6)
    assert _value_along(surf, np.pi / 2, np.pi / 2) == pytest.approx(d_yy, abs=0.2)
    # The surface is not an ellipsoid: at 45 deg in the xz plane the
    # frequency is the mean of d_xx and d_zz (b.D.b), which an ellipsoid
    # of those semi-axes does not give.
    mid = _value_along(surf, np.pi / 4, 0.0)
    assert mid == pytest.approx(0.5 * (d_xx + d_zz), abs=0.6)


def test_anisotropic_part_averages_to_zero_over_the_sphere():
    site = ch.Site("13C", delta_iso=120.0).add_csa(aniso=-80.0, asym=0.4)
    surf = tv.frequency_surface(site, relative_to="iso", n_theta=181, n_phi=361)
    weights = np.sin(surf["theta"])
    assert np.sum(surf["value"] * weights) / np.sum(weights) == pytest.approx(0.0, abs=0.2)


def test_first_order_satellite_surface_matches_the_textbook_formula():
    # I = 5/2: nu_Q = 3 C_Q / (2I(2I-1)); +1/2<->+3/2 at first order is
    # nu_Q (3 cos^2 theta - 1)/2 (eta = 0).
    site = ch.Site("27Al", delta_iso=0.0).add_quadrupolar(C_Q=4.0, eta_Q=0.0)
    surf = tv.frequency_surface(site, LARMOR_27AL, transition="+1/2↔+3/2", order="first")
    nu_q = 3 * 4.0e6 / (2 * 2.5 * 4.0)
    expected = nu_q * (3 * np.cos(surf["theta"]) ** 2 - 1) / 2 / LARMOR_27AL * 1e6
    assert np.abs(surf["value"]) == pytest.approx(np.abs(expected), rel=1e-6, abs=1e-6)


def test_central_transition_has_no_first_order_term():
    site = ch.Site("27Al", delta_iso=10.0).add_quadrupolar(C_Q=4.0, eta_Q=0.3)
    first = tv.frequency_surface(site, LARMOR_27AL, order="first", relative_to="iso")
    assert np.max(np.abs(first["value"])) < 1e-9
    second = tv.frequency_surface(site, LARMOR_27AL, order="second", relative_to="iso")
    assert np.max(np.abs(second["value"])) > 1.0


def test_csa_and_quadrupolar_add_direction_by_direction():
    base = ch.Site("27Al", delta_iso=10.0)
    quad = base.copy().add_quadrupolar(C_Q=4.0, eta_Q=0.3)
    both = quad.copy().add_csa(aniso=60.0, asym=0.2, euler=(0, 40, 0))
    csa_in_efg = ch._csa_frame_tensor(both)
    s_quad = tv.frequency_surface(quad, LARMOR_27AL, relative_to="iso")
    s_both = tv.frequency_surface(both, LARMOR_27AL, relative_to="iso")
    t, p = s_both["theta"], s_both["phi"]
    b = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)
    csa_part = np.einsum("...i,ij,...j->...", b, csa_in_efg, b)
    assert s_both["value"] == pytest.approx(s_quad["value"] + csa_part, abs=1e-9)


def test_euler_angles_tilt_the_csa_axes_in_the_efg_frame():
    site = (ch.Site("27Al").add_quadrupolar(C_Q=4.0, eta_Q=0.3)
            .add_csa(aniso=60.0, asym=0.2, euler=(0, 40, 0)))
    z_csa = tv.tensor_components(site)["csa"]["axes"][:, 2]
    assert np.degrees(np.arccos(z_csa @ [0, 0, 1])) == pytest.approx(40.0)


def test_efg_principal_values_are_traceless_in_c_q_units():
    site = ch.Site("27Al").add_quadrupolar(C_Q=4.0, eta_Q=0.3)
    v_xx, v_yy, v_zz = tv.tensor_components(site)["efg"]["principal"]
    assert v_zz == pytest.approx(4.0)
    assert v_xx + v_yy + v_zz == pytest.approx(0.0)
    assert (v_xx - v_yy) / v_zz == pytest.approx(0.3)                # eta_Q
    assert abs(v_zz) >= abs(v_yy) >= abs(v_xx)


def test_quadrupolar_surface_needs_the_field():
    site = ch.Site("27Al").add_quadrupolar(C_Q=4.0, eta_Q=0.3)
    with pytest.raises(ValueError, match="larmor"):
        tv.frequency_surface(site)


def test_default_transitions():
    assert tv.default_transition(ch.Site("27Al")) == "−1/2↔+1/2"
    assert tv.default_transition(ch.Site("13C")) == "−1/2↔+1/2"
    assert tv.default_transition(ch.Site("2H")) in tv.transition_labels(ch.Site("2H"))
    with pytest.raises(ValueError, match="no transition"):
        tv.frequency_surface(ch.Site("13C"), transition="+1/2↔+3/2")


def test_j_changes_no_shape_and_says_so():
    plain = ch.Site("13C", delta_iso=50.0).add_csa(aniso=40.0, asym=0.0)
    coupled = plain.copy().add_jcoupling("31P", n=1, J_Hz=150.0)
    assert tv.frequency_surface(coupled)["value"] == pytest.approx(
        tv.frequency_surface(plain)["value"])
    _fig, notes = tv.plot_tensor_surface(coupled, larmor_Hz=100e6, figure=plt.figure())
    assert any("J" in n and "shape" in n for n in notes)


def test_big_j_is_drawn_as_copies_of_the_surface():
    # A J that is a sizeable fraction of the radius: shown as wireframe copies.
    site = ch.Site("1H", delta_iso=2.0).add_jcoupling("31P", n=1, J_Hz=200.0)
    fig, notes = tv.plot_tensor_surface(site, larmor_Hz=400e6, figure=plt.figure())
    ax = fig.axes[0]
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    assert any(isinstance(c, Line3DCollection) for c in ax.collections)
    assert any("wireframe" in n for n in notes)


def test_surface_plot_is_equal_aspect_with_a_signed_colour_bar():
    site = ch.Site("13C", delta_iso=120.0).add_csa(aniso=-80.0, asym=0.4)
    fig, _notes = tv.plot_tensor_surface(site, relative_to="iso", figure=plt.figure())
    ax = fig.axes[0]
    assert ax.get_xlim() == ax.get_ylim() == ax.get_zlim()
    assert "δ" in fig.axes[1].get_ylabel()                              # the colour bar
    assert "13C" in fig._suptitle.get_text()


def test_ellipsoid_glyphs_and_their_caveats():
    both = (ch.Site("27Al", delta_iso=10.0).add_quadrupolar(C_Q=4.0, eta_Q=0.3)
            .add_csa(aniso=60.0, asym=0.2, euler=(0, 40, 0)).add_jcoupling("31P", n=1, J_Hz=15))
    _fig, notes = tv.plot_tensor_ellipsoids(both, figure=plt.figure())
    text = " ".join(notes)
    assert "traceless" in text and "Euler" in text and "no glyph" in text
    assert "mixed signs" in text          # delta_ii = 70, -14, -26 ppm here
    _fig, notes = tv.plot_tensor_ellipsoids(ch.Site("13C", delta_iso=120.0), figure=plt.figure())
    assert any("sphere of radius" in n for n in notes)
    _fig, notes = tv.plot_tensor_ellipsoids(
        ch.Site("13C", delta_iso=120.0).add_csa(aniso=-80.0, asym=0.4), figure=plt.figure())
    assert not any("mixed signs" in n for n in notes)                  # all positive: faithful


def test_plot_tensors_dispatches_and_rejects_unknown_styles():
    site = ch.Site("13C", delta_iso=120.0).add_csa(aniso=-80.0, asym=0.4)
    for style in ("surface", "ellipsoid"):
        fig, notes = tv.plot_tensors(site, style=style, figure=plt.figure())
        assert notes
    with pytest.raises(ValueError):
        tv.plot_tensors(site, style="glyph")
