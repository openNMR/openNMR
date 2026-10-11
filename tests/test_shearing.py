"""Shearing and scaling of 2D spectra (``opennmr.notebook.shearing``).

The reference for every shear test is built independently of the code
under test: a hypercomplex 2D made crystallite by crystallite, in which
the *expected* sheared spectrum is simply the same sum with each F1
interferogram's frequency moved to ``nu1 - kappa * nu2(bin)`` -- the
definition of a shear parallel to t2 in the mixed (t1, omega2) domain
(Grandinetti 2011, Eq. 193; checked numerically in
SCRIPT_openNMR_Shearing_20261009.ipynb).
"""

import math
import warnings

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import shearing as sh

TWO_PI = 2 * np.pi


def _bins(n, sw):
    return -np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sw))


def make_2d(crystallites, *, n1=48, n2=200, zf1=2, zf2=2, sw1=20000.0, sw2=50000.0,
            t2_decay=1.5e-3, t1_decay=2.0e-3, kappa_t2=0.0, t1_0=0.0, sf=104.2,
            carrier_hz=(800.0, 800.0), hypercomplex=True, f1_phc=(0.0, 0.0)):
    """A hypercomplex (F1-complex, F2-absorptive) processed 2D, in
    openNMR's conventions (descending axes, fftshifted bins, a peak at
    label frequency nu oscillating as exp(-i 2 pi nu t)).

    ``crystallites`` : iterable of (nu1, nu2, weight), carrier-relative Hz.
    ``kappa_t2`` : build the *sheared* spectrum directly (the reference).
    """
    N1, N2 = n1 * zf1, n2 * zf2
    t1 = t1_0 + np.arange(n1) / sw1
    t2 = np.arange(n2) / sw2
    nu2_bins = _bins(N2, sw2)
    out = np.zeros((N1, N2), complex)
    for nu1, nu2, w in crystallites:
        f2 = np.zeros(N2, complex)
        f2[:n2] = np.exp(-1j * TWO_PI * nu2 * t2) * np.exp(-t2 / t2_decay)
        l2 = np.real(np.fft.fftshift(np.fft.fft(f2)))
        keep = np.abs(l2) > 1e-6 * np.abs(l2).max()
        f1 = np.zeros((N1, N2), complex)
        f1[:n1, keep] = (np.exp(-1j * TWO_PI * np.outer(t1, nu1 - kappa_t2 * nu2_bins[keep]))
                         * np.exp(-(t1 - t1_0) / t1_decay)[:, None])
        out += w * np.fft.fftshift(np.fft.fft(f1, axis=0), axes=0) * l2[None, :]
    if any(f1_phc):
        from opennmr.io import bruker as io_bruker
        out = io_bruker.apply_recorded_phase(out, f1_phc[0], f1_phc[1], axis=0)
    c1, c2 = carrier_hz
    y_hz = np.linspace(c1 + sw1 / 2, c1 - sw1 / 2, N1)
    x_hz = np.linspace(c2 + sw2 / 2, c2 - sw2 / 2, N2)
    raw = {"fid_dt": 1.0 / sw2, "fid_dt_f1": 1.0 / sw1, "fn_mode": 5,
           "fid": np.zeros((2, 2)), "fid_raw": np.zeros((2, 2)),
           "recorded_phase0_f1": f1_phc[0], "recorded_phase1_f1": f1_phc[1],
           "acqus": {"MASR": 12500}}
    meta = NMRMetadata(vendor="bruker", nucleus="27Al", nucleus_indirect="27Al", sfo1=sf,
                       sfo1_indirect=sf, homonuclear=True, raw=raw, source_path="synthetic")
    axes = [NMRAxis(kind="frequency", ppm=y_hz / sf, hz=y_hz),
            NMRAxis(kind="frequency", ppm=x_hz / sf, hz=x_hz)]
    return NMRSpectrum(data=out.real.copy(), axes=axes, metadata=meta,
                       data_imag=out.imag.copy() if hypercomplex else None)


def two_site_mqmas(f1_sign=1, n_cryst=41, spin=1.5, sf=104.2, carrier=800.0):
    """Crystallites of two quadrupolar sites seen through the I = 3/2
    3QMAS echo pathway: delta_cs, delta_qis(CT) in ppm and a rank-4
    anisotropy spread over +-3 kHz on the CT."""
    ct = sh.symmetric_coherence(spin, -1)
    mq = sh.symmetric_coherence(spin, -3)
    sites = [(10.0, -6.0, 1.0), (-15.0, -12.0, 0.7)]
    out = []
    for dcs, dqis, amp in sites:
        u_cs = sf * dcs - carrier                      # carrier-relative CT CS offset (Hz)
        u_q = sf * dqis / ct["qis"]                    # per unit of the QIS coefficient
        for x in np.linspace(-1, 1, n_cryst):
            w4 = 3000.0 * x / ct["q4"]
            weight = amp * math.exp(-2 * x * x)
            nu2 = ct["cs"] * u_cs + ct["qis"] * u_q + ct["q4"] * w4
            nu1 = f1_sign * (mq["cs"] * u_cs + mq["qis"] * u_q + mq["q4"] * w4)
            out.append((nu1, nu2, weight))
    return out, sites


# ---------------------------------------------------------------------------
# Coefficients against the literature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spin,order,kappa", [
    (1.5, 3, -7 / 9), (2.5, 3, -19 / 12), (2.5, 5, -25 / 12),
    (3.5, 3, -101 / 45), (4.5, 3, -91 / 36),
])
def test_mqmas_ratios(spin, order, kappa):
    plan = sh.make_shear_plan("mqmas", spin=spin, mq_order=order)
    assert plan.kappa_t2 == pytest.approx(kappa, rel=1e-12)
    # chemical-shift convention: delta_F1 = delta_CS - 10/17 delta_QIS, for every spin
    assert plan.f1_cs_factor == pytest.approx(1.0)
    assert plan.f1_qis_factor == pytest.approx(-10 / 17)


@pytest.mark.parametrize("spin,kappa", [(1.5, -8 / 9), (2.5, -7 / 24), (3.5, -28 / 45),
                                        (4.5, -55 / 72)])
def test_stmas_ratios(spin, kappa):
    plan = sh.make_shear_plan("stmas", spin=spin)
    assert plan.kappa_t2 == pytest.approx(kappa, rel=1e-12)
    assert plan.f1_qis_factor == pytest.approx(-10 / 17)


def test_unified_convention_factors():
    p32 = sh.make_shear_plan("mqmas", spin=1.5, convention="unified")
    assert p32.f1_cs_factor == pytest.approx(17 / 8)
    # the well-known +1/32 (P_Q/nu0)^2 1e6 of I = 3/2: delta_QIS(CT) = -(1/40)(P_Q/nu0)^2
    assert p32.f1_qis_factor * (-1 / 40) == pytest.approx(1 / 32)
    p52 = sh.make_shear_plan("mqmas", spin=2.5, convention="unified")
    assert p52.f1_cs_factor == pytest.approx(17 / 31)


def test_nucleus_lookup_and_presets():
    assert sh.spin_of("27Al") == 2.5
    assert sh.spin_of("Na23") == 1.5
    labels = [p["label"] for p in sh.preset_kappas(2.5)]
    assert any("3Q" in x for x in labels) and any("ST1" in x for x in labels)
    plan = sh.make_shear_plan("mqmas", nucleus="23Na")
    assert plan.kappa_t2 == pytest.approx(-7 / 9)


def test_f1_sign_flips_ratio_but_not_meaning():
    plus = sh.make_shear_plan("mqmas", spin=1.5)
    minus = sh.make_shear_plan("mqmas", spin=1.5, f1_sign=-1)
    assert minus.kappa_t2 == pytest.approx(-plus.kappa_t2)
    assert minus.f1_cs_factor == pytest.approx(plus.f1_cs_factor)
    assert minus.f1_qis_factor == pytest.approx(plus.f1_qis_factor)
    assert minus.scale_f1 < 0 < plus.scale_f1


def test_invalid_inputs():
    with pytest.raises(ValueError):
        sh.make_shear_plan("mqmas", spin=0.5)
    with pytest.raises(ValueError):
        sh.make_shear_plan("mqmas", spin=1.5, mq_order=5)
    with pytest.raises(ValueError):
        sh.make_shear_plan("jres", convention="cs")     # silently "none" for jres...
    # ...but only when asked by name: the default is accepted
    sh.make_shear_plan("jres")
    with pytest.raises(ValueError):
        sh.make_shear_plan("custom_coefficients")


# ---------------------------------------------------------------------------
# The shear itself, against an independently built reference
# ---------------------------------------------------------------------------

def _rel_err(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_complex_route_matches_reference_exactly():
    cryst, _ = two_site_mqmas()
    kappa = -7 / 9
    src = make_2d(cryst)
    ref = make_2d(cryst, kappa_t2=kappa)
    plan = sh.make_shear_plan("custom_kappa", kappa_t2=kappa, convention="none")
    out = sh.shear_2d(src, plan)
    assert out.metadata.raw["shear"]["routes"]["shear parallel to t2"].startswith("complex")
    assert _rel_err(out.data, ref.data) < 1e-9
    assert _rel_err(out.data_imag, ref.data_imag) < 1e-9


def test_complex_route_needs_no_zero_fill():
    cryst, _ = two_site_mqmas()
    src = make_2d(cryst, zf1=1)
    ref = make_2d(cryst, zf1=1, kappa_t2=-7 / 9)
    out = sh.shear_2d(src, sh.make_shear_plan("custom_kappa", kappa_t2=-7 / 9, convention="none"))
    assert _rel_err(out.data, ref.data) < 1e-9


def test_real_route_exact_when_zero_filled_twofold():
    cryst, _ = two_site_mqmas()
    src = make_2d(cryst, hypercomplex=False)
    ref = make_2d(cryst, kappa_t2=-7 / 9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # no warning expected here
        out = sh.shear_2d(src, sh.make_shear_plan("custom_kappa", kappa_t2=-7 / 9,
                                                  convention="none"))
    assert out.data_imag is None
    assert _rel_err(out.data, ref.data) < 1e-9


def test_real_route_warns_without_zero_fill():
    cryst, _ = two_site_mqmas()
    src = make_2d(cryst, zf1=1, hypercomplex=False, t1_decay=1.0)
    with pytest.warns(UserWarning, match="zero-filled less than twofold"):
        out = sh.shear_2d(src, sh.make_shear_plan("custom_kappa", kappa_t2=-7 / 9,
                                                  convention="none"))
    assert out.metadata.raw["shear"]["warnings"]


def test_first_increment_time():
    cryst, _ = two_site_mqmas()
    t1_0 = 40e-6
    src = make_2d(cryst, t1_0=t1_0)
    ref = make_2d(cryst, t1_0=t1_0, kappa_t2=-7 / 9)
    plan = sh.make_shear_plan("custom_kappa", kappa_t2=-7 / 9, convention="none")
    good = sh.shear_2d(src, plan, t1_0=t1_0)
    bad = sh.shear_2d(src, plan)
    assert _rel_err(good.data, ref.data) < 1e-9
    assert _rel_err(bad.data, ref.data) > 1e-2


def test_recorded_f1_phase_is_carried_through():
    """A first-order F1 phase does not commute with a shift along F1;
    with the imaginary part present it is undone and redone exactly."""
    cryst, _ = two_site_mqmas()
    phc = (20.0, 35.0)
    src = make_2d(cryst, f1_phc=phc)
    ref = make_2d(cryst, kappa_t2=-7 / 9, f1_phc=phc)
    out = sh.shear_2d(src, sh.make_shear_plan("custom_kappa", kappa_t2=-7 / 9, convention="none"))
    assert _rel_err(out.data, ref.data) < 1e-9
    # and without the imaginary part it is flagged rather than silently wrong
    real_only = make_2d(cryst, f1_phc=phc, hypercomplex=False)
    with pytest.warns(UserWarning, match="first-order phase"):
        sh.shear_2d(real_only, sh.make_shear_plan("custom_kappa", kappa_t2=-7 / 9,
                                                  convention="none"))


# ---------------------------------------------------------------------------
# What the result means
# ---------------------------------------------------------------------------

def _peak_f1_ppm(spec, f2_window_ppm):
    x = spec.axes[1].ppm
    cols = (x <= max(f2_window_ppm)) & (x >= min(f2_window_ppm))
    proj = np.abs(spec.data[:, cols]).sum(axis=1)
    i = int(np.argmax(proj))
    return float(spec.axes[0].ppm[i]), proj


def test_mqmas_isotropic_positions_and_decoding():
    """After the shear and the chemical-shift scaling each site sits in
    F1 at delta_CS - 10/17 delta_QIS, and the two positions decode back
    to delta_CS and delta_QIS."""
    sf = 104.2
    cryst, sites = two_site_mqmas(sf=sf)
    src = make_2d(cryst, sf=sf)
    plan = sh.make_shear_plan("mqmas", spin=1.5)
    out = sh.shear_2d(src, plan)
    y = out.axes[0].ppm
    assert y[0] > y[-1]                                  # still descending
    dppm = abs(y[1] - y[0])
    for dcs, dqis, _ in sites:
        f2_cog = dcs + dqis
        expect_f1 = dcs - 10 / 17 * dqis
        got, proj = _peak_f1_ppm(out, (f2_cog - 30, f2_cog + 30))
        assert got == pytest.approx(expect_f1, abs=1.5 * dppm)
        # the ridge is now parallel to F2: one sharp line in F1
        assert (proj > 0.5 * proj.max()).sum() <= 4
        dec = sh.isotropic_from_peak(plan, expect_f1, f2_cog, spin=1.5, larmor_mhz=sf)
        assert dec["delta_cs"] == pytest.approx(dcs)
        assert dec["delta_qis"] == pytest.approx(dqis)
        pq = math.sqrt(-dqis * 1e-6 * 40 * sf ** 2)      # I = 3/2: delta_QIS = -(1/40)(PQ/nu0)^2
        assert dec["PQ_MHz"] == pytest.approx(pq)
    log = out.metadata.raw["shear"]["log"]
    assert "-7/9" in log and "10/17" in log


def test_unified_scaling_relabels_only():
    cryst, _ = two_site_mqmas()
    src = make_2d(cryst)
    cs = sh.shear_2d(src, sh.make_shear_plan("mqmas", spin=1.5))
    uni = sh.shear_2d(src, sh.make_shear_plan("mqmas", spin=1.5, convention="unified"))
    np.testing.assert_allclose(cs.data, uni.data)        # same data ...
    ratio = (cs.axes[0].hz[0] - cs.axes[0].hz[-1]) / (uni.axes[0].hz[0] - uni.axes[0].hz[-1])
    assert ratio == pytest.approx((1 + 7 / 9) / (3 + 7 / 9))   # ... different axis
    assert uni.metadata.raw["shear"]["sw_after"][0] == pytest.approx(20000 / (16 / 9))


def test_reversed_f1_gives_the_same_answer_with_f1_sign():
    cryst_p, _ = two_site_mqmas(f1_sign=1)
    cryst_m, _ = two_site_mqmas(f1_sign=-1)
    normal = sh.shear_2d(make_2d(cryst_p), sh.make_shear_plan("mqmas", spin=1.5))
    flipped = sh.shear_2d(make_2d(cryst_m), sh.make_shear_plan("mqmas", spin=1.5, f1_sign=-1))
    # same positions; the F1 grid differs by one point's reversal, so compare projections
    pn = np.abs(normal.data).sum(axis=1)
    pf = np.abs(flipped.data).sum(axis=1)
    assert abs(int(np.argmax(pn)) - int(np.argmax(pf))) <= 1
    assert normal.axes[0].ppm[0] > normal.axes[0].ppm[-1]
    assert flipped.axes[0].ppm[0] > flipped.axes[0].ppm[-1]


# ---------------------------------------------------------------------------
# Measuring kappa from the data
# ---------------------------------------------------------------------------

def test_estimate_finds_ratio_and_f1_sense():
    for f1_sign in (1, -1):
        cryst, _ = two_site_mqmas(f1_sign=f1_sign)
        spec = make_2d(cryst)
        spec.metadata.nucleus = "23Na"                  # I = 3/2, so the presets match
        scan = sh.estimate_shear(spec)
        assert scan["kappa"] == pytest.approx(f1_sign * -7 / 9, abs=0.02)
        top = scan["matches"][0]
        assert "3Q" in top["label"] and top["f1_sign"] == f1_sign
        assert scan["prominence"] > 0.2
        assert "f1_sign" in scan["summary"]


def test_estimate_on_already_sheared_data():
    cryst, _ = two_site_mqmas()
    spec = make_2d(cryst, kappa_t2=-7 / 9)
    scan = sh.estimate_shear(spec, kappa_range=(-2, 2), n_coarse=81)
    assert abs(scan["kappa"]) < 0.06
    assert "already sheared" in scan["summary"] and "split_t1=True" in scan["summary"]


# ---------------------------------------------------------------------------
# Solution state: the J-resolved tilt
# ---------------------------------------------------------------------------

def test_jres_tilt_collapses_multiplets():
    sf = 400.13
    J = 7.0
    cryst = []
    for delta_hz, n_lines in ((300.0, 3), (-450.0, 4)):
        for k in range(n_lines):
            m = k - (n_lines - 1) / 2
            w = math.comb(n_lines - 1, k)
            cryst.append((m * J, delta_hz + m * J, w))
    spec = make_2d(cryst, n1=32, n2=1024, sw1=60.0, sw2=2000.0, t1_decay=0.3, t2_decay=0.3,
                   sf=sf, carrier_hz=(0.0, 1880.0))
    spec.metadata.nucleus = spec.metadata.nucleus_indirect = "1H"
    out = sh.shear_2d(spec, sh.make_shear_plan("jres"))
    x_hz = out.axes[1].hz - 0.5 * (out.axes[1].hz[0] + out.axes[1].hz[-1])
    proj = np.abs(out.data).sum(axis=0)
    for delta_hz in (300.0, -450.0):
        near = np.abs(x_hz - delta_hz) < 6.0             # within one J of the shift
        far = (np.abs(x_hz - delta_hz) > 6.0) & (np.abs(x_hz - delta_hz) < 20.0)
        assert proj[near].max() > 5 * proj[far].max()
    # F1 (J) is centred on zero Hz, F2 untouched
    assert out.axes[0].hz[len(out.axes[0].hz) // 2] == pytest.approx(0.0, abs=1.0)
    np.testing.assert_allclose(out.axes[1].ppm, spec.axes[1].ppm)
    scan = sh.estimate_shear(spec, direction="t1", kappa_range=(-2, 2), n_coarse=81)
    assert scan["matches"][0]["kappa"] == 1.0 and scan["matches"][0]["f1_sign"] == 1
    assert "f1_sign = +1" in scan["summary"]
    # F1 displayed the other way round: the data say so
    mirrored = make_2d([(-a, b, w) for a, b, w in cryst], n1=32, n2=1024, sw1=60.0, sw2=2000.0,
                       t1_decay=0.3, t2_decay=0.3, sf=sf, carrier_hz=(0.0, 1880.0))
    scan = sh.estimate_shear(mirrored, direction="t1", kappa_range=(-2, 2), n_coarse=81)
    assert scan["matches"][0]["f1_sign"] == -1


# ---------------------------------------------------------------------------
# Sidebands, the notebook dict, bookkeeping
# ---------------------------------------------------------------------------

def test_sideband_positions():
    plan = sh.make_shear_plan("mqmas", spin=1.5, convention="none")
    nr = 12500.0
    pos = {(p["k"], p["j"]): p for p in sh.sideband_positions(plan, nr, sw=(nr, 50000.0))}
    s1 = plan.scale_f1
    assert pos[(1, 0)]["df1"] == pytest.approx(nr / s1)
    assert pos[(1, 0)]["df1_folded"] == pytest.approx(0.0, abs=1e-6)       # rotor-synchronised t1
    assert pos[(0, 1)]["df1"] == pytest.approx(7 / 9 * nr / s1)            # non-integer
    assert pos[(0, 1)]["df2"] == pytest.approx(nr)
    assert abs(pos[(0, 1)]["df1_folded"]) > 1000                           # does not fold away


def test_notebook_dict_input():
    cryst, _ = two_site_mqmas()
    spec = make_2d(cryst, hypercomplex=False)
    d = {"spectrum": spec.data, "x_ppm": spec.axes[1].ppm, "y_ppm": spec.axes[0].ppm,
         "x_Hz": spec.axes[1].hz, "y_Hz": spec.axes[0].hz, "SF1": 104.2, "SF2": 104.2,
         "SW1": 50000.0, "SW2": 20000.0, "SI1": spec.data.shape[1], "SI2": spec.data.shape[0],
         "NUC1": "27Al", "NUC2": "27Al", "pseudo_2d": False, "homonuclear": True}
    plan = sh.make_shear_plan("mqmas", spin=1.5)
    out_d = sh.shear_2d(d, plan)
    out_s = sh.shear_2d(spec, plan)
    assert isinstance(out_d, dict) and "shear" in out_d
    np.testing.assert_allclose(out_d["spectrum"], out_s.data, atol=1e-9 * np.abs(out_s.data).max())
    np.testing.assert_allclose(out_d["y_ppm"], out_s.axes[0].ppm)
    cropped = dict(d, x_ppm=d["x_ppm"][10:], x_Hz=d["x_Hz"][10:])
    with pytest.raises(ValueError, match="display limits"):
        sh.shear_2d(cropped, plan)


def test_rejects_pseudo_2d_and_1d():
    cryst, _ = two_site_mqmas()
    spec = make_2d(cryst)
    spec.metadata.pseudo_2d = True
    with pytest.raises(ValueError, match="pseudo-2D"):
        sh.shear_2d(spec, sh.make_shear_plan("mqmas", spin=1.5))
    one_d = NMRSpectrum(data=np.zeros(8), axes=[NMRAxis(kind="frequency", ppm=np.arange(8.0))],
                        metadata=NMRMetadata(vendor="bruker"))
    with pytest.raises(ValueError, match="2D"):
        sh.shear_2d(one_d, sh.make_shear_plan("mqmas", spin=1.5))


def test_result_metadata_is_clean():
    cryst, _ = two_site_mqmas()
    src = make_2d(cryst)
    out = sh.shear_2d(src, sh.make_shear_plan("mqmas", spin=1.5), source_label="exp 12")
    raw = out.metadata.raw
    assert not any(k.startswith("fid") for k in raw)
    assert "recorded_phase1_f1" not in raw
    assert raw["imag_partner_axis"] == 0
    assert raw["shear"]["source"] == "exp 12"
    assert src.metadata.raw.get("fid") is not None           # the input is untouched
    # a sheared spectrum can be sheared again (its imaginary part is still F1's)
    again = sh.shear_2d(out, sh.make_shear_plan("custom_kappa", kappa_t2=0.0, convention=1.0))
    np.testing.assert_allclose(again.data, out.data)


def test_scan_plot_draws():
    import matplotlib
    matplotlib.use("Agg")
    cryst, _ = two_site_mqmas()
    scan = sh.estimate_shear(make_2d(cryst), kappa_range=(-2, 2), n_coarse=41)
    ax = sh.plot_shear_scan(scan)
    assert ax.lines
    plan = sh.make_shear_plan("mqmas", spin=1.5)
    sh.plot_sideband_markers(ax, sh.sideband_positions(plan, 12500.0, orders_f1=range(-1, 2),
                                                       orders_f2=range(-1, 2)), 0.0, 0.0)


def test_double_shear_moves_peaks_where_the_algebra_says():
    """Shear parallel to t2 then parallel to t1 (Grandinetti Eqs.
    212-216): a narrow peak at (nu1, nu2) must land at
    (nu1 - k2 nu2, nu2 - k1 (nu1 - k2 nu2))."""
    k2, k1 = -0.6, 0.35
    peaks = [(3000.0, -4000.0, 1.0), (-2500.0, 6000.0, 0.8)]
    # F2 zero-filled threefold: after the first shear the F2-absorptive
    # plane's time signal *and its mirror* are displaced by up to
    # |k2| t1max, so twofold leaves no room (the code warns; see below).
    spec = make_2d(peaks, n1=64, n2=256, zf2=3, t1_decay=5e-3, t2_decay=5e-3)
    plan = sh.make_shear_plan("custom_kappa", kappa_t2=k2, kappa_t1=k1, convention="none")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sh.shear_2d(spec, plan)
    tight = make_2d(peaks, n1=64, n2=256, zf2=2, t1_decay=5e-3, t2_decay=5e-3)
    with pytest.warns(UserWarning, match="half-way along the time domain"):
        sh.shear_2d(tight, plan)
    n1, n2 = out.data.shape
    nu1_bins, nu2_bins = _bins(n1, 20000.0), _bins(n2, 50000.0)
    assert n2 == 768
    for nu1, nu2, _ in peaks:
        e1 = nu1 - k2 * nu2
        e2 = nu2 - k1 * e1
        e1 = (e1 + 10000.0) % 20000.0 - 10000.0          # F1 aliases into +-SW1/2
        i = int(np.argmin(np.abs(nu1_bins - e1)))
        j = int(np.argmin(np.abs(nu2_bins - e2)))
        window = out.data[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
        assert window.max() > 0.5 * out.data.max() * (0.8 if nu1 < 0 else 1.0)


def test_das_second_shear_centres_f2():
    plan = sh.make_shear_plan("das", spin=1.5, second_shear=True)
    assert plan.kappa_t2 == pytest.approx(-1.0, abs=1e-3)
    assert plan.cs_after[1] == pytest.approx(0.0, abs=1e-3)   # F2 holds no isotropic part
    assert plan.f1_cs_factor == pytest.approx(1.0)


def test_reachable_from_one_import():
    import opennmr.notebook as nmr
    assert nmr.shear_2d is sh.shear_2d
    assert nmr.shearing is sh


def test_processed_topspin_phase_from_proc2s():
    """TopSpin's processed 2rr/2ri carry proc2s PHC0/PHC1 in the data;
    they are taken off and put back around the shift like a raw read's."""
    cryst, _ = two_site_mqmas()
    phc = (-30.0, 50.0)
    src = make_2d(cryst, f1_phc=phc)
    ref = make_2d(cryst, kappa_t2=-7 / 9, f1_phc=phc)
    for key in ("fid", "fid_raw", "fid_dt", "fid_dt_f1", "fn_mode", "recorded_phase0_f1",
                "recorded_phase1_f1"):
        src.metadata.raw.pop(key, None)
    src.metadata.raw["proc2s"] = {"PHC0": phc[0], "PHC1": phc[1], "SW_p": 20000.0}
    src.metadata.raw["procs"] = {"PHC0": 0.0, "PHC1": 0.0, "SW_p": 50000.0}
    out = sh.shear_2d(src, sh.make_shear_plan("custom_kappa", kappa_t2=-7 / 9, convention="none"))
    assert _rel_err(out.data, ref.data) < 1e-9


def test_shear_and_unshear_round_trip_through_a_reversed_axis():
    """Shear with an F1 that comes out reversed (I = 5/2, chemical shift
    coefficient negative), then shear back: the original returns, F1
    mirrored, with its first-order F1 phase intact -- so the phase
    baked into a sheared spectrum is tracked through the reversal."""
    cryst, _ = two_site_mqmas()
    src = make_2d(cryst, f1_phc=(15.0, 40.0))
    k = -19 / 12
    first = sh.shear_2d(src, sh.make_shear_plan("custom_kappa", kappa_t2=k, cs=(-3.0, 1.0),
                                                convention="none"))
    assert first.metadata.raw["shear"]["plan"]["scale_f1"] < 0       # F1 was reversed
    back = sh.shear_2d(first, sh.make_shear_plan("custom_kappa", kappa_t2=k, convention="none"))
    z_back = back.data + 1j * back.data_imag
    z_src = src.data + 1j * src.data_imag
    expected = np.conj(z_src[::-1])           # F1 mirrored (and so conjugated)
    assert _rel_err(z_back, expected) < 1e-9


# ---------------------------------------------------------------------------
# Echo-antiecho: p and n pathways need opposite shears in their own frames
# ---------------------------------------------------------------------------

def _ea_spectrum(cryst_echo, *, k_e=0.0, k_a=0.0, n1=48, n2=200, sw1=20000.0, sw2=50000.0,
                 sf=104.2, carrier=800.0, align_echo=False):
    """FnMODE-6 row pairs built pathway by pathway (row 2k: the echo,
    evolving at nu1 in t1; row 2k+1: the antiecho, at -nu1), after
    openNMR's F2 FFT, recombined by the reader's own decoder and F1
    transformed. ``k_e``/``k_a`` shear each pathway in its own frame
    before recombination (the reference)."""
    from opennmr.processing.ft import decode_hypercomplex_f1
    N1, N2 = 2 * n1, 2 * n2
    t1, t2 = np.arange(n1) / sw1, np.arange(n2) / sw2
    nu2b = _bins(N2, sw2)
    E = np.zeros((n1, N2), complex)
    A = np.zeros_like(E)
    for nu1, nu2, w in cryst_echo:
        f2 = np.zeros(N2, complex)
        f2[:n2] = np.exp(-1j * TWO_PI * nu2 * t2) * np.exp(-t2 / 1.5e-3)
        line = 0.5j * w * np.fft.fftshift(np.fft.fft(f2))[None, :] * np.exp(-t1 / 2e-3)[:, None]
        E += line * np.exp(-1j * TWO_PI * np.outer(t1, nu1 - k_e * nu2b))
        A += line * np.exp(-1j * TWO_PI * np.outer(t1, -nu1 - k_a * nu2b))
    rows = np.empty((2 * n1, N2), complex)
    rows[0::2], rows[1::2] = E, A
    z = np.zeros((N1, N2), complex)
    z[:n1] = decode_hypercomplex_f1(rows, 6)
    S = np.fft.fftshift(np.fft.fft(z, axis=0), axes=0)
    y = np.linspace(carrier + sw1 / 2, carrier - sw1 / 2, N1)
    x = np.linspace(carrier + sw2 / 2, carrier - sw2 / 2, N2)
    raw = {"fid_dt": 1 / sw2, "fid_dt_f1": 1 / sw1, "fn_mode": 6, "align_echo": align_echo}
    meta = NMRMetadata(vendor="bruker", nucleus="23Na", nucleus_indirect="23Na", sfo1=sf,
                       sfo1_indirect=sf, homonuclear=True, raw=raw)
    return NMRSpectrum(data=S.real.copy(), data_imag=S.imag.copy(), metadata=meta,
                       axes=[NMRAxis(kind="frequency", ppm=y / sf, hz=y),
                             NMRAxis(kind="frequency", ppm=x / sf, hz=x)])


def test_echo_antiecho_shearing_the_recombined_data_shears_p_and_n_oppositely():
    cryst, sites = two_site_mqmas()
    spec = _ea_spectrum(cryst)
    scan = sh.estimate_shear(spec)
    f1_sign = scan["matches"][0]["f1_sign"]
    assert "3Q" in scan["matches"][0]["label"]
    plan = sh.make_shear_plan("mqmas", spin=1.5, f1_sign=f1_sign)
    out = sh.shear_2d(spec, plan)

    # reference: each pathway sheared in its own frame (n with kappa, p with -kappa)
    k = plan.kappa_t2
    relabel = sh.make_shear_plan("custom_kappa", kappa_t2=0.0, cs=plan.cs, convention=plan.scale_f1)
    ref = sh.shear_2d(_ea_spectrum(cryst, k_a=k, k_e=-k), relabel)
    assert _rel_err(out.data, ref.data) < 1e-9
    assert _rel_err(out.data_imag, ref.data_imag) < 1e-9

    # giving both pathways the same kappa is not a small error
    same = _ea_spectrum(cryst, k_a=k, k_e=k)
    opposite = _ea_spectrum(cryst, k_a=k, k_e=-k)
    assert _rel_err(same.data, opposite.data) > 0.3

    # and the sites land at delta_CS - 10/17 delta_QIS
    y, x = out.axes[0].ppm, out.axes[1].ppm
    dppm = abs(y[1] - y[0])
    for dcs, dqis, _ in sites:
        cols = np.abs(x - (dcs + dqis)) < 30
        got = y[int(np.argmax(np.abs(out.data[:, cols]).sum(axis=1)))]
        assert got == pytest.approx(dcs - 10 / 17 * dqis, abs=1.5 * dppm)


def test_warns_when_rows_were_aligned_to_their_own_echo_tops():
    cryst, _ = two_site_mqmas()
    spec = _ea_spectrum(cryst, align_echo=True)
    with pytest.warns(UserWarning, match="Align Echo"):
        out = sh.shear_2d(spec, sh.make_shear_plan("mqmas", spin=1.5, f1_sign=-1))
    assert any("Align Echo" in w for w in out.metadata.raw["shear"]["warnings"])


# ---------------------------------------------------------------------------
# Split-t1: the sequence shears (and scales) the data itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spin,frac_mq,frac_ct", [(1.5, "9/16", "7/16"), (2.5, "12/31", "19/31")])
def test_split_t1_plan(spin, frac_mq, frac_ct):
    cs = sh.make_shear_plan("mqmas", spin=spin, split_t1=True)
    assert cs.kappa_t2 == 0.0
    assert cs.f1_cs_factor == pytest.approx(1.0)
    assert cs.f1_qis_factor == pytest.approx(-10 / 17)
    text = cs.describe()
    assert frac_mq in text and frac_ct in text and "No shear" in text
    uni = sh.make_shear_plan("mqmas", spin=spin, split_t1=True, convention="unified")
    assert abs(uni.scale_f1) == pytest.approx(1.0)             # already in the unified scale
    assert abs(uni.f1_cs_factor) == pytest.approx(17 / 8 if spin == 1.5 else 17 / 31)
    with pytest.raises(ValueError):
        sh.make_shear_plan("jres", split_t1=True)


def _split_t1_crystallites(spin=1.5, sf=104.2, carrier=800.0):
    """Split-t1 data: each crystallite evolves in t1 at the time-weighted
    mix (nu_MQ - kappa nu_CT) / (1 + |kappa|) -- refocused crystallite by
    crystallite by the sequence, not bin by bin."""
    cryst, sites = two_site_mqmas(spin=spin, sf=sf, carrier=carrier)
    k = sh.make_shear_plan("mqmas", spin=spin).kappa_t2
    return [((nu1 - k * nu2) / (1 + abs(k)), nu2, w) for nu1, nu2, w in cryst], sites


def test_split_t1_data_land_where_the_literature_says():
    cryst, sites = _split_t1_crystallites()
    spec = make_2d(cryst, sw1=10000.0)
    spec.metadata.nucleus = spec.metadata.nucleus_indirect = "23Na"
    scan = sh.estimate_shear(spec, kappa_range=(-2, 2), n_coarse=81)
    assert abs(scan["kappa"]) < 0.06 and "split-t1" in scan["summary"]

    for convention, a, b in (("cs", 1.0, -10 / 17), ("unified", 17 / 8, -5 / 4)):
        out = sh.shear_2d(spec, sh.make_shear_plan("mqmas", spin=1.5, split_t1=True,
                                                   convention=convention))
        np.testing.assert_allclose(np.sort(np.abs(out.data).ravel()),
                                   np.sort(np.abs(spec.data).ravel()))   # not sheared again
        y, x = out.axes[0].ppm, out.axes[1].ppm
        dppm = abs(y[1] - y[0])
        for dcs, dqis, _ in sites:
            cols = np.abs(x - (dcs + dqis)) < 30
            got = y[int(np.argmax(np.abs(out.data[:, cols]).sum(axis=1)))]
            assert got == pytest.approx(a * dcs + b * dqis, abs=1.5 * dppm)

    # the ordinary (z-filter) plan on split-t1 data shears a second time: wrong
    wrong = sh.shear_2d(spec, sh.make_shear_plan("mqmas", spin=1.5))
    proj = np.abs(wrong.data).sum(axis=1)
    right = np.abs(sh.shear_2d(spec, sh.make_shear_plan(
        "mqmas", spin=1.5, split_t1=True)).data).sum(axis=1)
    assert (proj > 0.5 * proj.max()).sum() > 3 * (right > 0.5 * right.max()).sum()
