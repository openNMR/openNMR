

# ====================================================================== stage 3
# time-domain engine: dynamics under MAS and echo sequences

LARMOR_13C = 100.6e6
PPM_13C = np.linspace(250, -150, 2048)
PPM_2H_FINE = np.linspace(-150e3, 150e3, 4096) / LARMOR_2H * 1e6     # +-150 kHz, 73 Hz steps


def _csa_pair():
    a = Site("13C", 150.0, label="A", fwhm=2.0, lineshape="lorentzian").add_csa(80.0, 0.4, "Haeberlen")
    b = Site("13C", 50.0, label="B", fwhm=2.0, lineshape="lorentzian").add_csa(-60.0, 0.2, "Haeberlen")
    return M.MotionModel([a, b], hops=[{"from": 0, "to": 1, "k_Hz": 1.0}])


def _cone_2h():
    d = Site("2H", 0.0, label="D", fwhm=3.0, lineshape="lorentzian").add_quadrupolar(0.17, 0.0)
    return M.cone_jump(d, 40.0, 3, k_Hz=1.0)


def test_batched_expm_matches_scipy_including_stiff_matrices():
    from scipy.linalg import expm
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 4, 4)) + 1j * rng.normal(size=(6, 4, 4))
    assert np.allclose(M._batched_expm(A), [expm(a) for a in A], atol=1e-10)
    K = np.array([[-1e4, 1e4], [1e4, -1e4]], dtype=complex)
    stiff = (K + np.diag([2j * np.pi * 3e3, -2j * np.pi * 3e3])) * 1e-3
    assert np.allclose(M._batched_expm(stiff[None])[0], expm(stiff), atol=1e-9)


def test_echo_arguments_are_validated():
    assert M._parse_echo(None) is None
    assert M._parse_echo(("hahn", 1e-4)) == {"kind": "hahn", "tau_s": 1e-4}
    assert M._parse_echo({"kind": "solid", "tau_us": 50})["tau_s"] == pytest.approx(5e-5)
    for bad in ("hahn", {"kind": "xx", "tau_s": 1e-4}, {"kind": "hahn"}, ("hahn", -1.0)):
        with pytest.raises(ValueError):
            M._parse_echo(bad)
    with pytest.raises(ValueError):                                  # solid echo needs I > 1/2
        M.motion_spectrum(M.MotionModel([Site("13C", 0.0).add_csa(50.0, 0.0)]),
                          np.linspace(100, -100, 512), larmor_Hz=1e8, echo=("solid", 1e-4))
    with pytest.raises(ValueError):
        M.motion_spectrum(_csa_pair(), PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=-1.0)


@pytest.mark.parametrize("kind", ["hahn", "solid"])
def test_static_echo_without_motion_is_the_static_spectrum_with_t2_loss(kind):
    model, tau = _cone_2h(), 30e-6
    static = M.motion_spectrum(model, PPM_2H_FINE, larmor_Hz=LARMOR_2H)["real_spec"]
    r = M.motion_spectrum(model, PPM_2H_FINE, larmor_Hz=LARMOR_2H, echo={"kind": kind, "tau_s": tau})
    assert _cdf_gap(r["real_spec"], static) < 2e-3
    expect = math.exp(-math.pi * 3.0 * LARMOR_2H * 1e-6 * 2.0 * tau)     # R2 = pi * FWHM, over 2 tau
    assert r["td"]["echo"]["intensity"] == pytest.approx(expect, rel=2e-3)
    assert r["td"]["echo"]["tau_s"] == pytest.approx(tau)
    assert "echo amplitude" in r["log"]


def test_echo_intensity_is_lost_at_intermediate_rates_and_recovers_when_fast():
    model, tau = _cone_2h(), 30e-6
    inten = [M.motion_spectrum(model, PPM_2H_FINE, larmor_Hz=LARMOR_2H, k_modes={0: k},
                               echo=("solid", tau))["td"]["echo"]["intensity"]
             for k in (1e2, 1e5, 1e8)]
    assert inten[1] < 0.2 * inten[0]
    assert inten[2] > 5 * inten[1]


def test_hahn_echo_refocuses_a_csa_and_an_offset_but_the_solid_echo_does_not():
    site = Site("2H", 20.0, label="D", fwhm=3.0, lineshape="lorentzian").add_quadrupolar(
        0.17, 0.0).add_csa(100.0, 0.0, "Haeberlen")
    model, tau = M.MotionModel([site]), 30e-6
    hahn = M.motion_spectrum(model, PPM_2H_FINE, larmor_Hz=LARMOR_2H, echo=("hahn", tau))
    assert hahn["td"]["echo"]["intensity"] == pytest.approx(
        math.exp(-math.pi * 3.0 * LARMOR_2H * 1e-6 * 2.0 * tau), rel=2e-3)
    solid = M.motion_spectrum(model, PPM_2H_FINE, larmor_Hz=LARMOR_2H, echo=("solid", tau))
    assert "does not refocus shifts" in solid["log"]
    assert _cdf_gap(hahn["real_spec"], solid["real_spec"]) > 1e-3


def test_dynamic_mas_slow_limit_is_the_sum_of_the_single_state_mas_spectra():
    model = _csa_pair()
    lim = M.limit_spectra(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    r = M.motion_spectrum(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    assert _cdf_gap(r["real_spec"], lim["static"]) < 3e-3
    assert r["real_spec"].sum() == pytest.approx(2.0, rel=0.03)
    assert r["td"]["tail"] < 5e-3 and "time domain, MAS 5 kHz" in r["log"]


def test_dynamic_mas_reaches_the_fast_limit_and_not_before():
    model = _csa_pair()
    lim = M.limit_spectra(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    gaps = [_cdf_gap(M.motion_spectrum(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0,
                                       hop_scale=hs)["real_spec"], lim["fast"])
            for hs in (1e5, 1e7, 1e9)]
    assert gaps[0] > 0.1 > 0.03 > gaps[1] > gaps[2]
    assert gaps[2] < 5e-3


def test_isotropic_exchange_is_the_same_static_and_under_mas():
    a = Site("13C", 30.0, label="A", fwhm=1.0, lineshape="lorentzian")
    b = Site("13C", 0.0, label="B", fwhm=1.0, lineshape="lorentzian")
    model = M.MotionModel([a, b], hops=[{"from": 0, "to": 1, "k_Hz": 1500.0}])
    x = np.linspace(60, -30, 1024)
    s = M.motion_spectrum(model, x, larmor_Hz=LARMOR_13C)["real_spec"]
    m = M.motion_spectrum(model, x, larmor_Hz=LARMOR_13C, mas_Hz=8000.0)["real_spec"]
    assert _cdf_gap(s, m) < 1e-3
    assert np.max(np.abs(s - m)) < 0.01 * s.max()


def test_mas_hahn_echo_is_the_slow_spectrum_when_rotor_synchronised():
    model = _csa_pair()
    lim = M.limit_spectra(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    r = M.motion_spectrum(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0,
                          echo={"kind": "hahn", "tau_s": 2.0 / 5000.0})
    assert _cdf_gap(r["real_spec"], lim["static"]) < 3e-3
    expect = math.exp(-math.pi * 2.0 * LARMOR_13C * 1e-6 * 2.0 * (2.0 / 5000.0))
    assert r["td"]["echo"]["intensity"] == pytest.approx(expect, rel=5e-3)
    assert "rotor-synchronised" not in r["log"]
    off = M.motion_spectrum(model, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0,
                            echo=("hahn", 1.5 / 5000.0))
    assert "rotor-synchronised" in off["log"]


def test_single_state_under_mas_keeps_the_exact_engine_unless_an_echo_is_asked():
    one = M.MotionModel([_csa_pair().sites[0]])
    assert "td" not in M.motion_spectrum(one, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    assert M.motion_spectrum(one, PPM_13C, larmor_Hz=LARMOR_13C, mas_Hz=5000.0,
                             echo=("hahn", 2.0 / 5000.0))["td"] is not None


def test_time_domain_second_order_central_transition_matches_simulate_under_mas():
    site = Site("23Na", 0.0, label="Na", fwhm=1.0, lineshape="lorentzian").add_quadrupolar(2.0, 0.3)
    ref = simulate_combined(site, PPM_23NA, larmor_Hz=LARMOR_23NA, order="second",
                            transitions="ct", mas_Hz=15000.0, n_zcw=10946)["real_spec"]
    r = M.motion_spectrum(M.MotionModel([site]), PPM_23NA, larmor_Hz=LARMOR_23NA, mas_Hz=15000.0,
                          transitions="ct", order="second", echo=("hahn", 2.0 / 15000.0))
    assert _cdf_gap(r["real_spec"], ref) < 5e-3


def test_rate_series_and_notebook_code_carry_mas_and_echo():
    model = _csa_pair()
    ser = M.motion_rate_series(model, [1e3, 1e5], which="hops", ppm_axis=PPM_13C,
                               larmor_Hz=LARMOR_13C, mas_Hz=5000.0)
    assert len(ser["spectra"]) == 2 and ser["results"][0]["settings"]["mas_Hz"] == 5000.0
    assert {"static", "fast", "fast_tensor"} <= set(ser["limits"])
    code = M.motion_notebook_code(model, rates=[1e3], which="hops", larmor_MHz=100.6,
                                  mas_Hz=5000.0, echo={"kind": "hahn", "tau_s": 2e-4})
    compile(code, "notebook_code", "exec")
    assert "mas_Hz=5000" in code and "'hahn'" in code
