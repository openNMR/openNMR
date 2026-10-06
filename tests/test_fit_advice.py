"""notebook.fit_advice: plain-language advice on fits (explicit user request,
2026-10-06) -- each check against a synthetic spectrum whose truth is
known, so the advice is tested for being *right*, not just present."""

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import combined_fitting as cf
from opennmr.notebook import fit_advice as fa
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined
from opennmr.notebook.series_fitting import fit_series_with_lineshapes
from opennmr.processing.relaxation import fit_relaxation

NU0 = 100.6e6
X = np.linspace(-250, 450, 16384)


def _data(truth, x=X, nu0=NU0, mas=0.0, noise=1.0, n=2584, transitions="all", seed=0):
    y = simulate_combined(truth, x, nu0, mas_Hz=mas, n_zcw=n, transitions=transitions)["real_spec"]
    y = y / y.max() * 1000 + np.random.default_rng(seed).normal(0, noise, x.size)
    return {"x_ppm": x[::-1], "real_spectrum": y[::-1], "SFO1": nu0 / 1e6}


def _truth():
    return [Site("13C", 120, label="C1", fwhm=0.6).add_csa(-150, 0.4),
            Site("13C", 30, label="C2", fwhm=0.6).add_csa(60, 0.2)]


def _tells(items):
    return " | ".join(it["text"] for it in items if it["section"] == "tells")


def _nexts(items):
    return [it for it in items if it["section"] == "next"]


@pytest.fixture(scope="module")
def mas_data():
    return _data(_truth(), mas=8000)


# ---- the building blocks --------------------------------------------------------

def test_noise_is_measured_from_the_spectrum():
    x = np.linspace(0, 10, 20000)
    clean = 100 * np.exp(-((x - 5) / 0.05) ** 2)
    y = clean + np.random.default_rng(3).normal(0, 2.0, x.size)
    assert fa.estimate_noise(x, y, clean) == pytest.approx(2.0, rel=0.05)
    assert fa.estimate_noise(x, y) == pytest.approx(2.0, rel=0.1)


def test_leftover_signal_is_located():
    x = np.linspace(0, 10, 4000)
    rng = np.random.default_rng(0)
    fit = 50 * np.exp(-((x - 3) / 0.05) ** 2)
    y = fit + 10 * np.exp(-((x - 7) / 0.05) ** 2) + rng.normal(0, 0.5, x.size)
    lumps = fa.residual_lumps(x, y, fit, width_points=20)
    assert lumps[0]["centre"] == pytest.approx(7.0, abs=0.02) and lumps[0]["sign"] == 1
    assert fa.residual_lumps(x, fit + rng.normal(0, 0.5, x.size), fit, width_points=20) == []


# ---- single spectra --------------------------------------------------------------

def test_a_right_model_is_called_good_and_numbers_reliable(mas_data):
    fit = cf.fit_combined(mas_data, _truth(), mas_Hz=8000, n_zcw=987)
    items = fa.advise_combined_fit(fit)
    assert items[0]["tone"] == "good" and "within the noise" in items[0]["text"]
    assert "Well determined" in _tells(items)
    assert not _nexts(items)


def test_a_missing_peak_is_found_where_it_is_and_models_are_compared():
    truth3 = _truth() + [Site("13C", 70, label="C3", fwhm=0.6, amount=0.3)]
    data = _data(truth3, mas=8000)
    two = cf.fit_combined(data, _truth(), mas_Hz=8000, n_zcw=987)
    three = cf.fit_combined(data, truth3, mas_Hz=8000, n_zcw=987)
    items = fa.advise_combined_fit(two, alternatives=[("3 peaks", three)])
    adds = [it["action"] for it in _nexts(items) if it["action"]
            and it["action"]["type"] == "add_peak"]
    assert len(adds) == 1 and adds[0]["delta_iso"] == pytest.approx(70.0, abs=0.3)
    assert "“3 peaks” (3 peaks with CSA) describes your data clearly better" in _tells(items)
    back = fa.advise_combined_fit(three, alternatives=[("2 peaks", two)])
    assert back[0]["tone"] == "good"
    assert "This fit (3 peaks with CSA) describes your data clearly better than “2 peaks”" \
        in _tells(back)
    assert "far more than its extra parameters" in _tells(back)


def test_an_unneeded_extra_peak_loses_to_the_simpler_model():
    x = np.linspace(-5, 15, 8192)
    data = _data([Site("1H", 5, label="H", fwhm=0.1, lineshape="lorentzian")], x=x, nu0=400e6)
    one = cf.fit_combined(data, [Site("1H", 5, label="H", fwhm=0.1, lineshape="lorentzian")])
    two = cf.fit_combined(data, [Site("1H", 5, label="H", fwhm=0.1, lineshape="lorentzian"),
                                 Site("1H", 9, label="H2", fwhm=0.1, lineshape="lorentzian")])
    text = _tells(fa.compare_items(two, [("one peak", one)]))
    # The extra peak buys nothing: either a tie (prefer the simpler) or a
    # clear loss once its parameters are charged for.
    assert ("prefer the simpler one (“one peak”" in text
            or "“one peak” (1 peak, isotropic) describes your data" in text)
    # Different data are never compared.
    other = cf.fit_combined(_data([Site("1H", 5, label="H", fwhm=0.1, lineshape="lorentzian")],
                                  x=x, nu0=400e6, seed=5),
                            [Site("1H", 5, label="H", fwhm=0.1, lineshape="lorentzian")])
    assert fa.compare_items(one, [("other spectrum", other)]) == []


def test_a_wrong_mas_rate_is_diagnosed_with_its_direction(mas_data):
    fit = cf.fit_combined(mas_data, _truth(), mas_Hz=8080, n_zcw=987)
    items = fa.advise_combined_fit(fit)
    text = _tells(items)
    assert "MAS rate is probably lower than the 8080 Hz used" in text
    first_next = _nexts(items)[0]
    assert first_next["action"] == {"type": "free_mas"}
    # It is said once, first among the findings after the verdict.
    assert items[1]["text"].startswith("The spinning sidebands in your data sit")
    # Its sidebands are not mistaken for missing peaks.
    assert not any(it["action"] and it["action"]["type"] == "add_peak" for it in items)
    # The symptoms (η undetermined, trade-offs) wait until the cause is fixed.
    assert "will change once the suggestions below are tried" in text
    assert "trade off" not in text and "cannot pin down" not in text


def test_sidebands_too_weak_say_anisotropy_underestimated_and_held_means_free_it(mas_data):
    wrong = [Site("13C", 120, label="C1", fwhm=0.6).add_csa(-100, 0.4), _truth()[1]]
    fit = cf.fit_combined(mas_data, wrong, mas_Hz=8000, n_zcw=987,
                          vary={"C1": ["delta_iso", "amount", "fwhm"],
                                "C2": ["delta_iso", "amount", "fwhm", "aniso", "asym"]})
    items = fa.advise_combined_fit(fit)
    assert "the anisotropy is probably underestimated" in _tells(items)
    actions = [it["action"] for it in _nexts(items) if it["action"]]
    assert {"type": "free_parameter", "site": "C1", "name": "aniso"} in actions


def test_a_parameter_at_its_limit_offers_to_widen_it(mas_data):
    start = [Site("13C", 120, label="C1", fwhm=0.6).add_csa(-110, 0.4), _truth()[1]]
    fit = cf.fit_combined(mas_data, start, mas_Hz=8000, n_zcw=987,
                          bounds={"C1": {"aniso": (-120, -50)}})
    items = fa.advise_combined_fit(fit)
    assert "the anisotropy (Δ or span) of C1 stopped at its lower limit (-120 ppm)" \
        in _tells(items).replace("The anisotropy", "the anisotropy")
    widen = next(it["action"] for it in _nexts(items)
                 if it["action"] and it["action"]["type"] == "widen_bound")
    assert widen["site"] == "C1" and widen["name"] == "aniso" and widen["side"] == "lower"
    assert widen["value"] < -120


def test_a_quadrupolar_peak_missing_its_csa_is_told_to_add_one():
    x = np.linspace(-120, 100, 8192)
    truth = [Site("27Al", 10, label="Al", fwhm=1.5).add_quadrupolar(5.0, 0.3).add_csa(80, 0.5)]
    data = _data(truth, x=x, nu0=130.3e6, transitions="ct", n=6765)
    fit = cf.fit_combined(data, [Site("27Al", 10, label="Al", fwhm=1.5).add_quadrupolar(5.0, 0.3)],
                          transitions="ct", n_zcw=6765)
    items = fa.advise_combined_fit(fit)
    actions = [it["action"] for it in _nexts(items) if it["action"]]
    assert {"type": "add_interaction", "site": "Al", "kind": "csa"} in actions
    assert not any(a["type"] == "add_peak" and abs(a["delta_iso"] - 10) > 20 for a in actions)


def test_a_gaussian_on_a_lorentzian_line_is_told_to_try_pseudo_voigt():
    x = np.linspace(-5, 15, 8192)
    data = _data([Site("1H", 5, label="H", fwhm=0.1, lineshape="lorentzian")], x=x, nu0=400e6)
    fit = cf.fit_combined(data, [Site("1H", 5, label="H", fwhm=0.1, lineshape="gaussian")])
    items = fa.advise_combined_fit(fit)
    actions = [it["action"] for it in _nexts(items) if it["action"]]
    assert {"type": "set_lineshape", "site": "H", "lineshape": "pseudo_voigt"} in actions
    # Its wings are its shape, not missing peaks (a second peak beside it
    # is offered only as the alternative).
    assert not any(a["type"] == "add_peak" and abs(a["delta_iso"] - 5) > 0.5 for a in actions)
    assert "no peak in the model accounts for" not in _tells(items)


def test_an_unneeded_peak_is_flagged_for_removal():
    x = np.linspace(-5, 15, 8192)
    data = _data([Site("1H", 5, label="H", fwhm=0.1, lineshape="lorentzian")], x=x, nu0=400e6)
    fit = cf.fit_combined(data, [Site("1H", 5, label="H", fwhm=0.1, lineshape="lorentzian"),
                                 Site("1H", 9, label="H2", fwhm=0.1, lineshape="lorentzian")])
    items = fa.advise_combined_fit(fit)
    assert "H2 makes up only" in _tells(items)
    assert {"type": "remove_site", "site": "H2"} in [it["action"] for it in _nexts(items)]


def test_advice_text_is_plain():
    items = [fa._item("tells", "good", "Fine."), fa._item("next", "note", "Do this.")]
    assert fa.advice_text(items) == ("What the fit tells you:\n  ✓ Fine.\nWhat to try next:\n"
                                     "  → Do this.")


def test_the_report_carries_the_advice(mas_data, capsys):
    fit = cf.fit_combined(mas_data, _truth(), mas_Hz=8080, n_zcw=987)
    text = cf.print_combined_fit_report(fit)
    assert "What the fit tells you:" in text and "Let the MAS rate vary" in text
    assert "What the fit tells you" not in cf.print_combined_fit_report(fit, advice=False)


# ---- pseudo-2D series and relaxation curves ------------------------------------------

def _series(ta, tb):
    x = np.linspace(0, 10, 2000)
    g = lambda c: np.exp(-4 * np.log(2) * (x - c) ** 2 / 0.3 ** 2)  # noqa: E731
    t = np.geomspace(0.01, 3, 8)
    rng = np.random.default_rng(0)
    data = np.array([g(5) * np.exp(-ti / ta) + 0.8 * g(5.4) * np.exp(-ti / tb)
                     + rng.normal(0, 0.002, x.size) for ti in t])
    spectrum = NMRSpectrum(data=data, axes=[NMRAxis(kind="index", index=np.arange(1, 9)),
                                            NMRAxis(kind="frequency", ppm=x)],
                           metadata=NMRMetadata(vendor="x", nucleus="1H", sfo1=400,
                                                pseudo_2d=True))
    return fit_series_with_lineshapes(spectrum, x, [g(5) + 0.8 * g(5.4)], labels=["Site 1"])


def test_series_with_one_shape_is_good_and_a_changing_shape_is_caught():
    good = fa.advise_series(_series(0.5, 0.5))
    assert good[0]["tone"] == "good" and "only the intensities change" in good[0]["text"]
    bad = fa.advise_series(_series(0.2, 2.0))
    assert bad[0]["tone"] == "warn" and "The shape changes along the series" in bad[0]["text"]
    assert "slices 1–8" in bad[0]["text"]
    assert any("more peaks" in it["text"] for it in _nexts(bad))


def test_relaxation_delays_that_stop_short_are_called_out():
    rng = np.random.default_rng(1)
    t = np.geomspace(0.01, 3, 10)
    y = 1 - 2 * np.exp(-t / 2.0) + rng.normal(0, 0.01, t.size)
    items = fa.advise_relaxation(fit_relaxation(t, y, model="inversion_recovery"), "Peak 1")
    text = _tells(items)
    assert "Your longest delay (3 s) is only 1.5× T1" in text
    assert "assumes the curve carries on" in text          # not called well determined
    assert any("up to about 5× T1" in it["text"] for it in _nexts(items))


def test_relaxation_good_and_systematic_misfits():
    rng = np.random.default_rng(1)
    t = np.geomspace(0.02, 12, 12)
    good = fa.advise_relaxation(fit_relaxation(t, 1 - 2 * np.exp(-t / 2.0)
                                               + rng.normal(0, 0.01, t.size),
                                               model="inversion_recovery"))
    assert all(it["tone"] == "good" for it in good) and "well determined" in _tells(good)
    t = np.geomspace(0.01, 20, 16)
    stretched = np.exp(-(t / 1.0) ** 0.5) + rng.normal(0, 0.003, t.size)
    items = fa.advise_relaxation(fit_relaxation(t, stretched, model="exponential_decay"))
    assert "systematic way" in _tells(items)
    assert {"type": "set_relaxation_model", "model": "stretched_decay"} in \
        [it["action"] for it in _nexts(items)]
