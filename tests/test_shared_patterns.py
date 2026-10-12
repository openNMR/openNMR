"""Many sites with the same interactions (explicit user request,
2026-10-12: 1200 enumerated pNMR sites, one CSA): copy_interactions, and
simulate_combined simulating each shared set once and shifting it to each
site's δiso (notebook/combined_hamiltonian.py)."""
import time

import numpy as np
import pytest

from opennmr.notebook import combined_fitting as cfit
from opennmr.notebook import combined_hamiltonian as ch
from opennmr.notebook.combined_hamiltonian import Site, simulate_combined

AX = np.linspace(-1500.0, 1500.0, 8192)
H = AX[1] - AX[0]
INNER = np.abs(AX) < 1000


def _site(kind, d, amount=1.0, fwhm=5.0):
    s = Site("23Na" if kind == "quad" else "7Li", delta_iso=d, fwhm=fwhm, amount=amount,
             label=f"s{d:.5f}")
    if kind == "csa":
        s.add_csa(aniso=500.0, asym=0.3)
    elif kind == "j":
        s.add_csa(aniso=300.0, asym=0.5).add_jcoupling("31P", n=2, J_Hz=400.0)
    else:
        s.add_quadrupolar(C_Q=1.0, eta_Q=0.2).add_csa(aniso=200.0, asym=0.3,
                                                       euler=(10.0, 20.0, 30.0))
    return s


@pytest.mark.parametrize("mas", [0.0, 20000.0])
@pytest.mark.parametrize("kind", ["csa", "quad", "j"])
def test_shared_equals_separate_for_whole_point_shifts(kind, mas):
    """Shifts of whole points: the shared simulation is the separate one,
    to rounding."""
    sites = [_site(kind, 100 + H * k, a) for k, a in zip((0, 3, -7, 40), (1, 0.5, 2, 0.7))]
    shared = simulate_combined(sites, AX, B0_T=9.4, mas_Hz=mas, transitions="ct")
    alone = simulate_combined(sites, AX, B0_T=9.4, mas_Hz=mas, transitions="ct",
                              share_patterns=False)
    assert np.allclose(shared["real_spec"], alone["real_spec"], rtol=0,
                       atol=1e-10 * alone["real_spec"].max())
    for label in alone["sites"]:
        assert np.allclose(shared["sites"][label], alone["sites"][label], rtol=0,
                           atol=1e-10 * alone["real_spec"].max())
    assert "simulated once" in shared["log"]
    assert "simulated once" not in alone["log"]


@pytest.mark.parametrize("mas", [0.0, 20000.0])
def test_shared_matches_separate_for_any_shift_within_grid_error(mas):
    sites = [_site("csa", d, a) for d, a in zip((100, 103.3, 92.29, 141.2), (1, 0.5, 2, 0.7))]
    shared = simulate_combined(sites, AX, B0_T=9.4, mas_Hz=mas)["real_spec"]
    alone = simulate_combined(sites, AX, B0_T=9.4, mas_Hz=mas, share_patterns=False)["real_spec"]
    assert np.max(np.abs(shared - alone)[INNER]) < 2e-3 * alone.max()
    # Both are the same distance from a simulation on a grid 8x finer: the
    # sharing adds less than the grid already costs.
    fine = np.linspace(AX[0], AX[-1], (AX.size - 1) * 8 + 1)
    ref = simulate_combined(sites, fine, B0_T=9.4, mas_Hz=mas, share_patterns=False)
    ref = ref["real_spec"][::8]
    norm = [v / v.sum() for v in (shared, alone, ref)]
    err_shared = np.max(np.abs(norm[0] - norm[2]))
    err_alone = np.max(np.abs(norm[1] - norm[2]))
    assert err_shared < 1.5 * err_alone + 1e-12


def test_what_is_and_is_not_shared():
    a, b = _site("csa", 10.0), _site("csa", 50.0)
    c = _site("csa", 30.0, fwhm=9.0)                      # other broadening
    d = _site("j", 0.0)                                   # other interactions
    e = _site("csa", 70.0, fwhm=0.1)                      # too sharp to move off-grid
    groups = ch.shared_pattern_groups([a, b, c, d, e], "7Li", min_fwhm=2 * abs(H))
    assert [0, 1] in groups and [2] in groups and [3] in groups and [4] in groups
    q1, q2 = _site("quad", 0.0), _site("quad", 20.0)
    assert ch.shared_pattern_groups([q1, q2], "23Na", order="second") == [[0, 1]]
    assert ch.shared_pattern_groups([q1, q2], "23Na", order="exact") == [[0], [1]]
    other = Site("6Li", delta_iso=5.0).add_csa(aniso=500.0, asym=0.3)
    assert ch.shape_key(other, "7Li") is None              # another nucleus: placed by ratios


def test_many_sites_are_fast_and_summarised():
    rng = np.random.default_rng(0)
    sites = []
    for i, (d, amount) in enumerate(zip(rng.normal(300, 150, 400), rng.uniform(0.2, 1, 400))):
        s = Site("7Li", delta_iso=float(d), amount=float(amount), fwhm=8.0, label=f"Li {i}")
        s.add_csa(aniso=-600.0, asym=0.2)
        sites.append(s)
    t = time.perf_counter()
    result = simulate_combined(sites, AX, B0_T=9.4, mas_Hz=30000.0)
    assert time.perf_counter() - t < 20          # ~0.5 s; 400 separate simulations take minutes
    assert len(result["sites"]) == 400
    assert np.isclose(np.trapezoid(result["real_spec"], dx=abs(H)),
                      sum(np.trapezoid(v, dx=abs(H)) for v in result["sites"].values()))
    log = result["log"]
    assert "400 sites (7Li)" in log and "400 site(s): CSA Δ -600 ppm" in log
    assert len(log.splitlines()) < 10
    assert len(result["site_dicts"]) == 400
    lo, hi = ch.combined_axis_limits(sites, result["settings"]["larmor_Hz"], mas_Hz=30000.0)
    assert lo < min(s.delta_iso for s in sites) and hi > max(s.delta_iso for s in sites)


def test_plot_draws_the_total_only_for_many_sites():
    import matplotlib
    matplotlib.use("Agg")
    sites = [_site("csa", 10.0 * k) for k in range(20)]
    result = simulate_combined(sites, AX, B0_T=9.4)
    ax = ch.plot_combined(result, nuclide="7Li")
    assert len(ax.get_lines()) == 1
    assert any("20 sites" in t.get_text() for t in ax.texts)
    few = simulate_combined(sites[:3], AX, B0_T=9.4)
    assert len(ch.plot_combined(few, nuclide="7Li").get_lines()) == 4


def test_starting_model_shares_too():
    sites = [_site("csa", d) for d in (100.0, 160.0, 220.0)]
    sim = simulate_combined(sites, AX, B0_T=9.4)
    results = {"x_ppm": AX, "real_spectrum": sim["real_spec"],
               "SFO1": sim["settings"]["larmor_Hz"] / 1e6, "nuclide": "7Li"}
    guess = cfit.starting_model(results, sites, n_zcw=75025)
    # The amounts recovered by least squares are the true ones (1 each).
    amounts = np.array(list(guess["amounts"].values()))
    assert np.allclose(amounts / amounts.mean(), 1.0, atol=0.02)


def test_copy_interactions():
    sites = [Site("7Li", delta_iso=d, label=f"L{k}", fwhm=3.0) for k, d in
             enumerate((0.0, 10.0, 20.0))]
    sites.append(Site("1H", delta_iso=4.0, label="H"))
    sites[0].add_quadrupolar(C_Q=0.2, eta_Q=0.1).add_csa(aniso=-450.0, asym=0.3)
    sites[0].fwhm = 7.0
    out, report = ch.copy_interactions(sites, 0)
    assert report["copied"] == ["L1", "L2", "H"]
    assert report["no_quadrupolar"] == ["H"]                       # spin-1/2
    assert out[1].get("quadrupolar") and out[1].get("csa")
    assert out[1].delta_iso == 10.0 and out[1].fwhm == 3.0         # its own values kept
    assert not out[3].get("quadrupolar") and out[3].get("csa")
    assert "quadrupolar coupling, CSA copied to 3 site(s)" in report["text"]
    out, report = ch.copy_interactions(sites, "L0", targets=["L2"], broadening=True)
    assert report["copied"] == ["L2"] and out[2].fwhm == 7.0 and out[1].interactions == []
    # Dicts in, dicts out; a linked site is skipped.
    dicts = [s.to_dict() for s in sites]
    dicts[2]["link"] = {"to": "L1", "free": []}
    out, report = ch.copy_interactions(dicts, 0)
    assert isinstance(out[0], dict) and report["linked"] == ["L2"]
    with pytest.raises(ValueError):
        ch.copy_interactions(sites, "nope")


def test_exported_by_the_notebook_import():
    import opennmr.notebook as nmr
    assert nmr.copy_interactions is ch.copy_interactions
