"""notebook/simpson_fit.py: fit a measured spectrum with SIMPSON spectra."""
import numpy as np
import pytest

from opennmr.notebook import simpson_fit as sfit

X = np.linspace(-100, 100, 4001)


def line(x0, w, a=1.0):
    return a * w**2 / ((X - x0) ** 2 + w**2)


def test_recovers_amplitudes_with_no_shift_or_broadening():
    t1, t2 = line(-20, 3), line(30, 5)
    y = 2.0 * t1 + 0.5 * t2
    res = sfit.fit_simpson_templates(X, y, [{"x": X, "y": t1, "name": "A"},
                                            {"x": X, "y": t2, "name": "B"}],
                                     float_shift=False, baseline="none")
    assert [p["amplitude"] for p in res["params"]] == pytest.approx([2.0, 0.5], rel=1e-6)
    assert res["r_squared"] > 0.999999
    assert sum(p["fraction"] for p in res["params"]) == pytest.approx(1.0)


def test_recovers_a_shift_and_a_baseline():
    t = line(0, 4)
    y = 3.0 * line(7.5, 4) + 0.2 + 0.001 * X
    res = sfit.fit_simpson_templates(X, y, [{"x": X, "y": t, "name": "S"}],
                                     float_shift=True, shift_window=15, baseline="slope")
    p = res["params"][0]
    assert p["shift"] == pytest.approx(7.5, abs=1e-3)
    assert p["amplitude"] == pytest.approx(3.0, rel=1e-3)
    assert res["baseline_params"][0]["value"] == pytest.approx(0.2, abs=1e-3)


def test_extra_lorentzian_broadening_is_found():
    sfo = 100.0                                   # MHz: 1 ppm = 100 Hz
    t = line(0, 2)
    y = line(0, 2 + 300.0 / sfo / 2)              # +300 Hz FWHM = +1.5 ppm half width
    res = sfit.fit_simpson_templates(X, y, [{"x": X, "y": t, "name": "S"}], sfo1_mhz=sfo,
                                     float_shift=False, float_broadening=True,
                                     baseline="none")
    assert res["params"][0]["lb"] == pytest.approx(300.0, rel=0.1)


def test_fit_regions_ignore_the_rest():
    t = line(0, 3)
    y = 2 * t + line(60, 3, 5)                    # an unmodelled line far away
    res = sfit.fit_simpson_templates(X, y, [{"x": X, "y": t, "name": "S"}], float_shift=False,
                                     baseline="none", fit_regions=[(-20, 20)])
    assert res["params"][0]["amplitude"] == pytest.approx(2.0, rel=0.05)   # the far line's tail
    assert res["x"].min() >= -20 and res["x"].max() <= 20
    whole = sfit.fit_simpson_templates(X, y, [{"x": X, "y": t, "name": "S"}], float_shift=False,
                                       baseline="none")
    assert abs(whole["params"][0]["amplitude"] - 2.0) > abs(res["params"][0]["amplitude"] - 2.0)


def test_errors_and_checks():
    with pytest.raises(Exception):
        sfit.fit_simpson_templates(X, X, [], float_shift=False)


def test_front_end_takes_spectra(tmp_path):
    from opennmr.io import simpson
    import simpson_files as sf
    exp = simpson.read_1d(sf.write_spe(tmp_path / "e.spe"), nucleus="31P", field=9.4)
    sim = simpson.read_1d(sf.write_fid(tmp_path / "s.fid"), nucleus="31P", field=9.4)
    res = sfit.SIMPSONfit_spectrum(exp, sim, names=["sim"], float_shift=False, baseline="none")
    assert res["params"][0]["amplitude"] == pytest.approx(1.0, rel=1e-2)
    fig, ax = sfit.SIMPSONplot_fit(res)
    assert ax.get_xlabel()
