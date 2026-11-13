"""The aMAT spectral filter (explicit user request, 2026-11-13; Angew. Chem.
Int. Ed. 2024, e202408704, SI section S7): each F1 row of an aMAT 2D is
multiplied by a train of pulses at its own isotropic position + n ν_rot,
which removes the artefacts of intense signals that hide weak ones.

The synthetic spectrum: a strong site at 0 ppm and a weak one at −37 ppm
(the SI's para-proton), 1H at 500 MHz and 111 kHz MAS, with the strong
site's sideband manifold smeared along F1 (the artefact the SI shows)."""

import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.notebook import amat_filter as af
from test_gui import qapp  # noqa: F401

SF = 500.0
NU_ROT = 111e3
ROT_PPM = NU_ROT / SF


def _lor(x, c, w):
    return (w / 2) ** 2 / ((x - c) ** 2 + (w / 2) ** 2)


def _amat(artefact=2.0, imag=False, f1_shift=0.0):
    f2 = np.linspace(700, -700, 4096)
    f1 = np.linspace(150, -150, 257)
    data = np.zeros((f1.size, f2.size))
    for iso, amp in ((0.0, 100.0), (-37.0, 1.0)):
        row = _lor(f1[:, None], iso + f1_shift, 2.0)
        for n in range(-3, 4):
            data += amp * np.exp(-abs(n)) * row * _lor(f2[None, :], iso + n * ROT_PPM, 1.0)
    ghost = sum(np.exp(-abs(n)) * _lor(f2, n * ROT_PPM, 1.0) for n in range(-3, 4))
    data += artefact * ghost[None, :]
    meta = NMRMetadata(vendor="bruker", nucleus="1H", nucleus_indirect="1H", sfo1=SF,
                       homonuclear=True, raw={"acqus": {"MASR": NU_ROT}})
    return NMRSpectrum(data=data, axes=[NMRAxis("frequency", ppm=f1, hz=f1 * SF),
                                        NMRAxis("frequency", ppm=f2, hz=f2 * SF)],
                       metadata=meta, data_imag=0.5 * data if imag else None)


def _at(x, value):
    return int(np.argmin(np.abs(x - value)))


# ---- the pulse train ------------------------------------------------------------------

def test_pulse_train_is_periodic_and_centred():
    nu = np.linspace(-500, 500, 10001)
    f = af.pulse_train(nu, 30.0, 100.0, 30.0)
    for n in range(-5, 5):
        assert f[_at(nu, 30.0 + 100 * n)] == pytest.approx(1.0)
        assert f[_at(nu, 30.0 + 100 * n + 50)] == 0.0           # between pulses
    # Hann: half height at a quarter of the width from the centre
    assert f[_at(nu, 30.0 + 7.5)] == pytest.approx(0.5, abs=1e-3)
    assert f[_at(nu, 30.0 + 15.1)] == 0.0


def test_flat_pulses_are_flat():
    nu = np.linspace(-100, 100, 2001)
    f = af.pulse_train(nu, 0.0, 100.0, 30.0, shape="rect")
    assert set(np.unique(f)) == {0.0, 1.0}
    assert np.sum(f) * 0.1 == pytest.approx(2 * 30.0, rel=0.02)   # two pulses in 200


def test_pulse_train_refuses_impossible_widths():
    with pytest.raises(ValueError, match="no wider"):
        af.pulse_train([0.0], 0.0, 100.0, 150.0)
    with pytest.raises(ValueError, match="positive"):
        af.pulse_train([0.0], 0.0, 100.0, 0.0)
    with pytest.raises(ValueError, match="shape"):
        af.pulse_train([0.0], 0.0, 100.0, 10.0, shape="gauss")


def test_the_filter_ridge_lies_on_the_diagonal():
    f1 = np.linspace(100, -100, 41)
    f2 = np.linspace(300, -300, 1201)
    F = af.amat_filter_map(f1, f2, nu_rot_hz=NU_ROT, sf_mhz=SF)
    for i, v in enumerate(f1):
        assert F[i, _at(f2, v)] == pytest.approx(1.0, abs=0.01)
        for n in (-1, 1):                     # the sideband pulses inside the F2 range
            if abs(v + n * ROT_PPM) < 300:
                assert F[i, _at(f2, v + n * ROT_PPM)] == pytest.approx(1.0, abs=0.01)
            if abs(v + n * ROT_PPM / 2) < 300:
                assert F[i, _at(f2, v + n * ROT_PPM / 2)] == 0.0


# ---- the filter on a spectrum -----------------------------------------------------------

def test_weak_signal_is_freed_from_the_strong_ones_artefacts():
    spec = _amat()
    out = af.apply_amat_filter(spec)                      # rate from MASR
    row = _at(spec.axes[0].ppm, -37.0)
    f2 = spec.axes[1].ppm
    before, after = spec.data[row], out.data[row]
    # the strong site's ghost at 0 ppm (and its sidebands) is gone ...
    for n, ghost in ((-1, 0.5), (0, 1.5), (1, 0.5)):
        assert before[_at(f2, n * ROT_PPM)] > ghost
        assert after[_at(f2, n * ROT_PPM)] < 1e-6
    # ... the weak site's own centreband and sidebands are kept
    for n in (-1, 0, 1):
        k = _at(f2, -37.0 + n * ROT_PPM)
        assert after[k] == pytest.approx(before[k], rel=2e-3)     # row 0.5 ppm off its centre


def test_strong_site_is_untouched_on_its_own_row():
    spec = _amat()
    out = af.apply_amat_filter(spec, NU_ROT)
    row = _at(spec.axes[0].ppm, 0.0)
    f2 = spec.axes[1].ppm
    for n in range(-3, 4):
        k = _at(f2, n * ROT_PPM)
        assert out.data[row, k] == pytest.approx(spec.data[row, k], rel=1e-3)


def test_imaginary_part_is_filtered_alike_and_input_kept():
    spec = _amat(imag=True)
    copy = spec.data.copy()
    out = af.apply_amat_filter(spec, NU_ROT)
    assert np.allclose(out.data_imag, 0.5 * out.data)
    assert np.array_equal(spec.data, copy)
    assert "amat_filter" not in spec.metadata.raw


def test_log_records_what_was_done():
    out = af.apply_amat_filter(_amat(), NU_ROT, source_label="expno 115")
    rec = out.metadata.raw["amat_filter"]
    assert rec["width_hz"] == pytest.approx(NU_ROT / 3)
    assert rec["shape"] == "hann" and rec["nu_rot_hz"] == NU_ROT
    assert "expno 115" in rec["log"] and "S7" in rec["log"]
    assert 0 < rec["removed_fraction"] < 1


def test_width_in_hz_or_fraction():
    a = af.apply_amat_filter(_amat(), NU_ROT, width_fraction=0.25)
    b = af.apply_amat_filter(_amat(), NU_ROT, width_hz=0.25 * NU_ROT)
    assert np.allclose(a.data, b.data)


def test_too_narrow_for_the_homogeneous_linewidth_is_reported():
    # T2' = 20 µs: 1/(πT2') ≈ 15.9 kHz > half of 20 kHz
    out = af.apply_amat_filter(_amat(), NU_ROT, width_hz=20e3, t2_prime_s=20e-6)
    warnings = out.metadata.raw["amat_filter"]["warnings"]
    assert any("homogeneous linewidth" in w for w in warnings)
    ok = af.apply_amat_filter(_amat(), NU_ROT, t2_prime_s=1e-3)
    assert not ok.metadata.raw["amat_filter"]["warnings"]


def test_f1_offset_follows_a_misreferenced_f1():
    spec = _amat(f1_shift=5.0)        # F1 reads 5 ppm high
    plain = af.apply_amat_filter(spec, NU_ROT, width_fraction=0.03, shape="rect")
    fixed = af.apply_amat_filter(spec, NU_ROT, width_fraction=0.03, shape="rect",
                                 f1_offset_ppm=-5.0)
    row = _at(spec.axes[0].ppm, -37.0 + 5.0)
    k = _at(spec.axes[1].ppm, -37.0)
    assert plain.data[row, k] < 0.5 * spec.data[row, k]     # the weak site is cut
    assert fixed.data[row, k] == pytest.approx(spec.data[row, k], rel=1e-6)


def test_needs_a_rate_and_a_homonuclear_2d():
    spec = _amat()
    spec.metadata.raw = {}
    with pytest.raises(ValueError, match="MAS rate"):
        af.apply_amat_filter(spec)
    spec.metadata.nucleus_indirect = "13C"
    spec.metadata.homonuclear = False
    with pytest.raises(ValueError, match="homonuclear"):
        af.apply_amat_filter(spec, NU_ROT)


def test_a_1d_or_pseudo_2d_is_refused():
    spec = _amat()
    one = NMRSpectrum(data=spec.data[0], axes=[spec.axes[1]], metadata=spec.metadata)
    with pytest.raises(ValueError, match="2D"):
        af.apply_amat_filter(one, NU_ROT)
    spec.axes[0] = NMRAxis("index", index=np.arange(1, spec.data.shape[0] + 1))
    with pytest.raises(ValueError, match="pseudo-2D"):
        af.apply_amat_filter(spec, NU_ROT)


def test_the_nmr2dimport_dict_works_too():
    spec = _amat()
    d = {"spectrum": spec.data, "x_ppm": spec.axes[1].ppm, "y_ppm": spec.axes[0].ppm,
         "SF1": SF, "NUC1": "1H", "NUC2": "1H", "pseudo_2d": False, "homonuclear": False}
    out = af.apply_amat_filter(d, NU_ROT)
    assert np.allclose(out["spectrum"], af.apply_amat_filter(spec, NU_ROT).data)
    assert "log" in out["amat_filter"]


def test_trace_and_pictures():
    import matplotlib
    matplotlib.use("Agg")
    spec = _amat()
    out = af.apply_amat_filter(spec, NU_ROT)
    tr = af.amat_trace(spec, out, -37.0)
    assert tr["f1_ppm"] == pytest.approx(-37.0, abs=1.0)
    assert tr["train"].max() == pytest.approx(1.0, abs=0.01)
    assert np.allclose(tr["after"], tr["before"] * tr["train"])
    fig = af.plot_amat_trace(spec, out, -37.0)
    assert fig.axes[0].get_xlim()[0] > fig.axes[0].get_xlim()[1]       # ppm decreasing
    assert "$" in fig.axes[0].get_xlabel()                                # italic δ
    fig2 = af.plot_amat_filter_map(spec, out)
    assert len(fig2.axes[0].lines) == 1


def test_notebook_exports():
    import opennmr.notebook as nmr
    for name in ("amat_filter", "apply_amat_filter", "plot_amat_trace", "plot_amat_filter_map"):
        assert name in nmr.__all__ and hasattr(nmr, name)


# ---- GUI ----------------------------------------------------------------------------------

def test_gui_panel_filters_the_selected_2d(qapp):
    from opennmr.gui.amat_filter_panel import AmatFilterPanel
    panel = AmatFilterPanel()
    assert not panel.apply_button.isEnabled()
    spec = _amat()
    panel.set_source(spec, "aMAT 111 kHz")
    assert panel.apply_button.isEnabled()
    assert panel.mas_spin.value() == pytest.approx(111.0)          # from MASR
    got = []
    panel.result_ready.connect(lambda s, label, log: got.append((s, label, log)))
    result = panel.apply()
    assert result is not None and got and got[0][1] == "aMAT-filtered aMAT 111 kHz"
    assert "S7" in got[0][2]
    assert np.allclose(result.data, af.apply_amat_filter(
        spec, NU_ROT, width_fraction=panel.width_spin.value()).data)
    assert panel.width_spin.value() == pytest.approx(1 / 3, abs=1e-4)      # the SI's ν_rot/3
    assert panel.show_filter() and panel.show_trace()


def test_gui_width_modes_and_t2_check(qapp):
    from opennmr.gui.amat_filter_panel import AmatFilterPanel
    panel = AmatFilterPanel()
    panel.set_source(_amat(), "x")
    panel.width_mode_combo.setCurrentIndex(panel.width_mode_combo.findData("khz"))
    assert panel.width_spin.value() == pytest.approx(37.0, abs=0.01)      # ν_rot/3 in kHz
    panel.t2_spin.setValue(0.02)                                          # 20 µs
    panel.width_spin.setValue(20.0)
    args = panel.arguments()
    assert args["width_hz"] == pytest.approx(20e3) and args["t2_prime_s"] == pytest.approx(2e-5)
    result = panel.run()
    assert result.metadata.raw["amat_filter"]["warnings"]


def test_gui_python_code_reproduces_the_filter(qapp, monkeypatch):
    from opennmr.gui.amat_filter_panel import AmatFilterPanel
    import opennmr.io as oio
    spec = _amat()
    spec.metadata.source_path = "/data/amat/115"
    panel = AmatFilterPanel()
    panel.set_source(spec, "x")
    monkeypatch.setattr(oio, "load", lambda path, *a, **k: spec)
    scope = {}
    exec(panel.python_code().replace("print(", "(lambda *a, **k: None)("), scope)
    assert np.allclose(scope["filtered"].data, panel.run().data)


def test_gui_refuses_a_1d(qapp):
    from opennmr.gui.amat_filter_panel import AmatFilterPanel
    spec = _amat()
    one = NMRSpectrum(data=spec.data[0], axes=[spec.axes[1]], metadata=spec.metadata)
    panel = AmatFilterPanel()
    panel.set_source(one, "1D")
    assert not panel.apply_button.isEnabled() and "1D" in panel.source_label.text()


def test_main_window_has_the_section_and_adds_the_result(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    assert window.amat_filter_section.title() == "aMAT Spectral Filter (2D)"
    before = len(window.session)
    spec = af.apply_amat_filter(_amat(), NU_ROT)
    index = window._add_amat_filter_result(spec, "aMAT-filtered x", "log text")
    assert len(window.session) == before + 1 and index >= 0
