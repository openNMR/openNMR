"""Apply ▸ Signal-to-Noise (gui/snr_panel.py over notebook/snr.py):
tick spectra, type or drag the signal and noise regions, the same regions
for all by default or some spectra's own -- logged and marked † -- then
the S/N with its uncertainty, kept with each spectrum (explicit user
request, 2026-10-12)."""
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.gui.snr_panel import parse_window, parse_windows
from opennmr.notebook import snr

X = np.linspace(100.0, -100.0, 4096)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _dialogs(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else "")))
    return shown


@pytest.fixture
def window(qapp):
    from opennmr.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    return w


def _spectrum(height, seed, ns=16):
    rng = np.random.default_rng(seed)
    y = height * np.exp(-0.5 * (X / 0.6) ** 2) + rng.normal(size=X.size)
    return NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=X.copy())],
                       metadata=NMRMetadata(vendor="test", nucleus="7Li", ns=ns))


def _load(window, heights=(40.0, 80.0), ns=(16, 64)):
    for k, (h, n) in enumerate(zip(heights, ns)):
        window._add_derived_spectrum(_spectrum(h, k, n), f"snr{k}")
    window._show_side_tab("Integrate")
    window._snr_section.setExpanded(True)
    p = window.snr_panel
    p.refresh()
    p.set_checked([e["label"] for e in p._rows if e["label"].startswith("snr")])
    return p


def test_section_sits_in_integrate_after_dnp(window):
    # Apply until 2026-10-16; the Integrate tile since.
    from opennmr.gui.widgets.collapsible_section import CollapsibleSection
    side = [window.side_tabs.tabText(i) for i in range(window.side_tabs.count())]
    tab = window.side_tabs.widget(side.index("Integrate"))
    titles = [s.title() for s in tab.findChildren(CollapsibleSection)]
    # CP Build-up (2026-10-16), then the Offset Tracker (2026-10-17), follow it.
    assert titles.index("DNP") < titles.index("Signal-to-Noise") < titles.index("CP Build-up")
    assert titles[-1] == "Offset Tracker (HMQC / J-resolved, O2/O3)"


def test_parse_typed_regions():
    assert parse_window("12, 8") == parse_window("8 to 12") == parse_window("12-8") == (12.0, 8.0)
    assert parse_window("5 -5") == (5.0, -5.0)
    assert parse_windows("60, 40; -40, -60") == [(60.0, 40.0), (-40.0, -60.0)]
    with pytest.raises(ValueError):
        parse_window("12")


def test_shared_regions_typed_and_calculated(window):
    p = _load(window)
    assert not p.calculate_button.isEnabled()          # no regions yet
    p.signal_edit.setText("5, -5")
    p._on_signal_typed()
    p.noise_edit.setText("80, 40")
    p._on_noise_typed()
    assert p.calculate_button.isEnabled()
    result = p.calculate(explicit=True)
    assert result is not None and not result["regions_differ"]
    expected = snr.signal_to_noise_many([e["spectrum"] for e in p._entries], (5, -5), (80, 40),
                                        labels=p.labels())
    assert [r["snr"] for r in result["rows"]] == pytest.approx(
        [r["snr"] for r in expected["rows"]])
    assert "Same regions for every spectrum" in p.result_label.text()
    # Four times the scans and twice the height: √NS says 2.
    c = result["comparisons"][0]
    assert c["expected_from_scans"] == pytest.approx(2.0) and c["consistent_with_scans"]
    # The table shows each S/N; the log says what was done.
    assert "±" in p.spectra_table.item(0, 3).text()
    assert "Same regions for all" in p.log_view.toPlainText()
    # Drawn on the Compare view with the S/N written in.
    texts = [t.get_text() for ax in window.compare_canvas.figure.axes for t in ax.texts]
    assert any(t.startswith("S/N") for t in texts)


def test_result_is_kept_with_each_spectrum(window):
    p = _load(window)
    p.set_signal_region(5, -5)
    p.set_noise_region((80, 40))
    p.calculate()
    for e in p._entries:
        kept = window.session.spectra[e["index"]].metadata.raw["fits"]["snr"]
        assert kept["label"] == "Signal-to-noise" and "S/N" in kept["report"]
        assert any("signal 5 to -5 ppm" in line for line in kept["inputs_text"])


def test_drag_sets_the_region(window):
    p = _load(window)
    p.signal_drag_button.setChecked(True)
    assert p.armed == "signal" and window.compare_canvas._region_select_mode
    window.compare_canvas.regionSelected.emit(-4.0, 6.0)
    assert p.shared_signal == (6.0, -4.0) and p.armed is None
    assert p.signal_edit.text() == "6 to -4"
    p.noise_drag_button.setChecked(True)
    window.compare_canvas.regionSelected.emit(30.0, 70.0)
    assert p.shared_noise == [(70.0, 30.0)]
    # The DNP panels, not armed, ignored both drags.
    assert window.dnp_enhancement_panel.region is None


def test_own_regions_are_marked_logged_and_reported(window):
    p = _load(window)
    p.set_signal_region(5, -5)
    p.set_noise_region((80, 40))
    p.different_check.setChecked(True)
    second = p._entries[1]
    p.target_combo.setCurrentIndex(p.target_combo.findData(id(second["spectrum"])))
    p.noise_edit.setText("-40, -80")
    p._on_noise_typed()
    log = p.log_view.toPlainText()
    assert f"Noise region for {second['label']}: -40 to -80 ppm – DIFFERS" in log
    assert "†" in p.spectra_table.item(1, 2).text() and "†" not in p.spectra_table.item(0, 2).text()
    result = p.calculate()
    assert result["regions_differ"] and result["differing"] == [second["label"]]
    assert result["rows"][1]["noise_region"] == [(-40.0, -80.0)]
    assert result["rows"][1]["signal_region"] == (5.0, -5.0)
    assert "REGIONS DIFFERED BETWEEN SPECTRA" in p.result_label.text()
    assert "Regions DIFFERED" in p.log_view.toPlainText()
    assert "own regions †" in p.spectra_table.item(1, 6).text()
    kept = window.session.spectra[second["index"]].metadata.raw["fits"]["snr"]
    assert any("(OWN regions)" in line for line in kept["inputs_text"])
    # Switched off: the shared regions for all again, and said so.
    p.different_check.setChecked(False)
    assert not p.calculate()["regions_differ"]
    # Back to the shared region for that spectrum.
    p.different_check.setChecked(True)
    p.target_combo.setCurrentIndex(p.target_combo.findData(id(second["spectrum"])))
    p.reset_own()
    assert not p.calculate()["regions_differ"]


def test_changing_a_region_invalidates_the_result(window):
    p = _load(window)
    p.set_signal_region(5, -5)
    p.set_noise_region((80, 40))
    p.calculate()
    p.set_signal_region(3, -3)
    assert p.result is None and "calculate again" in p.result_label.text()
    assert not p.export_button.isEnabled()


def test_bad_regions_are_reported_not_raised(window, _dialogs):
    p = _load(window)
    p.set_signal_region(5, -5)
    p.set_noise_region((3, -20))                 # overlaps the signal
    assert p.calculate(explicit=True) is None
    assert any("overlaps" in text for _kind, text in _dialogs)


def test_notebook_code_and_csv(window, tmp_path, monkeypatch):
    p = _load(window)
    p.set_signal_region(5, -5)
    p.set_noise_region([(80, 60), (-60, -80)])
    p.calculate()
    code = p.notebook_code()
    assert "signal_region=(5.0, -5.0)" in code and "(-60.0, -80.0)" in code
    compile(code, "<snr>", "exec")
    path = tmp_path / "out.csv"
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    p._on_export()
    assert path.read_text(encoding="utf-8").startswith("spectrum,S/N")
