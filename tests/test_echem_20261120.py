"""Electrochemistry data (opennmr.notebook.echem)."""
import numpy as np
import pandas as pd
import pytest

from opennmr.notebook import echem

DT_H = 0.05
CURRENT = 0.1          # mA
MASS_MG = 5.0          # → 200 mAh/g for 10 h at 0.1 mA


def _profile(cycles=2, wobble=False):
    t, v, i = [], [], []
    clock = 0.0
    for _ in range(cycles):
        for sign, v0, v1 in ((+1, 3.0, 4.0), (-1, 4.0, 3.0)):
            n = int(10 / DT_H)
            for k in range(n + 1):
                frac = k / n
                volt = v0 + (v1 - v0) * frac
                if wobble and sign > 0:
                    volt += 0.15 * np.sin(2 * np.pi * 3 * frac)
                t.append(clock + k * DT_H)
                v.append(volt)
                i.append(sign * CURRENT)
            clock += 10 + DT_H
            for k in range(int(0.5 / DT_H)):
                t.append(clock + k * DT_H)
                v.append(v1)
                i.append(0.0)
            clock += 0.5
    return pd.DataFrame({"Time/s": np.array(t) * 3600, "Ewe/V": v, "I/mA": i})


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "cell.csv"
    _profile().to_csv(path, index=False)
    return path


@pytest.fixture
def cell(csv_path):
    return echem.load_echem(csv_path, mass_mg=MASS_MG, area_cm2=0.5, molar_mass=100.0, x0=1.0)


def test_table_is_read_and_standard_columns_added(cell):
    df = cell["data"]
    assert cell["reader"] == "table"
    for col in ("time_s", "time_h", "voltage_V", "current_mA", "state", "half_cycle", "cycle",
                "capacity_mAh", "capacity_spec_mAh_g", "capacity_area_mAh_cm2", "x"):
        assert col in df.columns
    assert "Ewe/V" in df.columns                      # the file's own columns stay
    assert set(df["state"]) == {"charge", "discharge", "rest"}
    assert sorted(df["cycle"].unique()) == [1, 2]
    assert sorted(df["half_cycle"].unique()) == [0, 1, 2, 3]


def test_capacity_resets_each_half_cycle_and_scales_with_mass_and_area(cell):
    df = cell["data"]
    first = df[df["half_cycle"] == 0]
    assert first["capacity_mAh"].max() == pytest.approx(1.0, rel=1e-2)       # 0.1 mA × 10 h
    assert first["capacity_spec_mAh_g"].max() == pytest.approx(200.0, rel=1e-2)
    assert first["capacity_area_mAh_cm2"].max() == pytest.approx(2.0, rel=1e-2)
    second = df[df["half_cycle"] == 1]
    assert second["capacity_mAh"].iloc[0] < 0.01      # reset
    assert second["capacity_mAh"].max() == pytest.approx(1.0, rel=1e-2)


def test_composition_follows_faradays_law_and_returns_after_a_full_cycle(cell):
    df = cell["data"]
    delta = 200.0 * 100.0 / echem.FARADAY_MAH_PER_MOL
    top = df[df["half_cycle"] == 0].iloc[-1]
    assert top["x"] == pytest.approx(1.0 - delta, rel=1e-2)
    assert df[df["half_cycle"] == 1].iloc[-1]["x"] == pytest.approx(1.0, abs=1e-2)
    anode = echem.load_echem(cell["source"], mass_mg=MASS_MG, molar_mass=100.0, x0=0.0,
                             charge_removes_ion=False)["data"]
    assert anode[anode["half_cycle"] == 0].iloc[-1]["x"] == pytest.approx(delta, rel=1e-2)


def test_composition_needs_all_three_and_says_what_is_missing(csv_path):
    loaded = echem.load_echem(csv_path, mass_mg=MASS_MG, molar_mass=100.0)
    assert "x" not in loaded["data"] and any("starting composition" in n for n in loaded["notes"])
    none = echem.load_echem(csv_path)
    assert "capacity_spec_mAh_g" not in none["data"] and none["notes"] == []


def test_formula_gives_the_molar_mass():
    assert echem.molar_mass_from_formula("LiFePO4") == pytest.approx(157.76, abs=0.05)
    with pytest.raises(ValueError, match="formula"):
        echem.molar_mass_from_formula("not a formula ((")


def test_units_of_the_file_are_converted(tmp_path):
    df = _profile()
    path = tmp_path / "units.csv"
    pd.DataFrame({"Time/s": df["Time/s"] / 3600, "Ewe/V": df["Ewe/V"],
                  "I/mA": df["I/mA"] / 1000}).to_csv(path, index=False)
    loaded = echem.load_echem(path, current_unit="A", time_unit="h", mass_mg=MASS_MG)["data"]
    assert loaded["time_h"].iloc[-1] == pytest.approx(df["Time/s"].iloc[-1] / 3600)
    assert loaded["capacity_spec_mAh_g"].max() == pytest.approx(200.0, rel=1e-2)
    with pytest.raises(ValueError, match="current_unit"):
        echem.derive(df, current_unit="kA")


def test_semicolon_and_decimal_comma_and_eclab_header(tmp_path):
    df = _profile()
    text = df.head(50).to_csv(index=False, sep=";", decimal=",")
    path = tmp_path / "eu.csv"
    path.write_text(text, encoding="utf-8")
    loaded = echem.load_echem(path)["data"]
    assert loaded["voltage_V"].iloc[1] == pytest.approx(df["Ewe/V"].iloc[1])
    path2 = tmp_path / "ec.txt"
    path2.write_text("EC-Lab ASCII FILE\nNb header lines : 4\nsomething\n"
                     + df.head(20).to_csv(index=False, sep="\t"), encoding="utf-8")
    assert len(echem.load_echem(path2)["data"]) == 20


def test_missing_file_bad_reader_and_missing_navani(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        echem.load_echem(tmp_path / "nope.csv")
    p = tmp_path / "x.mpr"
    p.write_bytes(b"not a real file")
    with pytest.raises(ValueError, match="reader"):
        echem.load_echem(p, reader="magic")
    import builtins
    real = builtins.__import__

    def refuse(name, *a, **k):
        if name.startswith(("navani", "galvani")):
            raise ImportError("no navani")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ImportError, match="pip install navani"):
        echem.load_echem(p)


def test_plottable_columns_lists_standard_then_the_files_own(cell):
    df = cell["data"].assign(Temperature=25.0)
    columns = echem.plottable_columns(df)
    assert columns[0] == "time_s" and "Temperature" in columns and "state" not in columns
    assert columns.index("voltage_V") < columns.index("Temperature")
    assert echem.axis_label("voltage_V") == "Voltage (V)" and echem.axis_label("Temperature") == "Temperature"


def test_select_cycles(cell):
    df = cell["data"]
    assert set(echem.select_cycles(df, 2, "charge")["cycle"]) == {2}
    assert set(echem.select_cycles(df, [1, 2], "discharge")["state"]) == {"discharge"}
    assert len(echem.select_cycles(df)) == len(df)
    with pytest.raises(ValueError, match="Cycle 9 does not exist"):
        echem.select_cycles(df, 9)
    with pytest.raises(ValueError, match="half"):
        echem.select_cycles(df, 1, "middle")


def test_ex_situ_point_needs_a_cycle_and_a_direction(cell):
    df = cell["data"]
    charge = echem.ex_situ_point(df, 3.5, cycle=1, half="charge")
    discharge = echem.ex_situ_point(df, 3.5, cycle=1, half="discharge")
    assert charge["row"]["capacity_spec_mAh_g"] == pytest.approx(100.0, rel=1e-2)
    assert discharge["row"]["capacity_spec_mAh_g"] == pytest.approx(100.0, rel=1e-2)
    assert charge["row"]["time_h"] < discharge["row"]["time_h"]
    later = echem.ex_situ_point(df, 3.5, cycle=2, half="charge")
    assert later["row"]["time_h"] > discharge["row"]["time_h"]
    assert charge["row"]["voltage_V"] == 3.5 and charge["n_crossings"] == 1 and charge["notes"] == []
    assert charge["row"]["x"] == pytest.approx(1.0 - 100.0 * 100.0 / echem.FARADAY_MAH_PER_MOL, rel=1e-2)


def test_ex_situ_point_interpolates_between_samples(cell):
    df = cell["data"]
    point = echem.ex_situ_point(df, 3.5012, cycle=1, half="charge")
    assert point["row"]["voltage_V"] == 3.5012
    assert point["row"]["capacity_spec_mAh_g"] == pytest.approx(200.0 * 0.5012, rel=2e-3)


def test_ex_situ_point_with_several_crossings_says_so_and_can_choose(tmp_path):
    path = tmp_path / "w.csv"
    _profile(cycles=1, wobble=True).to_csv(path, index=False)
    df = echem.load_echem(path, mass_mg=MASS_MG)["data"]
    first = echem.ex_situ_point(df, 3.5, cycle=1, half="charge", choose="first")
    last = echem.ex_situ_point(df, 3.5, cycle=1, half="charge", choose="last")
    assert first["n_crossings"] > 1 and first["notes"]
    assert first["row"]["capacity_mAh"] < last["row"]["capacity_mAh"]
    target = last["row"]["capacity_mAh"]
    near = echem.ex_situ_point(df, 3.5, cycle=1, half="charge", choose="nearest", near=target)
    assert near["row"]["capacity_mAh"] == pytest.approx(target)
    with pytest.raises(ValueError, match="near"):
        echem.ex_situ_point(df, 3.5, choose="nearest")


def test_ex_situ_point_errors_are_explained(cell):
    df = cell["data"]
    with pytest.raises(ValueError, match="not reached.*runs from 3"):
        echem.ex_situ_point(df, 4.5, cycle=1, half="charge")
    with pytest.raises(ValueError, match="no column"):
        echem.ex_situ_point(df, 3.5, quantity="nonsense")
    with pytest.raises(ValueError, match="choose"):
        echem.ex_situ_point(df, 3.5, choose="middle")
    results = echem.ex_situ_points(df, [{"value": 3.5, "label": "A"},
                                        {"value": 9.0, "label": "B"},
                                        {"value": 3.2, "cycle": 2, "half": "discharge"}])
    assert results[0]["label"] == "A" and "row" in results[0]
    assert results[1]["label"] == "B" and "not reached" in results[1]["error"]
    assert results[2]["label"] == "sample 3" and results[2]["cycle"] == 2


def test_other_quantities_can_locate_a_sample(cell):
    df = cell["data"]
    point = echem.ex_situ_point(df, 100.0, quantity="capacity_spec_mAh_g", cycle=1, half="charge")
    assert point["row"]["voltage_V"] == pytest.approx(3.5, abs=0.01)


def test_operando_times(tmp_path):
    starts, ends = echem.operando_times(4, start_s=100.0, per_spectrum_s=60.0, gaps_s=10.0)
    assert starts.tolist() == [100, 170, 240, 310] and ends.tolist() == [160, 230, 300, 370]
    starts, ends = echem.operando_times(3, durations_s=[10, 20, 30])
    assert starts.tolist() == [0, 10, 30] and ends.tolist() == [10, 30, 60]
    starts, ends = echem.operando_times(3, times_s=[0, 100, 300])
    assert ends.tolist() == [100, 300, 500]
    with pytest.raises(ValueError, match="Give"):
        echem.operando_times(3)
    with pytest.raises(ValueError, match="durations"):
        echem.operando_times(3, durations_s=[1, 2])
    with pytest.raises(ValueError, match="start times"):
        echem.operando_times(3, times_s=[1, 2])


def test_at_spectra_means_over_the_window_and_flags_outside(cell):
    df = cell["data"]
    starts, ends = echem.operando_times(3, start_s=0.0, per_spectrum_s=3600.0)
    table = echem.at_spectra(df, starts, ends)
    # Voltage rises 0.1 V per hour on the first charge: the mean over hour k is 3.05 + 0.1 k.
    np.testing.assert_allclose(table["voltage_V"], [3.05, 3.15, 3.25], atol=2e-3)
    assert table["capacity_spec_mAh_g_end"].tolist() == pytest.approx([20.0, 40.0, 60.0], rel=2e-2)
    assert table["x_end"].iloc[0] > table["x_end"].iloc[2]
    assert table["cycle_end"].tolist() == [1, 1, 1] and table.attrs["outside"] == 0
    late = echem.at_spectra(df, [1e9], [1e9 + 10])
    assert np.isnan(late["voltage_V"].iloc[0]) and late.attrs["outside"] == 1
    with pytest.raises(ValueError, match="time"):
        echem.at_spectra(pd.DataFrame({"a": [1, 2]}), [0], [1])


def test_spectrum_differences():
    m = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]])
    np.testing.assert_allclose(echem.spectrum_differences(m), [[0, 0], [2, 3], [3, 4]])
    np.testing.assert_allclose(echem.spectrum_differences(m, "from_first"), [[0, 0], [2, 3], [5, 7]])
    np.testing.assert_allclose(echem.spectrum_differences(m, "from_last"), [[-5, -7], [-3, -4], [0, 0]])
    with pytest.raises(ValueError, match="mode"):
        echem.spectrum_differences(m, "odd")
    with pytest.raises(ValueError, match="at least two"):
        echem.spectrum_differences(m[:1])


def test_read_value_table(tmp_path):
    path = tmp_path / "fit.csv"
    path.write_text("spectrum,shift,area\n1,10.5,3.0\n2,10.7,3.1\n", encoding="utf-8")
    table = echem.read_value_table(path)
    assert list(table.columns) == ["spectrum", "shift", "area"] and len(table) == 2
    one = tmp_path / "one.csv"
    one.write_text("a\n1\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two columns"):
        echem.read_value_table(one)


def test_formula_parser_without_pymatgen():
    assert echem.molar_mass_from_formula("NaMnO2") == pytest.approx(22.990 + 54.938 + 2 * 15.999, abs=1e-3)
    assert echem.molar_mass_from_formula("Na(Mn0.5Fe0.5)O2") == pytest.approx(
        22.990 + 0.5 * 54.938 + 0.5 * 55.845 + 2 * 15.999, abs=0.01)
    with pytest.raises(ValueError):
        echem.molar_mass_from_formula("Xx2")
    with pytest.raises(ValueError):
        echem.molar_mass_from_formula("Na(MnO2")


_MPR = r"C:\Users\quark\OneDrive\Documents\Electrochemistry\LIO\euanbassey_LIO003_cycle_test_10mAg-1_20230123_C06.mpr"


def test_mpr_reads_through_galvani_when_navani_is_missing(monkeypatch):
    import os, sys
    if not os.path.exists(_MPR):
        pytest.skip("real .mpr file not present")
    pytest.importorskip("galvani")
    monkeypatch.setitem(sys.modules, "navani", None)
    monkeypatch.setitem(sys.modules, "navani.echem", None)
    data = echem.load_echem(_MPR, mass_mg=1.0)["data"]
    assert data["cycle"].max() >= 2 and np.isfinite(data["voltage_V"]).all()
