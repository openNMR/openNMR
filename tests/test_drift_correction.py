"""Linear field-drift correction (notebook.drift_correction), a port of
Najbauer & Andreas's lindriftcomp / concatenate_ser_nuslist AU programs
(J. Magn. Reson. 305 (2019) 1-4).

Three kinds of check:

* **Translation**: the AU program's own index arithmetic and rotations,
  written out loop by loop exactly as in its C source, against the
  vectorised port (``endpoint=False`` reproduces the AU's ramp).
* **Physics, end to end**: a synthetic dataset whose peak moves with a
  known linear drift, read through openNMR's own Bruker reader with
  and without the correction, against the same dataset with no drift.
  This is what pins the signs (direct and every indirect mode), which
  the AU program cannot tell us about openNMR's conventions.
* **Files**: a Bruker experiment written to disk, corrected into a new
  EXPNO, and NUS blocks concatenated.
"""

import math
import os

import numpy as np
import pytest

from opennmr.io import bruker
from opennmr.notebook import drift_correction as dc

from test_bruker_raw_2d import _mock_2d


# ----------------------------------------------------------------------
# ramp and layout
# ----------------------------------------------------------------------

def test_drift_profile_ends_on_the_end_shift():
    shifts = dc.drift_profile(5, 1.0, 9.0)
    np.testing.assert_allclose(shifts, [1, 3, 5, 7, 9])


def test_drift_profile_au_mode_matches_lindriftcomp():
    # start + i*(end-start)/tot_points, as the AU program computes it.
    shifts = dc.drift_profile(4, 2.0, 10.0, endpoint=False)
    np.testing.assert_allclose(shifts, [2.0 + i * 8.0 / 4 for i in range(4)])


def _dims(*specs):
    return [dc.IndirectDim(name=f"F{i + 1}", n=n, kind=kind, dwell_s=1e-4, freq_ratio=0.25)
            for i, (n, kind) in enumerate(specs)]


def test_layout_3d_planes_321_matches_au_index_arithmetic():
    n1, n2 = 3, 4  # td1s/2, td2s/2
    rows, k = dc.acquisition_layout(_dims((n1, "states"), (n2, "states")), [0, 1], fntype=0)
    td2s = 2 * n2
    expected = []
    for j in range(n1):          # slow: F1
        for kk in range(n2):     # fast: F2
            rrr = 2 * j * td2s + 2 * kk
            rir = 2 * j * td2s + (2 * kk + 1)
            irr = (2 * j + 1) * td2s + 2 * kk
            iir = (2 * j + 1) * td2s + (2 * kk + 1)
            expected.append([rrr, rir, irr, iir])
    np.testing.assert_array_equal(rows, expected)
    assert k[1].tolist() == [0, 1]  # F1 index 0, F2 index 1


def test_layout_3d_planes_312_puts_f2_outermost():
    n1, n2 = 3, 2
    rows, k = dc.acquisition_layout(_dims((n1, "states"), (n2, "states")), [1, 0], fntype=0)
    td1s = 2 * n1
    expected = []
    for j in range(n2):          # slow: F2
        for kk in range(n1):     # fast: F1
            expected.append([2 * j * td1s + 2 * kk, 2 * j * td1s + 2 * kk + 1,
                             (2 * j + 1) * td1s + 2 * kk, (2 * j + 1) * td1s + 2 * kk + 1])
    np.testing.assert_array_equal(rows, expected)
    assert k[1].tolist() == [1, 0]  # second point: F1 index 1, F2 index 0


def test_layout_4d_planes_matches_au_block_offsets():
    n1, n2, n3 = 2, 2, 3
    rows, _ = dc.acquisition_layout(_dims((n1, "states"), (n2, "states"), (n3, "states")), fntype=0)
    td2s, td3s = 2 * n2, 2 * n3
    block = td2s * td3s * 2  # rows per i
    first = []
    for i in range(n1):
        for j in range(n2):
            for kk in range(n3):
                base = i * block
                r = [base + 2 * j * td3s + 2 * kk, base + 2 * j * td3s + 2 * kk + 1,
                     base + (2 * j + 1) * td3s + 2 * kk, base + (2 * j + 1) * td3s + 2 * kk + 1]
                r += [x + td2s * td3s for x in r]
                first.append(r)
    np.testing.assert_array_equal(rows, first)


def test_layout_points_and_nus_keep_each_points_fids_together():
    dims = _dims((2, "states"), (3, "states"))
    rows, _ = dc.acquisition_layout(dims, [0, 1], fntype=1)
    np.testing.assert_array_equal(rows[:2], [[0, 1, 2, 3], [4, 5, 6, 7]])
    nus = np.array([[0, 0], [1, 2], [0, 1]])
    rows, k = dc.acquisition_layout(dims, [0, 1], fntype=2, nus_indices=nus)
    np.testing.assert_array_equal(rows[2], [8, 9, 10, 11])
    np.testing.assert_array_equal(k, nus)


# ----------------------------------------------------------------------
# translation: the AU program's own loops, against the port
# ----------------------------------------------------------------------

def _rot(c, s, a, b):
    return c * a - s * b, s * a + c * b


def _au_3d_planes_321(fids, start, end, sw, gd, n1, n2, in1, in2, r1, r2):
    """lindriftcomp's 3D planes / AQSEQ 321 branch, as written (complex
    numbers in place of its re/im int pairs)."""
    out = fids.copy()
    td2s = 2 * n2
    tot = n1 * n2
    nt = fids.shape[1]
    dur = (np.arange(nt) - gd) / sw
    for j in range(n1):
        for k in range(n2):
            e = td2s // 2 * j + k
            delta = start + e * (end - start) / tot
            idx = [2 * j * td2s + 2 * k, 2 * j * td2s + 2 * k + 1,
                   (2 * j + 1) * td2s + 2 * k, (2 * j + 1) * td2s + 2 * k + 1]
            rrr, rir, irr, iir = (fids[i] * np.exp(2j * math.pi * delta * dur) for i in idx)
            a2 = delta * in2 * k * 2 * math.pi * r2
            c2, s2 = math.cos(a2), math.sin(a2)
            rrr, rir = _rot(c2, s2, rrr, rir)
            irr, iir = _rot(c2, s2, irr, iir)
            a3 = delta * in1 * j * 2 * math.pi * r1
            c3, s3 = math.cos(a3), math.sin(a3)
            rrr, irr = _rot(c3, s3, rrr, irr)
            rir, iir = _rot(c3, s3, rir, iir)
            for i, v in zip(idx, (rrr, rir, irr, iir)):
                out[i] = v
    return out


def _random_fids(n_rows, n_t2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_rows, n_t2)) + 1j * rng.normal(size=(n_rows, n_t2))


def test_port_reproduces_lindriftcomp_3d_planes_321():
    n1, n2, nt, sw, gd = 3, 4, 16, 5000.0, 2.5
    in1, in2, r1, r2 = 2e-4, 3e-4, 0.25, 0.1
    fids = _random_fids(4 * n1 * n2, nt)
    expected = _au_3d_planes_321(fids, 4.0, -6.0, sw, gd, n1, n2, in1, in2, r1, r2)
    dims = [dc.IndirectDim("F1", n1, "states", in1, r1), dc.IndirectDim("F2", n2, "states", in2, r2)]
    shifts = dc.drift_profile(n1 * n2, 4.0, -6.0, endpoint=False)
    got = dc.apply_drift_correction(fids, shifts, sw, dims, [0, 1], fntype=0, group_delay=gd)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_port_reproduces_lindriftcomp_3d_nus_321():
    n_t2, sw = 8, 4000.0
    in1, in2, r1, r2 = 1e-4, 2e-4, 0.4, 0.1
    nus = np.array([[0, 0], [2, 1], [1, 3], [3, 0]])  # (F1, F2) per entry
    fids = _random_fids(4 * len(nus), n_t2, seed=1)
    out = fids.copy()
    tot = len(nus) * 4
    dur = np.arange(n_t2) / sw
    for p, (f1, f2) in enumerate(nus):
        j = 4 * p
        delta = 1.0 + j * (5.0 - 1.0) / tot
        rows = [fids[j + i] * np.exp(2j * math.pi * delta * dur) for i in range(4)]
        a2 = delta * in2 * f2 * 2 * math.pi * r2
        c2, s2 = math.cos(a2), math.sin(a2)
        rows[0], rows[1] = _rot(c2, s2, rows[0], rows[1])
        rows[2], rows[3] = _rot(c2, s2, rows[2], rows[3])
        a3 = delta * in1 * f1 * 2 * math.pi * r1
        c3, s3 = math.cos(a3), math.sin(a3)
        rows[0], rows[2] = _rot(c3, s3, rows[0], rows[2])
        rows[1], rows[3] = _rot(c3, s3, rows[1], rows[3])
        out[j:j + 4] = rows
    dims = [dc.IndirectDim("F1", 4, "states", in1, r1), dc.IndirectDim("F2", 4, "states", in2, r2)]
    got = dc.apply_drift_correction(fids, dc.drift_profile(len(nus), 1.0, 5.0, endpoint=False), sw,
                                    dims, [0, 1], fntype=2, nus_indices=nus)
    np.testing.assert_allclose(got, out, atol=1e-12)


def test_port_reproduces_lindriftcomp_2d_echo_antiecho_via_states():
    """The AU converts each echo/antiecho pair to States, rotates, and
    writes States. The port keeps EA and phases echo/antiecho instead;
    converting its output the AU's way must give the AU's States rows."""
    n1, nt, sw, in1, r1 = 5, 12, 3000.0, 2e-4, 0.25
    fids = _random_fids(2 * n1, nt, seed=2)
    shifts = dc.drift_profile(n1, 3.0, -2.0, endpoint=False)
    dur = np.arange(nt) / sw

    def to_states(e, a):  # lindriftcomp's own EA -> States mapping
        return 1j * (e + a) / 2, -(e - a) / 2

    expected_c, expected_s = [], []
    for i in range(n1):
        e = fids[2 * i] * np.exp(2j * math.pi * shifts[i] * dur)
        a = fids[2 * i + 1] * np.exp(2j * math.pi * shifts[i] * dur)
        c_row, s_row = to_states(e, a)
        theta = shifts[i] * in1 * i * 2 * math.pi * r1
        c_row, s_row = _rot(math.cos(theta), math.sin(theta), c_row, s_row)
        expected_c.append(c_row)
        expected_s.append(s_row)
    got = dc.apply_drift_correction(fids, shifts, sw, [dc.IndirectDim("F1", n1, "echo_antiecho", in1, r1)])
    got_c, got_s = to_states(got[0::2], got[1::2])
    np.testing.assert_allclose(got_c, expected_c, atol=1e-12)
    np.testing.assert_allclose(got_s, expected_s, atol=1e-12)


def test_opennmr_convention_is_the_conjugate_of_bruker():
    fids = _random_fids(8, 16, seed=3)
    dims = [dc.IndirectDim("F1", 4, "states", 1e-4, 0.25)]
    shifts = dc.drift_profile(4, 2.0, 7.0)
    bruker_side = dc.apply_drift_correction(fids, shifts, 4000.0, dims)
    opennmr_side = dc.apply_drift_correction(np.conj(fids), shifts, 4000.0, dims, convention="opennmr")
    np.testing.assert_allclose(opennmr_side, np.conj(bruker_side), atol=1e-12)


def test_input_is_not_modified_and_partial_acquisition_is_corrected_up_to_its_end():
    dims = [dc.IndirectDim("F1", 6, "states", 1e-4, 0.25)]
    fids = _random_fids(9, 8, seed=4)  # 4.5 of 6 points acquired
    copy = fids.copy()
    out = dc.apply_drift_correction(fids, dc.drift_profile(6, 1.0, 2.0), 4000.0, dims)
    np.testing.assert_array_equal(fids, copy)
    np.testing.assert_array_equal(out[8], fids[8])  # the half point is left alone
    assert not np.allclose(out[:8], fids[:8])


# ----------------------------------------------------------------------
# physics end to end, through openNMR's own reader
# ----------------------------------------------------------------------

N1, N2, SW1, SW2 = 32, 512, 4000.0, 10000.0
BF_H, BF_X = 400.0, 100.6
F2_HZ, F1_HZ = 1500.0, 700.0


def _write_procs(path, params):
    with open(path, "w") as handle:
        handle.write("\n".join(f"##${k}= {v}" for k, v in params.items()) + "\n")


def _experiment(tmp_path, name="exp"):
    expt = tmp_path / name
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_procs(pdata / "procs", {"SW_p": SW2, "SI": 2048, "SF": BF_H, "WDW": 0})
    _write_procs(pdata / "proc2s", {"SW_p": SW1, "SI": 4 * N1, "SF": BF_X, "WDW": 0})
    return str(expt)


def _synthetic(fn_mode, start_hz, end_hz, pseudo=False):
    """Bruker-convention raw rows for one peak at (F1_HZ, F2_HZ), with
    the field drifting so that the peak sits ``-Δ_p`` from where it
    belongs while point ``p`` is acquired (the correction is then +Δ)."""
    t2 = np.arange(N2) / SW2
    deltas = dc.drift_profile(N1, start_hz, end_hz)
    ratio = BF_X / BF_H
    rows = []
    for k in range(N1):
        d = deltas[k]
        f2 = np.exp(-t2 * 40) * np.exp(2j * np.pi * (F2_HZ - d) * t2)
        if pseudo:
            rows.append(f2 * (0.97 ** k))
            continue
        t1 = k / SW1
        f1 = F1_HZ - d * ratio
        decay = np.exp(-t1 * 30)
        if fn_mode == 1:
            rows.append(decay * np.exp(-2j * np.pi * f1 * t1) * f2)
            continue
        c, s = decay * np.cos(2 * np.pi * f1 * t1) * f2, decay * np.sin(2 * np.pi * f1 * t1) * f2
        if fn_mode == 5 and k % 2:
            c, s = -c, -s
        if fn_mode == 6:
            rows += [-1j * c - s, -1j * c + s]
        else:
            rows += [c, s]
    return np.array(rows)


def _read(monkeypatch, path, fid, fn_mode, pseudo=False, **kwargs):
    acqus = {"NUC1": "1H", "BF1": BF_H, "O1": 0.0, "NS": 1.0, "SW_h": SW2, "TD": 2 * N2}
    acqu2s = {"NUC1": "off" if pseudo else "13C", "BF1": BF_X, "O1": 0.0, "FnMODE": fn_mode,
              "SW_h": SW1, "TD": fid.shape[0]}
    _mock_2d(monkeypatch, acqus, fid, acqu2s=acqu2s)
    return bruker.read_2d(path, homonuclear=False, autophase=False, **kwargs)


def _corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


@pytest.mark.parametrize("fn_mode", [1, 4, 5, 6])
def test_correction_restores_the_undrifted_2d(tmp_path, monkeypatch, fn_mode):
    path = _experiment(tmp_path)
    clean = _read(monkeypatch, path, _synthetic(fn_mode, 0.0, 0.0), fn_mode)
    drifted_fid = _synthetic(fn_mode, 2.0, 30.0)
    drifted = _read(monkeypatch, path, drifted_fid, fn_mode)
    fixed = _read(monkeypatch, path, drifted_fid, fn_mode,
                  drift_correction={"start_hz": 2.0, "end_hz": 30.0})
    assert _corr(drifted.data, clean.data) < 0.9      # the drift visibly hurts...
    assert _corr(fixed.data, clean.data) > 0.999       # ...and is undone
    record = fixed.metadata.raw["drift_correction"]
    assert record["start_hz"] == 2.0 and record["end_hz"] == 30.0
    assert "Najbauer" in record["citation"]
    np.testing.assert_array_equal(np.conj(fixed.metadata.raw["fid_raw_uncorrected"]), drifted_fid)


@pytest.mark.parametrize("fn_mode", [4, 6])
def test_wrong_sign_makes_it_worse(tmp_path, monkeypatch, fn_mode):
    path = _experiment(tmp_path)
    clean = _read(monkeypatch, path, _synthetic(fn_mode, 0.0, 0.0), fn_mode)
    drifted_fid = _synthetic(fn_mode, 2.0, 30.0)
    drifted = _read(monkeypatch, path, drifted_fid, fn_mode)
    wrong = _read(monkeypatch, path, drifted_fid, fn_mode,
                  drift_correction={"start_hz": -2.0, "end_hz": -30.0})
    assert _corr(wrong.data, clean.data) < _corr(drifted.data, clean.data)


def test_f1_scaling_matters(tmp_path, monkeypatch):
    """Correcting F2 alone (as for a pseudo-2D) leaves the F1 smear:
    the indirect rotation is doing real work."""
    path = _experiment(tmp_path)
    clean = _read(monkeypatch, path, _synthetic(4, 0.0, 0.0), 4)
    drifted_fid = _synthetic(4, 0.0, 60.0)
    full = _read(monkeypatch, path, drifted_fid, 4, drift_correction={"start_hz": 0.0, "end_hz": 60.0})
    direct_only = dc.apply_drift_correction(
        np.conj(drifted_fid), dc.drift_profile(N1, 0.0, 60.0), SW2,
        [dc.IndirectDim("F1", N1, "none")], convention="opennmr")
    f2_only = _read(monkeypatch, path, np.conj(direct_only), 4)
    assert _corr(full.data, clean.data) > _corr(f2_only.data, clean.data) + 0.01


def test_pseudo_2d_rows_are_shifted_back_and_never_mixed(tmp_path, monkeypatch):
    path = _experiment(tmp_path)
    clean = _read(monkeypatch, path, _synthetic(1, 0.0, 0.0, pseudo=True), 1, pseudo=True)
    drifted_fid = _synthetic(1, 0.0, 25.0, pseudo=True)
    fixed = _read(monkeypatch, path, drifted_fid, 1, pseudo=True,
                  drift_correction={"start_hz": 0.0, "end_hz": 25.0})
    for row in range(N1):
        assert _corr(fixed.data[row], clean.data[row]) > 0.999
    # Each row's integral is untouched: nothing leaked between slices.
    np.testing.assert_allclose(np.abs(fixed.data).sum(axis=1) / np.abs(clean.data).sum(axis=1), 1, atol=0.02)


def test_unsupported_tppi_says_so(tmp_path, monkeypatch):
    path = _experiment(tmp_path)
    with pytest.raises(NotImplementedError, match="TPPI"):
        _read(monkeypatch, path, _synthetic(4, 0.0, 1.0), 3, drift_correction={"start_hz": 0, "end_hz": 1})


def test_1d_loads_ignore_the_drift_kwarg_instead_of_crashing(monkeypatch):
    called = {}
    monkeypatch.setattr(bruker.os.path, "exists", lambda p: p.endswith("fid"))
    monkeypatch.setattr(bruker, "read_1d", lambda path, **kw: called.setdefault("kw", kw))
    bruker._read_auto("x", drift_correction={"start_hz": 1.0})
    assert "drift_correction" not in called["kw"]


# ----------------------------------------------------------------------
# files: write a corrected EXPNO, concatenate NUS blocks
# ----------------------------------------------------------------------

def _write_jcamp(path, params):
    with open(path, "w", encoding="latin-1") as handle:
        handle.write("##TITLE= synthetic\n")
        for k, v in params.items():
            handle.write(f"##${k}= {v}\n")
        handle.write("##END=\n")


def _bruker_3d(folder, fntype=0, nus=None, td=256, n1=3, n2=4, seed=0):
    os.makedirs(os.path.join(folder, "pdata", "1"), exist_ok=True)
    acqus = {"TD": td, "SW_h": 8000.0, "BF1": 600.0, "DTYPA": 0, "BYTORDA": 0, "FnTYPE": fntype,
             "AQSEQ": 0, "DECIM": 1, "DSPFVS": 20, "GRPDLY": 0, "NUC1": "<1H>"}
    if nus is not None:
        acqus["NusPOINTS"] = len(nus)
    _write_jcamp(os.path.join(folder, "acqus"), acqus)
    _write_jcamp(os.path.join(folder, "acqu"), acqus)
    _write_jcamp(os.path.join(folder, "acqu2s"), {"TD": 2 * n2, "SW_h": 2000.0, "BF1": 150.9,
                                                  "FnMODE": 5, "NUC1": "<13C>"})
    _write_jcamp(os.path.join(folder, "acqu3s"), {"TD": 2 * n1, "SW_h": 1500.0, "BF1": 60.8,
                                                  "FnMODE": 4, "NUC1": "<15N>"})
    n_fids = 4 * (len(nus) if nus is not None else n1 * n2)
    rng = np.random.default_rng(seed)
    data = rng.integers(-2 ** 20, 2 ** 20, size=n_fids * td, dtype=np.int32)
    data.astype("<i4").tofile(os.path.join(folder, "ser"))
    if nus is not None:
        with open(os.path.join(folder, "nuslist"), "w") as handle:
            for f1, f2 in nus:
                handle.write(f"{f2} {f1}\n")  # the file lists F2 first
    return data


def test_correct_bruker_dataset_writes_a_new_expno_and_leaves_the_original(tmp_path):
    src = str(tmp_path / "proj" / "12")
    original = _bruker_3d(src)
    result = dc.correct_bruker_dataset(src, None, 1.5, 9.0)
    assert result.destination.endswith(os.sep + "13")
    np.testing.assert_array_equal(np.fromfile(os.path.join(src, "ser"), "<i4"), original)

    setup = dc.bruker_drift_setup(src)
    assert [d.kind for d in setup.dims] == ["states", "states"]
    assert setup.dims[0].freq_ratio == pytest.approx(60.8 / 600.0)   # F1 = acqu3s
    assert setup.dims[1].freq_ratio == pytest.approx(150.9 / 600.0)  # F2 = acqu2s
    fids = original.astype(float).reshape(-1, 256)
    fids = fids[:, 0::2] + 1j * fids[:, 1::2]
    expected = dc.apply_drift_correction(fids, dc.drift_profile(12, 1.5, 9.0), 8000.0, setup.dims,
                                         setup.loop_order)
    written = np.fromfile(os.path.join(result.destination, "ser"), "<i4").astype(float).reshape(-1, 256)
    np.testing.assert_allclose(written[:, 0::2], np.rint(expected.real), atol=0.5)
    np.testing.assert_allclose(written[:, 1::2], np.rint(expected.imag), atol=0.5)
    with open(os.path.join(result.destination, "opennmr_drift_correction.txt")) as handle:
        assert "Najbauer" in handle.read()
    assert "Najbauer" in result.report()
    with pytest.raises(FileExistsError):
        dc.correct_bruker_dataset(src, result.destination, 1.5, 9.0)
    with pytest.raises(ValueError, match="new experiment"):
        dc.correct_bruker_dataset(src, src, 1.5, 9.0, overwrite=True)


def test_nus_blocks_are_corrected_then_concatenated(tmp_path):
    nus_a = [(0, 0), (1, 0), (0, 1)]
    nus_b = [(2, 3), (1, 2)]
    a, b = str(tmp_path / "p" / "20"), str(tmp_path / "p" / "21")
    _bruker_3d(a, fntype=2, nus=nus_a, seed=1)
    _bruker_3d(b, fntype=2, nus=nus_b, seed=2)
    ca = dc.correct_bruker_dataset(a, str(tmp_path / "p" / "120"), 0.0, 3.0)
    cb = dc.correct_bruker_dataset(b, str(tmp_path / "p" / "121"), 3.0, 6.0)
    assert ca.n_points == 3 and cb.n_points == 2
    out = dc.concatenate_nus_datasets([ca.destination, cb.destination], str(tmp_path / "p" / "130"))
    assert out["nus_points"] == 5 and set(out["updated_parameters"]) == {"acqus", "acqu"}
    joined = np.fromfile(os.path.join(out["destination"], "ser"), "<i4")
    parts = [np.fromfile(os.path.join(p, "ser"), "<i4") for p in (ca.destination, cb.destination)]
    np.testing.assert_array_equal(joined, np.concatenate(parts))
    setup = dc.bruker_drift_setup(out["destination"])
    np.testing.assert_array_equal(setup.nus_indices, nus_a + nus_b)
    with open(os.path.join(out["destination"], "nuslist")) as handle:
        assert handle.read().splitlines()[3] == "3 2"  # F2 F1, as the file had it


def test_concatenation_refuses_mismatched_blocks(tmp_path):
    a, b = str(tmp_path / "p" / "1"), str(tmp_path / "p" / "2")
    _bruker_3d(a, fntype=2, nus=[(0, 0)])
    _bruker_3d(b, fntype=2, nus=[(1, 0)], td=512)
    with pytest.raises(ValueError, match="does not match"):
        dc.concatenate_nus_datasets([a, b], str(tmp_path / "p" / "3"))
    c = str(tmp_path / "p" / "4")
    _bruker_3d(c)
    with pytest.raises(ValueError, match="not NUS"):
        dc.concatenate_nus_datasets([a, c], str(tmp_path / "p" / "5"))


def test_1d_experiment_is_refused_with_the_reason(tmp_path):
    folder = tmp_path / "1d"
    folder.mkdir()
    _write_jcamp(folder / "acqus", {"TD": 256, "SW_h": 8000.0, "BF1": 600.0})
    with pytest.raises(ValueError, match="summed"):
        dc.bruker_drift_setup(str(folder))


# ----------------------------------------------------------------------
# measuring the drift
# ----------------------------------------------------------------------

def _one_d(centre_ppm, sfo1=400.0, sr_hz=0.0, n=4096, width_ppm=0.02):
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    ppm = np.linspace(10, 0, n)
    data = 1.0 / (1 + ((ppm - centre_ppm) / width_ppm) ** 2)
    meta = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=sfo1, bf1=sfo1,
                       raw={"sr_recorded_hz": sr_hz} if sr_hz else {})
    return NMRSpectrum(data=data, axes=[NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)], metadata=meta)


def test_measure_drift_reads_a_sub_point_shift_with_the_right_sign():
    reference = _one_d(4.70)
    after = _one_d(4.70 - 3.3 / 400.0)  # drifted 3.3 Hz down -> needs +3.3
    assert dc.measure_drift_hz(reference, after, (5.0, 4.4)) == pytest.approx(3.3, abs=0.05)
    assert dc.measure_drift_hz(reference, reference, (5.0, 4.4)) == pytest.approx(0.0, abs=1e-9)


def test_measure_drift_compares_frequencies_not_ppm():
    """Same peak frequency, but one spectrum carries a +40 Hz SR (its 0
    ppm sits 40 Hz higher, so the peak reads ~0.1 ppm lower): no drift."""
    reference = _one_d(4.70)
    referenced = _one_d((4.70 * 400.0 - 40.0) / (400.0 + 40e-6), sr_hz=40.0)
    assert dc.measure_drift_hz(reference, referenced, (5.2, 4.4)) == pytest.approx(0.0, abs=0.05)
