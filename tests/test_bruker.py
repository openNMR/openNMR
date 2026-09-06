import os

import numpy as np
import pytest

from opennmr.io import bruker


def _write_params(path, params):
    lines = []
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"##${key}= (0..{len(value) - 1})")
            lines.append(" ".join(str(v) for v in value))
        elif isinstance(value, str):
            lines.append(f"##${key}= <{value}>")
        else:
            lines.append(f"##${key}= {value}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _make_1d_dataset(tmp_path, procno=1, n=8):
    expt = tmp_path / "expt1"
    pdata = expt / "pdata" / str(procno)
    pdata.mkdir(parents=True)

    _write_params(expt / "acqus", {
        "NUC1": "1H", "BF1": 400.0, "SFO1": 400.13, "O1": 2000.0, "NS": 4,
        "P": [10.0, 5.0] + [0.0] * 6,
    })
    _write_params(pdata / "procs", {
        "SW_p": 8000.0, "SI": n, "SF": 400.13, "NC_proc": 0,
    })

    real = np.arange(n, dtype="int32") - n // 2
    imag = np.zeros(n, dtype="int32")
    real.tofile(pdata / "1r")
    imag.tofile(pdata / "1i")
    return str(expt)


def test_read_1d_processed_axis_and_scaling(tmp_path):
    path = _make_1d_dataset(tmp_path, n=8)
    spectrum = bruker.read_1d_processed(path, mass=2.0)

    assert spectrum.ndim == 1
    assert len(spectrum.data) == 8
    assert spectrum.metadata.vendor == "bruker"
    assert spectrum.metadata.nucleus == "1H"
    # NC_proc=0 -> unscaled; mass=2, NS=4 -> divide by 8.
    expected = (np.arange(8) - 4) / 2.0 / 4.0
    np.testing.assert_allclose(spectrum.data, expected)
    # Descending ppm axis (NMR convention).
    assert spectrum.axes[0].ppm[0] > spectrum.axes[0].ppm[-1]


def test_read_1d_processed_crop(tmp_path):
    path = _make_1d_dataset(tmp_path, n=8)
    full = bruker.read_1d_processed(path)
    lo, hi = full.axes[0].ppm[-1], full.axes[0].ppm[0]
    mid = (lo + hi) / 2
    cropped = bruker.read_1d_processed(path, f1p=lo, f2p=mid)
    assert len(cropped.data) < len(full.data)
    assert cropped.axes[0].ppm.max() <= mid + 1e-9


def test_detect_requires_acqus_and_data(tmp_path):
    expt = tmp_path / "not_bruker"
    expt.mkdir()
    assert bruker._detect(str(expt)) is False

    (expt / "acqus").write_text("##$NUC1= <1H>\n")
    assert bruker._detect(str(expt)) is False  # no fid/ser/pdata yet

    (expt / "fid").write_bytes(b"\x00" * 16)
    assert bruker._detect(str(expt)) is True


def test_apply_bruker_window_none_is_noop():
    fid = np.ones(16, dtype=complex)
    out = bruker._apply_bruker_window(fid, {"WDW": 0}, sw_hz=8000.0)
    np.testing.assert_allclose(out, fid)


def test_apply_bruker_window_em_decays():
    fid = np.ones(16, dtype=complex)
    out = bruker._apply_bruker_window(fid, {"WDW": 1, "LB": 50.0}, sw_hz=8000.0)
    assert out[0] == pytest.approx(1.0)
    assert np.all(np.diff(np.abs(out)) <= 0)


def test_apply_bruker_window_unsupported_wdw_raises():
    fid = np.ones(16, dtype=complex)
    with pytest.raises(NotImplementedError):
        bruker._apply_bruker_window(fid, {"WDW": 2}, sw_hz=8000.0)


# ============================================================
# vdlist / vclist
# ============================================================

def test_read_vdlist_parses_suffixed_and_plain_values(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    (expt / "vdlist").write_text("10m\n1s\n500u\n2\n")

    result = bruker.read_vdlist(str(expt))
    np.testing.assert_allclose(result["seconds"], [0.01, 1.0, 0.0005, 2.0])
    assert result["raw"] == ["10m", "1s", "500u", "2"]
    assert result["name"] == "vdlist"


def test_read_vdlist_falls_back_to_lists_vd_subdir(tmp_path):
    expt = tmp_path / "expt1"
    (expt / "lists" / "vd").mkdir(parents=True)
    (expt / "lists" / "vd" / "vdlist").write_text("1m\n2m\n")

    result = bruker.read_vdlist(str(expt))
    np.testing.assert_allclose(result["seconds"], [0.001, 0.002])


def test_read_vdlist_missing_raises(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    with pytest.raises(FileNotFoundError):
        bruker.read_vdlist(str(expt))


def test_read_vclist_reads_plain_integer_counters(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    (expt / "vclist").write_text("1\n2\n4\n8\n16\n")

    result = bruker.read_vclist(str(expt))
    np.testing.assert_allclose(result["seconds"], [1, 2, 4, 8, 16])
    assert result["raw"] == ["1", "2", "4", "8", "16"]


# ============================================================
# diff.xml (PFG diffusion parameters)
# ============================================================

_DIFF_XML_TEMPLATE = """<?xml version="1.0" encoding="ISO-8859-1"?>
<diff>
    <delta>1.00</delta>
    <DELTA>20.0</DELTA>
    <exDiffCoff>6e-11</exDiffCoff>
    <xValues>16<List>0 27.84 11153.2 95950068.7 1 55.68 22306.4 383800000.0</List>
    </xValues>
</diff>
"""


def test_read_diff_xml_parses_gradient_and_b_values(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    (expt / "diff.xml").write_text(_DIFF_XML_TEMPLATE)

    result = bruker.read_diff_xml(str(expt))

    assert result["delta"] == pytest.approx(0.001)  # 1.00 ms -> s
    assert result["DELTA"] == pytest.approx(0.02)  # 20.0 ms -> s
    assert result["expD"] == pytest.approx(6e-11)
    np.testing.assert_allclose(result["gradient"], [0.2784, 0.5568])  # G/cm -> T/m
    np.testing.assert_allclose(result["B"], [95950068.7, 383800000.0])
    assert result["gamma"] is None


def test_read_diff_xml_includes_gamma_when_nucleus_given(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    (expt / "diff.xml").write_text(_DIFF_XML_TEMPLATE)

    result = bruker.read_diff_xml(str(expt), nucleus="1H")
    assert result["gamma"] is not None
    assert result["gamma"] > 0


# ============================================================
# read_2d_processed / _read_auto fallback (real-data regressions)
# ============================================================
#
# Found via real data (2H solid-state EXSY/HETCOR/BABA-DQ 2D
# experiments): (1) read_2d_processed's F1 block size silently came
# from procs (F2's own SI) instead of proc2s (F1's real SI) whenever
# homonuclear=True, since proc2s was skipped entirely in that branch;
# (2) nmrglue's own no-pulse-program shape-guessing has a real bug for
# DTYPA=2 (float64) 2D data with a complex direct dimension (a fixed
# 4-bytes-per-point divisor regardless of the actual 8-byte point
# size), which read_2d has no way to work around on its own -- there's
# no raw shape to determine at all, so _read_auto falls back to
# already-processed data instead, matching the escape-hatch role
# read_2d_processed's own docstring already describes.

def _make_2d_dataset_with_processed(tmp_path, n_f1=4, n_f2=8, name="expt2d"):
    """A genuine (non-pseudo) 2D dataset with real procs/proc2s SI
    values that differ from each other (n_f1 != n_f2 by default) --
    exactly the shape the read_2d_processed homonuclear/proc2s bug
    needs to be provable: using F2's SI for F1 too would silently
    produce a same-on-both-axes shape instead."""
    expt = tmp_path / name
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    _write_params(expt / "acqus", {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0, "NS": 1})
    _write_params(expt / "acqu2s", {"NUC1": "1H", "BF1": 400.0, "O1": 2000.0})
    _write_params(pdata / "procs", {"SW_p": 8000.0, "SI": n_f2, "SF": 400.13})
    _write_params(pdata / "proc2s", {"SW_p": 2000.0, "SI": n_f1, "SF": 400.13})
    data = np.arange(n_f1 * n_f2, dtype="int32").reshape(n_f1, n_f2)
    data.tofile(pdata / "2rr")
    (-data).tofile(pdata / "2ri")
    (expt / "ser").write_bytes(b"\x00")  # existence is all _read_auto checks; content unused (mocked)
    return str(expt)


def test_read_2d_processed_homonuclear_uses_real_proc2s_shape(tmp_path):
    path = _make_2d_dataset_with_processed(tmp_path, n_f1=4, n_f2=8)
    spectrum = bruker.read_2d_processed(path, homonuclear=True)
    assert spectrum.data.shape == (4, 8)  # not (8, 8) -- see this section's own docstring


def test_read_2d_processed_non_homonuclear_shape_unaffected(tmp_path):
    path = _make_2d_dataset_with_processed(tmp_path, n_f1=4, n_f2=8)
    spectrum = bruker.read_2d_processed(path, homonuclear=False)
    assert spectrum.data.shape == (4, 8)


def test_read_auto_falls_back_to_processed_when_read_2d_shape_fails(tmp_path, monkeypatch):
    path = _make_2d_dataset_with_processed(tmp_path, n_f1=4, n_f2=8)

    def fake_read_2d(_path, **_kwargs):
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(bruker, "read_2d", fake_read_2d)

    # read_2d_processed doesn't accept these -- confirms _read_auto
    # filters kwargs down rather than forwarding them all and raising
    # TypeError, since the GUI always sends raw-FID-reprocessing
    # kwargs once it's seen `ser` on disk.
    spectrum = bruker._read_auto(path, window="hann", zero_fill=999, fid_shift=1.0)

    assert spectrum.data.shape == (4, 8)
    assert spectrum.metadata.raw["read_2d_shape_guess_failed"] == "tuple index out of range"


def test_read_auto_reraises_when_no_processed_data_available(tmp_path, monkeypatch):
    path = _make_2d_dataset_with_processed(tmp_path, n_f1=4, n_f2=8)
    os.remove(os.path.join(path, "pdata", "1", "2rr"))

    def fake_read_2d(_path, **_kwargs):
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(bruker, "read_2d", fake_read_2d)

    with pytest.raises(IndexError):
        bruker._read_auto(path)


def test_read_auto_does_not_mask_unrelated_value_errors(tmp_path, monkeypatch):
    # Regression test: _read_auto's fallback used to catch (ValueError,
    # IndexError) -- broad enough to also silently swallow a genuine,
    # unrelated bug inside read_2d's own FID-processing steps (found
    # for real: align_echo's per-row auto-truncation disagreeing,
    # raising a plain ValueError from np.apply_along_axis, fixed
    # directly rather than via this fallback) into a fallback load that
    # discards whatever raw-FID processing was actually requested, with
    # no error shown to the user. Only IndexError -- what a genuine
    # nmrglue shape-guess failure actually produces (its own reshape
    # ValueError is already caught internally by nmrglue) -- should
    # trigger the fallback; any other ValueError must propagate.
    path = _make_2d_dataset_with_processed(tmp_path, n_f1=4, n_f2=8)

    def fake_read_2d(_path, **_kwargs):
        raise ValueError("could not broadcast input array from shape (4090,) into shape (4091,)")

    monkeypatch.setattr(bruker, "read_2d", fake_read_2d)

    with pytest.raises(ValueError):
        bruker._read_auto(path)


def test_read_auto_tags_shape_guess_failure_as_loaded_from_processed(tmp_path, monkeypatch):
    # metadata.raw["loaded_from_processed"] is the more general note
    # (see _read_auto's own docstring) meant to cover every already-
    # processed-data fallback uniformly, regardless of *why* -- confirm
    # this specific, existing failure mode sets it too, alongside the
    # older, narrower read_2d_shape_guess_failed it's always paired
    # with here.
    path = _make_2d_dataset_with_processed(tmp_path, n_f1=4, n_f2=8)

    def fake_read_2d(_path, **_kwargs):
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(bruker, "read_2d", fake_read_2d)

    spectrum = bruker._read_auto(path)

    assert spectrum.metadata.raw["loaded_from_processed"] == "tuple index out of range"


def test_read_auto_tags_loaded_from_processed_when_no_ser_present(tmp_path):
    # Previously silent: a genuine 2D with only pdata (2rr/2ri) on
    # disk, no raw ser at all, always fell back to read_2d_processed
    # with no indication anywhere that raw-FID reprocessing was never
    # even attempted (unlike the shape-guess-failure case above, which
    # at least set read_2d_shape_guess_failed).
    path = _make_2d_dataset_with_processed(tmp_path, n_f1=4, n_f2=8)
    os.remove(os.path.join(path, "ser"))

    spectrum = bruker._read_auto(path)

    assert spectrum.data.shape == (4, 8)
    assert spectrum.metadata.raw["loaded_from_processed"] == "No raw ser file found for this experiment."
    assert "read_2d_shape_guess_failed" not in spectrum.metadata.raw


def test_read_auto_tags_loaded_from_processed_for_1d_processed_only(tmp_path):
    # Same gap as the 2D case above, for the equivalent 1D-only
    # fallback (only 1r/1i on disk, no raw fid at all).
    path = _make_1d_dataset(tmp_path)

    spectrum = bruker._read_auto(path)

    assert spectrum.ndim == 1
    assert spectrum.metadata.raw["loaded_from_processed"] == "No raw fid file found for this experiment."


def test_resolve_magnitude_mode_follows_recorded_ph_mod():
    # Bruker PH_mod: 0 = none, 1 = pk (PHC0/PHC1), 2 = mc (magnitude),
    # 3 = ps (power). Only 2 appears in this project's real test data
    # (9 datasets, mostly gradient COSY); 3 is supported but unverified
    # against a real file.
    assert bruker._resolve_magnitude_mode(None, {"PH_mod": 2}) == "magnitude"
    assert bruker._resolve_magnitude_mode(None, {"PH_mod": 3}) == "power"
    assert bruker._resolve_magnitude_mode(None, {"PH_mod": 1}) is None
    assert bruker._resolve_magnitude_mode(None, {"PH_mod": 0}) is None
    assert bruker._resolve_magnitude_mode(None, {}) is None
    # Either dimension may carry the flag -- the real COSY that
    # motivated this records mc on F1 while F2 says "no".
    assert bruker._resolve_magnitude_mode(None, {"PH_mod": 0}, {"PH_mod": 2}) == "magnitude"
    # A None dict (e.g. a pseudo-2D's excluded proc2s) must not vote.
    assert bruker._resolve_magnitude_mode(None, {"PH_mod": 0}, None) is None


def test_resolve_magnitude_mode_explicit_override_beats_the_file():
    # True forces it on for anything; False forces it off even for a
    # file whose own PH_mod asks for it -- without that, the GUI's
    # "off" would silently flip back on for exactly the datasets the
    # toggle matters most for.
    assert bruker._resolve_magnitude_mode(True, {"PH_mod": 0}) == "magnitude"
    assert bruker._resolve_magnitude_mode(False, {"PH_mod": 2}) is None


def test_resolve_magnitude_mode_tolerates_unparseable_ph_mod():
    # PH_mod arrives from a hand-parsed JCAMP-DX file; a malformed
    # value should fall through to "phase-sensitive", not raise in the
    # middle of a load.
    assert bruker._resolve_magnitude_mode(None, {"PH_mod": "junk"}) is None
    assert bruker._resolve_magnitude_mode(None, {"PH_mod": None}) is None
    assert bruker._resolve_magnitude_mode(None, {"PH_mod": 2.0}) == "magnitude"


def test_apply_magnitude_mode_math():
    spectrum = np.array([3 + 4j, -3 - 4j, 0 + 0j])
    np.testing.assert_allclose(bruker._apply_magnitude_mode(spectrum, "magnitude"), [5.0, 5.0, 0.0])
    np.testing.assert_allclose(bruker._apply_magnitude_mode(spectrum, "power"), [25.0, 25.0, 0.0])
    np.testing.assert_array_equal(bruker._apply_magnitude_mode(spectrum, None), spectrum)


def test_read_bruker_binary_matches_dtype_and_endianness_rules(tmp_path):
    # Same rules nmrglue itself applies (confirmed by reading its
    # source): big-endian only when BYTORDA == 1, float64 only when
    # DTYPA == 2, int32 otherwise.
    values = np.array([1, -2, 3, -4, 5, -6], dtype="<i4")
    path = tmp_path / "fid"
    path.write_bytes(values.tobytes())
    got = bruker._read_bruker_binary(str(path), {"acqus": {"BYTORDA": 0, "DTYPA": 0}})
    np.testing.assert_array_equal(got, np.array([1 - 2j, 3 - 4j, 5 - 6j]))

    big = np.array([1, -2, 3, -4], dtype=">i4")
    path_big = tmp_path / "fid_big"
    path_big.write_bytes(big.tobytes())
    got_big = bruker._read_bruker_binary(str(path_big), {"acqus": {"BYTORDA": 1, "DTYPA": 0}})
    np.testing.assert_array_equal(got_big, np.array([1 - 2j, 3 - 4j]))

    floats = np.array([1.5, -2.5, 3.5, -4.5], dtype="<f8")
    path_f = tmp_path / "fid_float"
    path_f.write_bytes(floats.tobytes())
    got_f = bruker._read_bruker_binary(str(path_f), {"acqus": {"BYTORDA": 0, "DTYPA": 2}})
    np.testing.assert_array_equal(got_f, np.array([1.5 - 2.5j, 3.5 - 4.5j]))


def test_read_bruker_binary_truncates_a_leftover_tail_without_warning(tmp_path, recwarn):
    # The whole point of replacing nmrglue's own reshape here: a file
    # whose length doesn't divide exactly into the requested shape is
    # the *normal* case for real EXSY data, and used to emit
    # "UserWarning: (39583744,) cannot be shaped into (1002, 39488)".
    # Truncating is safe because the shape's element count is derived
    # from this same file's length by construction.
    values = np.arange(2 * (3 * 4) + 6, dtype="<i4")  # 3x4 complex, plus a leftover tail
    path = tmp_path / "ser"
    path.write_bytes(values.tobytes())

    got = bruker._read_bruker_binary(str(path), {"acqus": {"BYTORDA": 0, "DTYPA": 0}}, (3, 4))

    assert got.shape == (3, 4)
    assert not [w for w in recwarn if "cannot be shaped" in str(w.message)]


def test_read_bruker_binary_raises_when_the_file_is_too_short(tmp_path):
    # Genuinely truncated/corrupt data must not come back silently
    # short -- that would produce a plausible-looking but wrong
    # spectrum rather than an error.
    values = np.arange(4, dtype="<i4")  # only 2 complex points
    path = tmp_path / "ser"
    path.write_bytes(values.tobytes())
    with pytest.raises(ValueError, match="fewer than"):
        bruker._read_bruker_binary(str(path), {"acqus": {"BYTORDA": 0, "DTYPA": 0}}, (10, 10))


def test_read_2d_processed_missing_2rr_raises_a_clear_error(tmp_path):
    # Real, pre-existing crash found by sweeping every Bruker dataset in
    # test_data: `Cyclosporine/206/pdata/1` holds procs, proc2s, clevels
    # and a thumbnail but no 2rr. _load_block returns None for a missing
    # file (correct for the optional 2ri/2ii blocks) and that None used
    # to reach NMRSpectrum's constructor, failing with
    # "AttributeError: 'NoneType' object has no attribute 'ndim'" --
    # naming neither the file nor the experiment.
    expt = tmp_path / "expt"
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    for name, params in (
        ("procs", {"SW_p": 8000.0, "SI": 8, "SF": 400.13, "XDIM": 1}),
        ("proc2s", {"SW_p": 2000.0, "SI": 4, "SF": 400.13, "XDIM": 1}),
    ):
        with open(pdata / name, "w") as f:
            f.write("\n".join(f"##${k}= {v}" for k, v in params.items()) + "\n")
    for name in ("acqus", "acqu2s"):
        with open(expt / name, "w") as f:
            f.write("##$NUC1= <1H>\n##$BF1= 400.0\n##$O1= 2000.0\n##$NS= 1\n")

    with pytest.raises(FileNotFoundError, match="2rr"):
        bruker.read_2d_processed(str(expt))


def test_read_1d_processed_missing_1r_raises_a_clear_error(tmp_path):
    # The same guard for the last fallback _read_auto tries -- an
    # experiment with neither raw nor processed data (test_data's
    # `.../2010` is a real example) otherwise failed with a bare errno-2
    # from np.fromfile.
    expt = tmp_path / "expt"
    pdata = expt / "pdata" / "1"
    pdata.mkdir(parents=True)
    with open(pdata / "procs", "w") as f:
        f.write("##$SW_p= 8000.0\n##$SI= 8\n##$SF= 400.13\n")
    with open(expt / "acqus", "w") as f:
        f.write("##$NUC1= <1H>\n##$BF1= 400.0\n##$O1= 2000.0\n##$NS= 1\n")

    with pytest.raises(FileNotFoundError, match="1r"):
        bruker.read_1d_processed(str(expt))


def test_apply_recorded_phase_uses_topspin_index_ramp_convention():
    # phi(i) = PHC0 + PHC1 * i/(n-1), applied as exp(+i*phi) -- measured
    # directly against TopSpin, not assumed. Deliberately *not* the
    # pivot-based model processing.ft.apply_phase uses for a user's
    # interactive phase: routing the recorded constants through that one
    # is what left every 2D badly phased.
    spectrum = np.ones(5, dtype=complex)
    got = bruker.apply_recorded_phase(spectrum, 30.0, 90.0)
    expected = np.exp(1j * np.deg2rad(30.0 + 90.0 * np.arange(5) / 4))
    np.testing.assert_allclose(got, expected)

    # First point carries PHC0 alone; last carries PHC0 + PHC1.
    assert np.angle(got[0], deg=True) == pytest.approx(30.0)
    assert np.angle(got[-1], deg=True) == pytest.approx(120.0)


def test_apply_recorded_phase_axis_and_noop():
    data = np.ones((4, 6), dtype=complex)
    along_f1 = bruker.apply_recorded_phase(data, 0.0, 180.0, axis=0)
    assert np.angle(along_f1[0, 0], deg=True) == pytest.approx(0.0)
    assert np.angle(along_f1[-1, 0], deg=True) == pytest.approx(180.0)
    # constant across the other axis
    assert np.allclose(along_f1[2, :], along_f1[2, 0])
    # Zero phase is an exact no-op, not a multiply by ~1.0
    assert bruker.apply_recorded_phase(data, 0.0, 0.0) is data


@pytest.mark.skipif(
    not os.path.isdir(os.path.join("test_data", "bruker_PFG", "150")),
    reason="needs the real bruker_PFG dataset",
)
def test_apply_recorded_phase_reproduces_topspin_on_real_xf2_data():
    # The measurement this convention was derived from, pinned as a
    # regression test. bruker_PFG/150 is processed by TopSpin with xf2
    # only (proc2s SI == TD, so F1 was never transformed), which makes
    # its 2rr a clean F2-only reference.
    from opennmr.io.jcampdx import parse_jcampdx_params

    path = os.path.join("test_data", "bruker_PFG", "150")
    procs = parse_jcampdx_params(os.path.join(path, "pdata", "1", "procs"))
    reference = np.real(bruker.read_2d_processed(path, pseudo_2d=True).data)
    unphased = bruker.read_2d(path, pseudo_2d=True, autophase=False)
    complex_spectrum = unphased.data + 1j * unphased.data_imag

    phased = np.real(
        bruker.apply_recorded_phase(
            complex_spectrum, float(procs["PHC0"]), float(procs["PHC1"]), axis=-1,
        )
    )

    def _corr(a, b):
        a = np.asarray(a, float).ravel() - np.mean(a)
        b = np.asarray(b, float).ravel() - np.mean(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Essentially exact agreement with TopSpin's own processed data...
    assert _corr(phased, reference) > 0.999
    # ...and read_2d's own default now does exactly this, so a plain
    # load reproduces TopSpin too (it used to sit at ~0.51, routing
    # PHC0 through the pivot-based model and dropping PHC1 entirely).
    assert _corr(bruker.read_2d(path, pseudo_2d=True).data, reference) > 0.999


def test_acquisition_constants_names_parameters_the_way_a_pulse_programme_does():
    # User request: fitting equations should be able to use "values from
    # the input data (e.g. delays and constants and L or whatever)".
    # Matching TopSpin's own naming is the point -- someone writing an
    # equation reaches for d1 and l4 because that is what the pulse
    # programme calls them.
    path = os.path.join("test_data", "Bruker Example Data", "Cyclosporine", "203")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")

    spectrum = bruker.read_2d(path)
    constants = bruker.acquisition_constants(spectrum)

    assert constants["d1"] == pytest.approx(2.0)          # indexed array entries
    assert constants["p1"] == pytest.approx(8.01)
    assert constants["cnst2"] == pytest.approx(145.0)
    assert constants["gpz1"] == pytest.approx(80.0)
    assert constants["ns"] == pytest.approx(8.0)          # plain scalars
    assert constants["td"] == pytest.approx(2048.0)
    # Values as the vendor recorded them, not rescaled: d1 is seconds
    # and p1 microseconds, which is what those numbers mean in TopSpin.
    assert all(isinstance(v, float) for v in constants.values())


def test_acquisition_constants_is_empty_without_acqus():
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    spectrum = NMRSpectrum(
        data=np.zeros(4), axes=[NMRAxis(kind="frequency", ppm=np.arange(4.0))],
        metadata=NMRMetadata(vendor="test", nucleus="1H"),
    )
    assert bruker.acquisition_constants(spectrum) == {}
