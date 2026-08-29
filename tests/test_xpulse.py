import numpy as np

from nmrsuite.io import xpulse

FIXTURE = """\
##TITLE= synthetic test fixture
##$SFO1= 400.130000
##$SF= 400.130000
##$O1= 2400.500000
##.OBSERVENUCLEUS= 1H
##VARFORM= AFFN, AFFN, AFFN
##LAST= 0.000300
$$ SPINFLOW START
$$NP=4
$$Filter=10000
$$TDEFF=3
$$SI=8
$$ SPINFLOW END
##PAGE= N=1
##DATATABLE= (X++(R..R))
0.000000 1.0
0.000100 0.9
0.000200 0.8
0.000300 0.7
##PAGE= N=2
##DATATABLE= (X++(R..R))
0.000000 0.0
0.000100 0.0
0.000200 0.0
0.000300 0.0
##END=
"""


def _write_fixture(tmp_path):
    path = tmp_path / "spectrum.jdx"
    path.write_text(FIXTURE)
    return str(path)


def test_read_xpulse_jcamp_decodes_affn_fid(tmp_path):
    path = _write_fixture(tmp_path)
    fid_data = xpulse._read_xpulse_jcamp(path)
    np.testing.assert_allclose(fid_data["real"], [1.0, 0.9, 0.8, 0.7])
    np.testing.assert_allclose(fid_data["imag"], [0.0, 0.0, 0.0, 0.0])
    assert fid_data["params"]["$SFO1"] == "400.130000"
    assert fid_data["exp_params"]["NP"] == 4.0


def test_read_1d_produces_nmrspectrum(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, autophase=False, baseline_correct=False)

    assert spectrum.ndim == 1
    assert len(spectrum.data) == 4
    assert spectrum.metadata.vendor == "xpulse"
    assert spectrum.metadata.nucleus == "1H"
    assert spectrum.metadata.sfo1 == 400.13
    assert len(spectrum.axes[0].ppm) == 4
    assert len(spectrum.axes[0].hz) == 4
    # NMR convention: frequency axis decreasing left-to-right.
    assert spectrum.axes[0].hz[0] > spectrum.axes[0].hz[-1]


def test_truncate_true_uses_recorded_tdeff_not_literal_true(tmp_path):
    # Regression test: `truncate=True` must read TDEFF (3) from the
    # file, not fall through to isinstance(True, int) and truncate to
    # a single point (bool is an int subclass in Python).
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, truncate=True, autophase=False, baseline_correct=False)
    assert len(spectrum.data) == 3


def test_zero_fill_true_uses_recorded_si(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, zero_fill=True, autophase=False, baseline_correct=False)
    assert len(spectrum.data) == 8


def test_fid_shift_changes_output(tmp_path):
    path = _write_fixture(tmp_path)
    baseline = xpulse.read_1d(path, autophase=False, baseline_correct=False)
    shifted = xpulse.read_1d(path, autophase=False, baseline_correct=False, fid_shift=1.0)
    assert not np.allclose(baseline.data, shifted.data)


def test_align_echo_runs(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, autophase=False, baseline_correct=False, align_echo=True)
    assert len(spectrum.data) == 4


def test_baseline_correct_defaults_off_when_align_echo_is_on(tmp_path, monkeypatch):
    # Ported from leonmr's xpulse_image_import: a straight-line
    # baseline fit is a poor model for an echo-aligned profile, so
    # align_echo=True alone (baseline_correct left unspecified) should
    # skip baseline correction -- confirmed by checking whether
    # ft.baseline_correct actually got called.
    called = []
    from nmrsuite.processing import ft as processing_ft
    monkeypatch.setattr(processing_ft, "baseline_correct", lambda *a, **k: called.append(True) or a[1])

    path = _write_fixture(tmp_path)
    xpulse.read_1d(path, autophase=False, align_echo=True)
    assert called == []

    xpulse.read_1d(path, autophase=False, align_echo=False)
    assert called == [True]

    called.clear()
    xpulse.read_1d(path, autophase=False, align_echo=True, baseline_correct=True)
    assert called == [True]  # explicit override still honoured


def test_linear_predict_extends_fid_before_np_truncation(tmp_path):
    # NP=4 is recorded in the fixture; without re-deriving np_points
    # after linear prediction, the pipeline's own "truncate back to the
    # recorded NP" step would silently discard the 2 predicted points.
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(
        path, autophase=False, baseline_correct=False,
        linear_predict_points=2, linear_predict_order=1,
    )
    assert len(spectrum.data) == 6


def test_detect_rejects_non_jdx_files(tmp_path):
    path = tmp_path / "not_jcamp.txt"
    path.write_text("hello")
    assert xpulse._detect(str(path)) is False


def test_detect_accepts_xpulse_jdx(tmp_path):
    path = _write_fixture(tmp_path)
    assert xpulse._detect(path) is True


# ============================================================
# Arrayed experiments (e.g. PFG): array#/ subfolders as one
# pseudo-2D spectrum
# ============================================================

def _make_array_experiment(tmp_path, n_steps=3, valueslist=None):
    """A parent folder holding ``array1``..``arrayN`` subfolders, each
    with the same synthetic fixture -- mirrors X-Pulse's real PFG
    layout (``arrayN/<name>_avg.jdx`` plus a sibling ``raw/`` folder
    of individual-scan files, which must be ignored)."""
    root = tmp_path / "pfg_experiment"
    for i in range(1, n_steps + 1):
        array_dir = root / f"array{i}"
        array_dir.mkdir(parents=True)
        (array_dir / f"sample_{i}_avg.jdx").write_text(FIXTURE)
        raw_dir = array_dir / "raw"
        raw_dir.mkdir()
        (raw_dir / f"sample_{i}_1_raw.jdx").write_text(FIXTURE)
    if valueslist is not None:
        params_dir = root / "parameters"
        params_dir.mkdir(parents=True)
        (params_dir / "G1_valueslist.txt").write_text("\n".join(str(v) for v in valueslist) + "\n")
    return str(root)


def test_read_array_stacks_steps_as_pseudo_2d(tmp_path):
    path = _make_array_experiment(tmp_path, n_steps=3)
    spectrum = xpulse.read_array(path, autophase=False, baseline_correct=False)
    assert spectrum.ndim == 2
    assert spectrum.is_pseudo_2d is True
    assert spectrum.data.shape[0] == 3
    assert spectrum.metadata.raw["fid"].shape[0] == 3


def test_read_array_reads_gradient_relative_valueslist(tmp_path):
    path = _make_array_experiment(tmp_path, n_steps=3, valueslist=[0.1, 0.5, 0.9])
    spectrum = xpulse.read_array(path, autophase=False, baseline_correct=False)
    np.testing.assert_allclose(spectrum.metadata.raw["gradient_relative"], [0.1, 0.5, 0.9])


def test_read_array_gradient_relative_none_without_valueslist(tmp_path):
    path = _make_array_experiment(tmp_path, n_steps=3)
    spectrum = xpulse.read_array(path, autophase=False, baseline_correct=False)
    assert spectrum.metadata.raw["gradient_relative"] is None


def test_read_array_ignores_raw_subfolder_scans(tmp_path):
    # Each array folder's raw/ subfolder holds individual-scan repeats
    # sharing the same naming pattern -- must not be picked up as
    # "the" step file (or as a second, conflicting one).
    path = _make_array_experiment(tmp_path, n_steps=2)
    spectrum = xpulse.read_array(path, autophase=False, baseline_correct=False)
    assert spectrum.data.shape[0] == 2


def test_read_array_raises_without_array_dirs(tmp_path):
    empty = tmp_path / "not_an_experiment"
    empty.mkdir()
    try:
        xpulse.read_array(str(empty), autophase=False, baseline_correct=False)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_detect_recognizes_array_folder(tmp_path):
    path = _make_array_experiment(tmp_path, n_steps=2)
    assert xpulse._detect(path) is True


def test_detect_rejects_plain_folder_without_array_subdirs(tmp_path):
    plain = tmp_path / "plain_folder"
    plain.mkdir()
    assert xpulse._detect(str(plain)) is False


def test_read_auto_dispatches_array_folder_to_read_array(tmp_path):
    path = _make_array_experiment(tmp_path, n_steps=2)
    spectrum = xpulse._read_auto(path, autophase=False, baseline_correct=False)
    assert spectrum.ndim == 2
    assert spectrum.is_pseudo_2d is True


def test_read_auto_dispatches_single_file_to_read_1d(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse._read_auto(path, autophase=False, baseline_correct=False)
    assert spectrum.ndim == 1
