import glob
import os

import numpy as np
import pytest

from opennmr import io
from opennmr.io import xpulse

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


# Echo top (magnitude 10.0) at index 2, with index 1 and index 3 given
# equal magnitude (2.0) so find_echo_top's parabolic interpolation
# lands exactly on that integer sample (no fractional remainder) --
# keeps the regression test below an exact permutation check rather
# than a fuzzy one across align_echo's FFT-based fractional roll.
ECHO_FIXTURE = """\
##TITLE= synthetic echo test fixture
##$SFO1= 400.130000
##$SF= 400.130000
##$O1= 2400.500000
##.OBSERVENUCLEUS= 1H
##VARFORM= AFFN, AFFN, AFFN
##LAST= 0.000700
$$ SPINFLOW START
$$NP=8
$$Filter=10000
$$TDEFF=4
$$SI=8
$$ SPINFLOW END
##PAGE= N=1
##DATATABLE= (X++(R..R))
0.000000 5.0
0.000100 2.0
0.000200 10.0
0.000300 2.0
0.000400 1.0
0.000500 0.5
0.000600 0.25
0.000700 0.1
##PAGE= N=2
##DATATABLE= (X++(R..R))
0.000000 0.0
0.000100 0.0
0.000200 0.0
0.000300 0.0
0.000400 0.0
0.000500 0.0
0.000600 0.0
0.000700 0.0
##END=
"""


def _write_echo_fixture(tmp_path):
    path = tmp_path / "echo.jdx"
    path.write_text(ECHO_FIXTURE)
    return str(path)


def test_read_1d_records_probe_from_dollar_dollar_probe(tmp_path):
    path = tmp_path / "spectrum_with_probe.jdx"
    path.write_text(FIXTURE.replace("$$Filter=10000", "$$Filter=10000\n$$Probe=HFX"))
    spectrum = xpulse.read_1d(str(path), autophase=False, baseline_correct=False)

    assert spectrum.metadata.probe == "HFX"


def test_read_1d_probe_is_none_when_not_recorded(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, autophase=False, baseline_correct=False)

    assert spectrum.metadata.probe is None


def test_read_1d_records_pulse_program_from_dot_pulsesequence(tmp_path):
    path = tmp_path / "spectrum_with_pulseq.jdx"
    path.write_text(FIXTURE.replace(
        "##.OBSERVENUCLEUS= 1H", "##.OBSERVENUCLEUS= 1H\n##.PULSESEQUENCE= HSQC_HX",
    ))
    spectrum = xpulse.read_1d(str(path), autophase=False, baseline_correct=False)

    assert spectrum.metadata.pulse_program == "HSQC_HX"


def test_read_1d_pulse_program_is_none_when_not_recorded(tmp_path):
    path = _write_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, autophase=False, baseline_correct=False)

    assert spectrum.metadata.pulse_program is None


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
    from opennmr.processing import ft as processing_ft
    monkeypatch.setattr(processing_ft, "baseline_correct", lambda *a, **k: called.append(True) or a[1])

    path = _write_fixture(tmp_path)
    xpulse.read_1d(path, autophase=False, align_echo=True)
    assert called == []

    xpulse.read_1d(path, autophase=False, align_echo=False)
    assert called == [True]

    called.clear()
    xpulse.read_1d(path, autophase=False, align_echo=True, baseline_correct=True)
    assert called == [True]  # explicit override still honoured


def test_align_echo_then_truncate_discards_wrapped_pre_echo_tail(tmp_path):
    # Regression test: align_echo must run BEFORE truncate, not after.
    # align_echo rolls the echo top (here, magnitude 10.0 at index 2)
    # to index 0, which wraps the pre-echo portion (indices 0-1: 5.0,
    # 2.0) around to the tail of the array. Truncating first, against
    # the original unrolled timeline, does nothing to remove that
    # wrapped portion -- get the order backwards and the "aligned" FID
    # (10, 2, [wrapped: 5, 2] instead of the true post-echo decay 1,
    # 0.5) has a spurious bump spliced onto its tail: the "looks like
    # a time-reversal" artifact this ordering fixes.
    path = _write_echo_fixture(tmp_path)
    spectrum = xpulse.read_1d(
        path, autophase=False, baseline_correct=False, align_echo=True, truncate=True,
    )
    fid = spectrum.metadata.raw["fid"]
    assert len(fid) == 4
    np.testing.assert_allclose(np.real(fid), [10.0, 2.0, 1.0, 0.5], atol=1e-9)


def test_align_echo_caps_truncation_at_wrap_point_even_with_no_truncate_requested(tmp_path):
    # Regression test for the real remaining bug: previously, with no
    # `truncate` given at all (the actual default a user hits without
    # manually setting TDeff), align_echo alone did nothing to exclude
    # the wrapped pre-echo portion -- the file's own recorded TDEFF (4,
    # in this fixture) is what excluded it in the sibling test above,
    # not align_echo itself. Same fixture: echo top (10.0) at index 2,
    # wrap point at len(fid) - ceil(top) = 8 - 2 = 6. With truncate
    # left at its default (None/unset), align_echo must still cap the
    # result there on its own, not leave the wrapped [5.0, 2.0] tail in.
    path = _write_echo_fixture(tmp_path)
    spectrum = xpulse.read_1d(path, autophase=False, baseline_correct=False, align_echo=True)
    fid = spectrum.metadata.raw["fid"]
    assert len(fid) == 6
    np.testing.assert_allclose(np.real(fid), [10.0, 2.0, 1.0, 0.5, 0.25, 0.1], atol=1e-9)


def test_align_echo_caps_an_explicit_truncate_past_the_wrap_point(tmp_path):
    # An explicit truncate longer than the wrap point (here, 8 -- the
    # full untruncated length) must still be capped down to the wrap
    # point (6) once align_echo is on, not honoured as-is.
    path = _write_echo_fixture(tmp_path)
    spectrum = xpulse.read_1d(
        path, autophase=False, baseline_correct=False, align_echo=True, truncate=8,
    )
    fid = spectrum.metadata.raw["fid"]
    assert len(fid) == 6


def test_align_echo_does_not_widen_a_shorter_explicit_truncate(tmp_path):
    # An explicit truncate shorter than the wrap point must be
    # honoured exactly -- the cap only ever shortens, never lengthens.
    path = _write_echo_fixture(tmp_path)
    spectrum = xpulse.read_1d(
        path, autophase=False, baseline_correct=False, align_echo=True, truncate=3,
    )
    fid = spectrum.metadata.raw["fid"]
    assert len(fid) == 3
    np.testing.assert_allclose(np.real(fid), [10.0, 2.0, 1.0], atol=1e-9)


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


# ============================================================
# Genuine 2D (NTUPLES-format JCAMP-DX, e.g. HSQC)
# ============================================================

#: The initial F1 evolution time every real X-Pulse NTUPLES export
#: records as T1's own ``##FIRST=``, in seconds.
_T1_FIRST = 6e-06


def _make_ntuples_fixture(tmp_path, n1=8, n2=8, sw1=100.0, sw2=800.0, f1_hz=None, name="hsqc.jdx",
                          record_sw1=True, homonuclear=False, f2_hz=None):
    """A synthetic NTUPLES 2D fixture matching the real format
    confirmed by direct inspection of a real X-Pulse export.

    **The forward model is TPPI, and R/I are one FID.** Each ``##PAGE=
    T1=`` value carries *two* pages -- ``(T2++(R..R))`` and
    ``(T2++(I..I))`` -- and they are the real and imaginary halves of
    the same complex t2 FID, not a States cos/sin pair: both pages of
    a pair record the *same* T1, and the file declares
    ``##.ACQUISITION MODE= SIMULTANEOUS``. F1 sign discrimination
    comes instead from the ``##.ACQUISITION SCHEME= TPPI``
    90-degree-per-increment phase ramp, built in below.

    This fixture used to build States cos/sin pairs, which matched
    what the reader then did rather than what the files contain -- so
    it agreed with the reader while both were wrong, and the mirrored
    spectra a user reported went unnoticed. It now carries a known F2
    frequency as well as a known F1 one, precisely so a reflection in
    either dimension fails a test.

    Plain AFFN decimal text, not ASDF pseudo-digit -- deliberately
    paired with a ``##VARFORM= AFFN, AFFN, AFFN, AFFN`` header claim
    that's technically *correct* here (unlike the real file, which
    lies about this) precisely so the auto-detection path is exercised
    on both "really is AFFN" and (the real file's own) "claims AFFN
    but is ASDF" without needing two separate fixtures."""
    # ``n1`` counts acquired *FIDs*, matching ##VARDIM's own T1 entry
    # in the real files. F1's States quadrature uses two of them per
    # t1 increment, so there are n1 // 2 increments and 2 * n1 pages.
    n_t1 = max(n1 // 2, 1)
    if f1_hz is None:
        f1_hz = sw1 / n_t1        # exactly one F1 bin
    if f2_hz is None:
        f2_hz = 2 * (sw2 / n2)    # likewise an exact bin, well off centre
    t2 = np.arange(n2) / sw2
    decay = np.exp(-t2 * 5.0)
    pages = []
    for k in range(n_t1):
        t1 = k / sw1
        # **States quadrature: a cos- and a sin-modulated FID per t1
        # increment**, so four pages per increment once each is split
        # into its R and I halves. That is what MestReNova does with
        # the real files (its first-FT export halves 256 FIDs to 128
        # t1 points), and it is what gives F1 a single peak at the
        # full sweep width rather than a reflection or a halved
        # window.
        #
        # Earlier versions of this fixture built amplitude modulation
        # with a TPPI phase ramp. That was wrong about the format, and
        # because the reader was wrong the same way, the pair agreed
        # and the tests passed. A fixture cannot be what decides a
        # format question -- this one follows the reference data.
        # States-**TPPI**: successive t1 increments alternate sign (a
        # 180-degree phase alternation, which puts axial peaks at the
        # edge of F1 rather than through the middle of the spectrum).
        # That is the convention the real files use -- decoding them
        # as plain States (fn_mode 4) throws every peak to the axis
        # edges and scores r=-0.04 against MestReNova, against +0.85
        # for this one.
        sign = -1.0 if k % 2 else 1.0
        # (cos, -sin), not (cos, sin): F2 below is built as
        # exp(-2j*pi*f2*t2), so the same coherence evolves at -2*pi*f
        # and its t1 pair must follow. Getting this wrong puts the
        # fixture's own "diagonal" peak at +f in F2 and -f in F1 --
        # off its own diagonal, which no homonuclear dataset can be.
        for quadrature, sense in ((np.cos, 1.0), (np.sin, -1.0)):
            amplitude = sign * sense * quadrature(2 * np.pi * f1_hz * t1)
            complex_row = amplitude * decay * np.exp(-2j * np.pi * f2_hz * t2)
            pages.append(("R", np.real(complex_row), k))
            pages.append(("I", np.imag(complex_row), k))

    t2_last = (n2 - 1) / sw2
    lines = [
        "##TITLE= synthetic NTUPLES fixture",
        "##JCAMPDX= 6.0",
        "##DATACLASS= NTUPLES",
        "##NUMDIM= 2",
        "$$SpinFlow Sequence Parameters Start",
        f"$$NP1= {n1}",
        f"$$NP2= {n2}",
        # ``record_sw1=False`` mirrors a homonuclear sequence such as
        # NOESY_HH, whose ParameterBlock has no SW1 at all -- its F1
        # window is not independently settable. See _f1_sweep_width.
        *([f"$$SW1= {sw1}"] if record_sw1 else []),
        f"$$Filter= {sw2}",
        "$$O1= 0.0",
        "$$NS= 1",
        "$$TxPPM= 5.0",
        # Likewise TxPPMX: it is the X channel's offset, and a
        # homonuclear sequence has no X channel.
        *([] if homonuclear else ["$$TxPPMX= 100.0"]),
        "$$SpinFlow Sequence Parameters End",
        "##.OBSERVENUCLEUS= ^1H",
        "##$INDIRECTNUCLEUS= ^1H" if homonuclear else "##$INDIRECTNUCLEUS= ^13C",
        "##.ACQUISITION SCHEME= TPPI",
        "##$SF= 500.000000 $$ f2",
        f"##$SF= {'500.000000' if homonuclear else '125.000000'} $$ f1",
        "##$SFO1= 500.000000 $$ f2",
        f"##$SFO1= {'500.000000' if homonuclear else '125.000000'} $$ f1",
        "##NTUPLES= nD NMR FID",
        "##VARNAME= TIME1, TIME2, FID/REAL, FID/IMAG",
        "##SYMBOL= T1, T2, R, I",
        "##.NUCLEUS=  ^13C, ^1H",
        "##VARTYPE= INDEPENDENT, INDEPENDENT, DEPENDENT, DEPENDENT",
        "##VARFORM= AFFN, AFFN, AFFN, AFFN",
        f"##VARDIM= {n1}, {n2}, {n2}, {n2}",
        "##UNITS= SECONDS, SECONDS, ARBITRARY UNITS, ARBITRARY UNITS",
        # T1's own FIRST is the initial evolution time, **not** zero --
        # 6e-06 s in every real X-Pulse export seen. Pinned here as a
        # non-zero value on purpose: the zeroed ##FIRST= placeholder on
        # the next line also precedes the first ##PAGE=, so a reader
        # that takes the *last* pre-PAGE declaration silently reads
        # T1's first value as 0 and derives a slightly wrong F1 dwell
        # time. That was a real bug; a zero here would not have caught it.
        f"##FIRST= {_T1_FIRST}, 0, 0, 0",
        f"##LAST= {_T1_FIRST + (n1 - 1) / sw1}, {t2_last}, 0, 0",
        "##FIRST= 0,0,0,0",
    ]
    for kind, row, t1_index in pages:
        lines.append(f"##PAGE=\tT1= {t1_index}")
        lines.append("##FIRST= 0,0,0,0")
        lines.append(f"##DATATABLE= (T2++({kind}..{kind})), PROFILE")
        for i, (t2_val, value) in enumerate(zip(t2, row)):
            lines.append(f"{t2_val:.6f} {value:.6f}")
    lines.append("##END=")

    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path), f1_hz, f2_hz


def test_read_xpulse_ntuples_2d_pairs_the_pages_into_one_complex_fid(tmp_path):
    """``##SYMBOL= T1, T2, R, I``: R and I are the real and imaginary
    halves of **one** t2 FID, not a States cos/sin pair -- both pages
    of a pair carry the same ``##PAGE= T1=``. The reader still hands
    back the pages row-paired; read_2d is what combines them."""
    n1 = 8
    path, *_ = _make_ntuples_fixture(tmp_path, n1=n1)
    fid_data = xpulse._read_xpulse_ntuples_2d(path)
    assert fid_data["complex"].shape[0] == 2 * n1   # still one row per page
    spectrum = xpulse.read_2d(path, autophase=False)
    assert spectrum.metadata.raw["fid_raw"].shape[0] == n1   # one FID per page-pair
    assert np.iscomplexobj(spectrum.metadata.raw["fid_raw"])


def test_read_xpulse_2d_recovers_both_known_frequencies(tmp_path):
    """**The regression for "the spectra are like duplicated."**

    F2 mirrored because each page was transformed as a real FID
    instead of R and I being combined into one complex one. F1
    mirrored because the States pairing of *consecutive* FIDs was
    dropped, leaving its t1 series amplitude-modulated. Both are
    fixed, and neither reflection may come back."""
    path, f1_hz, f2_hz = _make_ntuples_fixture(tmp_path, n1=16, n2=32)
    spectrum = xpulse.read_2d(path, autophase=False)
    magnitude = np.abs(spectrum.data + 1j * spectrum.data_imag)

    f2_trace, f1_trace = magnitude.sum(axis=0), magnitude.sum(axis=1)
    f2_axis, f1_axis = spectrum.axes[1].hz, spectrum.axes[0].hz
    assert f2_axis[np.argmax(f2_trace)] == pytest.approx(f2_hz, abs=1.0)
    assert abs(f1_axis[np.argmax(f1_trace)]) == pytest.approx(abs(f1_hz), abs=1.0)

    def height_at(axis, trace, target):
        return trace[int(np.argmin(np.abs(axis - target)))] / trace.max()

    # F2: no reflection. That is the bug this pins.
    assert height_at(f2_axis, f2_trace, -f2_hz) < 0.05
    # F1: no reflection either. States quadrature gives genuine sign
    # discrimination, so a single peak at the full sweep width -- the
    # two are not a trade, which is what several earlier attempts got
    # wrong in both directions.
    assert height_at(f1_axis, f1_trace, -f1_hz) < 0.2


def test_the_f1_axis_spans_the_full_sweep_width(tmp_path):
    """**No half-window crop.** F1's width is what its own dwell time
    defines, and every transformed point is kept.

    User report: "the F1 dimension isn't matching ... you're ditching
    half of the data and so the spectral length is halved." Halving
    also contradicts the recorded parameters -- ``HSQC_HX`` records
    ``SW1= 4000`` with a measured dwell of exactly 1/4000 s, the
    States convention in which SW *is* the usable width."""
    n1, sw1 = 16, 100.0
    path, *_ = _make_ntuples_fixture(tmp_path, n1=n1, sw1=sw1)
    spectrum = xpulse.read_2d(path, autophase=False)
    # n1 acquired FIDs are n1 // 2 States t1 increments.
    assert spectrum.data.shape[0] == n1 // 2
    span_hz = spectrum.axes[0].hz.max() - spectrum.axes[0].hz.min()
    assert span_hz == pytest.approx(sw1, rel=0.15)


def test_a_homonuclear_f1_spans_the_same_window_as_f2(tmp_path):
    """The directly checkable consequence, and what the user reported
    against: both dimensions are the same nuclei on the same
    transmitter, so their sweep widths must agree. On the real
    ``NOESY_HH`` that is F1 -11.62..21.75 ppm against F2
    -11.73..21.75; cropped to half it was -3.24..13.37."""
    path, *_ = _make_ntuples_fixture(tmp_path, n1=16, n2=32, sw1=800.0, sw2=800.0,
                                     homonuclear=True, record_sw1=False)
    spectrum = xpulse.read_2d(path, autophase=False)
    f1, f2 = spectrum.axes[0].ppm, spectrum.axes[1].ppm
    # The *sweep width*, not the first-to-last span: those differ by
    # one bin, and F1 has far fewer bins than F2, so comparing spans
    # directly fails on the point count rather than on the width.
    def sweep(axis):
        return (axis.max() - axis.min()) * len(axis) / (len(axis) - 1)
    assert sweep(f1) == pytest.approx(sweep(f2), rel=0.02)


def test_detect_varform_ignores_unreliable_header_claim():
    # The real file's own ##VARFORM= claims AFFN for genuinely
    # ASDF-encoded data -- confirm detection goes by content.
    assert xpulse._detect_xpulse_datatable_varform(["0.5J23K11"]) == "ASDF"
    assert xpulse._detect_xpulse_datatable_varform(["0.5 1.23 -4.5e-2"]) == "AFFN"


def test_is_ntuples_2d_file_true_for_ntuples_false_for_plain_1d(tmp_path):
    ntuples_path, *_ = _make_ntuples_fixture(tmp_path)
    plain_path = _write_fixture(tmp_path)
    assert xpulse._is_ntuples_2d_file(ntuples_path) is True
    assert xpulse._is_ntuples_2d_file(plain_path) is False


def test_is_ntuples_2d_file_false_for_a_1d_file_that_still_declares_ntuples(tmp_path):
    # Real bug, found on a real file: a plain 1D X-Pulse FID export
    # (test_data/xpulse_MRI/14/*_avg.jdx) still declares
    # "##DATACLASS= NTUPLES" -- every X-Pulse export uses the NTUPLES
    # data-table format for its FID regardless of dimensionality, only
    # differing in "##NUMDIM=". The old detector checked for
    # "DATACLASS"/"NTUPLES" as bare substrings anywhere in the header,
    # so it misdetected this file as 2D and routed it into read_2d,
    # which then failed on this file's non-existent F1 spectral width.
    # _write_fixture's own "false" case above never exercised this --
    # its FIXTURE constant predates the NTUPLES format entirely, so it
    # never contained those substrings at all, discriminating nothing.
    path = tmp_path / "one_d_but_ntuples.jdx"
    path.write_text(
        "##TITLE= real-shaped 1D NTUPLES fixture\n"
        "##DATACLASS= NTUPLES\n"
        "##NUMDIM= 1\n"
        "##NTUPLES= NMRFID\n"
        "##VARNAME= TIME, FID/RE\n"
    )
    assert xpulse._is_ntuples_2d_file(str(path)) is False


def test_read_auto_dispatches_ntuples_file_to_read_2d(tmp_path):
    path, *_ = _make_ntuples_fixture(tmp_path)
    spectrum = xpulse._read_auto(path, autophase=False)
    assert spectrum.ndim == 2
    assert spectrum.metadata.nucleus_indirect == "13C"


# ============================================================
# Recorded (autophase) vs user phase
# ============================================================

def _phase_state(raw):
    return (
        float(raw.get("recorded_phase0", 0.0)), float(raw.get("recorded_phase1", 0.0)),
        float(raw.get("phase0", 0.0)), float(raw.get("phase1", 0.0)),
    )


def test_autophase_result_is_recorded_not_a_user_phase():
    # The bug this fixes: an autophase result stored in the *user* slot
    # is overwritten by a saved sidecar's own phase0 on load, and the
    # rebuild-from-raw then applies the user's correction with no base
    # under it -- leaving the spectrum essentially unphased.
    import os

    path = "test_data/xpulse_PFG/4/array1"
    if not os.path.isdir(path):
        pytest.skip(f"{path} not available")
    spectrum = xpulse.read_array("test_data/xpulse_PFG/4")
    recorded0, _r1, user0, user1 = _phase_state(spectrum.metadata.raw)
    assert recorded0 != 0.0          # autophase found something
    assert user0 == 0.0 and user1 == 0.0   # and it is not a user phase


def test_a_supplied_phase_is_a_user_phase_not_a_recorded_one():
    # The mirror case: a caller-supplied phase already *is* the user's,
    # and recording it as a base too would double-count it on rebuild.
    import os

    if not os.path.isdir("test_data/xpulse_PFG/4"):
        pytest.skip("test data not available")
    spectrum = xpulse.read_array("test_data/xpulse_PFG/4", autophase=False,
                                  phase0=31.0, phase1=2.0)
    recorded0, recorded1, user0, user1 = _phase_state(spectrum.metadata.raw)
    assert (recorded0, recorded1) == (0.0, 0.0)
    assert (user0, user1) == (31.0, 2.0)


def test_recorded_and_user_phase_use_the_shared_key_names():
    # Keyed through io.bruker's own accessors so the two vendors cannot
    # drift apart about which key means what.
    from opennmr.io import bruker

    # Converted into the recorded slot's own model, not copied into it:
    # that slot means TopSpin's index ramp (exp(+i*phi)) and autophase
    # returns the pivot model (exp(-i*phi)), so a verbatim copy applies
    # the correction backwards.
    applied = xpulse._phase_metadata(12.0, 3.0, autophased=True)
    assert applied[bruker.RECORDED_PHASE_KEYS["f2"][0]] == pytest.approx(-13.5)
    assert applied[bruker.USER_PHASE_KEYS["f2"][0]] == 0.0
    assert bruker.recorded_phase(applied, "f2") == pytest.approx((-13.5, 3.0))

    supplied = xpulse._phase_metadata(12.0, 3.0, autophased=False)
    assert bruker.recorded_phase(supplied, "f2") == (0.0, 0.0)
    assert supplied[bruker.USER_PHASE_KEYS["f2"][0]] == 12.0


# ============================================================
# Acquisition parameters, for fitting equations and factors
# ============================================================

def _bare_spectrum(raw):
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    return NMRSpectrum(
        data=np.zeros(4), axes=[NMRAxis(kind="frequency", ppm=np.arange(4.0))],
        metadata=NMRMetadata(vendor="xpulse", nucleus="1H", raw=raw),
    )


def test_acquisition_constants_reads_xpulse_exp_params():
    # The user's own case needs TAU by name, from an X-Pulse file --
    # acquisition_constants used to read Bruker's acqus only.
    from opennmr.core.acquisition import acquisition_constants

    constants = acquisition_constants(_bare_spectrum(
        {"exp_params": {"TAU": 1500.0, "P90": 8.78, "NS": 16.0, "Probe": "HFX"}}))
    assert constants["TAU"] == pytest.approx(1500.0)
    assert constants["P90"] == pytest.approx(8.78)
    # A lowercase alias too, since "tau" is what anyone types.
    assert constants["tau"] == pytest.approx(1500.0)
    # Non-numeric parameters are not quantities and are left out.
    assert "Probe" not in constants


def test_acquisition_constants_keeps_case_when_xpulse_needs_it():
    # A real PFG file records both "delta" (gradient duration) and
    # "DELTA" (diffusion delay) -- different numbers. Lowercasing
    # everything the Bruker branch's way would resolve one to the
    # other with nothing on screen to show it.
    from opennmr.core.acquisition import acquisition_constants

    constants = acquisition_constants(_bare_spectrum(
        {"exp_params": {"delta": 4000.0, "DELTA": 50000.0}}))
    assert constants["delta"] == pytest.approx(4000.0)
    assert constants["DELTA"] == pytest.approx(50000.0)


def test_acquisition_constants_reads_the_jcamp_header_too():
    from opennmr.core.acquisition import acquisition_constants

    constants = acquisition_constants(_bare_spectrum(
        {"params": {"$NS": 16.0, "$P1": 8.0, ".PULSESEQUENCE": "cpmg"}}))
    assert constants["NS"] == pytest.approx(16.0)
    assert constants["P1"] == pytest.approx(8.0)
    assert ".PULSESEQUENCE" not in constants


def test_acquisition_constants_on_a_real_xpulse_pfg_dataset():
    from opennmr.core.acquisition import acquisition_constants

    path = os.path.join("test_data", "xpulse_PFG", "4")
    if not os.path.isdir(path):
        pytest.skip(f"needs the real dataset at {path}")
    constants = acquisition_constants(xpulse.read_array(path))
    assert constants["delta"] == pytest.approx(4000.0)
    assert constants["DELTA"] == pytest.approx(50000.0)
    assert constants["P90"] == pytest.approx(8.78)


# ============================================================
# peek(): the JCAMP header only, never the data table
# ============================================================

def test_peek_stops_at_the_data_table():
    """X-Pulse is the one vendor here whose full read is dominated by
    *parsing* rather than processing -- 1063 ms of an 1142 ms read_2d,
    essentially all of it ASDF-decoding points the browser never looks
    at. The header scan must therefore stop, and stop in the right
    place: a long ``##TITLE=`` wraps onto a bare continuation line, so
    "the first line that isn't ##" would end the scan on line two."""
    path = "test_data/xpulse_HSQC_NUS/2/Quinine_HSQC_HX_avg.jdx"
    if not os.path.exists(path):
        pytest.skip("test_data/xpulse_HSQC_NUS not present")

    header = xpulse._peek_header(path)
    # Read past the wrapped title, all the way to the per-dimension
    # parameters near the end of the header...
    assert header["NUMDIM"] == ["2"]
    assert len(header["$TD"]) == 2
    # ...and stopped: a decoded data table would have been thousands
    # of lines more, and the datatable key is the last thing seen.
    assert "DATATABLE" in header


def test_peek_header_tolerates_a_file_that_is_all_header(tmp_path):
    """No data table at all must not run off the end or raise."""
    path = tmp_path / "header_only.jdx"
    path.write_text("##TITLE= x\nwrapped continuation\n##NUMDIM= 1\n##$TD= 64\n##END=\n")
    header = xpulse._peek_header(str(path))
    assert header["$TD"] == ["64"]
    # _peek_td halves: ##$TD counts real+imaginary points, so a TD of
    # 64 is 32 acquired points. (The preferred source is ##VARDIM,
    # which states the 32 outright -- absent here on purpose, so this
    # exercises the fallback.)
    assert xpulse._peek_td(header) == 32
    assert xpulse._peek_direct_points(header) == 32


def test_peek_td_is_zero_when_the_header_does_not_say():
    assert xpulse._peek_td({}) == 0
    assert xpulse._peek_td({"$TD": ["not a number"]}) == 0
    assert xpulse._peek_td({"$TD": ["64"]}, index=1) == 0


@pytest.mark.parametrize("path", [
    "test_data/xpulse_HSQC_NUS/2/Quinine_HSQC_HX_avg.jdx",   # NTUPLES 2D
    "test_data/xpulse_PFG/4",                                 # arrayed folder
])
def test_peek_agrees_with_a_real_load(path):
    """``##$TD`` counts real+imaginary points, so an acquired dimension
    is ``TD // 2``; the indirect dimension is F1's own ``TD`` for a 2D
    and the ``arrayN`` count for an arrayed folder. This pins that
    derivation against what a real load produces."""
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    info = xpulse.peek(path)
    spectrum = io.load(path, vendor="xpulse")
    assert tuple(info["shape"]) == tuple(spectrum.data.shape)
    assert info["ndim"] == spectrum.data.ndim
    assert info["nucleus"] == spectrum.metadata.nucleus


def test_peek_mirrors_read_auto_three_way_dispatch():
    """An arrayed folder is a pseudo-2D, a NUMDIM=2 file is a genuine
    2D, anything else is 1D -- the same three cases _read_auto splits
    on, so what the browser says matches what opening it gives."""
    cases = [
        ("test_data/xpulse_PFG/4", 2, True),
        ("test_data/xpulse_HSQC_NUS/2/Quinine_HSQC_HX_avg.jdx", 2, False),
        ("test_data/xpulse_MRI/14", 1, False),
    ]
    for path, ndim, pseudo in cases:
        target = path
        if os.path.isdir(path) and not xpulse._find_array_dirs(path):
            found = glob.glob(os.path.join(path, "*.jdx"))
            if not found:
                continue
            target = found[0]
        if not os.path.exists(target):
            continue
        info = xpulse.peek(target)
        assert info["ndim"] == ndim, target
        assert info["pseudo_2d"] is pseudo, target


# ============================================================
# Which gradient coil the experiment used
# ============================================================

def test_gradient_axis_is_read_from_the_spinflow_parameters():
    """**A triple-axis probe's three coils are not interchangeable.**
    Nothing read ``$$GradAxis``, so a PGSTE run on y fell back to the
    axis combo's default (z) and every D came out scaled by the ratio
    of the two coils' maxima. Reported directly: "Make sure grad_axis
    is being read from the parameters in the case of x-pulse for curve
    fitting"."""
    path = "test_data/xpulse_PFG/4"
    if not os.path.isdir(path):
        pytest.skip("test_data/xpulse_PFG not present")
    spectrum = xpulse.read_array(path)
    # The file's own header says "PGSTE 1H y gradient".
    assert spectrum.metadata.raw["gradient_axis"] == "y"


def test_gradient_axis_normalises_and_rejects_what_it_does_not_know():
    assert xpulse._gradient_axis_from_params({"GradAxis": "Y"}) == "y"
    assert xpulse._gradient_axis_from_params({"GradAxis": " z "}) == "z"
    # An unrecognised or missing value leaves the GUI on the user's own
    # selection -- the behaviour before this existed.
    assert xpulse._gradient_axis_from_params({"GradAxis": "diagonal"}) is None
    assert xpulse._gradient_axis_from_params({}) is None
    assert xpulse._gradient_axis_from_params(None) is None


# ============================================================
# F1 spectral width: recorded where a sequence records one,
# derived from the file's own T1 axis where none does
# ============================================================

def test_f1_sweep_width_prefers_the_recorded_sw1():
    """A sequence that states its own F1 window is believed. Deriving
    over the top of it would paper over a real mismatch rather than
    show it -- the derivation exists for the files that have no SW1,
    not to second-guess the ones that do."""
    # A T1 axis that would derive 1000 Hz, and a recorded 4000.
    assert xpulse._f1_sweep_width({"SW1": 4000.0}, 0.0, 0.127, 128) == 4000.0


def test_f1_sweep_width_derives_from_the_t1_axis_when_sw1_is_absent():
    """``NOESY_HH`` records no SW1 -- a homonuclear F1 window is not
    independently settable -- and reading one used to die on "Spectral
    width (F1) must be > 0 to compute dt" before producing anything,
    although the file states its F1 dwell time perfectly well in its
    own ##FIRST=/##LAST= T1 axis."""
    # 128 pairs from 6e-06 to 0.031756 s -> 2.5e-04 s -> 4000 Hz.
    assert xpulse._f1_sweep_width({}, 6e-06, 0.031756, 128) == pytest.approx(4000.0)
    # The initial evolution time is not ignorable: dropping it shifts
    # the answer, which is why ##FIRST= has to be the real declaration
    # and not the zeroed placeholder beside it.
    assert xpulse._f1_sweep_width({}, 0.0, 0.031756, 128) != pytest.approx(4000.0)


def test_f1_sweep_width_reports_nothing_rather_than_guessing():
    """No SW1 and no usable T1 axis is a genuine "can't tell" -- the
    caller raises with a message naming both, instead of a plausible
    number nobody can trace."""
    assert xpulse._f1_sweep_width({}, None, None, 128) == 0.0
    assert xpulse._f1_sweep_width({}, 0.0, 0.0, 128) == 0.0
    assert xpulse._f1_sweep_width({}, 0.0, 0.1, 1) == 0.0       # a single increment
    assert xpulse._f1_sweep_width({"SW1": "n/a"}, None, None, 8) == 0.0


@pytest.mark.skipif(not os.path.isfile("test_data/xpulse_HSQC_NUS/2/Quinine_HSQC_HX_avg.jdx"),
                    reason="test_data/xpulse_HSQC_NUS not present")
def test_the_derived_f1_width_matches_the_recorded_one_on_real_data():
    """**The check that makes the derivation trustworthy.** Quinine
    ``HSQC_HX`` records ``SW1= 4000.0`` *and* a T1 axis, so the two can
    be compared on a real file rather than a fixture built to agree.
    6e-06 -> 0.031756 s over 128 pairs is 2.5e-04 s exactly, i.e.
    4000.0000 Hz -- so the rule used on files with no SW1 reproduces
    the recorded value on the one file that has both."""
    path = "test_data/xpulse_HSQC_NUS/2/Quinine_HSQC_HX_avg.jdx"
    fid_data = xpulse._read_xpulse_ntuples_2d(path)
    n1 = fid_data["complex"].shape[0] // 2
    derived = xpulse._f1_sweep_width({}, fid_data["t1_first"], fid_data["t1_last"], n1)
    assert derived == pytest.approx(float(fid_data["exp_params"]["SW1"]))


@pytest.mark.skipif(not os.path.isfile("test_data/xpulse_HSQC_NUS/2/Quinine_HSQC_HX_avg.jdx"),
                    reason="test_data/xpulse_HSQC_NUS not present")
def test_global_first_is_the_declaration_not_the_zeroed_placeholder():
    """``##FIRST=`` appears twice before the first ``##PAGE=``: the real
    declaration, then the all-zero per-page placeholder. First wins."""
    fid_data = xpulse._read_xpulse_ntuples_2d(
        "test_data/xpulse_HSQC_NUS/2/Quinine_HSQC_HX_avg.jdx")
    assert fid_data["t1_first"] == pytest.approx(6e-06)
    assert fid_data["t1_last"] == pytest.approx(0.031756)


def test_ntuples_2d_reads_a_sequence_that_records_no_sw1(tmp_path):
    """End to end: the same fixture, minus its ``$$SW1=`` line, still
    loads and still gets the same F1 dwell time."""
    n1, sw1 = 8, 100.0
    with_sw1, *_ = _make_ntuples_fixture(tmp_path, n1=n1, sw1=sw1, name="with_sw1.jdx")
    without, *_ = _make_ntuples_fixture(tmp_path, n1=n1, sw1=sw1, name="no_sw1.jdx",
                                       record_sw1=False)

    recorded = xpulse.read_2d(with_sw1, autophase=False)
    derived = xpulse.read_2d(without, autophase=False)
    assert "SW1" not in derived.metadata.raw["exp_params"]
    assert derived.metadata.raw["fid_dt_f1"] == pytest.approx(1.0 / sw1)
    assert derived.metadata.raw["fid_dt_f1"] == pytest.approx(recorded.metadata.raw["fid_dt_f1"])
    np.testing.assert_allclose(derived.axes[0].ppm, recorded.axes[0].ppm)


def test_a_homonuclear_f1_takes_the_observe_channels_offset(tmp_path):
    """``TxPPMX`` is the *X channel's* offset, so a homonuclear
    sequence has none. Defaulting to 0.0 there would centre F1 at
    0 ppm while F2 sat at ``TxPPM`` (5.0), putting a NOESY's diagonal
    5 ppm off the diagonal."""
    path, *_ = _make_ntuples_fixture(tmp_path, homonuclear=True, record_sw1=False)
    spectrum = xpulse.read_2d(path, autophase=False)
    assert spectrum.metadata.nucleus == spectrum.metadata.nucleus_indirect == "1H"
    centre_f1 = 0.5 * (spectrum.axes[0].ppm.min() + spectrum.axes[0].ppm.max())
    centre_f2 = 0.5 * (spectrum.axes[1].ppm.min() + spectrum.axes[1].ppm.max())
    assert centre_f1 == pytest.approx(5.0, abs=1.0)
    assert centre_f1 == pytest.approx(centre_f2, abs=1.0)


def test_a_heteronuclear_f1_does_not_borrow_the_proton_offset(tmp_path):
    """The fallback is a deduction about one channel being two
    dimensions, not a general "use TxPPM if TxPPMX is missing" -- that
    would silently put a 13C axis on the 1H carrier."""
    path, *_ = _make_ntuples_fixture(tmp_path, homonuclear=False)
    spectrum = xpulse.read_2d(path, autophase=False)
    assert spectrum.metadata.nucleus_indirect == "13C"
    centre_f1 = 0.5 * (spectrum.axes[0].ppm.min() + spectrum.axes[0].ppm.max())
    assert centre_f1 == pytest.approx(100.0, abs=1.0)   # TxPPMX, not TxPPM


def test_no_f1_width_anywhere_raises_and_says_where_it_looked(tmp_path):
    """The error a user actually sees should name both places a width
    could have come from, not just the parameter that was missing."""
    import re
    path, *_ = _make_ntuples_fixture(tmp_path, record_sw1=False)
    # Collapse the T1 axis so nothing can be derived from it either.
    text = re.sub(r"##LAST= [^,]+,", "##LAST= 6e-06,", open(path).read(), count=1)
    assert "##LAST= 6e-06," in text
    (tmp_path / "flat.jdx").write_text(text)
    with pytest.raises(ValueError, match=r"(?s)F1.*SW1.*FIRST"):
        xpulse.read_2d(str(tmp_path / "flat.jdx"), autophase=False)


@pytest.mark.skipif(not os.path.isfile("test_data/xpulse_HSQC_NUS/2/Quinine_HSQC_HX_avg.jdx"),
                    reason="test_data/xpulse_HSQC_NUS not present")
def test_a_real_2d_has_no_reflection_about_the_transmitter():
    """The same regression on real data. Before the R/I pages were
    combined, Quinine's F2 projection carried an equal-height pair at
    4.05 and 5.95 ppm about its 5.0 ppm transmitter; 4.05 is its OCH3
    (3.95 ppm in the literature) and 5.95 was the reflection."""
    spectrum = xpulse.read_2d("test_data/xpulse_HSQC_NUS/2/Quinine_HSQC_HX_avg.jdx")
    ppm = spectrum.axes[1].ppm
    projection = np.abs(spectrum.data).sum(axis=0)
    carrier = float(spectrum.metadata.raw["exp_params"]["TxPPM"])

    peak_ppm = ppm[int(np.argmax(projection))]
    assert peak_ppm == pytest.approx(4.05, abs=0.3)
    # Its reflection about the transmitter must not also be a peak.
    mirror = carrier + (carrier - peak_ppm)
    height = projection[int(np.argmin(np.abs(ppm - mirror)))] / projection.max()
    assert height < 0.5, f"a reflection survives at {mirror:.2f} ppm"


def test_an_f1_window_is_applied_and_suppresses_truncation_ringing(tmp_path):
    """**F1 had no apodization at all.** An interferogram cut off at
    its last increment rings as sinc wiggles down F1, and F1 is where
    that hurts most -- a couple of hundred increments against a couple
    of thousand F2 points. User report: "I don't think the windowing is
    being applied to the indirect dimension"."""
    # **Half-way between two F1 bins on purpose.** The default fixture
    # frequency is an exact bin, where a truncated transform leaks
    # essentially nothing (measured: 3e-07) and there is no ringing for
    # a window to suppress -- the degenerate case, not the real one. A
    # half-bin offset is where truncation leakage is largest.
    # Enough F1 increments that the measurement means something: a
    # sine-bell over a handful of points broadens more than it
    # suppresses, and the off-peak window below would have almost
    # nothing left in it.
    n1, sw1 = 64, 100.0
    path, *_ = _make_ntuples_fixture(tmp_path, n1=n1, n2=32, sw1=sw1,
                                     f1_hz=1.5 * (sw1 / (n1 // 2)))
    plain = xpulse.read_2d(path, autophase=False)
    windowed = xpulse.read_2d(path, autophase=False,
                              window_f1="sine", window_kwargs_f1={"ssb": 2.0})

    assert plain.data.shape == windowed.data.shape
    assert not np.allclose(plain.data, windowed.data)  # it actually did something

    def ringing(spectrum):
        trace = np.abs(spectrum.data).sum(axis=1)
        peak = int(np.argmax(trace))
        # Well clear of the line itself: a window broadens the peak, so
        # the neighbouring bins legitimately gain intensity. What must
        # fall is the leakage further out.
        off = np.concatenate([trace[:max(peak - 5, 0)], trace[peak + 6:]])
        return off.mean() / trace[peak]

    assert ringing(windowed) < ringing(plain)


def test_no_f1_window_by_default_so_an_existing_load_is_unchanged(tmp_path):
    """Opting in is deliberate: the default stays "no F1 window", so
    adding the control cannot silently change what any dataset already
    on disk looks like when it is reopened."""
    path, *_ = _make_ntuples_fixture(tmp_path)
    default = xpulse.read_2d(path, autophase=False)
    explicit_none = xpulse.read_2d(path, autophase=False, window_f1=None)
    np.testing.assert_array_equal(default.data, explicit_none.data)


#: The NOESY that settled F1's encoding, together with MestReNova's
#: own export of it at four stages. In the repository rather than
#: referenced on a personal drive: this is the only thing pinning the
#: States-TPPI decode, and it took a reference dataset to establish
#: after three rounds of inferring it wrongly from the raw file alone.
_MNOVA_NOESY = os.path.join("test_data", "xpulse_NOESY", "4")


def _read_mnova_matrix(path):
    """A MestReNova CSV export: an F2 ppm header row, then one row per
    F1 point prefixed by its own ppm."""
    with open(path) as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    f2 = np.array([float(v) for v in lines[0].split("\t") if v.strip()])
    f1, rows = [], []
    for line in lines[1:]:
        parts = [p for p in line.split("\t") if p.strip()]
        f1.append(float(parts[0]))
        rows.append([float(p) for p in parts[1:]])
    return np.array(f1), f2, np.array(rows)


@pytest.mark.skipif(not os.path.isfile(os.path.join(_MNOVA_NOESY, "Mnova_exp.csv")),
                    reason="test_data/xpulse_NOESY not present")
def test_the_noesy_matches_mestrenovas_own_processing():
    """**The reference that settled F1**, after three rounds of
    getting it wrong from the data alone.

    MestReNova's export of this same file is the ground truth: its
    ``Processed_FID_1st_FT`` carries 128 non-zero rows of 256, i.e. it
    halves the acquired FIDs into States pairs exactly as this reader
    now does, and its finished F1 spans the full 2000 Hz with a single
    peak at 4.74 ppm rather than a reflected pair or a halved window.
    """
    m_f1, m_f2, reference = _read_mnova_matrix(f"{_MNOVA_NOESY}/Mnova_exp.csv")
    spectrum = xpulse.read_2d(f"{_MNOVA_NOESY}/SRB_AKS_B014_4_NOESY_HH_avg.jdx",
                              zero_fill_f1=len(m_f1))
    f1, f2 = spectrum.axes[0].ppm, spectrum.axes[1].ppm
    magnitude = np.abs(spectrum.data + 1j * spectrum.data_imag)

    # Same window, to within a bin.
    assert f1.min() == pytest.approx(m_f1.min(), abs=0.2)
    assert f1.max() == pytest.approx(m_f1.max(), abs=0.2)

    mine = magnitude.sum(axis=1)
    theirs = np.abs(reference).sum(axis=1)
    assert f1[int(np.argmax(mine))] == pytest.approx(m_f1[int(np.argmax(theirs))], abs=0.2)

    # And the whole F1 profile, not just its maximum -- a reflection
    # or a half-window would still put *a* peak near the right place.
    resampled = np.interp(m_f1, f1[::-1], (mine / mine.max())[::-1])
    assert np.corrcoef(resampled, theirs / theirs.max())[0, 1] > 0.7

    # The solvent self-correlation lands on the diagonal.
    i1, i2 = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    assert f1[i1] == pytest.approx(f2[i2], abs=0.3)


def test_missing_o1_is_zero_not_one_hertz(tmp_path):
    """No ``$O1`` in the file means no offset. The reader used to fall back
    to 1 Hz, which shifted every axis by 1 Hz (explicit user report)."""
    without = tmp_path / "no_o1.jdx"
    without.write_text(FIXTURE.replace("##$O1= 2400.500000\n", ""))
    zero = tmp_path / "zero_o1.jdx"
    zero.write_text(FIXTURE.replace("##$O1= 2400.500000", "##$O1= 0.0"))
    a = xpulse.read_1d(str(without), autophase=False, baseline_correct=False)
    b = xpulse.read_1d(str(zero), autophase=False, baseline_correct=False)
    assert a.metadata.o1 == 0.0
    np.testing.assert_allclose(a.axes[0].hz, b.axes[0].hz, atol=1e-9)
    np.testing.assert_allclose(a.axes[0].ppm, b.axes[0].ppm, atol=1e-12)
