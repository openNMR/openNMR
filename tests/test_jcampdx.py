from nmrsuite.io.jcampdx import parse_jcampdx_params

ACQUS_FIXTURE = """\
##TITLE= Parameter file, synthetic test fixture
##$NUC1= <1H>
##$BF1= 400.13
##$SFO1= 400.1316005
##$O1= 2400.5
##$NS= 16
##$P= (0..7)
10 5.5 20 0 0 0 0 12.3
##$SW_h= 8012.82
##$GPNAM= (0..3)
<> <SINE.100> <> <DiffBB_OW_2023-10-06_LP30_04G_150>
"""


def test_parses_string_value(tmp_path):
    path = tmp_path / "acqus"
    path.write_text(ACQUS_FIXTURE)
    params = parse_jcampdx_params(str(path))
    assert params["NUC1"] == "1H"


def test_parses_scalar_floats(tmp_path):
    path = tmp_path / "acqus"
    path.write_text(ACQUS_FIXTURE)
    params = parse_jcampdx_params(str(path))
    assert params["BF1"] == 400.13
    assert params["O1"] == 2400.5
    assert params["NS"] == 16.0
    assert params["SW_h"] == 8012.82


def test_parses_multiline_array(tmp_path):
    path = tmp_path / "acqus"
    path.write_text(ACQUS_FIXTURE)
    params = parse_jcampdx_params(str(path))
    assert params["P"] == [10.0, 5.5, 20.0, 0.0, 0.0, 0.0, 0.0, 12.3]


def test_ignores_non_dollar_header_lines(tmp_path):
    path = tmp_path / "acqus"
    path.write_text(ACQUS_FIXTURE)
    params = parse_jcampdx_params(str(path))
    assert "TITLE" not in params


def test_missing_key_absent_not_nan(tmp_path):
    path = tmp_path / "acqus"
    path.write_text(ACQUS_FIXTURE)
    params = parse_jcampdx_params(str(path))
    assert "GRPDLY" not in params


def test_parses_multiline_array_of_bracketed_names(tmp_path):
    # Some Bruker array parameters (GPNAM/SPNAM/CPDPRG) hold <name>
    # strings, including empty <> placeholders, not numbers -- found
    # via a real acqus file that broke a float()-only array parser.
    path = tmp_path / "acqus"
    path.write_text(ACQUS_FIXTURE)
    params = parse_jcampdx_params(str(path))
    assert params["GPNAM"] == ["", "SINE.100", "", "DiffBB_OW_2023-10-06_LP30_04G_150"]
