import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.io import persistence


def _make_spectrum(source_path, **meta_kwargs):
    axis = NMRAxis(kind="frequency", ppm=np.linspace(-1, 1, 8), hz=np.linspace(-400, 400, 8))
    metadata = NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0, source_path=source_path, **meta_kwargs)
    return NMRSpectrum(data=np.zeros(8), axes=[axis], metadata=metadata)


# ============================================================
# Sidecar path naming: directory- vs file-scoped source_path
# ============================================================

def test_sidecar_paths_live_inside_a_directory_source_path(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    assert persistence.metadata_sidecar_path(str(expt)) == str(expt / "opennmr_metadata.json")
    assert persistence.processing_sidecar_path(str(expt)) == str(expt / "opennmr_processing.json")


def test_sidecar_paths_are_a_sibling_file_for_a_file_source_path(tmp_path):
    # JEOL (and a bare-file X-Pulse dataset): several sibling
    # experiment files can share one directory with no per-experiment
    # subfolder -- a directory-scoped name would collide across them,
    # so the basename must be part of the sidecar's own name.
    expt = tmp_path / "carbon.jdf"
    expt.write_bytes(b"\x00")
    assert persistence.metadata_sidecar_path(str(expt)) == str(tmp_path / "carbon.jdf.opennmr_metadata.json")
    assert persistence.processing_sidecar_path(str(expt)) == str(tmp_path / "carbon.jdf.opennmr_processing.json")

    sibling = tmp_path / "proton.jdf"
    sibling.write_bytes(b"\x00")
    assert persistence.metadata_sidecar_path(str(sibling)) != persistence.metadata_sidecar_path(str(expt))


# ============================================================
# Metadata sidecar (data/identity facts)
# ============================================================

def test_save_and_load_metadata_sidecar_round_trips(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    spectrum = _make_spectrum(str(expt), nucleus_indirect="13C", procno=1, mass=0.045, probe="1.3 mm MAS BB")

    persistence.save_metadata_sidecar(spectrum)
    loaded = persistence.load_metadata_sidecar(str(expt))

    assert loaded["vendor"] == "bruker"
    assert loaded["nucleus"] == "1H"
    assert loaded["nucleus_indirect"] == "13C"
    assert loaded["probe"] == "1.3 mm MAS BB"
    assert loaded["sfo1_mhz"] == pytest.approx(400.0)
    assert loaded["procno"] == 1
    assert loaded["mass_g"] == pytest.approx(0.045)


def test_metadata_sidecar_is_a_real_json_file_at_the_expected_name(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    persistence.save_metadata_sidecar(_make_spectrum(str(expt)))
    assert (expt / "opennmr_metadata.json").exists()
    assert not (expt / "opennmr_metadata.json.tmp").exists()  # atomic write leaves no temp file behind


def test_metadata_sidecar_records_pseudo_2d_only_for_2d_spectra(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    axis2 = NMRAxis(kind="index", index=np.arange(1, 4))
    axis1 = NMRAxis(kind="frequency", ppm=np.linspace(-1, 1, 8), hz=np.linspace(-400, 400, 8))
    metadata = NMRMetadata(vendor="bruker", source_path=str(expt), pseudo_2d=True)
    spectrum2d = NMRSpectrum(data=np.zeros((3, 8)), axes=[axis2, axis1], metadata=metadata)

    persistence.save_metadata_sidecar(spectrum2d)
    assert persistence.load_metadata_sidecar(str(expt))["pseudo_2d"] is True

    # A 1D spectrum has no pseudo_2d concept at all -- must not record
    # a stale/meaningless True default from the dataclass's own field.
    spectrum1d = _make_spectrum(str(expt))
    persistence.save_metadata_sidecar(spectrum1d)
    assert persistence.load_metadata_sidecar(str(expt))["pseudo_2d"] is None


def test_load_metadata_sidecar_returns_none_when_missing(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    assert persistence.load_metadata_sidecar(str(expt)) is None


def test_load_metadata_sidecar_returns_none_for_corrupted_file(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    (expt / "opennmr_metadata.json").write_text("{not valid json")
    assert persistence.load_metadata_sidecar(str(expt)) is None


def test_save_metadata_sidecar_is_a_noop_without_a_real_source_path(tmp_path):
    # A synthetic/derived spectrum (no real file on disk) -- nothing
    # sensible to write next to, must not raise.
    spectrum = _make_spectrum(None)
    persistence.save_metadata_sidecar(spectrum)  # no exception
    spectrum2 = _make_spectrum(str(tmp_path / "does_not_exist"))
    persistence.save_metadata_sidecar(spectrum2)  # no exception, source_path doesn't exist on disk


# ============================================================
# Processing sidecar (live reprocessing recipe)
# ============================================================

def test_save_and_load_processing_sidecar_round_trips_live_state(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    spectrum = _make_spectrum(str(expt))
    spectrum.metadata.raw.update({
        "live_window": "hann", "live_zero_fill": 2048, "live_truncate_f1": 16,
        "phase0": 12.5, "phase1": -3.0, "baseline_correct": True, "baseline_method": "spline",
        "analysis_state": {"peaks": [["5.0", "0.05", "0.5", "0"]], "regions": []},
    })

    persistence.save_processing_sidecar(spectrum)
    loaded = persistence.load_processing_sidecar(str(expt))

    assert loaded["live_window"] == "hann"
    assert loaded["live_zero_fill"] == 2048
    assert loaded["live_truncate_f1"] == 16
    assert loaded["phase0"] == pytest.approx(12.5)
    assert loaded["phase1"] == pytest.approx(-3.0)
    assert loaded["baseline_correct"] is True
    assert loaded["baseline_method"] == "spline"
    assert loaded["analysis_state"] == {"peaks": [["5.0", "0.05", "0.5", "0"]], "regions": []}


def test_processing_sidecar_excludes_large_and_non_serializable_raw_entries(tmp_path):
    # acqus/procs/fid/fid_raw etc. are large, redundant with the
    # vendor's own on-disk files, and (for the FID arrays) not
    # JSON-serializable at all -- must never end up in the sidecar.
    expt = tmp_path / "expt1"
    expt.mkdir()
    spectrum = _make_spectrum(str(expt))
    spectrum.metadata.raw.update({
        "acqus": {"NUC1": "1H"}, "procs": {"SI": 65536}, "proc2s": {"SI": 1024},
        "fid": np.zeros(8), "fid_raw": np.zeros(8),
        "window_unsupported": "some diagnostic note", "read_2d_shape_guess_failed": "another note",
        "live_window": "hann",  # the one key that SHOULD survive, to prove the filter isn't wiping everything
    })

    persistence.save_processing_sidecar(spectrum)
    loaded = persistence.load_processing_sidecar(str(expt))

    assert loaded == {"live_window": "hann"}


def test_processing_sidecar_silently_drops_a_stray_non_serializable_value(tmp_path):
    # Robustness net: any *future* raw[key] = <numpy array or other
    # non-JSON type> addition should degrade to "not persisted" rather
    # than crashing the save -- confirms the filter is by actual
    # serializability, not just the hardcoded blocklist.
    expt = tmp_path / "expt1"
    expt.mkdir()
    spectrum = _make_spectrum(str(expt))
    spectrum.metadata.raw["sr_original_hz"] = np.array([1.0, 2.0, 3.0])
    spectrum.metadata.raw["sr_shift_hz"] = 5.0

    persistence.save_processing_sidecar(spectrum)  # no exception
    loaded = persistence.load_processing_sidecar(str(expt))
    assert loaded == {"sr_shift_hz": 5.0}


def test_load_processing_sidecar_returns_none_when_missing(tmp_path):
    expt = tmp_path / "expt1"
    expt.mkdir()
    assert persistence.load_processing_sidecar(str(expt)) is None


def test_save_processing_sidecar_is_a_noop_without_a_real_source_path(tmp_path):
    persistence.save_processing_sidecar(_make_spectrum(None))  # no exception


def test_processing_sidecar_round_trip_via_a_file_scoped_source_path(tmp_path):
    # The JEOL-style "several experiments share one directory" case --
    # confirms the round trip works end-to-end (not just the path
    # computation tested above) when source_path is a file, not a dir.
    expt = tmp_path / "carbon.jdf"
    expt.write_bytes(b"\x00")
    spectrum = _make_spectrum(str(expt))
    spectrum.metadata.raw["live_zero_fill"] = 4096

    persistence.save_processing_sidecar(spectrum)
    persistence.save_metadata_sidecar(spectrum)

    assert persistence.load_processing_sidecar(str(expt)) == {"live_zero_fill": 4096}
    assert persistence.load_metadata_sidecar(str(expt))["vendor"] == "bruker"
    assert (tmp_path / "carbon.jdf.opennmr_processing.json").exists()
    assert (tmp_path / "carbon.jdf.opennmr_metadata.json").exists()
