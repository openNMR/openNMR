"""Detecting cloud-backed data that is listed but not downloaded.

Placeholders cannot be created for real in a test (they are made by
OneDrive/iCloud's own file provider), so the two signals
:mod:`opennmr.io.cloud` reads are simulated at the ``os.stat`` boundary
-- which is exactly where the real ones appear.
"""

import os

import pytest

from opennmr.io import cloud


class _FakeStat:
    """Just the fields :mod:`opennmr.io.cloud` reads."""

    def __init__(self, size=1024, blocks=8, flags=0):
        self.st_size = size
        self.st_blocks = blocks
        self.st_flags = flags


def _fake_stat(monkeypatch, mapping):
    """Make ``os.stat`` answer from ``mapping`` (path -> _FakeStat),
    falling through to the real one for everything else."""
    real = os.stat

    def patched(path, *args, **kwargs):
        for marker, result in mapping.items():
            if str(path).endswith(marker):
                return result
        return real(path, *args, **kwargs)

    monkeypatch.setattr(cloud.os, "stat", patched)


def test_local_file_is_not_dataless(tmp_path):
    path = tmp_path / "acqus"
    path.write_text("##$NS= 16\n")

    assert cloud.is_dataless(str(path)) is False


def test_macos_dataless_flag_is_detected(tmp_path, monkeypatch):
    """macOS's own UF_DATALESS is the authoritative signal."""
    path = tmp_path / "ser"
    path.write_bytes(b"x" * 16)
    _fake_stat(monkeypatch, {"ser": _FakeStat(size=633_000_000, blocks=8,
                                              flags=cloud.UF_DATALESS)})

    assert cloud.is_dataless(str(path)) is True


def test_zero_blocks_with_real_size_is_detected(tmp_path, monkeypatch):
    """The portable fallback, for sync clients that use sparse
    placeholders rather than the macOS flag."""
    path = tmp_path / "ser"
    path.write_bytes(b"x" * 16)
    _fake_stat(monkeypatch, {"ser": _FakeStat(size=1_000_000, blocks=0, flags=0)})

    assert cloud.is_dataless(str(path)) is True


def test_empty_file_is_not_dataless(tmp_path, monkeypatch):
    """Zero blocks *and* zero size is an ordinary empty file, not a
    placeholder -- the size is what distinguishes them."""
    path = tmp_path / "ser"
    path.write_bytes(b"")
    _fake_stat(monkeypatch, {"ser": _FakeStat(size=0, blocks=0, flags=0)})

    assert cloud.is_dataless(str(path)) is False


def test_icloud_placeholder_sibling_is_detected(tmp_path):
    """Older iCloud evicts the file entirely and leaves a dotted
    placeholder beside it, so the original name does not exist."""
    (tmp_path / ".ser.icloud").write_bytes(b"plist")

    assert cloud.is_dataless(str(tmp_path / "ser")) is True


def test_missing_file_is_not_reported_as_cloud(tmp_path):
    """A path that simply is not there must surface as the caller's own
    error, not be disguised as a download-pending file."""
    assert cloud.is_dataless(str(tmp_path / "nothing-here")) is False


def test_parameters_are_local_for_a_real_experiment():
    source = os.path.join("test_data", "Bruker Example Data", "Cyclosporine", "204")
    if not os.path.isdir(source):
        pytest.skip(f"needs the real dataset at {source}")

    assert cloud.parameters_are_local(source) is True
    assert cloud.pending_download_bytes(source) == 0


def test_dataless_parameters_stop_an_experiment_being_peeked(tmp_path, monkeypatch):
    """The whole point: an experiment whose parameter files are still
    in the cloud must be reported as not-local, so the browser
    describes it from its name instead of downloading it."""
    experiment = tmp_path / "10"
    experiment.mkdir()
    (experiment / "acqus").write_text("##$NS= 16\n")
    (experiment / "ser").write_bytes(b"x" * 16)
    _fake_stat(monkeypatch, {
        "acqus": _FakeStat(size=4096, blocks=0, flags=0),
        "ser": _FakeStat(size=633_000_000, blocks=0, flags=0),
    })

    assert cloud.parameters_are_local(str(experiment)) is False
    assert cloud.pending_download_bytes(str(experiment)) == 633_000_000 + 4096


def test_downloaded_parameters_with_cloud_data_still_peek(tmp_path, monkeypatch):
    """Parameters local but bulk data still in the cloud is the common
    OneDrive case, and it is describable: peek only reads parameters,
    so the browser should still say what the experiment is -- while
    reporting what a load would cost."""
    experiment = tmp_path / "10"
    experiment.mkdir()
    (experiment / "acqus").write_text("##$NS= 16\n")
    (experiment / "ser").write_bytes(b"x" * 16)
    _fake_stat(monkeypatch, {"ser": _FakeStat(size=633_000_000, blocks=0, flags=0)})

    assert cloud.parameters_are_local(str(experiment)) is True
    assert cloud.pending_download_bytes(str(experiment)) == 633_000_000


@pytest.mark.parametrize("count,expected", [
    (0, "0 B"), (900, "900 B"), (1536, "1.5 KB"),
    (5 * 1024**2, "5.0 MB"), (633 * 1024**2, "633 MB"), (3 * 1024**3, "3.0 GB"),
])
def test_format_bytes(count, expected):
    assert cloud.format_bytes(count) == expected
