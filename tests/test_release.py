"""packaging/release.py: version handling and the git steps, in a throwaway repo."""

import importlib.util
import os
import subprocess

import pytest

PACKAGING = os.path.join(os.path.dirname(__file__), "..", "packaging")
_spec = importlib.util.spec_from_file_location("opennmr_release", os.path.join(PACKAGING, "release.py"))
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src" / "opennmr").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "opennmr"\nversion = "0.8.1"\n')
    (tmp_path / "src" / "opennmr" / "_version.py").write_text('_FALLBACK_VERSION = "0.8.1"\n')
    for cmd in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                ["add", "."], ["commit", "-qm", "start"]):
        subprocess.run(["git", *cmd], cwd=tmp_path, check=True)
    return str(tmp_path)


def test_parse_version_accepts_only_x_y_z():
    assert release.parse_version("0.8.12") == (0, 8, 12)
    assert release.parse_version("0.10.0") > release.parse_version("0.9.9")    # not a string comparison
    for bad in ("v0.8.2", "0.8", "0.8.2rc1", "", "0.8.2.1"):
        with pytest.raises(ValueError):
            release.parse_version(bad)


def test_set_version_changes_both_places_and_nothing_else(repo):
    release.set_version("0.9.0", repo)
    assert release.read_version(repo) == "0.9.0"
    assert open(os.path.join(repo, "src", "opennmr", "_version.py")).read() == '_FALLBACK_VERSION = "0.9.0"\n'
    assert 'name = "opennmr"' in open(os.path.join(repo, "pyproject.toml")).read()


def test_check_ready_rejects_old_dirty_and_existing(repo):
    assert release.check_ready("0.8.2", repo) == []
    assert release.check_ready("0.8.1", repo)                  # not newer
    assert release.check_ready("0.7.9", repo)                  # older
    assert release.check_ready("v0.8.2", repo)                 # malformed
    open(os.path.join(repo, "stray.txt"), "w").write("x")
    assert any("uncommitted" in p for p in release.check_ready("0.8.2", repo))
    os.remove(os.path.join(repo, "stray.txt"))
    subprocess.run(["git", "tag", "v0.8.2"], cwd=repo, check=True)
    assert any("already exists" in p for p in release.check_ready("0.8.2", repo))


def test_release_version_lines_match_the_real_files():
    """The patterns release.py edits must exist in this repository's real files."""
    repo = os.path.join(os.path.dirname(__file__), "..")
    assert release.read_version(repo)
    text = open(os.path.join(repo, "src", "opennmr", "_version.py"), encoding="utf-8").read()
    assert release.re.search(r'^_FALLBACK_VERSION\s*=\s*"[^"]+"', text, flags=release.re.MULTILINE)


def test_workflow_gates_publishing_on_every_installer():
    wf = open(os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "build.yml"),
              encoding="utf-8").read()
    for job in ("pypi:", "release:"):
        block = wf[wf.index(f"\n  {job}"):]
        assert "needs: build" in block.split("steps:")[0]
    assert "skip-existing: true" in wf and "SHA256SUMS" in wf
    publish = open(os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "publish.yml"),
                   encoding="utf-8").read()
    assert "tags:" not in publish            # a tag must not publish twice
