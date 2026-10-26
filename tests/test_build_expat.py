"""packaging/build.py bundles the Python's own libexpat on Linux when it is not the system's
(found when a conda Python's frozen app failed on Ubuntu with ``undefined symbol:
XML_SetAllocTrackerActivationThreshold``)."""
import importlib.util
import os

import pytest

_BUILD = os.path.join(os.path.dirname(__file__), "..", "packaging", "build.py")
_spec = importlib.util.spec_from_file_location("opennmr_build_expat", _BUILD)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

CONDA = """\
\tlinux-vdso.so.1 (0x00007ffd1c9f5000)
\tlibexpat.so.1 => /home/quark/miniconda3/lib/libexpat.so.1 (0x00007f1e5c000000)
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f1e5bc00000)
"""
SYSTEM = """\
\tlibexpat.so.1 => /lib/x86_64-linux-gnu/libexpat.so.1 (0x00007f1e5c000000)
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f1e5bc00000)
"""
USR = "\tlibexpat.so.1 => /usr/lib64/libexpat.so.1 (0x0000)\n"


def test_a_private_libexpat_is_found_and_a_system_one_is_not():
    assert build.private_expat_from_ldd(CONDA) == "/home/quark/miniconda3/lib/libexpat.so.1"
    assert build.private_expat_from_ldd(SYSTEM) is None
    assert build.private_expat_from_ldd(USR) is None
    assert build.private_expat_from_ldd("") is None
    assert build.private_expat_from_ldd("\tlibexpat.so.1 => not found\n") is None


def test_only_linux_gets_the_extra_binary():
    assert build._expat_args("win32") == [] and build._expat_args("darwin") == []


def test_the_library_is_staged_under_its_soname_not_the_versioned_name(tmp_path):
    """The bug that shipped a broken AppImage: the link's target (``libexpat.so.1.12.2``) was
    bundled, leaving nothing called ``libexpat.so.1`` for pyexpat to find."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libexpat.so.1.12.2").write_bytes(b"real library")
    try:
        os.symlink("libexpat.so.1.12.2", lib / "libexpat.so.1")
    except OSError:                                          # Windows without symlink rights
        pytest.skip("cannot create symlinks here")
    staged = build.stage_expat(str(lib / "libexpat.so.1"), str(tmp_path / "stage"))
    assert os.path.basename(staged) == "libexpat.so.1"
    assert not os.path.islink(staged) and open(staged, "rb").read() == b"real library"


def test_the_extra_binary_is_added_beside_the_executable(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libexpat.so.1").write_bytes(b"x")

    class Result:
        stdout = f"\tlibexpat.so.1 => {lib / 'libexpat.so.1'} (0x1)\n"

    args = build._expat_args("linux", run=lambda *a, **k: Result(), stage_dir=str(tmp_path / "stage"))
    if os.path.isfile(getattr(importlib.util.find_spec("pyexpat"), "origin", "") or ""):
        assert args[0] == "--add-binary" and args[1].endswith(os.pathsep + ".")
        assert os.path.isfile(tmp_path / "stage" / "libexpat.so.1")
    else:                                                    # a Python with a built-in pyexpat
        assert args == []
