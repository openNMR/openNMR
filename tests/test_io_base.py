import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.io import base


@pytest.fixture
def clean_registry(monkeypatch):
    """Every real vendor stays registered (peek() dispatch shouldn't
    need to know about them), but this test's own throwaway "fake"
    vendor is removed again afterward regardless of pass/fail."""
    yield
    base._REGISTRY.pop("fake", None)


def _fake_spectrum() -> NMRSpectrum:
    data = np.zeros((4, 8))
    axes = [
        NMRAxis(kind="index", index=np.arange(1, 5)),
        NMRAxis(kind="frequency", ppm=np.linspace(10, 0, 8), hz=np.linspace(4000, 0, 8)),
    ]
    metadata = NMRMetadata(vendor="fake", nucleus="1H", pseudo_2d=True)
    return NMRSpectrum(data=data, axes=axes, metadata=metadata)


def test_peek_uses_a_vendor_supplied_peek_when_registered(clean_registry):
    calls = []

    def fake_peek(path, **kwargs):
        calls.append(path)
        return {"vendor": "fake", "ndim": 2, "shape": (4, 8), "custom": True}

    def fail_read(path, **kwargs):
        raise AssertionError("peek() should not fall back to a full read when one is registered")

    base.register("fake", detect=lambda path: True, read=fail_read, peek=fake_peek)

    result = base.peek("some/path", vendor="fake")

    assert result == {"vendor": "fake", "ndim": 2, "shape": (4, 8), "custom": True}
    assert calls == ["some/path"]


def test_peek_falls_back_to_a_full_load_when_no_peek_is_registered(clean_registry):
    # Purely additive: a vendor that hasn't implemented peek() yet must
    # keep working (correctly, just not fast) rather than raising.
    base.register("fake", detect=lambda path: True, read=lambda path, **kwargs: _fake_spectrum())

    result = base.peek("some/path", vendor="fake")

    assert result == {
        "vendor": "fake", "ndim": 2, "shape": (4, 8),
        "nucleus": "1H", "nucleus_indirect": None, "pseudo_2d": True,
    }
