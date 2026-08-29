import numpy as np
import pytest

from nmrsuite.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from nmrsuite.processing import slicing


def _make_2d(n_f1=4, n_f2=6, pseudo=False, homonuclear=False):
    data = np.arange(n_f1 * n_f2, dtype=float).reshape(n_f1, n_f2)
    x_ppm = np.linspace(10, 0, n_f2)
    if pseudo:
        f1_axis = NMRAxis(kind="index", index=np.arange(1, n_f1 + 1))
    else:
        y_ppm = np.linspace(100, 90, n_f1)
        f1_axis = NMRAxis(kind="frequency", ppm=y_ppm, hz=y_ppm * 100.0)
    f2_axis = NMRAxis(kind="frequency", ppm=x_ppm, hz=x_ppm * 400.0)
    metadata = NMRMetadata(
        vendor="bruker", nucleus="1H", nucleus_indirect="13C",
        sfo1=400.0, sfo1_indirect=100.0, pseudo_2d=pseudo, homonuclear=homonuclear,
    )
    return NMRSpectrum(data=data, axes=[f1_axis, f2_axis], metadata=metadata)


def test_get_trace_row_by_index():
    spec = _make_2d()
    trace = slicing.get_trace(spec, axis="f1", index=2)
    assert trace.ndim == 1
    np.testing.assert_allclose(trace.data, spec.data[1, :])
    assert trace.metadata.raw["slice"]["slice_type"] == "row"
    assert trace.metadata.raw["slice"]["index"] == 2
    assert trace.metadata.nucleus == "1H"


def test_get_trace_column_by_shift_heteronuclear():
    spec = _make_2d()
    trace = slicing.get_trace(spec, axis="f2", shift=5.0)
    assert trace.ndim == 1
    assert trace.metadata.nucleus == "13C"
    assert trace.metadata.sfo1 == 100.0


def test_get_trace_width_averaging_matches_manual_sum():
    spec = _make_2d()
    trace = slicing.get_trace(spec, axis="f1", index=2, width=3, combine="sum")
    np.testing.assert_allclose(trace.data, spec.data[0:3, :].sum(axis=0))


def test_get_trace_pseudo2d_requires_index_not_shift():
    spec = _make_2d(pseudo=True)
    with pytest.raises(ValueError):
        slicing.get_trace(spec, axis="f1", shift=1.0)
    trace = slicing.get_trace(spec, axis="f1", index=1)
    assert trace.metadata.raw["slice"]["shift_unit"] == "index"


def test_get_slices_batch_rows_and_columns():
    spec = _make_2d()
    result = slicing.get_slices(spec, rows=[1, 2], column_shifts=[8.0, 2.0])
    assert len(result["rows"]) == 2
    assert len(result["columns"]) == 2
    indices = [t.metadata.raw["slice"]["index"] for t in result["columns"]]
    assert indices == sorted(indices)


def test_get_slices_normalise():
    spec = _make_2d()
    result = slicing.get_slices(spec, rows=[4], normalise=True)
    assert np.max(np.abs(result["rows"][0].data)) == pytest.approx(1.0)


def test_get_trace_window_crops_axis():
    spec = _make_2d()
    trace = slicing.get_trace(spec, axis="f1", index=1, window=(2.0, 6.0))
    assert trace.axes[0].ppm.max() <= 6.0
    assert trace.axes[0].ppm.min() >= 2.0
