import numpy as np
import pytest

from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
from opennmr.processing import slicing


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


def _make_trace(ppm, data, vendor="bruker", nucleus="1H", sfo1=400.0):
    axis = NMRAxis(kind="frequency", ppm=ppm, hz=ppm * sfo1)
    metadata = NMRMetadata(vendor=vendor, nucleus=nucleus, sfo1=sfo1)
    return NMRSpectrum(data=np.asarray(data, dtype=float), axes=[axis], metadata=metadata)


def test_build_pseudo2d_stacks_traces_in_order():
    ppm = np.linspace(0, 10, 5)
    traces = [_make_trace(ppm, np.full(5, i)) for i in range(3)]
    stacked = slicing.build_pseudo2d(traces)
    assert stacked.ndim == 2
    assert stacked.data.shape == (3, 5)
    np.testing.assert_allclose(stacked.data[0], np.zeros(5))
    np.testing.assert_allclose(stacked.data[2], np.full(5, 2))


def test_build_pseudo2d_marks_result_as_pseudo_2d():
    ppm = np.linspace(0, 10, 4)
    traces = [_make_trace(ppm, [1, 2, 3, 4]), _make_trace(ppm, [5, 6, 7, 8])]
    stacked = slicing.build_pseudo2d(traces)
    assert stacked.is_pseudo_2d is True
    assert stacked.axes[0].kind == "index"
    np.testing.assert_array_equal(stacked.axes[0].index, [1, 2])
    assert stacked.axes[1].kind == "frequency"
    np.testing.assert_allclose(stacked.axes[1].ppm, ppm)


def test_build_pseudo2d_inherits_metadata_from_first_trace():
    ppm = np.linspace(0, 10, 4)
    traces = [_make_trace(ppm, [1, 2, 3, 4], nucleus="19F", sfo1=376.5),
              _make_trace(ppm, [5, 6, 7, 8], nucleus="19F", sfo1=376.5)]
    stacked = slicing.build_pseudo2d(traces)
    assert stacked.metadata.vendor == "bruker"
    assert stacked.metadata.nucleus == "19F"
    assert stacked.metadata.sfo1 == pytest.approx(376.5)


def test_build_pseudo2d_rejects_a_single_trace():
    ppm = np.linspace(0, 10, 4)
    with pytest.raises(ValueError, match="at least two"):
        slicing.build_pseudo2d([_make_trace(ppm, [1, 2, 3, 4])])


def test_build_pseudo2d_rejects_mismatched_axes():
    ppm_a = np.linspace(0, 10, 4)
    ppm_b = np.linspace(0, 5, 4)
    with pytest.raises(ValueError, match="ppm axis"):
        slicing.build_pseudo2d([_make_trace(ppm_a, [1, 2, 3, 4]), _make_trace(ppm_b, [5, 6, 7, 8])])


def test_build_pseudo2d_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="ppm axis"):
        slicing.build_pseudo2d([
            _make_trace(np.linspace(0, 10, 4), [1, 2, 3, 4]),
            _make_trace(np.linspace(0, 10, 6), [1, 2, 3, 4, 5, 6]),
        ])


def test_build_pseudo2d_rejects_a_2d_trace():
    ppm = np.linspace(0, 10, 4)
    good = _make_trace(ppm, [1, 2, 3, 4])
    bad = _make_2d(n_f2=4)
    with pytest.raises(ValueError, match="not 1D"):
        slicing.build_pseudo2d([good, bad])


def test_build_pseudo2d_rejects_a_trace_with_no_ppm_axis():
    index_axis = NMRAxis(kind="index", index=np.arange(1, 5))
    no_ppm_trace = NMRSpectrum(
        data=np.array([1.0, 2.0, 3.0, 4.0]), axes=[index_axis],
        metadata=NMRMetadata(vendor="bruker"),
    )
    ppm = np.linspace(0, 10, 4)
    with pytest.raises(ValueError, match="ppm"):
        slicing.build_pseudo2d([no_ppm_trace, _make_trace(ppm, [1, 2, 3, 4])])
