"""Writers for the SIMPSON test files (shared by the SIMPSON tests): a
two-line signal whose lines sit at known frequencies, so every expected
position is exact."""
import numpy as np

SW = 20000.0
N = 512
LINES = ((3000.0, 300.0, 1.0), (-1500.0, 500.0, 0.5))      # Hz, decay rate (1/s), amplitude


def make_fid(n=N, sw=SW, lines=LINES):
    t = np.arange(n) / sw
    return sum(a * np.exp(2j * np.pi * f * t) * np.exp(-t * r) for f, r, a in lines)


def _write(path, kind, values, sw=SW):
    with open(path, "w") as handle:
        handle.write(f"SIMP\nNP={len(values)}\nSW={sw}\nTYPE={kind}\nDATA\n")
        for v in values:
            handle.write(f"{v.real:.10e} {v.imag:.10e}\n")
        handle.write("END\n")
    return str(path)


def write_fid(path, fid=None, sw=SW):
    return _write(path, "FID", make_fid(sw=sw) if fid is None else fid, sw)


def write_spe(path, fid=None, sw=SW):
    """The spectrum SIMPSON's own ``fft`` gives (frequency increasing from -SW/2)."""
    fid = make_fid(sw=sw) if fid is None else fid
    return _write(path, "SPE", np.fft.fftshift(np.fft.fft(fid)), sw)


def write_xreim(path, kind="spe", sw=SW):
    fid = make_fid(sw=sw)
    n = len(fid)
    if kind == "fid":
        x, values = np.arange(n) / sw, fid
    else:
        x, values = np.fft.fftshift(np.fft.fftfreq(n, 1 / sw)), np.fft.fftshift(np.fft.fft(fid))
    with open(path, "w") as handle:
        for xi, v in zip(x, values):
            handle.write(f"{xi:.10e} {v.real:.10e} {v.imag:.10e}\n")
    return str(path)


def write_point_fid(path, values, sw=SW):
    """A short FID whose first points are ``values`` (for the point series)."""
    return _write(path, "FID", np.asarray(values, dtype=complex), sw)


def line_hz(spectrum):
    """Frequencies (Hz) of the two strongest separated lines of a 1D spectrum."""
    hz = spectrum.axes[0].hz
    y = np.abs(spectrum.data + (1j * spectrum.data_imag if spectrum.data_imag is not None else 0))
    first = int(np.argmax(y))
    mask = np.abs(hz - hz[first]) > 600
    second = int(np.argmax(np.where(mask, y, -1)))
    return sorted([float(hz[first]), float(hz[second])])
