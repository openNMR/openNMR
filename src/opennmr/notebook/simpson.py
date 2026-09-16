"""
Tools for importing SIMPSON-simulated NMR spectra and pulse-shape
files, and for plotting them either alone or alongside spectra
imported with nmrtools.one_d.NMR1dimport.

Workflow
--------
    SIMPSONimport     -> import a SIMPSON .spe/.dat/.csdf(SIMP-text) spectrum
    CSDMimport        -> import a *genuine* CSDM/JSON file (needs csdmpy)
    read_shape        -> import a SIMPSON pulse-shape (.shape) file
    read_fid          -> import a raw SIMPSON FID (time domain, no FFT)
    apodization_window -> compute an apodization window on its own
    plot_fid          -> plot a FID, optionally with its apodization window
    read_single_point_fid -> first point of one single-point FID file
    read_fid_series   -> first point of a series of single-point FID files
    plot_1d_stack     -> stacked-spectrum plot from a list of result dicts
                        (mix NMR1dimport / SIMPSONimport / CSDMimport freely)
    plot_shape        -> amplitude/phase plot of a pulse-shape file
    plot_fid_series   -> plot first-point intensity vs a swept quantity

Design notes
------------
NMR1dplot (in one_d.py) is left untouched: it still only accepts
Bruker directory paths and calls NMR1dimport internally. plot_1d_stack
below re-implements its stacking/offset/colour/label logic against a
list of *already-imported* result dicts instead, so spectra from any
importer that produces a compatible dict can be plotted together.
This is a deliberate duplication of ~40 lines of one_d.py's plotting
loop, not a wrapper around it -- if NMR1dplot's stacking logic changes
later, this needs updating in step. Worth consolidating into one
shared function if that drift becomes a problem in practice.

SIMPSON files carry no observe frequency, so ppm conversion is
optional and only happens if you supply `nuclide` (+ `field`, or an
explicit `SFO1`). ppm conversion is attempted via
nmrtools.utils.find_gamma the way one_d.py calls it elsewhere, BUT:
every usage of find_gamma visible in one_d.py already has SFO1 known
in advance (read from a Bruker acqus file) before find_gamma is
called -- its apparent role there is picking the canonical nuclide
label given a known SFO1, not deriving SFO1 from a nuclide+field pair.
I do not have utils.py, so I can't confirm find_gamma can do the
latter at all. _hz_to_ppm() below calls it as asked, but checks for a
usable SFO1 in the result and raises a clear error if one isn't there,
rather than silently mis-scaling the ppm axis. If find_gamma doesn't
support this, the fix is either to add a small
nuclide+field->SFO1 helper (SFO1_MHz = field_T * gamma / (2*pi*1e6))
somewhere in utils.py, or to just pass SFO1 explicitly here.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx

from .utils import find_gamma, NUC_label
from .plotting import (
    setup_plotting, get_stack_colormap, apply_tick_style, apply_spine_style, save_figure,
)

try:
    import csdmpy as cp
    _HAS_CSDMPY = True
except ImportError:
    _HAS_CSDMPY = False


# ============================================================
# Format sniffing (content, not extension)
# ============================================================

def _sniff_format(path):
    """
    Identify a SIMPSON-related file by its content rather than its
    extension -- SIMPSON's `fsave` names files however the .tcl script
    asked, so a `.csdf` extension is no guarantee of JSON/CSDM content
    (see module docstring). Returns one of 'simp', 'dat', 'shape'.
    """
    with open(path, 'r', errors='ignore') as f:
        head_lines = [f.readline() for _ in range(3)]

    first = head_lines[0].strip()

    if first.startswith('{'):
        raise ValueError(
            f"{path} looks like JSON (genuine CSDM format), not SIMPSON "
            f"SIMP text. Use CSDMimport() instead of SIMPSONimport()."
        )

    if first == 'SIMP':
        return 'simp'

    # No header line: either a 3-column (freq, real, imag) .dat file,
    # or a 2-column (amplitude, phase) .shape file.
    ncols = len(first.split())
    if ncols == 3:
        return 'dat'
    elif ncols == 2:
        return 'shape'
    else:
        raise ValueError(
            f"Could not identify SIMPSON file format for {path} "
            f"(first line has {ncols} columns, expected 2 or 3)."
        )


# ============================================================
# Low-level readers
# ============================================================

def _read_simp_text(path):
    """
    Parse a SIMPSON 'SIMP' text-format file (.spe, or a .csdf that is
    actually SIMP text -- see module docstring). Returns NP, SW, TYPE,
    real, imag, path. Does not build an x-axis; SIMPSONimport does
    that, since it depends on TYPE (SPE vs FID).
    """
    NP = SW = TYPE = None
    real, imag = [], []
    in_data = False

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('NP='):
                NP = int(line[3:])
            elif line.startswith('SW='):
                SW = float(line[3:])
            elif line.startswith('TYPE='):
                TYPE = line[5:]
            elif line == 'DATA':
                in_data = True
            elif line == 'END':
                break
            elif in_data:
                re_val, im_val = line.split()
                real.append(float(re_val))
                imag.append(float(im_val))

    if NP is None or SW is None:
        raise ValueError(f"{path}: missing NP/SW header -- not a valid SIMPSON SIMP file.")
    if len(real) != NP:
        raise ValueError(
            f"{path}: header declares NP={NP} but {len(real)} data lines were read."
        )

    return {
        'NP': NP, 'SW': SW, 'TYPE': TYPE,
        'real': np.array(real), 'imag': np.array(imag), 'path': path,
    }


def _read_dat_text(path):
    """
    Parse a SIMPSON '-xreim' .dat file: three columns
    (freq_Hz, real, imag), no header. A single-point file (one data
    row -- e.g. an NP=1 efficiency-test FID) is loaded by np.loadtxt
    as a flat 1D array rather than an (1, 3) array, so that case is
    reshaped explicitly rather than being rejected.
    """
    data = np.loadtxt(path)
    if data.ndim == 1 and data.shape[0] == 3:
        data = data.reshape(1, 3)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns (freq, real, imag).")
    freq, real, imag = data[:, 0], data[:, 1], data[:, 2]
    return {'NP': len(freq), 'x_Hz': freq, 'TYPE': 'SPE', 'real': real, 'imag': imag, 'path': path}


def read_shape(path):
    """
    Read a SIMPSON pulse-shape file: two columns per line (amplitude,
    phase in degrees), no header, one row per point.

    Returns
    -------
    dict : 'amplitude', 'phase' (ndarrays), 'npoints', 'path'.
    """
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{path}: expected 2 columns (amplitude, phase).")
    amplitude, phase = data[:, 0], data[:, 1]
    return {'amplitude': amplitude, 'phase': phase, 'npoints': len(amplitude), 'path': path}


# ============================================================
# FID processing (only used if a file's TYPE is FID, not SPE)
# ============================================================

def apodization_window(npoints, dt, kind='exponential', lb=0.0, gb=0.0,
                        sine_offset=0.0, sine_end=1.0):
    """
    Compute a time-domain apodization (window) function on its own --
    split out from fft_simpson_fid so it can be inspected/plotted
    (see plot_fid) before deciding what to apply.

    Parameters
    ----------
    npoints : int
    dt : float
        Dwell time in seconds (1/SW).
    kind : {'none', 'exponential', 'gaussian', 'lorentz_to_gauss',
            'sine', 'sine_squared'}, optional
        'none'              -- all ones, no apodization.
        'exponential'       -- exp(-pi*lb*t); `lb` in Hz (line
                                broadening added, as in fsave/TopSpin
                                'em').
        'gaussian'          -- exp(-(pi*gb*t)**2); `gb` in Hz (pure
                                Gaussian broadening, no resolution
                                gain).
        'lorentz_to_gauss'  -- exp(-pi*lb*t) * exp((pi*gb*t)**2); the
                                standard Lorentz-to-Gauss resolution-
                                enhancement window (needs both `lb`
                                and `gb`; if `gb` is too small relative
                                to `lb` and the acquisition time, this
                                blows up rather than tapering the end
                                of the FID -- check with plot_fid).
        'sine'              -- sin(pi*(sine_offset + (sine_end -
                                sine_offset)*t/t_max)); `sine_offset`/
                                `sine_end` in units of pi (0 = start of
                                sine, 1 = end), the same convention as
                                TopSpin's SSB-derived sine bells.
        'sine_squared'      -- square of the 'sine' window above.
    lb, gb : float, optional
        Broadening constants in Hz for 'exponential'/'gaussian'/
        'lorentz_to_gauss'. Default 0 (no broadening).
    sine_offset, sine_end : float, optional
        Start/end phase (in units of pi) for 'sine'/'sine_squared'.
        Defaults (0, 1) give a plain sine bell over the full FID.

    Returns
    -------
    t, window : ndarrays
        Time axis (s) and the window function, both length npoints.
    """
    t = np.arange(npoints) * dt
    t_max = t[-1] if npoints > 1 else dt

    if kind == 'none':
        window = np.ones(npoints)
    elif kind == 'exponential':
        window = np.exp(-np.pi * lb * t)
    elif kind == 'gaussian':
        window = np.exp(-(np.pi * gb * t) ** 2)
    elif kind == 'lorentz_to_gauss':
        window = np.exp(-np.pi * lb * t) * np.exp((np.pi * gb * t) ** 2)
    elif kind == 'sine':
        phase = sine_offset + (sine_end - sine_offset) * t / t_max
        window = np.sin(np.pi * phase)
    elif kind == 'sine_squared':
        phase = sine_offset + (sine_end - sine_offset) * t / t_max
        window = np.sin(np.pi * phase) ** 2
    else:
        raise ValueError(
            f"Unknown apodization kind '{kind}' -- expected one of "
            f"'none', 'exponential', 'gaussian', 'lorentz_to_gauss', "
            f"'sine', 'sine_squared'."
        )

    return t, window


def apply_apodization(real, imag, SW, kind='exponential', **kwargs):
    """
    Multiply a FID by an apodization_window. Kept separate from
    fft_simpson_fid so the apodized time-domain FID itself (not just
    the resulting spectrum) can be inspected/plotted.

    Returns
    -------
    t, real_apod, imag_apod, window : ndarrays
    """
    npoints = len(real)
    dt = 1.0 / SW
    t, window = apodization_window(npoints, dt, kind=kind, **kwargs)
    fid = (np.asarray(real) + 1j * np.asarray(imag)) * window
    return t, fid.real, fid.imag, window


def fft_simpson_fid(real, imag, SW, apodization='exponential', apod_params=None,
                     line_broadening=0.0, zero_fill=None):
    """
    Fourier-transform a SIMPSON time-domain FID (TYPE=FID) into a
    frequency-domain spectrum, with an optional apodization window
    (see apodization_window for the available kinds) and zero-filling.
    This gives the same processing control over a SIMPSON FID that
    TopSpin has already silently applied to Bruker processed data by
    the time NMR1dimport reads it -- useful if you want simulated and
    experimental lineshapes processed identically rather than trusting
    SIMPSON's own `fsave -xreim` output.

    Parameters
    ----------
    real, imag : array-like
        Time-domain FID.
    SW : float
        Spectral width in Hz (dwell time = 1/SW).
    apodization : str, optional
        Window kind, passed to apodization_window. Default
        'exponential'. Pass 'none' to disable apodization entirely.
    apod_params : dict, optional
        Extra kwargs for apodization_window (e.g. {'gb': 20} for a
        Gaussian/Lorentz-to-Gauss window, or {'sine_offset': 0.5} for
        a shifted sine bell).
    line_broadening : float, optional
        Back-compatible shortcut for apod_params={'lb': ...} when
        apodization is 'exponential' (or 'lorentz_to_gauss'). Ignored
        if apod_params already sets 'lb'. Default 0 (none) -- this
        preserves the old fft_simpson_fid(real, imag, SW,
        line_broadening=...) call signature.
    zero_fill : int, optional
        Total number of points after zero-filling. Default: no
        zero-filling (uses len(real)).

    Returns
    -------
    x_Hz, real_spectrum, imag_spectrum : ndarrays
    """
    NP = len(real)
    apod_params = dict(apod_params) if apod_params else {}
    if line_broadening and 'lb' not in apod_params:
        apod_params['lb'] = line_broadening

    if apodization and apodization != 'none':
        _, real_a, imag_a, _ = apply_apodization(real, imag, SW, kind=apodization, **apod_params)
        fid = real_a + 1j * imag_a
    else:
        fid = np.asarray(real) + 1j * np.asarray(imag)

    dt = 1.0 / SW
    n_fft = zero_fill if zero_fill else NP
    spectrum = np.fft.fftshift(np.fft.fft(fid, n=n_fft))
    x_Hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt))

    return x_Hz, spectrum.real, spectrum.imag


def read_fid(path):
    """
    Read a SIMPSON time-domain FID file (TYPE=FID) without Fourier-
    transforming it -- for inspecting/apodizing/plotting the raw FID
    (see plot_fid) ahead of deciding what processing to use, or for
    feeding real/imag/SW into fft_simpson_fid by hand.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict : 'real', 'imag' (ndarrays), 'NP', 'SW', 'dt' (=1/SW),
    't' (time axis, s), 'path'.
    """
    fmt = _sniff_format(path)
    if fmt == 'shape':
        raise ValueError(
            f"{path} looks like a SIMPSON pulse-shape file (2 columns, "
            f"no header), not a FID. Use read_shape() instead."
        )

    raw = _read_simp_text(path) if fmt == 'simp' else _read_dat_text(path)
    if raw.get('TYPE') != 'FID':
        raise ValueError(
            f"{path}: TYPE is '{raw.get('TYPE')}', not FID -- this looks "
            f"like an already-processed spectrum. Use SIMPSONimport() "
            f"instead, or fft it yourself with fft_simpson_fid() if you "
            f"specifically need x_Hz/real_spectrum/imag_spectrum."
        )

    NP, SW = raw['NP'], raw['SW']
    dt = 1.0 / SW
    t = np.arange(NP) * dt

    return {'real': raw['real'], 'imag': raw['imag'], 'NP': NP, 'SW': SW,
            'dt': dt, 't': t, 'path': path}


# ============================================================
# ppm conversion helper (shared by SIMPSONimport / CSDMimport)
# ============================================================

def _hz_to_ppm(x_Hz, nuclide, field, SFO1):
    """
    Resolve an SFO1 (MHz) from an explicit SFO1, or via
    nmrtools.utils.find_gamma given (nuclide, field) -- see the
    caveat about find_gamma's actual capabilities in this module's
    docstring. Returns (x_ppm or None, SFO1 or None): None if no ppm
    axis could be resolved (caller decides whether that's fatal).
    """
    if SFO1 is not None:
        return x_Hz / SFO1, SFO1

    if nuclide is None:
        return None, None

    nucgamma = find_gamma(nuclide, field=field, SFO1=SFO1)
    resolved_SFO1 = nucgamma.get('SFO1') if isinstance(nucgamma, dict) else None

    if resolved_SFO1 is None:
        raise ValueError(
            f"find_gamma('{nuclide}', field={field}) did not return a "
            f"usable SFO1 -- it may not support deriving SFO1 from "
            f"nuclide+field alone (see this module's docstring). Pass "
            f"SFO1 explicitly instead, or extend find_gamma/utils.py."
        )

    return x_Hz / resolved_SFO1, resolved_SFO1


def _normalise(y):
    y = np.asarray(y, dtype=float)
    span = y.max() - y.min()
    if span == 0:
        return y
    return (y - y.min()) / span


# ============================================================
# SIMPSONimport
# ============================================================

def SIMPSONimport(path, nuclide=None, field=None, SFO1=None, mass=1,
                   f1p=0, f2p=0, line_broadening=0.0, zero_fill=None):
    """
    Import a SIMPSON-generated spectrum: .spe, .dat, or a .csdf file
    that is (despite the extension) actually SIMPSON's SIMP text
    format. Format is detected from file content, not extension.
    For genuine CSDM/JSON files (e.g. mrsimulator output), use
    CSDMimport instead.

    Parameters
    ----------
    path : str
        Path to the .spe / .dat / SIMP-text .csdf file.
    nuclide : str, optional
        e.g. "31P". If given with `field` or `SFO1`, an x_ppm axis is
        computed (x_ppm = x_Hz / SFO1[MHz], same convention as
        NMR1dimport). If omitted, x_ppm is None and only x_Hz is set.
    field : optional
        Passed to find_gamma alongside `nuclide` -- see the caveat in
        this module's docstring about whether this actually resolves
        an SFO1.
    SFO1 : float, optional
        Explicit observe frequency in MHz. Takes precedence over
        nuclide/field if given.
    mass : float, optional
        Scales real/imag spectra, mirroring NMR1dimport's convention
        (default 1 -- SIMPSON output isn't mass-normalised the way an
        experimental NS/mass-scaled spectrum is; this is mainly so the
        dict is drop-in compatible).
    f1p, f2p : float, optional
        ppm window to crop to (same convention as NMR1dimport).
        Requires a resolvable ppm axis.
    line_broadening, zero_fill : optional
        Only used if the file's TYPE is FID rather than SPE -- see
        fft_simpson_fid.

    Returns
    -------
    dict with the same keys as NMR1dimport's output (x_ppm, x_Hz,
    real_spectrum, normalised_real_spectrum, imag_spectrum,
    normalised_imag_spectrum, nuclide, BF1, SFO1, path), plus 'NP'
    and 'SW' for reference -- so it can be passed straight into
    plot_1d_stack alongside NMR1dimport results.
    """
    fmt = _sniff_format(path)

    if fmt == 'shape':
        raise ValueError(
            f"{path} looks like a SIMPSON pulse-shape file (2 columns, "
            f"no header), not a spectrum. Use read_shape() instead."
        )

    raw = _read_simp_text(path) if fmt == 'simp' else _read_dat_text(path)
    NP = raw['NP']

    if raw.get('TYPE') == 'FID':
        x_Hz, real, imag = fft_simpson_fid(
            raw['real'], raw['imag'], raw['SW'],
            line_broadening=line_broadening, zero_fill=zero_fill,
        )
    elif 'x_Hz' in raw:
        x_Hz, real, imag = raw['x_Hz'], raw['real'], raw['imag']
    else:
        x_Hz = np.linspace(-raw['SW'] / 2, raw['SW'] / 2, NP, endpoint=False)
        real, imag = raw['real'], raw['imag']

    x_ppm, resolved_SFO1 = (None, None)
    if nuclide is not None or SFO1 is not None:
        x_ppm, resolved_SFO1 = _hz_to_ppm(x_Hz, nuclide, field, SFO1)

    if f1p != 0 or f2p != 0:
        if x_ppm is None:
            raise ValueError(
                "f1p/f2p cropping needs a ppm axis -- supply nuclide "
                "(+field or SFO1)."
            )
        # NMR1dimport's crop (np.argmax(xAxppm < f1p)) relies on its
        # ppm axis always being descending (true for Bruker data, per
        # its own SR/O1 convention) -- argmax returns the *first*
        # match, which is only the crossover point if the array starts
        # False. SIMPSONimport's x_Hz/x_ppm is ascending (built via
        # linspace(-SW/2, SW/2, ...)), where that trick silently
        # returns index 0 instead of the crossover -- caught this by
        # testing against ad_echo.spe. Using an explicit mask instead
        # so cropping is correct regardless of axis order.
        lo, hi = (f1p, f2p) if f1p < f2p else (f2p, f1p)
        mask = (x_ppm >= lo) & (x_ppm <= hi)
        if not mask.any():
            raise ValueError(f"No points found between {f1p} and {f2p} ppm.")
        x_ppm, x_Hz = x_ppm[mask], x_Hz[mask]
        real, imag = real[mask], imag[mask]

    real = np.asarray(real) / mass
    imag = np.asarray(imag) / mass

    return {
        'x_ppm': x_ppm, 'x_Hz': x_Hz,
        'real_spectrum': real, 'normalised_real_spectrum': _normalise(real),
        'imag_spectrum': imag, 'normalised_imag_spectrum': _normalise(imag),
        'nuclide': nuclide, 'BF1': None, 'SFO1': resolved_SFO1,
        'path': path, 'NP': NP, 'SW': raw.get('SW'),
    }


# ============================================================
# CSDMimport (genuine CSDM/JSON, e.g. mrsimulator output)
# ============================================================

def CSDMimport(path, nuclide=None, field=None, SFO1=None, mass=1,
               dependent_variable_index=0, component_index=0):
    """
    Import a genuine CSDM/JSON file (e.g. from mrsimulator) via
    csdmpy, returning the same schema as SIMPSONimport/NMR1dimport.

    Verified against csdmpy directly (round-tripped a synthetic 1D
    CSDM file): `cp.load(path)` returns a CSDM object whose first
    dimension exposes `.coordinates.value` (ndarray) and
    `.coordinates.unit` (an astropy Unit -- compare via str()), and
    whose `dependent_variables[i].components[j]` is the (possibly
    complex) data array. Only 1D, single-dependent-variable files are
    handled; extend dependent_variable_index/component_index handling
    if you need more.

    Parameters
    ----------
    path : str
    nuclide, field, SFO1 : optional
        Only used if the file's axis is in Hz rather than already in
        ppm -- see _hz_to_ppm and its caveat about find_gamma.
    mass : float, optional
    dependent_variable_index, component_index : int, optional
        Which dependent variable / component to read.

    Returns
    -------
    dict, same schema as SIMPSONimport.
    """
    if not _HAS_CSDMPY:
        raise ImportError(
            "CSDMimport needs the 'csdmpy' package (pip install csdmpy) "
            "to read genuine CSDM/JSON files -- a different format from "
            "SIMPSON's SIMP text output. If your file starts with 'SIMP' "
            "rather than '{', it's SIMP text despite the .csdf extension "
            "-- use SIMPSONimport for that instead."
        )

    data = cp.load(path)
    dim = data.dimensions[0]
    dv = data.dependent_variables[dependent_variable_index]

    x = np.asarray(dim.coordinates.value)
    unit = str(dim.coordinates.unit)
    y = np.asarray(dv.components[component_index])

    real = np.real(y)
    imag = np.imag(y) if np.iscomplexobj(y) else np.zeros_like(real)

    if unit == 'ppm':
        x_ppm, x_Hz, resolved_SFO1 = x, None, SFO1
    elif unit == 'Hz':
        x_Hz = x
        x_ppm, resolved_SFO1 = _hz_to_ppm(x_Hz, nuclide, field, SFO1)
    else:
        raise ValueError(
            f"{path}: dimension unit is '{unit}' -- CSDMimport currently "
            f"only handles 'Hz' and 'ppm' axes, extend it for this case."
        )

    real = real / mass
    imag = imag / mass

    return {
        'x_ppm': x_ppm, 'x_Hz': x_Hz,
        'real_spectrum': real, 'normalised_real_spectrum': _normalise(real),
        'imag_spectrum': imag, 'normalised_imag_spectrum': _normalise(imag),
        'nuclide': nuclide, 'BF1': None, 'SFO1': resolved_SFO1,
        'path': path,
    }


# ============================================================
# Single-point FID series (e.g. pulse-sequence efficiency tests)
# ============================================================

def read_single_point_fid(path):
    """
    Read just the first data point of a SIMPSON FID file -- for
    single-point acquisitions used to evaluate pulse-sequence
    efficiency (e.g. NP=1 nutation/power/offset tests), where the
    quantity of interest is a single complex intensity rather than a
    lineshape. Works for any NP (only the first point is used), so a
    file with more points than strictly needed is handled fine too.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict : 'real', 'imag', 'magnitude' (floats), 'path'.
    """
    fmt = _sniff_format(path)
    if fmt == 'shape':
        raise ValueError(
            f"{path} looks like a SIMPSON pulse-shape file (2 columns, "
            f"no header), not a FID."
        )

    raw = _read_simp_text(path) if fmt == 'simp' else _read_dat_text(path)
    real0, imag0 = float(raw['real'][0]), float(raw['imag'][0])

    return {'real': real0, 'imag': imag0,
            'magnitude': float(np.hypot(real0, imag0)), 'path': path}


def read_fid_series(paths, x=None, xlabel=None):
    """
    Read the first point of each FID file in `paths` -- e.g. a series
    of single-point pulse-sequence-efficiency acquisitions collected
    across a swept parameter (pulse length, RF power, offset, ...) --
    and collect them into arrays ready for plot_fid_series.

    Parameters
    ----------
    paths : list of str
        FID file paths, in the order corresponding to `x`.
    x : array-like, optional
        The swept quantity, one value per path (e.g. pulse durations
        in \u00b5s, or RF amplitudes in kHz). If omitted, x is
        1..len(paths).
    xlabel : str, optional
        Axis label carried through to plot_fid_series (e.g.
        'Pulse length (\u00b5s)').

    Returns
    -------
    dict : 'x', 'real', 'imag', 'magnitude' (ndarrays, one entry per
    path, in `paths` order), 'xlabel', 'paths'.
    """
    if x is not None and len(x) != len(paths):
        raise ValueError(
            f"x has {len(x)} values but {len(paths)} paths were given -- "
            f"they must correspond 1:1."
        )

    real = np.empty(len(paths))
    imag = np.empty(len(paths))
    magnitude = np.empty(len(paths))
    for i, p in enumerate(paths):
        point = read_single_point_fid(p)
        real[i], imag[i], magnitude[i] = point['real'], point['imag'], point['magnitude']

    x_arr = np.asarray(x, dtype=float) if x is not None else np.arange(1, len(paths) + 1, dtype=float)

    return {'x': x_arr, 'real': real, 'imag': imag, 'magnitude': magnitude,
            'xlabel': xlabel, 'paths': list(paths)}


# ============================================================
# Plotting: stacked spectra from pre-imported result dicts
# ============================================================

def plot_1d_stack(spectra, labels=None, linestyles=None, colours=None,
                   xlims='default', xlabel=None, pl_width=3.375, pl_height=3.375,
                   normalise=True, yoffset=0.2, font='Arial', fontsize=12,
                   x_unit='ppm', output_format='',
                   tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                   tick_number=None, field='', legend=False,
                   save_fig=False, fig_name='nmr_1d_stack', save_format='png'):
    """
    Stacked plot of a list of already-imported 1D spectrum dicts --
    from NMR1dimport, SIMPSONimport, CSDMimport, or anything else
    producing the same schema (x_ppm/x_Hz, real_spectrum,
    normalised_real_spectrum, nuclide, SFO1). This is the mixed-source
    counterpart to one_d.NMR1dplot, which only accepts Bruker
    directory paths -- see this module's docstring for why the two
    aren't unified.

    Parameters
    ----------
    spectra : list of dict
        Result dicts to plot, in stacking order.
    labels : list of str, optional
        Legend label for each spectrum (e.g. "experimental",
        "SIMPSON sim"). If omitted, no per-spectrum legend labels are
        set (but see `legend`).
    linestyles : list of str, optional
        Matplotlib linestyle per spectrum (e.g. dash simulated spectra
        vs solid experimental ones). Defaults to solid for all.
    colours : list, optional
        Explicit colour per spectrum. If omitted, falls back to
        get_stack_colormap('black') stepped across the stack (same
        default as NMR1dplot).
    xlims, xlabel, pl_width, pl_height, normalise, yoffset, font,
    fontsize, x_unit, output_format, tick_dir, tick_length,
    tick_width, tick_pad, tick_number, field, save_fig, fig_name,
    save_format : optional
        Mirror the corresponding NMR1dplot parameters.
    legend : bool, optional
        Force-draw a legend even if no per-spectrum `labels` were
        given (matplotlib will just omit unlabelled lines).

    Returns
    -------
    fig, ax
    """
    if not spectra:
        raise ValueError("spectra must contain at least one result dict.")

    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height, output_format=output_format)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    n = len(spectra)

    if colours is None:
        cmap = get_stack_colormap('black')
        norm = plt.Normalize(vmin=0, vmax=max(n - 1, 1))
        scalarMap = cmx.ScalarMappable(norm=norm, cmap=cmap)
        colours = [scalarMap.to_rgba(i) for i in range(n)]

    if linestyles is None:
        linestyles = ['solid'] * n

    if labels is None:
        labels = [None] * n

    fig, ax = plt.subplots()

    yBase = 0
    xmin = xmax = labeltext = None

    for i, results in enumerate(spectra):
        x = results.get('x_ppm') if x_unit == 'ppm' else results.get('x_Hz')
        if x is None:
            raise ValueError(
                f"Spectrum {i} ({results.get('path')}) has no x_{x_unit} "
                f"axis -- for SIMPSON/CSDM imports this means "
                f"nuclide/field/SFO1 wasn't supplied at import time."
            )
        y = results['normalised_real_spectrum'] if normalise else results['real_spectrum']

        this_xmin, this_xmax = x[0], x[-1]
        if this_xmin > this_xmax:
            this_xmin, this_xmax = this_xmax, this_xmin
        xmin = this_xmin if xmin is None else min(xmin, this_xmin)
        xmax = this_xmax if xmax is None else max(xmax, this_xmax)

        if results.get('nuclide'):
            nucgamma = find_gamma(results['nuclide'], field=field, SFO1=results.get('SFO1'))
            labeltext = NUC_label(nucgamma['Nuclide'], x_unit=x_unit)

        ax.plot(x, y + yBase, color=colours[i], linestyle=linestyles[i],
                linewidth=1.5, label=labels[i])

        yBase += max(y) * (1 + yoffset)

    if xlims == 'default':
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(xlims[0], xlims[1])

    ax.set_xlabel(xlabel if xlabel is not None else labeltext)
    apply_spine_style(ax, top=False, right=False, bottom=True, left=False)
    ax.set_yticks([])
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad, num_ticks=tick_number, axis='x')
    ax.invert_xaxis()

    if legend or any(labels):
        ax.legend(frameon=False)

    fig.set_figheight(pl_height)
    fig.set_figwidth(pl_width)
    fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


# ============================================================
# Plotting: pulse shape
# ============================================================

def plot_shape(path, duration_us=None, pl_width=3.375, pl_height=2.5,
               font='Arial', fontsize=12, save_fig=False,
               fig_name='pulse_shape', save_format='png'):
    """
    Plot amplitude and phase of a SIMPSON pulse-shape file as two
    stacked panels. Deliberately separate from plot_1d_stack: a pulse
    shape isn't a spectrum (x-axis is point index/time, not
    frequency; data is amplitude/phase, not real/imaginary intensity)
    and overlaying it with spectra would misrepresent it.

    Parameters
    ----------
    path : str
    duration_us : float, optional
        Total pulse duration in microseconds. If given, the x-axis is
        time (us); if omitted, it's point index.

    Returns
    -------
    fig, (ax_amplitude, ax_phase)
    """
    shape = read_shape(path)
    n = shape['npoints']
    x = np.linspace(0, duration_us, n) if duration_us else np.arange(n)
    xlabel = 'Time (\u00b5s)' if duration_us else 'Point index'

    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, shape['amplitude'], color='black', linewidth=1.2)
    ax1.set_ylabel('Amplitude')
    apply_spine_style(ax1, top=False, right=False)

    ax2.plot(x, shape['phase'], color='black', linewidth=1.2)
    ax2.set_ylabel('Phase (\u00b0)')
    ax2.set_xlabel(xlabel)
    apply_spine_style(ax2, top=False, right=False)

    fig.set_figheight(pl_height)
    fig.set_figwidth(pl_width)
    fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, (ax1, ax2)


# ============================================================
# Plotting: FID and its apodization window
# ============================================================

def plot_fid(fid, apodization=None, apod_params=None, show_imag=True,
             show_window=True, overlay_apodized=True,
             pl_width=3.375, pl_height=2.5, font='Arial', fontsize=12,
             tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
             save_fig=False, fig_name='fid_plot', save_format='png'):
    """
    Plot a raw FID (as read by read_fid) against time, optionally with
    an apodization window (see apodization_window) and the resulting
    apodized FID overlaid -- so the effect of a chosen window can be
    checked before it's baked into fft_simpson_fid.

    Parameters
    ----------
    fid : dict
        Output of read_fid.
    apodization : str, optional
        Window kind (see apodization_window). If omitted/None, only
        the raw FID is plotted.
    apod_params : dict, optional
        Extra kwargs for apodization_window (lb, gb, sine_offset,
        sine_end).
    show_imag : bool, optional
        Also plot the imaginary channel of the raw FID. Default True.
    show_window : bool, optional
        Overlay the window function itself, scaled to the FID's peak
        amplitude so its shape is visible against the decay. Default
        True (only has an effect if `apodization` is given).
    overlay_apodized : bool, optional
        Also overlay the apodized real FID. Default True (only has an
        effect if `apodization` is given).
    pl_width, pl_height, font, fontsize, tick_dir, tick_length,
    tick_width, tick_pad, save_fig, fig_name, save_format : optional
        Mirror the plotting conventions used elsewhere in this module.

    Returns
    -------
    fig, ax
    """
    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    fig, ax = plt.subplots()
    t_us = fid['t'] * 1e6

    ax.plot(t_us, fid['real'], color='black', linewidth=1.0, label='real')
    if show_imag:
        ax.plot(t_us, fid['imag'], color='grey', linewidth=1.0,
                linestyle='--', label='imag')

    if apodization and apodization != 'none':
        apod_params = apod_params or {}
        _, window = apodization_window(fid['NP'], fid['dt'], kind=apodization, **apod_params)

        if show_window:
            peak = np.max(np.abs(fid['real']))
            scale = peak if peak > 0 else 1.0
            ax.plot(t_us, window * scale, color='C1', linewidth=1.2,
                    linestyle=':', label=f'{apodization} window')

        if overlay_apodized:
            _, real_a, _, _ = apply_apodization(fid['real'], fid['imag'], fid['SW'],
                                                 kind=apodization, **apod_params)
            ax.plot(t_us, real_a, color='C0', linewidth=1.0,
                    label='apodized (real)')

    ax.set_xlabel('Time (\u00b5s)')
    ax.set_ylabel('Intensity')
    apply_spine_style(ax, top=False, right=False)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad)
    ax.legend(frameon=False, fontsize=fontsize * 0.7)

    fig.set_figheight(pl_height)
    fig.set_figwidth(pl_width)
    fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax


# ============================================================
# Plotting: single-point FID series (e.g. pulse-sequence efficiency)
# ============================================================

def plot_fid_series(series, y='magnitude', xlabel=None, ylabel=None,
                     normalise=False, marker='o', linestyle='-', color='black',
                     pl_width=3.375, pl_height=2.5, font='Arial', fontsize=12,
                     tick_dir='in', tick_length=4.0, tick_width=1.0, tick_pad=4.0,
                     save_fig=False, fig_name='fid_series', save_format='png'):
    """
    Plot the first-point intensity from a series of single-point FIDs
    (see read_fid_series) against the swept quantity -- e.g. a
    nutation curve, an RF power-efficiency curve, an offset profile.

    Parameters
    ----------
    series : dict
        Output of read_fid_series.
    y : {'magnitude', 'real', 'imag'}, optional
        Which component of the first point to plot. Default
        'magnitude' (sign-insensitive, usually what you want for a
        pure efficiency check; use 'real' if phase/sign matters, e.g.
        a nutation curve that should go negative).
    xlabel, ylabel : str, optional
        Override axis labels. `xlabel` falls back to
        series['xlabel'], then plain 'x'; `ylabel` falls back to
        y.capitalize().
    normalise : bool, optional
        Scale y to [0, 1]. Default False.
    marker, linestyle, color : optional
        Mirror matplotlib's own plot() kwargs.
    pl_width, pl_height, font, fontsize, tick_dir, tick_length,
    tick_width, tick_pad, save_fig, fig_name, save_format : optional
        Mirror the plotting conventions used elsewhere in this module.

    Returns
    -------
    fig, ax
    """
    if y not in ('magnitude', 'real', 'imag'):
        raise ValueError("y must be 'magnitude', 'real', or 'imag'.")

    yvals = series[y]
    if normalise:
        yvals = _normalise(yvals)

    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    fig, ax = plt.subplots()
    ax.plot(series['x'], yvals, color=color, marker=marker, linestyle=linestyle,
            linewidth=1.2)

    ax.set_xlabel(xlabel if xlabel is not None else (series.get('xlabel') or 'x'))
    ax.set_ylabel(ylabel if ylabel is not None else y.capitalize())
    apply_spine_style(ax, top=False, right=False)
    apply_tick_style(ax, direction=tick_dir, length=tick_length, width=tick_width,
                      pad=tick_pad)

    fig.set_figheight(pl_height)
    fig.set_figwidth(pl_width)
    fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, ax
