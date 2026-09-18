"""
bms_spectrum.py
================

User-facing entry point for the Bulk Magnetic Susceptibility (BMS) shift
calculator (the ``bms`` sub-package: geometry/demag/shift/simulate/main/
mas_sidebands/spectrum), packaged to match the calling convention already
used by the other two spectrum producers in nmrtools:

    NMR1dimport(...)            -> results               (one_d.py, experimental)
    generate_spectrum_dict(...) -> (spectrum_dict, table) (pnmr_simulator.py, CIF-predicted)
    generate_bms_spectrum_dict(...) -> (spectrum_dict, table)   (here, BMS-predicted)

``spectrum_dict`` always has the same 10 keys:

    x_ppm, x_Hz, real_spectrum, normalised_real_spectrum,
    imag_spectrum, normalised_imag_spectrum, nuclide, BF1, SFO1, path

so a BMS-predicted lineshape plots and compares with
plotting.plot_spectrum_dict / plot_comparison / plot_stacked_spectra
exactly like an experimental or CIF-predicted one -- no BMS-specific
plotting code is needed anywhere.

What changed relative to the ported MATLAB-derived functions
--------------------------------------------------------------
bms.main.run_bms() (particles -> per-voxel DataFrame) and
bms.spectrum.assemble_spectrum() (per-voxel DataFrame -> spectrum arrays)
are both kept exactly as they were -- they're the validated numerical
core and there's no reason to touch them just for a calling-convention
change. What was missing was the layer generate_spectrum_dict() already
has for CIF-pNMR: a single top-level driver that (a) derives BF1/SFO1
from a nuclide + field via the *same* gyromagnetic-ratio table as every
other spectrum in the package, instead of requiring the caller to
compute and supply BF1/SFO1 themselves, and (b) returns the standard
dict shape directly, with the per-voxel table as a secondary diagnostic
return value (like generate_spectrum_dict's shift_table), rather than
requiring two separate manual calls (run_bms(), then assemble_spectrum())
with no shared referencing logic between them. That's what this module
adds.
"""

import os

from .utils import compute_BF1_SFO1
from .bms.main import run_bms as _run_bms_raw
from .bms.spectrum import assemble_spectrum


def _resolve_bms_output_dir(output_dir):
    """Mirrors pnmr_simulator._resolve_output_dir's pattern, but there is
    no source file to anchor a default location to here (BMS input is
    normally an in-memory particle list from io_utils.make_particle, not
    a file on disk) -- so the default is <cwd>/bms_simulation rather than
    "next to the input file"."""
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "bms_simulation")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_bms_spectrum_dict(
    particles,
    magnetic_susceptibility,
    nuclide,
    B0_tesla,
    spin_rate_hz,
    box_size=300.0,
    gridpoints=128,
    r=2,
    t=2,
    slope_filter=2000,
    threshold=0.99,
    equation="hexagonal",
    deformation="A",
    local_shift_contribution=None,
    rotor_contribution=None,
    offset_ppm=0.0,
    gamma_MHz_per_T=None,
    linewidth_hz=50.0,
    n_points=16384,
    sw_ppm=200.0,
    max_order=8,
    mu_grid=None,
    eta_grid=None,
    output_dir=None,
    save_voxel_table=True,
    n_jobs=None,
):
    """
    Run the full particle-packing -> demagnetizing-field -> BMS-shift ->
    MAS-sideband-broadened-spectrum pipeline, and package the result as
    a standard nmrtools spectrum_dict.

    Parameters
    ----------
    particles, magnetic_susceptibility, box_size, gridpoints, r, t,
    slope_filter, threshold, equation, deformation,
    local_shift_contribution, rotor_contribution, n_jobs
        Passed straight through to bms.main.run_bms() -- see that
        function's docstring, and io_utils.make_particle for how to
        build `particles` from plain Python objects (no LAMMPS dump
        file required), for what each of these means.

    nuclide : str
        NMR-isotope label used for referencing, e.g. "7Li", "27Al" --
        same lookup (utils.compute_BF1_SFO1 / find_gamma) used by
        NMR1dimport and generate_spectrum_dict, so a BMS-predicted
        spectrum references identically to an imported experimental one.

    B0_tesla : float
        Magnetic field strength of the (simulated) spectrometer, used
        with the nuclide's gyromagnetic ratio to compute BF1/SFO1.

    spin_rate_hz : float
        MAS spin rate in Hz, used both to convert delta (ppm) to reduced
        sideband order mu, and as the sideband spacing on the ppm axis.

    offset_ppm : float, optional
        Carrier offset from BF1, in ppm (0 => SFO1 == BF1). See
        utils.compute_BF1_SFO1.

    gamma_MHz_per_T : float, optional
        Override the built-in gyromagnetic-ratio table lookup entirely.
        See utils.compute_BF1_SFO1.

    linewidth_hz : float, optional
        Lorentzian linewidth (Hz) applied as a time-domain T2 decay.
        Deliberately NOT called `broadening` (the name
        generate_spectrum_dict uses for its Gaussian, ppm-valued
        broadening) and deliberately in Hz, not ppm: BMS spectra are
        built via histogram -> IFFT -> T2 decay -> FFT specifically so
        MAS spinning-sideband manifolds come out with the right relative
        intensities (see bms/mas_sidebands.py and bms/spectrum.py's
        module docstrings) -- a materially different lineshape mechanism
        from the CIF-pNMR pipeline's per-stick Gaussian, not merely a
        naming choice. Reusing `broadening` here would invite silently
        passing a ppm-sized number where a Hz-sized number is needed.

    n_points, sw_ppm : int, float, optional
        Number of points / total ppm sweep width of the output ppm axis
        (centered on 0 ppm). See bms.spectrum.assemble_spectrum.

    max_order, mu_grid, eta_grid : optional
        Sideband-manifold lookup-table resolution -- see
        bms.spectrum.assemble_spectrum / bms.mas_sidebands.sideband_manifold.

    output_dir : str, optional
        Directory used as the returned spectrum_dict['path'] and (if
        `save_voxel_table`) to save the voxel table. Defaults to
        <cwd>/bms_simulation -- see _resolve_bms_output_dir above for why
        this differs from generate_spectrum_dict's default.

    save_voxel_table : bool, optional
        Write the per-voxel (x, y, z, d_iso, delta, eta, N1, N2, N3)
        table to ``<output_dir>/<nuclide>_bms_voxel_table.csv`` -- the
        BMS equivalent of generate_spectrum_dict's
        ``save_shift_table``/``<nuclide>_shift_table.csv``.

    n_jobs : int, optional
        Worker processes for the per-particle demag/shift loop. None
        (default) -> os.cpu_count(). See bms.main.run_bms.

    Returns
    -------
    (spectrum_dict, voxel_table)
        spectrum_dict : the standard 10-key dict (see module docstring),
            directly usable with plotting.plot_spectrum_dict /
            plot_comparison / plot_stacked_spectra.
        voxel_table : the per-evaluation-point DataFrame from
            bms.main.run_bms() (x, y, z, d_iso, delta, eta, N1, N2, N3) --
            kept for inspection/diagnostics, and (by default) written to
            disk, exactly as generate_spectrum_dict returns its
            shift_table alongside the packaged spectrum.
    """
    BF1, SFO1, nuclide = compute_BF1_SFO1(
        nuclide, B0_tesla, offset_ppm=offset_ppm, gamma_MHz_per_T=gamma_MHz_per_T,
    )

    path = _resolve_bms_output_dir(output_dir)

    voxel_table = _run_bms_raw(
        particles=particles,
        magnetic_susceptibility=magnetic_susceptibility,
        box_size=box_size,
        gridpoints=gridpoints,
        r=r,
        t=t,
        slope_filter=slope_filter,
        threshold=threshold,
        equation=equation,
        deformation=deformation,
        local_shift_contribution=local_shift_contribution,
        rotor_contribution=rotor_contribution,
        out_dir=None,  # we write the voxel table ourselves, below, with a nuclide-labelled name
        n_jobs=n_jobs,
    )

    spectrum_dict = assemble_spectrum(
        voxel_table,
        spin_rate_hz=spin_rate_hz,
        BF1=BF1,
        SFO1=SFO1,
        nuclide=nuclide,
        path=path,
        n_points=n_points,
        sw_ppm=sw_ppm,
        linewidth_hz=linewidth_hz,
        max_order=max_order,
        mu_grid=mu_grid,
        eta_grid=eta_grid,
    )
    # bms.spectrum.assemble_spectrum stores 'path' as a pathlib.Path;
    # NMR1dimport/generate_spectrum_dict both store it as a plain str.
    # Normalised here rather than inside bms/spectrum.py, to avoid
    # touching that module's validated internals for a cosmetic type fix.
    spectrum_dict["path"] = str(spectrum_dict["path"])

    if save_voxel_table:
        voxel_table.to_csv(
            os.path.join(path, f"{nuclide}_bms_voxel_table.csv"),
            index=False, float_format="%.6f",
        )

    return spectrum_dict, voxel_table
