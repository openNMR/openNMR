"""Ported from main.m / print_summed_simu.m.

NOTE: main.m used two different "inside" thresholds -- sim_part.thresh
(0.99, from JSON, used inside shielding_bms.m) vs a hardcoded 0.9 (used
here to decide whether to add Local_Shift_Contribution/Rotor_Contribution).
That inconsistency is reproduced as-is (see INSIDE_SUM_THRESH below) --
worth deciding deliberately rather than me silently "fixing" it, since I
don't know which was intentional.

PARALLELIZATION: each particle's (demag_calc + evaluate) is completely
independent of every other particle -- they're only combined afterward
by summation -- so the per-particle loop is run across a process pool by
default. Each worker builds the box's k-space kernel exactly ONCE (via
_init_worker, called once per worker process, not per particle) rather
than every particle rebuilding it from scratch, which was the previous
behaviour even in serial code.

Windows/notebook note: ProcessPoolExecutor uses the 'spawn' start method
on Windows, which requires the target function to be importable from a
module rather than defined inline in a notebook cell. _run_one_particle
below lives in this module for exactly that reason -- calling
`from bms.main import run; run(...)` from a notebook works, but do not
copy _run_one_particle/_init_worker into a notebook cell yourself, or
multiprocessing will fail to pickle it on Windows.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .io_utils import inp_part, inp_sim, read_input, build_box
from .demag import build_kspace_kernel, demag_calc
from .shift import tensor2haeberlen

INSIDE_SUM_THRESH = 0.9  # matches main.m's hardcoded value; see note above

# Per-worker-process globals, set once by _init_worker -- NOT shared across
# processes, and not meaningful in the parent process itself.
_worker_sim_str = None
_worker_kernel = None


def _init_worker(sim_str):
    """Runs once per worker process when the pool starts (not once per
    particle) -- this is where the shared, box-dependent k-space kernel
    gets built exactly once per process."""
    global _worker_sim_str, _worker_kernel
    _worker_sim_str = sim_str
    _worker_kernel = build_kspace_kernel(sim_str)


def _run_one_particle(sim_part):
    from .simulate import evaluate  # local import: avoid import cost/cycles at worker startup
    demag_tensor, grid_vecs = demag_calc(_worker_sim_str, sim_part, kernel=_worker_kernel)
    r = evaluate(_worker_sim_str, sim_part, demag_tensor, grid_vecs)
    return r["demag_tensor"], r["shift_tensor"]


def _run_chunk(chunk):
    """Runs an entire slice of particles inside ONE worker call and sums
    them there, so only one (demag_sum, shift_sum) pair -- not one pair
    per particle -- ever crosses the process boundary back to the parent.

    This matters because each pair is a full (n_evalpoints, 3, 3) float64
    array -- e.g. ~144 MB apiece at a 128**3 grid -- and
    ProcessPoolExecutor.map submits every particle's task up front. With
    n_jobs=1 previously, that's n_particles separate large payloads
    pickled through the OS pipe essentially back-to-back; on Windows this
    exhausts pipe-buffer/non-paged-pool resources well before it's a
    Python-level memory problem, surfacing as
    `OSError: [WinError 1450] Insufficient system resources...` rather
    than a normal MemoryError. Chunking by worker cuts the number (and
    total volume) of cross-process transfers from n_particles down to
    n_jobs, with no change to the arithmetic itself."""
    demag_sum = None
    shift_sum = None
    for sim_part in chunk:
        d, s = _run_one_particle(sim_part)
        demag_sum = d if demag_sum is None else demag_sum + d
        shift_sum = s if shift_sum is None else shift_sum + s
    return demag_sum, shift_sum


def _chunk_particles(particles, n_chunks):
    """Split into n_chunks near-equal-sized contiguous slices (last
    chunks absorb the remainder), so each worker gets one balanced batch
    rather than one task per particle."""
    n_chunks = max(1, min(n_chunks, len(particles)))
    base, rem = divmod(len(particles), n_chunks)
    chunks, i = [], 0
    for c in range(n_chunks):
        size = base + (1 if c < rem else 0)
        chunks.append(particles[i:i + size])
        i += size
    return [c for c in chunks if c]


def run(input_data, out_dir=None, n_jobs=None):
    """
    input_data: either a dict with the same shape as the BMS_input*.json
        files ("Particles_properties", "Particles_shape", "Simulation",
        "Local_Shift_Contribution", "Rotor_Contribution"), or a path
        (str/Path) to such a JSON file on disk. Most notebook users will
        want run_bms() below instead, which builds this dict for you.
    out_dir: if given, BMS_out.csv is written there. If None (default),
        nothing is written to disk -- the DataFrame is only returned.
    n_jobs: worker processes for the per-particle loop.
        None (default) -> os.cpu_count().
        1              -> serial, single process (useful for debugging;
                           also avoids process-pool overhead for very
                           small particle counts).
    """
    data = input_data if isinstance(input_data, dict) else read_input(input_data)
    sim_str = inp_sim(data["Simulation"])
    particles = inp_part(data["Particles_properties"], data["Particles_shape"])

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, len(particles)))

    demag_sum = None
    shift_sum = None

    if n_jobs == 1:
        _init_worker(sim_str)
        d, s = _run_chunk(particles)
        demag_sum = d if demag_sum is None else demag_sum + d
        shift_sum = s if shift_sum is None else shift_sum + s
    else:
        chunks = _chunk_particles(particles, n_jobs)
        ctx = mp.get_context("spawn")  # 'spawn' is the only option on Windows anyway;
                                        # used everywhere for consistent behaviour.
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx,
                                  initializer=_init_worker, initargs=(sim_str,)) as ex:
            for d, s in ex.map(_run_chunk, chunks):
                demag_sum = d if demag_sum is None else demag_sum + d
                shift_sum = s if shift_sum is None else shift_sum + s

    local_shift = np.asarray(data["Local_Shift_Contribution"])
    rotor = np.asarray(data["Rotor_Contribution"])
    trace_sum = np.trace(demag_sum, axis1=-2, axis2=-1)
    add_extra = trace_sum > INSIDE_SUM_THRESH
    shift_sum = shift_sum + np.where(add_extra[:, None, None], (local_shift + rotor), 0.0)

    haeb_sum = tensor2haeberlen(shift_sum)
    eigvals = np.linalg.eigvalsh(demag_sum)  # ascending; matches eig()'s diag(D) up to ordering

    df = pd.DataFrame({
        "x": sim_str["evalpoints"][:, 0],
        "y": sim_str["evalpoints"][:, 1],
        "z": sim_str["evalpoints"][:, 2],
        "d_iso": haeb_sum[:, 0],
        "delta": haeb_sum[:, 1],
        "eta": haeb_sum[:, 2],
        "N1": eigvals[:, 0], "N2": eigvals[:, 1], "N3": eigvals[:, 2],
    })

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "BMS_out.csv", index=False, float_format="%.6f")

    return df


def run_bms(
    particles,
    magnetic_susceptibility,
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
    out_dir=None,
    n_jobs=None,
):
    """
    Run the full BMS demagnetization/shift calculation from plain Python
    objects -- no JSON input file and no LAMMPS dump file required. This
    is the entry point meant for interactive/notebook use; it builds the
    same input structure `run()` expects and calls it for you.

    particles: list of particle records, e.g. built with make_particle():
        particles = [
            make_particle(x=0, y=0, z=0, size=(50, 50, 50)),
            make_particle(x=100, y=0, z=0, size=(50, 50, 50),
                          quaternion=(0.707, 0.707, 0, 0)),
        ]
        (A raw (N,10)/(N,11) array in the LAMMPS dump column layout is
        also accepted -- see particles_to_array() in io_utils.py.)

    magnetic_susceptibility: (chi_1, chi_2, chi_3) -- principal-axis
        magnetic susceptibility, same units as the JSON's
        "Magnetic_Susceptibility".

    box_size: full side length of the (cubic, origin-centered) simulation
        box, or (sizex, sizey, sizez) for a non-cubic box.
    gridpoints: FFT grid resolution per axis.

    r, t, slope_filter, threshold, equation, deformation: shape-function
        parameters, same meaning as "Particles_shape" in the JSON (see
        geometry.py for what each Equation/Deformation combination does).
        Defaults here match BMS_input_full.json.

    local_shift_contribution, rotor_contribution: (3,3) arrays added to
        the shift tensor for evaluation points "inside" the summed
        particle shape (see INSIDE_SUM_THRESH above). Default to zero
        matrices if not given -- provide your real values for physically
        meaningful results.

    out_dir, n_jobs: see run().

    Returns the same per-evaluation-point DataFrame as run().
    """
    if local_shift_contribution is None:
        local_shift_contribution = np.zeros((3, 3))
    if rotor_contribution is None:
        rotor_contribution = np.zeros((3, 3))

    input_data = {
        "Particles_properties": {
            "data": particles,
            "Magnetic_Susceptibility": list(magnetic_susceptibility),
        },
        "Particles_shape": {
            "r": r,
            "t": t,
            "slope_filter": slope_filter,
            "threshold": threshold,
            "Equation": equation,
            "Deformation": deformation,
        },
        "Simulation": build_box(box_size, gridpoints),
        "Local_Shift_Contribution": np.asarray(local_shift_contribution).tolist(),
        "Rotor_Contribution": np.asarray(rotor_contribution).tolist(),
    }

    return run(input_data, out_dir=out_dir, n_jobs=n_jobs)


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "BMS_input.json",
        sys.argv[2] if len(sys.argv) > 2 else "./test_out")