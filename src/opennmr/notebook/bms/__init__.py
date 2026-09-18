"""Bulk magnetic susceptibility (BMS) shift simulation.

Notebook-friendly entry points:
    from bms import run_bms, make_particle

    particles = [
        make_particle(x=0, y=0, z=0, size=(50, 50, 50)),
    ]
    df = run_bms(particles, magnetic_susceptibility=(2.87e-3, 3.13e-3, 2.27e-3))

See main.run_bms for the full set of options. File-based usage
(bms.main.run(path_to_json, out_dir)) still works unchanged.
"""
from .main import run, run_bms
from .io_utils import make_particle, particles_to_array, build_box

__all__ = ["run", "run_bms", "make_particle", "particles_to_array", "build_box"]
