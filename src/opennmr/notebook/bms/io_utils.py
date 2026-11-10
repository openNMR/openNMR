"""Ported from inp_part.m and inp_sim.m, plus a small LAMMPS-dump reader
that main.m left to MATLAB's fopen/textscan."""
import json
import numpy as np
from .geometry import quat_to_rotmat


def read_particles_dump(filename):
    """Parses the LAMMPS 'ITEM: ATOMS ...' block. Returns an (N,11) array:
    columns = [id, x, y, z, v1, v2, v3, qi, qj, qk, qr]  (same layout
    inp_part.m assumed: cols 2-4 center, 5-7 semi-axes, 8-10 quat vector
    part, 11 quat scalar part / OrientationW).

    Reads exactly the N atoms declared under 'ITEM: NUMBER OF ATOMS' for
    the FIRST timestep in the file. Previously this read every non-blank
    line from 'ITEM: ATOMS' to end-of-file, which only worked by luck on
    a single-timestep dump: a real LAMMPS trajectory dump with more than
    one timestep would hit a second 'ITEM: TIMESTEP'/'ITEM: BOX BOUNDS'
    header line and crash with a ValueError out of float(), or -- worse,
    if a later section happened to tokenize as the right number of
    floats -- silently return wrong data mixed in from another
    timestep."""
    with open(filename) as f:
        lines = f.readlines()

    n_atoms = None
    start = None
    for i, line in enumerate(lines):
        if "ITEM: NUMBER OF ATOMS" in line:
            n_atoms = int(lines[i + 1])
        if "ITEM: ATOMS" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError("No 'ITEM: ATOMS' section found in dump file")
    if n_atoms is None:
        raise ValueError("No 'ITEM: NUMBER OF ATOMS' section found in dump file")

    data_lines = [l for l in lines[start:start + n_atoms] if l.strip()]
    if len(data_lines) < n_atoms:
        raise ValueError(
            f"Dump header declares {n_atoms} atoms but only {len(data_lines)} "
            f"data lines were found before end of file – dump may be truncated."
        )
    rows = [list(map(float, l.split())) for l in data_lines]
    return np.array(rows)


def make_particle(x, y, z, size, quaternion=(1.0, 0.0, 0.0, 0.0), id=None):
    """Build one particle record for the `particles=[...]` argument of
    `bms.main.run_bms()` (or for `particles_to_array()` directly) --
    no dump file required.

    x, y, z    : particle center, in the same units as your simulation box.
    size       : (v1, v2, v3) -- meaning depends on the "equation" you pass
                 to run_bms():
                   ellipsoid -> (semi-axis-x, semi-axis-y, semi-axis-z)
                   hexagonal -> (hex radius, unused, height)
                   donut     -> (major radius R0, minor radius r0, unused)
                   cuboid    -> (half side x, half side y, half side z);
                                a cube of edge 50: (25, 25, 25)
    quaternion : orientation as (qw, qx, qy, qz), scalar-first. Default
                 (1,0,0,0) is "no rotation". Needn't be pre-normalized.
    id         : optional integer particle id; auto-numbered if omitted.
    """
    qw, qx, qy, qz = quaternion
    v1, v2, v3 = size
    return {
        "id": id, "x": x, "y": y, "z": z,
        "v1": v1, "v2": v2, "v3": v3,
        "qi": qx, "qj": qy, "qk": qz, "qr": qw,
    }


def particles_to_array(particles):
    """Normalize the `particles=[...]` argument into the (N,11) array
    [id, x, y, z, v1, v2, v3, qi, qj, qk, qr] that `inp_part` expects --
    the same layout `read_particles_dump` produces from a dump file.

    Accepts, in order of preference:
      - a list of dicts from `make_particle(...)`
      - a list/array of raw rows, either 10 columns
        (x, y, z, v1, v2, v3, qi, qj, qk, qr) or 11 columns with a
        leading id column, matching the LAMMPS dump layout.
    """
    particles = list(particles)
    if not particles:
        raise ValueError("particles is empty")

    if isinstance(particles[0], dict):
        rows = []
        for i, p in enumerate(particles):
            pid = p.get("id")
            pid = i + 1 if pid is None else pid
            rows.append([
                pid, p["x"], p["y"], p["z"],
                p["v1"], p["v2"], p["v3"],
                p.get("qi", 0.0), p.get("qj", 0.0), p.get("qk", 0.0), p.get("qr", 1.0),
            ])
        return np.array(rows, dtype=float)

    arr = np.asarray(particles, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (10, 11):
        raise ValueError(
            "particles array must have 10 columns (x,y,z,v1,v2,v3,qi,qj,qk,qr) "
            "or 11 columns with a leading id column"
        )
    if arr.shape[1] == 10:
        ids = np.arange(1, arr.shape[0] + 1, dtype=float)[:, None]
        arr = np.hstack([ids, arr])
    return arr


def inp_part(properties, shape):
    """Returns a list of dicts, one per particle (mirrors inp_part.m's
    cell array of structs).

    properties["data"], if present, is used directly (via
    particles_to_array) instead of reading properties["filename"] from
    disk -- this is what lets run_bms() work entirely from in-memory
    Python objects."""
    sl = shape["slope_filter"]
    r = shape["r"]
    t = shape["t"]
    thresh = shape["threshold"]
    Chi = np.asarray(properties["Magnetic_Susceptibility"], dtype=float)
    equation = shape["Equation"]
    deformation = shape["Deformation"]

    if properties.get("data") is not None:
        data = particles_to_array(properties["data"])
    else:
        data = read_particles_dump(properties["filename"])

    particles = []
    for row in data:
        v1, v2, v3 = row[4], row[5], row[6]
        # The far-field dipole volume. A block's own (its corners make it
        # 6/π times the inscribed ellipsoid's); the other shapes keep the
        # ellipsoid formula the original code used (see packing._particle_volume).
        vol = 8.0 * v1 * v2 * v3 if equation == "cuboid" else (4.0 / 3.0) * np.pi * v1 * v2 * v3
        posc = row[1:4]
        qi, qj, qk, qr = row[7], row[8], row[9], row[10]
        q = np.array([qr, qi, qj, qk])
        q = q / np.linalg.norm(q)

        R = quat_to_rotmat(q)
        Chi_tensor = R.T @ np.diag(Chi) @ R

        particles.append({
            "Chi_PA": Chi,
            "Chi_tensor": Chi_tensor,
            "v1": v1, "v2": v2, "v3": v3,
            "vol": vol,
            "q": q,
            "r": r, "t": t, "sl": sl,
            "posc": posc,
            "Equation": equation,
            "Deformation": deformation,
            "thresh": thresh,
        })
    return particles


def build_box(size, gridpoints=128):
    """Build the "Simulation" dict for run_bms()/inp_sim() from a box size
    instead of six explicit Xmin/Xmax/.../Zmax bounds. The box is always
    centered on the origin.

    size : either a single number (cube of that full side length, applied
           to all three axes) or (sizex, sizey, sizez).
    """
    if np.isscalar(size):
        sizex = sizey = sizez = float(size)
    else:
        sizex, sizey, sizez = (float(v) for v in size)
    return {
        "Xmin": -sizex / 2, "Xmax": sizex / 2,
        "Ymin": -sizey / 2, "Ymax": sizey / 2,
        "Zmin": -sizez / 2, "Zmax": sizez / 2,
        "gridpoints": gridpoints,
    }


def inp_sim(simulation):
    """NOTE: fixes the bug in inp_sim.m where Ymin/Ymax/Zmin/Zmax were
    silently overwritten with Xmin/Xmax (see writeup) -- this version
    actually uses the Y/Z bounds from the JSON."""
    sizex = simulation["Xmax"] - simulation["Xmin"]
    sizey = simulation["Ymax"] - simulation["Ymin"]
    sizez = simulation["Zmax"] - simulation["Zmin"]
    gridpoints = simulation["gridpoints"]

    stepx, stepy, stepz = sizex / gridpoints, sizey / gridpoints, sizez / gridpoints

    x = np.arange(simulation["Xmin"], simulation["Xmax"] + stepx / 2, stepx)
    y = np.arange(simulation["Ymin"], simulation["Ymax"] + stepy / 2, stepy)
    z = np.arange(simulation["Zmin"], simulation["Zmax"] + stepz / 2, stepz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    return {
        "sizex": sizex, "sizey": sizey, "sizez": sizez,
        "gridpoints": gridpoints,
        "evalpoints": points,
    }


def read_input(json_path):
    with open(json_path) as f:
        return json.load(f)
