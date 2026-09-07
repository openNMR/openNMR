"""
Standalone check of what geometry.quat_to_rotmat + geometry._to_local_frame
actually compute, and a template for validating the orientation convention
against a known case.

Part 1 (always run, no ground truth needed): confirms the algebraic identity
described in geometry.py's module docstring --

    pts @ quat_to_rotmat(q)  ==  R_std(q) @ pt   for every point pt

where R_std(q) is the textbook (non-conjugated) quaternion-to-rotation-matrix,
built independently here so it isn't just checking the code against itself.

Part 2 (a template, not a finished test): shows how to sanity-check the
convention against a case you *can* reason about by eye -- a long, thin
ellipsoid rotated by a known angle about a known axis. Fill in the
"expected" direction for your own dump's quaternion convention (body->world
or world->body) and confirm the mask agrees. This script cannot tell you
which convention your particle dump actually uses -- only you (or a
validated MATLAB reference run) can.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bms.geometry import quat_to_rotmat, ellipsoid_mask


def R_std_textbook(q):
    """Independent, textbook quaternion->rotation-matrix (w,x,y,z), with
    NO conjugation/negation trick -- used only to cross-check the
    production code, not as a replacement for it."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2+z**2),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2+z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2+y**2)],
    ])


def part1_algebraic_identity():
    print("Part 1: checking pts @ quat_to_rotmat(q) == R_std(q) @ pt ...")
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        p = rng.normal(size=3)

        R_code = quat_to_rotmat(q)           # production function
        lhs = p @ R_code                     # what _to_local_frame computes
        rhs = R_std_textbook(q) @ p          # plain textbook rotation, no tricks

        max_err = max(max_err, np.max(np.abs(lhs - rhs)))
    print(f"  max abs error over 200 random quaternions/points: {max_err:.3e}")
    assert max_err < 1e-10, "identity does NOT hold -- geometry.py's net convention has changed!"
    print("  PASS: current code's net effect is pts@R == R_std(q) @ pt, as documented.")


def part2_orientation_sanity_check_template():
    print()
    print("Part 2: orientation sanity-check template (needs your judgement)")
    print("-" * 70)
    # A cigar-shaped particle: long axis is v3 (z, before rotation).
    sim_part = {"r": 2, "t": 2, "sl": 50, "v1": 5.0, "v2": 5.0, "v3": 40.0}

    # Rotate 90 deg about the world x-axis: q = [cos(45deg), sin(45deg), 0, 0]
    theta = np.pi / 2
    sim_part["q"] = np.array([np.cos(theta/2), np.sin(theta/2), 0.0, 0.0])

    # Sample the mask along the three world axes, near the origin, to see
    # which world direction the long axis now points along.
    axis_pts = {
        "x": (np.array([15.0]), np.array([0.0]), np.array([0.0])),
        "y": (np.array([0.0]), np.array([15.0]), np.array([0.0])),
        "z": (np.array([0.0]), np.array([0.0]), np.array([15.0])),
    }
    for label, (x, y, z) in axis_pts.items():
        val = ellipsoid_mask(sim_part, x, y, z)[0]
        print(f"  mask value 15 units out along world {label}-axis: {val:.4f}"
              f"  ({'inside/near particle -> long axis is here' if val > 0.3 else 'outside -> not the long axis'})")

    print()
    print("  A 90 deg rotation about world x should swap which of world y/z")
    print("  the long (v3) axis points along, leaving world x unaffected.")
    print("  Confirm the printed pattern matches what YOUR quaternion")
    print("  convention (body->world vs world->body, as produced by your")
    print("  particle generator) predicts -- this script does not know that")
    print("  convention for you.")


if __name__ == "__main__":
    part1_algebraic_identity()
    part2_orientation_sanity_check_template()
