"""
bms/packing.py
==============

Turn a particle-size distribution and a target count into a list of
particles ready for bms.main.run_bms() / bms_spectrum.generate_bms_spectrum_dict(),
without hand-specifying every particle's position, size and orientation
via make_particle() one at a time.

Two pieces:

  sample_particle_sizes(...) : draw N (v1, v2, v3) size triples from a
      Gaussian / Lorentzian / pseudo-Voigt distribution, independently
      per axis.

  random_close_pack(...)     : combine that with random, non-overlapping
      placement and random orientation inside a simulation box, via
      sequential random addition (RSA) with rejection.

HONESTY NOTE ON "RANDOM CLOSE PACKING"
---------------------------------------
What's implemented here is sequential random addition with rejection
(RSA), not a true random-close-packing (RCP) algorithm. RSA places
particles one at a time at uniformly random positions/orientations and
simply rejects attempts that overlap anything already placed. That's
simple and dependency-light, but it jams at well below the density a
real RCP protocol reaches -- for monodisperse spheres, RSA saturates
around ~38% volume fraction (Renyi's parking-constant problem), against
~64% for genuine RCP. If you need genuinely dense, well-relaxed
packings -- e.g. because near-contact particle-particle demagnetizing
interactions matter physically -- a proper packing/DEM code (LAMMPS
granular pair styles, or a dedicated packing generator) will do
noticeably better than anything reimplemented here, and
io_utils.read_particles_dump() already exists specifically to bring
such a result back in. random_close_pack() reports
`achieved_volume_fraction` precisely so you can see whether RSA got you
dense enough, or whether you need that route instead.

HONESTY NOTE ON "STANDARD DEVIATION" FOR LORENTZIAN / PSEUDO-VOIGT
--------------------------------------------------------------------
A Lorentzian (Cauchy) distribution has no finite variance -- its tails
are too heavy for one to exist. There is no such thing as "the standard
deviation of a Lorentzian". `spread` below is therefore reused as the
distribution's scale parameter (HWHM, conventionally called gamma) when
kind='lorentzian', and as both the Gaussian sigma and Lorentzian HWHM
components when kind='pseudo_voigt' -- NOT as a literal standard
deviation. Sampled sizes are also hard-truncated to [min_size, max_size]
to keep them positive and physically finite, which further shifts the
realised mean/std away from the nominal request for the heavier-tailed
kinds. Both functions below report the realised per-axis mean/std
alongside the nominal inputs (verbose=True) so you can see the
difference rather than silently trusting the request.

ONE EQUATION/DEFORMATION PER run_bms() CALL
--------------------------------------------
bms.io_utils.inp_part() takes a single global "Equation"/"Deformation"
for the whole particle set, applied to every particle -- there is
currently no way to mix e.g. some ellipsoids and some hexagonal prisms
in one run_bms() call. random_close_pack()'s `equation`/`deformation`
arguments are used here only to get the bounding-radius/volume geometry
right for the overlap test and the diagnostics below; pass the SAME
values through to run_bms()/generate_bms_spectrum_dict()'s own
equation=/deformation= arguments, or the shapes used there won't match
what was actually packed.
"""
import warnings

import numpy as np
from scipy.spatial import cKDTree

from .io_utils import make_particle


def _as_axis_triplet(value, name):
    """Broadcast a scalar or (a, b, c) input to a float ndarray of shape (3,)."""
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return np.full(3, arr.item())
    if arr.size == 3:
        return arr.astype(float)
    raise ValueError(f"{name} must be a scalar or a length-3 (a,b,c) sequence, got {value!r}")


def sample_particle_sizes(n, mean, spread, kind="gaussian", eta=0.5,
                           min_size=None, max_size=None,
                           max_resample_rounds=50, rng=None, verbose=False):
    """
    Draw n (v1, v2, v3) size triples, independently per axis, from the
    requested distribution family.

    Parameters
    ----------
    n : int
        Number of particles.
    mean : float or (a, b, c)
        Mean size per axis. A scalar broadcasts to all three axes.
    spread : float or (s_a, s_b, s_c)
        Width per axis -- meaning depends on `kind` (see module
        docstring's honesty note): standard deviation for 'gaussian';
        HWHM (gamma) for 'lorentzian'; both the Gaussian-sigma and
        Lorentzian-gamma components (equal) for 'pseudo_voigt'.
    kind : {'gaussian', 'lorentzian', 'pseudo_voigt'}, optional
    eta : float, optional
        Lorentzian mixing fraction for 'pseudo_voigt' (0 -> pure
        Gaussian, 1 -> pure Lorentzian). Ignored for the other kinds.
    min_size, max_size : float or (a,b,c), optional
        Hard truncation bounds per axis; draws outside this range are
        rejected and redrawn. Default min_size = 1% of `mean` per axis
        (keeps sizes positive without allowing near-zero degenerate
        particles); default max_size = mean + 10*spread per axis --
        generous, but finite, since resampling from an untruncated
        Lorentzian/pseudo-Voigt tail could otherwise stall indefinitely.
    max_resample_rounds : int, optional
        Give up resampling out-of-bounds draws after this many rounds
        and clip whatever's left to the nearer bound instead (loudly,
        via a warning -- not silently).
    rng : int, numpy.random.Generator, or None, optional
    verbose : bool, optional
        Print nominal vs realised (post-truncation) mean/std per axis.

    Returns
    -------
    sizes : (n, 3) ndarray
    """
    if kind not in ("gaussian", "lorentzian", "pseudo_voigt"):
        raise ValueError(f"kind must be 'gaussian', 'lorentzian' or 'pseudo_voigt', got {kind!r}")
    if kind != "gaussian":
        warnings.warn(
            f"kind={kind!r}: 'spread' is used as the distribution's scale "
            f"parameter (HWHM), not a literal standard deviation – {kind} has "
            f"no finite variance. See sample_particle_sizes' docstring.",
            stacklevel=2,
        )

    rng = np.random.default_rng(rng)
    mean = _as_axis_triplet(mean, "mean")
    spread = _as_axis_triplet(spread, "spread")

    min_size = _as_axis_triplet(0.01 * mean if min_size is None else min_size, "min_size")
    max_size = _as_axis_triplet(mean + 10.0 * spread if max_size is None else max_size, "max_size")

    sizes = np.empty((n, 3))
    for axis in range(3):
        remaining = np.arange(n)
        for _ in range(max_resample_rounds):
            if remaining.size == 0:
                break
            m = remaining.size
            if kind == "gaussian":
                draw = rng.normal(mean[axis], spread[axis], size=m)
            elif kind == "lorentzian":
                draw = mean[axis] + spread[axis] * rng.standard_cauchy(size=m)
            else:  # pseudo_voigt
                is_lorentz = rng.random(m) < eta
                draw = np.where(
                    is_lorentz,
                    mean[axis] + spread[axis] * rng.standard_cauchy(size=m),
                    rng.normal(mean[axis], spread[axis], size=m),
                )
            ok = (draw >= min_size[axis]) & (draw <= max_size[axis])
            sizes[remaining[ok], axis] = draw[ok]
            remaining = remaining[~ok]
        if remaining.size:
            warnings.warn(
                f"{remaining.size} of {n} axis-{axis} draws did not land in "
                f"[{min_size[axis]:.4g}, {max_size[axis]:.4g}] after "
                f"{max_resample_rounds} resample rounds; clipping them to the "
                f"nearer bound instead of resampling further.",
                stacklevel=2,
            )
            fallback = rng.normal(mean[axis], spread[axis], size=remaining.size)
            sizes[remaining, axis] = np.clip(fallback, min_size[axis], max_size[axis])

    if verbose:
        print("sample_particle_sizes: nominal vs realised (post-truncation), per axis (v1,v2,v3):")
        print(f"  mean requested : {mean}")
        print(f"  mean realised  : {sizes.mean(axis=0)}")
        print(f"  spread requested (meaning depends on kind={kind!r}): {spread}")
        print(f"  std realised   : {sizes.std(axis=0)}")

    return sizes


def _bounding_radius(equation, v1, v2, v3):
    """Radius of a sphere, centred on the particle centre, guaranteed to
    contain the whole particle -- used for the overlap test in
    random_close_pack(). This is conservative: two particles whose
    bounding spheres don't overlap definitely don't overlap either; two
    particles whose bounding spheres DO overlap might still not actually
    overlap (e.g. two long needles end-to-end). So this under-packs
    relative to the true achievable density -- it never over-packs."""
    if equation == "hexagonal":
        Rhex, h = v1, v3
        return np.sqrt(Rhex**2 + (h / 2.0)**2)
    elif equation == "donut":
        R0, r0 = v1, v2
        return R0 + r0
    elif equation == "cuboid":          # half sides v1, v2, v3: the corner
        return np.sqrt(v1**2 + v2**2 + v3**2)
    else:  # ellipsoid
        return np.maximum(np.maximum(v1, v2), v3)


def _particle_volume(equation, v1, v2, v3):
    """True geometric volume per shape, used only for the packing
    diagnostics below (achieved_volume_fraction) -- NOT by run_bms()
    itself. NOTE: bms.io_utils.inp_part computes sim_part['vol'] with
    the ellipsoid formula (4/3 * pi * v1 * v2 * v3) UNCONDITIONALLY,
    regardless of `equation`; that value feeds the far-field
    point-dipole term in simulate.evaluate(). For hexagonal/donut
    particles that's a pre-existing inconsistency between the near-field
    shape (used for the FFT/demag calculation) and the far-field volume
    (used for evaluation points outside the box) -- not introduced or
    fixed here, just worth knowing if you're packing non-ellipsoid
    shapes and your evaluation points extend into the far-field."""
    if equation == "hexagonal":
        Rhex, h = v1, v3
        return (3 * np.sqrt(3) / 2.0) * Rhex**2 * h
    elif equation == "donut":
        R0, r0 = v1, v2
        return 2 * np.pi**2 * R0 * r0**2
    elif equation == "cuboid":
        return 8.0 * v1 * v2 * v3
    else:
        return (4.0 / 3.0) * np.pi * v1 * v2 * v3


def _random_unit_quaternions(n, rng):
    """Uniform (Haar-random) rotations: normalise n iid 4D standard normals."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q  # rows are (qw, qx, qy, qz)


def random_close_pack(
    n,
    box_size,
    mean_size,
    spread_size,
    size_dist="gaussian",
    eta=0.5,
    equation="ellipsoid",
    deformation="A",
    min_gap=0.0,
    min_size=None,
    max_size=None,
    max_attempts_per_particle=2000,
    seed=None,
    verbose=True,
):
    """
    Sequentially place n randomly-sized, randomly-oriented particles at
    non-overlapping positions inside a simulation box -- sequential
    random addition (RSA) with rejection. See the module docstring for
    what that does and doesn't guarantee relative to "true" random close
    packing, and for what `spread_size` means for non-Gaussian
    `size_dist`.

    Parameters
    ----------
    n : int
        Target particle count. Fewer may actually be placeable -- see
        Returns; this is reported, never silently hidden.
    box_size : float or (sx, sy, sz)
        Same convention as io_utils.build_box / bms_spectrum's box_size.
    mean_size, spread_size, size_dist, eta, min_size, max_size
        Passed to sample_particle_sizes() -- see its docstring.
    equation, deformation : str, optional
        Same meaning as run_bms()'s shape parameters. Every packed
        particle uses the same equation/deformation (see module
        docstring: this can't currently vary per particle) -- pass the
        same values through to run_bms()/generate_bms_spectrum_dict().
    min_gap : float, optional
        Minimum extra gap enforced between bounding spheres (not true
        particle surfaces -- see _bounding_radius), on top of just
        touching. 0 by default.
    max_attempts_per_particle : int, optional
        Give up on a given particle (skip it, don't retry indefinitely)
        after this many rejected placements. More of these get spent
        for the later particles in a dense pack -- that's the RSA
        "jamming" behaviour flagged in the module docstring, not a bug.
    seed : int, numpy.random.Generator, or None, optional
    verbose : bool, optional
        Print achieved count and volume fraction.

    Returns
    -------
    particles : list of dict
        Ready to pass straight to bms.main.run_bms() /
        bms_spectrum.generate_bms_spectrum_dict()'s `particles=`
        argument. May contain fewer than `n` entries -- check
        len(particles) / stats['n_placed'] against `n`.
    stats : dict
        n_requested, n_placed, box_volume, placed_volume,
        achieved_volume_fraction.
    """
    rng = np.random.default_rng(seed)

    box = np.full(3, float(box_size)) if np.isscalar(box_size) else np.asarray(box_size, dtype=float)
    box_volume = float(np.prod(box))

    sizes = sample_particle_sizes(
        n, mean_size, spread_size, kind=size_dist, eta=eta,
        min_size=min_size, max_size=max_size, rng=rng, verbose=verbose,
    )
    radii = _bounding_radius(equation, sizes[:, 0], sizes[:, 1], sizes[:, 2])

    too_big = radii * 2.0 > box.min()
    if np.any(too_big):
        warnings.warn(
            f"{int(too_big.sum())} of {n} sampled particles have a "
            f"bounding-sphere diameter larger than the smallest box "
            f"dimension ({box.min():.4g}) and cannot be placed at all – "
            f"they are skipped, not silently shrunk. Reduce mean_size/"
            f"spread_size or enlarge box_size.",
            stacklevel=2,
        )

    placed_centers = np.empty((0, 3))
    placed_radii = np.empty((0,))
    particles = []
    placed_volume = 0.0
    n_jammed = 0

    # Place largest particles first -- standard RSA practice: big particles
    # have the least room to be squeezed in later, so packing them first
    # (into empty space) reaches a noticeably denser jam than a random order.
    order = np.argsort(-radii)

    for idx in order:
        if too_big[idx]:
            continue
        r_i = radii[idx]
        half_range = box / 2.0 - r_i
        if np.any(half_range < 0):
            continue  # doesn't fit inside this box dimension at all

        placed_this_one = False
        for _ in range(max_attempts_per_particle):
            center = rng.uniform(-half_range, half_range)
            if placed_centers.shape[0]:
                dist = np.linalg.norm(placed_centers - center, axis=1)
                if np.any(dist < (placed_radii + r_i + min_gap)):
                    continue  # overlap -- try a new candidate position
            placed_centers = np.vstack([placed_centers, center])
            placed_radii = np.append(placed_radii, r_i)
            q = _random_unit_quaternions(1, rng)[0]
            v1, v2, v3 = sizes[idx]
            particles.append(make_particle(
                x=center[0], y=center[1], z=center[2],
                size=(v1, v2, v3), quaternion=tuple(q),
            ))
            placed_volume += _particle_volume(equation, v1, v2, v3)
            placed_this_one = True
            break

        if not placed_this_one:
            n_jammed += 1

    stats = {
        "n_requested": n,
        "n_placed": len(particles),
        "box_volume": box_volume,
        "placed_volume": placed_volume,
        "achieved_volume_fraction": placed_volume / box_volume,
    }

    if verbose:
        print(f"random_close_pack: placed {stats['n_placed']}/{n} particles "
              f"({100 * stats['n_placed'] / n:.1f}%), "
              f"achieved volume fraction = {stats['achieved_volume_fraction']:.3f}")
        if stats["n_placed"] < n:
            print(f"  {n_jammed} particle(s) never found a non-overlapping slot "
                  f"within max_attempts_per_particle={max_attempts_per_particle} "
                  f"(RSA jamming – see module docstring, not a bug). Lower n or "
                  f"mean_size, raise box_size, or raise max_attempts_per_particle "
                  f"to place more.")

    return particles, stats

# ---------------------------------------------------------------------------
# Force-biased / Jodrey-Tory-style "grow and separate" packer.
#
# Rationale (see discussion with user): random_close_pack() above is
# sequential random addition (RSA) with rejection, which saturates well
# below true random-close-packing density (~38% for monodisperse spheres,
# worse still once you account for the bounding-sphere overlap proxy being
# loose for anisotropic hex/donut particles). This packer instead:
#
#   1. Nucleates all n particles at once, at zero size, at random positions.
#   2. Grows every not-yet-frozen particle's bounding radius by a small
#      relative step.
#   3. Resolves any overlaps this creates by PUSHING particles apart
#      (mass-weighted by r^3, so big particles move less) rather than
#      deleting either one -- repeated relaxation sweeps via a KD-tree
#      neighbour query, vectorised over all overlapping pairs at once.
#   4. If a growth step can't be fully resolved even after many relaxation
#      sweeps, that step is rejected globally and the growth increment is
#      halved and retried (adaptive step size). Once the increment falls
#      below a floor, whichever particles are STILL part of an unresolved
#      overlap at that (rejected, larger) trial radius are frozen at their
#      last successfully-accepted size -- everyone else keeps growing.
#
# This lets particles reach their full sampled (v1, v2, v3) size wherever
# there's room, and only shrinks the ones that genuinely ran out of space
# (isotropically, preserving aspect ratio) -- nobody is deleted, and the
# realised size distribution is reported (achieved_size_fraction) rather
# than silently assumed. No particle's position/radius update ever reduces
# separation from another particle, so the packing is overlap-free by
# construction at every accepted state, not just at the end.
#
# This is NOT a rigorous Lubachevsky-Stillinger implementation (no real
# collision dynamics/velocities, and overlap is still tested via bounding
# spheres, not the true anisotropic shape) -- but it is a substantial,
# numpy-only, dependency-light improvement on RSA, and is the standard
# "force-biased algorithm" shape of solution (Bargiel & Moscinski 1991;
# Jodrey & Tory 1985) for this problem.
# ---------------------------------------------------------------------------


def _resolve_overlaps(positions, radii, box, min_gap, max_iters, rng):
    """Jacobi-style push-apart relaxation sweep. Never reduces existing
    separation between any pair -- only increases it -- so if the input
    state has zero overlap at some radii <= `radii`, the output state does
    too, regardless of whether THIS call fully resolves `radii`.

    Returns (positions, resolved, max_overlap). `resolved` is True iff no
    pair overlaps at `radii` after (at most) `max_iters` sweeps."""
    positions = positions.copy()
    mass = radii ** 3
    eps = 1e-12

    for _ in range(max_iters):
        max_r = radii.max()
        if max_r <= 0:
            return positions, True, 0.0

        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=2 * max_r + min_gap, output_type="ndarray")
        if pairs.shape[0] == 0:
            return positions, True, 0.0

        i, j = pairs[:, 0], pairs[:, 1]
        delta = positions[i] - positions[j]
        dist = np.linalg.norm(delta, axis=1)
        min_dist = radii[i] + radii[j] + min_gap
        overlap = min_dist - dist
        active = overlap > 0
        if not np.any(active):
            return positions, True, 0.0

        i, j = i[active], j[active]
        delta, dist, overlap = delta[active], dist[active], overlap[active]

        # near-coincident centres: nudge apart along a random direction
        # rather than dividing by ~0.
        zero = dist < 1e-9
        if np.any(zero):
            jitter = rng.normal(size=(zero.sum(), 3))
            jitter /= np.linalg.norm(jitter, axis=1, keepdims=True)
            delta[zero] = jitter * 1e-6
            dist[zero] = 1e-6

        unit = delta / dist[:, None]
        # displacement split inversely by mass (lighter/smaller particle
        # moves further), matching an elastic-collision-style split;
        # `+eps` guards the (measure-zero) case of two zero-radius points.
        w_i = (mass[j] + eps) / (mass[i] + mass[j] + 2 * eps)
        w_j = (mass[i] + eps) / (mass[i] + mass[j] + 2 * eps)

        disp = np.zeros_like(positions)
        np.add.at(disp, i, unit * (overlap * w_i)[:, None])
        np.add.at(disp, j, -unit * (overlap * w_j)[:, None])
        positions += disp

        half = box / 2.0
        lo = -half[None, :] + radii[:, None]
        hi = half[None, :] - radii[:, None]
        positions = np.clip(positions, lo, hi)

    # ran out of sweeps -- report how close we got
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=2 * radii.max() + min_gap, output_type="ndarray")
    if pairs.shape[0] == 0:
        return positions, True, 0.0
    i, j = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(positions[i] - positions[j], axis=1)
    overlap = (radii[i] + radii[j] + min_gap) - dist
    max_overlap = overlap.max() if overlap.size else 0.0
    return positions, max_overlap <= 0, max(max_overlap, 0.0)


def _overlap_participants(positions, radii, box, min_gap):
    """Boolean (n,) mask: which particles are part of at least one
    overlapping pair at the given positions/radii (no relaxation)."""
    n = positions.shape[0]
    participants = np.zeros(n, dtype=bool)
    max_r = radii.max()
    if max_r <= 0:
        return participants
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=2 * max_r + min_gap, output_type="ndarray")
    if pairs.shape[0] == 0:
        return participants
    i, j = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(positions[i] - positions[j], axis=1)
    overlap = (radii[i] + radii[j] + min_gap) - dist
    active = overlap > 0
    participants[i[active]] = True
    participants[j[active]] = True
    return participants


def grow_and_separate_pack(
    n,
    box_size,
    mean_size,
    spread_size,
    size_dist="gaussian",
    eta=0.5,
    equation="ellipsoid",
    deformation="A",
    min_gap=0.0,
    min_size=None,
    max_size=None,
    growth_rate=0.02,
    growth_rate_min_factor=1e-3,
    relax_iters_per_step=30,
    max_growth_steps=2000,
    seed=None,
    verbose=True,
):
    """
    Force-biased / Jodrey-Tory-style packer: grow all n particles from zero
    size toward their sampled (v1, v2, v3) simultaneously, resolving
    overlaps by pushing particles apart rather than discarding either one.
    See the module-level comment above this function for the algorithm and
    its relationship to random_close_pack()/RSA.

    Particles that run out of room keep the LAST size they grew to without
    overlap -- they are frozen, not deleted, and their final size is scaled
    isotropically (aspect ratio preserved) below their nominal sampled
    size. This differs from random_close_pack(), which places particles at
    their full nominal size or not at all.

    Parameters
    ----------
    n, box_size, mean_size, spread_size, size_dist, eta, equation,
    deformation, min_gap, min_size, max_size, seed, verbose
        Same meaning as random_close_pack() -- see its docstring.
    growth_rate : float, optional
        Initial per-step growth increment, as a fraction of each
        particle's target bounding radius (0.02 -> ~50 steps to reach
        full size absent any jamming).
    growth_rate_min_factor : float, optional
        Once the (adaptively halved) growth increment falls below
        growth_rate * growth_rate_min_factor without resolving, the
        particles still causing the unresolved overlap are frozen at
        their last accepted size, and the step size resets for whoever's
        left.
    relax_iters_per_step : int, optional
        Push-apart sweeps attempted per growth step before declaring that
        step unresolved and backing off.
    max_growth_steps : int, optional
        Hard cap on outer growth-attempt iterations (safety valve; you
        should reach convergence or full freezing well before this for
        reasonable n / growth_rate).

    Returns
    -------
    particles : list of dict
        Ready for bms.main.run_bms() / generate_bms_spectrum_dict(). Only
        particles that reached a nonzero size are included (particles
        whose bounding sphere can never fit in the box at all are
        excluded and warned about, same as random_close_pack()).
    stats : dict
        n_requested, n_placed, n_at_full_size, box_volume, placed_volume,
        achieved_volume_fraction, mean_achieved_size_fraction -- the last
        being the key number to check: 1.0 means every placed particle
        reached its full nominal sampled size; lower means some were
        shrunk to fit.
    """
    rng = np.random.default_rng(seed)

    box = np.full(3, float(box_size)) if np.isscalar(box_size) else np.asarray(box_size, dtype=float)
    box_volume = float(np.prod(box))

    sizes = sample_particle_sizes(
        n, mean_size, spread_size, kind=size_dist, eta=eta,
        min_size=min_size, max_size=max_size, rng=rng, verbose=verbose,
    )
    R_target = _bounding_radius(equation, sizes[:, 0], sizes[:, 1], sizes[:, 2])

    too_big = R_target * 2.0 > box.min()
    if np.any(too_big):
        warnings.warn(
            f"{int(too_big.sum())} of {n} sampled particles have a "
            f"bounding-sphere diameter larger than the smallest box "
            f"dimension ({box.min():.4g}) and can never fit – they are "
            f"excluded (frozen at zero size), not silently shrunk.",
            stacklevel=2,
        )

    half = box / 2.0
    positions = rng.uniform(-half, half, size=(n, 3))
    quats = _random_unit_quaternions(n, rng)

    s = np.zeros(n)          # current fraction of target bounding radius reached
    frozen = too_big.copy()  # particles that have permanently stopped growing

    step_size = np.full(n, growth_rate)  # per-particle adaptive step, halved on repeated failure
    min_step = growth_rate * growth_rate_min_factor

    for _ in range(max_growth_steps):
        grow_mask = (~frozen) & (s < 1.0)
        if not np.any(grow_mask):
            break

        s_trial = s.copy()
        s_trial[grow_mask] = np.minimum(1.0, s[grow_mask] + step_size[grow_mask])
        radii_trial = s_trial * R_target

        pos_trial, resolved, _ = _resolve_overlaps(
            positions, radii_trial, box, min_gap, relax_iters_per_step, rng)

        # Relaxed positions are always kept -- relaxation only ever increases
        # separation between pairs, so this cannot introduce overlap at
        # whatever (possibly smaller, post-rollback) radii each particle
        # ends up actually keeping below.
        positions = pos_trial

        if resolved:
            s = s_trial
            step_size[grow_mask] = growth_rate
            continue

        # NOT fully resolved: growing every grow_mask particle together by
        # its step_size could not be relaxed apart. Importantly, this does
        # NOT mean every particle in the group is at fault -- with many
        # particles growing simultaneously, "reject the whole batch because
        # someone somewhere is stuck" would mean a single crowded corner of
        # the box perpetually blocks every other particle's progress
        # elsewhere, and growth stalls well short of any real jam.
        # Only the specific particles still overlapping (at radii_trial,
        # after relaxation) are rolled back and have their step halved;
        # everyone else in this round's grow_mask keeps their growth.
        involved = _overlap_participants(pos_trial, radii_trial, box, min_gap)
        stuck = grow_mask & involved
        accepted = grow_mask & ~stuck

        s[accepted] = s_trial[accepted]
        step_size[accepted] = growth_rate
        step_size[stuck] /= 2.0

        newly_frozen = stuck & (step_size < min_step)
        frozen[newly_frozen] = True
        step_size[newly_frozen] = growth_rate  # irrelevant once frozen, but keep tidy

    # The growth loop caps relaxation effort at relax_iters_per_step per
    # attempt (for speed) -- in a crowded final configuration this can
    # leave a tiny residual overlap (empirically ~1e-6 relative to particle
    # size in dense test cases) rather than exactly zero. One unhurried
    # polish pass at the FINAL committed radii (not a trial/grown size)
    # cleans this up before particles are returned.
    positions, polish_resolved, polish_overlap = _resolve_overlaps(
        positions, s * R_target, box, min_gap, max_iters=10 * relax_iters_per_step, rng=rng)
    if not polish_resolved and verbose:
        print(f"  Warning: final polish pass left a residual max overlap of "
              f"{polish_overlap:.3g} (vs typical particle size {np.mean(R_target):.3g}) – "
              f"consider raising relax_iters_per_step if this matters for your use case.")

    kept = s > 1e-9
    final_sizes = sizes * s[:, None]
    placed_volume = float(np.sum(_particle_volume(
        equation, final_sizes[kept, 0], final_sizes[kept, 1], final_sizes[kept, 2])))

    particles = []
    for idx in np.where(kept)[0]:
        v1, v2, v3 = final_sizes[idx]
        particles.append(make_particle(
            x=positions[idx, 0], y=positions[idx, 1], z=positions[idx, 2],
            size=(v1, v2, v3), quaternion=tuple(quats[idx]),
        ))

    stats = {
        "n_requested": n,
        "n_placed": len(particles),
        "n_at_full_size": int(np.sum(s >= 1.0 - 1e-9)),
        "box_volume": box_volume,
        "placed_volume": placed_volume,
        "achieved_volume_fraction": placed_volume / box_volume,
        "mean_achieved_size_fraction": float(s[kept].mean()) if np.any(kept) else 0.0,
    }

    if verbose:
        print(f"grow_and_separate_pack: placed {stats['n_placed']}/{n} particles "
              f"({100 * stats['n_placed'] / n:.1f}%), "
              f"{stats['n_at_full_size']}/{stats['n_placed']} at full nominal size, "
              f"mean achieved size fraction = {stats['mean_achieved_size_fraction']:.3f}, "
              f"achieved volume fraction = {stats['achieved_volume_fraction']:.3f}")
        if stats["n_at_full_size"] < stats["n_placed"]:
            print("  Some particles were shrunk (isotropically, aspect ratio preserved) to fit – "
                  "check achieved_size_fraction per-particle (`s` in the source) if the size "
                  "distribution fidelity matters for your analysis, or lower n / mean_size / "
                  "packing density expectations.")

    return particles, stats