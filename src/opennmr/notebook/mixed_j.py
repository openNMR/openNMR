"""
Weak (first-order) and strong (exact) J-coupled sets on one spectrum.

A real solution spectrum often has both: a CH2CH3 ethyl group whose
shifts differ by ~2 ppm, so J is a small perturbation and first-order
multiplets are exact to well within the linewidth, *and* an aromatic AB
pair with |Δν/J| ≈ 1, which needs the full Hamiltonian (roofing, extra
lines). :func:`mixed_j_spectrum` simulates each with the right model and
adds them on one ppm axis:

* **weak sites** -- as in :func:`~.simulations.multi_site_jcoupled_spectrum`
  (a shift, coupling groups, relative intensity, linewidth in ppm);
* **strong spins** -- as in :func:`~.spin_system.spin_system_spectrum`
  (spins and the J between them). The spins are split into their
  *independent* coupled sets first (connected components of the coupling
  graph): two sets with no J between them are two separate spin systems,
  which is exact -- and much cheaper than diagonalising their product, so
  several strong sets (an AB pair and an ABX, say) can sit on one spectrum.

Both parts use the same area scale (each observed nucleus contributes 1,
times a weak site's ``relative_intensity``), so their relative integrals are
the numbers of nuclei.
"""
from __future__ import annotations

import numpy as np

from . import simulations as _sim
from . import spin_system as _nss


def split_spin_system(spins, couplings) -> list[tuple[list[dict], list[dict]]]:
    """The independent coupled sets of a spin system: ``[(spins, couplings), ...]``
    in the order each set's first spin appears. A spin with no coupling is its
    own one-spin set; a coupling to a spin that is not listed raises ValueError."""
    spins = [dict(s) for s in spins]
    labels = [str(s.get("label", f"S{i + 1}")) for i, s in enumerate(spins)]
    for s, label in zip(spins, labels):
        s["label"] = label
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError("Two strongly coupled spins share a label.")
    parent = list(range(len(spins)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for c in couplings:
        for key in ("a", "b"):
            if str(c[key]) not in index:
                raise ValueError(f"Coupling {c['a']}-{c['b']}: no spin labelled {c[key]}.")
        ra, rb = find(index[str(c["a"])]), find(index[str(c["b"])])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(len(spins)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in sorted(groups.values(), key=min):
        names = {labels[i] for i in members}
        out.append(([spins[i] for i in members],
                    [c for c in couplings if str(c["a"]) in names]))
    return out


def mixed_j_spectrum(ppm_axis, larmor_Hz, B0_tesla, weak_sites=(), strong_spins=(),
                     strong_couplings=(), strong_fwhm_Hz=1.0, observe="1H",
                     lineshape="lorentzian", eta=0.0, weak_fwhm=None) -> dict:
    """Sum first-order (weak) sites and exactly solved (strong) sets.

    Parameters
    ----------
    ppm_axis, larmor_Hz, B0_tesla : the shared axis, the observe frequency and the
        field (the strong sets' shifts in Hz use the field and each nuclide).
    weak_sites : list of dict, as for
        :func:`~.simulations.multi_site_jcoupled_spectrum` (``ppm``, ``couplings``,
        ``relative_intensity``, ``fwhm`` in ppm).
    strong_spins, strong_couplings : as for
        :func:`~.spin_system.spin_system_spectrum`; split into independent sets.
    strong_fwhm_Hz : linewidth of every strong line (Hz).
    observe : the observed nuclide (strong sets with none of it are skipped).
    weak_fwhm : default linewidth (ppm) of a weak site without its own.

    Returns a dict: ``x_ppm``, ``real_spec`` (the sum), ``normalised_real_spec``,
    ``weak`` (the weak part's result, or None), ``strong`` (a list, one entry per
    set that has an observed spin: ``labels``, ``result``) and ``skipped`` (the
    labels of sets with nothing to observe).
    """
    x = np.asarray(ppm_axis, dtype=float)
    total = np.zeros_like(x)
    weak = None
    if len(weak_sites):
        default = weak_fwhm if weak_fwhm is not None else float(weak_sites[0].get("fwhm", 0.01))
        weak = _sim.multi_site_jcoupled_spectrum(
            list(weak_sites), larmor_Hz, x, fwhm=default, lineshape=lineshape, eta=eta)
        total += weak["real_spec"]
    strong, skipped = [], []
    if len(strong_spins):
        for set_spins, set_couplings in split_spin_system(strong_spins, strong_couplings):
            labels = [s["label"] for s in set_spins]
            if not any(s["nuclide"] == observe for s in set_spins):
                skipped.append(labels)
                continue
            result = _nss.spin_system_spectrum(
                set_spins, set_couplings, B0_tesla, x, fwhm_Hz=strong_fwhm_Hz,
                lineshape=lineshape, eta=eta, observe=observe)
            total += result["real_spec"]
            strong.append({"labels": labels, "result": result})
    peak = np.max(np.abs(total)) if total.size else 0.0
    return {"x_ppm": x, "real_spec": total,
            "normalised_real_spec": total / peak if peak else total,
            "weak": weak, "strong": strong, "skipped": skipped}
