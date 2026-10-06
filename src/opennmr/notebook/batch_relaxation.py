"""
Batch relaxation: one pseudo-2D per temperature (and per field), every
one measured and fitted the same way, then T1/T2 against temperature for a
BPP/Arrhenius fit (explicit user request, 2026-10-06: "T1 measurements as
a function of temperature, one pseudo-2D per temperature, all processed
with the same settings and grouped by field, then the BPP fit").

    from opennmr.notebook import batch_relaxation as br, vt_relaxation as vt
    batch = br.NMR2D_relaxation_batch(
        ["/data/vt/10", "/data/vt/11", "/data/vt/12"], integration="regions",
        regions=[(4.5, 5.5)], model="inversion_recovery",
        rows=list(range(1, 9)))
    print(br.batch_table(batch))
    datasets = br.batch_vt_datasets(batch)           # grouped by peak and field
    fit = vt.fit_vt_relaxation(datasets)

Each series is read as TopSpin processed it (``NMR2dimport``), measured by
:func:`opennmr.notebook.relaxation.NMR2D_relaxation_workflow` with the same
settings, its delays from its own ``vdlist`` unless given, its
temperature from its own ``acqus`` (``TE``, K) and its field from ``SFO1``.
A series that fails is reported and skipped, never silently dropped.
"""

from __future__ import annotations

import csv
import io
import os

import numpy as np

from .relaxation import NMR2D_relaxation_workflow


def read_temperature_and_field(datapath):
    """``(TE in K, SFO1 in MHz)`` from a Bruker experiment's ``acqus``;
    None for either it does not record."""
    te = sfo1 = None
    path = os.path.join(str(datapath), "acqus")
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("##$TE="):
                    te = float(line.split("=", 1)[1])
                elif line.startswith("##$SFO1="):
                    sfo1 = float(line.split("=", 1)[1])
    except (OSError, ValueError):
        pass
    return te, sfo1


def _quantity(rate_param):
    if not rate_param:
        return None
    r = str(rate_param)
    if r.upper().startswith("T1") and "rho" not in r.lower():
        return "T1"
    if r.upper().startswith("T2"):
        return "T2"
    if "rho" in r.lower():
        return "T1rho"
    return None


def rows_from_fits(fits, label, T_K, larmor_MHz):
    """One row per fitted peak/region of one series:
    {"experiment", "series", "T_K", "value", "error", "larmor_MHz",
    "quantity"} -- the shape Dynamics ▸ Relaxation vs Temperature takes."""
    rows = []
    for k, fit in enumerate(fits or []):
        rate = fit.get("rate_param")
        quantity = _quantity(rate)
        if quantity is None or rate not in (fit.get("params") or {}):
            continue
        err = (fit.get("errors") or {}).get(rate)
        rows.append({"experiment": str(label),
                     "series": str(fit.get("peak_label") or f"peak {k + 1}"),
                     "T_K": None if T_K is None else float(T_K),
                     "value": float(fit["params"][rate]),
                     "error": float(err) if err is not None else float("nan"),
                     "larmor_MHz": None if larmor_MHz is None else float(larmor_MHz),
                     "quantity": quantity})
    return rows


def NMR2D_relaxation_batch(datapaths, procno=1, temperatures=None, larmor_MHz=None,
                           labels=None, vdlists=None, **workflow_kwargs):
    """
    Run :func:`NMR2D_relaxation_workflow` on every pseudo-2D in
    ``datapaths`` with the same ``workflow_kwargs`` (integration, peaks or
    regions, model, rows, ...), and collect every series' fitted T1/T2
    with its temperature and field.

    ``temperatures`` (K) / ``larmor_MHz`` / ``vdlists`` / ``labels`` : one
    per path, overriding what each experiment's own files say (None in
    the list: read it). Plots and reports are off unless asked for.

    Returns a dict: ``rows`` (as :func:`rows_from_fits`, every series),
    ``runs`` (per path: label, T_K, larmor_MHz, fits or error), and
    ``failed`` (labels whose fit failed, with the reason).
    """
    paths = list(datapaths)
    n = len(paths)

    def per(values, name):
        if values is None:
            return [None] * n
        values = list(values)
        if len(values) != n:
            raise ValueError(f"{name}: {len(values)} values for {n} datasets.")
        return values

    temps, fields, vds = per(temperatures, "temperatures"), per(larmor_MHz, "larmor_MHz"), \
        per(vdlists, "vdlists")
    names = per(labels, "labels")
    kwargs = dict(workflow_kwargs)
    kwargs.setdefault("show_fit", False)
    kwargs.setdefault("show_report", False)
    kwargs.setdefault("show_warnings", False)
    rows, runs, failed = [], [], []
    for path, T, f, vd, name in zip(paths, temps, fields, vds, names):
        te, sfo1 = read_temperature_and_field(path)
        T = te if T is None else T
        f = sfo1 if f is None else f
        label = name or os.path.basename(os.path.normpath(str(path)))
        try:
            out = NMR2D_relaxation_workflow(path, procno=procno,
                                            **({**kwargs, "vdlist": vd} if vd is not None
                                               else kwargs))
        except Exception as exc:  # noqa: BLE001 -- one bad series must not sink the batch
            failed.append((label, str(exc)))
            runs.append({"label": label, "path": str(path), "T_K": T, "larmor_MHz": f,
                         "fits": None, "error": str(exc)})
            continue
        fits = out["relaxation_fits"]
        runs.append({"label": label, "path": str(path), "T_K": T, "larmor_MHz": f,
                     "fits": fits, "error": None})
        rows += rows_from_fits(fits, label, T, f)
    return {"rows": rows, "runs": runs, "failed": failed}


def group_rows(rows, field_tolerance_MHz=0.01):
    """``{(series, field MHz or None): [rows]}`` -- fields within
    ``field_tolerance_MHz`` count as one (SFO1 moves slightly with O1)."""
    groups: dict = {}
    for r in rows:
        f = r.get("larmor_MHz")
        key_f = None
        if f is not None:
            for (s, g) in groups:
                if s == r["series"] and g is not None and abs(g - f) <= field_tolerance_MHz:
                    key_f = g
                    break
            else:
                key_f = round(float(f), 4)
        groups.setdefault((r["series"], key_f), []).append(r)
    return groups


def batch_vt_datasets(batch_or_rows, field_tolerance_MHz=0.01):
    """The batch as :func:`opennmr.notebook.vt_relaxation.vt_dataset`
    datasets, one per (peak, field), ready for ``fit_vt_relaxation``."""
    from .vt_relaxation import vt_dataset
    rows = batch_or_rows["rows"] if isinstance(batch_or_rows, dict) else batch_or_rows
    out = []
    for (series, field), group in group_rows(rows, field_tolerance_MHz).items():
        group = [r for r in group if r["T_K"] is not None]
        if not group:
            continue
        out.append(vt_dataset([r["T_K"] for r in group], [r["value"] for r in group],
                              error=[r["error"] for r in group], larmor_MHz=field,
                              quantity=group[0]["quantity"],
                              label=f"{series}" + (f", {field:g} MHz" if field else "")))
    return out


def batch_table(batch_or_rows) -> str:
    """CSV text: experiment, series, T (K), field (MHz), quantity, value
    (s), error (s) -- one line per fitted series."""
    rows = batch_or_rows["rows"] if isinstance(batch_or_rows, dict) else batch_or_rows
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["experiment", "series", "T_K", "larmor_MHz", "quantity", "value_s", "error_s"])
    for r in rows:
        w.writerow([r.get("experiment", ""), r["series"],
                    "" if r["T_K"] is None else f"{r['T_K']:g}",
                    "" if r["larmor_MHz"] is None else f"{r['larmor_MHz']:.6g}",
                    r["quantity"], f"{r['value']:.6g}",
                    "" if not np.isfinite(r["error"]) else f"{r['error']:.3g}"])
    return buf.getvalue()
