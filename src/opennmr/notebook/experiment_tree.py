"""Open experiments as a tree of typed entries (explicit user request,
2026-11-13: label what is open as a spectrum, a fit, a profile fit or a
simulation, a multi-spectrum fit as a group, and bring a fit back by
opening it).

Pure data, no GUI: :func:`build_experiment_tree` takes what each open
experiment records about itself (its ``metadata.raw``) and returns nested
nodes; the GUI (gui/widgets/experiment_tree.py) only draws them.

Kinds
-----
``spectrum``     measured (or derived) data.
``simulation``   a simulated spectrum kept in Open experiments.
``fit``          a fit of a spectrum: the fit each Fit panel last made of
                 it, a spectral (peak) fit, a kept fit model, or a kept fit
                 listed as an experiment of its own.
``profile_fit``  a fit of values against a series (Fit Curve: relaxation,
                 diffusion, a custom curve).
``multi_fit``    a group: one multi-spectrum fit and the spectra in it.

A node is a dict: ``kind``, ``label``, ``type_label``, ``row`` (the Open
experiments row it selects: its own, or for a fit stored with a spectrum
that spectrum's), ``entry_id`` (of that row), ``virtual`` (True when it is
not itself a row), ``action`` (what opening it does: ``None`` to show the
row, or ``("stored_fit", key)``, ``("slice_fit",)``, ``("curve_fit",)``,
``("fit_model", id)``, ``("series_fit", kind)``), ``tooltip``, ``key`` (stable, for remembering what
is expanded) and ``children``.
"""

from __future__ import annotations

KIND_LABELS = {"spectrum": "Spectrum", "simulation": "Simulation", "fit": "Fit",
               "profile_fit": "Profile fit", "multi_fit": "Multi-fit"}

#: The Fit panels' names, as the GUI's own labels (main_window._FIT_LABELS).
PANEL_LABELS = {"csa": "CSA", "jcoupling": "J-coupling", "quadrupolar": "Quadrupolar",
                "czjzek": "Czjzek", "combined": "Solid-state NMR", "multi": "Multi-spectrum",
                "pnmr": "pNMR (CIF)", "simpson": "SIMPSON spectrum"}


def _node(kind, label, row, entry_id, *, virtual=False, action=None, tooltip="", key=None,
          type_label=None):
    return {"kind": kind, "label": str(label), "type_label": type_label or KIND_LABELS[kind],
            "row": row, "entry_id": entry_id, "virtual": bool(virtual), "action": action,
            "tooltip": tooltip or "", "key": key or f"{entry_id}:{kind}:{label}",
            "children": []}


def _text(value, default=""):
    return value if isinstance(value, str) and value.strip() else default


def entry_kind(raw: dict | None) -> str:
    """``spectrum``, ``simulation`` or ``fit`` for an open experiment with
    this ``metadata.raw``."""
    raw = raw or {}
    if raw.get("fit_model"):
        return "fit"
    if raw.get("simulation"):
        return "simulation"
    return "spectrum"


def series_fit_label(label: str, slices=None) -> str:
    """The tree's name for a model applied to every slice: the stored
    label ("Solid-state NMR model, every slice") read as what it is."""
    base = _text(label, "Model").replace(", every slice", "").strip()
    count = f", {slices} slices" if isinstance(slices, int) and slices > 1 else ""
    return f"{base} applied to every slice{count}"


def _stored_fit_nodes(entry) -> list[dict]:
    """The fits stored with one open experiment (not rows themselves)."""
    raw, row, eid = entry.get("raw") or {}, entry["row"], entry.get("entry_id")
    nodes = []
    listed = set(entry.get("listed_model_ids") or ())
    fits = raw.get("fits") if isinstance(raw.get("fits"), dict) else {}
    for key, record in fits.items():
        if key == "multi" or not isinstance(record, dict):
            continue
        label = _text(record.get("label"), PANEL_LABELS.get(key, str(key)))
        tip = "\n".join(t for t in (_text(record.get("target")), _text(record.get("time"))) if t)
        if str(key).startswith("series_"):
            # A fitted lineshape model applied to every slice of a series (no
            # panel owns it: it is shown as the per-slice fits it made).
            kind = str(key)[len("series_"):]
            inputs = record.get("inputs") if isinstance(record.get("inputs"), dict) else {}
            slices = inputs.get("slices")
            name = series_fit_label(label, slices)
            tip = "\n".join(t for t in (
                "The fitted lineshapes of one slice, held fixed, scaled to each slice in "
                "turn (amplitude per site per slice). Feeds Fit Curve.",
                _text(record.get("inputs_text")), tip) if t)
            nodes.append(_node("fit", name, row, eid, virtual=True,
                               action=("series_fit", kind), tooltip=tip,
                               key=f"{eid}:fit:{key}"))
            continue
        nodes.append(_node("fit", f"{label} fit", row, eid, virtual=True,
                           action=("stored_fit", key), tooltip=tip, key=f"{eid}:fit:{key}"))
    slices = raw.get("slice_fit_results")
    if isinstance(slices, (list, tuple)) and slices and isinstance(slices[0], dict) \
            and slices[0].get("popt") is not None:
        label = "Spectral fit" + (f" ({len(slices)} slices)" if len(slices) > 1 else "")
        nodes.append(_node("fit", label, row, eid, virtual=True, action=("slice_fit",),
                           key=f"{eid}:fit:slices"))
    curves = raw.get("relaxation_fit_results")
    if isinstance(curves, (list, tuple)) and curves:
        first = curves[0] if isinstance(curves[0], dict) else {}
        model = _text(first.get("model"))
        label = "Curve fit" + (f" ({model.replace('_', ' ')})" if model else "")
        if len(curves) > 1:
            label += f", {len(curves)} peaks"
        nodes.append(_node("profile_fit", label, row, eid, virtual=True, action=("curve_fit",),
                           key=f"{eid}:profile:curve"))
    for model in raw.get("fit_models") or []:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if model_id is not None and model_id in listed:
            continue                      # shown as its own row, below
        kind = "profile_fit" if model.get("kind") == "curve" else "fit"
        name = _text(model.get("name"), "kept fit")
        detail = _text(model.get("kind_label"))
        nodes.append(_node(kind, f"{name} ({detail})" if detail else name, row, eid,
                           virtual=True, action=("fit_model", model_id),
                           tooltip="Kept fit model" + (f", {model.get('time')}"
                                                       if _text(model.get("time")) else ""),
                           key=f"{eid}:model:{model_id}"))
    return nodes


def build_experiment_tree(entries) -> list[dict]:
    """The tree for ``entries``: one dict per open experiment, in row
    order, with ``row``, ``label``, ``entry_id`` and ``raw`` (its
    ``metadata.raw``). Returns the top-level nodes."""
    entries = [dict(e) for e in entries]
    by_id = {e.get("entry_id"): e for e in entries if e.get("entry_id") is not None}
    by_label = {}
    for e in entries:
        by_label.setdefault(e["label"], e)

    # Which experiment each kept-fit row belongs to, and the multi-fit groups.
    parent_of, kept_groups = {}, {}
    for e in entries:
        raw = e.get("raw") or {}
        e["kind"] = entry_kind(raw)
        if e["kind"] != "fit":
            continue
        model = raw.get("fit_model") if isinstance(raw.get("fit_model"), dict) else {}
        source = raw.get("fit_model_source") if isinstance(raw.get("fit_model_source"), dict) \
            else {}
        if model.get("kind") == "multi":
            kept_groups.setdefault(_text(model.get("name"), "Multi-spectrum fit"), []).append(e)
            continue
        parent = by_id.get(source.get("entry_id")) or by_label.get(source.get("spectrum"))
        if parent is not None and parent is not e:
            parent_of[e["row"]] = parent
            parent.setdefault("listed_model_ids", set()).add(source.get("id"))

    nodes, made = [], {}
    for e in entries:
        raw = e.get("raw") or {}
        model = raw.get("fit_model") if isinstance(raw.get("fit_model"), dict) else {}
        type_label = None
        if e["kind"] == "fit":
            type_label = "Profile fit" if model.get("kind") == "curve" else "Fit"
        node = _node(e["kind"], e["label"], e["row"], e.get("entry_id"),
                     key=f"{e.get('entry_id')}:row", type_label=type_label,
                     tooltip=_text(model.get("kind_label")))
        made[e["row"]] = node
    in_group = {e["row"] for members in kept_groups.values() for e in members}
    for e in entries:
        node = made[e["row"]]
        node["children"].extend(_stored_fit_nodes(e))
        if e["row"] in in_group:
            continue
        parent = parent_of.get(e["row"])
        if parent is not None:
            made[parent["row"]]["children"].append(node)
        else:
            nodes.append(node)

    # Multi-spectrum fits: the fit stored with each spectrum in it ...
    stored = {}
    for e in entries:
        fits = (e.get("raw") or {}).get("fits")
        record = fits.get("multi") if isinstance(fits, dict) else None
        if not isinstance(record, dict):
            continue
        inputs = record.get("inputs") if isinstance(record.get("inputs"), dict) else {}
        keys = tuple(str(x.get("key")) for x in inputs.get("experiments") or []
                     if isinstance(x, dict))
        stored.setdefault(keys or (_text(record.get("target"), e["label"]),), []).append(e)
    for keys, members in stored.items():
        n = max(len(keys), len(members))
        group = _node("multi_fit", f"Multi-spectrum fit ({n} spectra)", members[0]["row"],
                      members[0].get("entry_id"), virtual=True, action=("stored_fit", "multi"),
                      key="multi:" + "|".join(keys))
        for e in members:
            group["children"].append(_node(
                "fit", e["label"], e["row"], e.get("entry_id"), virtual=True,
                action=("stored_fit", "multi"), key=f"{e.get('entry_id')}:fit:multi"))
        nodes.append(group)
    # ... and those kept in Open experiments under one name.
    for name, members in kept_groups.items():
        group = _node("multi_fit", f"{name} ({len(members)} spectra)", members[0]["row"],
                      members[0].get("entry_id"), virtual=True, key=f"multi-kept:{name}")
        group["children"].extend(made[e["row"]] for e in members)
        nodes.append(group)
    return nodes


def iter_nodes(nodes, depth: int = 0):
    """Every node with its depth, in display order."""
    for node in nodes:
        yield node, depth
        yield from iter_nodes(node["children"], depth + 1)


def experiment_tree_text(nodes) -> str:
    """The tree as indented text, one entry per line with its type."""
    return "\n".join(f"{'    ' * depth}{node['label']}  [{node['type_label']}]"
                     for node, depth in iter_nodes(nodes))
