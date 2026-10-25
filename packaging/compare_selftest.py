"""Compare two self-test runs: from source, and from a built app.

Usage::

    python -m opennmr.selftest selftest-source
    dist/openNMR/openNMR --self-test selftest-app        # or the .exe / .app
    python packaging/compare_selftest.py selftest-source selftest-app

Passes (exit 0) when both runs passed every step, every number agrees,
and every image matches to within ``--tolerance`` (the fraction of
pixels allowed to differ; default 0 -- the same libraries on the same
machine should draw identically, so any difference is worth a look).
A ``diff_<name>.png`` is written beside the app run's images for each
one that differs, the differing pixels in red.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

IMAGES = ("figure_1d.png", "figure_2d.png", "figure_csa.png", "gui_main_window.png")


def compare_numbers(a: dict, b: dict, rtol: float = 1e-10, atol: float = 1e-12) -> list[str]:
    import numpy as np

    problems = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            problems.append(f"numbers: '{key}' is in only one run")
            continue
        x, y = np.asarray(a[key], dtype=float), np.asarray(b[key], dtype=float)
        if x.shape != y.shape:
            problems.append(f"numbers: '{key}' shapes differ {x.shape} vs {y.shape}")
        elif not np.allclose(x, y, rtol=rtol, atol=atol):
            worst = float(np.max(np.abs(x - y)))
            problems.append(f"numbers: '{key}' differs (largest difference {worst:.3g})")
    return problems


def compare_image(path_a: str, path_b: str, diff_path: str, tolerance: float) -> tuple[bool, str]:
    from PIL import Image, ImageChops

    a = Image.open(path_a).convert("RGB")
    b = Image.open(path_b).convert("RGB")
    if a.size != b.size:
        return False, f"sizes differ {a.size} vs {b.size}"
    diff = ImageChops.difference(a, b)
    if diff.getbbox() is None:
        return True, "identical"
    mask = diff.convert("L").point(lambda v: 255 if v else 0)
    changed = mask.histogram()[255] / (a.size[0] * a.size[1])
    marked = b.copy()
    marked.paste((255, 0, 0), mask=mask)
    marked.save(diff_path)
    ok = changed <= tolerance
    return ok, f"{changed:.4%} of pixels differ (see {os.path.basename(diff_path)})"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source_run")
    parser.add_argument("app_run")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="fraction of pixels allowed to differ per image (default 0)")
    args = parser.parse_args(argv)

    problems = []
    reports = {}
    for label, folder in (("source", args.source_run), ("app", args.app_run)):
        with open(os.path.join(folder, "report.json"), encoding="utf-8") as handle:
            reports[label] = json.load(handle)
        for name, step in reports[label]["steps"].items():
            if not step["ok"]:
                problems.append(f"{label} run: step '{name}' failed:\n{step.get('error', '')}")
    if not reports["app"].get("frozen"):
        problems.append("the 'app' run was not from a frozen build (sys.frozen not set)")

    with open(os.path.join(args.source_run, "numbers.json"), encoding="utf-8") as handle:
        numbers_a = json.load(handle)
    with open(os.path.join(args.app_run, "numbers.json"), encoding="utf-8") as handle:
        numbers_b = json.load(handle)
    problems += compare_numbers(numbers_a, numbers_b)

    lines = []
    for name in IMAGES:
        a, b = os.path.join(args.source_run, name), os.path.join(args.app_run, name)
        if not (os.path.exists(a) and os.path.exists(b)):
            problems.append(f"image {name}: missing from a run")
            continue
        ok, text = compare_image(a, b, os.path.join(args.app_run, f"diff_{name}"), args.tolerance)
        lines.append(f"  {name}: {text}")
        if not ok:
            problems.append(f"image {name}: {text}")

    print("Self-test comparison: source vs built app")
    for label in ("source", "app"):
        r = reports[label]
        timing = ", ".join(f"{k} {v['seconds']}s" for k, v in r["steps"].items())
        print(f"  {label}: {'passed' if r['ok'] else 'FAILED'} "
              f"(frozen={r.get('frozen')}, {r.get('platform')}, Python {r.get('python')}; {timing})")
    print("  numbers: " + ("all agree" if not any(p.startswith("numbers") for p in problems)
                           else "DIFFER"))
    print("\n".join(lines))
    if problems:
        print("\nPROBLEMS:\n- " + "\n- ".join(problems))
        return 1
    print("\nThe built app matches the source run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
