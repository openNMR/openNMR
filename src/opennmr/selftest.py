"""A self-check that the *installed* openNMR works -- above all the frozen
app an installer puts on a machine, whose Python, Qt and libraries are
PyInstaller's copies rather than the ones the test suite ran against.

Run it from source with ``python -m opennmr.selftest OUT_DIR`` and from
the built app with ``openNMR --self-test OUT_DIR``; then
``python packaging/compare_selftest.py SOURCE_DIR APP_DIR`` checks the
two agree -- the same numbers, and figures that look the same pixel for
pixel. That comparison, not "it launched", is what says a build is
right (explicit user request, 2026-10-15: "do not overwrite the current
set of scripts until we are certain everything looks the same").

What it exercises (each step recorded in ``OUT_DIR/report.json``; one
failing step does not stop the others):

1. **imports** -- every module of the package. PyInstaller follows
   imports it can see; a module only reached lazily, or a data file it
   reads, is what goes missing in a build, and this finds it first.
2. **version** -- the installed version matches the package metadata
   (a build without ``--copy-metadata`` silently reports the fallback).
3. **read_process** -- a synthetic Bruker 1D and 2D written to disk and
   read back through :func:`opennmr.io.load`.
4. **simulate / fit** -- a static CSA powder pattern (numpy/scipy) and a
   least-squares Lorentzian fit; results saved as numbers.
5. **cif_structure** -- a small partially occupied CIF read with
   pymatgen and its neighbour shells listed (Simulate/Fit ▸ pNMR's first
   step). pymatgen is imported lazily, so step 1 never loads it; its
   compiled neighbour search and data tables are what a build can miss.
6. **tls** -- HTTPS certificates are loadable (the update check's
   :func:`opennmr.update.ssl_context`); no network is used.
7. **figures** -- the 1D, 2D contour and CSA figures saved as PNG with
   the package's own plotting code (matplotlib, its bundled fonts).
8. **gui** -- a :class:`MainWindow` built offscreen, the 1D dataset
   opened in it, and the window captured as PNG.

Nothing here touches the user's ``~/.opennmr``: settings go to
``OUT_DIR/home``.
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import sys
import time
import traceback

#: Modules not imported in step 1: scripts with side effects on import.
SKIP_IMPORT_PREFIXES = ("opennmr.notebook.scripts",)

GUI_SIZE = (1400, 900)


def _write_params(path, params):
    lines = []
    for key, value in params.items():
        lines.append(f"##${key}= <{value}>" if isinstance(value, str) else f"##${key}= {value}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _lorentz(x, x0, w, a):
    return a * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def make_bruker_1d(folder, n=4096):
    """A processed 1D Bruker dataset: three Lorentzian lines on 1r."""
    import numpy as np

    expt = os.path.join(folder, "selftest_1d")
    pdata = os.path.join(expt, "pdata", "1")
    os.makedirs(pdata, exist_ok=True)
    _write_params(os.path.join(expt, "acqus"),
                  {"NUC1": "1H", "BF1": 400.13, "SFO1": 400.13, "O1": 2000.0, "NS": 16})
    _write_params(os.path.join(pdata, "procs"),
                  {"SW_p": 8000.0, "SI": n, "SF": 400.13, "OFFSET": 15.0, "NC_proc": 0})
    i = np.arange(n, dtype=float)
    data = (_lorentz(i, 900, 12, 2.0e6) + _lorentz(i, 2000, 20, 1.2e6)
            + _lorentz(i, 3100, 8, 0.8e6))
    data.astype("int32").tofile(os.path.join(pdata, "1r"))
    np.zeros(n, dtype="int32").tofile(os.path.join(pdata, "1i"))
    return expt


def make_bruker_2d(folder, n_f1=128, n_f2=256):
    """A processed 2D Bruker dataset: two Gaussian cross peaks on 2rr."""
    import numpy as np

    expt = os.path.join(folder, "selftest_2d")
    pdata = os.path.join(expt, "pdata", "1")
    os.makedirs(pdata, exist_ok=True)
    _write_params(os.path.join(expt, "acqus"), {"NUC1": "1H", "BF1": 400.13, "O1": 2000.0, "NS": 1})
    _write_params(os.path.join(expt, "acqu2s"), {"NUC1": "13C", "BF1": 100.61, "O1": 7546.0})
    _write_params(os.path.join(pdata, "procs"),
                  {"SW_p": 4000.0, "SI": n_f2, "SF": 400.13, "OFFSET": 10.0})
    _write_params(os.path.join(pdata, "proc2s"),
                  {"SW_p": 15000.0, "SI": n_f1, "SF": 100.61, "OFFSET": 150.0})
    y, x = np.mgrid[0:n_f1, 0:n_f2].astype(float)
    data = (1.0e6 * np.exp(-((x - 80) ** 2 / 50 + (y - 40) ** 2 / 30))
            + 0.6e6 * np.exp(-((x - 170) ** 2 / 80 + (y - 90) ** 2 / 40)))
    data.astype("int32").tofile(os.path.join(pdata, "2rr"))
    return expt


SELFTEST_CIF = """data_selftest
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   4.0
_cell_length_b   4.0
_cell_length_c   4.0
_cell_angle_alpha   90.0
_cell_angle_beta   90.0
_cell_angle_gamma   90.0
_symmetry_Int_Tables_number   1
_chemical_formula_sum   'Li2 Mn0.3 Fe0.7 O1'
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Li  Li1  1  0.0  0.0  0.0  1
  Li  Li2  1  0.5  0.5  0.0  1
  Mn  M1   1  0.5  0.0  0.0  0.3
  Fe  M1   1  0.5  0.0  0.0  0.7
  O   O1   1  0.0  0.5  0.5  1
"""


class _Report:
    def __init__(self):
        self.steps: dict[str, dict] = {}

    def run(self, name, fn):
        start = time.perf_counter()
        try:
            detail = fn()
            self.steps[name] = {"ok": True, "detail": detail}
        except Exception:  # noqa: BLE001 -- recorded, the other steps still run
            self.steps[name] = {"ok": False, "error": traceback.format_exc()}
        self.steps[name]["seconds"] = round(time.perf_counter() - start, 2)

    @property
    def ok(self):
        return all(step["ok"] for step in self.steps.values())


def run(out_dir: str) -> int:
    """Run every step into ``out_dir``; returns 0 if all passed."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # Keep the GUI's settings and simulation journal away from the real
    # home folder, and never check for updates from a self-test.
    home = os.path.join(out_dir, "home")
    os.makedirs(home, exist_ok=True)
    os.environ["HOME"] = home
    os.environ["USERPROFILE"] = home
    os.environ.setdefault("OPENNMR_SIMULATIONS_DIR", os.path.join(home, "simulations"))
    os.environ["OPENNMR_UPDATE_CHECK"] = "0"
    os.environ["OPENNMR_BACKGROUND_LOAD"] = "0"
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import numpy as np

    report = _Report()
    numbers: dict[str, list] = {}
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    def imports():
        import opennmr
        failed = {}
        count = 0
        for info in pkgutil.walk_packages(opennmr.__path__, "opennmr."):
            if info.name.startswith(SKIP_IMPORT_PREFIXES):
                continue
            try:
                importlib.import_module(info.name)
                count += 1
            except Exception as exc:  # noqa: BLE001
                failed[info.name] = f"{type(exc).__name__}: {exc}"
        # The two start-up modules beside the package: the launcher, and the
        # crash log it installs (a build that leaves the crash log out still
        # starts, silently without one -- caught here instead).
        for name in ("opennmr_launch", "opennmr_crashlog"):
            try:
                importlib.import_module(name)
                count += 1
            except Exception as exc:  # noqa: BLE001
                failed[name] = f"{type(exc).__name__}: {exc}"
        if failed:
            raise RuntimeError(f"{len(failed)} module(s) failed to import: "
                               + json.dumps(failed, indent=1))
        return {"modules": count}

    def version():
        from importlib import metadata
        from opennmr._version import __version__
        meta = metadata.version("opennmr")
        if meta != __version__:
            raise RuntimeError(f"_version says {__version__}, metadata says {meta}")
        return {"version": __version__}

    paths = {}

    def read_process():
        from opennmr import io
        paths["1d"] = make_bruker_1d(data_dir)
        paths["2d"] = make_bruker_2d(data_dir)
        s1 = io.load(paths["1d"], vendor="bruker")
        s2 = io.load(paths["2d"], vendor="bruker")
        s1 = s1[0] if isinstance(s1, list) else s1
        s2 = s2[0] if isinstance(s2, list) else s2
        paths["s1"], paths["s2"] = s1, s2
        numbers["spectrum_1d"] = np.real(np.asarray(s1.data)).ravel().tolist()
        numbers["spectrum_2d_sum"] = np.real(np.asarray(s2.data)).sum(axis=0).tolist()
        return {"shape_1d": list(np.shape(s1.data)), "shape_2d": list(np.shape(s2.data))}

    def simulate_and_fit():
        from scipy.optimize import curve_fit

        from opennmr.notebook import simulations
        ppm = np.linspace(-150, 250, 2048)
        csa = simulations.haeberlen_csa_tensor(50.0, 80.0, 0.4)
        sim = simulations.make_static_csa_spectrum(csa, ppm, fwhm=3.0, n_theta=300, n_phi=600)
        numbers["csa_static"] = np.asarray(sim["normalised_real_spec"]).tolist()
        paths["csa"] = (ppm, np.asarray(sim["normalised_real_spec"]))
        x = np.linspace(-10, 10, 2001)
        y = _lorentz(x, 1.234, 0.8, 5.0)
        popt, _ = curve_fit(_lorentz, x, y, p0=(1.0, 1.0, 4.0))
        numbers["lorentz_fit"] = popt.tolist()
        if not np.allclose(popt, (1.234, 0.8, 5.0), rtol=1e-6):
            raise RuntimeError(f"fit returned {popt}")
        return {"fit": popt.tolist()}

    def cif_structure():
        from opennmr.notebook import pnmr_simulator
        path = os.path.join(data_dir, "selftest.cif")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(SELFTEST_CIF)
        structure, site_data = pnmr_simulator.read_cif_with_occupancies(path, atom_of_interest="Li")
        table = pnmr_simulator.neighbour_distance_table(structure, site_data, "Li", max_r=3.0)
        numbers["cif_distances"] = table["Distance"].round(10).tolist()
        if len(structure) != 4 or table.empty:
            raise RuntimeError(f"{len(structure)} sites, {len(table)} neighbour pairs")
        return {"sites": len(structure), "pairs": len(table)}

    def tls():
        from opennmr import update
        stats = update.ssl_context().cert_store_stats()
        if not stats.get("x509_ca"):
            raise RuntimeError(f"no CA certificates loaded: {stats}")
        return {"ca_certificates": stats["x509_ca"]}

    def figures():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from opennmr.plotting.oned import plot_1d
        from opennmr.plotting.twod import plot_2d_contour
        written = []
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        plot_1d(ax, paths["s1"])
        fig.savefig(os.path.join(out_dir, "figure_1d.png"))
        # Every vector format Save Figure offers. matplotlib imports each writer by
        # name only when saving, so a frozen app that left one out passed every
        # bitmap check and failed here for the user (2026-10-29: "No module named
        # 'matplotlib.backends.backend_svg'").
        for extension in ("svg", "pdf", "eps"):
            name = f"figure_1d.{extension}"
            fig.savefig(os.path.join(out_dir, name))
            if os.path.getsize(os.path.join(out_dir, name)) < 500:
                raise RuntimeError(f"{name} is empty")
            written.append(name)
        plt.close(fig)
        written.append("figure_1d.png")

        result = plot_2d_contour(paths["s2"], font="DejaVu Sans")
        candidates = result if isinstance(result, (tuple, list)) else (result,)
        fig2 = next((c for c in candidates if hasattr(c, "savefig")), None) or plt.gcf()
        fig2.savefig(os.path.join(out_dir, "figure_2d.png"), dpi=100)
        plt.close("all")
        written.append("figure_2d.png")

        ppm, spec = paths["csa"]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(ppm, spec, color="black")
        ax.invert_xaxis()
        ax.set_xlabel("δ / ppm")
        fig.savefig(os.path.join(out_dir, "figure_csa.png"))
        plt.close(fig)
        written.append("figure_csa.png")
        return {"figures": written}

    def gui():
        from PySide6.QtWidgets import QApplication

        from opennmr.gui.main_window import MainWindow
        app = QApplication.instance() or QApplication([sys.argv[0]])
        window = MainWindow()
        window.resize(*GUI_SIZE)
        window.show()
        window._load(paths["1d"], vendor="bruker")
        for _ in range(20):
            app.processEvents()
        image = window.grab().toImage()
        image.save(os.path.join(out_dir, "gui_main_window.png"))
        detail = {"spectra_open": window.spectrum_list.count(),
                  "size": [image.width(), image.height()]}
        window.close()
        window.deleteLater()
        app.processEvents()
        if detail["spectra_open"] != 1:
            raise RuntimeError(f"expected 1 spectrum open, found {detail['spectra_open']}")
        return detail

    report.run("imports", imports)
    report.run("version", version)
    report.run("read_process", read_process)
    report.run("simulate_and_fit", simulate_and_fit)
    report.run("cif_structure", cif_structure)
    report.run("tls", tls)
    if report.steps["read_process"]["ok"] and report.steps["simulate_and_fit"]["ok"]:
        report.run("figures", figures)
        report.run("gui", gui)

    with open(os.path.join(out_dir, "numbers.json"), "w", encoding="utf-8") as handle:
        json.dump(numbers, handle)
    summary = {
        "ok": report.ok,
        "frozen": bool(getattr(sys, "frozen", False)),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "steps": report.steps,
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return 0 if report.ok else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir = argv[0] if argv else os.path.join(os.getcwd(), "opennmr-selftest")
    code = run(out_dir)
    # A windowed app on Windows has no console (sys.stdout is None); the
    # report file is the real output, this line is a courtesy.
    if sys.stdout is not None:
        print(f"openNMR self-test {'passed' if code == 0 else 'FAILED'}: see "
              f"{os.path.join(os.path.abspath(out_dir), 'report.json')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
