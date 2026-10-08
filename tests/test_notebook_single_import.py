"""One import for notebooks: ``import opennmr.notebook as nmr`` reaches
everything the tutorial notebooks used to import from half a dozen
sub-packages, and those names are the same objects as before (not copies
or wrappers), so old and new notebooks give identical results."""
import importlib

import pytest

import opennmr.notebook as nmr


# (attribute on nmr, module it used to be imported from, name there)
OLD_IMPORTS = [
    ("setup_plotting", "opennmr.notebook.plotting", "setup_plotting"),
    ("expand_pcov", "opennmr.notebook.fitting", "expand_pcov"),
    ("find_isotope_info", "opennmr.notebook.utils", "find_isotope_info"),
    ("compute_BF1_SFO1", "opennmr.notebook.utils", "compute_BF1_SFO1"),
    ("jcoupled_spectrum", "opennmr.notebook.simulations", "jcoupled_spectrum"),
    ("multi_site_jcoupled_spectrum", "opennmr.notebook.simulations",
     "multi_site_jcoupled_spectrum"),
    ("Site", "opennmr.notebook.combined_hamiltonian", "Site"),
    ("simulate_combined", "opennmr.notebook.combined_hamiltonian", "simulate_combined"),
    ("ordered_qualitative_colours", "opennmr.plotting.accessibility",
     "ordered_qualitative_colours"),
    ("load", "opennmr.io", "load"),
    ("results_from_spectrum", "opennmr.notebook.bridge", "results_from_spectrum"),
    ("transfer_sr", "opennmr.notebook.referencing", "transfer_sr"),
    ("save_state", "opennmr.notebook.state", "save_state"),
    ("load_state", "opennmr.notebook.state", "load_state"),
    ("describe_state", "opennmr.notebook.state", "describe_state"),
    ("run_bms", "opennmr.notebook.bms", "run_bms"),
    ("make_particle", "opennmr.notebook.bms", "make_particle"),
    ("particles_to_array", "opennmr.notebook.bms", "particles_to_array"),
    ("build_box", "opennmr.notebook.bms", "build_box"),
]

MODULE_ALIASES = [
    ("pt", "opennmr.notebook.periodic_table"),
    ("periodic_table", "opennmr.notebook.periodic_table"),
    ("pf", "opennmr.notebook.pnmr_fit"),
    ("pnmr_fit", "opennmr.notebook.pnmr_fit"),
    ("dw", "opennmr.notebook.dnp_workflow"),
    ("dnp_workflow", "opennmr.notebook.dnp_workflow"),
    ("acc", "opennmr.plotting.accessibility"),
    ("accessibility", "opennmr.plotting.accessibility"),
    ("bloch", "opennmr.simulate.bloch"),
    ("bms", "opennmr.notebook.bms"),
]


@pytest.mark.parametrize("attr, module, name", OLD_IMPORTS)
def test_old_import_reachable_from_nmr(attr, module, name):
    assert getattr(nmr, attr) is getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("alias, module", MODULE_ALIASES)
def test_module_alias(alias, module):
    assert getattr(nmr, alias) is importlib.import_module(module)


def test_jcoupling_fitting_public_names_reachable():
    # The notebook did `from opennmr.notebook.jcoupling_fitting import *`.
    jf = importlib.import_module("opennmr.notebook.jcoupling_fitting")
    public = [n for n, v in vars(jf).items()
              if not n.startswith("_") and callable(v)
              and getattr(v, "__module__", None) == jf.__name__]
    assert public
    missing = [n for n in public if getattr(nmr, n, None) is not getattr(jf, n)]
    assert not missing, missing


def test_all_names_resolve_and_are_unique():
    missing = [n for n in nmr.__all__ if not hasattr(nmr, n)]
    assert not missing, missing
    dupes = {n for n in nmr.__all__ if nmr.__all__.count(n) > 1}
    assert not dupes, dupes


def test_star_import_gives_bare_names():
    ns = {}
    exec("from opennmr.notebook import *", ns)
    for name in ("load", "Site", "pt", "pf", "dw", "acc", "bloch",
                 "expand_pcov", "results_from_spectrum", "run_bms"):
        assert name in ns, name
