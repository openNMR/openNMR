# openNMR -- notes for Claude sessions

Several people work on this repo, often through Claude. Most wasted
effort here has come from building something that already existed or
was already in progress, so start every session with the first section.

## Before changing anything

1. **`git pull` first.** Contributors push straight to `main`, and a
   local clone falls behind by dozens of commits within days.
2. **Read the top of `TODO.md`** ("Where things stand" and "What's
   next"). It is the shared record of what is built, what is deferred
   and why, and what was tried and rejected. The rounds below it are
   newest first.
3. **Check an "open" item against the code before building it.**
   `TODO.md` is append-only and goes stale: items have been found still
   marked open after they were implemented. `grep` for the feature
   first.
4. **Reuse before writing.** `src/opennmr/notebook/` (Euan's package:
   CSA fitting, SIMPSON, pNMR, BMS, simulations) is the fitting and
   simulation engine. The GUI calls into it, through
   `notebook/bridge.py`, rather than reimplementing it.
5. **Check `TODO.md`'s "What's next" section 0 before writing a
   reader or a PFG/diffusion feature.** Finished, tested modules are
   parked there unregistered so they cannot conflict with an in-flight
   merge -- currently `io/spinsolve.py` (Magritek Spinsolve PGSTE) and
   `io/bruker_pfg.py` (Bruker gradients without a `diff.xml`). They
   are invisible to `grep` for GUI features because nothing calls them
   yet, which is exactly how they get rebuilt from scratch by mistake.

## After changing something

- Add a round to `TODO.md`, newest first, directly under "How this file
  is organised": what was asked, what was wrong, what was built, and
  anything tried and rejected, so the next session does not repeat it.
  Update the test count in "Where things stand".
- Run the tests (below) before committing.

## Tests

```
pip install -e ".[test]"
pytest                          # full suite, ~9 minutes
pytest tests/test_gui.py -k toolbar   # a subset while iterating
```

- `tests/conftest.py` sets Qt to offscreen and pins loads to run
  synchronously; no display is needed. It also deletes every window a
  test leaves open, after each test -- without that, the suite's
  several hundred `MainWindow`s leak to ~13 GB and GitHub's runner kills
  the job. Tests need not close their own windows.
- Real instrument data lives in `test_data/`, which is **gitignored**.
  Tests that need it skip when it is absent, so a fresh clone runs the
  suite with those skipped. A new real-data test must do the same
  (`pytest.mark.skipif(not os.path.isdir(...))`).
- Share helpers between test files as `from test_gui import …`, never
  `from tests.test_gui import …`: CI runs plain `pytest`, where only
  `tests/` is on the import path, so the `tests.` form is a collection
  error that stops the whole run in about a minute. It still works under
  `python -m pytest` (which adds the repo root), so run plain `pytest`
  before pushing.
- `.github/workflows/tests.yml` runs the suite on every push and pull
  request to `main`, without `test_data/`. Check the Actions tab after
  pushing; still run the tests locally first, since CI cannot run the
  real-data ones.

## GUI layout (`src/opennmr/gui/main_window.py`)

- **Toolbar**: one row of launcher tiles (`self.tile_bar`, taken from
  `side_tabs`, a `WrappingTabWidget`), in the user's order: Open,
  Process, Compare, Fit, Simulate, Dynamics, Adv. Proc., Apply,
  Periodic Table, Settings. Panels are *built* in an older order (builders read each
  other's widgets) and *added* in this one -- see `__init__`. Ctrl+1..9
  (`_WINDOW_MENU_PANELS`) follow the toolbar order.
- **Periodic Table** (`gui/periodic_table_panel.py`, maths in
  `notebook/periodic_table.py`): its tile's panel is the Larmor
  calculator; the table replaces the plot through `self.plot_stack`
  (a QStackedWidget holding `canvas_tabs`), built on first use. Nuclear
  data come only from `notebook.utils._GAMMA_DF` -- do not add a copy.
- **Collapsible sections** (`widgets/collapsible_section.py`) start
  collapsed. Code that sends the user to one opens it with
  `setExpanded(True)`.
- **Open experiments** (`open_experiments_box`) sit under the side tabs
  in `left_splitter`, on every tile but Simulate.
- **Title / notes** open from a strip at the window's right edge
  (`self.notes_drawer`, `widgets/side_drawer.py`; View ▸ Title / Notes,
  Ctrl/Cmd+T), closed by default. Not under Open experiments any more.
- **Plot kinds and default styles**: a canvas announces its kind
  (`_enter_plot_kind`: 1d, 2d, pseudo2d, compare, curve) before drawing;
  MainWindow applies the saved default for that kind
  (`~/.opennmr/figure_styles.json`; "curve" falls back to "1d").
  Notebook figures shown in the GUI go through
  `MplCanvas.show_external_plot(draw, kind)`.
- **Settings** is a non-modal dialog (`_SettingsDialog`) hosting the
  imaging controls; add further settings there.
- **Simulate** draws on its own `simulation_canvas` (plain matplotlib),
  shown only while that panel is open; the maths is in
  `opennmr.simulate` (ports of `leonmr` code).
- **DNP** is Apply ▸ DNP (`_build_dnp_section`): "Enhancement" and
  "Field Sweep Profile", each a step-by-step `gui/dnp_panel.py`
  `DnpWorkflowPanel` over `notebook/dnp_workflow.py` (align on the
  reference line's shape, confirm, then measure). Not in Compare any more.
- **Dynamics** is Fit Curve, then two collapsible sections from
  `gui/dynamics_panels.py`: Relaxation vs Temperature (BPP / Arrhenius /
  Hendrickson-Bray; maths `notebook/vt_relaxation.py`) and Exchange
  Lineshape (ion hopping, Bloch-McConnell; `notebook/exchange.py`, the
  successor to Euan's XReSpecs script). Both draw on the Fit view.
- **pNMR fit** is Fit ▸ pNMR (CIF bond pathways) (`gui/pnmr_fit_panel.py` over
  `notebook/pnmr_fit.py`). Its structural model is Simulate ▸ pNMR (CIF)'s
  (`PnmrSimulationPanel.fit_model_kwargs()`) -- never add a second copy.
  Hover/click read-outs are `pnmr_fit.environments_at_shift`; plot extras go
  through `MplCanvas.show_slice_fit(..., overlay=)`, which redraws them on a
  restyle.
- **Interactions** (CSA, quadrupolar, J) are Fit ▸ / Simulate ▸ Combined
  Hamiltonian only (user request, 2026-10-06): add peaks, add interactions
  to each. Its "Strongly coupled peaks" model (`gui/combined_section.py`)
  is the J panel locked to spin-system mode. The old J-coupling,
  CSA/Dipolar and Quadrupolar panels are still built but sit in hidden
  hosts (`_retired_fit_host`, `_retired_sim_host`), because kept fits and
  saved states restore through them. Do not give them menu sections again.
  The fit's starting model is `combined_fitting.starting_model` (the live
  preview).
- **Several nuclei on one spectrum** (47Ti + 49Ti; a 63Cu coil line on
  23Na) are per-peak, in the Combined panels' Selected peak: Nucleus,
  Same chemistry as (a link), Not shared; Add isotope partner. Maths:
  `notebook/isotopes.py` (placement by IUPAC Ξ, the
  `"Frequency Ratio Xi (%)"` column of `_GAMMA_DF` -- never γ ratios,
  which miss reference shielding), `Site.link_to` / `isotope_partner`,
  `resolve_link_dicts`. GUI site dicts: `"isotope"` (absent = observed),
  `"link"`. A linked partner has fit rows only for its link's `free`.
- **Fit advice** ("what the fit tells you / what to try next", plain
  language) is `notebook/fit_advice.py`, shown by `widgets/advice_box.py`
  under Combined, Multi-spectrum, Isotropic, the Pseudo-2D series boxes
  and Fit Curve. A new fit should get an `advise_*` function there and
  an `AdviceBox`, not hand-written warnings. The MAS rate is a fittable
  parameter (`fit_combined(fit_mas=...)`), fixed by default.
- **Starting estimates** (one interaction kind at a time) are
  `notebook/start_estimates.py`, behind Combined's "Estimate Starting
  Values". CSA/quadrupolar use powder-pattern moments (M2 is kept under
  MAS); do not replace them with edge/horn rules (see TODO 2026-10-06 r3).
- **Batch relaxation** (Dynamics ▸ Batch, `gui/batch_relaxation_panel.py`;
  notebook `notebook/batch_relaxation.py`) replays the selected reference
  series' processing and analysis on each series through the ordinary
  handlers (`MainWindow._batch_run_one`), then sends rows to Relaxation
  vs Temperature. Add new per-series steps there, not in a parallel path.
- **Baselines**: `"spline_auto"` (`processing.fitting.auto_spline_baseline`)
  finds baseline anywhere by the Golotvin–Williams test; it shares
  `_baseline_kwargs` with every other method (`baseline_points` raw key).
- **Relaxometry extras** (three-exponential models, threshold regions,
  trapezoid/Simpson, correlation matrices) are `processing/relaxometry.py`,
  shared by *both* relaxation fitters (`processing/relaxation.py`, the GUI's,
  and `notebook/relaxation.py`, a separate copy) -- add a new relaxation
  model to both `RELAXATION_MODELS` or through this module. Fit Curve's
  model combo is saved by key (`model_key`): append new models, never insert.
- **DFT tensors** (CASTEP .magres, VASP OUTCAR, ORCA .out, Gaussian .log)
  are `notebook/dft_nmr.py`; Simulate ▸ DFT is `gui/dft_panel.py`, a
  `CombinedHamiltonianPanel` subclass (its sites, settings and Simulate are
  the Combined panel's). A new code's reader returns the same calc dict
  (atoms with `sigma`, `efg` in V/m², plus `method` and `g0`). Euler angles
  must be taken with `relative_euler(..., return_frame=True)`: the EFG axis
  signs are part of the answer. Shift scales are per-element calibrations
  from reference compounds (`notebook/dft_calibration.py`, σ_calc on δ_exp);
  pass them to `dft_sites(calibration=...)` -- do not add another
  referencing path. The VASP G=0 switch re-reads every file at once.
- **Shearing and scaling of 2D spectra** (MQMAS, STMAS, DAS, J-resolved,
  PASS/MAT, custom ratios or pathway coefficients) are
  `notebook/shearing.py` (`make_shear_plan`, `shear_2d`, `estimate_shear`,
  `sideband_positions`, `isotropic_from_peak`); Adv. Proc. ▸ Shearing and
  Scaling (2D) is `gui/shear_panel.py`. Coefficients and kappas are in the
  *displayed*-axis frame; F1's sense relative to the coherence is
  `f1_sign`, measured by `estimate_shear` -- never flip a kappa's sign by
  hand to "fix" a spectrum. Frequencies for the shear come from the FFT
  bins (carrier-relative), never from a re-referenced ppm axis. A sheared
  spectrum keeps the phase baked into its data as `baked_phase_f1/f2`
  (the recorded/user phase keys are dropped), so it can be sheared again.
  Derivation and checks: `SCRIPT_openNMR_Shearing_20261009.ipynb`.
- **Correlation matrices** of any fit are `notebook/fit_correlation.py`
  (`correlation_input` adapts each fit shape; add an adapter there for a
  new kind of fit). The GUI draws them through
  `MainWindow._show_correlation_matrix(canvas, fit)`.
- **Quantify** is one page: Peaks, Integration Regions, then the
  Spectral Fitting section (`_build_spectral_fit_section`). Whether the
  fit uses the regions is two explicit checkboxes there.
- **Analyse's and Simulate's sub-panels** live in `analysis_sub_tabs`
  and `simulate_sub_tabs`, `WrappingTabWidget`s
  (`gui/widgets/wrapping_tabs.py`) with their strips hidden. A
  sub-panel added with `addTab` appears in that button's dropdown
  automatically.
- **Plot views** (`canvas_tabs`: FID, Processing Preview, Spectrum,
  Spectral Fit, Fit, Compare) keep a visible strip directly above the
  plot. A toolbar "View" menu was tried and rejected as unintuitive.
- Switch panels and views by code with `setCurrentWidget` /
  `_show_side_tab(name)`; the toolbar follows `currentChanged`.
- **Toolbar icons**: the artwork exports are in `assets/tab_icons/`.
  After re-exporting one, run `python packaging/make_toolbar_icons.py`
  to regenerate `gui/resources/toolbar/`. The editable Affinity
  Designer sources are not in git.
- **Planned GUI work** (CSA fit/simulation split, view sidebar) is in
  the newest `TODO.md` rounds -- check there before starting any of it.
