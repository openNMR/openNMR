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
