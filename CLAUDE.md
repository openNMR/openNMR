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
- `.github/workflows/tests.yml` runs the suite on every push and pull
  request to `main`, without `test_data/`. Check the Actions tab after
  pushing; still run the tests locally first, since CI cannot run the
  real-data ones.

## GUI layout (`src/opennmr/gui/main_window.py`)

- **Toolbar** (`_build_main_toolbar`): Load, Process, Quantify, Analyse,
  Compare, Simulate -- icons only, one `ModeButton` each
  (`gui/widgets/mode_button.py`), with any dropdown arrow centred under
  its icon -- then a Settings gear at the far end. Each mode raises a
  left-hand panel held in `side_tabs` (tab names Data, Processing,
  Quantify, Analysis, Compare, Simulate).
- **Open experiments** (`open_experiments_box`) sit under the side tabs
  in the `left_panel` splitter, shown in Load, Analyse and Compare.
- **Settings** is a non-modal dialog (`_SettingsDialog`) hosting the
  imaging controls; add further settings there.
- **Simulate** draws on its own `simulation_canvas` (plain matplotlib),
  shown only while that panel is open; the maths is in
  `opennmr.simulate` (ports of `leonmr` code).
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
