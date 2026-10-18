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
   merge. (`io/spinsolve.py` and `io/bruker_pfg.py`, once parked
   there, have been wired in since 2026-09-23.)
6. **Magritek / Spinsolve** is `io/spinsolve.py`: PGSTE diffusion
   (`read_spectrum`) and plain 1D/2D (`read_spinsolve_1d/2d`, checked
   against Spinsolve's own `spectrum.pt1`/`cosySpectrum.pt2`). Read
   with this project's own code -- never nmrglue (user's instruction).
   A Spinsolve 2D type other than F1 `Complex` is refused until an
   example is checked; extend `_SPINSOLVE_WINDOWS` only with a window
   verified against Spinsolve's output.

## After changing something

- Add a round to `TODO.md`, newest first, directly under "How this file
  is organised": what was asked, what was wrong, what was built, and
  anything tried and rejected, so the next session does not repeat it.
  Update the test count in "Where things stand".
- Run the tests (below) before committing.

## Tests

```
pip install -e ".[test]"
pytest                          # full suite: long (~2,500 tests build a MainWindow)
pytest tests/test_gui.py -k toolbar   # a subset while iterating
```

- `tests/conftest.py` sets Qt to offscreen and pins loads to run
  synchronously; no display is needed. It also deletes every window a
  test leaves open, after each test -- without that, the suite's
  several hundred `MainWindow`s leak to ~13 GB and GitHub's runner kills
  the job. Tests need not close their own windows. **Parent every
  QObject a window or panel makes** (`QTimer(self)`; a timer made before
  `super().__init__` gets `setParent(self)` right after): a parentless
  timer/thread outlives the window and keeps it alive through its
  connections -- `tests/test_window_cleanup.py` fails if it does. A
  cancelled CI run ("The operation was canceled" partway) is this leak
  (out of memory) or the job's `timeout-minutes`.
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
  Process, Compare, Fit, Integrate, Simulate, Dynamics, Adv. Proc., Apply,
  Periodic Table, Settings. Panels are *built* in an older order (builders read each
  other's widgets) and *added* in this one -- see `__init__`. Ctrl+1..9
  (`_WINDOW_MENU_PANELS`) follow the toolbar order; a tenth tile gets its
  Window-menu entry but no number key.
- **Integrate** (`_build_integrate_panel`, since 2026-10-16): Integration
  Regions (built with the Fit panel as `_integration_content`, shown here),
  then DNP, Signal-to-Noise and CP Build-up (moved from Apply, which keeps
  eNMR and Imaging). There is no Integrate button: a region dragged (Select
  and Integrate Regions from Plot, Ctrl/Cmd+I), typed (the delegate's
  `commitData` -- a user's edit only, never the table being refilled on
  reselection), removed or found automatically is integrated at once by
  `_auto_integrate_regions`, which reports problems in the results line, not
  a dialog. `_on_integrate_regions_clicked()` (no argument) still shows
  dialogs; batch relaxation calls it. Regions are shaded on Fit and Integrate
  (`_regions_tab_active`).
- **A plot's own display options are on its right-click menu**, beside Plot
  Style… (user request, 2026-10-16), never a section of a side panel:
  `MplCanvas.register_plot_options(owner, label, callback)`, shown while the
  plot on show is `owner`'s -- `canvas.plot_owner = owner` after drawing
  (cleared by every rebuild, kept through a restyle); "curve_fit" and
  "fit_components" are recognised by the canvas itself. The options widget
  lives in a `gui/plot_options_dialog.PlotOptionsDialog` (non-modal, owned by
  the panel). Fit plot: components and colours, Curve Fit Plot (with the
  uncertainty band), Relaxation vs Temperature Plot, Exchange Lineshape Plot
  and Regularisation Plot work this way.
- **Periodic Table** (`gui/periodic_table_panel.py`, maths in
  `notebook/periodic_table.py`): its tile's panel is the Larmor
  calculator; the table replaces the plot through `self.plot_stack`
  (a QStackedWidget holding `canvas_tabs`), built on first use. Nuclear
  data come only from `notebook.utils._GAMMA_DF` -- do not add a copy.
- **Collapsible sections** (`widgets/collapsible_section.py`) start
  collapsed. Code that sends the user to one opens it with
  `setExpanded(True)`.
- **Open experiments** (`open_experiments_box`, inside
  `open_experiments_panel` with its strip `open_experiments_toggle`) sit
  under the side tabs in `left_splitter`, on every tile but Simulate and
  Periodic Table; minimised to the strip by View ▸ Open Experiments
  (Ctrl/Cmd+E) or a click, remembered (`open_experiments_open`).
- **Shortcuts** (2026-10-17): Ctrl/Cmd+1..9 then 0 are the tiles (0 = NMR
  Periodic Table); Reset View is Ctrl/Cmd+Shift+H; facts Ctrl/Cmd+B.
- **Buttons whose slot takes an optional first argument** must connect
  through a lambda: `clicked(bool)` passes `False` into it (Compare ▸ Plot
  never switched view; BPP Import CSV opened the file `False`).
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
  imaging controls; add further settings there. The Settings tile's
  panel (`_build_settings_panel`) holds the remembered interface choices
  in `~/.opennmr/ui_settings.json`: **Click and drag** (`drag_mode`,
  "zoom" default; `MplCanvas.set_drag_mode`, shown by the toolbar's
  checked Pan/Zoom button -- a bare `MplCanvas` still defaults to pan),
  **Sections shown** (`hidden_sections`; `MainWindow.OPTIONAL_SECTIONS`
  lists what can be hidden -- add a section there, never delete one;
  `"tab:<label>"` hides a whole launcher tile with its Window-menu entry),
  **Export size in** (`export_unit`; `widgets/typed_spinbox.LengthSpinBox`
  keeps inches inside, only the display converts -- never store cm),
  and the **default data folder** (`data_folder`, held app-wide in
  `gui/data_directory.py`; every "open data" dialog starts from
  `data_directory.start_directory()`).
- **Toolbar zoom/pan drags**: matplotlib's press connects a motion
  callback whose id is in `_zoom_info.cid` / `_pan_info.cid`. Never set
  those infos to None without disconnecting it (use
  `MplCanvas._cancel_navigation_drag`): the orphan raises on every mouse
  move (a Mac user's terminal, 2026-10-15).
- **Number boxes users type long values into** (DPI, Truncate to,
  zero-fill sizes) are `widgets/typed_spinbox.TypedSpinBox`: Qt's
  QSpinBox refuses a keystroke whose intermediate value passes the
  maximum, which read as "capped at three figures".
- **Simulate** draws on its own `simulation_canvas` (plain matplotlib),
  shown only while that panel is open; the maths is in
  `opennmr.simulate` (ports of `leonmr` code).
- **DNP** is Integrate ▸ DNP (`_build_dnp_section`; Apply ▸ DNP until 2026-10-16): "Enhancement" and
  "Field Sweep Profile", each a step-by-step `gui/dnp_panel.py`
  `DnpWorkflowPanel` over `notebook/dnp_workflow.py` (align on the
  reference line's shape, confirm, then measure). Not in Compare any more.
- **Signal-to-noise** is Integrate ▸ Signal-to-Noise (`_build_snr_section`,
  `gui/snr_panel.py` over `notebook/snr.py`): signal and noise regions
  typed or dragged on Compare, shared by default or per spectrum (marked
  †, logged, and stated in the result). S/N = height / σ of the detrended
  noise, with σ's uncertainty from the *independent* noise points
  (`correlation_time`), a conservative 95 % interval, TopSpin's sino
  (H/2σ, and `sino_topspin` with TopSpin's own arithmetic), Ph. Eur.
  2H/h, integral S/N and a detection threshold. 2D spectra:
  `signal_to_noise_2d` (F2 × F1 boxes, plane-detrended noise, τ per
  dimension). Every convention side by side: `snr.CONVENTIONS` /
  `snr_conventions(result)` -- add a new convention there. Kept as
  `fits["snr"]` through `_store_dnp`. Do not add another noise estimator.
  The panel's result is `snr.snr_summary(result)` (a row per quantity,
  a column per spectrum) in a `FitTable` sized to its rows; the
  sentences are under Full Report…. Its plot's text sizes come from the
  Compare canvas's legend style, read at draw time -- never hard-code
  a size there. Its spectra keep their Compare colours
  (`MainWindow._compare_colours_for` -> `plot_snr(colours=)`).
- **Dynamics** is Fit Curve, then collapsible sections: Relaxation vs
  Temperature (BPP / Arrhenius / Hendrickson-Bray; maths
  `notebook/vt_relaxation.py`) and Exchange Lineshape (ion hopping,
  Bloch-McConnell; `notebook/exchange.py`, the successor to Euan's XReSpecs
  script), both `gui/dynamics_panels.py`, and Regularisation
  (`gui/regularisation_panel.py` over `notebook/regularisation.py`). All draw
  on the Fit view. Fit Curve's peak browser (`curve_peak_bar`, ◀ peak # ▶)
  is the plot views' strip corner (`WrappingTabWidget.setCornerWidget`),
  shown while the Fit view shows a curve fit (`_sync_curve_peak_bar`).
- **Regularisation** (Tikhonov ILT; T1/T2/D distributions and T1-T2,
  T2-T2, D-T2 maps) is `notebook/regularisation.py`, a port of the user's
  MATLAB_RELAX_ALGORITHM scripts (Hansen's `csvd`/`tikhonov`, Rodriguez's
  `errestik`/`etanutik`), checked number for number against the MATLAB
  files run in Octave (`tests/test_regularisation.py`). Keep that
  equivalence: the "matlab" solver must still reproduce those numbers. The
  |x|-prior iterations weaken the smoothing as they go (iterations are a
  second regularisation parameter), and with fewer points than grid points
  the error estimate's λ falls into round-off -- both are said in the
  result's notes. `processing/ilt.py` (DOSY, NNLS + L-curve) is separate and
  diffusion-only; do not merge them without checking DOSY's numbers.
- **pNMR fit** is Fit ▸ pNMR (CIF bond pathways) (`gui/pnmr_fit_panel.py` over
  `notebook/pnmr_fit.py`). Its structural model is Simulate ▸ pNMR (CIF)'s
  (`PnmrSimulationPanel.fit_model_kwargs()`) -- never add a second copy.
  Hover/click read-outs are `pnmr_fit.environments_at_shift`; plot extras go
  through `MplCanvas.show_slice_fit(..., overlay=)`, which redraws them on a
  restyle.
- **Interactions** (CSA, quadrupolar, J) are Fit ▸ / Simulate ▸ CSA, Quad &
  J-coupling only (user request, 2026-10-06; named "Combined Hamiltonian"
  until 2026-10-12 -- the code still calls it `combined`): add peaks, add
  interactions to each. A "&" in a button's text must be written "&&"
  (Qt mnemonic); `CollapsibleSection` titles escape it themselves. pNMR's
  isotropic peaks go there through `notebook/pnmr_export.py` and the
  `gui/pnmr_export_row.py` row on both pNMR panels
  (`MainWindow._export_pnmr_sites`). Peaks are multi-select; "Copy to
  Peaks ▾" gives the selected/all peaks the current peak's interactions
  (`combined_hamiltonian.copy_interactions`). Sites with identical
  interactions + broadening are simulated **once and shifted** to each
  δiso (`broadened_sites`, used by `simulate_combined` and
  `starting_model`; not yet by the fitters' per-site sticks cache). Code
  that adds or changes many sites must use the panels' `bulk()` /
  `add_sites()` -- the Fit panel rebuilt its parameter table per site
  (6 min for 1200). Its "Strongly coupled peaks" model
  (`gui/combined_section.py`) is the J panel locked to spin-system mode. The old J-coupling,
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
  Since 2026-10-17 the menu is `baseline_family_combo` (Linear, Polynomial,
  Sinusoidal, Spline, Whittaker); the hidden `baseline_method_combo` holds
  the exact saved method. Polynomial/Sinusoidal = automatic fit + its
  coefficients (`ft.automatic_baseline_as_manual`) in `ManualBaselineBox`,
  steppable; stepping one makes it `manual_*`. Spline modes: `spline_auto`,
  `spline_points` (`baseline_spline_x`, `fitting.spline_through_points`),
  `spline` (smoothing, binned to ≤400 candidates -- never fit thousands).
- **Fonts**: `notebook/fonts.py` lists faces; Plot Style offers them; maths follows a named
  face at draw time. **Vector saves** go through `plotting/vector_export.vector_safe`
  (Illustrator drops paths over ~32k points; fonts embedded as TrueType) -- save new
  figures with `save_vector_safe`, never a bare `fig.savefig`.
- **Italic symbols**: `mathtext.default` is `"it"`; plain labels pass through
  `plotting.labels.italicise_symbols` (canvas restyle; `_italic_label` round every
  `set_xlabel`). Write descriptive subscripts `\mathrm{...}`.
- **Offset Tracker** (Integrate; `gui/offset_tracker_panel.py` over
  `notebook/offset_tracker.py`): integrals against O2/O3 in ppm (O/BF), drawn as
  bars, vertical lines or peaks (Gaussian/Lorentzian/pseudo-Voigt, user type and
  width, centred on the offsets -- the MAS spectrum when they sit on sidebands);
  what is drawn is kept as a derived 1D of the pulsed nucleus
  (`pattern_spectrum`). "triangles" was removed on 2026-10-18.
- **BPP temperature regions**: `vt_relaxation.fit_vt_regions` /
  `plot_arrhenius_regions` -- each region an independent fit.
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
  Echo-antiecho needs no special handling (the decoder's conjugation gives
  p and n opposite shears); Align Echo before shearing does break it, and
  split-t1 data must use `split_t1=True` (no shear, effective-coefficient
  referencing), never the ordinary plan.
  Derivation and checks: `SCRIPT_openNMR_Shearing_20261009.ipynb`.
- **Result read-outs** (results/summary/preview labels) are
  `widgets/report_label.py`'s `ReportLabel`: `setText(plain)` keeps the
  plain report (`text()`, logs, copies, saves) and shows it laid out by
  `gui/report_format.report_html`; `setReport(plain, html)` for a
  purpose-made layout (Simulate/Fit ▸ CSA, Quad & J-coupling: a site table
  with shared values said once, a metric row for fit quality). A new
  result label should be a `ReportLabel`, not a QLabel of long sentences.
- **Fit ▸ CSA, Quad & J-coupling's Parameters** have two layouts of the
  same `constraints` ("Show:" combo): a row per parameter
  (`parameter_table`) or a row per site (`site_parameter_table`,
  right-click for Free/Fixed/Limits; `set_cell_constraints`). The live
  preview and the fit redraw with `keep_zoom=True`
  (`MplCanvas.show_slice_fit`), so a zoom survives updates; Home resets.
- **Field boxes** (any B0 in tesla) are `widgets/field_input.FieldSpinBox` /
  `typed_spinbox.TypedDoubleSpinBox` (typed freely -- Qt's own refuses
  keystrokes on the way to e.g. 30 T), with a ¹H-MHz partner from
  `proton_frequency_box(field_spin)`; keep such partners out of panel
  state (underscore attribute names). CSA asymmetry outside its physical
  range is warned, never refused: `combined_hamiltonian.csa_asymmetry_warning`.
- **Correlation matrices** of any fit are `notebook/fit_correlation.py`
  (`correlation_input` adapts each fit shape; add an adapter there for a
  new kind of fit). The GUI draws them through
  `MainWindow._show_correlation_matrix(canvas, fit)`.
- **Quantify** is one page: Peaks, then the Spectral Fitting section
  (`_build_spectral_fit_section`); the Integration Regions it can use are
  made on the Integrate tile. Whether the fit uses the regions is two
  explicit checkboxes there.
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
- **1D/Compare y-limits** come from the full-resolution data with Plot Style's
  `baseline_gap` / `stack_gap` (`MplCanvas._fit_y_to_data`) -- never from the
  downsampled line's autoscale, which drops dips beside spikes.
- **Rows of a hypercomplex 2D** (FnMODE 4/5/6) carry F1's imaginary; `get_trace`
  rebuilds F2's for a row by `ft.hilbert_imaginary`. Kept slices find their 2D via
  `MainWindow._slice_parent_index` (identity, then `slice_parent_info`, then folder).
- **Double-click** a plot resets its view (`MplCanvas.set_double_click_resets`,
  Settings ▸ Interface, `double_click_reset`). **File ▸ Clear All History…** is
  `io.history.clear_experiment_files` + `MainWindow._clear_all_history`.
- The extracted-slice panel takes the same gaps (`slice_bottom_gap`,
  `slice_offset_fraction` from `_slice_panel_kwargs`).
- **Undo** (Edit ▸ Undo Last Processing Change, Ctrl+Alt+Z) is the processing
  log's previous state (`_record_history` -> `undo_stack`), restored through
  `_apply_restored_state`. New processing state only needs to be in the tracked
  state to be undoable.
- **Comparable intensities** are `notebook/intensity_scaling.py`: Bruker, Varian,
  Spinsolve and X-Pulse readers divide by NS at load; **JEOL does not**
  (`raw["per_scan"] = False`, divided by `total_scans`/`scans` when compared). No reader
  divides by RG (`receiver_gain`: Bruker linear RG only). Compare ▸ Normalize and CP
  Build-up both go through `comparison_factors` -- do not divide by NS anywhere else.
- **Width** (canvas toolbar, every plot): click a peak for its FWHM,
  `notebook/peak_width.py` (`peak_fwhm`, `peak_fwhm_2d`); overlap gives `estimate=True`
  and the word ESTIMATE on the plot and in the status bar.
- **Fit windows with several ranges**: `notebook/fit_ranges.parse_ppm_ranges`,
  `widgets/ranges_edit.PpmRangesEdit` ("Several ppm ranges" in the Fit window combos;
  Multi-spectrum's Window column). Gaps are shaded by `MainWindow._fit_gap_overlay`.
- **CP Build-up** is Integrate ▸ CP Build-up (`gui/cp_buildup_panel.py` over
  `notebook/cp_buildup.py`; models are Kolodziejski & Klinowski, Chem. Rev. 2002, eqs
  9, 10, 11, 28 -- keep the equation numbers in any new model).
- **Planned GUI work** (CSA fit/simulation split, view sidebar) is in
  the newest `TODO.md` rounds -- check there before starting any of it.
