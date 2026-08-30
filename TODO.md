# Backlog

Notes captured from real-data testing, not yet implemented. Grouped by area.

## Align Echo crash+masking, live-preview NS scaling, Imaging axis refresh (2026-08-30)

Three more real bugs from live testing, two of them root-causing what looked like several separate complaints.

- ~~**"When I hit align echo on a 2d or pseudo-2d... I can no longer phase anything"**~~ -- root-caused and fixed, a genuine crash silently masked by this same session's own earlier fallback fix. `read_2d`'s F2-row processing loop (`np.apply_along_axis`) calls `prepare_fid` once per row; `align_echo`'s own auto-truncation caps *each row* at its own `len(row) - ceil(echo_top)`. Fine for a single 1D FID, but a real (non-imaging) pseudo-2D series' rows have no genuine shared echo shape, so `find_echo_top` lands on a different, essentially noise-driven index every row -- the resulting per-row lengths disagree, and `np.apply_along_axis` cannot reconcile that into one array (`ValueError: could not broadcast...`). Reproduced directly on real Bruker T2 data (`bruker_T2/11`) via Align Echo. That `ValueError` was then getting caught by `_read_auto`'s own shape-guess fallback (added earlier this session for an unrelated real nmrglue bug) and silently swallowed into a `read_2d_processed` load instead -- no raw FID to reprocess from, and (for this dataset, no `2ri` on disk) no imaginary part at all, so `_capture_phase_reference` correctly refused to phase a spectrum with no `data_imag` -- explaining "can no longer phase anything" exactly.
  - **Real fix**: precompute one shared truncation cap (the *minimum* `echo_wrap_limit` across every row) before the per-row loop, and pass it in as an explicit `truncate=` for every row -- every row ends up the same final length regardless of where its own echo top happened to land, matching the same "err toward avoiding a wrapped-mirror artifact" reasoning the per-row cap already used, just applied uniformly.
  - **Also narrowed `_read_auto`'s fallback** from `except (ValueError, IndexError)` to `except IndexError` only -- `nmrglue.bruker.read_binary` already catches its own reshape `ValueError` internally (returns the unshaped 1D array with a warning printed instead of raising), so a genuine shape-guess failure always propagates as `read_2d`'s own `IndexError` (`data.shape[1]` on a 1D array) -- any other exception, like this `ValueError`, is something else going wrong and should surface to the caller normally, not be silently absorbed into a fallback that discards whatever raw-FID processing was actually requested. **General lesson**: a broad `except` added to work around one specific failure mode can end up masking a completely unrelated bug later, with no error ever shown to the user -- narrow to the actual, specific signature of the failure being handled, not just its exception type.
- ~~**"Phase is not being uniformly applied in the spectrum and processing tabs. Also the phase numbers are not staying up to date."**~~ -- investigated by directly comparing the Processing tab's live preview against the actually-loaded spectrum's own data (not just eyeballing): found a **real, large-magnitude bug**, not a phase bug at all. `_compute_processing_preview` builds its preview straight from the cached raw FID, but never divided by `NS` (scans) the way every vendor reader's own `read_1d`/`read_2d` does at load time (`data = spectrum / mass / ns`). On real data (`bruker_T2/11`, `NS=2600`) the preview came out ~2600x the amplitude of the actually-loaded spectrum -- confirmed by direct comparison, matched to floating-point precision (`~1e-11` diff) once the missing division was added. This is exactly the kind of scale mismatch that would look like "the two tabs don't agree" without actually being a phase-angle problem. The existing regression test for this preview (`test_main_window_processing_preview_phase_survives_slice_switch`) didn't catch it because its fixture defaults to `NS=1.0`, making the missing division a no-op -- added a new test with `NS=8.0` specifically to close that coverage gap. "Phase numbers not staying up to date" wasn't independently reproduced (`phase0_spin`'s value was confirmed to correctly persist across tab switches in direct testing) -- most likely the same visual-mismatch confusion, since a ~2600x-too-large preview would look wrong in ways easy to misattribute to phase specifically; revisit if it's still reported once this fix is in use.
- ~~**Imaging tab: "needs to refresh when the gradient is selected/changed and when the unit is changed... should also still be possible to scroll to zoom the axes, probably also because it isn't refreshing at all"**~~ -- root cause: changing the max-gradient value while the canvas was *already* showing the mm axis never re-derived the displayed x-data with the new gradient -- `_refresh_imaging_gradient` updated the stored gamma/gradient the canvas would use on the *next* unit switch, but never re-triggered the current one. Confirmed scroll-zoom itself was never actually broken (tested directly, in both ppm and mm modes) -- it just looked broken because the axis was frozen on stale data. Fixed: `_refresh_imaging_gradient` now re-calls `set_axis_unit("mm")` immediately whenever the canvas is already showing it; also falls back to `"ppm"` automatically if the gradient (or the selected spectrum's nucleus) becomes invalid while mm is active, rather than leaving a frozen, no-longer-meaningful conversion on screen.
- 625 tests passing (619 -> 625). Committed; not yet pushed.

## Post-plan feedback round: Compare-render slowness (real root cause), Output/live-canvas separation, phase-from-contour zoom, left-tab width again (2026-08-30)

Five items from live testing after the 6-phase plan finished. The first was reported as "loading a 2D ser should use acqus's XDIM to figure out shape, was very slow" -- investigation found something different and more valuable than what was asked for.

- ~~**"Loading the 2D ser should use acqus XDIM1/XDIM2... was very slow to load"**~~ -- **root cause found, and it wasn't shape-guessing.** Checked first: `XDIM` only ever exists in `procs`/`proc2s` (processed-data files) -- confirmed by grepping the real dataset's own `acqus`/`acqu2s`, neither has it. `read_2d_processed` already reads it correctly (fixed earlier this session). So the *raw* `ser` path structurally can't use "acqus's XDIM" -- there's no such field. Investigated the actual slowness instead (profiled with `cProfile` on the real, previously-crashing 2H EXSY dataset, 1024x65536 points): **`_load` took 32 seconds**, and profiling traced ~25s of that to `compare_canvas.show_compare()` being called with **1024 spectra** -- `_populate_compare_list_from_pseudo2d` (an earlier round's "Compare auto-fills from a pseudo-2D's own slices" feature) unconditionally renders a full "Stacked" Compare view (one subplot per slice) as a side effect of every fresh pseudo-2D selection, even though nobody asked to see Compare -- fine for the typical 8-32-slice T2/PFG case this was built for, catastrophic for a real 1024-slice dataset. Fixed: `_refresh_compare_view` now defers the actual render (`_compare_view_stale` flag) when Compare isn't the visible canvas tab and nothing explicitly asked to switch to it; the deferred render fires the moment Compare is actually visited (`_render_compare_view_if_stale`, wired into `_on_canvas_tab_changed`). **`_load` dropped from 32s to 5.65s** on the same real dataset -- the Compare tab itself still takes ~25s to actually render 1024 subplots the one time it's visited (a harder, separate problem -- matplotlib fundamentally doesn't scale to 1000+ subplots -- not addressed this round).
  - Separately, spent real effort trying to hand-derive the correct raw-`ser` shape for this real dataset from `acqus`/`acqu2s`'s `TD` values plus Bruker's documented 1024-byte row-padding convention -- two independent derivations both left a small, unexplained remainder against the real file size (not a clean division). Genuinely unresolved; did **not** ship a guessed fix (same reasoning as the X-Pulse TPPI deferral above) -- the existing `read_2d_processed` fallback (already correct, already fast on its own at 0.65s) remains the answer for this class of dataset until/unless the raw-ser byte-accounting is actually worked out with certainty.
- ~~**"Output should be separate from the normal spectrum window -- it's freaking out when plot size is changed"**~~ -- root cause confirmed: `set_style`'s DPI/figure-size handling applied straight to the *live* embedded canvas's own `figure`, which fights Qt's own automatic canvas-widget resizing (matplotlib and Qt each think they own the figure's size). Fixed by fully decoupling: new `MplCanvas.set_export_settings(dpi=, fig_width=, fig_height=)` only *stores* these, never touches the live figure; `export_full_resolution` (the existing single choke point every save already goes through) now applies them temporarily around the actual save call and restores the live figure's own size/dpi immediately after, success or failure. The Output tab's export fields now say explicitly that they don't affect the live view.
- ~~**"Minor ticks should be an option for the output (and a default)"**~~ -- done. `apply_tick_style`'s existing `minor` parameter (previously always the hardcoded default across all 8 call sites) is now threaded through `self._minor_ticks_visible` (default `True`) the same way linewidth/tick-length already were; new Output-tab checkbox.
- ~~**"Phasing shouldn't constantly reset the view of the spectra like axis limits"**~~ -- real, reproducible bug, found and fixed. Starting to phase a pseudo-2D spectrum from its default contour view (the common case -- every fresh selection lands there) silently reset any ppm zoom the user had set: there's no existing `Line2D` yet for `update_data` to refresh in place, so `_show_pseudo2d_slice` fell back to a full `show_spectrum` rebuild, which resets the view by design. Confirmed *not* "constantly" in the literal every-tick sense (an already-active slice/line view already correctly preserved zoom across repeated drags, verified directly) -- it was specifically the one-time contour-to-line transition. Fixed by preserving the x (ppm) range across that specific rebuild -- ppm means the same thing on a contour and a slice view, unlike y (slice number vs. intensity), so there's no reason to lose it.
- ~~**"Left tab is too wide again, especially in Analysis"**~~ -- found and fixed: two `QLabel`s added to the VOCS sub-tab this session were missing `.setWordWrap(True)` -- an unwrapped `QLabel`'s `sizeHint()` reports its full single-line width regardless, which propagates up through the layout (a `QTabWidget` sizes to fit its *widest* page). Measured directly: VOCS's own page `sizeHint` was 748px wide before the fix, 316px after -- and since `QTabWidget` sizing isn't per-currently-shown-tab, this was dragging the whole Analysis panel wide on every sub-tab, not just VOCS, matching "especially in Analysis." Also found and fixed the exact same miss in the Output tab's "Per-type defaults" label (never assigned to a variable at all, so `setWordWrap` couldn't have been called on it). **This is the second time this exact mistake has been made in this codebase** (the Compare tab's own intro label, fixed in an earlier round) -- added a systemic regression test (`test_main_window_no_long_unwrapped_labels_in_left_tabs`) scanning every `QLabel` in the built window for long (>60 char) text without `setWordWrap(True)`, so a third occurrence fails loudly in CI instead of needing to be manually reported again.
- 619 tests passing (611 -> 619). Committed; not yet pushed.

## Plan execution: Phase 5 done -- Imaging analysis (2026-08-30, all 6 phases now complete)

New "Imaging" Analysis sub-tab (alongside Spectral Fitting / Curve Fitting / VOCS): a "Max gradient (G/cm)" field, user-entered rather than looked up (no calibration table exists anywhere in the acquired data itself, matching this project's existing G/cm convention from `io.bruker.read_diff_xml`). Extends the Spectrum canvas's existing right-click ppm/Hz/kHz axis-unit switcher with a 4th option, "mm".

- `MplCanvas.set_axis_unit("mm")`: `position_mm = hz_offset / (gamma_Hz_per_T * max_gradient_G_per_cm * 1e-5)`, spelled out as three explicit unit-conversion steps in the code (G/cm -> T/m -> Hz/m -> Hz/mm) rather than one folded constant, so it's checkable at a glance. **Worth flagging**: this project's own earlier plan text (written before this phase was implemented) stated the formula as `hz_offset / (gamma_Hz_per_T * max_gradient_G_per_cm * 0.01)` -- that version is dimensionally wrong by a factor of 1000 (it produces *metres*, not millimetres, despite the name). Caught by working through the unit conversion by hand before implementing, then double-checked against a concrete sanity number (50 G/cm for 1H: ~21.3 kHz/mm, a physically reasonable microimaging gradient bandwidth) -- not just transcribed from the earlier planning text. New `MplCanvas.set_imaging_gradient_params(gamma, max_gradient)` supplies the two numbers; the "mm" menu entry is disabled (with a tooltip explaining why) until both are set, rather than silently producing a wrong result from stale/missing values.
- New `MainWindow._build_imaging_subtab`: the gradient field, and a status line showing the live Hz/mm conversion factor for whichever spectrum is selected. Gamma is resolved via the already-existing `core.registry.find_gamma` from the selected spectrum's own recorded nucleus -- recomputed on every reselection (not just every gradient edit), so a stale gamma from a previously-selected different-nucleus spectrum can't leak into the mm axis. The gradient value itself is siloed per spectrum (`metadata.raw["imaging_max_gradient_g_per_cm"]`, restored on reselect), matching how sample mass is already siloed the same way.
- **Verified end-to-end on real X-Pulse MRI data** (`test_data/xpulse_MRI/14`): loads, resolves 1H's gamma correctly (42.577 MHz/T), and at a representative 50 G/cm produces a plausible ~±11.7 mm position range for a compact microimaging probe.
- **T2-by-integration across a pseudo-2D imaging series needed no new backend or new feature at all**, exactly as scoped: this session's own Integration Regions feature (region table + drag-select + `NMR2D_integrate_slices` -> `NMR2D_fit_relaxation`) already works generically along whatever the Spectrum canvas's current x-axis is -- switch to mm via this phase's new toggle, then use Integration Regions on the Spectral Fitting sub-tab exactly as already documented, no separate workflow. Time-resolved uptake/exchange tracking is explicitly out of scope, per the original instruction.
- 611 tests passing (604 -> 611). Committed; not yet pushed.

**All six phases of the 2026-08-30 implementation plan are now done.** Not yet pushed to `origin/main` at any point this session (16 commits ahead as of Phase 4; more since) -- offer/ask before pushing, per this project's own established practice.

## Plan execution: Phase 4 done -- Output tab (2026-08-30)

New "Output" side tab (Data/Processing/Analysis/Compare/**Output**), scoped exactly per this session's own earlier clarifying questions: a *live* style panel (changes apply immediately to every canvas, not a separate export-only preview), covering core styling + export settings + per-type defaults -- directly validated by Euan's own independent ask for "buttons/menus for controlling typeface, fontsize, tick size, direction, pad, plot width and height."

- **Core styling** -- linewidth, font size, tick length/width, grid on/off, legend on/off + position, Compare colour map. `MplCanvas`'s `_AXIS_LINEWIDTH`/`_AXIS_LABELSIZE`/`_TICK_LENGTH`/`_TICK_WIDTH` (previously class-level constants) are now copied onto instance attributes in `__init__` -- every `show_*` method already read them as `self._AXIS_LINEWIDTH` etc., so this alone makes them per-instance-overridable with no other call site needing to change. New `MplCanvas.set_style(...)` updates whichever style is given both as the new default (future redraws pick it up automatically) *and* applied directly to every axes already on screen (so a change is visible immediately, not just after the next reselect) -- covers multi-axes views (a "Stacked" Compare, a 2D contour's projection panels) too, not just `self.ax`. Style changes apply to *every* canvas instance uniformly (`MainWindow._all_canvases()`) for one consistent app-wide look -- except legend on/off + position, which deliberately targets `self.canvas` (Spectrum) specifically, reading/writing the exact same `_legend_visible`/`_legend_loc` state its own existing right-click "Show Legend" toggle already uses, per this session's own explicit scoping ("read/write the same state, not a competing duplicate").
- **A real bug found and fixed while building this**: `Axes.grid(visible, **kwargs)` is a matplotlib quirk -- passing any style kwarg (`alpha=0.3`) alongside `visible=False` still shows/warns about the grid, regardless of the boolean. Fixed by only passing the kwarg when actually enabling the grid.
- **Export settings** -- DPI and figure width/height (inches), applied straight to `self.figure.set_dpi()`/`set_size_inches()`. The existing "Save Figure As..." path (matplotlib's own native toolbar save dialog, unmodified) reads the figure's own current dpi/size automatically, so no other code needed to change for this to take effect.
- **Per-type defaults** -- scoped down from a full named-template CRUD system to something simpler but still real: one saved style per spectrum type (1D / 2D contour / Pseudo-2D stack / Compare), "Save as default" / "Load default" buttons, persisted to `~/.opennmr/figure_styles.json` (a new user-level config location, survives an app update/reinstall). Deliberately manual (the user picks which type to save/load), not automatic per-selection application -- auto-detecting "what type is currently showing" and re-applying on every reselect would have meaningfully expanded this phase's scope; flagged here as a reasonable follow-up if the manual version turns out to be too much friction in practice.
- `plotting.CONTOUR_CMAP_NAMES`/`VALID_SAVE_FORMATS` (previously `plotting.style`-only) now re-exported from the `plotting` package's own `__init__.py`, matching every other style helper already exposed there.
- Verified with a rendered before/after screenshot (dramatically bumped linewidth/fontsize/tick-length + grid on) -- visibly, correctly different, applied live with no reselect needed.
- 604 tests passing (590 -> 604). Committed; not yet pushed.

## Plan execution: Phase 3 (X-Pulse 2D research), deliberately not implemented yet (2026-08-30)

Researched X-Pulse's actual 2D acquisition scheme on the real `test_data/xpulse_HSQC_NUS/2` dataset, per the plan's own "research before implementing" instruction -- checked `~/Documents/GitHub/leonmr` and `~/Documents/GitHub/x-pulse-sequences` first, then read the real slice files' own headers directly (not guessed at). Two genuine findings, and a deliberate decision **not** to write `io.xpulse.read_2d` yet:

- **Despite the folder name, this specific acquisition is not NUS**: each of the 128 `raw_slices/NNNN_slice.jdx` files' own header records `Sampling= 100%` (the `.par` file's `SamplingList= 100%,75% NUS,50% NUS,25% NUS` lists the *sequence's* available options, not what this run used) -- confirmed independently by the slice filenames themselves being densely numbered 1-128 with no gaps, matching `NP1=128` exactly. Good news: no compressed-sensing/IST reconstruction (the "separate, larger undertaking" the plan flagged as a real risk) is needed for this dataset.
- **The real acquisition scheme is TPPI** (Time-Proportional Phase Incrementation), recorded explicitly and consistently across every slice checked: `##.ACQUISITION SCHEME= TPPI`, `##.ACQUISITION MODE= SIMULTANEOUS`. This is a well-known, textbook (not X-Pulse-specific) quadrature-detection scheme -- but the *correct processing* for it is genuinely non-trivial: TPPI's own defining property is that it converts the F1 quadrature problem into what's mathematically a **real** (not complex) time-domain signal, processed with a real Fourier transform along F1, not a plain complex FFT the way a straightforward "stack 128 complex rows and FFT along axis 0" approach would do. Confirmed this isn't a simple complex-FFT case by reading `nmrglue`'s own NMRPipe-compatible processing code (`pipe_proc.py`'s `ft()`), which explicitly branches to a real transform specifically for TPPI/Real/Sequential-mode indirect dimensions (`FD2DPHASE in (0, 1)` -> `real = True`) -- the exact flag-combination logic needed to reproduce this correctly is nontrivial (`real`/`alt`/`neg`/`bruk` interacting), and is NMRPipe's own internal convention, not something exposed as a simple reusable public function for arbitrary (non-NMRPipe-format) input like this project's own `processing.ft` module would need.
- **No existing reference implementation anywhere to build from or verify against**: confirmed by direct search, not assumption -- `leonmr.py` has zero HSQC/TPPI/2D-hypercomplex-assembly code (`NMR2D` there is Bruker-`2rr`-only, unrelated), and `x-pulse-sequences` has zero genuine 2D-correlation pulse sequences at all (only 1D imaging/diffusion/relaxation/selective) -- this user has never had working X-Pulse 2D processing code before this. There is also no independently-known-correct quinine HSQC spectrum available here to check a from-scratch implementation's output against.
- **Deliberate decision: not implementing `read_2d` this round.** Getting TPPI's real-transform processing subtly wrong would produce exactly the "plausible-looking but wrong" 2D spectrum this project has been burned by twice before on much simpler cases (`align_echo`'s target-index default, FID-shift/truncate ordering) -- both caught only because a validated reference or real symptom was available to check against; neither exists here. This matches the same reasoning already applied to the Varian ppm-axis convention and the imaging gradient-calibration constant: research and document honestly rather than ship an unverifiable guess.
- **Concrete next step, when picked back up**: implement `io.xpulse.read_2d` following NMRPipe's own documented real-transform convention for TPPI (`pipe_proc.py`'s `ft()`/`tp()` as the reference, same "use the validated library's own logic, don't hand-roll" reasoning this project already applies to Bruker's digital-filter removal) as a first pass, explicitly flagged unverified in its own docstring (matching the Varian ppm-axis precedent) -- then render it and get the user to visually confirm the correlations make chemical sense for quinine (a well-characterized molecule) before trusting it as a default.
- Bruker 2D verification (the other half of this phase) is done -- see the entry directly below.

## Plan execution: Phase 3 (Bruker 2D verification), Euan's feedback (2026-08-30)

Euan added real feedback directly to the repo (`git push` to `origin/main`, a plain `notes_from_Euan.txt`, merged in as commit `b56d871`) while this round's Bruker 2D verification was already underway -- two of his items turned out to be the exact same real bugs real-data testing had just found, so this section covers both together.

**Bruker 2D verification, using the three real datasets the user dropped into `test_data/` this round** (`1p3mm_7Li_Li_TNb2O5_1p3V_20251124`, `20200804_NaMgMnO_Pristine_111218C_Check_23Na_1p3mm`, `2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119` -- gitignored, not committed, real instrument data). Genuine 2D-correlation experiments among them (EXSY, HETCOR, BABA-DQ) let this be real verification, not just Bruker T2/PFG pseudo-2D re-confirmation:

- ~~**CRITICAL (Euan): "tuple index out of range" loading dense data, no other warnings**~~ -- **root-caused, real nmrglue bug, fixed with a fallback**: `read_2d`'s raw-`ser` path failed on every real ``DTYPA=2`` (float64) 2D dataset with a complex direct dimension -- confirmed systematic (22/22 such experiments across all three datasets failed identically, 0 other-cause failures) and traced into `nmrglue.bruker.guess_shape`'s own no-pulse-program fallback formula, which divides file size by a fixed 4-bytes-per-point assumption regardless of the real (8-byte, float64) point size. Not something `read_2d` can route around on its own (there's no raw shape to determine at all, unlike the existing unsupported-window case, which still has a real FID to stay on). Fix: `io.bruker._read_auto` now catches this specific failure and falls back to `read_2d_processed` (TopSpin's own already-processed `2rr`/`2ri`) when available, tagging `metadata.raw["read_2d_shape_guess_failed"]` so the GUI can tell the user (a new "Loaded from processed data" notice) and correctly hide FID-reprocessing controls (`_fid_reprocessable` now checks this flag) instead of showing dead ones. Verified end-to-end on the real EXSY dataset that first surfaced it (and confirmed it also covers the exact experiment folder Euan separately reported hitting this on, `.../2016`, same `exsy_echo3` pulse sequence class) -- loads cleanly, renders a real contour plot with plausible axis ranges after toggling "Process Indirect Dimension".
- ~~**Real `read_2d_processed` bug found during the same investigation**~~ -- F1's own block size (`SI`/`XDIM`, from `proc2s`) was silently discarded and replaced with F2's own `SI` (from `procs`) whenever `homonuclear=True` was passed, since `proc2s` was only read in the `else` branch. Confirmed on the real EXSY data: F1's real SI=1024 got replaced by F2's SI=65536, reshaping the `2rr` block into a shape many times too large for the file (`ValueError: cannot reshape array of size 67108864 into shape (65536,65536)`, the correct shape being `(1024, 65536)`). Fixed: `proc2s` is now always read from disk regardless of `homonuclear` -- that flag only controls where F1's *chemical-shift-axis* parameters (nucleus/BF1/O1/SF/SW) come from, never the block-size parameters, which are always real once a file has actually been 2D-processed. (The GUI itself never happened to pass `homonuclear=True`, so this specific bug wasn't yet user-visible through the app -- found via direct API verification, fixed regardless since it's a real bug in a documented public parameter combination.)
- ~~**CRITICAL (Euan): "Instead of guessing at initial phases... read processing parameters if available... users who have carefully processed or prepared their data should not have to go back and re-process"**~~ -- **confirmed and fixed**: `read_1d`'s `phase_mode` default was `"apc0"` (always re-guess via a zero-order grid search on every fresh load, silently discarding whatever `PHC0`/`PHC1` TopSpin had recorded) and `read_2d` had no `phase_mode` concept at all (always `apc0`, unconditionally). Both now default to a new-for-2D `phase_mode="recorded"`, reproducing F2's own recorded `PHC0` -- either a user's careful manual phasing, or TopSpin's own automatic processing, both real prior work. `"apc0"`/`"joint"` are still available for a genuinely raw/never-processed acquisition or a deliberate fresh guess. Doesn't affect the GUI's "Apply Processing" reprocess flow at all (it already always sends the live on-screen phase explicitly, bypassing `phase_mode` entirely) -- only changes what a *fresh* load shows, which is exactly the reported problem.
- **Genuine (non-pseudo) 2D contour/phase machinery**: confirmed already working via the EXSY render above (real peak, plausible F1/F2 ppm ranges once "Process Indirect Dimension" is toggled on) -- no further Phase-3 work needed here beyond the two fixes above.

**Euan's remaining feedback, not yet actioned -- queued for a future round**:
- Mass input should default to mg, not g -- **done this round** (`_on_set_mass_action`'s dialog now reads/writes mg, converting to/from `metadata.mass`'s own gram storage at that one boundary; the info-panel display updated to match). Everything downstream (`session.set_mass`, comparison normalization) is untouched -- still grams internally.
- A running log file of what processing was done, when, re-importable to "remember" a session -- new feature, not started. Worth scoping against `metadata.raw`'s existing per-spectrum history (window/zero-fill/phase/etc. are already all recorded there per load) -- may be mostly a matter of exporting/re-importing that, not a new tracking mechanism.
- "Process Indirect Dimension" button discoverability -- UX-only, not started (candidates: a more prominent label/color when a spectrum is 2D and currently pseudo-2D-by-default, or a first-load tooltip/nudge).
- Typeface/fontsize/tick size-direction-pad/plot width-height controls -- **this is exactly Phase 4 (Output/Figure tab)**, already planned below; Euan's ask directly validates that scope, nothing new to add.
- Peak-fit error display: a separate box, not inline text under the peak definitions -- small UI polish, not started.
- S/N filter for auto-pick peak *detection* (distinct from prominence) -- related to but not the same as this round's Phase-1 S/N-gated *fitting* (which fixes an already-defined peak's amplitude to zero once S/N drops during a curve fit, not a detection-time filter for `auto_pick_peaks` itself). Not started -- `auto_pick_peaks` (`scipy.find_peaks`) could take an additional S/N-based height threshold alongside prominence.
- On changing a processing parameter (autophase, SI, ...), the view resets; should be toggleable so users can keep their zoom -- related to, but broader than, the pivot/phase-drag view-preservation work already done in this project (phase preserves the view already; other processing actions currently intentionally reset it, matching `autoscale_y`'s established "a one-shot result is worth re-fitting the view to" reasoning from an earlier round) -- Euan's ask is for this to be a user-toggleable *option*, not a fixed choice either way. Not started.
- Peak fitting: allow user-supplied min/max values -- **already done this session** (Phase 1's "Set Custom Bounds..." peak-table feature, per-parameter, added before this feedback arrived).
- 2D fitting: allow skipping certain slices (e.g. pure noise, which will never converge) -- new feature, not started.
- T1/T2 fit results: let users add fitted values onto plots; the display needs to be clearer/more prominent -- new feature, not started.
- Zero-filling "does nothing -- resolution and point count stay the same" -- **investigated, mechanism confirmed correct, not yet reproduced as a real bug**: tested both the standard zero-fill-size combo and the Custom-size spin box end-to-end (through the actual GUI reprocess path, not just the isolated `processing.ft.zero_fill` function) on real Bruker pseudo-2D data (`bruker_T2`, `bruker_PFG`) -- both correctly changed the spectrum's point count (e.g. 32768 -> 262144). No bug found in the paths tried. Possible explanations not yet ruled out: this may have been observed on a dataset that was hitting the "tuple index out of range" crash above (before today's fix) or the `read_2d_processed` fallback (no raw-FID reprocessing available at all, correctly hidden -- but that means zero-fill controls wouldn't be visible in the first place, not that they'd be visible-but-inert), or may be about digital vs. true resolution not being visually obvious at typical zoom levels rather than a code bug. Needs a specific repro (which dataset, 1D vs. 2D, which zero-fill control) to pin down further.

590 tests passing (582 -> 590 across Phase 3 verification + Euan's two critical fixes + the mass-unit fix). Committed; not yet pushed. Euan's `notes_from_Euan.txt` commit merged in from `origin/main`.

## Plan execution: Phase 2 done (2026-08-30)

- ~~`fid_raw` caching for Varian/JEOL~~ -- done, same one-line pattern Bruker/X-Pulse already had (`fid_raw = fid` captured immediately after reading, before shift/align/LP/window/zero-fill, added to `metadata.raw`). Closes the last gap in the Processing tab's live in-memory preview -- `_compute_processing_preview` returned `None` (silently no live preview) for these two vendors before this; regression tests confirm it now returns a real preview for both.
- ~~Phase 1's spin-box range, unbounded~~ -- done, `_PHASE_SPIN_RANGE` changed from a fixed +/-36000 to `float("inf")`. Applies to both Phase 0 and Phase 1 (they share `_make_phase_control`).
- ~~JEOL's `FILTER_FACTOR` digital-filter correction~~ -- done and verified against real data, not just the synthetic test fixture. `FILTER_FACTOR` (JEOL's GRPDLY-equivalent group-delay point count) is now corrected via `nmrglue.bruker.rm_dig_filter` called directly (`grpdly=FILTER_FACTOR`, bypassing the Bruker-dict-specific wrapper and its decim/dspfvs table lookup, which JEOL doesn't need since the delay is already a known point count) -- reusing the same validated algorithm Bruker's own digital-filter removal uses, rather than a hand-rolled reimplementation. **Verified against all four real `test_data/jeol_1d/*.jdf` files**: every one carries `FILTER_FACTOR=16`, and correcting shifts each file's FID magnitude peak left by exactly 16 points, every time -- a clean, consistent, real-data confirmation, not just a synthetic-fixture check (which is also in the test suite, `test_read_1d_filter_factor_shortens_and_corrects_fid`).
- 582 tests passing (576 -> 582). Committed; not yet pushed.

## Plan execution: Phase 0 + Phase 1 done, four mid-run bug reports fixed (2026-08-30)

Started execution of the implementation plan below at the scheduled time. Phase 0 and Phase 1 are both done and committed; four unrelated bug reports arrived mid-run (as separate messages, not part of the plan) and were fixed in place rather than deferred, since each was small and self-contained.

- ~~**Phase 0** -- Open File(s)... dialog~~ -- done as planned, no backend changes needed. `_on_open_files_clicked`/`_open_files` (main_window.py), `QFileDialog.getOpenFileNames` with a JEOL/X-Pulse-spanning filter, routes each file through the same `detect_vendor`/`_load` path folder-scanning already used. Unrecognised files are reported (not silently skipped, unlike a folder drop -- picking files one-by-one is an explicit choice).
- ~~**Phase 1** -- backend-exposure sweep~~ -- done:
  - Compare-tab coloring: both `canvas.py` call sites (`stack_colors`, `plot_1d_stack`) switched from `"black"` to `"viridis"`.
  - **VOCS** -- new Analysis sub-tab (`_build_vocs_subtab`). Apply VOCS to the selected spectrum (adds a derived spectrum); VOCS Sum across the Compare tab's checked stack (corrects + interpolates onto a common axis + sums/averages, also added as a derived spectrum). Pulse param (default P1) and window-filter checkbox shared by both operations.
  - **Spectrum arithmetic** -- new Compare-tab "Combine" control (Sum / Average / Subtract from first), wraps `processing.arithmetic.combine_spectra`, added as a derived spectrum.
  - New `Session.add_derived(spectrum, label)`: appends an in-memory-computed spectrum with `vendor="derived"` (no path/reload kwargs) -- deliberately not a real vendor name, so `_fid_reprocessable`'s existing whitelist naturally hides FID-reprocessing controls for it rather than needing a new special case.
  - **Fixed/bounded peak parameters** -- peak table gained a 4th "Fixed" column (checkbox; locks shift/fwhm/A/eta all at once for that peak) and a right-click "Set Custom Bounds..." dialog (optional 'low, high' per shift/fwhm/amplitude/eta, stored as `Qt.ItemDataRole.UserRole` data on the row's shift cell). Found and fixed a real bug while wiring this up: `_update_peak_table_from_fit` (writes fitted values back after a Fit Spectrum run) replaced column 0's `QTableWidgetItem` outright, which would have silently dropped any custom bounds attached to it -- now preserves them across the replacement. `analysis_state` save/restore (per-spectrum peak-table persistence) extended for both the Fixed flag and custom bounds, with backward-compatible reading of old saved states that predate this (plain 3-column peak rows, no `peak_bounds` key).
  - **S/N-gated peak fitting** -- new "Noise region (ppm)" text field + "S/N threshold" spin box in the Spectral Fitting sub-tab, threaded into `NMR2D_fit_slices`'s existing `noise_region`/`sn_threshold` kwargs via `_fit_peaks`. Blank noise region (the default) disables gating entirely, matching the backend's own default.
- 576 tests passing (555 -> 576 across Phase 0 + Phase 1). Committed after each phase; not yet pushed.

**Four mid-run bug reports, fixed as they arrived (not part of the original plan):**

- ~~Spectra in the left tab should be numerically ordered (not 1, 11, 12, 2, 3...) after being sorted by folder~~ -- real bug, `_scan_for_spectra`'s folder walk used a plain lexicographic `sorted(..., key=lambda e: e.name)`. New `MainWindow._natural_sort_key` (splits into digit/text chunks, digit chunks compared as ints) fixes ordering for Bruker's own numbered-experiment-folder convention (and any other numbered layout) when opening a parent folder.
- ~~2D phasing needs extra options or slice-based phasing, including windowing in both dimensions~~ -- investigated, not yet implemented (folded into Phase 3's scope below, since it's the same 2D-processing surface Phase 3 already touches). Two confirmed gaps: `bruker.read_2d`/`varian.read_2d` apply one shared window/LB across *both* F2 and F1 (no independent F1 window control); true (non-pseudo) 2D has no per-F1-row phase correction (pseudo-2D already has this via the per-slice Autophase button and per-slice phase editing).
- ~~The processing preview and spectrum windows need to stay updated and linked (or at least updated when the tab is changed)~~ -- investigated via direct empirical testing (a throwaway script driving a real `MainWindow`, not just reading the code): switching left-side tabs away from/back to Processing, reselecting a different spectrum while Processing Preview is showing, and switching the right-side canvas tab directly all correctly refresh/stay in sync already, matching the extensive existing machinery for exactly this (`_on_side_tab_changed`, `_on_canvas_tab_changed`, `_draw_selected_spectrum`'s "kept up to date regardless of which canvas tab is showing" comments, `_apply_phase_preview`'s dual-canvas updates). No bug found in the scenarios tried -- if this is still visibly wrong in real use, it needs a concrete repro (which control, which tab sequence) rather than a guess at a fix.
- ~~Phasing shouldn't reset when the pivot is changed~~ -- **real bug, root-caused and fixed**: `apply_phase` computes `phase0 + phase1*(frac(x) - pivot_frac)`, so moving the pivot while Phase 0/Phase 1's *numbers* stayed fixed silently changed the applied correction at every point by a constant `phase1*(old_pivot_frac - new_pivot_frac)` -- a visible jump across the whole spectrum, easily read as "phasing reset" even though neither spin box moved. New `_compensate_phase0_for_pivot_change`, called from both `_on_pivot_spin_changed` and `_on_pivot_slider_changed` (the slider path bypasses the spin box's own change handler via blocked signals, so needed the same fix separately -- caught by testing both paths, not just one), folds that same offset into Phase 0 so the displayed correction stays exactly continuous; only where *future* Phase 1 drags rotate around actually changes. Two pre-existing tests had encoded the old (buggy) behaviour as expected and were updated to match the corrected semantics.

## Implementation plan: backend-exposure sweep, X-Pulse 2D, Output/Figure tab, Imaging analysis (2026-08-30)

Planned ahead of execution, per explicit request ("prepare the plan... start to implement these things in a sensible order" at a named time). Two scoping questions asked and answered before finalizing (Output tab: live style panel, not a separate export-only preview; first version covers core styling + export settings + per-type defaults). Pulse-sequence planning/Bloch simulation explicitly out of scope for now (flagged as later interest). Ordered lowest-risk/most-scoped first.

**Phase 0 -- Open dialog: files, not just folders** (mid-planning addition: "sometimes the data isn't a folder (like jeol), so those files should be clickable for import, not just the folder. Same for the other vendors"). `_on_open_clicked` only ever offers `QFileDialog.getExistingDirectory` -- fine for Bruker/Varian's own multi-file experiment-folder layout, but JEOL (`.jdf`) and X-Pulse (`.jdx`/`.dx`) are single files, and the existing recursive folder-scan (`_scan_for_spectra`) already discovers and loads them as individual file paths once it finds them -- meaning `io.load()` on a bare file path almost certainly already works with **no backend changes needed**. Add a second "Open File(s)..." action (`QFileDialog.getOpenFileNames`, a vendor-spanning filter) alongside the existing folder picker, routing each picked file through the same `session.add`/vendor-detect path the folder scan already uses.

**Phase 1 -- Backend already built, GUI doesn't expose it yet** (lowest risk: no new DSP, matches this project's own recurring "check the backend before assuming a build" pattern):
- Compare-tab default coloring: `show_compare`'s two `plotting.stack_colors(len(spectra), "black")`/`plot_1d_stack(..., color=<default "black">)` call sites (`canvas.py`, stack and overlay branches) currently render every trace solid black. Switch to a continuous, non-washed-out-midpoint colormap (`"viridis"`, matching the same reasoning that already ruled out a diverging colormap for `show_slice_fit`'s shading this session) -- a two-line fix.
- **VOCS** (`processing/vocs.py`: `apply_vocs`, `vocs_sum`, `pulse_to_w1` -- fully built, tested, zero GUI). A natural fit as the Analysis tab's third sub-tab (Spectral Fitting / Curve Fitting / **VOCS**), matching the room already left for exactly this. Needs: a pulse-selection control (`apply_vocs`'s `pulse` param reads a recorded pulse length off the spectrum metadata) and a window-filter toggle for `apply_vocs`; `vocs_sum` combining several spectra (likely fed from a Compare-list-style multi-select).
- **Spectrum arithmetic** (`processing/arithmetic.py::combine_spectra` -- add/subtract/combine loaded spectra, fully tested, zero GUI). A small utility, probably reachable from the Compare tab (which already has a checkable multi-select list) -- pick an op (add/subtract/average/...), apply across the checked spectra, add the result back into the session as a new spectrum.
- **Fixed/bounded peak parameters** -- `peak_defs`'s `fixed`/`bounds` keys already work in `fitting.fit_spectrum`/`NMR2D_fit_slices`; the peak table has no way to set either (only the auto position-tolerance bound exists). Likely a "Fixed" checkbox column plus a right-click "Set custom bounds..." on a peak-table row.
- **S/N-gated peak fitting** (`NMR2D_fit_slices`'s `noise_region`/`sn_threshold` -- auto-fixes a peak's amplitude to zero once it decays into noise across a series, backend-only). A noise-region field + threshold spinbox near the peak table.

**Phase 2 -- Small, already-scoped mechanical fixes**:
- `fid_raw` caching for Varian/JEOL's `read_1d` (two-line addition, same pattern Bruker/X-Pulse already got) -- closes the last gap in the Processing tab's live in-memory preview.
- Phase 1's spin-box range is arbitrarily capped at +/-36000 degrees -- make it genuinely unbounded.
- JEOL's `FILTER_FACTOR` (digital-filter group-delay count, JEOL's GRPDLY equivalent) isn't corrected for yet -- the one real, already-documented gap from the JEOL importer's own verification (one real peak comes out phase-flipped under default autophasing).

**Phase 3 -- 2D plotting for X-Pulse, and Bruker 2D verification**. Bruker's `read_2d` + contour/phase machinery already exists and already works for genuine (non-pseudo) 2D -- this is verification against the user's incoming real Bruker 2D dataset, not new code, unless that verification surfaces a bug. X-Pulse has no true-2D reader at all (`read_1d`/`read_array` only, both 1D/pseudo-2D) -- **research before implementing**, matching this project's own established discipline (the Varian/Bruker 2D ports both needed real care around hypercomplex assembly): check `~/Documents/GitHub/leonmr`/`~/Documents/GitHub/x-pulse-sequences` and the real `test_data/xpulse_HSQC_NUS` file's own headers for X-Pulse's actual 2D acquisition scheme before writing `io.xpulse.read_2d` -- flag explicitly if that dataset turns out to need genuine NUS (non-uniform sampling) reconstruction, which is a separate, larger undertaking (IST/CS-style, not a plain FFT) from an ordinary States/TPPI/echo-antiecho 2D.

**Mid-run addition (2026-08-30, during Phase 0): "2D phasing needs extra options or slice-based phasing, including about windowing in both dimensions."** Folded into this phase since it's the same 2D-processing surface. Two confirmed gaps, found by reading the code (not guessed): (1) `bruker.read_2d`/`varian.read_2d` apply exactly one `window`/`window_kwargs` pair across *both* F2 and F1 -- there's no way to give the indirect dimension its own window function/LB, which real 2D processing routinely wants (F1 usually has far fewer points and often needs a different window than F2). (2) The GUI's `dim_combo` (Direct F2/Indirect F1) already lets phase0/phase1 be set *per dimension as a whole*, and pseudo-2D already has `autophase_slices_button` for independent per-slice phase -- but genuine (non-pseudo) 2D has no way to vary phase *by F1 row* (a single F1-wide phase0/phase1, not a per-row correction), which is the "slice-based phasing" the user is asking for extended to true 2D. Scope: add an F1-specific window-function control (own combo + LB/params, mirroring the existing F2 `window_combo`) threaded through `read_2d`'s indirect-dimension apodization call in both `bruker.py`/`varian.py`; investigate whether per-row F1 phase correction is actually needed for this user's real 2D data before building it (their workflows are overwhelmingly pseudo-2D, where per-slice phase already exists -- confirm with real data whether true-2D per-row phase is a genuine gap or already covered by the pseudo-2D path being the default).

**Phase 4 -- Output/Figure tab**. New side tab (Data/Processing/Analysis/Compare/**Output**), scoped via the two clarifying questions above: a *live* style panel (applies to whichever canvas tab is currently showing, not a separate export-only render path), covering core styling + export settings + per-type defaults in the first version.
- New `FigureStyle` dataclass (line width, color/colormap choice, font size, legend on/off + position, grid on/off, spine style, DPI, figure width/height, export format) + a small JSON-backed template store (named templates, one default per spectrum "type": 1D / 2D contour / pseudo-2D stack / Compare), auto-applied based on what's currently showing unless overridden.
- `MplCanvas`'s existing `_AXIS_LINEWIDTH`/`_AXIS_LABELSIZE`/`_TICK_LENGTH`/`_TICK_WIDTH` are class-level constants used by every `show_*` method already -- promote to instance attributes so the Output tab can actually change them, matching how `_legend_visible` (already a per-instance, right-click-togglable attribute) works today. Legend/grid controls in the new tab should read/write the *same* underlying state the existing right-click menu already toggles, not a competing duplicate.
- Color/colormap choice isn't currently parameterized at all (each `show_*` hardcodes its own); needs threading through as a new instance attribute the relevant plot calls read.
- "Save Figure As..." (`_on_save_figure_clicked`, currently overridden to always export full-resolution regardless of the live view) needs to read the chosen DPI/figure-size/format from the active style.

**Phase 5 -- Imaging analysis**. New "Imaging" Analysis sub-tab (alongside Spectral Fitting / Curve Fitting / VOCS): a "Max gradient (G/cm)" field (matching this project's existing G/cm convention from `read_diff_xml`), user-entered rather than looked up from a per-probe/axis calibration table -- deliberately the simpler, lower-risk direction already flagged as "more promising" earlier in this backlog, since no calibration table is recorded in the acquired data at all. Extends the canvas's existing ppm/Hz/kHz right-click axis-unit switcher with a 4th unit, "mm" (`position_mm = hz_offset / (gamma_Hz_per_T * max_gradient_G_per_cm * 0.01)`, gamma from the already-existing `core.registry.find_gamma`). Explicitly deferred, per direct instruction: time-resolved uptake/exchange tracking (`diffprof_exchange`-style, timestamped multi-row monitoring) -- **T2-by-integration across a pseudo-2D imaging series is the one enabled workflow**, and needs no new backend at all: this session's own Integration Regions feature (region table + drag-select + `NMR2D_integrate_slices` -> `NMR2D_fit_relaxation`) already does exactly this generically, in ppm -- once the mm axis exists, the same regions/integration/curve-fit pathway works unchanged with regions expressed in mm instead.

Execution starts 2026-08-30 14:11 local time (a `CronCreate` one-shot wakeup is scheduled for this, in case the conversation goes quiet before then). Committing after each phase, matching this session's established per-round cadence, rather than one giant commit at the end.

## Compare highlight fix, quick-phase buttons, processing-reset bug, integration regions (2026-08-30)

Two follow-ups from live testing plus a large new-feature message, all addressed in one round.

- ~~Compare-tab selection highlight still not showing~~ -- **the real bug, found this time**: the previous round's fix did get compare_list a "current" row, but nothing ever tinted the matching row in the plot -- and separately, an *earlier* round's "let peaks bleed into neighbouring rows" fix (`row_ax.patch.set_alpha(0.0)`) silently made any facecolor tint on that same patch invisible regardless, once one *was* wired up. `compare_list`'s current-row selection now drives a new `MplCanvas.set_compare_selection`, which makes the selected row's patch opaque just long enough to show its own tint (every other row stays transparent for the bleed). Also hardened the list's own native highlight against looking too faint without keyboard focus (forced the inactive/active palette highlight colors to match). Verified with a rendered screenshot -- the selected row is now unmistakably tinted yellow in both the list and the plot.
- ~~+/-90/+/-180 quick-phase buttons~~ -- done, a "Quick phase" row in the Phase box, later re-requested stacked vertically rather than side by side (too cramped in the narrow left panel) -- fixed.
- ~~Processing looks reset when switching slices or tabs; apply to all slices should default on; absolute phase should be noted in the internal data structure; save live processing into the spectrum data structure and use it in Processing~~ -- **real bug found**: a live phase drag already wrote the correction straight into `spectrum.data` (that part always worked) -- but nothing ever updated `metadata.raw["phase0"/"phase1"]` to match. The Processing tab's own live preview recomputes from the raw FID + `metadata.raw`, not from `spectrum.data` directly, so it kept showing the *un*-phased result the moment you switched slice and came back, even though the actual spectrum was never actually reset. `_capture_phase_reference` (the one place every reselection/slice-change funnels through) now commits the pending delta into `metadata.raw` right before re-anchoring, via a new `_commit_pending_phase_to_metadata` -- skipped for the two cases `metadata.raw` can't represent (a per-slice-only pseudo-2D edit, a conventional 2D's F1 phase), both already handled by the existing `_reapply_phase_post_reload` post-hoc mechanism instead. "Apply to every slice" now defaults to checked, since a shared correction across the whole series is the common case and it being unchecked made every slice switch look like a reset. Verified end-to-end: drag phase, switch slices, confirm `metadata.raw["phase0"]` now matches, confirm `_compute_processing_preview`'s own recomputation matches `spectrum.data` exactly.
- ~~Linear prediction should be able to back-predict points~~ -- the backend (`processing.ft.linear_predict(direction="backward")`) already fully worked (verified: recovers a clean synthetic decay's earlier points to machine precision) -- no GUI control ever exposed it. Added an "LP direction" combo (Forward/Backward), threaded through the live preview and Apply Processing, persisted per-spectrum like every other FID-processing control.
- ~~In Processing, the FID y-axis shouldn't say "FID"~~ -- done, now "FID Intensity".
- ~~TDeff: rename, and make it easier to go down from the max rather than up from 1~~ -- renamed to "Truncate to:"; the spin box now defaults to (and lets you step down from) the spectrum's own full raw-FID length instead of counting up from 1 to reach a useful value -- a real acquisition-recorded TDEFF, or a previously-saved choice, still takes priority when present.
- ~~Add integration regions as the most basic way to analyse PFG/T1/T2 data, also usable to bound/support peak fitting, selected by dragging on the spectrum after Cmd+I~~ -- done, a genuinely large addition; full detail in the "Add integration regions" commit and `_build_region_box`'s own docstring. Backend (`processing.relaxation.NMR2D_integrate_slices`) already existed and was already designed to be interchangeable with `NMR2D_fit_slices`'s own output -- this was GUI wiring, not a missing capability. New region table (Low/High ppm) + Add/Remove/Clear, a Cmd/Ctrl+I drag-select mode on the Spectrum canvas (`MplCanvas.set_region_select_mode`/`regionSelected`, with a live rubber-band shown mid-drag), persistent region shading (`MplCanvas.set_regions`, self-clearing by gid so a `figure.clear()` redraw can't leave stale patches), an "Integrate Regions" button that stores its result in the *same* `_slice_fit_results` slot Fit Spectrum uses (so Fit Curve needs no separate code path at all), and -- when any regions are defined -- the same regions bound the peak fit itself (`fit_regions`). Guarded every place that shared slot gets read back for display (`_restore_analysis_state`, the reselection-restore path) against an integration result's missing `popt`/`pcov`/`peak_defs`/`y_fit` reaching the peak-fit-only Spectral Fit viewer and crashing it -- caught by a dedicated regression test, not just inspection. Verified end-to-end: define a region around a known-T2 synthetic peak, Integrate Regions, Fit Curve, recovered T2 within 5% with no peak ever fit.

551 tests passing (539 -> 551, the middle number briefly hit 550 before an accidental test-body splice from a large edit was caught and fixed). Committed; not yet pushed.

## Analysis sub-tabs, dmfit-style fit shading, nav-bar menus, custom equations, context shortcuts, baseline rework (2026-08-30)

Seven items from one dense feedback message, all addressed in one round.

- ~~Sub tabs in Analysis: one for spectral fitting, one for curve fitting, room for more later (e.g. VOCS)~~ -- done. `_build_analysis_tab` now holds a `QTabWidget` (`analysis_sub_tabs`) with "Spectral Fitting" (peak table, Fit Spectrum, slice-fit browsing) and "Curve Fitting" (Fit Curve settings/button, curve-fit browsing) as its two pages, split out of what used to be one long stacked panel. `analysis_status_label` stays above both, unsplit, since it's a single selection-state message covering both steps together.
- ~~Spectral-fit visualisation: one peak fit can look disproportionately prominent; total fit should be dashed, components shaded like the user's own dmfit code~~ -- done. Read the reference (`leonmr.py`'s `dmfit`, ~line 470-525: raw data solid black at zorder 0, each component a `fill_between` region colored off a colormap and drawn tallest-height-first, total model a dashed line on top at zorder 100). `MplCanvas.show_slice_fit` now matches that shape: each peak is `ax.fill_between(..., alpha=0.4)` instead of a solid line, drawn in descending peak-height order so a small peak's shading isn't buried under a larger neighbour's, colored from `matplotlib.colormaps["tab10"]` (a qualitative palette -- tried dmfit's own "diverging"/BrBG first, but a diverging colormap's washed-out white midpoint made a middle-ranked peak nearly invisible once shaded, so switched to something that stays distinct regardless of peak count). Verified with a rendered before/after (one dominant peak + two much smaller ones) -- all three now clearly visible and proportional to their actual amplitude.
- ~~Nav bar should have Analysis/Fitting fields for closer control~~ -- done, scoped via a clarifying question rather than guessed: quick-access menus only, no new capability. New "Analysis" (Fit Spectrum, Auto-Pick Peaks, Pick from Plot toggle, Clear Peak Table) and "Fitting" (Fit Curve, Fit Type submenu, Model submenu) top-menu-bar entries -- every action delegates straight to the same handler/combo the Analysis tab's own widgets already use, so there's no separate state to keep in sync; the Fit Type/Model submenus are plain radio-style `QActionGroup`s that just set the corresponding combo's index, with checked-state refreshed on `aboutToShow` since the model list itself changes with fit type and with which spectrum is selected.
- ~~Peak pick from plot doesn't work~~ -- **real bug found**: pick mode only ever listened on `self.canvas` (the Spectrum tab), so clicking anywhere else reachable from the Analysis tab (Compare, Fit, Spectral Fit, Processing Preview) silently did nothing. `_on_pick_peak_toggled` now switches `canvas_tabs` to Spectrum when the toggle is turned on.
- ~~Compare tab: not obvious which spectrum is selected~~ -- **real bug found**: `_rebuild_compare_list` never set a "current" row after auto-populating from a multi-select, so nothing was ever highlighted. Now always establishes one, preferring to preserve the previously-current entry if it's still present after the rebuild.
- ~~Baseline correction isn't achieving much; make sure it applies across a whole pseudo-2D (hold off on conventional 2D for now); should also be usable for spectral fitting~~ -- done. Root cause: `processing.ft.baseline_correct` was a single unweighted `np.polyfit` over the *entire* spectrum, including the real peaks in it -- for anything but a dead-flat spectrum, the peaks pull the fit toward them and it barely corrects anything. Rewritten to call `fitting.estimate_baseline_robust` (the same sigma-clipped MAD estimator `fit_spectrum` already uses to hold a baseline fixed under a peak fit) instead, so the two now share one baseline algorithm -- literally satisfies "usable for spectral fitting" rather than needing a second implementation. Per-slice application across a whole pseudo-2D series *already* existed (`_on_baseline_correct_clicked`'s `apply_all_slices_checkbox` branch loops every row) -- the real gap was discoverability: the checkbox was labelled "Apply this phase to every slice" (phase-only wording) despite already gating Baseline Correct too, so renamed to "Apply to every slice" with a tooltip covering both. Conventional (non-pseudo) 2D deliberately untouched beyond the same underlying algorithm swap, per explicit instruction to hold off on redesigning that case. New tests in `tests/test_ft.py` compare directly against the old naive-polyfit behaviour on a synthetic sloped-baseline-plus-tall-peak spectrum (robust residual < 10% of the naive one, away from the peak).
- ~~Backspace-to-remove-spectrum is too easy to trigger by accident from another tab; should be context-dependent (e.g. remove a peak instead, if on the Analysis tab); rename the Spectra tab~~ -- done, both parts. `remove_action`'s keyboard accelerator is now `Qt.ShortcutContext.WidgetShortcut` scoped to `spectrum_list` specifically (`spectrum_list.addAction(remove_action)`) instead of firing window-wide -- the Edit-menu item itself is unaffected, still triggers from anywhere via a click. `peak_table` gets its own analogous Backspace-scoped `remove_peak_action`. Renamed the "Spectra" tab to "Data" throughout (tab label, `Window` menu's panel switcher, in-code comments/docstrings referencing it by name).

528 tests passing (510 -> 528, including this round's new coverage). Committed; not yet pushed.

## Bruker PFG load-speed regression, phase-drag zoom preservation, FID/Compare zoom behaviour (2026-08-30)

Four items from live testing, all diagnosed and fixed in one round.

- ~~Loading large Bruker PFG pseudo-2D data is slower than it was at
  first~~ -- done, a real, measured regression from the `prepare_fid`
  sharing work earlier tonight, not a vague slowdown. `read_2d`'s F2
  processing loop used to skip the whole shift/echo-align/linear-
  predict step per row entirely (three separate `if`s) whenever none
  of them were requested -- the common case for a fresh load. Folding
  the three into one shared `processing.ft.prepare_fid` call
  accidentally made the `np.apply_along_axis` wrapping it
  unconditional -- confirmed via a direct benchmark before fixing
  (~1200x slower for a synthetic 32x16384 case just from the per-row
  Python-level call overhead of a function that ends up doing nothing
  for every row). Restored the original one-line fast-path guard
  around the call; new regression test mocks `prepare_fid` to fail
  loudly if it's ever called when `fid_shift`/`align_echo`/
  `linear_predict_points` are all off, rather than only checking the
  result still looks right (which a slow-but-correct regression would
  have passed).
- ~~Keep the axis positioning constant while phasing on the
  Processing-preview tab, and show the pivot line there too~~ -- done.
  New `MplCanvas.update_processing_preview_spectrum` updates the
  preview's spectrum panel *in place* (same idea as the existing
  `update_data` for the plain Spectrum tab) instead of the full
  rebuild `show_processing_preview` does -- preserves whatever zoom/
  pan is already there. Wired into `_apply_phase_preview`'s "1d"/
  "pseudo2d" branches specifically (a new `_refresh_processing_preview_
  phase_only`, distinct from the existing full-rebuild `_refresh_
  processing_preview_if_active`) -- phase never touches the FID panel,
  so only the spectrum panel needs updating, and there's finally a
  real "current view" worth preserving through a drag the way there
  already was for the Spectrum tab. Discrete "just fix it" actions
  (Autophase, Revert Autophase, Baseline Correct) deliberately still
  use the full rebuild -- matches `autoscale_y`'s own established
  reasoning for those same three actions on the Spectrum tab (a
  one-shot result worth re-fitting the view to, not preserving the
  old one through). New `_pivot_marker_ax` generalizes the pivot-line
  mechanism (previously gated on `_current_1d_spectrum`, which is
  deliberately `None` for the Processing-preview view since it's not
  offering unit-switching there) to work on any axes a view opts into,
  not just the plain Spectrum tab's own. Verified with a rendered
  before/after screenshot of an actual zoomed-in small peak being
  phased -- the zoom held exactly, and the lineshape changed
  correctly.
- ~~The FID panel's scroll-zoom shouldn't have the pinned-baseline
  behaviour -- freeform, or centred on y=0~~ -- done. New
  `_freeform_zoom_axes` set (parallel to `_pivot_marker_ax`, same
  reset-then-opt-in pattern): `show_fid`'s own axes and
  `show_processing_preview`'s FID panel specifically (not its spectrum
  panel) now get a plain cursor-centred y-zoom -- both ends of the
  view move, matching how x-axis zoom already works -- instead of the
  pin-the-bottom zoom that makes sense for a phased spectrum (rising
  from a real baseline) but not an FID (oscillating around y=0, no
  baseline to pin).
- ~~Compare tab: let overlapping/zoomed spectra bleed into
  neighbouring rows instead of being clipped~~ -- done,
  `Line2D.set_clip_on(False)` on each row's plotted line in `show_
  compare`'s "Stacked" mode. Needed a second part too, found by
  actually rendering the result rather than assuming clip_on alone
  was enough: each row's own axes background is opaque by default, so
  a peak bleeding into a row drawn *after* it (rows render top-to-
  bottom) would just get painted over by that row's own white
  background -- `row_ax.patch.set_alpha(0.0)` makes every row
  transparent so a bleed stays visible regardless of which direction
  it goes or which row was drawn first. Verified with a rendered
  screenshot showing a deliberately zoomed-in peak correctly
  overlapping both neighbouring rows.

510 tests passing (504 -> 510). Committed and pushed.

## Compare-tab landing fix, and the real per-slice-preview performance fix (2026-08-30)

Two more items from live testing, addressed in the same round since the
second directly reopened -- and this time actually resolved -- the
per-slice-loading performance question flagged as deliberately
unattempted a few rounds back.

- ~~When a pseudo-2D loads, Compare is left showing instead of
  Spectrum~~ -- done. `_refresh_compare_view` gained a `switch_tab`
  parameter (default `True`, unchanged for every other call site --
  a manual multi-select, the Compare tab's own "Plot" button/context-
  menu action, Move Up/Down -- all genuinely deliberate "show me
  Compare" actions); the pseudo-2D auto-populate path in
  `_on_list_selection_changed` now passes `switch_tab=False`, and
  `_on_set_mass_action`'s own background refresh does too. Compare's
  *list* still auto-populates and its canvas still gets fully drawn in
  the background either way -- only which tab ends up visible changed.
  **A real regression surfaced while fixing this, caught by a test
  failing, not by inspection**: the guard that stops
  `_silo_other_canvas_tabs` from clearing a correctly-populated
  Compare view required `canvas_tabs.currentWidget() is
  self.compare_canvas` -- which, by design, is now never true for this
  exact path (the whole point of the fix is that the *canvas* stays on
  Spectrum). That guard was accidentally checking two different
  things at once (whether the Compare canvas's own *content* is
  correct, vs. whether the Compare *tab* is the active one) that only
  happened to coincide before this fix -- split into
  `compare_matches_list` (content-only, drives whether
  `_silo_other_canvas_tabs` should leave Compare alone) and
  `compare_tab_active_and_matches` (content AND active tab, drives
  whether `_draw_selected_spectrum` should skip its own Spectrum
  redraw/switch) once they actually diverged.
- ~~Phasing doesn't update the Processing-preview tab (still works on
  Spectrum)~~ -- done, and it directly reopened (with explicit,
  concrete direction this time -- "you don't need to cache the live
  fid, just the raw one, then load only the active phasing slice")
  the per-slice-loading performance item flagged as investigated-but-
  deliberately-not-attempted a few rounds ago. That's exactly what got
  built:
  - New `processing.ft.prepare_fid` -- the shift/echo-align/truncate/
    linear-predict step sequence (order matters a lot here, see its
    own docstring for the echo-wrap-artifact story this project has
    hit twice before), extracted from `io.xpulse.read_1d` into one
    shared function, now also used by `io.bruker.read_1d`/`read_2d`.
    Not a new implementation -- literally the same one, called from
    three places instead of duplicated across them, so there's no way
    for the GUI's own fast preview (below) to silently drift from what
    a real vendor reload actually does.
  - Every vendor reader that supports these steps now also stashes the
    *truly* raw FID (post digital-filter-removal only, before even
    shift/align) into `metadata.raw["fid_raw"]`, alongside the
    existing fully-processed `"fid"` -- cheap, since it's already
    sitting in memory as part of any real load regardless.
  - New `MainWindow._compute_processing_preview`: recomputes a preview
    spectrum entirely from `fid_raw` (one row, for a pseudo-2D --
    "only load the active phasing slice") via `prepare_fid` + apodize/
    zero_fill/fft_spectrum/apply_phase, all in memory -- no disk I/O,
    no reprocessing of any slice but the one being viewed. The ppm
    axis's own range is reused from the already-loaded spectrum
    (acquisition-fixed, doesn't depend on window/zero-fill choice) --
    deliberately not re-derived from O1/SF/SW a second time, avoiding
    a second implementation of that vendor-specific math that could
    quietly drift from the real one.
  - Because this is now genuinely cheap, phase-dragging (handle, spin
    box, or the pivot slider/spin) refreshes the Processing-preview on
    every tick, not just once a gesture finishes -- the
    `DragHandle.dragFinished`/`QSlider.sliderReleased` throttling from
    an earlier round is now dead code and was removed rather than left
    in place unused, along with its own now-obsolete test (replaced
    with one asserting live tick-level refresh, plus a new one
    numerically comparing the preview against the Spectrum tab's own
    phase to confirm they actually agree, not just that the preview
    changes at all).
  - New regression test asserts `io.load` is never called at all
    during a live preview refresh (mocked to fail loudly if it is) --
    checks the actual performance claim, not just that the result
    looks plausible.
  - **Scope, stated plainly**: Bruker and X-Pulse are covered (the
    two vendors with a multi-slice/pseudo-2D concept and the ones this
    round's `fid_raw` caching targeted). Varian and JEOL don't have
    `fid_raw` cached yet, so their Processing-preview tab will show
    correctly empty (not crash, not a stale/wrong preview) until that
    caching is added for them too -- a small, mechanical follow-up
    (the same two-line addition made to bruker.py/xpulse.py), not
    flagged as risky the way the original per-slice question was.

504 tests passing (501 -> 504). Committed and pushed.

## Autonomous overnight round 4: left-panel reflow -- measured, one real fix shipped, the rest connects to an already-open item (2026-08-30)

- ~~Left side panel needs better reflow at smaller window widths --
  "the left side tab is needing scrolling too often"~~ -- **measured
  precisely rather than guessed at**, one concrete fix shipped, and
  the remaining scope turns out to already be the SAME open question
  as the Analysis-tab sub-tabs item below -- not two separate things.
  Rendered each side-tab panel's actual content `sizeHint()` at the
  window's own current/default size (1100x680): Spectra 425px,
  Processing 469px, Compare 432px -- all comfortably under 680, no
  scrolling needed for any of them at a normal window size. Analysis
  alone needed **812px** -- confirmed visually too (rendered to a PNG,
  not just the number): at a perfectly ordinary 1100x680 window, the
  Analysis tab is already cut off mid-way through "Fit type:
  Relaxation (T1/T2)", well before reaching the Fit Curve button or
  anything below it. So this complaint is real, precisely located (the
  Analysis tab specifically, not a general problem across every tab),
  and reproducible even without a small window.
  - **One real, verified, unambiguous fix shipped**: `peak_table` (the
    pseudo-Voigt peak-list table) had no height cap at all, so Qt's
    own default `QTableWidget` size hint reserved generous space
    regardless of actual row count -- capped at 160px (comfortably ~4
    rows, scrolls internally for more), recovering ~32px (812 -> 780).
  - **Also checked and ruled out a false lead**: the *Compare* tab's
    own content `sizeHint()` reported 572px wide (well past the
    panel's actual ~280-340px width) -- looked like a second real bug
    at first (horizontal scrolling is deliberately disabled in
    `_scrollable`, so a too-wide `sizeHint` would mean invisible,
    silently clipped content). Rendered it to a PNG to check directly
    rather than trusting the number: it's a known Qt quirk where a
    word-wrapped `QLabel`'s `sizeHint()` over-reports its own "ideal"
    unconstrained width even though `heightForWidth` makes it actually
    reflow correctly once really constrained (which `QScrollArea`'s
    `setWidgetResizable(True)` does) -- the tab renders with no
    clipping at all. **Worth remembering**: a widget's raw
    `sizeHint()` number in this codebase is not reliable evidence of a
    real visual bug on its own, especially for word-wrapped labels --
    render it and look, the way this session's own established
    practice already does for canvas/plotting changes, not just for
    those.
  - **The remaining ~100px gap (780 vs. 680) doesn't have a further
    "obvious" fix** -- broke down the Analysis tab's own top-level
    layout item by item: it's dominated by two `QGroupBox`es (Peaks,
    328px; Fit Curve settings, 208px) that are already showing only
    their currently-relevant rows (`_on_delay_source_changed` already
    hides the Manual/Start+step/vclist rows that don't apply to the
    current selection) -- i.e. this height is genuinely load-bearing
    content, not slack left over from an oversight the way the peak
    table was. Closing the rest of this gap needs either (a) a broad,
    taste-driven spacing/padding tightening pass across every tab (not
    just Analysis), which is really the same thing as the still-open
    "beautification pass" item elsewhere in this file's history, or
    (b) a genuine structural change to the Analysis tab specifically --
    **which is exactly what the user's own "Analysis should have sub
    tabs or some other kind of organisational flow" ask (right below,
    still open) would also fix**, by not showing every group at once
    in the first place. Recommend treating these as the same item when
    picked back up, not two independent ones -- the sub-tabs question
    is a genuine design/taste call for the user to make, not a
    defensible judgment call to guess at overnight, so left as-is
    beyond the one concrete, low-risk fix above.

500 tests passing (500 -> 501, one new regression test). Committed and pushed.

**This closes out all five items from the "noted for next session"
list** (see that section below, now empty of anything not already
resolved) except the Analysis-tab sub-tabs question, which was always
explicitly open-ended and turns out to be the real fix for this item
too.

## Autonomous overnight round 3: Analysis-tab reorder and renames (2026-08-30)

- ~~"View peak fit" spin box should sit below the Fit Curve button/
  section, not above it~~ -- done. It (plus the uncertainty checkbox
  and its results label, which browse the same Fit Curve result) moved
  from between "Fit Spectrum" and the "Fit Curve settings" box down to
  after it -- the same "controls should follow, not precede, what they
  configure/what's fit before them" placement the Fit Curve button
  itself already used one round ago.
- ~~"Slice fit"/"peak fit" are confusing names -- should be "spectrum
  fit (slice #)" / "View curve fit (peak #)"~~ -- done, both the QLabel
  text and every comment/docstring quoting the old label text (found
  by grepping for the literal quoted strings, not just the two widgets
  themselves). Also gave "Spectrum fit (slice #)" the same left/right
  Prev/Next buttons "View curve fit (peak #)" already had (via the
  existing `_spin_with_step_buttons` helper) -- the user's own example
  phrasing ("this should have the left/right buttons, and then View
  curve fit (peak #)") named that as belonging to the first one
  specifically, and it was a real, small feature gap (a bare spin box,
  no buttons) once actually checked rather than assumed already there.
  Internal Python identifiers (`view_slice_fit_spin`, `_slice_fit_
  results`, etc.) deliberately left unchanged -- purely internal, no
  user-facing benefit to renaming them, only churn/breakage risk.
  Verified visually (rendered the Analysis tab to a PNG, not just the
  layout-order assertion), not just via the two new regression tests
  (row order via `QLayout.indexOf`, label text via `findChildren`).

The broader "sub tabs or some other organisational flow" part of this
item was explicitly open-ended (not spec'd further) -- left alone,
matching this project's own practice of not guessing at a vague design
ask when the two concrete asks alongside it were clear and actionable
on their own.

500 tests passing (498 -> 500). Committed and pushed.

## Autonomous overnight round 2: Processing-preview is now the default context for Align Echo/Autophase/phase-drag (2026-08-30)

Continuing unattended. Fixed the *routing* half of the "make
Processing-preview the default context for all processing steps"
item -- Align Echo, Autophase, Revert Autophase, Baseline Correct,
Autophase-each-slice-independently, and phase dragging (handle drag or
pivot slider) all now stay on the Processing-preview tab when that's
already active, instead of force-switching away to `fid_canvas`/
`self.canvas`. The other half (per-slice-only loading for pseudo-2D,
so the preview stays fast) turned out to need real vendor-level work
once actually investigated -- see the still-open item below for
exactly why, and what a real fix needs.

- `_draw_selected_spectrum`, `_apply_phase_preview` ("1d"/"pseudo2d"/
  "2d" branches), `_on_align_echo_toggled`, `_on_autophase_clicked`,
  `_on_revert_autophase_clicked`, `_on_baseline_correct_clicked`, and
  `_on_autophase_slices_clicked` all gained a "processing-preview tab
  already active? don't force-switch away" carve-out -- the same
  pattern the FID tab already had, applied consistently everywhere
  else that previously assumed Spectrum/FID were the only two
  possible "live" destinations. `_show_pseudo2d_slice` gained a
  `switch_tab` param (matching `_show_fid_preview`'s existing one) so
  Spectrum's own canvas can still be kept correct in the background
  without stealing focus.
- Discrete, one-shot actions (Autophase, Revert Autophase, Baseline
  Correct, Autophase-each-slice, `_on_align_echo_toggled`'s real
  commit) now also *refresh* the Processing-preview's content when
  it's active, not just avoid switching away from it -- cheap, since
  each only runs once per click.
- Continuous phase dragging (the drag-handle widgets, and the pivot
  slider) deliberately still does **not** refresh the Processing-
  preview on every intermediate tick -- for a pseudo-2D or a large
  conventional 2D, that would mean a full `Session.preview()` re-read
  (real I/O + FFT) many times a second, exactly the performance trap
  flagged when phase-dragging was first left out of the preview.
  Instead, new `DragHandle.dragFinished` (emitted once, on mouse
  release) and `QSlider.sliderReleased` (built into Qt already) each
  trigger exactly one refresh at the *end* of a drag gesture -- a
  reasonable middle ground: the preview updates once you let go,
  rather than either staying stale indefinitely or reloading on every
  pixel of mouse movement. For a plain 1D spectrum specifically, this
  same one-time-per-discrete-action refresh already happens via the
  "1d" branch's own logic (a single spectrum's own phase update was
  never expensive regardless of which canvas shows it), so 1D phase
  dragging refreshes correctly too, just via the same drag-finished
  path as everything else rather than needing special-casing.
- Real bug caught while adding the very first test for this:
  `_draw_selected_spectrum`'s own "else" branch (used whenever the
  selected spectrum isn't shown as a pseudo-2D stack or an
  already-correct Compare view) *also* unconditionally switched
  `canvas_tabs` to Spectrum, completely independently of the
  `_apply_phase_preview`-side fixes above -- meaning `_on_reprocess_
  fid_clicked` (which `_on_align_echo_toggled` calls for its real
  commit, and which ends by calling `_on_selection_changed`) was
  still yanking focus away even after the more obvious fix sites were
  patched. Caught by a real assertion failure, not by inspection --
  worth remembering that this kind of "canvas tab should stay put"
  fix has more call sites than the ones a user's bug report directly
  points at, given how many different code paths in this file
  ultimately redraw the Spectrum tab.

498 tests passing (492 -> 498, six new regression tests). Committed
and pushed.

## Autonomous overnight round: Compare auto-population fix, Stack-gap removal (2026-08-30)

Working through the "Noted for next session" list below unattended
(user asleep, per their own explicit go-ahead to continue without
waiting for clarification). Two items done and verified so far.

- ~~Compare no longer auto-populates from a pseudo-2D's own slices~~
  -- done, and the real cause was one layer deeper than the previous
  session's own diagnosis. That diagnosis (reordering would fix it)
  turned out to be *necessary but not sufficient* -- confirmed
  empirically by tracing every handler call during a real
  `_select_single_row(0)`: `spectrum_list` fires **two independent**
  signals for a single-selection change (`itemSelectionChanged` ->
  `_on_list_selection_changed`, *and* `currentRowChanged` ->
  `_on_selection_changed` directly), and Qt genuinely fires both for
  this exact gesture, with the direct one landing *last* regardless of
  which order the code's own logic ran in. A pure reorder (tried
  first) only moved the problem around: whichever handler happens to
  run last always clobbers the Compare view, since both eventually
  call the same "redraw Spectrum, clear Compare" logic. Real fix is
  order-independent instead: new `_compare_shows_current_pseudo2d_stack`
  checks whether `compare_list` already exactly matches the selected
  spectrum's own auto-populated slice stack, and both
  `_draw_selected_spectrum` (skip the force-switch back to Spectrum)
  and `_silo_other_canvas_tabs` (skip clearing `compare_canvas`, via a
  new `skip_compare` param) consult it -- so no matter how many times
  or what order these handlers fire for the *same* selection, an
  already-correct Compare view is left alone rather than redrawn over.
  New regression test asserts both `canvas_tabs.currentWidget()` and
  the actual plotted line count, not just that the list populates
  (the list was never actually broken -- only the display was).
- ~~Compare's "Stack gap" control does nothing~~ -- done, via removal
  rather than a rewire, per the user's own "should also be automatic"
  framing. Confirmed the control had **zero visible effect** for every
  realistic use (the `mode="stack"`, >1-spectrum case plots each row
  on its own independent, auto-scaled axes with a fixed `hspace=0.0`
  -- no offset concept left there at all; the *only* code path that
  still read the spin box's value was a degenerate `mode="stack"`-with-
  exactly-1-spectrum fallback, where a vertical offset is inherently
  meaningless since there's nothing else to offset from). Removed
  `stack_gap_spin` entirely (widget, form row, enable/disable wiring)
  and the now-pointless `y_offset` parameter on
  `MplCanvas.show_compare` itself -- the two remaining internal uses
  (overlay's `-1.0` collapse trick, the single-spectrum-stack
  fallback's `0.2`) are now fixed constants, not user-facing settings,
  since neither one is a "gap between multiple stacked spectra" in any
  case a person would actually see or want to tune.

492 tests passing (unchanged count -- one test gained new assertions,
one lost an obsolete one, net zero). Committed and pushed.

## Noted for next session (2026-08-29) -- not yet implemented

User feedback from trying an earlier round's changes -- explicitly "a
few pointers to note to do in the next session," not "implement now."
The two Compare-related items, the "make Processing-preview the
default context" routing work, and the Processing-preview performance
item are done -- see the rounds above (the performance one specifically
needed the user's own direction to actually resolve, not just more
investigation -- see the "Compare-tab landing fix, and the real
per-slice-preview performance fix" section for the full story). Still
open:

- **`fid_raw` caching for Varian/JEOL** -- small, mechanical follow-up
  to the Processing-preview performance fix above: Bruker and X-Pulse's
  `read_1d`/`read_2d` now stash the true raw FID into
  `metadata.raw["fid_raw"]` for the live in-memory preview to use;
  Varian and JEOL don't yet, so their Processing-preview tab shows
  correctly empty (not wrong, just unavailable) rather than a live
  preview. Same two-line addition as the other two vendors got, once
  someone confirms where in `read_1d` each one's own "raw, post
  digital-filter, pre-shift/align" point is.
- **Analysis tab organization** -- open-ended ("sub tabs or some other
  kind of organisational flow," not spec'd further); the two concrete
  asks alongside it (View peak fit's position, the slice-fit/peak-fit
  renames) are done -- see the "Autonomous overnight round 3" section
  above. This broader, unspec'd part is still genuinely open, and (see
  "Autonomous overnight round 4" above) turns out to be the actual fix
  the "left side tab needs scrolling too often" item was also asking
  for -- the Analysis tab alone needs 780px against a normal window's
  680px-tall panel, entirely from two group boxes that are already
  showing only their currently-relevant rows; sub-tabs (not showing
  every group at once) is a real structural fix for that, a broad
  spacing-tightening pass across every tab is the only other real
  lever, and a wider window floor / other tabs' own sizing were ruled
  out as not actually the problem (measured, not guessed).

## Spectrum-tab stacked-view fix, linked Compare scroll-zoom, Processing-tab live preview, button wording (2026-08-29)

- ~~Stacked view should only default in the Compare tab, not the
  Spectrum tab -- it breaks phasing there~~ -- done. Last round's
  pseudo-2D-defaults-to-Stacked change (`pseudo2d_view_combo`) was
  scoped too broadly: it changed the *Spectrum* tab's own default too,
  not just Compare's -- but the Spectrum tab is also where phasing
  happens (Phase group, per-slice/whole-dataset), which needs a plain
  single-slice/contour view, not the multi-axes Compare-style stack.
  Reverted `_refresh_selection_display`'s default back to Contour
  (index 0); "Stacked slices" stays available from the combo, just
  isn't the default. Runs unconditionally on every reselection (like
  the phase controls resetting alongside it), so it overrides a
  Stacked choice left over from a previous spectrum rather than
  carrying it forward, per "make sure this overrides whatever
  selection there was."
- ~~Y-scaling of the Compare stack via scroll wheel should be linked,
  unless one spectrum is selected (some kind of clickable thing per
  spectrum)~~ -- done. Scroll-zoom on a "Stacked" Compare view now
  zooms every row together by default (same relative factor per row,
  each still pinned at its own baseline) -- click a row (a plain
  click, not a drag -- distinguished via press/release pixel distance,
  see `MplCanvas._CLICK_DRAG_TOLERANCE_PX`) to select it, highlighted
  with a light amber tint (`_apply_compare_selection_highlight`);
  scroll then zooms only that row, regardless of which row is actually
  hovered, until it's clicked again to de-select. `_selected_compare_index`
  lives alongside `_compare_axes`, reset everywhere that already gets
  reset (the same 7 sites, via the same insert-after-`_fit_mode`
  technique as last round).
- ~~A distinct Processing-tab live preview (FID top, downsampled
  spectrum bottom, default first slice with a selector for
  (pseudo-)2D), separate from the FID/Spectrum tabs which should only
  show the results of processing~~ -- done, a real architecture
  change. New `processing_preview_canvas` (6th canvas tab, "Processing
  Preview"), with a new `MplCanvas.show_processing_preview` (two
  stacked subplots, no shared x -- FID is time, spectrum is ppm).
  Selecting the Processing (left) tab auto-switches the canvas over to
  it (`_on_side_tab_changed`); leaving switches back to Spectrum,
  *unless* the user had manually navigated elsewhere while still on
  the Processing tab (only switches back if the preview was actually
  the one showing). The pseudo-2D slice selector is the *existing*
  `slice_spin` (already the Dimension box's own per-slice picker) --
  reused, not duplicated, wired via a new `_on_slice_spin_changed`.
  New `Session.preview(index, **kwargs)` -- like `reload`, but a pure
  scratch re-read that never replaces the stored spectrum -- computes
  exactly what "Apply Processing" would currently produce, via the
  same kwargs-gathering the real commit uses (extracted into
  `_current_fid_processing_kwargs`, so the two can't drift apart).
  Every FID-processing control's live-changed signal now targets this
  new preview instead of hijacking the real FID tab (which is what
  broke it for its own actual job, e.g. reviewing a real result after
  reprocessing) -- `fid_shift_spin`/`linear_predict_points_spin`/
  `linear_predict_order_spin` gained live-preview wiring they never
  had at all before (previously only took effect on a real "Apply
  Processing" click). **Deliberately did not** route phase-dragging
  into this new preview -- phase edits mutate the real spectrum's data
  directly, in place, on every tick (no separate commit step the way
  FID-domain reprocessing has), so they're already "the result of
  processing," not a hypothetical -- and a live `Session.preview()`
  (full re-read + FFT) on every phase-drag tick would be far too slow
  regardless. Caught one real interaction bug while wiring the slice
  selector: `_apply_phase_preview`'s pseudo2d branch unconditionally
  force-switched `canvas_tabs` back to Spectrum whenever the phase
  target changed (which `slice_spin` also triggers) -- silently
  fighting the new preview's own tab-switch. Fixed by adding the same
  "don't force-switch away" carve-out the FID tab already had.
- ~~"Reprocess from raw FID" wording is confusing~~ -- done. Renamed to
  "Apply Processing" (the button itself, plus the one user-facing
  message and one docstring that named the old text).

492 tests passing (up from 483). Committed and pushed.

## Fit point exclusion, Compare subplots, pseudo-2D default, rendering caps, full-resolution export (2026-08-29)

- ~~Fit uncertainty is sometimes very unreliable; right-click to remove
  data points from the fit (and reset)~~ -- done. New
  `MplCanvas.fitPointExcludeRequested`/`fitPointsResetRequested` signals,
  captured from real matplotlib `event.xdata`/`event.ydata` at
  `button_press_event` time (button==3) rather than converting a Qt
  `QPoint` from `customContextMenuRequested` -- avoids the Qt-vs-
  matplotlib coordinate/HiDPI conversion entirely. Nearest-point search
  is axis-range-normalized so the delay/b-value and intensity axes
  (very different numeric scales) don't let one dominate. New
  `MainWindow` state, siloed per peak (`_relaxation_fit_original`,
  `_excluded_relaxation_indices: dict[int, set[int]]`) so excluding a
  point on one peak doesn't touch another's fit; `_refit_relaxation_peak`
  rolls back (returns `bool`, caller un-excludes on failure) rather than
  leaving a peak in a half-fit state if too few points remain. Excluded
  points still drawn (grey 'x') so it's clear what was removed, not just
  silently gone.
- ~~Rescaling the Compare plot should keep each baseline in place --
  maybe subplots with a linked x axis~~ -- done, and the "maybe" was
  right: a single shared linear y-axis genuinely *cannot* keep multiple
  stacked baselines fixed under a rescale (any y-zoom multiplies every
  distance from the pin point by the same factor, including the gaps
  *between* different traces' own baselines -- not fixable within one
  shared axes, confirmed mathematically before implementing, not just
  by trial and error). `show_compare`'s "Stacked" mode now builds one
  subplot per spectrum via `self.figure.subplots(n, 1, sharex=True)`,
  each independently y-autoscaled; the existing scroll-zoom pin-bottom
  logic already keyed off `event.inaxes` needed no changes at all to
  work per-row. Non-"Stacked" Compare modes (single overlay) unchanged.
- ~~In a pseudo-2D, stacked spectra should be plotted by default~~ --
  done, one-line change (`_pseudo2d_view_combo`'s default index 0 -> 1).
- ~~Some limit on how many points render at once, especially 2D/many
  stacked spectra, for dynamic-processing speed~~ -- done for 1D/stack
  (2D contour rendering was already downsampled via
  `plotting.style.downsample_2d`, pre-existing). New
  `DOWNSAMPLE_1D_MAX_POINTS = 8000` + `downsample_indices(n, max_points)`
  in `plotting/style.py`; `plot_1d` renders at most that many points
  (evenly-spaced indices) unless told not to. `xmin`/`xmax` returned are
  always the true full-resolution range regardless -- display-only,
  `spectrum` itself never touched.
- ~~Make sure exports are fully sampled, with an option to view at full
  resolution too~~ -- done, prompted directly by the downsampling change
  above (a real risk it introduced: silently downsampling with no way
  back would undercut a tool whose whole point is showing the actual
  data). `plot_1d` now always stashes the true-resolution arrays on the
  line itself (`line._opennmr_full_res`) whenever downsampling *would*
  apply, regardless of whether it's currently in effect -- so a later
  toggle or export never needs to re-plot from the spectrum object.
  `MplCanvas` gained a persistent `_full_resolution` flag (survives
  rebuilds, like the legend/aspect-lock flags already do), a checkable
  "Full Resolution" right-click action (only shown when
  `has_downsampled_data()`), and `export_full_resolution(save_fn)`
  (swaps every visible line to its full-res data, calls `save_fn`,
  restores the view afterward -- even if `save_fn` raises). Wired in at
  the toolbar level, not just the menu action: `_Toolbar.save_figure` is
  overridden so *both* the toolbar's own Save icon and the File menu's
  "Save Figure As…" always route through this, and can't be bypassed by
  using one path but not the other.
- ~~Scrolling to scale axes should also work for the y axis in 2D and
  pseudo-2D~~ -- done. New `_hovering_y_axis`/`_data_y_at` helpers mirror
  the existing x-axis-strip scroll-zoom logic (hovering past the right
  spine, matching `tick_right()`), gated to contour views only
  (`self._contour_factor is not None`).

483 tests passing (up from 457). Committed and pushed.

## Peak-fit arrows, relative +/- entry, layout fixes, uniform Compare spacing (2026-08-29)

- ~~View peak fit should have arrows like the phasing style~~ -- done.
  Factored the pseudo-2D slice picker's Prev/Next (◀/▶) buttons
  (added last round) into a shared `MainWindow._spin_with_step_buttons`
  helper, reused for both the slice spin box and the new "View peak
  fit" spin box.
- ~~Autophase isn't working for JEOL data; features should be added to
  all input styles since after import the data has the same basic
  structure~~ -- **investigated thoroughly, not guess-fixed**. Verified
  directly against real `test_data/jeol_1d/proton.jdf` through the
  actual GUI code path: Autophase computes phase0≈0 and changes
  nothing, because JEOL's own `read_1d` already autophases at load
  time (same as every vendor) -- the button correctly finds nothing
  further to do. The real, underlying issue is the JEOL importer's
  own already-documented gap (see `io/jeol.py`'s module docstring,
  written when the importer was first built): an uncorrected
  `FILTER_FACTOR` digital-filter delay leaves at least one peak
  phase-flipped regardless of how well the rest is phased -- not a
  gap in the Autophase *feature* (which is fully vendor-agnostic,
  confirmed: `data_imag` retained, `phase0`/`phase1` stored in
  `metadata.raw`, `pivot_frac` threaded through `jeol.read_1d` same as
  every other vendor). Investigated whether this specific DSP gap
  could be closed, the same way Bruker's GM window gap couldn't be
  (see below) -- (1) re-checked the `jeolconverter` JS reference this
  importer was ported from: it reads `FILTER_FACTOR` as metadata only,
  never applies a correction; (2) searched for JEOL's own group-delay
  correction algorithm -- confirmed proprietary/undisclosed, not
  published anywhere; (3) empirically tested the most plausible
  *generic* correction (treating `FILTER_FACTOR` like Bruker's GRPDLY
  -- shift the FID by that many points before FFT, via the
  already-existing `processing.ft.roll_fid`) against the real file --
  it did **not** cleanly fix the described symptom and showed signs of
  introducing a new, worse artifact (a large swing in the phased
  spectrum's overall sum, consistent with the circular shift wrapping
  significant tail energy into the front). Not shipped -- matches this
  project's own established practice (see the Bruker WDW=2/GM
  decision) of flagging a real, verified gap rather than guessing at
  an unverified DSP correction that could silently produce a wrong
  spectrum. Real, closable follow-up if this becomes a priority: get a
  JEOL spectrum with an independently-known-correct phase (e.g. cross-
  checked against Delta/JASON's own processed output) to calibrate
  against, since neither a public spec nor a working reference
  implementation exists to port from.
- ~~Fit Curve button should come below the choice of model~~ -- done,
  moved to the end of the "Fit Curve settings" box (below every
  setting it actually uses -- model, value source, delay/b-value
  source -- not just the one right above it, which is what "below the
  model" alone would have suggested).
- ~~Start/stop boxes hard to see, should be the same length and stack
  vertically~~ -- done (these are actually labelled "Start"/"Step").
  Two separate form rows now, each spin box a fixed, equal 120px wide,
  instead of both crammed side by side in one row of this narrow
  side panel.
- ~~Add/subtract by typing e.g. "+90" after "45" to get "135"~~ -- done.
  New `RelativeDoubleSpinBox` (`gui/widgets/relative_spinbox.py`)
  recognises a self-contained typed expression (`45+90`, `100-30`) and
  evaluates it instead of failing to parse -- swapped in for Phase
  0/Phase 1/Pivot. Deliberately requires *both* operands spelled out
  (not a bare `+90` alone, implicitly relative to the value before
  editing started): a bare leading `-` (e.g. selecting all and typing
  `-45` to set a literal negative value -- a completely normal thing
  to do) would otherwise be ambiguous between "set to -45" and
  "subtract 45," with no way to tell which from the committed text
  alone. A bare `+` has no such ambiguity, but supporting only one
  sign's bare form and not the other would be a confusing half-measure,
  so neither is supported -- only the unambiguous, fully-spelled-out
  expression form, which is also exactly the form the request's own
  example described ("type +90 *after* the 45").
- ~~Compare-tab spacing should be a uniform value, not relative to
  each spectrum's own max~~ -- done. `plot_1d_stack`'s per-step offset
  was `np.max(that spectrum's own y) * (1 + y_offset)` -- fine when
  `normalize=True` (every trace's own max is 1 either way), but the
  Compare tab's "None"/"By sample mass" modes pass `normalize=False`
  specifically so each trace keeps its true relative height, and *that*
  varying height was also scaling the *gap above* it, making the
  stack look uneven. Now uses one shared step size (the tallest
  spectrum's own height, computed once) for every gap, while each
  trace still plots at its own true height.

457 tests passing. Nothing committed yet from this round (not asked to).

## Contour defaults, apc0 robustness, per-peak relaxation-fit browsing, pivot direction (2026-08-29)

- ~~2D/pseudo-2D contours still wrong -- too many, lowest should be
  5-10% of max~~ -- done. `_CONTOUR_FACTOR_DEFAULT` 1e-3 -> 0.07,
  `_CONTOUR_CLEVELS_DEFAULT` 20 -> 7 (`gui/widgets/canvas.py` and the
  matching standalone defaults in `plotting/twod.py`'s
  `_prepare_contour`/`plot_2d_contour`/`overlay_plot`). The earlier
  round's level-overshoot formula fix was real and correct but didn't
  address this -- these are separate issues (a math bug vs. a default
  values choice) that both needed fixing for the reported symptom to
  actually go away.
- ~~Autophase still doesn't work, probably zeroing everything too
  much, even basic per-slice phase~~ -- **real bug found and fixed**,
  not just a data problem. `processing.ft.apc0` scored its grid search
  by raw `sum(real(corrected))` over the *whole* spectrum -- on
  anything that's mostly noise/baseline with only a small real peak
  (a late PFG gradient-step slice with decayed SNR, exactly what's
  being worked with right now), that sum is dominated by noise, not
  the peak, so the "best" phase it finds is essentially arbitrary.
  Confirmed with a direct synthetic side-by-side before touching the
  code: at moderate noise, the old scoring recovered a known phase to
  within 11 deg; `phase_score`'s "magnitude" penalty (already used by
  `autophase01`, phase-independent by construction so noise
  contributes comparably regardless of the phase tried) got within 2
  deg, with the gap widening further at higher noise. Both the
  Autophase button and "Autophase (each slice independently)" call
  `apc0` directly, so both are fixed by the same change. New
  regression test pins recovery to within 5 deg at a noise level where
  the old scoring would have failed by a wide margin.
- ~~Phase pivot slider direction should match the (reversed) axis~~ --
  done. The slider was mapping low value -> low ppm / high value ->
  high ppm, but the spectrum's own x-axis is plotted inverted (high
  ppm on the left) -- dragging the slider right visually moved the
  pivot marker *left*. Inverted the slider's mapping to match.
- ~~An easy way to phase per slice and pick easily~~ -- done. New
  Prev/Next (◀/▶) buttons next to the slice spin box -- quicker than
  the spin box's own tiny arrows or typing a number for stepping
  through "Autophase -> Next -> Autophase -> Next -> ..." one slice at
  a time. `setValue` clamps to the spin box's own range on its own, so
  these are safe no-ops at either end.
- ~~Fitting should be based on the user's own PFG fitting code, with
  (optionally) shaded-region error bars, one peak at a time with a way
  to switch~~ -- done. New `processing.relaxation.fit_curve_uncertainty`
  hand-rolls the same idea as the reference code's `lmfit`
  `eval_uncertainty` (a linearized/delta-method uncertainty band,
  propagated from `fit_relaxation`'s own parameter covariance via a
  numerically-estimated Jacobian, since this module fits with plain
  `scipy.curve_fit` rather than `lmfit`) -- returns `(t_fine, y_fine,
  y_band)` for a smooth curve + envelope, shaded in the GUI via
  `ax.fill_between`. `MplCanvas.show_relaxation_fit`'s signature
  changed from a list of every peak plotted together to ONE peak +
  optional `uncertainty` tuple, styled after the reference plots
  (black circles for data, dashed red fit line, grey shaded band --
  not a colour-per-peak legend, which stops meaning anything once only
  one peak is ever shown). New "View peak fit" spin box (Analysis tab,
  same pattern as the existing "View slice fit" one) browses between
  peaks; new "Show uncertainty band" checkbox toggles the shading.
  Three new GUI tests (spin box disabled before a fit, switches
  correctly between 2 peaks with only 2 lines ever on screen, checkbox
  toggles the shaded region on/off) plus three `processing/relaxation.py`
  unit tests for `fit_curve_uncertainty` itself (shape/positivity,
  scales linearly with `sigma`, tighter with less noise).

448 tests passing. Nothing committed yet from this round (not asked to).

## Phase workflow round 2, WDW/contour robustness, bidirectional pseudo-2D (2026-08-29)

- ~~Keep the phase information when saying reprocess~~ -- done, for
  every case where it's actually well-defined. New
  `MainWindow._current_phase_kwargs`/`_reapply_phase_post_reload`
  compute the *total* current phase (whatever a vendor's `read_1d`/
  `read_2d` already baked in via `metadata.raw["phase0"/"phase1"]`,
  plus the live delta still showing in the Phase spin boxes -- see
  `_capture_phase_reference`'s own docstring for why the spin boxes
  are deltas, not the total) and forward it into `session.reload(...)`
  as `phase0`/`phase1`/`pivot_frac`/`autophase=False`, for 1D, a whole
  2D dataset along F2, or a pseudo-2D with "apply to every slice"
  checked. The two cases that don't reduce to one scalar phase shared
  across the whole dataset -- a single pseudo-2D slice's own phase, or
  a conventional 2D's F1/indirect phase -- are instead reapplied
  post-hoc directly onto the freshly-reloaded spectrum, since neither
  `read_1d` nor `read_2d` has a way to phase just one row/dimension
  from a raw-FID call. `io/bruker.py::read_2d` gained `phase0`/
  `phase1`/`pivot_frac` params it didn't have at all before (previously
  zero-order-only, no way to preserve a caller's phase across a
  reload); every vendor's `read_1d` and `varian.read_2d` gained
  `pivot_frac` (was always hardcoded to 0.5/centre, silently
  re-pivoting away from wherever the user had actually set it).
  Wired into both "Reprocess from raw FID" and the pseudo-2D/
  conventional-2D toggle (`process_indirect_button`), since both
  reload from raw FID.
  **Real bug found while testing this, fixed**: the first version of
  `_current_phase_kwargs` didn't check whether `spectrum` was actually
  reprocessable from raw FID at all -- for a Bruker dataset with no
  `fid`/`ser` on disk (routed through `read_1d_processed`/
  `read_2d_processed`, which have no `phase0`/`phase1` concept), it
  still tried to forward them, raising `TypeError` inside a modal
  `QMessageBox.critical(...).exec()` call that blocks forever under a
  headless/offscreen Qt platform -- a full test-suite run hung with no
  error output. Same "silent hang = check for an unmocked call
  reaching a real modal dialog" lesson as the canvas-tabs-default bug
  two rounds ago. Fixed by gating on `_fid_reprocessable(spectrum)`.
- ~~Autophase should support 1st-order phasing~~ -- done. New "1st-order"
  checkbox next to the Autophase button switches it from `apc0`
  (zero-order-only) to `processing.ft.autophase01`'s joint search.
- ~~A revert option in case autophase was bad~~ -- done. New "Revert
  Autophase" button, disabled until an Autophase click actually
  happens, restores Phase 0/Phase 1 to their pre-autophase values.
  Invalidated (disabled again) on reselection/slice/dimension change
  (`_capture_phase_reference`) -- a stale snapshot from a *different*
  view would restore the wrong thing.
- ~~A slider bar for the pivot point~~ -- done. New `pivot_slider`
  (0-1000, mapped to the spectrum's own ppm range) next to the
  existing spin box, synced bidirectionally via `blockSignals` on
  whichever side didn't originate the change.
- ~~Make sure things aren't resetting when the pivot is changed~~ --
  verified: moving the pivot (spin or slider) only calls
  `_apply_phase_preview`, never `_capture_phase_reference` (which
  zeroes Phase 0/Phase 1) -- covered by a new regression test setting
  a nonzero phase, moving the pivot both ways, and asserting the phase
  is untouched.
- ~~Axis positioning should update to zero the baseline more
  effectively once phased~~ -- done, for *discrete* phase actions only
  (Autophase, Revert Autophase, Baseline Correct) -- new
  `MplCanvas.autoscale_y()` refits the y-axis to whatever's actually
  plotted now, computed directly from the main line's own data (not
  `Axes.relim()`/`autoscale_view()`, which silently no-ops once this
  canvas's own scroll-zoom/pan handlers have called `set_ylim`
  directly -- confirmed empirically, not assumed). Deliberately *not*
  wired into continuous dragging, which still preserves the current
  view exactly as before (a validated, tested design choice from an
  earlier round) -- a one-shot click and a drag gesture warrant
  different behaviour here.
- ~~A baseline correction option~~ -- done. New "Baseline Correct"
  button (`processing.ft.baseline_correct`, linear fit to the real
  part) targets whatever the Phase controls currently target -- same
  rules as Autophase, including the pseudo-2D single-slice case.
- ~~WDW function not found / make this more robust~~ -- done, **twice**:
  the first version fell back to `read_1d_processed`/`read_2d_processed`
  (TopSpin's own already-processed output) whenever the *recorded*
  Bruker window function wasn't one of the ones this package
  implements (e.g. GM) -- fixing the immediate crash
  (`NotImplementedError` bubbling through `_load` into a blocking
  error dialog) but at the cost of losing raw-FID reprocessing
  entirely for that spectrum. **The user pushed back**: "aren't we
  starting with unprocessed data? So why are we using the processed
  output?" -- exactly right, and a real miss against this project's
  own stated premise (every vendor reads the raw FID through one
  shared pipeline, `read_1d_processed`/`read_2d_processed` are meant
  as a last-resort *when no raw FID exists at all*, not a first
  choice degradation when one parameter isn't understood). Reworked:
  `read_1d`/`read_2d` now catch the `NotImplementedError` around only
  the windowing step, leave the FID unwindowed, and continue normally
  -- zero-fill, phase, echo-align, linear-predict, and "Reprocess"
  itself all keep working exactly as they would for any other
  spectrum; the user can pick a different window from the Processing
  tab immediately. `_read_auto`'s fallback-to-processed machinery
  (and the now-pointless `_filter_kwargs`) was reverted entirely --
  it's still there for the genuine "no raw FID on disk at all" case,
  unchanged from before this whole detour. Flagged via
  `metadata.raw["window_unsupported"]`, surfaced once via
  `QMessageBox.information` with much softer wording ("the recorded
  window wasn't applied," not "showing TopSpin's own data instead"),
  and `_fid_reprocessable` no longer treats this spectrum as
  unreprocessable, since it fully still is. Deliberately still did
  **not** attempt to actually implement GM -- see the module's own
  docstring for why (an unverified `LB`/`GB` parameter mapping risks
  silently producing a wrong spectrum).
- ~~Tuple error, some data from "the box" had issues~~ -- **not
  reproduced, need more detail from the user** (a traceback, or which
  file/vendor). Checked the obvious candidates (`_resolve_window_kwargs`,
  `_zero_fill_value`, the jcampdx multi-value array parser, `f1p`/`f2p`
  crop sorting) and found nothing tuple-shaped that looks wrong --
  didn't want to guess-fix something this vague and risk masking the
  real bug or introducing a new one.
- ~~2D contours start out scaled way too high~~ -- **real bug, found
  and fixed**: `_prepare_contour`'s power-law level spacing used
  `power * i / clevels` as the exponent, which only reaches exactly
  `power` (not 1) at the top level -- for `power > 1` (the default is
  1.25), that *overshoots* the real data max, so the top several
  "contour levels" sat above every real data point and were never
  actually drawn. The levels that *did* draw were squeezed into a
  smaller fraction of the true range than intended, several coarser
  than they should be -- moderate, real signal often fell entirely
  between two of them, looking like a 2D/pseudo-2D dataset "starts out
  scaled way too high" (barely anything visible) until the threshold
  was manually scrolled down a lot. Fixed by changing the exponent to
  `(i / clevels) ** power`, which always reaches exactly 1 (so the top
  level is exactly the data max, regardless of `power`) while still
  bunching more levels toward the noise floor as `power` increases --
  matching what the function's own docstring already claimed the old,
  buggy formula did. Two new regression tests pin both properties
  (top level == data max; steeper power bunches lower, not higher).
- ~~A way to switch between pseudo-2D and processing the indirect
  dimension in both directions~~ -- this already existed
  (`process_indirect_button`, a checkable toggle -- checked = real F1
  FFT, unchecked = pseudo-2D) and structurally supports both
  directions, but **every existing test only ever exercised one
  direction** (pseudo-2D -> conventional). Added a regression test
  that toggles both ways twice in a row on the same spectrum, which is
  exactly what caught the `_current_phase_kwargs` `TypeError`/hang
  above (only reachable from the *second* toggle, once `_phase_mode`
  had become `"2d"`).

439 tests passing (this round: several new phase/pivot/autophase/
baseline/contour/bruker-fallback tests). Nothing committed yet from
this round (not asked to).

## Phase-tab feedback, plot styling, menu bar (2026-08-29)

- ~~Default FID preview should be Real, not Magnitude~~ -- done.
  `fid_display_combo`'s items were just reordered (Real first) since
  the GUI reads the mode via `currentData()`, not index.
- ~~Phase 1's increment should be 1-2 orders of magnitude bigger than
  Phase 0's~~ -- done. New `_PHASE1_STEP_MULTIPLIER = 20.0` scales
  both the drag-handle sensitivity/wheel-step and the spin box's
  `singleStep` for Phase 1 only -- a linear (whole-spectrum)
  correction needs a much bigger raw-degree change than a zero-order
  one to visibly move anything.
- ~~A line showing where the pivot is while phasing~~ -- done. New
  `MplCanvas.set_pivot_marker(ppm)` draws/moves/hides a dashed orange
  `axvline` on the current 1D view. Deliberately *not* shown on every
  plain spectrum selection (tried that first -- broke a dozen tests
  asserting `len(ax.lines) == 1` after a plain load, and is genuinely
  the wrong behaviour anyway: "while phasing" means while the user is
  actually interacting with the Phase controls, not on every view).
  Wired into `_apply_phase_preview` (drag/spin/pivot changes),
  `_on_autophase_slices_clicked`, and pseudo-2D slice switching --
  not into plain spectrum (re)selection.
- ~~Missing autophase button/option~~ -- a plain 1D spectrum had *no*
  autophase control at all (the only existing one,
  `autophase_slices_button`, is pseudo-2D-only). New "Autophase"
  button in the Phase box runs `processing.ft.apc0` on whatever the
  Phase controls currently target (1D spectrum, the selected pseudo-2D
  slice or all slices, or the whole 2D dataset along the chosen
  dimension) and writes the result into the shared Phase 0 spin box
  (resetting Phase 1 to 0, since apc0 is zero-order-only) -- same
  live-preview path as dragging it by hand.
- ~~Axis linewidths/font sizes too small; style the fitting plots more
  like nmrtools2's own plotting functions~~ -- done. New shared style
  constants on `MplCanvas` (`_AXIS_LINEWIDTH=1.4`, `_AXIS_LABELSIZE=13`,
  `_TICK_LENGTH=5.5`, `_TICK_WIDTH=1.4` -- heavier than
  `plotting.style.setup_plotting`'s own manuscript-figure defaults,
  since a GUI canvas is viewed larger with more surrounding UI) applied
  to every 1D-line view (Spectrum, FID, Compare, Spectral Fit,
  Relaxation Fit). `apply_spine_style`/`apply_tick_style` gained
  `linewidth`/`labelsize` parameters to make this possible without
  `setup_plotting`'s global rcParams (never callable from GUI code --
  see that module's own docstring). `show_relaxation_fit` previously
  had *zero* styling calls at all (default matplotlib look, all four
  spines shown) -- now matches the others. Plotted-line widths in
  `show_slice_fit` bumped to match `nmrtools2`'s own
  `NMR1dplot_with_fit` convention (Data 1.5, Fit 2.0, components 1.2).
- ~~File/Edit/Window menu bar~~ -- done (this was already an open item
  in "App polish / packaging" below). Native on macOS, in-window on
  Windows/Linux, one `QMenuBar` via `MainWindow._build_menu_bar`. File:
  Open… (Cmd+O), Save Figure As… (Cmd+S, delegates to whichever canvas
  tab's own matplotlib toolbar `save_figure()`), Quit (Cmd+Q). Edit:
  Remove Selected Spectrum (Delete), Reset Phase, Autophase. Window:
  Minimize (Cmd+M), then Cmd+1-4 for the left-panel tabs
  (Spectra/Processing/Analysis/Compare) and Cmd+5-9 for the canvas
  tabs (labelled "... View" to disambiguate from the identically-named
  "Compare" panel tab).
- **Real latent bug found while testing the menu bar, fixed**:
  `canvas_tabs` defaulted to the *FID* tab (index 0, added first) on
  a fresh window, not Spectrum -- contradicting `self.canvas`'s own
  docstring ("the default/most-used one"). Harmless-looking most of
  the time since `_draw_selected_spectrum` switches to Spectrum on
  load *unless* the FID tab was already active AND the spectrum has no
  FID to preview (e.g. any plain processed/non-raw-FID spectrum) -- in
  which case neither branch fires and the window is left showing an
  empty FID tab with the actual spectrum silently drawn to the
  Spectrum tab behind it. Surfaced as an actual test hang, not just a
  failure: a "Save Figure As…" test mocked `canvas.toolbar.save_figure`
  but the *real* `fid_canvas.toolbar.save_figure()` ran instead (since
  that was the actually-current tab), popping a real native save
  dialog that blocks forever under a headless/offscreen Qt platform.
  Fixed with one line, `self.canvas_tabs.setCurrentWidget(self.canvas)`
  right after the tabs are built in `__init__`.

424 tests passing (410 -> 424 this round). Nothing committed yet from
this round (not asked to).

## Packaged-app crash fix + JEOL reprocessing gap (2026-08-29)

~~App built with `packaging/build.py` crashes on launch from
`/Applications`~~ -- fixed. User reported a real macOS crash report
(`EXC_CRASH`/`SIGABRT` inside
`QGuiApplicationPrivate::createPlatformIntegration()`). Root-caused via
`QT_DEBUG_PLUGINS=1` + `otool -L` + `conda list`, not guessed at: this
dev environment has a conda-forge `qtbase` 6.7.3 package installed
alongside pip-installed PySide6 6.11.2 (pulled in by `qtconsole`/PyQt6,
both also in the env) -- PyInstaller's dependency collector bundled the
*conda* Qt's platform plugin (`libqcocoa.dylib`, flat
`@rpath/libQt6Core.6.dylib` linkage, version 6.7.3) instead of
PySide6's own (framework-style `@rpath/QtCore.framework/...`, version
6.11.2), even though the core Qt libraries bundled alongside were
correctly 6.11.2 throughout. Qt's own plugin loader refuses to load a
mismatched-version platform plugin at all, so no window is ever shown
and the app aborts. Fixed via a new `_fix_qt_plugins()` post-build step
in `packaging/build.py` that wholesale-replaces every bundled
`PySide6/Qt/plugins` directory with a fresh copy from the actually-
installed PySide6, sidestepping the mismatch regardless of why
PyInstaller's own collection picked the wrong source. Verified: rebuilt
app's `libqcocoa.dylib` now links `QtCore.framework` 6.11.2
(`otool -L`); rebuilt `.app` launches and stays running (`ps aux`);
reinstalled to `/Applications` (replacing the old broken copy) and
re-confirmed launching cleanly from there too; fresh
`dist/opennmr-macos.dmg` rebuilt from the fixed `build.py` (the DMG
opened for the user earlier in this session predated the fix and would
still have crashed).

~~New JEOL spectra don't show FID reprocessing options~~ -- two related
bugs, fixed together. (1) `MainWindow._fid_reprocessable` (gates
whether the Processing tab's FID-reprocessing box is shown) only
checked for `"xpulse"`/`"varian"`, missing `"jeol"` entirely -- same bug
class as an earlier Varian fix, now hit again because adding a new
`io/` vendor doesn't automatically register it with every GUI-side
whitelist. (2) Even with the box shown, clicking Reprocess would have
raised `TypeError`: the GUI's reprocess handler unconditionally sends
`align_echo=...` to whichever vendor's `read_1d` is active, and
`io/jeol.py::read_1d` had no `align_echo` parameter at all (deliberately
left out originally, since real 1D JEOL data doesn't obviously need
it) -- every other vendor's `read_1d` already accepts it, so this broke
the established uniform kwarg interface. Fixed by adding
`align_echo: bool = False` to `jeol.read_1d` (applies
`processing.ft.align_echo` when set, same as the other vendors; no
wrap-truncation cap like X-Pulse's since there's no `truncate` concept
for JEOL). 2 new GUI regression tests + 1 new `test_jeol.py` unit test,
413 tests passing overall.

## JEOL Delta (`.jdf`) importer -- 1D (2026-08-29)

~~Find real JEOL example data (or build synthetic data from the
manual) to build a parser~~ -- done, a new `io/jeol.py` (`read_1d`),
registered like every other vendor. No official public spec exists for
this binary format, so this is a Python port of
[bjonnh/jeolconverter](https://github.com/bjonnh/jeolconverter)'s
`parseJEOL.js` (MIT licensed) -- read directly, not guessed at, since
its own field layout (including a real quirk: the header is always
big-endian, but the parameter/data sections that follow switch to
little-endian when the header's own `endian` byte says so) came from
actually tracing that implementation, not a spec document (none
appears to exist publicly).

**Verified two ways**, not just self-consistent:

1. Real data -- `test_data/jeol_1d/` (`proton.jdf`, `carbon.jdf`,
   `fluorine.jdf`, `phosphorus.jdf`), from
   [cheminfo/jeol-data-test](https://github.com/cheminfo/jeol-data-test)
   (MIT licensed, published there for testing purposes; original data
   credited to Guido Pauli's research group via Harvard Dataverse).
   Parsed header/parameter values match **exactly** what
   `jeolconverter`'s own test suite asserts for the same files (e.g.
   proton.jdf: nucleus `1H`, 32768 points, frequency
   `399782198.37825` Hz, offset `8.999999999999998` ppm) -- genuine
   ground-truth cross-validation against an independent
   implementation, not just internal consistency. These real files
   aren't wired into the pytest suite itself (matching this project's
   existing convention -- no other vendor's `test_data/` fixtures are
   either); `tests/test_jeol.py` uses a hand-built synthetic `.jdf`
   fixture instead, covering both endiannesses.
2. Ran the full pipeline (apodize/FFT/autophase/baseline-correct) on
   the real proton.jdf and confirmed the output is a plausible,
   mostly-correctly-phased real spectrum (positive on average across
   the actual signal region, not just noise) -- not merely "doesn't
   crash".

**Real files store a raw FID (time-domain), not a pre-computed
spectrum** -- every file checked has `dataUnits="Second"`, even though
the format can apparently also store an already-Fourier-transformed
spectrum (`dataUnits="Ppm"`, unhandled -- `read_1d` doesn't check for
this case at all yet, and will produce nonsense if ever handed one).
Given real files, this goes through the exact same
`processing.ft` apodize/FFT/phase/baseline pipeline X-Pulse and Varian
already use.

**Known gap, flagged not silently**: real files carry a
`FILTER_FACTOR` parameter (an integer point count) that's a
digital-filter group delay, analogous to Bruker's `GRPDLY`
(`nmrglue.bruker.remove_digital_filter`) -- not corrected for yet. The
FID's own magnitude peak lands a handful of points after index 0, not
at it, on every real file checked, and the default `phase_mode="apc0"`
(zero-order only) leaves at least one real peak visibly negative
(180° off) on the real proton.jdf even though most of the spectrum's
signal region phases correctly -- consistent with an uncorrected,
frequency-dependent (phase1-like) error from this, not a parsing bug
(the same peak's neighbours across the rest of the spectrum are
positive at the same global phase0). `phase_mode="joint"`
(`autophase01`) is available and may cope better, but its own
significance test didn't find a nonzero phase1 worth keeping on this
file either -- a real, proper correction (like Bruker's) is the actual
fix, not yet built.

**Not yet built**: 2D (COSY/HSQC/HMBC -- `read_1d` raises a clear
`ValueError` for these rather than silently misreading them; the
`jeolconverter` reference already documents the `Two_D` data-section
layout, so this is a well-scoped follow-up, not a mystery) and the
`dataUnits="Ppm"` (pre-FFT'd file) case mentioned above.

16 new tests (`tests/test_jeol.py`), 411 tests passing overall.

## Pin-bottom intensity zoom, Analysis-tab overflow (round 2), peak-pick tuning, per-slice autophase (2026-08-29)

User feedback, all done:

- ~~Scrolling on spectra should pin the bottom and move the top up and down~~ -- done, real redesign of the intensity-zoom branch of `_on_scroll` (added just last round). Was a cursor-centred zoom (`ymin`/`ymax` both move, scaled around the cursor's y-position) -- now pins whatever the bottom of the current view already is and only moves the top (`ax.set_ylim(ymin, ymin + (ymax - ymin) * factor)`), matching how an NMR viewer's own intensity slider behaves: the baseline shouldn't drift, only how tall a peak looks. Applied consistently to the main plot body and both contour projection panels -- `ax_f2` (top strip, normal orientation, baseline at the low/y end) pins the same way; `ax_f1` (left strip, plotted *negated* and rotated -- baseline ends up at its high/xmax end instead, see `plot_2d_contour`'s own `-f1_projection` plot call) pins `xmax` and moves `xmin` instead, the mirrored equivalent.
- ~~Analysis tab still overflows~~ -- **found the actual remaining cause**: last round's own new "Fit Curve (relaxation/diffusion)…" buttons (added to both the Analysis and Compare tabs) -- a `QPushButton`'s width doesn't shrink to fit a container the way a `QComboBox`'s can be told to (see below); a 33-character label was the single biggest content-width contributor added that round, self-inflicted. Shortened both to "Fit Curve…" (the full description already lives in the tooltip). Also fixed the underlying general issue while here: `QComboBox`'s default `AdjustToContentsOnFirstShow` policy sizes the *closed* box to whatever the longest item's text was the first time it was shown, and never shrinks back down even in a narrower container -- exactly the kind of fixed-content width `_scrollable`'s disabled horizontal scrollbar can't route around, so it surfaces as overflow instead. New `MainWindow._narrow_combo(combo, chars=14)` helper (`AdjustToMinimumContentsLengthWithIcon` + `setMinimumContentsLength`) applied to the Analysis tab's four combos (`fit_type_combo`, `relaxation_model_combo`, `value_source_combo`, `delay_source_combo` -- the last has "Relative gradient (valueslist)", the longest item in the whole tab) -- the closed box no longer forces extra width; the dropdown popup itself still shows every item at full length. Reorganized the Peaks button row into two rows while here (see below), which also helps since it was getting crowded.
- ~~Peak pick maybe too sensitive; need a clear-all button~~ -- done, both. Auto-Pick Peaks' default prominence raised from the backend's own 0.05 to 0.1 in the GUI, and made tunable via a new "Min. prominence" spin box next to it (confirmed via a two-peak synthetic test -- a weak second peak at 8% the strong one's height is found at 0.05 but correctly dropped as noise at the new 0.1 default). New "Clear All" button next to Add/Remove Peak, clears the whole peak table in one click.
- ~~Autophase button that can phase spectra in a pseudo-2d independently~~ -- done. New "Autophase (each slice independently)" button in the Processing tab's Dimension box (pseudo-2D only, next to the existing "Apply this phase to every slice" checkbox) -- genuinely different from that checkbox, which broadcasts *one* shared phase0/phase1 to every row: this finds and applies each slice's *own* zero-order phase independently (`processing.ft.apc0` per row, the same zero-order-only autophase already used elsewhere in the app, e.g. X-Pulse import's default `phase_mode="apc0"`), immediately, in place. Re-anchors the Phase controls afterward (`_capture_phase_reference`) so a subsequent manual drag starts from the new, autophased state rather than the pre-autophase snapshot. Coarser grid (0.5 deg vs. `apc0`'s own 0.1 deg default) to stay fast across a series with many slices.

390 tests passing.

## Echo-wrap fix, spectral-fit axis, FID display, Fit Curve buttons, value sources, auto-pick, left-panel sizing (2026-08-29)

User feedback, all done:

- ~~Echo top: still reversed unless cutting manually or using large lb~~ -- **the real remaining bug, found and fixed**. The previous fix (align_echo before truncate) only excluded the wrapped pre-echo tail when a `truncate` was *actually applied* long enough to stop before the wrap point -- with no `truncate` given at all (the default a user hits without manually setting TDeff), or with a `truncate`/file-recorded TDEFF that itself extends past the wrap point (TDEFF is set for the *original*, pre-roll acquisition timeline, not recomputed relative to where the echo top ends up after rolling), the wrap came right back in. Fixed: `read_1d` now always caps the effective truncation at the echo top's own wrap point (`len(fid) - ceil(echo_top)`) whenever `align_echo=True`, regardless of what `truncate` (or its absence) would otherwise imply -- only ever shortens, never lengthens, an explicit shorter `truncate`. This is what large `lb` was actually compensating for by decaying the wrapped tail toward zero rather than removing it -- no longer necessary. 3 new regression tests in `test_xpulse.py` covering: no truncate requested, an explicit truncate past the wrap point (gets capped), an explicit truncate short of it (honoured exactly).
- ~~1D spectral fits should use the same axis formalism as the normal spectrum~~ -- done. `show_slice_fit` was missing x-axis inversion, a nucleus-based label, and spine/tick styling entirely (just an unlabelled, un-inverted axis) -- now matches `show_spectrum`'s 1D view exactly, via a new optional `spectrum` param (the slice's own trace, `fit_result["slice"]`, for its nucleus/SFO1) reusing the same `find_gamma`/`NUC_label` lookup `plot_1d` does.
- ~~The FID should just be complex or real, not both~~ -- done. `show_fid` now plots one line (`mode="magnitude"` default, or `"real"`), not Real+Imag together -- a new "Preview:" combo in the FID processing box (Magnitude/Real) controls it, view-only, no reprocessing.
- ~~Still no Compare button~~ -- done, a real `QPushButton` this time (not just the right-click menu item added last round, which apparently wasn't discoverable enough) -- "Plot" and "Fit Curve (relaxation/diffusion)…" both added to the Compare tab itself, alongside the existing context-menu versions.
- ~~Fit Spectrum has a button but no button to fit the analysis (Fit Curve)~~ -- done, a real button in the Analysis tab too (`_on_fit_curve_button_clicked`, operates on the current selection; still also reachable via the Spectra-tab right-click for a row that isn't the current selection). Together with the Compare button above, this reverses the earlier "context-menu only" framing of "relaxation and pfg fitting should only be context-based options for pseudo-2ds" -- now read as "gated to when it's applicable," not "never a button."
- ~~Options for peak intensities vs. integrated regions~~ -- done. `NMR2D_fit_relaxation` gained `value_source="integral"` (default, unchanged behaviour) or `"amplitude"` (the peak's own fitted pseudo-Voigt/Gaussian/Lorentzian amplitude instead of its integrated area) -- a new "Value source" combo in the Analysis tab's Fit Curve settings box, siloed per spectrum like the other Fit Curve settings.
- ~~Some degree of auto peak picking should be an option~~ -- done. New `processing.fitting.auto_pick_peaks(spectrum, prominence=0.05, min_distance_fwhm=1.0, max_peaks=None)` (`scipy.signal.find_peaks`/`peak_widths` on the normalized spectrum, converted to peak-def dicts) -- a cleaned-up, decoupled version of the peak-finding core of `leonmr`'s `xf2_peak_pick` (TODO.md's own "nmrtools2/leonmr survey" item D), without that function's plotting/pandas/whole-pseudo-2D-stack-tracking coupling, none of which belongs at this level. New "Auto-Pick Peaks" button in the Analysis tab replaces the peak table with the detection results (slice 1, for a pseudo-2D series) -- still just initial guesses, refined by the fit like any manually-entered row.
- ~~Left tabs still not appropriately sized, content overflowing -- should be relative to the window, with an option to change~~ -- **best-effort fix, not visually confirmed**. Root cause: the left (Spectra/Processing/Analysis/Compare) panel had the `QSplitter` default stretch factor of 0 -- pinned at its initial ~300px forever however much the window grew (only the canvas panel had a stretch factor, so it alone got every pixel of a resize), proportionally shrinking the left panel to a sliver on a large/maximized window with nowhere for its own `QFormLayout` rows to go but overflow past the scrollbar-disabled `QScrollArea` viewport edge from an earlier round's fix. Fixed: both panels now get a share of any resize (`setStretchFactor(0, 1)` / `setStretchFactor(1, 3)` -- canvas still gets 3x more), a more generous initial split (340/760, was 300/800), and a usable minimum width floor (280px) so it can't be dragged/squeezed back into overflow either -- still fully draggable via the splitter handle either way ("an option to change"). Qt has no stretch-factor read-back API and this environment can't visually confirm real pixel behaviour or drive a real window `show()`/layout pass headlessly (this project has no established pattern for that either) -- if content still overflows after this, the next thing to check is whether some specific control (e.g. a `QComboBox` with long item text like "Relative gradient (valueslist)") has an outsized minimum width of its own that the panel-width fix alone can't route around.

390 tests passing.

## Canvas-tab layout, scroll-zoom redesign, contour tools, pseudo-2D-by-default (2026-08-29)

User feedback, all done:

- ~~FID tab should come before the Spectrum tab~~ -- done (separate small request, addressed first). Canvas tabs reordered to FID, Spectrum, Spectral Fit, Fit, Compare.
- ~~Right-side tabs: FID, Spectrum, Spectral Fit, then Fit (pseudo-2D only), then Compare (only when the left-hand Compare list is filled)~~ -- done. Reordered as asked; Fit and Compare now start hidden (`QTabWidget.setTabVisible`) and are kept in sync by two new helpers -- `_update_fit_tab_visibility(spectrum)` (called from `_refresh_selection_display`; also force-shown directly in `_run_curve_fit`, since a Compare-stack fit has no single "owning" spectrum for the selection-driven check to key off) and `_update_compare_tab_visibility()` (called from `_rebuild_compare_list` and everywhere `compare_list` gets cleared).
- ~~Left-side tabs: Spectra, Processing, Analysis, Compare~~ -- done (was Spectra, Processing, Compare, Analysis). ~~All of these should limit scrolling like in Compare~~ -- audited: `_scrollable()` (the shared `QScrollArea` wrapper every side-panel tab already goes through) already disables horizontal scrolling for all four uniformly; also found and fixed one more unwrapped dynamic-width `QLabel` (`_prompt_select_spectra`'s "Found N spectra under {base_dir}:" -- a folder path can be arbitrarily long) while auditing.
- ~~Scroll x-axis widens/narrows that axis only; scroll over the spectrum changes y-intensity only~~ -- done, real redesign of `MplCanvas._on_scroll`. Previously any scroll always zoomed x, and also zoomed y whenever the axes had visible y-ticks (so 1D/Compare/Fit views, which hide y-ticks, never got a y-zoom at all). Now: hovering the x-axis's own tick-label strip (detected via real pixel position + `Axes.get_window_extent()`, since `event.inaxes`/`xdata` are `None` there) zooms x only; hovering the plot body zooms y (intensity) only, regardless of tick visibility -- decoupled, matching common pro-plotting-software UX. Threaded through the whole test suite: rewrote `tests/test_gui_canvas.py`'s scroll tests with real pixel-position fake events (`_body_scroll_event`/`_x_strip_scroll_event`/`_outside_scroll_event`), since the old bare `SimpleNamespace(inaxes=..., xdata=...)` fakes had no pixel position to test the new hover logic with.
- ~~2D: x-axis behaviour the same; intensity independent for the projections~~ -- done. The main contour's x-axis strip now zooms x the same way; its body still adjusts the contour threshold (unchanged). `plot_2d_contour` now returns `fig, ax, proj_axes` (a new third value, `{"f1": Axes|None, "f2": Axes|None}`) instead of just `fig, ax` -- updated the two test call sites that unpacked it. `MplCanvas` tracks these as `self._proj_axes`, reset alongside `_contour_spectrum` everywhere that's cleared; `_on_scroll` checks them first, and a new `_zoom_projection_intensity` zooms just that panel's own intensity axis (`ax_f2`'s y, `ax_f1`'s x -- it's drawn rotated) independently of the main contour.
- ~~Right-click the contour to set the number of contours and contour scaling (start/step/max or power scaling)~~ -- done, new "Set Contour Levels…" dialog (offered whenever a contour is the current view). Fields: number of contours, start (lowest level, as a fraction of peak -- same value scroll-over-the-contour already adjusts), scaling (Power, with a tunable exponent -- higher bunches more levels near the noise floor -- or Linear, evenly spaced). `opennmr.plotting.twod._prepare_contour`/`plot_2d_contour` gained `level_mode`/`power` params (the old hardcoded `1.25` exponent is now `power`'s default, behaviour-preserving when not touched); `MplCanvas` gained `set_contour_levels(clevels, level_mode, power, factor=None)` plus a shared `_redraw_contour` helper (factored out of `_adjust_contour_levels`, which now just updates `_contour_factor` and calls it).
- ~~Missing a "grab slice" option for 2D data~~ -- done, new "Extract Slice…" right-click action (offered alongside Set Contour Levels). Prompts for axis (Row/fix-F1 or Column/fix-F2) and a 1-based index, extracts via `processing.slicing.get_trace`, and shows the resulting 1D trace in place of the contour -- the same "temporarily replace this canvas's view" pattern the Processing tab's pseudo-2D "Stacked slices" view already uses. Fills a real gap: previously the only way to isolate one row/column was the Processing tab's phase-preview slice spinner, which only appears for a pseudo-2D spectrum with retained imaginary data (`data_imag is not None`) -- nothing existed for a conventional 2D spectrum, or a pseudo-2D one without retained imaginary data. The dialog-construction/`exec()` flow itself isn't unit-tested (matches this project's existing convention for small modal dialogs, e.g. `_prompt_select_spectra` -- QDialog/QMenu modal loops aren't reliably interceptable headlessly); the reusable logic (`set_contour_levels`, `extract_slice`) is factored out and tested directly instead.
- ~~Compare context needs a Plot button~~ -- done, added to `_on_compare_list_context_menu` (offered alongside the existing Fit Curve action, same gating: at least one checked entry). Calls `_refresh_compare_view()` directly -- useful since the Compare canvas tab now starts hidden and only shows once something's plotted; toggling a checkbox already auto-refreshes/auto-switches too, this is just an explicit, discoverable way to do the same thing on demand.
- ~~Compare needs to be more automatic for any 2D data when pseudo-2D is used~~ -- done, as a direct consequence of the pseudo-2D-default change below: `_on_list_selection_changed`'s existing auto-populate-from-pseudo-2D check (`spectrum.ndim == 2 and spectrum.is_pseudo_2d`) already fires for any single-selected 2D spectrum that's pseudo-2D -- and since pseudo-2D is now the default treatment for *every* 2D spectrum regardless of vendor auto-detection, this now fires far more broadly than before, with no gating-logic change needed.
- ~~Pseudo-2D should be the default for 2D data; add a Process Indirect Dimension button (not a checkbox)~~ -- done, a real default-behaviour change, not just a relabel. `_load` now checks every freshly-loaded 2D spectrum and, if it didn't already come back pseudo-2D from the vendor's own auto-detection, immediately reloads it with `pseudo_2d=True` -- overriding that auto-detection rather than trusting it, since *under*-detecting pseudo-2D is the known failure mode on real data (see the 2026-08-28 entry below: real Bruker T2/PFG datasets that are pseudo-2D-by-content auto-detect as conventional 2D). The old `pseudo_2d_checkbox` (`QCheckBox`, "Force pseudo-2D") is now `process_indirect_button` (`QPushButton`, checkable, "Process Indirect Dimension") with inverted semantics -- checked = process F1 as a real frequency dimension (`pseudo_2d=False`, a genuine 2D-correlation experiment), unchecked (default) = pseudo-2D -- reprocessing immediately on toggle, same "immediate action + persistent state" pattern as Align Echo. Real ripple effect across the test suite (18 tests updated: renamed/rewrote 4 that tested the old checkbox directly, removed 18 now-redundant `.setChecked(True)` calls that used to opt *into* pseudo-2D and are now the default, added explicit `process_indirect_button.setChecked(True)` opt-outs to the handful of tests that specifically exercise conventional-2D behaviour) -- all intentional, tracked one by one against each test's actual purpose, not blanket-suppressed.

**Not independently verified against real conventional-2D (non-pseudo) data yet** -- Process Indirect Dimension reuses the exact same `session.reload(index, pseudo_2d=False)` path the old checkbox already used (itself verified earlier in the project, see the 2D/pseudo-2D processing section below), so it should be equivalent, but the pseudo-2D-by-default change itself is new and hasn't been smoke-tested end-to-end against a real genuine-2D-correlation dataset (if the user has one) -- worth doing before fully trusting it in a real workflow with such data.

376 tests passing.

## Compare-tab width, 1D-fit clarity/results, MRI echo time-reversal fix (2026-08-29)

User feedback, all done:

- ~~Compare tab text makes the tab scrollable/too wide~~ -- **real bug, fixed**. The intro `QLabel` in `_build_compare_tab` was missing `.setWordWrap(True)` (every other status/instruction label in the file already had it) -- its unwrapped single-line width exceeded the sidebar, and since `_scrollable()`'s `QScrollArea` had no horizontal-scroll policy set, this silently became a horizontal scrollbar instead of a taller window. Fixed both ways: added the missing `setWordWrap(True)`, and set `scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)` in `_scrollable()` as a systemic guard, so any future unwrapped wide content surfaces as an obvious wrapping bug instead of a scrollbar.
- ~~1D spectrum fitting still not a clear option~~ -- done. The Analysis tab's pseudo-2D-specific "Fit" box (Fit type/Model/Values, all relaxation/diffusion concepts) used to sit directly between Peaks and the "Fit Spectrum" button, making it look like configuration Fit Spectrum needed. Reordered so Fit Spectrum (+ its results/View-slice-fit spinner) comes immediately after Peaks; the settings box now comes after, renamed **"Fit Curve settings (relaxation / diffusion -- step 2)"** with a tooltip clarifying it configures the right-click Fit Curve action, not the button above it.
- ~~Fit results should show constituent peaks, not just the total fit, plus fit parameters~~ -- done. `MplCanvas.show_slice_fit` gained `components`/`peak_labels` params, plotting each peak's own curve (labelled `"Peak N (shift ppm)"`) and the baseline alongside Data/"Fit (total)". A new `spectral_fit_params_label` (Analysis tab, below View slice fit) shows each peak's fitted shift/FWHM/amplitude/eta ± 1-sigma error, updated by a shared `_display_slice_fit` helper (both `_on_view_slice_fit_changed` and `_silo_other_canvas_tabs`' reselection restore now go through it, rather than duplicating the plot/label logic).
- ~~Starting values should update with fit results~~ -- done. `_update_peak_table_from_fit` writes the first fitted slice's shift/FWHM/eta back into `peak_table`'s rows right after a successful Fit Spectrum, so a re-fit (or a later Fit Curve, which reuses these same `peak_defs`) starts from the converged position rather than the original guess.
- Curve-fit gating ("relaxation and pfg fitting should only be context-based options for pseudo-2ds") -- read as re-confirming the already-correct, just-built gating (context-menu only, never a button, never for a bare 1D spectrum), not as walking back the earlier explicit "or stacks of 1D data" instruction -- so the Compare-tab stack Fit Curve feature (`_on_fit_curve_compare_stack_action`) was kept, not removed. Flagging this reading here in case it's wrong.
- ~~MRI echo top found correctly but FID still "looks like a time-reversal"~~ -- **real bug, found and fixed**. `io/xpulse.py::read_1d` applied `truncate` (TDEFF) *before* `fid_shift`/`align_echo`. `align_echo` rolls the echo top to index 0, which wraps the pre-echo portion (dead time + the rise up to the echo top -- roughly a mirror image of the post-echo decay, for a symmetric full echo) around to the tail of the array; truncating first, against the original unrolled timeline, does nothing to remove that wrapped portion. Only a truncate applied to the *post-roll* array, short enough to stop before the wrap point, actually discards it -- otherwise the "aligned" FID has a mirror-image copy of its own decay spliced onto its tail, exactly the "time-reversal" symptom. Found by comparing against `leonmr/xpulse_imaging.py::xpulse_image_import`, which rolls the full FID first and only then hands it to `process_fid`'s own TDEFF truncation -- confirmed via the OneDrive gradient-calibration notebook (`X-pulse_gradient_calibration_new.ipynb`) that this is exactly the "beautiful" prior pipeline being referenced, not a different/undiscovered one. Fixed by reordering `read_1d` to roll first, truncate second (`read_array`, the pseudo-2D/arrayed X-Pulse reader, delegates straight to `read_1d` per step, so it's fixed too); added `test_xpulse.py::test_align_echo_then_truncate_discards_wrapped_pre_echo_tail`, an exact-permutation regression test that fails under the old order. Varian's importer has no `truncate`/TDEFF concept, so it wasn't affected.

366 tests passing.

## Analysis-workflow restructure + Varian/SR/echo-top fixes + drag-and-drop (2026-08-29)

User feedback, all done:

- ~~Spectral fitting should be possible for 1D~~ -- done. The "Fit" button is now **"Fit Spectrum"** (`_on_fit_spectrum_clicked`), and works directly on a plain 1D spectrum (fits it as a single "slice") as well as every slice of a pseudo-2D series, unlike before, where peak fitting only ever ran as a side effect of the combined pseudo-2D-only Fit button.
- ~~Workflow: spectral fitting before curve fitting~~ -- done, and now a real two-step gate, not just a suggestion: **Fit Spectrum is step 1** (peak fit, a button, works on 1D or pseudo-2D); **Fit Curve is step 2** (relaxation/diffusion fit) and reuses step 1's stored per-slice results when available, only re-running the spectral fit itself if step 1 hasn't been done yet.
- ~~Curve fitting should be a context menu, pseudo-2D/1D-stacks only~~ -- done. "Fit Curve (relaxation/diffusion)…" is a right-click item on a pseudo-2D row in the Spectra tab's list (`_on_fit_curve_pseudo2d_action`), or on the Compare tab's own list once it has at least one checked entry (`_on_fit_curve_compare_stack_action`) -- never a permanent button, and never offered for a plain 1D spectrum or an empty Compare list.
- ~~Compare should auto-fill with a pseudo-2D's own slices, in order~~ -- done. Selecting a single pseudo-2D spectrum (not multi-select) now populates the Compare list with every slice, checked, in order (`_populate_compare_list_from_pseudo2d`) -- the Compare list's entries are now `(session_index, slice_index | None)` tuples rather than plain ints, resolved on demand via `_resolve_compare_entry` (`get_trace` for a slice entry, the spectrum itself for a whole-spectrum entry). This is also what makes "Fit Curve across a stack of 1D data" possible at all -- a pseudo-2D's own slices and a manually-multi-selected set of separate files now go through the exact same Compare-list/curve-fit machinery.
- ~~Varian: missing all FID processing except SR and phasing~~ -- **real bug, fixed**. `_fid_reprocessable` only recognised `"xpulse"`/`"bruker"` -- `"varian"` was missing entirely, so the whole FID-processing box (window/zero-fill/fid-shift/align-echo/linear-predict) was hidden for every Varian spectrum even though `io.varian.read_1d` already supported all of it. Also found and fixed a second bug surfaced only once the box became visible: `io.varian.read_2d` didn't accept `fid_shift`/`align_echo`/`linear_predict_*` at all (the GUI always sends them) -- would have raised `TypeError` the first time anyone actually clicked Reprocess on a Varian 2D spectrum. Added per-row application of all three, matching `read_1d`'s own ordering.
- ~~SR should be a persistent value vs. SFO1, not a resetting delta~~ -- done, real redesign. `sr_shift_spin` is now the spectrum's own current SR (Hz, siloed per spectrum via `_restore_sr_control`), not a one-shot "Add" box that reset to 0 after every apply. Re-applying the same value is now idempotent (`_capture_original_axis` stashes the as-loaded hz/ppm once per spectrum; every apply recomputes from that fixed baseline, never compounds). Also fixed: applying SR used to silently reset the plot's axis-unit view back to ppm (a full `show_spectrum()` redraw always did that) -- now saves/restores the canvas's own `axis_unit` (a new public property on `MplCanvas`) around the redraw.
- ~~Echo top doesn't seem to work; make it a push button~~ -- done. `align_echo_checkbox` is now `align_echo_button`, a checkable push button that **immediately reprocesses and switches to the FID tab** when toggled (not just remembering a setting for a later separate "Reprocess" click) -- shows a dashed green "Echo top" marker (`find_echo_top`'s result) on the FID plot, which should land right at t=0 once actually aligned, so the effect is now directly visible/verifiable rather than silent.
- ~~Drag-and-drop folder import~~ -- done. `MainWindow.setAcceptDrops(True)` + `dragEnterEvent`/`dropEvent`, delegating to `_handle_dropped_paths` (kept separate from the Qt event glue so it's testable without constructing a real `QDropEvent`) -- a dropped folder goes through the exact same scan/prompt/load path as the Open… dialog (`_open_directory`, factored out of `_on_open_clicked` for this); a dropped single recognised file (e.g. one X-Pulse/Varian file) loads directly; anything unrecognised is silently skipped.

Also: the app icon/logo went through several rounds as the user iterated on the design (`src/opennmr/gui/resources/logo.png`, and a banner version at `assets/openNMR_logo_banner.png` now used in `README.md`) -- ended up PNG-only, not the originally-added SVG (`assets/openNMR_logo.svg`, kept as a design source but not loaded by any code), because the SVG renders its "openNMR" text in a specific Google font (Bai Jamjuree) with no guarantee that font is installed wherever the app runs or gets built, risking a silently wrong-looking (fallback-font) icon both live and in the packaged app.

364 tests passing.

## nmrtools2/leonmr survey -- candidate features not yet in opennmr (2026-08-29)

Per explicit instruction once the rest of this session's queue was done: "look through all the functions across nmrtools and pyleo [nmrtools2 and leonmr] to start adding or at least preparing features... anything ideated can be added first to the todo list and discussed." **Nothing below is implemented** -- this is a survey (via a dedicated read-through of `nmrtools2/src/bignmrtools/*.py` and `leonmr/*.py`/`xpulse_processing.py`/`xpulse_imaging.py`/`bloch_sim.py`/`bruker_diffprof.py`, cross-checked against what opennmr already has) for discussion, not a to-do list ready to just execute -- several of these are new capability *classes* (imaging/spatial profiles, pulse simulation) that would need their own design conversation, not just a port.

**A. Composable plotting functions** -- the biggest concrete gap: opennmr's fitting/relaxation backends produce numbers but have no matching "look at my fit" plot helpers (the GUI's own Analysis tab currently builds its plots by hand in `main_window.py` rather than calling a shared plotting function).
- `NMR1dplot_with_fit` (`bignmrtools/fitting.py:774`) -- 1D spectrum + fitted model overlay, optional per-peak components in gray.
- `NMR2D_plot_slices` (`bignmrtools/two_d.py:1561`) -- stack-plots extracted 2D rows/columns/both with optional fit-curve/component overlays and nQ crossing markers.
- `NMR2D_summary_plot` (`bignmrtools/two_d.py:2018`) -- one figure combining a 2D contour, extracted-slice stack, and integral-vs-position panels, with slice-position lines drawn on the contour. A genuinely GUI-worthy "one figure, whole story" export.
- `NMR2D_plot_integrals` (`bignmrtools/fitting.py:1391`) -- fitted peak integrals across a slice series, with error bars and multi-dataset overlay.
- `NMR2D_plot_relaxation` (`bignmrtools/relaxation.py:754`) -- plots one or more `fit_relaxation`/`NMR2D_fit_relaxation` results (data + curve, log-axis toggle, on-curve rate annotation) -- opennmr has a text report but nothing that draws this outside the GUI's own bespoke `show_relaxation_fit`.

**B. One-shot workflow orchestrators** -- map fairly directly onto a GUI "Analyze" button, if a batch/scripting entry point is ever wanted alongside the interactive GUI flow.
- `NMRFitWorkflow` (`bignmrtools/fitting.py:928`) -- import -> baseline estimate -> peak fit -> plot -> text report -> optional CSV export, one call.
- `NMR2D_workflow` (`bignmrtools/fitting.py:1634`) -- import -> extract rows/columns -> fit -> integrate -> plot slices -> plot integrals, one call.
- `NMR2D_relaxation_workflow` (`bignmrtools/relaxation.py:944`) -- the full T1/T2/T1rho/diffusion pipeline in one call; notably supports a `regions`-only mode that integrates raw regions instead of peak-fitting, for when peaks aren't resolved enough to model -- opennmr's Analysis tab currently has no non-peak-fit fallback at all.

**C. Imaging / spatial-profile analysis -- a capability class opennmr doesn't have at all yet**, distinct from PFG diffusion-attenuation fitting (which opennmr does have):
- `xpulse_image_import` (`leonmr/xpulse_imaging.py:293`) -- imports X-Pulse 1D gradient-echo/UTE imaging profiles and converts the frequency axis to physical position (mm) via `GRADIENT_CALIBRATION` (per-probe/axis, auto-selected from the file's own logged `Probe`/`GradAxis`) -- this is the same calibration table already flagged as needed for item 3 (mm axis "imaging" mode) in the GUI-polish-batch-3 section above; a real design/port target once that item is picked up, not just a one-off function.
- `diffprof_lwg_import` (`leonmr/bruker_diffprof.py:176`) -- the Bruker analog: raw-FID import of `diffprof.lwg` gradient-echo profiles (1D or pseudo-2D diffusion-weighted-T2 array) -> position (mm) vs echo-delay, with digital-filter removal/echo-alignment/autophasing/per-row baseline correction. Supersedes leonmr.py's older processed-`pdata`-only `diffprof`/`diffprof_2d`.
- Time-resolved uptake/exchange profile tracking, `diffprof_exchange` (`leonmr.py:2564`) -- re-normalizes an imaging-profile series against sample-boundary position and time-stamps each row, for monitoring solvent/gas uptake into a sample over hours. Flagged as "the idea is real, the code needs a rewrite" by the survey, not a clean port candidate.

**D. Lightweight peak-picking / intensity extraction**
- `xf2_peak_pick` (`leonmr.py:1580`) -- automatic peak detection (`scipy.signal.find_peaks`) across a pseudo-2D stack, tracking peak height *or* trapezoidal-integral over a ppm range across all slices at once -- a much lighter-weight alternative to opennmr's nonlinear `NMR2D_fit_slices` for a quick relaxation/diffusion trend without defining peak models first. Natural companion to item 8's tighter fit-position-bounds work above (a fast pre-check before committing to a full fit).

**E. Experiment-planning / simulation** -- not data analysis, but a natural companion feature in the same GUI:
- `sim_diffusion` (`leonmr.py:1284`) -- plots expected PFG attenuation vs gradient strength for a nucleus/delta/DELTA (sweeping D or for one supplied value) -- a pre-experiment planning aid for choosing gradient/delay settings, distinct from post-hoc fitting.
- Bloch simulator for shaped/selective pulses (`leonmr/bloch_sim.py`: `simulate_pulse`/`simulate_selective_pulse`/`generate_shape`/`plot_pulse_simulation`) -- reproduces the X-Pulse selective-pulse sequence family's own on-the-fly shape synthesis and plots the resulting excitation profile (Mx/My/Mz vs offset) plus the pulse shape -- useful as a pulse-design/QC check before running a sample.

**F. Misc file-format readers**
- `dmfit` (`leonmr.py:373`) -- reads/plots DMFit (solid-state NMR lineshape deconvolution) `.asc` exports: spectrum, overall model, and individual fitted lines. A distinct external-tool import format, not covered by any current opennmr reader.
- `xrdml_import` (`leonmr.py:3253`) -- PANalytical `.xrdml` XRD files (2theta/Q/intensity/wavelength). Off-nucleus (XRD, not NMR) -- same lab-workflow niche (materials characterization) but arguably outside "NMR suite" scope; flagged as lowest priority of this whole list.

**Checked and excluded as already covered** (either already ported, or superseded by a more general opennmr equivalent): `autophase01`, `linear_predict`, `NMR2D_integrate_slices`, S/N-gated peak fitting (`sn_threshold`/`noise_region` in `NMR2D_fit_slices`), `find_gamma`/`NUC_label`, `combine_spectra`, `NMR2D_multi_plot`/`NMR2D_overlay_plot`, and the older lmfit-based `T1`/`T2_Fit`/`DiffFit`.

## Installer / update framework (2026-08-29)

~~Build the framework~~ -- done, per explicit request ("start to build some kind of framework for installation that is system agnostic... a good way to implement updates. This doesn't need to be used just yet but it needs to be built in general"). Two independent pieces, both in `packaging/README.md` for the full writeup:

- **`packaging/build.py`** -- one script, run identically on any OS, produces a standalone app via PyInstaller (`pip install -e ".[build]"` first). **Actually built and launched successfully on macOS** (not just written and assumed to work) -- found and worked around two real environment issues along the way (an obsolete `pathlib` backport package that PyInstaller refuses to run alongside; PySide6+PyQt6 both installed, which PyInstaller also refuses to freeze together -- both documented in `packaging/README.md` in case they resurface on another machine). Also found and fixed a real bug during verification: pointing PyInstaller directly at `gui/app.py` builds without error but crashes at launch (`ImportError: attempted relative import with no known parent package`, since PyInstaller treats its target script as top-level, outside the `opennmr` package) -- fixed via `packaging/entrypoint.py`, a separate top-level script that imports `opennmr` as an installed package instead. Windows/Linux not yet tried.
- **`src/opennmr/update.py`** -- `check_for_update(manifest_url)` fetches a small JSON version manifest (`packaging/version_manifest.example.json`) and reports whether a newer version exists for the current platform. Deliberately read-only (never downloads/installs anything) -- see the README for why. 12 tests in `tests/test_update.py`. Not wired into the GUI (no menu item, no startup check) -- that's the natural next step once there's an actual manifest hosted somewhere to point it at.
- Still not built: native installers (DMG/MSI/AppImage) wrapping the raw PyInstaller output, code signing/notarization, CI automation, and the GUI-side "an update is available" affordance once update-checking is actually wired up.

## Beautification / workflows (open, 2026-08-29)

User: "consider proper workflows that people might want" + a general "beautification" pass, after the fix list and Varian importer above. Both given without specifics. ~~Separate canvas/tab for spectral (single-slice peak) fits~~ -- done (the one concrete piece of this given: `spectral_fit_canvas`, a 5th canvas tab, so viewing a slice's spectral fit no longer overwrites the main Spectrum canvas -- siloed per spectrum the same way as the Fit tab). Everything else here is still genuinely open -- rather than guess broadly at "beautification" (this codebase's actual feedback rounds have been very specific: "make the buttons bigger," not "improve aesthetics" in general -- guessing risks work that doesn't match what's actually wanted), this needs the user's own examples of rough edges or workflows they'd reach for. Worth asking about directly next time, rather than assuming.

## Varian/Agilent importer (2026-08-29)

~~Add a Varian vendor reader~~ -- done, `io/varian.py`, registered as `"varian"` in `io/base.py` alongside `bruker`/`xpulse`. Verified end-to-end against real `test_data/varian_1d`/`varian_2d` (a `.fid` directory of `fid`+`procpar`, VnmrJ's standard layout) -- `MainWindow._scan_for_spectra` needed **no changes** to pick these up, same as the X-Pulse array-folder loader above (the registry-based `detect_vendor` dispatch is genuinely vendor-agnostic). 17 new tests in `tests/test_varian.py`, using synthetic fixtures written via `nmrglue.varian.write` (round-tripping through nmrglue's own binary writer/reader rather than hand-fabricating Varian's block-header format, matching this project's `test_data/`-is-gitignored convention -- no test depends on the real local files).

- Like X-Pulse, Varian hands back only a raw FID (no vendor-processed fallback) -- `read_1d`/`read_2d` share the same `processing.ft` apodize/zero-fill/FFT/phase/baseline pipeline as the other two vendors. Raw binary read via `nmrglue.varian.read` (same reasoning as Bruker's use of nmrglue: reproducing Varian's block-header format and hypercomplex-2D interleaving by hand is exactly the kind of DSP code that's easy to get subtly wrong).
- **Read with real caution before trusting quantitative results from either path yet:**
  - `read_1d`'s ppm axis treats Varian's `tof` the same way `io.xpulse` treats X-Pulse's `$O1` (both = transmitter offset frequency) for consistency with this project's other two readers -- but this is **not cross-checked against an independently known chemical shift**, only confirmed to produce a single, plausible-looking peak on real data. nmrglue's own `guess_udic`/`unit_conversion` heuristics were tried first and rejected: `guess_udic` returned unusable placeholder values (999.99 for every field) on the real 1D test file, and `unit_conversion` (with `car=tof`) disagrees with the `io.xpulse`-style formula by a full sign flip on `tof` (peaks land ~112 ppm apart on the same real data) -- with no independent reference to arbitrate, the project-consistent choice was kept, but this needs a known-shift sample (or the original acquisition operator) to actually confirm.
  - `read_2d`'s States-hypercomplex F1 assembly (`fid[:, 0, :] + 1j*fid[:, 1, :]`, real/imaginary phase pair per `ni` increment) is structurally confirmed against real `test_data/varian_2d` (`nblocks == 2*ni`, reshapes cleanly) but **not validated against a known-good 2D lineshape** -- the real/imaginary-pairing order and which of the two phases is "real" hasn't been independently confirmed, only that it's *a* self-consistent standard-States interpretation. Falls back to reading every trace as its own row (no complex F1) if a dataset's `ntraces`/`ni` don't match the `2*ni` pattern this was built against.
  - Varian's `tn` procpar field is symbol-then-number ("Li7", "H1") -- normalized to this project's number-then-symbol convention ("7Li", "1H", matching `core.registry.find_gamma`) via `_normalize_nucleus`, a small regex swap, not flagged as risky.
- Not built: `f1p`/`f2p` ppm-crop kwargs (Bruker's `read_1d`/`read_2d` have them; skipped for Varian since nothing in the GUI's default load path currently uses them for any vendor); a Varian equivalent of Bruker's `read_1d_processed`/`read_2d_processed` fallback (Varian's raw FID is effectively always present, unlike a TopSpin dataset that might ship only `pdata`); true (non-pseudo) 2D F1 frequency support (`pseudo_2d=False` case) -- `read_2d` currently always treats F1 as an index dimension, no second FFT.

## GUI polish batch 3 (2026-08-29, 15-item feedback round)

User gave a 15-item feedback list, said "note this for later," then immediately "get started on those changes now," then (mid-implementation) "just get on with everything" -- worked straight through in the order below rather than stopping between items. 11 of 15 done; item 10 explicitly needs the user's own input before implementing, not a guess; items 2/3/7(partial)/9(now done) involved a new "nav bar"/imaging concept that's only partially designed.

1. ~~SR: single Hz value to add, maybe via right-click~~ -- done, both. Processing tab's Reference box is now one `sr_shift_spin` (Hz, additive) + Apply, replacing the old two-ppm-field "current/target" delta calculation. Also added to the Spectrum canvas's right-click menu ("Set Reference Shift (Hz)…", `MplCanvas.referenceShiftRequested` signal + `MainWindow._on_reference_shift_from_plot`), a `QInputDialog` prompt pre-filled with the running total (`metadata.raw["sr_shift_hz"]`).
2. **Pseudo-2D slice alignment, "nav bar" concept -- not built.** Vague as given; no design decided (what "alignment" means here -- shift correction between slices? a navigation panel for stepping through/scrubbing slices? both?). Needs a follow-up conversation before implementing, same as item 10 below -- don't guess at UI here.
   - **2026-08-29 follow-up from the user, still at the thinking-out-loud stage (not a spec):** the nav bar might be less about slice alignment specifically and more a way to *containerize different analysis types* (imaging, "imT2", etc. -- i.e. each analysis type gets its own configured workflow/tab rather than the current one-size-fits-all Analysis tab). Alongside that: "it might be good to have some way to grab values from the metadata in order to do the calculations. Maybe some kind of setup thing that can be saved locally per experiment" -- with a concrete example: a pseudo-2D `diffprof.lwg` (see the survey's `diffprof_lwg_import` entry below) needs `L6` (a Bruker acqus loop-counter parameter) *and* the slice number together to compute the actual per-slice value to fit against -- i.e. a per-analysis-type, savable "recipe" that maps `metadata.raw`/acqus fields (+ slice index) to the independent-variable calculation, generalizing what `_resolve_independent_values`'s current fixed set of sources (Auto/Manual/Start+step/vclist x factor/relative gradient) already does ad hoc, into something user-configurable and reusable per experiment rather than hardcoded per source type. Explicitly not ready to build -- the user said they need to think about it further.
3. **mm axis "imaging" mode + max-gradient-strength input -- not built.** Still blocked on the same thing flagged under "PFG / diffusion fitting" below: no gradient calibration is recorded in X-Pulse's acquired data (`G1` is a normalised -1..1 drive fraction only), so this needs either a user-entered max-gradient value (per the item's own suggestion) or a ported `xpulse_imaging.GRADIENT_CALIBRATION` table. A single global "max gradient strength" setting (item 7 below wants the same input) is the more promising direction -- do both together.
4. ~~Remove FID checkbox, click FID tab directly, TDeff + dashed marker~~ -- done. `show_fid_checkbox` removed entirely; `canvas_tabs.currentChanged` (`_on_canvas_tab_changed`) shows/refreshes the FID preview when the FID tab itself becomes current (clearing it if unavailable), instead of a separate opt-in toggle. Added a `tdeff_spin` (X-Pulse only, prefilled from the file's own recorded `TDEFF` when present) wired to a new `truncate` kwarg on Reprocess, and `MplCanvas.show_fid` draws a dashed red "TDeff cutoff" line at that point -- live-previewed as the spinbox changes, even before Reprocess is clicked. Reprocess now keeps both the Spectrum canvas (always redrawn) and the FID canvas (redrawn in place if it's the active tab) in sync, instead of only whichever the checkbox pointed at.
5. ~~Fit tab / right-side tabs not siloed per spectrum~~ -- **real bug, fixed**. Only the Analysis tab's own widgets were siloed before; `fid_canvas`/`compare_canvas`/`fit_canvas` weren't reset on reselection, so switching spectra could leave a previous experiment's relaxation-fit plot showing. `_on_fit_relaxation_clicked` now also stashes the plotted `results`/x-label into `metadata.raw["relaxation_fit_results"]`; `_silo_other_canvas_tabs` (called from every `_draw_selected_spectrum`) restores that per-spectrum fit plot (or clears the tab if the newly selected spectrum has none) and always clears the Compare tab (session-level, not per-spectrum -- see item 14).
6. ~~Start/End/Step -> 2 fields, count is already known~~ -- done. Dropped the End spinbox; "Start + step" now generates exactly `n_slices` values (`start + step * arange(n_slices)`), removing a real off-by-one/roundoff risk the old `np.arange(start, end+step/2, step)` had if End didn't land exactly on a slice boundary.
7. **Relative-gradient b-value source -- done; max-gradient-strength -> true b-value conversion NOT done.** Analysis tab's Values-source combo gained "Relative gradient (valueslist)", reading the new `metadata.raw["gradient_relative"]` (see item 9) directly as fit x-values -- verified against real `test_data/xpulse_PFG/4`. What's still missing: converting that relative (0..1) fraction into an actual b-value (s/m^2) via Stejskal-Tanner needs `b = (gamma*g*delta)^2*(DELTA-delta/3)` with `g = relative * max_gradient_strength` -- confirmed `delta`/`DELTA` **are** recorded per-file (`exp_params["delta"]`/`["DELTA"]`, in µs, verified on real data), but gamma's unit convention (`core.registry.find_gamma` returns Hz/T; the physics formula needs rad/s/T, i.e. a `2*pi` factor easy to get wrong) and gradient units (X-Pulse likely G/cm like Bruker, needs confirming) haven't been verified against a working reference -- `find_gamma`'s Hz/T vs. `read_diff_xml`'s docstring claiming "10^7 rad/s/T" for the same function's output is itself a discrepancy worth resolving first. Deliberately not hand-rolling this without checking against `leonmr`'s gradient-calibration notebook (`~/Library/CloudStorage/.../X-pulse_gradient_calibration_new.ipynb`) or the user directly -- getting a diffusion coefficient silently wrong by a factor of `2*pi` (or 100x on a unit mismatch) on real data is a worse outcome than leaving this as relative-gradient-only for now. The "maybe a global nav-bar setting" framing from the request also isn't resolved -- for now this would be a per-fit spinbox next to the Values-source combo, not global.
8. ~~Tighter default spectral-fit position constraints, with options~~ -- done. Position bounds used to default to fully unconstrained (`-inf, inf`), meaning a fit could slide a peak onto a neighbour. `_read_peak_defs` now sets `bounds={"shift": (shift - tol*fwhm, shift + tol*fwhm)}` for every peak, `tol` from a new `position_tolerance_spin` (default 0.5x each peak's own FWHM, range 0-20, siloed per spectrum) -- proportional to each peak's own width rather than a fixed ppm window, so narrow and broad peaks each get a sensibly-scaled leash.
9. ~~X-Pulse arrayed experiments (PFG etc.) read as one experiment~~ -- done, and this also resolves the "Still not built (X-Pulse PFG)" loader gap noted under "PFG / diffusion fitting" below. New `io.xpulse.read_array(path, **read_1d_kwargs)` reads every `arrayN/` subfolder's single `.jdx` (ignoring each one's own `raw/` per-scan-repeat files) as one pseudo-2D `NMRSpectrum`, one row per step in order, stashing `parameters/G1_valueslist.txt`'s relative gradient values (if present) in `metadata.raw["gradient_relative"]` for item 7 above. `xpulse._detect`/the registered reader now dispatch on `os.path.isdir` (array-folder vs. single file), so `MainWindow._scan_for_spectra` -- which already claims a directory as one match the moment any vendor's `detect()` recognises it, without descending further -- needed **no changes at all** to correctly offer `test_data/xpulse_PFG/4` as one match instead of 10. Verified end-to-end against that real dataset (`data.shape == (10, 8192)`, `gradient_relative == [0.05, 0.15, ..., 0.95]`).
10. **FID-shift redesign -- deliberately not implemented, needs a conversation.** User: "It shouldn't always be a rolling thing. sometimes we just need to cut the initial points." Currently `fid_shift_spin` always does a circular roll (`processing.ft.roll_fid`, Fourier-shift-theorem based, fractional-sample capable) via `io.*.read_1d`'s `fid_shift` kwarg. A destructive "cut N initial points" mode is a different operation (would need e.g. a mode toggle or a second control, and interacts with `fid_original_length`/zero-fill bookkeeping differently than a roll does) -- worth discussing when both are actually needed vs. one replacing the other, rather than guessing at the UI.
11. ~~FID processing controls shouldn't be hideable~~ -- done. Removed the collapsible `toggle_button`/`content`-widget pattern (a checkable `QPushButton`, not `QGroupBox.setCheckable`, which was itself a deliberate earlier workaround for a Qt gotcha -- see "Processing" section below) entirely; `_build_fid_processing_box`'s controls now attach straight to the box's own `QFormLayout` and are always visible whenever the box itself is (i.e. whenever the spectrum is FID-reprocessable at all).
12. ~~Bigger phasing buttons~~ -- done. `DragHandle` grew from 26x24 to 44x40px, glyph font bumped to 16pt to match.
13. ~~GUI window sometimes grows taller for no reason (bug)~~ -- **real bug, root-caused and fixed**. The side tabs (Processing/Analysis especially) show/hide whole groups of controls per selected spectrum (`fid_processing_box`, `dimension_box`, `start_end_step_row`, ...); each becoming-visible grows that tab's layout minimum-size hint, and Qt only ever grows a top-level window to fit a bigger minimum -- it never auto-shrinks back down once a narrower spectrum is selected, so the window ratcheted taller over a session and never came back. Fixed by wrapping each side-tab's content in its own `QScrollArea` (`MainWindow._scrollable`, `setWidgetResizable(True)`) -- content-height growth is now absorbed by internal scrolling instead of propagating up to the window's own size hint.
14. ~~Compare tab needs its own select/deselect/reorder context~~ -- done, real redesign. Added `compare_list` (a checkable `QListWidget` inside the Compare tab itself, populated from the Spectra tab's multi-selection but independently editable): uncheck an item to drop it from the plot without losing the underlying multi-select, and Move Up/Down now reorders *this list's own* local plot order rather than (as the old buttons did) the master session order via whatever row happened to be `spectrum_list.currentRow()`. Repopulating on a Spectra-tab selection change preserves check-state/order for rows still present (so ctrl-clicking one more spectrum in doesn't wipe out prior deselections/reordering) and drops rows no longer selected; removing a spectrum via the Spectra tab's Remove button rebuilds the list from scratch rather than risk stale indices (a spectrum removal doesn't always fire `itemSelectionChanged`, if the *set* of selected item objects doesn't itself change even though their row numbers shift). The old `_on_move_spectrum` (global list reordering) was removed along with its two tests, since its only caller was these now-repurposed buttons.
15. ~~Experiment selector: a way to pick a proc number~~ -- done. Spectrum list's right-click menu gained "Change proc number…" (Bruker only -- X-Pulse's `read_1d` has no `procno` concept), prompting for a `pdata/<n>` and reloading via `session.reload(index, procno=n)`. Default loading behaviour is unchanged (`io.bruker._read_auto` still prefers the raw `fid`/`ser` unconditionally); proc number only changes which recorded processing parameters (window/phase/zero-fill) get used as the *default recipe*, and which `pdata/<n>` a processed-fallback read would use if no raw file exists.

## GUI polish batch 2 (2026-08-28, following user's follow-up round)

- ~~If delays/b-values are left blank, fit over slice index~~ -- done. The Analysis tab's "Values" source combo (was a single free-text field) is now **Auto / Manual list / Start-End-Step / vclist × factor**. Auto tries vdlist/diff.xml first, then falls back to plain slice index (1, 2, 3, ...) rather than raising -- a fit is always possible, even if not physically meaningful. Manual list is the old comma-separated entry; Start/End/Step generates an evenly-spaced series from three spinboxes (easier than typing values by hand, as requested); vclist × factor reads `io.bruker.read_vclist` (a thin wrapper around `read_vdlist`, whose parser already tolerated unsuffixed plain integers) and multiplies by a user-set constant.
- ~~X-Pulse folder scan was reading individual per-scan `_raw.jdx` files~~ -- done, real bug fixed. `MainWindow._scan_for_spectra` now skips files matching `_XPULSE_RAW_SCAN_RE` (`..._N_raw.jdx`) -- verified on real data: `xpulse_PFG/4` went from offering 160 near-duplicate individual-scan matches down to the 10 actually-useful averaged spectra (one per gradient step); `xpulse_MRI` from 5 down to 1. Distinct from (and doesn't touch) `raw_slices/NNNN_slice.jdx`, which are genuine independent per-slice pseudo-2D data and must stay included.
- ~~Echo-top processing doesn't work well~~ -- **real bug found and fixed**. `processing.ft.align_echo` defaulted to rolling the echo peak to the array **centre** (`n // 2`); the validated reference (`leonmr/xpulse_imaging.py`'s `xpulse_image_import`) rolls to **index 0** instead, so what follows (window/zero-fill/FFT) sees an ordinary one-sided decay rather than a signal with its pre-echo half wrapped into the middle of the array. Default changed to `target_index=0` (still overridable). Along the way, also found (and fixed) that the reference implementation forces `baseline_correct=False` whenever echo-aligning (a straight-line baseline is a poor model for these top-hat imaging profiles) -- `io.xpulse.read_1d`'s `baseline_correct` is now `None`-defaulted to `not align_echo`, explicit `True`/`False` still overrides. Verified together on real `test_data/xpulse_MRI`: negative-going-points fraction went from 52% (no correction) to 81% (echo-align alone, the old bug) down to 41% (both fixes) -- a real, measured improvement, not just "should be better in theory."
- ~~Right-click the plot to switch ppm / Hz / kHz (mm "as relevant")~~ -- **ppm/Hz/kHz done**, mm deferred. `MplCanvas` tracks `_current_1d_spectrum` while a plain 1D spectrum/slice is shown; right-click → "Axis Units" submenu re-plots from the spectrum's own stored ppm/hz arrays (kHz = hz/1000), no recomputation. **mm is not built** -- it needs the same X-Pulse gradient-calibration data (`xpulse_imaging.GRADIENT_CALIBRATION`/`hz_to_mm`) already flagged as missing under "PFG / diffusion fitting" below; the two should be ported together (same missing piece, same follow-up).
- ~~Add a way to update the spectral reference (SR)~~ -- done. Processing tab gained a "Reference (SR)" box: enter the peak's current position (or "Pick from Plot", reusing the same click-to-pick infrastructure as the Analysis tab's peak picker -- the two pick modes are mutually exclusive, toggling one off the other) and what it should actually be; "Apply" shifts the direct-dimension ppm (and Hz, via the spectrum's own SFO1) axis by the difference, in place, no reprocessing needed.
- ~~Peak lists/analysis state should be siloed per experiment, not lingering~~ -- done. `MainWindow._save_current_analysis_state`/`_restore_analysis_state` snapshot the peak table, fit type/model, values source (+ its manual-text/start-end-step/vclist-factor inputs), and results text into `spectrum.metadata.raw["analysis_state"]` before switching away from a spectrum, and restore (or reset to fresh defaults for a never-touched spectrum) when switching to one -- mirrors the FID-processing persistence pattern from the previous batch.
- ~~See the spectral (1D peak) fits, not just the final T1/T2/D parameter~~ -- done. A "View slice fit" spinbox (enabled once a fit has run) plots the chosen slice's data + its fitted pseudo-Voigt curve (`MplCanvas.show_slice_fit`) in the Spectrum tab -- the intermediate per-slice fit that feeds the parameter fit, from `NMR2D_fit_slices`'s own per-slice output, stashed per-spectrum (`metadata.raw["slice_fit_results"]`) alongside the siloed analysis state above so it also survives a reselection.

## GUI polish batch (2026-08-28)

- ~~Scrolling over a 2D/pseudo-2D contour should scale the contour levels~~ -- done. `MplCanvas` tracks `_contour_spectrum`/`_contour_factor` while a contour is the current view; scroll up/down there now redraws with a higher/lower threshold (`plot_2d_contour`'s `factor`, clamped `[1e-6, 0.5]`) instead of zooming, preserving the current pan/zoom. Any other view (1D, FID, compare, fit) clears this state so scroll reverts to normal zoom.
- ~~Overlay the window function on the FID~~ -- done. The FID-preview canvas plots the currently-selected window's own shape (via `ft.apodize` on an all-ones array) as a dashed "Window" line, scaled to the FID's own peak magnitude -- updates live as the window/line-broadening controls change, before "Reprocess" is even clicked (`MainWindow._compute_window_envelope`/`_refresh_fid_preview_if_active`).
- ~~Zero-fill as a dropdown of standard NMR sizes~~ -- done. `zero_fill_combo` offers 1k..256k (all in `_ZERO_FILL_SIZES`) plus "Custom…" (reveals a spinbox). Defaults to the nearest preset at or above 2x the acquired points (`NP`, newly tracked as `metadata.raw["fid_original_length"]` by every raw-FID reader) -- falls back to Custom, pre-filled with the raw doubled value, when that exceeds every preset.
- ~~Processing parameters should persist per spectrum instead of resetting~~ -- done. `Session.get_load_kwargs(index)` exposes the kwargs a spectrum was actually last (re)loaded with (already tracked for `reload()`'s own benefit); `_refresh_fid_processing_controls` now restores window/zero-fill/shift/echo/LP controls from that instead of hardcoded defaults whenever the spectrum has been explicitly reprocessed before, falling back to fresh defaults for a never-touched spectrum.
- ~~A tab per plot kind (1D slice, 2D contour, fit) to navigate between~~ -- done. The single shared canvas is now four (`canvas` "Spectrum", `fid_canvas`, `compare_canvas`, `fit_canvas`) in a `QTabWidget`; each draw path targets its own and auto-switches to it, so e.g. running a relaxation fit doesn't clobber the spectrum view you were looking at, and you can navigate back.
- ~~Option to plot pseudo-2D as stacked slices~~ -- done. A "View: Contour / Stacked slices" combo next to the pseudo-2D checkbox; stacked view reuses `MplCanvas.show_compare` (extracting every slice via `slicing.get_slices`) since a pseudo-2D stack is exactly "many 1D traces, stacked". Resets to Contour on every fresh spectrum selection.

## PFG / diffusion fitting

See the "Analysis (peak fitting / relaxation)" section below for the T1/T2 relaxation analogue this parallels.

Investigated 2026-08-28 (user: "much of the pfg plotting and fitting should exist across pyleo and the x-pulse processing etc") -- found the existing, validated source to port from, rather than writing from scratch:
- **Bruker**: `~/Documents/GitHub/leonmr/leonmr.py`'s `diff_params_import(datapath, NUC)` parses `diff.xml` directly (`ElementTree`) -- delta/DELTA (ms->s), `exDiffCoff` (expected D), and critically the gradient **and b-value lists straight from TopSpin's own precomputed `xValues/List`** (strided `[1::4]` G/cm->T/m, `[3::4]` s/m^2) -- no need to derive b from delta/DELTA/gradient ourselves for Bruker, TopSpin already did it correctly (including the `Delta-delta/3` correction). `DiffFit(y, grad_params, expD=[...], stretch=[...], ...)` is the paired Stejskal-Tanner fit (`lmfit`, `I = I0*exp(-(D*B)^beta)`, multi-component capable) -- complete and mature.
- **X-Pulse**: no equivalent is anywhere near as clean. `xpulse_imaging.py`'s `GRADIENT_CALIBRATION` dict has empirically-measured max-gradient-per-probe/axis (G/cm, for `HFX`/`LOWGAMMA`) -- the one piece of hard-won calibration data that exists; without it (or the right probe identified) X-Pulse b-values can't be computed at all, since `G1` in the acquired data is only a normalised (-1..1) drive fraction. `~/Documents/GitHub/x-pulse-sequences/sequences/diffusion/*.py` documents the formula (`b ~ (gamma*G1_abs*delta)^2*(DELTA-delta/3)`, `G1_abs = G1 * FP{axis} * Gmax_from_table`) and duplicates the same calibration table (must be kept in sync, per that repo's own comments). The most complete X-Pulse-side logic (`delta_DELTA_from_exp_par` for PGSE-vs-PGSTE delta/DELTA/ramp-time conventions, `process_fid_array`/`process_fid_array_phased` for loading a `array#/*.jdx`-folder-per-gradient-step experiment -- exactly `test_data/xpulse_PFG/4`'s layout) lives only in a working notebook, `~/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Code Development/X-pulse_gradient_calibration_new.ipynb` -- correct/validated against known water D, but not packaged (duplicated cells, hardcoded absolute paths, reused variable names). `nmrtools2` has no diffusion-specific code at all yet, only the generic `relaxation.py` model-registry fitting framework this should plug into (same as the existing T1/T2 wiring).
- ~~Bruker PFG fitting~~ -- done. `io.bruker.read_diff_xml(path, nucleus=None)` (ported from `diff_params_import`) and `read_vclist` (a thin wrapper around `read_vdlist`, whose parser already tolerated unsuffixed plain numbers -- no new parsing needed). `processing.relaxation.RELAXATION_MODELS` gained `stejskal_tanner`/`stejskal_tanner_stretched` (`I0*exp(-(D*b)^beta)`) plus `format_diffusion_value` (scientific notation -- D doesn't have a clean SI-prefix convention the way times do). The Analysis tab has a "Fit type: Relaxation (T1/T2) / Diffusion (PFG)" combo that swaps the model list and relabels the delay/b-value entry row; b-values auto-read from `diff.xml` when present, or entered manually, exactly mirroring the vdlist flow. Verified end-to-end on real `test_data/bruker_PFG/150` (auto-read b-values, D = 1.10e-10 ± 0.36e-10 m^2/s -- right order of magnitude for the diff.xml's own `exDiffCoff=6e-11` hint, using only a naive auto-picked peak/width). Along the way, found and fixed a real numerical bug in `fit_relaxation` itself (affects T1/T2 fits too, not just diffusion, whenever parameters differ by many orders of magnitude): `curve_fit` without `x_scale="jac"` was hitting bogus premature `xtol` "convergence" on a Stejskal-Tanner fit (I0~1-10 next to D~1e-9), off by 30%+ from the known-true synthetic value; `x_scale="jac"` fixes it and is a no-op for already-well-scaled models (confirmed via the full existing relaxation test suite still passing).
- ~~X-Pulse array-folder loader (`array#/*.jdx`-per-gradient-step)~~ -- done 2026-08-29, see "GUI polish batch 3" item 9 above (`io.xpulse.read_array`). What that item's own writeup flags as still missing: a probe+axis gradient calibration (or a user-entered max-gradient value), needed to turn the relative gradient fractions it now exposes (`metadata.raw["gradient_relative"]`, from `parameters/G1_valueslist.txt`) into a true b-value -- see item 7 there for exactly what's blocking it (gamma-unit-convention discrepancy between `find_gamma` and `read_diff_xml`'s own docstring, unverified against the reference notebook). The `xpulse_imaging.GRADIENT_CALIBRATION` table (empirically-measured max-gradient-per-probe/axis, G/cm, for `HFX`/`LOWGAMMA`) is the one piece of hard-won calibration data that exists as an alternative/cross-check to a user-entered value -- port it if/when a probe-identification field is also added, since without knowing which probe was used the table alone doesn't resolve to one number. `~/Documents/GitHub/x-pulse-sequences/sequences/diffusion/*.py` documents the same formula and duplicates the same table (must be kept in sync, per that repo's own comments). The most complete X-Pulse-side logic (`delta_DELTA_from_exp_par` for PGSE-vs-PGSTE delta/DELTA/ramp-time conventions) still lives only in the working notebook below -- correct/validated against known water D, but not packaged (duplicated cells, hardcoded absolute paths, reused variable names) -- worth checking against directly before trusting a hand-rolled b-value formula on real data.
- This same `GRADIENT_CALIBRATION`/`hz_to_mm` porting work is also what "GUI polish batch 2" below needs for the canvas's ppm/Hz/kHz/**mm** axis-unit switch (mm is the one unit not yet built there) -- do both together when this gets picked up.

## Processing

- ~~More FID-processing options need to be reachable from the GUI~~ -- done. The Processing tab has a collapsible "FID processing (before FFT)" group (collapsed by default -- a checkable `QPushButton` toggle, deliberately *not* `QGroupBox.setCheckable`, which auto-*disables* its children while unchecked and would silently fight every `setEnabled()` call below): window/apodization choice (all of `ft._WINDOWS`, plus a "Recorded (TopSpin's own WDW)" option for Bruker), zero-fill size, echo alignment (`ft.align_echo`), FID shift (`ft.roll_fid`), and linear prediction (`ft.linear_predict`), applied via a "Reprocess from raw FID" button (not live -- these are raw-FID-domain operations, a full reprocess each time, unlike phase). Backed by new override kwargs on `io.bruker.read_1d`/`read_2d` (F2/direct-dimension only for 2D) and `io.xpulse.read_1d` (which already had window/zero-fill; gained shift/align/LP to match). Only shown/enabled for spectra actually loaded from a raw `fid`/`ser` file (`MainWindow._fid_reprocessable`) -- TopSpin-processed-fallback data can't be reprocessed this way. Also added an opt-in "Preview as FID (time-domain) instead of spectrum" checkbox (off/spectrum-view by default, per explicit request) that plots the actual processed FID (real+imag vs. time, stashed in `metadata.raw["fid"]`/`["fid_dt"]` by the readers above) instead of the FFT'd spectrum -- available for 1D and pseudo-2D (tracks the selected slice); not offered for conventional 2D (no single "the" FID to show). Verified on real `test_data/bruker_T2/11`/`bruker_PFG/150` and covered by new tests in `test_bruker_raw_fid.py`/`test_bruker_raw_2d.py`/`test_xpulse.py`/`test_gui.py`. In passing, fixed a real crash: `core/registry.py::NUC_label` raised `AttributeError` on a `None` nucleus (e.g. a conventional 2D with no recorded indirect-dimension nucleus) instead of degrading to a "?" label.
- Phase 1 range: still bounded (±36000° via the spinbox) -- make genuinely unbounded.
- **Bug**: switching between spectra doesn't reset the phase controls to *that spectrum's own* baseline correctly -- current behavior is described as "relative to the previous point," then zeroing out again. `_refresh_selection_display` is supposed to capture a fresh `_phase_reference` (that spectrum's own complex data) and zero the handles/spinboxes on every selection change -- needs a closer look at whether the reference capture or the reset is actually happening at the wrong time relative to Qt's signal ordering.

## 2D / pseudo-2D processing

- ~~Phase adjustment currently only works for 1D~~ -- done. The Processing tab now shows a "Dimension" box for any 2D/pseudo-2D spectrum with retained imaginary data (`data_imag`, now also read from `2ri` in `io/bruker.py::read_2d_processed`, which previously only read the real `2rr`):
  - Conventional 2D: an F2 (direct) / F1 (indirect) combo box; the correction applies uniformly across the whole dataset along whichever dimension is picked. Preview redraws the full contour (~60ms on a 16x262144 real diffusion dataset -- fine for interactive dragging).
  - Pseudo-2D: a 1-based slice spinbox plus an "Apply this phase to every slice" checkbox. Editing one slice only ever touches that slice (verified on real T2/PFG data -- other rows stay byte-identical); checking "apply to all" broadcasts the same correction to every slice, and unchecking it again reverts every slice but the one still selected. Live preview is a line plot of just that slice (`MainWindow._show_pseudo2d_slice`, via the now-complex-aware `processing.slicing.get_trace`), not a full contour redraw. Slices are labelled `"Slice N"` (or `"Slice N (index ...)"` if a recorded index differs from position) in the canvas title and status line.
  - Switching slice/dimension re-anchors the phase controls to that view's own current state and zeroes them, matching the existing 1D reselection behaviour (`MainWindow._capture_phase_reference`).
  - Covered by 15 new tests in `tests/test_gui.py` plus a manual smoke test against `test_data/bruker_T2/11` and `test_data/bruker_PFG/150` (both real pseudo-2D-by-content datasets that auto-detect as conventional 2D -- exactly the acqu2s-unreliable case the `pseudo_2d` override exists for).

## Pseudo-2D UI

- The "Force pseudo-2D" checkbox should be a switch/toggle control, not read as a radio button.
- For 2D and pseudo-2D plots: the contour panel itself should be square (locked aspect, distinct from the general aspect-ratio-lock feature -- this one should probably be automatic/default for 2D, not something the user has to turn on). For pseudo-2D specifically, there should be a toggle to turn the indirect-dimension display off entirely.
- ~~Pseudo-2D slices should be auto-labelled~~ -- done. `MainWindow._slice_label` labels every slice `"Slice N"`, or `"Slice N (index ...)"` when a recorded indirect-axis value differs from raw position -- used in the canvas title for both the phase-preview slice line and the FID-domain preview, and in the Processing tab's status text.

## Multi-spectrum comparison

- ~~Support loading several spectra to stack and compare~~ -- done. Ctrl/Cmd- or Shift-click several spectra in the Spectra tab's list (now `ExtendedSelection`) and the canvas switches to a new Compare tab's view; single-selection behaves exactly as before. Both layouts are the same call under the hood: `plotting.plot_1d_stack`'s `y_offset` (a large negative value collapses the accumulated stack offset to zero), exposed as "Stacked" (adjustable gap) vs. "Superimposed". Non-1D spectra in a multi-selection are silently skipped with a status note (2D/pseudo-2D overlay comparison is a separate, unbuilt feature).
- Normalisation: **None** (as loaded -- already per-scan, since every importer divides by NS at load time; found and fixed the same gap in `read_2d`/`read_2d_processed`, which weren't doing this despite `read_1d`/`read_1d_processed` always having), **Max intensity** (shape comparison regardless of concentration), or **By sample mass** -- mass is never recorded by any vendor file, so it's a new `NMRMetadata.mass` field set per-spectrum via the spectrum list's right-click "Set sample mass…" (mirrors "Override nucleus…"); spectra with no mass set are skipped with a status note when this mode is active. (X-Pulse's `.jdx` export doesn't record a scan count at all -- confirmed by inspecting a real file -- so a genuinely uniform "the raw counts, unmodified by any of this" view isn't recoverable for that vendor either way.)
- Reordering: "Move Up"/"Move Down" buttons in the Compare tab, swapping the current spectrum's load-order position (`Session.swap`) -- simpler and more robust than wiring up `QListWidget` drag-and-drop reordering. Found and fixed a real bug while building this: `QListWidget.setCurrentRow()` under `ExtendedSelection` mode doesn't clear prior selection (by design, so ctrl/shift-click gestures compose) -- meant opening a *second* spectrum silently left the first one selected too, auto-triggering Compare with two spectra nobody chose to compare. Fixed via `MainWindow._select_single_row`.
- ~~Each loaded spectrum should get a sensible default label~~ -- already covered by the existing `Session.label` (folder name + nucleus + vendor + procno), which reorders correctly under Move Up/Down; a bare "Spectrum N" fallback wasn't needed since `label()` always has more specific information available.

## Analysis (peak fitting / relaxation)

- ~~Absorb nmrtools2's analysis functions (e.g. T2 fitting) and wire them into the GUI~~ -- done, as a new Analysis tab. The backend (`processing/fitting.py`'s pseudo-Voigt/Gaussian/Lorentzian peak fitting, `processing/relaxation.py`'s T1/T2/T1rho models) was already fully ported from nmrtools2 this project -- nothing new needed there, only GUI exposure. Workflow, shown for any selected pseudo-2D spectrum: `processing.slicing.get_slices` (every slice) -> `processing.fitting.NMR2D_fit_slices` (pseudo-Voigt-fit the user's peak(s) in each slice) -> `processing.relaxation.NMR2D_fit_relaxation` (fit a chosen relaxation model to each peak's per-slice amplitude series) -> fitted T1/T2/... ± error (via `relaxation.format_time_value`) and a data+fit-curve plot.
  - Peaks are entered in a table (shift/FWHM/eta, all refined by the fit -- only the starting guess matters) rather than a full drag-to-place/drag-to-resize interactive editor (a much bigger build); a "Pick from Plot" toggle (`MplCanvas.peakPicked` signal, `set_pick_mode`) lets a single click on the spectrum fill in a row's shift, which covers most of the value of interactive placement without the width-dragging complexity.
  - Delays: auto-read from this experiment's own `vdlist` (`io.bruker.read_vdlist`) when present, or entered manually (comma-separated seconds) -- overrides vdlist when given, since not every dataset has one (confirmed on `test_data/bruker_T2/11`, which doesn't).
  - Verified two ways: end-to-end against synthetic pseudo-2D data with a known T2 (recovered 50.1 ± 0.1 ms against a true 50 ms), and against real `test_data/bruker_T2/11` (189Os, an extreme wideline nucleus -- SW ≈ 25,700 ppm) to confirm the pipeline runs correctly and fast (12-64 slices fit in well under a second) end to end; the real dataset's own *numeric* fit quality was poor with a naive default 5 ppm FWHM guess and no vdlist, which is expected for a wideline echo experiment needing a much wider guess and/or FID-domain processing (echo alignment) first, not a bug in the wiring.
  - Not built: region-integration as an alternative, simpler entry point (peak fitting was explicitly requested instead); fixed/bounded peaks (`peak_defs`' `fixed`/`bounds` keys exist in the backend but aren't exposed); S/N-gated amplitude fixing for peaks that decay into noise (`NMR2D_fit_slices`' `noise_region`/`sn_threshold`, backend-only); fitting a real (non-pseudo) 2D spectrum's single spectrum (`fitting.fit_spectrum`/`NMRFitWorkflow`-equivalent) rather than only a pseudo-2D series.

## Opening data

- ~~Loading should work at the *directory* level~~ -- done. Open now shows a folder picker (`QFileDialog.getExistingDirectory`); `MainWindow._scan_for_spectra` recursively scans the chosen folder for Bruker experiments and X-Pulse `.jdx`/`.dx` files (a Bruker experiment claims its own subtree so its `pdata` contents aren't re-matched separately), and `_prompt_select_spectra` shows a multi-select list when more than one match is found -- so both "pick one experiment folder directly" and "pick a parent folder full of numbered experiments" work, and X-Pulse pseudo-2D slices (nested under `raw/sliceN/`) are now selectable. Covered by tests in `tests/test_gui.py`.

## App polish / packaging

- ~~Custom app icon~~ -- logo added (`gui/resources/logo.png`), wired as the window icon. Still open: a real Dock/taskbar icon and custom app name needs actual packaging (see below), not just `setWindowIcon`.
- ~~A real top-level menu bar (native menu bar on macOS, in-window menu bar on Windows/Linux)~~ -- done, see "Phase-tab feedback, plot styling, menu bar" above.
- A proper installer/distributable build for Windows, macOS, and Linux -- should be "beyond easy" to get running on any of the three, not require the user to set up a conda environment and run `pip install -e .` by hand. Needs a packaging tool (e.g. `PyInstaller`, `briefcase`, or `py2app`/`cx_Freeze`) producing a standalone app (with the icon above baked in) per platform, ideally with a one-click/one-command install path.
