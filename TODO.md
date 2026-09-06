# Backlog

Notes captured from real-data testing, not yet implemented. Grouped by area.

## The file's own SR is now visible; imaging gradient reads GPZ1 (2026-09-06)

User: "As far as the sr goes, I want any sr that was noted in the import files to be applied already and to show in the processing tab. Also for imaging, the relative gradient strength doesn't seem to be calculated correctly for my Bruker data (should be gpz1 generally as a percentage value, e.g. gpz1=2 is 2%)."

### The SR was always applied -- nothing said so

`_ppm_hz_axis` has always folded `(SF - BF1) * 1e6` into every axis it builds, so a spectrum arrives already referenced. What was missing is that nothing recorded *how much*, so the Processing tab's SR box read 0.00 Hz for a dataset that plainly was referenced -- **315.3 Hz** on this project's own EXSY (`2p5mm/2019`), 0 on the Cyclosporine sets.

The readers now record it (`sr_recorded_hz`, `sr_recorded_hz_f1`), and it is blocklisted from the sidecar because it is re-derived from the vendor files on every load.

**This changes what the number in the box means**: it is now the **total** SR in effect, not a user-only delta on top of an invisible one. So `_apply_reference_shift_hz` applies `value - recorded` to the as-loaded axis. Leaving the box alone is then a no-op; typing a different number means "make the total this". Same adjustment in the ppm calibration handler and in the post-reprocess reapply, which now only reapplies a value that actually differs from the file's own (a fresh read has already folded that in).

**Known migration gap**: a sidecar written before this change stores `sr_shift_hz` as a user-only delta, and it will now be read as a total. Harmless where the recorded SR is 0 (most datasets); on one with a recorded SR it would shift the axis by that amount. Not worth a versioned migration for a feature this new, but recorded here rather than left to be discovered.

### Imaging gradient: GPZ1 was never read

The mm axis used the probe's **maximum** gradient as the applied one. A Bruker experiment runs at a *percentage* of it, recorded as `GPZ1` in `acqus` -- so at `gpz1=2` the scale came out 50x too small. Now read automatically (`GPZ[1]`; 80% on Cyclosporine/203, for instance) into a new "Gradient used" box, and the effective gradient is `maximum x percent / 100`.

Editable, and stored per spectrum, because the user's own word was "**generally** gpz1" -- another pulse programme may put its imaging gradient on a different channel. Falls back to 100% when there is no `GPZ` array at all, which is exactly the old behaviour.

### Two test fixtures were quietly self-contradictory

`_make_1d_dataset`/`_make_raw_2d_dataset` wrote `BF1: 400.0` alongside `SF: 400.13`. Nothing read both together before, so it never mattered; it now implies a **130 kHz** vendor reference, i.e. a 325 ppm offset, which is not what any test meant. Made self-consistent (`BF1: 400.13`).

That moved the fixtures' axes from about -320 ppm to a normal range, which surfaced two further things that had been passing by accident:

- Two pivot tests compared `pivot_spin.value()` against an exact centre with `pytest.approx`'s **relative** default. The spin box displays 3 decimal places, so its value is rounded to 5e-4 regardless -- at -320 ppm the relative default is larger than that rounding and it passed; at 5 ppm it is not. Now an explicit `abs=5e-4`.
- `test_..._own_pivot_not_the_other_ones` needed F1's centre to fall outside F2's range, and that held only because the bogus 130 kHz pushed F2 out to -320 ppm. The fixture now takes an `f1_o1`, and the test arranges the separation deliberately instead.

8 new tests. 1027 passing (1023 -> 1027).

## Planned, not built: load speed, and a folder browser in the Data tab

Recorded from a user request to plan rather than implement. Nothing below is done.

### Faster loading of large datasets

The EXSY benchmark (`2p5mm/2019`, 2048 x 65536) reads in **3.7 s** against a 9.268 s starting point, after vectorising the per-row loops, replacing `ng.bruker.read` with a direct binary read, and threading the FFTs. What is left is dominated by three things, and they want different remedies:

1. **The FFTs themselves.** Already `scipy.fft` with a capped worker count. Remaining headroom is in *not transforming what is not looked at* -- see (3).
2. **Peak memory.** The decoded array is ~1 GB in complex128 and several transient copies exist (window, zero-fill, transpose for F1). `complex64` throughout would halve it at ~7 digits of precision, which is far more than any NMR intensity carries; the risk is that comparisons against TopSpin (currently agreeing at +0.99) get noisier, so this needs measuring against the real-data suite rather than assuming.
3. **The user's own idea: Bruker's submatrix layout.** TopSpin stores processed 2D data in `XDIM` blocks precisely so a viewer can read *a region* without touching the whole file, and `read_2d_processed` already reconstructs those blocks -- it just reassembles all of them. A tiled reader that loads only the blocks intersecting the current view would make opening a large processed 2D near-instant, at the cost of a re-read on pan/zoom. This applies only to the **processed** path (`2rr`), not to raw `ser`, which has no such structure.
   A raw-`ser` equivalent exists but is different: transform F2 for every row (unavoidable), then transform **only the F1 columns in view**. That is a real restructure of `read_2d`, not a tweak.
4. **Sparse matrices** (also the user's idea) are worth being honest about: an NMR 2D is not sparse before thresholding -- every point carries noise. Sparsity only appears *after* a contour threshold is chosen, which is exactly where this project already saves work via `_MAX_CONTOUR_POINTS`. A sparse representation would help a *stored* peak list or a thresholded overlay, not the load.

Order of expected value: submatrix/tiled reads for processed data, then `complex64`, then view-limited F1 transforms. Each needs the real-data agreement suite re-run, not just timings.

### A folder browser in the Data tab

User: "the data tab might be nice to have a built-in file viewer where folders are collapsable tabs (like the MNova and bruker programmes). So a user can note a file directory and then click between folders within to load data."

A `QTreeView` over a `QFileSystemModel` rooted at a user-chosen directory, filtered to things this project can actually open, with the existing `io.load` dispatch on activation. The parts that need thought, not the widget:

- **What counts as a node.** A Bruker experiment is a numbered directory containing `acqus`; its `pdata/N` children are separate loadable procnos. The tree should show experiments, not raw filesystem depth, or it will be unusable on a real dataset directory.
- **Cost.** Deciding whether a directory is loadable means reading a parameter file. Doing that eagerly for a whole tree would be slow; it wants lazy evaluation per expanded node.
- **Overlap with what exists.** The Data tab already lists *loaded* spectra. This is a second, different list (things on disk, not yet loaded), so the two need distinguishing clearly rather than looking alike.

## Reference a 2D by clicking a peak and typing its shift (2026-09-06)

User: "I need a right click option on the 2D spectra to be able to set the sr (by setting ppm values)."

### There was no referencing action on a 2D at all

The existing right-click "Set Reference Shift (Hz)…" is gated on `_current_1d_spectrum`, which a contour never sets -- so it simply never appeared on a 2D. And an Hz offset is the wrong question to ask on one anyway: what a user knows is *where a peak should be*, not how many Hz the axis is out by.

New action, "Set Reference from Cursor (ppm)…": right-click the peak, type its true shift, and the SR follows. Both dimensions at once for a genuine 2D (a heteronuclear pair needs referencing on each, against `sfo1` and `sfo1_indirect` respectively); the direct one alone for a pseudo-2D, whose indirect axis is a slice index. No unit conversion is needed on the click, because `set_axis_unit` only ever applies to a 1D view -- a contour's axes are always ppm.

### The part that is easy to get backwards

The clicked position is read off the axis **as currently displayed**, which already carries whatever SR is applied. `_apply_reference_shift_hz` is idempotent *against the original as-loaded axis*, so the value it wants is `existing + (target - clicked) * base_MHz`, not the correction alone. Treating the correction as the whole value would be wrong twice over -- it would both discard the existing reference and re-apply the same correction on every subsequent use.

The regression test asserts this the way a user would notice it: re-referencing a peak that already reads correctly must change nothing.

Both dimensions are worked out before either is applied, for the reason recorded earlier in `_on_apply_sr_clicked` -- applying one calls `_restore_sr_control`, which rewrites the other's spin box from metadata.

### A hung test, and the seam that fixed it

The first version of the menu test patched `QMenu.exec`. That does not take: PySide6 resolves `exec` through the C++ binding, so the real modal menu opened and the headless suite **hung** rather than failing -- the same signature this project has hit before. Added a one-line `_exec_context_menu` seam on `MplCanvas` for the test to patch instead. (`QDialog.exec` *is* patchable at the class level, which is why the dialog tests need no equivalent.)

6 new tests. 1023 passing (1017 -> 1023).

## Slice linewidth, a clipped axis label, and a taller slice panel (2026-09-06)

User: "Linewidths of extracted slices should be 2. Be sure that the contour plot position resets back to the original load position (when I tried, the x-axis label got cut off). The extracted slice plot should be able to reach the top height of the top projection."

### Slice traces at linewidth 2

`slice_linewidth=2.0`, a parameter rather than a literal. The dashed lines on the contour stay thinner deliberately, so they read as annotation over the data rather than competing with it.

### The clipped x label: a rebuild never re-ran the layout

`show_spectrum` ends with `figure.tight_layout()`. `_redraw_contour` -- the preserve-the-view path used by a contour-level scroll and by adding or clearing a slice -- did not. So a rebuild kept whatever subplot margins were last in force, and once the slice panel (which brings its own x-axis label) changed what the figure needs, the contour's own label fell off the bottom. Now both paths run the same layout pass.

### Reset really resets

`_reset_navigation_view` called `toolbar.home()`, which only restores **axis limits** from the navigation stack -- and that stack still holds Axes from before any of this canvas's figure rebuilds, so it cannot restore a *layout* at all. New `MplCanvas.reset_view()`: for a 2D it rebuilds the contour at its natural limits (which now includes the layout pass); otherwise it falls back to `home()` plus a layout pass. Extracted slices survive it -- it resets the view, not the annotations.

### The slice panel reaches the top projection

`divider.append_axes` places the panel alongside the contour cell only. Its locator is now re-pointed at rows `0..n`, giving it the contour's height plus the pad and the F2 strip above it. Self-adapting: with no top projection there is one row and it is a no-op. Verified against the rendered geometry -- panel top matches the strip's top to within 1.5 px, panel bottom matches the contour's base.

### A test that could not fail, twice

The first version of the label test asserted only that the label was still inside the figure -- true either way in the synthetic case, where dropping the layout pass shifts the margins without clipping anything. The second compared a rebuild's axes position against a fresh `show_spectrum`'s, which fails **both** with and without the fix: `tight_layout` is not idempotent on an `AxesDivider` figure, so running it twice in sequence lands a hair off. Settled on asserting that the rebuild *runs* the layout pass, which is the actual invariant and does discriminate.

5 new tests, each verified to fail with its own change reverted. 1017 passing (1012 -> 1017).

## Contour redraws lost the Output style; new style defaults (2026-09-06)

User: "Scrolling the contours redraws the axis with defaults, not the ones set in output. These defaults should probably be: tick and line widths of 1.75 and fontsize of 16, and fig size of 8 by 8 (or 7.5 by 10 for 2D with extracted slices). Also the extracted projections should be normalised."

### The style leak

`set_style` stores the Output tab's choices on the instance and applies them to the axes on screen, and its docstring says "every ``show_*`` method reads ``self._AXIS_LINEWIDTH`` etc., so a future redraw picks these up automatically". True of the 1D views, which set those values directly -- and **not** true of the 2D path, which hands off to `plot_2d_contour` and never passed them. So every contour rebuild fell back to that function's own manuscript defaults (fontsize 12, tick width 1.0). A contour-level scroll triggers exactly such a rebuild, which is where the user hit it.

Fixed in two halves, because `plot_2d_contour` only accepts some of it: `_contour_style_kwargs()` passes `fontsize`/`tick_length`/`tick_width` into the call, and `_apply_current_axis_style()` applies the rest (spine linewidth, minor ticks) to every axes afterwards. `set_style` now delegates its own axis-style half to that same method rather than repeating it -- applying all of it unconditionally is equivalent to the old per-argument guards, since anything not passed in is re-applied at its existing value.

### New defaults

`_AXIS_LINEWIDTH` and `_TICK_WIDTH` 1.4 -> **1.75**, `_AXIS_LABELSIZE` 13 -> **16**. Tick length left at 5.5 (not mentioned).

Export figure size is now a real default rather than an accident: `_EXPORT_FIG_SIZE = (8.0, 8.0)`, and `_EXPORT_FIG_SIZE_WITH_SLICES = (7.5, 10.0)` for a 2D showing extracted slices, picked by `default_export_size()`. Before this the size was left alone entirely unless the Output tab had been edited, so an exported figure inherited whatever size Qt's layout happened to give the live widget; and the Output tab's own spin boxes were initialised from that same live size (6.4 x 4.8) rather than from anything chosen.

**Flagged to the user, implemented as stated**: 7.5 x 10 is taken literally as width x height, i.e. portrait. The sliced layout is contour + panel side by side, which is wider than it is tall, so 10 x 7.5 may have been meant. One number to swap either way.

### Extracted traces are normalised

Each trace now fills its own slot rather than sharing one amplitude scale across the stack. This reverses a deliberate earlier choice (a shared scale keeps relative intensities honest); the user asked for the other trade, which favours comparing lineshape between a weak row and a strong one.

4 new/updated tests, each verified to fail with its own change reverted. 1012 passing (1009 -> 1012).

## Crosshair removed (deferred, not abandoned); slice-panel scroll fixed (2026-09-06)

User: "crosshair is just filling the contour and not going away, and if I try to zoom then there is still an issue. Remove the crosshair for now (I still want it eventually) but keep the other improvements for the slicing."

### The crosshair is out

Removed entirely -- the class, the state, the lazy sync, its call in `_on_motion`, the import, and its tests. The `onmove`-vs-`ignore` fix from the previous entry was necessary but not sufficient: "filling the contour and not going away" is a *second* blit-coherence failure, and one that a headless test cannot see. Both symptoms are the same underlying thing -- a blitting widget and a canvas that also calls `draw_idle()` from its own scroll/pan/level handlers do not reliably agree on what the background is, so stale frames get restored and cursor lines accumulate instead of being erased.

### What a future attempt should start from

- Standing down in pan/zoom belongs in `onmove`, **never** in `ignore` -- `MultiCursor.clear`, the `draw_event` handler that recaptures the blit background, goes through `ignore` too (previous entry).
- That alone is not enough. The canvas issues `draw_idle()` from `_on_scroll`, the pan drag, and the contour-level adjust. Interleaving those with a widget that blits leaves stale regions on screen. A working version needs the crosshair to own its redraws -- i.e. every one of those paths has to invalidate/recapture the background, or the crosshair has to be drawn as a normal (non-blitted) artist and the redraw cost solved another way.
- The cost is why blitting was chosen: a plain redraw per mouse-move repaints the whole contour, measured at **295 ms** for a real HSQC and **1.34 s** for the EXSY.

### The real zoom defect that was left behind

Zoom itself was verified working throughout -- scroll on the x-strip, the y-strip and the body, both projection strips, a full toolbar rubber-band drag, and Home, with and without a slice panel. (An earlier sweep reported two projection strips as broken; that was my own crude event helper attributing the event to the wrong Axes, not the code. Rebuilt with properly-formed events, both zoom correctly.)

One genuine defect did survive: the slice panel lives in `_proj_axes` alongside the F1/F2 strips, and `_on_scroll` iterated that dict treating **every** entry as a projection. The panel's y axis is a normalised 0..1 stack of evenly-spread traces, so scrolling over it silently rescaled the stack and broke the layout -- and the panel sits exactly where the contour's right-hand "y strip" zoom region used to be empty. Now an explicit no-op, and the projection loop names `f1`/`f2` rather than iterating, so nothing else that dict grows later can be mistaken for a strip.

### Noted, not changed

`NavigationToolbar2`'s view stack is never cleared when this canvas rebuilds its figure (there is no `toolbar.update()` call anywhere), so it accumulates references to Axes that no longer exist. `home()` still behaves correctly in every case tested, so this is recorded rather than fixed -- clearing the stack on every contour redraw would throw away the user's zoom history, which is a worse trade.

1009 passing (1011 -> 1009: 4 crosshair tests removed, 2 scroll regression tests added).

## The crosshair froze the canvas: standing down belongs in onmove, not ignore (2026-09-06)

User: "zooming on the 2D no longer works. Something has been broken." Regression from the crosshair added an hour earlier, in the same session.

### Not what it looked like

Zooming was working. Every mechanism checked out headless -- scroll zoom on the x-strip and y-strip, contour-level scroll, and a full toolbar rubber-band press/drag/release all changed the limits correctly, with and without a slice panel present. The limits were right; the *picture* was not.

### The mechanism

`_NavigationAwareMultiCursor` stood down in pan/zoom by overriding `ignore()`. That looks like the natural hook -- it is the one `Widget.active` itself goes through -- but `MultiCursor` routes **two** handlers through it:

- `onmove` (motion) -- the one that should stand down, and
- `clear` (**draw_event**) -- which is what *recaptures the blit background*.

So while pan or zoom was selected, the background was frozen at whatever frame was current when the mode was entered. The next mouse move blitted that stale image straight back over the canvas. The zoom had happened; the display reverted.

**Pan is the default mode on load** (an earlier request in this project), so this was the normal case, not an edge one -- the background was stale from the moment a spectrum appeared.

Fixed by overriding `onmove` instead, leaving `clear` alone. Standing down also hides the lines with a plain `draw_idle` rather than a blit -- the whole point of the override is that the blit background cannot be trusted at the moment the mode changes -- and costs one redraw on the transition only, since after that the lines are already hidden.

### Worth keeping

Blitting widgets have two entry points, and a gate on the shared predicate catches both. When suppressing a blitting widget, suppress the *event handler*, never the background capture. And a "nothing responds" report is not always a broken handler: here every handler was fine and only the frame being painted was wrong, which is invisible to a headless test that asserts on state. The regression test asserts on the cached background itself for that reason.

2 tests (one rewritten, one new), both verified to fail with the `ignore()` gate restored. 1011 passing (1010 -> 1011).

## Slice-panel refinements and a blitted crosshair (2026-09-06)

Six follow-ups on the slice panel plus a cursor request.

### The connector lines are gone

"Drop the lines to the extracted slices, it seems too hard to maintain across different views." Correct call: they looked right in a saved figure, but in the GUI the contour and the panel zoom, pan and change units independently, and a line spanning two Axes has to be re-derived on every one of those. The shared colour is what pairs a line with its trace now. `slice_connect` and the `ConnectionPatch` import are removed rather than left as a disabled option.

### Traces now fill the panel

Was: a fixed offset of `0.55 x` the tallest trace, with the y-limit stretched to whatever the stack happened to reach -- so a short top trace left dead space above it. Now each trace gets one slot of `1/n` across a normalised 0..1 panel, with **a single shared amplitude scale** rather than a per-trace normalisation, so a weak slice still reads as weak next to the diagonal.

### The panel shares the contour's axis, rather than copying it once

Setting `panel.set_xlim(ax.get_xlim())` at draw time was not enough, and the reason is a genuinely subtle interaction: the GUI's own preserve-the-view redraw restores the contour's limits *after* `plot_2d_contour` has returned, so a zoomed contour got a full-range panel beside it. `sharex=ax` makes the two track through every later zoom and pan as well. A *column* panel's x axis is the contour's *y*, which matplotlib cannot share across orientations, so that case still takes a one-time copy.

Also: a matching x-axis label (the contour's own x label for a row panel, its y label for a column panel), no panel title, and the pad raised 0.45 -> 0.95 inches so the panel clears the contour's right-hand y-axis label.

### Adding a slice no longer resets the view

`_add_contour_slice`/`_clear_contour_slices` called `show_spectrum`, which rebuilds from scratch. They use the existing `_redraw_contour(preserve_view=True)` now -- adding a slice is an annotation, not a change of subject.

### Crosshair across the contour and both projections

**Two cursors, not one.** A single `MultiCursor` over all three Axes would also draw a horizontal line across the top strip and a vertical one down the left strip, in those panels' own unrelated intensity coordinates. So: one vertical across the contour + F2 strip, one horizontal across the contour + F1 strip.

**`MultiCursor` rather than a hand-rolled pair of lines, because it blits.** A plain redraw per mouse-move repaints the whole contour -- measured at 295 ms for a real HSQC and 1.34 s for the EXSY, nowhere near interactive.

**It stands down in pan/zoom mode** (`_NavigationAwareMultiCursor.ignore` also consults `toolbar.mode`), per the request: those have their own cursors and drag semantics.

**Rebuilt lazily from `_on_motion`, not from every redraw path.** This canvas clears and rebuilds its figure from about a dozen methods (`show_compare`, `show_fit`, `extract_slice`, ...), and a cursor still holding removed Axes would draw into them. Checking that the Axes it was built for are still the ones on the figure is one hook instead of twelve, and self-heals rather than depending on every future rebuild path remembering to call a teardown.

7 new/rewritten tests, each verified to fail with its own change reverted. 1010 passing (1004 -> 1010).

## Extracted-slice panel, indirect SR, per-spectrum nQ, pseudo-2D peak picking (2026-09-06)

Five requests in one round.

### 1. SR for the indirect dimension

The `Reference (SR)` box worked only on `axes[-1]`. The two dimensions need **independent** values against independent base frequencies (`sfo1` / `sfo1_indirect`) -- on a heteronuclear 2D they are different nuclei, so one Hz offset cannot reference both. Generalised over a dimension key (`_SR_KEYS`, direct keys left unsuffixed so older sidecars still restore), with a second spin box shown only when there is an indirect *frequency* axis to reference -- hidden for 1D and for a pseudo-2D, whose indirect axis is a slice index.

One bug found by running it: `_apply_reference_shift_hz` ends by calling `_restore_sr_control`, which rewrites **both** spin boxes from metadata -- so reading the second box after applying the first handed back the stored value (0) instead of what the user had just typed, and the indirect shift silently did nothing. Both values are now read up front. The post-reprocess reapply path also restored only the direct dimension.

### 2. The nQ line is now the spectrum's, not the app's

It genuinely belongs to the data: a diagonal means something on a homonuclear correlation and nothing on a heteronuclear one, and `n` is a property of the experiment. As one global switch it followed the user onto every other spectrum they selected. Now stored in `metadata.raw`, restored on selection, and persisted with the rest of the processing state. The restore path blocks signals so looking at a spectrum doesn't stamp a default onto it.

### 3. Extracted slices beside the contour

Built to a user-supplied reference figure: a dash-dot line across the contour at each slice's position, the extracted traces stacked in a panel to the right, one colour per slice, and -- the user's clarification -- **each line continues across the gap and joins its own trace's baseline**. Drawn as a `ConnectionPatch` with clipping off, added to the *figure* rather than either Axes, which is the only way to draw a line spanning two panels.

Choices, all the user's: panel on the right; traces stacked with offsets rather than overlaid; placed from the existing right-click "Extract Slice…", which gained a "Show as:" mode (`Show beside the contour` / `Replace the view`) and a label field.

Three things measured rather than assumed:

- **Panel width is relative (`axes_size.Scaled`), not inches.** The same function draws a standalone 7-inch figure and the GUI's embedded canvas; a fixed 3.5-inch panel left the embedded contour a few pixels wide.
- **Stacking order follows the contour.** The slice whose line sits highest gets the highest trace -- otherwise the connectors cross, which reads as an error even though every line joins the right trace.
- **The panel-side connector anchor is in axes coordinates, not data.** Data coordinates read `panel.get_xlim()` before the final `set_xlim`, so the endpoint landed outside the finished axis and the connector shot across the whole panel instead of stopping at its edge.

Slices are stored against the spectrum (a slice at F1 index 412 means nothing on another dataset), with `set_contour_slices` deliberately not emitting so restoring doesn't write straight back out.

### 4. Peak picking on a pseudo-2D

It really was impossible. Manual picking listens on the Spectrum canvas, which shows the contour/stack for a pseudo-2D, so a click there yields a 2D position with no single trace to attach a peak to; Auto-Pick was hard-wired to `index=1`; and a conventional 2D was refused outright.

Now there is a "Pick on slice" spinner (first by default, `setWrapping(True)` per the user: "pressing down on the lowest should cycle back to the highest"), turning pick mode on shows that slice on its own so there is something to click, and changing the slice while picking follows along. Both manual and Auto-Pick go through one `_peak_pick_target`.

**A latent bug fell out**: `extract_slice` read `self._contour_spectrum`, which `show_spectrum(trace)` clears because the trace is 1D -- so the source was `None` on any *second* call and `get_trace` raised. The right-click "Extract Slice…" was a one-shot, and picking silently stuck to whichever slice it started on. It now falls back to `_extracted_from_spectrum`. Found only because a broad `except Exception` I had just written was hiding it; narrowed to `IndexError`.

**A test that could not fail**: the first version of the Auto-Pick test passed with the hard-wired `index=1` still in place, because the shared 2D fixture tiles one identical row -- slice 1 and slice 3 are the same array. Rebuilt with per-row-distinct data and re-checked against the reverted fix.

### 5. Nothing pre-selected in Compare

`_rebuild_compare_list` always set a current row. This **narrows** the earlier "it is no longer obvious which spectrum is selected" feedback rather than reversing it: that was about the highlight being invisible once a row had been chosen, which still works. What is gone is the choice made on the user's behalf before any click. A current entry is still preserved across a rebuild, so Move Up/Down and check-state toggles don't lose their place.

18 new tests. Each fix verified to fail with itself reverted, independently. 1004 tests passing (986 -> 1004).

## Phasing a 2D snapped the zoom back to full scale on every pause (2026-09-06)

User: "Something is up with the GUI phase control for the indirect and it's jumping during phasing."

The jump was literal, and it is a view bug rather than a phase one -- the phase values themselves are smooth. Traced a real F1 drag tick by tick on `Cyclosporine/203`: each 0.5-degree tick changes the data by ~0.0102 relative, decreasing smoothly, with no discontinuity at a settle (0.00024) and `_phase_reference_delta_applied` tracking correctly. Nothing wrong there.

### What was actually jumping

Two 200 ms debounce timers -- `_live_reprocess_timer` and `_canvas_2d_refresh_timer` -- both restart on **every** drag tick, so on a pause they always fired as a pair. `_run_debounced_canvas_2d_refresh` carefully saves and restores the axis limits around its contour rebuild ("phase-editing shouldn't silently reset a deliberate zoom"). `_run_debounced_multi_row_reprocess` did not: a bare `show_spectrum`. It runs first, so its reset won.

Measured, with the view zoomed to 4..2 ppm x 60..20 ppm and one drag tick:

| baseline correction | timers that fire | zoom after |
|---|---|---|
| off | canvas only | preserved |
| **on** | reprocess, canvas | **reset to 12.7..-3.3 x 157.5..-7.5** |

Baseline correction is what puts the reprocess timer in play for a phase drag, which is why this shows up while phasing rather than at rest.

### Also: every settle rebuilt the identical contour twice

Both timers redrew, one straight after the other, for one frame. A contour rebuild measures **295 ms** for `Cyclosporine/203` and **1.34 s** for the EXSY (against a 7 ms and 140 ms phase tick), so the entire cost of a settle was being paid twice. The reprocess now cancels the canvas-refresh debounce, since it *is* that redraw.

### Fix

One `_redraw_2d_preserving_view` helper, used by both settle paths -- the zoom-preserving version was already written, it just wasn't shared. 5 new tests (parametrised over both dimensions and baseline on/off), verified to fail with each half of the fix reverted independently.

### Left alone deliberately

During a *continuous* drag the contour still does not redraw at all: both debounces keep restarting, so the update lands only when the drag pauses. That is the existing deliberate trade-off, and at 295 ms / 1.34 s per rebuild it is not obviously wrong -- a throttle would give feedback while dragging but would block the event loop for 1.34 s at a time on the EXSY. Flagged for the user rather than changed unilaterally.

986 tests passing (981 -> 986).

## The hourglass contours: F2's dispersion was leaking into F1 (2026-09-06)

User: "I'm still suffering from this hourglass shape in the contours, I feel like the phases of f2 and f1 are somehow offset from each other's phases."

That reading was right, and it names the artifact precisely. An hourglass (bowtie) contour **is** a phase-twist lineshape -- `Re[(A1 + iD1)(A2 + iD2)] = A1*A2 - D1*D2` -- which is what you get when the two dimensions' absorption and dispersion are mixed 90 degrees apart. Not a phase *setting* that could be dialled out: a structural error in the hypercomplex decode.

### The error, which both hypercomplex modes made

After the F2 FFT each row `a` is complex: `Re(a)` is F2 absorption, `Im(a)` F2 dispersion. The decode combined **whole complex rows** as `a + 1j*b`. That looks like the textbook hypercomplex step, but

```
a + 1j*b  ==  (Re(a) + 1j*Re(b))  +  1j*(Im(a) + 1j*Im(b))
```

-- the F2-real hypercomplex series *plus 1j times* the F2-imaginary one. F1-transform that and take the real part and you get `RR - II`, the phase twist. The wanted spectrum is `RR` alone, so the F2 imaginary has to be dropped (States) or cancelled (Echo-Antiecho) **before** the F1 transform.

The tell that this was structural rather than a phase offset: `a + 1j*b` is C-linear, so a global F2 phase commutes straight through it and through the F1 FFT. That accidental commuting property is exactly what let F2's dispersion into F1, and it is gone now -- see "consequence" below.

### Echo-Antiecho had a second, independent bug underneath

The raw "echo" row was conjugated **and time-reversed** before F2 processing. The stated reasoning -- the conjugate's own frequency reversal and the explicit reversal cancel -- is true for a bare FFT. It is false here, because **the window and the zero-fill sit in between**: reversing a decaying FID and then multiplying it by a decaying window applies the window backwards, and the zero-fill pads what is now the head of the signal. The echo rows were not mirrored, they were destroyed.

Measured against TopSpin's own `xf2` output (procno 2 of `Cyclosporine/203`, which the user had already processed that way):

| | our even (echo) rows | our odd (antiecho) rows |
|---|---|---|
| with the transform | **0.03** | 0.72 |
| transform removed | **0.73** | 0.72 |

Undoing it afterwards in the frequency domain recovers nothing (0.0008) -- the tell that information was lost rather than moved. With it removed, all 128 high-SNR rows match their own index.

### How the right answer was found, instead of guessed

Two earlier rounds guessed at sign/order permutations and "verified" the winner. This time the recombination was **solved**. TopSpin's own recombined F1 interferogram is recoverable from any finished 2D: undo `proc2s`'s `PHC0`/`PHC1`, then inverse-FFT along F1. Fitting `interf[k] ~ c1*a + c2*b + c3*conj(a) + c4*conj(b)` per t1 point then gives, with no ambiguity:

- coefficients **1.0000** on `conj(echo)` and on `antiecho`, **0.0003/0.0004** on the other two
- median relative residual **0.0172**, per-point `|corr|` **0.997-1.0000**
- a per-point scale whose phase is a **constant +90 degrees**, its magnitude just the F1 window (halved at k=0 by `FCOR`)

So `interferogram = 1j * (conj(echo) - antiecho)`. That constant 90 degrees is the whole ballgame: it turns F1 absorption into F1 dispersion, and dropping it costs everything -- end-to-end +0.9948 with it, **+0.0010** without. Guessing formulas would never have found a factor that looks like a no-op.

A prerequisite fell out of the same work: **which block is which dimension's imaginary partner had been recorded backwards.** Inverse-transform each candidate pair and only the true one comes back causal:

| pair | along F1 | along F2 |
|---|---|---|
| `2rr + 1j*2ir` | 49.55% | **94.32%** |
| `2rr + 1j*2ri` | **100.00%** | 49.36% |
| `2rr + 1j*2ii` | 50.20% | 49.66% |

`2ir` is F2's imaginary, `2ri` is **F1's** -- first letter F2, second F1. The 49-50% entries are the signature of a pairing that isn't one (an effectively real spectrum inverse-transforms Hermitian-symmetric, splitting energy between head and tail). An xf2-only dataset is the exception and stores only `2rr`/`2ii`. Recorded as `io.bruker._PROCESSED_BLOCK_NAMING_NOTE`.

### Result, against TopSpin's own `2rr`

| dataset | mode | before | after |
|---|---|---|---|
| `Cyclosporine/203` | Echo-Antiecho | +0.4448 | **+0.9948** |
| `Cyclosporine/204` | Echo-Antiecho | +0.0953 | **+0.9939** |
| `2p5mm/2011` | States-TPPI | +0.6883 | **+0.9986** |
| `2p5mm/2016` | States-TPPI | +0.7049 | **+0.9990** |
| `2p5mm/2019` (EXSY benchmark) | States-TPPI | +0.7021 | **+0.9956** |
| `2p5mm/2023` | States-TPPI | +0.6721 | **+0.9493** |
| `2p5mm/6` | States-TPPI | +0.6947 | **+0.9623** |

And the artifact itself, measured as negative/positive intensity in a box around each of the 6 strongest peaks (a phase twist has large negative diagonal wings; pure double absorption has almost none): **0.4227 -> 0.0311**, against TopSpin's own **0.0347**. The hourglass is gone, not reduced.

The EXSY load benchmark is unaffected: 3.7 s best of 3 (ceiling 9.268 s).

### Consequence: F2 phasing on a hypercomplex 2D can no longer be previewed live

The corrected decode is only R-linear, so an F2 phase no longer commutes through it -- correctly. It also means the stored `data`/`data_imag` pair for a hypercomplex 2D is **F1's** quadrature (TopSpin's `2rr`/`2ri`), which is exactly what F1 phasing wants and exactly what F2 phasing cannot use. F2 phase ticks for those spectra now defer to the debounced rebuild, which phases F2 before the decode where it still means something (`MainWindow._f2_quadrature_is_spent`). F1 phasing, the common interactive case, is unaffected and still live.

Not done instead: keeping the second hypercomplex plane (`2ir`/`2ii`) so F2 could be phased from four stored planes the way TopSpin does. That is the textbook answer, rejected on memory -- the EXSY's decoded array is already ~1 GB, and this project caps contour points at a quarter of that same dataset for the same reason.

A pleasant side effect: `read_2d` and `read_2d_processed` now agree on what `data_imag` means for a hypercomplex 2D, since the latter already returned `2ri`.

### Two copies of the deleted transform lived in the GUI

`_recompute_multi_row_data` and the per-row slice path both carried their own copy of the Echo-Antiecho raw-row transform. The suite's live-rebuild-equals-load tests caught both immediately. Worth noting the symmetry: this project's standing "two pipelines quietly diverge" warning is usually about a fix that reaches only one copy -- here it was a *deletion* that had to reach all three.

### Still broken, and not caused by this (both were equally broken before)

- `Cyclosporine/211` (Echo-Antiecho): +0.0143 -> +0.0113. F2 projection +0.71, F1 projection +0.03. Not the F1 LP (disabling it gives +0.0250), not a strip transform (`STSR=0`, `STSI=SI`). Its `proc2s` asks for an 8x LP extrapolation (`LPBIN=2048` from 128 complex points), which is the most suspicious thing about it but not yet demonstrated.
- `exam_CMCse_3/19` (States-TPPI): +0.0139 -> +0.0164, and its interferogram fits nothing (residual 0.998). Here it is **F2** that is wrong (F2 projection +0.51, F1 projection +0.90). Its `procs` has `TDeff=4096` with `SI=2048`, i.e. TopSpin windows over more points than it transforms -- a window computed over the wrong length changes the lineshape, which fits the symptom. Untested.

Both are single datasets with a specific parameter combination; neither is the hourglass.

5 new discriminating tests (all verified to fail with the fixes reverted) plus 8 real-data agreement tests that pin the table above and skip when the data isn't present. 981 tests passing (971 -> 981).

## Live phasing no longer "bounces" between baseline-corrected and not (2026-09-06)

User: "The indirect phasing sort of bounces like there is a mismatch to the current and read phases."

### It is not indirect-specific, and it is not a phase mismatch

Measured both dimensions with baseline correction on and off, which separates the two candidate causes cleanly:

| dim | baseline | live vs rebuilt |
|---|---|---|
| F2 | off | 0.000000 |
| F2 | **on** | **0.001701** |
| F1 | off | 0.000000 |
| F1 | **on** | **0.001701** |

Identical for F2 and F1, and zero without baseline correction. So the cause is **baseline correction**, not the indirect dimension and not a recorded-vs-current phase mismatch. (A double-baseline hypothesis was tested and rejected first -- baseline correction is exactly idempotent here, 0.000000 applied twice.)

### The mechanism

Baseline correction is deliberately **not** applied on every drag tick -- measured at **173 ms** for `Cyclosporine/203` and **1.5 s** for the EXSY, far too slow for a per-tick path, so the existing comment claiming as much is still correct. It is deferred to a debounced rebuild that fires after 200 ms of stillness.

The bug was what happened *next*: the following drag tick re-derived the display from the **pre-drag snapshot**, which has no baseline -- so the correction was thrown away again. Pause, correction appears; move, correction vanishes; pause, it reappears. That is the bouncing, and it repeats for the whole drag.

### The fix

After a rebuild, its result becomes the new phase reference, and the phase that result already carries is recorded (`_phase_reference_delta_applied`) so the next tick applies only the *remainder* rather than re-applying the whole phase to data that already has it. The drag then continues *from* the corrected data.

Live view after a settle, against the authoritative rebuild at the same phase: **0.001701 -> 0.000028**, for both dimensions.

**Restricted to rebuilds that came from the raw FID.** The processed-only fallback (`_recompute_multi_row_phase_and_baseline_only`) *derives* its result from this reference, so overwriting the reference with that result would make it self-referential -- baseline correction could then never be switched back off, because the uncorrected source it reverts to had been replaced by the corrected output. Caught by the existing pseudo-2D apply-all-slices test, which toggles baseline off and on again.

### Two measurement mistakes of my own, worth recording

- An early check called `_commit_pending_phase_to_metadata()` directly and appeared to show the phase doubling on commit. That is not a user-reachable path; re-running with **only real UI actions** showed the values stable. Direct calls to internal commit helpers do not reproduce user behaviour.
- The first anti-bounce check compared raw before/after magnitudes and "failed" at 0.007 for a 0.5-degree tick. That is the *correct* magnitude -- a phase rotation mixes in the dispersive component, so the change scales as `sin(0.5 deg) ~ 0.0087` of the imaginary part, not linearly with the phase. The meaningful comparison is against the authoritative rebuild, not against the previous frame.

3 new regression tests (parametrised over both dimensions), all verified to fail with the fix disabled. 971 tests passing (968 -> 971).

## Extracted slices now round-trip: phase on a slice, apply to the 2D (2026-09-06)

User: "I don't think the derived slice phasing is feeding back into the contour, it should be more like a temporary file and have some way to go back into the contour data."

Correct -- a kept slice was an independent copy, so phasing it went nowhere. It is now a **working copy with a live link home**.

**Provenance** is recorded when a slice is kept: the parent spectrum (a live object reference, *not* a session index, which shifts when spectra are removed or reordered), the axis, and the index. All three are blocklisted from the sidecar -- a spectrum object is not serialisable, and a within-session relationship means nothing after a reload.

**"Apply Phase to Source 2D"** appears in the Processing tab only for a slice whose parent is still loaded. It adds the phase dialled in on the slice into the parent's own *user* phase keys and rebuilds it through `_recompute_multi_row_data`, then switches back to the 2D. Going through the existing pipeline rather than mutating the parent's data directly is what makes the result **identical to having dialled the same phase on the 2D itself** -- verified to `< 1e-9` relative difference -- and what makes it survive the next rebuild.

**The dimension mapping is the subtle part** and is derived in one place: extracting with `axis="f1"` *fixes* F1 and traces **along F2**, so phasing that trace is an **F2** phase. Backwards would silently phase the wrong dimension of the 2D.

Only the phase added *since extraction* is transferred -- the slice inherited the parent's recorded phase when it was cut, so transferring that as well would apply it twice. The slice's own phase keys are cleared afterwards, so a second click cannot double-apply (it says "No phase to apply" instead).

### Two real bugs found by testing the round trip

- **Selecting the parent re-applied the phase to the slice.** `_capture_phase_reference` commits any pending phase into the *outgoing* target on the way out, so switching to the 2D wrote the still-live spin value straight back onto the slice, undoing the clearing. Fixed by zeroing the spin boxes *before* selecting. Found by running the round trip, not by reading it.
- **`parent not in self.session.spectra` raised.** `in` and `list.index()` compare with `==`, and `NMRSpectrum` is a dataclass holding numpy arrays, so equality returns an array: *"truth value of an array with more than one element is ambiguous"*. Both lookups now compare by identity. Caught by the test for the removed-parent case.

Also corrected in passing: an earlier equivalence check appeared to fail at 1.1 relative difference, which was my own harness not settling the debounced rebuild before comparing -- the real figure is 0.0.

968 tests passing (963 -> 968).

## Extracted-slice actions made visible (2026-09-06)

User: "There is no way to actually phase the extracted slice as it stands. maybe the extracted view should have new options to do something with the slice."

**The capability was there and worked -- the affordance was not.** Checked first rather than assuming a functional bug: after "Keep Slice", the phase controls are in the *identical* state as for a normally-loaded 1D (`_phase_mode == "1d"`, correct target index, handles enabled, same status text), and phasing genuinely changes the data. The problem was that the only way in was a **right-click context menu on the plot** -- not somewhere anyone looks, so in practice the feature may as well not have existed.

Fixed as suggested: the extracted view now has its own visible bar, shown **only** while a slice is being viewed:

- a label naming the slice and stating it is *not yet in the spectrum list*, so its status is obvious rather than inferred;
- **"Keep for Phasing/Fitting"** -- adds it to the session and selects it;
- **"Back to Contour"** -- returns to the 2D it came from.

Both buttons share the same helpers as the context-menu entries, which are kept so anyone who had found them is not broken. `_refresh_slice_bar` is driven from the single set/clear points, so the bar can never be left showing over an unrelated view.

**A real self-inflicted bug, caught by running it rather than by reading it**: a global text replace rewrote the signal emit *inside* `_emit_slice_keep_requested` into a call to the helper itself, so clicking Keep recursed to `RecursionError`. It has its own regression test now. A second, smaller one: the new canvas tests asserted `isVisible()`, which Qt reports as `False` for every child of a widget that was never shown -- `isHidden()` is the right check when the parent is not realised.

963 tests passing (959 -> 963).

## Navigation shortcuts, phasable extracted slices, and softer negative contours (2026-09-06)

Three requests, all done.

### (a) Navigation-mode shortcuts, pan by default

New **View** menu: **Pan Mode** (`Ctrl/Cmd+Shift+P`), **Zoom Mode** (`Ctrl/Cmd+Shift+Z`), **Toggle Pan/Zoom** (`Ctrl/Cmd+Z`, as requested -- there is no undo stack for it to collide with), and **Reset View** (`Ctrl/Cmd+0`). **Pan is switched on automatically once data is loaded**, and only for the first spectrum, so it never overrides a mode the user has since chosen.

Two things worth knowing about the implementation: matplotlib's toolbar owns this state and its `pan()`/`zoom()` methods **toggle**, so the helper checks the current mode first -- calling unconditionally would turn the requested mode straight back off. And the mode is applied to *every* canvas, not just the visible one, because it is a per-toolbar flag: otherwise switching canvas tab would silently change the interaction mode back.

### (b) Extracted slices can now be phased

`MplCanvas.extract_slice` only ever *displayed* a trace -- it never reached the session, and every phase control acts on session spectra, so an extracted slice could be looked at but not worked on. A new right-click **"Keep Slice (for phasing/fitting)"** (on an extracted slice) hands the exact displayed trace to `Session.add_derived` and selects it, so phasing, fitting and Compare all apply through their existing paths with no parallel machinery. `processing.slicing.get_trace` already carries `data_imag` through, which is what makes phasing work; when it is absent (a magnitude-mode or processed-only source) the slice is still kept, with a message explaining why phasing is unavailable rather than leaving the user to find a dead control.

**A pre-existing crash found while building this**: extracting a *column* (trace along F1) from a **pseudo-2D** raised `ValueError: This axis has no 'ppm' values (kind='index')` from `plot_1d` -- F1 there is a slice index, not a frequency. Reachable straight from the Extract Slice dialog. That option is now disabled for pseudo-2D, with a tooltip saying why (a trace along F1 is a decay curve; the Analysis tab is where that belongs) rather than silently removed.

### (c) Negative contours: absolute scaling, and teal

Both fixes to the same complaint -- "the negative contours look really prominent vs the positive".

**Scaling** was the substantive half. The negative side used to be scaled to `amin(data)` *independently* of the positive side, so on a spectrum with a big positive peak and small negative excursions its lowest level sat at `factor * |amin|` -- a far lower **absolute** threshold than the positive side's `factor * amax`. The negative contours therefore started much closer to the noise floor and drew far more rings, reading as prominent when the underlying signal was weaker. Both sides now scale to `max(|data|)`, and the negative levels are the exact mirror of the positive ones. On a 1000-vs-(-60) test spectrum, the negative side now draws **nothing at all**, which is correct: -60 is below the level the positive side itself starts at.

**Colour** changed from red to **teal** (`NEGATIVE_CONTOUR_COLOUR`, one constant so it can be retuned in one place). Red reads as an alarm colour and dominated the black positive contours even at equal intensity; teal stays clearly distinguishable from black -- and remains so under the common red-green colour-vision deficiencies, which red-against-black does not.

### Also cleaned up

- Removed an `_all_canvases` **duplicate I had just introduced** -- the class already had one. Caught by an assertion in my own edit script rather than shipping two methods with the same name.
- The menu tests looked menus up **positionally** (`menuBar().actions()[2]`), so adding the View menu broke three unrelated tests. Now looked up by name via a `_menu_by_title` helper, which will not break next time.

959 tests passing (953 -> 959).

## F1 linear prediction implemented; contour ceiling corrected to 0.25x (2026-09-06)

### F1 linear prediction

`io.bruker.apply_recorded_f1_linear_prediction` now extends the F1 interferogram exactly as TopSpin recorded, driven by `proc2s`'s `ME_mod`/`NCOEF`/`LPBIN`. **27 of the 54 2D datasets here ask for it and this project applied none.**

`LPBIN` counts **real** points (the same convention as `TD`) -- established by measurement rather than assumption: extending to `LPBIN/2` complex points reproduces TopSpin's F1 FWHM of **4** exactly on `Cyclosporine/203`, while treating `LPBIN` as a complex count over-narrows it to 1.

Effect on the dataset that prompted this (the "slanted" Echo-Antiecho):

| | F1 FWHM | tilt | corr vs TopSpin |
|---|---|---|---|
| no LP | 8 rows | -0.1006 | +0.3951 |
| **with LP** | **4 rows** | **-0.0665** | **+0.4454** |
| TopSpin | 4 rows | -0.0018 | -- |

So the F1 linewidth now matches TopSpin exactly and the tilt is substantially reduced -- but not eliminated, which is consistent with the recombination still being wrong. Wired into the GUI's live rebuild too, so recompute-vs-load stays **0.000000** across all five real datasets.

Forward-only modes (`ME_mod` 1/2) are implemented; the backward ones (3/4) reconstruct points *before* the start, a different operation, and no dataset here uses one -- deliberately not guessed at. `linear_predict_f1=False` skips it, since it is genuinely expensive (one AR fit per F2 column, ~1.5s for a 4096-column dataset).

### A real bug this exposed: unstable linear prediction

Three of the five LP datasets came back with peak magnitudes of **1e85 to 1e104** against an expected ~1e11, driving their correlation with TopSpin to *exactly* 0.0000. The unconstrained least-squares AR fit was putting poles outside the unit circle, so the "prediction" grew exponentially instead of extrapolating a decaying signal.

Fixed with **root reflection** (`_reflect_unstable_roots`): any pole outside the unit circle is replaced by `1/conj(r)` -- same frequency, decaying magnitude -- and the coefficients rebuilt. This is what TopSpin and NMRPipe do. Magnitudes are now sane (1e11/1e8/1e5) and nothing collapses to zero. Cost is 0.6 -> 0.9 ms per column.

**Scoped to forward prediction only**, and that distinction is subtle enough to be worth stating: backward prediction reverses the FID first, so a normally-decaying signal becomes a *growing* series whose poles legitimately sit outside the circle. Reflecting them predicts a *smaller* signal before t=0 when the true one is larger. Caught by the existing backward-reconstruction test, which reproduces a known signal and started failing by 75%.

### Contour ceiling corrected

`_MAX_CONTOUR_POINTS` is now defined as **a quarter of the EXSY benchmark's own point count** (`1002 * 65536 // 4` = 16,416,768), per direct instruction: "restrict any dataset to 0.25x the EXSY's number of points". It had been briefly set to half on an earlier reading of "at least half"; 0.25x is the preference. The EXSY now renders at 1002 x 16384, and its test asserts the *relationship* to the benchmark rather than a literal, so it survives future edits.

953 tests passing (948 -> 953).

## Positive projections; and the Echo-Antiecho "slant" measured and diagnosed (2026-09-06)

User: "the contours in the echo-antiecho are somehow slanted, and I'm not sure why. Also, the 2D projections should probably just be the positive projections (not the sum)."

### Projections -- done

`projection_mode="auto"` now resolves to `"max"` for **every** 2D, not just pseudo-2D, and `"max"` is the plain positive maximum. A sum projection of a phase-sensitive spectrum cancels its own positive and negative lobes and washes the peaks out, which is what prompted this. `"sum"` remains available explicitly. `_signed_max_magnitude` (added a few rounds ago to make `max` phase-sensitive) is now unused and was **deleted along with its three tests** rather than left as dead code -- `np.max` is phase-sensitive anyway, so the property that helper existed for is retained.

### The slant is real, and now quantified

Measured via the strongest peak's own 2D second moments:

| | F1 width | F2 width | F1-F2 covariance | tilt |
|---|---|---|---|---|
| ours | 4.48 rows | 2.41 pts | **-1.083** | **-4.32 deg** |
| TopSpin | 1.21 rows | 1.42 pts | -0.003 | axis-aligned |

So the observation was correct: our peaks are genuinely tilted (correlation coefficient -0.100) where TopSpin's are square to the axes.

### Root cause 1, confirmed: TopSpin uses linear prediction in F1 and we do not

`proc2s` for Cyclosporine/203 records **`ME_mod=2.0`** (linear prediction) with `NCOEF=32`, `LPBIN=512`, `TDeff=512` -- TopSpin extends F1 from its 128 complex points to 512 by LP before zero-filling to 1024. This project applies no F1 LP at all, which is exactly why our F1 lines are **~4x broader** (4.48 rows against 1.21).

This matters beyond appearance: **it caps how high the correlation against TopSpin can go regardless of whether the recombination is right**, which makes whole-array correlation a poor metric for the remaining Echo-Antiecho work. `processing.ft.linear_predict` already exists, so wiring F1 LP is a well-defined, independent improvement -- and probably the next thing to do.

### Root cause 2: still the unresolved recombination -- and a useful negative result

Tilt is a far more sensitive probe than correlation, so the recombination space was re-searched scoring by tilt. A candidate does reach TopSpin's tilt exactly:

    EA transform OFF, decode = -((a+b) - 1j*(a-b))   ->  tilt +0.0022 (TopSpin: -0.0018)

**But it is wrong**, and the check that caught it is worth keeping: its peaks land at F1 mirror positions (828 -> 196, 789 -> 235, 791 -> 233, 738 -> 286 in a 1024-point F1), matching only **3/8** of TopSpin's strongest peaks against the current form's **5/8**. Zero tilt was achieved by *symmetrising the mirror artifacts*, not by fixing the reconstruction.

**Lesson: tilt is necessary but not sufficient.** Any future candidate has to be scored on tilt **and** peak positions together -- either alone is satisfiable by a wrong answer.

Echo-Antiecho therefore remains unsolved at **+0.395**. The concrete next step is F1 linear prediction, both because it is correct in its own right and because it removes the linewidth confound from every subsequent measurement.

## Vendor conventions normalised at read time: `processing/` is now vendor-free (2026-09-06)

User: "many of the phase conventions etc should be fixed if the GUI reformats any read data into a common file type so it is then agnostic for vendor". Done for the two leaks that existed. Output is **identical on all five real datasets**, so this is purely structural.

### 1. The two phase models were provably redundant -- now one

Measured, not assumed: for the descending ppm axis every reader produces,

    apply_recorded_phase(s, PHC0, PHC1)  ==  apply_phase(s, ppm, -PHC0, +PHC1, pivot_frac=1.0)

to **5e-16**. `pivot_frac=1.0` puts the pivot at the array's first point, which is exactly what an index ramp starting at `PHC0` means; the `-PHC0` accounts for `apply_phase`'s `exp(-i*phi)` against TopSpin's `exp(+i*phi)`.

So `apply_recorded_phase` is no longer a second implementation of phasing -- it is now a thin, named *conversion* onto the single model, verified identical to the old ramp across 1D/2D/3D shapes and both axes. The conversion stays behind a name because the sign and pivot flips are measured rather than obvious, and belong in exactly one place.

### 2. `conjugate_f1` removed -- and it was not what it looked like

The flag looked like a leaked artifact of Bruker's up-front `np.conj(data)`. It is not. Testing a **fully conjugate-free** pipeline (plain F2 reversal after the FFT, no conjugate anywhere, recorded phase negated to absorb the difference) reproduces today's results exactly -- QF **+0.9999**, States-TPPI **+0.7075** -- and **still requires the flag**. So Bruker's F1 quadrature sense genuinely differs from X-Pulse's; it is a real vendor convention, not an artifact.

That makes the right fix a *data* normalisation rather than a flag: the reader now negates the sine-modulated rows before decoding, which is exactly equivalent (`a - 1j*b == a + 1j*(-b)`), and `processing.ft.decode_hypercomplex_f1` is back to `(data, fn_mode)` with **no vendor knowledge at all**. A test asserts its signature stays that way.

Also established along the way, and worth not re-deriving: `FFT(conj(x))` equals `conj(reverse(FFT(x)))` **only with a one-sample roll** -- exact to 1.6e-16 with it, wrong by O(1) without. That is the modular-wrap term this project had hit before.

### A real bug the test suite did not catch

The first version normalised for `fn_mode in (4, 5, 6)`. Echo-Antiecho (6) uses a different recombination that this sign is not part of, so it flipped real Cyclosporine/203 data from **+0.395 to -0.380** -- while **all 948 tests still passed**. Found only by the real-data check that runs alongside the suite. Now scoped to `(4, 5)`, with a regression test covering the distinction.

### Verification

All five real datasets match their pre-refactor values exactly (PFG +0.9999, T2 +0.9972, EXSY +0.7021, babaDQ +0.7049, EA 203 +0.3951), the recompute-vs-load acceptance test is still **0.000000** across 1D, pseudo-2D, both hypercomplex modes and magnitude mode, and 949 tests pass (948 -> 949).

### Not normalised (deliberately, for now)

The **frequency-direction** convention -- Bruker's global `np.conj(data)` versus a post-FFT reversal -- is left as-is. Both were measured as exactly equivalent for QF and the conjugate-free variant is marginally *better* for States-TPPI (+0.7075 vs +0.7021), but switching it changes what "zero phase" means for every existing spectrum and interacts with the recorded phase, so it wants its own pass rather than being bundled here.

## Normalise vendor conventions at read time (original plan, now superseded above)

User: "I think many of the phase conventions etc should be fixed if the GUI reformats any read data into a common file type so it is then agnostic for vendor, like we discussed."

**This is the right diagnosis, and there is now direct evidence for it.** The vendor-agnostic core model (`NMRSpectrum`) has existed since the start, but *conventions* were never normalised -- only the container was. The leaks are concrete and were all hit while debugging 2D:

- `processing.ft.decode_hypercomplex_f1` now takes a **`conjugate_f1` vendor flag** (Bruker `True`, X-Pulse `False`), because `io.bruker` conjugates the whole raw FID up front while `io.xpulse` does not. A "vendor-agnostic" processing function needing a vendor flag is the smell itself.
- **Two phase models** coexist (`apply_recorded_phase`'s TopSpin index ramp vs `processing.ft.apply_phase`'s pivot model), so `metadata.raw` has to carry `recorded_phase*` and `phase*` separately and `apply_dimension_phase` exists purely to compose them in the right order.
- Bruker's global `np.conj` (`_CONJUGATE_QUADRATURE_NOTE`) exists to fix F2's frequency direction, and then has knock-on effects on F1 that took several rounds to untangle.

**The proposal**: each reader normalises to one canonical internal convention at read time -- canonical quadrature sense, canonical frequency direction, and the *recorded* phase converted into the single internal phase model -- so `processing/` and the GUI never branch on vendor at all. Readers become the only place vendor knowledge lives, which is what the original layering intended.

**Why it is worth doing now rather than after stage 2**: the unsolved Echo-Antiecho reconstruction is exactly a convention problem (see below -- two coupled sign/direction errors, where each candidate fix interacts with the global conjugate and with the raw-row transform). Normalising first would collapse that search space instead of leaving each hypothesis entangled with three others. It is plausible this is a prerequisite, not a follow-up.

**Not started.** Scope needs its own pass: pick the canonical conventions, convert each reader, remove `conjugate_f1` and ideally one of the two phase models, with the existing byte-identity and recompute-vs-load checks as the acceptance tests.

## Stage 2, part 2: Echo-Antiecho -- raw-row transform proven wrong, decode still unsolved (2026-09-06)

The user supplied **`Cyclosporine/203` reprocessed with `xf2` only into procno 2** -- exactly the F2-only ground truth for hypercomplex data that was missing. Confirmed genuine: **256 nonzero rows** (matching the 256 raw rows) zero-filled to 1024, so its `2rr`/`2ir` really is "F2 processed, F1 untouched".

**Result 1 -- our F2 is correct for Echo-Antiecho, and the `fn_mode == 6` raw-row transform is wrong.** Comparing our F2-processed rows against procno 2 row by row, the ratio to TopSpin is:

| | magnitude ratio, row 0 |
|---|---|
| **without** the conj+time-reverse | **0.015 .. 0.016** (constant -> a pure phase difference) |
| with it | 0.012 .. 0.087 (scattered) |

A constant magnitude ratio with a per-row phase is exactly right: that residual per-row phase *is* the F1 evolution. So F2 is sound, and `data[0::2] = conj(data[0::2])[:, ::-1]` -- added empirically in an earlier round to suppress F1 mirror ghosts -- is not what TopSpin does.

**Result 2 -- removing it alone does not help, because the decode is also wrong.** End-to-end against TopSpin's procno 1, across both settings and six candidate recombinations, the best is **+0.6166** (`-1j*b`, with the transform still on). Every standard Rance-Kay form sits at 0.30-0.55. **`-1j*b` was deliberately not shipped**: it discards the echo row entirely, which would halve SNR in a *sensitivity-enhanced* experiment -- physically implausible, and far more likely an artifact of a second error than a real recombination.

So Echo-Antiecho has **at least two coupled errors** (the raw-row transform and the recombination), and testing them jointly over this candidate set does not reach a credible answer. This is the concrete reason the convention-normalisation above may be a prerequisite rather than a nicety.

**Still open**: Echo-Antiecho at **+0.395** as shipped. States-TPPI at **+0.70** after part 1.

## Stage 2, part 1: States-TPPI F1 quadrature sign identified and fixed (+0.03 -> +0.70) (2026-09-06)

The decode's States/States-TPPI recombination had F1's quadrature sense backwards for Bruker. Found by solving for it against ground truth, not by guessing.

### Two corrections to the previous entry's conclusions

**1. "F2 is wrong for States-TPPI" was wrong -- that was SNR, not a bug.** The dataset used (`exam_CMCse_3/19`) is a natural-abundance INADEQUATE with a **2rr peak SNR of 38.5**, against 13212 for the Cyclosporine HSQC. Ranking every hypercomplex dataset with all four blocks by SNR found far better States-TPPI targets -- the user's own EXSY data at **SNR 1194-2083**. On those, our raw row `2k` matches TopSpin's interferogram row `k` at **+0.999**: F2 was fine all along. **Lesson: check the validation target's own SNR before concluding the code is wrong -- a weak experiment cannot distinguish a correct pipeline from a broken one.**

**2. `TDeff` is not the cause either.** Tested (truncating before windowing, which is what TopSpin does): +0.5272 -> +0.5238 for the INADEQUATE, and 0.997203 -> 0.997225 for `bruker_T2/11`. No meaningful effect; not implemented.

### The actual bug, identified by least squares

Solving `interf[k] ~ c1*a + c2*b + c3*conj(a) + c4*conj(b)` against TopSpin's recovered F1 interferogram on high-SNR EXSY data fits with **residual 0.0015-0.0022**, and the coefficients come out unambiguous:

| k | coefficients | meaning |
|---|---|---|
| 0 | `+a` | (b unidentifiable -- the sine component is ~0 at t1=0) |
| 1 | `-a + 1j*b` | `-(a - 1j*b)` |
| 2 | `+a - 1j*b` | `+(a - 1j*b)` |

i.e. **`(-1)**k * (a - 1j*b)`**, where the code computed `(-1)**k * (a + 1j*b)`. The sine-modulated component's sign was inverted.

**Measured end-to-end against TopSpin on three independent datasets**: EXSY 2019 **+0.034 -> +0.702**, EXSY 2011 **+0.121 -> +0.688**, babaDQ 2016 **+0.142 -> +0.705**.

### It is a vendor convention, and the shared tests proved it

Flipping the sign outright broke `test_read_xpulse_ntuples_2d_recovers_known_f1_frequency` -- X-Pulse shares `decode_hypercomplex_f1`, and its own end-to-end test recovers a *known* F1 frequency, so that failure is authoritative rather than incidental. Bruker needs the opposite sense because `io.bruker` conjugates the whole raw FID up front (`_CONJUGATE_QUADRATURE_NOTE`); X-Pulse does not conjugate.

So it is exposed as `decode_hypercomplex_f1(..., conjugate_f1=False)`, **defaulting to the historical behaviour** so X-Pulse is untouched, with Bruker's two call sites (the reader and the GUI's live rebuild, kept in step as always) passing `True`. **Lesson: a shared function's cross-vendor tests are the cheapest possible check that a "fix" is not really a vendor convention in disguise.**

### Still open

**This is a real improvement, not a solution**: +0.70 is not +0.99. Something further remains in the F1 chain (window, zero-fill, or the F1 recorded-phase model). Echo-Antiecho is untouched at **+0.395** -- its own least-squares fit (residual 0.0022) reduced to `interf[k] ~ -1j*b`, the antiecho row alone, for every `k >= 1`, which matches no textbook Rance-Kay form and is still unexplained. Do not implement it without understanding it.

948 tests passing (946 -> 948).

## Rebuild stage 2 (investigation): F1 isolated against ground truth (2026-09-06)

Stage 2 started. **No code changed** -- this round built the diagnostic that isolates F1 and used it to narrow the problem substantially. Two findings are solid enough to build on; the fix itself is not yet identified.

### The diagnostic: recover TopSpin's own F1 interferogram

A fully 2D-processed dataset stores all four hypercomplex blocks (`2rr`, `2ir`, `2ri`, `2ii`), so TopSpin's own **complex** F1 spectrum can be assembled, its recorded F1 phase and window undone, and the result inverse-FFT'd along F1 to recover **the interferogram TopSpin actually fed into its F1 transform**. That is precisely the quantity `decode_hypercomplex_f1` produces, so the decode can be compared against ground truth *in isolation* instead of end-to-end.

### Bruker's block naming, resolved empirically (was guesswork before)

Both orderings were tried and scored by whether `2rr + i*<block>` behaves as a genuine complex pair (smooth phase across a strong row):

| pairing | phase-step std |
|---|---|
| `2rr + i*2ri` | **0.0006 rad** |
| `2rr + i*2ii` | 0.0072 rad |
| `2rr + i*2ir` | 0.0981 rad |

So **the first letter is F1 and the second is F2**: `2ri` is the F2-imaginary partner, and therefore **`2ir` is the F1-imaginary** -- the block needed for F1 work. Worth recording, because the two orderings are easy to assume and produce plausible-looking but wrong references either way. (Consistent with the xf2-only datasets, where the F2-imaginary lives in `2ii` because F1 has no real/imaginary distinction there at all.)

### Echo-Antiecho: F2 is right, and the recombination is *exactly* linear

- **F2 is correct at row level**: our raw row `2k` matches TopSpin's interferogram row `k` at **+0.999** (row 0), decaying gently to +0.925 by row 5 as signal decays -- i.e. agreement is limited by SNR, not by processing.
- **The recombination is an exact linear combination.** Solving `interf[k] ~ c1*a + c2*b + c3*conj(a) + c4*conj(b)` per increment (strong-signal points only) fits with **residual 0.0022**. That is a solved-form answer, not a fit that merely looks decent.
- **But the solution is surprising**: for every `k >= 1` the coefficients collapse to `interf[k] ~ -i * b`, i.e. **the odd (antiecho) row alone**, with `a`, `conj(a)` and `conj(b)` all at ~0.000. `k = 0` is the exception (both rows contribute). This does not match any textbook Rance-Kay recombination and is the concrete lead to chase next -- either TopSpin performs the echo/antiecho combination inside its own transform (so what is recovered here is post-recombination), or the `2ir` block carries something other than assumed. **Do not implement `-i*b` on the strength of this alone**; understand it first.
- Separately measured: the `fn_mode == 6` conj-and-time-reverse of even raw rows is **near-neutral** for F2 (profile 0.832 with it, 0.840 without). It was arrived at empirically in an earlier round to kill mirror ghosts, and this suggests it may be compensating for something else rather than being correct. Worth re-deriving.

### States-TPPI: fails much earlier, and differently

- **F2 itself is wrong here** -- F2 profile correlation against TopSpin is only **+0.52** (against +0.84 for EA and +0.9999 for the QF/xf2 datasets). Peak positions only partly coincide (bins 1312 and 1686 shared; 713/1472/1478 present in TopSpin but not ours, 1533/1893/1958 in ours but not TopSpin's).
- **No linear combination of our row pairs explains TopSpin's interferogram, under any pairing tried**: interleaved `(2k, 2k+1)`, blocked `(k, k+N/2)`, and reversed interleave all give residuals of **~0.999** -- i.e. no fit at all, versus 0.0022 for EA. Since a linear model that succeeds for EA fails completely here, **the problem is upstream of the decode**, consistent with the poor F2 profile.
- Note this dataset has F2 `TD=8192` but `SI=2048`, i.e. TopSpin **truncated** F2 rather than zero-filling it -- unlike every dataset where F2 currently verifies. That asymmetry is the first thing to check.

### Confirmed unchanged

The global `np.conj(data)` is genuinely required for hypercomplex data too: dropping it collapses the F2 profile from +0.84 to **-0.06** (EA) and +0.52 to **+0.02** (States-TPPI).

### Where to resume

1. Explain the Echo-Antiecho `-i * b` result before implementing anything from it.
2. Fix States-TPPI's **F2** first (start with the `SI < TD` truncation case); its decode cannot be judged until then.
3. Re-derive the `fn_mode == 6` raw-row transform, now that it is known to be near-neutral for F2.

## Rebuild stage 1 SHIPPED: recorded and user phase separated, F2 now +0.9999 by default (2026-09-06)

User: "start on that. Make sure everything is implemented cleanly and without redundancy." The deferred half of stage 1 is done -- the measured TopSpin convention is now the shipped default, with the GUI consistent with it.

### The design

Two phase models have to coexist, so they are stored and applied separately rather than merged:

* **recorded** -- the file's own `PHC0`/`PHC1` in TopSpin's index-ramp convention (`apply_recorded_phase`). Describes the file's own state.
* **user** -- whatever the operator has dialled since, in the pivot-based model (`processing.ft.apply_phase`) that makes interactive first-order phasing rotate about a chosen point.

Merging them into one number would make neither reproducible, which is exactly what the old code did (`phase0` held `PHC0` *and* the user's value, applied through the pivot model).

**One composition point**: `io.bruker.apply_dimension_phase(data, ppm, recorded, user, pivot_frac)` -- recorded first, user on top, on the last axis. Called by `read_1d`, `read_2d` (F2), and the GUI's three raw-FID rebuild sites. **One accessor**: `io.bruker.recorded_phase(raw, dim)`, with the key names held in `RECORDED_PHASE_KEYS`/`USER_PHASE_KEYS` so nothing re-derives them. Verified no direct key access survives anywhere outside those.

### Which sites actually needed changing -- three, not six

Auditing the six candidate sites showed only those that **rebuild from the raw FID** need the recorded phase re-applied (the reader's own application lives in the spectrum they replace, not in `fid_raw`):

| site | needs it | why |
|---|---|---|
| `_process_raw_fid_row` | **yes** | rebuilds a row from `fid_raw` |
| `_recompute_multi_row_data` F2 | **yes** | rebuilds every row from `fid_raw` |
| `_recompute_multi_row_data` F1 | **yes** | rebuilds F1 from `fid_raw` |
| `_recompute_multi_row_phase_and_baseline_only` | no | the *processed-only* fallback -- TopSpin already baked the phase into `2rr` |
| per-row live preview | no | applies the live delta to already-phased data |
| `_reapply_phase_post_reload` | no | runs after a fresh read, so the reader already applied it |

Getting this wrong in either direction double-applies or drops the phase, so the audit mattered more than the edit.

### Results

**Shipped default vs TopSpin, F2-processed datasets (n=17): median +0.9999, best +1.0000, 14/17 above +0.99** -- against a median of **+0.067** before this work. `bruker_PFG/150` went +0.513 -> +0.9999 and `bruker_T2/11` +0.816 -> +0.9972. F2's `PHC1` is now honoured for 2D as well (it was dropped entirely; `read_1d` always read both), and F1 has a recorded-phase step for the first time -- the "indirect phase is for sure not loading across from processing" report.

One outlier remains at +0.0262 (`20200804_NaMgMnO.../31`), not yet investigated.

**The acceptance test that blocked this last time now passes exactly.** Live rebuild vs load, relative difference:

| | before | after |
|---|---|---|
| 1D | -- | **0.000000** |
| T2 pseudo-2D | 0.46 | **0.000000** |
| PFG pseudo-2D | 0.76 | **0.000000** |
| Echo-Antiecho genuine-2D | -- | **0.000000** |
| States-TPPI genuine-2D | -- | **0.000000** |
| magnitude-mode COSY | 1.93 | **0.000000** |

Pinned as a parametrised regression test over all five real datasets, verified to discriminate (4 of 5 fail when the fix is reverted).

**A third divergence found and fixed while testing**: `_recompute_multi_row_data` never applied **magnitude mode**, so a magnitude-mode spectrum silently reverted to phase-sensitive the moment any control was touched (the 1.93 above). Same class as the phase gap -- a rebuild path not matching the reader -- so it is fixed the same way, calling the reader's own `_apply_magnitude_mode` last.

### Also cleaned up

- **Zeroing is now exact.** "Zero Phase 0/1" and "Reset" clear the *recorded* half at its source instead of trying to cancel it with an equal-and-opposite user phase -- which could never work for first-order phase, since the two use different models. `Reset` is now just both Zero calls rather than a third copy of the same logic. Both trigger a rebuild, since the recorded phase is applied during reconstruction.
- **The phase display** shows recorded + user + live delta, so "current phase" means what is actually applied.
- `recorded_phase*` is deliberately **not** blocklisted from the processing sidecar: once Zero can clear it, it is a user choice that has to survive a reload.

946 tests passing (940 -> 946). **Stage 2 (hypercomplex F1) unchanged and still open**: Echo-Antiecho +0.395, States-TPPI +0.034.

## Rebuild stage 1 (investigation): TopSpin's phase convention measured exactly (2026-09-06)

User: "start this rebuild". Stage 1 -- **validate and fix F2 in isolation** -- is solved, and the answer overturns an earlier conclusion in this file.

### The enabler: a genuine F2-only reference

`bruker_T2/11` and `bruker_PFG/150` are processed by TopSpin with **`xf2` only** -- `proc2s`'s `SI` equals its `TD`, so F1 was never transformed. Their `2rr` is therefore exactly "F2 processed, F1 untouched", which makes F2 testable without any F1 involvement. Better still, both store a **`2ii`** block that is the F2 *imaginary* partner (confirmed: identical size to `2rr`, and a smoothly varying phase across a peak), so `2rr + 1j*2ii` is TopSpin's own **complex** F2 spectrum -- and the per-point ratio against our own unphased output gives the applied phase **with no fitting at all**.

### The convention, measured not fitted

On `bruker_PFG/150` that ratio has **constant magnitude** (15.997..16.003 -- a pure phase difference, the 16 being `NC_proc` scaling) and a first-order fit residual of **0.001 degrees**:

| | measured | recorded |
|---|---|---|
| p0 | **-10.250** | `PHC0` = 349.742 (= -10.258 mod 360) |
| p1 | **-80.115** | `PHC1` = -80.100 |

So TopSpin's convention is exactly **`phi(i) = PHC0 + PHC1 * i/(SI-1)`**, applied as `exp(+i*phi)` across the transformed dimension's own points, in array order.

**This overturns the earlier "the recorded PHC values don't transfer" conclusion** (see the entry below). They transfer *exactly*. What was wrong was the **model**: this code routes them through `processing.ft.apply_phase`, which is pivot-based and normalised over the *ppm* range. That earlier round's measurements were correct about that model; the inference drawn from them was too broad. **Lesson: "applying the recorded constants makes it worse" falsifies the model you applied them with, not the constants.**

Applying the measured convention end-to-end:

| dataset | shipped | measured convention |
|---|---|---|
| `bruker_PFG/150` | +0.513 | **+0.9999** |
| `bruker_T2/11` | +0.816 | **+0.9972** |

And across the whole corpus, the F2-only (xf2-style) comparison now reaches **median +0.9999, best +1.0000 over 17 datasets**. F2 is solved.

**A self-inflicted near-miss worth recording**: an intermediate check had T2 apparently unfixable (best 0.087 over a full first-order scan). Cause was my own harness reading `2rr` with a plain `np.fromfile(...).reshape()`, bypassing the **XDIM submatrix block reconstruction** (`procs XDIM=4096`, `proc2s XDIM=4`) that `read_2d_processed` correctly performs. PFG has `proc2s XDIM=1` and so was unaffected, which is why one dataset "worked" and the other didn't. Always compare against `read_2d_processed`, never a raw `fromfile` of `2rr`.

### What shipped, and what deliberately did not

**Shipped**: `io.bruker.apply_recorded_phase(spectrum, phc0, phc1, axis)` -- the convention above, with the full measurement recorded in its docstring, plus regression tests including one that reproduces the +0.999 agreement on real `bruker_PFG/150` data and asserts it beats the shipped path by >0.4.

**Deliberately NOT shipped: the default-behaviour switch.** Wiring it into `read_1d`/`read_2d` was implemented and verified (the numbers above are from that build) and then **reverted**, because it made the GUI's live recompute diverge from the load -- measured at **0.46** (T2) and **0.76** (PFG) relative difference, i.e. touch any control and the spectrum jumps. The GUI applies the committed phase through the pivot model at six separate sites (`main_window.py` 3160, 3381, 3495, 3537, 4247, 4271); making the reader speak a different convention than those without converting them all is precisely this project's recurring "two pipelines quietly diverge" failure. Shipping half of it would have traded a known-wrong-but-stable state for an inconsistent one.

**The remaining work, now well-defined**: separate the *recorded* phase (Bruker convention, from the file) from the *user's* phase (pivot convention, from the controls) -- store them distinctly in `metadata.raw` rather than merged into `phase0`/`phase1` -- and have every reconstruction path apply recorded-then-user in that order. That is a mechanical but wide change across the reader and those six GUI sites, and it needs its own pass with the recompute-vs-load consistency check above as the acceptance test.

### Stage 2 (hypercomplex F1) remains open

With the correct F2 phase applied, `Cyclosporine/203` (Echo-Antiecho) improves from **-0.5433 to +0.395**, and States-TPPI datasets stay near 0. So F2 was a genuine and necessary prerequisite -- exactly as the user predicted ("the F2/F1 phasing doesn't need to be decoupled") -- but it is not sufficient: the hypercomplex F1 construction has its own separate defect still to find.

940 tests passing (937 -> 940).

## 2D-mode robustness survey: EVERY phase-sensitive 2D mode is wrong, not just Echo-Antiecho (2026-09-06)

User: investigate the Cyclosporine 203 HSQC defect, and "search through the sample bruker data for any other 2D processing modes to test our robustness". Plus a design steer: "the F2/F1 phasing doesn't need to be decoupled, maybe this is the difference to the bruker data. However it does need to apply properly in the 2D contour."

### The survey

54 datasets with a raw `ser`. Modes present:

| FnMODE (acqu2s) | MC2 (proc2s) | count |
|---|---|---|
| QF | QF | 25 |
| **Echo-Antiecho** | Echo-Antiecho | **15** |
| States-TPPI | States-TPPI | 11 |
| undefined | QF | 3 |

Echo-Antiecho is the **second-largest class**, not an edge case. No TPPI (3) or plain States (4) data exists here, so those two decode branches remain unvalidated against real data.

### The finding: this is much broader than the reported HSQC

Correlating `read_2d`'s genuine-2D output against `read_2d_processed` (TopSpin's own `2rr`), each dataset in its own recorded mode:

| mode | n | median | worst | best |
|---|---|---|---|---|
| Echo-Antiecho | 1 | **-0.543** | -0.543 | -0.543 |
| States-TPPI | 6 | **-0.022** | -0.104 | +0.000 |
| QF | 20 | +0.067 | -0.304 | **+0.861** |

**Essentially every phase-sensitive 2D reconstruction is uncorrelated with TopSpin.** The single good result (+0.861) is the magnitude-mode COSY -- and it is good *because* magnitude discards phase entirely. This is not a Cyclosporine-203 problem; it is the genuine-2D path in general.

### Hypotheses tested and eliminated

- **Not the recorded F1 phase.** `read_2d` never applies `proc2s`'s `PHC0`/`PHC1`, but applying Cyclosporine 203's own (`PHC0=-1.7`) moves it from -0.5433 to **-0.5464**. Worse.
- **Not any F1 phase.** Best achievable by a freely fitted F1 `(phase0, phase1)` is **+0.5574** for the HSQC and **+0.0030** for the States-TPPI INADEQUATE. A global F1 phase cannot fix either.
- **Not axis orientation.** F1-reversed, F2-reversed, both-reversed, half-rolled and fftshifted variants all drop the HSQC to |corr| < 0.01. The current orientation is right.
- **Not the coupling fix on its own.** The user's steer is well-founded in principle -- nmrglue's and NMRPipe's own order is *phase F2, delete imaginary (`di`), then build F1* -- and this code instead feeds the **complex** F2 data into `decode_hypercomplex_f1`. Implemented that (`Re()` of each F2 component before forming the F1 point) and measured: HSQC -0.5433 -> **-0.4027**, States-TPPI +0.0004 -> **-0.0005**. Better for one, no change for the other; not a fix.
- **Not phase at all, even jointly.** A full 4-parameter fit over F2 `(p0,p1)` and F1 `(p0,p1)`, tried both coupled and decoupled, tops out at **+0.4104** (coupled) and **+0.5549** (decoupled) for the HSQC.

### The real diagnostic, and where the next attempt should start

For Cyclosporine 203 the **magnitude** correlation is **+0.6074** while the signed correlation is **-0.5433**. Peaks are largely in the right *places* but with wrong signs -- and no global phase fixes that. Critically, even the magnitude only reaches 0.61, so the reconstruction differs beyond phase as well.

That combination (right positions, wrong signs, phase-unfixable, magnitude also imperfect) points at the hypercomplex construction itself rather than at any phase step. **The F2 phase being wrong is the more likely upstream cause**: an earlier round established that F2's recorded `PHC0`/`PHC1` do not transfer to this reconstruction at all (see that entry), and for hypercomplex data F1 is built *from* the F2-phased result -- so an F2 phase error propagates into F1 in a frequency-dependent way that no single F1 phase can undo. Getting F2 right is therefore a prerequisite, not an independent task; the two are coupled exactly as the user said.

**Not fixed.** Four separate hypotheses were measured and none held; shipping any of them would have been a guess. This needs a focused rebuild of the hypercomplex path validated against TopSpin at each stage (F2-only, then decode, then F1), rather than end-to-end correlation fitting.

### Shipped from this round: two real robustness crashes, found by sweeping all 156 Bruker experiments

- **Missing `2rr` crashed with `AttributeError: 'NoneType' object has no attribute 'ndim'`** from inside `NMRSpectrum`'s constructor, naming neither the file nor the experiment. `_load_block` returns `None` for a missing file -- correct for the *optional* `2ri`/`2ii` blocks, wrong for the real block. Real dataset: `Bruker Example Data/Cyclosporine/206/pdata/1` holds `procs`, `proc2s`, `clevels` and a thumbnail but no `2rr`. Now a clear `FileNotFoundError`. **Pre-existing, not introduced by any recent change** (confirmed against `HEAD`).
- **An experiment with neither raw nor processed data** (`.../2010`: `acqus` and `pdata/1`, but no `fid`, `ser` or `1r` -- an aborted acquisition) failed with a bare errno-2 from `np.fromfile`. Now the same clear message.

After both, **155 of 156 Bruker experiments load cleanly** through `_read_auto`; the one failure is the genuinely empty `2010`, which now says so. 937 tests passing (935 -> 937).

## Four-item round: zero-phase semantics, density cap raised, own binary reader (nmrglue warning gone) (2026-09-06)

Three of four items done; the fourth (Cyclosporine 203's indirect phase) turned up a **bigger, separate defect** than the one reported -- quantified below, not fixed.

- ~~"Zeroing phases should actually zero the values, not just reset to the loaded values."~~ -- done, and the report was exactly right. The Phase spin boxes hold only the *delta* since the last `_capture_phase_reference`; the total in force is that delta **plus** whatever is committed in `metadata.raw` (on a freshly loaded Bruker spectrum, the file's own recorded `PHC0`/`PHC1`). Setting the delta to 0 therefore returned to the *loaded* phase -- on a spectrum not yet dragged, it visibly did nothing at all. Both "Zero Phase 0/1" and "Reset" now set the delta to `-committed`, driving the total to zero while leaving every existing preview/commit/persistence path untouched (rather than reaching into `metadata.raw` behind their back). New `_phase_committed_value(order)` helper reads the same F1/F2-aware keys `_refresh_current_phase_displays` already uses, so the two can't disagree about which dimension is being edited.
- ~~"Set a max value of data density that is at least half of those EXSYs."~~ -- done. `_MAX_CONTOUR_POINTS` raised from 16,777,216 (2^24) to **33,554,432** (2^25); the pseudo-2D EXSY is 1002x65536 = 65,667,072, so half is 32,833,536 and the new ceiling clears it. The EXSY now renders at 1002x33487 (about half its F2 resolution rather than a quarter), costing ~2.0s of contour time against ~4.0s at full resolution. Every other real dataset remains untouched, as before. A new test pins the "at least half the EXSY" property against the constant so it can't quietly be walked back.
- ~~"It seems we're still using nmrglue's fileio... I still get these errors"~~ -- **correct, and now fixed for the binary read.** To be precise about what was and wasn't rewritten: the *JCAMP-DX parameter* parser (`io/jcampdx.py`) has been this project's own from the start, but the raw `ser`/`fid` **binary** read still went through `nmrglue.bruker.read`, which warns and returns a flat, unshaped array whenever the requested shape doesn't divide the file exactly -- the `(39583744,) cannot be shaped into (1002, 39488)` message. That was cosmetic (this module immediately truncated and reshaped it, always safe here since the shape's element count is derived from that same array's length) but it surfaced to users on real EXSY data and read as a genuine error. New `_read_bruker_binary` reads the file directly with the same dtype/endianness rules nmrglue itself applies (`BYTORDA == 1` -> big-endian, `DTYPA == 2` -> float64; confirmed by reading its source). **Verified byte-identical to nmrglue on every real Bruker dataset in `test_data`: 54/54 `ser` files and 100/100 `fid` files, zero mismatches** -- and it additionally *succeeds* on one dataset (`.../2019`'s sibling `.../5`) where `ng.bruker.read` itself raises `IndexError`. The warning is confirmed gone on 2011 and 2019. `nmrglue` is still used deliberately for the parts it has genuinely validated against real Bruker output -- digital-filter/group-delay removal and the window functions -- and for the acqus parameter dicts.

**Not fixed -- Cyclosporine 203's HSQC is worse than "the indirect phase isn't loading across".** The recorded F1 phase genuinely isn't read (`proc2s` `PH_mod=1`, `PHC0=-1.7`, `PHC1=0.0`, and `read_2d` has no F1 recorded-phase step at all -- see the entry below). But applying it does **not** fix this dataset, measured: correlation against `read_2d_processed` is **-0.5433** as shipped and **-0.5464** with the recorded `PHC0=-1.7` applied. The output is *anti-correlated* with TopSpin's, and the best achievable by **any** F1 phase is only **+0.5574** -- so unlike the T2/PFG case (where a fitted phase reaches 0.9995), phase alone cannot explain this one. This is `FnMODE=6` (Echo-Antiecho) with `MC2=5`, the hypercomplex path with the documented conjugate-and-time-reverse row-pair handling; the defect is most likely in that reconstruction, not in phasing. **Needs its own focused investigation** -- flagged rather than patched, since applying the recorded F1 phase would have looked like an answer while leaving a 0.55-correlation spectrum in place.

935 tests passing (928 -> 935).

## Tried applying the recorded `PHC1` to 2D -- it makes things WORSE, not shipped (2026-09-05)

User: "Let's try it" (re: applying F2's recorded `PHC1`, which `read_1d` applies and `read_2d` does not). **Tried, measured, and deliberately abandoned** -- the naive fix is a large regression. No code changed; this entry is the result.

**Measured against `read_2d_processed` (TopSpin's own `2rr`), row 0, correlation:**

| dataset | recorded PHC0 / PHC1 | shipped now (PHC0 only) | applying recorded PHC0+PHC1 | best achievable by *any* phase |
|---|---|---|---|---|
| `bruker_T2/11` | -205.704 / 908.800 | **+0.848** | **-0.241** | +0.99947 |
| `bruker_PFG/150` | 349.742 / -80.100 | **+0.513** | **-0.134** | +0.99995 |

Applying the recorded values takes both datasets from usable to *anti-correlated*. Shipping that would have been a clear regression, caught only because the change was measured against real data before being written.

**But the residual genuinely is a phase error, and it is large.** A freely fitted `(phase0, phase1)` reaches **0.99947** and **0.99995** -- essentially perfect -- so nothing else (window, zero-fill, scaling, baseline) meaningfully differs from TopSpin. This is a real, fixable defect worth pursuing, not an acceptable limitation; `bruker_PFG/150` in particular ships at only 0.513.

**Why the recorded values don't transfer -- ruled out, so the next attempt doesn't repeat this:**
- Not a pivot convention: scanned `pivot_frac` 0.0/0.25/0.5/0.75/1.0 and both signs. Best over that whole grid was +0.863 against a +0.848 baseline -- marginal, and at an arbitrary point.
- Not a simple scale/sign relationship: fitted-vs-recorded first-order ratios are **-0.900** (T2) and **-1.083** (PFG). Inconsistent between datasets, so there is no constant convention factor to apply.
- Not frequency-axis reversal: converting the fit into Bruker's own index order (`PHC0' = p0+p1`, `PHC1' = -p1`) leaves T2 off by ~91 degrees on *both* terms and PFG off by 104 and 6.6 -- again inconsistent.
- Not residual group delay: T2 records `GRPDLY=-1` (no digital filter at all) yet needs 2.27 points of delay to match; PFG records `GRPDLY=76` and needs -0.24. No correspondence.

**Recommended next step, not started**: rather than trying to transplant TopSpin's own recorded constants into a differently-reconstructed spectrum, give `read_2d` the joint phase0/phase1 autophasing `read_1d` already has (`processing.ft.autophase01`; `read_2d`'s `phase_mode="apc0"` currently fits phase0 only). The fits above show a correct phase exists and is findable from the data alone -- which is a more robust route than reverse-engineering a convention that two datasets already disagree about.

## Projections were phase-blind: `max(abs())` replaced with signed max-magnitude (2026-09-05)

User: "The 2D data still isn't matching the phase window phasing info", clarified mid-investigation as "specifically the projections".

**Confirmed with an exact, not subtle, mechanism.** `plot_2d_contour`'s `"max"` projection mode -- which `"auto"` selects for *every* pseudo-2D, i.e. the default for essentially every 2D in this app -- computed `np.max(np.abs(data), axis=...)`. Absolute value is **phase-blind by construction**, so the projections could not respond to phasing at all. Measured on real data (`bruker_T2/11`): applying `phase0=180` degrees inverts every peak, changing the phase window's own slice by 2.0 (a complete inversion) while both projections changed by **exactly 0.00000**.

| phase0 | F2 projection change | F1 projection change | phase-window slice change |
|---|---|---|---|
| 45 deg | 0.55747 | 0.15220 | 0.55754 |
| 90 deg | 0.96357 | 0.33995 | 1.25751 |
| **180 deg** | **0.00000** | **0.00000** | **2.00000** |

**Fixed** with a new `_signed_max_magnitude(data, axis)`: per column, the single value with the largest *magnitude*, keeping its own sign. After the fix the same measurements track the slice (45 deg: 0.564, 90 deg: 1.272, 180 deg: 2.025 against the slice's 2.0).

Why not the obvious alternatives, both rejected for concrete reasons rather than taste: `sum` is phase-sensitive but washes out the slice-to-slice amplitude changes that `max` was chosen for in the first place (a relaxation/diffusion series exists to show exactly those); plain `max(data)` is phase-sensitive but silently loses negative peaks, returning a wholly-negative column's *least* negative value -- the noise floor -- instead of its peak. Signed max-magnitude keeps all three properties, and is the same rule `plotting.style.downsample_2d` already uses for the same reason.

**No visual change for correctly-phased data**: for an all-positive array the two rules are *identical* (asserted directly, and confirmed against real data), so the difference appears only where there is genuine negative or dispersive signal -- exactly when it is needed.

4 new regression tests, including an end-to-end one through `plot_2d_contour` verified to fail on the old `max(abs(...))` version (where the two projections were bit-identical before and after a sign flip). 928 tests passing (924 -> 928).

**Also confirmed while investigating, still NOT fixed** (see the entry further down): `read_2d` reports and applies `phase0` from the file's `PHC0` but leaves `phase1` at **0.0** while the file records a real value -- `PHC1=77.15` for EXSY 2019, `PHC1=908.8` for `bruker_T2/11`. `read_1d` reads and applies both. So for those datasets the phase window honestly reports what was applied, but what was applied is missing TopSpin's own first-order term entirely. This is a *separate* mismatch from the projection bug above and is the most likely remaining source of "the 2D doesn't match" reports -- **ask before changing**, since it alters the default phase of every 2D load; `PH_mod` (now read, see the magnitude round) is available to gate it correctly.

## Magnitude mode implemented: recorded `PH_mod` honoured, plus a manual toggle for anything (2026-09-05)

User: "if magnitude mode is used somewhere, you can implement that. Or better yet, do that and have the option to switch it on for anything." Both done, commit `cf58b51`.

**It is genuinely used**: surveyed every Bruker dataset in `test_data` -- **9 record `PH_mod=2` (mc/magnitude)**, mostly gradient-COSY experiments (`exam_CMCse_3/12`, `exam_CMCse_1/2`, `exam_CMCse_2/3`+`/5`, `Menthyl-Anthranilate/20`+`/71`, `exam_CMCse_3/14`+`/16`, and two 2H datasets). None record `PH_mod=3` (ps/power). Ignoring it is exactly what made those load as an unphased, dispersive-looking phase-sensitive spectrum: the COSY correlated **+0.0107** with TopSpin's own processed output, against **+0.8612** for the magnitude of the same complex data.

All four Bruker readers (`read_1d`, `read_2d`, `read_1d_processed`, `read_2d_processed`) take `magnitude=None|True|False`: `None` follows the recorded `PH_mod`, `True` forces it on for any spectrum (1D or 2D), `False` forces it off even for a file that asks for it. Applied **last**, after windowing/FFT/phasing in every dimension -- which is where it was verified to reproduce TopSpin; taking it between F2's and F1's transforms is a different, wrong quantity. For a 2D, either dimension's parameters may carry the flag (the COSY records mc on F1 while F2 says "no"), but a pseudo-2D's `proc2s` is excluded since its indirect dimension was never transformed. `data_imag` becomes `None`, which is already how the GUI disables its phase controls -- a magnitude spectrum has no phase left to correct.

**GUI**: a "Magnitude Mode" toggle next to "Process Indirect Dimension", shown for Bruker 1D and 2D. It **reloads** rather than recomputing in memory -- magnitude is a discrete on/off with no responsiveness argument for a second implementation, and this project has repeatedly been bitten by "two pipelines quietly diverge". The user's explicit choice is stored as `magnitude_override`, separately from the reader's descriptive `magnitude_mode`, because "off" and "never set" are otherwise indistinguishable: without it, switching magnitude off on a `PH_mod=mc` file would silently turn back on at the next load. `_load` reads it before the read, for the same reason `pseudo_2d` is decided there. `magnitude_mode_source` is blocklisted from the sidecar (re-derived every load; "requested" describes only the session that set it).

**Two real bugs found while building this, both caught by running the tests rather than by inspection:**
1. The processed-data readers didn't accept `magnitude`, so `_read_auto` forwarded it into them and raised `TypeError` on every processed-fallback dataset -- which surfaced as a *hung test run*, because the GUI handler's error path opens a modal dialog. Fixed by supporting it there too (and adding `magnitude` to the fallback's own kwarg filter).
2. A file that records magnitude mode has **no `2ri`/`1i` on disk** -- TopSpin's stored data is already the magnitude (confirmed: every value in the COSY's `2rr` is non-negative). The first version's `spectrum_imag is not None` guard therefore silently reported no mode at all for exactly those files, so the GUI button would have shown "off" for a magnitude dataset. Now the mode is recorded for the UI and the data passed through untouched, rather than having `abs()` applied a second time; an explicit user request still applies `abs()`, since a real-only array is all there is.

**Benchmarks unaffected**: EXSY 2019, bruker_T2 and bruker_PFG all record `PH_mod=1`/`0` and still load phase-sensitive with `data_imag` retained.

9 new regression tests. 924 tests passing (916 -> 924). **Still open** from the entry below: `read_2d` ignoring F2's recorded `PHC1` and having no F1 recorded-phase step -- unchanged by this round, and now with `PH_mod` available to gate it correctly.

## F1 projection was plotted with inverted polarity (2026-09-05)

User: "The phase of the contours and the projections are definitely inverted and doesn't match the processing window", clarified mid-investigation as "it's more about the matching of the contour phase to the projection phase".

**Confirmed and fixed.** `plot_2d_contour` plotted the F1 (left) projection as `-f1_projection` while the F2 (top) projection plotted its own un-negated. A positive (black-contour) peak therefore appeared as a *negative-going* trace in the F1 strip, and a negative (red-contour) peak as positive -- so the two projections disagreed with each other, and the F1 one disagreed with the contour sitting beside it. Proven with a controlled synthetic fixture carrying one known-positive and one known-negative peak at different F1/F2 positions: before the fix, F2 read `+4913 / -5114` (correct) while F1 read `-5114 / +4913` (inverted); after, both read `+4913 / -5114`.

**The negation was doing real layout work, and that is preserved**: this panel sits to the *left* of the contour with intensity along x, and negating made positive peaks point leftward, away from the contour, instead of into it. Fixed by reversing the *axis* rather than the data -- plot the true values on a descending xlim -- which points the peaks the same way while leaving the plotted numbers honest. `MplCanvas._zoom_projection_intensity` needed no change: it was already written in terms of pinning whichever end `xmax` is, and the baseline is still that end (its docstring, which described the old negation, was updated).

3 new regression tests, all verified to discriminate by restoring the negated version and confirming they fail. 916 tests passing (913 -> 916).

## Found while investigating the above, NOT fixed -- read_2d ignores recorded phase that read_1d honours

Two real gaps, found by comparing `read_2d`'s genuine-2D output against `read_2d_processed` (TopSpin's own trusted result) on the COSY. Flagged rather than fixed, because both change the default phase of every 2D load and the user's report had already been narrowed to the projection issue above -- **ask before changing**.

- **F2's recorded `PHC1` is never applied.** `read_1d` reads both (`phase0 = procs["PHC0"]`, `phase1 = procs["PHC1"]`, `bruker.py:347-348`) but `read_2d` reads only `PHC0` (`bruker.py:661`). Real effect: the EXSY 2019 dataset records `PHC0=309.42, PHC1=77.15` -- the first-order term is silently dropped, so its F2 is left with a real, visible phase error that TopSpin itself does not have.
- **F1 has no recorded-phase step at all.** There is no `proc2s.get("PHC0")`/`proc2s.get("PHC1")` anywhere in `read_2d`. The COSY's strongest TopSpin peaks are cleanly absorptive (all `+1.3e8` to `+1.7e8` across the peak) while `read_2d` swings `-6.8e8` to `+5.9e8` over the same points -- a dispersive lineshape, i.e. an unphased indirect dimension, not an inverted one.

**Also unhandled, and relevant to judging the above**: `PH_mod` is never consulted. The COSY records `procs PH_mod=0` (no F2 phase correction) and `proc2s PH_mod=2` (**magnitude mode** in F1) -- so TopSpin displays that dataset in magnitude mode, which is why its peaks are uniformly positive. Any fix for the two gaps above should read `PH_mod` first rather than unconditionally applying `PHC0`/`PHC1`, or it will be wrong for exactly this class of dataset.



## Big-data loading performance: profiled, planned, and the top item shipped (2026-09-05)

User: "Plan out some methods to make the loading of big data faster and add it to the list." Profiled a real load first (`cProfile` against the "2019" EXSY dataset, `2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119/2019`, 2048x65536 points -- the same dataset from the T1-noise investigation elsewhere in this file) rather than guessing where the time goes, per this project's own standing discipline.

**Benchmark, fixed as the reference point for all future load-speed work**: `io.bruker.read_2d` on the "2019" EXSY dataset above (2048x65536 points), OS file cache warm -- **baseline 9.268s** (`cProfile`-measured, before any of the changes below). Re-measure against this exact dataset and this exact baseline number for any future load-speed change, not a different file or a re-run of an already-modified baseline.

**Finding: `io.bruker.read_2d` spends ~9.3s total on this dataset, and ~6.2s of that (67%) is in six `np.apply_along_axis` calls that could be replaced with native batched numpy calls with no change in output.** F2's window/zero-fill/FFT (`bruker.py:634`/`638`, `645`, `646`) loop once per raw row (1002 calls here, confirmed via the profile's own per-line call counts); F1's window/zero-fill/FFT (`bruker.py:690`/`696`, `703`, `704`) loop once per F2 *column* instead (65536 calls here) -- the F1 loop is the more extreme case, since each call only processes a short column, so fixed Python/array-allocation overhead dominates even more heavily relative to actual FFT work than in F2's loop. Concretely, of the 6.2s: `numpy.fft`'s own C-level compute (`_raw_fft`) accounts for only ~3.3s of it -- the remaining ~2.9s is pure per-call Python/wrapper overhead from doing 66,000+ individual one-row/column-at-a-time calls instead of one batched call.

All three per-row steps are either already batch-safe or trivially made so -- checked by reading the actual implementations, not assumed:
- **FFT** (`processing/ft.py::fft_spectrum`) is *already* fully batch-safe -- its own docstring says as much ("there's no reason this function itself has to be wrong about its own output shape for 2D input"); it already uses `np.fft.fft(fid, axis=-1)`/`fftshift(..., axes=-1)`, which numpy already batches correctly across a leading axis. Callers just need to stop wrapping it in `apply_along_axis` and call it directly on the whole 2D array (transposed first for F1, since that's an axis-0 operation and the shared function operates on the last axis).
- **Window**, the recorded-WDW path (`_apply_bruker_window`) is *also* already batch-safe for free -- confirmed by reading `nmrglue.proc_base.em`/`sp`'s own source directly, both use `data.shape[-1]` and broadcast (`apod * data`), so they already work correctly on a full 2D array unchanged. The custom-window override path (`processing.ft.apodize`) is *not* yet safe -- it computes `weights` from `len(fid)`, which reads `shape[0]` (wrong axis) for a 2D input; needs the same `shape[-1]` fix `fft_spectrum` already got, per that function's own documented history of this exact class of bug.
- **Zero-fill** (`processing.ft.zero_fill`) is *not* yet batch-safe -- both its length check and its `np.pad` width tuple assume 1D input. Needs a shape-aware pad-width (`[(0,0)]*(fid.ndim-1) + [(0, n - fid.shape[-1])]`-style construction) rather than just a `len()` swap.

**Plan, in priority order:**
1. ~~Fix `apodize`/`zero_fill` to be axis/shape-aware~~ -- done. `apodize` now sizes its window from `fid.shape[-1]` instead of `len(fid)` (the exact class of bug `fft_spectrum` itself already had fixed, per that function's own docstring); `zero_fill` now truncates/pads its last axis specifically (`fid[..., :n]`, a shape-aware `np.pad` width) instead of assuming 1D. Both changes are no-ops for every existing 1D call site (`len(fid) == fid.shape[-1]` whenever `fid.ndim == 1`) -- confirmed by the full existing suite passing unchanged. 2 new regression tests (`test_zero_fill_batches_2d_along_last_axis`, `test_apodize_batches_2d_along_last_axis`) checking the batched result matches calling per-row, one row at a time, exactly.
2. ~~In `read_2d`, replace the 6 `apply_along_axis` calls with direct whole-array calls~~ -- done. F2's window/zero-fill/FFT now run once each on the whole `(n_rows, n_f2)` array; F1's own three transpose once (`spectrum.T`), run the same whole-array calls, transpose back once -- rather than 65536 separate per-column calls. **Verified two ways, not just re-profiled**: (a) re-profiling the exact benchmark dataset above gives **4.966s** (down from the 9.268s baseline -- a ~46% reduction on this run; three repeated direct-timing runs landed 5.4-7.0s, some variance, all well under the 9.3s baseline); (b) a git-stash-based exact A/B on the same real file -- saved `read_2d(...).data` before and after this change, byte-identical (`np.array_equal` True, max abs diff `0.0`) -- the vectorization changed *only* how the arrays are computed, not what they contain. Remaining time is now dominated by `_remove_digital_filter` (~1.9s, nmrglue-internal, untouched by this fix) and the FFT's own genuine compute cost (~3.2s total across every FFT call in the read, batched or not -- same FLOPs either way); window/zero-fill overhead no longer shows up in the profile at all. Full suite: 898 passing (896 -> 898).
3. Apply the identical fix to `_recompute_multi_row_data` (`main_window.py`) -- it has its own explicit per-row Python loop for the same reason (the "two pipelines quietly diverge" pattern this project keeps hitting, see the rigorous-audit entry just below). Lower priority than `read_2d` itself since it runs on already-loaded data during interactive phase editing, not cold load, and its loop also handles echo-alignment/linear-prediction, which genuinely are row-dependent -- likely needs a "fast batched path only when neither is active" split rather than unconditional vectorization. **Not started.**
4. Once batched, consider `scipy.fft` (with `workers=-1`) in place of `numpy.fft` for the now-single big FFT calls -- multi-threading only helps once there's one large call to thread, not 66,000 small ones, so this is only worth doing after step 2. **Not started** -- now unblocked by item 2 above landing, and planned in detail in the threading section below.

**Lower-confidence/heavier-weight ideas, not scoped in detail yet:**
- `_remove_digital_filter` (nmrglue's own `remove_digital_filter`/`fsh2`) accounts for ~1.7s (19%) and is mostly outside this project's own control -- it's already using two single-shot batched FFTs internally, not a per-row loop. Large-array allocation overhead shows up here too (`np.zeros_like` alone costs ~0.8s across the load, for arrays this size -- ~2GB per complex128 copy at this dataset's dimensions -- though the profile's flat call listing doesn't pin down exactly which call sites those 6 allocations belong to, so this is noted, not fully traced). *Speculative, unverified*: for datasets where `GRPDLY` happens to be a whole number of points, a plain integer roll could skip the frequency-shift FFT method entirely -- not checked how often that's actually true across this project's real test data, and not worth shortcutting a validated correctness-critical step without checking first.
- A background-thread load (`QThread`) wouldn't reduce compute time but would keep the GUI responsive (progress indicator, cancellable) during whatever time remains after the above -- worth doing once the actual remaining compute time is known, so the UX design matches the real wait rather than today's.
- Caching the reconstructed array (keyed to file mtime + processing params, e.g. alongside the existing sidecar) would only help *re*-opening the same large file, not first load, and adds real cache-invalidation complexity -- lowest priority tier.
- The already-known, already-documented **Compare-tab 1024-subplot rendering cost** (~25s, matplotlib doesn't scale to 1000+ subplots -- see the "Loading the 2D ser..." entry further down this file) is squarely part of "big data feels slow to open" from the user's perspective even though it's display-side, not `read_2d` itself -- a candidate fix (not scoped) is rendering very large slice counts as a single 2D image/heatmap instead of N separate line-plot subplots, or lazily rendering only the visible/nearby slices.

## Adaptive contour ceiling + threaded FFT shipped; contour threading ruled out (2026-09-05)

User: "Do it" (re: the contour render, then the threading items). One item shipped, one closed as **not possible**, one escalated as a user decision.

### Shipped: `scipy.fft` with a bounded, overridable worker count

`processing.ft.fft_spectrum` now uses `scipy.fft.fft(..., workers=fft_workers())` instead of `numpy.fft`. Same pocketfft underneath -- the switch buys only the `workers` parameter. `read_2d` on the benchmark: **4.385s serial -> 3.630s threaded**; full GUI load 8.02s -> 7.42s (a smaller share now, since the load is no longer FFT-dominated).

**Accuracy gate, checked before anything else**: output is **bit-identical to `numpy.fft`** at workers 1/2/4/8 across five real array shapes (`np.array_equal`, not `allclose` -- a tolerance would hide exactly the thread-count-dependent drift worth ruling out). Threading splits the batch across independent 1D transforms, so no individual transform's summation order changes. The five-dataset end-to-end A/B is still byte-identical to the **original pre-all-fixes baseline**.

Worker policy follows the system-agnosticism contract written in the threading plan below:
- `_available_cpus()` uses `os.process_cpu_count()` (3.13+) -> `os.sched_getaffinity` (Linux only, probed not assumed) -> `os.cpu_count()` -> 1. Never bare `os.cpu_count()`, which reports the machine rather than this process's allowance and is wrong under cgroup limits and affinity masks.
- Capped at 4 (`_FFT_WORKER_CAP`), never `workers=-1`: the measured F2 plateau is 4, and scipy/numpy's own BLAS/OpenMP pools are already live in-process, so uncapped threading risks oversubscription for no measured gain.
- `OPENNMR_FFT_WORKERS` pins the count; `1` selects the fully serial path, which is measurably numpy's own performance and bit-identical output, so "threads off" is genuinely supported rather than degraded. A malformed or non-positive value falls back to automatic rather than raising -- a bad env var must not stop a spectrum loading. **The full suite passes under `OPENNMR_FFT_WORKERS=1` as well as the default** (908 tests, both ways).

### Ruled out with evidence: threading the contour render

The threading plan below flagged `contourpy(thread_count=...)` as promising-but-unverified. **Now verified, and it is a dead end** -- two independent reasons, either sufficient:

1. **matplotlib never passes it.** `matplotlib/contour.py` constructs `contourpy.contour_generator(x, y, z, name=..., corner_mask=..., line_type=..., fill_type=..., chunk_size=self.nchunk)` -- no `thread_count`, and no public route to inject one. (`thread_count` also only does anything with chunking enabled, which matplotlib leaves at `nchunk=0` by default.)
2. **contourpy does not release the GIL.** Measured directly: 14 levels on a `(1002, 16384)` array take 7.76s serially, **8.28s** across 4 threads with a generator per call, and **7.99s** across 4 threads sharing one generator -- i.e. 0.94x and 0.97x. Threading contour generation cannot help from Python at all, whatever API is used.

**Also measured and worth recording**: the contour cost is linear in *levels x points*, not in the number of `ax.contour()` calls. Collapsing the positive and negative passes into one call with per-level colours was tried and gives **no speedup at all** (4.019s -> 4.066s), because contourpy calls `create_contour` once per level regardless -- and it also silently loses the dashed-negative visual cue, since matplotlib only applies `negative_linestyles` when the level set is signed the way two separate calls make it. Not shipped.

### Escalated rather than decided: contour resolution

With threading ruled out, **resolution is the only remaining lever** on the ~4.0s contour. Measured on the benchmark (880x660 px canvas), downsampling F2 with the existing max-preserving `downsample_2d`:

| F2 points | contour time | speedup | pixels differing >2% |
|---|---|---|---|
| 65536 (full) | 4.012s | 1.0x | -- |
| 32768 | 1.977s | 2.0x | 0.102% |
| 16384 | 0.995s | 4.0x | 0.108% |
| 8192 | 0.499s | 8.0x | 0.177% |
| 4096 | 0.251s | 16.0x | 0.191% |
| 2048 | 0.126s | 32.0x | 0.186% |

The difference is small and does not grow much with further downsampling (consistent with anti-aliasing on the contour lines rather than lost features -- `downsample_2d` keeps each bin's largest-magnitude value since the earlier EXSY-diagonal fix, so thin ridges survive). **But it is not zero, and the full-resolution Spectrum tab is an explicit standing user instruction** ("I need to have the fully sampled spectrum in the spectrum tab") -- so rather than silently trading it away under a "make it fast" mandate, the options were put to the user, who chose **"adaptive: full res unless huge."**

### Shipped: an adaptive contour ceiling (`_cap_contour_points`)

`_prepare_contour` now clamps the resolution handed to matplotlib to `_MAX_CONTOUR_POINTS` (16,777,216) total, on top of whatever the caller asked for. Deliberately a safety valve for outliers, not a downsampling policy: **every other dataset in this project's own test data is far below it and comes back as the bit-for-bit identical full-resolution array** -- verified directly, not just by shape (COSY 2048x2048, bruker_T2 64x32768, bruker_PFG 16x262144 all `np.array_equal` to `spectrum.data`). Only the 65.7M-point EXSY class is capped, to 1002x16743.

The reduction comes out of F2 first (`_MIN_CAPPED_CONTOUR_X` floor of 4096), since F1 is typically far smaller and far more sensitive to losing rows -- the EXSY keeps all 1002 F1 rows. `spectrum.data` itself is never touched; the cap affects only what is handed to the contour generator, so every downstream consumer (slices, fitting, projections, phase) still sees full resolution. Peak magnitude survives exactly (`abs().max()` identical before and after capping), since `downsample_2d` is max-preserving.

**EXSY benchmark load: 7.42s -> 4.77s.** Cumulative for the whole effort: **17.4s -> 4.8s (3.6x)**, with `spectrum.data` byte-identical to the original pre-all-fixes baseline on all five datasets throughout.

**A real bug in this function's own first draft, caught by its own test rather than by review**: reducing F2 alone isn't always sufficient, because F2 has a floor -- a request whose F1 is itself oversized (e.g. a hypothetical 65536x65536) came back **16x over the ceiling**. Now it reduces F1 too once F2 is at its floor. Verified the test discriminates by restoring the buggy logic and confirming the failure, then restoring and `diff`-confirming byte-identity.

9 new regression tests. 913 tests passing (908 -> 913).

## Both traced duplications fixed: 17.4s -> 8.0s per load, byte-identical output (2026-09-05)

User: "Fix them however you think is best", with a follow-up steer mid-work: "The goal is accuracy and speed." Both duplications from the trace entry below are fixed. **Output is byte-identical on all five real datasets checked** (`np.array_equal` True, max abs diff `0.0`) -- accuracy was treated as the gate, speed as the result.

| dataset | before | after | |
|---|---|---|---|
| EXSY 2019 (2048x65536) | 17.08s | **8.02s** | 2.13x |
| COSY genuine 2D | 0.242s | 0.100s | 2.42x |
| bruker_T2 pseudo-2D | 0.414s | 0.205s | 2.02x |
| bruker_PFG pseudo-2D | 0.806s | 0.429s | 1.88x |
| Bruker 1D | 0.059s | 0.035s | 1.69x |

Re-running the duplication trace afterwards confirms **every** duplicate is gone: `_draw_selected_spectrum`, `read_2d`, `_remove_digital_filter`, `_read_raw_2d_data`, `_refresh_selection_display` and `_prepare_contour` are all 1 call now, down from 2. (The one remaining N-per-load pattern, `get_trace` x1002 at 0.122s, is inherent to populating the Compare list and was already ruled not worth touching.) The test suite itself got faster as a side effect: 93s -> 67s.

**Fix 1 -- redraw once, in `_select_single_row`.** The three list mutations now run inside `blockSignals(True)`/`blockSignals(False)` (in a `try`/`finally`, so an exception can't leave the widget permanently signal-deaf), followed by one explicit `_on_list_selection_changed()` call. Deliberately fixed at the single call site that emits both signals for one programmatic selection, rather than by removing `_on_list_selection_changed`'s own explicit `_on_selection_changed` call -- that call is load-bearing for the multi- to single-selection case its comment describes, and every user-driven selection still goes through the normal unblocked path untouched. `_on_list_selection_changed` (not `_on_selection_changed`) is the right single entry point because it dispatches the single-vs-multi branch itself and its single-selection branch already runs the pseudo-2D Compare-population step too.

**Fix 2 -- read once, with the right flag.** `_load` now reads this experiment's saved pseudo-2D preference from `path` *before* loading (every vendor reader sets `metadata.source_path` to exactly the path handed to it, so the sidecar is addressable without having read the file yet) and passes `pseudo_2d=` into the first `session.add`. `io.bruker._read_auto` drops `pseudo_2d` on its 1D branches, which is what makes passing it safe before anyone knows whether the path is 1D or 2D. Scoped to Bruker deliberately: `io.xpulse.read_2d` has no `pseudo_2d` parameter at all, and the other vendors keep the original read-then-reload path rather than being switched on an assumption about readers this trace never measured.

**A real bug in the first draft of fix 2, caught by its own regression test rather than by inspection**: the old read-then-reload branch was left reachable, and for a *saved genuine-2D* preference (`pseudo_2d: False`) the freshly-loaded spectrum is correctly not pseudo-2D -- which satisfied `not spectrum.is_pseudo_2d` and triggered a second, byte-for-byte identical read, reintroducing the exact duplicate the change exists to remove, for the one case that doesn't take the common path. Now gated on `"pseudo_2d" not in add_kwargs`.

**A flaw in my own verification harness, worth recording because it nearly produced a false alarm**: the first A/B pointed *every* dataset's sidecar at one shared temp file, so each dataset's saved state leaked into the next one's load and 4 of 5 datasets appeared to "change". Nothing was wrong with the product code -- the harness needed per-source-path isolation (hash the path into the filename). The standing sidecar-isolation practice in this project is "redirect to a fresh temp dir"; that is not sufficient when a single run loads more than one dataset.

5 new regression tests, each verified to discriminate by reverting its own fix independently and confirming the failure, then restoring and `diff`-confirming the file came back byte-identical. 903 tests passing (898 -> 903).

## Traced: every duplicated/redundant call in a big-data load -- 54% of the wall time is doing work twice (2026-09-05)

User: "Can you trace all the duplicated and redundant calls that slow us down?" Done by **instrumenting the real GUI load** (wrapping each expensive function in place, recording every call's duration plus its opennmr-side caller chain, then running `MainWindow._load` on the EXSY benchmark dataset) rather than reading call sites and inferring -- so every claim below has a caller stack behind it. Traced, not fixed: nothing here is implemented yet.

**Total: `_load` = 17.4s, of which ~9.4s (54%) is duplicated work.** Only two root causes, each with one clear mechanism:

### 1. The spectrum is fully redrawn twice -- ~4.24s wasted

`_draw_selected_spectrum` runs **twice** per load, 4.265s + 4.238s, on the same spectrum and same row, through the identical `show_spectrum` -> `plot_2d_contour` path. The two caller chains differ by exactly one frame, which is what identifies the mechanism:

```
call 1:  _select_single_row:1225 -> _on_list_selection_changed:1948 -> _on_selection_changed:7445 -> _draw_selected_spectrum
call 2:  _select_single_row:1225 -> _on_selection_changed:7445 -> _draw_selected_spectrum
```

`MainWindow` connects **two** signals from the same widget to two handlers that both end in a redraw (`main_window.py:752-753`): `currentRowChanged -> _on_selection_changed` and `itemSelectionChanged -> _on_list_selection_changed`. `_select_single_row`'s own `setCurrentRow(row)` (line 1225) fires **both**. `_on_list_selection_changed`'s single-selection branch then calls `_on_selection_changed(...)` *explicitly* -- deliberately, with a comment giving a real reason ("Multi- -> single-selection doesn't always change `currentRow()` ... so `currentRowChanged` wouldn't fire on its own"). That reasoning is correct for the multi->single case it was written for, but the call is unconditional, so in the common case -- a fresh load, where `currentRowChanged` *does* fire -- the spectrum gets drawn twice. **This is a guard for one case silently doubling the work in another**, not a stray call; a fix needs to keep the multi->single path working (that's what its own comment is protecting) rather than just deleting the explicit call. `_refresh_selection_display` is duplicated by the same mechanism (0.376s across 2 calls, minor by comparison).

### 2. Every genuine 2D file is read from disk twice -- ~5.19s wasted

`read_2d` runs **twice**, 5.432s + 2.824s:

```
call 1 (5.432s):  _load:1054 -> session.add:35        -> io.load -> _read_auto -> read_2d          # genuine 2D
call 2 (2.824s):  _load:1095 -> session.reload:50     -> _load_with:84 -> ... -> read_2d           # pseudo-2D
```

`_load` calls `session.add(path)` (a full genuine-2D read, F1 FFT included), then -- for any 2D that isn't already pseudo-2D -- `session.reload(index, pseudo_2d=wants_pseudo_2d)`, and `Session.reload` calls `_load_with`, a **complete second read from disk**, not an in-memory reinterpretation. The first read's entire result is discarded; the spectrum that survives is `pseudo_2d: True`, shape `(1002, 65536)`.

Measured directly to size the waste precisely, rather than assuming the two reads cost the same: **the discarded genuine-2D read is 5.188s; the pseudo-2D read that's actually kept is 2.826s.** So loading once with the right flag would cost 2.83s instead of 8.01s. The pseudo-2D default itself is correct and deliberate (see the "Pseudo-2D-by-default" round) -- the waste is paying for a genuine-2D reconstruction in order to *discover* that. `_load` already reads `saved_metadata` early (before the pseudo-2D branch, for an unrelated ordering reason it documents), so the deciding information is available before `session.add` is ever called. Knock-on duplicates that vanish with this one: `_remove_digital_filter` (2 calls, 3.62s), `_read_raw_2d_data` (2), `nmrglue.bruker.read` (2, 0.311s).

### Explicitly ruled out (checked, not assumed)

- **Redundant parameter-file parsing: not a problem.** `parse_jcampdx_params` runs 4x and `_read_procs` 4x, but that is 2x per read (`procs` + `proc2s`) purely as a consequence of duplicate #2, and it costs **0.001s total** -- there is no caching worth adding here.
- **`slicing.get_trace` runs 1002 times** (once per slice, from `_refresh_compare_view` -> `_resolve_compare_entry` populating the Compare list) but totals **0.123s** -- a real N-per-load pattern, but not a cost worth touching. The expensive part of Compare is already deferred (`_compare_view_stale`, from an earlier round) and correctly did not render here.
- **`downsample_2d` never ran** -- consistent with the Spectrum tab's full-resolution default from the earlier downsampling round.

### Sizing

Fixing both would take the benchmark load from ~17.4s to roughly ~8s, without threading, without touching any numerics, and without changing a single output value -- both fixes are "stop doing the identical work a second time." That is a larger win than every item in the threading plan below combined, which is why they belong first. Caveat carried from the threading entry: these numbers come from a headless `offscreen` Qt platform, and some matplotlib text-metric time in that profile may not be representative of a real display -- the *duplication counts* are platform-independent, but the exact per-redraw seconds may differ.

## Threading plan for big-data loading -- measured scope, and an explicit system-agnosticism contract (2026-09-05)

User: "start planning out the threading and make sure it'll be system-agnostic." Planning only -- nothing below is implemented. Profiled the **real, user-facing** GUI load first (not just `read_2d` in isolation) so the plan is scoped against what a user actually waits for, and the result substantially reframes what threading can and can't buy.

**The headline measurement, and it is not what the `read_2d` benchmark alone suggests**: `MainWindow._load` on the EXSY benchmark dataset takes **~18.0s end-to-end** (17.5s under `cProfile`), even though `read_2d` itself is now ~5s. The split:
- **~8.5s (48%) is `plotting.twod.plot_2d_contour`**, called *twice* during one load, of which ~6.9s is matplotlib/contourpy's own `_make_paths_from_contour_generator`. This is the single largest cost in a big-data load -- larger than reading the file.
- **~7.8s (44%) is `io.load`/`read_2d`, called *twice*** -- `session.add` (4.7s) then `session.reload` (3.0s). Of that, `_remove_digital_filter` is ~3.7s across the two reads.
- The remainder is sidecar/widget/selection-cascade work, individually small.

**Consequence for the threading design, stated up front because it determines everything below: threading cannot touch the 48%.** matplotlib figure/canvas manipulation and every Qt widget call must happen on the GUI thread -- that is a hard Qt/Cocoa rule, not a tuning choice (see the system-agnosticism contract below). So a background-load thread makes the ~7.8s read non-blocking, and does nothing at all for the ~8.5s render. Threading here is a **responsiveness** feature, not primarily a speed feature, and the plan should be honest about that rather than selling it as "loads get 2x faster."

### Found while measuring: two non-threading wins that should land first

Both are cheaper and higher-value than the threading work, and doing threading first would just mean threading redundant work:

1. **`_load` reads every genuine 2D file twice.** `session.add(path)` does a full read, and then -- for any 2D that isn't already pseudo-2D -- `session.reload(index, pseudo_2d=wants_pseudo_2d)` does a *complete second read from disk* (`Session.reload` -> `_load_with`, not an in-memory reinterpretation). On the benchmark that is 4.7s of genuine-2D work (including the full F1 FFT) computed and then **thrown away**, since the final spectrum comes back `pseudo_2d: True`. The pseudo-2D default is deliberate and correct (see the "Pseudo-2D-by-default" round); what's wasteful is *discovering* it after paying for the genuine-2D read. Fix direction (not yet designed in detail): decide `pseudo_2d` from the sidecar + vendor metadata *before* the first read, so the file is read once with the right flag -- `_load` already loads `saved_metadata` early for exactly this kind of ordering reason, so the information is available before `session.add` is called today. Worth ~3-4.7s on this dataset, with no threading and no numerical change.
2. **The contour render happens twice per load** and is the biggest single cost. Not yet traced to its two call sites (main canvas + Processing preview is the likely pair, unconfirmed) -- do that before optimizing. Options once traced: render once and share, defer the non-visible one the way `_compare_view_stale` already defers Compare (an established pattern in this codebase), reduce contour level count for very large arrays, or use `contourpy` directly. *Confirmed available, not yet tried*: `contourpy.contour_generator` accepts a `thread_count` parameter (contourpy 1.3.1 is what's installed) -- contour generation itself is C++-side and genuinely parallelizable even though the matplotlib artist creation that follows is not. Whether matplotlib 3.10's own `Axes.contour` exposes a way through to it is **unverified** -- check before planning around it.

### Part 1 -- Compute threading: `scipy.fft` with a bounded worker count

`scipy` is already a hard dependency (`pyproject.toml`), so this adds no new install surface. **Measured on this machine (10 cores, macOS/darwin), on the benchmark's own real array shapes:**

| shape | numpy | scipy w=1 | w=2 | w=4 | w=8 | w=-1 |
|---|---|---|---|---|---|---|
| F2, `(1002, 65536)` | 0.442s | 0.437s | 0.231s | **0.140s** | 0.145s | 0.143s |
| F1, `(65536, 2048)` | 0.434s | 0.435s | 0.234s | 0.143s | 0.112s | **0.108s** |

Three things to take from this rather than just "use `workers=-1`":
- `workers=1` matches numpy exactly -- so scipy is a drop-in with no regression risk at the serial setting, which makes a "threads off" fallback genuinely free.
- **F2 plateaus at 4 workers** (memory-bandwidth-bound: few, very long transforms) while **F1 keeps scaling to 8+** (many short transforms). A single global worker count is a compromise; the honest version is to let the two call sites differ, or accept ~3x instead of ~4x for one of them.
- Realistic total win: the FFTs are ~3.2s of the current ~5s `read_2d`, so ~2.2-2.4s back, i.e. `read_2d` roughly 5s -> ~2.7s. Meaningful, but *smaller than the double-read fix above*, which is why that is sequenced first.

*Also measured, and it is a real effect worth handling*: FFT on the **transposed (non-contiguous) view** the F1 path now produces costs 0.221s vs 0.143s contiguous at `workers=4` -- a ~1.5x penalty. An explicit `np.ascontiguousarray` before the F1 FFT may pay for itself, but it also allocates another full copy (~2GB at this dataset's size), so this needs measuring end-to-end, not assuming.

### Part 2 -- GUI threading: a `QThread` load worker

Today there is **zero** threading anywhere in `src/` (confirmed by grep -- no `QThread`, `threading`, `multiprocessing`, or `concurrent.futures`), so this is a from-scratch addition and the first concurrency in the codebase. Shape:

- A worker object doing **only** `io.load(...)` (plus, once fixed, the single correctly-flagged read), moved to a `QThread` via `moveToThread` -- not by subclassing `QThread` and overriding `run`, so the worker stays a plain testable object.
- Results come back by **signal** carrying the finished `NMRSpectrum`; every widget/canvas/`QMessageBox` call in `_load` stays exactly where it is, on the GUI thread, running in the slot.
- `Session` is already Qt-free by design (built that way from the start for the phase-2 web UI), which makes it a clean thing to call from a worker. But `Session.add`/`reload` **mutate shared lists** (`self.spectra`, `self._load_info`); the worker must not touch them. Do the `io.load` in the worker, hand the spectrum back, and let the GUI thread do the `Session` mutation.
- Progress/cancel: a cancel flag checked between stages is realistic; true mid-FFT cancellation is not, so scope the UX to "cancel between stages" rather than promising instant abort.
- Lifecycle: parent/own the thread and `quit()`+`wait()` it on window close, or Qt emits `QThread: Destroyed while thread is still running` and can hard-crash on exit.

### Part 3 -- The system-agnosticism contract (the user's explicit requirement)

Rules the implementation must satisfy, each with the specific cross-platform reason:

1. **Threads, never `multiprocessing`/`fork`, for this work.** `multiprocessing` defaults differ by platform -- `fork` on Linux, `spawn` on macOS (since 3.8) and Windows. `spawn` re-imports the parent module, which requires an `if __name__ == "__main__"` guard and behaves differently again inside a PyInstaller-packaged app (the packaging path this project already has queued). It would also need `NMRSpectrum` to pickle cleanly across the boundary. Threads sidestep every one of those, and numpy/scipy/nmrglue release the GIL during their C-level compute, so threads genuinely deliver here. **This is the single most important system-agnosticism decision in the plan.**
2. **Never create or touch a Qt widget, matplotlib figure, or canvas off the GUI thread.** Universal Qt rule, and macOS/Cocoa enforces it hardest (AppKit requires the *main* thread specifically, not merely a consistent one) -- so code that happens to work on Linux can crash on macOS. This is also exactly why the 48% render cost is not threadable.
3. **Never hardcode a worker count, and don't trust `os.cpu_count()`.** It reports the machine's cores, not what this process may use -- wrong under cgroup/container CPU limits and under CPU affinity masks (both common on Linux and in CI). `os.process_cpu_count()` is the correct API but is **Python 3.13+**, and this project is `requires-python = ">=3.10"` -- so it needs a guarded fallback (`os.process_cpu_count()` if present, else `len(os.sched_getaffinity(0))` where available -- Linux-only -- else `os.cpu_count()`, else 1).
4. **Cap the worker count; do not ship `workers=-1`.** The table above shows F2 plateauing at 4 on a 10-core machine, so `-1` buys nothing there while risking oversubscription against numpy/scipy's own BLAS/OpenMP thread pools on many-core machines. A small cap (start at `min(4, available)`, revisit with measurements on a many-core Linux box, which nobody has run this on yet) is the safer default.
5. **Provide an environment-variable escape hatch and a serial path.** `workers=1` is measurably identical to numpy, so "threads off" must be a supported, tested configuration -- for CI determinism, for constrained machines, and for debugging. Tests should pin it rather than inheriting whatever the host has.
6. **No `sleep`-based waiting in tests.** Headless GUI tests already run under `QT_QPA_PLATFORM=offscreen`; a threaded load needs deterministic synchronisation (`QSignalSpy`, or a signal-driven event loop with a timeout) or it will pass on a fast dev machine and flake in CI on a slow one -- a genuinely platform-dependent failure mode.
7. **Verify on real data, byte-for-byte, exactly as the vectorization round did.** A threaded FFT must produce output identical to the serial path (FFT is deterministic; thread count must not change results). The git-stash A/B against the benchmark dataset (`np.array_equal`, max abs diff `0.0`) is the standard to repeat -- and it is what would catch, for instance, a transpose/contiguity change quietly altering numerics.

### Suggested order

1. Eliminate the double read in `_load` (biggest single win, no threading, no numerical change).
2. Trace and fix the double contour render (biggest remaining cost; still no threading).
3. `scipy.fft` with the bounded, environment-overridable worker policy above (items 3-5 of the contract).
4. `QThread` load worker for responsiveness on whatever time remains.
5. Only then revisit `contourpy(thread_count=...)`, if `Axes.contour` turns out to expose a route to it.

## Rigorous 2D-processing-flow audit: two real data-consistency bugs found and fixed, xf2-slice default reverted (2026-09-05)

User: "I get a flicker as it saves and it seems to reload something old" while phasing any 2D, plus an explicit request for a rigorous audit ("make sure that every update/save state is updating the same data, from the same data... be extra sure about the updates and where the processed spectra read in and out from") and a design clarification ("2D processing should use slice one of the xf2... not index 1 of the fully processed data").

**Bug 1, the main suspect for the flicker**: `_recompute_multi_row_data` never applied any window function to F1 at all. `read_2d` always does (defaulting to `proc2s`'s own recorded WDW -- there's no live GUI control for F1's window at all, unlike F2's own `window_combo`). Confirmed on real COSY data (`exam_CMCse_3/12`, proc2s WDW=4/QSINE): a live recompute with *no* phase or FID-processing change at all produced data ~80% different from the original load -- not subtle. Found by first confirming F2's own per-row processing matched `read_2d`'s own primitives exactly (0.0 diff), which narrowed the search to F1 specifically.

**Bug 2**: the same method only ever read the *committed* phase0/phase1, never the live, not-yet-committed drag delta. Baseline correction is deliberately deferred to this debounced recompute (too slow to re-run every drag tick) rather than applied inline -- restarted on every tick while `baseline_correct` is on. If the 200ms debounce fired while a drag was paused but not released, this method rebuilt `spectrum.data` from raw using the *old* phase, silently reverting the live-dragged display. Confirmed: a live 77-degree drag followed by this firing showed a ~1.35e9 magnitude difference before the fix, exact match after. Fixed for both F2 and F1.

**Design reversion**: `_compute_processing_preview`'s genuine-2D branch (see the "Processing tab's 'Slice 1' preview didn't match the 2D contour" entry earlier in this file) is reverted back to xf2/raw-row-based, per direct user clarification -- that earlier fix optimized for "match the contour" over "show what F2's controls alone would do to the raw acquisition," which is this tab's actual stated purpose.

**Not yet done, flagged rather than rushed**: a numeric "choose which slice" control for genuine 2D (pseudo-2D already has `slice_spin`; genuine 2D's own `_phase_2d_slice_index` is still hardcoded to 1 with no user-facing control). "Grab a slice from the spectrum" already exists (`MplCanvas.extract_slice`, with "Show Contour" to get back) and covers part of the same request.

4 new regression tests (2 for the bugs above, verified to discriminate by reverting each fix independently; 1 replacing the now-invalid genuine-2D-preview test; the pre-existing xf2-mismatch test needed no changes, confirming the revert was clean). 896 tests passing (894 -> 896). Commit `f7d464e`, pushed.

## Five-part follow-up: X-Pulse 2D crash, real COSY F1-mirror bug (fixed), contour colour contrast, phase-button layout, projections source (2026-09-05)

User sent five items in one message, plus a direct question answered inline. In order:

- ~~"Please add colour contrast for positive and negative contours"~~ -- done. `plot_2d_contour` now draws a second, mirrored set of contour levels for negative-going data in red (matplotlib's own default dashed linestyle for negative levels adds a second visual cue on top). See commit `3e4e01d`.
- ~~"The phase zeroing buttons should be in a block with the other quick phase buttons"~~ -- done, pure layout move. See commit `d5a0e40`.
- **"This COSY... contours are flipped in the indirect dimension"** -- real, confirmed, fixed bug, genuinely different from the T1-noise investigation above (this dataset is QF/FnMODE 1, not States-TPPI). `read_2d`'s own module-level conjugate fix (`_CONJUGATE_QUADRATURE_NOTE`) was, by that fix's own honest admission, only ever verified for F2 -- "since Bruker's raw complex encoding is presumably one consistent convention library-wide, F1 too, though only F2 has actually been checked... every real dataset available while building this was pseudo-2D, which never exercises the F1 FFT branch." This COSY (`test_data/Bruker Example Data/exam_CMCse_3/12`, PULPROG `cosygpmfppqf`) is the first genuine QF-mode 2D to actually test that assumption, and it's wrong for F1: conjugating a signal before its own FFT reverses that FFT's own frequency-bin order. Confirmed directly: `read_2d_processed`'s own trusted diagonal peaks (always exactly F1==F2 for a homonuclear COSY, an unambiguous check) sat at up to ~135x higher intensity in `read_2d`'s own output once F1 was reversed to match, at 8 independently-checked positions. Fixed by reversing F1's own output array for FnMODE not in (4,5,6) -- hypercomplex modes don't need this, their own row-pair recombination already produces the correct direction (already validated end-to-end elsewhere). New regression test uses a deliberately *signed* synthetic F1 frequency and checks the recovered sign directly -- the existing States-TPPI end-to-end test only ever compared `abs(abs(peak_hz) - f1_hz)`, blind to direction, and would not have caught this. Found the same bug's sibling in `_recompute_multi_row_data` (independently rebuilds F1's FFT) via the full suite -- two existing tests cross-checking it against a fresh `read_2d` call failed until it got the identical fix. Commits `c741568`, `d5a0e40`.
- **"Importing 2D x-pulse causes a crash"** -- confirmed and fixed: `io.xpulse.read_2d` has no `pseudo_2d` parameter at all (every genuine 2D load from it is unconditionally the real thing), but the "Process Indirect Dimension" toggle was shown regardless and, on switching to pseudo-2D, forwarded `pseudo_2d=` straight into `read_2d`, raising `TypeError`. Fixed by hiding the button for this vendor, the same way it's already hidden for 1D data. Commit `d5a0e40`.
- ~~"Please check where the projections for the 2D contour are coming from"~~ -- traced directly: `plotting.twod.plot_2d_contour` sets `data = spectrum.data` once (line 162) and both the main contour *and* both projection panels (`np.sum`/`np.max` along the opposite axis) read from that exact same array -- not a separate or re-derived source. Whatever affects the main contour (correctness bugs, the T1-noise-like artifact investigated above) automatically, consistently affects the projections too, since they share one source of truth.
- **"What parameters do you need from TopSpin and from which file?"** (in answer to the still-open T1-noise-artifact investigation, not the COSY bug above -- see the "2D processing bug fixed" round further up this file) -- checked the two leading candidates directly against `proc2s` for `test_data/.../2011`: `SYMM = 0.0` (no symmetrization) and `Mdd_mod = 0.0` (no compressed-sensing/NUS reconstruction) -- both off, ruling out both theories that were on the table. The T1-noise-artifact root cause (why `read_2d_processed`'s own output doesn't show it, when nothing in `proc2s` obviously explains why) remains open and unresolved as of this entry -- distinct from, and not fixed by, the COSY F1-mirror bug immediately above, which is a different dataset (QF vs States-TPPI) and a different, now-confirmed-and-fixed root cause.

19 new/updated regression tests total (COSY F1-mirror x2 files, X-Pulse button gating, negative-contour contrast x2 files). 894 tests passing (889 -> 894, after 2 briefly regressed and re-fixed mid-round -- see above). Three commits (`c741568`, `d5a0e40`, `3e4e01d`), all pushed.

## Investigated experiment 2011's raw-`ser` shape; shipped pulse-programme name + a "loaded from processed data" note across all four vendors (2026-09-04)

User: "Can you check the read for the EXSY data in experiment 2011 from one of the test data? There was some issue with the read of the shape. Also in the read info, the experiment (pulse programme) name should be noted. It would be also good to note when the processed data is being read in instead of raw data (bruker)." Mid-turn follow-up: "you should also be able to parse the pulse programme from x-pulse and maybe varian data. I'm not sure about jeol but you can look."

**The "shape issue" investigation itself did not land on a confirmed bug** -- worth recording in detail since a lot of ground was covered, including one real mistake caught before it shipped:

- `test_data/.../2011`'s raw `ser` shape guess (`_guess_raw_2d_shape`) lands on 504 rows via the existing floor-division fallback (`acqu2s`'s own `TD=500` doesn't divide the file exactly), leaving 53% of one more row's worth of bytes unaccounted for at the end of the file. Confirmed via direct byte-level analysis, not assumed.
- First hypothesis: this leftover is itself evidence the shape guess is wrong, since an incomplete/misaligned set of t1 increments would be a real F1-correctness risk. Implemented `_RawShapeUncertain`, raised from `_read_raw_2d_data` above a 1%-of-one-row threshold, caught by `_read_auto` the same way a hard `_guess_raw_2d_shape` failure already is (falls back to `read_2d_processed`).
- **Caught before shipping, by checking the one dataset this project has relied on most**: `test_data/.../2019` (the EXSY States-TPPI dataset used extensively in the immediately preceding rounds -- pivot-fix, cyclic-phase0, the genuine-2D processing-preview fix) has an *even larger* leftover (42% of one row) on the exact same `exsy_echo3.enb` pulse sequence, and its raw path has already been cross-validated multiple times against independent references in this project's own history. The new check would have silently broken raw-FID reprocessing for it. Reverted `_RawShapeUncertain` entirely before committing anything -- see `_read_raw_2d_data`'s own docstring for the full account. **How to apply generally**: before trusting a new heuristic threshold, check it against the *already-validated* real datasets this project relies on most, not just the one dataset currently under investigation -- a threshold tuned to one file can look reasonable in isolation and still regress something else.
- Also checked and ruled out, each with real evidence rather than assumption: (a) `tdeff_f1_spin.maximum()` showing `1048576` instead of the expected `252` -- a test-ordering artifact on my own part (the control is disabled, showing Qt's raw range ceiling, until "Process Indirect Dimension" is toggled; correctly shows 252 once enabled); (b) the F2 ppm axis spanning roughly ±3200 ppm looking implausible for a 2H spectrum -- confirmed `read_2d_processed` (TopSpin's own numbers) shows the *identical* range, so this is a faithful reflection of `procs["SW_p"]` as actually recorded (plausible for a quadrupolar 2H acquisition with a deliberately wide window), not a computation bug; (c) the saved `pseudo_2d: false` sidecar setting appearing not to be restored on load -- actually just my own test isolating the sidecar path to an empty scratch directory (correct practice to avoid touching real files) rather than reflecting the real, pre-existing sidecar's content; restoring the real sidecar's content into the isolated path confirmed this actually works correctly.
- Real, useful difference found along the way: `2011` has a genuine pre-existing sidecar (`opennmr_metadata.json`/`opennmr_processing.json`, with saved phase0/phase0_f1 values) from earlier real use, while `2019` has none -- not itself an explanation for "an issue with the shape", but confirms real prior work exists on this file worth being careful not to disturb (used copies in an isolated scratch path throughout, never touched the real files).
- **Net result**: rendered both the full-range and zoomed-in contour for the properly-restored (real sidecar) state -- a plausible, if unremarkable/noisy, EXSY diagonal region, no obvious garbling. Nothing in this investigation reproduces a concrete "shape" defect; flagged back to the user for more specific reproduction detail (what was actually seen -- an error, a specific wrong-looking number, a visual artifact) rather than continuing to guess blindly.

**What did ship, real and tested**: `NMRMetadata` gained `pulse_program`, populated from the vendor's own closest equivalent to a pulse-sequence name -- Bruker `PULPROG`, Varian `seqfil`, X-Pulse `.PULSESEQUENCE`, and JEOL `experiment` (found by dumping every parameter this project's own JEOL parser exposes for a real file and checking by eye -- `experiment: 'single_pulse_dec'` -- confirmed consistent, with a sensible-looking value, across all four real JEOL test files available). All four verified directly against real data (`bruker_T2/11` -> `"hahn-echo.rjm"`, `varian_1d/...` -> `"onepul"`, `xpulse_HSQC_NUS/...` -> `"HSQC_HX"`, `jeol_1d/proton.jdf` -> `"single_pulse_dec"`), not just synthetic fixtures, before committing. Shown in the info panel alongside probe/nucleus/etc.

Separately, `io.bruker._read_auto` now tags `metadata.raw["loaded_from_processed"]` with a short human-readable reason on *every* path that returns already-processed data instead of the raw-FID one -- including two branches (no raw `ser`/`fid` present at all) that previously did this completely silently, unlike the existing (narrower) `read_2d_shape_guess_failed` case. Shown persistently in the info panel (not just the existing one-time load popup, which stays for the harder shape-guess-failure case specifically) so it's visible any time the spectrum is reselected later, not only right after loading. Bruker-only for now, matching the user's own explicit "(bruker)" scoping of this specific request -- the other three vendors have no equivalent "silently fell back to a different data source" path to note in the first place (X-Pulse/Varian/JEOL are always-raw-FID formats with no processed-data fallback at all).

13 new regression tests across `test_bruker.py`/`test_bruker_raw_2d.py`/`test_bruker_raw_fid.py`/`test_varian.py`/`test_xpulse.py`/`test_jeol.py`/`test_gui.py`. Found and fixed a real, unrelated omission while investigating JEOL's own parameters for this: `read_1d` for JEOL is the only one of the four vendor readers that has never populated `NMRMetadata.probe` at all, despite a `probe_id` parameter existing in the real file dumped during this investigation -- not fixed this round (out of scope, not what was asked), but worth a follow-up.

## Processing tab's "Slice 1" preview didn't match the 2D contour for a genuine 2D -- real bug (2026-09-04)

User: "still something different between the processing and 2D contour. Maybe just the first slice but Im not sure."

**Real bug found**: `_compute_processing_preview` built its F2-direct preview by reprocessing `raw_fid[0]` (or whichever row `_phase_2d_slice_index` picks) through F2's own windowing/FFT pipeline alone. For a *genuine* 2D that's a fundamentally different quantity from what the preview claims to show -- "F1 frequency bin 1", i.e. `spectrum.data[0]` -- since an F1 bin is a weighted FFT combination *across every raw row*, not any single row's own trace. Whenever real F1 modulation exists (an EXSY, where slice-to-slice variation *is* the signal), the two can look completely different: confirmed directly against real States-TPPI EXSY data (`test_data/2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119/2019`), ~94% relative difference for the default "Slice 1" preview, while adjacent *raw* rows alone already differ from each other by ~200% -- consistent with "different quantities", not a computation bug (the same raw-row pipeline matches `spectrum.data` exactly, to floating-point precision, whenever the two really are the same quantity, i.e. pseudo-2D or any 1D case -- confirmed both still match exactly, unaffected).

**Fix**: mirrors how `_show_2d_indirect_processing_preview` already handles F1 -- for a genuine 2D, trust `spectrum.data` itself (already the single source of truth for the current total phase, including any live phase-drag delta -- `_apply_phase_preview`'s own "2d" branch writes that straight into `spectrum.data` before this ever runs) rather than re-deriving an approximation from one raw row. The FID-domain panel (window/TDeff/align-echo preview) is unaffected -- a single raw row's own time-domain signal remains a legitimate, genuinely per-row concept even though the spectrum-domain panel no longer derives from it.

**A second, smaller robustness fix found while implementing the above, then partly reverted after an existing test caught an over-correction**: the existing row-selection code computed which *raw*-array row to preview (`_phase_2d_proportional_row`, a deliberately approximate mapping, per its own docstring, since the raw and processed arrays are different lengths), then reused `row + 1` as the index into the *processed* `spectrum` array for the new `spectrum.data`-trusting fix above. That reuse only happened to be correct because `_phase_2d_slice_index` is currently hardcoded to `1` (an earlier round's fix, replacing a largest-magnitude pick) -- `_phase_2d_proportional_row`'s formula algebraically returns `0` whenever the target is index `1`, regardless of array sizes, so `row + 1 == _phase_2d_slice_index` always held today, but only as a side effect of that specific hardcoding, not by construction. Fixed by using `_phase_2d_slice_index` directly (a new `spectrum_row` variable) wherever the code needs to index into `spectrum` itself, i.e. for `get_trace(spectrum, ...)` -- avoids silently breaking again if a largest-magnitude-style pick is ever reintroduced.

First pass over-applied this and used `spectrum_row` for the slice *title* too, which broke a pre-existing, deliberately-constructed regression test (`test_main_window_processing_preview_slice_matches_spectrum_tab_across_zero_fill_mismatch`, from an earlier round -- caught immediately by the full suite, not missed). Root cause of the over-correction: `title` is set on the FID panel specifically (`MplCanvas.show_processing_preview` calls `fid_ax.set_title(title, ...)`, not anything on the spectrum axes below it) -- and the FID panel's own data is, correctly and unavoidably, still `raw_fid[row]` (a genuinely raw-domain concept; window/TDeff/align-echo apply per raw row). So the title must keep describing *that* row (`row + 1`, raw-domain), not `spectrum_row` (processed-domain) -- reverted the title back to `row + 1`, kept `spectrum_row` for the `get_trace` call. `_phase_2d_proportional_row` exists precisely so `row + 1` remains an accurate (if approximate) label even for the spectrum panel below whenever the two diverge, e.g. after a live F1 zero-fill/truncate -- exactly what that existing test deliberately constructs.

Verified end-to-end against real data (exact match vs. `spectrum.data[0]`, previously ~94% off; live phase-drag still flows through correctly) and with a new synthetic fixture built specifically to discriminate the bug (every raw row given a different F2 frequency, guaranteeing real F1 modulation): the old code path was off by ~102% relative to `spectrum.complex_data[0]` on this fixture, confirmed by temporarily reverting the fix and re-running the new test (fails as expected); fixed, it matches to floating-point precision. 1 new regression test (`test_main_window_processing_preview_uses_spectrum_data_for_genuine_2d`); the full suite (876 tests) is what caught the title over-correction before it shipped.

## F2/F1 phase reapplication used the wrong dimension's pivot -- real bug, reported against a real States-TPPI EXSY (2026-09-04)

User: "There's still something off with the phasing between direct and indirect dimensions and how they're applied in the 2D. This is also for an exsy with states tppi." Investigated by re-checking every phase-application code path once more, not assuming the earlier round's fixes (dimension-switch commit-key, toggle preservation, round-trip preservation, Echo-Antiecho preview) were the whole story.

**Real bug found**: `_recompute_multi_row_data` (the live in-memory recompute for any F2-only control change on a conventional 2D) reapplies F2's own committed phase *and* F1's own committed phase separately -- but both computed their own `pivot_frac` from the *same* single, shared `self.pivot_spin.value()`, which is only ever meaningful for whichever dimension the Phase panel's Dimension combo currently has focused (`_capture_phase_reference` recentres it on every dimension switch). A nonzero `phase1` (or `phase1_f1`) committed while the *other* dimension was focused got reapplied around a pivot borrowed from a completely unrelated ppm range on every subsequent live recompute -- not a cosmetic difference, since `pivot_frac` defines *where* the phase1 ramp is zero. Confirmed directly with a synthetic fixture whose F1/F2 ranges don't even overlap: reusing F1's own centre (0.0 ppm) as F2's pivot, where F2's own range doesn't contain 0 at all, produced a result ~129% off (relative) from the correct, centre-pivoted reference; fixed, it matches to floating-point precision.

**Fix**: mirrors the same "always centre (0.5) unless this specific dimension is the one currently live-targeted" reasoning already established and shipped for `_current_phase_kwargs`/`_reapply_phase_post_reload` in the previous round -- added the same `is_phase_target` guard (confirming the live phase-editing session is actually targeting *this* spectrum, not some other one) plus a per-dimension "is the pivot actually live for this one" check, for both the F2 and F1 reapplication blocks.

**Not actually States-TPPI-specific** -- the bug is about pivot reuse, unrelated to FnMODE/hypercomplex decoding at all; it would affect any genuine 2D with a nonzero phase1 correction regardless of encoding. The user's EXSY-with-States-TPPI report is a real, representative trigger (EXSY data commonly needs a genuine phase1 correction across a wide F1 window) rather than evidence the bug was mode-specific -- worth remembering that a user's own framing of *when* they saw a bug doesn't necessarily mean the root cause is scoped that narrowly.

Also checked, while investigating: `_apply_phase_preview`'s own live-drag application (only ever touches whichever single dimension is currently focused, never both at once) and `_recompute_multi_row_phase_and_baseline_only` (the no-raw-FID fallback, which uses one unified `_phase_reference_full`/`_phase_freq_ppm` for whatever's currently targeted, not two separate reapplications) -- neither has the same bug, confirmed by tracing their own logic, not assumed safe by similarity alone.

Verified by reverting and re-observing the original ~129% discrepancy before restoring the fix. 1 new regression test, built with a synthetic fixture with deliberately non-overlapping F1/F2 ranges specifically because the real EXSY dataset's own F1/F2 centres happened to be numerically close by coincidence, which would have made a real-data-only test unreliable as a discriminator. 870 tests passing (869 -> 870).

## Zero-order phase made cyclic (2026-09-04)

User: "can you make 0th order phase cyclic from 0 to 359?" Phase 0 is a full rotation (`exp(i*phase0)`) -- 360deg away is physically identical -- so wrapping it doesn't change the actual correction applied, only how it's tracked/displayed. Wrapped in two places: `_commit_pending_phase_to_metadata` (the committed/stored total, `% 360.0`, so it doesn't grow unbounded across many separate drag sessions) and `_refresh_current_phase_displays` (the live label, so it stays cyclic continuously during an active drag too, not just once committed). Applies to both `"phase0"` (F2) and `"phase0_f1"` (F1) -- whichever key a given session is targeting. Phase 1 deliberately excluded -- a ramp *rate* across the spectrum, not a rotation, so "modulo 360" has no equivalent physical meaning for it (matches `_PHASE_SPIN_RANGE`'s own genuinely-unbounded design for the live delta itself). 5 new tests (wrap-up past 360, wrap-down for a negative delta, phase1 confirmed unwrapped, the display wrapping mid-drag before commit, and F1's own `phase0_f1` wrapping the same way).

## "Extract Slice" had no way back to the contour; the Processing tab's F2 preview was wrong for Echo-Antiecho data (2026-09-04)

Two more real reports, same session as the phase-preservation round above: "the phase is 90 offset from indirect processed and not. There is also not way to get back to the contour once I grab a slice."

**No way back after Extract Slice -- real, confirmed gap.** `MplCanvas.extract_slice` replaces the contour with a plain 1D trace via `show_spectrum(trace)`, which (correctly) clears `_contour_spectrum` -- but nothing ever offered a way back except reselecting the spectrum in the list (losing whatever else was mid-edit). Confirmed directly: grepped for any "show contour"/"back to contour" mechanism anywhere in `canvas.py`/`main_window.py` -- none existed. Fixed with a new `_extracted_from_spectrum` attribute (set only by `extract_slice`, right before it overwrites the contour) and a right-click "Show Contour" menu action, shown whenever it's set. Cleared unconditionally at the top of every `show_spectrum` call -- correct any time this canvas's own content genuinely changes, since `show_spectrum` is the only entry point for this particular canvas.

**"Phase is 90 offset ... indirect processed and not" -- a real, precisely-located bug, not a phase issue as such.** Traced directly: `io.bruker.read_2d` caches the raw FID into `metadata.raw["fid_raw"]` *before* applying its own Echo-Antiecho (FnMODE 6) fix -- the extra even-row ("echo") conjugate-and-reversal every echo row needs before F2 processing (see that fix's own detailed comment in `read_2d`, and the real, user-reported "mirrored in x" bug an earlier, incomplete version of it once caused). `_recompute_multi_row_data` (the GUI's own live-recompute path) already re-applies this exact transform -- it was fixed for this same class of bug in an earlier round. `_compute_processing_preview` (the Processing tab's own live single-row preview, used for the default "Slice 1" view in *both* pseudo-2D and genuine-2D modes) reads a row directly out of the *cached, untransformed* `fid_raw` and was never updated to match -- exactly the "two pipelines quietly diverge" class `_recompute_multi_row_data`'s own comment already warned about. Confirmed directly against real Echo-Antiecho data (`Cyclosporine/203`, FnMODE 6): a ~208% relative difference for the default "Slice 1" preview (row 0, an echo/even row) against the actual, correctly-processed spectrum -- not a rounding-level discrepancy, and a plausible real-world description of "the phase is 90 offset" for a conjugated-but-unreversed trace, which looks wrong/mirrored at a glance without being a literal phase error. Both modes happened to select the same (echo) row for this dataset, so the bug wasn't actually mode-dependent in its root cause -- fixed by applying the identical transform `_recompute_multi_row_data` already uses, scoped to just the one row this method actually needs (not the whole array).

**Verified**: the fixed preview matches the actual, correctly-committed spectrum's own row exactly (0.0 diff) on real data; reverting the fix reproduces the ~208% discrepancy, confirming it's genuinely discriminating. 4 new regression tests (3 for Show Contour in `test_gui_canvas.py`, 1 for the Echo-Antiecho preview fix in `test_gui.py`, isolated from an unrelated baseline-correction timing detail found while writing it -- a fresh `read_2d`-built spectrum has no baseline correction baked in yet, but `_compute_processing_preview` applies it live on every call, which very nearly produced a false failure unrelated to the fix under test). 869 tests passing (865 -> 869). Checked for sibling instances of the same "reads an unfixed cache" bug class: `_show_fid_preview` also extracts a single row directly, but from a *different* cache (`metadata.raw["fid"]`, the fully F2-processed pre-FFT FID, built from `data` *after* the Echo-Antiecho fix already ran) -- confirmed unaffected, not a parallel bug.

## 2D phase preservation across dimension switches and Process Indirect Dimension toggles -- three real bugs (2026-09-04)

User, with explicit top priority ("nothing else is as important as getting the basic import and processing steps correct"): "Phasing of 2D data before and after processing the indirect dimension should use the same info. the phase in the contour should also match the projections." Asked for a full logic-sequence walkthrough before any fix. Investigated by direct code tracing plus empirical reproduction against real data (`Bruker Example Data/Cyclosporine/203`) at every step, per this project's own standing discipline, given the stated stakes -- three real, independently-locatable bugs found, all in the GUI's own phase-state bookkeeping (`main_window.py`), none in `io.bruker.read_2d`'s own DSP.

**Bug 1 -- dimension switch commits a pending phase delta under the wrong key.** `_commit_pending_phase_to_metadata` (called from `_capture_phase_reference`, itself triggered by `dim_combo.currentIndexChanged`) decided which key to commit to (`"phase0"/"phase1"` for F2, `"phase0_f1"/"phase1_f1"` for F1) by reading `self.dim_combo.currentIndex()` live -- but Qt updates a combo box's `currentIndex()` *before* emitting `currentIndexChanged`, so by the time this runs, the combo already reflects the *new* dimension, not the one the pending delta actually belongs to. Confirmed directly: dragging F2's phase then switching the Dimension combo to F1 committed the F2 delta into `phase0_f1` instead of `phase0`. Fixed with a new `_phase_target_dim` attribute, set once (in `_capture_phase_reference`'s own "2d" branch) at the point a session's target dimension is actually established, and read from there instead of the live combo box.

**Bug 2 -- the "Process Indirect Dimension" toggle only preserved whichever dimension was currently UI-focused.** `_current_phase_kwargs` (forwards F2's phase into the `session.reload` call the toggle triggers) returned `{}` whenever the Dimension combo was on F1, silently letting the reload fall back to reading F2's phase fresh from the vendor's own recorded default -- discarding whatever F2 phase the user had actually established. `_reapply_phase_post_reload`'s own F1 branch had the exact mirror-image bug, gated on the combo being on F1, so F1's phase was lost whenever F2 was focused instead. Confirmed directly: toggling with F1 focused reverted an established F2 phase (552.8° -> a different vendor-default value) even though nothing about F2 itself should have been affected by that toggle. Fixed by making both methods always read/forward the already-*committed* `metadata.raw` values regardless of which dimension the UI happens to be showing -- a dataset's own established phase has nothing to do with what's currently on screen. `_on_process_indirect_toggled` now explicitly commits any pending delta before anything else, so both methods can rely on `metadata.raw` being fully current rather than needing to separately track/add a live spin-box delta.

**Bug 3 -- found while verifying bug 2's fix: F1's phase survived exactly one toggle, not a round trip.** Even with bug 2 fixed, `_reapply_phase_post_reload`'s F1 branch only ever *baked* F1's phase into the freshly-reloaded spectrum's `.data` -- it never wrote `"phase0_f1"/"phase1_f1"` back into that spectrum's own `metadata.raw`. Since `read_2d` always builds a brand-new `metadata.raw` from scratch on every reload (no F1-phase concept of its own), a genuine-2D -> pseudo-2D -> genuine-2D round trip (e.g. briefly checking the pseudo-2D relaxation view and coming back) silently lost F1's phase entirely -- confirmed directly, and confirmed it survived a *single* toggle in isolation, which is what made this one easy to miss. Fixed: `_on_process_indirect_toggled` now unconditionally carries `phase0_f1`/`phase1_f1` over into the freshly-reloaded spectrum's own `metadata.raw` on every toggle, in either direction -- a dormant value sitting unread in a pseudo-2D spectrum's metadata is harmless (nothing reads it in that state) and is exactly what makes it available again next time.

**"The contour should match the projections" -- confirmed already true by construction, not a fourth bug.** `plotting.twod.plot_2d_contour` computes the contour and both projection panels from the exact same `spectrum.data` binding, within one function call -- they cannot diverge from each other as a rendering matter. Confirmed directly (F2 projection's own values checked element-for-element against `np.sum(spectrum.data, axis=0)`). The user's report was a real, correctly-observed symptom of bugs 1-3 above (both views correctly reflected the *same*, wrongly-reset phase), not a separate rendering bug -- fixing the phase-preservation bugs is the whole fix here.

**Verified extensively, given the stated stakes**: every bug reproduced directly against real data *before* being fixed, and the fix re-verified the same way afterward (both switch directions for bug 1; both toggle directions, and specifically leaving the *other* dimension focused, for bug 2; two separate round trips for bug 3). The final `spectrum.data` after a full round trip was checked numerically against an independent from-scratch reference computation (`io.bruker.read_2d` + `processing.ft.apply_phase` called directly, bypassing the GUI entirely) -- identical to floating-point precision, not just "the metadata looks right." 4 new regression tests in `test_gui.py`. 865 tests passing (861 -> 865).

**A real mishap during investigation, corrected**: an early reproduction script (before adopting a properly isolated scratch sidecar path) wrote real `opennmr_metadata.json`/`opennmr_processing.json` sidecar files into the actual `test_data/Bruker Example Data/Cyclosporine/203/` folder, overwriting what was almost certainly a pre-existing, legitimate `pseudo_2d: false` sidecar from earlier genuine-2D testing in this project's own history (this dataset has been used as the reference HSQC example throughout). Caught by an unexpected `is_pseudo_2d: True` on what should have been a clean fresh load, diagnosed via `git check-ignore` (confirming `test_data/` is untracked, so there was no git history to recover from), and corrected by re-establishing `pseudo_2d: false` through the real toggle-and-save path rather than hand-writing the JSON. All reproduction scripts after this point used a properly isolated (and, after a second related mishap -- reusing the same scratch path across multiple separate script runs, causing phase deltas to accumulate across runs -- a genuinely fresh-per-run) sidecar location. Flagged here in full rather than glossed over, matching this project's own standing "measure twice" discipline -- worth remembering for any future direct-script reproduction against real `test_data/` paths: monkeypatch `io.persistence._sidecar_base` to an isolated, per-run-fresh directory *before* the first `_load` call, not just before any specific writing step.

## nmrtools2/leonmr survey, category A (partial): composable fit-result plotting functions (2026-09-04)

User picked this off the survey's own discussion list (below) as the next thing to build, flagged there as "the biggest concrete gap" -- opennmr's fitting/relaxation backends produce numbers but had no matching "look at my fit" plot helpers of their own, so the GUI's own Analysis tab built its plots by hand in `main_window.py` instead of calling a shared function.

**New module `plotting/fit.py`**, ported from `nmrtools2`'s `fitting.py`/`two_d.py`/`relaxation.py` (read directly, not from a summary, per the standing reuse-first discipline):

- `plot_1d_with_fit(fit_result, ...)` -- a spectrum + its fitted model (+ optional per-peak components), from `NMR1dplot_with_fit`. Simplified for opennmr's own data model: takes one `NMR2D_fit_slices` per-slice result dict directly (already carries its own `x`/`y`/`y_fit`/`components`, computed once inside `NMR2D_fit_slices` itself) rather than re-deriving them from `popt`/`peak_defs` the way the original had to.
- `plot_2d_integrals(fits, ...)` -- fitted (or region-integrated) peak integrals across a slice series, from `NMR2D_plot_integrals`. Ported closely, with one deliberate scope cut: dropped the `x_axis="vdlist"` mode (opennmr's own `processing.slicing.get_slices` doesn't stash a recoverable experiment `datapath` the way `nmrtools2`'s did, so there's nothing to auto-read a vdlist from here) -- for a real delay/gradient axis, use `NMR2D_fit_relaxation` + `plot_relaxation` instead, which already has its own established vdlist-reading.
- `plot_relaxation(fit_results, ...)` -- one or more fitted relaxation/diffusion curves (data + fit, optional rate-constant annotation, log-axis, per-series colour with an automatic paler fit-curve tint), from `NMR2D_plot_relaxation`. The cleanest port of the three: opennmr's own `fit_relaxation`/`NMR2D_fit_relaxation` already return this *exact* dict shape (`t`, `y`, `y_err`, `params`, `errors`, `model`, `param_names`, `rate_param`, `peak_label`) -- unsurprising, since both sides trace back to the same `nmrtools2` origin, just with the plotting half never carried over until now. Ports over essentially unchanged.

**Deliberately not ported this round**: `NMR2D_plot_slices` (stack-plot extracted 2D rows/columns, nQ-crossing markers, legend/text label styles, a `mode='both'` layout) and `NMR2D_summary_plot` (one figure combining a contour + slice stack + integrals panel). Both are substantially larger than the three above (`NMR2D_plot_slices` alone is 250+ lines with many options), and their core capability -- stacking extracted slices with an optional fit overlay -- already substantially overlaps with the GUI's own already-working Compare-tab/pseudo-2D-stack rendering, so the incremental value is lower than the three shipped here. Left as an explicit, flagged follow-up rather than silently dropped -- ask before picking this back up.

**GUI wiring not done this round either** -- the survey's own framing ("the GUI... builds its plots by hand... rather than calling a shared plotting function") implies the endgame is replacing that hand-built code with these new functions, but that means touching the Analysis tab's own already-working, well-tested rendering paths (`_display_slice_fit`/`show_relaxation_fit` and friends) -- a separate, higher-risk task from shipping the new backend functions themselves. Backend-only this round, by design.

**Verified**: 16 new tests (`test_plotting_fit.py`) -- `plot_1d_with_fit`: fit genuinely tracks data (not just "renders something"), component toggle, nucleus-based/custom xlabel, axis embedding. `plot_2d_integrals`: pseudo-2D correctly falls back to a slice-number axis, plotted values trace the true (known, synthetic) inversion-recovery shape end-to-end (not just structurally), normalised-integral math, empty/invalid-input rejection. `plot_relaxation`: plotted curve matches evaluating the model function directly with the fit's own fitted parameters, rate-constant annotation text is numerically correct, `show_series` selects the identical series whether given by label or by index, unknown-label/empty-input rejection, log-axis, and the default paler-fit-curve-tint colour behaviour. Also real-data checks against `test_data/bruker_T2/11` (`plot_1d_with_fit`/`plot_2d_integrals` render correctly on real, noisy data with no crash; `plot_relaxation` not separately real-data-checked -- this file has no vdlist, and the function was already verified end-to-end, exact-value, against synthetic data). Two real test-authoring mistakes found and fixed along the way, both in the new tests themselves (not the shipped functions): a malformed manual 2D-wrapper fixture for a "plain 1D" fit case (fixed by reusing the exact one-slice pseudo-2D pattern already established in `test_fitting.py`), and two assertions built on a wrong assumption about how many `Line2D` objects `ax.errorbar()` creates when real per-point uncertainties are present (fixed by asserting on identified artists/curve values instead of a hardcoded line count).

## nmrtools2/leonmr survey -- candidate features not yet in opennmr (2026-08-29)

## Command-line function caller -- noted, not yet scoped (2026-09-04)

User: "It might be nice to also add a command line function caller as a shortcut for many of the existing functions." Real idea, genuinely underspecified as given -- "many of the existing functions" spans wildly different call shapes (`io.load` takes a path; `processing.fitting.fit_spectrum`/`NMR2D_fit_slices` need a loaded `NMRSpectrum` plus a `peak_defs` list of dicts, not trivially a CLI flag; the new composable plotting functions, once built, need fit results as input, not raw files). Needs a scoping conversation before starting, not a guess: which functions (just `io`+basic processing, i.e. a load/reprocess/export pipeline? or fitting too, and if so how do peak defs get expressed on a command line -- a flag syntax, or a config/JSON file?), and what CLI framework (stdlib `argparse` -- zero new dependencies, matches this project's own lean-dependency pattern so far -- vs `click`/`typer` for nicer subcommand ergonomics at the cost of a new dependency). Ask before starting.

## Square-contour fix, round 2: the F1/F2 projections were never actually broken -- the panel itself wasn't really square (2026-09-04)

Follow-up to "Pseudo-2D UI" section's square-contour-aspect item (below) -- user, immediately after that shipped: "Make sure those 2D projections are linked axes with the contour plot size." Investigated directly (real-data rendering, not just re-reading the code) rather than assuming the earlier fix was already complete.

**A real, deeper bug found**: `ax.set_box_aspect(1.0)` (the round-1 fix) shrinks *only* the main contour axes in isolation -- it has no knowledge of the F1 (left)/F2 (top) projection axes, which are separate sibling axes created via `mpl_toolkits.axes_grid1`'s `divider.append_axes(...)` and positioned relative to the main axes' *original* (pre-shrink) size. Confirmed directly: on a deliberately non-square figure, the projections' own width/height stayed matched to the main panel's *un-shrunk* nominal size while the main panel itself shrank and re-centred -- a visible, measured misalignment (bbox positions checked numerically, and the rendered image showed a real gap plus a projection wider than the now-square panel below it).

**First fix attempt, also wrong (caught before shipping)**: switching to the *divider's own* aspect mechanism (`divider.set_aspect(True)`, which coordinates the main cell and every appended cell together, avoiding the isolation problem above) fixed the coordination but not the actual squareness -- `AxesDivider`'s default reference size (`Size.AxesX`/`Size.AxesY`) is the axes' own **data-coordinate range**, not physical size, so `divider.set_aspect(True)` was equalising "1 ppm on F2 == 1 ppm on F1 physically" (a data-aspect lock), not making the panel a square box. F1/F2 ppm spans are essentially never numerically equal for real NMR data (confirmed directly: a real HSQC's F2 span was ~17.6 ppm against F1's ~181.5 ppm, a ~10x difference) -- produced a panel roughly 1 inch by 10 inches, nowhere near square, despite `divider.set_aspect(True)` being set and the projections correctly tracking it. A first synthetic verification (`np.random.rand(20, 20)`, equal x/y data ranges by coincidence) missed this entirely -- exactly the kind of test-fixture blind spot this project has hit before with equal-by-coincidence values masking a real bug.

**Actual fix**: construct the `AxesDivider` manually with `xref=Size.Scaled(1), yref=Size.Scaled(1)` (a plain, data-independent relative unit) instead of `make_axes_locatable(ax)`'s data-range-based default. Verified against both real datasets on a deliberately non-square figure: genuine-2D (`Cyclosporine/203`, F1/F2 spans ~10x apart) and pseudo-2D (`bruker_T2/11`) both render with an exactly-square main panel (checked numerically, `width_inches == height_inches`) and projections correctly matched to it (`f1.height == ax.height`, `f2.width == ax.width`), confirmed against the actual rendered image, not just bbox numbers.

`plot_2d_contour` gained a `square_aspect: bool = True` parameter (docstring explains the whole isolation-vs-coordinated-divider reasoning above); `MplCanvas` passes `square_aspect=self._aspect_ratio is None` from both `show_spectrum` and `_redraw_contour`, so a user's own explicit general aspect-ratio lock still wins outright (unchanged from round 1). `_apply_view_options` reverted to its original single unconditional `self.ax.set_box_aspect(self._aspect_ratio)` line -- the divider now owns squaring entirely, so the generic per-view box_aspect call no longer needs any contour-specific branching at all.

- New `_make_2d_asymmetric_range` test fixture (`test_gui_canvas.py`) -- F1/F2 spans deliberately ~20x apart, since the existing `_make_2d`'s equal 10-ppm/10-ppm spans can't discriminate this bug at all (documented directly in the fixture's own docstring, as a standing lesson for this file).
- Rewrote `test_2d_contour_defaults_to_a_square_box_aspect` (checked `get_box_aspect()`, which stays `None` under the real fix -- squaring is divider-driven now, not box_aspect-driven) into `test_2d_contour_defaults_to_a_square_physical_panel`, checking the actual physical position instead. New `test_2d_square_default_keeps_f1_f2_projections_matched_to_the_panel` and updated `test_2d_square_default_survives_a_contour_redraw`. **Confirmed discriminating by temporarily reverting to a plain `make_axes_locatable` divider**: the physical-squareness tests correctly failed; the "projections matched" test did *not* -- a genuinely useful finding in its own right, since it clarified that the projections were never actually out of sync with the main panel's *own* cell (the divider always coordinates that correctly) -- what broke was only whether that shared cell resolved to a square shape in the first place. Kept the "projections matched" test anyway as a distinct, still-worth-having invariant (see its own updated comment).
- 845 tests passing (844 -> 845).

## Bounding peak amplitude from Integration Region integrals -- the last fitting-wizard-roadmap item (2026-09-04)

Closes out the one remaining piece of the "Fitting wizard and beyond" roadmap item (see "Analysis (peak fitting / relaxation)" below): "bounding possibly from integrals." Opt-in (default off) -- new "Bound amplitude using region integrals" checkbox, Peak Picking's Advanced section.

**Mechanism**: `processing.fitting.NMR2D_fit_slices` gained `amplitude_bound_regions` (a list of `(lo, hi)` regions -- the GUI passes the exact same Integration Regions list already used for `fit_regions`, not a separate concept). For each peak whose *starting* shift falls inside one of these regions, that slice's own raw (unfitted) data is trapezoidally integrated over the region, then converted to an amplitude upper bound via a new `_peak_area_per_unit_amplitude` helper (`area = A * fwhm * factor`, factor depending on lineshape -- the same Gaussian-area formula already independently verified in the fit-report CSV export's own writeup, reused not re-derived). Recomputed fresh per slice (not frozen from a single reference slice), so it correctly tracks a genuinely-varying true amplitude across a relaxation/diffusion series. Deliberately loose: an entire region's own raw integral is used as-is (no extra margin) as the cap for *each* peak inside it -- still a true physical upper bound for any single peak sharing the region, even with several peaks/baseline contributing, since no single one can individually exceed what the whole region measures. When a peak's shift falls inside more than one given region, the tightest resulting bound wins. A peak's own explicit `bounds["amplitude"]`/`bounds["A"]` (e.g. the GUI's "Set Bounds..." dialog) always wins outright and is never overridden. A region whose raw integral comes out <= 0 is skipped for that peak/slice rather than forcing a near-zero cap -- S/N-based gating (`noise_region`/`sn_threshold`) already exists as the deliberate, explicit tool for "pin amplitude to zero once a peak is gone"; this shouldn't do that as a side effect of an unrelated, mildly-negative trapz artifact on genuinely negative-going data (e.g. inversion recovery before the null point).

**A real, independently-valuable bug found and fixed while building this**: `fit_spectrum`'s own initial parameter guess (`initialize_params_with_map`, e.g. amplitude read straight off the data at a peak's shift) was never clipped into whatever bounds would actually be used -- already latent for the *existing*, manual "Set Bounds..." dialog whenever a user set an amplitude bound tighter than that raw data value (there was even a dead-giveaway diagnostic, `print("BAD INITIAL PARAMETER"...)`, that detected exactly this without ever fixing it), but this new feature makes hitting it easy (a narrow region routinely produces a bound below the data's own raw height). scipy's bounded `curve_fit` raises immediately in that case ("Initial guess is outside of provided bounds") rather than clipping it itself. Fixed with `p0[is_free] = np.clip(p0[is_free], lower[is_free], upper[is_free])` right before the fit call -- clips only the free subset, so a `fixed` parameter's value is never silently altered. New dedicated regression test (`test_fit_spectrum_clips_data_driven_initial_guess_into_a_tight_manual_bound`) covers this independently of the new feature, using a plain manual bound.

**Verified**: 6 new backend tests (`test_fitting.py`) -- exact hand-computed bound value against a known analytic Gaussian, confirms the fit genuinely converges to the bound's own edge (not coincidentally already there), explicit user bounds never overridden, off by default, tightest-of-overlapping-regions wins, a non-positive region integral is skipped rather than forcing a zero cap. 3 new GUI tests (`test_gui.py`) through the real `_on_fit_spectrum_clicked` call path -- checkbox default, checked-vs-unchecked genuinely changes the fitted result, persistence round-trip through spectrum reselection (mirrors `seed_mode`'s own pattern exactly). Real-data smoke test against `test_data/bruker_T2/11` (a real pseudo-2D series) confirmed no crash and sane bound values across several slices with a deliberately generous (non-binding) region.

## 2D phasing redesign: Spectrum tab always shows the 2D, Processing tab is the only 1D slice view (2026-09-03)

Real, repeated user report -- a regression against this session's own earlier "stuck on slice" fix, plus a direct design reversal of an even earlier request: "2D processing is still messed up. It still turns the spectrum view into a 1D, gets crazy phasing that can't be matched up from the 1D slice to the 2D contour, and I still have to repeatedly select and deselect the process indirect button to get back the 2D. Spectrum tab should always show the 2D or pseudo 2D where there is one, and the processing tab should only have a 1D that is matched to the fid tab."

**Root cause**: the earlier "stuck on slice" fix (this session) only ever addressed the *non-phase* settle path (`_run_debounced_multi_row_reprocess`). The *phase-drag* path itself -- `_apply_phase_preview`'s own "pseudo2d"/"2d" branches, plus `_on_autophase_slices_clicked` -- was untouched, and by design switched `self.canvas` to a single-slice line the moment any phase interaction happened (a real, earlier request: "the phasing for a 2D should still default to a 1D slice"). That earlier request is now explicitly superseded.

**Fix, a genuine architecture change, not a patch**: `self.canvas` never renders a 2D/pseudo-2D phase slice of its own any more -- `_show_2d_phase_slice`/`_show_pseudo2d_slice` (the two methods that did this) are deleted outright, since removing every real caller left them dead code. It always shows the full contour/stack, refreshed live during a drag via a new debounced timer (`_canvas_2d_refresh_timer`/`_schedule_canvas_2d_refresh`, 200ms, same pattern as the existing `_live_reprocess_timer`) -- a full contour rebuild is measured at ~0.35s on a real, large array (`plotting.style.downsample_2d`'s own docstring), so redrawing on literally every mouse-move tick would make the drag itself feel frozen; the phase correction itself is still applied to `spectrum.data` on every tick (cheap), only the expensive on-screen redraw waits for a short quiet period. The debounced refresh explicitly preserves both x *and* y limits around the rebuild (a contour's y axis is a second ppm/index axis, just as meaningful to preserve as x -- the old single-slice-line view only ever needed to preserve x). The Processing tab's own 1D preview (`_compute_processing_preview` for F2/direct, `_show_2d_indirect_processing_preview` for F1/indirect, both already built in earlier rounds this session) is now the *only* place a judgeable 1D slice shows during 2D phase editing -- "the processing tab should only have a 1D that is matched to the fid tab" is already true for pseudo-2D (both key off the same `slice_spin` row); for a genuine 2D there's no FID-tab equivalent to match against at all (gated off entirely, `_fid_preview_available`), so the Processing tab's own established "slice 1" convention is the closest analogous, internally-consistent choice -- flagged here in case that's not what was meant.

**This also resolves "crazy phasing that can't be matched up from the 1D slice to the 2D contour," without a separate fix**: the old Spectrum-tab slice (`_show_2d_phase_slice`, F1-*frequency*-bin-1 of the processed spectrum) and the Processing-tab's own F2 preview (`_phase_2d_proportional_row`, raw t1-*time*-domain row 1) were never the same kind of slice to begin with -- comparing them was comparing two different domains that only happened to share the ordinal "1," not two views of the same thing. Removing the Spectrum tab's own slice display entirely removes the comparison that produced this complaint, rather than trying to reconcile two domains that were never meant to match.

**Also this round**: the `+180`/`-180` quick-phase buttons nudge by the identical amount (phase is 360-degree periodic) -- real feedback that having both was pointless; one `180°` button (no sign) replaces them. New "Zero Phase 0"/"Zero Phase 1" buttons, distinct from the existing combined Reset (zeros both) -- explicit request ("a button to zero 0th or 1st order phase").

- New tests: `test_main_window_2d_phase_stays_a_contour_not_a_slice`, `test_main_window_2d_phase_never_needs_reprocessing_process_indirect_to_recover_contour` (a real multi-step workflow -- drag, dimension switch, tab switch, per-slice autophase -- confirming `self.canvas` is a contour at every single point along the way, matching the user's own "repeatedly select and deselect" framing directly), `test_main_window_zero_phase_buttons_zero_only_their_own_order`. Renamed/rewritten (asserted the now-forbidden opposite behaviour before): `test_main_window_2d_phase_slice_follows_dimension_switch`, `test_main_window_pseudo2d_phase_sets_processing_preview_slice_title` (from `..._sets_canvas_slice_title`). Strengthened: `test_main_window_phase_from_contour_preserves_ppm_zoom` now actually settles the new debounce and checks both x and y limits, rather than passing for a weaker reason (no rebuild had happened yet to reset anything).

## Genuine 2D Spectrum tab getting stuck showing a slice; always slice 1 (2026-09-03)

**Spectrum tab silently stuck on a 1D slice**: "now when processing 2d data, I only see the 1d slice unless I unprocess and reprocess the indirect dimension. Something is being linked where it shouldn't." Root cause, reproduced directly: `_run_debounced_multi_row_reprocess` (the settle callback for *any* live FID-processing control change -- window, zero-fill, shift, not just phase) unconditionally called `_show_2d_phase_slice`/`_show_pseudo2d_slice` on `self.canvas`, regardless of whether the Spectrum tab was even the currently-visible one, and regardless of whether it was already showing its default contour. So changing an unrelated control (confirmed: F1 zero-fill) while looking at the Processing tab silently converted the Spectrum tab's content to a slice in the background -- with nothing to ever convert it back except a full reload (`process_indirect_button` off/on), since `_on_canvas_tab_changed` has no "refresh on switching back to Spectrum" branch at all. Fixed: both branches now check whether `self.canvas` was already showing a contour before this settle -- if so, refresh it *as* a contour with the new data; only switch to a slice if it was already a slice (i.e. genuinely mid an active phase-editing session, which is what the "avoid flicker" reasoning this code already had was actually meant to cover).

**Always slice 1, not largest-magnitude**: "the processing should just always use slice 1 of the 'xf2' (unprocessed indirect)." Replaces the earlier largest-total-magnitude row/column pick (`_capture_phase_reference`'s own `np.argmax(magnitude.sum(...))`) with a fixed `_phase_2d_slice_index = 1`. Deliberate simplification, not just a preference: the magnitude pick needed the *processed* F1 axis to even evaluate, which is exactly what made the raw-vs-processed row-count mismatch this session's earlier fixes had to work around (`_phase_2d_proportional_row`) -- index 1 is a valid position in both the raw and the processed array either way, no mapping needed. Every consumer of `_phase_2d_slice_index` picked this up for free (one change, not several).

- Updated tests: `test_main_window_2d_phase_defaults_to_slice_1` (renamed/updated from `..._the_max_magnitude_row`), `test_main_window_2d_phase_slice_follows_dimension_switch` (updated to check canvas title, since the index itself no longer varies by dimension).

## X-Pulse MRI load crash: 1D files misdetected as 2D NTUPLES (2026-09-03)

"Spectral width (F1) must be > 0 to compute dt." Root cause: `io.xpulse._is_ntuples_2d_file` (decides `read_1d` vs. `read_2d` for a single `.jdx` file) checked for the bare substrings `"DATACLASS"`/`"NTUPLES"` in the header -- but *every* X-Pulse export uses the NTUPLES data-table format for its FID regardless of dimensionality, including a plain 1D file (confirmed directly: `test_data/xpulse_MRI/14/CsBr_LWG_1D-Image-Echo_H_avg.jdx`, a real 1D MRI profile, declares `##DATACLASS= NTUPLES` same as a genuine 2D file does) -- so a 1D file got misdetected as 2D and routed into `read_2d`, which then failed on this file's nonexistent F1 spectral width. Fixed: check `##NUMDIM= 2` specifically (the field that actually distinguishes them) via regex, instead of the bare substring check. The *existing* test for this detector (`test_is_ntuples_2d_file_true_for_ntuples_false_for_plain_1d`) never caught this -- its "false" fixture predates the NTUPLES format entirely (no `DATACLASS`/`NTUPLES` string in it at all), so it couldn't discriminate the actual failure mode; added a second test using a minimal 1D-but-NTUPLES-declaring fixture, confirmed to fail against the old substring check before trusting it.

- New test: `test_is_ntuples_2d_file_false_for_a_1d_file_that_still_declares_ntuples` (`tests/test_xpulse.py`).

## Open button style, brought back without the earlier functional risk (2026-09-03)

"Change the look or style of the open button. The import view we briefly had that didn't work was quite a nice look." That was the single-non-native-`QFileDialog`-in-`ExistingFiles`-mode attempt (`a8c0021`), reverted for *functional* reasons (folder selection unreliable in that mode, restrictive default filter) -- not for its look. Brought the look back without the functional risk: both existing native dialogs (`getExistingDirectory` for the plain click, `getOpenFileNames` for the dropdown) now pass `options=QFileDialog.Option.DontUseNativeDialog`, keeping each dialog's own already-correct, single-purpose file-mode semantics (Directory-only vs. Files-only) unchanged -- the specific file-vs-folder click ambiguity that broke the original attempt was a property of mixing both in one `ExistingFiles`-mode dialog, which doesn't apply to either of these. **Not interactively verified** (this dev environment is headless) -- worth a quick real check that folder selection in the new styled dialog still feels right, consistent with this project's standing lesson about not trusting an untested-in-this-environment UI claim as settled.

## Genuine 2D Processing-tab preview: full contour instead of a slice, and no F1 support at all (2026-09-03)

Real user report, against real data (`test_data/20200804_NaMgMnO_Pristine_111218C_Check_23Na_1p3mm/31`, a genuine 23Na 2D): "I get a different contour in processing vs the spectrum/import tabs. The spectrum one is correct but the processing one isn't. It needs also to be updated so that the processing is just for a single slice (first or largest magnitude), and should be independently possible for the indirect dimension. This is what I have been asking for for all 2D data."

**Root cause, part 1 -- full contour instead of a slice**: `_show_processing_preview` had an explicit fallback, `if ... (spectrum.ndim == 2 and not spectrum.is_pseudo_2d): self.processing_preview_canvas.show_spectrum(spectrum)` -- a full contour redraw of the whole dataset, unconditionally, for *any* genuine 2D, any time the Processing tab's preview needed a full rebuild (landing on the tab, or any FID-processing-control change). This predates `_compute_processing_preview`'s own later genuine-2D support and was never removed once that support existed. Reproduced directly: the contour *was* numerically correct (same `spectrum` object, confirmed by identity), but redrawing a full contour on every phase-drag tick is exactly the cost `_show_2d_phase_slice`'s own docstring says single-slice preview exists to avoid -- and visually inconsistent with the Spectrum tab's own already-correct single-slice phase preview showing at the same time.

**Root cause, part 2 -- F1 (indirect) had no preview path at all**: `_compute_processing_preview` (the raw-FID-based preview used for F2/direct) was hardcoded to `get_trace(spectrum, axis="f1", ...)` -- always an F2 row, never an F1 column -- so switching `dim_combo` to F1 didn't change anything; the Processing tab kept showing whatever F2 was showing (or the full contour, per part 1).

**Fix**: `_show_processing_preview`/`_refresh_processing_preview_phase_only` now route a genuine 2D to one of two single-slice paths depending on `dim_combo`: F2 (direct) keeps using `_compute_processing_preview`'s existing raw-FID pipeline (now applied unconditionally -- always the largest-magnitude row, `_phase_2d_slice_index`, not gated behind an active phase-drag or a `dim_combo`-index check that used to make it silently no-op outside of phasing); F1 (indirect) is a new method, `_show_2d_indirect_processing_preview`. F1 can't reuse F2's raw-FID approach -- an F1 trace only exists once *every* raw row has already been through its own F2-domain processing, which isn't cheap enough to redo per preview tick -- so it instead reads a trace straight off `spectrum.data` (already kept current by the debounced whole-dataset recompute, `_recompute_multi_row_data`) via `get_trace(spectrum, axis="f2", ...)`, mirroring `_show_2d_phase_slice`'s own already-trusted F1 approach on the Spectrum tab exactly.

**A real bug caught mid-fix, before it shipped**: the first version of `_show_2d_indirect_processing_preview` applied `phase0_f1`/`phase1_f1` (baked-in) plus the live Phase spin-box delta to the extracted trace, mirroring `_compute_processing_preview`'s own F2 formula. Cross-checking its output numerically against `_show_2d_phase_slice`'s (the trusted reference, same dimension, same state) on the real dataset showed a large discrepancy (~24000, comparable to the data's own scale, not a rounding gap) -- because `spectrum.data` *already* has the live phase delta baked in by the time this runs (`_apply_phase_preview`'s own "2d" branch writes it in, uniformly across the array, before calling into this code), so re-applying it was silently double-phasing. Fixed by dropping the phase-application step entirely -- extract and display, nothing else, exactly matching `_show_2d_phase_slice`.

**A test-design pitfall found while writing the regression test for that bug**: the first version of the new test compared rendered `Line2D` y-data between the two implementations, using the existing raw-2D test fixture's default FID (identical across every F1 row). That FID's F1 dimension FFTs to a pure single-point real delta -- and `NMRSpectrum.normalized()` (what the canvas renders) scales the real part to `[-1, 1]` by its own max-abs, which collapses any single-nonzero-point array to the same `[0, 0, +-1, 0]` shape *regardless of that point's actual phase or magnitude*. The test passed even with the double-phasing bug deliberately reintroduced. Fixed by building a small fixture with a distinct complex weight per F1 row (not the tiled default), keeping the F1 trace multi-point and genuinely phase-sensitive after normalization -- confirmed discriminating both ways (fails with the bug reintroduced, passes with the fix) before trusting it.

- New tests: `test_main_window_2d_genuine_processing_preview_shows_a_slice_not_a_contour`, `..._supports_f1_dimension`, `..._does_not_double_apply_phase` (`tests/test_gui.py`).

## 2D phasing preview mismatch between the Spectrum and Processing tabs (2026-09-03)

Real user report: "there is something happening still with phasing for 2D data, especially when switching between tabs." Reproduced directly: while genuine-2D phasing, the Spectrum tab shows the row picked by `_phase_2d_slice_index` (the max-magnitude row on the *processed* F1 axis), but `_compute_processing_preview` (the Processing tab's own live preview) was reading `slice_spin.value()` instead -- a control that's hidden and stuck at 1 for a genuine 2D -- so the two tabs could show completely different, unrelated rows while switching between them mid-edit.

First fix attempt used `_phase_2d_slice_index` directly as a `raw_fid` row index -- wrong, because `_phase_2d_slice_index` indexes the *processed*, zero-filled F1 axis (`spectrum.data.shape[0]`), while `raw_fid` (`metadata.raw["fid_raw"]`) is the pre-zero-fill (and, for a hypercomplex FnMODE, pre-decode) acquired array -- confirmed on real data the two lengths differ substantially (256 raw rows vs. 1024 processed). A zero-filled processed index has no true inverse into the raw array (it doesn't correspond to any single real acquired point), so a direct/clamped use just saturated at the raw array's last row for any out-of-range index -- a wrong, silently-plausible-looking answer, not a crash.

Fixed with a new proportional-mapping helper, `_phase_2d_proportional_row` -- maps `_phase_2d_slice_index`'s fractional position along the processed axis onto the equivalent fractional position in the raw array ("roughly the same relative position in the acquisition"), used only in the branch where genuine-2D phasing is actually active (`not spectrum.is_pseudo_2d and self._phase_mode == "2d"` and F2/direct dimension selected and a slice index has been captured); every other case (pseudo-2D, F1-dimension phasing, no active 2D-phase session) is unchanged, still driven by `slice_spin`.

`_show_fid_preview` was checked for the same bug and found NOT to need a fix -- `_fid_preview_available` (which gates every one of its ~9 call sites) always excludes a genuine 2D, so its own `fid.ndim == 2` branch is unreachable for this case in practice; left as-is with a comment explaining why, instead of shipping a dead-code edit.

- Covered by a new test, `test_main_window_processing_preview_slice_matches_spectrum_tab_across_zero_fill_mismatch`, deliberately zero-filling F1 from 4 raw rows to 16 processed rows so the three candidate behaviours (the old dead `slice_spin` path, the first-attempt direct/clamped index, and the proportional mapping) give three different, known answers ("Slice 1" / "Slice 4" / "Slice 3") -- confirmed discriminating by running it against both broken variants directly, not just against the final fix.

## Echo-Antiecho fix, round 2: the conjugate-only version shipped a real F2-mirror bug (2026-09-03)

Real user report, caught by actually using the app: "I'm pretty sure the echo-anti echo data is now mirrored in x." Correct, and a genuine mistake in this project's own earlier "fixed and verified" work (the "Echo-Antiecho F1-mirror bug... found and fixed" entry, 2026-09-02) -- worth a full, honest account, not just a quiet patch.

**What went wrong**: round 1's fix conjugated the raw "echo" row's time-domain FID before F2 processing, verified by measuring the F1-mirror ghost ratio (dropped from ~0.8-1.2x to ~0.005-0.02x -- looked like a clean win). What that verification *didn't* check: `FFT(conj(x))[k] == conj(FFT(x)[-k])` -- conjugating a raw time-domain row doesn't just fix the F1/hypercomplex phase relationship, it also **reverses that row's own F2 frequency axis**, an unavoidable side effect of conjugating a raw signal that's pre-FFT. Checked directly, after the report: the round-1 fix's own F1 ratio was still excellent (~0.01), but its F2 ratio was ~0.95 -- an equal-magnitude ghost at each peak's F2-mirror position, exactly matching the report.

**The real fix**: conjugate *and time-reverse* the same row (order doesn't matter -- conjugation is elementwise, reversal is a pure reordering, so they commute). The two frequency-reversals (one from the conjugate, one from the explicit reversal) cancel, leaving F2 untouched, while the conjugate's own value-negation still supplies the F1 relationship. Verified on the same real dataset (`test_data/Bruker Example Data/Cyclosporine/203`) with both axes checked *together* this time (specifically because that's exactly what round 1's own verification missed) -- F1 and F2 mirror ratios both land at ~0.005-0.01 (noise floor) across the 10 strongest real peaks, simultaneously. Also visually confirmed: the full contour render went from a busy, cluttered pattern (genuine peaks + ghosts, easy to mistake for "a lot of real correlations") to a clean, sparse, correctly-assigned HSQC-shaped map.

**How to apply generally, beyond this one fix**: when a fix touches a *raw, pre-transform* signal (here: a time-domain FID before its own FFT) and the verification only measures *one* of the transform's several possible failure modes (here: F1 mirroring, checked; F2 mirroring, a mathematically-predictable side effect of the exact same operation, not checked) -- checking one axis clean is not the same as checking the operation is correct. The fix's own mechanism (`FFT(conj(x))` reverses frequency) should have been enough to predict the F2 risk analytically, without needing the user to find it first; worth remembering for any future fix in this codebase that manipulates a raw time-domain signal directly.

- Corrected in three places: `io.bruker.read_2d`'s own `fn_mode == 6` branch (`data[0::2] = np.conj(data[0::2])[:, ::-1]`, was missing the `[:, ::-1]`), `MainWindow._recompute_multi_row_data`'s mirrored fix (the GUI's own separate in-memory live-recompute path), and `decode_hypercomplex_f1`'s own docstring (now documents the full two-round story, not just the round-1 result).
- Updated the existing regression test (`test_read_2d_echo_antiecho_undoes_the_conjugate_only_for_the_echo_row` -> `..._conjugates_and_reverses_the_echo_row`) to lock in the corrected behaviour, including a "would have failed under round 1's own fix" discriminating check.
- All existing tests still pass; no other regression found from this correction.

## 2D roadmap, overlay multiple 2D contours -- another already-ported backend function wired up (2026-09-03)

Continuing item 2 of the "LWG roadmap" list -- "a way to overlay multiple 2D contours." Same pattern as the diagonal-line item above: `plotting.twod.overlay_plot` already existed, ported from `nmrtools2`'s own `NMR2D_overlay_plot` per the earlier project-plan document, complete with its own compatibility checking (`_check_overlay_compatible` -- same F1/F2 nuclides, conventional-vs-pseudo-2D, homo-vs-heteronuclear) and flat-colour-per-spectrum rendering -- just never called from GUI code.

- New `MplCanvas.show_2d_overlay(spectra, labels)` -- thin wrapper around `overlay_plot`. **Found and fixed a real bug while building this**: calling `self.figure.add_subplot(111)` a second time right after `self.clear()` (which already leaves `self.ax` as a fresh subplot) created two overlapping, visually broken axes -- caught immediately from an actual rendered image, not just a "did it crash" check.
- Wired into `MainWindow._refresh_compare_view` as a dedicated early branch (new `_refresh_compare_2d_overlay`): when 2-3 *and only* 2-3 checked Compare-tab entries are all genuine 2D, this overlay path runs instead of the existing 1D stack/superimpose logic -- deliberately kept as a separate method rather than folded into the 1D path, since mass-normalization and per-row plot highlighting don't have 2D-overlay equivalents. A mixed 1D/2D selection, a single 2D spectrum, or more than 3 2D spectra all fall through to the existing 1D-only logic unchanged (already reports non-1D entries as skipped). An incompatible pair (e.g. different F1 nuclides) surfaces `overlay_plot`'s own `ValueError` as quiet status-label text, matching how "not 1D"/"no sample mass set" are already reported -- not a popup, and doesn't force a tab switch, since this is an expected/reportable mismatch, not an exceptional failure.
- Verified against the real Cyclosporine dataset loaded twice (both a broken-axes and a fixed render inspected directly as images, not just checked for exceptions).
- 6 new tests (`test_gui.py`: a working 2-spectrum overlay, an incompatible-nuclide rejection, a 4-spectra selection correctly falling through to the 1D-only skip path; `test_gui_canvas.py`: `show_2d_overlay` clears `_contour_spectrum` and renders a legend, raises for incompatible spectra).

## 2D roadmap, Extract Slice now defaults to the cursor's own position (2026-09-03)

Continuing item 2 of the "LWG roadmap" list -- "Row/slice extraction should default to the cursor's current position rather than a fixed index." The right-click "Extract Slice…" dialog's own `index_spin` previously always started at 1 (`QSpinBox`'s own range-minimum default, never explicitly set) regardless of where the contour was actually right-clicked.

- `MplCanvas._on_press`'s own right-click position capture (`_last_right_click_xy`) was gated on `_fit_mode` (only used for the fit-tab's "Exclude Nearest Point" action) -- widened to capture on *every* right-click, still returning early only for the fit-mode case so nothing else about that path changed.
- New `MplCanvas._nearest_slice_index(spectrum, axis, cursor_value)`: the 1-based row/column index nearest a cursor's data-coordinate value -- ppm-based nearest-match for a genuine frequency axis (same approach `processing.slicing.get_trace`'s own `shift` parameter already uses), or round-and-clip for a pseudo-2D's index-kind F1 axis (which has no ppm at all, but whose plotted Y value already *is* approximately the 1-based slice number).
- `_prompt_extract_slice`'s `index_spin` now defaults from this (using the cursor's Y position when the "Row (fix F1)" axis is selected, X when "Column (fix F2)" is), re-deriving live if the axis choice is switched inside the dialog -- falls back to the old fixed default of 1 if the menu was somehow opened without a tracked right-click position at all.
- Verified against the real Cyclosporine dataset (a right-click at a specific ppm position correctly resolves to the nearest row/column index, matching the axis's own recorded ppm values).
- 7 new tests (`test_gui_canvas.py`: `_nearest_slice_index` on a frequency axis, out-of-range clipping, the pseudo-2D index-axis branch; the dialog's own pre-filled default following the cursor for both axis choices, and falling back to 1 without a tracked cursor position).

## 2D roadmap, indirect-dimension carrier frequency -- verified correct against nmrtools2 (2026-09-03)

Continuing item 2 of the "LWG roadmap" list -- explicit standing ask: "Verify the indirect dimension's own carrier frequency is calculated correctly... cross-check against nmrtools2's own code." Read nmrtools2's `two_d.py::NMR2dimport` directly (not a summary) to get its exact F1-axis formula, then compared line-by-line against `io.bruker._ppm_hz_axis` (the shared F1/F2 axis function `read_2d`/`read_2d_processed` both already use).

**Exact match, including the homonuclear edge case**: nmrtools2's own formula is `SR = (SF-OBS)*1e6`, `true_centre = O1-SR`, a linspace spanning `true_centre +/- SW/2`, `ppm = Hz/SF` -- applied identically to F1 (using `acqu2s`/`proc2s`'s own `O1`/`BF1`/`SF`/`SW`) and F2 (using `acqus`/`procs`'s own), character-for-character the same as `_ppm_hz_axis`. nmrtools2 also special-cases `homonuclear=True` (F1 borrows F2's own `O1`/`SR` from `acqus` rather than reading `acqu2s`/`proc2s` at all, since a homonuclear 2D's indirect dimension shares the direct dimension's transmitter) -- confirmed `read_2d`/`read_2d_processed` already implement this exact same branch (`if homonuclear: obs_1, o1_1, sf_1, sw_1 = obs, o1, sf, sw`). **No bug found -- this item is closed as verified, not needing a fix.**

## 2D roadmap, diagonal-line (nQ) overlay wired to the GUI (2026-09-03)

Continuing item 2 of the "LWG roadmap" list below. Per that entry's own note, checked `plotting.twod.plot_2d_contour` first (per the standing reuse-first instruction) -- confirmed it already has a fully working `show_nQ_line`/`nQ`/`nQ_colors` implementation (draws dashed `y = n*x` line(s) directly in the axes' own data coordinates, already used at both existing `plot_2d_contour` call sites in `twod.py`), just never called with anything but the defaults from GUI code -- genuinely just needed wiring, not building.

- New `MplCanvas._show_nq_line`/`_nq_values`, `_nq_line_kwargs()`/`set_nq_line()` -- same pattern as the existing `_contour_fraction_kwargs()`/`set_contour_fraction()` (rebuilds a currently-shown 2D contour immediately on change). Wired into both of `canvas.py`'s own `plot_2d_contour` call sites (`show_spectrum`, `_redraw_contour`).
- GUI: new "Show diagonal (nQ) line" checkbox + "Diagonal n:" spin box in the Output tab, next to the existing 2D-downsample control -- applies to every canvas via `MainWindow._all_canvases()`, same as contour fraction. Off by default (matches `show_nQ_line`'s own default), `n` defaults to 1 (the main diagonal -- the common case, e.g. EXSY/NOESY/COSY).
- Verified against the real Cyclosporine HSQC dataset (a real, if not physically-meaningful-for-that-particular-heteronuclear-dataset, rendering check -- the line draws exactly where `y = n*x` should in the displayed ppm coordinates regardless of whether the two axes share a nucleus, confirming the mechanism itself works correctly).
- 6 new tests (`test_gui.py`: checkbox default/every-canvas-toggle/n-value-change, a real-2D-dataset line-appears-and-disappears check; `test_gui_canvas.py`: default kwargs, `set_nq_line` changing them, rebuilding a currently-shown contour).

## Single Open button for both files and folders (2026-09-03)

Last item from the original request ("by the way there should just be one button that can open files or folders"). The two separate "Open…" (folder) / "Open File(s)…" (file) buttons, menu actions, and dialogs are now one -- `_on_open_clicked` builds a non-native `QFileDialog` instance in `ExistingFiles` mode instead of using the static `getExistingDirectory`/`getOpenFileNames` convenience functions. Confirmed first, not assumed: Qt's native file dialogs don't support picking files *or* folders in one session on any platform (a well-known, still-open Qt limitation, per a web search and the Qt Centre forum's own community-standard answer) -- the workaround used here (a non-native `QFileDialog`, `ExistingFiles` mode) is the documented, standard cross-platform fix: Qt's own file-list view lets a single click select any row regardless of whether it's a file or folder (double-click still navigates into a folder), unlike a native panel where only files are selectable in that mode. Tradeoff disclosed, not hidden: this uses Qt's own custom-drawn dialog rather than the native OS one, a small, deliberate cost for actually working the same way everywhere. New `_open_paths` consolidates the file/folder routing previously duplicated between the old file-picker handler and drag-and-drop's own `_handle_dropped_paths` (which now calls it too, with `report_unrecognised=False` to keep its existing "silently skip OS clutter" behaviour, unlike the dialog's own "report anything unrecognised" default for a deliberate pick). New test confirms the actual real-world payoff: one dialog session, a folder and an individual file picked together, both load.

## "SI for both dimensions" -- checked, does not reproduce (2026-09-03)

Resumed the one item explicitly paused during the phase/window round above ("hold off on adding the new 2D stuff for now"). Changed F2's own zero-fill and F1's (`zero_fill_f1_spin`) *together*, not just one at a time the way every pre-existing zero-fill test covered -- against the real Cyclosporine echo-antiecho dataset (grow both, ~2x each) and a synthetic fixture cross-checked directly against `io.bruker.read_2d` (grow both; separately, shrink both) -- shapes, values, and ppm axes all matched exactly in every case, no crash, and a phase interaction (including the pivot slider) right after a combined resize worked cleanly too. Also checked Reset afterward: correctly restores the original size and keeps the genuine-2D interpretation (doesn't revert to pseudo-2D). New regression test `test_main_window_zero_fill_f1_and_f2_together_matches_read_2d_directly`. Most likely explanation: this concern and the pivot-slider crash (fixed just above) were the same underlying issue -- both are triggered by a live resize of `spectrum.data`/`axes`, and the shape-gated re-anchor fix covers a resize from *either or both* dimensions changing, not just one.

## Pivot-slider crash fixed, genuine-2D phasing now slice-based by default (2026-09-03)

User asked to act on the two previously-noted-only errors, plus a new explicit requirement: "the phasing for a 2D should still default to a 1D slice like it does before you process the indirect dimension rather than trying to phase the entire thing all at once."

- **NSOpenPanel console warning** -- confirmed harmless and not fixable from this codebase: already on the latest PySide6 (6.11.2, checked directly, not assumed); a web search turned up other PySide6-on-macOS users hitting the identical message on recent macOS versions, consistently described as console noise from Qt's own native file-dialog wrapper interacting with AppKit, not an application-level bug. No code change made.
- **Pivot-slider crash, real root cause found**: `_phase_reference_full`/`_phase_freq_ppm` (the phase-editing session's own one-time snapshot, per `_capture_phase_reference`'s own docstring "never called mid-drag") went stale the moment a live FID-processing change (e.g. zero-fill) resized `spectrum.data`/`axes` via the debounced `_recompute_multi_row_data` -- the next phase/pivot tick then wrote that stale-shaped snapshot straight into `spectrum.data`, mismatched against the just-updated, correctly-new-shaped `spectrum.axes`. Fixed by re-anchoring (`_capture_phase_reference()`) inside `_run_debounced_multi_row_reprocess`, but **only when the array's shape actually changed** -- an earlier, broader version (re-anchor on every settle) broke the pseudo-2D baseline-correction on/off toggle instead, since `_phase_reference_full` is deliberately a *clean, phase-only* reference the baseline-correction path is supposed to bypass; re-capturing it from already-baseline-corrected data baked the correction into what's meant to be the "revert to this" snapshot, caught immediately by an existing regression test (`test_main_window_baseline_correct_pseudo2d_respects_apply_all_slices`) failing. New regression test (`test_main_window_pivot_change_after_live_zero_fill_does_not_crash`) confirmed to fail without the fix and pass with it.
- **2D phasing now defaults to a single 1D slice while editing, matching pseudo-2D's own established behaviour** -- the correction itself still applies uniformly to the *whole* array (a genuine-2D phase is one shared correction across every row/column, unlike pseudo-2D's own per-slice option), only the *display* during editing changed: a new `_show_2d_phase_slice` (mirroring `_show_pseudo2d_slice`) shows a single representative row (dim="f2"/direct) or column (dim="f1"/indirect), picked once per `_capture_phase_reference` call as whichever has the largest total magnitude -- not a full contour redraw (the expensive `downsample_2d` call) on every drag tick. Verified end-to-end against the real Cyclosporine echo-antiecho dataset: starts as a contour, switches to a slice line ("F1 row 831 (phase preview)") on the first phase interaction, correctly re-picks a different representative slice ("F2 column 1139") when switching to F1-direction phasing, and the pivot slider works without crashing. 4 new regression tests (default slice-index picking, contour-to-slice display switch, uniform whole-array phase application still holding, slice-index following a dimension switch).
- 785 tests passing (780 -> 785). Not yet committed (writeup in progress).

## Two real errors, noted only per explicit request ("note these for later") -- not yet investigated (2026-09-02)

- **Console warning on opening the file dialog (likely benign, may not even be fixable from this codebase)**: `The class 'NSOpenPanel' overrides the method identifier. This method is implemented by class 'NSWindow'` -- a macOS AppKit method-override warning, printed at `QFileDialog` open time. Looks like a known PySide6-on-macOS Qt-internal quirk (Qt's native file-dialog wrapper overriding an `NSWindow` method `NSOpenPanel` itself already implements) rather than anything in this project's own code -- worth a quick check of whether a newer PySide6 version silences it before spending real time on it, since app-level code has no direct hook into Qt's own native-dialog wrapper internals.
- **Real crash, pseudo-2D phase-pivot slider, x/y shape mismatch**: `ValueError: x and y must have same first dimension, but have shapes (8192,) and (4096,)`, raised in `plotting.oned.plot_1d`'s own `ax.plot(x_plot, y_plot + y_offset, ...)` call. Full traceback: `_on_pivot_slider_changed` -> `_apply_phase_preview` -> `_show_pseudo2d_slice` -> `canvas.show_spectrum(trace)` -> `plot_1d`. `trace` here is a single pseudo-2D slice (`processing.slicing.get_trace`'s own output) -- the mismatch is between that trace's own `x` (ppm axis, 8192) and `y` (data, 4096), a factor of exactly 2x, smelling like a stale/mismatched zero-fill: either the trace's ppm axis was built against a different point count than the data it's now being asked to plot (e.g. a live zero-fill change landing on the data but not the cached axis, or vice versa), specifically surfaced by dragging the phase pivot slider on a pseudo-2D slice. Needs direct reproduction (a pseudo-2D dataset, a live zero-fill change, then a pivot-slider drag) before diagnosing further -- not yet attempted.

## Echo-Antiecho F1-mirror bug, missing F1 apodization, Gaussian/Sine/QSine window parameters, phase-preservation check (2026-09-02)

Session resumed after a "pause on item 2" break; item 2 (2D roadmap remainder) stayed paused per the user's own later instruction ("hold off on adding the new 2D stuff for now -- just finish up all this phase and window function stuff"). Before this round: merged and pushed 4 origin-only commits from collaborator Euan Bassey (a new `src/opennmr/notebook/` subpackage port + a `style.py` font-handling tweak in an unrelated function, `setup_plotting` -- no conflict with this project's own `downsample_2d` changes) -- `git log main..origin/main` showed the 4 commits before merging, `git merge` was clean (auto-merged), 768 tests passed post-merge, pushed as `40d47b5`.

User then reported real Bruker echo-anti-echo 2D data (`test_data/Bruker Example Data/Cyclosporine/203`, a real HSQC-shaped 1H-13C dataset, FnMODE=6) showing "mirroring or something", plus two general worries (window-function change resetting phase; SI for both dimensions) and a "windowing needs real parameters" follow-up.

- **Echo-Antiecho (FnMODE=6) F1-mirror artifact -- found and fixed, real-data verified.** `processing.ft.decode_hypercomplex_f1`'s own docstring had flagged this exact risk beforehand ("no real Echo-Antiecho dataset exists... getting the row order wrong would mirror F1 peaks... without otherwise looking obviously broken") -- now there was real data to check it against. Cross-checked `io.bruker.read_2d` (raw-FID reprocessing) against `read_2d_processed` (TopSpin's own trusted `2rr`) on the real dataset: every genuine peak landed at its correct position, but was accompanied by an *equal-magnitude* ghost at that peak's F1-mirror row (ratio ~0.8-1.2x the genuine peak, across the 10 strongest real peaks) -- confirming the flagged risk was real. Tried all four sign/order permutations of `decode_hypercomplex_f1`'s own post-F2-FFT combine formula (`(echo±antiecho) + i*(echo∓antiecho)`) -- none helped (ghost ratio stayed ~0.8-1.2x for every one), which makes sense in hindsight: FFT does not commute with conjugation (`FFT(conj(x))[k] == conj(FFT(x)[-k])`, not `conj(FFT(x))[k]`), so a fix needs to act on the *raw, pre-F2-FFT* time-domain row, not the combine formula. **Real fix**: conjugate the "echo" (even-indexed) raw row of each pair, in `read_2d`, before F2's own window/zero-fill/FFT -- on top of the pre-existing unconditional `data = np.conj(data)` every row already gets (see that line's own `_CONJUGATE_QUADRATURE_NOTE`), the two conjugates cancel for echo rows (leaving them exactly as stored on disk) while antiecho rows still get the one original conjugate. Dropped the ghost ratio to ~0.005-0.02x (noise floor) on the same 10 real peaks. The GUI's own *separate*, in-memory live-recompute path (`MainWindow._recompute_multi_row_data`, used whenever an F2-only control changes live on an already-loaded 2D) re-implements this same hypercomplex decode independently from cached `fid_raw` -- initially missed the same fix, caught by an existing regression test (`test_main_window_f1_truncate_row_pair_aware_matches_read_2d_directly`, which cross-checks the two paths against each other) failing after the `read_2d` fix landed -- fixed there too, mirroring the identical raw-row conjugate. New test `test_read_2d_echo_antiecho_undoes_the_conjugate_only_for_the_echo_row` (`test_bruker_raw_2d.py`) locks in the actual code behaviour with a hand-constructed single-t1-point fixture (compares the resulting F2 row directly against processing the known on-disk echo FID through the identical window/zero-fill/FFT pipeline, sidestepping the need to interpret the ppm/Hz axis by hand) -- deliberately *not* built as a "recovers a known F1 frequency" fixture the way States/States-TPPI's own test is, since the existing idealized P/N-type synthetic fixture (`_synthetic_hypercomplex_2d_fid`, `fn_mode=6` branch) doesn't model the real raw-quadrature relationship this fix depends on (confirmed empirically: running it through the fixed code recovers the *wrong* frequency, -466.67 Hz instead of the fixture's own true 375 Hz) -- documented as a known synthetic-fixture limitation in that test's own updated comment, not silently left looking like it validates something it doesn't.
- **F1 (indirect-dimension) apodization was completely unimplemented** -- found while investigating the above. `read_2d`'s own docstring claimed "F1 apodization still always follows TopSpin's own recorded proc2s", but the actual F1 pipeline (hypercomplex decode -> zero-fill -> FFT) never applied *any* window to F1 at all, recorded or otherwise -- a real, previously-undetected gap (every real 2D dataset in this project's `test_data/` up to now happened to be pseudo-2D or QF-mode, where this line of code just never ran). The real dataset's own `proc2s` records `WDW=4`/`SSB=2` (QSINE) for F1, same as F2. **Fixed**: `read_2d` gained `window_f1`/`window_kwargs_f1` params (default `None` -> TopSpin's own recorded `proc2s` WDW via the same `_apply_bruker_window` F2 already uses, with the same "leave unwindowed + `window_unsupported_f1` note" fallback for an unsupported WDW code) -- verified against the real dataset (no `window_unsupported_f1`, mirror ratio still ~0.02, no crash).
- **Window-function change resetting phase -- checked, does not reproduce.** Code reading showed `phase0`/`phase1` fed into `apply_phase` (both the single-row live-preview path and the multi-row recompute path) are computed purely from `spectrum.metadata.raw["phase0"/"phase1"]` (+ the live drag delta for the single-row case) -- no window-related term anywhere in that expression. A first empirical check (comparing the *output spectrum's* phase at the discrete magnitude-argmax sample before/after a window change) showed a spurious ~0.5° "difference" -- traced to each window's own different lineshape/spectral-leakage shifting exactly which sample is the argmax for a non-bin-exact synthetic peak, a measurement artefact of that check, not a real phase change. Verified the right way instead: spied directly on the actual `(phase0, phase1)` arguments passed into `processing_ft.apply_phase` across 5 different window selections (exponential/hann/gaussian/sine/qsine) -- identical every time. New regression test `test_main_window_changing_window_function_does_not_change_committed_phase`.
- **Gaussian/Sine/QSine window choices had zero tunable parameters** -- they existed as `window_combo` menu items, but `_resolve_window_kwargs` returned an empty kwargs dict for all three, and the backend's own `_window_gaussian` took a meaningless `std` (a bare sample count, scipy's own generic textbook Gaussian, nothing like Bruker's LB/GB-based GM). Real user report: "EM is fine but things like Gaussian I think need a negative lb, and ssb for qsin processing etc." Checked nmrtools2/leonmr first (per the standing reuse-first instruction) -- neither has *any* Gaussian/SP/QSINE window implementation to port (leonmr only has a plain EM + parameterless Hann), so this was genuinely new work. **Fixed**: every window in `processing.ft`'s registry (bar parameter-free Hann) now goes through `nmrglue.process.proc_base`'s own NMRPipe-compatible `em`/`gm`/`sp` primitives -- the *same* functions `io.bruker._apply_bruker_window` already uses (and has real-data validation for) on Bruker's own recorded `WDW`. `exponential` switched from a hand-rolled `np.exp` call to `proc_base.em` (confirmed algebraically identical first). `sine`/`qsine` gained a real `ssb` parameter (Bruker's own convention, `off = 1/ssb if ssb >= 2 else 0` -- the exact formula `_apply_bruker_window` already uses) -- low-risk, reuses already-validated math. `gaussian` gained real `lb`/`gb` parameters (both in Hz, matching NMRPipe's own published GM formula, `e = pi*i*g1/sw`, `g = 0.6*pi*g2*(g3*(size-1)-i)/sw`, found via a web search after an initial guess -- that "GB is a 0-1 fraction of the acquisition time" -- produced a window crushed to ~0 everywhere except one edge point for any realistic array size; NMRPipe's own docs confirm `g1`/`g2` are *both* "in Hz", not a fraction) -- `g3` (location of the Gaussian maximum) fixed at 0.5 rather than exposed, matching TopSpin's own 2-parameter (LB/GB only) GM UI. **Not verified against real Bruker GM-processed data** (no real GM/WDW=2 dataset exists in this project's `test_data/` to cross-check against, same standing as `_apply_bruker_window`'s own already-deliberate WDW=2/GM gap) -- flagged in `_window_gaussian`'s own docstring, not silently presented as certain, same pattern as the Varian ppm-axis and JEOL `FILTER_FACTOR` gaps. GUI gained `gb_spin`/`ssb_spin` (new), `lb_spin`'s range extended to allow negative values (needed for Gaussian's own sign convention), all three enabled/disabled per the current `window_combo` selection via a new shared `_update_window_param_spin_enabled` (replacing 3x duplicated single-line `lb_spin.setEnabled(...)` calls), persisted/restored across reselection the same way `lb_spin` already was. 9 new tests across `test_ft.py` (window-shape/parameter correctness, including a cross-check that the generic registry's `ssb` produces the identical window `_apply_bruker_window`'s own recorded-WDW path does) and `test_gui.py` (enable-state switching, reprocess-doesn't-crash, persistence).
- **Zero-filling "looks like a bunch of zero points appended, instead of interpolated"** -- not a bug, standard NMR convention: zero-filling *is* literally appending zeros to the time-domain FID's tail (confirmed the code's own window-then-zero-fill order is correct throughout, matching every vendor's `read_1d`/`read_2d`) -- the "interpolation" it provides is an emergent property of the FFT of a zero-padded array (sinc interpolation in the frequency domain), not something that shows up in a time-domain FID preview. Explained to the user rather than changed.
- **"SI for both dimensions" (F1+F2 combined zero-fill) -- investigation started, then explicitly paused** ("hold off on adding the new 2D stuff for now") before reaching a conclusion. Not yet reproduced against real data or the GUI's own combined-zero-fill code path -- next step whenever this is picked back up.
- Still outstanding from the same user message, not yet started: two error reports the user referenced as "attached below" but that never actually arrived in the message text -- need to ask for the real error content before these can be investigated. Also outstanding: "there should just be one button that can open files or folders" (currently two separate actions/buttons) -- a small, well-scoped UI request, not started this round.
- 780 tests passing (768 -> 780, +12 net). Not yet committed or pushed this round (git history/commit split still pending as this write-up lands).

## Autonomous 01:42 session, item 1: the three real-EXSY-data reports (2026-09-02)

Cron-fired continuation of the "wait 2 hours" request. Real EXSY data (`test_data/2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119/2019` and `.../2011`) let all three of the previously "noted only" EXSY items get genuine direct reproduction. **Methodology note for future sessions**: reproducing against this real data via the actual `MainWindow`/`io.persistence` sidecar-save machinery writes real `opennmr_metadata.json`/`opennmr_processing.json` files into the experiment folder as a side effect (the sidecar-save timer fires even in a headless script with no `app.exec()`, confirmed directly, not assumed) -- monkeypatch `io.persistence.metadata_sidecar_path`/`processing_sidecar_path` to a scratch directory before repeated reproduction scripts against real data, and always check for + clean up any accidentally-written sidecar files in real experiment folders afterward (confirmed via file timestamp which ones were mine before removing them -- a `2023/` experiment in the same parent folder had its own pre-existing sidecar files from unrelated real usage, left untouched).

- **"F1 dimension truncate is stuck at 16 points"** -- thoroughly investigated against both real EXSY experiments (2019: 1002 raw rows -> 501-point F1 max; 2011: 504 raw rows -> 252-point F1 max), through a realistic full interaction sequence (load, toggle Process Indirect Dimension on/off/on, set a live truncate value, reselect away and back) -- **does not reproduce with the current codebase.** Confirmed `tdeff_f1_spin`'s own max/default computation (`_refresh_fid_processing_controls`) already uses `.shape[0]`/`.shape[-1]` correctly, not `len()` -- a different code path than the X-Pulse array-step-count bug fixed earlier this session, and not affected by the same class of bug. Most likely explanation: already fixed by this session's own earlier F1 Truncate/Zero-fill build (which predates when this report was actually investigated). **Not closed with full certainty** -- if the user still sees this on their own real data, the next step is getting the specific dataset/procno it happens on, since it doesn't reproduce on the two EXSY files available locally.
- **"Reset in processing 2Ds defaults it to a 1D"** -- reproduced directly: not a literal dimensionality collapse (`spectrum.ndim` stayed 2, the Spectrum tab correctly kept showing a real contour, not a line plot, and every pseudo-2D-specific widget's own `isHidden()` state was correct throughout) -- the real, confirmed behavior is that `_on_reset_processing_clicked` reverted a genuine-2D dataset's own "Process Indirect Dimension" opt-in back to pseudo-2D (the `_load`-time default), on top of discarding the window/zero-fill/phase/truncate choices it was actually meant to discard. A real, understandable point of confusion, even though technically "working as designed" -- pseudo-2D's own per-slice/index treatment is what read as "a 1D" to the user, not a genuine ndim change. **Fixed**: Reset now preserves whatever pseudo-2D/genuine-2D interpretation is currently in effect (`spectrum.is_pseudo_2d` carried forward into the reset's own `pseudo_2d` kwarg) instead of reverting it to the load-time snapshot -- "Process Indirect Dimension" is a structural interpretation choice, not a processing recipe, so Reset (which only ever promised to discard "processing choices") shouldn't have been touching it either way.
- **"Processing tab vs. Spectrum tab showing two different 2Ds... EXSY diagonal in spectrum but other different contours in processing"** -- the most concerning of the three, confirmed as a *real*, previously-unknown bug (not user error, not a stale-data bug either -- both canvases were already provably showing the *exact same spectrum object*, confirmed directly: `processing_preview_canvas._contour_spectrum is window.canvas._contour_spectrum is spectrum`). Root cause: `plotting.style.downsample_2d` (used for the Processing tab's own default-downsampled preview, ~16x fewer total points at the default 0.25 contour fraction vs. the Spectrum tab's own full-resolution default) picked evenly-spaced *indices* and threw away everything else -- exactly the algorithm this project ported from `nmrtools2` unchanged. For a thin, sparse feature like an EXSY/NOESY diagonal ridge (often 1-2 points wide at real digital resolution), that can step clean over the whole feature and lose most or all of it, even though the underlying data never changed -- confirmed directly on the real EXSY file (a synthetic irrational-slope ridge test showed the old method losing ~93% of a ridge that should have survived). **Fixed**: `downsample_2d` now keeps each output bin's own largest-magnitude (signed) value instead of an index pick -- checked `nmrtools2` first per the standing "reuse existing code" instruction and confirmed its own `downsample_2d` has the identical naive-decimation bug (this project's own copy was ported from it unchanged), so there was nothing better to port and this is genuinely new work. **Known, accepted performance trade-off**: the new method has to touch every input element at least once (unlike index-skipping, which only ever touched the output-sized subset) -- measured directly at ~0.35s on the real 1024x65536 EXSY array, vs. a few ms for the old method. Acceptable for a one-off settled redraw, a real cost during an active *live 2D phase-drag* on data this large specifically (redraws on every mouse-move tick, no in-place contour update path exists). Not optimized further -- the actual fix for 2D drag responsiveness at any data size is the already-queued "per-row/column-only 2D phasing" item below, not squeezing more speed out of a full-contour downsample that a proper per-row drag wouldn't even need to run.
- 8 new regression tests (`test_plotting_style.py`, new file: ridge preservation vs. the old naive method side-by-side, sign preservation, no-op when already within limits, non-divisible axis lengths, output shape correctness; `test_gui.py`: Reset preserving vs. correctly-still-reverting the Process Indirect Dimension choice, plus an existing F1-truncate Reset test updated for the new expected behavior). 768 tests passing (760 -> 768). Committed; not yet pushed. **Item 1 of the roadmap is now closed out -- item 2 (2D roadmap remaining items) is next.**

## LWG roadmap 2026-09-01: I/O -> 2D/downsampling -> fitting wizard, in that order

## Scheduled: autonomous continuation at 01:42 CEST 2026-09-02

User: "I want you to wait 2 hours and then get on with the rest of the to dos that we discussed earlier." A one-shot `CronCreate` job was set for 01:42 local (2 hours from 23:42 CEST 2026-09-01) to resume without further prompting -- same "note now, autonomously resume at a specific clock time" pattern as the earlier 2026-09-01 02:01 session (see that entry further down), including the same reasoning for writing full context here immediately: the session may summarize/compact before the scheduled time.

**Real EXSY test data found and confirmed while writing this note** (wasn't known to be available when the three EXSY items below were first written up as "noted only, not yet investigated") -- `test_data/2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119/2019` (mixing time 3.2 ms) and `.../2011` (mixing time 0.2 ms), both `##$PULPROG= <exsy_echo3.enb>`, a real 2H EXSY 2D dataset. This means all three EXSY items below can now get genuine direct reproduction, not just static-analysis guessing -- start there.

**Priority order for this session, matching this project's own standing discipline throughout (investigate/reproduce directly before fixing, write regression tests, run the full suite, commit each logical round, don't push to origin/main without asking)**:

1. **The three "LWG Notes 2026-09-01, real EXSY 2D data" items below** (F1 truncate stuck at 16 for Bruker/EXSY specifically -- note this is a *different* root cause than the just-fixed X-Pulse array-step-count bug, since Bruker's own F1 `n1_native` computation already uses `.shape[0]` correctly, not `len()` -- so don't assume the same fix applies; Reset showing a 2D as "a 1D"; Processing tab vs. Spectrum tab showing different contours for the same 2D, the most concerning of the three) -- now directly reproducible against the real file above. Load it into the GUI (or a headless script) exactly as a user would, reproduce each report literally, THEN diagnose and fix.
2. **2D roadmap remaining items** (see "LWG roadmap" entry below for the full list: per-row/column-only 2D phasing for responsiveness, cursor-default row/slice extraction, the diagonal-line overlay -- check `plotting.twod.plot_2d_contour`'s own already-existing `show_nQ_line`/`nQ`/`nQ_colors` params first, per the standing "reuse existing code" instruction, before building anything new -- 2D contour overlay, indirect-dimension carrier-frequency verification against `nmrtools2`'s own code, and a contour/heatmap toggle for pseudo-2D/stacked-1D data).
3. **Fitting improvements** (see "Fitting wizard and beyond" entry below) -- implement the concrete, well-specified pieces (per-parameter fixing rather than only a whole-peak Fixed checkbox, bounding "possibly from integrals", a too-close-to-region-edge warning, a more readable/exportable fit report, slice-to-slice parameter propagation/seeding). **Hold off on committing to the full "fitting wizard" UI/workflow restructure itself** -- that's a bigger design commitment than the rest of this list, better suited to a real-time check-in with the user than an unattended autonomous session; implement the underlying capabilities the wizard would need, but don't build the wizard flow itself without further input.

No requirement to finish the entire list in one sitting -- work steadily down it in this order (matching how the earlier 2026-09-01 02:01 session did items 1/2/3 one at a time, each written up and committed separately), stop at a natural/sensible boundary, and write up TODO.md/the project memory file for whatever's actually done vs. still deferred.

## X-Pulse: Truncate maximum wrongly capped at the array-step count, not the real FID length (2026-09-01)

Real user report: "there is a default about truncation that only goes up to 16, meaning that I can't process more than 16 data points." Root-caused by direct reproduction (a synthetic 16-step arrayed X-Pulse dataset), not guessed at.

- `_refresh_fid_processing_controls`'s own `tdeff_spin` (F2 truncate, X-Pulse-only) maximum was `len(raw_fid)` -- correct for a plain single-file X-Pulse load (`fid_raw` is 1D), but for an *arrayed* experiment (`io.xpulse.read_array` -- a PFG/T1/T2 series with `array1/`..`arrayN/` subfolders), `metadata.raw["fid_raw"]` is genuinely 2D: `(n_steps, NP)`. `len()` reads axis 0 (the *step* count), not `.shape[-1]` (the real per-FID point count) -- so a real 16-step series capped Truncate's own maximum at 16 regardless of the true acquired length. Same "`len()` reads the wrong axis for 2D input" bug class already found and fixed once this session, in `processing.ft.fft_spectrum`. Fixed with the same one-line pattern: `raw_fid.shape[-1]` instead of `len(raw_fid)` -- correct for both the 1D (single-file) and 2D (arrayed) case.
- **Audited "for all vendors" per the user's own explicit ask, not just patched the one report**: grepped every `len(...)` call on a raw-FID-like variable in `main_window.py` -- this was the only one. F1's own truncate/zero-fill (`tdeff_f1_spin`/`zero_fill_f1_spin`, already vendor-agnostic since this session's earlier "F1 Truncate/Zero-fill" round) already used `.shape[0]`/`.shape[-1]` explicitly, not `len()` -- confirmed not affected by this same bug.
- 1 new regression test (`test_gui.py`, a synthetic 16-step arrayed X-Pulse fixture with a 4-point-per-FID fixture -- deliberately step-count != point-count so the old-buggy-value (16) and the fixed value (4) can't be confused with each other by coincidence). 760 tests passing. Committed; not yet pushed.
- **User flagged this as the first of "some old and some new bugs with x-pulse data"** -- more X-Pulse reports are expected; this is not necessarily the whole list.

## Phase display redesigned: non-editable current-value labels, not a read-only-label-plus-editable-box (2026-09-01)

Follow-up correction to the previous round's fix: "can you just display non-editable numbers with current phase instead of that editable box? That way you can use whatever numbers internally and the user sees the relevant values. We don't need to see the zeroed out phase values."

- Removed the separate `current_phase_label` line added last round; instead **replaced Phase 0/Phase 1's own visible editable spin boxes with non-editable `QLabel`s** (`phase0_current_display`/`phase1_current_display`, selectable-but-not-editable, showing the real absolute total, e.g. "45.00°") placed directly next to each drag handle, in the spin boxes' own old spot.
- **Internals unchanged, exactly as the user suggested** ("use whatever numbers internally"): `phase0_spin`/`phase1_spin` (the old editable `RelativeDoubleSpinBox`es) still exist and are still the real drag-delta state `DragHandle` stays synced with -- just never added to any layout any more, so every one of the ~30 existing call sites that read/write `.value()` (`_apply_phase_preview`, `_commit_pending_phase_to_metadata`, `_compute_processing_preview`, `_current_phase_kwargs`, quick-phase nudge buttons, autophase/revert-autophase, pivot-move phase-folding) needed zero changes. `_pair_widget`'s signature generalized (`display: QWidget` instead of `spin: QDoubleSpinBox`) since it now pairs a handle with a label, not a handle with a spin box.
- `_refresh_current_phase_displays` (renamed from last round's `_refresh_current_phase_label`) now updates both labels directly instead of one combined line, same two call sites as before (`_capture_phase_reference`, `_apply_phase_preview`).
- Updated `_set_processing_enabled`'s widget list to also grey out the two new displays when phase editing isn't available, and updated a pre-existing test (`phase_spin_box_wide_enough_for_large_values` -> the displays, not the now-hidden spin boxes) plus rewrote the 3 tests from last round's own fix. 759 tests passing. Committed; not yet pushed.

**Also, a standing instruction restated with emphasis, not specific to this item**: "Many of the things I've requested definitely exist inside the codebase I initially provided [nmrtools2/leonmr]. Always use that code as a basis instead of writing from scratch." Saved as a dedicated feedback memory (`feedback_reuse_reference_code.md`) rather than left as a one-off -- check those two repos for an existing implementation before building any new opennmr feature from scratch, going forward. (Checked for this specific phase-display item -- confirmed nmrtools2/leonmr have no GUI code at all, so there was genuinely nothing to reuse here; this rule matters most for the still-queued 2D-roadmap and fitting-wizard items, which are much more likely to have real processing/plotting analogues worth checking first.)

## Current-phase display fix, user-adjustable 2D downsample fraction, per-probe gradient config (2026-09-01)

Three quick items ahead of (and the start of) the 2D roadmap's downsampling piece.

- **"The phase value is not only stored but also read into the GUI, it resets to zero when switching between experiments"**: Phase 0/Phase 1 are (deliberately, by design -- `_capture_phase_reference`'s own docstring) a drag *delta* from whatever the view looked like when editing started, always zeroed on reselection -- not a bug, but nowhere showed the *actual* applied value, so it read as the phase itself being lost. Rather than changing the drag/delta semantics (would need touching every phase-application call site -- `_apply_phase_preview`, `_commit_pending_phase_to_metadata`, `_compute_processing_preview`, `_current_phase_kwargs` -- real regression risk across this app's most heavily-tested interaction), added a new read-only `current_phase_label` ("Current phase: X° / Y°") next to the Phase 0/Phase 1 controls, refreshed from `_capture_phase_reference` (every reselection/slice/dimension change) and live during a drag (`_apply_phase_preview`). Spin boxes' own delta behavior is unchanged.
- **User-adjustable 2D downsample fraction**: `plotting.plot_2d_contour`'s own hardcoded default (`contour_fraction_x/y=0.25`, "4x less") is now `MplCanvas._contour_fraction` (still defaulting to 0.25), user-adjustable via a new "2D downsample" % spin box in the Output tab (`_on_contour_fraction_changed`, applies to every canvas). New `MplCanvas.set_contour_fraction` mirrors `set_full_resolution`'s own rebuild-if-currently-shown pattern; recorded even while Full Resolution is on (so it's already correct the moment that's turned back off), but has no visible effect until then.
- **Per-probe app-wide config**: explicit request -- "a configuration that each user can save directly into the app to store info like max gradient strength that can be associated automatically with the probe." New `~/.opennmr/probe_config.json` (same user-dotfile convention as the existing `figure_styles.json`), a plain `{probe_string: {max_gradient_g_per_cm: ...}}` dict. A new "Save as default for this probe" button next to the imaging Max gradient spin box writes the *current* value under this spectrum's own recorded probe; on any other spectrum reselection/load with no per-experiment gradient of its own already set, this now applies **automatically** (no separate "load" step, unlike the figure-style templates' own explicit save/load pair) -- but a per-experiment value, once set, always wins over the probe default (not written back into `metadata.raw`, so a later probe-config change can't silently overwrite an experiment that already has its own explicit choice).
- 18 new tests (`test_gui.py`: current-phase-label dash/live-drag/reselection-survival; `test_gui_canvas.py`: `set_contour_fraction` changing the default kwargs, rebuilding a shown contour, recorded-but-inert under Full Resolution; `test_gui.py` again: contour-fraction spin box default/applies-to-every-canvas, probe-gradient save/no-probe-message/auto-apply/per-experiment-override/different-probe-no-op). 758 tests passing (745 -> 758). Committed; not yet pushed.

## LWG roadmap 2026-09-01: I/O -> 2D/downsampling -> fitting wizard, in that order

Three-part message, explicit ordering confirmed by the user as sensible. **I/O and 2D/downsampling are both now fully done** (as of 2026-09-03) -- every sub-item below shipped across this and earlier sessions, most recently the EXSY items (closed just above) and this session's own genuine-2D slice/preview rework. **Fitting wizard is the one remaining phase, not yet started.**

**2D/downsampling** (all done): downsample fraction (user-adjustable, default 0.25) -- done. Phasing a 2D operating on one row/column at a time -- done (this session, several rounds, most recently "always slice 1"). Row/slice extraction defaulting to the cursor's position -- done ("2D roadmap, Extract Slice..."). Diagonal-line (nQ) overlay -- done ("2D roadmap, diagonal-line (nQ) overlay..."), reusing `plotting.twod.plot_2d_contour`'s already-existing `show_nQ_line`/`nQ`/`nQ_colors` params as suspected. Multiple-2D-contour overlay -- done ("2D roadmap, overlay multiple 2D contours..."). Indirect-dimension carrier frequency verified against `nmrtools2` -- done, exact formula match, no fix needed. Pseudo-2D contour-vs-heatmap toggle -- done. EXSY "processing vs. spectrum different contours" -- closed, see above.

**Fitting wizard and beyond** (started 2026-09-03, every concrete item now done): a better overall fitting workflow. ~~Individual parameters (not just an entire line) should be fixable~~ -- done, see "Analysis (peak fitting / relaxation)" below. ~~Bounding "possibly from integrals"~~ -- done 2026-09-04, see "Bounding peak amplitude from Integration Region integrals" above (opt-in "Bound amplitude using region integrals" checkbox, derives an amplitude upper bound per peak per slice from its own Integration Region's raw integral). ~~A sensible warning when a fit result lands too close to its own region edge~~ -- done, see "Analysis (peak fitting / relaxation)" below. ~~A more readable, exportable fit report~~ -- done, see "Analysis (peak fitting / relaxation)" below. ~~A way to fit one slice first, then either apply those same parameters to every other slice or use them as each next slice's own starting guess~~ -- done, see "Analysis (peak fitting / relaxation)" below. **Every concrete fitting-wizard item from the original roadmap message is now done.** The suggested UI shape: a "fitting wizard" walking the user through the steps, rather than the current all-at-once Peak Picking/Integration tab layout, is still not started -- explicit user-set boundary, hold off on that restructure without a real-time check-in first. Longer-term: multivariate fitting, specifically named for PFG and T1/T2.

## I/O: persistent sidecar files for metadata and processing parameters (2026-09-01)

First of a three-part roadmap the user laid out (I/O, then 2D/downsampling, then a fitting wizard) -- explicit instruction to do I/O first and ask clarifying questions before starting, which happened via three targeted questions (write trigger, auto-load-on-reopen, per-procno-vs-shared scope) before any code was written. Answers: live/debounced writes, always auto-load on reopen, one shared processing file for the whole experiment (not per procno).

- **New `io.persistence` module**: two JSON sidecar files per experiment, `opennmr_metadata.json` (sample-level facts: vendor/nucleus/nucleus_indirect/probe/sfo1/procno/mass_g/pseudo_2d) and `opennmr_processing.json` (the *entire* `metadata.raw` dict, filtered to whatever's actually JSON-serializable minus a small blocklist of large/redundant vendor-param-dump and raw-FID-array keys) -- deliberately not a hand-maintained key list for the processing file, so any *future* `raw[key] = ...` addition is automatically covered without this module needing a matching update. Written directly inside `source_path` when it's a directory (Bruker/Varian/folder-per-experiment X-Pulse); as `<basename>.opennmr_*.json` alongside it when `source_path` is a single file instead (JEOL, and a bare-file X-Pulse dataset -- several JEOL experiments in this project's own test_data share one directory with no per-experiment subfolder, so a directory-scoped name there would collide).
- **`NMRMetadata` gained a `probe` field**, explicit user ask ("definitely should for Bruker under probhd, and may also exist for x-pulse in some cases") -- confirmed both exist on real test_data (`##$PROBHD` for Bruker, `$$Probe` for X-Pulse) and wired into every read path for both, plus Varian's own `procpar` `probe` field (found while checking, not asked for specifically -- JEOL's own reference parser wasn't checked for an equivalent, left `None` there rather than guessing a field name). Shown in the Data tab's info label when recorded ("take note of" -- the user's own phrasing suggested visibility, not just internal bookkeeping).
- **GUI wiring**: `MainWindow._load` now loads both sidecar files (if any) before selecting the freshly-loaded spectrum, merges the saved processing dict straight into `metadata.raw` (so the *existing* restore-on-reselection machinery -- `_refresh_fid_processing_controls`, `_restore_analysis_state`, etc. -- picks it up with no new restore logic needed), then a new `_apply_loaded_processing_sidecar` actually recomputes `spectrum.data` from it (synchronously, not through the usual debounce, since this is a one-shot load-time step) and reapplies a saved SR shift. A saved "Process Indirect Dimension" opt-in (`pseudo_2d: false`) now also overrides `_load`'s own existing "always default a fresh 2D to pseudo-2D" behaviour, so a genuine-2D user doesn't have to redo that opt-in after every restart.
- **Debounced auto-save**: a new `_sidecar_save_timer` (800ms, longer than the existing 200ms recompute-debounce timer since a disk write has no on-screen result to wait for), scheduled from every place that already persists live state into `metadata.raw` in-memory (FID-processing controls, phase commit, baseline toggle/options, analysis-state save, nucleus override, mass, SR shift, imaging gradient, Process Indirect Dimension toggle, Reset). **Reselecting away from a spectrum synchronously flushes its own pending save first** (mirroring the existing `_save_current_analysis_state` "commit the outgoing spectrum's own state" pattern) -- otherwise the single-pending-index debounce (same limitation `_live_reprocess_timer` already has) would silently drop an edit made right before switching spectra. **Reset also schedules a save** so a discarded processing recipe doesn't quietly resurrect itself from an old sidecar file on the next reopen.
- **Known limitation, not fixed this round**: the single-pending-index debounce can still drop an edit if two *different* spectra are edited within the same 800ms window without a reselection between them (reselection is the only flush point) -- narrow, matches an existing pre-existing limitation in the unrelated recompute timer, not fixed here.
- 30 new tests (`test_persistence.py`: sidecar path naming for directory- vs file-scoped experiments, save/load round-trips for both files, serializability filtering, corrupted-file fail-soft, `pseudo_2d` recorded only for real 2D data; `test_gui.py`: live-edit-schedules-a-save, reopening-in-a-fresh-MainWindow restores zero-fill/phase/mass/nucleus/F1-truncate, the outgoing-spectrum flush-on-reselection fix, Reset updating the saved sidecar, the no-sidecar-yet no-op case, the info-label probe display; plus `probe`-field tests for Bruker `read_1d`/`read_2d`, X-Pulse `read_1d`, Varian `read_1d`). 745 tests passing (715 -> 745). Committed; not yet pushed.

## LWG Notes 2026-09-01, real EXSY 2D data -- all three items closed, none currently reproduce (2026-09-03)

User's own words, verbatim, three separate reports against a real EXSY dataset. Investigated 2026-09-03 by direct reproduction against both real files (`test_data/2p5mm_HX_2H_23Na_113Cd_Na2CdFeCN6_D2O_150126A_20260119/2011` and `.../2019`, mixing times 0.2/3.2 ms, both real `##$PULPROG= <exsy_echo3.enb>` States-TPPI 2D datasets, `FnMODE=5`) -- all three already resolved, none by new work this round:

- **"F1 dimension truncate is stuck at 16 points for some reason."** Does not reproduce on either real file: `tdeff_f1_spin.maximum()` is 252 (2011) and 501 (2019) -- correctly `fid_raw.shape[0] // 2` (row-pair-halved for `FnMODE=5`), not 16. `fid_raw` is populated on both (`read_2d_shape_guess_failed` unset) -- the "raw ser shape guessing fell back to a small guess" mechanism this note originally suspected isn't in play for these files.
- **"Reset in processing 2Ds defaults it to a 1D."** Does not reproduce: `_on_reset_processing_clicked` on a genuine 2D correctly keeps `spectrum.is_pseudo_2d == False` and `process_indirect_button` checked. **Already fixed by an earlier round** ("Autonomous 01:42 session, item 1" below) -- Reset now carries `spectrum.is_pseudo_2d` forward into its own reset kwargs instead of reverting to the load-time pseudo-2D-by-default snapshot; this note predates that fix.
- **"Seeing two different 2Ds between the processing and spectrum views... EXSY diagonal in spectrum but other different contours in processing."** The literal symptom this described -- the Processing tab attempting its own full contour that could diverge from the Spectrum tab's -- is now categorically impossible: per this session's own "Genuine 2D Processing-tab preview" fix (above), the Processing tab never draws a contour for a genuine 2D any more, always a single slice. What it shows instead (confirmed well-formed on both real EXSY files -- finite, sensibly-scaled, correctly ppm-ranged) is **raw t1-increment 1's own F2-only spectrum** (literally "xf2, unprocessed indirect," per this session's own later "always slice 1" request) -- a *time-domain* slice, deliberately different in kind from the Spectrum tab's *frequency-domain* F1-bin-1 slice shown during phase editing (confirmed by direct comparison: the two don't numerically match, which is expected and correct, not a discrepancy -- they're slices along different axes of the same correct dataset, not two different datasets).

## Manual peak picking now shows triangle markers; region shading survives reselection (2026-09-01)

User: "manual peak picking also needs triangle markers." Auto-Pick already called `canvas.set_peak_markers(...)` explicitly after filling the peak table, but nothing else did -- a manual add (the Add Peak button, or click-to-pick via `_on_peak_picked`, both routed through `_add_peak_row`), Remove Peak, Clear All, and a direct in-table edit all left the canvas either showing nothing or showing a stale marker set.

- New `_refresh_peak_markers()` (mirrors the existing `_sync_region_shading()`, same "table and canvas must always agree" reasoning), reading every row's own shift and calling `canvas.set_peak_markers`. Wired into `_add_peak_row` (covers manual add + click-to-pick + Auto-Pick's own loop, so Auto-Pick's separate explicit call was removed as redundant), a new `_on_remove_peak_clicked`/`_on_clear_peaks_clicked` follow-up call, and `peak_table.itemChanged` (a typed-in edit moves the marker too, matching `region_table`'s own existing `itemChanged` wiring). `clear_peaks_button`/the menu action now call `_on_clear_peaks_clicked` instead of a bare `lambda: peak_table.setRowCount(0)`.
- **A second, real bug found and fixed along the way, not just the reported one**: reselecting a spectrum didn't restore its own saved peak markers (or region shading) at all -- confirmed by direct reproduction before fixing. Root cause: `_restore_analysis_state`'s own `_sync_region_shading()` call runs *before* `_draw_selected_spectrum`'s redraw in `_on_selection_changed`'s call order, and `MplCanvas.show_spectrum`'s `figure.clear()` discards the old marker/shading state, throwing away that earlier call. This was already true for region shading before this round touched anything -- confirmed by reproducing it in isolation first. Fixed by also calling `_sync_region_shading()`/`_refresh_peak_markers()` from the end of `_draw_selected_spectrum` itself, the actual choke point every redraw (reselection, pseudo-2D view combo, etc.) goes through -- both calls are safe no-ops via their own existing canvas-side guards (`_pivot_marker_ax`/`_current_1d_spectrum` being `None`, a stack view, etc.) when there's nothing sensible to draw on.
- 6 new regression tests (`test_gui.py`): manual add, click-to-pick, remove, in-table edit, and reselection-survival for peaks; reselection-survival for regions (the same underlying bug, confirmed broken first via a standalone repro before the fix). `test_main_window_clear_all_peaks_button` updated to call the real `_on_clear_peaks_clicked` handler instead of asserting against the old bare `setRowCount(0)` it used to describe. 715 tests passing (709 -> 715). Committed; not yet pushed.

## Autonomous 02:01 session, item 3: F1 (indirect dimension) Truncate/Zero-fill, any vendor (2026-09-01)

Closes out the "LWG Notes. 31/08/2026 23:16" list -- all three items now done.

- ~~**"I want the TD and SI options to appear for both dimensions when any 2D is loaded"**~~ -- F2 already had `tdeff_spin`/`zero_fill_custom_spin`, but F2's own truncate control is X-Pulse-only (Bruker's `read_1d`/`read_2d` had no `truncate`-equivalent exposed), and neither vendor had an F1 equivalent at all. Added two new vendor-agnostic controls, `tdeff_f1_spin`/`zero_fill_f1_spin` ("Truncate F1 to:"/"Zero-fill F1 to:"), enabled whenever the selected spectrum is a genuine 2D with a real F1 FFT -- `spectrum.ndim == 2 and not spectrum.is_pseudo_2d` -- for *any* vendor, not gated on `is_xpulse` the way F2's `tdeff_spin` is.
  - **io level**: `io.bruker.read_2d` gained `zero_fill_f1: int | None`/`truncate_f1: int | None` params (X-Pulse's `read_2d` already had `truncate_f1`/`zero_fill_f1`, just not wired into the GUI yet). Both are row-*pair*-aware: applied to raw row pairs before `decode_hypercomplex_f1` whenever `FnMODE` calls for hypercomplex decoding (4/5/6), so a truncation request never separates a real row from its matching imaginary one -- applied directly to raw rows otherwise (FnMODE 1/QF, every genuine 2D dataset actually in this project's own `test_data`). Same convention both vendors already used for F2's own truncate.
  - **GUI live-recompute**: `_recompute_multi_row_data`'s F1 section (previously a fixed no-op re-FFT back to whatever `spectrum.data.shape[0]` already was, since no live F1 control existed to change it) now reads `metadata.raw["live_truncate_f1"]`/`["live_zero_fill_f1"]` and applies the same row-pair-aware truncate before `decode_hypercomplex_f1`, then zero-fills to the live target instead of the old fixed length. The F1 ppm/Hz axis is rebuilt at the new length via `np.linspace(old_ppm[0], old_ppm[-1], new_si1)` -- F1's ppm *span* depends only on the acquisition itself (transmitter offset + spectral width), never on how many points it's sampled into, exactly the same reasoning F2's own live zero-fill (`ppm2 = np.linspace(reference_ppm[0], reference_ppm[-1], data.shape[1])`, already in this function) already relies on -- kept vendor-agnostic this way rather than re-deriving `o1_1`/`sf_1`/`obs_1`/`sw_1` from vendor-specific raw params.
  - **A real ordering bug caught before it shipped**: the first draft gated visibility on `self._phase_mode == "2d"` (matching an inline comment left on the widget constructors from before this session's compaction) -- but `_capture_phase_reference` (which sets `_phase_mode`) runs *after* `_refresh_fid_processing_controls` in `_refresh_selection_display`'s own call order, so `_phase_mode` is always one selection stale at the point these controls are refreshed. Fixed by computing `is_genuine_2d` directly from `spectrum.ndim`/`spectrum.is_pseudo_2d` instead (matching the same computation `_refresh_selection_display`'s own Dimension-controls visibility already uses a few lines later), not from `_phase_mode` at all.
  - Persistence follows the exact same mechanism as every other live FID-processing control: `_persist_fid_processing_controls_to_metadata` writes `live_truncate_f1`/`live_zero_fill_f1` into `spectrum.metadata.raw` whenever the corresponding spin box is enabled, `_refresh_fid_processing_controls` restores them on reselection. Reset (`_on_reset_processing_clicked`) needs no special-case code -- it already replaces the spectrum object wholesale via `session.reset`, which naturally drops these along with everything else live.
  - **Confirmed truncate-alone doesn't shrink the final row count** -- a separately-recorded zero-fill target still applies afterward and can zero-fill back up past the truncation point, exactly mirroring F2's own already-established truncate/zero-fill independence (and the io-level tests in `test_bruker_raw_2d.py` from earlier this session). Not a bug -- both new GUI tests set truncate and zero-fill together to actually observe a smaller final spectrum.
- New tests: 3 io-level (`test_bruker_raw_2d.py`, Bruker `read_2d`'s own `truncate_f1`/`zero_fill_f1`, row-pair-aware vs. direct-row cases) plus 6 GUI-level (`test_gui.py`: gating enabled only for genuine 2D across both vendors, live truncate+zero-fill actually shrinking the row count for Bruker and X-Pulse, a direct cross-check of the live in-memory recompute path against calling `io.bruker.read_2d(truncate_f1=..., zero_fill_f1=...)` directly on the same FnMODE-6 States-TPPI-style synthetic data -- catches a row-pairing/axis-length bug a shape-only assertion wouldn't -- persistence across reselection, and Reset clearing both). Full suite green (700 -> 709 passing). Committed; not yet pushed.

## Autonomous 02:01 session, item 2: main Spectrum tab no longer silently downsampled (2026-09-01)

Continuing the "LWG Notes. 31/08/2026 23:16" list, in order. Item 2 done; item 3 (TD/SI for both dimensions) is next.

- ~~**"The spectrum is severely downsampled right now. Only the spectrum in fid processing should be... I need to have the fully sampled spectrum in the spectrum tab"**~~ -- confirmed exactly as diagnosed before deferring: `MplCanvas.show_spectrum`'s 2D branch calls `plotting.plot_2d_contour` without overriding its own `contour_fraction_x`/`contour_fraction_y` defaults (0.25 each, capped at 16384/1024 points) -- meaning *every* 2D contour render, including the main Spectrum tab, silently rendered at a quarter of its own real resolution with no way to tell short of already knowing to look for it. Confirmed directly on the real 128x2048 HSQC dataset: rendered at 32x512 by default.
  - This project already had the right mechanism for exactly this, just never extended to 2D: `MplCanvas._full_resolution` (an existing per-canvas toggle, previously 1D-only -- `plot_1d`'s own rendering-only downsampling, with a working right-click "Full Resolution" menu action and an `export_full_resolution` escape hatch that always uses true full resolution for a save regardless of the live view). Extended all three to also cover a 2D contour: `show_spectrum`/`_redraw_contour` now pass `contour_fraction_x=None, contour_fraction_y=None` (via a new shared `_contour_fraction_kwargs()` helper) when `_full_resolution` is on; `has_downsampled_data` (gates whether the menu toggle is even offered) now also returns true whenever a 2D contour is showing at all; `set_full_resolution`/`export_full_resolution` both now trigger a full `show_spectrum` rebuild for the 2D case (a `QuadContourSet`, unlike a `Line2D`, has no reusable "already-plotted full-resolution data" to swap back in place) -- `export_full_resolution`'s own rebuild explicitly saves/restores the current pan/zoom around it, so a save/export never silently resets the view.
  - **Per-canvas default, not a global one, matching the user's own explicit split**: `MplCanvas.__init__` gained a `full_resolution: bool = False` constructor parameter; `MainWindow` passes `full_resolution=True` only for `self.canvas` (the main Spectrum tab) -- `processing_preview_canvas` (FID processing) and every other canvas keep the existing fast/approximate default, matching "only the spectrum in fid processing should be [downsampled]" precisely. This also means the main Spectrum tab's own **1D** rendering is now full-resolution by default too (the same `_full_resolution` flag already threaded through `plot_1d`) -- the user's report happened to be about 2D data, but the underlying default applied to both, so both are fixed together rather than leaving 1D on the old silently-downsampled default.
- 8 new canvas-level tests (default fractions per-construction-arg, `set_full_resolution` actually changing what would render, `has_downsampled_data` recognizing a 2D contour, `export_full_resolution` for 2D both when a swap is and isn't needed, zoom preserved across an export rebuild) plus 1 new `MainWindow`-level test confirming only the Spectrum canvas gets the new default, every other named canvas keeps the old one. 700 tests passing (691 -> 700). Committed; not yet pushed.

## Autonomous 02:01 session, item 1: X-Pulse 2D crash fixed (2026-09-01)

Started on the "LWG Notes. 31/08/2026 23:16" list below, in the order requested. Item 1 (the crash) is fully root-caused and fixed; items 2 (downsampling) and 3 (TD/SI for both dimensions) are next.

- ~~**X-Pulse 2D phase-drag crash**~~ -- root cause confirmed exactly as diagnosed before deferring, then traced one level deeper to the real bug. `processing.ft.fft_spectrum` used `n = len(fid)` for its own frequency-axis length -- correct for a 1D FID (the only case it had ever been used for before this session's X-Pulse 2D work), but `len()` reads a 2D array's *first* dimension, not its last. `io.xpulse.read_2d`'s own F1 FFT step calls `fft_spectrum(spectrum.T, dt1)` directly on a full 2D array (128x2048, transposed to 2048x128) rather than looping per-column via `np.apply_along_axis` the way Bruker's own `read_2d` does for the identical operation -- so `freq_hz` (and therefore `spectrum.axes[0].ppm`, F1's own axis) came out length 2048 (matching the *row* count of the transposed array) instead of 128 (the actual FFT length), while the data itself was correctly shaped. `plot_2d_contour`'s F1 projection panel (correctly length 128, computed from the real data) then plotted against this wrongly-128-vs-2048-mismatched axis and crashed.
  - Fixed at the source: `fft_spectrum` now uses `fid.shape[-1]` (identical to `len()` for the 1D case every existing caller already uses, but correct for 2D input too) and explicit `axis=-1`/`axes=-1` on its own FFT/fftshift calls (a bare `np.fft.fftshift(spectrum)` with no `axes=` would have *also* incorrectly shifted the row axis for a 2D input). `io.xpulse.read_2d`'s own call site needed no further change once this was fixed -- it was already structured to do the right thing, just relying on a function that wasn't actually 2D-safe yet.
  - **A separate, real robustness gap found and fixed while reproducing this**: `MainWindow._draw_selected_spectrum` (the initial-render path) already wraps its own `show_spectrum` call in a try/except showing `QMessageBox.critical` on failure -- but `_apply_phase_preview`'s "2d" branch (the live phase-drag path, where the user's own pasted traceback actually came from) had no such guard at all, so this exact class of rendering failure printed an unhandled traceback once per mouse-move tick for the whole drag. Deliberately **not** given the same `QMessageBox.critical` treatment -- a dialog popping once per drag tick would be a far worse failure mode than the original bug (a wall of stacked/blocking dialogs during an interactive drag) -- instead wrapped the same way `_baseline_correct_if_enabled` already is elsewhere this session: best-effort, skip this tick's redraw silently, let the next tick (or the drag settling) recover on its own.
  - Also explains why this reproduced as a genuine *hang* (not a visible crash) when first tried headlessly: the initial render already hit the same bug, `_draw_selected_spectrum`'s own `QMessageBox.critical` popped, and a headless/offscreen test has no one to click "OK" -- confirmed directly (low CPU time despite minutes of wall-clock, the same signature this project has hit and diagnosed before for an unmocked modal dialog). Not a new gap to fix -- `_draw_selected_spectrum`'s dialog is correct, appropriate behaviour for a real (non-headless) session; the reproduction script just needed to mock it, same as every other headless test in this suite already does for exactly this reason.
- 2 new regression tests (a minimal synthetic NTUPLES fixture, deliberately asymmetric F1<<F2 like the real 128x2048 HSQC dataset, since that shape is what turns this into a loud crash rather than a silent wrong-value bug a square array wouldn't catch) plus 2 new `fft_spectrum` unit tests (2D input axis length, and that each row FFTs independently -- confirming the `axes=-1` fftshift fix specifically). 691 tests passing (689 -> 691). Committed; not yet pushed.

## LWG Notes. 31/08/2026 23:16. Scheduled to start autonomously 01/09/2026 02:01.

- **X-Pulse 2D import error, now with the real traceback (arrived in a follow-up message)** -- confirmed structure detail from the user directly first: "there is one 2d file and NP number of slice files", matching this session's own earlier finding for `test_data/xpulse_HSQC_NUS`. The actual error, repeated many times in a row (once per live-drag tick):
  ```
  ValueError: x and y must have same first dimension, but have shapes (128,) and (2048,)
  ```
  at `plotting/twod.py:275`, `ax_f1.plot(-f1_projection + ..., y_axis, 'k')`, reached via
  `MainWindow._on_phase_handle_changed` -> `_apply_phase_preview` -> `self.canvas.show_spectrum(spectrum)` -> `plotting.plot_2d_contour`.
  **Diagnosed (not yet fixed) directly from the traceback, before deferring to the scheduled start below**: this happens specifically while *phase-dragging* an X-Pulse 2D spectrum, not on the initial load (already confirmed working standalone via `io.load()` during development). Strong working hypothesis: `read_2d`/`_read_xpulse_ntuples_2d` (committed `4ebe9f4` this session) were only wired in at the *load* level -- `MainWindow._apply_phase_preview`'s "2d" branch and the live FID-recompute path (`_process_raw_fid_row`/`_recompute_multi_row_data`) are Bruker-shaped in their assumptions (`procs`/`proc2s`-based window/zero-fill, `fn_mode` handling built and tested only against Bruker fixtures) and were never adapted for X-Pulse's own very different raw-FID structure/metadata schema (`exp_params`, `TxPPM`-based axis, no `procs`) -- a live phase drag most likely runs that Bruker-oriented recompute regardless of vendor, and produces a spectrum whose F1 axis (`spectrum.axes[0]`) and data (`spectrum.data.shape[0]`) end up mismatched (2048 vs 128) as a result. Needs direct reproduction (load the real HSQC-NUS file, drag phase) to confirm this specific mechanism before fixing -- this is a diagnosis, not a confirmed root cause.
- **TD and SI options for both dimensions on any 2D load** -- explicit ask, broader than the X-Pulse-only io-level support added this session (`read_2d(zero_fill_f1=..., truncate_f1=...)`, no GUI exposure yet). User wants F1 *and* F2 TD/SI spin boxes to appear in the Processing tab's FID-processing box whenever *any* 2D spectrum (any vendor) is loaded -- i.e. also extending Bruker's own `read_2d`, which has never exposed F1 zero-fill/truncate at all (a deliberate earlier-round scope decision, now superseded by this explicit ask) -- not just X-Pulse.
- **Main Spectrum tab is severely downsampled; should be fully sampled** -- "Only the spectrum in fid processing should be [downsampled]... In spectral fit, this is fine already, but I need to have the fully sampled spectrum in the spectrum tab." **A real, concrete lead found while diagnosing the crash above, not yet confirmed as the actual cause**: `plotting.twod.plot_2d_contour`'s own defaults (`contour_fraction_x=0.25, contour_fraction_y=0.25`) downsample to 25% of points (capped at 16384/1024) *unconditionally* via `_prepare_contour` -> `downsample_2d` -- worth checking directly whether the main Spectrum canvas's own `show_spectrum` call passes these same fast-render defaults through with no override, when it should be requesting full resolution (unlike a genuinely fast-preview context, e.g. Compare, where this downsampling was originally built for -- see the "Compare-render slowness" fix from an earlier round). The FID-processing preview and Spectral Fit tab (already correct, per the user) are the two reference points to compare against directly.
- **Scheduled**: user asked to "note these and start them autonomously at 2.01am" -- a one-shot cron job was set for 01/09/2026 02:01 local to begin work on this list without further prompting. Start with the X-Pulse 2D phase-drag crash (reproduce directly against real `test_data/xpulse_HSQC_NUS` first -- load it, drag phase, confirm the diagnosis above before fixing), then the downsampling lead (also likely fixable alongside, since both trace through the same 2D contour-rendering code path), then TD/SI for both dimensions. Matches this project's own standing discipline: investigate and reproduce before fixing, write regression tests, run the full suite, commit each round.

## Multicomponent (biexponential) PFG/diffusion fitting (2026-08-31)

Last item from the previous round's feedback message: "multicomponent fitting to the relaxation and especially PFG modules is highly desired." Relaxation (T1/T2) already had this -- `RELAXATION_MODELS` already includes `biexponential_decay`/`biexponential_recovery`, fully wired into the GUI already. PFG/diffusion had no multicomponent variant at all -- that's what this round actually adds.

- ~~**Biexponential Stejskal-Tanner diffusion**~~ -- new `processing.relaxation.stejskal_tanner_decay_biexponential(b, I0a, Da, I0b, Db)`, same shape as the existing `biexponential_decay` with `b`/`D` substituted for `t`/rate constant (exact substitution already established for the single-component `stejskal_tanner_decay`). Registered as `"stejskal_tanner_biexponential"`, `rate_param=None` (two D values, not one) -- the existing generic "no single rate_param -> show every fitted parameter" fallback in both `NMR2D_fit_relaxation`'s callers already handles this correctly with zero GUI code changes needed, same as the pre-existing T1/T2 biexponential models. New "Stejskal-Tanner, biexponential (Da/Db)" entry in the Diffusion model combo.
- **A real nonlinear-fitting reliability issue found and fixed while writing the round-trip test, not just theorized**: the first, reasonably-argued initial-guess heuristic for the two D values (spread symmetrically around `1/b_max`, mirroring `_p0_biexp_decay`'s own 0.3x/2x spread for T1/T2) reliably converged to the *wrong* solution on a real synthetic round-trip test (well-separated true D values, no noise) -- not a close-but-imprecise result, a materially different one (amplitudes swapped, both D values off by an order of magnitude). Traced to `fit_relaxation`'s own `curve_fit` call using explicit bounds + `x_scale="jac"` (added in an earlier round for exactly this kind of orders-of-magnitude parameter-scale problem) -- confirmed directly that the *identical* p0 converges correctly under a bare/unbounded `curve_fit` call but not under the bounded one `fit_relaxation` actually uses, since providing real bounds changes which scipy solver runs at all. Fixed by empirically searching for (not just theorizing) a p0 spread that converges reliably under `fit_relaxation`'s own actual call signature -- a wider, higher-centred spread (20/b_max, 2/b_max, not 5/b_max, 0.5/b_max) gets there. **How to apply generally**: for any future multicomponent/nonlinear model added to this registry, verify convergence through `fit_relaxation` itself on a real round-trip test, not through a standalone `curve_fit` call with the same p0 -- the bounds + `x_scale="jac"` combination this project's own fit function always uses can behave differently even when the starting guess is identical.
- 689 tests passing (685 -> 689). Committed; not yet pushed.
- **All items from this feedback round are now done**: baseline polynomial/sinusoidal + default-on, X-Pulse 2D import, F1 SI/TD (io-level), and now multicomponent PFG fitting. Not yet pushed to `origin/main` at any point across this session's several rounds -- offer/ask before pushing.

## Baseline: polynomial/sinusoidal methods, default-on; X-Pulse 2D import finally works (2026-08-31)

New feedback message, four items: baseline correction "still bad" + y-positioning "too high" (should be polynomial/sinusoidal, spline maybe better as a future interactive click-to-define module), multicomponent fitting for relaxation/PFG (not started this round -- see below), X-Pulse 2D import "still doesn't work" (root cause was found two rounds ago, not yet implemented until now). A mid-round follow-up also asked for F1 SI/TD (zero-fill/truncate) controls for indirect-dimension FID processing.

- ~~**"Baseline should be polynomial, which includes linear fits, or sinusoidal"**~~ -- done. `processing.fitting.estimate_baseline_polynomial(x, y, degree=...)` generalizes the existing robust-linear estimator to any degree (1 = identical to the old linear fit -- verified they agree exactly, not just similarly); `estimate_baseline_robust` is now a thin, signature-preserving wrapper around it, so `fit_spectrum`/`NMR2D_fit_relaxation` (its other real callers) are untouched. New `estimate_baseline_sinusoidal` fits `a0 + sum a_k*cos(kwx) + b_k*sin(kwx)` (linear in its own coefficients, same sigma-clip candidate-selection loop, reused via a new shared `_select_baseline_candidates` helper -- three near-duplicate copies of that selection logic was the threshold for pulling it out). `processing.ft.baseline_correct` gained `method="polynomial"`/`"sinusoidal"` alongside the existing linear/spline, with `degree`/`n_harmonics`/`period` params. GUI: baseline method combo gained the two new options plus a degree spin box (visible only for Polynomial, same collapsible-visibility pattern the smoothing spin already used for Spline).
- ~~**"Maybe splining should be left for an interactive module where the user can click on the spectrum to define the baseline"**~~ -- explicitly a "maybe," and a real, separate feature (a whole new click-to-define-baseline-points interaction mode) -- not attempted this round; noted here as a scoped future idea, not silently skipped.
- ~~**"Y-positioning is like too high... a well-phased spectrum should generally land close to zero"**~~ -- partially addressed, honestly not fully root-caused. Direct investigation on two real datasets (a wideline 7Li static and a sharper 23Na spectrum) found the *uncorrected* baseline already sitting reasonably close to zero on both -- didn't reproduce a clear "still bad" case via those two spot checks. The one concrete, well-justified fix made: **baseline correction now defaults ON for every vendor** (`MainWindow._load` used to only seed this default for X-Pulse, since only X-Pulse's own `read_1d` silently baseline-corrects at load time already -- Bruker/Varian/JEOL never have, so most of this user's own real data was never getting *any* baseline correction unless the toggle was found and switched on manually). Given large recorded first-order phase values are common on the quadrupolar/wideline nuclei this user's own real datasets are full of (confirmed directly: -1748.5 deg and 1581.9 deg PHC1 on two real spectra), and a large phase1 concentrates residual curvature toward the spectrum's edges -- exactly where the default candidate-baseline-region selection draws its points from -- the new polynomial/sinusoidal options give a real tool for exactly that failure mode, even though "y sits too high" wasn't independently reproduced as its own separate bug. Left the "pin the bottom of the view" scroll-zoom behavior (from an earlier round) untouched -- if baseline positioning is genuinely more reliable now, that older fix should already feel more correct as a consequence, rather than needing its own separate change.
- ~~**"The 2D x-pulse import still doesn't work"**~~ -- **actually fixed this time**, building directly on this session's own new `processing.ft.decode_hypercomplex_f1` (built for Bruker States-TPPI earlier this session) -- the real HSQC test file's own `##.ACQUISITION SCHEME= TPPI` plus its real/imaginary PAGE-pairing (confirmed by direct inspection: `(T2++(R..R))` immediately followed by `(T2++(I..I))`, same recorded `T1=` value) is exactly States-TPPI's row-pairing convention, so the same function applies directly with no new decoding logic needed.
  - New `io.xpulse._read_xpulse_ntuples_2d`/`read_2d` parse the NTUPLES/PAGE format the root-cause investigation two rounds ago identified but didn't implement. Reused the plain-1D reader's own ASDF/AFFN pseudo-digit decoder (`_parse_pseudo`/`_parse_affn_pac`/`_parse_data`, pulled out to module level -- a pure, behavior-preserving move, confirmed by the existing 23-test suite staying green throughout) rather than reimplementing it.
  - **Two more real, confirmed bugs found and fixed while validating against the real file, not assumed**: (1) the file's own `##VARFORM= AFFN, AFFN, AFFN, AFFN` header is simply wrong for the R/I (FID amplitude) columns -- their data lines are unambiguously ASDF pseudo-digit text (confirmed by direct inspection), and trusting the header decoded fragments of pseudo-digit text as if they were plain AFFN numerals, producing occasional wild (~1e10) spike values at effectively random positions in every row. Fixed by detecting ASDF vs AFFN from the data's own content (any ASDF-alphabet letter present) instead of trusting the header -- self-correcting regardless of what any future export's header happens to claim, verified via a dedicated test using a real, technically-correct `AFFN` fixture specifically so both paths are exercised. (2) A related per-line point-count mismatch (isolated single-line ASDF decoding doesn't always agree with the true multi-line-continuous decode, since DIF-mode carries state across line boundaries) -- the existing length-mismatch fallback (already written for the plain-1D reader's own reasoning) correctly self-corrects to a uniform time grid once the VARFORM bug above stopped masking it, and a uniform per-point F2 dwell time is the physically correct model anyway (real FID acquisition is evenly sampled), not just a degraded fallback.
  - **F1 zero-fill/truncate (SI/TD) now supported at the io level**, per the mid-round follow-up: `read_2d(zero_fill_f1=..., truncate_f1=...)`. `truncate_f1` operates on raw row *pairs* (before hypercomplex decode) so a request never separates a real row from its matching imaginary one. **GUI-level exposure (spin boxes in the Processing tab's FID-processing box) is not done this round** -- this is io-level plumbing only; wiring live F1 SI/TD controls into the GUI (and, if wanted for parity, doing the same for Bruker's own `read_2d`, which has never exposed this either, by explicit earlier-round design choice) is a real, separate next step.
  - **F1 (and F2) ppm-axis referencing is a best-effort, explicitly flagged, not a verified convention** -- the NTUPLES export has no `$O1` field for either dimension (confirmed absent), so `read_1d`'s own `$O1`/`$SF`/`$SFO1`-based Hz formula doesn't apply; used the file's own `TxPPM`/`TxPPMX` (transmitter offset, directly in ppm) to centre each axis instead, which is at least the semantically correct field for the purpose, chosen only after directly ruling out an alternative (`exp_params["O1"]`, a different, unrelated field of the same name) that produced a physically impossible 1H range (-60 to -30 ppm) on the real dataset. Even with `TxPPM`/`TxPPMX`, the strongest recovered peak on the one real Quinine HSQC dataset available sits at an implausible ~21 ppm 1H shift -- likely an unphased artifact riding on real signal rather than proof the axis itself is wrong, but genuinely not confirmed against an independent reference. Flagged in the function's own docstring and here, same standing as the Varian ppm-axis and JEOL `FILTER_FACTOR` gaps already on record.
  - `MainWindow._scan_for_spectra` no longer offers a `raw_slices/` folder's 100+ individual per-F1-increment files as separate matches once a real NTUPLES 2D file is found alongside it -- the originally-reported symptom ("has both raw slice data, and the complete data set"). A `raw_slices/` folder with no NTUPLES sibling (genuine standalone per-slice pseudo-2D data, e.g. a PFG series) is untouched.
- **Not started this round**: multicomponent fitting for relaxation/PFG (see the standing note above -- `RELAXATION_MODELS` already has `biexponential_decay`/`biexponential_recovery` for T1/T2, but PFG/diffusion has no multicomponent Stejskal-Tanner variant at all yet); GUI-level F1 SI/TD controls.
- 685 tests passing (665 -> 685). Committed; not yet pushed.

## Analysis tab restructure: Peak Picking / Integration split, fit-tuning tucked behind Advanced (2026-08-31)

Item 2 of the four-part message from two rounds ago -- the last genuinely-unstarted piece, now done. User follow-up when asked what "integration" should concretely mean: "I'm not really sure. I just know the fitting part is too busy and there's too much information all at once for casual users." Investigated the actual current sub-tab (not assumptions from memory) before proposing anything.

- ~~**"Spectral Fitting" sub-tab was doing too much at once"**~~ -- confirmed concretely, not just taken on faith: the single sub-tab stacked ~17 interactive elements with no visual hierarchy -- peak table, 3 button rows (add/remove/clear, pick-from-plot/auto-pick, prominence), 2 more tuning rows (position tolerance, S/N region+threshold), a full Integration Regions box (table, 3 buttons, select-from-plot, integrate), then Fit Spectrum + results + a slice-fit viewer. Integration already existed as a real, working feature here (region table, drag-select, integrate -> feeds Fit Curve) -- it wasn't missing, just visually flattened into the same wall as everything else, which is why "add integration" wasn't the actual task.
- ~~**Split into Peak Picking and Integration sub-tabs**~~ -- `_build_spectral_fitting_subtab` renamed `_build_peak_picking_subtab` (peak table, add/remove/clear, pick-from-plot, auto-pick + prominence, Fit Spectrum + results + slice-fit viewer -- kept together since picking peaks and fitting them is one continuous flow, not two); new `_build_integration_subtab` (just the existing Integration Regions box + its own new `integration_results_label`). `analysis_sub_tabs` is now Peak Picking / Integration / Curve Fitting / VOCS / Imaging.
  - **The two "step 1"s sharing one results label needed splitting too**: `_show_integration_results` used to write into `analysis_results_label` (Fit Spectrum's own label) since both lived on the same tab -- moved to its own `integration_results_label` so a user on the Integration tab actually sees their own result, not one that silently landed on a different tab. Threaded through the existing per-spectrum `_capture_analysis_state`/`_restore_analysis_state` persistence the same way `analysis_results_label`'s text already was, and through the full-reset path in `_on_clear_clicked` -- same pattern, not a new mechanism.
- ~~**"Fitting bounds" (position tolerance, S/N noise region + threshold) separated out**~~ -- not given a third parallel sub-tab (would just be one more empty-looking tab a casual user clicks through for nothing, trading "busy" for "same amount of stuff behind more clicks," not actually fixing the complaint) -- instead a new `_collapsible_section` helper (Qt has no native collapsible group box; a checkable `QToolButton` with a flipping arrow indicator driving a content widget's visibility) wraps them as a collapsed-by-default "Advanced (fit tolerance and S/N)" section inside Peak Picking. Same widgets/attributes as before, still fully usable once expanded -- nothing removed, just not competing for a first-time user's attention by default. Auto-Pick + prominence stayed in the default (uncollapsed) view since they're a primary picking mechanism a casual user would reach for first, not a tuning knob.
- 4 new GUI tests (sub-tab split confirmed via `isAncestorOf`, Advanced section starts collapsed and its content really does contain the three tuning controls once expanded, integration results now persist independently per spectrum) plus the one existing test that referenced the old shared label updated. **Caught a real Qt testing gotcha while writing these**: `QWidget.isVisible()` is always `False` in this project's headless tests regardless of any toggle, since the window itself is never actually `.show()`n (composed ancestor visibility, not the widget's own flag) -- this project's own established convention (`fid_processing_box` et al.) already uses `isHidden()` instead; documented directly in `_collapsible_section`'s own docstring so the next collapsible-widget test doesn't have to rediscover this.
- Also verified interactively (not just via the test suite): built a real headless `MainWindow`, confirmed tab order/labels, the Advanced section's arrow flips and content becomes a real ancestor of the tuning controls on toggle, and both new tabs contain the right child widgets.
- 665 tests passing (662 -> 665). Committed; not yet pushed.
- **All four items from the "Set up that decoding... peak picking... integration... fitting bounds..." message are now done.** Not yet pushed to `origin/main` at any point across this session's several rounds -- offer/ask before pushing, per this project's own established practice.

## Hypercomplex F1 decoding implemented (States/States-TPPI/Echo-Antiecho) -- the concrete next step flagged in the last several rounds (2026-08-31)

Item 1 of the four-part message from the previous round ("Set up that decoding"), picked up immediately after that round's baseline/Apply-Processing fixes. `read_2d`'s F1 FFT previously just complex-FFT'd whatever raw rows arrived, which is only correct for QF (FnMODE=1, what every genuine 2D dataset actually in this project's test_data used) -- States/States-TPPI/Echo-Antiecho data needs adjacent raw row-pairs combined into a genuine complex F1 point *first*. This is the gap identified two rounds ago while investigating "appears... often inverted," deferred until the raw-`ser` shape problem (a prerequisite) was solved.

- ~~**States/States-TPPI/Echo-Antiecho row-pair decoding**~~ -- new `processing.ft.decode_hypercomplex_f1(data, fn_mode)`, wired into `read_2d` right before F1's own zero-fill/FFT (a no-op for FnMODE 0-3, unchanged behaviour). `FnMODE`'s own integer mapping (0 undefined, 1 QF, 2 QSEQ, 3 TPPI, 4 States, 5 States-TPPI, 6 Echo-Antiecho) confirmed directly from `nmrglue.fileio.bruker.guess_udic`'s own source, not assumed -- the authoritative source for *this exact integer table*, since Bruker's own manual documents acquisition behaviour, not this internal numbering. States/States-TPPI formulas (row `2k`=real/"cosine", `2k+1`=imaginary/"sine" for t1 increment `k`; States-TPPI additionally alternates sign every other point) cross-checked against an independent source (NMR Wiki's States-TPPI page) and derived from first-principles Euler's-formula reasoning, landing on the same result both ways.
- **Real, decisive validation against real data, not just synthetic**: cross-checked all 5 real States-TPPI (FnMODE=5) datasets in test_data against `read_2d_processed` (TopSpin's own numbers) -- F1 ppm axis range matches TopSpin's *exactly* (not approximately) on every one. More importantly, directly compared the new decoded F1 projection's correlation with TopSpin's own F1 projection against what the *old* (undecoded) code would have produced on the same real files: old code's correlation with TopSpin ranged from ~0 to **-0.46** (i.e. genuinely anti-correlated -- close to a literal mirror image on that dataset, matching "appears... inverted" almost exactly) across the three datasets checked in depth; the new decoded path is positive on all three (0.83, 0.12, 0.05). The weaker two are plausibly explained by those specific experiments' own lower intrinsic F1 SNR/peakiness (checked directly: TopSpin's own F1 projection is itself much less peaked on those two) rather than a formula error -- F2 projection correlation (unaffected by this fix, a sanity check) was consistently strong (~0.92-0.93) on all three, confirming the datasets themselves aren't just noise. Every real dataset checked improved, several dramatically, none got worse.
- **Echo-Antiecho implemented but explicitly flagged, not silently presented as equally certain**: `(echo+antiecho) + 1j*(echo-antiecho)`, the standard sensitivity-enhanced/PEP recombination -- derived from first-principles physics (verified self-consistent: a synthetic round-trip test recovers a known F1 frequency at the correct magnitude) and cross-checked against nmrglue's own `process.nmrtxt.rance_kay.bruk_ranceY` (present in the installed dependency, confirming the general shape of the recombination). **What's specifically NOT verified**: Bruker's own raw row *order* (echo-then-antiecho vs. the reverse) -- getting this one detail backwards would mirror F1 peaks across the spectrum centre without otherwise looking broken. No Echo-Antiecho (FnMODE=6) dataset exists anywhere in this project's test_data to check against (only FnMODE 0/1/5 are actually present in real data here) -- same standing as the Varian ppm-axis and JEOL `FILTER_FACTOR` gaps already documented in this file: implemented with the best-justified choice, not guessed-and-hidden. Revisit if/when a real Echo-Antiecho dataset becomes available.
- **Also fixed, found while wiring this in**: the GUI's own live multi-row recompute (`MainWindow._recompute_multi_row_data`, this project's own "same source" live-processing path from the previous round) redoes F1's FFT whenever an F2-only control changes on a conventional 2D -- without also redoing this same hypercomplex decode there, any live F2 control change on a States/States-TPPI/Echo-Antiecho dataset would have silently reverted to the old, wrong (pre-this-round) behaviour the instant the user touched window/zero-fill/shift/etc., even though the *initial* load would have been correct. Fixed by threading the newly-recorded `metadata.raw["fn_mode"]` through to that path too, so both pipelines stay genuinely the same source, not just at load time.
- 9 new unit tests (`decode_hypercomplex_f1` in isolation, including physical round-trip frequency-recovery checks and a regression test proving the decode step is load-bearing) plus 4 new `read_2d` integration tests (mocked synthetic fixtures, through the real pipeline including axis construction/NS-normalization) plus 1 new GUI test for the live-recompute integration gap below (confirmed to actually fail without that fix, not just pass with it). 662 tests passing (648 -> 662).
- **Not started, the other genuinely-unstarted item from the same four-part message**: restructuring the Spectral Fitting tab into separate peak-picking/integration/fitting-bounds sections -- see the previous round's entry below for the same note, still accurate.

## Same-source architecture extended to every FID-domain control; baseline hang fixed; Apply Processing removed (2026-08-31)

Same-session chat-message round, four items: hypercomplex F1 decoding (not started -- see below), Spectral Fitting tab restructuring (not started -- see below), spline baseline hang, and "for the last time, remove the [Apply Processing] button, we can have a reset button." The last two turned out to share one root cause with two already-known complaints (this file's own "processing is reset when changing between experiments" and "the saving removed the baseline"), so all four were fixed together as one architectural change rather than four patches.

- ~~**"Apply processing is still resetting everything"/ "the saving removed the baseline"**~~ -- root cause: every FID-domain control (window, zero-fill, shift, align-echo, linear-predict, truncate) only ever updated a separate, disconnected preview canvas. The actual `spectrum.data` only changed via a full `session.reload()` (a disk re-read), which discarded any state -- baseline, phase, anything -- not explicitly threaded back through reload kwargs. This is exactly why saving/reprocessing looked like it "removed the baseline": it did, because baseline correction wasn't a reload kwarg. Fixed by extending last round's "processing and spectrum should update from the same source" principle (previously phase- and baseline-only) to *every* FID-domain control: new `_process_raw_fid_row` (shift/align/truncate/LP -> window -> zero-fill -> FFT -> NS-normalize) is the one shared per-row pipeline, called directly from every control's own signal handler (`_on_fid_processing_control_changed`) and writing straight into `spectrum.data`/`data_imag`/`axes` -- no separate reload/commit step anywhere in the loop.
- ~~**"For the last time, remove the [Apply Processing] button, we can have a reset button"**~~ -- done, now that every control is genuinely live there's nothing left for that button to commit. Replaced with **Reset**, wired to a new `Session.reset(index, **kwargs)` (like `reload`, but *replaces* the stored load kwargs instead of merging) -- reverts to `self._original_load_kwargs[index]`, snapshotted in `_load()` right after the pseudo-2D-default handling, so Reset goes back to "as originally loaded, including the GUI's own pseudo-2D-by-default convention," not raw vendor auto-detection.
- ~~**"Still the processing is reset when changing between experiments"**~~ -- fixed as a direct consequence of the above: every control change now persists into `spectrum.metadata.raw["live_*"]` (`_persist_fid_processing_controls_to_metadata`), read back on reselection by `_refresh_fid_processing_controls` -- replaces the old `session.get_load_kwargs()`-based restore, which only ever updated on an actual `reload()` (no longer happening for live changes, so it had gone stale the moment this round's live-control work started).
- ~~**"The spline baselining doesn't work and hangs, user inputs are fine if needed to bound the solutions"**~~ -- root cause **wasn't** the spline math (the smoothing-parameter spin box from the previous round's baseline rewrite already *is* the user-adjustable bound this asked for, and it isn't touched by this fix) -- it was **zero throttling**. `DragHandle.mouseMoveEvent` fires `valueChanged` synchronously on every pixel of mouse movement; for a pseudo-2D/2D spectrum this was, until this round, recomputing baseline (spline or linear) inline on *every single tick* of a phase or FID-control drag. Reproduced directly: 30 rapid ticks against a real 500-row baseline-enabled dataset would previously have meant 30-60+s of blocked UI. Fixed with a 200ms debounce (`_live_reprocess_timer`/`_schedule_multi_row_reprocess`/`_run_debounced_multi_row_reprocess`) that separates cheap, always-instant work (phase application, still live on every tick) from expensive per-row work (spline baseline, full FID reprocessing), which now only runs once after the drag settles. Verified: the same 30-tick drag now completes in 24ms, with the real recompute landing ~381ms after the last tick.
- **Two new real bugs found writing tests for this, both fixed**:
  - **2D zero-fill silently truncated F2 to F1's row count** -- `_refresh_fid_processing_controls`'s zero-fill default used `len(spectrum.data)`, which for a 2D array returns the row (F1) count, not F2's column count. A live recompute on a conventional 2D would silently shrink F2 down to however many F1 rows/increments existed. Fixed to `spectrum.data.shape[-1]`.
  - **Linear-predict order (fixed UI default of 20) could crash a live Qt signal handler** -- found while fixing the LP-direction test against this file's own tiny synthetic X-Pulse fixture (`NP=4`, `TDEFF=3`): `linear_predict` correctly raises `ValueError` when `order >= len(fid)-1` (a real, order-20-vs-3-points case), but nothing caught it -- it propagated straight out of a `valueChanged` slot. Not just a test-fixture artifact: any real TDEFF-truncated-short FID with LP order left at its default would hit exactly this. Fixed at the processing layer, not just papered over in the GUI: `processing.ft.prepare_fid` now clamps the order to the largest value actually valid for the FID length at hand (skipping LP outright only in the genuinely-degenerate case where even `order=1` isn't valid), so a real caller gets a real LP result instead of a crash. Added GUI-level defense in depth too (`_process_raw_fid_row` now wraps `prepare_fid` in the same "best effort, fall back to the unmodified row" try/except `_baseline_correct_if_enabled` already established last round) in case anything else in that pipeline fails on genuinely degenerate data.
- Other latent gaps closed as part of generalizing the "same source" pipeline: `fid_canvas` (the plain FID tab) wasn't refreshed by the new unified control handler, only the Processing tab's own preview was -- fixed. `metadata.raw["fid"]`/`["fid_dt"]` (the FID-preview tab's own source data) previously only updated on a full reload and would go stale after any live control change -- now updated directly by both the single-row and multi-row commit paths. X-Pulse's `read_1d` has a hidden `baseline_correct = not align_echo` default applied once at load with no record kept anywhere retrievable -- since the new live pipeline never replicates that implicitly, `_load()` now seeds `metadata.raw["baseline_correct"]`/`["baseline_method"]` explicitly for X-Pulse at load time so the live pipeline's own explicit toggle starts from the same place the old hidden default did.
- **Not started this round** (items 1-2 of the same chat message, genuinely deferred rather than rushed): States/States-TPPI/Echo-Antiecho hypercomplex F1 row-pair decoding in `read_2d` (the shape-resolution work is done -- see the round below -- this is the next, separate piece, flagged there already); restructuring the Spectral Fitting tab into separate peak-picking / integration / fitting-bounds sections (integration in particular may need real new UI, not just reorganizing existing controls -- no region-integration-from-the-Spectral-Fitting-tab UI currently exists, distinct from the Imaging tab's own already-working Integration Regions feature noted in the 2026-08-30 Phase 5 entry below).
- 648 tests passing (645 -> 648). Committed; not yet pushed.

## LWG Notes. 30/08/2026 21:37.

-  **ouput**
When testing, changing the dpi didn't seem to change the output (very low quality). I also need the options to save in different file types like vector graphics (pdf/svg).

- **x-pulse 2D**
Still not being read, being treated as a 1D and then fails. Either learn to read these in and ignore the raw slices, or read in the array of raw slices as a pseudo-2D and do the fft for the indirect dimension.

## Raw ser shape resolution actually solved -- read a real compiled pulse program directly, cross-checked against real data (2026-08-30)

User's direct follow-up to the "fundamental operation requirements" note below: "use [the compiled pulse programmes already present in each experiment folder] and compare to the processed data to try to build a clear and reproducible parser." Doing exactly that resolved the raw-`ser` shape problem this project had failed to close across three separate prior rounds.

- ~~**Row length**~~ -- confirmed two independent ways, not guessed. Read Bruker's own official "Acquisition Commands and Parameters" manual (already downloaded last round) for `DIGMOD`/`DECIM`: "oversampling... results in a larger number of points than specified... Before the data are written to disk, they are digitally filtered which reduces (decimates) their number to TD" -- i.e. the stored data is already at the decimated rate, ruling out an oversampling-adds-extra-points theory. Then measured the real per-row word count **directly from the raw bytes**, not a formula: autocorrelation of the raw `ser` file's own magnitude envelope (FFT-based, on the full real dataset) shows a strong, unambiguous periodicity at exactly `ceil(TD0/128)*128` words (float64/`DTYPA=2`) on two independent real datasets (one at acf=0.88, about as clean a signal as this gets) -- confirming the formula from two rounds ago was *already right*, and ruling out nmrglue's own guess (which implied a spurious "extra factor of ~2" for some of these files, now understood to be a coincidental exact divisor of the total word count, not anything physically meaningful -- checked and rejected directly, see below).
- ~~**Row count**~~ -- confirmed by reading a real compiled pulse program directly (`pulseprogram.precomp`, present in every experiment folder as the user pointed out -- previously only partially read for its loop-variable *names*, not its actual expanded instructions). For a real States-TPPI dataset, the `mc`-macro's own expansion is fully explicit: `ST1CNT = trunc(td1/2)`, then `lo to LBLSTS1 times 2` (one States pair: 2 rows) nested inside `lo to LBLF1 times ST1CNT` (one iteration per t1 step) -- total rows written = `2 * trunc(TD/2)`, i.e. `TD` itself directly (not doubled, not halved) for an even `TD`. This is corroborated by the official manual's own `FnMODE` reference table ("States: successive fids will be acquired incrementing the time interval **after every second fid**") -- the pairing is *inside* what `TD` already counts, not an additional multiplier on top of it.
- **The exact byte count still doesn't always divide perfectly** -- checked and genuinely not resolved: on otherwise-clean acquisitions (no abort-and-restart in their own audit trail), the real file holds a small, inconsistent handful of rows beyond `TD` (+4, +4, +2 across three independent clean datasets -- not proportional to `TD`, not a fixed constant either). Ruled out several plausible explanations directly against real data: oversampling (manual explicitly rules this out, see above), the file's own `audita.txt` "double data hash MD5: TD0 * TD2" line (just restates already-known acqus/acqu2s values, no hidden multiplier), a fixed header/trailer (the excess is way too large -- megabytes, not a small header would explain), an aborted-and-restarted acquisition contaminating the file (checked -- the worst offender, a 32%-oversized file, *does* have a "terminated... raw data deleted" entry in its own audit trail; excluded from the "clean" analysis for exactly that reason). Bruker's own "Pulse Programming Manual" (a *different*, more specialized manual than the one already obtained -- the acquisition-parameters manual explicitly refers to it for `mc`-statement specifics) is the most likely place a definitive answer lives; not fetched this round.
- ~~**Implemented despite the small residual, rather than continuing to block on it**~~ -- new `io.bruker._guess_raw_2d_shape`/`_read_raw_2d_data`: computes the row length as above (now trusted, not guessed), tries `TD` (acqu2s) as the row count first, and falls back to `file_size // row_bytes` (floor division, always in-bounds) whenever `TD` doesn't divide the file exactly -- so a real acquisition with this small, still-unexplained discrepancy still loads and captures every genuinely-present row, instead of crashing outright the way relying on `TD` alone (or nmrglue's own guess) both did. Still goes through `nmrglue.bruker.read` itself (passing the computed shape explicitly, bypassing only its own internal `guess_shape`) rather than reading the file directly, preserving the exact same `ng.bruker.read`-based mocking seam every test in this project's suite already uses.
- ~~**Validated hard, not just "stopped crashing"**~~ -- every genuine (non-pseudo) 2D Bruker dataset actually available across **three independent real sample sets** now loads successfully (11 experiments total, up from 0) with no shape-guess-failure fallback. Cross-checked one directly against `read_2d_processed` (TopSpin's own validated numbers) on the same experiment: F2 ppm axis range matched exactly, and the top-5 peak positions in a full-dataset F2 projection matched almost exactly (4 of 5 the same values, the 5th one row of resolution apart) -- real evidence the raw reconstruction recovers the same chemical shifts TopSpin does, not just "produces *a* spectrum." Confirmed zero regression on the two real pseudo-2D datasets this project's test suite already exercised successfully (`bruker_T2/11`, `bruker_PFG/150`) -- bit-identical output, not just matching shape.
- **Still open, and load-bearing**: this only fixes *reading the raw bytes into the right shape*. `read_2d`'s F1 (indirect) FFT still does not perform States/States-TPPI/Echo-Antiecho hypercomplex row-pair decoding before the FFT (the gap identified two rounds ago, investigating "appears... inverted") -- correctly-shaped raw data is a prerequisite for that fix, not a substitute for it. F1 will still look wrong (aliased/mirrored) on real hypercomplex-encoded 2D data until that's built. Natural next step now that shape-guessing is no longer the blocker.
- Existing tests that mocked `ng.bruker.read` directly for `read_2d` coverage (11 in `test_bruker_raw_2d.py`, 2 in `test_bruker_raw_fid.py`, 4 in `test_gui.py`) all needed a one-line addition (`monkeypatch.setattr(bruker, "_guess_raw_2d_shape", lambda dic, ser_path: fid.shape)`) since the new shape-resolution step runs before the mocked call -- their own small, block-rounded synthetic fixtures don't reflect real-file-scale `TD`/`DTYPA` arithmetic anyway. Also found and fixed a related latent bug while doing this: a GUI regression test that simulated a shape-guess failure via a mocked `IndexError` silently started hanging the whole suite (an unmocked `QMessageBox.critical` reached via `_load`'s own exception handler) once the *real* failure signature changed from `IndexError` (nmrglue's old shape-guess-then-reshape-then-crash chain, which no longer exists) to `KeyError` (this project's own new `_guess_raw_2d_shape` failing louder and earlier on a genuinely missing `TD`) -- `_read_auto`'s fallback now catches both, with an updated docstring explaining why.
- 4 new dedicated tests for the new shape-resolution functions themselves (real on-disk binary `ser` files, not mocked) plus the existing suite: 645 tests passing (641 -> 645). Committed; not yet pushed.

## LWG added notes 30/08/2026 17:36

- **Fundamental operation requirements**
I think that nmrglue may be holding us back in many places. I think fixing our i/o is the key focus from now until it works extremely well. This means that we need to figure out any parsing challenges from the fid and ser files in the bruker system. Read in detail for clues on deciphering this file structure and implement as a first task. This may require a bespoke i/o approach that doesn't include nmrglue but could use that code as a basis.

I think many issues are occurring because the data input is not coming from these raw files, therefore the spectrum object is not complete and isn't being updated or read from correctly. The spectrum object is a critical piece of being vendor agnostic so this is of primary importance.

- **x-pulse 2D not being recognised as 2D**
The pseudo-2D data is being pulled in correctly but it is different for the actual 2D data which has both raw slice data (raw_slices), and the complete data set.

- **Metadata persistence**
The metadata for process parameters and the processed data itself should persist when changing back and forth from datasets, then populating the process window with it.

- **2D slice extraction**
Slice extraction should be based on the cursor location on the contour. It should be able to get the projection at that cursor location. Maybe this could be part of an analysis tab.

- **Context**
I am thinking more and more that the right side plot tabs should mostly exist in the context of the left side option tabs: processing for the processing tab, analysis should be a specific spectrum which has markers like peak picking, integration regions, etc., spectral fitting should have a spectrum with the fits, and so on.

- **Baseline correction**
Baseline correction is no good right now. This needs work, baseline correction should be applied to the processed spectrum but the correction should be able to be switched on and off.

- **Apply processing button**
Since all the processing is shown live, there is no need for this button, which currently just resets everything anyway. This also means that doing things like "Align Echo" need to happen in the same way and be applied properly to the spectrum plot.

## Response to the notes above, plus a same-session chat-message round: same-source 1D phase, baseline correction rewrite (spline + toggle), peak-pick markers, X-Pulse 2D root cause found (2026-08-30)

Four items arrived as a chat message (1D phase Processing<->Spectrum mismatch, metadata persistence, baseline correction quality, auto-pick visual markers) immediately followed by the "LWG added notes" section above. Overlapping items are answered together here rather than twice.

- ~~**"A bug with phasing 1d from the processing to the spectrum tab... spectrum should be populated from the processing window data and processing metadata"**~~ -- root-caused and fixed, and it was worse than it sounded: reproduced directly (change a window/zero-fill choice live on the Processing tab without clicking Apply, then drag phase) and the Spectrum tab ended up showing a **different point count** than the Processing tab's own preview, not just a different value -- `_apply_phase_preview`'s "1d" branch phased a fixed snapshot taken back when the Phase controls were last anchored, completely independent of whatever the Processing tab's live controls currently showed. Fixed: the "1d" branch now calls `_compute_processing_preview` itself (the exact function the Processing tab's own preview already uses) and writes its result straight into `spectrum.data`/`data_imag`/`axes` -- the same source, not a second computation of it -- falling back to the old snapshot-based phasing only when there's no raw FID to recompute from at all (a processed-only fallback load). `MplCanvas.update_data`/`update_processing_preview_spectrum` both gained a length-mismatch fallback to a full rebuild, since an in-place y-only update assumed a phase change never touches point count, which is no longer true.
  - **Found and fixed a real, more consequential bug while verifying this**: the zero-fill combo's default (absent an explicit saved choice) was a "2x NP, rounded to nearest preset" *suggestion*, not what the vendor reader's own default zero-fill actually produced (X-Pulse in particular: no zero-fill beyond the raw length at all) -- meaning the very first phase drag (or even just opening the Processing tab) on a freshly-loaded, never-reprocessed spectrum would silently zero-fill it, changing its resolution as a side effect of an unrelated action. Fixed: defaults to `len(spectrum.data)` (the spectrum's own actual current length) instead of a suggestion -- correct for every vendor uniformly, since it doesn't need to track each reader's own internal default separately. This is very likely what "metadata isn't there or at least not populated... when switching to other experiment folders" was actually reporting too (the underlying phase-persistence *mechanism* was directly tested and confirmed already working -- `metadata.raw["phase0"]`/`spectrum.data` both survive a switch-away-and-back correctly -- but a live recompute picking up this mismatched zero-fill default would look exactly like "the metadata isn't there").
- ~~**Baseline correction ("doesn't work so well... options to spline... or something", and "should be applied to the processed spectrum but... switched on and off")**~~ -- rewritten. `processing.ft.baseline_correct` gained `method="linear"|"spline"` (new `fitting.estimate_baseline_spline`, a robust sigma-clipped cubic smoothing spline -- reuses the same candidate-point selection as the existing linear estimator, but scores outlier-rejection against the *spline's own* fit each iteration rather than a forced straight line, since a real curved baseline's own points otherwise get wrongly sigma-clipped out as if they were peaks). Verified on a synthetic quadratic baseline + peak: linear correction left residual std 28x the true noise floor (couldn't touch the curvature at all -- confirms the "doesn't work so well" report wasn't about the robustness rewrite from an earlier round, but a structural limit of a straight line); spline correction matched the true noise floor almost exactly. The one-shot "Baseline Correct" button is now a **checkable toggle** (`baseline_correct_button`, matching Align Echo's own established pattern) with a method combo and a smoothing spin box (0/"Auto" picks a data-driven default from the baseline points' own noise level); state (`on/off`, method, smoothing) lives in `metadata.raw`, restored per spectrum on reselection -- same persistence mechanism as phase. Wired into all three phase-mode branches (1D via the same live-recompute pipeline as the phase fix above; pseudo-2D and conventional-2D via the same per-row/whole-dataset targeting phase itself already uses, baseline applied on top).
- ~~**"Auto peak pick should leave a visual indication on the spectrum"**~~ -- done. New `MplCanvas.set_peak_markers(shifts)` draws a small triangle just above the trace at each peak's own shift (reusing the same live spectrum data already on screen to find each marker's height, not a separate lookup) -- called from `_on_auto_pick_peaks_clicked` right after populating the table. Cleared on the next full rebuild (`show_spectrum`/`clear`) the same way the existing pivot marker already was; while fixing this, also closed a **pre-existing gap in that pivot-marker lifecycle itself** -- `MplCanvas.clear()` never dropped `_pivot_line`/`_pivot_marker_ax`, which could raise on a stale, no-longer-attached Line2D if `set_pivot_marker` were called right after a `clear()`.
- ~~**"x-pulse 2D not being recognised as 2D... has both raw slice data (raw_slices), and the complete data set"**~~ -- **root cause found precisely, not yet implemented.** `io.xpulse.read_array` (the only existing "assemble many files into one 2D spectrum" path) only recognizes `arrayN/` *subfolders* (`_find_array_dirs`) and always marks the result `pseudo_2d=True` -- structurally wrong for this case even where it did apply. The real HSQC-NUS test dataset (`test_data/xpulse_HSQC_NUS/2`) uses a completely different, **currently unhandled** layout: a `raw_slices/NNNN_slice.jdx` flat-file-per-F1-increment folder (128 files, matching `NP1=128`) *alongside* one top-level `..._avg.jdx` file that is itself a **self-contained standard JCAMP-DX 6.0 NTUPLES 2D block** (`##NUMDIM=2`, `##NTUPLES= nD NMR FID`, 256 `##PAGE= T1=...` blocks -- one real + one imaginary per F1 increment, each using the exact same pseudo-digit-encoded `##DATATABLE=` format `read_1d` already decodes for a plain 1D file's own two data tables) -- confirmed by direct inspection, including `$$SW1=4000.0` (F1's own spectral width) and `##.NUCLEUS= ^13C, ^1H` (a heteronuclear experiment -- F1 needs its *own* gyromagnetic-ratio-aware axis, not F2's). Loading this file through the existing `read_1d` today silently mis-parses it (0 points found for both real/imag) and then crashes (`IndexError`). `_scan_for_spectra` currently offers this folder as 129 disconnected matches (128 individual slices + the one averaged file) instead of recognizing it as one dataset -- exactly the reported symptom. **This is genuinely confirmed 100%-sampled for the one test dataset available** (`Sampling 100%` in `parameters/HSQC_HX.par` -- the `SamplingList= 100%,75% NUS,50% NUS,25% NUS` line lists what the pulse sequence *supports*, not what this acquisition used), so a real fix does **not** need a non-uniform-sampling reconstruction algorithm to handle this specific file -- only a new NTUPLES/PAGE-aware parser (reusing the existing per-block ASDF decoder once per page) and a proper heteronuclear F1 axis. A true sub-100%-sampled dataset would still need NUS reconstruction (compressed sensing/IST) on top of this -- flagged as a separate, later concern, consistent with the standing `processing/ft.py`-module note about NUS being a natural addition once linear prediction was in place. Scoped as the concrete next task, not attempted this round -- wanted room to verify a brand-new parser + heteronuclear axis physics properly rather than rush it.
- ~~**"Fundamental operation requirements... nmrglue may be holding us back... figure out any parsing challenges from the fid and ser files"**~~ -- picked back up from the "loading the 2D ser... was very slow" item further down this file (two independent hand-derivation attempts had already stalled there) with real new research, still not resolved. Found and read Bruker's own official 274-page "Acquisition Commands and Parameters" reference manual directly (not a summary) -- its authoritative definition of `TD` states plainly that in the indirect (F1) direction "it represents the number of increments," interpreted by the pulse program's own `lo to n times td1`/`mc` statement -- i.e. **there is no universal formula for the true raw row count**; it's determined by how each specific pulse program's own loop/phase-cycle structure is written, which is exactly why `nmrglue`'s own loop-substitution shape-guessing (even with the pulse program re-enabled) fails on the real `babaDQ.zw`-based test dataset: its loop-counter variable names (`l1`, `ST1CNT`, `tdav`) aren't in nmrglue's hardcoded `td0`-`td3` substitution table. Separately found a **real, citable bug in nmrglue itself**: `guess_shape`'s own file-size-based fallback correctly uses a DTYPA-aware word-count rounding (128-word blocks for float64, matching its own comment), but then converts that word count to bytes using a hardcoded `*4` (int32's own size) regardless of DTYPA, instead of `*8` for float64 -- explains why its own guessed leading dimension ("2", "5", ...) was numerologically plausible but never actually validated. Using the corrected DTYPA-aware byte accounting directly (bypassing nmrglue's guesser entirely) got file-size-based row-count inference to ~99% of an exact integer division across three independent real datasets -- much closer than either prior attempt, but still not exact, and a leftover ~0.4-0.6-of-a-row remainder wasn't explained by anything found in the manual, nmrglue's source, or the dataset's own `audita.txt` (which does record a "double data hash MD5: TD0 * TD2" line -- checked as a possible independent ground-truth shape source, but it turned out to just restate the acqus/acqu2s TD values already known, not reveal any hidden multiplier). **Practical upshot, stated plainly**: the raw-`ser` path for a genuine (non-pseudo) 2D Bruker dataset still does not reliably work on any real test data available, for reasons that are now well-understood at the "why" level (pulse-program-dependent row multiplier with no generic formula) even though the exact byte-level fix for these specific real files remains open. Given the manual's own confirmation that this is fundamentally pulse-program-specific, a *bespoke* I/O layer replacing nmrglue wouldn't sidestep this either -- it would face the identical need to either resolve each pulse program's own loop structure or fall back to the same file-size-inference approach explored here. The most promising concrete next step isn't more hand-derivation: it's getting one real dataset's *actual* raw row count confirmed independently (TopSpin itself, or Bruker support) as ground truth to finally close the ~1% gap, rather than a fourth attempt at guessing it blind.
- **Not addressed this round** (both from the "LWG added notes" above, genuinely deferred rather than rushed): cursor-position-based 2D slice/projection extraction on the contour view (a real, well-scoped new feature -- "part of an Analysis tab" per the note); the "right-side canvas tabs should follow the left-side option tab's context" architecture idea (Processing tab -> live processing view, Analysis tab -> spectrum with peak/integration markers, Spectral Fit tab -> spectrum with fits) -- a genuine, larger design direction worth its own dedicated round rather than a partial, rushed pass folded into this one.
- **Apply Processing button**: partially addressed as a side effect of the phase/baseline fixes above (phase and baseline correction are now both fully live for every phase mode, no "Apply" step needed for either) -- but window/zero-fill/shift/align-echo/linear-predict/truncate still go through the FID-domain reprocess button (`_on_reprocess_fid_clicked`), which remains a discrete "Apply Processing" step, not yet live. Making *those* live too is a bigger change (they're pre-FFT, not post-FFT like phase/baseline, so "live" means re-running the whole FFT pipeline on every control tick rather than just re-deriving from already-FFT'd data) -- left as the next piece of this same theme rather than attempted alongside everything else this round.
- 641 tests passing (630 -> 641). Committed; not yet pushed.

## Phase-box polish, F1/indirect phase persistence, and a major raw-2D shape-guessing finding (2026-08-30)

Six items from live testing: two small polish fixes, two real correctness bugs (one root-causing three separate-sounding complaints), and one large, honestly-unresolved finding uncovered while chasing them.

- ~~**"Phase number box is too small to read the number, needs at least 3-4 digits"**~~ -- done. `_make_phase_control`'s spin box now gets `setMinimumWidth(80)`; Qt's default `sizeHint` (50px) couldn't show a value like "-1234.56", especially now that the phase range is unbounded (an earlier round's fix). Covered by `test_main_window_phase_spin_box_wide_enough_for_large_values`.
- ~~**"FID intensity should be vs time (with appropriate units) and maybe a secondary option for points"**~~ -- done. New `fid_x_unit_combo` ("Time (s)" / "Points"), threaded into `MplCanvas.show_fid`/`show_processing_preview` as `x_unit`/`fid_x_unit`. Along the way, found the actual root cause was narrower than it looked: the FID data was *already* time-based (`t = np.arange(len(fid)) * dt`) in both panels, but `show_processing_preview`'s own FID axis was simply missing an `set_xlabel` call entirely (`show_fid` had one, this one didn't) -- so the Processing tab's FID panel had no x-axis label at all, not wrong units. Covered by `test_main_window_fid_x_unit_combo_switches_axis_label`.
- ~~**"Processing figures disappear when the indirect dimension has been processed"**, and **"processing and spectrum should update from the same source"**~~ -- same fix. `_show_processing_preview` was calling `processing_preview_canvas.clear()` for any conventional (non-pseudo) 2D by design (its own FID+spectrum dual preview has no way to represent one) -- from the user's side this just looked like the figure vanishing the moment "Process Indirect Dimension" was checked. Fixed by falling back to `processing_preview_canvas.show_spectrum(spectrum)` -- literally the same spectrum object the Spectrum tab's own canvas shows, not a second computation of it -- instead of clearing. The live F1/F2 phase-drag handler (`_apply_phase_preview`'s `"2d"` branch) now mirrors into `processing_preview_canvas` too when it's the active tab, so dragging phase while looking at the Processing tab stays live instead of only catching up on the next unrelated refresh. Covered by `test_main_window_processing_preview_mirrors_spectrum_for_conventional_2d`.
- ~~**"when not on the processing left tab, the pivot line should go away"**~~ -- done. `_on_side_tab_changed` now calls `set_pivot_marker(None)` on both canvases when leaving the Processing (left) side-tab -- it was never cleared before, just eventually overwritten by the next phase drag or full rebuild, so it lingered as a stray dashed line on Data/Analysis/Compare. Covered by `test_main_window_pivot_marker_clears_when_leaving_processing_tab`.
- ~~**"This indirect dimension is also impossible to phase independently"**~~ -- **real, reproducible bug, found and fixed**, though see the shape-guessing finding below for why it may not have been the whole story on real data. F1/indirect phase on a conventional 2D was always a *live-only* correction: every drag correctly wrote straight into `spectrum.data`/`data_imag` (never the problem), but nothing ever recorded it anywhere durable, unlike F2's own `phase0`/`phase1` (baked into `metadata.raw` by every `read_1d`/`read_2d` call and re-supplied on every reload via `_current_phase_kwargs`). The moment the spin boxes got re-zeroed by a later `_capture_phase_reference` (switching dim, reselecting) and *then* something reloaded the spectrum -- Apply Processing being the most direct case, but also Align Echo or any other FID-processing control -- F1's phase silently reverted to unphased with no way to tell it had ever been touched. Fixed via new `metadata.raw["phase0_f1"/"phase1_f1"]` fields (`_commit_pending_phase_to_metadata` now folds the F1 delta in here instead of no-op-ing) and threading the *pre-reload* spectrum's own `metadata.raw` into `_reapply_phase_post_reload` (new `previous_raw` parameter -- the freshly-reloaded spectrum never has these fields itself, since `read_2d` has no F1-phase kwarg to record them the way it does for F2) at both call sites (`_on_process_indirect_toggled`, `_on_reprocess_fid_clicked`). Verified end-to-end with a synthetic conventional-2D dataset (`_make_raw_2d_dataset`): drag F1 phase, re-anchor (zeroing the spin boxes), click Apply Processing, confirm the reloaded spectrum's data matches the original data with that exact phase correction re-applied along F1 -- `test_main_window_f1_phase_persists_across_apply_processing`.
- **"This indirect dimension... appears to often be inverted"** -- investigated, not fixed, and turned into a much bigger finding than expected. Tried to reproduce directly against real, non-pseudo 2D Bruker data in `test_data/` (a States-TPPI 2H/1H BaBa DQ dataset, `FnMODE=5`) via `read_2d`, to check both the inversion claim and (as a candidate root cause) whether `read_2d`'s F1 FFT is missing proper States/States-TPPI/Echo-Antiecho hypercomplex row-pair assembly before it does a plain complex FFT along F1 (confirmed via `nmrglue`'s own source: `ng.bruker.read()` does *not* do this decoding itself -- it's a separate step, `ng.proc_base.decode_States`/`c2ri`, that `read_2d` never calls; `bruker.py` has zero mentions of "states"/"tppi"/"hyper" anywhere). **Every single genuine 2D Bruker dataset actually available in `test_data/`** (8 real experiments checked, across two different sample datasets, `FnMODE` 0 and 5 both) **fails nmrglue's raw-`ser` shape-guessing outright** -- confirmed via `IndexError` in `read_2d` and directly re-derived in `nmrglue.bruker.guess_shape`: without the pulse program (`read_2d` deliberately passes `read_pulseprogram=False`, an earlier round's own speed/robustness choice), the shape guess falls back to a cruder file-size-based heuristic that doesn't reshape cleanly for any of them. Re-enabling pulse-program reading (`pprog_file="pulseprogram.precomp"`, the only filename present on these datasets) gets the file parsed but *still* doesn't resolve the shape -- this pulse program's own loop-counter variables (`l1`, `l8`, `l0`, `ST1CNT`, `tdav`) aren't in nmrglue's hardcoded `td0`/`td1`/`td2`/`td3` substitution table, so `guess_shape`'s loop-based branch bails out unresolved regardless. Also tried deriving the raw per-row byte count directly from `TD`/`FILE_SIZE` arithmetic, independent of nmrglue entirely -- came out non-integer for every dataset tried, meaning the simple "`ceil(TD/256)*256` int32 words per row" formula this project already uses elsewhere doesn't hold here either (very likely oversampling/digital-filter-related, i.e. the same open, previously-documented byte-padding problem from the *"loading the 2D ser... was very slow"* item below, now confirmed to affect essentially all real conventional-2D data, not just one pathological dataset). **Practical upshot**: the raw-FID path for a genuine (non-pseudo) 2D Bruker dataset essentially never actually loads today -- every real one silently falls back to `read_2d_processed` (TopSpin's own `2rr`/`2ri`), same as before this round. This is very likely the *primary* reason F1 phasing read as "impossible" in the first place (the live-phase-drag mechanism above works fine against the fallback-loaded data too, so the persistence fix still helps there) -- and the missing States/TPPI/Echo-Antiecho decode is a second, independent correctness gap that would need implementing once shape-guessing is fixed, before the raw path could be trusted for real 2D data at all. Did not attempt to hand-derive either fix -- same reasoning as the two prior unresolved shape-guessing attempts below: this needs either real Bruker-format DSP expertise or a validated reference to check against, not another guess.
- **"The truncate box keeps greying out, especially when the SI is changing"** -- investigated, not resolved. Confirmed `tdeff_spin.setEnabled(reprocessable and is_xpulse)` (X-Pulse-only by design) has exactly one call site (`_refresh_fid_processing_controls`, from spectrum reselection) and directly tested cycling every `zero_fill_combo`/`zero_fill_custom_spin` value against a real loaded X-Pulse dataset -- enabled state and value never changed, disconfirming the most obvious theory. Best remaining working theory, unconfirmed: this is the same shape-guess-failure mechanism above, manifesting as `fid_processing_box` (which contains the truncate control) being hidden outright (`setVisible(reprocessable)`) rather than merely disabled, whenever a conventional-2D Bruker selection's raw path fails to load -- "greying out" may describe that rather than a literal enabled/disabled toggle. Needs more specific repro steps (which vendor/dataset, exact click sequence) to pin down further.
- **"Pressing apply processing resets everything but it shouldn't. Honestly it shouldn't be needed at all if everything is updating live"** -- partially addressed as a side effect of the F1-phase-persistence fix above (less state is actually lost across an Apply Processing click now), but the architectural question -- should this be a live-only pipeline with no explicit "commit" step at all -- is a real design decision, not something to resolve unilaterally; still open.
- 630 tests passing (625 -> 630). Committed; not yet pushed.

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

- ~~**`fid_raw` caching for Varian/JEOL**~~ -- done (found already fixed while sweeping this file for stale entries, 2026-09-04). Both `io/varian.py` and `io/jeol.py` already stash `metadata.raw["fid_raw"]` (added alongside their respective "processing-options gap" bug fixes, above), and `MainWindow._fid_reprocessable`'s docstring explicitly lists `"xpulse"`/`"varian"`/`"jeol"` together as always-raw-FID vendors -- this bullet just never got struck through.
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

**A. Composable plotting functions** -- the biggest concrete gap: opennmr's fitting/relaxation backends produce numbers but have no matching "look at my fit" plot helpers (the GUI's own Analysis tab currently builds its plots by hand in `main_window.py` rather than calling a shared plotting function). **Partially done 2026-09-04** -- see "nmrtools2/leonmr survey, category A (partial)" above for the full writeup.
- ~~`NMR1dplot_with_fit` (`bignmrtools/fitting.py:774`) -- 1D spectrum + fitted model overlay, optional per-peak components in gray.~~ done, ported as `plotting.fit.plot_1d_with_fit`.
- `NMR2D_plot_slices` (`bignmrtools/two_d.py:1561`) -- stack-plots extracted 2D rows/columns/both with optional fit-curve/component overlays and nQ crossing markers. **Not ported** -- deliberately deferred, see above (large, and substantially overlaps the GUI's own already-working stack rendering).
- `NMR2D_summary_plot` (`bignmrtools/two_d.py:2018`) -- one figure combining a 2D contour, extracted-slice stack, and integral-vs-position panels, with slice-position lines drawn on the contour. A genuinely GUI-worthy "one figure, whole story" export. **Not ported** -- same reason as `NMR2D_plot_slices` above.
- ~~`NMR2D_plot_integrals` (`bignmrtools/fitting.py:1391`) -- fitted peak integrals across a slice series, with error bars and multi-dataset overlay.~~ done, ported as `plotting.fit.plot_2d_integrals` (the `x_axis="vdlist"` mode dropped -- see above).
- ~~`NMR2D_plot_relaxation` (`bignmrtools/relaxation.py:754`) -- plots one or more `fit_relaxation`/`NMR2D_fit_relaxation` results (data + curve, log-axis toggle, on-curve rate annotation) -- opennmr has a text report but nothing that draws this outside the GUI's own bespoke `show_relaxation_fit`.~~ done, ported as `plotting.fit.plot_relaxation`. GUI still uses its own bespoke `show_relaxation_fit` for now -- not rewired to call this yet, see above.

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
- ~~Phase 1 range: still bounded (±36000° via the spinbox) -- make genuinely unbounded~~ -- done (found already fixed while sweeping this file for stale entries, 2026-09-04). `_PHASE_SPIN_RANGE = float("inf")` (`main_window.py:55`), with a comment explicitly citing "the previous +/-36000 cap" -- superseded by an intervening round, this bullet just never got struck through.
- ~~**Bug**: switching between spectra doesn't reset the phase controls to *that spectrum's own* baseline correctly~~ -- done (found already fixed while sweeping this file for stale entries, 2026-09-04), superseded by the "Phase display redesigned"/"Current-phase display fix" rounds (2026-09-01, above): `_capture_phase_reference` now explicitly commits any pending phase into the *previous* target's own `metadata.raw` before re-anchoring (so nothing is silently lost switching away), then zeros the drag delta -- and `_refresh_current_phase_displays` shows the *total* committed phase (baked-in + live delta) via non-editable labels, not the delta itself, which is what this old report's "relative to the previous point, then zeroing out again" was actually describing (a display-confusion, not a data bug, per that round's own writeup).

## 2D / pseudo-2D processing

- ~~Phase adjustment currently only works for 1D~~ -- done. The Processing tab now shows a "Dimension" box for any 2D/pseudo-2D spectrum with retained imaginary data (`data_imag`, now also read from `2ri` in `io/bruker.py::read_2d_processed`, which previously only read the real `2rr`):
  - Conventional 2D: an F2 (direct) / F1 (indirect) combo box; the correction applies uniformly across the whole dataset along whichever dimension is picked. Preview redraws the full contour (~60ms on a 16x262144 real diffusion dataset -- fine for interactive dragging).
  - Pseudo-2D: a 1-based slice spinbox plus an "Apply this phase to every slice" checkbox. Editing one slice only ever touches that slice (verified on real T2/PFG data -- other rows stay byte-identical); checking "apply to all" broadcasts the same correction to every slice, and unchecking it again reverts every slice but the one still selected. Live preview is a line plot of just that slice (`MainWindow._show_pseudo2d_slice`, via the now-complex-aware `processing.slicing.get_trace`), not a full contour redraw. Slices are labelled `"Slice N"` (or `"Slice N (index ...)"` if a recorded index differs from position) in the canvas title and status line.
  - Switching slice/dimension re-anchors the phase controls to that view's own current state and zeroes them, matching the existing 1D reselection behaviour (`MainWindow._capture_phase_reference`).
  - Covered by 15 new tests in `tests/test_gui.py` plus a manual smoke test against `test_data/bruker_T2/11` and `test_data/bruker_PFG/150` (both real pseudo-2D-by-content datasets that auto-detect as conventional 2D -- exactly the acqu2s-unreliable case the `pseudo_2d` override exists for).

## Pseudo-2D UI

- ~~The "Force pseudo-2D" checkbox should be a switch/toggle control, not read as a radio button~~ -- superseded (found while sweeping this file for stale entries, 2026-09-04): the checkbox itself no longer exists -- renamed to `process_indirect_button` (a checkable `QPushButton`, "Process Indirect Dimension") in the "Pseudo-2D-by-default" round (2026-08-29, see "Canvas-tab layout, scroll-zoom redesign..." above), for unrelated reasons (the default flipped to pseudo-2D, needing an opt-*in* control with inverted semantics). Not literally a switch widget, but categorically not a radio button either -- leaving this closed rather than guessing further; flag if it still reads wrong in practice.
- ~~For 2D and pseudo-2D plots: the contour panel itself should be square~~ -- done 2026-09-04. `MplCanvas._apply_view_options` now defaults `ax.set_box_aspect(1.0)` whenever a 2D/pseudo-2D contour is showing (`self._contour_spectrum is not None`) and the user hasn't explicitly locked their own aspect ratio via the existing general `_aspect_ratio` feature (which still wins outright when set -- an explicit width:height choice is deliberate, this default shouldn't override it). Every other view (1D, FID, Compare, Processing preview) is unaffected -- still fully auto unless locked, exactly as before. Verified against real data: both a genuine-2D (`Cyclosporine/203`) and a pseudo-2D (`bruker_T2/11`) render with `box_aspect == 1.0`, and the rendered contour panel is visibly square (checked the actual saved figure, not just the numeric property). 5 new tests in `test_gui_canvas.py` (square by default, survives a `_redraw_contour` rebuild -- a separate code path from `show_spectrum` that also rebuilds the axes -- explicit lock still overrides it, and switching back to 1D doesn't leave the box stuck at 1.0).
- ~~For pseudo-2D specifically, there should be a toggle to turn the indirect-dimension display off entirely~~ -- found already done while investigating the square-aspect item above, 2026-09-04: the existing `pseudo2d_view_combo` ("Contour" / "Stacked slices" / "Heatmap", wired to `_on_pseudo2d_view_changed`) already covers this -- "Stacked slices" plots every slice as an offset 1D line with no F1 axis at all, exactly "the indirect-dimension display off entirely." This bullet just never got struck through when that combo was built.
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
  - **This bullet was stale** -- everything it once listed as "not built" turned out to already be built by an intervening round except the last item: region-integration (an "Integrate Regions" button/menu action, `_on_integrate_regions_clicked`), whole-peak fixed and custom bounds (the peak table's "Fixed" checkbox + right-click "Set Bounds / Fixed Parameters…" dialog), and S/N-gated amplitude fixing (`sn_noise_region_edit`/`sn_threshold_spin`) are all already exposed in the GUI -- found while investigating the fitting-wizard roadmap item below, not previously corrected here. Per-parameter fixing (not just a whole peak) was the one genuine gap, closed 2026-09-03 (see below). Still not built: fitting a real (non-pseudo) 2D spectrum's single spectrum (`fitting.fit_spectrum`/`NMRFitWorkflow`-equivalent) rather than only a pseudo-2D series.
  - ~~Individual parameters (not just an entire peak) should be fixable~~ -- done 2026-09-03, part of the "fitting wizard" roadmap phase. The peak table's row-level "Fixed" checkbox already existed (fixes all four parameters together) and custom bounds were already exposed via a right-click "Set Custom Bounds…" dialog -- both predated this round and were incorrectly listed as unbuilt just above. Renamed that dialog to "Set Bounds / Fixed Parameters…" and gave each parameter field (Shift/FWHM/Amplitude/Eta) its own "Fixed" checkbox alongside its bounds box, stored on the peak row's own column-0 item under a second custom Qt data role (`_PEAK_FIXED_ITEM_ROLE`) so it round-trips through `_update_peak_table_from_fit` (a real risk found by inspection before it could ship as a bug -- that method already had to explicitly preserve custom bounds across a fresh `QTableWidgetItem` for the same reason) and the `analysis_state` sidecar (`peak_fixed_params`, mirroring `peak_bounds`'s own persistence exactly) the same way custom bounds already did. The row checkbox still wins outright when checked (fixes everything, unchanged behaviour); per-parameter choices only take effect when it's left unchecked. `processing.fitting.build_p0`'s own `fixed` dict already supported this per-parameter (nothing to change backend-side) -- but had **no existing test at any partial (non-all-four) granularity**, backend or GUI; added `test_fit_spectrum_fixed_dict_locks_only_the_named_parameter` (a deliberately-wrong-starting-shift check: fixed stays exactly pinned, not pulled toward the true value, while FWHM/amplitude still converge) to close that real verification gap before trusting the newly-more-prominent feature.
  - ~~A sensible warning when a fit result lands too close to its own region edge~~ -- done 2026-09-03. New `_warn_if_fit_near_region_edge(results)`, called after every fresh spectral (peak) fit -- Fit Spectrum, and the two Fit Curve entry points that run their own spectral fit first when none exists yet. For each fitted peak, "its own region" is whichever Integration Region contained *that peak's starting shift* (not the fitted result, which may already have drifted outside every region entirely -- a more serious case this same check still catches, since a shift outside every region trivially fails the "close to the one it started in" test); a no-op entirely when no regions are defined, since an unconstrained fit has no edge to land near. Aggregated into one summary `QMessageBox.warning` across every slice (naming which peak, and how many of how many slices were affected) rather than one popup per slice -- a pseudo-2D fit can run across hundreds of slices, where a popup-per-slice would make the feature unusable. Threshold: within 5% of the region's own width from either edge (`_FIT_REGION_EDGE_WARNING_FRACTION`), arbitrary but conservative. **Verified against a real fit, not just constructed to match the code**: a synthetic peak (true centre 5.0 ppm) with an integration region narrower than the peak needs (4.9-4.98) genuinely pushes the fitted shift to the region's own edge (5.0 ppm, i.e. `hi - fitted_shift == 0`) because the region masks out the data the fit can see, not because of an unrelated constraint -- confirmed directly that the position-tolerance bound (`position_tolerance_spin`, default 0.5×FWHM) binds *first* at its own default value, which would have made the test pass for the wrong reason; widened it in the test fixture specifically to rule that out.
  - ~~A more readable, exportable fit report~~ -- done 2026-09-03. New "Export Fit Report…" button (Peak Picking sub-tab, below the per-slice spectral-fit display) writes a CSV covering *every* slice's fitted peak parameters (shift/FWHM/amplitude/eta ± 1-sigma error) -- unlike the existing in-app label, which only ever shows one slice at a time. A second section (blank line, then "Curve fit -- one row per peak") is appended when a Fit Curve result already exists for this spectrum, with columns for whichever parameters that model actually has (`param_names` -- differs per relaxation/diffusion model) plus RMSE. Building the CSV (`_build_fit_report_csv`) is a pure string function with no Qt/file I/O, specifically so it's testable directly against its own output rather than only "did a file get written"; the thin GUI wrapper (`_on_export_fit_report_clicked`) handles the "nothing fitted yet" case (an info message, no file dialog at all) and the save dialog itself. Verified end-to-end on a real synthetic T1-recovery series (not just structurally): the exported integral-based `M0` matched a hand-computed Gaussian-area formula (`amplitude × FWHM × sqrt(pi/(4·ln2))`) almost exactly, and the recovered `T1` matched the series' own true synthetic value to 5 decimal places.
  - ~~A way to fit one slice first, then either apply those same parameters to every other slice or use them as each next slice's own starting guess~~ -- done 2026-09-03. `processing.fitting.NMR2D_fit_slices` gained `seed_mode` (`"independent"` default, unchanged; `"propagate"` -- each slice's own converged shift/FWHM/eta seeds the next slice's starting guess; `"apply_first"` -- shift/FWHM/eta fixed at the first slice's own converged values for every slice after it, only amplitude still fit), exposed as a new "Seed each slice from:" combo in the Peak Picking sub-tab's existing "Advanced" section, persisted through `analysis_state` the same way `position_tolerance`/`sn_threshold` already are. Amplitude is deliberately never touched by either mode -- it's expected to vary across a relaxation/diffusion series, that's the entire point of fitting one. **A real, non-obvious interaction bug found and fixed by testing through the actual GUI call path, not just the bare backend function**: the GUI always attaches a shift *bound* (`_read_peak_defs`'s own position-tolerance window, centred on the table's original guess) that a first version of seeding left untouched -- so propagating a new starting guess still left the fit constrained to the *original* narrow window, silently capping how far it could ever actually move regardless of the new seed. A bare backend call with no bounds at all (the first, seemingly-successful verification) never exercised this at all. Fixed with a new `_reseed_peak_position` helper that recentres any existing shift-bound window onto the new seed, preserving its width. **Verified twice, at two different levels, deliberately**: a backend test with a synthetic peak drifting 0.15 ppm/slice (unbounded `peak_defs`, matching the first verification attempt) confirms `"propagate"` tracks the drift exactly and `"independent"` provably cannot; a separate GUI-level end-to-end test through `_on_fit_spectrum_clicked` itself is what actually caught the bounds-recentring gap, since only that path attaches the bound the bug depended on.

## Opening data

- ~~Loading should work at the *directory* level~~ -- done. Open now shows a folder picker (`QFileDialog.getExistingDirectory`); `MainWindow._scan_for_spectra` recursively scans the chosen folder for Bruker experiments and X-Pulse `.jdx`/`.dx` files (a Bruker experiment claims its own subtree so its `pdata` contents aren't re-matched separately), and `_prompt_select_spectra` shows a multi-select list when more than one match is found -- so both "pick one experiment folder directly" and "pick a parent folder full of numbered experiments" work, and X-Pulse pseudo-2D slices (nested under `raw/sliceN/`) are now selectable. Covered by tests in `tests/test_gui.py`.

## App polish / packaging

- ~~Custom app icon~~ -- logo added (`gui/resources/logo.png`), wired as the window icon. Still open: a real Dock/taskbar icon and custom app name needs actual packaging (see below), not just `setWindowIcon`.
- ~~A real top-level menu bar (native menu bar on macOS, in-window menu bar on Windows/Linux)~~ -- done, see "Phase-tab feedback, plot styling, menu bar" above.
- A proper installer/distributable build for Windows, macOS, and Linux -- should be "beyond easy" to get running on any of the three, not require the user to set up a conda environment and run `pip install -e .` by hand. Needs a packaging tool (e.g. `PyInstaller`, `briefcase`, or `py2app`/`cx_Freeze`) producing a standalone app (with the icon above baked in) per platform, ideally with a one-click/one-command install path.
