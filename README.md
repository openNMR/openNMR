Vendor-agnostic NMR import, processing, plotting, and a desktop GUI.

Every vendor importer reads raw acquisition data (not a vendor's own
pre-processed output) and produces the same `NMRSpectrum` object, which
the same processing and plotting code then operates on. Currently
supported: Bruker (TopSpin `fid`/`ser`), X-Pulse (JCAMP-DX FID, including
arrayed/PFG experiments), Varian/Agilent (`fid`/`procpar`), JEOL Delta
(`.jdf`, 1D only so far).

## Layout

- `opennmr.core` — vendor-agnostic data model (`NMRSpectrum`, `NMRAxis`,
  `NMRMetadata`) and nucleus/gyromagnetic-ratio lookups.
- `opennmr.io` — per-vendor readers, all producing `NMRSpectrum`.
- `opennmr.processing` — FID processing (apodization, FFT, phasing),
  VOCS, spectrum arithmetic, 2D slice extraction, peak fitting,
  relaxation and diffusion fitting, PFG b-values and gradient
  calibration, DOSY by inverse Laplace transform, and multivariate
  component resolution (SCORE/OUTSCORE).
- `opennmr.plotting` — matplotlib plotting, `ax`-first so it embeds
  cleanly in a GUI canvas as well as standalone notebook figures.
- `opennmr.gui` — PySide6 desktop shell.

## Install from PyPI

```
pip install opennmr
opennmr-gui
```

A splash screen shows while the program loads, and a second launch brings the
running window to the front instead of opening another (`opennmr-gui
--new-instance` or `OPENNMR_MULTI_INSTANCE=1` opens a second window on purpose;
`OPENNMR_SPLASH=0` turns the splash off).

Python 3.12 or newer. The same package serves a notebook (`from opennmr
import notebook`) and the GUI.

### Updating

Settings ▸ Updates shows the installed version and checks PyPI when you
press **Check for Updates** (a check at start-up is off unless you turn it
on). When a newer version exists, **Update to X and restart** closes
openNMR, upgrades it with pip in the same Python environment, and opens it
again; nothing is shown while it runs. If the new version does not start,
the old one is put back, and Settings ▸ Updates says so the next time.
Details are kept in `~/.opennmr/update.log`. Where pip may not change the
environment (a system Python, say), the button is not offered and the
command to run yourself is shown instead: `pip install --upgrade opennmr`.

## Install (editable, into an existing environment)

```
pip install -e ".[test]"
```

## Run the GUI

```
opennmr-gui
```

## Add an OpenNMR button to TopSpin

```
opennmr-topspin-install
```

Finds the TopSpin installations on the local machine, offers to copy the
OpenNMR plugin into each installation's `exp/stan/nmr/py/user/` folder,
and to add the `opennmr` launch command to TopSpin's `autostart.mac`.
At the TopSpin command line, `opennmr` puts the button on the main
toolbar and `opennmr -kill` takes it off again.

**The plugin records the absolute path of the environment you run the
installer from**, rather than looking for `opennmr-gui` on PATH when the
button is clicked. Depending on how TopSpin is launched, shell startup
files may not be read, so a conda install is invisible to it, and on
Windows neither conda nor python.org's installer puts itself on PATH by
default. The cost is that moving or recreating the environment can break
the button; re-run the installer if it does.

`--topspin FOLDER` defines a TopSpin installation directory the search
missed; `--uninstall` removes the plugin and its autostart entry;
`--list` shows what was found. Writing into TopSpin's own folder may
need administrator rights.

## Run tests

```
pytest
```

## Getting good data for diffusion and relaxation analysis

The analysis methods in this package differ enormously in how much they
ask of the data, and **the demanding ones fail quietly**: they return a
confident-looking answer with an excellent fit statistic. This section
is what each one actually needs. The GUI repeats the essentials in the
relevant dialogs.

### The single most important thing

**Fit quality does not tell you whether a multivariate resolution is
right.** Measured directly on synthetic mixtures in this repository's
own test suite: an OUTSCORE run whose diffusion coefficients were **69%
wrong** reported R² = 0.99940, against 0.99937 for the run that got them
right. The error surface really is that flat — wildly different answers
reproduce the data equally well. A good R² means "this many components
can fit", never "these are the right components".

This is why `processing.score` reports cross-talk alongside R², warns
when the two methods disagree while fitting equally well, and says in
its own report which number the run actually optimised.

### Acquisition

| | Peak-by-peak fit | DOSY (ILT) | SCORE / OUTSCORE |
|---|---|---|---|
| Gradient levels / delays | 8+ | 12+ | 8–10 minimum, 15+ preferred |
| Attenuation of the fastest component | to ~20% | to ~5% | to ~10% |
| Phase / baseline | helps | required | required |
| Resolves overlapped peaks | no | no | yes |

* **Sample the decay, don't just start it.** A component that loses
  only 20% of its intensity across the whole series is nearly a
  constant, and its decay constant is poorly determined no matter how
  good the fit looks. There is a test in `tests/test_score.py` pinning
  this down: a component attenuating by 8% is recovered only to within
  a factor, which is a property of the experiment and not of the
  algorithm. The fix is a longer diffusion time or stronger gradients.
* **Space the levels sensibly.** Equal steps in *b* (equal steps in
  gradient *squared*) sample a decay far better than equal steps in
  gradient amplitude, which crowds points into the undecayed start.
* **Keep signal-to-noise up on the last levels too**, not just the
  first. The most-attenuated points carry most of the information about
  the slow components.
* **Don't let peaks move.** A bilinear model — the assumption behind
  SCORE, OUTSCORE and DOSY alike — says each component has one fixed
  spectrum. A peak that drifts across the series (temperature,
  exchange, referencing) breaks that, and shows up as a spurious extra
  component rather than as an error.

### Processing, before any of these methods

1. **Phase properly, in absorption mode.** Non-negativity is the single
   most stabilising constraint available to SCORE, OUTSCORE and the
   ILT, and it means nothing on a dispersive spectrum.
2. **Baseline-correct.** A rolling baseline is not a non-negative
   superposition of component spectra; it will be absorbed into
   whichever component can best accommodate it.
3. **Reference consistently across the series.**
4. **Use real b-values.** The GUI refuses to run DOSY or SCORE when the
   x-axis has fallen back to plain slice indices, because the resulting
   "diffusion coefficients" would be in no physical unit while being
   labelled m²/s. Either read them from a Bruker `diff.xml`, or supply
   δ/Δ/gradient and let `processing.diffusion.b_value` compute them —
   and get the **gradient shape factor** right: treating a stock Bruker
   `SMSQ10` sequence as a perfect rectangle overestimates *b* by 23%,
   which lands directly on D.

### Where the x-values come from

The Curve Fitting sub-tab's **Values** source sets the x-axis for Fit
Curve, DOSY and SCORE alike — one setting, so the three cannot end up
fitting the same series against different axes. The line under it
always shows what the current source resolves to, against the selected
experiment.

| Source | What it reads |
|---|---|
| Auto | this experiment's own `vdlist`/`diff.xml`, else slice index |
| Manual list | values typed in, comma- or whitespace-separated (`10m` etc. accepted) |
| Start + step | an evenly-spaced series, one value per slice |
| **List file × factor** | **any one-column text file of numbers, scaled by a constant or an acquisition parameter** |
| Relative gradient | an X-Pulse arrayed PFG series' own gradient fractions |
| b from gradients | a true b-value computed from δ/Δ/G |

**List file × factor** is the vendor-neutral one. `opennmr.io.lists`
reads any one-column list — a Bruker `vdlist`/`vclist`/`difflist`, an
X-Pulse `parameters/…_valueslist.txt`, your own `delays.txt` — and
`find_value_lists` offers whatever is actually sitting with the data
(the experiment folder, Bruker's `lists/` tree, X-Pulse's `parameters/`,
and the parent folder), with Browse… for anything else. Nothing is
inferred from the filename: the Factor is what turns a loop counter or
a relative fraction into a physical quantity.

The **Factor accepts an expression in the experiment's own acquisition
parameters**, not only a number — the same names a custom fitting
equation uses, from `opennmr.core.acquisition.acquisition_constants`
(`TAU`, `P90`, `D70` for X-Pulse; `d1`, `p1`, `cnst2`, `l4` for
Bruker). So an X-Pulse CPMG echo-count list becomes a T2 delay series
with a factor of `TAU*2`, with no arithmetic done by hand. The panel
shows what the expression resolved to and which parameters went into
it. Parameters are used **exactly as the vendor recorded them** — an
X-Pulse `TAU` is in µs and stays in µs, so any conversion belongs in
the expression; only the person reading the pulse programme knows the
unit it was written in.

Everything in the panel is **per experiment**. Selecting a different
spectrum re-scans *its* folder for list files and restores *its* own
factor, unit and source — nothing is carried over from the one before.

**Unit.** The Unit combo says what the numbers in this panel *are*,
not only how they are displayed — a typed list of `10, 20, 40` and a
list *file* holding `10, 20, 40` both mean 10, 20, 40 of that unit.
It defaults to **ms**; fits are always done in seconds. For a list
file it applies to the whole product, list × Factor, which makes the
X-Pulse CPMG case simpler rather than harder: with the Unit on µs an
echo-count list takes a factor of plain `TAU*2`, since `TAU` is
already recorded in µs.

Two things are never rescaled by it, because they already carry a unit
of their own: a value that names one (`10m` is 10 ms whatever the combo
says — a `vdlist` written that way is taken at its word) and the **Auto**
source, which is the vendor's own values in the vendor's own unit. A
diffusion fit has no Unit row at all: a b-value has no comparable
prefix convention.

The unit reaches the **whole** panel, not just the conversion: the Fit
tab's x-axis and label, the guess table's time rows (in both
directions — a `T2` guess typed as `50` under a ms heading reaches the
model as 0.05 s), "adopt fit as guess", the bounds error messages, and
the CSV export. Only *time* parameters scale —
`relaxation._TIME_PARAMS`, the same set that decides which reported
values get a time unit printed after them — so an amplitude, a stretch
exponent and a diffusion coefficient are left alone. The stored fit is
never modified; the unit is a display transform, so changing it redraws
rather than refits.

Fitted values in the report keep `format_time_value`'s automatic
ms/µs/ns scaling rather than being forced to the chosen unit: it is
never wrong, and pinning it would print a 2 s result as "2000 ms".

**Slices.** The Curve Fitting panel's Slices box limits the analysis to
the first N slices of a series ("all" by default). One setting feeds
Fit Spectrum, Fit Curve, DOSY and SCORE, so they cannot disagree about
how much of a series is in use; DOSY and SCORE cut the data matrix with
it, not only the x-values.

A value list **longer** than the data is not an error — the surplus is
simply unused, the way TopSpin uses the first F1-TD entries of a longer
`vdlist`. A shorter one cannot be fitted, and says so with the slice
count to set. Pseudo-2D row counts are capped at `acqu2s`'s own `TD`
(the *status* parameter file): the floor-division fallback in
`_guess_raw_2d_shape` only exists to find a reshapable shape, and rows
past TD are not acquired data.

**Relative amplitudes.** Any model with more than one amplitude reports
each one as a fraction of the total — on the parameter's own row and
again in a section pairing it with the rate it belongs to
(`relaxation.amplitude_fractions`). That is the form a multi-component
result is quoted and compared in; a raw `A1 = 0.0532` is on whatever
arbitrary intensity scale the spectra were processed on and says
nothing on its own. The compact per-peak summary and the global fit's
per-curve block quote amplitudes *only* as shares, since comparing
curves is the whole reason those blocks exist.

Amplitudes are recognised by name — `M0`, `A`/`A1`/`A2`, `I0`/`I0a`/
`I0b`, `I1`…`In` — the same way `_TIME_PARAMS` decides what gets a time
unit. A **custom equation gets no ratios**: its parameters are whatever
you named them, and treating the wrong symbol as an amplitude would be
worse than showing nothing. A share outside 0–100% (components of
opposite sign, usually an over-parameterised model) is flagged rather
than hidden.

**Export curve fit (CSV)** writes the data behind the plot — the x-axis
in the displayed unit, then every peak's measured integral or
amplitude at each point with the fitted curve beside it
(`processing.relaxation.curve_fit_to_csv`). Peaks are not assumed to
share an x-axis: a point excluded from one peak's fit leaves an empty
cell rather than a number that was never measured.

### SCORE fits the same model Fit Curve does

SCORE's per-component decay is not a private exponential: it is
whichever `processing.relaxation.RELAXATION_MODELS` entry the Model
combo names, evaluated through `processing.score.build_kernel`. So a
stretched or inversion-recovery model resolves components with a β or
an inversion efficiency of their own, and a Stejskal-Tanner fit and a
SCORE resolution of the same series are the same function asked two
different questions.

The model's **amplitude** parameter is pinned at 1 rather than fitted —
each component's size lives entirely in its resolved spectrum, which is
solved for linearly, and fitting both would be exactly redundant. Which
parameter that is gets measured (doubling it doubles the curve), not
matched against a list of names, so a custom equation works too. A
model that already describes several populations (a biexponential) has
no such parameter and is refused with that reason: SCORE's own
component count is doing that job.
5. **Calibrate the gradient if you don't know it.** A maximum gradient
   10% wrong makes every D **21%** wrong, with nothing in the fit to
   show for it. `Analysis ▸ Calibrate Gradient…` measures it from a
   standard of known D.
6. **Check the gradient axis.** A probe's maximum gradient is saved per
   axis, because the three coils of a triple-axis probe are rarely the
   same strength — one number shared between them quietly rescales
   every D measured on the other two. Set the axis alongside the
   maximum; "Save gradient for probe" then stores that axis's value
   and leaves the others alone. It is the same setting the Imaging tab
   shows, and the same number: one experiment, one coil, one maximum.

### Reference deconvolution (`processing.refdeconv`)

> **Not wired into the GUI, and not usable on a processed spectrum.**
> It inverse-transforms the spectrum back to its FID, so the input must
> be the transform of a *causal* FID. A TopSpin `2rr`/`2ii` is not — it
> carries the digital-filter group delay and has apodisation and
> baseline correction folded in. Applied to one, it destroys the line
> (measured: a peak of 2397 spanning 0.0159 ppm became 16 spanning
> 0.536 ppm). It needs the raw FID path; that wiring does not exist yet.


Corrects a lineshape distortion that **every peak shares** — a bad
shim, a non-spinning sample, or the electrodes in an eNMR cell. Pick an
isolated resonance, assert the shape it should have had, and the ratio
of ideal to measured is the distortion; divide it out and every peak is
corrected at once.

```python
from opennmr.processing import refdeconv
result = refdeconv.reference_deconvolve(
    spectrum_row, ppm, region=(-0.2, 0.2), fwhm=0.01)   # target width, ppm
```

Validated against a constructed truth (build an FID, distort it,
transform): a pure phase distortion broadening a line from 0.147 to
0.244 ppm is corrected back to 0.159, **and peaks outside the reference
region are corrected too** — the method's whole claim. Recovery degrades
as the distortion destroys more of the FID (0.031 s of surviving signal
recovers only 0.574 → 0.366); that limit is information, not
implementation.

`REGULARISATIONS` is the method, not a detail — the reference FID decays
into noise and an undamped reciprocal diverges. `wiener` (default)
rolls off smoothly; `truncate` is sharper but rings; `none` exists to
demonstrate the failure.

It does **not** fix convection or electro-osmotic flow (per-scan phase
modulation, not one fixed envelope), needs a genuinely isolated
reference, and gives you whatever linewidth you ask for — asking for one
the acquisition cannot support amplifies noise until the fit looks sharp
and means nothing.

### eNMR (`processing.enmr`, and the Analysis ▸ eNMR tab)

The phase a resonance picks up under an applied field is linear in the
voltage: `Δφ = γ·g·δ·Δ·μ·V/d`.

**The slope is the result** — it is what the data determines, in rad/V,
and needs nothing but the data. Converting it to a mobility needs the
cell's electrode gap and the sequence's gradient and timings, which live
in a lab notebook rather than any vendor file, so that step is optional:

```python
from opennmr.processing import enmr
phases = enmr.phase_series(rows, ppm, region=(3.18, 3.42))

enmr.fit_enmr(voltages, phases)                      # slope alone
enmr.fit_enmr(voltages, phases, electrode_gap=0.0385,
               gamma=2 * np.pi * 42.577e6, gradient=0.3,
               delta=1e-3, big_delta=0.1)            # ... and a mobility
```

**Two levels of pairing, and they are different things.**

*Within one experiment*, each voltage is acquired at **both
polarities**. Reversing the field reverses the drift, so
`(φ₊ − φ₋)/2` doubles the signal and cancels every phase error that is
not field-driven. `polarity_pairs` detects this rather than assuming
it.

*Across experiments*, the whole run is repeated with the **gradient
reversed**, which flips the sign of the fitted slope. `combine_experiments`
aligns the signs and averages — a plain average would cancel the
measurement away — and reports the `spread`, because two runs of one
sample agreeing is the evidence that either measured anything. On a real
reversed pair: +1.244e-2 and −1.396e-2 → **+1.320e-2 ± 1.5e-3 rad/V**,
spread 8%.

**Unwrap with `phase_ramp`, not `phase_series`.** Consecutive points of
a +V/−V series are opposite polarity and legitimately far apart, so
unwrapping across them makes `np.unwrap` read a polarity jump as a wrap
and insert a 2π step that was never there. On real data that turned a
clean ramp (R² 0.88) into no correlation at all (R² 0.009) — it does not
look like a broken calculation, it looks like a negative result.
`phase_ramp` takes the voltages and picks the stride itself.

**Validated against an accepted literature value**, the only check that
catches a wrong 2π, a wrong drift time or a unit slip, all of which
still produce a beautiful straight line. On a TMABr standard
(`test_data/eNMR/...2024-05-13/21`):

| species | slope | R² | μ |
|---|---|---|---|
| TMA⁺ | 7.90e-3 ± 2.2e-4 rad/V | 0.9954 | **3.79e-8 m²/Vs** (lit. 3.79e-8) |
| water | 3.3e-5 ± 8.8e-5 rad/V | 0.023 | zero to within error |

Water sits in the same tube, field and shim, so it controls for
everything except charge.

Phase comes from the **complex integral** over the peak region, which on
this data gives R² = 0.995 with no lineshape fit at all; a full complex
fit is for *overlapped* peaks, not for phase accuracy as such.

**The PFG constants read themselves.** `pfg_params(spectrum)` — or
`fit_enmr(..., spectrum=spectrum)` — fills in γ, g, δ and Δ, so only the
electrode gap needs typing. Which source each comes from was measured
across five real datasets, because the pulse programme's legend is not
enough: `Δ = d20` reproduces `diff.xml` exactly, but `δ` does not
(`d17+d18` is 67–83% out on four of five), and the gradient in T/m needs
the probe calibration that only `diff.xml`/`diffConf.xml` carries. So
`diff.xml` is the source of record, `d20` the cross-check, and each
value reports its provenance. γ is converted to rad/s/T once, on the way
out.

`electrode_gap` has no default, and a half-supplied set of constants is
rejected rather than silently skipped: μ scales directly with the gap,
no vendor file records it, and a plausible-looking default would quietly
rescale every result.

### Choosing a method

* **One clean, well-separated peak per species** → ordinary curve
  fitting (`Analysis ▸ Curve Fitting`). It is the only one of these
  that gives a diffusion coefficient with a meaningful uncertainty.
* **Peaks belonging to the same molecule** → tick *Fit all peaks
  together* and share D. Fitting separately and averaging understates
  the uncertainty.
* **Overlapped peaks, species differing ≳30% in D** → **SCORE**.
* **Overlapped peaks, species differing by only a few percent in D** →
  **OUTSCORE**, which minimises cross-talk rather than residual. It
  separates down to ~3% where SCORE needs ~30% — but it assumes the
  components have genuinely *different* peaks. For species that legitimately
  share most of their resonances, minimising overlap pushes toward
  something untrue; the report warns when that has happened.
* **Unknown number of components, or a continuous distribution** →
  the **DOSY map** (`Analysis ▸ DOSY Map (ILT)`). Treat peak *positions*
  as the result and peak *widths* as an artefact of the regularisation.

Both `SCORE`/`OUTSCORE` and the curve fits take the **number of
components as an input, not an output**. Run at one more and one fewer
and prefer the answer whose component spectra look like plausible,
separate molecules.
