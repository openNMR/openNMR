# openNMR

Vendor-agnostic NMR import, processing, plotting, and a desktop GUI.

Every vendor importer reads raw acquisition data (not a vendor's own
pre-processed output) and produces the same `NMRSpectrum` object, which
the same processing and plotting code then operates on. Currently
supported: Bruker (TopSpin `fid`/`ser`), X-Pulse (JCAMP-DX FID, including
arrayed/PFG experiments), Varian/Agilent (`fid`/`procpar`).

## Layout

- `opennmr.core` — vendor-agnostic data model (`NMRSpectrum`, `NMRAxis`,
  `NMRMetadata`) and nucleus/gyromagnetic-ratio lookups.
- `opennmr.io` — per-vendor readers, all producing `NMRSpectrum`.
- `opennmr.processing` — FID processing (apodization, FFT, phasing),
  VOCS, spectrum arithmetic, 2D slice extraction, peak fitting,
  relaxation fitting.
- `opennmr.plotting` — matplotlib plotting, `ax`-first so it embeds
  cleanly in a GUI canvas as well as standalone notebook figures.
- `opennmr.gui` — PySide6 desktop shell.

## Install (editable, into an existing environment)

```
pip install -e ".[test]"
```

## Run the GUI

```
opennmr-gui
```

## Run tests

```
pytest
```
