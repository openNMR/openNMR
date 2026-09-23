"""Where a PFG series' b-values and timings come from -- one answer for
the GUI and for notebooks.

The two functions here are what the GUI's diffusion fit calls (the
"Auto" Values source, and "Read from file" for delta/Delta), so a
notebook gets exactly the numbers the GUI would, with the same record of
where each came from::

    from opennmr import io
    from opennmr.notebook import diffusion

    spectrum = io.load("path/to/experiment")
    found = diffusion.recorded_b_values(spectrum)
    found["b"], found["source"], found["warnings"]

Vendors covered (TODO.md section 0, seams 1-2 and 4):

* **Bruker** -- :func:`opennmr.io.bruker_pfg.read_pfg_params`: TopSpin's
  own b-values from ``diff.xml`` where it exists; otherwise ``difflist``
  gradients (the ``dosy`` AU) with delta/Delta read from the pulse
  programme's own parameter documentation.
* **Spinsolve** -- the b-values :func:`opennmr.io.spinsolve.read_spectrum`
  stored at load (``delayTimes.txt``).
* **X-Pulse** -- timings only, via
  :func:`opennmr.processing.diffusion.pfg_params_from_metadata` (it
  records relative gradient steps, not b-values).

Nothing is guessed. A value the files do not give is absent, and
anything a reader wants the user to see before trusting a number is in
``warnings``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..io import bruker_pfg as _bruker_pfg
from ..processing import diffusion as _diffusion

#: Errors a Bruker PFG read can raise on an incomplete or odd dataset.
#: Reported as warnings rather than raised: a missing gradient list
#: should fall back to the next Values source, not stop the fit.
_READ_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError)


def _read_bruker(spectrum) -> tuple[Any, list[str]]:
    path = getattr(spectrum.metadata, "source_path", None)
    if not path:
        return None, []
    try:
        return _bruker_pfg.read_pfg_params(path, nucleus=spectrum.metadata.nucleus), []
    except (FileNotFoundError, NotADirectoryError):
        return None, []
    except _READ_ERRORS as exc:  # a dataset the reader could not parse
        return None, [f"could not read this experiment's PFG parameters: {exc}"]


def _bruker_b_source(pfg) -> str:
    if pfg.sources.get("b") == "diff.xml":
        return "diff.xml (TopSpin's own b-values)"
    parts = [f"difflist gradients ({pfg.sources.get('difflist', 'G/cm')})"]
    if "delta" in pfg.sources:
        parts.append(f"δ from {pfg.sources['delta']}")
    if "big_delta" in pfg.sources:
        parts.append(f"Δ from {pfg.sources['big_delta']}")
    if "tau" in pfg.sources:
        parts.append(f"τ from {pfg.sources['tau']}")
    if "b" in pfg.sources:
        parts.append(f"b {pfg.sources['b']}")
    return "; ".join(parts)


def _bruker_warnings(pfg) -> list[str]:
    # The diff.xml branch carries one informational line ("timings are
    # reported for display") -- a note about provenance, not something
    # to flag with a warning sign.
    if pfg.sources.get("b") == "diff.xml":
        return []
    return list(pfg.warnings)


def recorded_b_values(spectrum) -> dict | None:
    """The b-values (s/m^2) this experiment's own files give, or ``None``
    when its vendor records none.

    Returns ``{"b": ndarray | None, "source": str | None,
    "warnings": list[str]}``. ``b`` is ``None`` (with the reason in
    ``warnings``) when the vendor is covered but this dataset is
    incomplete -- e.g. a Bruker ``dosy`` experiment whose pulse
    programme does not document delta.
    """
    vendor = getattr(spectrum.metadata, "vendor", None)
    raw = spectrum.metadata.raw or {}

    if vendor == "spinsolve" and raw.get("b_values") is not None:
        return {"b": np.asarray(raw["b_values"], dtype=float),
                "source": raw.get("b_source", "delayTimes.txt"), "warnings": []}

    if vendor == "bruker":
        pfg, errors = _read_bruker(spectrum)
        if pfg is None:
            return {"b": None, "source": None, "warnings": errors} if errors else None
        if pfg.b is None:
            return {"b": None, "source": None, "warnings": _bruker_warnings(pfg)}
        return {"b": np.asarray(pfg.b, dtype=float), "source": _bruker_b_source(pfg),
                "warnings": _bruker_warnings(pfg)}

    return None


def recorded_pfg_params(spectrum) -> dict:
    """delta, Delta, tau (seconds), sequence and gradient shape as this
    experiment's own files record them -- ``{}`` where nothing is
    recorded.

    Shaped like :func:`opennmr.processing.diffusion.pfg_params_from_metadata`
    (so ``delta_is_plateau_only`` says whether ``delta`` still needs a
    ramp added), plus ``sources`` (value name -> where it came from) and
    ``warnings``.
    """
    vendor = getattr(spectrum.metadata, "vendor", None)
    raw = spectrum.metadata.raw or {}

    if vendor == "spinsolve" and raw.get("pfg_params"):
        params = raw["pfg_params"]
        found = {key: params[key] for key in ("delta", "DELTA", "sequence", "shape")
                 if params.get(key) is not None}
        found["delta_is_plateau_only"] = False  # data.2d.txt's encoding time includes the ramp
        found["sources"] = dict(params.get("sources", {}))
        found["warnings"] = []
        return found

    if vendor == "bruker":
        pfg, errors = _read_bruker(spectrum)
        if pfg is None:
            return {"sources": {}, "warnings": errors} if errors else {}
        found = pfg.as_pfg_params()
        found["sources"] = {("DELTA" if key == "big_delta" else key): text
                            for key, text in pfg.sources.items()
                            if key in ("delta", "big_delta", "tau")}
        found["warnings"] = _bruker_warnings(pfg)
        if pfg.sources.get("b") == "diff.xml":
            found["b_note"] = ("b-values come from diff.xml directly; these timings are "
                               "for display and for the 'b from gradients' source")
        return found

    found = _diffusion.pfg_params_from_metadata(spectrum)
    if found:
        found["sources"] = {key: f"{vendor} acquisition parameters"
                            for key in ("delta", "DELTA", "ramp") if key in found}
        found["warnings"] = []
    return found
