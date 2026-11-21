"""
Electrochemistry data for NMR: read a cycler file, derive capacity and composition,
select cycles, and find where an ex situ sample or an operando spectrum sits on the curve.

Biologic ``.mpr``/``.mpt`` files are read with galvani, which comes with openNMR, and plain
delimited tables (``.csv``/``.tsv``/``.txt``/``.dat``) are read
with pandas, so any export with a header row works. Every column of the file is
kept, so *whatever is in the file can be plotted*. The standard columns below are added where
they can be worked out.

======================  ===============================================================
``time_s``              time since the start, seconds
``time_h``              the same in hours
``voltage_V``           potential, volts
``current_mA``          current in mA (positive = charge, i.e. oxidation of the sample)
``state``               ``"charge"``, ``"discharge"`` or ``"rest"``
``half_cycle``          0, 1, 2 … counting each run of charge or discharge
``cycle``               1, 2, 3 … (two half cycles per cycle)
``capacity_mAh``        charge passed in this half cycle, from 0, mAh
``capacity_spec_mAh_g`` that per gram of active material (needs the mass)
``capacity_area_mAh_cm2``  that per cm² of electrode (needs the area)
``x``                   composition of the mobile ion per formula unit (needs the molar
                        mass and the starting composition)
======================  ===============================================================

Composition (Faraday's law)::

    x = x0 − s · Q_net · M / F        Q_net: net charge passed since the start, per gram
                                      of active material (mAh/g); M: g/mol;
                                      F = 26801.48 mAh/mol; s = +1 when the ion leaves on
                                      charge (a cathode), −1 when it enters (an anode).

Notebook::

    from opennmr.notebook import echem
    cell = echem.load_echem("cell.mpr", mass_mg=5.2, formula="LiFePO4", x0=1.0)
    df = cell["data"]
    point = echem.ex_situ_point(df, 3.45, cycle=2, half="charge")      # one sample
    table = echem.at_spectra(df, *echem.operando_times(40, per_spectrum_s=900))
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

#: Faraday constant in mAh/mol (96485.33212 C/mol ÷ 3.6 C per mAh).
FARADAY_MAH_PER_MOL = 96485.33212 / 3.6

_NAVANI_EXTENSIONS = (".mpr", ".mpt", ".res", ".xlsx", ".xls", ".nda", ".ndax", ".bdf",
                      ".bdf.csv", ".bdf.gz", ".bdf.parquet")
_TABLE_EXTENSIONS = (".csv", ".tsv", ".txt", ".dat")

#: Plain-language names for the standard columns (axis labels).
AXIS_LABELS = {
    "time_s": "Time (s)",
    "time_h": "Time (h)",
    "voltage_V": "Voltage (V)",
    "current_mA": "Current (mA)",
    "capacity_mAh": "Capacity (mAh)",
    "capacity_spec_mAh_g": "Specific capacity (mAh g$^{\\mathregular{-1}}$)",
    "capacity_area_mAh_cm2": "Areal capacity (mAh cm$^{\\mathregular{-2}}$)",
    "x": "Composition, $x$",
    "cycle": "Cycle",
    "half_cycle": "Half cycle",
}

_CANDIDATES = {
    "time_s": ("Time", "time/s", "Time (s)", "time", "Test_Time(s)", "Test Time (s)", "t/s",
               "Total Time(s)", "Time/s"),
    "voltage_V": ("Voltage", "Ewe/V", "Voltage (V)", "Voltage(V)", "Potential (V)", "Ewe",
                  "E/V", "Potential/V", "Voltage/V", "Voltage_V", "V"),
    "current_mA": ("Current", "I/mA", "<I>/mA", "Current (mA)", "Current(mA)", "I (mA)",
                   "Current/mA", "Current_mA"),
}


def _trapezoid(y, x):
    return (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(y, x)


# ============================================================
# Reading
# ============================================================

def _first_column(df: pd.DataFrame, names) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        hit = lowered.get(name.strip().lower())
        if hit is not None:
            return hit
    return None


def _read_table(path: str) -> pd.DataFrame:
    """A delimited text file with a header row: comma, tab, semicolon or whitespace."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        head = "".join(fh.readline() for _ in range(60))
    skip = 0
    # EC-Lab exports declare how many header lines there are.
    for line in head.splitlines():
        if line.lower().startswith("nb header lines"):
            try:
                skip = int(line.split(":")[1]) - 1
            except (IndexError, ValueError):
                skip = 0
            break
    # The delimiter is the one that appears the same (non-zero) number of times on every
    # line of the table. Counting characters would mistake a decimal comma for a separator.
    body = [ln for ln in head.splitlines()[skip:] if ln.strip()][:20]
    sep = r"\s+"
    for candidate, name in (("\t", "\t"), (";", ";"), (",", ",")):
        counts = {ln.count(candidate) for ln in body}
        if len(counts) == 1 and counts != {0}:
            sep = name
            break
    # A semicolon-separated export from a European locale writes 3,45 for 3.45.
    decimal = "," if (sep == ";" and re.search(r"\d,\d", head)) else "."
    df = pd.read_csv(path, sep=sep, skiprows=skip, encoding="utf-8-sig", engine="python",
                     decimal=decimal)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _biologic_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Time, voltage and current from the columns of a Biologic file (``.mpr`` or ``.mpt``).
    Current is the file's ``I/mA`` (``<I>/mA``) when it has one, otherwise the charge step over
    the time step, which is how navani gets it too."""
    data = data.rename(columns=lambda c: str(c).replace("<", "").replace(">", ""))
    if "time/s" not in data or "Ewe/V" not in data:
        raise ValueError("The Biologic file has no time/s or Ewe/V column.")
    out = pd.DataFrame({"Time": data["time/s"].to_numpy(float),
                        "Voltage": data["Ewe/V"].to_numpy(float)})
    dt = np.diff(out["Time"].to_numpy(float), prepend=out["Time"].iloc[0])
    if "I/mA" in data:
        out["Current"] = data["I/mA"].to_numpy(float)
    else:
        charge = next((c for c in ("dQ/mA.h", "dq/mA.h") if c in data), None)
        if charge is not None:
            dq = data[charge].to_numpy(float)
        elif "Q charge/discharge/mA.h" in data:
            dq = np.diff(data["Q charge/discharge/mA.h"].to_numpy(float), prepend=0.0)
        else:
            raise ValueError("The Biologic file has neither a current nor a charge column.")
        with np.errstate(divide="ignore", invalid="ignore"):
            current = dq / (dt / 3600.0)
        out["Current"] = np.where(np.isfinite(current), current, 0.0)
    return out


def _read_biologic(path: str) -> pd.DataFrame:
    """A Biologic ``.mpr`` or ``.mpt`` through galvani alone. galvani is a core dependency;
    navani is not needed for these two."""
    from galvani import BioLogic
    with open(path, "rb") as handle:
        if path.lower().endswith(".mpt"):
            data = pd.DataFrame(BioLogic.MPTfile(handle, encoding="latin-1")[0])
        else:
            data = pd.DataFrame(BioLogic.MPRfile(handle).data)
    return _biologic_frame(data)


_BIOLOGIC_EXTENSIONS = (".mpr", ".mpt")


def _read_navani(path: str) -> pd.DataFrame:
    """Biologic files through galvani, which comes with openNMR. Any other instrument format is
    handed to navani only if it happens to be installed; it is not a dependency, because it
    cannot be installed on every supported Python, so everyone is told the same thing:
    export a csv."""
    if path.lower().endswith(_BIOLOGIC_EXTENSIONS):
        try:
            return _read_biologic(path)
        except ImportError:
            pass  # galvani missing from an odd install: try navani below
    try:
        from navani import echem as navani_echem
    except ImportError as exc:
        raise ImportError(
            "openNMR reads Biologic .mpr and .mpt files and plain .csv/.txt tables. For any "
            "other cycler software, export the data as a .csv table.") from exc
    return navani_echem.echem_file_loader(path)


def load_echem(path, mass_mg: float | None = None, area_cm2: float | None = None,
               formula: str | None = None, molar_mass: float | None = None,
               x0: float | None = None, charge_removes_ion: bool = True,
               reader: str = "auto", current_unit: str = "mA", time_unit: str = "s",
               capacity_from: str = "integrated", rest_fraction: float = 1e-4) -> dict:
    """Read a cycler file and add the standard columns (see the module notes).

    ``reader``: ``"auto"`` (a plain table for ``.csv``/``.tsv``/``.txt``/``.dat``, galvani for
    Biologic files, then navani if it is installed, as the fallback for a table it cannot make
    sense of), ``"navani"`` or ``"table"``. ``current_unit`` / ``time_unit`` say what the file's
    columns are in (``"A"``, ``"mA"``, ``"uA"``; ``"s"``, ``"min"``, ``"h"``). The mass,
    area, molar mass (or ``formula``) and ``x0`` are optional; each derived column
    appears only when what it needs is given. Returns ``{"data", "source", "reader",
    "settings", "notes"}``."""
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if reader not in ("auto", "navani", "table"):
        raise ValueError("reader must be 'auto', 'navani' or 'table'.")
    lower = path.lower()
    notes: list[str] = []
    used = reader
    if reader == "auto":
        used = "table" if lower.endswith(_TABLE_EXTENSIONS) else "navani"
    if used == "table":
        try:
            df = _read_table(path)
            if _first_column(df, _CANDIDATES["voltage_V"]) is None and reader == "auto":
                raise ValueError("no voltage column")
        except Exception as exc:  # noqa: BLE001 -- fall back to navani, saying why
            if reader == "table":
                raise
            notes.append(f"Read as a plain table failed ({exc}); trying navani.")
            df, used = _read_navani(path), "navani"
    else:
        df = _read_navani(path)
    df = df.reset_index(drop=True)
    settings = {"mass_mg": mass_mg, "area_cm2": area_cm2, "formula": formula,
                "molar_mass": molar_mass, "x0": x0, "charge_removes_ion": charge_removes_ion,
                "current_unit": current_unit, "time_unit": time_unit,
                "capacity_from": capacity_from}
    df, more = derive(df, mass_mg=mass_mg, area_cm2=area_cm2, formula=formula,
                      molar_mass=molar_mass, x0=x0, charge_removes_ion=charge_removes_ion,
                      current_unit=current_unit, time_unit=time_unit,
                      capacity_from=capacity_from, rest_fraction=rest_fraction)
    return {"data": df, "source": path, "reader": used, "settings": settings,
            "notes": notes + more}


# ============================================================
# Derived columns
# ============================================================

def molar_mass_from_formula(formula: str) -> float:
    """g/mol of a chemical formula, e.g. ``"LiNi0.5Mn1.5O4"`` or ``"Na(Mn0.5Fe0.5)O2"``.
    Uses the package's own periodic table: nothing extra to install."""
    import re
    from . import periodic_table as pt
    text = str(formula).strip().replace(" ", "")
    if not text:
        raise ValueError("The formula is empty.")
    token = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)|(\()|(\))(\d*\.?\d*)")
    stack: list[dict] = [{}]
    pos = 0
    while pos < len(text):
        m = token.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"Could not read the formula {formula!r} at {text[pos:pos + 6]!r}.")
        pos = m.end()
        if m.group(1):
            n = float(m.group(2)) if m.group(2) else 1.0
            stack[-1][m.group(1)] = stack[-1].get(m.group(1), 0.0) + n
        elif m.group(3):
            stack.append({})
        else:
            if len(stack) < 2:
                raise ValueError(f"Unbalanced brackets in {formula!r}.")
            n = float(m.group(5)) if m.group(5) else 1.0
            inner = stack.pop()
            for k, v in inner.items():
                stack[-1][k] = stack[-1].get(k, 0.0) + v * n
    if len(stack) != 1:
        raise ValueError(f"Unbalanced brackets in {formula!r}.")
    total = 0.0
    for symbol, count in stack[0].items():
        element = pt._BY_SYMBOL.get(symbol)
        if element is None:
            raise ValueError(f"Unknown element {symbol!r} in the formula {formula!r}.")
        total += count * float(element.mass)
    return total


# ============================================================
# Derived columns
# ============================================================

_CURRENT_TO_MA = {"A": 1e3, "mA": 1.0, "uA": 1e-3, "µA": 1e-3}
_TIME_TO_S = {"s": 1.0, "min": 60.0, "h": 3600.0}


def derive(df: pd.DataFrame, mass_mg=None, area_cm2=None, formula=None, molar_mass=None,
           x0=None, charge_removes_ion: bool = True, current_unit: str = "mA",
           time_unit: str = "s", capacity_from: str = "integrated",
           rest_fraction: float = 1e-4) -> tuple[pd.DataFrame, list[str]]:
    """Add the standard columns to a copy of ``df`` (the file's own columns stay).

    ``capacity_from``: ``"integrated"`` (∫ I dt within each half cycle, the same for every
    instrument) or ``"file"`` (the file's own ``Capacity`` column, where it has one).
    ``rest_fraction``: a current smaller than this fraction of the largest is rest."""
    if current_unit not in _CURRENT_TO_MA:
        raise ValueError(f"current_unit must be one of {sorted(_CURRENT_TO_MA)}.")
    if time_unit not in _TIME_TO_S:
        raise ValueError(f"time_unit must be one of {sorted(_TIME_TO_S)}.")
    if capacity_from not in ("integrated", "file"):
        raise ValueError("capacity_from must be 'integrated' or 'file'.")
    out = df.copy()
    notes: list[str] = []
    tcol = _first_column(out, _CANDIDATES["time_s"])
    vcol = _first_column(out, _CANDIDATES["voltage_V"])
    icol = _first_column(out, _CANDIDATES["current_mA"])
    if tcol is not None:
        out["time_s"] = pd.to_numeric(out[tcol], errors="coerce") * _TIME_TO_S[time_unit]
        out["time_h"] = out["time_s"] / 3600.0
    else:
        notes.append("No time column found: choose another axis, or rename the time column.")
    if vcol is not None:
        out["voltage_V"] = pd.to_numeric(out[vcol], errors="coerce")
    else:
        notes.append("No voltage column found.")
    if icol is not None:
        out["current_mA"] = pd.to_numeric(out[icol], errors="coerce") * _CURRENT_TO_MA[current_unit]
    if "time_s" in out and "current_mA" in out:
        t = out["time_s"].to_numpy(float)
        i = out["current_mA"].to_numpy(float)
        biggest = np.nanmax(np.abs(i)) if i.size else 0.0
        threshold = rest_fraction * biggest
        sign = np.where(np.abs(i) <= threshold, 0, np.sign(i)).astype(int)
        state = np.where(sign > 0, "charge", np.where(sign < 0, "discharge", "rest"))
        # A new half cycle starts when the sign of the (non-rest) current changes; rest points
        # belong to the half cycle before them.
        nonrest = sign != 0
        change = np.zeros(len(sign), dtype=float)
        if nonrest.any():
            signs = sign[nonrest]
            flips = np.concatenate(([0], (signs[1:] != signs[:-1]).astype(int)))
            change[nonrest] = np.cumsum(flips)
        labelled = pd.Series(np.where(nonrest, change, np.nan))
        half = labelled.ffill().fillna(0).to_numpy(int)
        out["state"] = state
        out["half_cycle"] = half
        out["cycle"] = half // 2 + 1
        dt = np.diff(t, prepend=t[0])
        step = 0.5 * (i + np.concatenate(([i[0]], i[:-1]))) * dt / 3600.0
        step = np.nan_to_num(step)
        net = np.cumsum(step)
        capacity = pd.Series(np.abs(step)).groupby(half).cumsum().to_numpy(float)
        if capacity_from == "file":
            file_cap = _first_column(out, ("Capacity", "Capacity(mAh)", "Capacity (mAh)",
                                           "Q charge/discharge/mA.h", "Q-Qo/mA.h"))
            if file_cap is None:
                notes.append("The file has no capacity column; the capacity was integrated "
                             "from the current instead.")
            else:
                capacity = pd.to_numeric(out[file_cap], errors="coerce").to_numpy(float)
        out["capacity_mAh"] = capacity
        out["_net_charge_mAh"] = net
        if mass_mg:
            out["capacity_spec_mAh_g"] = capacity / (float(mass_mg) / 1000.0)
        if area_cm2:
            out["capacity_area_mAh_cm2"] = capacity / float(area_cm2)
        mm = molar_mass
        if mm is None and formula:
            mm = molar_mass_from_formula(formula)
        if mass_mg and mm and x0 is not None:
            s = 1.0 if charge_removes_ion else -1.0
            q_spec = out["_net_charge_mAh"] / (float(mass_mg) / 1000.0)
            out["x"] = float(x0) - s * q_spec * float(mm) / FARADAY_MAH_PER_MOL
        elif x0 is not None or mm or formula:
            missing = [name for name, ok in (("the active mass", bool(mass_mg)),
                                             ("the molar mass or formula", bool(mm)),
                                             ("the starting composition x₀", x0 is not None))
                       if not ok]
            if missing:
                notes.append("Composition needs " + ", ".join(missing) + ".")
        out = out.drop(columns=["_net_charge_mAh"])
    return out, notes


def plottable_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric column of the data: the standard ones first, then the file's own."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    standard = [c for c in AXIS_LABELS if c in numeric]
    return standard + [c for c in numeric if c not in standard]


def axis_label(column: str) -> str:
    return AXIS_LABELS.get(column, str(column))


# ============================================================
# Selecting
# ============================================================

def select_cycles(df: pd.DataFrame, cycles=None, half: str = "both") -> pd.DataFrame:
    """The rows of the chosen cycle(s) (an int, a list, or ``None`` for all) and half cycle
    (``"charge"``, ``"discharge"`` or ``"both"``; rest points between are kept with
    ``"both"`` and dropped otherwise)."""
    if half not in ("charge", "discharge", "both"):
        raise ValueError("half must be 'charge', 'discharge' or 'both'.")
    if "cycle" not in df:
        raise ValueError("The data have no cycle column (they need a current and a time).")
    mask = np.ones(len(df), dtype=bool)
    if cycles is not None:
        wanted = [cycles] if isinstance(cycles, (int, np.integer)) else list(cycles)
        available = sorted(set(int(c) for c in df["cycle"].unique()))
        missing = [c for c in wanted if c not in available]
        if missing:
            raise ValueError(f"Cycle {missing[0]} does not exist: the data have cycles "
                             f"{available[0]} to {available[-1]}.")
        mask &= df["cycle"].isin(wanted).to_numpy()
    if half != "both":
        mask &= (df["state"] == half).to_numpy()
    return df[mask]


# ============================================================
# Ex situ: one value → one point
# ============================================================

def ex_situ_point(df: pd.DataFrame, value: float, quantity: str = "voltage_V", cycle: int = 1,
                  half: str = "charge", choose: str = "first", near: float | None = None,
                  near_quantity: str = "capacity_mAh") -> dict:
    """Where on the curve a sample sits, given one number (usually its voltage).

    The same voltage is reached on every cycle, and on both the charge and the discharge,
    so the **cycle and the half cycle must be said**. Within that one half cycle a value can
    still be crossed more than once (a plateau that wobbles, a relaxation): ``choose`` says
    which crossing is meant, ``"first"``, ``"last"``, or ``"nearest"`` to ``near`` (a value of
    ``near_quantity``, such as the capacity the sample was stopped at). Every crossing is
    returned in ``"crossings"`` so a doubtful choice can be seen.

    The point is linearly interpolated between the two samples either side. Returns
    ``{"row": {column: value}, "index", "crossings", "n_crossings", "cycle", "half",
    "notes"}``."""
    if choose not in ("first", "last", "nearest"):
        raise ValueError("choose must be 'first', 'last' or 'nearest'.")
    if quantity not in df.columns:
        raise ValueError(f"The data have no column {quantity!r}.")
    if choose == "nearest" and near is None:
        raise ValueError("'nearest' needs near=, the value to be nearest to.")
    segment = select_cycles(df, cycle, half)
    if len(segment) < 2:
        raise ValueError(f"Cycle {cycle} has no {half} half cycle with data.")
    y = segment[quantity].to_numpy(float)
    below = y - float(value)
    a, b = below[:-1], below[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        hit = np.isfinite(a) & np.isfinite(b) & ((a == 0) | (a * b < 0))
        fraction = np.where(a == 0, 0.0, a / (a - b))
    crossings = [(int(k), float(fraction[k])) for k in np.flatnonzero(hit)]
    if below[-1] == 0:
        crossings.append((len(y) - 1, 0.0))
    if not crossings:
        raise ValueError(
            f"{quantity} = {value:g} is not reached in the {half} of cycle {cycle}: it runs "
            f"from {np.nanmin(y):.4g} to {np.nanmax(y):.4g}.")
    numeric = segment.select_dtypes(include="number")
    rows = []
    for k, frac in crossings:
        k1 = min(k + 1, len(segment) - 1)
        lo, hi = numeric.iloc[k], numeric.iloc[k1]
        row = (lo + frac * (hi - lo)).to_dict()
        row[quantity] = float(value)
        rows.append((int(segment.index[k]), frac, row))
    if choose == "first":
        pick = 0
    elif choose == "last":
        pick = len(rows) - 1
    else:
        if near_quantity not in df.columns:
            raise ValueError(f"The data have no column {near_quantity!r}.")
        pick = int(np.argmin([abs(r[2][near_quantity] - near) for r in rows]))
    notes = []
    if len(rows) > 1:
        notes.append(f"{quantity} = {value:g} is crossed {len(rows)} times in the {half} of cycle "
                     f"{cycle}; the {choose} was taken. Check 'crossings' if that is not the "
                     "one meant.")
    index, frac, row = rows[pick]
    return {"row": row, "index": index + frac, "crossings": [r[2] for r in rows],
            "n_crossings": len(rows), "cycle": int(cycle), "half": half, "notes": notes}


def ex_situ_points(df: pd.DataFrame, samples, quantity: str = "voltage_V") -> list[dict]:
    """:func:`ex_situ_point` for several samples. ``samples``: a list of dicts with
    ``"value"`` and optionally ``"cycle"``, ``"half"``, ``"choose"``, ``"near"``, ``"label"``.
    A sample that cannot be placed gives ``{"label", "error"}`` instead of stopping the rest."""
    results = []
    for n, sample in enumerate(samples):
        sample = dict(sample)
        label = sample.pop("label", f"sample {n + 1}")
        try:
            point = ex_situ_point(df, quantity=quantity, **sample)
            point["label"] = label
            results.append(point)
        except ValueError as exc:
            results.append({"label": label, "error": str(exc)})
    return results


# ============================================================
# Operando: spectra ↔ time
# ============================================================

def operando_times(n: int, start_s: float = 0.0, per_spectrum_s: float | None = None,
                   durations_s=None, gaps_s=0.0, times_s=None):
    """The start and end time (s, on the cycler's clock) of each of ``n`` spectra.

    * ``per_spectrum_s``: each takes this long, one after another from ``start_s``
      (``gaps_s``: dead time between, a number or a list);
    * ``durations_s``: a list of durations instead;
    * ``times_s``: the start time of each, given directly (the end is the next start, the
      last as long as the one before).

    ``start_s`` is when the first spectrum began **on the cycler's time axis**: the two
    instruments' clocks must be related by hand, since neither records the other's."""
    n = int(n)
    if times_s is not None:
        starts = np.asarray(times_s, dtype=float)
        if starts.size != n:
            raise ValueError(f"{starts.size} start times for {n} spectra.")
        ends = np.empty(n)
        ends[:-1] = starts[1:]
        ends[-1] = starts[-1] + (starts[-1] - starts[-2] if n > 1 else 0.0)
        return starts, ends
    if durations_s is not None:
        durations = np.asarray(durations_s, dtype=float)
        if durations.size != n:
            raise ValueError(f"{durations.size} durations for {n} spectra.")
    elif per_spectrum_s is not None:
        durations = np.full(n, float(per_spectrum_s))
    else:
        raise ValueError("Give per_spectrum_s, durations_s or times_s.")
    gaps = np.broadcast_to(np.asarray(gaps_s, dtype=float), (n,)).copy()
    gaps[-1] = 0.0
    steps = durations + gaps
    starts = float(start_s) + np.concatenate(([0.0], np.cumsum(steps)[:-1]))
    return starts, starts + durations


def at_spectra(df: pd.DataFrame, starts_s, ends_s, columns=None) -> pd.DataFrame:
    """The cell's state for each spectrum: the time-weighted **mean over its acquisition
    window** of every column in ``columns`` (default voltage, current), and the value **at its
    end** of the capacity and composition columns, which are cumulative. A window outside the
    recorded data gives NaN (and is counted in ``attrs["outside"]``)."""
    if "time_s" not in df:
        raise ValueError("The data have no time column.")
    starts, ends = np.asarray(starts_s, float), np.asarray(ends_s, float)
    t = df["time_s"].to_numpy(float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    columns = list(columns) if columns is not None else [
        c for c in ("voltage_V", "current_mA") if c in df]
    end_columns = [c for c in ("capacity_mAh", "capacity_spec_mAh_g", "capacity_area_mAh_cm2",
                               "x", "cycle", "half_cycle") if c in df]
    out = {"start_s": starts, "end_s": ends, "mid_s": 0.5 * (starts + ends)}
    outside = (starts < t[0]) | (ends > t[-1])
    for c in columns:
        y = df[c].to_numpy(float)[order]
        means = np.full(len(starts), np.nan)
        for k, (a, b) in enumerate(zip(starts, ends)):
            if outside[k] or b <= a:
                continue
            grid = np.concatenate(([a], t[(t > a) & (t < b)], [b]))
            values = np.interp(grid, t, y)
            means[k] = _trapezoid(values, grid) / (b - a)
        out[c] = means
    for c in end_columns:
        y = df[c].to_numpy(float)[order]
        if c in ("cycle", "half_cycle"):
            values = y[np.clip(np.searchsorted(t, ends, side="right") - 1, 0, len(t) - 1)]
        else:
            values = np.interp(ends, t, y)
        values = np.where(outside, np.nan, values)
        out[f"{c}_end"] = values
    table = pd.DataFrame(out)
    table.attrs["outside"] = int(outside.sum())
    return table


# ============================================================
# Spectra of a series
# ============================================================

def spectrum_differences(matrix, mode: str = "successive") -> np.ndarray:
    """Differences between the spectra of an operando series, one row per spectrum.

    ``matrix``: rows are spectra on a common axis. ``"successive"``: row *i* minus row
    *i* − 1 (the first row is zero, so the result lines up with the spectra);
    ``"from_first"``: each minus the first; ``"from_last"``: each minus the last."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] < 2:
        raise ValueError("A series of at least two spectra, as a 2D array, is needed.")
    if mode == "successive":
        d = np.zeros_like(m)
        d[1:] = m[1:] - m[:-1]
        return d
    if mode == "from_first":
        return m - m[0]
    if mode == "from_last":
        return m - m[-1]
    raise ValueError("mode must be 'successive', 'from_first' or 'from_last'.")


def read_value_table(path) -> pd.DataFrame:
    """A table of values to put beside the cell data (fitted shifts, areas and so on): a
    ``.csv``/``.tsv``/``.txt`` with a header row. Columns are returned as they are."""
    df = _read_table(os.fspath(path))
    if df.shape[1] < 2:
        raise ValueError("The table needs at least two columns: an x (time, voltage, spectrum "
                         "number …) and a value.")
    return df
