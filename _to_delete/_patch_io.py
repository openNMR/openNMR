import re, pathlib
p = pathlib.Path("src/opennmr/io/simpson.py")
s = p.read_text()

def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:60], s.count(old))
    s = s.replace(old, new)

# ---- A: sniff_format
a = s.index("def sniff_format")
b = s.index("def parse_simp_text")
s = s[:a] + '''def sniff_format(path: str) -> str:
    """``"simp"``, ``"dat"`` (three columns), ``"shape"`` (two columns) or
    ``"tcl"`` (a SIMPSON input script), by content. Raises ValueError for JSON
    (genuine CSDM) and anything else."""
    first = ""
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                first = line.strip()
                break
    if first.startswith("{"):
        raise ValueError(
            f"{path} looks like JSON (genuine CSDM format), not SIMPSON text. Use "
            "opennmr.notebook.CSDMimport() for that.")
    if first == "SIMP":
        return "simp"
    if str(path).lower().endswith(".tcl") or first.startswith("#") or first.split()[:1] in (
            ["spinsys"], ["par"], ["proc"]):
        return "tcl"
    columns = len(first.split())
    if columns == 3:
        return "dat"
    if columns == 2:
        return "shape"
    raise ValueError(f"Could not identify {path} as a SIMPSON file (first line has "
                     f"{columns} columns; expected the SIMP header, or 3 columns for -xreim).")


''' + s[b:]

# ---- B: NELEM
sub('''    if len(real) != np_header:
        raise ValueError(f"{path}: header declares NP={np_header} but {len(real)} data lines "
                         "were read.")
    return {"NP": np_header, "SW": float(header["SW"]), "TYPE": header.get("TYPE", "").upper() or None,
            "real": np.asarray(real, dtype=float), "imag": np.asarray(imag, dtype=float),
            "path": path, "header": header}''',
'''    nelem = max(int(float(header.get("NELEM", 1) or 1)), 1)
    if len(real) != np_header * nelem:
        raise ValueError(f"{path}: header declares NP={np_header}"
                         + (f" and NELEM={nelem}" if nelem > 1 else "")
                         + f" ({np_header * nelem} data lines) but {len(real)} were read.")
    return {"NP": np_header, "SW": float(header["SW"]), "NELEM": nelem,
            "TYPE": header.get("TYPE", "").upper() or None,
            "real": np.asarray(real, dtype=float), "imag": np.asarray(imag, dtype=float),
            "path": path, "header": header}''')
sub('''"""A ``SIMP`` file: ``NP``, ``SW``, ``TYPE``, ``real``, ``imag``, ``path``
    (plus any other ``KEY=value`` header lines under ``header``)."""''',
'''"""A ``SIMP`` file: ``NP``, ``SW``, ``NELEM``, ``TYPE``, ``real``, ``imag``,
    ``path`` (plus any other ``KEY=value`` header lines under ``header``).

    With ``NELEM=n`` the file holds ``n`` FIDs (or spectra) of ``NP`` points
    one after the other; ``real``/``imag`` then have ``n*NP`` entries and
    :func:`load_raw` picks one with ``element=``."""''')

# ---- C: xreim classification
sub('''_TIME_UNITS = {"s": 1.0''', '''def classify_columns(x) -> str:
    """``"xreim"`` when the first column is one evenly spaced axis (strictly
    monotonic, and no point further than a quarter of a step from a straight
    line -- SIMPSON writes x to about six figures, so it is not exact), else
    ``"table"``: x repeating while another column steps, a parameter scan."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return "xreim"
    steps = np.diff(x)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        return "table"
    index = np.arange(x.size)
    slope, intercept = np.polyfit(index, x, 1)
    if slope == 0:
        return "table"
    return "xreim" if np.max(np.abs(x - (intercept + slope * index))) <= 0.25 * abs(slope) else "table"


def uniform_axis(x):
    """``(axis, step)``: the straight-line fit through ``x`` -- the axis the
    rounded values in the file stand for."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return x.copy(), 0.0
    index = np.arange(x.size)
    slope, intercept = np.polyfit(index, x, 1)
    return intercept + slope * index, float(slope)


_TIME_UNITS = {"s": 1.0''')

# ---- D: load_raw
a = s.index("def load_raw")
b = s.index("# ---------------------------------------------------------------------------\n# Observe frequency")
s = s[:a] + '''def n_elements(path: str) -> int:
    """``NELEM`` of a SIMP file (1 when absent or for any other file)."""
    try:
        with open(path, "r", errors="ignore") as handle:
            if handle.readline().strip() != "SIMP":
                return 1
            for line in handle:
                line = line.strip()
                if line == "DATA":
                    break
                if line.upper().startswith("NELEM="):
                    return max(int(float(line.split("=", 1)[1])), 1)
    except (OSError, ValueError):
        pass
    return 1


def describe(path: str) -> dict:
    """What a file is, without reading its data: ``{"role", "detail"}``.

    ``role`` is ``"data"`` (a spectrum or FID: SIMP text, or x/re/im columns),
    ``"table"`` (x, y, z columns of a parameter scan), ``"shape"`` (a pulse
    shape), ``"script"`` (a .tcl input) or ``"unknown"``."""
    try:
        fmt = sniff_format(path)
    except (ValueError, OSError) as exc:
        return {"role": "unknown", "detail": str(exc)}
    if fmt == "simp":
        return {"role": "data", "detail": "SIMP text", "elements": n_elements(path)}
    if fmt == "shape":
        return {"role": "shape", "detail": "pulse shape (amplitude, phase)"}
    if fmt == "tcl":
        return {"role": "script", "detail": "SIMPSON input script"}
    try:
        data = np.loadtxt(path, ndmin=2)
    except ValueError as exc:
        return {"role": "unknown", "detail": str(exc)}
    if data.shape[1] != 3:
        return {"role": "unknown", "detail": "expected 3 columns"}
    if classify_columns(data[:, 0]) == "table":
        return {"role": "table", "detail": "x, y, z table"}
    return {"role": "data", "detail": "x, real, imaginary columns"}


def load_raw(path: str, kind: str | None = None, time_unit: str = "s",
             sw: float | None = None, element: int = 0) -> dict:
    """Read ``path`` into ``{"kind": "FID"|"SPE", "NP", "SW", "real", "imag",
    "x_Hz" (SPE), "format", "NELEM", "element"}``.

    ``kind`` : ``None`` takes the file's own ``TYPE`` (SIMP text). A headerless
    x/re/im ``.dat`` has none: it is called a spectrum when its x column runs
    through zero (frequencies about the carrier) and a FID when it starts at or
    after zero (times); give ``"fid"`` or ``"spe"`` to say otherwise. ``sw`` : a
    ``.dat`` FID's spectral width, if its time column cannot give it.
    ``element`` : which of an ``NELEM`` file's FIDs/spectra (0-based)."""
    fmt = sniff_format(path)
    if fmt == "shape":
        raise ValueError(f"{path} looks like a SIMPSON pulse-shape file (2 columns, no header), "
                         "not a spectrum or FID. Use SIMPSONread_shape().")
    if fmt == "tcl":
        raise ValueError(f"{path} is a SIMPSON input script (.tcl), not data. Its nucleus and "
                         "field can be read with read_tcl_settings().")
    wanted = kind.upper() if kind else None
    if wanted not in (None, "FID", "SPE"):
        raise ValueError("kind must be 'fid' or 'spe' (or None to read it from the file).")
    if fmt == "simp":
        raw = parse_simp_text(path)
        found = raw["TYPE"]
        if found not in ("FID", "SPE"):
            raise ValueError(f"{path}: TYPE is {found!r}; expected FID or SPE.")
        if wanted and wanted != found:
            raise ValueError(f"{path} holds a {found}, not a {wanted}.")
        nelem = raw["NELEM"]
        element = int(element)
        if not 0 <= element < nelem:
            raise ValueError(f"{path} holds {nelem} element(s); element={element} is out of "
                             f"range (0 to {nelem - 1}).")
        if nelem > 1:
            block = slice(element * raw["NP"], (element + 1) * raw["NP"])
            raw["real"], raw["imag"] = raw["real"][block], raw["imag"][block]
        raw["element"] = element
        raw["kind"] = found
        raw["format"] = "SIMP"
        if found == "SPE":
            raw["x_Hz"] = -raw["SW"] / 2 + np.arange(raw["NP"]) * (raw["SW"] / raw["NP"])
        return raw
    raw = parse_xreim(path)
    if classify_columns(raw["x"]) == "table":
        raise ValueError(
            f"{path} has three columns, but the first is not one evenly spaced axis (it repeats "
            "or jumps): it is a table of x, y, z values from a parameter scan, not x, real, "
            "imaginary. Read it with read_table() / SIMPSONread_table().")
    raw.update({"NELEM": 1, "element": 0})
    guessed = wanted is None
    if guessed:
        wanted = "SPE" if (raw["x"].min() < 0 < raw["x"].max()) else "FID"
    raw["kind"] = wanted
    raw["kind_guessed"] = guessed
    raw["format"] = "xreim"
    axis, step = uniform_axis(raw["x"])
    if wanted == "SPE":
        raw["x_Hz"] = axis
        raw["SW"] = float(abs(step) * raw["NP"]) if raw["NP"] > 1 else float(sw or 0.0)
    else:
        scale = _TIME_UNITS.get(time_unit)
        if scale is None:
            raise ValueError(f"time_unit must be one of {sorted(_TIME_UNITS)}.")
        if sw is None:
            if raw["NP"] < 2:
                raise ValueError(f"{path}: one point has no dwell time; give sw= (Hz).")
            dt = abs(step) * scale
            if dt <= 0:
                raise ValueError(f"{path}: the time column does not increase.")
            sw = 1.0 / dt
        raw["SW"] = float(sw)
    return raw


def read_table(path: str) -> dict:
    """A headerless three-column parameter-scan table (``fsave`` of a scan, or
    ``-xyz``-style output): ``{"x", "y", "z"}`` columns, and when x and y form a
    complete grid, ``"xs"``, ``"ys"`` and ``"Z"`` (``Z[j, i]`` at ``ys[j]``,
    ``xs[i]``)."""
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns (x, y, z).")
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    out = {"path": path, "x": x, "y": y, "z": z, "grid": False}
    xs, ys = np.unique(x), np.unique(y)
    if xs.size * ys.size == x.size and xs.size > 1 and ys.size > 1:
        grid = np.full((ys.size, xs.size), np.nan)
        grid[np.searchsorted(ys, y), np.searchsorted(xs, x)] = z
        if not np.isnan(grid).any():
            out.update({"grid": True, "xs": xs, "ys": ys, "Z": grid})
    return out


_NUMBER_NAMES = ("spin_rate", "proton_frequency", "gamma_angles", "sw", "np")


def _eval_number(expression: str, names: dict):
    """A number from ``8192*6`` / ``spin_rate*gamma_angles``, or ``None``."""
    import ast
    try:
        tree = ast.parse(expression.strip().replace("$par(", "").replace(")", "") if "$" in expression
                         else expression.strip(), mode="eval")
    except SyntaxError:
        return None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return float(names[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult,
                                                               ast.Div, ast.Pow)):
            left, right = walk(node.left), walk(node.right)
            return {ast.Add: left + right, ast.Sub: left - right, ast.Mult: left * right,
                    ast.Div: left / right, ast.Pow: left ** right}[type(node.op)]
        raise ValueError("not a plain number")

    try:
        return walk(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def read_tcl_settings(path: str) -> dict:
    """What a SIMPSON input script says about the experiment, for pre-filling
    the nucleus and field boxes: ``{"nucleus", "channels", "proton_mhz",
    "spin_rate", "sw", "np"}`` (each ``None`` if absent or not a plain number).

    Only the ``spinsys`` and ``par`` blocks are looked at, and only arithmetic
    on numbers and earlier ``par`` entries is evaluated -- the script is never
    run."""
    text = open(path, "r", errors="ignore").read()
    text = "\\n".join(line.split("#", 1)[0] for line in text.splitlines())
    out: dict = {"nucleus": None, "channels": [], "proton_mhz": None, "spin_rate": None,
                 "sw": None, "np": None}
    match = re.search(r"\\bchannels\\s+([^\\n]+)", text)
    if match:
        out["channels"] = match.group(1).split()
        out["nucleus"] = out["channels"][0] if out["channels"] else None
    block = re.search(r"\\bpar\\s*\\{(.*?)\\n\\s*\\}", text, flags=re.S)
    values: dict = {}
    if block:
        for line in block.group(1).splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2 or parts[0] == "variable":
                continue
            number = _eval_number(parts[1].replace("$par(", "").replace(")", ""), values) \\
                if "$" in parts[1] else _eval_number(parts[1], values)
            if number is not None:
                values[parts[0]] = number
    if "proton_frequency" in values:
        out["proton_mhz"] = values["proton_frequency"] / 1e6
    for key in ("spin_rate", "sw", "np"):
        out[key] = values.get(key)
    return out


''' + s[b:]

# ---- E: read_1d
sub('''            phase_penalty: str = "magnitude", mass: float = 1.0, **_ignored) -> NMRSpectrum:''',
    '''            phase_penalty: str = "magnitude", mass: float = 1.0, element: int = 0,
            **_ignored) -> NMRSpectrum:''')
sub('''    raw = load_raw(path, kind=kind, time_unit=time_unit, sw=sw)
    observe, b0''', '''    raw = load_raw(path, kind=kind, time_unit=time_unit, sw=sw, element=element)
    title = os.path.basename(path)
    if raw["NELEM"] > 1:
        title += f" [{raw['element'] + 1}/{raw['NELEM']}]"
    observe, b0''')
sub('''"kind": raw["kind"], "format": raw["format"], "conjugate": bool(conjugate)}''',
    '''"kind": raw["kind"], "format": raw["format"], "conjugate": bool(conjugate),
             "element": raw["element"], "n_elements": raw["NELEM"],
             "kind_guessed": bool(raw.get("kind_guessed", False))}''')
assert s.count('"title": os.path.basename(path)') == 2
s = s.replace('"title": os.path.basename(path)', '"title": title')

# ---- G: peek
sub('''"simpson_type": header.get("TYPE")}''', '''"simpson_type": header.get("TYPE"),
            "n_elements": int(float(header.get("NELEM", 1) or 1))}''')
p.write_text(s)
print("ok")
