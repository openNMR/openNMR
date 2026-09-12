"""
Vendor-neutral one-column value lists.

Every vendor labels the slices of a pseudo-2D series with a plain text
file holding one number per line, and every vendor gives it a different
name: Bruker writes ``vdlist``/``vclist``/``difflist`` (and keeps copies
under ``lists/vd/``, ``lists/vc/``), X-Pulse writes
``parameters/G1_valueslist.txt``, Varian arrays live in ``procpar``
alongside a hand-written list as often as not, and a user with a
one-off experiment writes ``delays.txt`` next to the data.

They are all the same file. :func:`read_value_list` reads any of them,
and :func:`find_value_lists` finds the ones sitting with a given
dataset, so the GUI can offer a list *file* rather than one hard-coded
vendor's list name.

What this does **not** do is decide what the numbers mean. A vdlist is
in seconds, a vclist is a loop counter, a valueslist is a relative
gradient fraction -- the same reader returns all three, and the caller
(the Values-source combo's own factor, see
``MainWindow._resolve_independent_values``) is what turns a counter or a
fraction into the quantity being fitted against. Guessing here, from a
filename, is exactly how a delay list ends up silently scaled by
something.
"""

from __future__ import annotations

import os

import numpy as np

#: Bruker's own time-unit suffixes, as written in a ``vdlist``
#: (``"10m"`` = 10 ms). A line with no suffix is a plain number and is
#: returned unchanged -- which is what a ``vclist`` counter, an X-Pulse
#: gradient fraction, and a hand-written b-value all are.
_UNIT_SCALE = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "s": 1.0}

#: Line prefixes skipped as comments. ``#`` and ``;`` are the usual
#: conventions; ``$$`` is JCAMP-DX's own, which shows up in files
#: written alongside X-Pulse data.
_COMMENT_PREFIXES = ("#", ";", "$$", "//")

#: Filenames that are a value list by convention, checked exactly (no
#: extension) inside an experiment folder. Ordered most-specific-first
#: so :func:`find_value_lists` offers the likeliest candidate first.
_KNOWN_LIST_NAMES = ("vdlist", "vclist", "vplist", "difflist", "valuelist",
                     "fq1list", "fq2list", "fq3list")

#: Sub-paths (relative to an experiment folder) also scanned for list
#: files, covering Bruker's ``lists/`` tree and X-Pulse's
#: ``parameters/`` folder.
_LIST_SUBDIRS = (os.path.join("lists", "vd"), os.path.join("lists", "vc"),
                 os.path.join("lists", "vp"), os.path.join("lists", "f1"),
                 "lists", "parameters")

#: Extensions a discovered file may carry and still be offered. A list
#: is text; anything binary would only waste the user's click.
_LIST_EXTENSIONS = ("", ".txt", ".list", ".dat", ".csv")

#: Fewest entries a file must hold to be *discovered* as a value list.
#: One number is a parameter, not a series -- and a pseudo-2D with a
#: single slice is not something anyone fits a curve to. Only applies
#: to discovery; :func:`read_value_list` will happily read a
#: deliberately-chosen one-entry file.
_DISCOVERY_MIN_ENTRIES = 2

#: Most lines any candidate file is read for during *discovery*. A
#: discovery scan parses every file it finds to confirm it really is a
#: one-column list, and a dataset folder can hold a multi-megabyte text
#: file that is not one -- reading the first few lines is enough to
#: tell, and bounds the cost of opening a folder.
_DISCOVERY_MAX_LINES = 64


def has_unit_suffix(token: str) -> bool:
    """Whether ``token`` carries its own unit letter (``"10m"``,
    ``"500u"``) rather than being a bare number.

    A value that names its own unit is an **absolute time** and must be
    taken at its word; a bare number means whatever the caller says it
    means (the GUI's Unit combo). The one thing that must never happen
    is applying both -- see ``MainWindow._resolve_independent_values``.

    ``"1e5"`` is not a suffixed value: the trailing letter has to
    follow a complete number for the rest to parse.
    """
    token = token.strip()
    if len(token) < 2 or token[-1] not in _UNIT_SCALE:
        return False
    try:
        float(token[:-1])
    except ValueError:
        return False
    return True


def parse_value(token: str) -> float:
    """One list entry as a number.

    A Bruker time suffix (``"10m"`` -> ``0.01``) is applied; anything
    else is a plain float. Raises ``ValueError`` on a token that is
    neither, which is what :func:`find_value_lists` uses to tell a
    value list from an arbitrary text file.
    """
    token = token.strip()
    if not token:
        raise ValueError("empty token")
    if token[-1] in _UNIT_SCALE:
        # Guard against "1e5" -- 'e5' is not a suffix, and float()
        # handles the whole token. Only a trailing unit letter
        # following a complete number is a suffix.
        try:
            return float(token[:-1]) * _UNIT_SCALE[token[-1]]
        except ValueError:
            pass
    return float(token)


def read_value_list(path: str, max_lines: int | None = None) -> dict:
    """Read a one-column list of numbers from ``path``.

    Blank lines and comment lines (see :data:`_COMMENT_PREFIXES`) are
    skipped. Each remaining line contributes its **first**
    whitespace/comma-separated token, so a file that carries a second
    column of notes still reads as the list it is; a line whose first
    token is not a number raises ``ValueError`` naming the line, rather
    than being skipped -- a list with an unreadable entry in it has the
    wrong length, and a silently short list is the one failure mode
    that produces a plausible-looking wrong fit.

    ``max_lines`` : stop after this many *data* lines. Used by
    :func:`find_value_lists` to sniff a candidate cheaply; ``None``
    (the default) reads the whole file.

    Returns ``{"name", "path", "raw", "values", "seconds",
    "units_explicit"}``. ``values`` and ``seconds`` are the same array
    -- ``seconds`` is kept because
    :func:`opennmr.processing.relaxation.NMR2D_fit_relaxation` and
    :func:`opennmr.io.bruker.read_vdlist` already speak that key, and
    is a misnomer for any list that is not a delay list (see this
    module's own docstring).

    ``units_explicit`` says whether the file **named its own units** --
    true when any entry carried a suffix, as a Bruker ``vdlist``
    normally does. That is the difference between a list whose numbers
    are already absolute times and one whose numbers mean whatever the
    caller decides, and a caller that applies its own unit to the
    former would scale an already-absolute value a second time.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such value list: {path!r}")

    raw: list[str] = []
    values: list[float] = []
    units_explicit = False
    with open(path, "r", errors="ignore") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith(_COMMENT_PREFIXES):
                continue
            token = line.replace(",", " ").split()[0]
            try:
                values.append(parse_value(token))
            except ValueError as exc:
                raise ValueError(
                    f"{os.path.basename(path)} line {number}: {line!r} is not a number."
                ) from exc
            # Any suffixed entry makes the whole list absolute. A file
            # that mixes "10m" with bare numbers is pathological, and
            # taking the whole list at its word is the reading that
            # cannot silently rescale a value that already had a unit.
            units_explicit = units_explicit or has_unit_suffix(token)
            raw.append(line)
            if max_lines is not None and len(values) >= max_lines:
                break

    if not values:
        raise ValueError(f"{os.path.basename(path)} holds no numbers.")

    array = np.array(values, dtype=float)
    return {"name": os.path.basename(path), "path": path, "raw": raw,
            "values": array, "seconds": array, "units_explicit": units_explicit}


def looks_like_value_list(path: str) -> bool:
    """Whether ``path`` parses as a one-column numeric list -- the test
    :func:`find_value_lists` applies to every candidate it finds, so a
    ``title`` or a stray ``README.txt`` sitting in the data folder is
    never offered as a delay list."""
    try:
        parsed = read_value_list(path, max_lines=_DISCOVERY_MAX_LINES)
    except (OSError, ValueError):
        return False
    return len(parsed["values"]) >= _DISCOVERY_MIN_ENTRIES


def _name_suggests_a_list(name: str) -> bool:
    """Whether a file *named* ``name`` is plausibly a value list.

    Applied only to **extensionless** files, and the reason is a real
    measurement rather than tidiness: a Bruker experiment folder is
    full of extensionless text files that parse perfectly well as
    columns of numbers -- ``gpnam5``, ``scon2``, ``EA``, a shim file,
    a preemphasis table. Offering those alongside the ``difflist`` that
    is actually wanted turns a one-click choice into a guessing game.
    A file with a text extension is offered on its contents alone,
    since that is where a user's own ``delays.txt`` lives.
    """
    stem = os.path.splitext(name)[0].lower()
    return stem in _KNOWN_LIST_NAMES or "list" in stem


def find_value_lists(datapath: str, include_parent: bool = True) -> list[str]:
    """Value-list files sitting with the dataset at ``datapath``.

    Looks in the experiment folder itself, the vendor list sub-folders
    under it (:data:`_LIST_SUBDIRS`), and -- since X-Pulse's
    ``##_valueslist.txt`` convention and a user's own ``delays.txt``
    are as likely to live beside the experiment as inside it -- the
    parent folder, unless ``include_parent`` is False.

    ``datapath`` may be a file (an X-Pulse ``.jdx``, a JEOL ``.jdf``),
    in which case its own folder is the experiment folder. Every
    candidate is parsed before being offered (see
    :func:`looks_like_value_list`), so the returned paths are lists
    that actually read, not names that merely look promising.

    Returned most-likely-first: known vendor list names, then
    ``*valueslist*`` / ``*list*`` names, then anything else, each group
    in sorted order. Deduplicated by real path, so Bruker's copy of a
    ``vdlist`` under ``lists/vd/`` is not offered twice.
    """
    if not datapath:
        return []
    root = datapath if os.path.isdir(datapath) else os.path.dirname(datapath)
    if not os.path.isdir(root):
        return []

    directories = [root]
    directories += [os.path.join(root, sub) for sub in _LIST_SUBDIRS]
    if include_parent:
        parent = os.path.dirname(os.path.normpath(root))
        if parent and parent != root:
            directories.append(parent)

    found: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name.lower())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            extension = os.path.splitext(entry.name)[1].lower()
            if extension not in _LIST_EXTENSIONS:
                continue
            if not extension and not _name_suggests_a_list(entry.name):
                continue
            key = os.path.realpath(entry.path)
            if key in seen:
                continue
            seen.add(key)
            if looks_like_value_list(entry.path):
                found.append(entry.path)

    return sorted(found, key=_discovery_rank)


def _discovery_rank(path: str) -> tuple:
    """Sort key putting the likeliest list first: a known vendor list
    name, then anything whose name contains "list", then the rest --
    alphabetically within each group."""
    name = os.path.basename(path).lower()
    stem = os.path.splitext(name)[0]
    if stem in _KNOWN_LIST_NAMES:
        group = 0
    elif "valueslist" in name or "valuelist" in name:
        group = 1
    elif "list" in name:
        group = 2
    else:
        group = 3
    return (group, name, path)
