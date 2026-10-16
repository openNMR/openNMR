"""Build openNMR's built-in fact list for Cmd/Ctrl+E (a maintainer's tool,
run by hand; not part of the app).

    python packaging/fetch_facts.py            # fetch, parse, write
    python packaging/fetch_facts.py --dry-run  # fetch and parse, show samples

Reads the two lists the Cmd/Ctrl+E box draws from (explicit user request,
2026-10-16) and writes them, one fact per line, zlib-compressed, to
``src/opennmr/gui/resources/facts.dat`` -- not plain text in an install, and
packaged with the other GUI resources (pyproject package-data; PyInstaller
adds the whole resources folder). Commit that file after checking the
samples this prints.

Sources:
- "No Such Thing As A Fish" master fact list (Google Sheet, every tab): the
  "Fact" column of each tab, read from the workbook's .xlsx export.
- "1,339 QI Facts To Make Your Jaw Drop" (sources.qi.com/1339): the sentence
  after each "Page N, Position M" label (citations skipped).

Standard library only, so it runs in any Python.
"""

from __future__ import annotations

import argparse
import html
import io
import os
import re
import sys
import urllib.request
import zipfile
import zlib
from xml.etree import ElementTree as ET

SHEET_ID = "1f8ITcZz8DfrTDVwY5f1gOphq4wjEu3VJvVqZ1SSjSWU"
SHEET_XLSX = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"
QI_URL = "https://sources.qi.com/1339"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "opennmr", "gui", "resources", "facts.dat")
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_UA = {"User-Agent": "Mozilla/5.0 (openNMR fact list builder)"}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=120) as r:
        return r.read()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(text))).strip()


# ---- the Google Sheet (.xlsx, read with the standard library) ---------------------

def _col(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def _cell_text(cell, shared) -> str:
    kind = cell.get("t")
    if kind == "s":
        v = cell.find("m:v", _NS)
        return shared[int(v.text)] if v is not None and v.text else ""
    if kind == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{{{_NS['m']}}}t"))
    v = cell.find("m:v", _NS)
    return v.text if v is not None and v.text else ""


def facts_from_xlsx(data: bytes) -> list[str]:
    """The "Fact" column of every sheet of a workbook."""
    z = zipfile.ZipFile(io.BytesIO(data))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    out = []
    sheets = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
    for name in sheets:
        root = ET.fromstring(z.read(name))
        fact_col = None
        for row in root.iter(f"{{{_NS['m']}}}row"):
            cells = {_col(c.get("r")): _cell_text(c, shared) for c in row.findall("m:c", _NS)
                     if c.get("r")}
            if fact_col is None:
                for col, text in cells.items():
                    if clean(text).lower() == "fact":
                        fact_col = col
                continue
            text = clean(cells.get(fact_col, ""))
            if text:
                out.append(text)
    return out


# ---- sources.qi.com ------------------------------------------------------------

_LABEL = re.compile(r"Page\s+\d+\s*,\s*Position\s+\d+\s*:?", re.IGNORECASE)


def _html_lines(page: str) -> list[str]:
    page = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h\d|tr|td|section|article|blockquote)>", "\n", page)
    page = re.sub(r"<[^>]+>", " ", page)
    return [clean(line) for line in page.split("\n") if clean(line)]


def _is_citation(line: str) -> bool:
    low = line.lower()
    return ("http" in low or "www." in low or "sources.qi.com" in low
            or low.startswith(("source", "[source"))) or bool(_LABEL.match(line))


def facts_from_qi(page: str) -> list[str]:
    """The claim after each "Page N, Position M" label."""
    lines = _html_lines(page)
    out = []
    for i, line in enumerate(lines):
        m = _LABEL.search(line)
        if not m:
            continue
        rest = clean(line[m.end():])
        candidates = ([rest] if rest else []) + lines[i + 1:i + 4]
        for text in candidates:
            if _LABEL.search(text):
                break
            if not _is_citation(text) and len(text) >= 15:
                out.append(text)
                break
    return out


# ---- putting it together ---------------------------------------------------------

def tidy(facts) -> list[str]:
    seen, out = set(), []
    for f in facts:
        f = clean(f).strip('"“” ')
        if len(f) < 15 or f.lower() in seen:
            continue
        seen.add(f.lower())
        out.append(f)
    return out


def write(facts: list[str], path: str = OUT) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(zlib.compress("\n".join(facts).encode("utf-8"), 9))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="parse and show, write nothing")
    args = parser.parse_args(argv)
    found = {}
    for name, url, parse in (("No Such Thing As A Fish sheet", SHEET_XLSX, facts_from_xlsx),
                             ("QI 1,339 facts", QI_URL, lambda b: facts_from_qi(
                                 b.decode("utf-8", "replace")))):
        try:
            found[name] = parse(fetch(url))
        except Exception as exc:  # noqa: BLE001 -- report and carry on with the other
            print(f"{name}: FAILED ({exc})", file=sys.stderr)
            found[name] = []
        print(f"{name}: {len(found[name])} facts")
        for sample in found[name][:3]:
            print(f"   e.g. {sample[:150]}")
    facts = tidy([f for v in found.values() for f in v])
    print(f"{len(facts)} facts after removing duplicates and fragments.")
    if not facts:
        print("Nothing to write.", file=sys.stderr)
        return 1
    if not args.dry_run:
        write(facts)
        print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
