"""Edit the built-in Cmd/Ctrl+E fact list (a maintainer's tool, run by hand).

    python packaging/edit_facts.py --export facts.txt   # facts.dat -> editable text
    python packaging/edit_facts.py --import facts.txt   # edited text -> facts.dat

``src/opennmr/gui/resources/facts.dat`` is zlib-compressed UTF-8, one fact
per line, so it is not plain text in an install. Export it, edit the text
file in any editor, import it back, and commit facts.dat (``/facts.txt`` is
git-ignored). On import, blank lines and lines starting with # are dropped.
The list is curated by hand: nothing here downloads or regenerates it.
"""

from __future__ import annotations

import argparse
import os
import sys
import zlib

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "opennmr", "gui", "resources", "facts.dat")


def write(facts: list[str], path: str = OUT) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(zlib.compress("\n".join(facts).encode("utf-8"), 9))


def read(path: str = OUT) -> list[str]:
    with open(path, "rb") as handle:
        text = zlib.decompress(handle.read()).decode("utf-8")
    return [line for line in text.splitlines() if line.strip()]


def export_text(text_path: str, path: str = OUT) -> int:
    facts = read(path)
    with open(text_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(facts) + "\n")
    return len(facts)


def import_text(text_path: str, path: str = OUT) -> int:
    with open(text_path, encoding="utf-8-sig") as handle:
        lines = [line.strip() for line in handle.read().splitlines()]
    facts = [line for line in lines if line and not line.startswith("#")]
    write(facts, path)
    return len(facts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", metavar="TEXT_FILE",
                       help="write facts.dat out as an editable text file, one fact per line")
    group.add_argument("--import", dest="import_", metavar="TEXT_FILE",
                       help="replace facts.dat with this text file, one fact per line")
    args = parser.parse_args(argv)
    if args.export:
        print(f"Wrote {export_text(args.export)} facts to {args.export}")
    else:
        print(f"Wrote {import_text(args.import_)} facts to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
