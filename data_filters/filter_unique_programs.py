#!/usr/bin/env python3
"""
filter_unique_programs.py
-------------------------

Read a text file that stores Brain‑Fuck programmes in the usual format

    ###
    <one or more lines of code>
    -> n1 n2 ...               # optionally, any number list
    ###

and write a new file that contains *at most one* block for every **unique
programme text**.  Two blocks are considered identical iff the *canonicalised*
source code (see canonical() below) matches byte‑for‑byte.

The output‑number lines are ignored for the purpose of deduplication, so it is
perfectly fine if several unique programmes all print the same sequence of
bytes – they are *different* to us and every one of them will be retained.
"""

from __future__ import annotations
import sys, re
from pathlib import Path
from typing import List, Set, Tuple

DELIM = "###"
ARROW_RE = re.compile(r'^->')            # first arrow line ends the programme


# ---------------------------------------------------------------------------

def canonical(code_lines: List[str]) -> str:
    """
    Return a canonical representation of *code_lines* suitable for hashing.

    The default strategy – which is usually sufficient – keeps exactly the raw
    BF tokens and discards everything else (whitespace, comments, carriage
    returns…).  If you want stricter equivalence (e.g. include comments, or
    normalise space but keep newlines), adjust this function only.
    """
    BF_TOKENS = set("><+-.,[]")
    return ''.join(c for line in code_lines for c in line if c in BF_TOKENS)


# ---------------------------------------------------------------------------

def iter_blocks(path: Path):
    """
    Yield blocks (“### … ###”) as lists of raw lines *including newlines*.
    The opening line that starts with '###' remains inside the block; the
    trailing delimiter belongs to the *next* block (mirrors the earlier filter
    behaviour).
    """
    block = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(DELIM) and block:
                yield block
                block = []
            block.append(line)
    if block:
        yield block


def extract_program(block: List[str]) -> List[str]:
    """
    Return the programme part of *block* as a list of raw lines (without the
    leading '###', without the first '-> …' line, without anything after it).
    """
    prog_lines = []
    start = 1 if block and block[0].startswith(DELIM) else 0
    for line in block[start:]:
        if ARROW_RE.match(line):
            break
        prog_lines.append(line)
    return prog_lines


# ---------------------------------------------------------------------------

def filter_unique_programs(src: Path, dst: Path) -> None:
    """
    Write *dst* so that it contains only the first block that introduced each
    canonicalised programme text.
    """
    seen: Set[str] = set()
    kept_blocks: List[List[str]] = []

    for block in iter_blocks(src):
        key = canonical(extract_program(block))
        if key in seen:
            continue
        seen.add(key)
        kept_blocks.append(block)

    with dst.open("w", encoding="utf‑8") as fh:
        for b in kept_blocks:
            fh.writelines(b)


# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv
    if len(argv) != 3:
        prog = Path(argv[0]).name
        print(f"Usage: {prog} <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    src, dst = map(Path, argv[1:])
    filter_unique_programs(src, dst)


if __name__ == "__main__":
    main()

