#!/usr/bin/env python3
"""
filter_bf_k.py

Filter a text file that contains Brainfuck programs grouped like

    ###
    <program …>
    -> 255 254
    ###

Retain **at most K programs** for every distinct output sequence.
The first K occurrences in file order are preserved; the rest are
discarded.

Usage
-----
    python filter_bf_k.py <input_file> <output_file> <K>

    <input_file>  : source file with Brainfuck hits
    <output_file> : destination file
    <K>           : positive integer, maximum programs to keep per
                    distinct output (K = 1 reproduces earlier behaviour)

Example
-------
    python filter_bf_k.py bf_hits_nonempty.txt bf_hits_k3.txt 3
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional


OUTPUT_RE = re.compile(r"^\s*->\s*(.+?)\s*$")


def split_blocks(lines: List[str]) -> List[List[str]]:
    """
    Split *lines* into blocks delimited by lines beginning with '###'.
    The delimiter line is kept as part of its block.
    """
    blocks: List[List[str]] = []
    current: List[str] = []

    for line in lines:
        if line.startswith("###"):
            if current:
                blocks.append(current)
                current = []
        current.append(line)
    if current:
        blocks.append(current)

    return blocks


def extract_output(block: List[str]) -> Optional[Tuple[int, ...]]:
    """
    Return the numeric output sequence found in '-> …' line inside *block*.
    If absent, return None.  The sequence is normalised to a tuple of ints
    so it can be used as a dict key.
    """
    for line in block:
        m = OUTPUT_RE.match(line)
        if m:
            # allow non‑numeric tokens but normal numeric case is expected
            tokens = m.group(1).split()
            try:
                return tuple(int(tok) for tok in tokens)
            except ValueError:
                return tuple(tokens)  # fall back to raw strings
    return None


def filter_blocks(src: Path, dst: Path, k: int) -> None:
    lines = src.read_text(encoding="utf‑8", errors="replace").splitlines(keepends=True)

    kept_blocks: List[List[str]] = []
    seen_count: Dict[Tuple[int, ...], int] = {}

    for block in split_blocks(lines):
        key = extract_output(block)

        if key is None:
            # blocks without explicit output are always retained
            kept_blocks.append(block)
            continue

        if seen_count.get(key, 0) < k:
            kept_blocks.append(block)
            seen_count[key] = seen_count.get(key, 0) + 1
        # else: skip block (already stored k times)

    with dst.open("w", encoding="utf‑8") as fh:
        for block in kept_blocks:
            fh.writelines(block)


def main() -> None:
    if len(sys.argv) != 4:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} <input_file> <output_file> <K>", file=sys.stderr)
        sys.exit(1)

    src_path = Path(sys.argv[1])
    dst_path = Path(sys.argv[2])

    try:
        k_val = int(sys.argv[3])
        if k_val <= 0:
            raise ValueError
    except ValueError:
        print("K must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    filter_blocks(src_path, dst_path, k_val)


if __name__ == "__main__":
    main()

