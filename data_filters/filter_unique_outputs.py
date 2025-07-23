from __future__ import annotations
import re, sys
from pathlib import Path
from typing import List, Tuple, Set

ARROW_RE = re.compile(r'^->(?:\s+\d+)+\s*$')     # copy from training code
DELIM = "###"


def _split_blocks(lines: List[str]) -> List[List[str]]:
    """Split on lines that **start** with '###' (same rule as before)."""
    blocks, buf = [], []
    for line in lines:
        if line.startswith(DELIM) and buf:
            blocks.append(buf)
            buf = []
        buf.append(line)
    if buf:
        blocks.append(buf)
    return blocks


def _first_clean_output(block: List[str]) -> Tuple[int, ...] | None:
    """
    Return the *first* arrow line that is acceptable to the training loader;
    else None.  Trailing comments, NULs, etc. are ignored.
    """
    for line in block:
        if ARROW_RE.match(line):
            nums = tuple(int(tok) for tok in line.split()[1:])
            return nums
    return None


def filter_by_output(src: Path, dst: Path) -> None:
    lines = src.read_text(encoding="utf‑8", errors="replace").splitlines(keepends=True)

    kept: List[List[str]] = []
    seen: Set[Tuple[int, ...]] = set()

    for block in _split_blocks(lines):
        key = _first_clean_output(block)

        if key is None:                 # no valid arrow -> keep verbatim
            kept.append(block)
        elif key not in seen:           # first time we meet this output
            kept.append(block)
            seen.add(key)               # remember it
        # else: duplicate → skip entire block

    with dst.open("w", encoding="utf‑8") as fh:
        for b in kept:
            fh.writelines(b)


def main() -> None:
    if len(sys.argv) != 3:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    src, dst = map(Path, sys.argv[1:])
    filter_by_output(src, dst)


if __name__ == "__main__":
    main()

