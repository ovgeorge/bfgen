#!/usr/bin/env python3
from __future__ import annotations
import re, sys
from pathlib import Path
from collections import defaultdict

ARROW_RE = re.compile(r'^->(?:\s+\d+)+\s*$')      # ← same as in training code


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name}  <corpus.txt>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    occ = defaultdict(list)                       # {nums‑tuple: [lineno, …]}

    with path.open(encoding="utf‑8", errors="replace") as fh:
        for ln, line in enumerate(fh, 1):
            if ARROW_RE.match(line):
                nums = tuple(int(tok) for tok in line.split()[1:])
                occ[nums].append(ln)

    total_pairs   = sum(len(v) for v in occ.values())
    distinct_out  = len(occ)
    redundant     = total_pairs - distinct_out

    print(f"\nScanned {total_pairs:,} arrow lines.")
    print(f"Distinct output sequences : {distinct_out:,}")
    print(f"Redundant (duplicate) pairs: {redundant:,}")

    # ---------------------------------------------------------------------‑
    # show the 20 sequences that appear most often
    # ---------------------------------------------------------------------‑
    dups = [(k, v) for k, v in occ.items() if len(v) > 1]
    if not dups:
        print("\n✔  No duplicates at all — one program per output, as intended.")
        return

    print("\nTop duplicate outputs (up to 20):")
    for i, (nums, lns) in enumerate(sorted(dups, key=lambda kv: -len(kv[1]))[:20], 1):
        head = ", ".join(map(str, nums[:6]))
        if len(nums) > 6:
            head += " …"
        locs = ", ".join(str(n) for n in lns[:5])
        if len(lns) > 5:
            locs += ", …"
        print(f"{i:>2}. occurs {len(lns):>3}× – first numbers [{head}] – lines {locs}")

    print("\n(Use the reported line numbers to inspect those places in the file.)")


if __name__ == "__main__":
    main()

