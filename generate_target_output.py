#!/usr/bin/env python3
"""
bf_eval.py  ―  Brainf*** synthesiser evaluation (v2.2)

v2.2 (2025‑07‑20)
─────────────────
* Added live progress bar (tqdm) that shows running stats.
* No other behavioural changes.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TextIO, Tuple  

import torch
from tiny_bf import (
    BOS, EOS, PAD,
    DeepBF,
    SRC_OFFSET,
    ids_to_bf,
    load_pairs,
)
from search import search                # ← beam‑search now lives here

from tqdm.auto import tqdm             

# ── silence selected warnings ──────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*")

# ╔════════════════════════════════ configuration ═════════════════════════╗
@dataclass(frozen=True)
class SearchCfg:
    beam: int = 1
    alpha: float = 0.6
    max_len: int = 128                   # decoder cap


@dataclass(frozen=True)
class RunCfg:
    device: str
    batch: int
    max_len: int                         # caps *target* length
    search: SearchCfg
    timeout: float = 2.0
    seed: int = 0xBEEF
    no_exec: bool = False


@dataclass
class RunStats:                          # mutable counters
    generated: int = 0
    syntax_ok: int = 0
    timeout: int = 0
    failed: int = 0
    skipped: int = 0
    dropped: int = 0


# ╔════════════════════════════ utilities ═════════════════════════════════╗
def encode_targets(
    targets: List[List[int]],
    device: str,
    pos_limit: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    kept, seqs = [], []
    for idx, nums in enumerate(targets):
        seq = [BOS] + [n + SRC_OFFSET for n in nums] + [EOS]
        if len(seq) <= pos_limit:
            kept.append(idx)
            seqs.append(seq)

    if not seqs:                         # nothing fits
        return (
            torch.empty(0, 0, dtype=torch.long, device=device),
            torch.empty(0, 0, dtype=torch.bool, device=device),
            kept,
        )

    max_len = max(map(len, seqs))
    src = torch.full((len(seqs), max_len), PAD, dtype=torch.long, device=device)
    for row, seq in zip(src, seqs):
        row[: len(seq)] = torch.tensor(seq, device=device)

    return src, src == PAD, kept


def run_bf(code: str, *, cfg: RunCfg, stats: RunStats) -> bytes:
    """Run Brainf*** (unless --no-exec)."""
    stats.generated += 1
    if cfg.no_exec:
        stats.skipped += 1
        return b""

    interp = shutil.which("beef") or shutil.which("bf")
    if interp is None:
        raise FileNotFoundError("Need ‘beef’ or ‘bf’ interpreter")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".bf") as tmp:
        tmp.write(code)
        path = tmp.name

    try:
        proc = subprocess.run(
            [interp, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=cfg.timeout,
            check=False,
        )
        if proc.returncode != 0:
            stats.failed += 1
            raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
        stats.syntax_ok += 1
        return proc.stdout
    except subprocess.TimeoutExpired:
        stats.timeout += 1
        raise
    finally:
        Path(path).unlink(missing_ok=True)


def decode_batch(
    model: DeepBF,
    targets: List[List[int]],
    run_cfg: RunCfg,
    pos_limit: int,
) -> List[Tuple[int, str]]:
    src, smask, kept = encode_targets(targets, run_cfg.device, pos_limit)
    programmes: List[Tuple[int, str]] = []

    for i_in_batch, orig_idx in enumerate(kept):
        seq = search(model,
                     src[i_in_batch : i_in_batch + 1],
                     smask[i_in_batch : i_in_batch + 1],
                     cfg=run_cfg.search)
        programmes.append((orig_idx, ids_to_bf(seq)))
    return programmes


# ╔════════════════════════════ evaluation loop ════════════════════════════╗
@torch.no_grad()
def eval_loop(
    model: DeepBF,
    pairs: list[Tuple[List[int], str]],
    run_cfg: RunCfg,
    stats: RunStats,
    log_fp: TextIO,
) -> Tuple[int, int]:
    """
    Returns (syntactically_ok, perfect_matches).
    A tqdm progress bar keeps the user informed.
    """
    rng = random.Random(run_cfg.seed)
    rng.shuffle(pairs)

    synt_ok = valid = 0
    src_limit = getattr(model, "L", 128)
    total = len(pairs)

    with tqdm(total=total, ncols=100, unit="ex") as bar:
        for b0 in range(0, total, run_cfg.batch):
            batch_pairs = pairs[b0 : b0 + run_cfg.batch]

            # filter examples that overflow limits
            filtered = [
                (idx, out)
                for idx, (out, _) in enumerate(batch_pairs)
                if len(out) + 2 <= src_limit and len(out) <= run_cfg.max_len
            ]
            if not filtered:
                stats.dropped += len(batch_pairs)
                bar.update(len(batch_pairs))
                bar.set_postfix(
                    ok=synt_ok, perfect=valid,
                    drop=stats.dropped, fail=stats.failed,
                    tmo=stats.timeout,
                )
                continue

            tgt_list = [out for _, out in filtered]
            idx_map = [idx for idx, _ in filtered]

            coded = decode_batch(model, tgt_list, run_cfg, pos_limit=src_limit)
            prog_per_target = {idx_map[i]: prog for i, prog in coded}

            for local_idx, (expected, _) in enumerate(batch_pairs):
                if local_idx not in prog_per_target:
                    stats.dropped += 1
                    continue

                prog = prog_per_target[local_idx]
                

                try:
                    out = run_bf(prog, cfg=run_cfg, stats=stats)
                    synt_ok += 1
                    

                    
                    if list(out) == expected:
                        valid += 1
                except Exception:
                    out = "Exception"
                    pass     # stats updated inside run_bf
                    
                
                if not run_cfg.no_exec:
                        rec = {
                            "bf": prog,
                            "target": expected,
                            "output": list(out),
                        }
                        json.dump(rec, log_fp)
                        log_fp.write("\n")


            bar.update(len(batch_pairs))
            bar.set_postfix(
                ok=synt_ok, perfect=valid,
                drop=stats.dropped, fail=stats.failed,
                tmo=stats.timeout,
            )

    return synt_ok, valid


# ╔════════════════════════════── CLI / main ─══════════════════════════════╗
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("pairs", type=Path, help="file with (output, program) pairs")
    p.add_argument("ckpt", type=Path, help="checkpoint *.pt")
    p.add_argument("--beam", type=int, default=1, help="beam width (1=greedy)")
    p.add_argument("--alpha", type=float, default=0.6, help="length‑penalty α")
    p.add_argument("--max_len", type=int, default=128,
                   help="max decoder tokens **and** max target length")
    p.add_argument("--batch", type=int, default=256,
                   help="batch size of targets")
    p.add_argument("--no-exec", action="store_true",
                   help="skip running the generated code")
    p.add_argument("-n", "--max_outputs", type=int, default=512,
                   help="evaluate at most this many DISTINCT outputs "
                        "(0 = all)")
    p.add_argument("--log", type=Path, default='generated.json',            
                   help="write every syntactically‑correct programme as "
                        "JSONL to this file")
    return p.parse_args(argv)

#╔════════════════════ pick distinct outputs ═════════════════════════════╗
def pick_unique_outputs(
    pairs: List[Tuple[List[int], str]],
    max_outputs: int,
    seed: int,
) -> List[Tuple[List[int], str]]:
    """
    Keep at most *max_outputs* distinct output sequences.
    For each distinct output choose ONE programme at random; then if the
    number of distinct outputs still exceeds the cap, sample down to the cap.
    """
    by_out: Dict[Tuple[int, ...], List[Tuple[List[int], str]]] = {}
    for nums, prog in pairs:
        by_out.setdefault(tuple(nums), []).append((nums, prog))

    rng = random.Random(seed)
    unique_pairs = [rng.choice(v) for v in by_out.values()]

    if 0 < max_outputs < len(unique_pairs):
        unique_pairs = rng.sample(unique_pairs, max_outputs)

    rng.shuffle(unique_pairs)             # final random order
    return unique_pairs

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    random.seed(0xBEEF)
    torch.manual_seed(0xBEEF)

    device = "cuda"# if torch.cuda.is_available() else "cpu"
    run_cfg = RunCfg(
        device=device,
        batch=args.batch,
        max_len=args.max_len,
        search=SearchCfg(
            beam=args.beam,
            alpha=args.alpha,
            max_len=args.max_len,
        ),
        no_exec=args.no_exec,
    )
    stats = RunStats()

    # load model & data -----------------------------------------------------
    model = DeepBF(d=256, h=8, enc_layers=12, dec_layers=12,
                   ff=1024, L=128).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    pairs = load_pairs(args.pairs)
    pairs = pick_unique_outputs(
        pairs, max_outputs=args.max_outputs, seed=run_cfg.seed
    )
    log_fp = args.log.open("w", encoding="utf-8")
    # evaluate --------------------------------------------------------------
    t0 = time.time()
    synt_ok, valid = eval_loop(model, pairs, run_cfg, stats, log_fp=log_fp)
    dt = time.time() - t0

    # report ----------------------------------------------------------------
    total = len(pairs) - stats.dropped
    print(f"\n✓ syntactically correct : {synt_ok}/{total}")
    print(f"✓ perfect matches       : {valid}/{total}")
    print(f"dropped (too long)      : {stats.dropped}")
    print(f"elapsed                 : {dt:.1f}s")
    print("\nExecution stats:", json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()

