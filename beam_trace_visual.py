#!/usr/bin/env python3
"""
beam_trace_vis.py ― DeepBF beam‑search tracer
2025‑07‑20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tiny_bf import BOS, EOS, PAD, SRC_OFFSET, DeepBF, ids_to_bf
from target_eval import search

# ╔══════════════════════════ config & helpers ═══════════════════════════╗
@dataclass(frozen=True)
class SearchCfg:
    beam: int = 4
    alpha: float = 0.6
    max_len: int = 128


def _encode_targets(nums: List[int], dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    seq = [BOS] + [n + SRC_OFFSET for n in nums] + [EOS]
    src = torch.tensor(seq, dtype=torch.long, device=dev).unsqueeze(0)  # [1,T]
    return src, src == PAD

# ╔════════════════════════ beam‑trace wrapper ═══════════════════════════╗
def trace_beam(
    model: DeepBF,
    nums: List[int],
    dev: str,
    *,
    beam: int = 4,
    alpha: float = 0.6,
    steps: int | None = None,
    plot: bool = False,
) -> None:
    if steps is None:
        steps = getattr(model, "L", 128)
    cfg = SearchCfg(beam=beam, alpha=alpha, max_len=steps)

    src, smask = _encode_targets(nums, dev)
    log: List[Dict[str, Any]] = []
    search(model, src, smask, cfg, trace=log)

    for t in log:
        print(f"\nStep {t['step']}")
        for i, (prog, sc) in enumerate(zip(t["beams"], t["scores"])):
            print(f"  beam {i:2d}: {prog:<25} log‑p={sc:8.2f}")

    if plot:
        import matplotlib.pyplot as plt

        for i in range(len(log[0]["beams"])):
            y = [step["scores"][i] for step in log if i < len(step["scores"])]
            x = list(range(1, len(y) + 1))
            plt.plot(x, y, label=f"beam {i}")
        plt.xlabel("decoding step")
        plt.ylabel("cumulative log‑probability")
        plt.title("Beam‑search score trajectories")
        plt.legend()
        plt.tight_layout()
        save_to = "plot.png"
        plt.savefig(save_to, dpi=512, bbox_inches="tight")
        print(f"[beam_trace_vis] plot saved → {save_to}")
        plt.show()


# ╔══════════════════════════ CLI glue ═══════════════════════════════════╗
def _build_model(ckpt: Path, dev: str) -> DeepBF:
    model = DeepBF(d=256, h=8, enc_layers=12, dec_layers=12,
                   ff=1024, L=128).to(dev)
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def _parse_nums(arg: str) -> List[int]:
    try:
        return [int(x) for x in arg.split(",") if x]
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise DeepBF beam search")
    p.add_argument("ckpt", type=Path, help="checkpoint *.pt")
    p.add_argument("target", type=_parse_nums,
                   help="comma‑separated output bytes to condition on")
    p.add_argument("--beam", type=int, default=4, help="beam width")
    p.add_argument("--alpha", type=float, default=0.6, help="length‑penalty α")
    p.add_argument("--steps", type=int, default=30,
                   help="max decoding steps to trace")
    p.add_argument("--plot", action="store_true",
                   help="draw matplotlib chart besides console trace")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(args.ckpt, dev)

    trace_beam(model, args.target, dev,
               beam=args.beam, alpha=args.alpha,
               steps=args.steps, plot=args.plot)


if __name__ == "__main__":
    main()

