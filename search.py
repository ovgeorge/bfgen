from __future__ import annotations

import torch

from tiny_bf import (          # project‑local helper
    BOS, EOS, PAD,
    DeepBF,
    SRC_OFFSET,
    ids_to_bf,
    load_pairs,
)

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple


# ╔════════════════════════════════ configuration ═════════════════════════╗
@dataclass(frozen=True)
class SearchCfg:
    beam: int = 1
    alpha: float = 0.6          # length‑penalty α
    max_len: int = 128          # decoder limit  (tokens generated)



# ╔══════════════════════════ traced search ══════════════════════════════╗
@torch.no_grad()
def search(
    model: DeepBF,
    src: torch.Tensor,
    smask: torch.Tensor,
    cfg: SearchCfg,
    *,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    beam_width, alpha, max_len = cfg.beam, cfg.alpha, cfg.max_len
    dev = src.device

    beams = torch.full((1, 1), BOS, dtype=torch.long, device=dev)
    scores = torch.zeros(1, device=dev)
    alive = torch.ones(1, dtype=torch.bool, device=dev)

    def _bf_strings(mat: torch.Tensor) -> List[str]:
        return [
            ids_to_bf([t for t in row[1:].tolist() if t not in (PAD, EOS)])
            for row in mat
        ]

    for step in range(1, max_len + 1):
        B = beams.size(0)
        logits = model(
            src.expand(B, -1),
            beams,
            smask.expand(B, -1),
            beams == PAD,
        )[:, -1]
        logp = torch.log_softmax(logits, -1)

        cand = scores.unsqueeze(1) + logp
        cand[~alive] = -1e9

        flat = cand.view(-1)
        topk = flat.topk(min(beam_width, flat.size(0)))
        parent = topk.indices // logits.size(1)
        token = topk.indices % logits.size(1)

        beams = torch.cat([beams[parent], token.unsqueeze(1)], 1)
        scores = topk.values
        alive = alive[parent] & token.ne(EOS)

        if trace is not None:
            trace.append(
                {
                    "step": step,
                    "beams": _bf_strings(beams.cpu()),
                    "scores": scores.cpu().tolist(),
                }
            )

        if not alive.any():
            break

    lengths = beams.ne(PAD).sum(1).float()
    lp = ((5 + lengths) / 6).pow(alpha)
    best = (scores / lp).argmax().item()

    seq = beams[best, 1:].tolist()
    return seq[: seq.index(EOS)] if EOS in seq else seq

