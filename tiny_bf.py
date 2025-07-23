"""
tiny_bf_transformer_file.py
Loads training pairs from a txt file formatted as:

###
BF code (one or more lines)
-> n1 n2 n3 ...
###

The int list is the ASCII‑code output (zeros suppressed).
"""

import os, re, sys, random, subprocess, tempfile, math
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor


from pathlib import Path
from typing import List, Tuple

import torch
torch.backends.cuda.enable_flash_sdp(True)
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
from torch.cuda.amp import autocast, GradScaler

# ------------------------------------------------------------------ vocab
PAD, BOS, EOS = 0, 1, 2
SRC_OFFSET, SRC_VOCAB = 2, 258            # 0 pad, 1 bos, 2‑257 = 1‑255
BF_TOKENS = "><+-.,[]"
BF_TO_ID = {c: i + 3 for i, c in enumerate(BF_TOKENS)}
ID_TO_BF = {v: k for k, v in BF_TO_ID.items()}
TGT_VOCAB = len(BF_TO_ID) + 3             # 0‑13

# ------------------------------------------------------------------ data

import re
arrow_re = re.compile(r'^->(?:\s+\d+)+\s*$')   # e.g. "-> 255 1 3"


# ---- constants -------------------------------------------------------------
MAX_LEN = 512          # hard limit for encoder and decoder sequences
# ---------------------------------------------------------------------------

def load_pairs(path: Path) -> List[Tuple[List[int], str]]:
    """Parse the file and keep only examples that fit MAX_LEN after tokenisation."""
    pairs, buf = [], []
    for raw in path.read_text(encoding="utf8").splitlines():
        line = raw.strip()
        if not line or line.startswith("###"):
            continue
        if arrow_re.match(line):                      # delimiter
            nums = [int(x) for x in line[2:].split()]
            prog = "\n".join(buf).strip()
            buf.clear()

            # --- length test ------------------------------------------------
            if (len(nums) + 2) <= MAX_LEN:            # +2 for BOS/EOS in src
                tgt_len = 2 + sum(c in BF_TOKENS for c in prog)  # BOS/EOS in tgt
                if tgt_len <= MAX_LEN:
                    pairs.append((nums, prog))
            # ----------------------------------------------------------------
        else:
            buf.append(line)

    if buf:
        raise ValueError("File ended before a delimiter ('-> ...') line.")
    return pairs

# ───────────────────────── data: unconditional corpus ──────────────────────
def load_bf_programs(path: Path, max_len: int = MAX_LEN) -> list[torch.Tensor]:
    """
    File format: each Brainfuck program is separated by a blank line.
    Lines beginning with '#' are ignored.
    Only programs whose tokenised length ≤ MAX_LEN are kept.

    Example
    -------
        +++++[>++++++++<-]>+      # prints 'A'
                                    ← blank line
        +[-->-[>>+>-----<<]<--<---] # prints '0'
    """
    progs, buf = [], []
    for raw in path.read_text(encoding="utf8").splitlines():
        ln = raw.strip()
        if not ln:                                  # blank → flush buffer
            if buf:
                code = "\n".join(buf)
                buf.clear()
                if 2 + sum(c in BF_TOKENS for c in code) <= max_len:
                    progs.append(bf_ids(code))
            continue
        if ln.startswith("#"):
            continue
        buf.append(ln)
    if buf:                                         # file ended without blank
        code = "\n".join(buf)
        if 2 + sum(c in BF_TOKENS for c in code) <= max_len:
            progs.append(bf_ids(code))
    return progs



def ascii_ids(nums):      # src
    return torch.tensor([BOS] + [n + SRC_OFFSET for n in nums] + [EOS])


def bf_ids(code):         # tgt
    return torch.tensor([BOS] + [BF_TO_ID[c] for c in code if c in BF_TO_ID] + [EOS])


class PairDS(Dataset):
    def __init__(self, pairs):
        self.src = [ascii_ids(a) for a, _ in pairs]
        self.tgt = [bf_ids(b) for _, b in pairs]

    def __len__(self): return len(self.src)
    def __getitem__(self, i): return self.src[i], self.tgt[i]


def collate(batch):
    s, t = zip(*batch)
    s = pad_sequence(s, True, PAD)
    t = pad_sequence(t, True, PAD)
    return s, t[:, :-1], t[:, 1:]

# ------------------------------------------------------------------ model (better)
import math, torch, torch.nn as nn
from torch import Tensor

# ------------------------------------------------ multi‑layer model
import math, torch, torch.nn as nn
from torch import Tensor

# --- rotary position --------------------------------------------------------
class _Rotary(nn.Module):
    def __init__(self, d: int, max_len: int = 2048):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, d, 2) / d))
        t = torch.arange(max_len)
        f = torch.einsum("i,j->ij", t, inv)          # (L, d/2)
        self.register_buffer("sin", f.sin(), False)
        self.register_buffer("cos", f.cos(), False)

    def forward(self, x: Tensor) -> Tensor:          # (B,T,D)
        sin, cos = self.sin[: x.size(1)], self.cos[: x.size(1)]
        x1, x2   = x[..., 0::2], x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), -1).flatten(-2)

# ---------------------------------------------------------------------------
class DeepBF(nn.Module):
    """
    Separate #layers for encoder/decoder.
    Defaults: 12 enc + 12 dec, d=512.
    """
    def __init__(
        self,
        d: int = 512,
        h: int = 8,
        enc_layers: int = 12,
        dec_layers: int = 12,
        ff: int = 4 * 512,
        L: int = 2048,
        dropout: float = 0.1,
        tie_embed: bool = True,
    ):
        super().__init__()
        self.se = nn.Embedding(SRC_VOCAB, d, padding_idx=PAD)
        self.te = nn.Embedding(TGT_VOCAB, d, padding_idx=PAD)
        self.rope = _Rotary(d, L)
        enc_block = nn.TransformerEncoderLayer(
            d, h, ff, dropout, batch_first=True, activation="gelu")
        dec_block = nn.TransformerDecoderLayer(
            d, h, ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_block, enc_layers)
        self.decoder = nn.TransformerDecoder(dec_block, dec_layers)
        self.head = nn.Linear(d, TGT_VOCAB, bias=False)
        if tie_embed:
            self.head.weight = self.te.weight
        self.drop = nn.Dropout(dropout)

    def _rope_embed(self, emb: Tensor) -> Tensor:
        return self.rope(self.drop(emb))

    def forward(
        self,
        src: Tensor,
        tgt_inp: Tensor,
        smask: Tensor | None = None,
        tmask: Tensor | None = None,
    ) -> Tensor:
        src = self._rope_embed(self.se(src))
        tgt = self._rope_embed(self.te(tgt_inp))

        T = tgt.size(1)
        causal = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), 1)

        mem = self.encoder(src, src_key_padding_mask=smask)
        dec = self.decoder(
            tgt,
            mem,
            tgt_mask=causal,
            tgt_key_padding_mask=tmask,
            memory_key_padding_mask=smask,
        )
        return self.head(dec)


# ---------------------------------------------------------------- utils
def ids_to_bf(seq):
    return "".join(ID_TO_BF.get(i, "") for i in seq
                   if i not in (PAD, BOS, EOS))


        

from torch.cuda.amp import autocast, GradScaler
import math
# ────────────────────────── datasets & collation ────────────────────────────
class CodeLMDS(Dataset):
    """Pure decoder‑side LM: returns (y_in , y_out)."""
    def __init__(self, code_seqs: List[torch.Tensor]):
        self.seq = code_seqs
    def __len__(self):  return len(self.seq)
    def __getitem__(self, i):
        s = self.seq[i]
        return s[:-1], s[1:]                    # shift‑left

def collate_lm(batch):
    y_in , y_out = zip(*batch)
    return (pad_sequence(y_in , True, PAD),
            pad_sequence(y_out, True, PAD))

# ─────────────────────────── helper losses ─────────────────────────────────
def _seq_nll(logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Negative‑log‑likelihood of each whole sequence in batch."""
    ce = nn.functional.cross_entropy(
        logits.view(-1, TGT_VOCAB), tgt.view(-1),
        ignore_index=PAD, reduction='none')
    tok = ce.view(tgt.size()).sum(1)            # (B,)
    return tok                                  # nll = −log p(seq)

def set_loss(logits: torch.Tensor,
             tgt   : torch.Tensor,
             src   : torch.Tensor) -> torch.Tensor:
    """
    Exact log‑sum‑exp over all programs that share the same ASCII list
      L = −log( 1/m Σ p(prog_j|ascii) ).
    """
    nll  = _seq_nll(logits, tgt)                # (B,)
    logp = -nll                                 # log p
    keys = [tuple(s.tolist()) for s in src]     # hashable
    buckets: dict[list[int]] = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)

    losses = []
    for idx in buckets.values():
        m  = len(idx)
        lp = logp[idx] - math.log(m)
        losses.append(-torch.logsumexp(lp, dim=0))
    return torch.stack(losses).mean()

# ─────────────────────────── training loops ────────────────────────────────
@torch.no_grad()
def _make_dummy_src(bs: int, device):
    return torch.full((bs, 2), BOS, dtype=torch.long, device=device).\
               scatter_(1, torch.tensor([[1]], device=device), EOS)

# ───────────────────── training utilities (re‑worked) ──────────────────────
def save_ckpt(obj: dict, ckpt_dir: str, tag: str):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(obj, Path(ckpt_dir) / f"{tag}.pt")


def train_lm(model, dl, opt, device, epochs=12,  # ← longer schedule
             save_every=2, ckpt_dir="lm_ckpts", max_norm=1.0):
    crit, scaler = nn.CrossEntropyLoss(ignore_index=PAD), GradScaler()
    for ep in range(1, epochs + 1):
        model.train(); tot, tok = 0., 0
        for y_in, y_out in dl:
            y_in, y_out = y_in.to(device), y_out.to(device)
            src = _make_dummy_src(y_in.size(0), device)
            with autocast():
                lg   = model(src, y_in, src == PAD, y_in == PAD)
                loss = crit(lg.view(-1, TGT_VOCAB), y_out.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * y_out.size(0); tok += y_out.size(0)
        print(f"[LM]  epoch {ep}: {tot/tok:.4f}")
        if save_every and ep % save_every == 0:
            save_ckpt({"epoch": ep,
                       "model_state_dict": model.state_dict(),
                       "optimizer_state_dict": opt.state_dict(),
                       "scaler_state_dict": scaler.state_dict()},
                      ckpt_dir, f"lm_ep{ep:03d}")

def freeze_backbone(model: DeepBF, freeze: bool):
    """(Un)freeze embeddings + first half of decoder layers."""
    for p in model.se.parameters(): p.requires_grad = not freeze
    for p in model.te.parameters(): p.requires_grad = not freeze
    k = len(model.decoder.layers) // 2             # first half
    for i, layer in enumerate(model.decoder.layers):
        if i < k:
            for p in layer.parameters():
                p.requires_grad = not freeze


def train_cond(model, dl, opt, device, epochs=40,
               save_every=5, ckpt_dir="cond_ckpts",
               freeze_epochs=4, accum=2, max_norm=1.0):
    scaler = GradScaler(); step, warm_steps = 0, 0
    steps_per_epoch = math.ceil(len(dl) / accum)
    tot_steps = epochs * steps_per_epoch
    warm_steps = int(0.10 * tot_steps)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1,
                                           total_iters=warm_steps),
         torch.optim.lr_scheduler.CosineAnnealingLR(
             opt, T_max=tot_steps - warm_steps, eta_min=1e-5)],
        [warm_steps])

    freeze_backbone(model, True)                   # start frozen

    for ep in range(1, epochs + 1):
        if ep == freeze_epochs + 1:
            print("⇢ unfreezing embeddings + early decoder layers")
            freeze_backbone(model, False)

        model.train(); tot, tok = 0., 0
        for i, (src, y_in, y_out) in enumerate(dl, 1):
            src, y_in, y_out = src.to(device), y_in.to(device), y_out.to(device)
            with autocast():
                lg = model(src, y_in, src == PAD, y_in == PAD)
                loss = set_loss(lg, y_out, src) / accum
            scaler.scale(loss).backward()
            if i % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                sched.step(); step += 1
            tot += loss.item() * y_out.size(0); tok += y_out.size(0)
        print(f"[COND] epoch {ep}: {tot/tok:.4f} | lr {sched.get_last_lr()[0]:.2e}")

        if save_every and ep % save_every == 0:
            save_ckpt({"epoch": ep,
                       "model_state_dict": model.state_dict(),
                       "optimizer_state_dict": opt.state_dict(),
                       "scaler_state_dict": scaler.state_dict()},
                      ckpt_dir, f"cond_ep{ep:03d}")

# ──────────────────────────── main ───────────────────────────────
if __name__ == "__main__":
    pairs_file   = Path(sys.argv[1])               # conditional pairs
    corpus_file  = Path(sys.argv[2])               # pure‑LM BF corpus

    pairs = load_pairs(pairs_file)
    print(f"→ conditional pairs: {len(pairs):,}")

    lm_code = load_bf_programs(corpus_file)
    print(f"→ unconditional programs: {len(lm_code):,}")

    # datasets / loaders -----------------------------------------------------
    ds_cond = PairDS(pairs)                        # (src , tgt)
    ds_lm   = CodeLMDS(lm_code)                    # purely decoder‑side

    dl_cond = DataLoader(ds_cond, 64, shuffle=True, collate_fn=collate)
    dl_lm   = DataLoader(ds_lm  , 64, shuffle=True, collate_fn=collate_lm)

    # model & optimiser ------------------------------------------------------
    dev   = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepBF(d=256, h=8, enc_layers=12, dec_layers=12,
                   ff=1024, L=MAX_LEN, tie_embed=True, dropout=0.10).to(dev)
    opt    = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-2)

    # 1) long unconditional warm‑up + checkpoints
    train_lm(model, dl_lm, opt, dev,
             epochs=12, save_every=2, ckpt_dir="lm_ckpts")

    # 2) conditional fine‑tuning with early freeze
    train_cond(model, dl_cond, opt, dev,
               epochs=40, save_every=5, ckpt_dir="cond_ckpts",
               freeze_epochs=4)





