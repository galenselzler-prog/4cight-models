# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Idea encoder: turns an idea-bearing utterance into a vector so ideas can
be compared by meaning ("tape the wheels" == "put sticky tape on the tires").

Trained on the same-idea pairs file plus the idea links in the utterance file
(a `repeat` is the same idea as the one it links to, `develops`/`combines` are
related). Cosine-similarity thresholds for same / related / different are
calibrated on held-out pairs and shipped with the model."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from . import labels as L
from .text_encoder import Pooled, load_base, pick_device, reload_encoder

TARGET = {"same": 1.0, "related": 0.5, "different": 0.0}
#: A `new` idea is, by the guide's definition, not the same as any earlier idea in
#: its session, but it may still be related. These pairs get a one-sided loss that
#: only pushes similarity down to the "related" level, never further.
NOT_SAME = "not_same"
MAX_LEN = 64
DIM = 128


class IdeaEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, dim: int = DIM):
        super().__init__()
        self.pooled = Pooled(encoder)
        self.proj = nn.Linear(self.pooled.dim, dim)

    def forward(self, input_ids, attention_mask):
        return nn.functional.normalize(self.proj(self.pooled(input_ids, attention_mask)), dim=-1)


def pairs_from_links(utts: pd.DataFrame, max_not_same: int = 3, seed: int = 0) -> pd.DataFrame:
    """Extra training pairs from idea links inside whole labeled sessions:
    repeat -> same, develops/combines -> related, new vs. earlier new ideas -> not_same."""
    rng = random.Random(seed)
    rows = []
    for sid, g in utts.groupby("session_id", sort=False):
        origin = {r.idea_id: r.text for r in g.itertuples() if r.idea_id}
        earlier_new: list[str] = []
        for r in g.itertuples():
            if r.idea_link == "new":
                for prev in rng.sample(earlier_new, min(max_not_same, len(earlier_new))):
                    rows.append(dict(idea_a_text=prev, idea_b_text=r.text, final_label=NOT_SAME))
                earlier_new.append(r.text)
            if r.idea_link not in ("repeat", "develops", "combines") or not r.linked_ideas:
                continue
            for iid in str(r.linked_ideas).split(";"):
                if iid.strip() in origin:
                    rows.append(dict(idea_a_text=origin[iid.strip()], idea_b_text=r.text,
                                     final_label="same" if r.idea_link == "repeat" else "related"))
    return pd.DataFrame(rows, columns=["idea_a_text", "idea_b_text", "final_label"])


def hash_split(pairs: pd.DataFrame, holdout: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic split by pair_id so the same pair is always held out."""
    h = pairs["pair_id"].map(lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    return pairs[h >= holdout].reset_index(drop=True), pairs[h < holdout].reset_index(drop=True)


@torch.no_grad()
def embed(model: IdeaEncoder, tok, texts: list[str], batch_size: int = 128) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        b = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
        out.append(model(b["input_ids"], b["attention_mask"]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.proj.out_features), dtype=np.float32)


def cosine_pairs(model, tok, df: pd.DataFrame) -> np.ndarray:
    a, b = embed(model, tok, list(df["idea_a_text"])), embed(model, tok, list(df["idea_b_text"]))
    return (a * b).sum(1)


def calibrate(sims: np.ndarray, gold: list[str]) -> tuple[dict, float]:
    """Pick t_related < t_same maximising 3-way macro-F1 on held-out pairs."""
    grid = np.round(np.linspace(-0.2, 0.99, 120), 4)
    best = (-1.0, 0.5, 0.8)
    for i, tr in enumerate(grid):
        for ts in grid[i + 1:]:
            pred = np.where(sims >= ts, "same", np.where(sims >= tr, "related", "different"))
            f = f1_score(gold, pred, average="macro", labels=L.PAIR_LABEL, zero_division=0)
            if f > best[0]:
                best = (f, float(tr), float(ts))
    return {"related": best[1], "same": best[2]}, round(best[0], 4)


def classify(sim: float, thresholds: dict) -> str:
    return "same" if sim >= thresholds["same"] else ("related" if sim >= thresholds["related"] else "different")


def train(pairs: pd.DataFrame, utts_train: pd.DataFrame, base: str, out_dir, epochs: int = 6, lr: float = 2e-5,
          batch_size: int = 32, seed: int = 0, device: str | None = None, log=print,
          utts_val: pd.DataFrame | None = None) -> dict:
    """Held-out pairs (plus same/related links from validation sessions, if
    given) pick the best epoch and calibrate the thresholds."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = pick_device(device)
    tr_pairs, ho_pairs = hash_split(pairs)
    ho_pairs = ho_pairs[["idea_a_text", "idea_b_text", "final_label"]]
    if utts_val is not None and len(utts_val):
        vl = pairs_from_links(utts_val, seed=seed)
        ho_pairs = pd.concat([ho_pairs, vl[vl["final_label"] != NOT_SAME]], ignore_index=True)
    train_df = pd.concat([tr_pairs[["idea_a_text", "idea_b_text", "final_label"]], pairs_from_links(utts_train)],
                         ignore_index=True)
    corpus = list(pairs["idea_a_text"]) + list(pairs["idea_b_text"]) + list(utts_train["text"])
    tok, enc = load_base(base, texts_for_tiny=corpus, seed=seed)
    if base == "tiny" and lr < 1e-3:
        lr = 1e-3
    model = IdeaEncoder(enc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    A, B = list(train_df["idea_a_text"]), list(train_df["idea_b_text"])
    T = torch.tensor([TARGET.get(x, TARGET["related"]) for x in train_df["final_label"]])
    ONE_SIDED = torch.tensor([x == NOT_SAME for x in train_df["final_label"]])
    steps = epochs * math.ceil(len(A) / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps))

    def enc_batch(texts):
        b = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
        return model(b["input_ids"], b["attention_mask"])

    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(A))
        total = 0.0
        for i in range(0, len(A), batch_size):
            idx = order[i:i + batch_size]
            sim = (enc_batch([A[j] for j in idx]) * enc_batch([B[j] for j in idx])).sum(1)
            err = sim - T[idx].to(dev)
            err = torch.where(ONE_SIDED[idx].to(dev), err.clamp(min=0), err)
            loss = (err ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(idx)
        th, f1 = calibrate(cosine_pairs(model, tok, ho_pairs), list(ho_pairs["final_label"]))
        log(f"  idea epoch {ep + 1}/{epochs}: train loss {total / len(A):.4f}, held-out pair macro-F1 {f1:.3f}")
        if f1 > best:
            best, best_state, best_th = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, th
    model.load_state_dict(best_state)
    save(model, tok, out_dir, base, best_th)
    return {"pair_macro_f1": best, "thresholds": best_th, "n_train_pairs": len(A), "n_holdout_pairs": len(ho_pairs)}


def save(model: IdeaEncoder, tok, out_dir, base: str, thresholds: dict):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.pooled.encoder.save_pretrained(out / "encoder")
    tok.save_pretrained(out / "tokenizer")
    torch.save(model.proj.state_dict(), out / "proj.pt")
    (out / "config.json").write_text(json.dumps({"base": base, "dim": model.proj.out_features, "max_len": MAX_LEN,
                                                 "thresholds": thresholds}, indent=2))


def load(folder, device=None):
    folder = Path(folder)
    cfg = json.loads((folder / "config.json").read_text())
    tok, enc = reload_encoder(folder)
    model = IdeaEncoder(enc, cfg["dim"])
    model.proj.load_state_dict(torch.load(folder / "proj.pt", weights_only=True))
    return model.to(pick_device(device)).eval(), tok, cfg["thresholds"]
