# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Multi-task utterance model: one encoder, four heads (sentiment, move,
critical-thinking skill, argument part). One model file serves the sentiment
signal AND the per-utterance evidence for the critical-thinking and
creativity scores, so the device runs a single text model per utterance."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch import nn

from . import labels as L
from .data import model_input_text
from .text_encoder import Pooled, load_base, pick_device, reload_encoder

HEADS = L.UTTERANCE_HEADS
MAX_LEN = 128


class UtteranceModel(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.pooled = Pooled(encoder)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({h: nn.Linear(self.pooled.dim, len(v)) for h, v in HEADS.items()})

    def forward(self, input_ids, attention_mask):
        z = self.drop(self.pooled(input_ids, attention_mask))
        return tuple(self.heads[h](z) for h in HEADS)  # fixed order = HEADS order (ONNX outputs)


def _texts(df: pd.DataFrame) -> list[str]:
    return [model_input_text(t, c) for t, c in zip(df["text"], df["context_prev"])]


def _targets(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    return {h: torch.tensor([v.index(x) for x in df[h]]) for h, v in HEADS.items()}


def _class_weights(y: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse square-root frequency: rare labels (e.g. self_regulation) count
    more without letting a handful of examples dominate."""
    counts = torch.bincount(y, minlength=n).float().clamp(min=1)
    w = counts.sum() / counts.sqrt()
    return w / w.mean()


def train(splits, base: str, out_dir: str | Path, epochs: int = 4, lr: float = 3e-5, batch_size: int = 16,
          seed: int = 0, device: str | None = None, log=print) -> dict:
    """Fine-tunes on splits.train, keeps the epoch with the best validation
    mean macro-F1, saves the model to out_dir, returns validation metrics."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = pick_device(device)
    tok, enc = load_base(base, texts_for_tiny=_texts(splits.train), seed=seed)
    if base == "tiny" and lr < 1e-3:
        lr = 1e-3  # randomly initialised tiny model needs a larger step
    model = UtteranceModel(enc).to(dev)
    X, Y = _texts(splits.train), _targets(splits.train)
    losses = {h: nn.CrossEntropyLoss(weight=_class_weights(Y[h], len(v)).to(dev)) for h, v in HEADS.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(X) / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps))

    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(X))
        total = 0.0
        for i in range(0, len(X), batch_size):
            idx = order[i:i + batch_size]
            b = tok([X[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
            out = model(b["input_ids"], b["attention_mask"])
            loss = sum(losses[h](o, Y[h][idx].to(dev)) for h, o in zip(HEADS, out))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += loss.item() * len(idx)
        metrics = evaluate(model, tok, splits.validation, dev)
        score = float(np.mean([metrics[h]["macro_f1"] for h in HEADS]))
        log(f"  utterance epoch {ep + 1}/{epochs}: train loss {total / len(X):.3f}, val mean macro-F1 {score:.3f}")
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    save(model, tok, out_dir, base)
    return evaluate(model, tok, splits.validation, dev)


@torch.no_grad()
def predict_proba(model: UtteranceModel, tok, df: pd.DataFrame, device=None, batch_size: int = 64) -> dict[str, np.ndarray]:
    dev = device or next(model.parameters()).device
    model.eval()
    X = _texts(df)
    outs = {h: [] for h in HEADS}
    for i in range(0, len(X), batch_size):
        b = tok(X[i:i + batch_size], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
        for h, o in zip(HEADS, model(b["input_ids"], b["attention_mask"])):
            outs[h].append(torch.softmax(o, -1).cpu().numpy())
    return {h: np.concatenate(v) if v else np.zeros((0, len(HEADS[h]))) for h, v in outs.items()}


def predict_labels(model, tok, df: pd.DataFrame, device=None) -> pd.DataFrame:
    """Copy of df with the four label columns replaced by model predictions
    (plus `sentiment_pos/neu/neg` probabilities for the app)."""
    p = predict_proba(model, tok, df, device)
    out = df.copy()
    for h, v in HEADS.items():
        out[h] = [v[i] for i in p[h].argmax(1)]
    for i, s in enumerate(L.SENTIMENT):
        out[f"sentiment_{s[:3]}"] = p["sentiment"][:, i]
    return out


def evaluate(model, tok, df: pd.DataFrame, device=None) -> dict:
    p = predict_proba(model, tok, df, device)
    res = {}
    for h, v in HEADS.items():
        y = [v.index(x) for x in df[h]]
        yhat = p[h].argmax(1).tolist()
        res[h] = {
            "accuracy": round(accuracy_score(y, yhat), 4),
            "macro_f1": round(f1_score(y, yhat, average="macro", labels=list(range(len(v))), zero_division=0), 4),
            "kappa": round(cohen_kappa_score(y, yhat), 4) if len(set(y)) > 1 else None,
            "n": len(y),
        }
    return res


def save(model: UtteranceModel, tok, out_dir, base: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.pooled.encoder.save_pretrained(out / "encoder")
    tok.save_pretrained(out / "tokenizer")
    torch.save({h: m.state_dict() for h, m in model.heads.items()}, out / "heads.pt")
    (out / "config.json").write_text(json.dumps({"base": base, "heads": HEADS, "max_len": MAX_LEN,
                                                 "guide_version": L.GUIDE_VERSION}, indent=2))


def load(folder, device=None):
    folder = Path(folder)
    tok, enc = reload_encoder(folder)
    model = UtteranceModel(enc)
    for h, sd in torch.load(folder / "heads.pt", weights_only=True).items():
        model.heads[h].load_state_dict(sd)
    return model.to(pick_device(device)).eval(), tok
