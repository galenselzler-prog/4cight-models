# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Pre-fill labeling tasks with a trained utterance model, so labelers correct
instead of starting from scratch.

  fourc prep-labeling --prelabel runs/pretrain

For each utterance the model's best guess for move, critical-thinking skill and
argument part is attached to the Label Studio task as a *prediction*, but only
when the model is confident (MIN_CONFIDENCE). Unsure fields are left empty so
the labeler has to choose.

Guarding against labelers just accepting the guess ("anchoring"):

  * BLIND_FRACTION of utterances (picked by a hash of the utterance id, so it is
    the same every run) get NO prediction at all. Those are labeled cold.
  * Every task carries `prelabel` = shown | blind | none and `model_label`
    (what the model guessed, even for blind tasks). `fourc import-labels` copies
    both into utterances.csv, so after labeling you can compare:
      - model vs. human on BLIND tasks = how good the model really is, unbiased;
      - human agreement on shown vs. blind tasks = whether guesses are pulling
        labelers toward them.

Sentiment is not pre-filled: the weak meeting labels barely cover it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from . import labels as L

PRELABEL_HEADS = ("move", "ct_skill", "argument")
MIN_CONFIDENCE = 0.6
BLIND_FRACTION = 0.20


def is_blind(utterance_id: str, fraction: float = BLIND_FRACTION) -> bool:
    """Stable choice: the same utterance is always blind or always shown."""
    h = int(hashlib.sha256(("prelabel:" + utterance_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < fraction


def prediction_result(guesses: dict[str, tuple[str, float]], version: str) -> dict | None:
    """Label Studio prediction for {head: (label, confidence)}; None when nothing is confident."""
    result = [{"from_name": h, "to_name": "text", "type": "choices", "score": round(float(p), 4),
               "value": {"choices": [label]}} for h, (label, p) in guesses.items()]
    if not result:
        return None
    return {"model_version": version, "score": round(min(r["score"] for r in result), 4), "result": result}


class Prelabeler:
    def __init__(self, folder: str | Path, min_confidence: float = MIN_CONFIDENCE,
                 blind_fraction: float = BLIND_FRACTION, device: str | None = None):
        from . import utterance_model  # torch is only needed when pre-labeling

        self.model, self.tok = utterance_model.load(folder, device)
        self._predict = utterance_model.predict_proba
        self.min_confidence = min_confidence
        self.blind_fraction = blind_fraction
        self.version = f"prelabel:{Path(folder).name}"

    def guess(self, tasks: list[dict]) -> list[dict[str, tuple[str, float]]]:
        """Per task: {head: (label, confidence)} for every pre-filled head, confident or not."""
        if not tasks:
            return []
        df = pd.DataFrame({"text": [t["text"] for t in tasks], "context_prev": [t.get("context_prev", "") for t in tasks]})
        proba = self._predict(self.model, self.tok, df)
        out = []
        for i in range(len(tasks)):
            g = {}
            for h in PRELABEL_HEADS:
                p = proba[h][i]
                k = int(p.argmax())
                g[h] = (L.UTTERANCE_HEADS[h][k], float(p[k]))
            out.append(g)
        return out

    def annotate(self, tasks: list[dict]) -> list[dict | None]:
        """Adds `prelabel` and `model_label` to each task's data (in place) and returns, per task,
        the prediction to attach (None for blind tasks and when the model is not confident)."""
        preds: list[dict | None] = []
        for t, g in zip(tasks, self.guess(tasks)):
            t["model_label"] = "|".join(g[h][0] for h in PRELABEL_HEADS)
            if is_blind(t["utterance_id"], self.blind_fraction):
                t["prelabel"] = "blind"
                preds.append(None)
                continue
            pred = prediction_result({h: v for h, v in g.items() if v[1] >= self.min_confidence}, self.version)
            t["prelabel"] = "shown" if pred else "none"
            preds.append(pred)
        return preds
