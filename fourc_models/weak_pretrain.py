# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Pretrain the utterance model on the weak AMI/ICSI labels (meeting_labels.py).

The result is a starting point, not a finished model: it has seen adult work
meetings, so it knows what questions, suggestions, challenges and reasons
sound like, but not students. Fine-tune it on classroom labels with

    fourc train --data <classroom data> --out <run> --init-utterance <this output>

and compare with a run that starts from the plain base model (same command without
--init-utterance) on the same classroom data before trusting that it helps.

Empty labels in the weak file are skipped by the loss and the metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import data, utterance_model

LABEL_COLUMNS = ["text", "context_prev", "sentiment", "move", "ct_skill", "argument", "split"]


def load_weak(path: str | Path, max_train: int | None = None, max_eval: int | None = None, seed: int = 0) -> data.Splits:
    """The weak-label CSV as train/validation/test (split by meeting team, already in the file).
    max_train / max_eval take a random sample, for a quick first run."""
    df = pd.read_csv(path, keep_default_na=False, usecols=LABEL_COLUMNS, dtype=str)
    sp = data.split(df)
    if max_train and len(sp.train) > max_train:
        sp.train = sp.train.sample(max_train, random_state=seed).reset_index(drop=True)
    if max_eval:
        sp.validation = sp.validation.sample(min(max_eval, len(sp.validation)), random_state=seed).reset_index(drop=True)
        sp.test = sp.test.sample(min(max_eval, len(sp.test)), random_state=seed).reset_index(drop=True)
    return sp


def pretrain(weak_csv, out_dir, base: str = "tiny", epochs: int = 1, max_train: int | None = None,
             max_eval: int | None = 5000, device: str | None = None, seed: int = 0, log=print) -> dict:
    sp = load_weak(weak_csv, max_train, max_eval, seed)
    log(f"weak labels: train {len(sp.train):,} / validation {len(sp.validation):,} / test {len(sp.test):,} utterances")
    val = utterance_model.train(sp, base, Path(out_dir), epochs=epochs, seed=seed, device=device, log=log)
    model, tok = utterance_model.load(out_dir, device)
    test = utterance_model.evaluate(model, tok, sp.test)
    report = {"base": base, "epochs": epochs, "train_utterances": len(sp.train), "validation": val, "test": test,
              "note": "weak labels on adult meetings; a starting point for fine-tuning, not a classroom model"}
    (Path(out_dir) / "pretrain_report.json").write_text(json.dumps(report, indent=2))
    for h, m in test.items():
        log(f"  {h:<10} test macro-F1 {m['macro_f1']:.3f} (n={m['n']:,})")
    return report
