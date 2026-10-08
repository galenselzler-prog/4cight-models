# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Pretraining on weak labels with unknown (empty) cells, then starting another run from it."""

import random

import pandas as pd

from fourc_models import data, synthetic, utterance_model as U, weak_pretrain as W


def _weak_csv(path, n=240):
    rng = random.Random(0)
    qs = ["who did that", "what if we try it", "why is it slow", "which one do we pick"]
    ss = ["we could use a curved case", "maybe add a second button", "let us try a bigger wheel"]
    rows = []
    for i in range(n):
        if rng.random() < 0.5:
            t, mv, ar = rng.choice(qs), "question", ""            # argument unknown
        else:
            t, mv, ar = rng.choice(ss), "new_idea", "claim"
        rows.append(dict(text=t, context_prev="", sentiment="", move=mv, ct_skill="", argument=ar,
                         split="train" if i < n * 0.7 else ("validation" if i < n * 0.85 else "test")))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_unknown_labels_are_skipped_by_loss_and_metrics():
    df = pd.DataFrame({"text": ["a", "b"], "context_prev": ["", ""], "sentiment": ["", ""],
                       "move": ["question", ""], "ct_skill": ["", ""], "argument": ["", "claim"]})
    y = U._targets(df)
    assert y["move"].tolist() == [L_index("move", "question"), U.UNKNOWN]
    assert y["sentiment"].tolist() == [U.UNKNOWN, U.UNKNOWN]


def L_index(head, label):
    return U.HEADS[head].index(label)


def test_pretrain_then_start_another_run_from_it(tmp_path):
    csv = tmp_path / "weak.csv"
    _weak_csv(csv)
    rep = W.pretrain(csv, tmp_path / "pre", base="tiny", epochs=3, max_eval=None, log=lambda *_: None)
    assert rep["test"]["move"]["n"] > 0 and rep["test"]["sentiment"]["n"] == 0, "no sentiment labels -> not scored"
    assert rep["test"]["move"]["macro_f1"] > 0.4, rep["test"]["move"]
    # a classroom-style dataset (all labels present), starting from the pretrained model
    d = tmp_path / "class"
    synthetic.write_dataset(d, n_sessions=12, seed=1)
    sp = data.split(data.load_utterances(d / "utterances.csv"))
    out = U.train(sp, "tiny", tmp_path / "ft", epochs=1, log=lambda *_: None, init_from=tmp_path / "pre")
    assert set(out) == set(U.HEADS)
