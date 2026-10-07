# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Communication and collaboration evidence per person per segment, computed
from utterance labels (the model's predictions at run time, or gold labels).

Same idea as ct_features: every feature is a plain share or rate a teacher
can read back ("took 22% of the turns, 40% of them built on someone else"),
so each score can say why.

  communication  - taking a fair part, being clear, explaining, asking
  collaboration  - responding to others, coordinating, building, staying on task
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ct_features import unit_key

COMM_FEATURES = [
    "log_turns", "turn_share", "share_balance", "log_mean_words",
    "rate_question", "rate_reasoning", "rate_explanation", "rate_new_idea",
    "rate_stuck", "rate_off_task", "rate_positive", "rate_negative",
]
COLLAB_FEATURES = [
    "turn_share", "share_balance", "rate_builds_on", "rate_coordinates",
    "rate_question", "rate_positive", "rate_negative", "rate_stuck", "rate_off_task",
    "responds_to_other", "builds_after_other", "rate_challenge", "revision_after_challenge",
]


def _words(text: str) -> int:
    t = str(text).strip()
    return len(t.split()) if t else 0


def social_features(utts: pd.DataFrame) -> pd.DataFrame:
    """One row per (unit, speaker) with COMM_FEATURES, COLLAB_FEATURES and n_turns."""
    if "speaker" not in utts.columns:
        raise ValueError("utterances need a `speaker` column (de-identified placeholder) to score people")
    key = unit_key(utts)
    rows = []
    for unit, g in utts.groupby(key, sort=False):
        g = g.reset_index(drop=True)
        speakers = list(g["speaker"].unique())
        fair = 1.0 / max(len(speakers), 1)
        prev_speaker = g["speaker"].shift(1)
        for spk in speakers:
            mine = g[g["speaker"] == spk]
            n = len(mine)
            share = n / len(g)
            after_other = (prev_speaker.loc[mine.index].notna()) & (prev_speaker.loc[mine.index] != spk)
            collab_move = mine["move"].isin(["builds_on", "coordinates", "question"])
            f = {key: unit, "speaker": spk, "n_turns": n}
            f["log_turns"] = float(np.log1p(n))
            f["turn_share"] = float(share)
            f["share_balance"] = float(1 - min(abs(share - fair) / fair, 1)) if len(speakers) > 1 else 1.0
            f["log_mean_words"] = float(np.log1p(np.mean([_words(t) for t in mine["text"]])))
            for name, col, val in [
                ("rate_question", "move", "question"), ("rate_reasoning", "move", "reasoning"),
                ("rate_new_idea", "move", "new_idea"), ("rate_builds_on", "move", "builds_on"),
                ("rate_coordinates", "move", "coordinates"), ("rate_stuck", "move", "stuck"),
                ("rate_off_task", "move", "off_task"), ("rate_explanation", "ct_skill", "explanation"),
                ("rate_positive", "sentiment", "positive"), ("rate_negative", "sentiment", "negative"),
                ("rate_challenge", "argument", "challenge"),
            ]:
                f[name] = float((mine[col] == val).mean())
            f["responds_to_other"] = float((collab_move & after_other).mean())
            f["builds_after_other"] = float(((mine["move"] == "builds_on") & after_other).mean())
            rev = 0
            for i in mine.index:
                before = g.iloc[max(0, i - 5):i]
                if g.at[i, "argument"] == "revision" and (
                        (before["argument"] == "challenge") & (before["speaker"] != spk)).any():
                    rev += 1
            f["revision_after_challenge"] = rev / n
            rows.append(f)
    return pd.DataFrame(rows)
