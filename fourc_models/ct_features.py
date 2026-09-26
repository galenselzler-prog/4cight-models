# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Critical-thinking evidence per person per segment, computed from
utterance labels (the model's predictions at run time, or gold labels).

Every feature is a plain count or rate a teacher can read back ("revised
their own claim twice after a challenge"), which is what makes the score
explainable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import labels as L

SKILLS = [s for s in L.CT_SKILL if s != "none"]
PARTS = [a for a in L.ARGUMENT if a != "none"]
WINDOW = 5  # turns to look back for claim->evidence and challenge->revision chains

CT_FEATURES = (
    ["log_turns"]
    + [f"rate_{s}" for s in SKILLS]
    + ["skill_variety"]
    + [f"rate_{a}" for a in PARTS]
    + ["evidence_after_claim", "revision_after_challenge", "rate_question", "rate_reasoning", "rate_off_task"]
)


def unit_key(df: pd.DataFrame) -> str:
    """Segments are rated per person. If utterances carry a segment_id the
    unit is (segment, speaker); otherwise the whole session is the segment."""
    return "segment_id" if "segment_id" in df.columns else "session_id"


def ct_features(utts: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (unit, speaker) with CT_FEATURES plus n_turns."""
    if "speaker" not in utts.columns:
        raise ValueError("utterances need a `speaker` column (de-identified placeholder) to score people")
    key = unit_key(utts)
    rows = []
    for unit, g in utts.groupby(key, sort=False):
        g = g.reset_index(drop=True)
        for spk in g["speaker"].unique():
            mine = g[g["speaker"] == spk]
            n = len(mine)
            f = {key: unit, "speaker": spk, "n_turns": n, "log_turns": float(np.log1p(n))}
            for s in SKILLS:
                f[f"rate_{s}"] = float((mine["ct_skill"] == s).mean())
            f["skill_variety"] = mine.loc[mine["ct_skill"] != "none", "ct_skill"].nunique() / len(SKILLS)
            for a in PARTS:
                f[f"rate_{a}"] = float((mine["argument"] == a).mean())
            ev = rev = 0
            for i in mine.index:
                before = g.iloc[max(0, i - WINDOW):i]
                if g.at[i, "argument"] == "evidence" and (before["argument"] == "claim").any():
                    ev += 1
                if g.at[i, "argument"] == "revision" and (
                        (before["argument"] == "challenge") & (before["speaker"] != spk)).any():
                    rev += 1
            f["evidence_after_claim"] = ev / n
            f["revision_after_challenge"] = rev / n
            f["rate_question"] = float((mine["move"] == "question").mean())
            f["rate_reasoning"] = float((mine["move"] == "reasoning").mean())
            f["rate_off_task"] = float((mine["move"] == "off_task").mean())
            rows.append(f)
    return pd.DataFrame(rows)
