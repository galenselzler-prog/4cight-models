# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Load and validate the three labeled files the labeling guide produces:
utterances, same-idea pairs, and segment ratings. Validation is strict — a
bad row fails loudly with its line number instead of silently skewing a model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import labels as L


class DataError(ValueError):
    pass


def _read(path: str | Path, required: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{path}: missing columns {missing}")
    return df


def _check_values(df: pd.DataFrame, col: str, allowed: list[str], path, allow_empty=False):
    bad = df[~df[col].isin(allowed + ([""] if allow_empty else []))]
    if len(bad):
        row = bad.index[0] + 2  # +1 header, +1 one-based
        raise DataError(f"{path}: line {row}: {col}={bad.iloc[0][col]!r} not in {allowed}")


UTTERANCE_COLUMNS = [
    "utterance_id", "session_id", "activity_id", "text", "context_prev", "grade_band",
    "sentiment", "move", "ct_skill", "argument", "idea_id", "idea_link", "linked_ideas",
    "teacher_idea", "flag_unclear", "split", "guide_version",
]


def load_utterances(path: str | Path) -> pd.DataFrame:
    """Final (adjudicated) utterance labels. Rows keep file order, which within
    a whole session is the order the utterances were spoken."""
    df = _read(path, UTTERANCE_COLUMNS)
    for col, allowed in [("sentiment", L.SENTIMENT), ("move", L.MOVE), ("ct_skill", L.CT_SKILL),
                         ("argument", L.ARGUMENT), ("grade_band", L.GRADE_BANDS),
                         ("split", ["train", "validation", "test"])]:
        _check_values(df, col, allowed, path)
    _check_values(df, "idea_link", L.IDEA_LINK, path, allow_empty=True)
    if df["utterance_id"].duplicated().any():
        raise DataError(f"{path}: duplicate utterance_id {df[df['utterance_id'].duplicated()].iloc[0]['utterance_id']}")
    # A whole session must never straddle splits (prevents leakage between train and test).
    per_session = df.groupby("session_id")["split"].nunique()
    leaky = per_session[per_session > 1]
    if len(leaky):
        raise DataError(f"{path}: session {leaky.index[0]} appears in more than one split")
    return df.reset_index(drop=True)


PAIR_COLUMNS = ["pair_id", "activity_id", "idea_a_text", "idea_b_text", "final_label"]


def load_pairs(path: str | Path) -> pd.DataFrame:
    df = _read(path, PAIR_COLUMNS)
    _check_values(df, "final_label", L.PAIR_LABEL, path)
    return df.reset_index(drop=True)


SEGMENT_COLUMNS = ["segment_id", "activity_id", "grade_band", "rated", "skill", "final_level"]


def load_segment_ratings(path: str | Path) -> pd.DataFrame:
    """One row per (segment, person, skill) with the final level. Rows rated
    NE (not enough evidence) are kept for reference but have level None."""
    df = _read(path, SEGMENT_COLUMNS)
    _check_values(df, "skill", L.RATED_SKILLS, path)
    _check_values(df, "final_level", ["1", "2", "3", "4", L.NOT_ENOUGH_EVIDENCE], path)
    df = df.copy()
    df["level"] = df["final_level"].map(lambda v: None if v == L.NOT_ENOUGH_EVIDENCE else int(v))
    return df.reset_index(drop=True)


def model_input_text(text: str, context_prev: str) -> str:
    """How an utterance is shown to the model: the utterance first (so it is
    never truncated away), then up to two previous turns as context."""
    return f"{text} [CTX] {context_prev}" if context_prev else text


@dataclass
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split(df: pd.DataFrame) -> Splits:
    return Splits(*(df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test")))
