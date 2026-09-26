# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Creativity evidence per person per segment (PISA-style facets):

  fluency      distinct ideas the person contributed (duplicates by MEANING removed)
  flexibility  distinct kinds of idea (clusters of related ideas)
  originality  how rare each idea is across other groups doing the same activity
  elaboration  how often the person developed or combined ideas

Idea links are found with the idea encoder: an idea-bearing utterance is a
repeat / develops / combines / new idea depending on its similarity to the
ideas already said in the session. Originality needs an IdeaBank per
activity_id, built from past sessions (training split only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .ct_features import unit_key

IDEA_MOVES = ("new_idea", "builds_on")

CR_FEATURES = ["log_turns", "fluency", "fluency_rate", "flexibility", "originality_mean", "originality_max",
               "elaboration", "elaboration_rate", "rate_builds_on", "rate_repeat"]


def link_ideas(utts: pd.DataFrame, emb: np.ndarray, thresholds: dict) -> pd.DataFrame:
    """Adds `pred_link` (new/repeat/develops/combines or "") and `pred_idea`
    (index of the new idea it is / links to) for each utterance, session by
    session in spoken order. `emb` holds one vector per utterance row."""
    out = utts.copy()
    link, idea = [""] * len(out), [-1] * len(out)
    for _, g in out.groupby("session_id", sort=False):
        seen: list[int] = []  # row positions of new ideas so far
        for pos in (out.index.get_indexer(g.index)):
            if out.iloc[pos]["move"] not in IDEA_MOVES:
                continue
            if seen:
                sims = emb[seen] @ emb[pos]
                j = int(sims.argmax())
                if sims[j] >= thresholds["same"]:
                    link[pos], idea[pos] = "repeat", seen[j]
                    continue
                related = sims >= thresholds["related"]
                if related.sum() >= 2:
                    link[pos], idea[pos] = "combines", seen[j]
                    continue
                if related.any():
                    link[pos], idea[pos] = "develops", seen[j]
                    continue
            link[pos], idea[pos] = "new", pos
            seen.append(pos)
    out["pred_link"], out["pred_idea"] = link, idea
    return out


@dataclass
class IdeaBank:
    """Past ideas per activity, deduplicated by meaning, with how many
    sessions produced each and which kind (related cluster) it belongs to."""
    thresholds: dict
    centroids: dict = field(default_factory=dict)      # activity -> (k, d) array
    sessions: dict = field(default_factory=dict)       # activity -> list[int] sessions containing idea
    kinds: dict = field(default_factory=dict)          # activity -> list[int] kind id
    n_sessions: dict = field(default_factory=dict)     # activity -> int

    @classmethod
    def build(cls, linked: pd.DataFrame, emb: np.ndarray, thresholds: dict) -> "IdeaBank":
        bank = cls(thresholds)
        new = linked["pred_link"].to_numpy() == "new"
        for act, g in linked[new].groupby("activity_id"):
            pos = linked.index.get_indexer(g.index)
            cents: list[np.ndarray] = []
            sess: list[set] = []
            for p, sid in zip(pos, g["session_id"]):
                v = emb[p]
                if cents:
                    sims = np.stack(cents) @ v
                    j = int(sims.argmax())
                    if sims[j] >= thresholds["same"]:
                        sess[j].add(sid)
                        continue
                cents.append(v); sess.append({sid})
            C = np.stack(cents)
            kinds = [-1] * len(C)
            k = 0
            for i in range(len(C)):
                if kinds[i] < 0:
                    for j in np.where((C @ C[i]) >= thresholds["related"])[0]:
                        if kinds[j] < 0:
                            kinds[j] = k
                    k += 1
            bank.centroids[act], bank.sessions[act] = C, [len(s) for s in sess]
            bank.kinds[act], bank.n_sessions[act] = kinds, linked.loc[linked["activity_id"] == act, "session_id"].nunique()
        return bank

    def lookup(self, activity: str, v: np.ndarray) -> tuple[float, str]:
        """(originality 0-1, kind label). Unknown activity -> 0.5 and a kind of its own."""
        if activity not in self.centroids:
            return 0.5, "unbanked"
        sims = self.centroids[activity] @ v
        j = int(sims.argmax())
        if sims[j] >= self.thresholds["same"]:
            return 1.0 - self.sessions[activity][j] / max(1, self.n_sessions[activity]), f"k{self.kinds[activity][j]}"
        kind = f"k{self.kinds[activity][j]}" if sims[j] >= self.thresholds["related"] else "novel"
        return 1.0, kind  # never seen before in any past session

    def save(self, path):
        np.savez_compressed(Path(path).with_suffix(".npz"), **{f"c::{a}": c for a, c in self.centroids.items()})
        Path(path).with_suffix(".json").write_text(json.dumps(
            {"thresholds": self.thresholds, "sessions": self.sessions, "kinds": self.kinds,
             "n_sessions": self.n_sessions}, indent=1))

    @classmethod
    def load(cls, path) -> "IdeaBank":
        meta = json.loads(Path(path).with_suffix(".json").read_text())
        arr = np.load(Path(path).with_suffix(".npz"))
        return cls(meta["thresholds"], {k[3:]: arr[k] for k in arr.files}, meta["sessions"], meta["kinds"],
                   meta["n_sessions"])


def creativity_features(linked: pd.DataFrame, emb: np.ndarray, bank: IdeaBank) -> pd.DataFrame:
    if "speaker" not in linked.columns:
        raise ValueError("utterances need a `speaker` column (de-identified placeholder) to score people")
    key = unit_key(linked)
    rows = []
    for unit, g in linked.groupby(key, sort=False):
        pos_all = linked.index.get_indexer(g.index)
        for spk in g["speaker"].unique():
            m = (g["speaker"] == spk).to_numpy()
            mine, pos = g[m], pos_all[m]
            n = len(mine)
            new_pos = [p for p, l in zip(pos, mine["pred_link"]) if l == "new"]
            orig, kinds = [], set()
            for p in new_pos:
                o, k = bank.lookup(linked.iloc[p]["activity_id"], emb[p])
                orig.append(o)
                kinds.add(k if k not in ("novel", "unbanked") else f"{k}{p}")
            elab = int(mine["pred_link"].isin(["develops", "combines"]).sum())
            rows.append({key: unit, "speaker": spk, "n_turns": n, "log_turns": float(np.log1p(n)),
                         "fluency": len(new_pos), "fluency_rate": len(new_pos) / n, "flexibility": len(kinds),
                         "originality_mean": float(np.mean(orig)) if orig else 0.0,
                         "originality_max": float(np.max(orig)) if orig else 0.0,
                         "elaboration": elab, "elaboration_rate": elab / n,
                         "rate_builds_on": float((mine["move"] == "builds_on").mean()),
                         "rate_repeat": float((mine["pred_link"] == "repeat").mean())})
    return pd.DataFrame(rows)
