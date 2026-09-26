# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Ordinal 1-4 scorer (proportional-odds / cumulative logistic model).

  P(level > k) = sigmoid(w . x - b_k),  b_1 < b_2 < b_3

Trained on teacher segment ratings; features are the explainable evidence
from ct_features / creativity_metrics. It is small on purpose: a linear
model on readable features gives each score a "because" list, trains on a
few hundred ratings, and ships as a JSON file the TypeScript engine runs
directly (no model file needed on the device)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

from . import labels as L

K = len(L.LEVELS)


class OrdinalScorer:
    def __init__(self, features: list[str], skill: str, min_turns: int = 3, l2: float = 1e-2):
        self.features, self.skill, self.min_turns, self.l2 = list(features), skill, min_turns, l2
        self.mean = self.std = self.w = self.b = None

    # ---- training -------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, epochs: int = 600, lr: float = 0.05, seed: int = 0):
        torch.manual_seed(seed)
        A = X[self.features].to_numpy(dtype=np.float64)
        self.mean, self.std = A.mean(0), A.std(0) + 1e-6
        Z = torch.tensor((A - self.mean) / self.std, dtype=torch.float32)
        Y = torch.tensor(np.asarray(y, dtype=int) - 1)  # 0..K-1
        w = torch.zeros(Z.shape[1], requires_grad=True)
        b0 = torch.zeros(1, requires_grad=True)
        gaps = torch.zeros(K - 2, requires_grad=True)  # thresholds kept ordered via softplus gaps
        opt = torch.optim.Adam([w, b0, gaps], lr=lr)
        for _ in range(epochs):
            b = self._thresholds_t(b0, gaps)
            p = self._level_probs_t(Z @ w, b)
            loss = -torch.log(p[torch.arange(len(Y)), Y].clamp(min=1e-9)).mean() + self.l2 * (w ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        self.w = w.detach().numpy().astype(np.float64)
        self.b = self._thresholds_t(b0, gaps).detach().numpy().astype(np.float64)
        return self

    @staticmethod
    def _thresholds_t(b0, gaps):
        return torch.cat([b0, b0 + torch.cumsum(torch.nn.functional.softplus(gaps) + 1e-3, 0)])

    @staticmethod
    def _level_probs_t(eta, b):
        gt = torch.sigmoid(eta[:, None] - b[None, :])  # P(level > k), k=1..K-1
        ones, zeros = torch.ones_like(eta[:, None]), torch.zeros_like(eta[:, None])
        cum = torch.cat([ones, gt, zeros], 1)
        return (cum[:, :-1] - cum[:, 1:]).clamp(min=0)

    # ---- inference (pure numpy; mirrored in TypeScript) ------------------
    def level_probs(self, X: pd.DataFrame) -> np.ndarray:
        z = (X[self.features].to_numpy(dtype=np.float64) - self.mean) / self.std
        gt = 1 / (1 + np.exp(-(z @ self.w)[:, None] + self.b[None, :]))
        cum = np.concatenate([np.ones((len(z), 1)), gt, np.zeros((len(z), 1))], 1)
        return np.clip(cum[:, :-1] - cum[:, 1:], 0, 1)

    def predict(self, X: pd.DataFrame) -> list:
        """Level 1-4, or "NE" when the person said too little to judge."""
        levels = self.level_probs(X).argmax(1) + 1
        return [L.NOT_ENOUGH_EVIDENCE if n < self.min_turns else int(l) for l, n in zip(levels, X["n_turns"])]

    def expected_display(self, X: pd.DataFrame) -> np.ndarray:
        """Smooth 1-10 display value: expected level mapped like LEVEL_TO_DISPLAY."""
        return self.level_probs(X) @ np.array([L.LEVEL_TO_DISPLAY[l] for l in L.LEVELS])

    def evidence(self, row: pd.Series, top: int = 3) -> list[tuple[str, float]]:
        """Features that pushed this person's score up or down the most."""
        c = ((row[self.features].to_numpy(dtype=np.float64) - self.mean) / self.std) * self.w
        order = np.argsort(-np.abs(c))[:top]
        return [(self.features[i], round(float(c[i]), 3)) for i in order]

    # ---- evaluation / persistence ----------------------------------------
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        pred = np.array(self.level_probs(X).argmax(1) + 1)
        y = np.asarray(y, dtype=int)
        return {"n": int(len(y)), "qwk": round(float(cohen_kappa_score(y, pred, weights="quadratic")), 4),
                "exact": round(float(accuracy_score(y, pred)), 4),
                "within_1": round(float(np.mean(np.abs(y - pred) <= 1)), 4),
                "mae": round(float(mean_absolute_error(y, pred)), 4)}

    def to_json(self, version: str) -> dict:
        return {"kind": "ordinal-logistic", "skill": self.skill, "version": version, "levels": L.LEVELS,
                "levelToDisplay": {str(k): v for k, v in L.LEVEL_TO_DISPLAY.items()},
                "minTurns": self.min_turns, "features": self.features, "mean": self.mean.tolist(),
                "std": self.std.tolist(), "weights": self.w.tolist(), "thresholds": self.b.tolist()}

    def save(self, path, version: str):
        Path(path).write_text(json.dumps(self.to_json(version), indent=2))

    @classmethod
    def load(cls, path) -> "OrdinalScorer":
        d = json.loads(Path(path).read_text())
        s = cls(d["features"], d["skill"], d["minTurns"])
        s.mean, s.std = np.array(d["mean"]), np.array(d["std"])
        s.w, s.b = np.array(d["weights"]), np.array(d["thresholds"])
        return s


def join_ratings(features: pd.DataFrame, ratings: pd.DataFrame, skill: str) -> pd.DataFrame:
    """Matches feature rows to teacher ratings: segment_id <-> unit, rated <-> speaker.
    NE ratings are dropped (they are not a level)."""
    key = "segment_id" if "segment_id" in features.columns else "session_id"
    r = ratings[(ratings["skill"] == skill) & ratings["level"].notna()]
    r = r.rename(columns={"segment_id": key, "rated": "speaker"})[[key, "speaker", "level"]]
    return features.merge(r, on=[key, "speaker"], how="inner")
