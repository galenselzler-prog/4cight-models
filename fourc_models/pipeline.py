# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""End-to-end training run for the critical-thinking, creativity, communication
and collaboration models.

  1. utterance model (sentiment, move, CT skill, argument part)
  2. idea encoder (same / related / different ideas)
  3. predict every utterance, link ideas, build the IdeaBank (train split only)
  4. evidence features per person per segment
  5. ordinal 1-4 scorers fitted to teacher ratings
  6. report.json with validation AND test metrics

Everything the scorers are trained and evaluated on is computed from MODEL
PREDICTIONS, exactly as on the device, so test numbers reflect the real
pipeline rather than perfect labels."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import __version__, data, idea_model, utterance_model
from . import labels as L
from .creativity_metrics import CR_FEATURES, IDEA_MOVES, IdeaBank, creativity_features, link_ideas
from .ct_features import CT_FEATURES, ct_features
from .scorer import OrdinalScorer, join_ratings
from .social_features import COLLAB_FEATURES, COMM_FEATURES, social_features


def _embed_ideas(model, tok, df: pd.DataFrame) -> np.ndarray:
    emb = np.zeros((len(df), model.proj.out_features), dtype=np.float32)
    mask = df["move"].isin(IDEA_MOVES).to_numpy()
    if mask.any():
        emb[mask] = idea_model.embed(model, tok, list(df.loc[mask, "text"]))
    return emb


def _fit_scorer(feats, ratings, skill, names, key, split_of, log):
    joined = join_ratings(feats, ratings, skill)
    joined["split"] = joined[key].map(split_of)
    tr = joined[joined["split"] == "train"]
    if tr["level"].nunique() < 2:
        raise ValueError(f"{skill}: need ratings at 2+ levels in the train split")
    s = OrdinalScorer(names, skill).fit(tr, tr["level"].to_numpy())
    res = {sp: s.evaluate(g, g["level"].to_numpy()) for sp, g in joined.groupby("split") if len(g)}
    log(f"  {skill} scorer: " + ", ".join(f"{k} QWK {v['qwk']:.2f}" for k, v in res.items()))
    return s, res, joined


def run(data_dir, out_dir, base: str = "tiny", idea_base: str | None = None, epochs: int = 4,
        idea_epochs: int = 6, device: str | None = None, seed: int = 0, log=print,
        init_utterance: str | None = None) -> dict:
    t0 = time.time()
    data_dir, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    utts = data.load_utterances(data_dir / "utterances.csv")
    pairs = data.load_pairs(data_dir / "pairs.csv")
    ratings = data.load_segment_ratings(data_dir / "segments.csv")
    sp = data.split(utts)
    log(f"data: {len(utts)} utterances ({len(sp.train)}/{len(sp.validation)}/{len(sp.test)}), "
        f"{len(pairs)} pairs, {len(ratings)} ratings")

    log("[1/5] utterance model")
    um_val = utterance_model.train(sp, base, out / "utterance", epochs=epochs, seed=seed, device=device, log=log,
                                    init_from=init_utterance)
    um, utok = utterance_model.load(out / "utterance", device)
    um_test = utterance_model.evaluate(um, utok, sp.test)

    log("[2/5] idea encoder")
    im_res = idea_model.train(pairs, sp.train, idea_base or base, out / "idea", epochs=idea_epochs, seed=seed,
                              device=device, log=log, utts_val=sp.validation)
    im, itok, th = idea_model.load(out / "idea", device)

    log("[3/5] predict + link ideas + idea bank")
    pred = utterance_model.predict_labels(um, utok, utts)
    emb = _embed_ideas(im, itok, pred)
    linked = link_ideas(pred, emb, th)
    tr_mask = (linked["split"] == "train").to_numpy()
    bank = IdeaBank.build(linked[tr_mask].reset_index(drop=True), emb[tr_mask], th)
    bank.save(out / "idea_bank")
    gold = utts["idea_link"] != ""
    test_links = gold & (utts["split"] == "test")
    link_f1 = f1_score(utts.loc[test_links, "idea_link"], linked.loc[test_links, "pred_link"].replace("", "none"),
                       average="macro", labels=sorted(set(utts.loc[test_links, "idea_link"])), zero_division=0) if test_links.any() else None

    log("[4/5] evidence features")
    ctf = ct_features(pred)
    crf = creativity_features(linked, emb, bank)
    key = "segment_id" if "segment_id" in utts.columns else "session_id"
    split_of = utts.groupby(key)["split"].first()

    log("[5/5] scorers")
    version = f"{__version__}+g{L.GUIDE_VERSION}"
    ct, ct_res, ct_joined = _fit_scorer(ctf, ratings, "critical_thinking", CT_FEATURES, key, split_of, log)
    cr, cr_res, _ = _fit_scorer(crf, ratings, "creativity", CR_FEATURES, key, split_of, log)
    ct.save(out / "scorer_critical_thinking.json", version)
    cr.save(out / "scorer_creativity.json", version)

    # Communication and collaboration: fitted only when teachers rated them, so
    # runs on older data (critical thinking and creativity only) still work.
    socf = social_features(pred)
    social_res = {}
    for skill, names in (("communication", COMM_FEATURES), ("collaboration", COLLAB_FEATURES)):
        if not (ratings["skill"] == skill).any():
            log(f"  {skill}: no teacher ratings in this data, skipped")
            continue
        sc, res, _ = _fit_scorer(socf, ratings, skill, names, key, split_of, log)
        sc.save(out / f"scorer_{skill}.json", version)
        social_res[skill] = res

    # Diagnostic: same CT scorer on features from GOLD labels = ceiling if the utterance model were perfect.
    gold_ct = join_ratings(ct_features(utts), ratings, "critical_thinking")
    gold_ct = gold_ct[gold_ct[key].map(split_of) == "test"]
    ct_ceiling = ct.evaluate(gold_ct, gold_ct["level"].to_numpy()) if len(gold_ct) else None

    example = ct_joined[ct_joined["split"] == "test"].head(1)
    report = {
        "version": version, "base": base, "idea_base": idea_base or base, "seconds": round(time.time() - t0, 1),
        "data": {"utterances": len(utts), "pairs": len(pairs), "ratings": len(ratings),
                 "sessions": int(utts["session_id"].nunique())},
        "utterance_model": {"validation": um_val, "test": um_test},
        "idea_model": {**im_res, "test_idea_link_macro_f1": None if link_f1 is None else round(link_f1, 4)},
        "critical_thinking_scorer": {**ct_res, "test_on_gold_labels": ct_ceiling},
        "creativity_scorer": cr_res,
        "communication_scorer": social_res.get("communication"),
        "collaboration_scorer": social_res.get("collaboration"),
        "example_evidence": {"speaker": example["speaker"].iloc[0], "level": int(example["level"].iloc[0]),
                             "predicted": ct.predict(example)[0], "because": ct.evidence(example.iloc[0])}
        if len(example) else None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"done in {report['seconds']}s -> {out / 'report.json'}")
    return report
