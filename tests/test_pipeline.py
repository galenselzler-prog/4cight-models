# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""End to end on synthetic data with tiny offline models: data -> train ->
report -> ONNX export -> onnxruntime. Proves the plumbing, not accuracy."""

import json

import numpy as np
import pandas as pd
import pytest

from fourc_models import data
from fourc_models.scorer import OrdinalScorer
from fourc_models.synthetic import write_dataset


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from fourc_models.pipeline import run
    root = tmp_path_factory.mktemp("e2e")
    write_dataset(root / "data", n_sessions=40, seed=1)
    run(root / "data", root / "run", base="tiny", epochs=3, idea_epochs=4, device="cpu", log=lambda *_: None)
    return root


def test_report_has_all_metrics(run_dir):
    rep = json.loads((run_dir / "run" / "report.json").read_text())
    for head in ("sentiment", "move", "ct_skill", "argument"):
        assert rep["utterance_model"]["test"][head]["n"] > 0
    for k in ("critical_thinking_scorer", "creativity_scorer"):
        assert {"train", "validation", "test"} <= set(rep[k])
    # Synthetic levels drive behaviour, so a working pipeline beats chance on held-out
    # sessions. Validation and test are averaged because each holds only ~24 people.
    for k in ("critical_thinking_scorer", "creativity_scorer"):
        assert (rep[k]["validation"]["qwk"] + rep[k]["test"]["qwk"]) / 2 > 0.25, k
    assert rep["example_evidence"]["because"]


def test_export_onnx_runs(run_dir):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from fourc_models.export import export

    man = export(run_dir / "run", run_dir / "dist", log=lambda *_: None)
    out = run_dir / "dist"
    assert [m["task"] for m in man["models"]] == ["utterance", "idea"]
    tok = Tokenizer.from_file(str(out / "utterance.tokenizer.json"))
    enc = tok.encode("what if we tape the wheels")
    feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
             "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
    outs = ort.InferenceSession(str(out / "utterance.onnx")).run(None, feeds)
    assert [o.shape[1] for o in outs] == [len(v) for v in man["labels"].values()]
    tok2 = Tokenizer.from_file(str(out / "idea.tokenizer.json"))
    e2 = tok2.encode("tape the wheels")
    vec = ort.InferenceSession(str(out / "idea.onnx")).run(None, {
        "input_ids": np.array([e2.ids], dtype=np.int64), "attention_mask": np.array([e2.attention_mask], dtype=np.int64)})[0]
    assert abs(np.linalg.norm(vec) - 1) < 1e-3
    s = json.loads((out / "scorer_creativity.json").read_text())
    assert len(s["weights"]) == len(s["features"]) and len(s["thresholds"]) == 3
    assert json.loads((out / "idea_bank.json").read_text())["activities"]


def test_scorer_json_roundtrip_and_ne(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "n_turns": 10})
    y = np.clip(np.round(X["a"] * 1.2 + 2.5), 1, 4).astype(int)
    s = OrdinalScorer(["a"], "critical_thinking").fit(X, y)
    assert s.evaluate(X, y)["qwk"] > 0.8
    assert np.all(np.diff(s.b) > 0)  # thresholds stay ordered
    s.save(tmp_path / "s.json", "t")
    s2 = OrdinalScorer.load(tmp_path / "s.json")
    assert np.allclose(s.level_probs(X), s2.level_probs(X))
    X.loc[0, "n_turns"] = 1
    assert s2.predict(X)[0] == "NE"


def test_validation_catches_bad_rows(tmp_path):
    paths = write_dataset(tmp_path, n_sessions=4, seed=2)
    u = pd.read_csv(paths["utterances"], dtype=str, keep_default_na=False)
    u.loc[3, "move"] = "arguing"
    u.to_csv(paths["utterances"], index=False)
    with pytest.raises(data.DataError, match="line 5"):
        data.load_utterances(paths["utterances"])
    u.loc[3, "move"] = "other"
    u.loc[u["session_id"] == u["session_id"].iloc[0], "split"] = ["train", "test"] * (len(u[u["session_id"] == u["session_id"].iloc[0]]) // 2)
    u.to_csv(paths["utterances"], index=False)
    with pytest.raises(data.DataError, match="more than one split"):
        data.load_utterances(paths["utterances"])


def test_deberta_family_exports_to_onnx(tmp_path):
    """The planned real base (DeBERTa-v3) exports and matches PyTorch; checked
    with a tiny random config so no download is needed."""
    from transformers import DebertaV2Config, DebertaV2Model
    from fourc_models.export import _onnx
    from fourc_models.text_encoder import train_local_tokenizer
    from fourc_models.utterance_model import HEADS, UtteranceModel

    tok = train_local_tokenizer(["what if we tape the wheels", "that will not work because it failed"] * 20)
    cfg = DebertaV2Config(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                          intermediate_size=128, max_position_embeddings=256, relative_attention=True,
                          position_buckets=32, max_relative_positions=-1, pos_att_type=["p2c", "c2p"],
                          position_biased_input=False, type_vocab_size=0, norm_rel_ebd="layer_norm",
                          share_att_key=True)
    diff = _onnx(UtteranceModel(DebertaV2Model(cfg)), tok, tmp_path / "d.onnx", list(HEADS),
                 ["what if we tape the wheels", "that will not work [CTX] ok"])
    assert diff < 1e-3
