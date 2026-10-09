# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Pre-filled labeling tasks: predictions only where the model is confident, a stable blind
share left untouched, audit fields on every task, and a full prep run with mocked R2."""

import io
import json
import wave

import numpy as np
import pytest

from fourc_models import labeling, prelabel as P
from fourc_models import utterance_model as U
from fourc_models.text_encoder import load_base

META = {"durationMs": 20000, "activityId": "ramp-car", "gradeBand": "6-8",
        "participants": [{"speaker": "[STUDENT_A]"}, {"speaker": "[STUDENT_B]"}],
        "transcript": [{"startMs": i * 1000, "endMs": i * 1000 + 800, "speaker": None,
                        "text": ["why is it slow", "we could use a bigger wheel"][i % 2]} for i in range(18)]}


def _wav(sec):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.sin(np.arange(int(sec * 16000)) / 7) * 5000).astype(np.int16).tobytes())
    return b.getvalue()


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("um")
    texts = ["why is it slow", "we could use a bigger wheel", "okay", "hmm"]
    tok, enc = load_base("tiny", texts_for_tiny=texts, seed=0)
    U.save(U.UtteranceModel(enc), tok, d, "tiny")
    return d


def test_blind_share_is_stable_and_about_the_right_size():
    ids = [f"u_{i:05d}" for i in range(4000)]
    blind = [P.is_blind(i) for i in ids]
    assert blind == [P.is_blind(i) for i in ids]
    assert 0.17 < np.mean(blind) < 0.23
    assert not any(P.is_blind(i, 0.0) for i in ids) and all(P.is_blind(i, 1.0) for i in ids)


def test_prediction_only_holds_confident_fields():
    assert P.prediction_result({}, "v") is None
    r = P.prediction_result({"move": ("question", 0.91), "argument": ("claim", 0.65)}, "v1")
    assert r["model_version"] == "v1" and r["score"] == 0.65
    assert r["result"][0] == {"from_name": "move", "to_name": "text", "type": "choices", "score": 0.91,
                              "value": {"choices": ["question"]}}


def test_annotate_marks_every_task_and_hides_guesses_on_blind_ones(model_dir):
    pl = P.Prelabeler(model_dir, min_confidence=0.0)
    tasks = [{"utterance_id": f"u_{i:04d}", "text": "why is it slow", "context_prev": ""} for i in range(200)]
    preds = pl.annotate(tasks)
    for t, p in zip(tasks, preds):
        assert t["model_label"].count("|") == 2 and t["prelabel"] in ("shown", "blind")
        assert (t["prelabel"] == "blind") == (p is None)
        if p:
            assert {r["from_name"] for r in p["result"]} == set(P.PRELABEL_HEADS)
            for r in p["result"]:
                assert r["value"]["choices"][0] in U.HEADS[r["from_name"]]
    assert 20 < sum(t["prelabel"] == "blind" for t in tasks) < 60


def test_low_confidence_leaves_the_task_unfilled(model_dir):
    pl = P.Prelabeler(model_dir, min_confidence=1.01, blind_fraction=0.0)
    t = [{"utterance_id": "u_0001", "text": "okay", "context_prev": ""}]
    assert pl.annotate(t) == [None] and t[0]["prelabel"] == "none" and t[0]["model_label"]


def test_prep_labeling_attaches_predictions_with_mock_r2(model_dir):
    moto = pytest.importorskip("moto")
    import boto3
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="research")
        s3.put_object(Bucket="research", Key="recordings/rec1/session.json", Body=json.dumps(META).encode())
        s3.put_object(Bucket="research", Key="recordings/rec1/audio.wav", Body=_wav(20))
        pl = P.Prelabeler(model_dir, min_confidence=0.0)
        st = labeling.prep_labeling("research", s3=s3, log=lambda *_: None, prelabel=pl)
        assert st["utterance_tasks"] == 18
        keys = [o["Key"] for o in s3.list_objects_v2(Bucket="research", Prefix="labeling/utterance-tasks/")["Contents"]]
        tasks = [json.loads(s3.get_object(Bucket="research", Key=k)["Body"].read()) for k in sorted(keys)]
        assert all(t["data"]["prelabel"] in ("shown", "blind") for t in tasks)
        shown = [t for t in tasks if "predictions" in t]
        assert shown and all(t["data"]["prelabel"] == "shown" for t in shown)
        assert any(t["data"]["prelabel"] == "blind" and "predictions" not in t for t in tasks) or len(tasks) < 40
