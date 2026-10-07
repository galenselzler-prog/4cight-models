# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Recording -> labeling tasks (R2 mocked) and Label Studio export -> training
CSVs. The export fixtures were produced by a real Label Studio 1.23.1 with
labeling/utterance_config.xml and rating_config.xml."""

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from fourc_models import data, labeling

FIX = Path(__file__).parent / "fixtures"
META = {"durationMs": 20000, "activityId": "ramp-car",
        "participants": [{"speaker": "[STUDENT_A]"}, {"speaker": "[STUDENT_B]"}],
        "transcript": [{"startMs": 500, "endMs": 2500, "speaker": None, "text": "what if we tape the wheels"},
                       {"startMs": 6000, "endMs": 7000, "speaker": None, "text": "  "},
                       {"startMs": 19800, "endMs": 20000, "speaker": "[STUDENT_B]", "text": "done"}]}


def _wav(sec):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.sin(np.arange(int(sec * 16000)) / 7) * 5000).astype(np.int16).tobytes())
    return b.getvalue()


def test_tasks_skip_blank_lines_keep_context_and_pad_clips():
    tasks = labeling.utterance_tasks("abcdef123", META, lambda n: f"s3://b/{n}.wav")
    assert [t[2]["text"] for t in tasks] == ["what if we tape the wheels", "done"]
    assert tasks[0][:2] == (200, 2800)
    assert tasks[1][1] == 20000, "clips never run past the end of the recording"
    assert tasks[1][2]["context_prev"] == "[?]: what if we tape the wheels"
    assert [o["value"] for o in tasks[0][2]["speaker_options"]] == ["[STUDENT_A]", "[STUDENT_B]", "[OTHER]"]
    assert {t[2]["split"] for t in tasks} == {labeling.split_for("abcdef123")}
    r = labeling.rating_tasks("abcdef123", META, "s3://b/a.wav")
    assert [x["rated"] for x in r] == ["[STUDENT_A]", "[STUDENT_B]"] and "[00:19] [STUDENT_B]: done" in r[0]["transcript"]


def test_cut_wav():
    with wave.open(io.BytesIO(labeling.cut_wav(_wav(3), 1000, 2500))) as w:
        assert (w.getframerate(), w.getnframes()) == (16000, 24000)


def test_prep_labeling_with_mock_r2():
    moto = pytest.importorskip("moto")
    import boto3
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="research")
        for rid, complete in [("rec1", True), ("rec2", False)]:
            s3.put_object(Bucket="research", Key=f"recordings/{rid}/session.json", Body=json.dumps(META).encode())
            if complete:
                s3.put_object(Bucket="research", Key=f"recordings/{rid}/audio.wav", Body=_wav(20))
        st = labeling.prep_labeling("research", s3=s3, log=lambda *_: None)
        assert st == {"recordings": 1, "utterance_tasks": 2, "rating_tasks": 2, "skipped": 0}, "rec2 has no audio yet"
        assert labeling.prep_labeling("research", s3=s3, log=lambda *_: None)["skipped"] == 1
        task = json.loads(s3.get_object(Bucket="research", Key="labeling/utterance-tasks/rec1/0000.json")["Body"].read())
        assert task["data"]["audio"] == "s3://research/labeling/clips/rec1/0000.wav"


def test_import_real_label_studio_export(tmp_path):
    rep = labeling.import_labels(FIX / "ls_utterance_export.json", FIX / "ls_rating_export.json", tmp_path,
                                 default_grade_band="3-5", log=lambda *_: None)
    assert rep["labeled"] == 2 and rep["needs_adjudication"] == 1, "a sentiment disagreement waits for review"
    u = data.load_utterances(tmp_path / "utterances.csv")
    assert list(u["speaker"]) == ["[STUDENT_A]", "[STUDENT_B]"]
    assert list(u["move"]) == ["new_idea", "reasoning"] and u.loc[0, "idea_id"] == "I1"
    r = data.load_segment_ratings(tmp_path / "segments.csv")
    ct = r[r.skill == "critical_thinking"].set_index("rated")["final_level"]
    assert ct["[STUDENT_A]"] == "4" and ct["[STUDENT_B]"] == "2", "3 vs 4 -> rounded mean; 2 vs 2 -> 2"
    assert r[(r.skill == "creativity") & (r.rated == "[STUDENT_B]")]["final_level"].iloc[0] == "NE"
    data.load_pairs(tmp_path / "pairs.csv")


def test_import_needs_grade_band(tmp_path):
    with pytest.raises(ValueError, match="grade band"):
        labeling.import_labels(FIX / "ls_utterance_export.json", None, tmp_path, log=lambda *_: None)


def test_label_studio_configs_use_the_guide_vocabulary():
    import re
    from fourc_models import labels as L
    cfg = (Path(__file__).parent.parent / "labeling" / "utterance_config.xml").read_text()
    def aliases(name):
        block = re.search(rf'<Choices name="{name}".*?</Choices>', cfg, re.S).group(0)
        return re.findall(r'alias="([^"]+)"', block)
    assert aliases("sentiment") == L.SENTIMENT and aliases("move") == L.MOVE
    assert aliases("ct_skill") == L.CT_SKILL and aliases("argument") == L.ARGUMENT
    assert aliases("idea_link") == L.IDEA_LINK


def test_upload_models_to_mock_r2(tmp_path):
    moto = pytest.importorskip("moto")
    import boto3
    f = tmp_path / "speaker.onnx"
    f.write_bytes(b"x" * 1000)
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="models")
        assert labeling.upload_models([f], "models", s3=s3, log=lambda *_: None) == ["speaker.onnx"]
        assert s3.head_object(Bucket="models", Key="speaker.onnx")["ContentLength"] == 1000
        with pytest.raises(ValueError):
            labeling.upload_models([tmp_path / "notes.txt"], "models", s3=s3)


def test_rating_config_rates_all_four_skills():
    import re
    from fourc_models import labels as L
    cfg = (Path(__file__).parent.parent / "labeling" / "rating_config.xml").read_text()
    names = re.findall(r'<Choices name="([^"]+)"', cfg)
    assert names == L.RATED_SKILLS
