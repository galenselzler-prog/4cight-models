# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""One-microphone behavior features and scoring. The golden file is shared with
the app: ils-app/engine-tests/behavior.test.cjs must reproduce the same numbers.
Regenerate (and copy to the app) with:  python tests/test_behavior.py"""

import json
import math
import shutil
from pathlib import Path

from fourc_models import behavior as B

HERE = Path(__file__).parent
GOLDEN = HERE / "fixtures" / "behavior_golden.json"
APP_GOLDEN = HERE.parent.parent / "ils-app" / "engine-tests" / "fixtures" / "behavior_golden.json"
APP_NORMS = HERE.parent.parent / "ils-app" / "src" / "engine" / "reference" / "meetingNorms.ts"

CASES = {
    "two_alternate": {"speakers": ["a", "b"], "segments": [
        ["a", 0, 8], ["b", 8.5, 16], ["a", 16.5, 24], ["b", 25, 33], ["a", 33.2, 41], ["b", 42, 50]]},
    "one_dominates": {"speakers": ["a", "b", "c", "d"], "segments": [
        ["a", 0, 60], ["b", 61, 66], ["a", 67, 130], ["c", 131, 135], ["a", 136, 200], ["d", 203, 206], ["a", 207, 290]]},
    "silent_member": {"speakers": ["a", "b", "c"], "segments": [
        ["a", 0, 20], ["b", 21, 41], ["a", 42, 60], ["b", 61, 80], ["a", 81, 100]]},
    "pauses_and_merges": {"speakers": ["a", "b"], "segments": [
        ["a", 0, 5], ["a", 6.9, 10], ["b", 10.5, 20], ["a", 25, 30], ["a", 33.5, 40], ["b", 41, 47]]},
    "ignores_strangers": {"speakers": ["a", "b"], "segments": [
        ["a", 0, 30], ["x", 30, 60], ["b", 61, 90], ["a", 91, 120]]},
}


def _norms():
    text = APP_NORMS.read_text()
    return json.loads(text[text.index("= {") + 2: text.rindex("};") + 1])


def compute(case, norms):
    segs = [tuple(s) for s in case["segments"]]
    g, per = B.session_features(segs, case["speakers"])
    band = B.pick_band(norms, g["minutes"])
    return {"group": g, "speakers": per, "band": band["name"],
            "group_score": B.score_group(g, band),
            "person_scores": {k: B.score_person(v, band) for k, v in per.items()}}


def golden():
    norms = _norms()
    return {"cases": CASES, "expected": {k: compute(c, norms) for k, c in CASES.items()}}


def test_percentile_handles_ties_edges_and_the_middle():
    q = [0, 0, 0, 1, 2, 3, 3]
    assert B.percentile(-5, q) == 0.0 and B.percentile(9, q) == 1.0
    assert B.percentile(0, q) == 1 / 6          # middle of the run of zeros (indices 0-2)
    assert math.isclose(B.percentile(1.5, q), (3 + 0.5) / 6)


def test_collapse_keeps_first_speaker_and_drops_slivers():
    out = B.collapse_overlaps([("a", 0, 10), ("b", 5, 12), ("c", 6, 6.1)])
    assert out == [("a", 0, 10), ("b", 10, 12)]


def test_features_make_sense():
    g, per = B.session_features([("a", 0, 60), ("b", 61, 66), ("a", 67, 130)], ["a", "b", "c"])
    assert g["participation_rate"] == 2 / 3 and per["c"]["share_balance"] == 0.0
    assert g["top_share"] > 0.9 and g["reply_rate"] == 1.0
    turns = B.turns_from([("a", 0, 5), ("a", 6.9, 10), ("b", 10.5, 20), ("a", 25, 30)])
    assert [(t[0], t[1], t[2]) for t in turns] == [("a", 0, 10), ("b", 10.5, 20), ("a", 25, 30)]
    assert turns[1][3] == 0.5 and turns[2][3] == 5


def test_lopsided_scores_lower_than_even():
    if not APP_NORMS.exists():
        return
    n = _norms()
    even = compute(CASES["two_alternate"], n)["group_score"]
    lop = compute(CASES["one_dominates"], n)["group_score"]
    assert lop["communication"] < even["communication"]


def test_golden_file_is_up_to_date():
    if not APP_NORMS.exists():
        return
    assert json.loads(GOLDEN.read_text()) == json.loads(json.dumps(golden())), \
        "behavior.py or the norms changed: run `python tests/test_behavior.py` and commit the golden files"


if __name__ == "__main__":
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(golden(), indent=1, sort_keys=True) + "\n")
    APP_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(GOLDEN, APP_GOLDEN)
    print("wrote", GOLDEN, "and", APP_GOLDEN)
