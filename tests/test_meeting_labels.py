# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Mapping AMI/ICSI annotations onto the 4Cight label vocabulary (weak labels)."""

import pandas as pd

from fourc_models import labels as L
from fourc_models import meeting_labels as ML
from fourc_models.meetings import ACTS


def test_dialogue_acts_map_to_moves():
    w = ML.weak_labels
    assert w("question", "who did that")["move"] == "question"
    assert w("suggest", "we could use a curved case")["move"] == "new_idea"
    assert w("inform", "because it is cheaper")["move"] == "reasoning"
    assert w("backchannel", "mm hm")["move"] == "other"
    assert w("inform", "we meet on friday")["move"] == "", "plain statements stay unknown"
    assert w("inform", "so that's the point")["move"] == "", "'so that's' is not a reason"


def test_argument_annotation_drives_argument_and_builds_on():
    w = ML.weak_labels
    assert w("inform", "it is too heavy for that", "STA", "NEG>B")["argument"] == "challenge"
    assert w("inform", "and it is also red and shiny", "STA", "ELA>B")["move"] == "builds_on"
    assert w("inform", "and it is also red and shiny", "STA", "ELA>B")["argument"] == "evidence"
    assert w("inform", "yes", "STA", "ELA>B")["move"] != "builds_on", "one word cannot build on anything"
    assert w("inform", "which one do we pick", "OIS")["move"] == "question"
    assert w("disagree", "no")["argument"] == "", "a bare 'no' is a reply, not a challenge"
    assert w("disagree", "no that will not fit")["argument"] == "challenge"


def test_unsupported_labels_are_empty_and_values_stay_in_vocabulary():
    for act in ACTS:
        for text in ("", "hello there everyone", "because it works", "i don't know"):
            r = ML.weak_labels(act, text)
            assert r["move"] in [""] + L.MOVE and r["argument"] in [""] + L.ARGUMENT
            assert r["ct_skill"] in [""] + L.CT_SKILL and r["sentiment"] in [""] + L.SENTIMENT
    assert ML.weak_labels("inform", "we meet on friday")["sentiment"] == ""


def test_split_keeps_a_team_together_and_is_stable():
    assert ML.split_for("ES2002") == ML.split_for("ES2002")
    assert {ML.split_for(f"X{i}") for i in range(200)} == {"train", "validation", "test"}
    assert ML._group("AMI", "ES2002a") == ML._group("AMI", "ES2002d") == "ES2002"
    assert ML._group("ICSI", "Bed003") == "Bed003"


def test_link_is_credited_to_the_later_speaker():
    acts = pd.DataFrame([
        ("AMI", "M1", "M", "A", "", 10.0, 12.0, "we should use rubber", "suggest", "sug"),
        ("AMI", "M1", "M", "B", "", 20.0, 23.0, "and it could be blue", "inform", "inf"),
    ], columns=["corpus", "meeting_id", "meeting_kind", "speaker", "role", "start", "end", "text", "act", "act_raw"])
    structs = pd.DataFrame([("sA", "M1", "A", "STA", 10.0, 12.0), ("sB", "M1", "B", "STA", 20.0, 23.0)],
                           columns=["struct_id", "meeting_id", "speaker", "kind", "start", "end"])
    rels = pd.DataFrame([("sA", "sB", "ELA")], columns=["source", "target", "rel"])  # file order: earlier -> later
    out = ML.attach_arguments(acts, structs, rels)
    assert out.loc[0, "arg_links"] == "" and out.loc[1, "arg_links"] == "ELA>A"
    assert out.loc[1, "arg_types"] == "STA"
