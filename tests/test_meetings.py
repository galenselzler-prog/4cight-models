# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""AMI/ICSI loaders and behavior features, on tiny files in the corpora's real
NXT format (layouts copied from the public annotation releases)."""

import numpy as np
import pandas as pd

from fourc_models import meetings as M

NS = 'xmlns:nite="http://nite.sourceforge.net/"'


def _words(path, prefix, words, idfmt):
    body = "\n".join(
        f'<w nite:id="{idfmt.format(i)}" starttime="{s}" endtime="{e}">{t}</w>' for i, (t, s, e) in enumerate(words))
    path.write_text(f'<?xml version="1.0" encoding="ISO-8859-1"?>\n<nite:root nite:id="{prefix}" {NS}>\n{body}\n</nite:root>')


def _ami(tmp_path):
    root = tmp_path / "ami"
    for d in ("words", "dialogueActs", "ontologies", "corpusResources"):
        (root / d).mkdir(parents=True)
    (root / "ontologies/da-types.xml").write_text(
        f'<da-type nite:id="cmrda" {NS}><da-type nite:id="ami_da_1" name="bck"/>'
        '<da-type nite:id="ami_da_4" name="inf"/><da-type nite:id="ami_da_5" name="el.inf"/></da-type>')
    (root / "corpusResources/meetings.xml").write_text(
        f'<nite:root {NS}><meeting observation="ES9999a"><speaker nxt_agent="A" role="PM"/>'
        '<speaker nxt_agent="B" role="UI"/></meeting></nite:root>')
    # A talks 0-10 s then 12-20 s; B says "mm-hm" at 5 s, asks a question at 9.5 s (over A), talks to 30 s.
    _words(root / "words/ES9999a.A.words.xml", "a", [("we", 0, 4), ("need", 4, 10), (".", 10, 10), ("a", 12, 16), ("plan", 16, 20)],
           "ES9999a.A.words{}")
    _words(root / "words/ES9999a.B.words.xml", "b", [("mm", 5, 5.5), ("why", 9.5, 11), ("because", 21, 30)],
           "ES9999a.B.words{}")
    da = lambda i, t, a, b, f: (f'<dact nite:id="x.{i}"><nite:pointer role="da-aspect" href="da-types.xml#id({t})"/>'
                                f'<nite:child href="ES9999a.{f}.words.xml#id(ES9999a.{f}.words{a})..id(ES9999a.{f}.words{b})"/></dact>')
    (root / "dialogueActs/ES9999a.A.dialog-act.xml").write_text(
        f'<nite:root {NS}>{da(1, "ami_da_4", 0, 2, "A")}{da(2, "ami_da_4", 3, 4, "A")}</nite:root>')
    (root / "dialogueActs/ES9999a.B.dialog-act.xml").write_text(
        f'<nite:root {NS}>{da(1, "ami_da_1", 0, 0, "B")}{da(2, "ami_da_5", 1, 1, "B")}{da(3, "ami_da_4", 2, 2, "B")}</nite:root>')
    return root


def _icsi(tmp_path):
    root = tmp_path / "ICSI"
    for d in ("Words", "DialogueActs"):
        (root / d).mkdir(parents=True)
    _words(root / "Words/Bxx001.A.words.xml", "a", [("hi", 1, 2), ("there", 2, 3)], "Bxx001.w.{}")
    (root / "DialogueActs/Bxx001.A.dialogue-acts.xml").write_text(
        f'<nite:root {NS}><dialogueact nite:id="d1" starttime="1" endtime="3" type="qy^d" participant="mn1">'
        '<nite:child href="Bxx001.A.words.xml#id(Bxx001.w.0)..id(Bxx001.w.1)"/></dialogueact></nite:root>')
    return root


def test_ami_loader_reads_text_times_roles_and_acts(tmp_path):
    a = M.load_ami(_ami(tmp_path))
    assert len(a) == 5
    first = a[(a.speaker == "A")].iloc[0]
    assert (first.text, first.start, first.end, first.role, first.act) == ("we need", 0, 10, "PM", "inform")
    b = a[a.speaker == "B"].set_index("act")
    assert set(b.index) == {"backchannel", "question", "inform"}
    assert b.loc["question", "text"] == "why"


def test_icsi_loader_and_tag_mapping(tmp_path):
    i = M.load_icsi(_icsi(tmp_path))
    assert i.iloc[0].act == "question" and i.iloc[0].text == "hi there"
    for tag, want in [("b", "backchannel"), ("s^bk", "backchannel"), ("s^aa", "agree"), ("s^ar", "disagree"),
                      ("s^cs", "suggest"), ("fh", "floor"), ("%--", "fragment"), ("qw", "question"), ("s", "inform")]:
        assert M.icsi_act(tag) == want, tag


def test_turns_ignore_backchannels_and_flag_talking_over(tmp_path):
    a = M.load_ami(_ami(tmp_path))
    t = M.build_turns(a)
    # A (0-10), B's question starts at 9.5 while A is still talking, A again, B again.
    assert list(t["speaker"]) == ["A", "B", "A", "B"]
    assert bool(t.loc[1, "talks_over"]) and not bool(t.loc[3, "talks_over"])
    assert t.loc[0, "end"] == 10 and t.loc[2, "start"] == 12
    assert np.isnan(t.loc[0, "gap"]) and t.loc[2, "gap"] == 1.0, "A starts 1 s after B's question ends (11 s)"


def test_features_shares_add_up(tmp_path):
    a = M.load_ami(_ami(tmp_path))
    sp, mt = M.behavior_features(a, min_seconds=10)
    assert np.isclose(sp["talk_share"].sum(), 1.0)
    assert mt.iloc[0]["speakers"] == 2 and 0 <= mt.iloc[0]["lopsided_index"] <= 1
    assert sp.set_index("speaker").loc["B", "rate_question"] > 0
