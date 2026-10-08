# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""AMI/ICSI -> weak utterance labels in the 4Cight vocabulary (labels.py).

Why: the utterance model (move, argument, CT skill, sentiment) normally learns
from classroom utterances that people label by hand. AMI and ICSI already carry
expert annotations of what each statement DOES in the conversation (suggests,
asks, agrees, disagrees, elaborates, argues against ...). Mapping those onto our
labels gives >100k utterances to pretrain on, so fewer classroom labels are
needed later.

These are WEAK labels:
  * adults in work meetings, not students;
  * the mapping is a judgement call, written out below so it can be argued with;
  * a label we cannot support is left EMPTY (= unknown), never guessed. Empty
    cells must be masked, not treated as a class, when training on this file.

AMI additionally has human-annotated argument structure (statements, issues,
and typed links POS/NEG/UNC/OPT/ELA/SPE/SUB/EXC/REQ between speakers' points).
That is used where it exists; ICSI only has dialogue acts.

Nothing here ships in the app. Output is training data only.
"""

from __future__ import annotations

import bisect
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from . import labels as L
from .meetings import NITE, _Words, load_ami, load_icsi

WEAK_COLUMNS = [
    "utterance_id", "session_id", "corpus", "speaker", "start", "end", "text", "context_prev",
    "act", "act_raw", "arg_types", "arg_links",
    "sentiment", "move", "ct_skill", "argument",
    "grade_band", "split", "label_source", "guide_version",
]

# Cue words. Deliberately short and conservative; each is a plain English phrase
# a person could check by eye in the report's samples.
_REASON = re.compile(
    r"\b(because|therefore|which means|that means|that's why|that is why|the reason|"
    r"in order to|as a result|due to)\b", re.I)
_EXAMPLE = re.compile(r"\b(for example|for instance|such as|e\.g\.|like when)\b", re.I)
_STUCK = re.compile(r"\b(i don't know|i do not know|no idea|i'm not sure|i am not sure|i don't understand|"
                    r"i don't get it|i'm confused|i'm lost)\b", re.I)
_COORD = re.compile(r"\b(who wants to|who's going to|who is going to|shall we move|next topic|moving on|"
                    r"let's move on|let's start|let's get started|next item|on the agenda|can you take|"
                    r"you take notes|time check|we have \w+ minutes)\b", re.I)

MIN_CLAIM_WORDS = 4  # a "claim" needs something to say


def _words(text: str) -> int:
    return len(text.split())


# --- AMI argument annotation ---------------------------------------------------

_ID = re.compile(r"#id\(([^)]+)\)")


def _type_names(path: Path, tag: str) -> dict[str, str]:
    out = {}
    for el in ET.parse(path).getroot().iter(tag):
        if el.get("name") and el.get(NITE + "id"):
            out[el.get(NITE + "id")] = el.get("name")
    return out


def load_ami_arguments(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(structs, relations). structs: one row per annotated statement/issue with
    its speaker, time span and type (STA, WST, OIS, AIS, YIS, OTH). relations:
    source struct -> target struct with a type (POS NEG UNC OPT ELA SPE SUB EXC REQ)."""
    root = Path(root)
    ae = _type_names(root / "ontologies" / "ae-types.xml", "ae-type")
    ar = _type_names(root / "ontologies" / "ar-types.xml", "ar-type")
    rows, cache = [], {}
    for f in sorted((root / "argumentation" / "ae").glob("*.argumentstructs.xml")):
        meeting, agent = f.name.split(".")[:2]
        wpath = root / "words" / f"{meeting}.{agent}.words.xml"
        if not wpath.exists():
            continue
        words = cache.setdefault(wpath, _Words(wpath))
        for el in ET.parse(f).getroot().iter("ae"):
            kind = ""
            for ptr in el.findall(NITE + "pointer"):
                m = _ID.search(ptr.get("href", ""))
                if ptr.get("role") == "type" and m:
                    kind = ae.get(m.group(1), "")
            idx: list[int] = []
            for ch in el.findall(NITE + "child"):
                idx += words.indices(ch.get("href"))
            start, end, _ = words.span(idx)
            if start != start:
                continue
            rows.append((el.get(NITE + "id"), meeting, agent, kind, start, end))
    structs = pd.DataFrame(rows, columns=["struct_id", "meeting_id", "speaker", "kind", "start", "end"])
    rels = []
    for f in sorted((root / "argumentation" / "ar").glob("*.argumentationrels.xml")):
        for el in ET.parse(f).getroot().iter("ar"):
            ends = {}
            for ptr in el.findall(NITE + "pointer"):
                m = _ID.search(ptr.get("href", ""))
                if m:
                    ends[ptr.get("role")] = m.group(1)
            if {"source", "target", "type"} <= ends.keys():
                rels.append((ends["source"], ends["target"], ar.get(ends["type"], "")))
    return structs, pd.DataFrame(rels, columns=["source", "target", "rel"])


def attach_arguments(acts: pd.DataFrame, structs: pd.DataFrame, rels: pd.DataFrame) -> pd.DataFrame:
    """Add arg_types ('STA|OIS') and arg_links ('NEG>B', 'ELA>A') to each AMI dialogue act.
    A struct and a dialogue act of the same speaker are matched when they overlap by at least half of the shorter one.
    Link text is REL>target speaker, only when the target is somebody else's earlier point."""
    info = {r.struct_id: (r.speaker, r.start) for r in structs.itertuples(index=False)}
    out_links: dict[str, list[tuple[str, str]]] = {}
    for s, t, r in rels.itertuples(index=False):
        if s in info and t in info:
            (s_spk, s_start), (t_spk, t_start) = info[s], info[t]
            # Only links between two different people count. The file's source/target
            # order does not say who answered whom (it is mostly earlier -> later), so
            # the LATER point is the response and the link is credited to it.
            if t_spk != s_spk:
                later, earlier_spk = (s, t_spk) if s_start >= t_start else (t, s_spk)
                out_links.setdefault(later, []).append((r, earlier_spk))
    by_key: dict[tuple, tuple[list[float], list]] = {}
    for key, g in structs.sort_values("start").groupby(["meeting_id", "speaker"]):
        by_key[key] = (g["start"].tolist(), list(g.itertuples(index=False)))
    types, links = [], []
    for r in acts.itertuples(index=False):
        t_set, l_set = [], []
        if r.corpus == "AMI" and (r.meeting_id, r.speaker) in by_key:
            starts, items = by_key[(r.meeting_id, r.speaker)]
            i = bisect.bisect_left(starts, r.start - 30)  # structs are short; start near the act
            while i < len(items) and items[i].start <= r.end:
                s = items[i]
                # covers at least half of the shorter of the two (a struct can span several acts)
                dur = max(min(s.end - s.start, r.end - r.start), 1e-6)
                overlap = min(s.end, r.end) - max(s.start, r.start)
                if overlap / dur >= 0.5:
                    t_set.append(s.kind)
                    l_set += [f"{rel}>{spk}" for rel, spk in out_links.get(s.struct_id, [])]
                i += 1
        types.append("|".join(sorted(set(t_set))))
        links.append("|".join(sorted(set(l_set))))
    acts = acts.copy()
    acts["arg_types"], acts["arg_links"] = types, links
    return acts


# --- the mapping ----------------------------------------------------------------

def weak_labels(act: str, text: str, arg_types: str = "", arg_links: str = "") -> dict[str, str]:
    """One utterance's weak labels. Empty string = unknown.

    MOVE (what the turn does)
      question     dialogue act question, or an AMI issue (open / A-B / yes-no)
      builds_on    AMI: elaborates/specializes/qualifies (ELA/SPE/SUB) someone else's earlier point
      new_idea     dialogue act suggest, or AMI option (OPT/EXC) link
      reasoning    says WHY (because, therefore, ...)
      coordinates  runs the meeting (who's going to, moving on, ...)
      stuck        says they do not know / are lost
      other        backchannel, floor-holding, fragments, agree/disagree/assess without more
      (off_task is never produced: these meetings have no such annotation)
    ARGUMENT
      challenge    AMI NEG link to someone else, or dialogue act disagree
      evidence     AMI ELA/SPE/SUB link, or an explicit example
      claim        a suggestion/assessment/statement of >= 4 words that is an AMI statement
                   (STA), or a suggestion
      none         backchannel, floor, fragment, agree
      (revision is never produced)
    CT_SKILL    evaluation for assess/challenge, explanation for "why" statements, none for
                backchannel/floor/fragment; everything else unknown
    SENTIMENT   positive/negative only from the explicit be.pos / be.neg acts
    """
    t = text or ""
    kinds = set(filter(None, arg_types.split("|")))
    rels = {x.split(">")[0] for x in filter(None, arg_links.split("|"))}
    no_content = act in ("backchannel", "floor", "fragment")
    reason = bool(_REASON.search(t))
    example = bool(_EXAMPLE.search(t))

    # move
    if act == "question" or kinds & {"OIS", "AIS", "YIS"}:
        move = "question"
    elif rels & {"ELA", "SPE", "SUB"} and _words(t) >= MIN_CLAIM_WORDS:
        move = "builds_on"
    elif act == "suggest" or rels & {"OPT", "EXC"}:
        move = "new_idea"
    elif reason and not no_content:
        move = "reasoning"
    elif _COORD.search(t) and not no_content:
        move = "coordinates"
    elif _STUCK.search(t) and not no_content:
        move = "stuck"
    elif no_content or act in ("agree", "disagree", "assess", "positive", "negative"):
        move = "other"
    else:
        move = ""  # a plain statement: could be anything

    # argument
    if "NEG" in rels or (act == "disagree" and _words(t) >= 3):  # a bare "no" is a reply, not an argument
        argument = "challenge"
    elif rels & {"ELA", "SPE", "SUB"} or example:
        argument = "evidence"
    elif (act == "suggest" or "STA" in kinds) and _words(t) >= MIN_CLAIM_WORDS and not no_content:
        argument = "claim"
    elif no_content or act == "agree":
        argument = "none"
    else:
        argument = ""

    # ct skill
    if no_content:
        ct = "none"
    elif argument == "challenge" or act == "assess":
        ct = "evaluation"
    elif reason:
        ct = "explanation"
    else:
        ct = ""

    sentiment = {"positive": "positive", "negative": "negative"}.get(act, "")
    assert move in ("",) + tuple(L.MOVE) and argument in ("",) + tuple(L.ARGUMENT)
    assert ct in ("",) + tuple(L.CT_SKILL) and sentiment in ("",) + tuple(L.SENTIMENT)
    return {"sentiment": sentiment, "move": move, "ct_skill": ct, "argument": argument}


def split_for(group: str) -> str:
    """70/15/15 by hash, one split per GROUP so the same team or meeting never straddles splits."""
    h = int(hashlib.sha1(group.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("validation" if h < 85 else "test")


def _group(corpus: str, meeting: str) -> str:
    # AMI sessions a-d are the same four people on one design project.
    return meeting[:-1] if corpus == "AMI" else meeting


def build_weak_labels(ami_root=None, icsi_root=None) -> pd.DataFrame:
    parts = []
    if ami_root:
        acts = load_ami(ami_root)
        structs, rels = load_ami_arguments(ami_root)
        parts.append(attach_arguments(acts, structs, rels))
    if icsi_root:
        acts = load_icsi(icsi_root)
        acts["arg_types"] = ""
        acts["arg_links"] = ""
        parts.append(acts)
    if not parts:
        raise ValueError("give at least one of --ami / --icsi")
    df = pd.concat(parts, ignore_index=True)
    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df = df.sort_values(["corpus", "meeting_id", "start", "end"]).reset_index(drop=True)

    labs = [weak_labels(a, t, k, r) for a, t, k, r in zip(df["act"], df["text"], df["arg_types"], df["arg_links"])]
    for h in ("sentiment", "move", "ct_skill", "argument"):
        df[h] = [x[h] for x in labs]

    # previous two substantive turns in the same meeting (backchannels carry no context)
    ctx = []
    for _, g in df.groupby(["corpus", "meeting_id"], sort=False):
        prev: list[str] = []
        for text, act in zip(g["text"], g["act"]):
            ctx.append(" ".join(prev[-2:]))
            if act not in ("backchannel", "floor", "fragment"):
                prev.append(text)
    df["context_prev"] = ctx

    df["session_id"] = df["corpus"] + "-" + df["meeting_id"]
    df["utterance_id"] = df["session_id"] + "-" + (df.groupby("session_id").cumcount() + 1).astype(str)
    df["grade_band"] = "adult"
    df["split"] = [split_for(_group(c, m)) for c, m in zip(df["corpus"], df["meeting_id"])]
    df["label_source"] = ["weak:ami-da+argument" if a else f"weak:{c.lower()}-da"
                          for c, a in zip(df["corpus"], df["arg_types"] != "")]
    df["guide_version"] = L.GUIDE_VERSION
    return df[WEAK_COLUMNS]


# --- report ---------------------------------------------------------------------

def write_label_report(df: pd.DataFrame, out_dir: str | Path, samples: int = 6, seed: int = 0) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(df)
    lines = [
        "Weak utterance labels from AMI + ICSI (adults in work meetings; labels are MAPPED, not hand-labeled).",
        "Empty = unknown, to be masked in training. Read the samples; the mapping is a judgement call.",
        "",
        f"{n:,} utterances from {df['session_id'].nunique()} meetings "
        f"({', '.join(f'{c}: {k:,}' for c, k in df['corpus'].value_counts().items())})",
        f"AMI utterances with argument annotation: {int((df['arg_types'] != '').sum()):,} "
        f"({(df['arg_types'] != '').sum() / max(1, (df['corpus'] == 'AMI').sum()):.0%} of AMI)",
        f"split by meeting/team: {', '.join(f'{s}: {k:,}' for s, k in df['split'].value_counts().items())}",
        "",
    ]
    for head in ("move", "argument", "ct_skill", "sentiment"):
        vc = df[head].replace("", "(unknown)").value_counts()
        known = (df[head] != "").mean()
        lines.append(f"{head}  - known for {known:.0%} of utterances")
        for lab, k in vc.items():
            lines.append(f"    {lab:<16}{k:>9,}  {k / n:6.1%}")
        lines.append("")
    lines.append("Do the labels agree with dialogue acts the way you'd expect? (rows = move)")
    lines.append(pd.crosstab(df["move"].replace("", "(unknown)"), df["act"]).to_string())
    lines.append("")
    lines.append("Samples to eyeball (random; text only, speakers are de-identified letters)")
    rng_df = df.sample(frac=1.0, random_state=seed)
    for head in ("move", "argument"):
        for lab in [x for x in getattr(L, head.upper()) if x in set(df[head])]:
            lines.append(f"  [{head} = {lab}]")
            for r in rng_df[rng_df[head] == lab].head(samples).itertuples():
                lines.append(f"    - ({r.act}) {r.text[:140]}")
    text = "\n".join(lines)
    (out / "labels_report.txt").write_text(text + "\n")
    return text


def export(ami_root, icsi_root, out_dir: str | Path) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_weak_labels(ami_root, icsi_root)
    df.to_csv(out / "meeting_weak_labels.csv", index=False)
    write_label_report(df, out)
    return df
