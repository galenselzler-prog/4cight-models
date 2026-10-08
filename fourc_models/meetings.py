# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Group-meeting corpora (AMI, ICSI) -> who spoke when -> behavior features.

Why: the app can measure how a group talks (who talks, how long, who talks over
whom, how quickly people answer each other, how often they ask) without keeping
any audio. These corpora are free (CC BY 4.0) and have timed, per-speaker
transcripts, so we can compute the same kinds of numbers on real group talk and
check that they behave sensibly before we have classroom data.

They are ADULTS in work meetings, not students. They tell us whether the
numbers behave sensibly on real conversation; they cannot tell us what a good
student discussion is. Only the annotation files are needed (no audio).

  AMI   Edinburgh/AMI consortium, CC BY 4.0   (ami_public_manual_1.6.2.zip)
  ICSI  ICSI Meeting Corpus, CC BY 4.0 (some annotations; read LICENCE.txt)

Every speaker is de-identified: a letter within the meeting (A, B, C...) and,
for AMI scenario meetings, the assigned role (PM, ME, UI, ID). No real names.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

NITE = "{http://nite.sourceforge.net/}"
_HREF = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")

#: Dialogue-act classes shared by both corpora.
ACTS = ["inform", "suggest", "assess", "question", "agree", "disagree",
        "backchannel", "floor", "fragment", "positive", "negative", "other"]

ACT_COLUMNS = ["corpus", "meeting_id", "meeting_kind", "speaker", "role",
               "start", "end", "text", "act", "act_raw"]

# --- dialogue-act tag -> shared class ---------------------------------------

_AMI_ACT = {
    "bck": "backchannel", "stl": "floor", "fra": "fragment", "inf": "inform",
    "sug": "suggest", "ass": "assess", "el.inf": "question", "el.sug": "question",
    "el.ass": "question", "el.und": "question", "off": "suggest",
    "und": "other", "be.pos": "positive", "be.neg": "negative", "oth": "other",
}


def ami_act(short_name: str) -> str:
    return _AMI_ACT.get(short_name, "other")


def icsi_act(tag: str) -> str:
    """MRDA tags are several marks joined by '^' and '|', e.g. 's^bk|s'. The
    first mark is the main kind (s statement, q question, b backchannel, fh/fg
    floor holder/grabber, h hold, % abandoned); the marks after it refine
    statements (aa/aap agree, ar/arp reject, cs suggestion, ba assessment,
    bk acknowledgement). This is a coarse mapping on purpose."""
    t = (tag or "").strip()
    if not t:
        return "other"
    first = re.split(r"[|^:.]", t.lstrip("%"))[0] if not t.startswith("%") else "%"
    marks = set(re.split(r"[|^:.]", t))
    if t.startswith("%") or first in ("%", "z"):
        return "fragment"
    if first.startswith("q"):
        return "question"
    if first == "b":
        return "backchannel"
    if first in ("fh", "fg", "h"):
        return "floor"
    if "bk" in marks:
        return "backchannel"
    if marks & {"aa", "aap"}:
        return "agree"
    if marks & {"ar", "arp"}:
        return "disagree"
    if "cs" in marks:
        return "suggest"
    if "ba" in marks:
        return "assess"
    if first == "s":
        return "inform"
    return "other"


# --- reading the NXT XML -----------------------------------------------------

class _Words:
    """The words of one speaker's channel, in order, with times."""

    def __init__(self, path: Path):
        self.index: dict[str, int] = {}
        self.text: list[str] = []
        self.start: list[float] = []
        self.end: list[float] = []
        for el in ET.parse(path).getroot():
            if el.tag != "w":
                continue
            word = (el.text or "").strip()
            try:
                s, e = float(el.get("starttime")), float(el.get("endtime"))
            except (TypeError, ValueError):
                s = e = float("nan")
            self.index[el.get(NITE + "id")] = len(self.text)
            # Punctuation keeps its slot (so ranges work) but has no text/time of its own.
            real = any(ch.isalnum() for ch in word)
            self.text.append(word if real else "")
            self.start.append(s if real else float("nan"))
            self.end.append(e if real else float("nan"))

    def indices(self, href: str) -> list[int]:
        m = _HREF.search(href or "")
        if not m:
            return []
        a = self.index.get(m.group(1))
        b = self.index.get(m.group(2)) if m.group(2) else a
        if a is None or b is None:
            return []
        return list(range(min(a, b), max(a, b) + 1))

    def span(self, idx: list[int]) -> tuple[float, float, str]:
        starts = [self.start[i] for i in idx if self.text[i] and self.start[i] == self.start[i]]
        ends = [self.end[i] for i in idx if self.text[i] and self.end[i] == self.end[i]]
        words = [self.text[i] for i in idx if self.text[i]]
        if not starts or not ends:
            return float("nan"), float("nan"), ""
        return min(starts), max(ends), " ".join(words)


def _kind(meeting_id: str) -> str:
    return re.match(r"[A-Za-z]+", meeting_id).group(0) if re.match(r"[A-Za-z]+", meeting_id) else meeting_id


def load_ami(root: str | Path) -> pd.DataFrame:
    """Every dialogue act in the AMI manual annotations, with who, when and what."""
    root = Path(root)
    short: dict[str, str] = {}
    for el in ET.parse(root / "ontologies" / "da-types.xml").getroot().iter("da-type"):
        if el.get("name") and el.get(NITE + "id", "").startswith("ami_da_"):
            short[el.get(NITE + "id")] = el.get("name")
    roles: dict[tuple[str, str], str] = {}
    meetings_xml = root / "corpusResources" / "meetings.xml"
    if meetings_xml.exists():
        for m in ET.parse(meetings_xml).getroot().iter("meeting"):
            for s in m.iter("speaker"):
                roles[(m.get("observation"), s.get("nxt_agent"))] = s.get("role") or ""
    rows = []
    cache: dict[Path, _Words] = {}
    for f in sorted((root / "dialogueActs").glob("*.dialog-act.xml")):
        meeting, agent = f.name.split(".")[:2]
        wpath = root / "words" / f"{meeting}.{agent}.words.xml"
        if not wpath.exists():
            continue
        words = cache.setdefault(wpath, _Words(wpath))
        for dact in ET.parse(f).getroot().iter("dact"):
            tag = ""
            for ptr in dact.findall(NITE + "pointer"):
                m = _HREF.search(ptr.get("href", ""))
                if m and m.group(1).startswith("ami_da_"):
                    tag = short.get(m.group(1), "")
            idx: list[int] = []
            for ch in dact.findall(NITE + "child"):
                idx += words.indices(ch.get("href"))
            start, end, text = words.span(idx)
            if start != start:
                continue
            rows.append(("AMI", meeting, _kind(meeting), agent, roles.get((meeting, agent), ""),
                         start, end, text, ami_act(tag), tag))
    return pd.DataFrame(rows, columns=ACT_COLUMNS)


def load_icsi(root: str | Path) -> pd.DataFrame:
    """Every dialogue act in the ICSI core annotations (point `root` at the ICSI folder)."""
    root = Path(root)
    rows = []
    cache: dict[Path, _Words] = {}
    for f in sorted((root / "DialogueActs").glob("*.dialogue-acts.xml")):
        meeting, chan = f.name.split(".")[:2]
        wpath = root / "Words" / f"{meeting}.{chan}.words.xml"
        if not wpath.exists():
            continue
        words = cache.setdefault(wpath, _Words(wpath))
        for da in ET.parse(f).getroot().iter("dialogueact"):
            idx: list[int] = []
            for ch in da.findall(NITE + "child"):
                idx += words.indices(ch.get("href"))
            start, end, text = words.span(idx)
            if start != start:
                try:
                    start, end = float(da.get("starttime")), float(da.get("endtime"))
                except (TypeError, ValueError):
                    continue
            tag = da.get("type") or ""
            rows.append(("ICSI", meeting, _kind(meeting), chan, "", start, end, text, icsi_act(tag), tag))
    return pd.DataFrame(rows, columns=ACT_COLUMNS)


# --- turns and behavior features --------------------------------------------

def build_turns(acts: pd.DataFrame, pause_s: float = 2.0, talk_over_s: float = 0.2) -> pd.DataFrame:
    """Group one meeting's acts into turns. A turn is one person's run of speech
    with nobody else taking the floor; backchannels ("mm-hm") do not take it.
    `talks_over` marks a turn that starts while someone else is still speaking."""
    turns = []
    cur = None
    for a in acts.sort_values(["start", "end"], kind="stable").itertuples():
        if a.act == "backchannel":
            continue
        if cur is not None and a.speaker == cur["speaker"] and a.start - cur["end"] <= pause_s:
            cur["end"] = max(cur["end"], a.end)
            cur["n_acts"] += 1
            cur["words"] += len(a.text.split())
            cur["questions"] += int(a.act == "question")
            continue
        if cur is not None:
            turns.append(cur)
        prev = cur
        cur = {"speaker": a.speaker, "start": a.start, "end": a.end, "n_acts": 1,
               "words": len(a.text.split()), "questions": int(a.act == "question"),
               "after_other": prev is not None and prev["speaker"] != a.speaker,
               "gap": (a.start - prev["end"]) if prev is not None and prev["speaker"] != a.speaker else np.nan,
               "talks_over": prev is not None and prev["speaker"] != a.speaker and a.start < prev["end"] - talk_over_s}
    if cur is not None:
        turns.append(cur)
    return pd.DataFrame(turns)


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total, cur_end = 0.0, -np.inf
    for s, e in sorted(intervals):
        if e <= cur_end:
            continue
        total += e - max(s, cur_end)
        cur_end = e
    return total


def _overlap_and_silence(intervals_by_speaker: dict[str, list[tuple[float, float]]]) -> tuple[float, float]:
    """(share of speech time with 2+ people talking, share of the meeting nobody talks)."""
    events = []
    for ivs in intervals_by_speaker.values():
        for s, e in ivs:
            events += [(s, 1), (e, -1)]
    if not events:
        return 0.0, 0.0
    events.sort()
    active, last, any_t, multi_t = 0, events[0][0], 0.0, 0.0
    for t, d in events:
        if active >= 1:
            any_t += t - last
        if active >= 2:
            multi_t += t - last
        active += d
        last = t
    span = events[-1][0] - events[0][0]
    return (multi_t / any_t if any_t else 0.0), (1 - any_t / span if span else 0.0)


def behavior_features(acts: pd.DataFrame, min_speakers: int = 2, min_seconds: float = 300.0):
    """(per_speaker, per_meeting) tables of behavior features. Meetings with
    fewer than `min_speakers` talkers or less than `min_seconds` of talk are skipped."""
    sp_rows, mt_rows = [], []
    for (corpus, meeting), g in acts.groupby(["corpus", "meeting_id"], sort=True):
        speakers = sorted(g["speaker"].unique())
        span = g["end"].max() - g["start"].min()
        if len(speakers) < min_speakers or span < min_seconds:
            continue
        iv = {s: g.loc[g["speaker"] == s, ["start", "end"]].itertuples(index=False, name=None) for s in speakers}
        iv = {s: [(a, b) for a, b in v] for s, v in iv.items()}
        talk = {s: _union_length(iv[s]) for s in speakers}
        total_talk = sum(talk.values()) or 1.0
        fair = 1 / len(speakers)
        turns = build_turns(g)
        floor_acts = g[g["act"] != "backchannel"]
        overlap, silence = _overlap_and_silence(
            {s: [(a, b) for a, b in floor_acts.loc[floor_acts["speaker"] == s, ["start", "end"]].itertuples(index=False, name=None)]
             for s in speakers})
        minutes = span / 60
        shares = np.array([talk[s] / total_talk for s in speakers])
        entropy = float(-(shares[shares > 0] * np.log(shares[shares > 0])).sum() / np.log(len(speakers)))
        sorted_shares = np.sort(shares)
        gini = float((2 * np.arange(1, len(shares) + 1) - len(shares) - 1).dot(sorted_shares) / len(shares))
        for s in speakers:
            mine, my_turns = g[g["speaker"] == s], turns[turns["speaker"] == s]
            others_floor_s = total_talk - talk[s]
            share = talk[s] / total_talk
            n_turns = max(len(my_turns), 1)
            responded = my_turns[my_turns["after_other"] & (my_turns["gap"] <= 2.0)]
            sp_rows.append({
                "corpus": corpus, "meeting_id": meeting, "meeting_kind": g["meeting_kind"].iloc[0],
                "speaker": s, "role": mine["role"].iloc[0],
                "talk_s": talk[s], "talk_share": share,
                "share_balance": 1 - min(abs(share - fair) / fair, 1),
                "turns": len(my_turns), "turn_share": len(my_turns) / max(len(turns), 1),
                "mean_turn_s": float((my_turns["end"] - my_turns["start"]).mean()) if len(my_turns) else 0.0,
                "longest_turn_s": float((my_turns["end"] - my_turns["start"]).max()) if len(my_turns) else 0.0,
                "words": int(mine["text"].str.split().str.len().sum()),
                "rate_talk_over": float(my_turns["talks_over"].mean()) if len(my_turns) else 0.0,
                "responds_to_other": len(responded) / n_turns,
                "median_response_gap_s": float(my_turns["gap"].median()) if my_turns["gap"].notna().any() else np.nan,
                "backchannels_per_other_min": float((mine["act"] == "backchannel").sum() / (others_floor_s / 60 or 1)),
                "rate_question": float((mine["act"] == "question").mean()),
                "rate_suggest": float((mine["act"] == "suggest").mean()),
                "rate_agree": float(mine["act"].isin(["agree", "positive"]).mean()),
                "rate_disagree": float(mine["act"].isin(["disagree", "negative"]).mean()),
            })
        mt_rows.append({
            "corpus": corpus, "meeting_id": meeting, "meeting_kind": g["meeting_kind"].iloc[0],
            "speakers": len(speakers), "minutes": minutes,
            "turns_per_min": len(turns) / minutes,
            "top_share": float(shares.max()), "lopsided_index": float((shares.max() - fair) / (1 - fair)),
            "gini": gini, "balance_entropy": entropy,
            "overlap_share": overlap, "silence_share": silence,
            "talk_over_rate": float(turns["talks_over"].mean()) if len(turns) else 0.0,
            "median_response_gap_s": float(turns["gap"].median()) if turns["gap"].notna().any() else np.nan,
            "questions_per_min": float((g["act"] == "question").sum() / minutes),
            "backchannels_per_min": float((g["act"] == "backchannel").sum() / minutes),
        })
    return pd.DataFrame(sp_rows), pd.DataFrame(mt_rows)


# --- the report ---------------------------------------------------------------

def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda v: f"{v:.2f}")


def write_report(per_speaker: pd.DataFrame, per_meeting: pd.DataFrame, out_dir: str | Path) -> str:
    """Write speakers.csv, meetings.csv and report.txt; return the report text."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_speaker.to_csv(out / "speakers.csv", index=False)
    per_meeting.to_csv(out / "meetings.csv", index=False)
    cols = ["turns_per_min", "top_share", "lopsided_index", "balance_entropy", "overlap_share",
            "silence_share", "talk_over_rate", "median_response_gap_s", "questions_per_min", "backchannels_per_min"]
    lines = [
        "Behavior features on public group-meeting corpora (adults, not students).",
        "AMI Meeting Corpus and ICSI Meeting Corpus, CC BY 4.0 - cite the corpora if you publish.",
        "",
        f"{len(per_meeting)} meetings, {len(per_speaker)} speakers "
        f"({', '.join(f'{c}: {n}' for c, n in per_meeting['corpus'].value_counts().items())})",
        "",
        "1) How the numbers are spread (per meeting)",
        _fmt(per_meeting.groupby("corpus")[cols].describe().T.loc[(slice(None), ["mean", "25%", "50%", "75%"]), :]),
        "",
        "2) Most lopsided meetings: one person dominates (check these against talk_share below)",
    ]
    lop = per_meeting.sort_values("lopsided_index", ascending=False).head(5)
    lines.append(_fmt(lop[["corpus", "meeting_id", "speakers", "minutes", "top_share", "lopsided_index", "turns_per_min"]]))
    lines += ["", "3) Most balanced meetings"]
    bal = per_meeting.sort_values("lopsided_index").head(5)
    lines.append(_fmt(bal[["corpus", "meeting_id", "speakers", "minutes", "top_share", "lopsided_index", "turns_per_min"]]))
    lines += ["", "4) Busiest and quietest conversation (turns per minute)"]
    lines.append(_fmt(per_meeting.sort_values("turns_per_min", ascending=False).head(3)[["corpus", "meeting_id", "turns_per_min", "overlap_share", "median_response_gap_s"]]))
    lines.append(_fmt(per_meeting.sort_values("turns_per_min").head(3)[["corpus", "meeting_id", "turns_per_min", "overlap_share", "median_response_gap_s"]]))
    roles = per_speaker[per_speaker["role"] != ""]
    if len(roles):
        lines += ["", "5) Sanity check - AMI scenario roles (PM = project manager, who runs the meeting; "
                      "ME marketing, UI user-interface, ID industrial designer)",
                  _fmt(roles.groupby("role")[["talk_share", "turn_share", "mean_turn_s", "rate_question", "rate_suggest"]].mean())]
    lines += ["", "6) Do the numbers move together the way you'd expect? (correlations, per meeting)",
              _fmt(per_meeting[["lopsided_index", "balance_entropy", "turns_per_min", "overlap_share", "talk_over_rate",
                                "questions_per_min"]].corr())]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    return text
