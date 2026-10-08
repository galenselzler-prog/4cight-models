# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Group behavior features that a single microphone can measure, and a score that
says where a group falls among real meetings. No human rater is involved.

This file is the REFERENCE for the app: src/engine/behavior.ts must compute the
same numbers (tests/fixtures/behavior_golden.json is checked by both).

What one microphone can tell us (and what it can't)
  The app hears the room through one microphone and works out who is speaking
  from their enrolled voice. It can measure how long each person talks, how long
  their turns run, how quickly one person follows another, how many turns happen
  per minute, and how much of the time nobody speaks. It cannot reliably measure
  two people talking at the same moment, so overlap and "mm-hm" backchannels are
  NOT used. The AMI/ICSI recordings have one microphone per person, so before
  computing norms their speech is collapsed onto one timeline the way a single
  microphone would hear it (collapse_overlaps).

What the score means
  A score is NORM-REFERENCED: it says how this group's participation compares
  with about 170 hours of real adult meetings (AMI, ICSI), 1-10 where about 5 is a
  typical meeting. Which direction is "better" for each number is a design
  assumption (listed in DIRECTION and in the score formulas), not something
  learned from human ratings. It measures participation and interaction, not the
  quality of what was said. Critical thinking and creativity are not scored here.
"""

from __future__ import annotations

import json
import math
from typing import Iterable

import numpy as np

REPLY_WITHIN_S = 2.0          # a turn that starts within this of someone else's end is a reply
MERGE_GAP_S = 2.0             # same person pausing less than this is still one turn
MIN_BURST_S = 0.25            # shortest stretch of speech (matches the app's detector)
BRIDGE_S = 0.30               # pauses shorter than this do not split speech (matches the app)
PRESENT_S = 5.0               # talked at least this long = took part
QUANTILES = 21                # norm tables: 0%, 5%, ... 100%
NORMS_VERSION = "meetings-0.1.0"

GROUP_FEATURES = ["balance_entropy", "top_share", "participation_rate", "turns_per_min", "reply_rate", "silence_share"]
SPEAKER_FEATURES = ["share_balance", "turns_per_min", "reply_rate"]

#: +1 = a higher number is better, -1 = lower is better. A design assumption.
DIRECTION = {"balance_entropy": 1, "top_share": -1, "participation_rate": 1, "turns_per_min": 1,
             "reply_rate": 1, "silence_share": -1, "share_balance": 1}

Segment = tuple  # (speaker, start_s, end_s)


# --- turning the corpora's per-person channels into one-microphone speech ----------

def bursts(word_times: Iterable[tuple[float, float]], bridge: float = BRIDGE_S, min_len: float = MIN_BURST_S):
    """Words of ONE speaker -> stretches of speech, bridging pauses shorter than `bridge`."""
    out, cur = [], None
    for s, e in sorted(word_times):
        if cur is not None and s - cur[1] <= bridge:
            cur[1] = max(cur[1], e)
        else:
            if cur is not None:
                out.append(tuple(cur))
            cur = [s, e]
    if cur is not None:
        out.append(tuple(cur))
    return [(s, e) for s, e in out if e - s >= min_len]


def collapse_overlaps(segments: list[Segment], min_len: float = MIN_BURST_S) -> list[Segment]:
    """One microphone hears one voice at a time: where two people overlap, the
    person who started first keeps the time and the later start is trimmed
    (dropped if what is left is shorter than `min_len`, e.g. a quick "mm-hm")."""
    out: list[Segment] = []
    cover = -math.inf
    for spk, s, e in sorted(segments, key=lambda x: (x[1], x[2])):
        s2 = max(s, cover)
        if e - s2 >= min_len:
            out.append((spk, s2, e))
            cover = max(cover, e)
    return out


# --- the features (the app computes exactly these) ----------------------------------

def turns_from(segments: list[Segment], merge_gap: float = MERGE_GAP_S):
    """[(speaker, start, end, gap_after_other)] — a run of one person's speech with
    no one else in between; gap is None unless the previous turn was someone else's."""
    turns: list[list] = []
    for spk, s, e in sorted(segments, key=lambda x: (x[1], x[2])):
        if turns and turns[-1][0] == spk and s - turns[-1][2] <= merge_gap:
            turns[-1][2] = max(turns[-1][2], e)
        else:
            prev = turns[-1] if turns else None
            gap = (s - prev[2]) if prev is not None and prev[0] != spk else None
            turns.append([spk, s, e, gap])
    return [tuple(t) for t in turns]


def _entropy(shares: list[float]) -> float:
    n = len(shares)
    if n < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in shares if p > 0)
    return h / math.log(n)


def session_features(segments: list[Segment], speakers: list[str]):
    """(group, per_speaker) features for the speech heard so far.
    `speakers` is everyone enrolled (people who have not spoken count as silent).
    Speech from anyone not in `speakers` is ignored."""
    ids = list(dict.fromkeys(speakers))
    segs = [(s, a, b) for s, a, b in segments if s in set(ids)]
    if not segs:
        return None, {}
    n = len(ids)
    talk = {i: 0.0 for i in ids}
    for spk, a, b in segs:
        talk[spk] += b - a
    total = sum(talk.values())
    first, last = min(a for _, a, _ in segs), max(b for _, _, b in segs)
    span = max(last - first, 1e-9)
    minutes = span / 60
    shares = [talk[i] / total for i in ids]
    turns = turns_from(segs)
    replies = [t for t in turns[1:] if t[3] is not None and t[3] <= REPLY_WITHIN_S]
    group = {
        "balance_entropy": _entropy(shares),
        "top_share": max(shares),
        "participation_rate": sum(1 for i in ids if talk[i] >= PRESENT_S) / n,
        "turns_per_min": len(turns) / minutes,
        "reply_rate": len(replies) / max(len(turns) - 1, 1),
        "silence_share": max(0.0, 1 - total / span),
        "minutes": minutes,
        "talk_s": total,
    }
    fair = 1 / n
    per = {}
    for i, share in zip(ids, shares):
        mine = [t for t in turns if t[0] == i]
        my_replies = [t for t in mine if t[3] is not None and t[3] <= REPLY_WITHIN_S]
        per[i] = {
            "share_balance": 1 - min(abs(share - fair) / fair, 1) if n > 1 else 1.0,
            "turns_per_min": len(mine) / minutes,
            "reply_rate": len(my_replies) / len(mine) if mine else 0.0,
            "talk_share": share,
            "turns": len(mine),
        }
    return group, per


# --- norms -----------------------------------------------------------------------

#: How long the session has run decides which table a group is compared with: a
#: group 5 minutes in is judged against other meetings' first 5 minutes, not against
#: whole meetings (which are more even by the end). name, from-minute, prefixes to sample.
BANDS = [("short", 3, [4, 6, 8]), ("medium", 10, [12, 16, 20]), ("long", 25, [30, 40, 60])]


def _prefix(segments: list[Segment], minutes: float):
    t0 = min(a for _, a, _ in segments)
    cut = t0 + minutes * 60
    return [(s, a, min(b, cut)) for s, a, b in segments if a < cut]


def build_norms(meetings: dict[str, list[Segment]], min_talk_s: float = 60.0, source: str = "") -> dict:
    """`meetings` maps a meeting id to its collapsed one-microphone segments. For
    each band, every meeting long enough contributes its first N minutes (what the
    app sees N minutes into a session): one group row, and one row per person
    (people who stayed silent included)."""
    bands = []
    for name, from_min, prefixes in BANDS:
        grows, srows = {k: [] for k in GROUP_FEATURES}, {k: [] for k in SPEAKER_FEATURES}
        n = 0
        for mid, segs in meetings.items():
            everyone = sorted({s for s, _, _ in segs})
            if len(everyone) < 2:
                continue
            length = (max(b for _, _, b in segs) - min(a for _, a, _ in segs)) / 60
            for m in prefixes:
                if length < m:
                    continue
                part = _prefix(segs, m)
                if sum(b - a for _, a, b in part) < min_talk_s:
                    continue
                g, per = session_features(part, everyone)
                n += 1
                for k in GROUP_FEATURES:
                    grows[k].append(g[k])
                for p in per.values():
                    for k in SPEAKER_FEATURES:
                        srows[k].append(p[k])
        qs = np.linspace(0, 1, QUANTILES)
        table = lambda rows: {k: [round(float(v), 6) for v in np.quantile(rows[k], qs)] for k in rows}
        bands.append({"name": name, "from_minutes": from_min, "prefix_minutes": prefixes, "rows": n,
                      "group": table(grows), "speaker": table(srows)})
    return {"version": NORMS_VERSION, "source": source, "meetings": len(meetings), "bands": bands}


def pick_band(norms: dict, minutes: float) -> dict:
    """The table for a session that has run `minutes` (the longest band it has reached)."""
    chosen = norms["bands"][0]
    for b in norms["bands"]:
        if minutes >= b["from_minutes"]:
            chosen = b
    return chosen


def percentile(value: float, q: list[float]) -> float:
    """Where `value` falls in a quantile table, 0..1 (ties take the middle of the run)."""
    L = len(q)
    same = [i for i, x in enumerate(q) if x == value]
    if same:
        return (same[0] + same[-1]) / 2 / (L - 1)
    if value < q[0]:
        return 0.0
    if value > q[-1]:
        return 1.0
    a = sum(1 for x in q if x < value)          # q[a-1] < value < q[a]
    lo, hi = q[a - 1], q[a]
    return (a - 1 + (value - lo) / (hi - lo)) / (L - 1)


def _good(feature: str, value: float, table: dict) -> float:
    p = percentile(value, table[feature])
    return p if DIRECTION[feature] > 0 else 1 - p


def score_group(g: dict, band: dict) -> dict:
    """Group communication and collaboration, 1-10, from group features.
      communication  = fair spread of talk (even shares, no one dominating), and everyone took part
      collaboration  = people answer each other quickly, often, with little dead air, and evenly
    """
    t = band["group"]
    comm = (_good("balance_entropy", g["balance_entropy"], t) + _good("top_share", g["top_share"], t)) / 2
    comm *= 0.5 + 0.5 * g["participation_rate"]
    collab = (_good("turns_per_min", g["turns_per_min"], t) + _good("reply_rate", g["reply_rate"], t)
              + _good("silence_share", g["silence_share"], t) + _good("balance_entropy", g["balance_entropy"], t)) / 4
    return {"communication": 1 + 9 * comm, "collaboration": 1 + 9 * collab}


def score_person(p: dict, band: dict) -> dict:
    """One person's communication and collaboration, 1-10.
      communication  = how close their share of the talking is to a fair share, and how often they take a turn
      collaboration  = how often their turns answer someone else, and how often they take a turn
    """
    t = band["speaker"]
    turns = _good("turns_per_min", p["turns_per_min"], t)
    comm = (_good("share_balance", p["share_balance"], t) + turns) / 2
    collab = (_good("reply_rate", p["reply_rate"], t) + turns) / 2
    return {"communication": 1 + 9 * comm, "collaboration": 1 + 9 * collab}


def dump_norms(norms: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(norms, f, indent=1, sort_keys=True)
        f.write("\n")


def norms_as_typescript(norms: dict) -> str:
    """The same tables as a TypeScript module for the app (src/engine/reference/meetingNorms.ts)."""
    body = json.dumps(norms, indent=2, sort_keys=True)
    return ("// GENERATED by `fourc export-norms` from fourc_models/behavior.py. Do not edit by hand.\n"
            "// Copyright © 2026 4Cight Inc. All rights reserved.\n"
            "// Where a group's behavior falls among real meetings (AMI + ICSI corpora, CC BY 4.0;\n"
            "// only word timings were used). See behavior.ts for how it is used.\n\n"
            "import type { MeetingNorms } from '../behavior';\n\n"
            f"export const MEETING_NORMS: MeetingNorms = {body};\n")
