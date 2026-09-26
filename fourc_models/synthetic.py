# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Synthetic classroom data in the exact labeling-guide format.

FOR TESTING THE PIPELINE ONLY. It lets every stage (loading, training,
scoring, export) run end to end before real labels exist. Each simulated
student has a hidden critical-thinking and creativity level (1-4) that drives
what they say; the teacher ratings are those levels plus noise, so a working
pipeline should recover them well above chance. Numbers from synthetic data
say nothing about real-world accuracy.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from . import labels as L

# activity -> idea category -> paraphrases of distinct ideas (each inner list = one idea)
IDEA_BANK = {
    "ramp-car": {
        "wheels": [["tape the wheels", "put sticky tape on the tires"],
                   ["use bigger wheels", "swap in the large wheels"],
                   ["add rubber bands to the wheels", "wrap rubber bands around the tires"]],
        "weight": [["make the car lighter", "take weight off the car"],
                   ["move the weight to the front", "put the heavy part at the front"]],
        "ramp": [["make the ramp steeper", "raise the ramp higher"],
                 ["make the ramp smoother", "cover the ramp with paper so it is smooth"]],
        "unusual": [["roll the car on pencils like logs", "put pencils under it like rollers"]],
    },
    "bridge": {
        "shape": [["make a triangle shape", "use triangles for the frame"],
                  ["build an arch", "curve the bridge into an arch"]],
        "material": [["roll the paper into tubes", "make paper straws for beams"],
                     ["fold the paper like a fan", "pleat the paper into folds"]],
        "support": [["add a pillar in the middle", "put a support under the center"]],
        "unusual": [["hang it with string like a suspension bridge", "use string cables to hold it up"]],
    },
}

CT_TEMPLATES = {
    "interpretation": ("question", "none", ["so you mean {x}?", "wait, are you saying {x}?"]),
    "analysis": ("reasoning", "claim", ["it is either {x} or {y}", "there are two things, {x} and {y}"]),
    "evaluation": ("reasoning", "challenge", ["that will not work because {x} failed before", "but {x} did not help last time"]),
    "inference": ("reasoning", "claim", ["if we {x} it should go farther", "so {x} would probably fix it"]),
    "explanation": ("reasoning", "evidence", ["it worked because we {x}", "the reason is that we {x}"]),
    "self_regulation": ("reasoning", "revision", ["wait I was wrong, maybe {x}", "actually let me check, maybe {x} instead"]),
}
PLAIN = [
    ("other", "neutral", ["okay", "sure", "number three"]),
    ("coordinates", "neutral", ["you hold it and I will measure", "whose turn is it"]),
    ("off_task", "neutral", ["did you watch the game", "lunch is pizza today"]),
    ("stuck", "negative", ["this is so annoying nothing works", "I do not get what to do"]),
    ("other", "positive", ["yes we did it", "that was so cool"]),
]


def _session(rng: random.Random, session_id: str, activity: str, grade: str, split: str,
             n_students: int, n_turns: int):
    categories = IDEA_BANK[activity]
    all_ideas = [(cat, i) for cat, ideas in categories.items() for i in range(len(ideas))]
    students = [f"[STUDENT_{chr(65 + i)}]" for i in range(n_students)]
    ct_level = {s: rng.randint(1, 4) for s in students}
    cr_level = {s: rng.randint(1, 4) for s in students}

    rows, ideas_so_far = [], []  # ideas_so_far: list of (idea_id, cat, idx)
    prev = []
    for turn in range(n_turns):
        s = rng.choice(students)
        ct, cr = ct_level[s], cr_level[s]
        row = dict(sentiment="neutral", move="other", ct_skill="none", argument="none",
                   idea_id="", idea_link="", linked_ideas="", teacher_idea="0")
        roll = rng.random()
        if roll < 0.12 * cr:  # idea-bearing turn
            fresh = [x for x in all_ideas if x[:2] not in [(c, i) for _, c, i in ideas_so_far]]
            if ideas_so_far and rng.random() < 0.45:
                iid, cat, idx = rng.choice(ideas_so_far)
                if rng.random() < 0.5:
                    text = rng.choice(categories[cat][idx])
                    row.update(move="builds_on", idea_link="repeat", linked_ideas=iid, text=f"yeah {text}")
                else:
                    text = rng.choice(categories[cat][idx]) + " and make it stronger"
                    row.update(move="builds_on", idea_link="develops", linked_ideas=iid,
                               ct_skill="inference", text=f"and {text}")
            elif fresh:
                # higher creativity -> more likely to pick an unusual idea
                weights = [(3.0 * cr if c == "unusual" else 2.0) for c, _ in fresh]
                cat, idx = rng.choices(fresh, weights=weights)[0]
                iid = f"I{len(ideas_so_far) + 1}"
                ideas_so_far.append((iid, cat, idx))
                text = rng.choice(categories[cat][idx])
                row.update(move="new_idea", idea_id=iid, idea_link="new",
                           sentiment="positive" if rng.random() < 0.4 else "neutral",
                           text=f"what if we {text}")
            else:
                move, sent, opts = rng.choice(PLAIN)
                row.update(move=move, sentiment=sent, text=rng.choice(opts))
        elif roll < 0.12 * cr + 0.14 * ct:  # critical-thinking turn
            skill = rng.choice(list(CT_TEMPLATES))
            move, arg, opts = CT_TEMPLATES[skill]
            x = rng.choice(rng.choice(list(categories.values())))[0]
            y = rng.choice(rng.choice(list(categories.values())))[0]
            row.update(move=move, ct_skill=skill, argument=arg,
                       text=rng.choice(opts).format(x=x, y=y))
        else:
            move, sent, opts = rng.choice(PLAIN)
            row.update(move=move, sentiment=sent, text=rng.choice(opts))
        row.update(utterance_id=f"u_{session_id}_{turn:03d}", session_id=session_id,
                   activity_id=activity, context_prev=" ".join(prev[-2:]), grade_band=grade,
                   flag_unclear="0", split=split, guide_version=L.GUIDE_VERSION, speaker=s)
        rows.append(row)
        prev.append(f"{s}: {row['text']}")

    ratings = []
    for s in students:
        for skill, lvl in (("critical_thinking", ct_level[s]), ("creativity", cr_level[s])):
            noisy = min(4, max(1, lvl + rng.choice([0, 0, 0, 1, -1])))
            ratings.append(dict(segment_id=session_id, activity_id=activity, grade_band=grade,
                                rated=s, skill=skill, final_level=str(noisy)))
    return rows, ratings


def make_dataset(n_sessions: int = 60, seed: int = 0, n_students: int = 4, n_turns: int = 40):
    """Returns (utterances, pairs, segment_ratings) DataFrames in guide format.
    Utterances carry an extra `speaker` column (the de-identified placeholder),
    which the real export pipeline adds from the transcript's speaker labels."""
    rng = random.Random(seed)
    utts, ratings = [], []
    for i in range(n_sessions):
        split = "train" if i < n_sessions * 0.7 else ("validation" if i < n_sessions * 0.85 else "test")
        rows, r = _session(rng, f"s{i:03d}", rng.choice(list(IDEA_BANK)), rng.choice(L.GRADE_BANDS),
                           split, n_students, n_turns)
        utts += rows
        ratings += r
    pairs = []
    k = 0
    for activity, cats in IDEA_BANK.items():
        flat = [(c, idea) for c, ideas in cats.items() for idea in ideas]
        for c1, idea1 in flat:
            for c2, idea2 in flat:
                a, b = rng.choice(idea1), rng.choice(idea2)
                label = "same" if idea1 is idea2 else ("related" if c1 == c2 else "different")
                pairs.append(dict(pair_id=f"p{k:04d}", activity_id=activity, idea_a_text=a,
                                  idea_b_text=b, final_label=label))
                k += 1
    return pd.DataFrame(utts), pd.DataFrame(pairs), pd.DataFrame(ratings)


def write_dataset(out_dir: str | Path, **kw) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    u, p, r = make_dataset(**kw)
    paths = {"utterances": out / "utterances.csv", "pairs": out / "pairs.csv", "segments": out / "segments.csv"}
    u.to_csv(paths["utterances"], index=False)
    p.to_csv(paths["pairs"], index=False)
    r.to_csv(paths["segments"], index=False)
    return paths
