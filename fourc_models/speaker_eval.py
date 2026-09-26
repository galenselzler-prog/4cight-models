# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Measure speaker recognition before any student is recorded.

Builds classroom-like group recordings from a folder of voices where we know
exactly who spoke when, runs the SAME steps as the app (speech detection ->
1.5 s pieces -> voice embedding -> closest enrolled voiceprint) and reports
how much talk time lands on the right student. Sweeps the match threshold
and recommends one for src/engine/speaker.ts.

voices/<speaker>/*.wav      one folder per speaker (any sample rate, 16-bit WAV)

Voices can be synthetic (`fourc synth-voices`, macOS text-to-speech) to get
started, or real consented recordings / Mozilla Common Voice for numbers you
can trust. Synthetic voices are cleaner and more distinct than children in a
classroom, so treat synthetic results as an upper bound.

The detection and attribution code below mirrors src/engine/audio.ts and
src/engine/speaker.ts in the app; keep them in step.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .emotion_export import read_wav

SR = 16000
OUTSIDER = "__outsider__"


# ---- mirror of src/engine/audio.ts detectSpeech --------------------------------
def detect_speech(x: np.ndarray, sr: int = SR, frame_ms=30, floor_db=-50.0, margin_db=10.0,
                  bridge_ms=300, min_speech_ms=250, noise_floor_db=None) -> list[tuple[int, int]]:
    frame = max(1, round(sr * frame_ms / 1000))
    n = len(x) // frame
    if n == 0:
        return []
    f = x[: n * frame].reshape(n, frame).astype(np.float64)
    db = 10 * np.log10((f * f).mean(1) + 1e-12)
    noise = np.sort(db)[int(0.1 * (n - 1))]
    if noise_floor_db is not None:  # room level remembered from quieter moments
        noise = min(noise, noise_floor_db)
    thr = max(floor_db, noise + margin_db)
    bridge, min_frames = round(bridge_ms / frame_ms), round(min_speech_ms / frame_ms)
    regions, start, last = [], -1, -1
    for i in range(n):
        if db[i] >= thr:
            if start < 0:
                start = i
            elif i - last - 1 > bridge:
                regions.append((start, last + 1))
                start = i
            last = i
    if start >= 0:
        regions.append((start, last + 1))
    return [(a * frame, b * frame) for a, b in regions if b - a >= min_frames]


# ---- mirror of src/engine/speaker.ts -------------------------------------------
def split_segments(regions, seg_len: int, min_len: int):
    """Cut speech regions into ~seg_len pieces; a short tail joins the piece before."""
    out = []
    for a, b in regions:
        pieces, s = [], a
        while s < b:
            e = min(b, s + seg_len)
            if e - s < min_len and pieces:
                pieces[-1] = (pieces[-1][0], e)
            else:
                pieces.append((s, e))
            s = e
        out += [p for p in pieces if p[1] - p[0] >= min_len]
    return out


def make_voiceprint(embed, audio: np.ndarray, chunk_sec=3.0, min_speech_sec=3.0) -> np.ndarray:
    speech = [audio[a:b] for a, b in detect_speech(audio)]
    joined = np.concatenate(speech) if speech else np.zeros(0, np.float32)
    if len(joined) < min_speech_sec * SR:
        # Someone who reads without pausing leaves no silence to measure the room
        # against; fall back to the plain loudness gate for enrollment.
        speech = [audio[a:b] for a, b in detect_speech(audio, noise_floor_db=-200.0)]
        joined = np.concatenate(speech) if speech else np.zeros(0, np.float32)
    if len(joined) < min_speech_sec * SR:
        raise ValueError(f"only {len(joined) / SR:.1f} s of speech; need {min_speech_sec} s")
    step = int(chunk_sec * SR)
    chunks = [joined[i:i + step] for i in range(0, len(joined) - int(1.5 * SR) + 1, step)] or [joined]
    v = np.mean([embed(c) for c in chunks], axis=0)
    return v / np.linalg.norm(v)


def attribute(embed, audio, regions, voiceprints: dict, threshold: float, margin: float,
              seg_sec=1.5, min_seg_sec=0.75):
    ids = list(voiceprints)
    V = np.stack([voiceprints[i] for i in ids])
    out = []
    for a, b in split_segments(regions, int(seg_sec * SR), int(min_seg_sec * SR)):
        sims = V @ embed(audio[a:b])
        order = np.argsort(-sims)
        best = sims[order[0]]
        second = sims[order[1]] if len(ids) > 1 else -1.0
        who = ids[order[0]] if best >= threshold and best - second >= margin else None
        out.append((a, b, who, float(best)))
    return out


# ---- building test sessions ------------------------------------------------------
@dataclass
class Voice:
    name: str
    enroll: np.ndarray
    pool: list[np.ndarray]


def load_voices(voices_dir, enroll_sec=12.0) -> list[Voice]:
    voices = []
    for d in sorted(p for p in Path(voices_dir).iterdir() if p.is_dir()):
        clips = [read_wav(f) for f in sorted(d.glob("*.wav"))]
        clips = [c for c in clips if len(c) > SR // 2]
        if not clips:
            continue
        enroll, pool, have = [], [], 0.0
        for c in clips:  # first clips (about enroll_sec) enroll, the rest are for sessions
            if have < enroll_sec:
                enroll.append(c)
                have += len(c) / SR
            else:
                pool.append(c)
        if pool:
            voices.append(Voice(d.name, np.concatenate(enroll), pool))
    return voices


def pink_noise(n: int, rng) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white) / np.sqrt(np.arange(1, n // 2 + 2))
    x = np.fft.irfft(spec, n)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def build_session(group: list[Voice], rng, n_turns=24, snr_db=15.0, outsider: Voice | None = None,
                  outsider_share=0.15):
    """Returns (audio, truth): per-sample speaker index into `group`, -1 = nobody,
    -2 = the outsider (a voice nobody enrolled, like the teacher or a
    neighbouring group), who takes about `outsider_share` of the turns."""
    parts, truth = [], []
    for _ in range(n_turns):
        if outsider is not None and rng.random() < outsider_share:
            k, voice = -2, outsider
        else:
            k = rng.randrange(len(group))
            voice = group[k]
        clip = rng.choice(voice.pool)
        clip = clip / (np.abs(clip).max() + 1e-9) * rng.uniform(0.25, 0.6)  # distance to the iPad varies
        gap = np.zeros(int(rng.uniform(0.2, 1.0) * SR), np.float32)
        parts += [gap, clip.astype(np.float32)]
        truth += [np.full(len(gap), -1), np.full(len(clip), k)]
    audio = np.concatenate(parts)
    speech_rms = np.sqrt(np.mean(audio[np.concatenate(truth) != -1] ** 2))
    noise = pink_noise(len(audio), np.random.default_rng(rng.randrange(1 << 30)))
    audio = audio + noise * speech_rms / (10 ** (snr_db / 20)) / (np.sqrt(np.mean(noise ** 2)) + 1e-9)
    return np.clip(audio, -1, 1).astype(np.float32), np.concatenate(truth)


def evaluate(onnx_path, voices_dir, out=None, sessions=20, group_size=4, snr_db=15.0, seed=0, log=print) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def embed(x):
        return sess.run(["embedding"], {"audio": x[None].astype(np.float32)})[0][0]

    voices = load_voices(voices_dir)
    if len(voices) < 2:
        raise ValueError(f"{voices_dir}: need at least 2 speaker folders with enough audio")
    group_size = min(group_size, len(voices))
    rng = random.Random(seed)
    prints = {v.name: make_voiceprint(embed, v.enroll) for v in voices}
    log(f"  {len(voices)} voices enrolled; {sessions} test sessions of {group_size} speakers at {snr_db:.0f} dB SNR")

    # Embed once; threshold sweep reuses the raw similarities.
    trials = []
    for _ in range(sessions):
        group = rng.sample(voices, group_size)
        others = [v for v in voices if v not in group]
        outsider = rng.choice(others) if others else None
        audio, truth = build_session(group, rng, snr_db=snr_db, outsider=outsider)
        vp = {v.name: prints[v.name] for v in group}
        pieces = split_segments(detect_speech(audio), int(1.5 * SR), int(0.75 * SR))
        ids = [v.name for v in group]
        V = np.stack([vp[i] for i in ids])
        for a, b in pieces:
            sims = V @ embed(audio[a:b])
            lab = truth[a:b]
            spoken = lab[lab != -1]
            if not len(spoken):
                majority = None  # only room noise
            else:
                vals, counts = np.unique(spoken, return_counts=True)
                top = vals[counts.argmax()]
                majority = OUTSIDER if top == -2 else ids[top]
            trials.append((sorted(zip(sims.tolist(), ids), reverse=True), majority, b - a))
    speech_total = sum(n for _, m, n in trials if m not in (None, OUTSIDER))
    outsider_total = sum(n for _, m, n in trials if m == OUTSIDER)

    sweep = []
    for thr in np.round(np.arange(0.0, 0.81, 0.05), 2):
        for margin in (0.0, 0.05, 0.1):
            right = wrong = unknown = noise_claimed = outsider_claimed = 0
            for ranked, truth_id, n in trials:
                best, who = ranked[0]
                second = ranked[1][0] if len(ranked) > 1 else -1
                pred = who if best >= thr and best - second >= margin else None
                if truth_id is None:
                    noise_claimed += n if pred else 0
                elif truth_id == OUTSIDER:
                    outsider_claimed += n if pred else 0
                elif pred is None:
                    unknown += n
                elif pred == truth_id:
                    right += n
                else:
                    wrong += n
            sweep.append({"threshold": float(thr), "margin": margin, "correct": right / speech_total,
                          "wrong": wrong / speech_total, "unknown": unknown / speech_total,
                          "outsider_credited": outsider_claimed / outsider_total if outsider_total else None,
                          "noise_assigned_sec": round(noise_claimed / SR, 1)})
    # Wrong student <= 5% and an unenrolled voice credited to students <= 20%;
    # among those, the most talk time on the right student.
    ok = [s for s in sweep if s["wrong"] <= 0.05 and (s["outsider_credited"] or 0) <= 0.20]
    best = (max(ok, key=lambda s: (s["correct"], -s["wrong"])) if ok
            else min(sweep, key=lambda s: s["wrong"] + (s["outsider_credited"] or 0)))
    report = {"voices": len(voices), "sessions": sessions, "group_size": group_size, "snr_db": snr_db,
              "speech_minutes": round(speech_total / SR / 60, 1), "recommended": best,
              "note": ("Talk time: correct = on the right student, wrong = on another student, unknown = "
                       "unassigned. outsider_credited = share of an unenrolled voice's speech wrongly "
                       "credited to a student."),
              "sweep": sweep}
    log(f"  recommended threshold {best['threshold']} margin {best['margin']}: "
        f"{best['correct']:.0%} of talk time on the right student, {best['wrong']:.0%} wrong, {best['unknown']:.0%} unassigned"
        + (f"; unenrolled voice credited to a student {best['outsider_credited']:.0%} of the time"
           if best["outsider_credited"] is not None else ""))
    if out:
        Path(out).write_text(json.dumps(report, indent=2))
    return report


# ---- synthetic voices (macOS text-to-speech) --------------------------------------
SENTENCES = [
    "I think the car goes farther when the ramp is steeper.", "What if we put tape on the wheels?",
    "That did not work last time, remember?", "Can you hold it while I measure the distance?",
    "Wait, I was wrong, it is the weight that matters.", "Let's try building a triangle for the frame.",
    "My idea is to roll the paper into tubes.", "Why do you think the bridge fell down?",
    "Okay, whose turn is it to write the answer?", "I agree with you, but we should test it twice.",
    "Maybe the rubber bands make the wheels grip better.", "We measured forty two centimeters that time.",
    "Hold on, let me check the instructions again.", "That was so cool, it went all the way to the wall!",
    "I don't get it, why is ours slower than theirs?", "So if we move the weight to the front, it should work.",
    "Let's write down what we changed before we try again.", "Could we use string like a suspension bridge?",
    "I think we need more supports in the middle.", "Great job, that was our best try so far.",
]
NOVELTY = {"Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Good News", "Jester", "Organ",
           "Superstar", "Trinoids", "Whisper", "Wobble", "Zarvox", "Deranged", "Hysterical", "Pipe Organ"}


def mac_voices(locale_prefix="en_") -> list[str]:
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    names = []
    for line in out.splitlines():
        parts = line.split("#")[0].split()
        if len(parts) >= 2 and parts[-1].startswith(locale_prefix):
            name = " ".join(parts[:-1])
            if name not in NOVELTY and "(" not in name:
                names.append(name)
    return sorted(set(names))


def synth_voices(out_dir, n_voices=8, sentences_per_voice=16, rate_jitter=True, seed=0, log=print) -> list[str]:
    if sys.platform != "darwin" or not shutil.which("say"):
        raise RuntimeError("synth-voices uses macOS text-to-speech (`say`); run it on the Mac")
    rng = random.Random(seed)
    names = mac_voices()[:n_voices]
    if len(names) < 2:
        raise RuntimeError("fewer than 2 English system voices installed (System Settings > Accessibility > Spoken Content)")
    for name in names:
        d = Path(out_dir) / name.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(rng.sample(SENTENCES, min(sentences_per_voice, len(SENTENCES)))):
            rate = str(rng.randint(150, 210)) if rate_jitter else "175"
            subprocess.run(["say", "-v", name, "-r", rate, "-o", str(d / f"{i:02d}.wav"),
                            "--file-format=WAVE", "--data-format=LEI16@16000", text], check=True)
    log(f"  wrote {len(names)} synthetic voices to {out_dir}: {', '.join(names)}")
    return names
