# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""From consented research recordings to training data, through Label Studio.

  fourc labeling-setup   create the two Label Studio projects and connect them to R2
  fourc prep-labeling    turn new recordings in R2 into labeling tasks (in R2)
  (label in Label Studio, then Export -> JSON for each project)
  fourc import-labels    Label Studio exports -> utterances.csv / segments.csv / pairs.csv

Recordings arrive from the app as recordings/<id>/audio.wav + session.json
(names already replaced by [STUDENT_A] ...). prep-labeling writes, in the
same private bucket:

  labeling/clips/<id>/<n>.wav                one audio clip per utterance
  labeling/utterance-tasks/<id>/<n>.json     one task per utterance
  labeling/rating-tasks/<id>/<student>.json  one task per student (teacher ratings)
  labeling/prepared/<id>.json                marker: this recording is done

Label Studio reads the task files from R2 and shows the audio through
short-lived signed links, so audio never becomes public.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import wave
from collections import Counter
from pathlib import Path

from . import labels as L
from .data import UTTERANCE_COLUMNS

CLIP_PAD_MS = 300
UTTERANCE_FIELDS = ["sentiment", "move", "ct_skill", "argument"]
UTTERANCE_EXTRA = ["speaker"]


# ---------------------------------------------------------------- task building
def split_for(recording_id: str) -> str:
    """Whole recordings go to one split (70/15/15), stable across runs."""
    h = int(hashlib.sha256(recording_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < 0.70 else ("validation" if h < 0.85 else "test")


def grade_band(meta: dict) -> str:
    g = meta.get("gradeBand")
    return g if g in L.GRADE_BANDS else ""


def utterance_tasks(recording_id: str, meta: dict, clip_uri) -> list[tuple[int, int, dict]]:
    """[(clip_start_ms, clip_end_ms, task_data)]; clip_uri(n) -> where clip n is stored."""
    segs = [s for s in meta.get("transcript", []) if str(s.get("text", "")).strip()]
    speakers = [{"value": p["speaker"]} for p in meta.get("participants", [])] + [{"value": "[OTHER]"}]
    out, prev = [], []
    for n, seg in enumerate(segs):
        start = max(0, int(seg["startMs"]) - CLIP_PAD_MS)
        end = min(int(meta.get("durationMs", seg["endMs"] + CLIP_PAD_MS)), int(seg["endMs"]) + CLIP_PAD_MS)
        data = {
            "utterance_id": f"u_{recording_id[:8]}_{n:04d}",
            "session_id": recording_id,
            "activity_id": meta.get("activityId") or "",
            "grade_band": grade_band(meta),
            "text": seg["text"],
            "context_prev": " ".join(prev[-2:]),
            "audio": clip_uri(n),
            "speaker_options": speakers,
            "suggested_speaker": seg.get("speaker") or "",
            "split": split_for(recording_id),
            "guide_version": L.GUIDE_VERSION,
        }
        out.append((start, max(end, start + 500), data))
        prev.append(f"{seg.get('speaker') or '[?]'}: {seg['text']}")
    return out


def rating_tasks(recording_id: str, meta: dict, audio_uri: str) -> list[dict]:
    lines = []
    for seg in meta.get("transcript", []):
        if str(seg.get("text", "")).strip():
            t = int(seg["startMs"]) // 1000
            lines.append(f"[{t // 60:02d}:{t % 60:02d}] {seg.get('speaker') or '[?]'}: {seg['text']}")
    transcript = "\n".join(lines)
    return [{
        "segment_id": recording_id,
        "rated": p["speaker"],
        "activity_id": meta.get("activityId") or "",
        "grade_band": grade_band(meta),
        "audio": audio_uri,
        "transcript": transcript,
        "split": split_for(recording_id),
    } for p in meta.get("participants", [])]


def cut_wav(wav_bytes: bytes, start_ms: int, end_ms: int) -> bytes:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        sr, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        a = min(w.getnframes(), start_ms * sr // 1000)
        b = min(w.getnframes(), max(a, end_ms * sr // 1000))
        w.setpos(a)
        frames = w.readframes(b - a)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(ch); out.setsampwidth(width); out.setframerate(sr)
        out.writeframes(frames)
    return buf.getvalue()


# ---------------------------------------------------------------- R2 (S3 API)
def r2_endpoint() -> str:
    """R2's S3 endpoint for the account (R2_ENDPOINT overrides, e.g. for tests)."""
    return os.environ.get("R2_ENDPOINT") or f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"


def r2_client():
    import boto3
    return boto3.client("s3", endpoint_url=r2_endpoint(), region_name="auto",
                        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"])


def prep_labeling(bucket: str | None = None, s3=None, log=print) -> dict:
    """Makes labeling tasks for every uploaded recording not prepared yet."""
    s3 = s3 or r2_client()
    bucket = bucket or os.environ.get("R2_RESEARCH_BUCKET", "4cight-research")
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="recordings/"):
        keys += [o["Key"] for o in page.get("Contents", [])]
    rec_ids = sorted({k.split("/")[1] for k in keys if k.endswith("/session.json")
                      and f"recordings/{k.split('/')[1]}/audio.wav" in keys})
    stats = {"recordings": 0, "utterance_tasks": 0, "rating_tasks": 0, "skipped": 0}
    for rid in rec_ids:
        marker = f"labeling/prepared/{rid}.json"
        try:
            s3.head_object(Bucket=bucket, Key=marker)
            stats["skipped"] += 1
            continue
        except Exception:
            pass
        meta = json.loads(s3.get_object(Bucket=bucket, Key=f"recordings/{rid}/session.json")["Body"].read())
        audio = s3.get_object(Bucket=bucket, Key=f"recordings/{rid}/audio.wav")["Body"].read()
        clip_key = lambda n: f"labeling/clips/{rid}/{n:04d}.wav"  # noqa: E731
        utts = utterance_tasks(rid, meta, lambda n: f"s3://{bucket}/{clip_key(n)}")
        for n, (a, b, data) in enumerate(utts):
            s3.put_object(Bucket=bucket, Key=clip_key(n), Body=cut_wav(audio, a, b), ContentType="audio/wav")
            s3.put_object(Bucket=bucket, Key=f"labeling/utterance-tasks/{rid}/{n:04d}.json",
                          Body=json.dumps({"data": data}).encode(), ContentType="application/json")
        rates = rating_tasks(rid, meta, f"s3://{bucket}/recordings/{rid}/audio.wav")
        for t in rates:
            name = t["rated"].strip("[]").lower()
            s3.put_object(Bucket=bucket, Key=f"labeling/rating-tasks/{rid}/{name}.json",
                          Body=json.dumps({"data": t}).encode(), ContentType="application/json")
        s3.put_object(Bucket=bucket, Key=marker, ContentType="application/json",
                      Body=json.dumps({"utterances": len(utts), "ratings": len(rates)}).encode())
        stats["recordings"] += 1
        stats["utterance_tasks"] += len(utts)
        stats["rating_tasks"] += len(rates)
        log(f"  {rid}: {len(utts)} utterances, {len(rates)} students")
    log(f"  prepared {stats['recordings']} new recordings ({stats['skipped']} already done)")
    return stats


def upload_models(paths: list[str | Path], bucket: str | None = None, s3=None, log=print) -> list[str]:
    """Puts the app's .onnx files in the private models bucket (the Cloudflare
    dashboard can't upload files over 300 MB; this uses multipart uploads)."""
    s3 = s3 or r2_client()
    bucket = bucket or os.environ.get("R2_MODELS_BUCKET", "4cight-models")
    done = []
    for p in map(Path, paths):
        if p.suffix != ".onnx":
            raise ValueError(f"{p}: only .onnx model files go in the models bucket")
        s3.upload_file(str(p), bucket, p.name, ExtraArgs={"ContentType": "application/octet-stream"})
        size = s3.head_object(Bucket=bucket, Key=p.name)["ContentLength"]
        if size != p.stat().st_size:
            raise RuntimeError(f"{p.name}: uploaded {size} bytes, expected {p.stat().st_size}")
        log(f"  uploaded {p.name} ({size / 1e6:.0f} MB) to r2://{bucket}/")
        done.append(p.name)
    return done


# ---------------------------------------------------------------- Label Studio API
class LabelStudio:
    """Minimal Label Studio API client. `token` is a Personal Access Token
    (Account & Settings -> Personal Access Token); legacy tokens also work."""

    def __init__(self, url: str, token: str):
        import requests
        self.url, self.http = url.rstrip("/"), requests.Session()
        r = self.http.post(f"{self.url}/api/token/refresh", json={"refresh": token}, timeout=30)
        auth = f"Bearer {r.json()['access']}" if r.ok and "access" in r.json() else f"Token {token}"
        self.http.headers["Authorization"] = auth

    def call(self, method: str, path: str, **kw):
        r = self.http.request(method, f"{self.url}{path}", timeout=120, **kw)
        if not r.ok:
            raise RuntimeError(f"Label Studio {method} {path}: {r.status_code} {r.text[:300]}")
        return r.json() if r.content else None


def labeling_setup(ls_url: str, ls_token: str, bucket: str | None = None, s3=None, log=print) -> dict:
    """Creates the two projects and connects each to its task folder in R2.
    Safe to run again: existing projects and storages are reused and re-synced."""
    bucket = bucket or os.environ.get("R2_RESEARCH_BUCKET", "4cight-research")
    s3 = s3 or r2_client()
    for prefix in ("labeling/utterance-tasks/", "labeling/rating-tasks/"):
        # Label Studio refuses to connect to an empty folder.
        s3.put_object(Bucket=bucket, Key=f"{prefix}.keep", Body=b"")
    here = Path(__file__).resolve().parent.parent / "labeling"
    ls = LabelStudio(ls_url, ls_token)
    existing = {p["title"]: p for p in ls.call("GET", "/api/projects?page_size=100")["results"]}
    out = {}
    for title, config, prefix, desc in [
        ("4Cight utterances", "utterance_config.xml", "labeling/utterance-tasks/",
         "Label each utterance: speaker, sentiment, move, CT skill, argument, idea link (guide v1.2)."),
        ("4Cight segment ratings", "rating_config.xml", "labeling/rating-tasks/",
         "Teachers: rate each student's critical thinking and creativity, 1-4 or NE (guide v1.2)."),
    ]:
        project = existing.get(title) or ls.call("POST", "/api/projects", json={
            "title": title, "description": desc, "label_config": (here / config).read_text(),
            "maximum_annotations": 2, "show_collab_predictions": False})
        storages = ls.call("GET", f"/api/storages/s3?project={project['id']}")
        if not storages:
            storage = ls.call("POST", "/api/storages/s3", json={
                "project": project["id"], "title": "R2 research bucket", "bucket": bucket, "prefix": prefix,
                "regex_filter": r".*\.json$", "use_blob_urls": False, "presign": True, "presign_ttl": 15,
                "recursive_scan": True,
                "aws_access_key_id": os.environ["R2_ACCESS_KEY_ID"],
                "aws_secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"],
                "s3_endpoint": r2_endpoint(),
                "region_name": "auto"})
            storages = [storage]
        for st in storages:
            ls.call("POST", f"/api/storages/s3/{st['id']}/sync")
        out[title] = project["id"]
        log(f"  {title}: project {project['id']}, synced from r2://{bucket}/{prefix}")
    return out


# ---------------------------------------------------------------- export -> CSVs
def _values(annotation: dict) -> dict:
    out = {}
    for r in annotation.get("result", []):
        v = r.get("value", {})
        if "choices" in v:
            out[r["from_name"]] = v["choices"]
        elif "text" in v:
            out[r["from_name"]] = [t.strip() for t in v["text"] if t.strip()]
    return out


def _final(annotations: list[dict], field: str, multi: bool = False):
    """ground truth > agreement > the only annotation. Returns (value, status)."""
    anns = [a for a in annotations if not a.get("was_cancelled")]
    gt = [a for a in anns if a.get("ground_truth")]
    pick = lambda a: _values(a).get(field, [])  # noqa: E731
    if gt:
        return pick(gt[0]), "adjudicated"
    vals = [tuple(sorted(pick(a))) for a in anns]
    if not vals:
        return [], "missing"
    if len(set(vals)) == 1:
        return list(vals[0]), "agreed" if len(vals) > 1 else "single"
    return None, "conflict"


def import_labels(utterance_export: str | Path, rating_export: str | Path | None, out_dir: str | Path,
                  default_grade_band: str | None = None, log=print) -> dict:
    """`default_grade_band` fills in recordings made without one (the app
    doesn't ask for it yet); import stops if it's needed and missing."""
    if default_grade_band is not None and default_grade_band not in L.GRADE_BANDS:
        raise ValueError(f"grade band must be one of {L.GRADE_BANDS}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path(utterance_export).read_text())
    rows, report = [], Counter()
    for t in tasks:
        d, anns = t["data"], [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            report["unlabeled"] += 1
            continue
        final, statuses = {}, []
        for f in UTTERANCE_FIELDS + UTTERANCE_EXTRA + ["idea_link", "idea_id", "linked_ideas", "flags", "corrected_text"]:
            v, st = _final(anns, f)
            final[f] = v
            if f in UTTERANCE_FIELDS + UTTERANCE_EXTRA:
                statuses.append(st)
        if "conflict" in statuses or any(not final[f] for f in UTTERANCE_FIELDS + UTTERANCE_EXTRA):
            report["needs_adjudication"] += 1
            continue
        flags = final["flags"] or []
        a0 = _values(anns[0])
        a1 = _values(anns[1]) if len(anns) > 1 else {}
        fmt = lambda v: "|".join((v.get(f) or ["?"])[0] for f in UTTERANCE_FIELDS) if v else ""  # noqa: E731
        rows.append({
            "utterance_id": d["utterance_id"], "session_id": d["session_id"], "activity_id": d.get("activity_id", ""),
            "text": (final["corrected_text"] or [d["text"]])[0], "context_prev": d.get("context_prev", ""),
            "grade_band": d.get("grade_band") or default_grade_band or "",
            **{f: final[f][0] for f in UTTERANCE_FIELDS},
            "idea_id": (final["idea_id"] or [""])[0], "idea_link": (final["idea_link"] or [""])[0],
            "linked_ideas": (final["linked_ideas"] or [""])[0],
            "teacher_idea": "1" if "teacher_idea" in flags else "0", "flag_unclear": "1" if "unclear" in flags else "0",
            "labeler_a": str(anns[0].get("completed_by", "")), "labeler_b": str(anns[1].get("completed_by", "")) if len(anns) > 1 else "",
            "label_a": fmt(a0), "label_b": fmt(a1),
            "adjudicated": "1" if any(a.get("ground_truth") for a in anns) else "0",
            "split": d.get("split", "train"), "guide_version": d.get("guide_version", L.GUIDE_VERSION),
            "speaker": final["speaker"][0],
        })
        report["labeled"] += 1
    rows.sort(key=lambda r: r["utterance_id"])  # file order = spoken order within a session
    missing_band = sum(1 for r in rows if not r["grade_band"])
    if missing_band:
        raise ValueError(f"{missing_band} utterances have no grade band; rerun with --grade-band (one of {L.GRADE_BANDS})")
    cols = UTTERANCE_COLUMNS + [c for c in ["labeler_a", "labeler_b", "label_a", "label_b", "adjudicated", "speaker"]
                                if c not in UTTERANCE_COLUMNS]
    _write(out / "utterances.csv", cols, rows)

    seg_rows = []
    if rating_export:
        for t in json.loads(Path(rating_export).read_text()):
            d, anns = t["data"], [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
            for skill in ("critical_thinking", "creativity"):
                given = [(_values(a).get(skill) or [None])[0] for a in anns]
                given = [g for g in given if g]
                gt = [(_values(a).get(skill) or [None])[0] for a in anns if a.get("ground_truth")]
                if gt and gt[0]:
                    final = gt[0]
                elif not given:
                    report["ratings_missing"] += 1
                    continue
                elif len(set(given)) == 1:
                    final = given[0]
                elif L.NOT_ENOUGH_EVIDENCE not in given and max(map(int, given)) - min(map(int, given)) <= 1:
                    final = str(round(sum(map(int, given)) / len(given) + 1e-9))  # neighbours: round the mean
                else:
                    report["ratings_need_adjudication"] += 1
                    continue
                for a in anns:
                    lvl = (_values(a).get(skill) or [""])[0]
                    seg_rows.append({"segment_id": d["segment_id"], "activity_id": d.get("activity_id", ""),
                                     "grade_band": d.get("grade_band") or default_grade_band or "", "rated": d["rated"], "skill": skill,
                                     "rater_id": str(a.get("completed_by", "")), "level": lvl, "final_level": final,
                                     "rubric_version": L.GUIDE_VERSION})
                report[f"rated_{skill}"] += 1
    _write(out / "segments.csv", ["segment_id", "activity_id", "grade_band", "rated", "skill", "rater_id", "level",
                                  "final_level", "rubric_version"], _one_row_per_rating(seg_rows))
    pairs = out / "pairs.csv"
    if not pairs.exists():  # same-idea pairs are labeled later, from the idea links above
        _write(pairs, ["pair_id", "activity_id", "idea_a_text", "idea_b_text", "labeler_a", "labeler_b", "label_a",
                       "label_b", "final_label", "guide_version"], [])
    report = dict(report)
    log("  " + ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in report.items()))
    return report


def _one_row_per_rating(rows: list[dict]) -> list[dict]:
    """The training loader wants one row per (segment, student, skill) with the final level."""
    seen, out = set(), []
    for r in rows:
        k = (r["segment_id"], r["rated"], r["skill"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _write(path: Path, cols: list[str], rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
