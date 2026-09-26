# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Command line:

  fourc synth  --out data/synthetic                 # fake data in the guide's format
  fourc check  --data data/labeled                  # validate labeled CSVs only
  fourc train  --data data/labeled --out runs/r1 --base microsoft/deberta-v3-small \
               --idea-base sentence-transformers/all-MiniLM-L6-v2
  fourc export --run runs/r1 --out dist/r1          # ONNX + JSON for the app
  fourc convert-emotion --model-dir models/m2-emotion --out dist/emotion [--clips clips/]
  fourc convert-speaker --out dist/speaker                  # downloads the open ECAPA model once
  fourc synth-voices --out voices/synthetic                 # macOS text-to-speech test voices
  fourc speaker-eval --model dist/speaker/speaker.onnx --voices voices/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="fourc")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="write synthetic data in the labeling-guide format")
    s.add_argument("--out", required=True)
    s.add_argument("--sessions", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("check", help="validate utterances.csv, pairs.csv, segments.csv")
    c.add_argument("--data", required=True)
    t = sub.add_parser("train", help="train both models and write report.json")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--base", default="tiny", help='HF id or local folder; "tiny" = offline test model')
    t.add_argument("--idea-base", default=None, help="base for the idea encoder (default: --base)")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--idea-epochs", type=int, default=6)
    t.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("export", help="export a run to ONNX + JSON for the app")
    e.add_argument("--run", required=True)
    e.add_argument("--out", required=True)
    m = sub.add_parser("convert-emotion", help="convert the existing M2 emotion model to ONNX (+ int8)")
    m.add_argument("--model-dir", required=True, help="folder with config.json and model.safetensors")
    m.add_argument("--out", required=True)
    m.add_argument("--clips", default=None, help="optional folder of real .wav clips to check the conversion on")
    m.add_argument("--no-int8", action="store_true", help="skip the smaller int8 copy")
    m.add_argument("--version", default="2.0.0")
    cs = sub.add_parser("convert-speaker", help="convert the open speaker-recognition model to ONNX")
    cs.add_argument("--source", default="speechbrain/spkrec-ecapa-voxceleb", help="Hugging Face id or local folder")
    cs.add_argument("--out", required=True)
    cs.add_argument("--cache", default="models/speaker-source", help="where the downloaded model is kept")
    sv = sub.add_parser("synth-voices", help="make synthetic test voices with macOS text-to-speech")
    sv.add_argument("--out", required=True)
    sv.add_argument("--voices", type=int, default=8)
    se = sub.add_parser("speaker-eval", help="measure speaker recognition on test group recordings")
    se.add_argument("--model", required=True, help="speaker.onnx")
    se.add_argument("--voices", required=True, help="folder with one sub-folder of .wav files per speaker")
    se.add_argument("--sessions", type=int, default=20)
    se.add_argument("--group-size", type=int, default=4)
    se.add_argument("--snr", type=float, default=15.0, help="speech-to-classroom-noise ratio in dB")
    se.add_argument("--report", default=None, help="write the full report JSON here")
    a = p.parse_args(argv)

    if a.cmd == "synth":
        from .synthetic import write_dataset
        paths = write_dataset(a.out, n_sessions=a.sessions, seed=a.seed)
        print("wrote", *paths.values(), sep="\n  ")
        print("Synthetic data is for testing the pipeline only; its scores mean nothing about real accuracy.")
    elif a.cmd == "check":
        from . import data
        d = Path(a.data)
        u = data.load_utterances(d / "utterances.csv")
        pr = data.load_pairs(d / "pairs.csv")
        r = data.load_segment_ratings(d / "segments.csv")
        print(f"OK: {len(u)} utterances in {u['session_id'].nunique()} sessions, {len(pr)} pairs, "
              f"{len(r)} ratings ({int(r['level'].isna().sum())} NE)")
        if "speaker" not in u.columns:
            print("WARNING: utterances.csv has no `speaker` column; the scorers need it.")
    elif a.cmd == "train":
        from .pipeline import run
        rep = run(a.data, a.out, base=a.base, idea_base=a.idea_base, epochs=a.epochs, idea_epochs=a.idea_epochs,
                  device=a.device, seed=a.seed)
        print(json.dumps({k: rep[k] for k in ("critical_thinking_scorer", "creativity_scorer")}, indent=2))
    elif a.cmd == "export":
        from .export import export
        export(a.run, a.out)
    elif a.cmd == "convert-speaker":
        from .speaker_export import convert as convert_speaker
        convert_speaker(a.source, a.out, savedir=a.cache)
    elif a.cmd == "synth-voices":
        from .speaker_eval import synth_voices
        synth_voices(a.out, n_voices=a.voices)
        print("Synthetic voices are for testing only; real accuracy needs real, consented recordings.")
    elif a.cmd == "speaker-eval":
        from .speaker_eval import evaluate
        evaluate(a.model, a.voices, out=a.report, sessions=a.sessions, group_size=a.group_size, snr_db=a.snr)
    elif a.cmd == "convert-emotion":
        from .emotion_export import convert
        convert(a.model_dir, a.out, clip_dir=a.clips, int8=not a.no_int8, version=a.version)


if __name__ == "__main__":
    main()
