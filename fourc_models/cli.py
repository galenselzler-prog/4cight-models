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
  fourc labeling-setup --url https://fourcight-labeling.onrender.com   # once (needs LABEL_STUDIO_TOKEN, R2_*)
  fourc upload-models dist/emotion/emotion.int8.onnx dist/speaker/speaker.onnx   # app models -> private R2
  fourc prep-labeling                                               # new recordings -> labeling tasks
  fourc meeting-report --ami data/corpora/ami --icsi data/corpora/icsi/ICSI --out runs/meetings   # behavior features on public meetings
  fourc meeting-labels --ami data/corpora/ami --icsi data/corpora/icsi/ICSI --out runs/meetings   # weak utterance labels for pretraining
  fourc export-norms --ami data/corpora/ami --icsi data/corpora/icsi/ICSI --out runs/meetings/meeting-norms.json --ts ../ils-app/src/engine/reference/meetingNorms.ts
  fourc import-labels --utterances utt.json --ratings ratings.json --out data/labeled --grade-band 3-5
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
    e.add_argument("--data", default=None, help="training data folder: check int8 on its test utterances")
    e.add_argument("--no-int8", action="store_true", help="ship full-precision ONNX only")
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
    ls = sub.add_parser("labeling-setup", help="create the Label Studio projects and connect them to R2")
    ls.add_argument("--url", required=True, help="Label Studio address")
    um = sub.add_parser("upload-models", help="upload the app's .onnx files to the private R2 models bucket")
    um.add_argument("files", nargs="+")
    pl = sub.add_parser("prep-labeling", help="turn new research recordings in R2 into labeling tasks")
    pl.add_argument("--bucket", default=None)
    mr = sub.add_parser("meeting-report", help="behavior features on the public AMI/ICSI meeting corpora (annotations only)")
    mr.add_argument("--ami", default=None, help="folder from ami_public_manual_1.6.2.zip")
    mr.add_argument("--icsi", default=None, help="the ICSI folder inside ICSI_core_NXT.zip")
    mr.add_argument("--out", required=True)
    ml = sub.add_parser("meeting-labels", help="AMI/ICSI annotations -> weak utterance labels (move, argument, ct_skill, sentiment) + report")
    ml.add_argument("--ami", default=None)
    ml.add_argument("--icsi", default=None)
    ml.add_argument("--out", required=True)
    en = sub.add_parser("export-norms", help="build the real-meeting reference tables the app scores groups against")
    en.add_argument("--ami", default=None)
    en.add_argument("--icsi", default=None)
    en.add_argument("--out", required=True, help="JSON file")
    en.add_argument("--ts", default=None, help="also write the TypeScript module for the app")
    il = sub.add_parser("import-labels", help="Label Studio JSON exports -> training CSVs")
    il.add_argument("--utterances", required=True, help="export of the '4Cight utterances' project (JSON)")
    il.add_argument("--ratings", default=None, help="export of the '4Cight segment ratings' project (JSON)")
    il.add_argument("--out", required=True)
    il.add_argument("--grade-band", default=None, help="for recordings made without one: K-2, 3-5, 6-8, 9-12 or adult")
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
        print(json.dumps({k: rep[k] for k in ("critical_thinking_scorer", "creativity_scorer", "communication_scorer", "collaboration_scorer")}, indent=2))
    elif a.cmd == "export":
        from .export import export
        export(a.run, a.out, data_dir=a.data, int8=not a.no_int8)
    elif a.cmd == "labeling-setup":
        import os
        from .labeling import labeling_setup
        token = os.environ.get("LABEL_STUDIO_TOKEN")
        if not token:
            p.error("set LABEL_STUDIO_TOKEN (Label Studio -> Account & Settings -> Personal Access Token)")
        labeling_setup(a.url, token)
    elif a.cmd == "upload-models":
        from .labeling import upload_models
        upload_models(a.files)
    elif a.cmd == "prep-labeling":
        from .labeling import prep_labeling
        prep_labeling(a.bucket)
    elif a.cmd == "meeting-report":
        import pandas as pd
        from . import meetings
        if not (a.ami or a.icsi):
            p.error("give --ami and/or --icsi")
        parts = ([meetings.load_ami(a.ami)] if a.ami else []) + ([meetings.load_icsi(a.icsi)] if a.icsi else [])
        per_speaker, per_meeting = meetings.behavior_features(pd.concat(parts, ignore_index=True))
        print(meetings.write_report(per_speaker, per_meeting, a.out))
        print(f"\nWrote {a.out}/report.txt, meetings.csv, speakers.csv")
    elif a.cmd == "export-norms":
        from . import behavior, meetings
        if not (a.ami or a.icsi):
            p.error("give --ami and/or --icsi")
        norms = behavior.build_norms(
            meetings.one_mic_meetings(a.ami, a.icsi),
            source="AMI Meeting Corpus + ICSI Meeting Corpus, CC BY 4.0 (word timings only)")
        behavior.dump_norms(norms, a.out)
        if a.ts:
            Path(a.ts).parent.mkdir(parents=True, exist_ok=True)
            Path(a.ts).write_text(behavior.norms_as_typescript(norms))
        print(f"{norms['meetings']} meetings; rows per band:", {b['name']: b['rows'] for b in norms['bands']})
    elif a.cmd == "import-labels":
        from .labeling import import_labels
        import_labels(a.utterances, a.ratings, a.out, default_grade_band=a.grade_band)
        print(f"Next: fourc check --data {a.out}")
    elif a.cmd == "convert-speaker":
        from .speaker_export import convert as convert_speaker
        convert_speaker(a.source, a.out, savedir=a.cache)
    elif a.cmd == "synth-voices":
        from .speaker_eval import synth_voices
        synth_voices(a.out, n_voices=a.voices)
        print("Synthetic voices are for testing only; real accuracy needs real, consented recordings.")
    elif a.cmd == "meeting-labels":
        from .meeting_labels import export
        df = export(a.ami, a.icsi, a.out)
        print(f"wrote {len(df):,} weak-labeled utterances to {a.out}/meeting_weak_labels.csv and {a.out}/labels_report.txt")
    elif a.cmd == "speaker-eval":
        from .speaker_eval import evaluate
        evaluate(a.model, a.voices, out=a.report, sessions=a.sessions, group_size=a.group_size, snr_db=a.snr)
    elif a.cmd == "convert-emotion":
        from .emotion_export import convert
        convert(a.model_dir, a.out, clip_dir=a.clips, int8=not a.no_int8, version=a.version)


if __name__ == "__main__":
    main()
