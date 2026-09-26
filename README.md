# 4cight-models

Copyright © 2026 4Cight Inc. All rights reserved. Proprietary and confidential.

Trains 4Cight's own **critical-thinking** and **creativity** models from the labels defined in the
*Classroom Sentiment Labeling Guide* (v1.2) and exports them for the ILS app (iPhone, iPad, web).
No third-party scoring service is involved. Open base models are used only as a starting point, and
their licenses are recorded in the manifest.

## How it scores

```
utterance text ──► utterance model ──► sentiment · move · CT skill · argument part
                   (one encoder,           │
                    4 heads, ONNX)         ├─► CT evidence per person ──► CT scorer ──► level 1-4 + "because"
                                           │
idea-bearing turns ► idea encoder ──► links (new/repeat/develops/combines)
                   (ONNX)                 + IdeaBank originality ──► creativity evidence ──► creativity scorer ──► level 1-4 + "because"
```

| Piece | File | What it learns from |
|---|---|---|
| Utterance model | `utterance_model.py` | Utterance labels (`sentiment`, `move`, `ct_skill`, `argument`) |
| Idea encoder | `idea_model.py` | Same-idea pairs and the idea links in whole sessions |
| CT evidence | `ct_features.py` | Nothing to learn: rates of each Delphi skill and argument part, claim→evidence and challenge→revision chains |
| Creativity evidence | `creativity_metrics.py` | Fluency, flexibility, originality (IdeaBank per `activity_id`) and elaboration |
| Scorers | `scorer.py` | Teacher segment ratings (ordinal 1–4; NE ratings excluded) |

The scorers are small on purpose. A linear ordinal model on readable evidence gives every score a
"because" list for teachers, trains on a few hundred ratings, and ships as JSON that the TypeScript
engine runs directly.

The scorers are trained and tested on features computed from **model predictions**, the same way
the device will compute them. The report also shows the critical-thinking scorer on **gold** labels,
which is the ceiling if the utterance model were perfect. It shows where to spend labeling effort.

## Install (Mac)

```bash
cd ~/Developer/ILS-App/4cight-models
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q            # about a minute; tiny offline models, synthetic data
```

## Data

Put the three files from the labeling workflow in one folder: `utterances.csv`, `pairs.csv` and
`segments.csv`, with the columns in the guide's *Output format* section. Two additions:

- `speaker` (required for scoring): the de-identified placeholder for who said the utterance
  (`[STUDENT_A]`). It must match `rated` in `segments.csv`.
- `segment_id` (optional): if segments are shorter than whole sessions, tag each utterance with its
  segment. Without it, the whole session is the segment.

Keep one whole session inside one split. The loader refuses data that breaks this.

```bash
fourc check --data data/labeled
```

## Train and export

```bash
# Real run (downloads the open base models once; uses Apple GPU via MPS)
fourc train --data data/labeled --out runs/2026-10-r1 \
  --base microsoft/deberta-v3-small --idea-base sentence-transformers/all-MiniLM-L6-v2

fourc export --run runs/2026-10-r1 --out dist/2026-10-r1
```

`runs/<name>/report.json` holds validation and test metrics for every piece. `dist/<name>/` holds
what the app ships: `utterance.onnx`, `idea.onnx`, both tokenizers, the two scorer JSON files,
`idea_bank.json` and `manifest.json` (ModelSpec entries with sha256 and base-model license). Every
ONNX file is checked against PyTorch before it is written.

`data/`, `runs/` and `dist/` are git-ignored. Student data and weights never go in the repo.

## Try it without real data

```bash
fourc synth --out data/synthetic
fourc train --data data/synthetic --out runs/synthetic      # tiny model, about a minute on CPU
fourc export --run runs/synthetic --out dist/synthetic
```

Synthetic data only proves the plumbing works. Its scores say nothing about real accuracy.

## Existing M2 emotion model → app

The 2024 speech-emotion model (wav2vec2-large, 316M parameters, labels background / curiosity /
happy / negativity / neutral) feeds the communication and collaboration scores. Convert it once:

```bash
# models/m2-emotion/ = config.json, preprocessor_config.json, model.safetensors from the Drive "Models/model" folder
fourc convert-emotion --model-dir models/m2-emotion --out dist/emotion
# optional: also check the conversion on real classroom clips (16-bit WAV)
fourc convert-emotion --model-dir models/m2-emotion --out dist/emotion --clips clips/
```

Output: `emotion.onnx` (full precision, about 1.3 GB, reference copy) and `emotion.int8.onnx`
(about 355 MB, the one the app loads). Both take raw 16 kHz mono audio and normalise it inside the
graph. The int8 copy is fine for iPhone and iPad; it is heavy for Chromebooks, so a smaller
distilled emotion model is the follow-up for the web app.

## Speaker recognition (who is talking)

The app tells students apart with voiceprints made at enrollment. The voice model is SpeechBrain's
ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, Apache-2.0, 21M parameters, 84 MB), converted once:

```bash
fourc convert-speaker --out dist/speaker        # downloads the open model once, checks it, writes speaker.onnx
```

Measure it before any student is recorded. `synth-voices` makes test voices with macOS text-to-speech;
`speaker-eval` builds noisy group recordings from them (who spoke when is known), runs the same steps
as the app, and recommends the match threshold for `src/config.ts`:

```bash
fourc synth-voices --out voices/synthetic
fourc speaker-eval --model dist/speaker/speaker.onnx --voices voices/synthetic --report dist/speaker/eval.json
```

Synthetic voices are cleaner and more different from each other than children in a classroom, so
their numbers are a best case. For numbers you can trust, put real consented recordings (or Mozilla
Common Voice clips) in `voices/<speaker>/*.wav` and run `speaker-eval` again. `voices/` is git-ignored.

Voiceprints are biometric data about children: the app keeps them in memory for the current session
only, never uploads or saves them. Keeping them between sessions needs a legal/privacy decision first.

## Release gates (real data)

| Piece | Gate |
|---|---|
| Utterance heads | Test macro-F1 and kappa at or near labeler-vs-labeler agreement |
| Idea encoder | Held-out pair macro-F1 ≥ 0.75; idea-link macro-F1 reported |
| Scorers | Test QWK ≥ 0.60 and ≥ 90% within one level, per grade band |
| Speaker recognition | ≥ 90% of talk time on the right student, ≤ 5% on the wrong one, on real classroom recordings |

## Still to do

- Wire into the app: add `'utterance' | 'idea'` to `ModelTask` in `src/engine/models/runtime.ts`,
  run the tokenizers in TypeScript, and port `ct_features`, `link_ideas` and the scorer math
  (all plain arithmetic).
- Per-grade-band and fairness breakdowns in the report once real data exists.
