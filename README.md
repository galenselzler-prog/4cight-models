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

## Release gates (real data)

| Piece | Gate |
|---|---|
| Utterance heads | Test macro-F1 and kappa at or near labeler-vs-labeler agreement |
| Idea encoder | Held-out pair macro-F1 ≥ 0.75; idea-link macro-F1 reported |
| Scorers | Test QWK ≥ 0.60 and ≥ 90% within one level, per grade band |

## Still to do

- Wire into the app: add `'utterance' | 'idea'` to `ModelTask` in `src/engine/models/runtime.ts`,
  run the tokenizers in TypeScript, and port `ct_features`, `link_ideas` and the scorer math
  (all plain arithmetic).
- Per-grade-band and fairness breakdowns in the report once real data exists.
