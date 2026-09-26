# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Export a training run to what the app ships:

  utterance.onnx       input_ids, attention_mask -> sentiment, move, ct_skill, argument logits
  idea.onnx            input_ids, attention_mask -> unit-length idea vector
  *.tokenizer.json     tokenizers for both (run in TypeScript)
  scorer_*.json        ordinal scorers (run directly in the TypeScript engine)
  idea_bank.json       past ideas per activity for originality
  manifest.json        ModelSpec entries matching src/engine/models/runtime.ts

Each ONNX file is checked against PyTorch with onnxruntime before it is
written to the manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import idea_model, utterance_model
from .creativity_metrics import IdeaBank

LICENSES = {  # base model -> license, for IP due diligence (PROPRIETARY.md)
    "tiny": "none (randomly initialised, test only)",
    "microsoft/deberta-v3-small": "MIT",
    "microsoft/deberta-v3-xsmall": "MIT",
    "sentence-transformers/all-MiniLM-L6-v2": "Apache-2.0",
    "bert-base-uncased": "Apache-2.0",
}


class _Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids, attention_mask)


def _onnx(model, tok, path: Path, output_names: list[str], sample: list[str]) -> float:
    import onnxruntime as ort

    model = model.cpu().eval()
    b = tok(sample, padding=True, return_tensors="pt")
    args = (b["input_ids"], b["attention_mask"])
    axes = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
            **{o: {0: "batch"} for o in output_names}}
    torch.onnx.export(_Wrap(model).eval(), args, str(path), input_names=["input_ids", "attention_mask"],
                      output_names=output_names, dynamic_axes=axes, opset_version=17, dynamo=False)
    # Verify: same inputs through onnxruntime must match PyTorch.
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    b2 = tok(sample[::-1] + ["a much longer different sentence to change the padded length"], padding=True,
             return_tensors="pt")
    ort_out = sess.run(None, {"input_ids": b2["input_ids"].numpy().astype(np.int64),
                              "attention_mask": b2["attention_mask"].numpy().astype(np.int64)})
    with torch.no_grad():
        pt = model(b2["input_ids"], b2["attention_mask"])
    pt = pt if isinstance(pt, tuple) else (pt,)
    diff = max(float(np.abs(o - p.numpy()).max()) for o, p in zip(ort_out, pt))
    if diff > 1e-3:
        raise RuntimeError(f"{path.name}: ONNX output differs from PyTorch by {diff}")
    return diff


def _spec(path: Path, id_: str, task: str, version: str, base: str) -> dict:
    return {"id": id_, "task": task, "version": version, "format": "onnx", "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sizeBytes": path.stat().st_size,
            "baseModel": base, "baseModelLicense": LICENSES.get(base, "CHECK BEFORE SHIPPING"),
            "trainingData": "4Cight classroom labels (guide v1.2); see training report"}


def export(run_dir, out_dir, log=print) -> dict:
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((run / "report.json").read_text())
    version = report["version"]
    sample = ["what if we tape the wheels", "that will not work because it failed before [CTX] try again"]

    um, utok = utterance_model.load(run / "utterance", "cpu")
    d1 = _onnx(um, utok, out / "utterance.onnx", list(utterance_model.HEADS), sample)
    utok.backend_tokenizer.save(str(out / "utterance.tokenizer.json"))
    im, itok, th = idea_model.load(run / "idea", "cpu")
    d2 = _onnx(im, itok, out / "idea.onnx", ["embedding"], sample)
    itok.backend_tokenizer.save(str(out / "idea.tokenizer.json"))
    log(f"  ONNX verified (max diff {max(d1, d2):.2e})")

    for f in ("scorer_critical_thinking.json", "scorer_creativity.json"):
        shutil.copy(run / f, out / f)
    bank = IdeaBank.load(run / "idea_bank")
    (out / "idea_bank.json").write_text(json.dumps({
        "version": version, "thresholds": th,
        "activities": {a: {"centroids": np.round(c, 5).tolist(), "sessions": bank.sessions[a],
                           "kinds": bank.kinds[a], "nSessions": bank.n_sessions[a]} for a, c in bank.centroids.items()}}))

    manifest = {
        "version": version,
        "models": [_spec(out / "utterance.onnx", "fourc-utterance", "utterance", version, report["base"]),
                   _spec(out / "idea.onnx", "fourc-idea", "idea", version, report["idea_base"])],
        "outputs": {"utterance.onnx": list(utterance_model.HEADS), "idea.onnx": ["embedding"]},
        "labels": utterance_model.HEADS,
        "maxLen": {"utterance": utterance_model.MAX_LEN, "idea": idea_model.MAX_LEN},
        "ideaThresholds": th,
        "scorers": ["scorer_critical_thinking.json", "scorer_creativity.json"],
        "ideaBank": "idea_bank.json",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"  exported to {out}")
    return manifest
