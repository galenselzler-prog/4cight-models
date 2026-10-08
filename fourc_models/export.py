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
written to the manifest. By default both models are also quantised to int8
(*.int8.onnx, about 4x smaller) and the int8 copy is what the manifest ships,
but only if it agrees with full precision (MIN_INT8_AGREEMENT per utterance
head, MIN_INT8_COSINE for idea vectors); otherwise the fp32 file ships and a
warning is printed. Pass the training data folder to check on its test
utterances instead of a few built-in sentences."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import idea_model, utterance_model
from . import labels as L
from .creativity_metrics import IdeaBank
from .licenses import license_for

MIN_INT8_AGREEMENT = 0.95  # top label per utterance head, int8 vs fp32
MIN_INT8_COSINE = 0.98     # mean cosine between int8 and fp32 idea vectors
_CHECK_SENTENCES = [
    "I think the bridge fell because the base was too narrow",
    "what if we use straws instead of tape",
    "can you pass me the scissors",
    "no that is wrong, it already failed twice",
    "maybe we combine both ideas and make it taller",
    "I am bored",
    "my evidence is that the heavier side tipped first",
    "let's split up the jobs, you measure and I cut",
]


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
            "baseModel": base, "baseModelLicense": license_for(base),
            "trainingData": "4Cight classroom labels (guide v1.2); see training report"}


def _check_texts(data_dir, sample: list[str]) -> tuple[list[str], list[str], str]:
    """(utterance-model inputs, plain texts, description) used to compare int8 with fp32."""
    if data_dir is None:
        texts = sample + _CHECK_SENTENCES
        return texts, texts, "built-in sentences"
    from .data import load_utterances
    u = load_utterances(Path(data_dir) / "utterances.csv")
    t = u[u["split"] == "test"]
    t = (t if len(t) else u).head(1000)
    return utterance_model._texts(t), list(t["text"]), f"{len(t)} test utterances"


def _run_onnx(path: Path, tok, texts: list[str], max_len: int, batch: int = 32) -> list[np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = None
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="np")
        r = sess.run(None, {"input_ids": b["input_ids"].astype(np.int64),
                            "attention_mask": b["attention_mask"].astype(np.int64)})
        outs = r if outs is None else [np.concatenate([a, c]) for a, c in zip(outs, r)]
    return outs


def _quantize(fp32: Path) -> Path:
    """Weight-only int8. Gather is included because the token-embedding table is
    most of a DeBERTa-v3 model (128k-token vocabulary). Per-channel scales keep
    small encoders close to fp32 (MiniLM: mean cosine 0.947 per-tensor, 0.990
    per-channel, same 23 MB)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q = fp32.with_name(fp32.stem + ".int8.onnx")
    quantize_dynamic(str(fp32), str(q), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm", "Gather"], per_channel=True)
    return q


def _int8(out: Path, utok, itok, data_dir, sample: list[str], log) -> tuple[Path, Path, dict]:
    u32, i32 = out / "utterance.onnx", out / "idea.onnx"
    uq, iq = _quantize(u32), _quantize(i32)
    utexts, itexts, src = _check_texts(data_dir, sample)
    ref, got = _run_onnx(u32, utok, utexts, utterance_model.MAX_LEN), _run_onnx(uq, utok, utexts, utterance_model.MAX_LEN)
    agree = {h: round(float(np.mean(a.argmax(1) == b.argmax(1))), 4) for h, a, b in zip(utterance_model.HEADS, ref, got)}
    e32 = _run_onnx(i32, itok, itexts, idea_model.MAX_LEN)[0]
    e8 = _run_onnx(iq, itok, itexts, idea_model.MAX_LEN)[0]
    cos = (e32 * e8).sum(1) / (np.linalg.norm(e32, axis=1) * np.linalg.norm(e8, axis=1) + 1e-9)
    checks = {"int8_checked_on": src, "utterance_int8_top_label_agreement": agree,
              "idea_int8_mean_cosine": round(float(cos.mean()), 4), "idea_int8_min_cosine": round(float(cos.min()), 4)}
    log(f"  int8 checked on {src}: utterance top-label agreement "
        + ", ".join(f"{h} {v:.0%}" for h, v in agree.items())
        + f"; idea mean cosine {cos.mean():.3f}")
    ship_u = uq if min(agree.values()) >= MIN_INT8_AGREEMENT else u32
    ship_i = iq if cos.mean() >= MIN_INT8_COSINE else i32
    for shipped, fp in ((ship_u, u32), (ship_i, i32)):
        if shipped == fp:
            log(f"  WARNING: {fp.stem}.int8.onnx differs too much from full precision; manifest ships {fp.name}")
    if data_dir is None:
        log("  note: pass --data <training data folder> to check int8 on real test utterances")
    return ship_u, ship_i, checks


def export(run_dir, out_dir, log=print, data_dir=None, int8: bool = True) -> dict:
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
    ship_u, ship_i, checks = out / "utterance.onnx", out / "idea.onnx", {"fp32_max_diff": max(d1, d2)}
    if int8:
        ship_u, ship_i, q = _int8(out, utok, itok, data_dir, sample, log)
        checks.update(q)

    scorer_files = [f"scorer_{k}.json" for k in L.RATED_SKILLS if (run / f"scorer_{k}.json").exists()]
    for f in scorer_files:
        shutil.copy(run / f, out / f)
    bank = IdeaBank.load(run / "idea_bank")
    (out / "idea_bank.json").write_text(json.dumps({
        "version": version, "thresholds": th,
        "activities": {a: {"centroids": np.round(c, 5).tolist(), "sessions": bank.sessions[a],
                           "kinds": bank.kinds[a], "nSessions": bank.n_sessions[a]} for a, c in bank.centroids.items()}}))

    manifest = {
        "version": version,
        "models": [_spec(ship_u, "fourc-utterance", "utterance", version, report["base"]),
                   _spec(ship_i, "fourc-idea", "idea", version, report["idea_base"])],
        "outputs": {ship_u.name: list(utterance_model.HEADS), ship_i.name: ["embedding"]},
        "labels": utterance_model.HEADS,
        "maxLen": {"utterance": utterance_model.MAX_LEN, "idea": idea_model.MAX_LEN},
        "ideaThresholds": th,
        "scorers": scorer_files,
        "ideaBank": "idea_bank.json",
        "sizes": {p.name: p.stat().st_size for p in sorted(out.glob("*.onnx"))},
        "checks": checks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"  exported to {out}: ships {ship_u.name} ({ship_u.stat().st_size / 1e6:.0f} MB) and "
        f"{ship_i.name} ({ship_i.stat().st_size / 1e6:.0f} MB)")
    return manifest
