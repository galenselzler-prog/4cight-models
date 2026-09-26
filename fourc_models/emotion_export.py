# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Convert the existing M2 speech-emotion model (wav2vec2 + classification
head, 5 labels) into the ONNX files the app runs.

  emotion.onnx        full precision, the reference copy
  emotion.int8.onnx   about 4x smaller, the one the app ships
  manifest.json       ModelSpec entries + labels + audio format

The graph takes raw 16 kHz mono audio (float32, -1..1) and does the
per-clip normalisation itself, so the app never has to copy the Python
preprocessing. Outputs are `logits` and `probs` in the app's label order.

Both head layouts in circulation are supported:
  - "speech-classification" head (classifier.dense -> tanh -> classifier.out_proj)
    used by the 2024 M2 training notebook;
  - the Hugging Face Wav2Vec2ForSequenceClassification head (projector -> classifier).
"""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn

SAMPLE_RATE = 16000
#: Checkpoint label name -> the app's EmotionSignal key (src/engine/types.ts).
APP_LABELS = {"background noise": "background", "background": "background", "curiosity": "curiosity",
              "happy": "happy", "negativity": "negativity", "neutral": "neutral"}


class EmotionModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: str, head_weights: dict, pooling: str, normalize: bool):
        super().__init__()
        self.encoder, self.head, self.pooling, self.normalize = encoder, head, pooling, normalize
        h = encoder.config.hidden_size
        if head == "speech-classification":
            n = head_weights["classifier.out_proj.weight"].shape[0]
            self.dense, self.out_proj = nn.Linear(h, h), nn.Linear(h, n)
            self.dense.load_state_dict({"weight": head_weights["classifier.dense.weight"],
                                        "bias": head_weights["classifier.dense.bias"]})
            self.out_proj.load_state_dict({"weight": head_weights["classifier.out_proj.weight"],
                                           "bias": head_weights["classifier.out_proj.bias"]})
        elif head == "hf-sequence-classification":
            p, n = head_weights["projector.weight"].shape[0], head_weights["classifier.weight"].shape[0]
            self.projector, self.classifier = nn.Linear(h, p), nn.Linear(p, n)
            self.projector.load_state_dict({"weight": head_weights["projector.weight"],
                                            "bias": head_weights["projector.bias"]})
            self.classifier.load_state_dict({"weight": head_weights["classifier.weight"],
                                             "bias": head_weights["classifier.bias"]})
        else:
            raise ValueError(head)

    def forward(self, audio):
        x = audio
        if self.normalize:  # Wav2Vec2FeatureExtractor(do_normalize=True), per clip
            x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-7)
        hidden = self.encoder(x).last_hidden_state
        if self.head == "hf-sequence-classification":
            logits = self.classifier(self.projector(hidden).mean(1))
        else:
            pooled = {"mean": lambda t: t.mean(1), "sum": lambda t: t.sum(1),
                      "max": lambda t: t.max(1).values}[self.pooling](hidden)
            logits = self.out_proj(torch.tanh(self.dense(pooled)))
        return logits, torch.softmax(logits, -1)


def _load_state(model_dir: Path) -> dict:
    st = model_dir / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st))
    bin_ = model_dir / "pytorch_model.bin"
    if bin_.exists():
        return torch.load(bin_, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"{model_dir}: no model.safetensors or pytorch_model.bin")


def load(model_dir: str | Path) -> tuple[EmotionModel, dict]:
    """Returns (model in eval mode, info with labels, base model and head layout)."""
    from transformers import Wav2Vec2Model

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    pre_path = model_dir / "preprocessor_config.json"
    pre = json.loads(pre_path.read_text()) if pre_path.exists() else {}
    if pre.get("sampling_rate", SAMPLE_RATE) != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, preprocessor says {pre['sampling_rate']}")

    state = _load_state(model_dir)
    if {"classifier.dense.weight", "classifier.out_proj.weight"} <= state.keys():
        head = "speech-classification"
    elif {"projector.weight", "classifier.weight"} <= state.keys():
        head = "hf-sequence-classification"
    else:
        other = sorted(k for k in state if not k.startswith("wav2vec2."))
        raise ValueError(f"unknown classification head; non-encoder keys: {other[:10]}")
    head_weights = {k: v for k, v in state.items() if not k.startswith("wav2vec2.")}
    del state

    encoder, info = Wav2Vec2Model.from_pretrained(model_dir, output_loading_info=True)
    missing = [k for k in info["missing_keys"] if "masked_spec_embed" not in k]
    if missing:
        raise ValueError(f"encoder weights missing from checkpoint: {missing[:10]}")
    model = EmotionModel(encoder, head, head_weights, cfg.get("pooling_mode", "mean"),
                         pre.get("do_normalize", True)).eval()

    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    names = [id2label[i] for i in range(len(id2label))]
    unknown = [n for n in names if n.lower() not in APP_LABELS]
    if unknown:
        raise ValueError(f"labels {unknown} have no EmotionSignal key; update APP_LABELS and types.ts together")
    return model, {"labels": [APP_LABELS[n.lower()] for n in names], "checkpoint_labels": names,
                   "base": cfg.get("_name_or_path", "unknown"), "head": head,
                   "params": sum(p.numel() for p in model.parameters())}


def read_wav(path: str | Path) -> np.ndarray:
    """16-bit PCM WAV -> float32 mono at 16 kHz (linear resampling; fine for checks)."""
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        x = x.reshape(-1, w.getnchannels()).mean(1)
        sr = w.getframerate()
    if sr != SAMPLE_RATE:
        t = np.arange(0, len(x) / sr, 1 / SAMPLE_RATE)
        x = np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)
    return x


def _check_clips(clip_dir: str | Path | None) -> list[np.ndarray]:
    """Real clips if given, otherwise synthetic test signals of different lengths."""
    if clip_dir:
        clips = [read_wav(p) for p in sorted(Path(clip_dir).glob("*.wav"))]
        if clips:
            return clips
    rng = np.random.default_rng(0)
    t1, t2 = np.arange(int(1.0 * SAMPLE_RATE)) / SAMPLE_RATE, np.arange(int(3.3 * SAMPLE_RATE)) / SAMPLE_RATE
    return [(0.3 * np.sin(2 * np.pi * (200 + 300 * t1) * t1)).astype(np.float32),
            (0.1 * rng.standard_normal(len(t2))).astype(np.float32),
            (0.2 * np.sin(2 * np.pi * 150 * t2) * (1 + np.sin(2 * np.pi * 3 * t2))).astype(np.float32)]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def convert(model_dir, out_dir, clip_dir=None, int8: bool = True, version: str = "2.0.0", log=print) -> dict:
    import onnxruntime as ort

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, info = load(model_dir)
    log(f"  loaded {info['params'] / 1e6:.0f}M-parameter model ({info['head']} head), labels {info['labels']}")

    fp32 = out / "emotion.onnx"
    example = torch.zeros(1, SAMPLE_RATE)
    torch.onnx.export(model, (example,), str(fp32), input_names=["audio"], output_names=["logits", "probs"],
                      dynamic_axes={"audio": {0: "batch", 1: "samples"}, "logits": {0: "batch"}, "probs": {0: "batch"}},
                      opset_version=17, dynamo=False)

    clips = _check_clips(clip_dir)
    with torch.no_grad():
        ref = [model(torch.from_numpy(c)[None])[1].numpy()[0] for c in clips]
    sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    got = [sess.run(["probs"], {"audio": c[None]})[0][0] for c in clips]
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
    if diff > 1e-3:
        raise RuntimeError(f"emotion.onnx differs from PyTorch by {diff}")
    log(f"  emotion.onnx verified on {len(clips)} clips (max prob diff {diff:.1e})")
    report = {"fp32_max_prob_diff": diff, "check_clips": "real" if clip_dir else "synthetic", "n_clips": len(clips)}
    normalized = model.normalize
    del model, sess  # free ~2.5 GB before quantising the large model
    import gc
    gc.collect()

    shipped = fp32
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = out / "emotion.int8.onnx"
        # Only the transformer's matrix multiplies are quantised; the convolutional
        # front end stays float so it runs on every ONNX Runtime backend (incl. web).
        quantize_dynamic(str(fp32), str(q), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        qs = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
        qp = [qs.run(["probs"], {"audio": c[None]})[0][0] for c in clips]
        agree = float(np.mean([a.argmax() == b.argmax() for a, b in zip(ref, qp)]))
        qdiff = max(float(np.abs(a - b).max()) for a, b in zip(ref, qp))
        report.update(int8_top_label_agreement=agree, int8_max_prob_diff=qdiff)
        log(f"  emotion.int8.onnx: top label matches full precision on {agree:.0%} of clips, max prob diff {qdiff:.3f}")
        if clip_dir and agree < 0.9:
            log("  WARNING: int8 disagrees on >10% of real clips; ship emotion.onnx until this is investigated")
        shipped = q

    spec = {"id": "fourc-emotion", "task": "emotion", "version": version, "format": "onnx", "file": shipped.name,
            "sha256": _sha(shipped), "sizeBytes": shipped.stat().st_size, "baseModel": info["base"],
            "baseModelLicense": "CHECK BEFORE SHIPPING (see model card of the base model)",
            "trainingData": "4Cight M2 classroom emotion data (2024)"}
    manifest = {"models": [spec], "labels": info["labels"], "checkpointLabels": info["checkpoint_labels"],
                "input": {"name": "audio", "sampleRate": SAMPLE_RATE, "channels": 1, "dtype": "float32",
                          "range": [-1, 1], "normalizedInGraph": normalized,
                          "recommendedClipSec": [1, 10]},
                "outputs": ["logits", "probs"], "sizes": {p.name: p.stat().st_size for p in out.glob("*.onnx")},
                "checks": report}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"  wrote {out}: " + ", ".join(f"{k} {v / 1e6:.0f} MB" for k, v in manifest["sizes"].items()))
    return manifest
