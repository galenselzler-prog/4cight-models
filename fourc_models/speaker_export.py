# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Convert an open speaker-embedding model (SpeechBrain ECAPA-TDNN, trained
on VoxCeleb, Apache-2.0) into the ONNX file the app uses to tell students
apart.

  speaker.onnx     raw 16 kHz mono audio -> unit-length voice embedding (192 numbers)
  manifest.json    ModelSpec + input format + checks

Two recordings of the same person give embeddings pointing the same way
(cosine similarity near 1); different people point apart. The app compares
each few seconds of classroom speech with the voiceprints made at enrollment.

SpeechBrain computes its filterbank features with a complex-valued FFT that
ONNX cannot export, so the same maths (Hamming window, 25 ms / 10 ms,
power spectrum) is rebuilt here from ordinary convolutions and checked
against SpeechBrain's own output before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .licenses import license_for

SAMPLE_RATE = 16000
DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class ConvPowerSpectrum(nn.Module):
    """|STFT|^2 identical to torch.stft(center=True, pad_mode='constant',
    onesided=True) followed by SpeechBrain's spectral_magnitude(power=1),
    written as two strided convolutions so it exports to ONNX."""

    def __init__(self, n_fft: int, win_length: int, hop_length: int, window: torch.Tensor):
        super().__init__()
        if win_length != n_fft:
            raise ValueError("only win_length == n_fft is supported")
        self.n_fft, self.hop = n_fft, hop_length
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        ang = 2 * math.pi * k * n / n_fft
        w = window.to(torch.float64)
        self.register_buffer("cos", (torch.cos(ang) * w).float()[:, None, :])
        self.register_buffer("sin", (-torch.sin(ang) * w).float()[:, None, :])

    def forward(self, audio):  # [B, N] -> [B, frames, bins]
        x = nn.functional.pad(audio[:, None, :], (self.n_fft // 2, self.n_fft // 2))
        re = nn.functional.conv1d(x, self.cos, stride=self.hop)
        im = nn.functional.conv1d(x, self.sin, stride=self.hop)
        return (re * re + im * im).transpose(1, 2)


class SpeakerEmbeddingModel(nn.Module):
    """audio [B, N] -> L2-normalised embedding [B, D]."""

    def __init__(self, compute_features, mean_var_norm, embedding_model):
        super().__init__()
        stft = compute_features.compute_STFT
        if getattr(compute_features, "deltas", False) or getattr(compute_features, "context", False):
            raise ValueError("Fbank with deltas/context is not supported")
        if stft.normalized_stft or not stft.center or stft.pad_mode != "constant" or not stft.onesided:
            raise ValueError("unsupported STFT settings")
        self.spectrum = ConvPowerSpectrum(stft.n_fft, stft.win_length, stft.hop_length, stft.window)
        self.fbanks = compute_features.compute_fbanks
        if mean_var_norm.norm_type != "sentence":
            raise ValueError(f"unsupported input normalisation {mean_var_norm.norm_type!r}")
        self.std_norm = bool(mean_var_norm.std_norm)
        self.embedding_model = embedding_model

    def forward(self, audio):
        feats = self.fbanks(self.spectrum(audio))
        feats = feats - feats.mean(1, keepdim=True)
        if self.std_norm:
            feats = feats / feats.std(1, keepdim=True).clamp(min=1e-10)
        emb = self.embedding_model(feats).squeeze(1)
        return nn.functional.normalize(emb, dim=-1)


def load(source: str = DEFAULT_SOURCE, savedir: str | Path | None = None):
    """Loads a SpeechBrain speaker model (Hugging Face id or local folder with
    hyperparams.yaml). Returns (export module, reference encoder, info)."""
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(source=str(source), savedir=str(savedir) if savedir else None,
                                         run_opts={"device": "cpu"})
    enc.mods.eval()
    model = SpeakerEmbeddingModel(enc.mods.compute_features, enc.mods.mean_var_norm,
                                  enc.mods.embedding_model).eval()
    base = str(source) if not Path(str(source)).exists() else f"local:{Path(source).name}"
    return model, enc, {"base": base, "dim": int(model(torch.zeros(1, SAMPLE_RATE)).shape[-1]),
                        "params": sum(p.numel() for p in model.parameters())}


def _check_clips() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    out = []
    for sec, f0 in [(1.0, 140), (2.7, 210), (4.3, 110)]:
        t = np.arange(int(sec * SAMPLE_RATE)) / SAMPLE_RATE
        voice = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 8)) * (1 + np.sin(2 * np.pi * 3 * t))
        out.append((0.1 * voice + 0.01 * rng.standard_normal(len(t))).astype(np.float32))
    return out


def convert(source: str = DEFAULT_SOURCE, out_dir="dist/speaker", savedir=None, version: str = "1.0.0",
            log=print) -> dict:
    import onnxruntime as ort

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, enc, info = load(source, savedir)
    log(f"  loaded {info['params'] / 1e6:.1f}M-parameter speaker model ({info['base']}), {info['dim']}-number voiceprints")

    clips = _check_clips()
    with torch.no_grad():
        ours = [model(torch.from_numpy(c)[None])[0].numpy() for c in clips]
        theirs = [nn.functional.normalize(enc.encode_batch(torch.from_numpy(c)[None]).squeeze(1), dim=-1)[0].numpy()
                  for c in clips]
    rebuild = max(float(np.abs(a - b).max()) for a, b in zip(ours, theirs))
    if rebuild > 1e-3:
        raise RuntimeError(f"rebuilt features differ from SpeechBrain's by {rebuild}")
    log(f"  matches SpeechBrain's own pipeline (max diff {rebuild:.1e})")

    path = out / "speaker.onnx"
    torch.onnx.export(model, (torch.zeros(1, SAMPLE_RATE),), str(path), input_names=["audio"],
                      output_names=["embedding"],
                      dynamic_axes={"audio": {0: "batch", 1: "samples"}, "embedding": {0: "batch"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = [sess.run(["embedding"], {"audio": c[None]})[0][0] for c in clips]
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ours, got))
    if diff > 1e-3:
        raise RuntimeError(f"speaker.onnx differs from PyTorch by {diff}")
    log(f"  speaker.onnx verified on {len(clips)} clips (max diff {diff:.1e})")

    spec = {"id": "fourc-speaker", "task": "speaker", "version": version, "format": "onnx", "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sizeBytes": path.stat().st_size,
            "baseModel": info["base"], "baseModelLicense": license_for(info["base"]),
            "trainingData": "VoxCeleb 1+2 (base model; not fine-tuned by 4Cight yet)"}
    manifest = {"models": [spec], "embeddingDim": info["dim"],
                "input": {"name": "audio", "sampleRate": SAMPLE_RATE, "channels": 1, "dtype": "float32",
                          "range": [-1, 1], "recommendedClipSec": [1.5, 10]},
                "outputs": ["embedding"], "checks": {"speechbrain_max_diff": rebuild, "onnx_max_diff": diff}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"  wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return manifest
