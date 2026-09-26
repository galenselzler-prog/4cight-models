# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Emotion-model conversion, checked with tiny random wav2vec2 checkpoints
laid out exactly like the real ones (no download)."""

import json
import wave

import numpy as np
import onnxruntime as ort
import pytest
import torch
from safetensors.torch import save_file
from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification, Wav2Vec2Model

from fourc_models.emotion_export import convert, load

LABELS = {"0": "Background Noise", "1": "Curiosity", "2": "Happy", "3": "Negativity", "4": "Neutral"}


def _cfg():
    return Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=37,
                          conv_dim=(32, 32, 32), conv_kernel=(10, 3, 3), conv_stride=(5, 2, 2),
                          num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, do_stable_layer_norm=True,
                          feat_extract_norm="layer", vocab_size=55, id2label=LABELS,
                          label2id={v: int(k) for k, v in LABELS.items()}, classifier_proj_size=16)


def _write_pre(d):
    (d / "preprocessor_config.json").write_text(json.dumps({"do_normalize": True, "sampling_rate": 16000}))


@pytest.fixture
def speech_head_dir(tmp_path):
    """Like the 2024 M2 checkpoint: wav2vec2.* weights with old weight-norm names
    + classifier.dense / classifier.out_proj head + custom config keys."""
    torch.manual_seed(0)
    cfg = _cfg()
    enc = Wav2Vec2Model(cfg)
    sd = {"wav2vec2." + k.replace("parametrizations.weight.original0", "weight_g")
          .replace("parametrizations.weight.original1", "weight_v"): v.contiguous() for k, v in enc.state_dict().items()}
    sd.update({"classifier.dense.weight": torch.randn(32, 32) * 0.3, "classifier.dense.bias": torch.zeros(32),
               "classifier.out_proj.weight": torch.randn(5, 32), "classifier.out_proj.bias": torch.zeros(5)})
    save_file(sd, str(tmp_path / "model.safetensors"))
    d = cfg.to_dict()
    d.update(architectures=["Wav2Vec2ForSpeechClassification"], pooling_mode="mean",
             _name_or_path="lighteternal/wav2vec2-large-xlsr-53-greek")
    (tmp_path / "config.json").write_text(json.dumps(d))
    _write_pre(tmp_path)
    return tmp_path, enc, sd


def test_speech_head_matches_independent_math(speech_head_dir):
    d, enc, sd = speech_head_dir
    model, info = load(d)
    assert info["head"] == "speech-classification"
    assert info["labels"] == ["background", "curiosity", "happy", "negativity", "neutral"]
    x = torch.randn(1, 8000) * 0.2
    with torch.no_grad():
        z = (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-7)
        h = enc.eval()(z).last_hidden_state.mean(1)
        ref = torch.tanh(h @ sd["classifier.dense.weight"].T) @ sd["classifier.out_proj.weight"].T
        assert torch.allclose(model(x)[0], ref, atol=1e-5)


def test_convert_speech_head(speech_head_dir, tmp_path):
    d, _, _ = speech_head_dir
    man = convert(d, tmp_path / "out", log=lambda *_: None)
    assert man["checks"]["fp32_max_prob_diff"] < 1e-3
    assert man["models"][0]["file"] == "emotion.int8.onnx" and man["models"][0]["task"] == "emotion"
    assert man["models"][0]["baseModelLicense"] == "Apache-2.0"
    s = ort.InferenceSession(str(tmp_path / "out" / "emotion.int8.onnx"))
    p = s.run(["probs"], {"audio": np.zeros((2, 24000), np.float32)})[0]
    assert p.shape == (2, 5) and np.allclose(p.sum(1), 1, atol=1e-4)


def test_convert_hf_head_with_real_wav_clips(tmp_path):
    torch.manual_seed(1)
    Wav2Vec2ForSequenceClassification(_cfg()).save_pretrained(tmp_path / "m")
    _write_pre(tmp_path / "m")
    clips = tmp_path / "clips"
    clips.mkdir()
    for i, sr in enumerate([16000, 44100]):  # second clip checks resampling
        with wave.open(str(clips / f"c{i}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((np.sin(np.arange(sr * 2) / 7) * 8000).astype(np.int16).tobytes())
    man = convert(tmp_path / "m", tmp_path / "out", clip_dir=clips, log=lambda *_: None)
    assert man["checks"]["check_clips"] == "real" and man["checks"]["n_clips"] == 2
    assert man["models"][0]["baseModelLicense"] == "CHECK BEFORE SHIPPING"  # unknown base stays flagged
    assert man["checks"]["fp32_max_prob_diff"] < 1e-3
