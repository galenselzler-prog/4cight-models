# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Speaker-model conversion, checked offline with a tiny random ECAPA laid
out exactly like speechbrain/spkrec-ecapa-voxceleb (same hyperparams file
structure, local checkpoint files)."""

import numpy as np
import onnxruntime as ort
import pytest
import torch

HPARAMS = """
n_mels: 80
pretrained_path: {path}
compute_features: !new:speechbrain.lobes.features.Fbank
    n_mels: !ref <n_mels>
mean_var_norm: !new:speechbrain.processing.features.InputNormalization
    norm_type: sentence
    std_norm: False
embedding_model: !new:speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN
    input_size: !ref <n_mels>
    channels: [32, 32, 32, 32, 96]
    kernel_sizes: [5, 3, 3, 3, 1]
    dilations: [1, 2, 3, 4, 1]
    attention_channels: 16
    lin_neurons: 24
modules:
    compute_features: !ref <compute_features>
    mean_var_norm: !ref <mean_var_norm>
    embedding_model: !ref <embedding_model>
pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    loadables:
        embedding_model: !ref <embedding_model>
    paths:
        embedding_model: !ref <pretrained_path>/embedding_model.ckpt
"""


@pytest.fixture
def tiny_source(tmp_path):
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    torch.manual_seed(0)
    m = ECAPA_TDNN(80, channels=[32, 32, 32, 32, 96], kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                   attention_channels=16, lin_neurons=24)
    src = tmp_path / "src"
    src.mkdir()
    torch.save(m.state_dict(), src / "embedding_model.ckpt")
    (src / "hyperparams.yaml").write_text(HPARAMS.format(path=src))
    return src


def test_conv_spectrum_matches_torch_stft():
    from fourc_models.speaker_export import ConvPowerSpectrum
    win = torch.hamming_window(400)
    x = torch.randn(2, 12345)
    ref = torch.view_as_real(torch.stft(x, 400, 160, 400, win, True, "constant", False, True,
                                        return_complex=True)).pow(2).sum(-1).transpose(1, 2)
    got = ConvPowerSpectrum(400, 400, 160, win)(x)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-3)


def test_convert_tiny_ecapa(tiny_source, tmp_path):
    from fourc_models.speaker_export import convert
    man = convert(str(tiny_source), tmp_path / "out", savedir=tmp_path / "cache", log=lambda *_: None)
    assert man["embeddingDim"] == 24 and man["models"][0]["task"] == "speaker"
    assert man["checks"]["speechbrain_max_diff"] < 1e-3 and man["checks"]["onnx_max_diff"] < 1e-3
    s = ort.InferenceSession(str(tmp_path / "out" / "speaker.onnx"))
    e = s.run(["embedding"], {"audio": (np.random.default_rng(0).standard_normal((3, 30000)) * 0.1).astype(np.float32)})[0]
    assert e.shape == (3, 24) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)


def _burst_clip(total, bursts, seed=1):
    rng = np.random.default_rng(seed)
    x = 0.001 * rng.standard_normal(int(total * 16000))
    for a, b in bursts:
        t = np.arange(int(a * 16000), int(b * 16000))
        x[t] += 0.3 * np.sin(2 * np.pi * 220 * t / 16000) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t / 16000))
    return x.astype(np.float32)


def test_detect_speech_mirrors_app_behaviour():
    """Same scenario as the app's engine test (engine-tests/live.test.cjs)."""
    from fourc_models.speaker_eval import detect_speech
    r = [(a / 16000, b / 16000) for a, b in detect_speech(_burst_clip(10, [(1, 2.5), (2.6, 3.0), (6, 7.5), (9, 9.05)]))]
    assert len(r) == 2
    assert abs(r[0][0] - 1) < 0.05 and abs(r[0][1] - 3.0) < 0.05
    assert abs(r[1][0] - 6) < 0.05 and abs(r[1][1] - 7.5) < 0.05


def test_split_segments():
    from fourc_models.speaker_eval import split_segments
    # 4 s region -> 1.5 + 1.5 + 1.0 ; 1.7 s -> 1.5 + 0.2 tail merged -> one 1.7 piece; 0.5 s dropped
    assert split_segments([(0, 64000), (100000, 127200), (200000, 208000)], 24000, 12000) == [
        (0, 24000), (24000, 48000), (48000, 64000), (100000, 127200)]


def test_speaker_eval_end_to_end(tiny_source, tmp_path):
    """Plumbing only: voices are distinct synthetic tones, the model is random."""
    import wave
    from fourc_models.speaker_eval import evaluate
    from fourc_models.speaker_export import convert
    convert(str(tiny_source), tmp_path / "m", savedir=tmp_path / "cache", log=lambda *_: None)
    for k, f0 in enumerate([110, 180, 260, 400]):
        d = tmp_path / "voices" / f"v{k}"
        d.mkdir(parents=True)
        for i in range(10):
            t = np.arange(int(2.5 * 16000)) / 16000
            x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6)) * (1 + 0.5 * np.sin(2 * np.pi * (2 + i % 3) * t))
            with wave.open(str(d / f"{i}.wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes((x / np.abs(x).max() * 12000).astype(np.int16).tobytes())
    rep = evaluate(tmp_path / "m" / "speaker.onnx", tmp_path / "voices", out=tmp_path / "r.json", sessions=3,
                   group_size=3, log=lambda *_: None)
    assert rep["voices"] == 4 and rep["speech_minutes"] > 0
    assert any(s["outsider_credited"] is not None for s in rep["sweep"]), "sessions include an unenrolled voice"
    rec = rep["recommended"]
    assert abs(rec["correct"] + rec["wrong"] + rec["unknown"] - 1) < 1e-6
    assert (tmp_path / "r.json").exists()
