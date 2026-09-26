# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Shared text-encoder plumbing for both models.

Real training starts from an open base model downloaded once on the Mac
(for example `microsoft/deberta-v3-small`, MIT). Tests and the synthetic
smoke run use `base="tiny"`: a small randomly initialised BERT plus a
tokenizer trained locally on the data, so nothing is downloaded.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel, PreTrainedTokenizerFast

TINY = "tiny"
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[CTX]"]


def train_local_tokenizer(texts: list[str], vocab_size: int = 2000) -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL))
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]", special_tokens=[("[CLS]", cls), ("[SEP]", sep)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                   additional_special_tokens=["[CTX]"])


def load_base(base: str, texts_for_tiny: list[str] | None = None, seed: int = 0):
    """Returns (tokenizer, encoder). `base` is a Hugging Face id, a local
    folder, or "tiny"."""
    torch.manual_seed(seed)
    if base == TINY:
        tok = train_local_tokenizer(texts_for_tiny or ["hello"])
        cfg = BertConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=128, max_position_embeddings=256)
        return tok, BertModel(cfg, add_pooling_layer=False)
    tok = AutoTokenizer.from_pretrained(base)
    if "[CTX]" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["[CTX]"]})
    enc = AutoModel.from_pretrained(base)
    enc.resize_token_embeddings(len(tok))
    return tok, enc


def reload_encoder(folder: str | Path):
    folder = Path(folder)
    return AutoTokenizer.from_pretrained(folder / "tokenizer"), AutoModel.from_pretrained(folder / "encoder")


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


class Pooled(nn.Module):
    """Encoder -> mean-pooled sentence vector. Mean pooling (not [CLS]) works
    for every base model family and exports cleanly to ONNX."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.dim = encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return mean_pool(out.last_hidden_state, attention_mask)


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple silicon Macs
    return torch.device("cpu")
