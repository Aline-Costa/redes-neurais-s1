"""Tests for the layers and the assembled encoder."""

import numpy as np

from bert_cpu.transformer import BERTModel, EncoderLayer
from bert_cpu.attention import MultiHeadAttention


def test_encoder_layer_shape():
    """An encoder layer preserves ``(batch, seq, d_model)``."""
    raise NotImplementedError


def test_multihead_attention_shape():
    raise NotImplementedError


def test_bert_forward_shape():
    """BERTModel returns ``(batch, seq, d_model)`` hidden states."""
    raise NotImplementedError


def test_overfit_tiny_batch():
    """Smoke test: the model can drive the loss down on a tiny fixed batch."""
    raise NotImplementedError
