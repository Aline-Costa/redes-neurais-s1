"""Transformer encoder and BERT-style model assembly."""

from __future__ import annotations

from typing import Optional

from bert_cpu.attention import MultiHeadAttention
from bert_cpu.engine import Tensor
from bert_cpu.nn import Dropout, Embedding, LayerNorm, Linear, Module


class PositionwiseFeedForward(Module):
    """Two-layer feed-forward network with GELU: ``Linear -> GELU -> Linear``."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class EncoderLayer(Module):
    """A single Transformer encoder block.

    Post-norm/pre-norm residual structure around multi-head self-attention and
    a position-wise feed-forward network.
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError


class TransformerEncoder(Module):
    """Stack of ``num_layers`` :class:`EncoderLayer` blocks."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError


class BERTEmbeddings(Module):
    """Token + positional + (optional) segment embeddings, summed and normed."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        raise NotImplementedError

    def forward(self, input_ids, token_type_ids=None) -> Tensor:
        raise NotImplementedError


class BERTModel(Module):
    """BERT-style bidirectional Transformer encoder.

    Embeds the input ids, runs the encoder stack, and returns the per-token
    contextual representations (the encoder's last hidden state).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        d_ff: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        raise NotImplementedError

    def forward(self, input_ids, attention_mask=None, token_type_ids=None) -> Tensor:
        raise NotImplementedError


class BERTForMaskedLM(Module):
    """BERT encoder with a masked-language-modelling head for pretraining."""

    def __init__(self, bert: BERTModel, vocab_size: int) -> None:
        raise NotImplementedError

    def forward(self, input_ids, attention_mask=None, token_type_ids=None) -> Tensor:
        """Return vocabulary logits of shape ``(batch, seq, vocab_size)``."""
        raise NotImplementedError
