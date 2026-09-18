"""Attention mechanisms for the Transformer encoder."""

from __future__ import annotations

from typing import Optional

from bert_cpu.engine import Tensor
from bert_cpu.nn import Dropout, Linear, Module


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[Dropout] = None,
) -> Tensor:
    """Compute ``softmax(q @ k^T / sqrt(d_k) + mask) @ v``.

    Parameters
    ----------
    q, k, v : Tensor
        Query, key and value tensors of shape ``(..., seq, d_k)``.
    mask : Tensor, optional
        Additive mask broadcast onto the attention scores (e.g. ``-inf`` at
        padding positions).
    dropout : Dropout, optional
        Dropout applied to the attention weights.
    """
    raise NotImplementedError


class MultiHeadAttention(Module):
    """Multi-head self-attention.

    Projects the input into ``num_heads`` query/key/value subspaces, applies
    scaled dot-product attention per head, concatenates and projects back.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        raise NotImplementedError

    def _split_heads(self, x: Tensor) -> Tensor:
        """Reshape ``(batch, seq, d_model)`` -> ``(batch, heads, seq, d_k)``."""
        raise NotImplementedError

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Inverse of :meth:`_split_heads`."""
        raise NotImplementedError

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError
