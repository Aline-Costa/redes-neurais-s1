"""Loss functions.

Where does the loss sit relative to the model? *On the same graph.* A loss
function is **not** special machinery — it is simply a few more ``Tensor`` ops
appended to the front of the graph the model already built. The scalar it
returns becomes the graph's new **root**, and because it is stitched together
from differentiable engine ops (``softmax``, ``log``, ``*``, ``sum``, ``/``), its
``_prev`` chain runs unbroken all the way down to the model's parameters.

The crucial mental model: the ``logits`` handed to a loss are **not an input or
a leaf** — they are an *interior* node, the visible handle onto a deep model
subgraph beneath them::

        L                       <- the loss appends these few nodes
        |
       nll
        |
    logp = log(probs)
        |
     probs = softmax(logits)
        |
     logits    <--- one Tensor variable you can see...
        |
      (Wᵀ @ ...)                <- ...but a whole model hangs below it
        |
     x, W1, W2, W3              <- the real leaves (inputs + parameters)

So calling ``loss = cross_entropy(logits, targets)`` does not start a new
computation; it *extends the existing graph upward*. That is why a single
``loss.backward()`` suffices: it seeds ``dL/dL = 1`` at the new root ``L`` and
walks that one continuous ``_prev`` chain back to every parameter ``W``. The
loss's only job is to keep the chain intact — build the scalar from ``Tensor``
ops, never raw NumPy, or the chain snaps and ``backward()`` never reaches the
weights. (Contrast ``optim.step()``, which is deliberately *off*-graph: it only
*reads* the gradients this backward pass produces.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from bert_cpu.engine import Tensor


def cross_entropy(logits: Tensor, targets, ignore_index: int = -100) -> Tensor:
    """Mean softmax cross-entropy over a batch of predictions.

    Computes ``mean_over_valid_positions( -log softmax(logits)[correct class] )``
    and returns it as a **scalar ``Tensor`` still wired into the graph** — so
    ``loss.backward()`` propagates ``dL/dlogits = (softmax - one_hot) / N`` back
    through ``logits`` and into every parameter that produced them (see the
    module docstring: the loss is just more nodes on the same graph).

    Everything below is built from differentiable engine ops, with one
    exception: the label-derived arrays (``one_hot``, the validity mask) are
    plain **constants** (``requires_grad=False``). Labels are *data*, not
    parameters — gradient must flow into ``logits``, never into the targets — so
    they are computed once in NumPy and injected as constant tensors.

    Parameters
    ----------
    logits : Tensor
        Unnormalised scores of shape ``(..., num_classes)`` — the **class axis is
        last** (matches ``Tensor.softmax``'s default ``axis=-1``).
    targets : array-like of int
        Ground-truth class ids, one per position; shape ``logits.shape[:-1]``.
    ignore_index : int
        Target value whose positions are excluded from the loss (e.g. non-masked
        tokens during MLM training). Such positions contribute zero loss *and*
        zero gradient, and are left out of the averaging denominator.
    """
    num_classes = logits.shape[-1]
    targets = np.asarray(targets)

    # --- Label-derived CONSTANTS (no gradient flows into these) --------------
    # Which positions actually count toward the loss.
    valid = targets != ignore_index
    # Ignored positions may hold sentinel ids (e.g. -100) that would index out of
    # range, so clamp them to a safe class before building the one-hot.
    safe = np.where(valid, targets, 0)
    # one_hot[..., c] == 1 for the correct class c; rows of ignored positions are
    # zeroed out, so their contribution to nll below is exactly 0.
    one_hot = np.eye(num_classes, dtype=logits.data.dtype)[safe]
    one_hot = one_hot * valid[..., None]
    one_hot = Tensor(one_hot, requires_grad=False)   # a constant node in the graph

    # Number of positions we average over (guard the all-ignored edge case). This
    # is a constant scalar, not something we differentiate.
    n_valid = max(float(valid.sum()), 1.0)

    # --- Differentiable path: these ops extend the model's graph -------------
    # softmax already subtracts the per-row max internally (as untracked NumPy),
    # so probs are numerically stable and log(probs) = log_softmax. Composing the
    # engine's softmax and log reproduces the exact cross-entropy gradient by the
    # chain rule (verified in test/test_loss.py).
    probs = logits.softmax(axis=-1)                  # (..., C)
    logp = probs.log()                               # (..., C)
    # Multiplying by the constant one_hot and summing the class axis is the
    # in-graph way to "gather" each position's correct-class log-prob.
    nll = -(logp * one_hot).sum(axis=-1)             # (...) ; 0 at ignored positions
    # Mean over valid positions. Dividing a Tensor by a constant keeps the graph
    # intact, so the returned scalar's _prev chain still reaches logits.
    return nll.sum() / n_valid


def masked_lm_loss(logits: Tensor, labels, ignore_index: int = -100) -> Tensor:
    """Cross-entropy restricted to masked positions for MLM pretraining."""
    raise NotImplementedError
