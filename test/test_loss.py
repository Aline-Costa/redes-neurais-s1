"""Software tests for the loss functions.

The most valuable check here is again a **finite-difference gradient check**:
``cross_entropy`` adds no new backward rule — it is built from the engine's
``softmax``/``log``/``*``/``sum`` — so what we are really verifying is that the
loss stays wired into the graph and that ``loss.backward()`` produces the exact
cross-entropy gradient ``(softmax - one_hot) / N`` at the logits.

The numerical-gradient reference is reused from ``test_engine`` so there is a
single, well-documented finite-difference implementation.

Run them with::

    pytest test/test_loss.py
"""

import numpy as np

from bert_cpu import engine as cpu
from bert_cpu import nn
from bert_cpu.loss import cross_entropy
from test.test_engine import numeric_gradient


def _manual_cross_entropy(logits: np.ndarray, targets: np.ndarray, ignore_index=-100) -> float:
    """Reference cross-entropy computed in plain NumPy (class axis last)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    valid = targets != ignore_index
    safe = np.where(valid, targets, 0)
    picked = np.take_along_axis(logp, safe[..., None], axis=-1)[..., 0]
    n_valid = max(float(valid.sum()), 1.0)
    return float(-(picked * valid).sum() / n_valid)


# ====================================================================== #
# Value correctness
# ====================================================================== #
def test_cross_entropy_matches_manual():
    """The Tensor loss equals a from-scratch NumPy cross-entropy."""
    cpu.set_seed(0)
    logits = cpu.Tensor(np.random.randn(4, 5))          # (batch, num_classes)
    targets = np.array([0, 2, 4, 1])

    loss = cross_entropy(logits, targets)
    assert loss.shape == ()                              # a scalar to differentiate
    assert np.isclose(loss.data, _manual_cross_entropy(logits.data, targets))


def test_uniform_logits_give_log_C():
    """Equal logits -> uniform softmax -> loss is exactly log(num_classes)."""
    C = 7
    logits = cpu.Tensor(np.zeros((3, C)))
    targets = np.array([0, 3, 6])
    loss = cross_entropy(logits, targets)
    assert np.isclose(loss.data, np.log(C))


def test_confident_correct_is_near_zero():
    """A large logit on the correct class drives the loss toward 0."""
    logits = cpu.Tensor(np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]))
    targets = np.array([0, 2])
    loss = cross_entropy(logits, targets)
    assert loss.data < 1e-3


# ====================================================================== #
# Gradient — the loss stays on one graph back to the logits
# ====================================================================== #
def test_cross_entropy_gradient_is_softmax_minus_onehot():
    """Analytic ``dL/dlogits`` equals ``(softmax - one_hot) / N``."""
    cpu.set_seed(1)
    logits = cpu.Tensor(np.random.randn(4, 5))
    targets = np.array([1, 0, 3, 2])

    logits.zero_grad()
    cross_entropy(logits, targets).backward()

    # Reference gradient of mean cross-entropy w.r.t. the logits.
    sm = np.exp(logits.data - logits.data.max(axis=-1, keepdims=True))
    sm /= sm.sum(axis=-1, keepdims=True)
    one_hot = np.eye(5)[targets]
    expected = (sm - one_hot) / 4.0                      # N = 4 valid rows
    assert np.allclose(logits.grad, expected, atol=1e-6)


def test_cross_entropy_gradcheck():
    """Finite-difference gradient check of the loss w.r.t. the logits."""
    cpu.set_seed(2)
    logits = cpu.Tensor(np.random.randn(3, 6))
    targets = np.array([0, 5, 2])

    logits.zero_grad()
    cross_entropy(logits, targets).backward()

    numeric = numeric_gradient(lambda: float(cross_entropy(logits, targets).data), logits.data)
    assert np.allclose(logits.grad, numeric, atol=1e-5)


# ====================================================================== #
# ignore_index
# ====================================================================== #
def test_ignore_index_excludes_positions():
    """Ignored positions drop out of both the value and the gradient."""
    cpu.set_seed(3)
    logits = cpu.Tensor(np.random.randn(4, 5))
    targets = np.array([1, -100, 3, -100])               # rows 1 and 3 ignored

    logits.zero_grad()
    loss = cross_entropy(logits, targets, ignore_index=-100)
    loss.backward()

    # Value equals cross-entropy over only the two valid rows.
    keep = np.array([0, 2])
    sub = cpu.Tensor(logits.data[keep])
    expected = cross_entropy(sub, targets[keep])
    assert np.isclose(loss.data, expected.data)

    # Ignored rows receive exactly zero gradient.
    assert np.allclose(logits.grad[[1, 3]], 0.0)
    assert np.any(logits.grad[[0, 2]] != 0.0)


# ====================================================================== #
# End to end: the loss connects a parameter to backward() in one graph
# ====================================================================== #
def test_loss_backward_reaches_layer_weight():
    """A Linear -> cross_entropy -> backward fills the layer's weight gradient."""
    cpu.set_seed(4)
    layer = nn.Linear(3, 4)                               # column-oriented: (in, batch)
    x = cpu.Tensor(np.random.randn(3, 2))
    logits = layer(x).T                                   # -> (batch, num_classes)
    targets = np.array([0, 3])

    layer.zero_grad()
    cross_entropy(logits, targets).backward()

    # The single backward() reached the parameter through the whole graph.
    assert np.any(layer.weight.grad != 0.0)
