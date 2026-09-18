"""Software tests for the parameter optimisers.

An optimiser adds no autograd rule of its own — it only reads ``p.grad`` and updates
``p.data`` in place. These tests check that mechanical update on two levels:

- **exact single-step algebra** (SGD's update, momentum accumulation, and AdamW's
  decoupled shrink each match a hand computation), and
- **it actually optimises**: a short training loop over a closed-form convex
  objective ``f(p) = sum((p - target) ** 2)`` drives ``p`` to ``target``.

A tiny optimisation loop is fair game *here* (unlike in ``exercises/``, which
never train): the objective is closed-form and its only purpose is to prove the
step rule descends.

Run them with::

    pytest test/test_optim.py
"""

import numpy as np

from bert_cpu import engine as cpu
from bert_cpu import optim


# ====================================================================== #
# SGD — exact single-step algebra
# ====================================================================== #
def test_sgd_single_step_matches_manual():
    """One plain-SGD step is exactly ``p <- p - lr * grad``."""
    cpu.set_seed(0)
    p = cpu.Tensor(np.random.randn(3, 4))
    p0 = p.data.copy()
    p.grad = np.random.randn(3, 4)                 # pretend backward filled it

    lr = 0.1
    optim.SGD([p], lr=lr).step()

    assert np.allclose(p.data, p0 - lr * p.grad)


def test_sgd_momentum_accumulates():
    """With momentum, a constant gradient makes the second step larger.

    Velocity builds up as ``v <- momentum*v + g``, so on a fixed gradient the
    step sizes grow ``lr*g``, ``lr*(1+momentum)*g``, ... — the second move is
    strictly bigger than the first.
    """
    p = cpu.Tensor(np.ones((2, 2)))
    g = np.full((2, 2), 0.5)
    lr, momentum = 0.1, 0.9
    opt = optim.SGD([p], lr=lr, momentum=momentum)

    before1 = p.data.copy()
    p.grad = g.copy()
    opt.step()
    move1 = np.abs(p.data - before1)

    before2 = p.data.copy()
    p.grad = g.copy()
    opt.step()
    move2 = np.abs(p.data - before2)

    assert np.all(move2 > move1)
    # First step is plain lr*g; second is lr*(1 + momentum)*g.
    assert np.allclose(move1, lr * np.abs(g))
    assert np.allclose(move2, lr * (1.0 + momentum) * np.abs(g))


def test_sgd_minimises_quadratic():
    """SGD drives ``p`` to the minimiser of ``sum((p - target)**2)``."""
    cpu.set_seed(1)
    target = np.random.randn(5, 1)
    p = cpu.Tensor(np.zeros((5, 1)))
    opt = optim.SGD([p], lr=0.1, momentum=0.9)

    for _ in range(300):
        opt.zero_grad()
        loss = ((p - cpu.Tensor(target)) ** 2).sum()
        loss.backward()
        opt.step()

    assert np.allclose(p.data, target, atol=1e-4)


# ====================================================================== #
# Adam
# ====================================================================== #
def test_adam_minimises_quadratic():
    """Adam converges on the same convex objective."""
    cpu.set_seed(2)
    target = np.random.randn(5, 1)
    p = cpu.Tensor(np.zeros((5, 1)))
    opt = optim.Adam([p], lr=0.1)

    for _ in range(500):
        opt.zero_grad()
        loss = ((p - cpu.Tensor(target)) ** 2).sum()
        loss.backward()
        opt.step()

    assert np.allclose(p.data, target, atol=1e-3)


def test_adam_first_step_is_signed_lr():
    """Adam's very first step is ``-lr * sign(g)`` (up to eps).

    With ``m`` and ``v`` both starting at zero, bias correction makes the first
    update ``lr * m_hat / sqrt(v_hat) = lr * g / |g|`` for every coordinate —
    a hallmark of Adam that also confirms bias correction is applied.
    """
    p = cpu.Tensor(np.zeros((4,)))
    g = np.array([2.0, -3.0, 0.5, -10.0])
    p.grad = g.copy()
    lr = 0.01
    optim.Adam([p], lr=lr, eps=0.0).step()

    assert np.allclose(p.data, -lr * np.sign(g))


def test_adam_weight_decay_shrinks():
    """With zero gradient, decoupled weight decay alone shrinks the weight."""
    p = cpu.Tensor(np.array([1.0, -2.0, 3.0]))
    p0 = p.data.copy()
    p.grad = np.zeros(3)                            # no gradient signal at all
    lr, wd = 0.1, 0.1
    optim.Adam([p], lr=lr, weight_decay=wd).step()

    # Zero grad -> the adaptive term is 0; only the decoupled shrink acts.
    assert np.allclose(p.data, p0 - lr * wd * p0)
    assert np.all(np.abs(p.data) < np.abs(p0))


# ====================================================================== #
# Shared bookkeeping
# ====================================================================== #
def test_zero_grad_resets():
    """``optimizer.zero_grad`` zeros the gradient of every parameter."""
    a = cpu.Tensor(np.random.randn(2, 3))
    b = cpu.Tensor(np.random.randn(3, 2))
    (a @ b).sum().backward()
    assert np.any(a.grad != 0.0) and np.any(b.grad != 0.0)

    optim.SGD([a, b], lr=0.1).zero_grad()
    assert np.all(a.grad == 0.0) and np.all(b.grad == 0.0)
