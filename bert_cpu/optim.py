"""Parameter optimisers.

An optimiser is the last piece of the training loop and, mechanically, the
smallest. The engine's ``loss.backward()`` does all the hard work: it fills every
parameter's ``.grad`` with ``dloss/dp``. All an optimiser then does is walk that
list of parameters and nudge each one a little way *downhill* — in the direction
that (locally) makes the loss smaller. The whole training loop is::

    optimizer.zero_grad()   # clear last step's gradients
    loss = forward(...)     # build the graph, get a scalar
    loss.backward()         # fill every p.grad
    optimizer.step()        # p.data <- p.data - (something) * p.grad

Everything the optimisers need already lives on ``Tensor``: ``p.data`` (the
NumPy array of values), ``p.grad`` (the NumPy array of gradients, initialised to
zeros — never ``None``), and ``p.zero_grad()``. The updates below mutate
``p.data`` **in place with plain NumPy**, deliberately *not* through ``Tensor``
operations: an optimiser step is not part of any loss, so it must not be recorded
on the computational graph.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from bert_cpu.engine import Tensor


class Optimizer:
    """Base optimiser interface.

    For a visual and intuitive explanation of gradient descent, which provides
    the mathematical basis for many optimization algorithms used to train neural
    networks, see Grant Sanderson's video: 
    https://youtu.be/IHZwWFHWa-w?si=xRMYNni40hGop9p

    This class holds the list of parameters to update and provides the machinery shared by
    every optimiser — materialising the parameter list and zeroing gradients.
    The actual update rule lives in ``step``, which each subclass overrides.
    """

    def __init__(self, parameters: Iterable[Tensor]) -> None:
        # Materialise once: callers often pass ``model.parameters()`` which may
        # be a generator, and we iterate the list on every ``step``/``zero_grad``.
        self.parameters: List[Tensor] = list(parameters)

    def step(self) -> None:
        """Apply one update to every parameter from its current gradient."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter.

        Call this at the top of each iteration: the engine *accumulates* into
        ``p.grad`` (``self.grad += ...`` in every ``_backward``), so without a
        reset the next ``backward()`` would add on top of the previous step's
        gradients. Offered here so training code can call
        ``optimizer.zero_grad()`` instead of ``model.zero_grad()`` (they do the
        same thing), matching the PyTorch idiom.
        """
        for p in self.parameters:
            p.zero_grad()


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum.

        For visual and intuitive introductions to the underlying ideas, see
        Grant Sanderson's video on gradient descent: https://youtu.be/IHZwWFHWa-w?si=xRMYNni40hGop9pk

        For a focused explanation of stochastic gradient descent, see
        Bachir El Khadir's video: https://youtu.be/VbYTp0CIJkY?si=03lkskw00rXR6z2Y

    Plain SGD takes a fixed-size step straight down the gradient:

        p <- p - lr * g            (g = p.grad)

    **Momentum** replaces the raw gradient with a running, exponentially-weighted
    average of the recent gradients — the *velocity* ``v``::

        v <- momentum * v + g
        p <- p - lr * v

    You may think of ``v`` as the velocity of a ball rolling down the loss surface:
    directions in which the gradient is *consistent* accumulate speed, while
    directions that keep flipping sign cancel out. This damps the zig-zagging
    that plain SGD suffers in narrow ravines and accelerates progress along the
    consistent descent direction. ``momentum = 0`` recovers vanilla SGD.

    Parameters
    ----------
    parameters : Iterable[Tensor]
        The parameters to update (e.g. ``model.parameters()``).
    lr : float
        Learning rate — the step size along the (velocity-smoothed) gradient.
    momentum : float
        The velocity retention factor in ``[0, 1)``. ``0`` disables momentum.
    """

    def __init__(
        self, parameters: Iterable[Tensor], lr: float = 1e-3, momentum: float = 0.0
    ) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        # Per-parameter velocity buffers, keyed by id(param) and created lazily.
        self._velocity: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        # Update each parameter tensor independently. This method does not build
        # or traverse the computational graph. It assumes that loss.backward() has
        # already been called and that p.grad contains the gradient of the loss with
        # respect to the parameter p.
        for p in self.parameters:
            g = p.grad                       # already computed by loss.backward()
            if self.momentum == 0.0:
                p.data -= self.lr * g        # in-place, off-graph update
                continue

            # Lazily start the velocity buffer at zero (same shape/dtype as p).
            v = self._velocity.get(id(p))
            if v is None:
                v = np.zeros_like(p.data)
            v = self.momentum * v + g
            self._velocity[id(p)] = v
            p.data -= self.lr * v


class Adam(Optimizer):
    """Adam optimizer with per-coordinate adaptive step sizes.

    Adam keeps two running averages for each parameter tensor:

    - the first moment ``m``: an exponentially weighted moving average (EWMA)
      of the gradient ``g``. It acts like momentum and provides a smoothed
      estimate of the descent direction.
    - the second moment ``v``: an exponentially weighted moving average (EWMA)
      of the squared gradient ``g**2``. It tracks the recent scale or magnitude
      of the gradient for each coordinate.


    For visual and intuitive introductions to the underlying ideas, see
    Grant Sanderson's video on gradient descent: https://youtu.be/IHZwWFHWa-w?si=xRMYNni40hGop9pk

    For an intuitive explanation of Adam, see Zachary Huang's video:
    https://youtu.be/IWvTU6swl_E?si=WrVDBEL2M7LM3STo

    Adam combines the momentum idea from SGD with adaptive learning rates in
    the spirit of AdaGrad/RMSProp. Instead of applying the same effective step
    size to every coordinate, Adam divides the smoothed gradient by the square
    root of the smoothed squared gradient. Coordinates with consistently large
    gradients therefore take proportionally smaller steps, while coordinates
    with smaller gradients can take relatively larger steps:

        m <- b1 * m + (1 - b1) * g
        v <- b2 * v + (1 - b2) * g**2

    Here, ``g`` is the gradient of the output with respect to parameter ``p``.
    The coefficients ``b1`` and ``b2`` control how much past gradient
    information is retained in the first- and second-moment estimates.

    Since ``m`` and ``v`` are initialized at zero, their early values are biased
    toward zero. Adam corrects this startup bias by dividing each estimate by
    ``1 - b**t``. This correction is largest during the first steps and becomes
    negligible as the step count ``t`` increases::

        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)

        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    The term ``eps`` is added for numerical stability. In short, ``m_hat`` is a
    bias-corrected, forgetful average of the descent direction, while ``v_hat``
    is a bias-corrected, forgetful average of the gradient scale for each
    coordinate.
    

    Parameters
    ----------
    parameters : Iterable[Tensor]
        The parameters to update.
    lr : float
        Base learning rate.
    betas : tuple
        ``(b1, b2)`` — the EMA retention factors for the first and second
        moments. The defaults ``(0.9, 0.999)`` are near-universal.
    eps : float
        Numerical-stability term added to the denominator.
    weight_decay : float
        Decoupled (AdamW) weight-decay coefficient.
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._t = 0                              # global step count (for bias correction)
        self._m: Dict[int, np.ndarray] = {}      # first-moment buffers, keyed by id(param)
        self._v: Dict[int, np.ndarray] = {}      # second-moment buffers, keyed by id(param)

    def step(self) -> None:
        self._t += 1
        b1, b2 = self.beta1, self.beta2
        # Bias-correction denominators depend only on the step count, so compute
        # them once per step rather than per parameter.
        bias_correction1 = 1.0 - b1 ** self._t
        bias_correction2 = 1.0 - b2 ** self._t

        # Independent per-parameter update from the precomputed gradient; state
        # buffers follow the parameter via id(p), not the (ephemeral) graph.
        for p in self.parameters:
            g = p.grad                       # already computed by loss.backward()

            # Lazily start both moment buffers at zero (same shape/dtype as p).
            m = self._m.get(id(p))
            if m is None:
                m = np.zeros_like(p.data)
            v = self._v.get(id(p))
            if v is None:
                v = np.zeros_like(p.data)

            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)
            self._m[id(p)] = m
            self._v[id(p)] = v

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Decoupled weight decay (AdamW): a separate shrink toward zero.
            if self.weight_decay != 0.0:
                p.data -= self.lr * self.weight_decay * p.data
