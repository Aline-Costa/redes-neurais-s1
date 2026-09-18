"""Neural-network building blocks built on top of the ``Tensor`` engine.

Mirrors micrograd's ``nn.py`` (``Module`` base + composable layers) but the
primitives are now tensor-valued: ``Linear``, ``Embedding``, ``LayerNorm``,
``Dropout`` and container modules.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np

from bert_cpu import engine
from bert_cpu.engine import Tensor, cat


class Parameter(Tensor):
    """A ``Tensor`` that is registered as a learnable parameter of a module.

    ``Parameter`` is *behaviourally* identical to ``Tensor`` — it adds no new
    maths. Its only job is to act as a **marker**: when a ``Module`` walks its
    own attributes (see ``Module.named_parameters``), the ones that are
    ``Parameter`` instances are the things an optimizer should update. A plain
    ``Tensor`` stored on a module (e.g. a cached constant) is therefore *not*
    collected, while ``self.weight = Parameter(...)`` is. This is exactly how
    PyTorch tells learnable weights apart from incidental tensors.
    """


class Module:
    """Base class for all neural-network modules.

    A ``Module`` is just an object that (a) computes something in ``forward`` and
    (b) owns some learnable ``Parameter`` tensors. Unlike a purely abstract base,
    this class is *mostly concrete*: the only method a subclass must override is
    ``forward``. Everything else — collecting parameters, zeroing gradients,
    flipping the train/eval flag — is provided here and works by **introspection**
    over the subclass's own attributes.

    How parameter discovery works
    -----------------------------
    When you write a layer like::

        class Linear(Module):
            def __init__(self, n_in, n_out):
                self.weight = Parameter(...)   # learnable


    the parameters are stored as ordinary attributes. ``named_parameters`` simply
    reads ``vars(self)`` (the instance ``__dict__``) and yields every attribute
    that is a ``Parameter``. It also recurses into attributes that are themselves
    ``Module`` instances (or lists/tuples of them), so a model built from nested
    sub-modules exposes *all* of its parameters through a single
    ``model.parameters()`` call. No manual registration is needed; storing a
    ``Parameter`` (or a child ``Module``) is the registration.

    Subclasses implement ``forward``; calling the module (``module(x)``) invokes
    it via ``__call__``.
    """

    # Train/eval state. Layers like Dropout read this flag to decide their
    # behaviour. It defaults to training mode, matching PyTorch.
    training: bool = True

    def __call__(self, *args, **kwargs) -> Tensor:
        """Invoke ``forward`` (so ``module(x)`` works like ``module.forward(x)``)."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        """Compute the module's output. Must be overridden by subclasses."""
        raise NotImplementedError

    def _child_modules(self) -> Iterator["Module"]:
        """Yield the direct sub-modules stored as attributes of this module.

        Looks through ``vars(self)`` for attributes that are ``Module`` instances,
        plus any ``Module`` found inside an attribute that is a list/tuple (e.g.
        ``Sequential``'s held modules). Used to recurse for ``train``/``eval``.
        """
        for attr in vars(self).values():
            if isinstance(attr, Module):
                yield attr
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        yield item

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        """Yield ``(name, parameter)`` pairs for this module and its sub-modules.

        Walks the instance attributes once; ``Parameter`` attributes are yielded
        directly, child ``Module``s are recursed into (their names are prefixed,
        e.g. ``"encoder.weight"``), and lists/tuples are indexed
        (``"layers.0.weight"``). The dotted names mirror the attribute path, so a
        parameter is easy to locate in the model tree.
        """
        for name, attr in vars(self).items():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(attr, Parameter):
                yield full, attr
            elif isinstance(attr, Module):
                yield from attr.named_parameters(full)
            elif isinstance(attr, (list, tuple)):
                for i, item in enumerate(attr):
                    item_name = f"{full}.{i}"
                    if isinstance(item, Parameter):
                        yield item_name, item
                    elif isinstance(item, Module):
                        yield from item.named_parameters(item_name)

    def parameters(self) -> List[Parameter]:
        """Return the flat list of learnable parameters in this module (and children)."""
        return [p for _, p in self.named_parameters()]

    def zero_grad(self) -> None:
        """Zero the gradient of every parameter (call before each backward pass)."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        """Set training mode on this module and all sub-modules (affects e.g. Dropout)."""
        self.training = mode
        for m in self._child_modules():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode (shorthand for ``train(False)``)."""
        return self.train(False)


class Linear(Module):
    """Linear layer ``y = Wᵀ @ x`` using the *bias trick* (no separate bias).

    Rather than keeping a separate bias vector, the bias is folded into the
    weight matrix as a leading row ``w_0``, and the input is augmented with a
    matching leading constant ``x_0 = 1``. This is the exact convention used by
    the engine's didactic demo (``y = tanh(wᵀ @ x)`` with ``x_0 = 1``).

    Inputs are **column-oriented**: features run down the first axis and samples
    across the second, so ``x`` has shape ``(in_features, batch)`` (a single
    sample is ``(in_features, 1)``). With the augmentation, the stored weight has
    shape ``(in_features + 1, out_features)`` and::

        x_aug = [ 1 ; x ]                 # prepend a constant row -> (in+1, batch)
        y     = Wᵀ @ x_aug                # (out, in+1) @ (in+1, batch) -> (out, batch)

    Writing out the first row makes the bias explicit::

        y = W[1:]ᵀ @ x  +  W[0]           # the w_0 row plays the role of the bias

    Because ``x_0 = 1`` is constant, ``Wᵀ @ x_aug`` is mathematically identical
    to "weights·x + bias", but everything lives in one matrix. The augmentation
    uses the engine's differentiable ``cat``, so gradients flow into ``W`` (every
    row, including the bias row ``w_0``) and back into the real features of ``x``.
    ``Linear`` therefore adds no new backward rule of its own.

    Initialisation
    --------------
    The weight rows are Glorot/Xavier uniform (see ``xavier_uniform``); the bias
    row ``w_0`` starts at zero, the usual convention. ``weight`` is a
    ``Parameter``, so it is collected automatically by ``parameters()`` /
    ``zero_grad()``.

    Parameters
    ----------
    in_features : int
        Number of input features (the real ones, before augmentation).
    out_features : int
        Number of output features.
    bias : bool
        If ``True`` (default) the weight carries a leading bias row and the input
        is augmented with ``x_0 = 1``. If ``False`` the layer is the pure linear
        map ``Wᵀ @ x`` with ``W`` of shape ``(in_features, out_features)``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Xavier-initialised weight rows, shape (in_features, out_features).
        W = xavier_uniform(in_features, out_features).data
        if bias:
            # Fold the bias in as a leading row w_0, starting at zero. The
            # matching input augmentation x_0 = 1 happens in forward().
            bias_row = np.zeros((1, out_features), dtype=W.dtype)
            W = np.concatenate([bias_row, W], axis=0)        # (in + 1, out)
        self.weight: Parameter = Parameter(W)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias:
            # Bias trick: prepend a constant row x_0 = 1 so the bias row of W is
            # applied like any other weight inside Wᵀ @ x.
            ones_shape = (1,) if x.ndim == 1 else (1, x.shape[1])
            ones = Tensor(np.ones(ones_shape, dtype=x.data.dtype), requires_grad=False)
            x = cat([ones, x], axis=0)
        return self.weight.T @ x


class Embedding(Module):
    """Lookup table mapping integer ids to dense, learnable vectors.

    An ``Embedding`` is a matrix ``weight`` of shape ``(num_embeddings,
    embedding_dim)`` — one row per vocabulary id. The forward pass is a pure
    **gather**: given a batch of ids, it returns the corresponding rows. There is
    no matrix multiply and, notably, **no backward rule of its own** — exactly the
    way ``Linear`` is "just ``cat`` + ``@``", an ``Embedding`` is "just an index".

    Why indexing already gives the right gradient
    ---------------------------------------------
    The lookup is a single call into the engine's ``Tensor.__getitem__``. On the
    backward pass that op runs ``np.add.at(weight.grad, idx, out.grad)``, which
    *accumulates* rather than overwrites. So when the same id appears at many
    positions in a batch (a common token like ``[PAD]`` or "the"), every
    occurrence's upstream gradient is summed back into that one shared row — which
    is precisely ``d loss / d weight[id]``. Ids that never appear receive no
    contribution and keep a zero gradient, so the optimizer only moves rows the
    batch actually used.

    Layout
    ------
    Unlike ``Linear`` (column-oriented ``(features, batch)``), ``Embedding``
    follows the standard PyTorch/BERT layout so it feeds the attention stack
    directly: ``weight`` is ``(V, D)`` and, for ids of any shape, the output is
    ``(*ids.shape, D)`` — e.g. ids ``(batch, seq)`` give ``(batch, seq, D)``.

    Parameters
    ----------
    num_embeddings : int
        Vocabulary size ``V`` (number of rows in the table).
    embedding_dim : int
        Size ``D`` of each embedding vector.
    padding_idx : int, optional
        If given, that row is initialised to the zero vector (the usual
        convention for the ``[PAD]`` token). It is still a normal learnable row —
        this only sets its starting value.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # BERT-style small normal init; the table is a single learnable Parameter.
        W = normal_((num_embeddings, embedding_dim)).data
        if padding_idx is not None:
            W[padding_idx] = 0.0            # pad token starts as the zero vector
        self.weight: Parameter = Parameter(W)

    def forward(self, idx) -> Tensor:
        """Gather rows of the table for the given integer ids.

        ``idx`` may be a Python list, a NumPy integer array, or a ``Tensor`` of
        ids (its ``data`` is used); it is coerced to an integer array. The single
        indexing op below builds the whole forward *and* backward — see the class
        docstring for why ``np.add.at`` makes the gathered gradient correct.
        """
        if isinstance(idx, Tensor):
            idx = idx.data
        idx = np.asarray(idx, dtype=int)
        return self.weight[idx]            # (*idx.shape, embedding_dim)


class LayerNorm(Module):
    """Layer normalisation over the last dimension with learnable affine."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Dropout(Module):
    """Inverted dropout; a no-op when in eval mode."""

    def __init__(self, p: float = 0.1) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Sequential(Module):
    """Container that chains modules end to end."""

    def __init__(self, *modules: Module) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# Weight initialisation
# ---------------------------------------------------------------------- #
def xavier_uniform(in_features: int, out_features: int) -> Tensor:
    """Glorot/Xavier uniform initialisation for a weight matrix.

    Draws every weight uniformly from ``[-a, a]`` where::

        a = sqrt( 6 / (in_features + out_features) )

    The idea (Glorot & Bengio, 2010) is to choose the spread so that the variance
    of a layer's *outputs* matches the variance of its *inputs*, and likewise for
    the gradients flowing backward. For a uniform distribution on ``[-a, a]`` the
    variance is ``a^2 / 3``; plugging in the ``a`` above gives a per-weight
    variance of ``2 / (in + out)``, the Xavier target that balances the
    forward (``fan_in``) and backward (``fan_out``) signal. Keeping that variance
    near 1 across layers is what stops activations from exploding or vanishing as
    a network gets deeper.

    Randomness flows through the engine's global RNG, so a prior
    ``set_seed`` makes the initialisation reproducible. The dtype follows the
    engine's current ``default_dtype`` so precision settings are respected.

    Returns
    -------
    Tensor
        A ``(in_features, out_features)`` weight tensor (``requires_grad=True``).
    """
    a = np.sqrt(6.0 / (in_features + out_features))
    data = np.random.uniform(-a, a, size=(in_features, out_features))
    return Tensor(data.astype(engine.default_dtype))


def normal_(shape, mean: float = 0.0, std: float = 0.02) -> Tensor:
    """BERT-style normal initialisation for a weight tensor.

    Draws every element independently from a Gaussian ``N(mean, std**2)``. BERT
    initialises its embedding tables and linear layers this way, with the small
    default ``std = 0.02`` (Devlin et al., 2019): keeping the initial weights tiny
    stops the summed token/position/segment embeddings — and the deep residual
    stack above them — from starting with an exploding variance.

    Like ``xavier_uniform``, the draw flows through NumPy's global RNG, so a prior
    ``set_seed`` makes it reproducible, and the dtype follows the engine's current
    ``default_dtype``.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the tensor to create (e.g. ``(num_embeddings, embedding_dim)``).
    mean, std : float
        Mean and standard deviation of the Gaussian (BERT defaults ``0.0`` /
        ``0.02``).

    Returns
    -------
    Tensor
        A weight tensor of the requested shape (``requires_grad=True``).
    """
    data = np.random.normal(mean, std, size=shape)
    return Tensor(data.astype(engine.default_dtype))
