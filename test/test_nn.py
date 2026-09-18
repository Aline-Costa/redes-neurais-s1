"""Software tests for the neural-network layers built on top of the engine.

Ordinary correctness checks. The most valuable one is again a **finite-difference
gradient check**: a ``Linear`` layer adds no new backward rule (it is the
engine's ``cat`` + ``@``), so what we are really verifying is that gradients flow
correctly into a layer's ``Parameter`` weight — including the folded-in bias row
— and back into the input.

``Linear`` follows the engine's *bias trick* and *column-vector* convention:
inputs are ``(in_features, batch)``, the bias is row 0 of the weight matrix, and
the forward pass augments the input with ``x_0 = 1`` and computes ``Wᵀ @ x``.

The numerical-gradient reference is reused from ``test_engine`` so there is a
single, well-documented implementation of the finite-difference check.

Run them with::

    pytest test/test_nn.py

The didactic *visualisation* (a layer as a matrix, and the chain rule over a
stack of activated layers) now lives separately, in ``learn/viz02_nn.py`` (run it
with ``python -m learn.viz02_nn``).
"""

import numpy as np

from bert_cpu import engine as cpu
from bert_cpu import nn
from test.test_engine import numeric_gradient


# ====================================================================== #
# Module / parameter bookkeeping
# ====================================================================== #
def test_linear_registers_single_weight_parameter():
    """With the bias folded in, a Linear exposes exactly one weight parameter."""
    layer = nn.Linear(3, 5)

    params = layer.parameters()
    assert len(params) == 1
    assert layer.weight in params
    assert all(isinstance(p, nn.Parameter) for p in params)

    names = dict(layer.named_parameters())
    assert set(names) == {"weight"}
    # The weight carries an extra leading row for the bias: (in + 1, out).
    assert names["weight"].shape == (4, 5)


def test_bias_row_starts_at_zero():
    """The folded-in bias row (row 0 of the weight) is initialised to zero."""
    layer = nn.Linear(3, 5)
    assert np.all(layer.weight.data[0] == 0.0)
    # ...while the real weight rows are not all zero.
    assert np.any(layer.weight.data[1:] != 0.0)


def test_linear_no_bias_weight_shape():
    """With ``bias=False`` the weight is a plain ``(in, out)`` map, no bias row."""
    layer = nn.Linear(4, 2, bias=False)
    assert layer.weight.shape == (4, 2)
    assert layer.parameters() == [layer.weight]


def test_nested_module_collects_all_parameters():
    """A module holding sub-modules exposes their parameters with dotted names."""

    class TwoLayer(nn.Module):
        def __init__(self):
            self.fc1 = nn.Linear(3, 4)
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc2(self.fc1(x).relu())

    model = TwoLayer()
    # One weight parameter per Linear (bias folded in) -> 2 total.
    assert len(model.parameters()) == 2
    names = set(dict(model.named_parameters()))
    assert names == {"fc1.weight", "fc2.weight"}


def test_zero_grad_resets_the_weight():
    """``Module.zero_grad`` zeros the gradient of every parameter."""
    layer = nn.Linear(3, 2)
    x = cpu.Tensor(np.random.randn(3, 6))      # (in_features, batch)
    layer(x).sum().backward()

    assert np.any(layer.weight.grad != 0.0)
    layer.zero_grad()
    assert np.all(layer.weight.grad == 0.0)


def test_train_eval_toggles_flag_recursively():
    """``train``/``eval`` flip the ``training`` flag on a module and its children."""

    class Wrapper(nn.Module):
        def __init__(self):
            self.fc = nn.Linear(2, 2)

        def forward(self, x):
            return self.fc(x)

    model = Wrapper()
    assert model.training and model.fc.training  # default is training mode

    model.eval()
    assert not model.training and not model.fc.training

    model.train()
    assert model.training and model.fc.training


# ====================================================================== #
# Forward correctness
# ====================================================================== #
def test_linear_forward_shape_and_value():
    """``Linear`` computes ``Wᵀ @ [1; x]`` with the expected output shape."""
    cpu.set_seed(0)
    d_in, d_out, n = 3, 5, 6
    layer = nn.Linear(d_in, d_out)
    x = cpu.Tensor(np.random.randn(d_in, n))   # column-oriented: (in, batch)

    y = layer(x)
    assert y.shape == (d_out, n)

    # Hand-compute the augmented forward pass using the layer's own weight.
    x_aug = np.concatenate([np.ones((1, n)), x.data], axis=0)
    expected = layer.weight.data.T @ x_aug
    assert np.allclose(y.data, expected)


def test_bias_trick_matches_explicit_affine():
    """``Wᵀ @ [1; x]`` equals ``W_weights·x + bias`` (the trick is just bookkeeping)."""
    cpu.set_seed(0)
    layer = nn.Linear(3, 5)
    # Give the bias row a non-zero value so the comparison is meaningful.
    layer.weight.data[0] = np.random.randn(5)
    x = cpu.Tensor(np.random.randn(3, 4))

    y = layer(x)
    bias = layer.weight.data[0][:, None]       # (out, 1), broadcast over batch
    weights = layer.weight.data[1:]            # (in, out)
    expected = weights.T @ x.data + bias
    assert np.allclose(y.data, expected)


def test_linear_no_bias_forward():
    """With ``bias=False`` the output is exactly ``Wᵀ @ x`` (no augmentation)."""
    cpu.set_seed(0)
    layer = nn.Linear(4, 2, bias=False)
    x = cpu.Tensor(np.random.randn(4, 3))
    assert np.allclose(layer(x).data, layer.weight.data.T @ x.data)


# ====================================================================== #
# Gradient check
# ====================================================================== #
def test_linear_gradcheck():
    """Finite-difference gradient check for the weight (incl. bias row) and input.

    Objective: ``loss = mean( (Wᵀ @ [1; x]) ** 2 )``. We verify the analytic
    gradients from ``backward`` against central finite differences for both the
    weight ``W`` and the input ``x`` — confirming gradients flow correctly
    through the bias-trick augmentation into every row of ``W``.
    """
    cpu.set_seed(0)
    layer = nn.Linear(3, 5)
    x = cpu.Tensor(np.random.randn(3, 4))
    W = layer.weight

    def forward():
        return (layer(x) ** 2).mean()

    for t in (x, W):
        t.zero_grad()
    forward().backward()

    for name, t in (("x", x), ("W", W)):
        numeric = numeric_gradient(lambda: float(forward().data), t.data)
        assert np.allclose(t.grad, numeric, atol=1e-5), f"gradient mismatch for {name}"


# ====================================================================== #
# Initialisation
# ====================================================================== #
def test_xavier_uniform_bounds_and_reproducibility():
    """Xavier init stays within ``[-a, a]`` and is reproducible under a seed."""
    in_f, out_f = 20, 30
    a = np.sqrt(6.0 / (in_f + out_f))

    cpu.set_seed(0)
    w1 = nn.xavier_uniform(in_f, out_f)
    assert w1.shape == (in_f, out_f)
    assert np.all(np.abs(w1.data) <= a)

    # Same seed -> identical draw (centralised RNG, a core project guarantee).
    cpu.set_seed(0)
    w2 = nn.xavier_uniform(in_f, out_f)
    assert np.allclose(w1.data, w2.data)


# ====================================================================== #
# Chain rule across stacked, differently-activated layers
# ====================================================================== #
def _act_derivative(name: str, z: np.ndarray) -> np.ndarray:
    """Return ``act'(z)`` via the engine itself.

    For an elementwise activation, ``act(z).sum().backward()`` leaves
    ``z.grad[i] = act'(z_i)`` — the same local slope the engine multiplies by in
    its backward pass. (The ``learn/viz02_nn.py`` walkthrough uses the same trick.)
    """
    t = cpu.Tensor(z.copy())
    getattr(t, name)().sum().backward()
    return t.grad


def test_three_layer_chain_rule():
    """The chain identity ``dL/dz = dL/da ⊙ act'(z)`` holds at every layer, and
    the stacked gradients match finite differences.

    This is the correctness backbone of the ``learn/viz02_nn.py`` walkthrough: it
    confirms that what the demo *narrates* the chain rule does is exactly what the
    engine *computes*, for all three differently-activated layers.
    """
    # A lively seed: all ReLU units fire, so gradients are non-trivially non-zero
    # and stay clear of the ReLU kink at z = 0.
    cpu.set_seed(5)
    x = cpu.Tensor(np.random.randn(2, 1) * 0.5)
    layer1, layer2, layer3 = nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 1)

    z1 = layer1(x);   a1 = z1.relu()
    z2 = layer2(a1);  a2 = z2.tanh()
    z3 = layer3(a2);  a3 = z3.gelu()

    for p in (x, layer1.weight, layer2.weight, layer3.weight):
        p.zero_grad()
    a3.sum().backward()

    # Per-layer chain identity: the pre-activation grad is the upstream grad
    # times the activation's local slope.
    for z, a, act in ((z1, a1, "relu"), (z2, a2, "tanh"), (z3, a3, "gelu")):
        assert np.allclose(z.grad, a.grad * _act_derivative(act, z.data), atol=1e-6)

    # End-to-end gradient check of the whole stack against finite differences.
    def forward():
        return layer3(layer2(layer1(x).relu()).tanh()).gelu().sum()

    params = (("x", x), ("W1", layer1.weight), ("W2", layer2.weight), ("W3", layer3.weight))
    for name, t in params:
        numeric = numeric_gradient(lambda: float(forward().data), t.data)
        assert np.allclose(t.grad, numeric, atol=1e-5), f"gradient mismatch for {name}"


# ====================================================================== #
# Embedding — a gather, not a matmul
# ====================================================================== #
def test_embedding_registers_weight_parameter():
    """An Embedding exposes exactly one ``weight`` parameter of shape (V, D)."""
    emb = nn.Embedding(10, 4)

    params = emb.parameters()
    assert len(params) == 1
    assert emb.weight in params
    assert all(isinstance(p, nn.Parameter) for p in params)

    names = dict(emb.named_parameters())
    assert set(names) == {"weight"}
    assert names["weight"].shape == (10, 4)


def test_embedding_forward_shape_and_gather():
    """``emb(ids)`` gathers table rows: ids (batch, seq) -> (batch, seq, D)."""
    cpu.set_seed(0)
    V, D = 10, 4
    emb = nn.Embedding(V, D)
    ids = np.array([[1, 3, 3], [0, 9, 2]])          # (batch=2, seq=3)

    out = emb(ids)
    assert out.shape == (2, 3, D)
    # Every output vector is exactly the corresponding row of the table.
    for b in range(ids.shape[0]):
        for s in range(ids.shape[1]):
            assert np.allclose(out.data[b, s], emb.weight.data[ids[b, s]])


def test_embedding_accepts_tensor_ids():
    """Ids may arrive as a Tensor; its data is used for the gather."""
    cpu.set_seed(0)
    emb = nn.Embedding(6, 3)
    ids = np.array([2, 5, 0])
    from_array = emb(ids).data
    from_tensor = emb(cpu.Tensor(ids, requires_grad=False)).data
    assert np.allclose(from_array, from_tensor)


def test_embedding_padding_idx_is_zero():
    """With ``padding_idx`` set, that row starts as the zero vector."""
    emb = nn.Embedding(8, 5, padding_idx=0)
    assert np.all(emb.weight.data[0] == 0.0)
    # ...while the other rows are not all zero.
    assert np.any(emb.weight.data[1:] != 0.0)


def test_embedding_repeated_ids_accumulate_grad():
    """Repeated ids sum their gradients into one row; unused rows stay at zero.

    This is the correctness heart of ``Embedding``: it adds no backward rule, so
    what we verify is that the engine's ``__getitem__`` (``np.add.at``) routes and
    *accumulates* the upstream gradient per id. With ``loss = emb(ids).sum()`` each
    used cell contributes 1, so a row's gradient equals how many times its id
    appears.
    """
    cpu.set_seed(0)
    V, D = 5, 3
    emb = nn.Embedding(V, D)
    ids = np.array([[1, 1, 2]])                     # id 1 twice, id 2 once

    emb(ids).sum().backward()

    assert np.allclose(emb.weight.grad[1], 2.0)     # id 1 used twice
    assert np.allclose(emb.weight.grad[2], 1.0)     # id 2 used once
    # Ids 0, 3, 4 never appear -> no gradient.
    for unused in (0, 3, 4):
        assert np.allclose(emb.weight.grad[unused], 0.0)


def test_embedding_gradcheck():
    """Finite-difference gradient check of the table for a scalar objective.

    ``loss = mean( emb(ids) ** 2 )``. The analytic ``weight.grad`` from
    ``backward`` must match central finite differences — including the correct
    accumulation over the repeated id.
    """
    cpu.set_seed(0)
    emb = nn.Embedding(6, 4)
    ids = np.array([[0, 2, 2], [5, 1, 2]])          # id 2 repeats across the batch
    W = emb.weight

    def forward():
        return (emb(ids) ** 2).mean()

    W.zero_grad()
    forward().backward()

    numeric = numeric_gradient(lambda: float(forward().data), W.data)
    assert np.allclose(W.grad, numeric, atol=1e-5)


# ====================================================================== #
# normal_ initialisation
# ====================================================================== #
def test_normal_init_stats_and_reproducibility():
    """``normal_`` draws N(mean, std) of the right shape and is seed-reproducible."""
    shape = (400, 300)

    cpu.set_seed(0)
    w1 = nn.normal_(shape, mean=0.0, std=0.02)
    assert w1.shape == shape
    # Large sample -> empirical mean/std close to the requested Gaussian.
    assert abs(float(w1.data.mean())) < 1e-3
    assert abs(float(w1.data.std()) - 0.02) < 1e-3

    # Same seed -> identical draw (centralised RNG guarantee).
    cpu.set_seed(0)
    w2 = nn.normal_(shape, mean=0.0, std=0.02)
    assert np.allclose(w1.data, w2.data)
