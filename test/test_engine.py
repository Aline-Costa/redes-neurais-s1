"""Software tests for the autograd engine.

Ordinary correctness checks (``test_*`` functions). The most valuable one is a
numerical gradient check (compare analytic gradients from ``backward`` against
finite differences), since broadcasting in the backward pass is the easiest
thing to get subtly wrong. The shared ``numeric_gradient`` helper used by those
checks lives here and is reused by ``test_nn.py``.

Run them with::

    pytest test/test_engine.py

The didactic *visualisation* of the gradient graph now lives separately, in
``learn/viz01_engine.py`` (run it with ``python -m learn.viz01_engine``).
"""

import numpy as np

from bert_cpu import engine as cpu


# ====================================================================== #
# Helpers
# ====================================================================== #
def numeric_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Estimate the gradient of a scalar function ``f()`` w.r.t. ``x`` numerically.

    Why this exists
    ---------------
    This is the independent *reference* against which we check the gradients
    computed by our autograd engine, i.e., by ``Tensor.backward()``. In this
    library, ``Tensor.backward()`` traverses the computational graph in reverse
    topological order and calls each node's local ``_backward`` closure to apply
    the chain rule.

    The trick of a *gradient check* is to compute the same gradient a second way,
    using a method that knows nothing about the engine's internals, and compare.
    If the numerical and analytic gradients agree within tolerance, we gain
    confidence that ``Tensor.backward()`` and the local ``_backward`` rules are
    implemented correctly. If they disagree, then some part of the backward pass
    is likely wrong: either a local ``_backward`` rule, gradient accumulation, or
    the chain-rule traversal. (See ``test_gradcheck_mlp`` and
    ``test_softmax_sums_to_one``.)

    What exactly is ``f``?
    ----------------------
    ``f`` is the function we are differentiating: it re-runs the forward pass
    and returns the single scalar output as a plain Python float. Crucially it
    takes **no arguments** -- it does not receive ``x``. Instead it reads ``x``
    *indirectly*, by closing over the very same array object that this routine
    mutates. So ``f`` and ``x`` are wired to the same memory: when we poke
    ``x[i]``, the next call to ``f()`` recomputes the output using that poked
    value. That shared-array link is the whole mechanism -- it is how perturbing
    one number changes the output we measure.

    The maths
    ---------
    The gradient is the vector of partial derivatives, and each partial is just
    a derivative -- the limit of a difference quotient from first-year calculus:

        df/dx_i = lim_{h->0} ( f(x + h*e_i) - f(x) ) / h

    where ``e_i`` is the unit step along coordinate ``i``. We cannot take the
    limit numerically, so we approximate it with a small finite ``h = eps``.
    Instead of the one-sided quotient above we use the **central difference**,
    which perturbs ``x_i`` both up and down::

        df/dx_i ≈ ( f(x + eps*e_i) - f(x - eps*e_i) ) / (2 * eps)

    The central form is used because its error shrinks as O(eps^2) (the first
    error term cancels by symmetry), whereas the one-sided form is only O(eps),
    so for the same ``eps`` the central estimate is far more accurate.

    How it is computed
    ------------------
    There is no analytic shortcut here: we literally evaluate ``f`` twice for
    every single element of ``x`` (perturbed up, then down), filling the
    gradient entry by entry. That is why this is O(size of ``x``) forward passes
    and only ever used on tiny tensors inside tests -- never to actually train.



    Concretely, a caller writes (see ``test_gradcheck_mlp``)::

        x = cpu.Tensor(np.random.randn(4, 3))          # a leaf in the graph
        def forward():
            return ((x @ W + b).gelu() ** 2).mean()  # rebuilds graph, scalar out
        numeric_gradient(lambda: float(forward().data), x.data)

    Here ``f = lambda: float(forward().data)`` rebuilds the expression and
    returns the loss, and the ``x`` we differentiate is ``x.data`` -- the *same*
    NumPy buffer the Tensor ``x`` holds. Because ``forward`` reads ``x`` (and
    thus ``x.data``) every time it runs, editing ``x.data[i]`` in place is seen
    by the next ``forward()``. (Pass a *copy* of the array and the link breaks:
    ``f`` would keep reading the original and every partial would come out 0.)

    Parameters
    ----------
    f : Callable[[], float]
        Zero-argument closure that re-evaluates the forward pass over the
        *current* contents of ``x`` and returns the scalar output as a float.
    x : np.ndarray
        The exact array object that ``f`` reads (e.g. a Tensor's ``.data``), not
        a copy. It is perturbed element by element and always restored to its
        original value before returning.
    eps : float
        The finite step ``h``. A trade-off: too large and the difference
        quotient is a poor approximation of the limit (truncation error); too
        small and floating-point round-off dominates. ``1e-6`` is a good middle
        ground for ``float64``.

    Returns
    -------
    np.ndarray
        An array of the same shape as ``x`` holding the estimated gradient.
        Compared against the engine's ``.grad`` with a tolerance (e.g.
        ``atol=1e-5``), since finite differences are never bit-exact.
    """
    grad = np.zeros_like(x)
    # Iterate over every element of x, tracking its multi-dimensional index.
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        original = x[i]

        x[i] = original + eps        # perturb coordinate i up   -> f(x + eps*e_i)
        plus = f()
        x[i] = original - eps        # perturb coordinate i down -> f(x - eps*e_i)
        minus = f()
        x[i] = original              # restore so the next element starts clean

        # Central difference: the slope of f along coordinate i at x.
        grad[i] = (plus - minus) / (2.0 * eps)
        it.iternext()
    return grad


# ====================================================================== #
# Software tests
# ====================================================================== #
def test_add_broadcast_backward():
    """Adding a (d,) bias to a (n, d) tensor should sum grad over the batch."""
    n, d = 4, 3
    x = cpu.Tensor(np.random.randn(n, d))
    b = cpu.Tensor(np.random.randn(d))

    (x + b).sum().backward()

    # Every element of x feeds the sum once -> all-ones gradient.
    assert np.allclose(x.grad, np.ones((n, d)))
    # The bias is broadcast across all n rows, so its grad sums over the batch.
    assert np.allclose(b.grad, np.full(d, float(n)))


def test_matmul_backward():
    """matmul backward should match grad @ b.T and a.T @ grad."""
    a = cpu.Tensor(np.random.randn(4, 3))
    b = cpu.Tensor(np.random.randn(3, 5))

    out = a @ b
    out.backward()  # seeds out.grad with ones of shape (4, 5)

    upstream = np.ones((4, 5))
    assert np.allclose(a.grad, upstream @ b.data.T)
    assert np.allclose(b.grad, a.data.T @ upstream)


def test_softmax_sums_to_one():
    """Softmax output is a valid probability distribution along its axis."""
    logits = cpu.Tensor(np.random.randn(6, 10))
    probs = logits.softmax(axis=-1)

    assert np.allclose(probs.data.sum(axis=-1), 1.0)
    assert np.all(probs.data >= 0.0)

    # And its Jacobian is correct: gradient check against finite differences.
    # A fixed weight matrix keeps the scalar objective deterministic so that
    # the analytic and numeric passes measure the same function.
    coeff = np.random.randn(6, 10)

    def f():
        x = logits.data - logits.data.max(axis=-1, keepdims=True)
        e = np.exp(x)
        sm = e / e.sum(axis=-1, keepdims=True)
        return float((sm * coeff).sum())

    logits.zero_grad()
    (logits.softmax(axis=-1) * cpu.Tensor(coeff)).sum().backward()
    analytic = logits.grad.copy()
    numeric = numeric_gradient(f, logits.data)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_cat_routes_gradient_to_slices():
    """``cat`` joins tensors and sends each its own slice of the upstream grad.

    Concatenation has no local derivative to multiply by — it only places values
    into contiguous slices — so the backward pass simply cuts ``out.grad`` back
    into those slices. Here a constant ``ones`` row (``requires_grad=False``) is
    prepended to ``x`` (the "bias trick"), and the gradient must flow only into
    ``x``'s slice.
    """
    x = cpu.Tensor(np.random.randn(3, 4))
    ones = cpu.Tensor(np.ones((1, 4)), requires_grad=False)

    out = cpu.cat([ones, x], axis=0)
    assert out.shape == (4, 4)

    coeff = np.random.randn(4, 4)
    (out * cpu.Tensor(coeff)).sum().backward()

    # x receives exactly the coefficients lined up with its slice (rows 1..).
    assert np.allclose(x.grad, coeff[1:])


def test_gradcheck_mlp():
    """Finite-difference gradient check of a small composite expression.

    Computes ``loss = mean(gelu(x @ W + b) ** 2)`` and verifies the analytic
    gradients w.r.t. every input against central finite differences.
    """
    np.random.seed(0)
    x = cpu.Tensor(np.random.randn(4, 3))
    W = cpu.Tensor(np.random.randn(3, 5))
    b = cpu.Tensor(np.random.randn(5))

    def forward():
        return ((x @ W + b).gelu() ** 2).mean()

    for t in (x, W, b):
        t.zero_grad()
    forward().backward()

    for name, t in (("x", x), ("W", W), ("b", b)):
        numeric = numeric_gradient(lambda: float(forward().data), t.data)
        assert np.allclose(t.grad, numeric, atol=1e-5), f"gradient mismatch for {name}"
