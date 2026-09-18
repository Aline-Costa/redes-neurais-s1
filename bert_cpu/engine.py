"""Tensor-based reverse-mode autodiff engine.

This module implements a NumPy-backed successor to micrograd's [1] scalar
``Value`` class. A ``Tensor`` wraps an ``np.ndarray`` and records the operations
applied to it so that gradients can be computed by reverse-mode automatic
differentiation.

The core autograd machinery is identical in spirit to micrograd: the engine
builds a computational graph, sorts it topologically, and calls each node's
``_backward`` closure in reverse order. The main difference is that each node
now stores a whole NumPy array rather than a single scalar value.

In micrograd, arithmetic is performed through Python-level scalar
operations, and a vector or matrix computation would be represented as many
small scalar nodes. In this tensor-based engine, operations such as elementwise
multiplication, reductions, broadcasting, and matrix multiplication are
represented as single graph nodes, while the underlying numerical computation
is delegated to NumPy's optimized compiled backend.

This makes the graph more compact and brings the implementation closer to the
design of real deep learning frameworks, where graph nodes usually represent
tensor operations rather than individual scalar operations. 

References:
----------

[1] https://github.com/karpathy/micrograd
"""


from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np

ArrayLike = Union["Tensor", np.ndarray, float, int]
Axis = Optional[Union[int, Tuple[int, ...]]]

# The engine uses ``default_dtype`` to control the numerical precision used
# internally by new tensors. By default, this is usually set to ``np.float64``
# because it gives better numerical stability, especially for finite-difference
# gradient checks. Gradient checking is sensitive to very small perturbations,
# so lower precision types may introduce round-off errors that make a correct
# backward rule look wrong.
#
# The precision can be changed globally by modifying ``engine.default_dtype``:
#
#     import numpy as np
#     import bert_cpu.engine as engine
#
#     engine.default_dtype = np.float32
#
# Common NumPy floating-point options are: np.float16, np.float32, mp.float64
# In this educational implementation, the default favors clarity and numerical
# reliability over speed.

default_dtype: np.dtype = np.float64 

# Exotic data types such as float8 could also work if you supply a dtype object from
# a package like ``ml_dtypes``.


# --------------------------------------------------------------------------- #
# Floating-point-operation (FLOP) counter
# --------------------------------------------------------------------------- #
# A single global tally of the floating-point operations the engine executes, so
# a training loop can report the compute it actually performs (see
# ``exercises/q05_binary_classification.ipynb``). It is instrumented in two places:
# the *forward* cost of every op is added in ``Tensor.__init__`` (derived from the
# op label and the output size), and the *matmul backward* — the dominant cost of
# the backward pass — is added inside ``__matmul__``. Conventions: a matmul counts
# ``2 * output_elements * shared_dim`` (one multiply + one add per MAC); an
# elementwise op counts one FLOP per output element; pure data movement
# (reshape / transpose / cat / indexing) counts zero. So the tally is *exact* for
# the matmuls that dominate and approximate for the cheap ops. Wrap the work you
# want to measure between ``reset_flops()`` and ``flop_count()``.
_flop_count: int = 0


def reset_flops() -> None:
    """Zero the global FLOP counter."""
    global _flop_count
    _flop_count = 0


def flop_count() -> int:
    """Return the number of FLOPs counted since the last ``reset_flops()``."""
    return _flop_count


def _add_flops(n: int) -> None:
    global _flop_count
    _flop_count += int(n)


# Ops that only move or compare data — no floating-point arithmetic to tally.
_ZERO_FLOP_OPS = frozenset({"reshape", "transpose", "getitem", "cat", "max"})


def _forward_flops(op: str, out_data: np.ndarray, children) -> int:
    """FLOPs of one forward op, from its label, output size and (ordered) operands."""
    if not op:
        return 0                                   # a leaf tensor: nothing computed
    n = out_data.size
    if op == "@":                                  # 2 * out_elements * shared inner dim
        return 2 * n * children[0].data.shape[-1]
    if op in _ZERO_FLOP_OPS:
        return 0
    if op in ("sum", "mean", "var"):               # one read/add per input element
        return children[0].data.size if children else n
    if op == "softmax":                            # exp + sum + divide, roughly
        return 3 * n
    if op == "gelu":                               # several mul/add/tanh per element
        return 8 * n
    return n                                       # elementwise: +, *, **k, exp, log, ...


def set_seed(seed: int) -> None:
    """Seed the single, library-wide NumPy random number generator.

    All randomness in the project — parameter initialization, dataset
    shuffling, token masking, dropout, and negative sampling — flows through
    NumPy's *global* random state (the ``np.random.*`` functions, which is also
    what ``randn`` below uses). Seeding that one source therefore makes the
    whole experiment reproducible. Centralizing these sources of randomness
    makes experiments easier to reproduce and helps researchers compare model
    variants under more controlled conditions.

    As discussed by Pham et al. [2] and Chen et al. [3], the interaction between
    software-level randomness and hardware-level** nondeterminism, using regular
    Deep Learning frameworks, makes it difficult to determine whether an observed
    performance change was caused by a proposed architectural or algorithmic 
    modification or merely by uncontrolled experimental variation. Centralizing 
    the explicit sources of randomness in a single NumPy generator therefore helps
    researchers conduct more controlled comparisons during early prototyping. 
    
    **GPUs are widely used for deep learning training because their large number of 
    processing cores enables substantial parallelism. However, this parallel execution 
    can also introduce nondeterminism in floating-point computations. Floating-point 
    operations are affected by rounding and are not mathematically associative in 
    finite precision; therefore, different execution or reduction orders may lead to
    small numerical differences across runs [2, 4].


    Parameters
    ----------
    seed:
        Integer used to initialize NumPy's pseudo-random number generator.
        Reusing the same seed reproduces the same random sequence, provided
        that random operations are executed in the same order.

    How to use it
    -------------
    Call it **once, at the very start**, before creating any tensors or data;

        import bert_cpu
        bert_cpu.set_seed(0)        # same seed -> identical run every time
        x = bert_cpu.randn(3, 4)    # reproducible

    Call ``set_seed`` again with another value for a different but equally
    repeatable run; omit it entirely for a fresh random run.

    Returns
    -------
    None
        The seed is applied to NumPy's global random state in place.

    References
    ----------
    [2] H. V. Pham, S. Qian, J. Wang, T. Lutellier, J. Rosenthal, L. Tan,
        Y. Yu, and N. Nagappan, "Problems and Opportunities in Training Deep
        Learning Software Systems: An Analysis of Variance," Proceedings of
        the 35th IEEE/ACM International Conference on Automated Software
        Engineering, pp. 771–783, 2020.
    [3] B. Chen, M. Wen, Y. Shi, D. Lin, G. K. Rajbahadur, and Z. M. Jiang,
        “Towards Training Reproducible Deep Learning Models,”
        Proceedings of the 44th International Conference on Software Engineering,
        2022.
    [4] David Goldberg. 1991. What Every Computer Scientist Should Know About
        Floating-Point Arithmetic. ACM Comput. Surv. 23, 1 (1991), 5–48


    """
    np.random.seed(seed)



def _expand_reduced(grad: np.ndarray, axis: Axis, keepdims: bool, target_shape: Tuple[int, ...]) -> np.ndarray:

    """Expand a reduction's upstream gradient back to the input shape.

    Consider the gradient of ``(w.T @ x).sum()``. Let ``z = w.T @ x`` be an array
    with shape ``(2, 3)`` (e.g. ``w`` is ``(2, 2)`` and ``x`` is ``(2, 3)``), and
    let ``out = z.sum()`` be a scalar:

        z = w.T @ x = [ z00  z01  z02 ]    out = z.sum()
                      [ z10  z11  z12 ]

        out = z00 + z01 + z02 + z10 + z11 + z12 

    Because ``out`` is the sum of every element of ``z``, each element has local
    derivative:

        d out / d z_ij = 1.

    When ``out.backward()`` is called, the engine populates ``out.grad`` with the
    upstream gradient. See ``Tensor.sum`` for the derivative rule used to compute
    ``d out / d z`` when ``out = z.sum()``. 
    
    To obtain the gradient with respect to ``z``, this scalar
    must be expanded to the same shape as ``z``. Since every element of ``z`` has
    local derivative 1, the upstream gradient is repeated across all positions
    using NumPy broadcasting [5]. For an upstream gradient equal to 1:

        np.broadcast_to(grad, (2, 3)) = [ 1  1  1 ]
                                        [ 1  1  1 ]

    More generally, if the upstream scalar were ``g``, every element would receive
    ``g``.

    This is a full reduction because ``axis is None``. No reduced axes need to be
    reinserted before broadcasting the scalar to the original input shape.

    For a partial reduction, the removed dimensions must first be restored. For
    example, ``z.sum(axis=1)`` collapses the three columns and produces an output
    with shape ``(2,)``. Its upstream gradient also has shape ``(2,)``. Before it
    can be broadcast to ``z``'s shape, axis 1 is reinserted with
    ``np.expand_dims``, producing shape ``(2, 1)``:

        grad.shape                         -> (2,)
        np.expand_dims(grad, axis=1)       -> (2, 1)
        np.broadcast_to(grad, (2, 3))      -> (2, 3)

    This restores the gradient shape required by the input of the reduction.

    The ``keepdims`` flag decides whether that first step is needed. With
    ``keepdims=True`` the reduced axis is *kept* as a size-1 dimension (e.g.
    ``z.sum(axis=1, keepdims=True)`` already has shape ``(2, 1)``), so the
    gradient is already aligned with the input and ``np.expand_dims`` is skipped
    — only the broadcast runs. The reinsertion therefore happens exactly when an
    axis was reduced *and* dropped, i.e. ``axis is not None and not keepdims``.

    Parameters
    ----------
    grad : np.ndarray
        Upstream gradient, carrying the reduction's (smaller) output shape.
    axis : int, tuple of int, or None
        Axis/axes that were reduced; ``None`` means a full reduction.
    keepdims : bool
        Whether the reduction kept the reduced axes as size-1 dimensions.
    target_shape : tuple of int
        Shape of the reduction's input — the shape to expand the gradient to.

    Returns
    -------
    np.ndarray
        ``grad`` expanded (re-inserted if needed, then broadcast) to
        ``target_shape``.

    References
    ----------
    [5] NumPy Developers, "Broadcasting," NumPy User Guide:
    https://numpy.org/devdocs/user/basics.broadcasting.html
    """
    if axis is not None and not keepdims:
        # Put the removed axes back as size-1 so the array re-aligns with target.
        grad = np.expand_dims(grad, axis)
    # Stretch the size-1 axes up to the full input shape (copy across the
    # elements that were collapsed into each output entry).
    return np.broadcast_to(grad, target_shape)


class Tensor:
    """An n-dimensional array node in the autograd computational graph.

    For an accessible introduction to computational graphs, see Andrew Ng's
    explanation here: https://youtu.be/hCP1vGoCdYU?si=DvIRDH0MucRckYcU 
    
    A ``Tensor`` stores both a numerical value and the information required to
    propagate gradients through the operation that created it. Tensors may be
    leaf nodes, such as model parameters and inputs, or intermediate nodes
    produced by arithmetic operations.

    Consider the expression:

        out = (w.T @ x).sum()

    where ``@`` is **matrix multiplication** — the linear-algebra row-by-column
    product, *not* the elementwise ``*`` — and ``wᵀ`` is the transpose of ``w``.
    The ``wᵀ @ x`` form is the usual way to write a linear layer. (For matrices
    ``A`` and ``B``, ``A @ B`` requires the inner dimensions to match:
    ``(m, k) @ (k, n) -> (m, n)``.) With ``w`` of shape ``(2, 2)`` and ``x`` of
    shape ``(2, 3)``, ``z = wᵀ @ x`` has shape ``(2, 3)``. The engine constructs
    the following computational graph:

        ● out          (sum -> scalar)
        └─ ● z          (@ -> shape (2, 3))
           ├─ ● w
           └─ ● x

    Here, ``z = wᵀ @ x`` is an intermediate ``Tensor`` whose parent nodes are
    ``w`` and ``x``. The scalar ``out`` is another ``Tensor``, created by
    reducing all elements of ``z``.

    When ``out.backward()`` is called, the engine seeds ``out.grad`` with 1,
    corresponding to:

        d out / d out = 1

    It then traverses the graph in reverse topological order. The ``sum`` node
    uses ``_expand_reduced`` to restore the reduced dimensions and broadcast its
    upstream gradient back to ``z.shape``. The matmul node then applies the
    chain rule: each input's gradient is the upstream gradient ``d out / d z``
    times that input's local slope of ``z``:

        d out / d x = (d out / d z) · (d z / d x),   with   d z / d x = w
        d out / d w = (d out / d z) · (d z / d w),   with   d z / d w = x

    Written so the matrix shapes line up, those products are:

        d out / d x = w @ (d out / d z)
        d out / d w = x @ (d out / d z).T

    The backward (derivative) rule for each operation lives in its own method:
    see ``Tensor.__matmul__`` for the matrix multiplication and ``Tensor.sum``
    for the sum.

    The resulting gradients are accumulated in ``x.grad`` and ``w.grad``.
    Gradient accumulation is necessary because the same tensor may contribute
    to the final output through more than one path in the graph.

    Attributes
    ----------
    data : np.ndarray
        Numerical value computed. For example, the ``data`` stored by ``z`` in
        ``z = w.T @ x`` is the matrix product of ``w.data.T`` and ``x.data``.

    grad : np.ndarray
        Gradient of the final scalar output, ``out``, with respect to this
        tensor's ``data``. It has the same shape as ``data`` and is initialized
        to zeros.

        Gradient contributions are accumulated with addition rather than
        overwritten because the same tensor may influence ``out`` through
        multiple paths in the computational graph. For example, consider:

            out = x**2 + 3 * x

        The tensor ``x`` contributes to ``out`` through two paths. Drawing the
        graph the way the ``learn`` walkthroughs do (output on top, operands
        hanging below), ``x`` is reached twice — once under each path:

            ● out          (the + node)
            ├─ ● x²         (the **2 node)
            │  └─ ● x
            └─ ● 3·x        (the * node)
               └─ ● x       ⤴ (the same x as above)

        By the sum rule, the total derivative with respect to ``x`` is the sum
        of the derivative contributions from both paths:

            d out / d x = d(x**2) / d x + d(3x) / d x
                        = 2x + 3

        During the backward pass, each path therefore adds its contribution to
        ``x.grad``:

            x.grad += 2 * x.data
            x.grad += 3

        If gradients were overwritten instead of accumulated, the contribution
        from one path would replace the contribution from the other. Accumulation
        ensures that ``x.grad`` stores the complete derivative obtained from all
        paths connecting ``x`` to ``out``.


    requires_grad : bool
        Whether this tensor should accumulate gradients. When ``False``, the
        tensor is treated as a constant during the backward pass.

    _backward : Callable[[], None]
        Closure containing the local derivative rule for the operation that
        created this tensor.

        The gradient stored in this tensor is the upstream gradient received from
        the next operation in the computational graph. In other words, if this
        tensor is ``z``, then ``self.grad`` represents:

            d out / d z

        where ``out`` is the final scalar output. The closure combines this
        upstream gradient with the local derivatives of the operation that created
        ``z`` and accumulates the resulting contributions into the parent tensors.

        For example, consider:

            z = w.T @ x

        In the closure stored in ``z._backward``, ``self`` refers to ``z``.
        Therefore:

            self.grad == z.grad == d out / d z

        Applying the chain rule gives:

            d out / d x = (d z / d x) @ (d out / d z)
            d out / d w = (d z / d w) @ (d out / d z)

        For compatible vector and matrix shapes, the closure conceptually performs:

            x.grad += w.data @ z.grad
            w.grad += np.outer(x.data, z.grad)

        Thus, ``z.grad`` controls how strongly the local derivatives of the
        matrix-multiplication operation contribute to the gradients of ``x`` and
        ``w``. These contributions are accumulated because the same parent tensor
        may influence the final output through multiple paths.



    _prev : set[Tensor]
        Parent tensors used to create this tensor. For ``z = wᵀ @ x``,
        ``z._prev`` contains ``w`` and ``x``. Leaf tensors normally have no
        parents.

    _op : str
        Human-readable label for the operation that created this tensor, such
        as ``"*"``, ``"+"``, ``"@"``, or ``"sum"``. It is used for debugging and
        computational-graph visualization.
    """

    def __init__(
        self,
        data: ArrayLike,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
        requires_grad: bool = True,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        # If the input is already a Tensor, extract only its numerical data.
        # This avoids wrapping a Tensor inside another Tensor.
        if isinstance(data, Tensor):
            data = data.data

        # Store the numerical value of this node as a NumPy array. The element
        # type is flexible so the engine can run in float64/32/16/... :
        #   * an explicit ``dtype`` always wins;
        #   * otherwise a float array is kept as-is, so the precision chosen
        #     for the leaves propagates through every op result unchanged;
        #   * anything else (Python lists/scalars, integer arrays) adopts the
        #     library-wide ``default_dtype`` (float64 by default, which keeps
        #     numerical gradient checks stable).
        if dtype is not None:
            self.data: np.ndarray = np.asarray(data, dtype=dtype)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            self.data = data
        else:
            self.data = np.asarray(data, dtype=default_dtype)

        # The gradient mirrors data's shape AND dtype, so precision is uniform.
        self.grad: np.ndarray = np.zeros_like(self.data)
        self.requires_grad: bool = requires_grad
        self._backward: Callable[[], None] = lambda: None
        # ``_children`` are the tensors this one was built from; storing them as
        # ``_prev`` is what records the *edges* of the computational graph. Each
        # op (``__add__``, ``__matmul__``, ``cat``, ...) passes its operands here,
        # so the graph is assembled incrementally as the forward pass runs. Leaf
        # tensors pass nothing, so ``_prev`` stays empty and a backward traversal
        # stops there.
        self._prev: set = set(_children)
        self._op: str = _op

        # Tally this op's forward FLOPs into the global counter (0 for leaves and
        # pure data-movement ops). ``_children`` is still ordered here, which the
        # matmul cost needs (its inner dimension comes from the first operand).
        _add_flops(_forward_flops(_op, self.data, tuple(_children)))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_tensor(x: ArrayLike) -> "Tensor":
        """Wrap a raw value as a non-grad constant tensor (or pass through)."""
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the underlying array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        """Transpose of the last two axes (matrix transpose)."""
        if self.data.ndim < 2:
            return self
        axes = list(range(self.data.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return self.transpose(*axes)

    # ------------------------------------------------------------------ #
    # Elementwise binary ops
    # ------------------------------------------------------------------ #
    def __add__(self, other: ArrayLike) -> "Tensor":
        """Elementwise addition with broadcasting.

        Backward: d out/d self = 1 and d out/d other = 1, so each parent just
        receives ``out.grad`` (``_unbroadcast`` undoes any broadcasting).
        """
        other = self._as_tensor(other)
        out = Tensor(
            data=self.data + other.data,
            # ``_children`` are the operands this op consumed. The constructor
            # stores them as ``out._prev`` (see ``Tensor.__init__``), and *that*
            # is what builds the computational graph: passing ``(self, other)``
            # here is the edge "out was produced from self and other". Leaves
            # (tensors you create directly) pass nothing, so their ``_prev`` is
            # empty — which is exactly where a backward traversal stops.
            _children=(self, other),
            _op="+",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike) -> "Tensor":
        """Elementwise multiplication with broadcasting.

        Backward: d out/d self = other and d out/d other = self, i.e.
        ``self.grad += other * out.grad`` and ``other.grad += self * out.grad``.
        """
        other = self._as_tensor(other)
        out = Tensor(
            data=self.data * other.data,
            _children=(self, other),   # recorded as out._prev -> wires out into the graph
            _op="*",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += self._unbroadcast(other.data * out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, other: Union[int, float]) -> "Tensor":
        """Elementwise power by a constant exponent.

        Backward: for ``out = self ** n``, d out/d self = n * self**(n-1), so
        ``self.grad += n * self**(n-1) * out.grad``.
        """
        assert isinstance(other, (int, float)), "exponent must be a scalar constant"
        out = Tensor(
            data=self.data ** other,
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op=f"**{other}",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Batched matrix multiplication (``@``).

        Backward: for ``out = A @ B``, dA = out.grad @ Bᵀ and dB = Aᵀ @ out.grad
        (transposing the last two axes; ``_unbroadcast`` folds any batch dims).
        """
        other = self._as_tensor(other)
        out = Tensor(
            data=self.data @ other.data,
            _children=(self, other),   # recorded as out._prev -> wires out into the graph
            _op="@",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad @ np.swapaxes(other.data, -1, -2)   # dA = grad @ Bᵀ
                _add_flops(2 * self.data.size * out.grad.shape[-1])
                self.grad += self._unbroadcast(grad, self.shape)
            if other.requires_grad:
                grad = np.swapaxes(self.data, -1, -2) @ out.grad    # dB = Aᵀ @ grad
                _add_flops(2 * other.data.size * self.data.shape[-2])
                other.grad += self._unbroadcast(grad, other.shape)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Unary / activation ops
    # ------------------------------------------------------------------ #
    def exp(self) -> "Tensor":
        """Elementwise natural exponential."""
        out = Tensor(
            data=np.exp(self.data),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="exp",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        """Elementwise natural logarithm."""
        out = Tensor(
            data=np.log(self.data),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="log",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        """Elementwise square root."""
        out = Tensor(
            data=np.sqrt(self.data),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="sqrt",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (0.5 / out.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        """Elementwise hyperbolic tangent."""
        out = Tensor(
            data=np.tanh(self.data),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="tanh",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (1.0 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        """Elementwise rectified linear unit."""
        out = Tensor(
            data=np.maximum(0.0, self.data),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="relu",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """Gaussian Error Linear Unit (tanh approximation, as used by BERT)."""
        c = np.sqrt(2.0 / np.pi)
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = Tensor(
            data=0.5 * x * (1.0 + t),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="gelu",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                dinner = c * (1.0 + 3.0 * 0.044715 * x ** 2)
                dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dinner
                self.grad += dgelu * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Shape ops
    # ------------------------------------------------------------------ #
    def reshape(self, *shape: int) -> "Tensor":
        """Return a view of the tensor with a new shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(
            data=self.data.reshape(shape),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="reshape",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        """Permute the axes of the tensor."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if len(axes) == 0:
            axes = tuple(reversed(range(self.data.ndim)))
        out = Tensor(
            data=self.data.transpose(axes),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="transpose",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                inv = tuple(np.argsort(axes))
                self.grad += out.grad.transpose(inv)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Reductions (must broadcast the upstream grad back on the way down)
    # ------------------------------------------------------------------ #
    def sum(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Sum over ``axis``.

        Backward: d out/d self = 1 for every summed element, so ``out.grad`` is
        broadcast back to ``self.shape`` via ``_expand_reduced``.
        """
        out = Tensor(
            data=self.data.sum(axis=axis, keepdims=keepdims),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="sum",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _expand_reduced(out.grad, axis, keepdims, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Arithmetic mean over ``axis``."""
        out = Tensor(
            data=self.data.mean(axis=axis, keepdims=keepdims),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="mean",
            requires_grad=self.requires_grad,
        )
        # Number of elements collapsed into each output entry.
        n = self.data.size / out.data.size

        def _backward() -> None:
            if self.requires_grad:
                g = _expand_reduced(out.grad, axis, keepdims, self.shape)
                self.grad += g / n

        out._backward = _backward
        return out

    def var(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Variance over ``axis`` (used by LayerNorm). Uses population (ddof=0)."""
        out = Tensor(
            data=self.data.var(axis=axis, keepdims=keepdims),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="var",
            requires_grad=self.requires_grad,
        )
        mu = self.data.mean(axis=axis, keepdims=True)
        n = self.data.size / out.data.size

        def _backward() -> None:
            if self.requires_grad:
                g = _expand_reduced(out.grad, axis, keepdims, self.shape)
                self.grad += (2.0 / n) * (self.data - mu) * g

        out._backward = _backward
        return out

    def max(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        """Maximum over ``axis`` (used for numerically stable softmax)."""
        out = Tensor(
            data=self.data.max(axis=axis, keepdims=keepdims),
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="max",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                md = self.data.max(axis=axis, keepdims=True)
                mask = (self.data == md).astype(self.data.dtype)
                # Split the gradient evenly across tied maxima.
                mask /= mask.sum(axis=axis, keepdims=True)
                g = _expand_reduced(out.grad, axis, keepdims, self.shape)
                self.grad += mask * g

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Composite ops
    # ------------------------------------------------------------------ #
    def softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable softmax along ``axis``."""
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        sm = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(
            data=sm,
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="softmax",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                dot = (out.grad * sm).sum(axis=axis, keepdims=True)
                self.grad += sm * (out.grad - dot)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Autograd
    # ------------------------------------------------------------------ #
    def backward(self) -> None:
        """Run reverse-mode autodiff from this (scalar) tensor.

        Builds the topological order of the graph, seeds ``self.grad`` with
        ones, then invokes each node's ``_backward`` in reverse order.
        """
        topo: list = []
        visited: set = set()

        def build(v: "Tensor") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self) -> None:
        """Reset this tensor's gradient to zeros."""
        self.grad = np.zeros_like(self.data)

    # ------------------------------------------------------------------ #
    # Broadcasting helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Sum ``grad`` so that it matches ``shape``.

        Reverses NumPy broadcasting: any axis that was expanded during the
        forward pass is summed out in the backward pass.
        """
        # Collapse the leading axes that broadcasting prepended.
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        # Collapse axes that were size-1 in the original but expanded.
        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # ------------------------------------------------------------------ #
    # Reflected / derived operators
    # ------------------------------------------------------------------ #
    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-self._as_tensor(other))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return (-self) + other

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        return self * (self._as_tensor(other) ** -1.0)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return (self ** -1.0) * other

    def __getitem__(self, idx) -> "Tensor":
        """Indexing / slicing (used for embedding lookups)."""
        out = Tensor(
            data=self.data[idx],
            _children=(self,),   # recorded as out._prev -> wires out into the graph
            _op="getitem",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                # ``add.at`` accumulates correctly when ``idx`` repeats rows.
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, "
            f"data={self.data!r})"
        )


# ---------------------------------------------------------------------- #
# Tensor constructors
# ---------------------------------------------------------------------- #
def zeros(*shape: int, requires_grad: bool = True, dtype: Optional[np.dtype] = None) -> Tensor:
    """Create a tensor of zeros (``dtype`` defaults to ``default_dtype``)."""
    return Tensor(np.zeros(shape, dtype=dtype or default_dtype), requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = True, dtype: Optional[np.dtype] = None) -> Tensor:
    """Create a tensor of ones (``dtype`` defaults to ``default_dtype``)."""
    return Tensor(np.ones(shape, dtype=dtype or default_dtype), requires_grad=requires_grad)


def randn(*shape: int, requires_grad: bool = True, dtype: Optional[np.dtype] = None) -> Tensor:
    """Create a tensor of standard-normal samples (``dtype`` defaults to ``default_dtype``)."""
    return Tensor(np.random.randn(*shape).astype(dtype or default_dtype), requires_grad=requires_grad)


# ---------------------------------------------------------------------- #
# Joining ops
# ---------------------------------------------------------------------- #
def cat(tensors: Iterable["Tensor"], axis: int = 0) -> Tensor:
    """Concatenate tensors along ``axis`` into a single node.

    The forward value is just ``np.concatenate``; the interesting part is the
    backward rule. Concatenation does not mix or scale any element — it only
    *places* each input's values into a contiguous slice of the output. So for
    ``out = cat([A, B], axis=0)`` the upstream gradient ``out.grad`` is simply
    **cut back into the same slices** and handed to each input unchanged:

        A ── row block 0 ──┐
                           ├──> out      grad(A) = out.grad[rows of A]
        B ── row block 1 ──┘             grad(B) = out.grad[rows of B]

    There is no local derivative to multiply by (it is the identity on each
    slice), which is why this is used for the "bias trick": prepending a constant
    row ``x_0 = 1`` to an input lets the gradient w.r.t. the real features flow
    straight back through the corresponding slice, while the constant row (a
    ``requires_grad=False`` tensor) is ignored.

    Parameters
    ----------
    tensors : iterable of Tensor
        The tensors to join. All must share the same shape except along ``axis``.
    axis : int
        The axis along which to concatenate.

    Returns
    -------
    Tensor
        The concatenated tensor, wired so each input receives its own slice of
        the upstream gradient.
    """
    tensors = [Tensor._as_tensor(t) for t in tensors]
    out = Tensor(
        data=np.concatenate([t.data for t in tensors], axis=axis),
        _children=tuple(tensors),   # every input recorded as out._prev -> graph edges
        _op="cat",
        requires_grad=any(t.requires_grad for t in tensors),
    )

    # Sizes along the concat axis -> the split points in the output gradient.
    sizes = [t.shape[axis] for t in tensors]
    split_points = np.cumsum(sizes)[:-1]

    def _backward() -> None:
        # Slice out.grad back into the per-input blocks and route each one home.
        grads = np.split(out.grad, split_points, axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                t.grad += g

    out._backward = _backward
    return out
