"""Didactic visualisation of the autograd engine's gradient graph.

This walkthrough prints an input vector and an ASCII drawing of the
computational graph, annotating every node with its forward value *and* its
gradient, so a reader can literally see how reverse-mode autodiff propagates the
chain rule from the output back to the inputs. The demo computes
``y = tanh(wᵀ @ x)`` — a tiny linear layer — using the "bias trick" (a leading
``x_0 = 1`` so the bias is just another weight ``w_0``).

It is *not* a test: the engine's correctness is checked in ``test/test_engine.py``.
This file is here to be **read and run**. Run it on its own with::

    python -m learn.viz_engine                               # defaults: float64, random seed
    python -m learn.viz_engine --precision float32 --seed 0  # pick dtype and seed
"""

import numpy as np

from bert_cpu import engine as cpu
from learn.console import console


# ====================================================================== #
# Drawing the gradient graph
# ====================================================================== #
# Column headers for the graph drawing: each node shows its name, forward value
# and gradient. The actual tree (who feeds whom) is drawn by
# ``console.format_graph``; here we only say *what* to print beside each node.
_GRAPH_HEADERS = ["node", "value", "grad"]


def _graph_columns(known: set):
    """Return a ``node -> [name, value, grad]`` cell function for ``format_graph``.

    Gradients of nodes whose id is not yet in ``known`` print as ``-`` (not
    computed yet), so drawing the graph with a growing ``known`` set animates how
    ``backward`` fills the gradients in.
    """
    def columns(node: cpu.Tensor) -> list:
        name = getattr(node, "label", None) or node._op or "leaf"
        # Genuine 2D matrices (e.g. a weight or its transpose) print as a stacked
        # table; scalars and vectors stay on one line.
        value = console.fmt_auto_plain(node.data)
        grad = console.fmt_auto_plain(node.grad) if id(node) in known else "-"
        return [name, value, grad]

    return columns


def draw_graph(root: cpu.Tensor, known: set) -> None:
    """Print the computational graph of ``root`` as a vertical tree (see
    :meth:`console.format_graph`), annotating each node with its value and grad.
    """
    print()
    console.print_graph(root, columns=_graph_columns(known), headers=_GRAPH_HEADERS)
    print()


def _T(name: str) -> str:
    """Toggle a trailing transpose marker ``ᵀ`` on a variable name.

    So ``_T("x") == "xᵀ"`` and ``_T("wᵀ") == "w"`` (a transpose cancels itself),
    which keeps the matmul backward formulas readable.
    """
    return name[:-1] if name.endswith("ᵀ") else name + "ᵀ"


def _backward_rule(v: cpu.Tensor) -> "tuple[str, str]":
    """Return ``(formula, note)`` for how node ``v``'s backward feeds its inputs.

    Spells out the local derivative used by ``v._backward`` so the reader sees
    exactly how each child gradient is computed (the chain-rule step). The
    children are listed in the same sorted order ``draw_graph`` uses. Note that
    ``_prev`` is a *set*, so the left/right order of a matmul's operands is not
    recoverable here — the matmul note is therefore phrased generically.
    """
    vname = getattr(v, "label", "") or v._op or "?"
    kids = [
        getattr(c, "label", "") or c._op or "?"
        for c in sorted(
            v._prev, key=lambda c: (getattr(c, "label", "") or c._op or "", id(c))
        )
    ]

    if v._op == "tanh" and len(kids) == 1:
        z = kids[0]
        return (f"grad({z}) = grad({vname}) * (1 - {vname}^2)",
                f"local derivative of tanh: d{vname}/d{z} = 1 - {vname}^2")
    if v._op == "relu" and len(kids) == 1:
        z = kids[0]
        return (f"grad({z}) = grad({vname}) * ({z} > 0)",
                "relu passes the gradient where its input was positive, blocks it elsewhere")
    if v._op == "gelu" and len(kids) == 1:
        z = kids[0]
        return (f"grad({z}) = grad({vname}) * gelu'({z})",
                "gelu's smooth local derivative")
    if v._op == "@" and len(kids) == 2:
        a, b = kids
        return (f"grad({a}), grad({b}): upstream grad times the *other* factor, transposed",
                "matmul backward: dA = grad @ Bᵀ, dB = Aᵀ @ grad (A, B in forward order)")
    if v._op == "cat":
        return (f"each input gets its own slice of grad({vname})",
                "concatenation just routes each contiguous slice of the gradient back")
    if v._op == "transpose" and len(kids) == 1:
        c = kids[0]
        return (f"grad({c}) = grad({vname})ᵀ",
                "transpose just transposes the gradient straight back")
    if v._op == "sum" and len(kids) == 1:
        c = kids[0]
        return (f"grad({c}) = grad({vname}) copied into every element of {c}",
                "sum sends the same upstream gradient to each summand")
    if v._op == "*" and len(kids) == 2:
        a, b = kids
        return (f"grad({a}) = grad({vname}) * {b},   grad({b}) = grad({vname}) * {a}",
                "elementwise product rule: each factor's grad is the upstream grad "
                "times the other factor")
    # Generic fallback for any other op.
    return (f"grad(child) = grad({vname}) * d{vname}/dchild", "chain rule")


def draw_backward_steps(root: cpu.Tensor) -> None:
    """Run backprop one node at a time, redrawing the graph after each reveal.

    Replays exactly what ``Tensor.backward`` does (topological order, seed the
    output, then call each node's ``_backward`` in reverse), but pauses after
    every step to redraw the graph. A node's gradient is only revealed once
    *all* of its consumers have run their ``_backward`` (so the accumulation is
    complete); until then it prints as ``-``.
    """
    # Topological order: children before parents (same as Tensor.backward).
    topo: list = []
    visited: set = set()

    def build(v: cpu.Tensor) -> None:
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

    build(root)

    # For each node, the set of consumers that feed gradient back into it.
    consumers = {id(n): set() for n in topo}
    for n in topo:
        for child in n._prev:
            consumers[id(child)].add(id(n))

    # Fresh start: zero every gradient, then seed the output with dy/dy = 1.
    for n in topo:
        n.grad = np.zeros_like(n.data)
    root.grad = np.ones_like(root.data)
    known = {id(root)}

    out_name = getattr(root, "label", "") or root._op or "out"
    step = 0
    print(console.label(f"\nSTEP {step}") + console.text(": seed the output, ")
          + console.math(f"grad({out_name}) = d{out_name}/d{out_name} = 1")
          + console.text("; everything else is '-'"))
    draw_graph(root, known)

    processed: set = set()
    for v in reversed(topo):
        v._backward()
        processed.add(id(v))
        # A child is now fully known once every consumer of it is processed.
        newly = [
            c for c in v._prev
            if id(c) not in known and consumers[id(c)] <= processed
        ]
        if not newly:
            continue
        for c in newly:
            known.add(id(c))
        step += 1
        vname = getattr(v, "label", "") or v._op or "leaf"
        filled = ", ".join(getattr(c, "label", "") or c._op or "leaf" for c in newly)
        formula, note = _backward_rule(v)
        print(console.label(f"\nSTEP {step}") + console.text(": backprop through ") + console.math(vname)
              + console.text(" fills the gradient of ") + console.math(filled))
        print(console.text("        ") + console.math(formula) + console.text(f"\n        [{note}]"))
        draw_graph(root, known)



def demo_gradient_graph() -> None:
    """Build a tiny linear unit, run autodiff, and visualise the gradient graph.

    The unit computes ``y = tanh(wᵀ @ x)`` — a single matrix multiplication (a
    linear layer) followed by a tanh. The "bias trick" augments the input with a
    leading constant ``x_0 = 1`` so the bias is just another weight ``w_0``, and
    ``x`` and ``w`` are column vectors so ``wᵀ @ x`` is their weighted sum.
    """
    print('\n')
    print(console.text("=" * 64))
    print(
        console.text("Gradient-graph demo:  ")
        + console.math("y = tanh( wᵀ @ x )")
        + console.text(",  with ")
        + console.math("x_0 = 1")
    )
    print(console.text("=" * 64))

    # Real features of the input (random each run; seed via cpu.set_seed).
    features = np.random.randn(2)
    # Augmented input column vector x = [x_0=1, x_1, x_2]ᵀ, shape (3, 1).
    x = cpu.Tensor([[1.0], [features[0]], [features[1]]]);   x.label = "x"
    # Weight column vector w = [w_0(bias), w_1, w_2]ᵀ, shape (3, 1). Kept small
    # (* 0.3) so the pre-activation stays in tanh's active region and the
    # gradients don't vanish into a saturated tanh.
    w = cpu.Tensor(np.random.randn(3, 1) * 0.3);             w.label = "w"

    print(console.text("\nReal features      : "), end="")
    print(console.fmt(features))
    # x and w are (3, 1) column vectors, so they print stacked vertically, the
    # way a column vector is written in maths.
    print(console.text("Augmented input  ") + console.math("x =")
          + console.text("    (column (3,1); ") + console.math("x_0 = 1") + console.text(" prepended)"))
    print(console.fmt_auto(x.data, indent="    "))
    print(console.text("Weights          ") + console.math("w =")
          + console.text("    (column (3,1); ") + console.math("w_0") + console.text(" is the bias)"))
    print(console.fmt_auto(w.data, indent="    "))

    # Forward pass: transpose w, then the matmul (linear layer), then tanh.
    wt = w.T;       wt.label = "wᵀ"   # transpose -> row (1, 3)
    z = wt @ x;     z.label = "z"     # (1, 3) @ (3, 1) -> (1, 1)
    y = z.tanh();   y.label = "y"

    print(console.text("\nForward pass:"))
    console.kv("  wᵀ           = ", console.fmt_auto(wt.data), "    (w transposed -> row)", color=console.math)
    console.kv("  z = wᵀ @ x   = ", console.fmt_auto(z.data), "    (matrix product = weighted sum + bias)", color=console.math)
    console.kv("  y = tanh(z)  = ", console.fmt_auto(y.data), color=console.math)

    print(console.text("\nEach node stores a ")
        + console.label("node name") + console.text(", a forward ")
        + console.label("value") + console.text(", and a ") + console.label("grad") + console.text("."))
    print(console.text("The output here is ") + console.math("y")
        + console.text(", so the ") + console.label("grad") + console.text(" of a node means ")
        + console.math("dy/d(node)"))
    print(console.text("— the derivative of the output ") + console.math("y")
        + console.text(" with respect to that node. For example,"))
    print(console.label("grad") + console.text("(") + console.math("z") + console.text(") is ")
        + console.math("dy/dz") + console.text(", and ") + console.label("grad") + console.text("(")
        + console.math("w") + console.text(") is ") + console.math("dy/dw") + console.text("."))
    print(console.text("The table below lays out the computational graph, one row per node, and is\nwhere each ")
        + console.label("grad") + console.text(" gets computed (still '-' until backprop reaches it):\n"))

    print(console.text("For an accessible introduction to computational graphs, see Andrew Ng's\nexplanation here: ")
        + console.value("https://youtu.be/hCP1vGoCdYU?si=DvIRDH0MucRckYcU"))

    draw_graph(y, known=set())


    print(console.text("\nBackprop begins by seeding the output, ")
        + console.math("dy/dy = 1")
        + console.text("."))

    print(console.text("(In regular training, this initial value would be ")
        + console.math("dL/dy")
        + console.text(", where L \nis a loss function; here, however, we are differentiating ")
        + console.math("y")
        + console.text(" itself.)"))

    print(console.math("dy/dy")
        + console.text(" is then used to compute ")
        + console.math("dy/dz = grad(z)")
        + console.text("."))

    # Replay backprop step by step, redrawing the graph as each grad is filled.
    draw_backward_steps(y)


# ====================================================================== #
# Standalone entry point
# ====================================================================== #
def main(argv=None) -> None:
    """Run the didactic walkthrough with a chosen precision and RNG seed.

    The precision and seed are configured on the engine *before* anything is
    built, so they govern the whole run::

        python -m learn.viz_engine --precision float32 --seed 0
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        default="float64",
        help="NumPy float dtype for the engine (e.g. float16, float32, float64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the NumPy RNG (reproducible run). Omit for a random run.",
    )
    args = parser.parse_args(argv)

    # Resolve the precision string to a NumPy dtype (clear error if invalid).
    try:
        dtype = np.dtype(args.precision)
    except TypeError as exc:
        parser.error(f"unknown precision {args.precision!r}: {exc}")
    if not np.issubdtype(dtype, np.floating):
        parser.error(f"precision must be a floating type, got {args.precision!r}")

    # Configure the engine globally, before any tensor is created. Only seed
    # when asked, so an unseeded run draws fresh random tensors each time.
    cpu.default_dtype = dtype.type
    if args.seed is not None:
        cpu.set_seed(args.seed)

    seed_str = "random" if args.seed is None else str(args.seed)
    print(console.text("Engine config: ") + console.label("seed") + console.text("=")
          + console.value(seed_str) + console.text("  ") + console.label("precision")
          + console.text("=") + console.value(dtype.name))
    demo_gradient_graph()


if __name__ == "__main__":
    main()
