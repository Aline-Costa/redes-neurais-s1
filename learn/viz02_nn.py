"""Didactic visualisation: a linear layer is a matrix, and stacking layers
builds a chain of derivatives.

Where ``viz01_engine.py`` shows a *single* node-level graph, this walkthrough
moves up one level of abstraction and shows what ``nn.py`` is really about:

1. a **linear layer is just a matrix** (with the bias folded in as row 0), then
   a nonlinearity — ``a = act(Wᵀ @ x)``; and
2. when several such layers are **stacked**, each with its own nonlinear
   activation, the backward pass becomes a **chain of derivatives**: the
   gradient is propagated layer by layer, multiplied at every step by that
   layer's activation slope ``act'(z)`` and its weight matrix.

It is *not* a test: the layers' correctness is checked in ``test/test_nn.py``.
This file is here to be **read and run**::

    python -m learn.viz02_nn                               # defaults: float64, random seed
    python -m learn.viz02_nn --precision float32 --seed 5  # pick dtype and seed
"""

import numpy as np

from bert_cpu import engine as cpu
from bert_cpu import nn
from learn.console import console
from learn.viz01_engine import draw_graph, draw_backward_steps


def demo_linear_chain() -> None:
    """Visualise the chain rule over a stack of three activated linear layers.

    Stacks three ``nn.Linear`` layers, each with a different nonlinear activation
    (ReLU → tanh → GELU), draws the whole stack as one computational graph, then
    replays the backward pass **one layer at a time**, spelling out the
    chain-rule factors that connect the loss to every weight.
    """
    print("\n")
    print(console.text("=" * 70))
    print(console.text("NN demo:  stacking Linear layers and computing its chain of derivatives"))
    print(console.text("=" * 70))

    print(console.text("\nThe network consists of three fully connected layers. In each layer,\nthe affine transformation ")
          + console.math("Wᵀ @ [1; ·]") 
          + console.text(", where the leading ") + console.math("1") + console.text(" accounts for\nthe bias term, is followed by a nonlinear activation function."))
    
    print(console.text("\nRead the set of equations describing the network from bottom to top, \nfollowing the forward pass from the input ")
          + console.math("x") + console.text(" to the prediction ") + console.math("ŷ") + console.text(":\n"))
    
    print(console.math("  ŷ  = gelu(W3ᵀ @ [1; a2])"))
    print(console.math("  a2 = tanh(W2ᵀ @ [1; a1])"))
    print(console.math("  a1 = relu(W1ᵀ @ [1; x ])"))

    x = cpu.Tensor(np.random.randn(2, 1));  x.label = "x"   # column input (2, 1)
    layer1 = nn.Linear(2, 2);  layer1.weight.label = "W1"
    layer2 = nn.Linear(2, 2);  layer2.weight.label = "W2"
    layer3 = nn.Linear(2, 1);  layer3.weight.label = "W3"

    z1 = layer1(x);   z1.label = "z1";   a1 = z1.relu();  a1.label = "a1"
    z2 = layer2(a1);  z2.label = "z2";   a2 = z2.tanh();  a2.label = "a2"
    z3 = layer3(a2);  z3.label = "z3";   a3 = z3.gelu();  a3.label = "ŷ"
    loss = a3.sum();  loss.label = "L"

    # The three weight matrices ARE the network: together they hold every
    # parameter (row 0 of each is that layer's bias). Show them, and the input.
    print(console.text("\nThe network is its three weight matrices ") + console.math("W1, W2, W3")
          + console.text(" (first row of each is\nthe neuron's bias in that layer). Together they hold every parameter the network has:\n"))

    print(console.text("For how a network's layers are modelled as matrices, see Andrew Ng's two\nvideos in sequence: ")
        + console.value("https://youtu.be/CcRkHl75Z-Y?si=JRsu9U6zsEjpkzfC")
        + console.text("\n                    ")
        + console.value("https://youtu.be/rMOdrD61IoU?si=2FNLuNEHVKkG6TiB") + console.text("\n"))

    for w in (layer1.weight, layer2.weight, layer3.weight):
        print(console.math(f"  {w.label}  (shape {w.shape[0]}x{w.shape[1]}) ="))
        print(console.fmt_matrix(w.data, indent="      "))

    print(console.text("\nThe input is the column vector ") + console.math("x (shape 2x1)\n"))
    print(console.fmt_matrix(x.data, indent="      "))

    def show_matmul(eq: str, weight, prev_data: np.ndarray, z) -> None:
        """Print ``z = Wᵀ @ [1; prev]`` as actual matrices: Wᵀ, the augmented
        input column ``[1; prev]``, and the resulting ``z`` — side by side."""
        aug = np.vstack([np.ones((1, prev_data.shape[1])), prev_data])  # prepend x_0 = 1
        print(console.math("  " + eq + " ="))
        block = console.hjoin([
            (console.fmt_matrix_plain(weight.data.T), console.value),   # Wᵀ
            ("@", console.math),
            (console.fmt_matrix_plain(aug), console.value),             # [1; prev]
            ("=", console.math),
            (console.fmt_matrix_plain(z.data), console.value),          # z
        ])
        print("\n".join("      " + ln for ln in block.split("\n")))

    print(console.text("\nThe ")+console.label("Forward pass")+console.text(": each pre-activation ") + console.math("z")
          + console.text(" is a matrix product ") + console.math("Wᵀ @ [1; ·]")
          + console.text(", then an activation:\n"))
    
    show_matmul("z1 = W1ᵀ @ [1; x ]", layer1.weight, x.data, z1)
    console.kv("  a1 = relu(z1)        = ", console.fmt_auto(a1.data), color=console.math)
    show_matmul("z2 = W2ᵀ @ [1; a1]", layer2.weight, a1.data, z2)
    console.kv("  a2 = tanh(z2)        = ", console.fmt_auto(a2.data), color=console.math)
    show_matmul("z3 = W3ᵀ @ [1; a2]", layer3.weight, a2.data, z3)
    console.kv("  ŷ  = gelu(z3)        = ", console.fmt_auto(a3.data), color=console.math)
    console.kv("  L  = sum(ŷ)          = ", console.fmt_auto(loss.data), "    (a scalar to differentiate)", color=console.math)

    # ------------------------------------------------------------------ #
    # Why it is one graph: follow ._prev iteratively from ŷ
    # ------------------------------------------------------------------ #
    print(console.text("\nHow do we know these separate steps are ") + console.label("one graph")
          + console.text("? Every ") + console.math("Tensor")
          + console.text(" keeps the\ntensors it was built from in its ") + console.math("._prev")
          + console.text(" set. Start the frontier at ") + console.math("ŷ")
          + console.text(" and\nrepeatedly replace it with the union of its operands ")
          + console.math("(node._prev)") + console.text(":"))

    def node_name(n):
        # The only unlabelled nodes are the bias-trick constant, whose value is 1.
        return getattr(n, "label", None) or n._op or "1"

    print(console.text("\n  start:    { ") + console.math("ŷ") + console.text(" }"))
    frontier = {a3}                                       # a3 is labelled "ŷ"
    while True:
        nxt = set()
        for node in frontier:
            nxt |= node._prev                             # follow every link
        if not nxt:
            print(console.text("  ._prev →  { }   ") + console.text("(empty — the leaves: the input ")
                  + console.math("x") + console.text(", the weights, the bias ")
                  + console.math("1") + console.text(")"))
            break
        names = ", ".join(sorted(node_name(n) for n in nxt))
        print(console.text("  ._prev →  { ") + console.math(names) + console.text(" }"))
        frontier = nxt

    print(console.text("\nFollowing ") + console.math("._prev")
          + console.text(" alone swept from ") + console.math("ŷ")
          + console.text(" to every leaf, so the steps really are ") + console.label("one graph")
          + console.text(".\nHere it is drawn, the output ") + console.math("L")
          + console.text(" on top down to the inputs ") + console.math("x, W1, W2, W3")
          + console.text(":"))
    draw_graph(loss, known=set())
    print(console.text("It is exactly the engine's graph from ") + console.math("viz01_engine")
          + console.text(", just bigger: three ") + console.math("Wᵀ @ [1; ·]")
          + console.text(" blocks chained through their activations."))

    # Now replay the backward pass over that SAME graph, node by node — exactly
    # what viz01_engine does, just on a bigger graph. Each step seeds/reveals one
    # more gradient, redraws the table, and prints the chain-rule step it used.
    print(console.text("\nNow the ") + console.label("backward pass")
          + console.text(": seed ") + console.math("dL/dL = 1")
          + console.text(" and walk the SAME graph in reverse. Each step\nfills in one more ")
          + console.label("grad") + console.text(" and shows the chain-rule computation between the tables:"))
    draw_backward_steps(loss)

    print(console.text("\nEvery node's ") + console.label("grad") + console.text(" is now filled — the weights ")
          + console.math("W1, W2, W3") + console.text(" included. Calling ") + console.math("loss.backward()")
          + console.text("\ndoes exactly this, automatically, from the output ") + console.math("L")
          + console.text(" back to every input."))


# ====================================================================== #
# Standalone entry point (mirrors viz01_engine.main)
# ====================================================================== #
def main(argv=None) -> None:
    """Run the layer/chain walkthrough with a chosen precision and RNG seed.

    The precision and seed are configured on the engine *before* anything is
    built, so they govern the whole run::

        python -m learn.viz02_nn --precision float32 --seed 5
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
        help="Seed for the NumPy RNG (reproducible run). Omit for a random run. "
             "Seed 5 is a good lively choice (all ReLU units fire).",
    )
    args = parser.parse_args(argv)

    try:
        dtype = np.dtype(args.precision)
    except TypeError as exc:
        parser.error(f"unknown precision {args.precision!r}: {exc}")
    if not np.issubdtype(dtype, np.floating):
        parser.error(f"precision must be a floating type, got {args.precision!r}")

    cpu.default_dtype = dtype.type
    if args.seed is not None:
        cpu.set_seed(args.seed)

    seed_str = "random" if args.seed is None else str(args.seed)
    print(console.text("Engine config: ") + console.label("seed") + console.text("=")
          + console.value(seed_str) + console.text("  ") + console.label("precision")
          + console.text("=") + console.value(dtype.name))
    demo_linear_chain()


if __name__ == "__main__":
    main()
