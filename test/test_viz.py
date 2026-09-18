"""Smoke tests for the didactic walkthroughs in ``learn/``.

These do not check any maths (that is the job of ``test_engine.py`` and
``test_nn.py``); they only guarantee the visualisations still *run* end to end,
so the learning material does not silently rot as the engine and layers evolve.
Run them — and see the actual output — with::

    pytest -s test/test_viz.py
"""

from bert_cpu import engine as cpu
from learn.viz01_engine import demo_gradient_graph
from learn.viz02_nn import demo_linear_chain


def test_viz_engine_runs():
    """The engine gradient-graph walkthrough runs without error."""
    cpu.set_seed(0)
    demo_gradient_graph()


def test_viz_nn_runs():
    """The layer/chain walkthrough runs without error (seed 5 keeps it lively)."""
    cpu.set_seed(5)
    demo_linear_chain()
