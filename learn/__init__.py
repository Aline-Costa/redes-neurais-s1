"""Didactic visualisations for BERT-cpu.

This package holds the *runnable walkthroughs* (the ``viz_*`` modules) that print
a computational graph and animate reverse-mode autodiff, so the maths of the
library can be read line by line. They are kept separate from ``test/`` (which
holds only the software/correctness tests) and are meant to be run directly::

    python -m learn.viz01_engine     # the autograd engine, node by node
    python -m learn.viz02_nn         # a linear layer + the chain rule over a stack
"""
