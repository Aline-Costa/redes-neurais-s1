"""Ensures the project root is on ``sys.path`` so ``import bert_cpu`` works.

Pytest inserts the directory containing the first ``conftest.py`` it finds
(the rootdir) onto ``sys.path``; placing this file at the project root makes the
``bert_cpu`` package importable from the tests without installing it.
"""
