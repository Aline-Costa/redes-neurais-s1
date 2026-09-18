"""Loaders for the example datasets bundled with BERT-cpu.

Two datasets live here, for the two things this project trains:

- **UCI Adult / Census-Income** (``load_adult``) — a *tabular* classification set
  (predict whether a person earns ``>50K`` a year from 14 demographic features).
  The raw files live in ``datasets/adult/`` — ``adult.data`` (train) and
  ``adult.test`` (test).
- **Flatland** (``load_flatland``) — a *text* corpus, the full public-domain
  novella *Flatland: A Romance of Many Dimensions* (Edwin Abbott, Project
  Gutenberg eBook #97), in ``datasets/flatland/corpus.txt``. This is the corpus
  for the tokenizer and masked-language-model side of the project: raw prose to
  learn a vocabulary from and to mask-and-predict over.

Everything below the Adult section (encoding, standardisation, one-hot) is
specific to that tabular set; the Flatland loader at the bottom is much simpler —
it just cleans the Gutenberg boilerplate and hands back text.

The loader returns data that is **ready to feed the engine**: every feature is
numeric and the layout follows the project's *column-oriented* convention
(``X`` is ``(n_features, n_samples)``, the same orientation ``nn.Linear``
expects), and labels are integers in ``{0, 1}``. Nothing here depends on
``bert_cpu`` — it produces plain NumPy arrays, so you decide when to wrap them in
a ``Tensor``.

Encoding choices (all standard, kept deliberately simple and inspectable):

- **Continuous** features are *standardised* (z-scored): ``(x - mean) / std``.
- **Categorical** features are *one-hot* encoded; the missing-value token ``"?"``
  is treated as its own category, so no rows are dropped.
- The **label** ``>50K`` -> ``1``, ``<=50K`` -> ``0``.

Crucially, the encoder (the continuous means/stds and the category vocabularies)
is **fit on the training split only** and then *applied* to whichever split you
load. Fitting on test would leak information from the evaluation set into the
features — a classic mistake this loader avoids by construction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# The raw files sit in datasets/adult/ regardless of the caller's working dir.
_ADULT_DIR = Path(__file__).resolve().parent / "adult"

# Column schema, in file order (from datasets/adult/adult.names). 14 features
# followed by the income label.
_COLUMNS: List[str] = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]
_CONTINUOUS: List[str] = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]
_CATEGORICAL: List[str] = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
# The label field (last column). ">50K" is the positive class; the test file
# writes the labels with a trailing period (">50K."), stripped below.
_POSITIVE_LABEL = ">50K"

_SPLIT_FILES = {"train": "adult.data", "test": "adult.test"}


class AdultDataset:
    """A loaded, fully-numeric split of the Adult dataset.

    Attributes
    ----------
    X : np.ndarray
        Feature matrix, shape ``(n_features, n_samples)`` — *column-oriented*,
        matching ``nn.Linear`` (features down axis 0, samples across axis 1).
        Continuous columns are standardised; categoricals are one-hot.
    y : np.ndarray
        Integer labels, shape ``(n_samples,)``, values in ``{0, 1}``
        (``1`` == income ``>50K``).
    feature_names : list of str
        Name of each of the ``n_features`` rows of ``X``. One-hot columns are
        named ``"field=value"`` (e.g. ``"sex=Male"``), so a row of ``X`` is easy
        to trace back to its meaning.
    categories : dict[str, list[str]]
        The vocabulary (fit on train) used to one-hot each categorical field.
    split : str
        Which split this is (``"train"`` or ``"test"``).
    """

    def __init__(self, X, y, feature_names, categories, split) -> None:
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.categories = categories
        self.split = split

    @property
    def n_features(self) -> int:
        return self.X.shape[0]

    @property
    def n_samples(self) -> int:
        return self.X.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (f"AdultDataset(split={self.split!r}, "
                f"n_features={self.n_features}, n_samples={self.n_samples})")


def _read_rows(path: Path) -> List[List[str]]:
    """Parse a raw Adult file into a list of ``[field, ...]`` string rows.

    Skips blank lines and comment lines (the test file opens with a
    ``|1x3 Cross validator`` header), and any row that does not have the
    expected 15 comma-separated fields.
    """
    expected = len(_COLUMNS) + 1                      # 14 features + label
    rows: List[List[str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("|"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != expected:
                continue
            rows.append(parts)
    return rows


def _fit_encoder(train_rows: List[List[str]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """Learn the encoding *from the training split only*.

    Returns the per-feature mean and std for the continuous columns and the
    (sorted) category vocabulary for each categorical column. These are applied,
    unchanged, to whatever split is later encoded — so the test set never
    influences the transformation.
    """
    cont_idx = [_COLUMNS.index(c) for c in _CONTINUOUS]
    cont = np.array([[float(r[i]) for i in cont_idx] for r in train_rows], dtype=np.float64)
    means = cont.mean(axis=0)
    stds = cont.std(axis=0)
    stds[stds == 0.0] = 1.0                            # guard constant columns

    categories: Dict[str, List[str]] = {}
    for c in _CATEGORICAL:
        j = _COLUMNS.index(c)
        categories[c] = sorted({r[j] for r in train_rows})   # includes "?" if present
    return means, stds, categories


def _encode(
    rows: List[List[str]],
    means: np.ndarray,
    stds: np.ndarray,
    categories: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Turn parsed string rows into a numeric ``(X_rows_by_features, y, names)``.

    ``X`` is built sample-major ``(n_samples, n_features)`` here; the public
    loader transposes it to the column-oriented ``(n_features, n_samples)``.
    """
    n = len(rows)

    # Continuous block: standardise with the train statistics.
    cont_idx = [_COLUMNS.index(c) for c in _CONTINUOUS]
    cont = np.array([[float(r[i]) for i in cont_idx] for r in rows], dtype=np.float64)
    cont = (cont - means) / stds

    # Categorical blocks: one-hot against the train vocabulary. A value unseen in
    # training (rare) maps to an all-zero row for that field rather than erroring.
    feature_names: List[str] = list(_CONTINUOUS)
    blocks = [cont]
    for c in _CATEGORICAL:
        j = _COLUMNS.index(c)
        vocab = categories[c]
        index = {v: k for k, v in enumerate(vocab)}
        block = np.zeros((n, len(vocab)), dtype=np.float64)
        for row_i, r in enumerate(rows):
            k = index.get(r[j])
            if k is not None:
                block[row_i, k] = 1.0
        blocks.append(block)
        feature_names.extend(f"{c}={v}" for v in vocab)

    X = np.hstack(blocks)                              # (n_samples, n_features)

    # Label: strip the test file's trailing "." then map to {0, 1}.
    y = np.array(
        [1 if r[-1].rstrip(".").strip() == _POSITIVE_LABEL else 0 for r in rows],
        dtype=np.int64,
    )
    return X, y, feature_names


def load_adult(split: str = "train") -> AdultDataset:
    """Load a split of the Adult dataset as model-ready NumPy arrays.

    Parameters
    ----------
    split : {"train", "test"}
        Which split to return. The encoder is always fit on ``"train"`` and
        applied to the requested split (see the module docstring).

    Returns
    -------
    AdultDataset
        With ``.X`` ``(n_features, n_samples)``, ``.y`` ``(n_samples,)`` in
        ``{0, 1}``, plus ``.feature_names`` and ``.categories``.

    Examples
    --------
    >>> import datasets
    >>> train = datasets.load_adult("train")
    >>> train
    AdultDataset(split='train', n_features=108, n_samples=32561)
    >>> train.X.shape, train.y.shape
    ((108, 32561), (32561,))
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(_SPLIT_FILES)}, got {split!r}")

    # Fit on train, then encode the requested split with those same statistics.
    train_rows = _read_rows(_ADULT_DIR / _SPLIT_FILES["train"])
    means, stds, categories = _fit_encoder(train_rows)

    rows = train_rows if split == "train" else _read_rows(_ADULT_DIR / _SPLIT_FILES[split])
    X, y, feature_names = _encode(rows, means, stds, categories)

    # Transpose to the project's column-oriented (n_features, n_samples) layout.
    return AdultDataset(
        X=X.T, y=y, feature_names=feature_names, categories=categories, split=split
    )


# ====================================================================== #
# Flatland — a plain-text corpus for the tokenizer / masked-LM
# ====================================================================== #

# The novella sits next to this module regardless of the caller's working dir.
_FLATLAND_FILE = Path(__file__).resolve().parent / "flatland" / "corpus.txt"

# Project Gutenberg wraps the actual book between these two marker lines. Anything
# before the first / after the second is licensing and catalogue boilerplate.
_GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG"
_GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG"

# A deliberately *naive* sentence splitter: break after ., ! or ? when followed by
# whitespace. It has no idea about abbreviations ("Mr.", "e.g."), so it will
# occasionally over-split — fine for a didactic corpus, and documented as such.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class FlatlandDataset:
    """The Flatland corpus, cleaned of Gutenberg boilerplate, in three forms.

    The same text is offered at three granularities so the caller picks whatever
    a task needs — the whole string, paragraphs, or sentences. A ``Tokenizer``
    trains on any list of strings, so ``build_vocab(ds.sentences)`` or
    ``build_vocab(ds.paragraphs)`` both work directly.

    Attributes
    ----------
    text : str
        The full book text with the Gutenberg header/footer removed.
    paragraphs : list of str
        Blank-line-separated paragraphs, each with internal whitespace collapsed
        to single spaces (illustration/section markers like ``[Illustration]`` are
        dropped).
    sentences : list of str
        The paragraphs split into sentences by a naive ``.!?`` rule.
    """

    def __init__(self, text: str, paragraphs: List[str], sentences: List[str]) -> None:
        self.text = text
        self.paragraphs = paragraphs
        self.sentences = sentences

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    def __len__(self) -> int:
        # Sentences are the natural training unit, so len() counts them.
        return len(self.sentences)

    def __repr__(self) -> str:
        return (f"FlatlandDataset(n_paragraphs={self.n_paragraphs}, "
                f"n_sentences={self.n_sentences}, chars={len(self.text)})")


def _strip_gutenberg(raw: str) -> str:
    """Return only the text between the Gutenberg START and END marker lines.

    Falls back to the whole input if the markers are absent, so a plain text file
    without Gutenberg wrapping still loads.
    """
    lines = raw.splitlines()
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if line.startswith(_GUTENBERG_START):
            start = i + 1
        elif line.startswith(_GUTENBERG_END):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def _split_paragraphs(text: str) -> List[str]:
    """Split cleaned text into paragraphs on blank lines.

    Each paragraph's internal newlines and runs of spaces are collapsed to single
    spaces, and pure markup markers (a line that is entirely ``[...]``, e.g.
    ``[Illustration]``) are dropped.
    """
    paragraphs: List[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        para = " ".join(chunk.split())          # collapse all whitespace
        if not para:
            continue
        if para.startswith("[") and para.endswith("]"):
            continue                            # illustration / section marker
        paragraphs.append(para)
    return paragraphs


def _split_sentences(paragraphs: List[str]) -> List[str]:
    """Break each paragraph into sentences with the naive ``.!?`` rule."""
    sentences: List[str] = []
    for para in paragraphs:
        for sent in _SENTENCE_BOUNDARY.split(para):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences


def load_flatland() -> FlatlandDataset:
    """Load the Flatland text corpus as cleaned paragraphs and sentences.

    Reads ``datasets/flatland/corpus.txt``, strips the Project Gutenberg
    header/footer, and returns a :class:`FlatlandDataset` exposing ``.text``,
    ``.paragraphs`` and ``.sentences``. Pure text in, plain Python strings out —
    nothing here depends on ``bert_cpu``.

    Returns
    -------
    FlatlandDataset
        With ``.text`` (str), ``.paragraphs`` (list of str) and ``.sentences``
        (list of str).

    Examples
    --------
    >>> import datasets
    >>> flat = datasets.load_flatland()
    >>> flat.sentences[0]
    'I call our world Flatland, not because we call it so, ...'
    >>> from bert_cpu import Tokenizer
    >>> tok = Tokenizer()
    >>> tok.build_vocab(flat.sentences, max_size=2000)   # learn WordPiece from it
    """
    raw = _FLATLAND_FILE.read_text(encoding="utf-8")
    text = _strip_gutenberg(raw)
    paragraphs = _split_paragraphs(text)
    sentences = _split_sentences(paragraphs)
    return FlatlandDataset(text=text, paragraphs=paragraphs, sentences=sentences)
