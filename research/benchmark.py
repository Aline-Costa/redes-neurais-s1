"""The measurement instrument for the research questions.

Every question in ``01_research_questions.ipynb`` is a trade-off — *does this idea
buy accuracy per FLOP?* — so every experiment has to report two numbers, not one.
This module makes that cheap and, more importantly, **comparable**: two students
answering two different questions should produce tables that can be read side by
side, and against the q05 / q06 baselines.

    from research.benchmark import run_adult, baseline_mlp, repeat, summarize

    rows = repeat(lambda seed: run_adult(baseline_mlp, seed=seed, name="baseline"))
    print(summarize(rows))

What it gives you:

- :func:`run_adult` — the q05 training loop (Adult income, full-batch Adam),
  parameterised so a question can swap the model, the loss, the optimiser or the
  schedule without forking it. Returns a :class:`Result` carrying train/val/test
  accuracy, **training FLOPs**, **inference FLOPs** and wall time.
- :func:`run_sgns` — the q06 loop (word2vec on *Flatland*), scored by the
  silhouette of the representative word groups before and after training.
- :func:`repeat` / :func:`summarize` / :func:`pareto` — three seeds, mean ± std,
  and the accuracy-versus-FLOPs picture that answers most of the questions.

The notebooks in ``exercises/`` are the didactic version of these loops, written
out cell by cell; this file is the compact instrument. It is checked against them:
``run_adult(baseline_mlp, seed=0)`` reproduces q05's 0.8543 test accuracy and its
84,227,782,200 training FLOPs exactly, and ``run_sgns`` reproduces q06's
-0.058 -> +0.141 silhouette. If you change this file, keep that true.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
from bert_cpu import engine as cpu
from bert_cpu import nn
from bert_cpu import optim
from bert_cpu.loss import cross_entropy
from bert_cpu.tokenizer import Tokenizer


# ============================================================================ #
# Results
# ============================================================================ #
@dataclass
class Result:
    """One run of one configuration, with its cost.

    ``score`` is the headline number (test accuracy on Adult, silhouette on
    Flatland) so that :func:`summarize` and :func:`pareto` work for both tasks;
    the task-specific numbers live beside it and in ``extra``.
    """

    name: str
    seed: int
    score: float
    train_flops: int
    infer_flops: int = 0
    seconds: float = 0.0
    extra: Dict = field(default_factory=dict)

    @property
    def gflops(self) -> float:
        return self.train_flops / 1e9


class count_flops:
    """Context manager around the engine's global FLOP tally.

        with count_flops() as c:
            model(x)
        print(c.value)

    Note it is *global*: it measures every engine op executed inside the block.
    """

    def __enter__(self) -> "count_flops":
        cpu.reset_flops()
        self.value = 0
        return self

    def __exit__(self, *exc) -> None:
        self.value = cpu.flop_count()


# ============================================================================ #
# Adult income classification (the q05 task)
# ============================================================================ #
_ADULT_CACHE: Dict[str, object] = {}


def load_adult_cached() -> Tuple[object, object]:
    """``datasets.load_adult`` for both splits, parsed once per process."""
    if "train" not in _ADULT_CACHE:
        _ADULT_CACHE["train"] = datasets.load_adult("train")
        _ADULT_CACHE["test"] = datasets.load_adult("test")
    return _ADULT_CACHE["train"], _ADULT_CACHE["test"]


def train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2):
    """Column-oriented hold-out split (identical to the one in q05)."""
    n = X.shape[1]
    perm = np.random.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return X[:, tr_idx], y[tr_idx], X[:, val_idx], y[val_idx]


class MLP(nn.Module):
    """``Linear -> act -> Linear`` — the q05 model, with the width and the
    activation left as knobs (several questions vary exactly those)."""

    def __init__(self, n_features: int, hidden: int = 64, n_classes: int = 2,
                 activation: Callable = lambda t: t.relu()) -> None:
        self.activation = activation
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


def baseline_mlp(n_features: int) -> MLP:
    """The q05 reference model: 108 -> 64 -> ReLU -> 2."""
    return MLP(n_features, hidden=64)


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    """Arg-max accuracy from a pure forward pass."""
    logits = model(cpu.Tensor(X, requires_grad=False)).data
    return float((logits.argmax(axis=0) == y).mean())


def default_loss(model: nn.Module, X: cpu.Tensor, y: np.ndarray) -> cpu.Tensor:
    """Softmax cross entropy on column-oriented logits (the q05 objective)."""
    return cross_entropy(model(X).T, y)


def run_adult(
    model_factory: Callable[[int], nn.Module] = baseline_mlp,
    *,
    epochs: int = 100,
    lr: float = 1e-2,
    seed: int = 0,
    val_frac: float = 0.2,
    loss_fn: Callable[[nn.Module, cpu.Tensor, np.ndarray], cpu.Tensor] = default_loss,
    opt_factory: Callable = lambda params, lr: optim.Adam(params, lr=lr),
    on_epoch: Optional[Callable[[int, nn.Module, object], None]] = None,
    subset: Optional[np.ndarray] = None,
    name: str = "adult",
    verbose: bool = False,
) -> Result:
    """Train a classifier on Adult and report accuracy *and* the FLOPs it cost.

    Parameters you are meant to vary
    --------------------------------
    model_factory : ``n_features -> Module``
        Anything with a ``forward`` producing ``(n_classes, batch)`` logits.
    loss_fn : ``(model, X, y) -> scalar Tensor``
        Full control of the forward and the objective — this is where mixup,
        R-Drop, distillation or a reweighted loss goes. Defaults to the q05
        objective. Whatever it does is counted in the FLOP tally.
    opt_factory : ``(params, lr) -> Optimizer``
        e.g. ``lambda p, lr: optim.SGD(p, lr=lr, momentum=0.9)``, or your own.
    on_epoch : ``(epoch, model, opt) -> None``
        Called after every step — for pruning masks, LR schedules, logging.
    subset : array of column indices
        Train on a subset of the training rows (data-pruning questions). The
        validation split is carved out *before* the subset is applied.

    Returns
    -------
    Result
        ``score`` is test accuracy. ``train_flops`` is everything the engine
        executed during training (forward + backward + the validation forwards);
        ``infer_flops`` is one forward pass over the test split, which is the
        number to quote when a question is about inference cost.
    """
    cpu.set_seed(seed)
    train_ds, test_ds = load_adult_cached()

    X_tr, y_tr, X_val, y_val = train_val_split(train_ds.X, train_ds.y, val_frac)
    if subset is not None:
        X_tr, y_tr = X_tr[:, subset], y_tr[subset]

    model = model_factory(train_ds.n_features)
    opt = opt_factory(model.parameters(), lr)

    Xt = cpu.Tensor(X_tr, requires_grad=False)
    Xv = cpu.Tensor(X_val, requires_grad=False)

    train_hist: List[float] = []
    val_hist: List[float] = []
    total_flops = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        cpu.reset_flops()

        opt.zero_grad()
        loss = loss_fn(model, Xt, y_tr)
        loss.backward()
        opt.step()

        val_loss = float(default_loss(model, Xv, y_val).data)
        total_flops += cpu.flop_count()

        train_hist.append(float(loss.data))
        val_hist.append(val_loss)
        if on_epoch is not None:
            on_epoch(epoch, model, opt)
        if verbose:
            print(f"  epoch {epoch:3d}/{epochs}   train {train_hist[-1]:.4f}"
                  f"   val {val_loss:.4f}")

    seconds = time.time() - started

    with count_flops() as c:
        test_acc = accuracy(model, test_ds.X, test_ds.y)

    return Result(
        name=name,
        seed=seed,
        score=test_acc,
        train_flops=total_flops,
        infer_flops=c.value,
        seconds=seconds,
        extra={
            "acc_train": accuracy(model, X_tr, y_tr),
            "acc_val": accuracy(model, X_val, y_val),
            "acc_test": test_acc,
            "epochs": epochs,
            "lr": lr,
            "n_params": sum(p.data.size for p in model.parameters()),
            "train_hist": train_hist,
            "val_hist": val_hist,
            "model": model,
        },
    )


# ============================================================================ #
# word2vec on Flatland (the q06 task)
# ============================================================================ #
REPRESENTATIVE: Dict[str, List[str]] = {
    "shapes": ["triangle", "square", "pentagon", "hexagon", "circle", "polygon"],
    "people": ["women", "men", "man", "woman", "wife", "husband"],
    "space":  ["dimension", "dimensions", "space", "plane", "solid", "line"],
}

_FLAT_CACHE: Dict[Tuple[int, int], Tuple] = {}


def flatland_pairs(min_count: int = 5, window: int = 3):
    """Corpus -> ``(word2id, id2word, freqs, centers, contexts)``, cached.

    Deterministic and free of RNG draws, so caching it does not disturb the
    random stream a seeded run depends on.
    """
    key = (min_count, window)
    if key in _FLAT_CACHE:
        return _FLAT_CACHE[key]

    flat = datasets.load_flatland()
    counts: Dict[str, int] = {}
    for sent in flat.sentences:
        for tok in Tokenizer._basic_tokenize(sent):
            if tok.isalpha():
                counts[tok] = counts.get(tok, 0) + 1
    kept = sorted((w for w, c in counts.items() if c >= min_count),
                  key=lambda w: (-counts[w], w))
    word2id = {w: i for i, w in enumerate(kept)}
    freqs = np.array([counts[w] for w in kept], dtype=np.float64)

    centers: List[int] = []
    contexts: List[int] = []
    for sent in flat.sentences:
        ids = [word2id[t] for t in Tokenizer._basic_tokenize(sent)
               if t.isalpha() and t in word2id]
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    centers.append(ids[i])
                    contexts.append(ids[j])

    out = (word2id, kept, freqs,
           np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64))
    _FLAT_CACHE[key] = out
    return out


def negative_sampler(freqs: np.ndarray, power: float = 0.75):
    """Sampler over the unigram distribution raised to ``power`` (word2vec's 3/4)."""
    weights = freqs ** power
    probs = weights / weights.sum()

    def sample(shape) -> np.ndarray:
        return np.random.choice(len(probs), size=shape, p=probs)

    return sample


class SkipGramNS(nn.Module):
    """Two embedding tables: input (center) and output (context)."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        self.in_emb = nn.Embedding(vocab_size, dim)
        self.out_emb = nn.Embedding(vocab_size, dim)


def log_sigmoid(x: cpu.Tensor) -> cpu.Tensor:
    """``log sigma(x)``, composed from engine primitives."""
    return (((-x).exp() + 1.0) ** -1.0).log()


def ns_loss(model: SkipGramNS, centers, contexts, negatives, dim: int) -> cpu.Tensor:
    """Skip-gram negative-sampling loss for one batch (one graph, one backward)."""
    b = centers.shape[0]
    v = model.in_emb(centers)
    u_pos = model.out_emb(contexts)
    u_neg = model.out_emb(negatives)
    pos_score = (v * u_pos).sum(axis=1)
    neg_score = (u_neg * v.reshape(b, 1, dim)).sum(axis=2)
    total = log_sigmoid(pos_score).sum() + log_sigmoid(-neg_score).sum()
    return -total / b


def silhouette_samples(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-point silhouette coefficient under cosine distance."""
    X = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    dist = 1.0 - X @ X.T
    np.fill_diagonal(dist, 0.0)

    labels = np.asarray(labels)
    uniq = set(labels.tolist())
    scores = np.zeros(len(labels))
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == lab].mean() for lab in uniq if lab != labels[i])
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return scores


def silhouette_score(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient — the q06 headline metric."""
    return float(silhouette_samples(vectors, labels).mean())


def representative_vectors(table: np.ndarray, word2id: Dict[str, int]):
    """``(vectors, labels, words)`` for the representative groups that are in-vocab."""
    vecs, labels, words = [], [], []
    ci = 0
    for _, group in REPRESENTATIVE.items():
        kept = [w for w in group if w in word2id]
        if len(kept) < 2:
            continue
        for w in kept:
            vecs.append(table[word2id[w]])
            labels.append(ci)
            words.append(w)
        ci += 1
    return np.array(vecs), np.array(labels), words


def run_sgns(
    *,
    dim: int = 32,
    window: int = 3,
    neg_k: int = 5,
    min_count: int = 5,
    epochs: int = 100,
    batch: int = 2048,
    lr: float = 0.02,
    seed: int = 0,
    tie_tables: bool = False,
    opt_factory: Callable = lambda params, lr: optim.Adam(params, lr=lr),
    name: str = "sgns",
    verbose: bool = False,
) -> Result:
    """Train word2vec (skip-gram + negative sampling) on *Flatland*.

    ``score`` is the silhouette of the representative word groups **after**
    training; ``extra["silhouette_before"]`` is the same measure on the random
    initialisation, which is the honest zero point. Every hyper-parameter here is
    also a FLOP knob — ``dim``, ``neg_k`` and ``epochs`` all multiply the cost.
    """
    cpu.set_seed(seed)
    word2id, id2word, freqs, centers, contexts = flatland_pairs(min_count, window)
    sample_negatives = negative_sampler(freqs)

    model = SkipGramNS(len(word2id), dim)
    if tie_tables:                       # one table playing both roles
        model.out_emb = model.in_emb

    vecs, labels, _ = representative_vectors(model.in_emb.weight.data, word2id)
    before = silhouette_score(vecs, labels)

    opt = opt_factory(model.parameters(), lr)
    n_pairs = centers.shape[0]
    total_flops = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        cpu.reset_flops()
        perm = np.random.permutation(n_pairs)
        running, n_batches = 0.0, 0
        for start in range(0, n_pairs, batch):
            idx = perm[start:start + batch]
            neg = sample_negatives((len(idx), neg_k))
            opt.zero_grad()
            loss = ns_loss(model, centers[idx], contexts[idx], neg, dim)
            loss.backward()
            opt.step()
            running += float(loss.data)
            n_batches += 1
        total_flops += cpu.flop_count()
        if verbose:
            print(f"   epoch {epoch}/{epochs}   mean loss = {running / n_batches:+.3f}")

    seconds = time.time() - started
    vecs, labels, _ = representative_vectors(model.in_emb.weight.data, word2id)
    after = silhouette_score(vecs, labels)

    return Result(
        name=name,
        seed=seed,
        score=after,
        train_flops=total_flops,
        seconds=seconds,
        extra={
            "silhouette_before": before,
            "silhouette_after": after,
            "vocab": len(word2id),
            "pairs": int(n_pairs),
            "dim": dim,
            "neg_k": neg_k,
            "epochs": epochs,
            "n_params": sum(p.data.size for p in model.parameters()),
            "model": model,
            "word2id": word2id,
            "id2word": id2word,
        },
    )


# ============================================================================ #
# Seeds, tables, pictures
# ============================================================================ #
def repeat(run: Callable[[int], Result], seeds: Sequence[int] = (0, 1, 2)) -> List[Result]:
    """Run one configuration under several seeds.

    One seed is an anecdote. Three is the minimum this project accepts, because
    the differences these questions chase are often smaller than seed noise —
    which is itself a finding worth reporting.
    """
    return [run(s) for s in seeds]


def summarize(results: Sequence[Result], baseline: Optional[str] = None) -> str:
    """Mean ± std per configuration name, with cost, as a printable table."""
    groups: Dict[str, List[Result]] = {}
    for r in results:
        groups.setdefault(r.name, []).append(r)

    header = (f"{'configuration':<28}{'n':>3}  {'score':>16}  {'train GFLOP':>12}"
              f"  {'infer MFLOP':>12}  {'params':>9}  {'s/run':>7}")
    lines = [header, "-" * len(header)]
    ref = None
    for cfg, rows in groups.items():
        scores = [r.score for r in rows]
        mean = statistics.fmean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        gf = statistics.fmean([r.gflops for r in rows])
        mf = statistics.fmean([r.infer_flops for r in rows]) / 1e6
        pars = statistics.fmean([r.extra.get("n_params", 0) for r in rows])
        secs = statistics.fmean([r.seconds for r in rows])
        if cfg == baseline:
            ref = mean
        lines.append(f"{cfg:<28}{len(rows):>3}  {mean:>8.4f} ± {std:<5.4f}"
                     f"  {gf:>12.2f}  {mf:>12.1f}  {pars:>9,.0f}  {secs:>7.1f}")

    if ref is not None:
        lines.append("")
        lines.append(f"(deltas against '{baseline}')")
        for cfg, rows in groups.items():
            if cfg == baseline:
                continue
            mean = statistics.fmean([r.score for r in rows])
            std = statistics.stdev([r.score for r in rows]) if len(rows) > 1 else 0.0
            delta = mean - ref
            verdict = "within noise" if abs(delta) <= std else "outside 1 sd"
            lines.append(f"  {cfg:<26} {delta:+.4f}   ({verdict})")
    return "\n".join(lines)


def pareto(results: Sequence[Result], *, cost: str = "train_flops",
           title: str = "accuracy vs compute", annotate: bool = True):
    """Scatter the score against its cost — the picture most of these questions want.

    Needs matplotlib (``pip install matplotlib``); prints a note and returns
    ``None`` if it is missing, so a headless run still works.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed — `pip install matplotlib` for the plot.")
        return None

    groups: Dict[str, List[Result]] = {}
    for r in results:
        groups.setdefault(r.name, []).append(r)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cfg, rows in groups.items():
        xs = [getattr(r, cost) for r in rows]
        ys = [r.score for r in rows]
        ax.errorbar(statistics.fmean(xs), statistics.fmean(ys),
                    yerr=(statistics.stdev(ys) if len(ys) > 1 else 0.0),
                    fmt="o", capsize=3, label=cfg)
        if annotate:
            ax.annotate(cfg, (statistics.fmean(xs), statistics.fmean(ys)),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(cost.replace("_", " ") + "  (log scale)")
    ax.set_ylabel("score")
    ax.set_title(title)
    fig.tight_layout()
    return ax
