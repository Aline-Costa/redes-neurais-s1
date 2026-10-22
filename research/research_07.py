import pathlib
import sys
import gc
import os
import itertools
import psutil

ROOT = pathlib.Path.cwd()
if not (ROOT / "bert_cpu").is_dir():
    ROOT = ROOT.parent
if not (ROOT / "bert_cpu").is_dir():
    raise RuntimeError("I could not find the bert_cpu folder. Run the script from the root of the repository.")

sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bert_cpu import engine as cpu
from bert_cpu import nn
import research.benchmark as benchmark
from research.benchmark import MLP, baseline_mlp, pareto, run_adult, run_sgns, summarize

process = psutil.Process(os.getpid())

OUT = ROOT / "results" / "research_07"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEEDS = (0, 1, 2, 3, 4)
EPOCHS_ADULT = 50
LR_ADULT = 1e-2

# Reproduce the baseline from q05/q06 prior to any changes (as required by the prompt).
RUN_BASELINE_CHECK = False #True

SGNS_CONFIG = {
    "dim": 32,
    "window": 3,
    "neg_k": 5,
    "min_count": 5,
    "epochs": EPOCHS_ADULT,
    "batch": 2048,
    "lr": 0.02,
}


def ram_gb():
    return process.memory_info().rss / (1024 ** 3)


# Activations

def sigmoid(x):
    return 1.0 / (1.0 + (-x).exp())


def silu(x):
    return x * sigmoid(x)


def activation_triplet(x):
    if isinstance(x, np.ndarray):
        relu = np.maximum(x, 0.0)
        gelu = 0.5 * x * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
        )
        silu_x = x / (1.0 + np.exp(-np.clip(x, -700, 700)))
        return relu, gelu, silu_x
    return x.relu(), x.gelu(), silu(x)


def softmax_triplet(a, b, c):
    ea, eb, ec = a.exp(), b.exp(), c.exp()
    total = ea + eb + ec
    return ea / total, eb / total, ec / total


def tensor_values(t):
    return np.asarray(t.data, dtype=np.float64).reshape(-1)


# Learnable mixtures

class LearnableActivation(nn.Module):
    """Non-normalized linear mixture of ReLU, GELU, and SiLU."""
    def __init__(self):
        super().__init__()
        self.alpha_relu = nn.Parameter(cpu.ones(1, 1))
        self.alpha_gelu = nn.Parameter(cpu.ones(1, 1))
        self.alpha_silu = nn.Parameter(cpu.ones(1, 1))

    def coefficients(self):
        return {
            "alpha_relu": float(self.alpha_relu.data.item()),
            "alpha_gelu": float(self.alpha_gelu.data.item()),
            "alpha_silu": float(self.alpha_silu.data.item()),
        }

    def forward(self, x):
        return (
            self.alpha_relu * x.relu()
            + self.alpha_gelu * x.gelu()
            + self.alpha_silu * silu(x)
        )


class NormalizedLearnableActivation(nn.Module):
    """Mixture with coefficients normalized by softmax."""
    def __init__(self):
        super().__init__()
        self.beta_relu = nn.Parameter(cpu.zeros(1, 1))
        self.beta_gelu = nn.Parameter(cpu.zeros(1, 1))
        self.beta_silu = nn.Parameter(cpu.zeros(1, 1))

    def _weights(self):
        return softmax_triplet(self.beta_relu, self.beta_gelu, self.beta_silu)

    def coefficients(self):
        w = self._weights()
        return {
            "pi_relu": float(w[0].data.item()),
            "pi_gelu": float(w[1].data.item()),
            "pi_silu": float(w[2].data.item()),
        }

    def forward(self, x):
        a, b, c = self._weights()
        return a * x.relu() + b * x.gelu() + c * silu(x)


class Mixture(nn.Module):
    def __init__(self, normalized=False):
        super().__init__()
        self.normalized = normalized
        if normalized:
            self.a = nn.Parameter(cpu.zeros(1, 1))
            self.b = nn.Parameter(cpu.zeros(1, 1))
            self.c = nn.Parameter(cpu.zeros(1, 1))
        else:
            self.a = nn.Parameter(cpu.ones(1, 1) / 3.0)
            self.b = nn.Parameter(cpu.ones(1, 1) / 3.0)
            self.c = nn.Parameter(cpu.ones(1, 1) / 3.0)

    def coefficients(self):
        if self.normalized:
            return softmax_triplet(self.a, self.b, self.c)
        return self.a, self.b, self.c

    def forward(self, x):
        xr, xg, xs = activation_triplet(x)
        a, b, c = self.coefficients()
        return a * xr + b * xg + c * xs


class UnitMixture(Mixture):
    def __init__(self, hidden, normalized=False):
        super().__init__(normalized=normalized)
        self.hidden = hidden

        # Forma (hidden, 1) para entradas no formato (hidden, batch).
        if normalized:
            self.a = nn.Parameter(cpu.zeros(hidden, 1))
            self.b = nn.Parameter(cpu.zeros(hidden, 1))
            self.c = nn.Parameter(cpu.zeros(hidden, 1))
        else:
            self.a = nn.Parameter(cpu.ones(hidden, 1) / 3.0)
            self.b = nn.Parameter(cpu.ones(hidden, 1) / 3.0)
            self.c = nn.Parameter(cpu.ones(hidden, 1) / 3.0)


class TwoHiddenMLP(nn.Module):
    """MLP: n_features -> hidden1 -> hidden2 -> 2."""
    def __init__(self, n_features, mode="global", normalized=False,
                 hidden1=64, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)
        self.mode = mode
        self.normalized = normalized

        if mode == "global":
            self.shared_act = Mixture(normalized)
        elif mode == "per_layer":
            self.act1 = Mixture(normalized)
            self.act2 = Mixture(normalized)
        elif mode == "per_unit":
            self.act1 = UnitMixture(hidden1, normalized)
            self.act2 = UnitMixture(hidden2, normalized)
        elif mode not in ("relu", "gelu", "silu"):
            raise ValueError(f"Unknown mode: {mode}")

    def apply_activation(self, x, layer):
        if self.mode == "relu":
            return x.relu()
        if self.mode == "gelu":
            return x.gelu()
        if self.mode == "silu":
            return silu(x)
        if self.mode == "global":
            return self.shared_act(x)
        if self.mode in ("per_layer", "per_unit"):
            return (self.act1 if layer == 1 else self.act2)(x)
        raise ValueError(self.mode)

    def forward(self, x):
        x = self.apply_activation(self.fc1(x), 1)
        x = self.apply_activation(self.fc2(x), 2)
        return self.fc3(x)


# Coefficients and logging

def coefficients_for_activation(activation, prefix=""):
    if isinstance(activation, (LearnableActivation, NormalizedLearnableActivation)):
        return {f"{prefix}{k}": v for k, v in activation.coefficients().items()}
    if isinstance(activation, Mixture):
        vals = activation.coefficients()
        arrays = [tensor_values(t) for t in vals]
        names = ("pi_relu", "pi_gelu", "pi_silu") if activation.normalized else (
            "alpha_relu", "alpha_gelu", "alpha_silu"
        )
        out = {}
        for n, a in zip(names, arrays):
            out[f"{prefix}{n}"] = float(a.mean())
            if a.size > 1:
                # Dispersion among units (per-unit): the average conceals the structure.
                out[f"{prefix}{n}_std"] = float(a.std())
        return out
    return {}


def collect_model_coefficients(model):
    if hasattr(model, "activation") and model.activation is not None:
        return coefficients_for_activation(model.activation)
    if getattr(model, "mode", None) == "global":
        return coefficients_for_activation(model.shared_act)
    if getattr(model, "mode", None) in ("per_layer", "per_unit"):
        return {
            **coefficients_for_activation(model.act1, "layer1_"),
            **coefficients_for_activation(model.act2, "layer2_"),
        }
    return {}


def make_coefficient_logger(history):
    def on_epoch(epoch, model, opt):
        row = {"epoch": epoch}
        row.update(collect_model_coefficients(model))
        if len(row) > 1:
            history.append(row)
    return on_epoch


# Settings Adult

ORIGINAL_CONFIGS = [
    ("ReLU", baseline_mlp, False),
    ("GELU", lambda n: MLP(n, hidden=64, n_classes=2, activation=lambda x: x.gelu()), False),
    ("SiLU", lambda n: MLP(n, hidden=64, n_classes=2, activation=silu), False),
    ("Learnable", lambda n: MLP(n, hidden=64, n_classes=2, activation=LearnableActivation()), True),
    ("Normalized", lambda n: MLP(n, hidden=64, n_classes=2, activation=NormalizedLearnableActivation()), True),
    # Part of the statement: "would the parameters have yielded better results with 2 additional hidden units?"
    ("ReLU h=66", lambda n: MLP(n, hidden=66, n_classes=2, activation=lambda x: x.relu()), False),
    ("GELU h=66", lambda n: MLP(n, hidden=66, n_classes=2, activation=lambda x: x.gelu()), False),
    ("SiLU h=66", lambda n: MLP(n, hidden=66, n_classes=2, activation=silu), False),
]

# (name, mode, normalized, TwoHiddenMLP kwargs)
EXTENDED_CONFIGS = [
    ("TwoHidden-ReLU", "relu", False, {}),
    ("TwoHidden-GELU", "gelu", False, {}),
    ("TwoHidden-SiLU", "silu", False, {}),
    ("Global", "global", False, {}),
    ("Per-layer", "per_layer", False, {}),
    ("Per-unit", "per_unit", False, {}),
    ("Global-normalized", "global", True, {}),
    ("Per-layer-normalized", "per_layer", True, {}),
    ("Per-unit-normalized", "per_unit", True, {}),
    # Width for 2 layers: +2 units in the 1st layer
    # = +2*(n_features+1) + 2*hidden2 parameters (approx. the 288 from the Per-unit model).
    ("TwoHidden-ReLU h1=66", "relu", False, {"hidden1": 66}),
    ("TwoHidden-GELU h1=66", "gelu", False, {"hidden1": 66}),
    ("TwoHidden-SiLU h1=66", "silu", False, {"hidden1": 66}),
]

TRANSFER_MODELS = {
    "Normalized", "Global", "Per-layer", "Per-unit",
    "Global-normalized", "Per-layer-normalized", "Per-unit-normalized",
}

# Paired comparison groups versus the BEST fixed activation.
PAIRED_GROUPS = [
    {
        "label": "1 camada oculta",
        "fixed": ["ReLU", "GELU", "SiLU"],
        "widened": {"ReLU": "ReLU h=66", "GELU": "GELU h=66", "SiLU": "SiLU h=66"},
        "mixtures": ["Learnable", "Normalized"],
    },
    {
        "label": "2 camadas ocultas",
        "fixed": ["TwoHidden-ReLU", "TwoHidden-GELU", "TwoHidden-SiLU"],
        "widened": {
            "TwoHidden-ReLU": "TwoHidden-ReLU h1=66",
            "TwoHidden-GELU": "TwoHidden-GELU h1=66",
            "TwoHidden-SiLU": "TwoHidden-SiLU h1=66",
        },
        "mixtures": [
            "Global", "Per-layer", "Per-unit",
            "Global-normalized", "Per-layer-normalized", "Per-unit-normalized",
        ],
    },
]

# Statistical analysis (paired by seed)
# >>> ANALYSIS_BEGIN

def bootstrap_ci(d, n_boot=10000, seed=0, alpha=0.05):
    """IC percentil (bootstrap) da média das diferenças pareadas."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def sign_flip_pvalue(d, max_exact=20, n_mc=100000, seed=0):
    """Teste de permutação por inversão de sinais (bilateral).

    Exato para n <= max_exact. Com n = 5 o menor p possível é 2/32 = 0.0625,
    então nunca fica abaixo de 0.05: use também as diferenças individuais.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    if n <= max_exact:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_mc, n))
    stats = np.abs((signs * d).mean(axis=1))
    return float((stats >= obs - 1e-12).mean())


def best_fixed(df, fixed_names):
    """Seleciona a melhor ativação fixa pela acurácia média de validação."""
    required = {"name", "seed", "acc_val"}
    missing_cols = required - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Colunas ausentes: {sorted(missing_cols)}"
        )

    sub = df[df["name"].isin(fixed_names)].copy()

    missing_names = set(fixed_names) - set(sub["name"])
    if missing_names:
        raise ValueError(
            f"Configurações fixas ausentes: {sorted(missing_names)}"
        )

    if sub["acc_val"].isna().any():
        raise ValueError(
            "Existem resultados sem acc_val:\n"
            + sub.loc[
                sub["acc_val"].isna(), ["name", "seed"]
            ].to_string(index=False)
        )

    seed_sets = {
        name: set(group["seed"])
        for name, group in sub.groupby("name")
    }

    reference_seeds = seed_sets[fixed_names[0]]
    inconsistent = {
        name: sorted(seeds)
        for name, seeds in seed_sets.items()
        if seeds != reference_seeds
    }

    if inconsistent:
        raise ValueError(
            "As configurações fixas não têm as mesmas sementes: "
            f"{inconsistent}"
        )

    return (
        sub.groupby("name")["acc_val"]
        .mean()
        .reindex(fixed_names)
        .idxmax()
    )


def paired_vs(df, reference, candidates, metric="score"):
    #Difference between candidate and reference, paired by seed, with CI and cost.
    piv = df.pivot_table(index="seed", columns="name", values=metric)
    cost = df.groupby("name")[["n_params", "train_gflops", "infer_flops", "seconds"]].mean()
    rows = []
    for cand in candidates:
        if cand not in piv.columns or reference not in piv.columns:
            continue
        sub = piv[[cand, reference]].dropna()
        if len(sub) < 2:
            continue
        d = (sub[cand] - sub[reference]).to_numpy()
        lo, hi = bootstrap_ci(d)
        if lo > 0:
            verdict = "melhor (IC > 0)"
        elif hi < 0:
            verdict = "pior (IC < 0)"
        else:
            verdict = "dentro do ruído"

        def pct(col):
            ref_v = cost.loc[reference, col]
            return 100.0 * (cost.loc[cand, col] - ref_v) / ref_v if ref_v else np.nan

        rows.append({
            "reference": reference,
            "candidate": cand,
            "n_seeds": len(d),
            "mean_diff": d.mean(),
            "ci_low": lo,
            "ci_high": hi,
            "wins": f"{int((d > 0).sum())}/{len(d)}",
            "p_signflip": sign_flip_pvalue(d),
            "verdict": verdict,
            "per_seed_diffs": ";".join(f"{x:+.4f}" for x in d),
            "extra_params": cost.loc[cand, "n_params"] - cost.loc[reference, "n_params"],
            "d_train_flops_pct": pct("train_gflops"),
            "d_infer_flops_pct": pct("infer_flops"),
            "d_seconds_pct": pct("seconds"),
        })
    return pd.DataFrame(rows)


def run_paired_analysis(df, groups, metric="score"):
    #Compares mixtures with complete baselines that have been selected via validation.
    names = set(df["name"])
    out = []

    for g in groups:
        expected = set(g["fixed"])
        missing = expected - names

        if missing:
            raise ValueError(
                f"Baselines ausentes no grupo {g['label']}: "
                f"{sorted(missing)}"
            )

        # Requires the three baselines in order to select the best one fairly.
        best = best_fixed(df, g["fixed"])
        wide = g["widened"].get(best)

        parts = [
            paired_vs(df, best, g["mixtures"], metric)
        ]

        if wide is not None:
            if wide not in names:
                raise ValueError(
                    f"Baseline alargada ausente: {wide}"
                )

            parts.append(
                paired_vs(df, wide, g["mixtures"], metric)
            )
            parts.append(
                paired_vs(df, best, [wide], metric)
            )

        parts = [p for p in parts if not p.empty]

        if not parts:
            continue

        table = pd.concat(parts, ignore_index=True)
        table.insert(0, "group", g["label"])
        table["best_fixed"] = best
        out.append(table)

    return (
        pd.concat(out, ignore_index=True)
        if out else pd.DataFrame()
    )


def print_paired(table):
    if table.empty:
        print("Análise pareada: sem dados.")
        return
    show = table.copy()
    for c in ("mean_diff", "ci_low", "ci_high"):
        show[c] = show[c].round(4)
    show["p_signflip"] = show["p_signflip"].round(3)
    for c in ("d_train_flops_pct", "d_infer_flops_pct", "d_seconds_pct"):
        show[c] = show[c].round(1)
    show["extra_params"] = show["extra_params"].round(0)
    cols = ["group", "reference", "candidate", "mean_diff", "ci_low", "ci_high",
            "wins", "p_signflip", "verdict", "extra_params",
            "d_train_flops_pct", "d_infer_flops_pct", "d_seconds_pct"]
    print(show[cols].to_string(index=False))


def coefficient_convergence(history_df, final_df):
    """Os coeficientes saem da inicialização? Convergem ao mesmo lugar entre sementes?

    shift_over_seed_std = deslocamento médio / desvio entre sementes do valor final.
    Valores altos: o treino move os coeficientes de forma consistente.
    Valores baixos: o movimento é comparável à variação entre sementes.
    """
    if history_df.empty or final_df.empty:
        return pd.DataFrame()
    coef_cols = [c for c in final_df.columns if c not in ("method", "seed", "epoch")]
    rows = []
    for method, g in final_df.groupby("method"):
        for col in coef_cols:
            if col not in g.columns or g[col].isna().all():
                continue
            init = history_df[(history_df["method"] == method) & (history_df["epoch"] == 0)]
            if col not in init.columns:
                continue
            m = g[["seed", col]].merge(init[["seed", col]], on="seed", suffixes=("_final", "_init"))
            if m.empty:
                continue
            final = m[f"{col}_final"].to_numpy(dtype=float)
            start = m[f"{col}_init"].to_numpy(dtype=float)
            shift = np.abs(final - start).mean()
            std = final.std(ddof=1) if len(final) > 1 else np.nan
            rows.append({
                "method": method,
                "coefficient": col,
                "init_mean": start.mean(),
                "final_mean": final.mean(),
                "std_across_seeds": std,
                "min_final": final.min(),
                "max_final": final.max(),
                "mean_abs_shift": shift,
                "shift_over_seed_std": shift / std if (std and std > 0) else np.nan,
            })
    return pd.DataFrame(rows)

# <<< ANALYSIS_END
 
# BASELINE_CHECK_BEGIN (reproduction of q05 and q06)

BASELINE_REF = {
    "adult_acc": 0.8543,
    "adult_train_flops": 84_227_782_200,
    "sgns_before": -0.058,
    "sgns_after": 0.141,
}


def check_baseline_reproduction(tol_acc=1e-4, tol_sil=2e-3):
    """Reproduz os números documentados em research/benchmark.py.

    Adult: baseline_mlp, seed 0, 100 épocas -> 0.8543 e 84.227.782.200 FLOPs.
    SGNS : run_sgns padrão sem ativação, seed 0 -> -0.058 -> +0.141.
    Como o benchmark.py foi alterado (activation_factory), isto confirma que
    o caminho sem ativação continua idêntico ao original.
    """
    rows = []

    r = run_adult(baseline_mlp, epochs=100, lr=1e-2, seed=0, name="baseline-check")
    rows.append(("Adult acurácia de teste", r.score, BASELINE_REF["adult_acc"],
                 abs(r.score - BASELINE_REF["adult_acc"]) <= tol_acc))
    rows.append(("Adult FLOPs de treino", r.train_flops, BASELINE_REF["adult_train_flops"],
                 r.train_flops == BASELINE_REF["adult_train_flops"]))
    r.extra.pop("model", None)

    s = run_sgns(seed=0, activation_factory=None, name="baseline-check")
    before = s.extra["silhouette_before"]
    after = s.extra["silhouette_after"]
    rows.append(("SGNS silhouette antes", before, BASELINE_REF["sgns_before"],
                 abs(before - BASELINE_REF["sgns_before"]) <= tol_sil))
    rows.append(("SGNS silhouette depois", after, BASELINE_REF["sgns_after"],
                 abs(after - BASELINE_REF["sgns_after"]) <= tol_sil))
    s.extra.pop("model", None)

    table = pd.DataFrame(rows, columns=["check", "obtained", "expected", "ok"])
    print("\nBaseline check:")
    for _, row in table.iterrows():
        flag = "OK    " if row["ok"] else "FALHOU"
        print(f"  [{flag}] {row['check']}: obtido={row['obtained']}  esperado={row['expected']}")
    if not table["ok"].all():
        print("WARNING: the baseline is NOT reproducible. Review benchmark.py before relying on the results.")
    return table

# <<< BASELINE_CHECK_END


baseline_table = None
if RUN_BASELINE_CHECK:
    baseline_table = check_baseline_reproduction()
    baseline_table.to_csv(OUT / "q07_baseline_check.csv", index=False)
    gc.collect()

# Execution Adult

adult_results = []
adult_models = {}
coefficient_histories = {}


def run_adult_captured(name, factory, track, seed):
    #It captures the trained model and records the coefficients from epoch 0.
    captured = {}
    history = []

    def factory_capture(n):
        model = factory(n)
        captured["model"] = model
        if track:
            # Época 0: valores de inicialização (necessários para medir o deslocamento).
            row = {"epoch": 0}
            row.update(collect_model_coefficients(model))
            if len(row) > 1:
                history.append(row)
        return model

    result = run_adult(
        factory_capture,
        epochs=EPOCHS_ADULT,
        lr=LR_ADULT,
        seed=seed,
        name=name,
        on_epoch=make_coefficient_logger(history) if track else None,
    )
    model = captured.get("model")
    adult_results.append(result)


    if name in TRANSFER_MODELS and model is not None:
        adult_models[(name, seed)] = model

    if track:
        coefficient_histories[(name, seed)] = history

    # The model required for the transfer is already in adult_models. 
    # This avoids keeping a redundant reference in every result.
    result.extra.pop("model", None)

    print(
        f"Adult | {name} | seed={seed} | score={result.score:.5f} | "
        f"time={result.seconds:.2f}s | RAM={ram_gb():.2f} GB"
    )
    return result


for seed in SEEDS:
    for name, factory, track in ORIGINAL_CONFIGS:
        result = run_adult_captured(name, factory, track, seed)
        del result
        gc.collect()

    for name, mode, normalized, kwargs in EXTENDED_CONFIGS:
        factory = lambda n, m=mode, norm=normalized, kw=kwargs: TwoHiddenMLP(
            n, mode=m, normalized=norm, **kw
        )
        result = run_adult_captured(
            name, factory,
            mode in ("global", "per_layer", "per_unit"),
            seed,
        )
        del result
        gc.collect()


# MLP de 1 camada — baseline ReLU
original_names = {name for name, _, _ in ORIGINAL_CONFIGS}
original_results = [r for r in adult_results if r.name in original_names]
print("\nResumo Adult — MLP (baseline: ReLU)")
print(summarize(original_results, baseline="ReLU"))

# TwoHiddenMLP — baseline TwoHidden-ReLU
extended_names = {cfg[0] for cfg in EXTENDED_CONFIGS}
extended_results = [r for r in adult_results if r.name in extended_names]
print("\nResumo Adult — TwoHiddenMLP (baseline: TwoHidden-ReLU)")
print(summarize(extended_results, baseline="TwoHidden-ReLU"))


# Exportação Adult

def result_to_row(r, task):
    extra = getattr(r, "extra", {}) or {}

    row = {
        "task": task,
        "name": getattr(r, "name", ""),
        "seed": getattr(r, "seed", np.nan),
        "score": getattr(r, "score", np.nan),
        "acc_train": extra.get("acc_train", np.nan),
        "acc_val": extra.get("acc_val", np.nan),
        "acc_test": extra.get("acc_test", np.nan),
        "train_flops": getattr(r, "train_flops", np.nan),
        "train_gflops": getattr(r, "train_flops", np.nan) / 1e9,
        "infer_flops": (getattr(r, "infer_flops", np.nan)
                                        if str(task).startswith("Adult")
                                        else np.nan
                                ),
        "seconds": getattr(r, "seconds", np.nan),
        "n_params": extra.get("n_params", np.nan),
        
    }


   # Preserves other scalar metrics, such as silhouette_before
   # and silhouette_after, without storing large objects like models.
    for k, v in extra.items():
        if (k not in ("model", "word2id", "id2word") and np.isscalar(v)):
            row[k] = v

    return row


adult_df = pd.DataFrame([result_to_row(r, "Adult") for r in adult_results])
adult_df.to_csv(OUT / "q07_adult_results.csv", index=False)

history_rows = [
    {"method": method, "seed": seed, **item}
    for (method, seed), hist in coefficient_histories.items()
    for item in hist
]
history_df = pd.DataFrame(history_rows)
history_df.to_csv(OUT / "q07_coefficient_history.csv", index=False)

final_rows = [
    {"method": method, "seed": seed, **hist[-1]}
    for (method, seed), hist in coefficient_histories.items() if hist
]
final_df = pd.DataFrame(final_rows)
final_df.to_csv(OUT / "q07_final_coefficients.csv", index=False)

if not final_df.empty:
    final_df.groupby("method").mean(numeric_only=True).to_csv(OUT / "q07_coefficient_means.csv")
    final_df.groupby("method").std(numeric_only=True).to_csv(OUT / "q07_coefficient_stds.csv")

# Paired comparison against the best fixed activation (central question of Q7).
paired_df = run_paired_analysis(adult_df, PAIRED_GROUPS)
paired_df.to_csv(OUT / "q07_paired_vs_best_fixed.csv", index=False)
print("\nComparação pareada por semente (melhor fixo escolhido pela acurácia de validação):")
print_paired(paired_df)
print(
    f"\nNota: com {len(SEEDS)} sementes o teste de inversão de sinais tem p mínimo de "
    f"{2 / 2 ** len(SEEDS):.4f}; confie no IC e nas diferenças por semente."
)

# Do the coefficients depend on the initialization? Do they converge to the same values ​​across different seeds?
convergence_df = coefficient_convergence(history_df, final_df)
convergence_df.to_csv(OUT / "q07_coefficient_convergence.csv", index=False)
if not convergence_df.empty:
    print("\nConvergência dos coeficientes entre sementes:")
    print(convergence_df.round(4).to_string(index=False))


# Charts

def plot_mean_std(df, metric, ylabel, title, filename, group_col="name"):
    if metric not in df.columns or group_col not in df.columns:
        print(f"Coluna ausente para gráfico: {metric}")
        return
    data = df.dropna(subset=[metric])
    if data.empty:
        return
    summary = data.groupby(group_col)[metric].agg(["mean", "std"])
    summary["std"] = summary["std"].fillna(0)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(summary.index.astype(str), summary["mean"], yerr=summary["std"], capsize=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paired(table):
    """Diferenças pareadas (com IC) contra o melhor fixo."""
    if table.empty:
        return
    data = table[table["reference"] == table["best_fixed"]]
    for group, g in data.groupby("group"):
        g = g.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.5 * len(g) + 1.5)))
        y = np.arange(len(g))
        err = np.vstack([g["mean_diff"] - g["ci_low"], g["ci_high"] - g["mean_diff"]])
        ax.errorbar(g["mean_diff"], y, xerr=err, fmt="o", capsize=4)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(g["candidate"])
        ax.set_xlabel(f"Δ acurácia de teste vs. melhor fixo ({g['best_fixed'].iloc[0]}), IC 95%")
        ax.set_title(f"Pareado por semente — {group}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        safe = str(group).replace(" ", "_")
        fig.savefig(FIG / f"paired_vs_best_fixed_{safe}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


# Specific charts for the Adult -> SGNS transfer

def plot_transfer_bar(df, metric, ylabel, title, filename, higher_is_better=True):
    """Gráfico horizontal por método, com média e desvio-padrão."""
    required = {"method", metric}
    if not required.issubset(df.columns):
        print(f"Colunas ausentes para {filename}: {required - set(df.columns)}")
        return

    data = df[["method", metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
    if data.empty:
        print(f"Sem dados válidos para {filename}.")
        return

    summary = data.groupby("method")[metric].agg(["mean", "std", "count"])
    summary["std"] = summary["std"].fillna(0)
    summary = summary.sort_values("mean", ascending=not higher_is_better)

    height = max(5, 0.48 * len(summary))
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(summary.index, summary["mean"], xerr=summary["std"], capsize=4)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(row["mean"] + row["std"], i, f'  n={int(row["count"])}',
                va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_scatter(df, x_metric, x_label, title, filename):
    """Compara desempenho e custo, identificando cada método."""
    required = {"method", "score", x_metric}
    if not required.issubset(df.columns):
        print(f"Colunas ausentes para {filename}: {required - set(df.columns)}")
        return

    data = df[["method", "score", x_metric]].copy()
    for col in ("score", x_metric):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        print(f"Sem dados válidos para {filename}.")
        return

    summary = data.groupby("method").agg(
        score_mean=("score", "mean"),
        score_std=("score", "std"),
        cost_mean=(x_metric, "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(summary["cost_mean"], summary["score_mean"], s=65)
    for _, row in summary.iterrows():
        ax.annotate(row["method"], (row["cost_mean"], row["score_mean"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Silhouette score")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_coefficients(df):
    """Compara os coeficientes das misturas transferidas."""
    coefficient_cols = ["alpha_relu", "alpha_gelu", "alpha_silu"]
    required = {"method", *coefficient_cols}
    if not required.issubset(df.columns):
        print("Colunas ausentes para o gráfico de coeficientes:", required - set(df.columns))
        return

    data = df[["method", *coefficient_cols]].copy()
    for col in coefficient_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    summary = data.groupby("method")[coefficient_cols].mean()
    summary = summary.dropna(how="all")
    if summary.empty:
        print("Não há coeficientes válidos para plotar.")
        return

    summary = summary.rename(columns={
        "alpha_relu": "ReLU", "alpha_gelu": "GELU", "alpha_silu": "SiLU",
    })
    ax = summary.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_xlabel("Modelo treinado no Adult")
    ax.set_ylabel("Coeficiente transferido")
    ax.set_title("Coeficientes das misturas transferidas para SGNS")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Ativação")
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(FIG / "transfer_coefficients.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_before_after(df):
    """Compara silhouette antes e depois do treinamento, se disponível."""
    before_col, after_col = "silhouette_before", "silhouette_after"
    if not {before_col, after_col, "method"}.issubset(df.columns):
        print("Comparação antes/depois ignorada: silhouette_before ou silhouette_after não está disponível.")
        return

    data = df[["method", before_col, after_col]].copy()
    for col in (before_col, after_col):
        data[col] = pd.to_numeric(data[col], errors="coerce")

    summary = data.groupby("method")[[before_col, after_col]].mean()
    summary = summary.dropna(how="all")
    if summary.empty:
        return

    summary = summary.rename(columns={
        before_col: "Antes do treinamento",
        after_col: "Depois do treinamento",
    })
    ax = summary.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_xlabel("Método")
    ax.set_ylabel("Silhouette score")
    ax.set_title("SGNS: qualidade dos embeddings antes e depois do treinamento")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(FIG / "sgns_silhouette_before_after.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_histories(method):
    hist_by_seed = {
        seed: coefficient_histories.get((method, seed), [])
        for seed in SEEDS
    }
    hist_by_seed = {s: h for s, h in hist_by_seed.items() if h}
    columns = sorted({
        k for hist in hist_by_seed.values()
        for row in hist for k in row if k != "epoch"
    })
    for col in columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for seed, hist in hist_by_seed.items():
            valid = [r for r in hist if col in r]
            if valid:
                ax.plot([r["epoch"] for r in valid], [r[col] for r in valid], label=f"Seed {seed}")
        ax.set_xlabel("Época (0 = inicialização)")
        ax.set_ylabel(col)
        ax.set_title(f"{method}: evolução de {col}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / f"{method}_{col}.png".replace("/", "_"), dpi=220, bbox_inches="tight")
        plt.close(fig)


for method, _, track in ORIGINAL_CONFIGS:
    if track:
        plot_histories(method)
for method, mode, _, _ in EXTENDED_CONFIGS:
    if mode in ("global", "per_layer", "per_unit"):
        plot_histories(method)

plot_mean_std(adult_df, "score", "Score", "Adult: desempenho", "adult_score.png")
plot_mean_std(adult_df, "seconds", "Tempo (s)", "Adult: tempo de treinamento", "adult_time.png")
plot_mean_std(adult_df, "train_gflops", "GFLOPs", "Adult: FLOPs de treinamento", "adult_train_gflops.png")
plot_mean_std(adult_df, "n_params", "Parâmetros", "Adult: número de parâmetros", "adult_parameters.png")
plot_paired(paired_df)

# Pareto broken down by architecture — in-memory data

def plot_pareto_memory(results, architecture_label, architecture_tag):
    """
    Uses the already-calculated Result objects directly. 
    Does not read CSVs and does not re-run the experiments. 
    """

    if not results:
        print(f"No results for {architecture_label}.")
        return

    # Aggregates the seeds for each configuration.
    rows = []
    for name, group in itertools.groupby(
        sorted(results, key=lambda r: r.name),
        key=lambda r: r.name,
    ):
        group = list(group)

        rows.append({
            "name": name,
            "score": np.mean([r.score for r in group]),
            "score_std": (
                np.std([r.score for r in group], ddof=1)
                if len(group) > 1 else 0.0
            ),
            "train_flops": np.mean([
                r.train_flops for r in group
            ]),
            "infer_flops": np.mean([
                r.infer_flops for r in group
            ]),
        })

    summary = pd.DataFrame(rows)

    for cost_col, cost_tag, cost_label in [
        ("train_flops", "train", "FLOPs de treinamento"),
        ("infer_flops", "infer", "FLOPs de inferência"),
    ]:
        data = summary.copy()
        data[cost_col] = pd.to_numeric(
            data[cost_col], errors="coerce"
        )
        data["score"] = pd.to_numeric(
            data["score"], errors="coerce"
        )
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=[cost_col, "score"])

        # Zero is not a valid computational cost for this graph.
        data = data[data[cost_col] > 0]

        if data.empty:
            print(
                f"Gráfico ignorado: não há valores positivos válidos "
                f"para {cost_label} em {architecture_label}."
            )
            continue

        # Sorts by increasing cost and identifies non-dominated points
        data = data.sort_values(
            [cost_col, "score"],
            ascending=[True, False],
        )

        best_score = -np.inf
        frontier_rows = []

        for _, row in data.iterrows():
            if row["score"] > best_score:
                frontier_rows.append(row)
                best_score = row["score"]

        frontier = pd.DataFrame(frontier_rows)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(
            data[cost_col],
            data["score"],
            yerr=data["score_std"],
            fmt="o",
            capsize=3,
            label="Média ± desvio-padrão",
        )

        for _, row in data.iterrows():
            ax.annotate(
                row["name"],
                (row[cost_col], row["score"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax.plot(
            frontier[cost_col],
            frontier["score"],
            marker="o",
            linestyle="--",
            label="Fronteira de Pareto",
        )

        ax.set_xscale("log")
        ax.set_xlabel(f"{cost_label} (escala logarítmica)")
        ax.set_ylabel("Acurácia de teste")
        ax.set_title(
            f"{architecture_label}: acurácia versus {cost_label}"
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        filename = f"pareto_{architecture_tag}_{cost_tag}.png"
        path = FIG / filename
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        print(f"Gráfico salvo: {path}")


plot_pareto_memory(
    original_results,
    architecture_label="MLP — uma camada oculta",
    architecture_tag="mlp",
)

plot_pareto_memory(
    extended_results,
    architecture_label="TwoHiddenMLP — duas camadas ocultas",
    architecture_tag="twohidden",
)

# SGNS: transfer of frozen coefficients

def frozen_factory(coefficients):
    a, b, c = map(float, coefficients)

    class FrozenMixture:
        def __call__(self, x):
            xr, xg, xs = activation_triplet(x)
            return a * xr + b * xg + c * xs
    return lambda: FrozenMixture()


def original_normalized_coefficients(model):
    if not isinstance(model.activation, NormalizedLearnableActivation):
        raise TypeError("O modelo capturado não possui NormalizedLearnableActivation.")
    c = model.activation.coefficients()
    return c["pi_relu"], c["pi_gelu"], c["pi_silu"]


def extended_transfer_coefficients(model):
    if model.mode == "global":
        values = model.shared_act.coefficients()
        arrays = tuple(tensor_values(v) for v in values)
    elif model.mode in ("per_layer", "per_unit"):
        c1, c2 = model.act1.coefficients(), model.act2.coefficients()
        arrays = tuple(
            np.concatenate([tensor_values(c1[i]), tensor_values(c2[i])])
            for i in range(3)
        )
    else:
        raise ValueError(f"Modelo {model.mode} não tem mistura transferível.")
    return tuple(float(np.mean(v)) for v in arrays)


sgns_rows = []
transfer_coefficient_rows = []

for seed in SEEDS:
    sgns_methods = [
        ("SGNS-baseline", None),
        ("SGNS-ReLU", lambda: (lambda x: activation_triplet(x)[0])),
        ("SGNS-GELU", lambda: (lambda x: activation_triplet(x)[1])),
        ("SGNS-SiLU", lambda: (lambda x: activation_triplet(x)[2])),
    ]

    for method_name, factory in sgns_methods:
        result = run_sgns(
            **SGNS_CONFIG, seed=seed, name=method_name,
            activation_factory=factory,
        )
        row = result_to_row(result, "q07-SGNS")
        row["method"] = method_name
        sgns_rows.append(row)
        print(f"{method_name} | seed={seed} | silhouette={result.score:.5f} | time={result.seconds:.2f}s")
        del result
        gc.collect()

    transfer_sources = [
        ("Normalized", "original"),
        ("Global", "extended"),
        ("Per-layer", "extended"),
        ("Per-unit", "extended"),
        ("Global-normalized", "extended"),
        ("Per-layer-normalized", "extended"),
        ("Per-unit-normalized", "extended"),
    ]

    for method, source_type in transfer_sources:
        model = adult_models.get((method, seed))
        if model is None:
            print(f"AVISO: modelo Adult {method}, seed={seed} não foi capturado; transferência ignorada.")
            continue

        coeffs = (
            original_normalized_coefficients(model)
            if source_type == "original"
            else extended_transfer_coefficients(model)
        )


        transfer_coefficient_rows.append({
            "method": method,
            "seed": seed,
            "normalized": (
                method == "Normalized"
                or method.endswith("-normalized")
            ),
            "alpha_relu": coeffs[0],
            "alpha_gelu": coeffs[1],
            "alpha_silu": coeffs[2],
            "sum_coefficients": sum(coeffs),
        })

        method_name = f"SGNS-frozen-{method}"
        result = run_sgns(
            **SGNS_CONFIG, seed=seed, name=method_name,
            activation_factory=frozen_factory(coeffs),
        )
        row = result_to_row(result, "q07-SGNS")
        row.update({
            "method": method_name, "source_method": method,
            "alpha_relu": coeffs[0], "alpha_gelu": coeffs[1],
            "alpha_silu": coeffs[2],
        })
        sgns_rows.append(row)
        print(f"{method_name} | seed={seed} | silhouette={result.score:.5f} | time={result.seconds:.2f}s")
        del result
        gc.collect()

sgns_df = pd.DataFrame(sgns_rows)
sgns_df.to_csv(OUT / "q07_transfer_results.csv", index=False)

transfer_df = pd.DataFrame(transfer_coefficient_rows)
transfer_df.to_csv(OUT / "q07_transfer_coefficients.csv", index=False)

if not sgns_df.empty:
    cols = [c for c in ("score", "seconds", "train_gflops", "n_params") if c in sgns_df.columns]
    sgns_df.groupby("method")[cols].agg(["mean", "std"]).to_csv(OUT / "q07_transfer_summary.csv")
    plot_mean_std(sgns_df, "score", "Silhouette", "SGNS: comparação de métodos", "sgns_silhouette.png", "method")
    plot_mean_std(sgns_df, "seconds", "Tempo (s)", "SGNS: tempo de treinamento", "sgns_time.png", "method")
    plot_mean_std(sgns_df, "train_gflops", "GFLOPs", "SGNS: FLOPs de treinamento", "sgns_train_gflops.png", "method")

    plot_transfer_bar(sgns_df, metric="score", ylabel="Silhouette score",
                      title="Adult -> SGNS: comparação de desempenho",
                      filename="transfer_silhouette.png", higher_is_better=True)
    plot_transfer_bar(sgns_df, metric="seconds", ylabel="Tempo de treinamento (s)",
                      title="Adult -> SGNS: custo temporal",
                      filename="transfer_time.png", higher_is_better=False)
    plot_transfer_bar(sgns_df, metric="train_gflops", ylabel="GFLOPs de treinamento",
                      title="Adult -> SGNS: custo computacional",
                      filename="transfer_train_gflops.png", higher_is_better=False)
    plot_transfer_scatter(sgns_df, x_metric="seconds", x_label="Tempo de treinamento (s)",
                          title="SGNS: desempenho versus tempo",
                          filename="transfer_score_vs_time.png")
    plot_transfer_scatter(sgns_df, x_metric="train_gflops", x_label="GFLOPs de treinamento",
                          title="SGNS: desempenho versus custo computacional",
                          filename="transfer_score_vs_gflops.png")
    plot_transfer_before_after(sgns_df)

if not transfer_df.empty:
    plot_transfer_coefficients(transfer_df)


# Metadata

pd.DataFrame([{
    "seeds": ",".join(map(str, SEEDS)),
    "adult_epochs": EPOCHS_ADULT,
    "adult_lr": LR_ADULT,
    "sgns_config": str(SGNS_CONFIG),
    "extended_architecture": "n_features -> 64 -> 32 -> 2",
    "best_fixed_selection": "acurácia de validação (acc_val)",
    "baseline_check_ran": RUN_BASELINE_CHECK,
    "baseline_check_all_ok": bool(baseline_table["ok"].all()) if baseline_table is not None else None,
    "transfer_aggregation": "mean coefficients for per-layer/per-unit",
    "ram_gb_at_end": ram_gb(),
}]).to_csv(OUT / "q07_metadata.csv", index=False)

print("\nExperimentos concluídos.")
print("Resultados:", OUT)
print("Gráficos:", FIG)