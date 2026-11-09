"""The two-tower model by the same protocol as the baselines (DESIGN.md, "Day 2: two-tower retriever").

1. **Validation.** Each grid configuration (temperature, logQ on or off, history length) trains on the training
   positives; after every epoch its exported embeddings rank every user's validation item. The configurations are the
   (temperature, logQ) pairs crossed with the history lengths, in that order (pairs outer). The candidates are
   (configuration, epoch) pairs and the choice is the highest validation NDCG@10 (ties to the earlier configuration,
   then the earlier epoch). Every configuration's best epoch gets the paired interval of (chosen - it) NDCG@10, as
   item-kNN's grid rows do. The history length became a grid dimension on day 3 (DESIGN.md, "Day 3"): day 2 fixed it
   at 50 and a post-hoc ablation found 5 far better on validation.
2. **Against item-kNN on validation.** Item-kNN is tuned on its own grid; "the two-tower model beats item-kNN" means the
   paired validation interval of (two-tower - item-kNN) NDCG@10 lies above zero, not just a higher mean.
3. **Test.** The chosen configuration is retrained on training + validation for the chosen number of epochs with each
   seed; the first seed's run is the reported one (with intervals and paired differences against both baselines, also
   refit on training + validation), and the others give the spread over seeds.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import scipy
import torch

from signalrec import evaluate as ev
from signalrec import experiment
from signalrec import twotower as tt
from signalrec.baselines import ItemKNN, Popularity
from signalrec.prepare import Prepared

PRIMARY = experiment.PRIMARY
PRIMARY_K = experiment.PRIMARY_K
SEED = experiment.SEED
DIAGNOSTIC = experiment.DIAGNOSTIC
# (temperature, logQ correction on?) pairs. Day 2's grid crossed each temperature with logQ on and off; every logQ-off
# configuration lost about 0.03 validation NDCG@10 with an interval excluding zero, so from day 3 the default keeps logQ
# on (a validation result, carried forward) and spends the time on the history length instead.
DEFAULT_GRID: tuple[tuple[float, bool], ...] = tuple((t, True) for t in (0.05, 0.1, 0.2))
# Each user's most recent n positives feed the user tower.
DEFAULT_HISTORIES: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
DEFAULT_EPOCHS = 20
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_BASE = tt.Config()

Progress = Callable[[str], None]


@dataclass(frozen=True)
class Candidate:
    config: tt.Config
    epoch: int
    summary: dict[str, float]
    per_user: ev.FloatArray  # validation NDCG@10 per user


def _summary_row(summary: dict[str, float]) -> dict[str, float]:
    return {name: summary[name] for name in ("recall@10", "ndcg@10", "recall@50", "ndcg@50", "coverage@10")}


def _config_label(config: tt.Config) -> dict[str, Any]:
    return {"temperature": config.temperature, "logq": config.logq, "max_history": config.max_history}


class _Validator:
    """Called after every epoch: scores validation, appends to the learning curve, keeps the best epoch."""

    def __init__(self, p: Prepared, seq: tt.Sequences, config: tt.Config, progress: Progress | None) -> None:
        self.p = p
        self.seq = seq
        self.history = p.history(include_val=False)
        self.config = config
        self.progress = progress
        self.curve: list[dict[str, Any]] = []
        self.per_user: list[ev.FloatArray] = []  # every epoch's per-user validation NDCG@10, in curve order
        self.best: Candidate | None = None

    def __call__(self, stat: tt.EpochStats, net: tt.TwoTowerNet) -> None:
        t0 = time.perf_counter()
        embeddings = tt.embed(net, self.seq, self.config.max_history)
        evaluation = ev.evaluate(embeddings, self.history, self.p.val_items, "val")
        summary = evaluation.summary()
        self.curve.append(
            {
                "epoch": stat.epoch,
                "loss": stat.loss,
                **_summary_row(summary),
                "train_seconds": stat.seconds,
                "eval_seconds": time.perf_counter() - t0,
            }
        )
        per_user = ev.ndcg_at(evaluation.ranks, PRIMARY_K)
        self.per_user.append(per_user)
        # Strictly greater: a tie keeps the earlier epoch.
        if self.best is None or summary[PRIMARY] > self.best.summary[PRIMARY]:
            self.best = Candidate(self.config, stat.epoch, summary, per_user)
        if self.progress is not None:
            self.progress(
                f"temperature {self.config.temperature:g} logq {self.config.logq} history {self.config.max_history}: "
                f"epoch {stat.epoch} "
                f"loss {stat.loss:.4f} val {PRIMARY} {summary[PRIMARY]:.4f}"
            )


def tune(
    p: Prepared,
    grid: Sequence[tuple[float, bool]],
    epochs: int,
    base: tt.Config = DEFAULT_BASE,
    progress: Progress | None = None,
    histories: Sequence[int] = DEFAULT_HISTORIES,
) -> tuple[list[dict[str, Any]], list[Candidate], list[list[ev.FloatArray]]]:
    """Train every configuration (``grid`` crossed with ``histories``) on the training positives, scoring validation
    after every epoch.

    Returns one row per configuration (its learning curve and best epoch), each configuration's best candidate, and
    every epoch's per-user validation NDCG@10 (one list per configuration).
    """
    seq = tt.sequences(p, include_val=False)
    genres, _ = tt.genre_matrix(p.genres)
    rows: list[dict[str, Any]] = []
    bests: list[Candidate] = []
    per_user: list[list[ev.FloatArray]] = []
    for temperature, logq, max_history in configurations(grid, histories):
        config = replace(base, temperature=temperature, logq=logq, epochs=epochs, max_history=max_history)
        validator = _Validator(p, seq, config, progress)
        tt.train(seq, genres, config, validator)
        if validator.best is None:  # train calls back once per epoch and epochs >= 1
            raise RuntimeError("training ran no epochs")
        rows.append({**_config_label(config), "best epoch": validator.best.epoch, "curve": validator.curve})
        bests.append(validator.best)
        per_user.append(validator.per_user)
    return rows, bests, per_user


def configurations(grid: Sequence[tuple[float, bool]], histories: Sequence[int]) -> list[tuple[float, bool, int]]:
    """The tuning order: each (temperature, logQ) pair with every history length, pairs outer."""
    return [(temperature, logq, n) for temperature, logq in grid for n in histories]


def _check_inputs(
    grid: Sequence[tuple[float, bool]],
    epochs: int,
    seeds: Sequence[int],
    knn_grid: Sequence[tuple[int, float]],
    base: tt.Config,
    histories: Sequence[int],
) -> None:
    """Everything that could fail later, checked before the first minute of training (review round 1: an empty
    item-kNN grid, a repeated configuration and an unusable later seed all failed only after tuning)."""
    if not grid:
        raise ValueError("the grid is empty")
    if len(set(grid)) != len(grid):
        raise ValueError(f"the grid repeats a configuration: {list(grid)}")
    if not histories:
        raise ValueError("no history lengths")
    if len(set(histories)) != len(histories):
        raise ValueError(f"history lengths repeat: {list(histories)}")
    if not seeds:
        raise ValueError("no seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds repeat: {list(seeds)}")
    if not knn_grid:
        raise ValueError("the item-kNN grid is empty")
    for temperature, logq, max_history in configurations(grid, histories):
        for seed in seeds:
            replace(base, temperature=temperature, logq=logq, epochs=epochs, max_history=max_history, seed=seed).check()
    for k, shrink in knn_grid:
        ItemKNN(k=k, shrink=shrink)  # its constructor checks k and shrink


def _paired(a: ev.Evaluation, b: ev.Evaluation) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ev.KS:
        for metric, fn in (("recall", ev.recall_at), ("ndcg", ev.ndcg_at)):
            x, y = fn(a.ranks, k), fn(b.ranks, k)
            low, high = ev.paired_bootstrap_ci(x, y, seed=SEED)
            out[f"{metric}@{k}"] = {"value": float(x.mean() - y.mean()), "ci95": [low, high]}
    return out


def run(
    p: Prepared,
    grid: Sequence[tuple[float, bool]] = DEFAULT_GRID,
    epochs: int = DEFAULT_EPOCHS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    knn_grid: Sequence[tuple[int, float]] = experiment.DEFAULT_GRID,
    base: tt.Config = DEFAULT_BASE,
    progress: Progress | None = None,
    *,
    histories: Sequence[int] = DEFAULT_HISTORIES,
) -> dict[str, Any]:
    _check_inputs(grid, epochs, seeds, knn_grid, base, histories)
    base = replace(base, seed=seeds[0])
    rows, bests, per_epoch = tune(p, grid, epochs, base, progress, histories)
    chosen_index = max(range(len(bests)), key=lambda i: bests[i].summary[PRIMARY])
    chosen = bests[chosen_index]
    for row, candidate in zip(rows, bests, strict=True):
        row["best"] = _summary_row(candidate.summary)
        low, high = ev.paired_bootstrap_ci(chosen.per_user, candidate.per_user, seed=SEED)
        row[DIAGNOSTIC] = [low, high]
    indistinguishable = [
        _config_label(c.config)
        for i, (c, row) in enumerate(zip(bests, rows, strict=True))
        if i != chosen_index and row[DIAGNOSTIC][0] <= 0
    ]
    # Every candidate, not just each configuration's best epoch (review round 1: DESIGN asks for "every candidate").
    close_candidates = []
    for i, (row, values) in enumerate(zip(rows, per_epoch, strict=True)):
        for point, per_user in zip(row["curve"], values, strict=True):
            low, high = ev.paired_bootstrap_ci(chosen.per_user, per_user, seed=SEED)
            point[DIAGNOSTIC] = [low, high]
            if low <= 0 and not (i == chosen_index and point["epoch"] == chosen.epoch):
                label = {
                    "temperature": row["temperature"],
                    "logq": row["logq"],
                    "max_history": row["max_history"],
                    "epoch": point["epoch"],
                }
                close_candidates.append(label)

    knn = experiment.tune_item_knn(p, knn_grid)
    knn_best = knn.best
    low, high = ev.paired_bootstrap_ci(chosen.per_user, knn.per_user[knn.chosen], seed=SEED)
    versus_knn_val = {
        "value": float(chosen.per_user.mean() - knn.per_user[knn.chosen].mean()),
        "ci95": [low, high],
        "two-tower beats item-knn (interval above 0)": low > 0,
    }

    # Test: refit everything on training + validation.
    test_history = p.history(include_val=True)
    test_seq = tt.sequences(p, include_val=True)
    genres, genre_names = tt.genre_matrix(p.genres)
    final = replace(chosen.config, epochs=chosen.epoch)
    seed_runs: list[dict[str, Any]] = []
    evaluations: list[tuple[ev.Evaluation, float]] = []
    for seed in seeds:
        t0 = time.perf_counter()
        net, stats = tt.train(test_seq, genres, replace(final, seed=seed))
        seconds = time.perf_counter() - t0
        evaluation = ev.evaluate(tt.embed(net, test_seq, final.max_history), test_history, p.test_items, "test")
        seed_runs.append({"seed": seed, **_summary_row(evaluation.summary()), "final loss": stats[-1].loss})
        evaluations.append((evaluation, seconds))
        if progress is not None:
            progress(f"test seed {seed}: {PRIMARY} {evaluation.summary()[PRIMARY]:.4f} ({seconds:.1f} s)")
    reported, reported_seconds = evaluations[0]  # the first seed is the one validation chose with
    popularity = Popularity()
    popularity.fit(test_history)
    pop_eval = ev.evaluate(popularity, test_history, p.test_items, "test")
    knn_model = ItemKNN(k=int(knn_best["k"]), shrink=float(knn_best["shrink"]))
    knn_model.fit(test_history)
    knn_eval = ev.evaluate(knn_model, test_history, p.test_items, "test")
    primary_by_seed = [r[PRIMARY] for r in seed_runs]

    return {
        "protocol": (
            "leave-last-out; (temperature, logQ, history length, epochs) chosen on validation by "
            + PRIMARY
            + "; retrained on train+validation; test"
        ),
        "users": p.n_users,
        "items": p.n_items,
        "genres": list(genre_names),
        "examples": {
            "validation runs": len(tt.example_positions(tt.sequences(p, include_val=False))[0]),
            "test runs": len(tt.example_positions(test_seq)[0]),
        },
        "fixed": {
            k: v for k, v in base.as_dict().items() if k not in ("temperature", "logq", "max_history", "epochs", "seed")
        },
        "grid": {"temperature, logq": [list(pair) for pair in grid], "max_history": list(histories)},
        "validation": {
            "two-tower": rows,
            "item-knn": {"k": knn_best["k"], "shrink": knn_best["shrink"], **_summary_row(knn_best)},
        },
        "chosen": {"two-tower": {**_config_label(chosen.config), "epochs": chosen.epoch}},
        "validation configurations indistinguishable from the chosen one (paired ndcg@10 interval reaches 0)": (
            indistinguishable
        ),
        "validation candidates indistinguishable from the chosen one (paired ndcg@10 interval reaches 0)": {
            "of": sum(len(row["curve"]) for row in rows) - 1,
            "count": len(close_candidates),
            "candidates": close_candidates,
        },
        "two-tower minus item-knn, validation ndcg@10 (paired)": versus_knn_val,
        "test": {
            "two-tower": {**experiment._with_intervals(reported), "seed": seeds[0], "train_seconds": reported_seconds},
            "item-knn": {"k": knn_best["k"], "shrink": knn_best["shrink"], **experiment._with_intervals(knn_eval)},
            "popularity": experiment._with_intervals(pop_eval),
        },
        "test, two-tower by seed": {
            "runs": seed_runs,
            f"{PRIMARY} mean": float(np.mean(primary_by_seed)),
            f"{PRIMARY} min": float(np.min(primary_by_seed)),
            f"{PRIMARY} max": float(np.max(primary_by_seed)),
        },
        "two-tower minus item-knn (paired, test)": _paired(reported, knn_eval),
        "two-tower minus popularity (paired, test)": _paired(reported, pop_eval),
        "bootstrap": {"resamples": 1000, "seed": SEED},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "torch threads": torch.get_num_threads(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
    }


def table(result: dict[str, Any]) -> str:
    """The test results as a Markdown table, the two-tower row last."""
    lines = [
        "| Model | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Coverage@10 |",
        "|---|---|---|---|---|---|",
    ]
    chosen = result["chosen"]["two-tower"]
    knn = result["test"]["item-knn"]
    labels = {
        "popularity": "popularity",
        "item-knn": f"item-kNN (k={knn['k']}, shrink={knn['shrink']:g})",
        "two-tower": (
            f"two-tower (τ={chosen['temperature']:g}, logQ {'on' if chosen['logq'] else 'off'}, "
            f"history {chosen['max_history']}, {chosen['epochs']} epochs, seed {result['test']['two-tower']['seed']})"
        ),
    }
    for name, label in labels.items():
        r = result["test"][name]

        def cell(metric: str, r: dict[str, Any] = r) -> str:
            value = r[metric]["value"]
            if "ci95" not in r[metric]:
                return f"{value:.3f}"
            low, high = r[metric]["ci95"]
            return f"{value:.4f} [{low:.4f}, {high:.4f}]"

        lines.append(
            f"| {label} | {cell('recall@10')} | {cell('ndcg@10')} | {cell('recall@50')} | {cell('ndcg@50')} "
            f"| {cell('coverage@10')} |"
        )
    return "\n".join(lines)
