"""Retrieval → ranking, by the protocol (DESIGN.md, "Day 3: FAISS, ranking, the feature store").

1. **Validation lists.** The reported two-tower configuration (from ``twotower.json``) trained on the training positives
   retrieves every user's best ``n`` unseen movies with the exact FAISS index, their training positives excluded. The
   label is the position of the user's validation movie in that list (absent if it wasn't retrieved). Features come
   from the training positives only (``features.offline(include_val=False)``).
2. **Choosing the ranker.** The validation users are split once, by a seeded permutation: 80% train the ranker, 20%
   choose it. Each hidden size trains for ``epochs`` epochs; the candidates are (hidden size, epoch) pairs and the
   choice is the highest NDCG@10 on the 20% (ties to the earlier hidden size, then epoch), also compared with the
   retrieved order on the same users (paired).
3. **Test.** The two-tower model retrained on training + validation (as in ``twotower.json``'s test run) retrieves the
   test lists, features from training + validation, and the chosen ranker re-orders them. Reported: the retrieved
   order and the re-ranked order on test, with intervals and the paired difference. The ranker never trains on a test
   label, and the movies it re-orders at test time were retrieved by a model that never saw them.

A held-out movie that wasn't retrieved counts as a miss at every cut-off (rank ``n + 1``), so Recall@K and NDCG@K are
the full-ranking metrics for K up to ``n``, and Recall@n is the ceiling the ranker works under.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from signalrec import evaluate as ev
from signalrec import experiment
from signalrec import features as fe
from signalrec import ranking as rk
from signalrec import retrieval as rt
from signalrec import twotower as tt
from signalrec.prepare import Prepared

PRIMARY = experiment.PRIMARY  # "ndcg@10"
PRIMARY_K = experiment.PRIMARY_K
SEED = experiment.SEED
N_CANDIDATES = 200
DEFAULT_HIDDEN = (0, 16, 64)
DEFAULT_EPOCHS = 20
SELECT_SHARE = 0.2
KS = (10, 20, 50)

Progress = Callable[[str], None]


@dataclass(frozen=True)
class Retrieved:
    lists: rk.Lists
    seconds: dict[str, float]


def build_lists(
    p: Prepared, config: tt.Config, include_val: bool, n: int = N_CANDIDATES, progress: Progress | None = None
) -> Retrieved:
    """Train the retriever, retrieve every user's ``n`` unseen candidates, compute their features and labels."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    seq = tt.sequences(p, include_val)
    net, _ = tt.train(seq, tt.genre_matrix(p.genres)[0], config)
    embeddings = tt.embed(net, seq, config.max_history)
    timings["train two-tower"] = time.perf_counter() - t0
    catalogue = fe.Catalogue.from_prepared(p)
    users, popularity = fe.offline(p, include_val)
    context = rk.Context.build(catalogue, embeddings.items)
    retriever = rt.Retriever(embeddings.items, rt.IndexSpec("flat"))
    targets = p.test_items if include_val else p.val_items
    width = min(n, p.n_items)
    feats = np.zeros((p.n_users, width, len(rk.FEATURES)), dtype=np.float64)
    mask = np.zeros((p.n_users, width), dtype=bool)
    labels = np.full(p.n_users, -1, dtype=np.int64)
    search = features = 0.0
    for u in range(p.n_users):
        t1 = time.perf_counter()
        ids, scores = retriever.search(embeddings.users[u], n, users[u].seen)
        t2 = time.perf_counter()
        feats[u, : len(ids)] = rk.candidate_features(users[u], ids, scores, popularity, context)
        search += t2 - t1
        features += time.perf_counter() - t2
        mask[u, : len(ids)] = True
        hit = np.flatnonzero(ids == targets[u])
        if len(hit):
            labels[u] = int(hit[0])
    timings["retrieve"] = search
    timings["features"] = features
    if progress is not None:
        split = "test" if include_val else "validation"
        progress(f"{split} lists: {int((labels >= 0).sum())} of {p.n_users} held-out movies retrieved in the top {n}")
    return Retrieved(rk.Lists(feats, mask, labels), timings)


def metrics(ranks: ev.IntArray, n: int) -> dict[str, ev.FloatArray]:
    """Per-user Recall@K and NDCG@K for each K in ``KS`` up to ``n``, and Recall@n.

    Only cut-offs up to ``n``: a movie that wasn't retrieved has rank ``n + 1``, which a larger cut-off would count as
    a hit (with 20 candidates, Recall@50 would have counted every miss).
    """
    out = {}
    for k in KS:
        if k <= n:
            out[f"recall@{k}"] = ev.recall_at(ranks, k)
            out[f"ndcg@{k}"] = ev.ndcg_at(ranks, k)
    out[f"recall@{n}"] = ev.recall_at(ranks, n)
    return out


def _with_intervals(per_user: dict[str, ev.FloatArray]) -> dict[str, Any]:
    return {
        name: {"value": float(values.mean()), "ci95": list(ev.bootstrap_ci(values, seed=SEED))}
        for name, values in per_user.items()
    }


def _paired(a: dict[str, ev.FloatArray], b: dict[str, ev.FloatArray]) -> dict[str, Any]:
    return {
        name: {
            "value": float(a[name].mean() - b[name].mean()),
            "ci95": list(ev.paired_bootstrap_ci(a[name], b[name], seed=SEED)),
        }
        for name in a
    }


def split_users(n_users: int, share: float = SELECT_SHARE, seed: int = SEED) -> tuple[ev.IntArray, ev.IntArray]:
    """(fit, select): a seeded permutation of the users, the first ``share`` of it to select with."""
    order = np.random.default_rng(seed).permutation(n_users).astype(np.int64)
    n_select = max(1, round(share * n_users))
    if n_select >= n_users:
        raise ValueError(f"{n_users} users can't be split into a training and a selection part")
    return np.sort(order[n_select:]), np.sort(order[:n_select])


def without(lists: rk.Lists, feature: int) -> rk.Lists:
    """The same lists with one feature constant (0): after standardisation it carries nothing."""
    features = lists.features.copy()
    features[:, :, feature] = 0.0
    return rk.Lists(features, lists.mask, lists.labels)


def feature_ablation(
    fit: rk.Lists,
    select: rk.Lists,
    hidden: int,
    epochs: int,
    seed: int,
    chosen: dict[str, ev.FloatArray],
    n: int,
) -> list[dict[str, Any]]:
    """Post hoc, on the selection users (validation): the chosen ranker's settings retrained with each feature removed,
    and the paired NDCG@10 of (without it - the chosen ranker). Nothing is chosen from it."""
    rows = []
    for i, name in enumerate(rk.FEATURES):
        weights, _ = rk.train(without(fit, i), rk.RankerConfig(hidden=hidden, epochs=epochs, seed=seed))
        per_user = metrics(rk.ranks_after(weights, without(select, i)), n)
        low, high = ev.paired_bootstrap_ci(per_user[PRIMARY], chosen[PRIMARY], seed=SEED)
        rows.append(
            {
                "without": name,
                PRIMARY: float(per_user[PRIMARY].mean()),
                f"minus the chosen ranker, {PRIMARY}": float(per_user[PRIMARY].mean() - chosen[PRIMARY].mean()),
                "paired ci95": [low, high],
            }
        )
    return rows


def _check_inputs(hidden: Sequence[int], epochs: int, n: int, seed: int) -> None:
    if not hidden:
        raise ValueError("no hidden sizes")
    if len(set(hidden)) != len(hidden):
        raise ValueError(f"hidden sizes repeat: {list(hidden)}")
    if type(n) is not int or n < PRIMARY_K:
        raise ValueError(
            f"the candidate count must be an integer >= {PRIMARY_K} (the choice is by {PRIMARY}), got {n!r}"
        )
    for h in hidden:
        rk.RankerConfig(hidden=h, epochs=epochs, seed=seed).check()


def run(
    p: Prepared,
    config: tt.Config,
    hidden: Sequence[int] = DEFAULT_HIDDEN,
    epochs: int = DEFAULT_EPOCHS,
    n: int = N_CANDIDATES,
    seed: int = 0,
    progress: Progress | None = None,
    ablate: bool = True,
) -> dict[str, Any]:
    _check_inputs(hidden, epochs, n, seed)
    config.check()
    val = build_lists(p, config, include_val=False, n=n, progress=progress)
    fit_users, select_users = split_users(p.n_users)
    fit, select = val.lists.subset(fit_users), val.lists.subset(select_users)
    retrieved_select = metrics(rk.ranks_after(None, select), n)

    rows: list[dict[str, Any]] = []
    best: tuple[float, int, int, rk.RankerWeights, dict[str, ev.FloatArray]] | None = None
    for h in hidden:
        snapshots: list[rk.RankerWeights] = []
        t0 = time.perf_counter()
        _, losses = rk.train(fit, rk.RankerConfig(hidden=h, epochs=epochs, seed=seed), snapshots)
        seconds = time.perf_counter() - t0
        curve = []
        for epoch, weights in enumerate(snapshots, start=1):
            per_user = metrics(rk.ranks_after(weights, select), n)
            value = float(per_user[PRIMARY].mean())
            curve.append({"epoch": epoch, "loss": losses[epoch - 1], PRIMARY: value})
            if best is None or value > best[0]:  # strictly: a tie keeps the earlier candidate
                best = (value, h, epoch, weights, per_user)
        rows.append({"hidden": h, "train_seconds": seconds, "curve": curve})
        if progress is not None:
            top = max(curve, key=lambda c: c[PRIMARY])
            progress(f"ranker hidden {h}: best selection {PRIMARY} {top[PRIMARY]:.4f} at epoch {top['epoch']}")
    if best is None:  # hidden is non-empty and epochs >= 1
        raise RuntimeError("no ranker trained")
    _, chosen_hidden, chosen_epoch, chosen, chosen_select = best
    ablation = feature_ablation(fit, select, chosen_hidden, chosen_epoch, seed, chosen_select, n) if ablate else None

    test = build_lists(p, config, include_val=True, n=n, progress=progress)
    retrieved_test = metrics(rk.ranks_after(None, test.lists), n)
    ranked_test = metrics(rk.ranks_after(chosen, test.lists), n)
    return {
        "protocol": (
            f"two-tower retrieves the top {n} unseen movies (FAISS flat); ranker trained on 80% of the validation "
            f"lists, (hidden, epochs) chosen on the other 20% by {PRIMARY}; test lists from the model retrained on "
            "training + validation"
        ),
        "users": p.n_users,
        "items": p.n_items,
        "candidates": n,
        "features": list(rk.FEATURES),
        "two-tower configuration": config.as_dict(),
        "validation": {
            "held-out movie retrieved (share of users)": float((val.lists.labels >= 0).mean()),
            "users": {"fit": len(fit_users), "select": len(select_users)},
            "fit lists with the held-out movie": int((fit.labels >= 0).sum()),
            "retrieved order on the selection users": {name: float(v.mean()) for name, v in retrieved_select.items()},
            "rankers": rows,
            "seconds": val.seconds,
        },
        "chosen": {"hidden": chosen_hidden, "epochs": chosen_epoch},
        "chosen minus retrieved order, selection users (paired)": _paired(chosen_select, retrieved_select),
        "feature ablation (selection users, post hoc)": ablation,
        "test": {
            "held-out movie retrieved (share of users)": float((test.lists.labels >= 0).mean()),
            "retrieved order": _with_intervals(retrieved_test),
            "re-ranked": _with_intervals(ranked_test),
            "seconds": test.seconds,
        },
        "re-ranked minus retrieved order (paired, test)": _paired(ranked_test, retrieved_test),
        "ranker weights": {
            "mean": chosen.mean.tolist(),
            "scale": chosen.scale.tolist(),
            "layers": [[w.tolist(), b.tolist()] for w, b in chosen.layers],
        },
        "bootstrap": {"resamples": 1000, "seed": SEED},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "faiss": rt.FAISS_VERSION,
            "machine": platform.machine(),
            "system": platform.system(),
        },
    }
