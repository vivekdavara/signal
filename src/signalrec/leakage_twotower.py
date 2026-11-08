"""What is the split's calendar-time leak worth to the two-tower model? (DESIGN.md, "Known leakage")

Day 1 measured it for popularity by recounting as of each user's test time. A learned model can't be retrained once per
user, so users are grouped into buckets by test time (equal numbers of users), and for each bucket a model is trained
only on the training and validation positives strictly before the bucket's **earliest** test time. That data is a
subset of the leaky model's (training + validation, at any time), so the two differ only in what happened after the
cut. It is also stricter than per-user as-of for every user but the bucket's first (it drops what happened between the
cut and their own test time), so it removes somewhat more than the leak; more buckets narrow that. Two more things
differ from a clean "the leak's worth" (review round 2): an early bucket's model sees fewer examples, so with the same
number of epochs it also takes fewer optimiser steps; and the configuration was chosen on the leaky validation run.

Each bucket's users are then ranked exactly as in the reported test run: user vectors from their own training +
validation history (the last 50 positives), that history excluded, ties to the smaller index. The "leaky" side is the
reported configuration trained on the whole training + validation history, as the two-tower test row is. The paired
difference over all users is what the leak is worth to the model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from signalrec import evaluate as ev
from signalrec import twotower as tt
from signalrec.prepare import Prepared

IntArray = npt.NDArray[np.int64]
SEED = 20261107
LEAKY = "trained on all training + validation (leaky)"
AS_OF = "trained as of the bucket's cut"
COMPARISON = "two-tower, leaky minus as of the cut (paired)"


def bucket_users(test_times: IntArray, buckets: int) -> list[IntArray]:
    """Users sorted by (test time, user index) and cut into ``buckets`` groups of equal size (the first ones one
    larger when it doesn't divide)."""
    if buckets < 1:
        raise ValueError("buckets must be at least 1")
    order = np.lexsort((np.arange(len(test_times)), test_times)).astype(np.int64)
    return [part for part in np.array_split(order, buckets) if len(part)]


def events_before(p: Prepared, cut: int) -> tt.Sequences:
    """Every user's training and validation positives with a time strictly before ``cut``, as sequences in (time,
    movie id) order.

    Not the test positives: the leaky model never trains on them, so the as-of model mustn't either, or it would know
    things the model it is compared with doesn't (review round 2; day 1's popularity count did include them).
    """
    users = np.concatenate([p.train_users, np.arange(p.n_users)]).astype(np.int64)
    items = np.concatenate([p.train_items, p.val_items]).astype(np.int64)
    times = np.concatenate([p.train_times, p.val_times]).astype(np.int64)
    keep = times < cut
    users, items, times = users[keep], items[keep], times[keep]
    order = np.lexsort((items, times, users))
    counts = np.bincount(users, minlength=p.n_users).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return tt.Sequences(items[order], starts)


def _ranks(net: tt.TwoTowerNet, p: Prepared, users: IntArray, history_seq: tt.Sequences, max_history: int) -> IntArray:
    embeddings = tt.embed(net, history_seq, max_history)
    history = p.history(include_val=True)[users]
    return ev.ranks(embeddings.score(users), history, p.test_items[users])


def run(p: Prepared, config: tt.Config, buckets: int, ks: tuple[int, ...] = (10, 50)) -> dict[str, Any]:
    """The reported configuration (``config``, with its epochs and seed) as of each bucket's cut, against itself
    trained on the whole training + validation history."""
    config.check()
    genres, _ = tt.genre_matrix(p.genres)
    test_seq = tt.sequences(p, include_val=True)
    leaky_net, _ = tt.train(test_seq, genres, config)
    leaky = _ranks(leaky_net, p, np.arange(p.n_users, dtype=np.int64), test_seq, config.max_history)

    as_of = np.empty(p.n_users, dtype=np.int64)
    rows = []
    for index, users in enumerate(bucket_users(p.test_times, buckets)):
        cut = int(p.test_times[users].min())
        seq = events_before(p, cut)
        n_examples = len(tt.example_positions(seq)[0])
        row: dict[str, Any] = {
            "bucket": index,
            "users": len(users),
            "cut (earliest test time)": datetime.fromtimestamp(cut, UTC).isoformat(),
            "latest test time": datetime.fromtimestamp(int(p.test_times[users].max()), UTC).isoformat(),
            "positives before the cut": len(seq.items),
            "training examples": n_examples,
        }
        if n_examples >= 2:
            net, _ = tt.train(seq, genres, config)
            as_of[users] = _ranks(net, p, users, test_seq, config.max_history)
            row["trained"] = True
        else:
            # Nothing to learn from yet: the untrained model (its initial weights) is what existed then.
            torch.manual_seed(config.seed)
            net = tt.TwoTowerNet(genres, config.dim, config.hidden)
            as_of[users] = _ranks(net, p, users, test_seq, config.max_history)
            row["trained"] = False
        for k in ks:
            row[f"ndcg@{k}"] = {
                "leaky": float(ev.ndcg_at(leaky[users], k).mean()),
                "as of the cut": float(ev.ndcg_at(as_of[users], k).mean()),
            }
        rows.append(row)

    comparison: dict[str, Any] = {}
    for k in ks:
        for metric, fn in (("recall", ev.recall_at), ("ndcg", ev.ndcg_at)):
            a, b = fn(leaky, k), fn(as_of, k)
            low, high = ev.paired_bootstrap_ci(a, b, seed=SEED)
            comparison[f"{metric}@{k}"] = {
                LEAKY: float(a.mean()),
                AS_OF: float(b.mean()),
                "difference": float(a.mean() - b.mean()),
                "difference ci95": [low, high],
            }
    return {
        "users": p.n_users,
        "configuration": config.as_dict(),
        "buckets": rows,
        COMPARISON: comparison,
        "bootstrap": {"resamples": 1000, "seed": SEED},
    }
