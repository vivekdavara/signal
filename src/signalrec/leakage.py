"""How much does the per-user split leak calendar time? (DESIGN.md, "Known leakage")

Leave-last-out holds out each user's last positive, but other users' training positives can be later in calendar time
than that test item, so a model fit on the training data has seen "the future". Two measurements:

- ``future_share``: for each user, the share of all training positives that are later than their test item.
- ``popularity_as_of``: popularity counted only from positives strictly before each user's test time (every user's
  train, validation and test positives that had happened by then), against the usual popularity fit on the whole
  training + validation history. Both exclude the same history and rank the same test items, so the paired difference
  is what the leak is worth to popularity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from signalrec import evaluate as ev
from signalrec.baselines import Popularity
from signalrec.prepare import Prepared

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SEED = 20261107


def future_share(p: Prepared) -> FloatArray:
    """Per user: the share of all training positives with a time later than the user's test time."""
    times = np.sort(p.train_times)
    later = len(times) - np.searchsorted(times, p.test_times, side="right")
    share: FloatArray = later / len(times)
    return share


def popularity_as_of(p: Prepared) -> IntArray:
    """The test item's rank under popularity counted from positives strictly before each user's test time.

    One sweep in time order: users sorted by test time, events added while they are earlier than the user's time.
    History (training + validation) is excluded as everywhere else; ties go to the smaller index.
    """
    users = np.arange(p.n_users, dtype=np.int64)
    event_items = np.concatenate([p.train_items, p.val_items, p.test_items])
    event_times = np.concatenate([p.train_times, p.val_times, p.test_times])
    order = np.argsort(event_times, kind="stable")
    event_items, event_times = event_items[order], event_times[order]
    history = p.history(include_val=True)
    counts = np.zeros(p.n_items, dtype=np.float64)
    ranks = np.empty(p.n_users, dtype=np.int64)
    added = 0
    for user in users[np.argsort(p.test_times, kind="stable")]:
        cutoff = p.test_times[user]
        upto = int(np.searchsorted(event_times, cutoff, side="left"))
        if upto > added:
            np.add.at(counts, event_items[added:upto], 1.0)
            added = upto
        ranks[user] = ev.ranks(counts[None, :], history[[user]], p.test_items[[user]])[0]
    return ranks


def run(p: Prepared, ks: tuple[int, ...] = (10, 50)) -> dict[str, Any]:
    share = future_share(p)
    history = p.history(include_val=True)
    model = Popularity()
    model.fit(history)
    leaky = ev.evaluate(model, history, p.test_items, "test").ranks
    as_of = popularity_as_of(p)
    comparison: dict[str, Any] = {}
    for k in ks:
        for metric, fn in (("recall", ev.recall_at), ("ndcg", ev.ndcg_at)):
            a, b = fn(leaky, k), fn(as_of, k)
            low, high = ev.paired_bootstrap_ci(a, b, seed=SEED)
            comparison[f"{metric}@{k}"] = {
                "global (leaky)": float(a.mean()),
                "as of test time": float(b.mean()),
                "difference": float(a.mean() - b.mean()),
                "difference ci95": [low, high],
            }
    return {
        "users": p.n_users,
        "training positives later than the user's test item": {
            "mean share": float(share.mean()),
            "median share": float(np.median(share)),
            "users with any": float((share > 0).mean()),
            "users with over half": float((share > 0.5).mean()),
        },
        "popularity, global minus as of test time (paired)": comparison,
        "bootstrap": {"resamples": 1000, "seed": SEED},
    }
