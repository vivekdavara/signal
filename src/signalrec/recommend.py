"""Recommendations for one user, by title: the qualitative check next to the metrics.

The model is fit on training + validation (as for test), so the list is what the test evaluation ranked; the user's
test movie is marked if it made the list.
"""

from __future__ import annotations

import numpy as np

from signalrec import evaluate as ev
from signalrec.baselines import ItemKNN, Popularity
from signalrec.prepare import Prepared


def user_index(p: Prepared, raw_user_id: int) -> int:
    at = int(np.searchsorted(p.user_raw_ids, raw_user_id))
    if at == p.n_users or p.user_raw_ids[at] != raw_user_id:
        raise KeyError(f"user {raw_user_id} is not in the prepared data (dropped by the k-core, or never rated)")
    return at


def recent_likes(p: Prepared, user: int, n: int) -> list[int]:
    """The user's last ``n`` history movies (training then validation), oldest first."""
    mine = [*p.train_items[p.train_users == user].tolist(), int(p.val_items[user])]
    return [int(i) for i in mine[-n:]]


def recommend(p: Prepared, model: Popularity | ItemKNN, user: int, k: int) -> list[int]:
    history = p.history(include_val=True)
    model.fit(history)
    top = ev.top_k(model.score(np.array([user], dtype=np.int64)), history[[user]], k)[0]
    return [int(i) for i in top if i >= 0]


def describe(p: Prepared, model: Popularity | ItemKNN, raw_user_id: int, k: int = 10, recent: int = 5) -> str:
    user = user_index(p, raw_user_id)
    lines = [f"user {raw_user_id}: last {recent} liked (oldest first)"]
    lines += [f"  {p.titles[i]}" for i in recent_likes(p, user, recent)]
    label = model.name if isinstance(model, Popularity) else f"{model.name} (k={model.k}, shrink={model.shrink:g})"
    lines.append(f"{label}: top {k}")
    test_item = int(p.test_items[user])
    for rank, item in enumerate(recommend(p, model, user, k), start=1):
        mark = "   <- the held-out test movie" if item == test_item else ""
        lines.append(f"  {rank:2d}. {p.titles[item]}{mark}")
    lines.append(f"held-out test movie: {p.titles[test_item]}")
    return "\n".join(lines)
