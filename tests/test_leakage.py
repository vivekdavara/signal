"""The leakage measurements against direct computations."""

from __future__ import annotations

import numpy as np
import pytest

from signalrec import evaluate as ev
from signalrec import leakage, movielens, prepare, synthetic
from signalrec.prepare import Prepared


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=120, n_items=150, seed=3)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


def test_popularity_as_of_matches_a_direct_count_per_user(prepared: Prepared) -> None:
    items = np.concatenate([prepared.train_items, prepared.val_items, prepared.test_items])
    times = np.concatenate([prepared.train_times, prepared.val_times, prepared.test_times])
    history = prepared.history(include_val=True)
    want = []
    for user in range(prepared.n_users):
        counts = np.bincount(items[times < prepared.test_times[user]], minlength=prepared.n_items).astype(float)
        want.append(int(ev.ranks(counts[None, :], history[[user]], prepared.test_items[[user]])[0]))
    assert leakage.popularity_as_of(prepared).tolist() == want


def test_future_share_by_hand(prepared: Prepared) -> None:
    share = leakage.future_share(prepared)
    for user in (0, 5, 17):
        later = int((prepared.train_times > prepared.test_times[user]).sum())
        assert share[user] == later / len(prepared.train_times)


def test_events_at_the_same_second_are_not_in_the_past() -> None:
    # User 0's test item (movie 2) is at t=10, the second users 1 and 2 liked movie 3. Strictly before t=10 only movies
    # 0 and 1 have likes, so movies 2 and 3 tie at 0 and movie 2 ranks 1st among user 0's candidates {2, 3}. Counting
    # t <= 10 would give movie 3 two likes and movie 2 one (user 0's own test item), and rank movie 2 2nd.
    p = Prepared(
        train_users=np.array([0, 1, 2], dtype=np.int64),
        train_items=np.array([0, 3, 3], dtype=np.int64),
        train_times=np.array([1, 10, 10], dtype=np.int64),
        val_items=np.array([1, 0, 0], dtype=np.int64),
        val_times=np.array([2, 11, 11], dtype=np.int64),
        test_items=np.array([2, 1, 2], dtype=np.int64),
        test_times=np.array([10, 12, 13], dtype=np.int64),
        user_raw_ids=np.array([1, 2, 3], dtype=np.int64),
        item_raw_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        titles=("a", "b", "c", "d"),
        genres=((), (), (), ()),
    )
    # User 1 at t=12: counts 0:3, 1:1, 2:1, 3:2; candidates {1, 2} tie at 1, movie 1 first. User 2 at t=13: movie 1
    # now has 2 likes (user 1's test item at t=12) against movie 2's 1, so movie 2 ranks 2nd.
    assert leakage.popularity_as_of(p).tolist() == [1, 1, 2]
    assert leakage.future_share(p).tolist() == [0.0, 0.0, 0.0]


def test_the_report_names_its_numbers(prepared: Prepared) -> None:
    report = leakage.run(prepared)
    cell = report["popularity, global minus as of test time (paired)"]["ndcg@10"]
    assert cell["difference"] == pytest.approx(cell["global (leaky)"] - cell["as of test time"])
    low, high = cell["difference ci95"]
    assert low <= cell["difference"] <= high
    assert 0 <= report["training positives later than the user's test item"]["mean share"] <= 1
