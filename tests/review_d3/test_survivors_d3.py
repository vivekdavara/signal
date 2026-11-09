"""Tests for the mutants that survived day 3's review (the reviewer's mutation run at 3afe5de, 18 of 32 survived).

Each test's docstring names the mutant it kills. ``.scratch/mutate_d3.py`` re-applies them to today's code.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import redis

from signalrec import cli, movielens, prepare, synthetic
from signalrec import evaluate as ev
from signalrec import experiment_ranking as er
from signalrec import experiment_twotower as ext
from signalrec import features as fe
from signalrec import ranking as rk
from signalrec import retrieval as rt
from signalrec import twotower as tt
from signalrec.featurestore import FeatureStore, FeatureStoreError
from signalrec.prepare import Prepared

URL = os.environ.get("SIGNALREC_REDIS_URL", "redis://127.0.0.1:6383/0")
TT = tt.Config(dim=16, hidden=16, batch_size=64, lr=1e-2, epochs=3, max_history=5)
N = 30
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import bench_retrieval  # noqa: E402


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=150, n_items=160)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


@pytest.fixture(scope="module")
def client() -> redis.Redis:
    c = redis.Redis.from_url(URL)
    try:
        c.ping()
    except redis.ConnectionError as exc:
        if os.environ.get("SIGNALREC_REQUIRE_REDIS") == "1":
            pytest.fail(f"SIGNALREC_REQUIRE_REDIS=1 but no Redis at {URL}: {exc}")
        pytest.skip(f"no Redis at {URL}")
    return c


CATALOGUE = fe.Catalogue(("Drama", "War"), ((0,), (0, 1), (1,)), np.array([1990, 2000, fe.NO_YEAR], dtype=np.int64))


@pytest.fixture
def store(client: redis.Redis) -> Iterator[FeatureStore]:
    s = FeatureStore(client, CATALOGUE, namespace=f"test-{uuid.uuid4().hex}")
    yield s
    s.clear()


# --- features.py ----------------------------------------------------------------------------------------------


def test_validation_events_carry_the_validation_time(prepared: Prepared) -> None:
    """fe-val-times: validation events were stamped with the test time and nothing noticed."""
    events = fe.events(prepared, include_val=True)
    val = {(e.user, e.movie): e.time for e in events}
    for u in range(prepared.n_users):
        assert val[(u, int(prepared.val_items[u]))] == int(prepared.val_times[u])
    assert (prepared.val_times != prepared.test_times).any()


def test_popularity_covers_movies_nobody_liked(prepared: Prepared) -> None:
    """fe-popularity-minlength: without minlength the popularity array is short when the last movies have no
    positives, and indexing it by such a movie fails."""
    extra = replace(
        prepared,
        item_raw_ids=np.append(prepared.item_raw_ids, prepared.item_raw_ids.max() + 1),
        titles=(*prepared.titles, "Nobody Liked This (2001)"),
        genres=(*prepared.genres, ("Drama",)),
    )
    _, popularity = fe.offline(extra, include_val=True)
    assert len(popularity) == extra.n_items
    assert popularity[-1] == 0


# --- featurestore.py ------------------------------------------------------------------------------------------


def test_a_movie_without_a_year_adds_no_year(store: FeatureStore) -> None:
    """fs-no-year-counted: the script added -1 to year_sum and 1 to year_count for a movie with no year."""
    assert store.record(fe.Event(1, 2, 10))  # movie 2 has no year
    got = store.user(1)
    assert (got.year_sum, got.year_count, got.positives) == (0, 0, 1)


def test_clear_does_not_delete_a_namespace_that_extends_its_name(client: redis.Redis) -> None:
    """fs-clear-prefix: matching "<ns>*" instead of "<ns>:*" deleted the keys of every namespace starting with it."""
    base = f"test-{uuid.uuid4().hex}"
    short, longer = FeatureStore(client, CATALOGUE, base), FeatureStore(client, CATALOGUE, base + "x")
    try:
        short.record(fe.Event(1, 0, 1))
        longer.record(fe.Event(1, 1, 1))
        short.clear()
        assert longer.user(1).seen.tolist() == [1]
    finally:
        short.clear()
        longer.clear()


def test_a_genre_index_just_past_the_catalogue_is_an_error(store: FeatureStore) -> None:
    """fs-genre-bound: with <= the index equal to the genre count passed the check and raised IndexError instead."""
    store.client.hset(f"{store.namespace}:u:2:genres", str(len(CATALOGUE.genre_names)), "1")
    with pytest.raises(FeatureStoreError, match="outside the"):
        store.user(2)


def test_a_final_batch_of_one_event_is_applied(store: FeatureStore) -> None:
    """fs-last-batch-of-one: a replay whose last batch held a single event dropped it."""
    stats = store.replay([fe.Event(1, 0, 1), fe.Event(1, 1, 2), fe.Event(1, 2, 3)], batch=2)
    assert stats.applied == 3
    assert store.user(1).recent.tolist() == [0, 1, 2]


def test_a_read_is_one_snapshot_while_events_arrive(client: redis.Redis) -> None:
    """fs-read-not-transaction: without MULTI/EXEC another client's event can land between the four reads, so the
    recent list, the seen set and the positives count disagree."""
    n = 3000
    catalogue = fe.Catalogue(("Drama",), tuple((0,) for _ in range(n)), np.full(n, 1990, dtype=np.int64))
    namespace = f"test-{uuid.uuid4().hex}"
    writer = FeatureStore(redis.Redis.from_url(URL), catalogue, namespace)
    reader = FeatureStore(client, catalogue, namespace)
    done = threading.Event()

    def write() -> None:
        for movie in range(n):
            writer.record(fe.Event(1, movie, movie))
        done.set()

    thread = threading.Thread(target=write)
    thread.start()
    reads = 0
    try:
        deadline = time.monotonic() + 30
        while not done.is_set() and time.monotonic() < deadline:
            got = reader.user(1)
            assert len(got.seen) == got.positives == got.year_count == int(got.genre_counts.sum())
            assert len(got.recent) == min(got.positives, fe.RECENT)
            if got.positives:
                assert int(got.recent[-1]) == got.positives - 1  # movies arrive in order 0, 1, 2, ...
            reads += 1
    finally:
        thread.join(timeout=60)
        reader.clear()
    assert reads > 50  # the reads really overlapped the writes


# --- ranking.py -----------------------------------------------------------------------------------------------


class _Constant(rk.RankerWeights):
    def __init__(self) -> None:
        super().__init__(np.zeros(len(rk.FEATURES)), np.ones(len(rk.FEATURES)), ())

    def score(self, features: rk.Float64Array) -> rk.Float64Array:
        out: rk.Float64Array = np.zeros(len(features))
        return out


def test_rerank_keeps_the_retrieved_order_of_a_large_tie() -> None:
    """rk-unstable-rerank: NumPy's default sort is stable only below 17 elements, so a 64-way tie survived the
    4-candidate test."""
    candidates = np.arange(100, 164, dtype=np.int64)
    order = _Constant().rerank(candidates, np.zeros((64, len(rk.FEATURES))))
    assert order.tolist() == candidates.tolist()


def test_a_list_whose_held_out_movie_is_first_is_trained_on() -> None:
    """rk-usable-gt0: labels > 0 dropped every list whose held-out movie was retrieved first."""
    rng = np.random.default_rng(0)
    lists = rk.Lists(rng.normal(size=(20, 5, len(rk.FEATURES))), np.ones((20, 5), dtype=bool), np.zeros(20, np.int64))
    _, losses = rk.train(lists, rk.RankerConfig(hidden=0, epochs=2))
    assert len(losses) == 2


# --- experiment_ranking.py ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result(prepared: Prepared) -> dict[str, Any]:
    return er.run(prepared, TT, (0, 4), 3, N, ablate=False)


def test_validation_features_use_training_popularity_only(prepared: Prepared) -> None:
    """er-popularity-leak: validation lists' popularity counted the validation positives (the labels)."""
    lists = er.build_lists(prepared, TT, include_val=False, n=N).lists
    seq = tt.sequences(prepared, include_val=False)
    net, _ = tt.train(seq, tt.genre_matrix(prepared.genres)[0], TT)
    embeddings = tt.embed(net, seq, TT.max_history)
    users, popularity = fe.offline(prepared, include_val=False)
    context = rk.Context.build(fe.Catalogue.from_prepared(prepared), embeddings.items)
    retriever = rt.Retriever(embeddings.items)
    for u in range(0, prepared.n_users, 7):
        ids, scores = retriever.search(embeddings.users[u], N, users[u].seen)
        want = rk.candidate_features(users[u], ids, scores, popularity, context)
        assert np.array_equal(lists.features[u, : len(ids)], want), u


def test_the_test_share_retrieved_is_the_test_lists(prepared: Prepared, result: dict[str, Any]) -> None:
    """er-test-share-from-val: the test's share retrieved was the validation lists'."""
    test_lists = er.build_lists(prepared, TT, include_val=True, n=N).lists
    val_lists = er.build_lists(prepared, TT, include_val=False, n=N).lists
    assert result["test"]["held-out movie retrieved (share of users)"] == float((test_lists.labels >= 0).mean())
    assert (test_lists.labels >= 0).mean() != (val_lists.labels >= 0).mean()


def test_recall_at_n_counts_the_last_place_and_not_the_one_after() -> None:
    """er-recall-at-n-plus-1: Recall@n computed at n + 1 counted every movie that wasn't retrieved."""
    got = er.metrics(np.array([30, 31, 1], dtype=np.int64), 30)["recall@30"]
    assert got.tolist() == [1.0, 0.0, 1.0]


def test_the_selection_difference_is_chosen_minus_retrieved(result: dict[str, Any]) -> None:
    """er-selection-paired-swapped: the selection users' paired difference had its sign swapped."""
    best = max(c["ndcg@10"] for row in result["validation"]["rankers"] for c in row["curve"])
    retrieved = result["validation"]["retrieved order on the selection users"]["ndcg@10"]
    cell = result["chosen minus retrieved order, selection users (paired)"]["ndcg@10"]
    assert cell["value"] == pytest.approx(best - retrieved)
    assert best != retrieved
    low, high = cell["ci95"]
    assert low <= cell["value"] <= high


def test_tied_rankers_go_to_the_first_hidden_size_and_epoch(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    """er-ties-to-later: >= for > chose the last of equal candidates."""
    real = rk.ranks_after

    def same(weights: rk.RankerWeights | None, lists: rk.Lists) -> ev.IntArray:
        return real(None, lists)  # every ranker "re-ranks" to the retrieved order: all candidates tie

    monkeypatch.setattr(rk, "ranks_after", same)
    tied = er.run(prepared, TT, (0, 4), 3, N, ablate=False)
    assert tied["chosen"] == {"hidden": 0, "epochs": 1}


# --- experiment_twotower.py -----------------------------------------------------------------------------------


def test_close_candidates_name_their_history_length(prepared: Prepared, monkeypatch: pytest.MonkeyPatch) -> None:
    """tt-close-label-history: the indistinguishable candidates' labels lost their history length. Every evaluation
    returns the same ranks, so every candidate ties and is listed; the labels must name each (length, epoch)."""

    def same_ranks(model: Any, history: Any, targets: Any, split: str, *args: Any, **kwargs: Any) -> ev.Evaluation:
        n_users, n_items = history.shape
        return ev.Evaluation(
            "fake", split, np.full(n_users, 3, dtype=np.int64), n_items, np.zeros((n_users, 50), dtype=np.int64)
        )

    monkeypatch.setattr(ev, "evaluate", same_ranks)
    base = tt.Config(dim=16, hidden=16, batch_size=64, lr=1e-2)
    out = ext.run(prepared, ((0.1, True),), 2, (0,), ((5, 0.0),), base, histories=(2, 5))
    listed = out["validation candidates indistinguishable from the chosen one (paired ndcg@10 interval reaches 0)"]
    got = sorted((c["max_history"], c["epoch"]) for c in listed["candidates"])
    assert got == [(2, 2), (5, 1), (5, 2)]  # all but the chosen (history 2, epoch 1)


# --- scripts/bench_retrieval.py -------------------------------------------------------------------------------


def test_the_benchmarks_held_out_movie_is_the_test_movie(tmp_path: Path) -> None:
    """br-targets-val: the share "held out in the top n" was computed for the validation movie, which is in the
    history and excluded, so it was always 0."""
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "250", "--items", "200"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    source = tmp_path / "twotower.json"
    args = ["twotower", "synthetic", "--data-dir", str(tmp_path), "--grid", "0.1:on", "--histories", "5"]
    assert cli.main([*args, "--epochs", "1", "--seeds", "0", "--knn-grid", "5:0", "--quiet", "--out", str(source)]) == 0
    out = tmp_path / "retrieval.json"
    argv = ["synthetic", "--data-dir", str(tmp_path), "--from", str(source), "--n", "10", "--out", str(out)]
    assert bench_retrieval.main([*argv, "--synthetic-items", "0"]) == 0
    flat = json.loads(out.read_text())["real catalogue"]["indexes"][0]
    p = prepare.load(tmp_path / "processed" / "synthetic", "synthetic")
    config = cli._reported_config(source, "synthetic", p.manifest)
    seq = tt.sequences(p, include_val=True)
    net, _ = tt.train(seq, tt.genre_matrix(p.genres)[0], config)
    top = ev.top_k(tt.embed(net, seq, config.max_history).score(np.arange(p.n_users)), p.history(True), 10)
    want = float(np.mean([p.test_items[u] in top[u] for u in range(p.n_users)]))
    assert flat["held-out movie in the top 10 (share of users)"] == want
    assert want > 0


def test_a_read_goes_through_multi_exec(store: FeatureStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """fs-read-not-transaction, which the concurrent test above can't kill: Redis runs a client's four pipelined
    commands from one read buffer before serving the writer, so a plain pipeline behaves atomically in practice. The
    guarantee shouldn't rest on that, so this checks the read asks for MULTI/EXEC."""
    seen: list[bool] = []
    real = store.client.pipeline

    def spy(transaction: bool = True, shard_hint: Any = None) -> Any:
        seen.append(transaction)
        return real(transaction=transaction, shard_hint=shard_hint)

    monkeypatch.setattr(store.client, "pipeline", spy)
    store.user(1)
    assert seen == [True]
