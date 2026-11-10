"""serving.py: the request path against the offline pipeline, the filter, the stages; export.py's pieces.

Needs Redis like test_featurestore.py (skips without it unless SIGNALREC_REQUIRE_REDIS=1).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import redis

from signalrec import export, movielens, prepare, synthetic
from signalrec import features as fe
from signalrec import ranking as rk
from signalrec import retrieval as rt
from signalrec import serving as sv
from signalrec import twotower as tt
from signalrec.featurestore import FeatureStore
from signalrec.prepare import Prepared, PrepareError
from signalrec.tower import Model

URL = os.environ.get("SIGNALREC_REDIS_URL", "redis://127.0.0.1:6383/0")
CONFIG = tt.Config(dim=16, hidden=32, epochs=3, batch_size=64, max_history=4)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> movielens.Dataset:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=80, n_items=120)
    return movielens.load("synthetic", raw)


@pytest.fixture(scope="module")
def prepared(dataset: movielens.Dataset) -> Prepared:
    return prepare.prepare(dataset, prepare.Params())


def random_ranker(seed: int = 5) -> rk.RankerWeights:
    rng = np.random.default_rng(seed)
    f = len(rk.FEATURES)
    return rk.RankerWeights(
        rng.normal(size=f),
        rng.uniform(0.5, 2.0, size=f),
        ((rng.normal(size=(8, f)), rng.normal(size=8)), (rng.normal(size=(1, 8)), rng.normal(size=1))),
    )


@pytest.fixture(scope="module")
def model(prepared: Prepared) -> Model:
    return export.build_model(prepared, CONFIG, random_ranker(), {"dataset": "synthetic"})


@pytest.fixture(scope="module")
def client() -> redis.Redis:
    c = redis.Redis.from_url(URL)
    try:
        c.ping()
    except redis.ConnectionError as exc:
        if os.environ.get("SIGNALREC_REQUIRE_REDIS") == "1":
            pytest.fail(f"SIGNALREC_REQUIRE_REDIS=1 but no Redis at {URL}: {exc}")
        pytest.skip(f"no Redis at {URL} (start one on 6383, or set SIGNALREC_REDIS_URL)")
    return c


@pytest.fixture(scope="module")
def loaded(client: redis.Redis, model: Model, prepared: Prepared, dataset: movielens.Dataset) -> Iterator[FeatureStore]:
    store = FeatureStore(client, model.catalogue, namespace=f"test-{uuid.uuid4().hex}")
    export.load_store(store, prepared, dataset.ratings)
    yield store
    store.clear()


@pytest.fixture
def fresh(client: redis.Redis, model: Model) -> Iterator[FeatureStore]:
    store = FeatureStore(client, model.catalogue, namespace=f"test-{uuid.uuid4().hex}")
    yield store
    store.clear()


def offline_lists(prepared: Prepared, model: Model, n: int) -> list[fe.IntArray]:
    """The protocol's test-time pipeline, offline: PyTorch user vectors, offline features, the positives excluded."""
    seq = tt.sequences(prepared, include_val=True)
    net, _ = tt.train(seq, tt.genre_matrix(prepared.genres)[0], CONFIG)
    emb = tt.embed(net, seq, CONFIG.max_history)
    np.testing.assert_array_equal(emb.items, model.items)  # the same training: deterministic for a seed
    users, popularity = fe.offline(prepared, include_val=True)
    context = rk.Context.build(model.catalogue, emb.items)
    retriever = rt.Retriever(emb.items, rt.IndexSpec("flat"))
    out = []
    for u in range(prepared.n_users):
        ids, scores = retriever.search(emb.users[u], n, users[u].seen)
        feats = rk.candidate_features(users[u], ids, scores, popularity, context)
        out.append(model.ranker.rerank(ids, feats))
    return out


def test_the_request_path_reproduces_the_offline_pipeline(
    loaded: FeatureStore, model: Model, prepared: Prepared
) -> None:
    rec = sv.Recommender(model, loaded, n_candidates=50)
    want = offline_lists(prepared, model, 50)
    for u in range(prepared.n_users):
        got = rec.recommend(u, k=50, exclude="liked")
        assert got.movies.tolist() == want[u].tolist(), u
        assert got.candidates == 50


def test_the_default_filter_removes_every_rated_movie(
    loaded: FeatureStore, model: Model, prepared: Prepared, dataset: movielens.Dataset
) -> None:
    rec = sv.Recommender(model, loaded, n_candidates=40)
    rated: dict[int, set[int]] = {}
    for e in fe.rated_events(prepared, dataset.ratings):
        rated.setdefault(e.user, set()).add(e.movie)
    users, _ = fe.offline(prepared, include_val=True)
    differ = 0
    for u in range(prepared.n_users):
        got = rec.recommend(u, k=40)
        removed = rated.get(u, set()) | set(users[u].seen.tolist())
        assert not removed & set(got.movies.tolist())
        assert len(got.movies) == 40
        differ += got.movies.tolist() != rec.recommend(u, k=40, exclude="liked").movies.tolist()
        # The test movie was rated after everything in the store: it can still be recommended.
        assert int(prepared.test_items[u]) not in removed
    assert differ > 0  # the synthetic users rated movies they didn't like, so the filters differ


def test_excluded_genres_never_appear_and_k_is_still_met(loaded: FeatureStore, model: Model) -> None:
    rec = sv.Recommender(model, loaded, n_candidates=30)
    names = model.catalogue.genre_names
    for genre in names[:3]:
        g = names.index(genre)
        for u in range(0, 80, 7):
            got = rec.recommend(u, k=10, exclude_genres=[genre])
            assert len(got.movies) == 10
            assert all(g not in model.catalogue.movie_genres[m] for m in got.movies.tolist())
    two = rec.recommend(3, k=10, exclude_genres=[names[0], names[1], names[0]])
    assert all(not {0, 1} & set(model.catalogue.movie_genres[m]) for m in two.movies.tolist())
    with pytest.raises(KeyError, match="unknown genre"):
        rec.recommend(3, exclude_genres=["Space Opera"])


def test_without_the_ranker_the_retrieved_order_is_kept(loaded: FeatureStore, model: Model) -> None:
    rec = sv.Recommender(model, loaded, n_candidates=25)
    features, rated = loaded.serving_view(9)
    excluded = np.unique(np.concatenate([rated, features.seen]))
    want, scores = rec.retriever.search(model.tower.vector(features.recent), 25, excluded)
    got = rec.recommend(9, k=10, rank=False)
    assert got.movies.tolist() == want[:10].tolist()
    np.testing.assert_array_equal(got.scores, scores[:10])
    ranked = rec.recommend(9, k=25)
    assert sorted(ranked.movies.tolist()) == sorted(want.tolist())  # the same candidates, re-ordered


def test_every_stage_is_timed(loaded: FeatureStore, model: Model) -> None:
    got = sv.Recommender(model, loaded).recommend(0)
    assert tuple(got.seconds) == sv.STAGES
    assert all(s >= 0 for s in got.seconds.values())
    assert got.total == pytest.approx(sum(got.seconds.values()))


def test_an_event_changes_the_next_recommendation(fresh: FeatureStore, model: Model) -> None:
    rec = sv.Recommender(model, fresh, n_candidates=30)
    first = rec.recommend(1, k=10)
    assert len(first.movies) == 10  # a user with no events still gets recommendations (the empty history's vector)
    top = int(first.movies[0])
    assert fresh.record(fe.Event(1, top, 100))
    after = rec.recommend(1, k=10)
    assert top not in after.movies.tolist()
    assert after.movies.tolist() != first.movies.tolist()
    low = int(after.movies[0])
    assert fresh.record_rating(fe.Event(1, low, 101))
    again = rec.recommend(1, k=10)
    assert low not in again.movies.tolist()
    # A rating that isn't a positive leaves the vector alone: the rest keep their order.
    assert [m for m in after.movies.tolist() if m != low] == again.movies.tolist()[:9]


def test_fewer_eligible_movies_than_k(fresh: FeatureStore, model: Model) -> None:
    rec = sv.Recommender(model, fresh, n_candidates=10)
    for m in range(model.n_items - 4):
        fresh.record_rating(fe.Event(2, m, m))
    got = rec.recommend(2, k=10)
    assert sorted(got.movies.tolist()) == list(range(model.n_items - 4, model.n_items))
    assert got.candidates == 4
    with pytest.raises(ValueError):
        rec.recommend(2, k=0)
    with pytest.raises(ValueError):
        rec.recommend(2, exclude="nothing")


def test_k_above_the_candidate_count_widens_retrieval(loaded: FeatureStore, model: Model) -> None:
    rec = sv.Recommender(model, loaded, n_candidates=10)
    got = rec.recommend(4, k=30)
    assert len(got.movies) == 30 and got.candidates == 30


def test_a_store_for_another_catalogue_is_refused(client: redis.Redis, model: Model) -> None:
    other = fe.Catalogue(model.catalogue.genre_names, model.catalogue.movie_genres[:-1], model.catalogue.years[:-1])
    with pytest.raises(ValueError, match="catalogue"):
        sv.Recommender(model, FeatureStore(client, other, namespace="unused"))


# --- export.py ----------------------------------------------------------------------------------------------------


def test_ranker_from_results_reads_the_chosen_weights(tmp_path: Path) -> None:
    weights = random_ranker()
    result = {
        "dataset": "synthetic",
        "two-tower configuration": CONFIG.as_dict(),
        "ranker weights": {
            "mean": weights.mean.tolist(),
            "scale": weights.scale.tolist(),
            "layers": [[w.tolist(), b.tolist()] for w, b in weights.layers],
        },
    }
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps(result))
    back = export.ranker_from_results(path, "synthetic", CONFIG)
    x = np.random.default_rng(0).normal(size=(5, len(rk.FEATURES)))
    np.testing.assert_array_equal(back.score(x), weights.score(x))
    with pytest.raises(PrepareError, match="not 'ml-1m'"):
        export.ranker_from_results(path, "ml-1m", CONFIG)
    with pytest.raises(PrepareError, match="another two-tower configuration"):
        export.ranker_from_results(path, "synthetic", tt.Config(max_history=3))
    path.write_text("{}")
    with pytest.raises(PrepareError, match="not a ranking results file"):
        export.ranker_from_results(path, "synthetic", CONFIG)


def test_load_store_is_idempotent(fresh: FeatureStore, prepared: Prepared, dataset: movielens.Dataset) -> None:
    positives, rated = export.load_store(fresh, prepared, dataset.ratings)
    assert positives.applied == positives.events
    assert 0 < rated.applied <= rated.events  # a positive's rating is already in the set: those are no-ops
    again_positives, again_rated = export.load_store(fresh, prepared, dataset.ratings)
    assert again_positives.applied == 0 and again_rated.applied == 0
    users, popularity = fe.offline(prepared, include_val=True)
    for u in range(prepared.n_users):
        assert fresh.user(u) == users[u]
    np.testing.assert_array_equal(fresh.popularity(), popularity)
