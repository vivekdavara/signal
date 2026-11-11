"""serving.py: the request path against the offline pipeline, the filter, the stages; export.py's pieces.

Needs Redis (the fixtures are in conftest.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import redis

from signalrec import export, movielens
from signalrec import features as fe
from signalrec import ranking as rk
from signalrec import retrieval as rt
from signalrec import serving as sv
from signalrec import twotower as tt
from signalrec.featurestore import FeatureStore
from signalrec.prepare import Prepared, PrepareError
from signalrec.tower import Model
from tests.serving.conftest import CONFIG, random_ranker


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
    assert len(first.movies) == 10 and first.source == "popularity"  # no events yet: the most popular (day 5)
    top = int(first.movies[0])
    assert fresh.record(fe.Event(1, top, 100))
    after = rec.recommend(1, k=10)
    assert after.source == "two-tower"
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


# --- users with no positives (day 5) -----------------------------------------------------------------------------


def popular_order(popularity: fe.IntArray, excluded: set[int]) -> list[int]:
    """Most popular first, ties to the smaller index, by brute force."""
    movies = [m for m in range(len(popularity)) if m not in excluded]
    return sorted(movies, key=lambda m: (-int(popularity[m]), m))


def test_a_user_with_no_positives_gets_the_most_popular_movies(
    loaded: FeatureStore, model: Model, prepared: Prepared
) -> None:
    rec = sv.Recommender(model, loaded)
    popularity = loaded.popularity()
    _, offline_popularity = fe.offline(prepared, include_val=True)
    np.testing.assert_array_equal(popularity, offline_popularity)
    new_user = prepared.n_users + 3  # nothing stored under it
    got = rec.recommend(new_user, k=15)
    assert got.source == "popularity"
    assert got.movies.tolist() == popular_order(popularity, set())[:15]
    np.testing.assert_array_equal(got.scores, popularity[got.movies].astype(np.float64))
    assert got.candidates == min(sv.N_CANDIDATES, model.n_items)
    assert tuple(got.seconds) == sv.STAGES and all(s >= 0 for s in got.seconds.values())
    assert rec.recommend(None, k=15).movies.tolist() == got.movies.tolist()


def test_the_popular_list_excludes_rated_movies_and_genres(fresh: FeatureStore, model: Model) -> None:
    rec = sv.Recommender(model, fresh)
    for user, movie in [(10, 3), (11, 3), (12, 3), (10, 7), (11, 7), (10, 1)]:
        assert fresh.record(fe.Event(user, movie, 0))
    assert fresh.record_rating(fe.Event(20, 7, 0))  # user 20 rated movie 7 below the threshold: no positives
    genre = model.catalogue.genre_names[model.catalogue.movie_genres[3][0]]
    in_genre = {
        m for m, genres in enumerate(model.catalogue.movie_genres) if model.catalogue.movie_genres[3][0] in genres
    }
    got = rec.recommend(20, k=12, exclude_genres=[genre])
    assert got.source == "popularity"
    assert got.movies.tolist() == popular_order(fresh.popularity(), {7} | in_genre)[:12]
    assert 7 not in got.movies.tolist() and not in_genre & set(got.movies.tolist())


def test_the_first_positive_switches_to_the_two_tower_model(fresh: FeatureStore, model: Model) -> None:
    rec = sv.Recommender(model, fresh)
    assert rec.recommend(30).source == "popularity"
    fresh.record(fe.Event(30, 5, 0))
    got = rec.recommend(30)
    assert got.source == "two-tower" and 5 not in got.movies.tolist()
    np.testing.assert_array_equal(
        got.movies, sv.Recommender(model, fresh).recommend(30).movies
    )  # deterministic: the same store, the same answer


def test_nobody_reads_nothing_from_the_store(
    fresh: FeatureStore, model: Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = sv.Recommender(model, fresh)

    def boom(user: int) -> None:
        raise AssertionError("serving_view called for a user the store has never seen")

    monkeypatch.setattr(fresh, "serving_view", boom)
    assert rec.recommend(None, k=5).source == "popularity"


def test_the_popular_list_is_read_once_per_ttl(
    fresh: FeatureStore, model: Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = []
    real = fresh.popularity

    def counting() -> fe.IntArray:
        reads.append(1)
        return real()

    monkeypatch.setattr(fresh, "popularity", counting)
    now = [1000.0]
    rec = sv.Recommender(model, fresh, popular_ttl=1.0, clock=lambda: now[0])
    first = rec.recommend(None, k=5)
    assert len(reads) == 1
    fresh.record(fe.Event(40, int(model.n_items - 1), 0))  # the last movie becomes the most popular
    now[0] += 0.5
    assert rec.recommend(None, k=5).movies.tolist() == first.movies.tolist()  # within the TTL: the cached list
    assert len(reads) == 1
    # Exclusions are per request even from the cached order: user 41's rated movie is excluded at once.
    fresh.record_rating(fe.Event(41, int(first.movies[0]), 0))
    assert int(first.movies[0]) not in rec.recommend(41, k=5).movies.tolist()
    assert len(reads) == 1
    now[0] += 0.5  # the TTL is up
    assert rec.recommend(None, k=5).movies[0] == model.n_items - 1
    assert len(reads) == 2


def test_a_zero_ttl_reads_every_time(fresh: FeatureStore, model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = sv.Recommender(model, fresh, popular_ttl=0)
    assert rec.recommend(None, k=3).movies.tolist() == [0, 1, 2]  # nobody has liked anything: index order
    fresh.record(fe.Event(50, 9, 0))
    assert rec.recommend(None, k=3).movies.tolist() == [9, 0, 1]


@pytest.mark.parametrize("ttl", [-1, 3600, float("nan"), float("inf"), True, "1"])
def test_bad_ttls_are_refused(fresh: FeatureStore, model: Model, ttl: object) -> None:
    with pytest.raises(ValueError):
        sv.Recommender(model, fresh, popular_ttl=ttl)  # type: ignore[arg-type]
