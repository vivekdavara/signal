"""api.py through FastAPI's test client: the endpoints, their errors, and the experiment arm.

Needs Redis (the fixtures are in conftest.py).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
import redis
from fastapi.testclient import TestClient

from signalrec import abtest, export, movielens
from signalrec import serving as sv
from signalrec.api import create_app
from signalrec.featurestore import FeatureStore
from signalrec.prepare import Prepared
from signalrec.tower import Model


@pytest.fixture
def store(client: redis.Redis, model: Model, prepared: Prepared, dataset: movielens.Dataset) -> Iterator[FeatureStore]:
    s = FeatureStore(client, model.catalogue, namespace=f"test-{uuid.uuid4().hex}")
    export.load_store(s, prepared, dataset.ratings)
    yield s
    s.clear()


@pytest.fixture
def recommender(model: Model, store: FeatureStore) -> sv.Recommender:
    return sv.Recommender(model, store, n_candidates=40)


@pytest.fixture
def api(recommender: sv.Recommender) -> TestClient:
    return TestClient(create_app(recommender))


def raw_user(model: Model, dense: int) -> int:
    return int(model.user_raw_ids[dense])


def test_recommendations_are_the_recommenders_in_movielens_ids(
    api: TestClient, recommender: sv.Recommender, model: Model
) -> None:
    for dense in (0, 7, 33):
        r = api.get(f"/recommendations/{raw_user(model, dense)}", params={"k": 12})
        assert r.status_code == 200, r.text
        body = r.json()
        want = recommender.recommend(dense, k=12)
        assert [item["movie"] for item in body["items"]] == [int(model.item_raw_ids[m]) for m in want.movies]
        assert [item["score"] for item in body["items"]] == want.scores.tolist()
        first = body["items"][0]
        m = model.movie_index(first["movie"])
        assert first["title"] == model.titles[m]
        assert first["genres"] == [model.catalogue.genre_names[g] for g in model.catalogue.movie_genres[m]]
        assert body["arm"] is None and body["k"] == 12 and body["user"] == raw_user(model, dense)
        stages = [part.split(";")[0] for part in r.headers["server-timing"].split(", ")]
        assert stages == [s.replace(" ", "-") for s in sv.STAGES]


def test_genres_are_excluded_by_name(api: TestClient, model: Model) -> None:
    names = model.catalogue.genre_names
    r = api.get(
        f"/recommendations/{raw_user(model, 3)}", params=[("exclude_genre", names[0]), ("exclude_genre", names[2])]
    )
    assert r.status_code == 200
    for item in r.json()["items"]:
        assert names[0] not in item["genres"] and names[2] not in item["genres"]
    bad = api.get(f"/recommendations/{raw_user(model, 3)}", params={"exclude_genre": "Space Opera"})
    assert bad.status_code == 422 and "Space Opera" in bad.json()["detail"]


@pytest.mark.parametrize("k", [0, -1, 101, "ten"])
def test_bad_k_is_a_422(api: TestClient, model: Model, k: object) -> None:
    assert api.get(f"/recommendations/{raw_user(model, 0)}", params={"k": k}).status_code == 422


def test_unknown_users_are_a_404(api: TestClient, model: Model) -> None:
    missing = int(model.user_raw_ids.max()) + 1
    r = api.get(f"/recommendations/{missing}")
    assert r.status_code == 404 and str(missing) in r.json()["detail"]
    assert api.get("/recommendations/abc").status_code == 422


def test_a_positive_event_removes_the_movie_and_a_retry_is_a_no_op(api: TestClient, model: Model) -> None:
    user = raw_user(model, 5)
    top = api.get(f"/recommendations/{user}").json()["items"][0]["movie"]
    r = api.post("/events", json={"user": user, "movie": top, "rating": 5.0, "time": 2_000_000_000})
    assert r.status_code == 200 and r.json() == {"applied": True, "positive": True}
    assert api.post("/events", json={"user": user, "movie": top, "rating": 5.0}).json()["applied"] is False
    after = [item["movie"] for item in api.get(f"/recommendations/{user}").json()["items"]]
    assert top not in after


def test_a_low_rating_only_filters(api: TestClient, model: Model, store: FeatureStore) -> None:
    user = raw_user(model, 6)
    before = store.user(6)
    top = api.get(f"/recommendations/{user}").json()["items"][0]["movie"]
    r = api.post("/events", json={"user": user, "movie": top, "rating": 2})
    assert r.json() == {"applied": True, "positive": False}
    assert store.user(6) == before
    assert top not in [item["movie"] for item in api.get(f"/recommendations/{user}").json()["items"]]


@pytest.mark.parametrize(
    "body",
    [
        {"user": 1.0, "movie": 1, "rating": 5},
        {"user": True, "movie": 1, "rating": 5},
        {"user": "1", "movie": 1, "rating": 5},
        {"user": 1, "movie": 1, "rating": 5.5},
        {"user": 1, "movie": 1, "rating": 0},
        {"user": 1, "movie": 1, "rating": "5"},
        {"user": 1, "movie": 1},
        {"user": 1, "movie": 1, "rating": 4, "extra": 1},
        {"user": 0, "movie": 1, "rating": 4},
        {"user": 1, "movie": 1, "rating": 4, "time": -1},
    ],
)
def test_malformed_events_are_a_422(api: TestClient, body: dict[str, object]) -> None:
    assert api.post("/events", json=body).status_code == 422


def test_events_for_unknown_users_or_movies_are_a_404(api: TestClient, model: Model) -> None:
    missing_movie = int(model.item_raw_ids.max()) + 1
    r = api.post("/events", json={"user": raw_user(model, 0), "movie": missing_movie, "rating": 5})
    assert r.status_code == 404 and str(missing_movie) in r.json()["detail"]
    missing_user = int(model.user_raw_ids.max()) + 1
    assert api.post("/events", json={"user": missing_user, "movie": 1, "rating": 5}).status_code == 404


def test_health(api: TestClient, model: Model) -> None:
    r = api.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["movies"] == model.n_items


def test_redis_down_is_a_503(model: Model) -> None:
    dead = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2)
    rec = sv.Recommender(model, FeatureStore(dead, model.catalogue, namespace="unused"), n_candidates=20)
    api = TestClient(create_app(rec))
    assert api.get("/health").status_code == 503
    assert api.get(f"/recommendations/{raw_user(model, 0)}").status_code == 503
    assert (
        api.post(
            "/events", json={"user": raw_user(model, 0), "movie": int(model.item_raw_ids[0]), "rating": 5}
        ).status_code
        == 503
    )


def test_foreign_values_in_redis_are_a_500(api: TestClient, model: Model, store: FeatureStore) -> None:
    store.client.sadd(f"{store.namespace}:u:2:rated", "not-a-movie")
    r = api.get(f"/recommendations/{raw_user(model, 2)}")
    assert r.status_code == 500 and "feature store" in r.json()["detail"]


def test_the_experiment_arm_decides_whether_the_ranker_runs(recommender: sv.Recommender, model: Model) -> None:
    experiment = abtest.Experiment("ranker-v1", 0.5)
    api = TestClient(create_app(recommender, experiment))
    arms = set()
    for dense in range(30):
        user = raw_user(model, dense)
        body = api.get(f"/recommendations/{user}").json()
        arm = abtest.assign(experiment, user)
        assert body["arm"] == arm
        arms.add(arm)
        want = recommender.recommend(dense, rank=arm == abtest.TREATMENT)
        assert [item["movie"] for item in body["items"]] == [int(model.item_raw_ids[m]) for m in want.movies]
    assert arms == {abtest.CONTROL, abtest.TREATMENT}
    assert api.get("/health").json()["experiment"] == "ranker-v1"
