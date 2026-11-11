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
    assert bad.status_code == 422 and "Space Opera" in bad.json()["detail"][0]["msg"]


@pytest.mark.parametrize("k", [0, -1, 101, "ten"])
def test_bad_k_is_a_422(api: TestClient, model: Model, k: object) -> None:
    assert api.get(f"/recommendations/{raw_user(model, 0)}", params={"k": k}).status_code == 422


def test_a_user_with_no_events_gets_the_popular_list_and_is_not_registered(
    api: TestClient, model: Model, store: FeatureStore
) -> None:
    missing = int(model.user_raw_ids.max()) + 1
    r = api.get(f"/recommendations/{missing}", params={"k": 7})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "popularity" and body["user"] == missing
    popularity = store.popularity()
    want = sorted(range(model.n_items), key=lambda m: (-int(popularity[m]), m))[:7]
    assert [item["movie"] for item in body["items"]] == [int(model.item_raw_ids[m]) for m in want]
    assert store.registered(missing, len(model.user_raw_ids)) is None  # a GET registers nobody
    assert api.get("/recommendations/abc").status_code == 422


def test_known_users_are_served_by_the_two_tower_model(api: TestClient, model: Model) -> None:
    assert api.get(f"/recommendations/{raw_user(model, 0)}").json()["source"] == "two-tower"


def test_a_positive_event_removes_the_movie_and_a_retry_is_a_no_op(api: TestClient, model: Model) -> None:
    user = raw_user(model, 5)
    top = api.get(f"/recommendations/{user}").json()["items"][0]["movie"]
    r = api.post("/events", json={"user": user, "movie": top, "rating": 5.0, "time": 2_000_000_000})
    assert r.status_code == 200 and r.json() == {"applied": True, "positive": True, "new_user": False}
    assert api.post("/events", json={"user": user, "movie": top, "rating": 5.0}).json()["applied"] is False
    after = [item["movie"] for item in api.get(f"/recommendations/{user}").json()["items"]]
    assert top not in after


def test_a_low_rating_only_filters(api: TestClient, model: Model, store: FeatureStore) -> None:
    user = raw_user(model, 6)
    before = store.user(6)
    top = api.get(f"/recommendations/{user}").json()["items"][0]["movie"]
    r = api.post("/events", json={"user": user, "movie": top, "rating": 2})
    assert r.json() == {"applied": True, "positive": False, "new_user": False}
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


def test_events_for_unknown_movies_are_a_404_and_register_nobody(
    api: TestClient, model: Model, store: FeatureStore
) -> None:
    missing_movie = int(model.item_raw_ids.max()) + 1
    r = api.post("/events", json={"user": raw_user(model, 0), "movie": missing_movie, "rating": 5})
    assert r.status_code == 404 and str(missing_movie) in r.json()["detail"]
    new_user = int(model.user_raw_ids.max()) + 1
    assert api.post("/events", json={"user": new_user, "movie": missing_movie, "rating": 5}).status_code == 404
    assert store.registered(new_user, len(model.user_raw_ids)) is None


def test_a_new_user_is_registered_by_their_first_event_and_then_personalised(
    api: TestClient, model: Model, store: FeatureStore
) -> None:
    n_users = len(model.user_raw_ids)
    new_user = int(model.user_raw_ids.max()) + 10
    first = api.get(f"/recommendations/{new_user}").json()
    assert first["source"] == "popularity"
    disliked = first["items"][0]["movie"]
    r = api.post("/events", json={"user": new_user, "movie": disliked, "rating": 1})
    assert r.json() == {"applied": True, "positive": False, "new_user": True}
    assert store.registered(new_user, n_users) == n_users
    after_dislike = api.get(f"/recommendations/{new_user}").json()
    assert after_dislike["source"] == "popularity"  # still no positives
    assert [i["movie"] for i in after_dislike["items"]] == [i["movie"] for i in first["items"][1:]] + [
        after_dislike["items"][-1]["movie"]
    ]
    liked = after_dislike["items"][3]["movie"]
    r = api.post("/events", json={"user": new_user, "movie": liked, "rating": 5})
    assert r.json() == {"applied": True, "positive": True, "new_user": False}
    personal = api.get(f"/recommendations/{new_user}").json()
    assert personal["source"] == "two-tower"
    served = {i["movie"] for i in personal["items"]}
    assert disliked not in served and liked not in served
    # Their features are exactly one positive's, under the new index.
    features = store.user(n_users)
    assert features.positives == 1 and features.recent.tolist() == [model.movie_index(liked)]
    # A second new user gets the next index; the first keeps theirs.
    other = new_user + 1
    assert api.post("/events", json={"user": other, "movie": liked, "rating": 4}).json()["new_user"] is True
    assert store.registered(other, n_users) == n_users + 1
    assert store.registered(new_user, n_users) == n_users


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


def test_every_422_has_the_same_shape(api: TestClient, model: Model) -> None:
    user = raw_user(model, 0)
    for r in (
        api.get(f"/recommendations/{user}", params={"k": 101}),
        api.get(f"/recommendations/{user}", params={"exclude_genre": "Space Opera"}),
        api.get(f"/recommendations/{user}", params={"k": "x"}),
        api.post("/events", json={"user": user}),
    ):
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert isinstance(detail, list) and detail and {"loc", "msg", "type"} <= set(detail[0])


def test_the_api_reports_the_package_version(api: TestClient) -> None:
    import signalrec

    assert api.get("/openapi.json").json()["info"]["version"] == signalrec.__version__ == "1.0.0"


def test_users_on_the_popular_list_are_in_no_arm(recommender: sv.Recommender, model: Model) -> None:
    experiment = abtest.Experiment("ranker-v1", 0.5)
    api = TestClient(create_app(recommender, experiment))
    first_new = int(model.user_raw_ids.max()) + 1
    arms = {abtest.assign(experiment, u) for u in range(first_new, first_new + 20)}
    assert arms == {abtest.CONTROL, abtest.TREATMENT}  # both arms would be drawn
    for user in range(first_new, first_new + 20):
        body = api.get(f"/recommendations/{user}").json()
        assert body["source"] == "popularity" and body["arm"] is None


@pytest.mark.parametrize("field", ["user", "movie"])
@pytest.mark.parametrize("value", [10**18, 2**63, 2**64])
def test_event_ids_beyond_what_a_get_accepts_are_a_422(
    api: TestClient, model: Model, store: FeatureStore, field: str, value: int
) -> None:
    body = {"user": raw_user(model, 0), "movie": int(model.item_raw_ids[0]), "rating": 5, field: value}
    r = api.post("/events", json=body)
    assert r.status_code == 422, r.text
    assert store.registered(min(value, 2**63 - 1), len(model.user_raw_ids)) is None


def test_the_largest_id_an_event_takes_can_be_asked_for(api: TestClient, model: Model) -> None:
    user = 10**18 - 1
    r = api.post("/events", json={"user": user, "movie": int(model.item_raw_ids[0]), "rating": 5})
    assert r.json()["new_user"] is True
    got = api.get(f"/recommendations/{user}")
    assert got.status_code == 200 and got.json()["source"] == "two-tower"
