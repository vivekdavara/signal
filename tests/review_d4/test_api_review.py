"""Review D4: api.py's validation and error mapping."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import redis
from fastapi.testclient import TestClient

from signalrec import export, movielens
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
def api(model: Model, store: FeatureStore) -> TestClient:
    return TestClient(create_app(sv.Recommender(model, store, n_candidates=40)), raise_server_exceptions=False)


@pytest.mark.parametrize("rating", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_a_non_finite_rating_is_a_422_not_a_500(api: TestClient, model: Model, rating: str) -> None:
    """Defect (api.py:38, EventIn.rating): Python's JSON parser (which FastAPI uses) accepts NaN, Infinity and 1e309
    (inf). The ge/le bounds reject them, but the 422 body then echoes the input, and NaN/inf are not JSON: rendering
    the validation error raises ValueError, so the client gets a 500 for a malformed body.

    Input: POST /events {"user": <known>, "movie": <known>, "rating": NaN} (raw body).
    Expected: 422. Actual: 500 Internal Server Error.
    """
    user, movie = int(model.user_raw_ids[0]), int(model.item_raw_ids[0])
    body = f'{{"user": {user}, "movie": {movie}, "rating": {rating}}}'
    r = api.post("/events", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422, (r.status_code, r.text[:200])


def _underscored_id(model: Model) -> tuple[int, str]:
    """A known user whose id has at least two digits, written with an underscore between its digits."""
    for raw in model.user_raw_ids.tolist():
        text = str(raw)
        if len(text) >= 2:
            return raw, text[0] + "_" + text[1:]
    pytest.skip("no user id with two digits")


@pytest.mark.parametrize("spelling", ["underscore", "plus", "float"])
def test_a_malformed_user_id_in_the_path_is_refused_not_resolved_to_a_user(
    api: TestClient, model: Model, spelling: str
) -> None:
    """Defect (api.py:85, ``user: int`` in lax mode): the path id is parsed by Python-style int coercion, so
    ``/recommendations/1_0`` serves user 10, ``/recommendations/+5`` user 5 and ``/recommendations/5.0`` user 5. The
    API promises strict integers; an id spelled any other way should be a 422 (or 404), never another user's list.

    Input: GET /recommendations/<d>_<rest> for a known two-digit user id (and +<id>, <id>.0).
    Expected: 422 or 404. Actual: 200 with that user's recommendations.
    """
    raw, underscored = _underscored_id(model)
    path = {"underscore": underscored, "plus": f"+{raw}", "float": f"{raw}.0"}[spelling]
    r = api.get(f"/recommendations/{path}")
    assert r.status_code in (404, 422), (path, r.status_code, r.json())


@pytest.mark.parametrize("k", ["5.0", "1_0", "+5", " 5"])
def test_a_non_integer_k_is_a_422(api: TestClient, model: Model, k: str) -> None:
    """Defect (api.py:87, ``k: Annotated[int, Query(ge=1, le=MAX_K)]``): DESIGN.md says a bad k is a 422 with strict
    integers ("1.0" refused), but the query parameter is coerced: k=5.0 and k=+5 serve 5 movies, k=1_0 serves 10.

    Input: GET /recommendations/<user>?k=5.0 (and 1_0, +5, " 5").
    Expected: 422. Actual: 200.
    """
    r = api.get(f"/recommendations/{int(model.user_raw_ids[0])}", params={"k": k})
    assert r.status_code == 422, (k, r.status_code, r.json().get("k"))


def test_health_names_the_served_models_hash(model: Model, store: FeatureStore, tmp_path: Path) -> None:
    """Defect (api.py:79): /health's "model" is ``model.info.get("npz_sha256", dataset)``, but the hash is written
    to model.json beside ``info``, never into it, and ``tw.load`` doesn't put it there: /health always answers the
    dataset name, so it can't tell which model a server is running (the docstring: "which model is served").

    Input: a model saved with tw.save, loaded with tw.load, served; GET /health.
    Expected: "model" is the npz SHA-256 save returned. Actual: "synthetic".
    """
    from signalrec import tower as tw

    meta = tw.save(model, tmp_path / "m")
    served = tw.load(tmp_path / "m")
    api = TestClient(create_app(sv.Recommender(served, store, n_candidates=20)))
    assert api.get("/health").json()["model"] == meta["npz_sha256"]
