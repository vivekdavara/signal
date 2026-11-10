"""Review D4 round 2: api.py's digit-only path and query integers."""

from __future__ import annotations

import uuid

import pytest
import redis
from fastapi.testclient import TestClient

from signalrec.api import create_app
from signalrec.featurestore import FeatureStore
from signalrec.serving import Recommender
from signalrec.tower import Model


@pytest.mark.parametrize("path", ["/recommendations/0{u}", "/recommendations/00{u}", "/recommendations/{u}?k=05"])
def test_leading_zeros_are_not_another_spelling_of_an_id(client: redis.Redis, model: Model, path: str) -> None:
    """Defect (api.py:35, DIGITS): round 1 made ids and k digit-only because FastAPI's lax ints let several strings
    name one user ("1_0" served user 10). ``^[0-9]{1,18}$`` still allows leading zeros, so /recommendations/01,
    /recommendations/001 and /recommendations/1 are three URLs for user 1 (and the body says "user": 1 to a client
    that asked for "01"), and k=05 is k=5: the aliasing round 1 removed, by another spelling. MovieLens ids have no
    leading zeros; the canonical form is ``0|[1-9][0-9]*``.

    Input: GET /recommendations/0<u>, /recommendations/00<u>, /recommendations/<u>?k=05 for a known user u.
    Expected: 422, like "1_0" and "+5".
    Actual: 200, served as user u (k 5).
    """
    store = FeatureStore(client, model.catalogue, namespace=f"test-{uuid.uuid4().hex}")
    try:
        http = TestClient(create_app(Recommender(model, store)), raise_server_exceptions=False)
        u = int(model.user_raw_ids[0])
        r = http.get(path.format(u=u))
        assert r.status_code == 422, (r.status_code, r.text[:200])
    finally:
        store.clear()
