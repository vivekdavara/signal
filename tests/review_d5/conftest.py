"""Review D5: the serving fixtures (a synthetic dataset, a small trained model, Redis under fresh test-<uuid>
namespaces), shared from tests/serving/conftest.py, plus a loaded store and an API per test."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
import redis
from fastapi.testclient import TestClient

from signalrec import export, movielens
from signalrec import serving as sv
from signalrec.api import create_app
from signalrec.featurestore import FeatureStore
from signalrec.prepare import Prepared
from signalrec.tower import Model
from tests.serving.conftest import client, dataset, fresh, loaded, model, prepared  # noqa: F401


@pytest.fixture
def store(client: redis.Redis, model: Model, prepared: Prepared, dataset: movielens.Dataset) -> Iterator[FeatureStore]:  # noqa: F811
    s = FeatureStore(client, model.catalogue, namespace=f"test-{uuid.uuid4().hex}")
    export.load_store(s, prepared, dataset.ratings)
    yield s
    s.clear()


@pytest.fixture
def api(model: Model, store: FeatureStore) -> TestClient:  # noqa: F811
    return TestClient(create_app(sv.Recommender(model, store, n_candidates=40)), raise_server_exceptions=False)
