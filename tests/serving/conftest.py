"""Shared by the serving and API tests: a synthetic dataset, a small trained model, and Redis (as
test_featurestore.py: skipped without a server unless SIGNALREC_REQUIRE_REDIS=1)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import numpy as np
import pytest
import redis

from signalrec import export, movielens, prepare, synthetic
from signalrec import ranking as rk
from signalrec import twotower as tt
from signalrec.featurestore import FeatureStore
from signalrec.prepare import Prepared
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
