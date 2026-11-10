"""Review D4 round 2: tests that kill mutants of round 1's fixes that survived tests/review_d4 and the unit tests."""

from __future__ import annotations

import dataclasses
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
import redis

from signalrec import features as fe
from signalrec import tower as tw
from signalrec.featurestore import FeatureStore, FeatureStoreError
from signalrec.serving import Recommender
from signalrec.tower import Model


def _saved(model: Model, directory: Path) -> dict[str, Any]:
    tw.save(model, directory)
    meta: dict[str, Any] = json.loads((directory / "model.json").read_text())
    return meta


def _write(directory: Path, meta: dict[str, Any], rehash: bool = True) -> None:
    if rehash:
        meta["sha256"] = tw._meta_hash(meta)  # a self-consistent model.json: only the type checks can refuse it
    (directory / "model.json").write_text(json.dumps(meta))


def test_info_is_covered_by_model_json_hash(model: Model, tmp_path: Path) -> None:
    """Survivor (tower.py _meta_hash): leaving ``info`` out of the hashed fields. info carries min_rating, which
    decides what POST /events treats as a positive, so editing it must be refused."""
    meta = _saved(model, tmp_path)
    meta["info"]["min_rating"] = 1.0
    _write(tmp_path, meta, rehash=False)
    with pytest.raises(tw.ModelError):
        tw.load(tmp_path)


def test_max_history_true_is_refused(model: Model, tmp_path: Path) -> None:
    """Survivor (tower.py _strict_int): ``isinstance(value, int)`` for ``type(value) is int`` accepts true (a bool is
    an int) as max_history 1."""
    meta = _saved(model, tmp_path)
    meta["max_history"] = True
    _write(tmp_path, meta)
    with pytest.raises(tw.ModelError):
        tw.load(tmp_path)


@pytest.mark.parametrize("genre", [1.0, True])
def test_a_genre_index_that_is_not_an_int_is_refused(model: Model, tmp_path: Path, genre: Any) -> None:
    """Survivor (tower.py load, the movie_genres check): without it, int() coerces 1.0 and true to genre 1."""
    meta = _saved(model, tmp_path)
    meta["movie_genres"][0] = [genre]
    _write(tmp_path, meta)
    with pytest.raises(tw.ModelError):
        tw.load(tmp_path)


@pytest.mark.parametrize("field", ["titles", "genre_names"])
def test_a_title_or_genre_name_that_is_not_a_string_is_refused(model: Model, tmp_path: Path, field: str) -> None:
    """Survivor (tower.py load, the titles/genre names check): without it, str() turns 7 into the title "7"."""
    meta = _saved(model, tmp_path)
    meta[field][0] = 7
    _write(tmp_path, meta)
    with pytest.raises(tw.ModelError):
        tw.load(tmp_path)


def test_a_store_keeping_exactly_max_history_serves_the_model(client: redis.Redis, model: Model) -> None:
    """Survivor (serving.py Recommender, the max_history check): ``>=`` for ``>`` refuses a store whose recent cap
    equals the model's max_history, which keeps every position the tower reads."""
    small = dataclasses.replace(model, tower=dataclasses.replace(model.tower, max_history=4))
    store = FeatureStore(client, model.catalogue, namespace=f"test-{uuid.uuid4().hex}", recent=4)
    Recommender(small, store)  # no ValueError
    with pytest.raises(ValueError):
        Recommender(small, FeatureStore(client, model.catalogue, namespace=f"test-{uuid.uuid4().hex}", recent=3))


def test_record_maps_wrongtype_to_feature_store_error(fresh: FeatureStore) -> None:
    """Survivor (featurestore.py record): dropping the WRONGTYPE mapping; round 1's test accepts either exception."""
    fresh.client.set(f"{fresh.namespace}:u:0:rated", "x")
    with pytest.raises(FeatureStoreError):
        fresh.record(fe.Event(0, 0, 0))
