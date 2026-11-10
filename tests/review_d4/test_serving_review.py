"""Review D4: training-serving skew in the request path (serving.py)."""

from __future__ import annotations

import uuid

import numpy as np
import redis

from signalrec import export, movielens
from signalrec import twotower as tt
from signalrec.featurestore import FeatureStore
from signalrec.prepare import Prepared
from signalrec.serving import Recommender
from signalrec.tower import Model


def test_a_store_keeping_fewer_recent_positives_than_the_towers_history_is_refused(
    client: redis.Redis, model: Model, prepared: Prepared, dataset: movielens.Dataset
) -> None:
    """Defect (serving.py:58-73, Recommender.__init__): the user tower reads the last ``max_history`` positives from
    the store's ``recent`` list, but nothing checks that the store keeps at least that many. A twotower.json chosen
    with ``--histories 100`` (max_history 100 > features.RECENT = 50) is served from a store capped at 50: every heavy
    user's vector is computed from half their history, not what was evaluated offline, silently.

    Input: the fixture model (max_history 4) and a FeatureStore with recent=2, loaded with the same data.
    Expected: Recommender(...) raises ValueError (or serving reproduces the offline user vector).
    Actual: accepted, and the served vectors differ from ``twotower.embed``'s for users with more than 2 positives.
    """
    assert model.tower.max_history == 4
    store = FeatureStore(client, model.catalogue, namespace=f"test-{uuid.uuid4().hex}", recent=2)
    try:
        export.load_store(store, prepared, dataset.ratings)
        try:
            rec = Recommender(model, store)
        except ValueError:
            return  # refused: the fix
        # Accepted: then it must at least compute what offline computes. It doesn't.
        seq = tt.sequences(prepared, include_val=True)
        lengths = np.diff(seq.starts)
        heavy = int(np.flatnonzero(lengths > 2)[0])
        features, _ = store.serving_view(heavy)
        full = seq.items[seq.starts[heavy] : seq.starts[heavy + 1]]
        np.testing.assert_allclose(
            rec.model.tower.vector(features.recent),
            model.tower.vector(full),
            atol=1e-6,
            err_msg="served user vector differs from the offline one (store recent < max_history)",
        )
    finally:
        store.clear()
