"""Building what serving reads (DESIGN.md, "Day 4: serving and experiments").

- ``build_model``: the reported two-tower configuration retrained on training + validation (as the test runs of
  ``twotower`` and ``ranking`` do; deterministic for a seed on one machine), its user tower exported to NumPy and its
  item vectors computed, plus the ranker chosen by ``signalrec ranking`` (its weights are in ``ranking.json``).
- ``load_store``: the feature store as of just before every user's test movie: the training and validation positives
  replayed through the event script, and every earlier rating into the rated set.

So the served model is the one the offline test numbers describe, and ``serving_check`` can require the request path
to reproduce them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from signalrec import features as fe
from signalrec import ranking as rk
from signalrec import twotower as tt
from signalrec.featurestore import FeatureStore, ReplayStats
from signalrec.movielens import Ratings
from signalrec.prepare import Prepared, PrepareError
from signalrec.tower import Model, export_user_tower


def ranker_from_results(path: Path, dataset: str, config: tt.Config) -> rk.RankerWeights:
    """The chosen ranker's weights from a ``ranking.json``, which must have ranked this dataset with ``config``."""
    try:
        result = json.loads(path.read_text())
        if result["dataset"] != dataset:
            raise PrepareError(f"{path} holds {result['dataset']!r}, not {dataset!r}")
        if result["two-tower configuration"] != config.as_dict():
            raise PrepareError(
                f"{path} ranked candidates from another two-tower configuration than the reported one; "
                "run signalrec ranking again"
            )
        w = result["ranker weights"]
        weights = rk.RankerWeights(
            np.array(w["mean"], dtype=np.float64),
            np.array(w["scale"], dtype=np.float64),
            tuple((np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)) for a, b in w["layers"]),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PrepareError):
            raise
        raise PrepareError(f"{path}: not a ranking results file ({exc!r}); run signalrec ranking") from None
    return weights


def build_model(p: Prepared, config: tt.Config, ranker: rk.RankerWeights, info: dict[str, Any]) -> Model:
    """Train the two-tower on training + validation with ``config`` and bundle it with ``ranker``."""
    config.check()
    seq = tt.sequences(p, include_val=True)
    net, _ = tt.train(seq, tt.genre_matrix(p.genres)[0], config)
    emb = tt.embed(net, seq, config.max_history)
    model = Model(
        tower=export_user_tower(net, config.max_history),
        items=emb.items,
        ranker=ranker,
        item_raw_ids=p.item_raw_ids.astype(np.int64),
        user_raw_ids=p.user_raw_ids.astype(np.int64),
        titles=p.titles,
        catalogue=fe.Catalogue.from_prepared(p),
        info={**info, "two-tower configuration": config.as_dict()},
    )
    model.check()
    return model


def load_store(store: FeatureStore, p: Prepared, ratings: Ratings) -> tuple[ReplayStats, ReplayStats]:
    """Replay the training and validation positives, then every rating before each user's test time."""
    positives = store.replay(fe.events(p, include_val=True))
    rated = store.replay_ratings(fe.rated_events(p, ratings))
    return positives, rated
