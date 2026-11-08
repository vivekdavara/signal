"""Review D2: ``experiment_twotower.run`` and ``twotower.Config`` refuse bad inputs before spending training time."""

from __future__ import annotations

import math

import numpy as np
import pytest

from signalrec import experiment_twotower as ext
from signalrec import movielens, prepare, synthetic
from signalrec import twotower as tt
from signalrec.prepare import Prepared

BASE = tt.Config(dim=8, hidden=8, batch_size=64, lr=1e-2)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=60, n_items=80)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


class _Trained(Exception):
    pass


def _no_training(*args: object, **kwargs: object) -> None:
    raise _Trained("run trained a model before refusing its arguments")


def test_an_empty_knn_grid_is_refused_before_training(prepared: Prepared, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run`` checks its own grid and seeds are non-empty but not item-kNN's grid, so an empty one fails only after
    every two-tower configuration has trained (with "max() iterable argument is empty")."""
    monkeypatch.setattr(tt, "train", _no_training)
    with pytest.raises(ValueError, match="grid"):
        ext.run(prepared, ((0.1, True),), 1, (0,), (), BASE)


def test_an_unusable_later_seed_is_refused_before_training(prepared: Prepared, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run`` checks only that seeds are non-empty, so a negative later seed (``numpy.random.default_rng`` refuses it)
    fails only after tuning and the first seed's test run."""
    with pytest.raises(ValueError):
        np.random.default_rng(-1)  # the seed itself is unusable
    monkeypatch.setattr(tt, "train", _no_training)
    with pytest.raises(ValueError, match="seed"):
        ext.run(prepared, ((0.1, True),), 1, (0, -1), ((5, 0.0),), BASE)


def test_a_temperature_whose_reciprocal_overflows_float32_is_refused() -> None:
    """``Config.check`` (and ``--grid``) accept any finite temperature > 0, but below ~3e-39 the float32 logits overflow
    and training silently produces a NaN loss (then evaluation fails on a non-finite score)."""
    config = tt.Config(dim=4, hidden=4, epochs=1, batch_size=4, temperature=1e-50)
    try:
        config.check()
    except ValueError:
        return
    seq = tt.Sequences(np.arange(6, dtype=np.int64), np.array([0, 3, 6], dtype=np.int64))
    _, stats = tt.train(seq, np.zeros((6, 0), dtype=np.float32), config)
    assert math.isfinite(stats[-1].loss)
