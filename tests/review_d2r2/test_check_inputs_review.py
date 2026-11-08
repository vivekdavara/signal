"""Review D2 round 2: ``experiment_twotower._check_inputs`` misses an item-kNN grid that fails only after training."""

from __future__ import annotations

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


def test_a_non_integer_knn_k_is_refused_before_training(prepared: Prepared, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_check_inputs`` relies on ``ItemKNN``'s constructor, which accepts k=5.0, but ``neighbours`` slices with k, so
    ``run`` raises a TypeError ("slice indices must be integers") only after every two-tower configuration trained."""
    monkeypatch.setattr(tt, "train", _no_training)
    with pytest.raises((ValueError, TypeError)):
        ext.run(prepared, ((0.1, True),), 1, (0,), ((5.0, 0.0),), BASE)  # type: ignore[arg-type]
