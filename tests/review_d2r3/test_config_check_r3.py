"""Review D2 round 3: ``Config.check`` after round 2's temperature floor and int/float rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from signalrec import movielens, prepare, synthetic
from signalrec import twotower as tt
from signalrec.prepare import Prepared


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=60, n_items=80)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


@pytest.mark.parametrize("field", ["temperature", "lr"])
def test_an_integer_too_large_for_a_float_is_a_value_error(field: str) -> None:
    """The int/float rule admits any Python int, and ``math.isfinite`` on an int too large for a float raises
    OverflowError, so ``check`` escapes its ValueError contract (and the callers that catch only ValueError)."""
    with pytest.raises(ValueError):
        tt.Config(**{field: 10**400}).check()  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["temperature", "lr"])
def test_a_numpy_float_is_accepted(field: str) -> None:
    """``type(value) not in (int, float)`` refuses ``np.float64(0.1)``, a subclass of float that every earlier version
    accepted: round 2 meant to refuse bools and also refuses valid NumPy floats (ItemKNN's k rule accepts NumPy
    integers)."""
    tt.Config(**{field: np.float64(0.1)}).check()  # type: ignore[arg-type]


def test_a_learning_rate_the_check_accepts_trains_on_finite_losses(prepared: Prepared) -> None:
    """Round 2 floored the temperature because the check let a model train "successfully" on NaN losses, but the
    learning rate has no such bound: ``lr=1e30`` passes ``check`` and trains on NaN losses into NaN embeddings."""
    config = tt.Config(dim=8, hidden=8, epochs=2, batch_size=64, lr=1e30)
    try:
        config.check()
    except ValueError:
        return
    seq = tt.sequences(prepared, include_val=False)
    net, stats = tt.train(seq, tt.genre_matrix(prepared.genres)[0], config)
    assert all(math.isfinite(stat.loss) for stat in stats), [stat.loss for stat in stats]
    embeddings = tt.embed(net, seq, config.max_history)
    assert np.isfinite(embeddings.users).all() and np.isfinite(embeddings.items).all()
