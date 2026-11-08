"""Review D2 round 2: ``Config.check``'s new temperature rule and the bool fields it still lets through."""

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


@pytest.mark.parametrize("temperature", [3e-39, 1e-38])
def test_a_temperature_the_check_accepts_trains_on_finite_losses(prepared: Prepared, temperature: float) -> None:
    """The round-1 bound (1/temperature a finite float32) is too loose: logits differ by up to 2/temperature and a
    temperature below float32's smallest normal is a denormal, so 3e-39 still trains on NaN losses (NaN embeddings,
    which evaluation then refuses after the grid has trained) and 1e-38 on infinite ones."""
    config = tt.Config(dim=8, hidden=8, epochs=1, batch_size=64, lr=1e-2, temperature=temperature)
    try:
        config.check()
    except ValueError:
        return
    seq = tt.sequences(prepared, include_val=False)
    net, stats = tt.train(seq, tt.genre_matrix(prepared.genres)[0], config)
    assert all(math.isfinite(stat.loss) for stat in stats), [stat.loss for stat in stats]
    embeddings = tt.embed(net, seq, config.max_history)
    assert np.isfinite(embeddings.users).all() and np.isfinite(embeddings.items).all()


@pytest.mark.parametrize("field", ["temperature", "lr"])
def test_a_bool_temperature_or_learning_rate_is_refused(field: str) -> None:
    """Round 1 made ``logq`` and ``seed`` refuse the wrong type, but ``temperature=True`` and ``lr=True`` still pass
    ``Config.check`` as 1.0 (so a results file saying ``"temperature": true`` is a usable configuration)."""
    with pytest.raises(ValueError):
        tt.Config(**{field: True}).check()
