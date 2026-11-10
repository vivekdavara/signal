"""tower.py: the NumPy user tower against PyTorch's, and the model files."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from signalrec import movielens, prepare, synthetic
from signalrec import ranking as rk
from signalrec import retrieval as rt
from signalrec import tower as tw
from signalrec import twotower as tt
from signalrec.features import Catalogue
from signalrec.prepare import Prepared

MAX_HISTORY = 5


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=120, n_items=150)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


@pytest.fixture(scope="module")
def trained(prepared: Prepared) -> tuple[tt.TwoTowerNet, tt.Sequences]:
    seq = tt.sequences(prepared, include_val=True)
    config = tt.Config(dim=16, hidden=32, epochs=3, batch_size=64, max_history=MAX_HISTORY)
    net, _ = tt.train(seq, tt.genre_matrix(prepared.genres)[0], config)
    return net, seq


def _torch_vector(net: tt.TwoTowerNet, history: list[int]) -> tw.Float32Array:
    row = [net.pad] * (MAX_HISTORY - len(history)) + history[-MAX_HISTORY:]
    net.eval()
    with torch.no_grad():
        out: tw.Float32Array = net.users(torch.tensor([row], dtype=torch.int64)).numpy()[0]
    net.train()
    return out


def _ranker(n_features: int = len(rk.FEATURES)) -> rk.RankerWeights:
    rng = np.random.default_rng(3)
    return rk.RankerWeights(
        rng.normal(size=n_features),
        rng.uniform(0.5, 2, size=n_features),
        ((rng.normal(size=(4, n_features)), rng.normal(size=4)), (rng.normal(size=(1, 4)), rng.normal(size=1))),
    )


def _model(prepared: Prepared, net: tt.TwoTowerNet, seq: tt.Sequences) -> tw.Model:
    emb = tt.embed(net, seq, MAX_HISTORY)
    return tw.Model(
        tower=tw.export_user_tower(net, MAX_HISTORY),
        items=emb.items,
        ranker=_ranker(),
        item_raw_ids=prepared.item_raw_ids,
        user_raw_ids=prepared.user_raw_ids,
        titles=prepared.titles,
        catalogue=Catalogue.from_prepared(prepared),
        info={"note": "test"},
    )


def test_every_users_vector_matches_pytorch(trained: tuple[tt.TwoTowerNet, tt.Sequences]) -> None:
    net, seq = trained
    tower = tw.export_user_tower(net, MAX_HISTORY)
    emb = tt.embed(net, seq, MAX_HISTORY)
    worst = 0.0
    for u in range(seq.n_users):
        history = seq.items[seq.starts[u] : seq.starts[u + 1]]
        got = tower.vector(history)
        assert got.dtype == np.float32
        worst = max(worst, float(np.abs(got - emb.users[u]).max()))
    # float32 sums in another order: a few units in the last place of numbers below 1.
    assert worst < 1e-6


@pytest.mark.parametrize("history", [[], [0], [3, 7], [1, 2, 3, 4, 5], list(range(12)), [9, 9, 9]])
def test_short_long_empty_and_repeated_histories(
    trained: tuple[tt.TwoTowerNet, tt.Sequences], history: list[int]
) -> None:
    net, _ = trained
    tower = tw.export_user_tower(net, MAX_HISTORY)
    got = tower.vector(np.array(history, dtype=np.int64))
    np.testing.assert_allclose(got, _torch_vector(net, history), atol=1e-6, rtol=0)
    assert abs(float(np.linalg.norm(got.astype(np.float64))) - 1.0) < 1e-6


def test_only_the_last_max_history_positives_are_read(trained: tuple[tt.TwoTowerNet, tt.Sequences]) -> None:
    net, _ = trained
    tower = tw.export_user_tower(net, MAX_HISTORY)
    long = np.arange(20, dtype=np.int64)
    np.testing.assert_array_equal(tower.vector(long), tower.vector(long[-MAX_HISTORY:]))
    assert not np.array_equal(tower.vector(long), tower.vector(long[-MAX_HISTORY - 1 : -1]))


def test_retrieval_returns_the_same_movies_from_either_vector(
    prepared: Prepared, trained: tuple[tt.TwoTowerNet, tt.Sequences]
) -> None:
    net, seq = trained
    tower = tw.export_user_tower(net, MAX_HISTORY)
    emb = tt.embed(net, seq, MAX_HISTORY)
    retriever = rt.Retriever(emb.items, rt.IndexSpec("flat"))
    for u in range(seq.n_users):
        history = seq.items[seq.starts[u] : seq.starts[u + 1]]
        a, _ = retriever.search(emb.users[u], 30, history)
        b, _ = retriever.search(tower.vector(history), 30, history)
        assert a.tolist() == b.tolist(), u


def test_a_history_movie_outside_the_catalogue_is_refused(trained: tuple[tt.TwoTowerNet, tt.Sequences]) -> None:
    tower = tw.export_user_tower(trained[0], MAX_HISTORY)
    for bad in ([-1], [tower.n_items], [0, 10**9]):
        with pytest.raises(ValueError, match="outside the catalogue"):
            tower.vector(np.array(bad, dtype=np.int64))
    # The padding row is not a movie either.
    with pytest.raises(ValueError, match="outside the catalogue"):
        tower.vector(np.array([tower.n_items], dtype=np.int64))


def test_save_and_load_round_trip(
    tmp_path: Path, prepared: Prepared, trained: tuple[tt.TwoTowerNet, tt.Sequences]
) -> None:
    model = _model(prepared, *trained)
    tw.save(model, tmp_path / "m")
    back = tw.load(tmp_path / "m")
    for name in ("history_table", "w1", "b1", "w2", "b2"):
        np.testing.assert_array_equal(getattr(back.tower, name), getattr(model.tower, name))
    np.testing.assert_array_equal(back.items, model.items)
    np.testing.assert_array_equal(back.item_raw_ids, model.item_raw_ids)
    np.testing.assert_array_equal(back.user_raw_ids, model.user_raw_ids)
    assert back.titles == model.titles
    assert back.catalogue.genre_names == model.catalogue.genre_names
    assert back.catalogue.movie_genres == model.catalogue.movie_genres
    np.testing.assert_array_equal(back.catalogue.years, model.catalogue.years)
    assert back.info == {"note": "test"}
    assert back.tower.max_history == MAX_HISTORY
    features = np.random.default_rng(0).normal(size=(7, len(rk.FEATURES)))
    np.testing.assert_array_equal(back.ranker.score(features), model.ranker.score(features))


def test_a_changed_npz_is_refused(
    tmp_path: Path, prepared: Prepared, trained: tuple[tt.TwoTowerNet, tt.Sequences]
) -> None:
    tw.save(_model(prepared, *trained), tmp_path)
    data = bytearray((tmp_path / "model.npz").read_bytes())
    data[-100] ^= 1
    (tmp_path / "model.npz").write_bytes(bytes(data))
    with pytest.raises(tw.ModelError, match="SHA-256"):
        tw.load(tmp_path)


def test_a_missing_or_foreign_model_json_is_refused(tmp_path: Path) -> None:
    with pytest.raises(tw.ModelError, match="no readable model"):
        tw.load(tmp_path)
    (tmp_path / "model.json").write_text(json.dumps({"format_version": 99}))
    (tmp_path / "model.npz").write_bytes(b"")
    with pytest.raises(tw.ModelError, match="format version"):
        tw.load(tmp_path)


def test_movie_and_user_ids_map_both_ways(prepared: Prepared, trained: tuple[tt.TwoTowerNet, tt.Sequences]) -> None:
    model = _model(prepared, *trained)
    for i in (0, 5, model.n_items - 1):
        assert model.movie_index(int(model.item_raw_ids[i])) == i
    assert model.user_index(int(model.user_raw_ids[-1])) == len(model.user_raw_ids) - 1
    missing = int(model.item_raw_ids.max()) + 1
    with pytest.raises(KeyError):
        model.movie_index(missing)
    with pytest.raises(KeyError):
        model.user_index(0)  # raw MovieLens ids start at 1


def test_inconsistent_models_are_refused(prepared: Prepared, trained: tuple[tt.TwoTowerNet, tt.Sequences]) -> None:
    good = _model(prepared, *trained)
    bad_pad = good.tower.history_table.copy()
    bad_pad[-1, 0] = 1.0
    nan_ranker = _ranker()
    nan_ranker.layers[0][0][0, 0] = np.nan
    cases = [
        replace(good, tower=replace(good.tower, history_table=bad_pad)),
        replace(good, tower=replace(good.tower, max_history=0)),
        replace(good, items=good.items[:-1]),
        replace(good, item_raw_ids=good.item_raw_ids[::-1].copy()),
        replace(good, items=good.items.astype(np.float64)),
        replace(good, ranker=_ranker(5)),
        replace(good, ranker=nan_ranker),
        replace(good, ranker=replace(_ranker(), scale=np.zeros(len(rk.FEATURES)))),
        replace(good, catalogue=replace(good.catalogue, movie_genres=good.catalogue.movie_genres[:-1])),
        replace(good, catalogue=replace(good.catalogue, movie_genres=((99,), *good.catalogue.movie_genres[1:]))),
        replace(good, catalogue=replace(good.catalogue, years=good.catalogue.years[:-1])),
    ]
    for model in cases:
        with pytest.raises(tw.ModelError):
            model.check()
    good.check()


def test_info_with_int_keys_round_trips_and_nan_is_refused(
    tmp_path: Path, prepared: Prepared, trained: tuple[tt.TwoTowerNet, tt.Sequences]
) -> None:
    model = replace(_model(prepared, *trained), info={"by k": {9: "a", 10: "b"}})
    tw.save(model, tmp_path / "ints")
    assert tw.load(tmp_path / "ints").info == {"by k": {"9": "a", "10": "b"}}
    with pytest.raises(tw.ModelError, match="JSON"):
        tw.save(replace(model, info={"x": float("nan")}), tmp_path / "nan")
