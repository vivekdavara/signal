"""One user's recommendations by title."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signalrec import cli, movielens, prepare, recommend, synthetic
from signalrec.baselines import ItemKNN, Popularity
from signalrec.prepare import Prepared


def small() -> Prepared:
    # Three users, five movies. User 0 liked movies 0 then 1 (validation), test movie 2; movie 2 is the most liked
    # among the rest, so popularity ranks it first for user 0.
    return Prepared(
        train_users=np.array([0, 1, 1, 2, 2], dtype=np.int64),
        train_items=np.array([0, 2, 3, 2, 4], dtype=np.int64),
        train_times=np.array([1, 1, 2, 1, 2], dtype=np.int64),
        val_items=np.array([1, 0, 0], dtype=np.int64),
        val_times=np.array([2, 3, 3], dtype=np.int64),
        test_items=np.array([2, 4, 3], dtype=np.int64),
        test_times=np.array([3, 4, 4], dtype=np.int64),
        user_raw_ids=np.array([10, 20, 30], dtype=np.int64),
        item_raw_ids=np.array([1, 2, 3, 4, 5], dtype=np.int64),
        titles=("Alpha (1990)", "Beta (1991)", "Gamma (1992)", "Delta (1993)", "Epsilon (1994)"),
        genres=((), (), (), (), ()),
    )


def test_user_ids_are_raw_movielens_ids() -> None:
    p = small()
    assert recommend.user_index(p, 20) == 1
    with pytest.raises(KeyError, match="user 25 is not in the prepared data"):
        recommend.user_index(p, 25)
    with pytest.raises(KeyError):
        recommend.user_index(p, 99)


def test_recent_likes_end_with_the_validation_movie() -> None:
    assert recommend.recent_likes(small(), 0, 5) == [0, 1]
    assert recommend.recent_likes(small(), 1, 1) == [0]


def test_recommendations_skip_the_history_and_mark_the_test_movie() -> None:
    p = small()
    assert recommend.recommend(p, Popularity(), 0, 3) == [2, 3, 4]
    text = recommend.describe(p, Popularity(), 10, k=2)
    assert "   1. Gamma (1992)   <- the held-out test movie" in text
    assert "Alpha" not in text.split("top 2")[1]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=60, n_items=90)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


def test_item_knn_lists_are_k_long_and_unseen(prepared: Prepared) -> None:
    history = prepared.history(include_val=True)
    for user in range(0, prepared.n_users, 7):
        top = recommend.recommend(prepared, ItemKNN(k=5), user, 10)
        assert len(top) == len(set(top)) == 10
        assert not set(top) & set(history[user].indices.tolist())


def test_the_recommend_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "60", "--items", "90"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert cli.main(["recommend", "synthetic", "--data-dir", str(tmp_path), "--user", "1"]) == 0
    assert "item-knn (k=3, shrink=10): top 10" in capsys.readouterr().out
    assert cli.main(["recommend", "synthetic", "--data-dir", str(tmp_path), "--user", "100000"]) == 1
    assert "not in the prepared data" in capsys.readouterr().err
