"""scripts/cold_start.py: the hold-out, the replayed histories, the ranks by brute force, and the script end to end."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signalrec import cli, prepare
from signalrec import features as fe
from signalrec import twotower as tt
from signalrec.tower import UserTower

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cold_start.py"
sys.path.insert(0, str(SCRIPT.parent))
import cold_start  # noqa: E402


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("data")
    assert cli.main(["synthetic", "--data-dir", str(d), "--users", "80", "--items", "120"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(d)]) == 0
    return d


@pytest.fixture(scope="module")
def prepared(data_dir: Path) -> prepare.Prepared:
    return prepare.load(data_dir / "processed" / "synthetic", "synthetic")


def test_the_hold_out_is_a_fixed_share_of_ids() -> None:
    ids = np.arange(1, 20_001, dtype=np.int64)
    mask = cold_start.held_out(ids, 0.2)
    assert abs(mask.mean() - 0.2) < 0.01
    np.testing.assert_array_equal(mask, cold_start.held_out(ids, 0.2))  # a pure function of the id
    assert not (cold_start.held_out(ids, 0.1) & ~mask).any()  # a smaller share is a subset
    assert (cold_start.held_out(ids, 0.2, salt="other") != mask).any()
    for share in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            cold_start.held_out(ids, share)


def test_without_users_keeps_exactly_the_other_users(prepared: prepare.Prepared) -> None:
    drop = cold_start.held_out(prepared.user_raw_ids, 0.3)
    assert 0 < drop.sum() < prepared.n_users
    kept = cold_start.without_users(prepared, drop)
    assert kept.n_users == int((~drop).sum()) and kept.n_items == prepared.n_items
    np.testing.assert_array_equal(kept.user_raw_ids, prepared.user_raw_ids[~drop])
    before = fe.user_sequences(prepared, include_val=True)
    after = fe.user_sequences(kept, include_val=True)
    kept_users = np.flatnonzero(~drop)
    assert len(after) == len(kept_users)
    for new, old in enumerate(kept_users):
        np.testing.assert_array_equal(after[new], before[old])
    np.testing.assert_array_equal(kept.test_items, prepared.test_items[~drop])
    assert len(kept.train_users) == sum(len(before[u]) - 1 for u in kept_users)  # minus each validation item
    assert (np.diff(kept.train_users) >= 0).all()
    with pytest.raises(ValueError):
        cold_start.without_users(prepared, drop[:-1])


def test_known_keeps_the_last_h() -> None:
    seqs = [np.array([5, 6, 7], dtype=np.int64), np.array([9], dtype=np.int64)]
    assert [s.tolist() for s in cold_start.known(seqs, 0)] == [[], []]
    assert [s.tolist() for s in cold_start.known(seqs, 2)] == [[6, 7], [9]]
    assert [s.tolist() for s in cold_start.known(seqs, None)] == [[5, 6, 7], [9]]
    with pytest.raises(ValueError):
        cold_start.known(seqs, -1)


def brute_rank(scores: np.ndarray, excluded: set[int], target: int) -> int:
    order = sorted((m for m in range(len(scores)) if m not in excluded), key=lambda m: (-scores[m], m))
    return order.index(target) + 1


def random_tower(n_items: int, dim: int, seed: int) -> UserTower:
    rng = np.random.default_rng(seed)

    def f32(*shape: int) -> Any:
        return rng.normal(size=shape).astype(np.float32)

    table = f32(n_items + 1, dim)
    table[n_items] = 0
    return UserTower(table, f32(8, dim), f32(8), f32(dim, 8), f32(dim), max_history=2)


def test_two_tower_ranks_by_brute_force() -> None:
    n_items, dim = 30, 4
    tower = random_tower(n_items, dim, 1)
    rng = np.random.default_rng(2)
    items = rng.normal(size=(n_items, dim)).astype(np.float32)
    items[7] = items[3]  # an exact tie
    histories = [np.array([], dtype=np.int64), np.array([1, 2, 5], dtype=np.int64), np.array([4], dtype=np.int64)]
    targets = np.array([7, 3, 29], dtype=np.int64)
    got = cold_start.two_tower_ranks(tower, items, histories, targets)
    for i, (h, t) in enumerate(zip(histories, targets.tolist(), strict=True)):
        scores = items @ tower.vector(h)
        assert got[i] == brute_rank(scores, set(h.tolist()), t)


def test_popularity_ranks_by_brute_force_with_ties() -> None:
    popularity = np.array([5, 9, 9, 0, 5, 2, 9], dtype=np.int64)
    histories = [np.array([], dtype=np.int64), np.array([1], dtype=np.int64), np.array([1, 2], dtype=np.int64)]
    targets = np.array([2, 6, 4], dtype=np.int64)
    got = cold_start.popularity_ranks(popularity, histories, targets)
    want = [brute_rank(popularity.astype(float), set(h.tolist()), t) for h, t in zip(histories, targets, strict=True)]
    assert got.tolist() == want == [2, 2, 3]  # 6 (9), then 0 and 4 tied at 5: the smaller index first


def test_the_unpaired_interval() -> None:
    a, b = np.full(50, 0.3), np.full(70, 0.1)
    low, high = cold_start.unpaired_ci(a, b)
    assert low == pytest.approx(0.2) and high == pytest.approx(0.2)
    rng = np.random.default_rng(3)
    x, y = rng.normal(1.0, 1.0, 400), rng.normal(0.0, 1.0, 300)
    low, high = cold_start.unpaired_ci(x, y)
    assert low < x.mean() - y.mean() < high and low > 0


def test_the_held_out_users_shape_nothing(prepared: prepare.Prepared, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training sees exactly the other users' positives, and popularity counts only theirs."""
    seen: list[tt.Sequences] = []
    real_train = tt.train

    def spy(seq: tt.Sequences, *args: Any, **kwargs: Any) -> Any:
        seen.append(seq)
        return real_train(seq, *args, **kwargs)

    monkeypatch.setattr(tt, "train", spy)
    config = tt.Config(dim=8, hidden=16, epochs=1, batch_size=64, max_history=3)
    result = cold_start.measure(prepared, config, 0.3)
    drop = cold_start.held_out(prepared.user_raw_ids, 0.3)
    want = fe.user_sequences(prepared, include_val=True)
    (seq,) = seen
    trained = [seq.items[a:b] for a, b in zip(seq.starts[:-1], seq.starts[1:], strict=True)]
    assert len(trained) == int((~drop).sum())
    for got, user in zip(trained, np.flatnonzero(~drop), strict=True):
        np.testing.assert_array_equal(got, want[user])
    assert result["users"] == {"held out": int(drop.sum()), "trained on": int((~drop).sum())}


def test_the_script_end_to_end(data_dir: Path, tmp_path: Path) -> None:
    base = ["twotower", "synthetic", "--data-dir", str(data_dir), "--grid", "0.1:on", "--epochs", "2", "--seeds", "0"]
    source = tmp_path / "twotower.json"
    assert cli.main([*base, "--knn-grid", "5:0", "--histories", "3", "--quiet", "--out", str(source)]) == 0
    out = tmp_path / "cold-start.json"
    argv = ["synthetic", "--data-dir", str(data_dir), "--source", str(source), "--out", str(out)]
    assert cold_start.main(argv) == 0
    result = json.loads(out.read_text())
    rows = result["by known positives"]
    assert list(rows) == ["0", "1", "2", "3", "all"]
    p = prepare.load(data_dir / "processed" / "synthetic", "synthetic")
    assert result["users"]["held out"] + result["users"]["trained on"] == p.n_users
    for row in rows.values():
        for group in ("held out", "trained on"):
            for cell in row[group].values():
                assert 0 <= cell["value"] <= 1 and cell["ci95"][0] <= cell["value"] <= cell["ci95"][1]
    assert result["configuration"]["max_history"] == 3
    assert cold_start.main([*argv[:-1], str(tmp_path / "x.json"), "--share", "0.999999"]) == 1  # nobody to train on
