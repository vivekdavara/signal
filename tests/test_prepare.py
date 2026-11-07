"""The pipeline: k-core, the split, indexing, and the on-disk format."""

from __future__ import annotations

import itertools
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from signalrec import movielens, prepare, synthetic
from signalrec.movielens import Dataset, Movie, Ratings
from signalrec.prepare import Params, PrepareError
from tests.conftest import real_raw_dir

# A module-level synthetic dataset for the property test (Hypothesis can't take function-scoped fixtures). It is
# loaded into memory, so its temporary directory is removed straight away.
with tempfile.TemporaryDirectory(prefix="signal-syn-") as _tmp:
    SYNTHETIC = movielens.load("synthetic", synthetic.write(Path(_tmp) / "raw", 80, 120))


def arr(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.int64)


rows = st.lists(st.tuples(st.integers(0, 3), st.integers(0, 3)), max_size=14, unique=True)


@settings(max_examples=300, deadline=None)
@given(rows, st.integers(1, 4), st.integers(1, 4))
def test_k_core_is_the_largest_row_set_meeting_both_bounds(
    pairs: list[tuple[int, int]], min_user: int, min_item: int
) -> None:
    users = arr([u for u, _ in pairs])
    items = arr([i for _, i in pairs])
    keep = prepare.k_core(users, items, min_user, min_item)

    # Brute force: a choice of users and items is valid if every chosen user and item keeps enough rows among the
    # rows they share; the k-core is the union of every valid choice (a union of valid choices is valid).
    best = np.zeros(len(pairs), dtype=bool)
    for us in itertools.product([False, True], repeat=4):
        for its in itertools.product([False, True], repeat=4):
            sel = np.array([us[u] and its[i] for u, i in pairs], dtype=bool)
            user_ok = all(sel[users == u].sum() >= min_user for u in range(4) if us[u])
            item_ok = all(sel[items == i].sum() >= min_item for i in range(4) if its[i])
            if user_ok and item_ok:
                best |= sel
    assert keep.tolist() == best.tolist()


def test_k_core_iterates_until_both_bounds_hold() -> None:
    # Item 9 has one row, so it goes in the first pass; that leaves user 2 with one row, so user 2 goes in the second.
    users = arr([1, 1, 2, 2, 3, 3])
    items = arr([7, 8, 8, 9, 7, 8])
    keep = prepare.k_core(users, items, 2, 2)
    kept = set(zip(users[keep].tolist(), items[keep].tolist(), strict=True))
    for u in {u for u, _ in kept}:
        assert sum(1 for x, _ in kept if x == u) >= 2
    for i in {i for _, i in kept}:
        assert sum(1 for _, x in kept if x == i) >= 2
    assert kept == {(1, 7), (1, 8), (3, 7), (3, 8)}


timed_rows = st.lists(
    st.tuples(st.integers(0, 4), st.integers(0, 30), st.integers(0, 3)),
    min_size=1,
    max_size=40,
    unique_by=lambda r: (r[0], r[1]),
)


@settings(max_examples=300, deadline=None)
@given(timed_rows)
def test_leave_last_out_holds_out_each_users_latest_two(data: list[tuple[int, int, int]]) -> None:
    users = arr([u for u, _, _ in data])
    items = arr([i for _, i, _ in data])
    times = arr([t for _, _, t in data])
    per_user = {u: sum(1 for x, _, _ in data if x == u) for u in set(users.tolist())}
    if min(per_user.values()) < 3:
        with pytest.raises(PrepareError):
            prepare.leave_last_out(users, items, times)
        return
    train, val, test = prepare.leave_last_out(users, items, times)
    assert sorted([*train, *val, *test]) == list(range(len(data)))
    assert users[val].tolist() == users[test].tolist() == sorted(per_user)
    for v, t in zip(val, test, strict=True):
        u = users[t]
        mine = sorted((int(times[r]), int(items[r]), int(r)) for r in range(len(data)) if users[r] == u)
        assert mine[-1][2] == t
        assert mine[-2][2] == v
    key = np.lexsort((items[train], times[train], users[train]))
    assert key.tolist() == list(range(len(train)))


def test_ties_in_time_break_by_movie_id() -> None:
    users = arr([1, 1, 1, 1])
    items = arr([40, 10, 30, 20])
    times = arr([5, 5, 5, 5])
    train, val, test = prepare.leave_last_out(users, items, times)
    assert (items[train].tolist(), items[val].tolist(), items[test].tolist()) == ([10, 20], [30], [40])


def dataset(rows: list[tuple[int, int, float, int]]) -> Dataset:
    ratings = Ratings(
        arr([r[0] for r in rows]),
        arr([r[1] for r in rows]),
        np.array([r[2] for r in rows], dtype=np.float32),
        arr([r[3] for r in rows]),
    )
    ids = sorted({r[1] for r in rows})
    return Dataset("hand", ratings, {i: Movie(i, f"Movie {i} (2000)", ("Drama",) if i % 2 else ()) for i in ids})


def hand_dataset() -> Dataset:
    # Users 10 and 20 like movies 100, 200, 300 (and 400 for user 20); user 30 likes too few; 3.5 is not a positive.
    return dataset(
        [
            (10, 100, 4.0, 1),
            (10, 200, 5.0, 2),
            (10, 300, 4.5, 3),
            (10, 400, 3.5, 4),
            (20, 300, 4.0, 1),
            (20, 100, 4.0, 2),
            (20, 200, 4.0, 3),
            (20, 400, 5.0, 3),
            (30, 100, 5.0, 9),
            (30, 400, 4.0, 9),
        ]
    )


def test_prepare_on_a_hand_made_dataset() -> None:
    p = prepare.prepare(hand_dataset(), Params(min_rating=4.0, min_user_positives=3, min_item_positives=2))
    # User 30 has 2 positives: dropped. Movie 400 then has only user 20's positive: dropped. User 20 keeps 3.
    assert p.user_raw_ids.tolist() == [10, 20]
    assert p.item_raw_ids.tolist() == [100, 200, 300]
    assert p.train_users.tolist() == [0, 1]
    assert p.train_items.tolist() == [0, 2]  # user 10: movie 100 at t=1; user 20: movie 300 at t=1
    assert p.val_items.tolist() == [1, 0]
    assert p.test_items.tolist() == [2, 1]
    assert p.titles == ("Movie 100 (2000)", "Movie 200 (2000)", "Movie 300 (2000)")
    assert prepare.counts(p) == {
        "users": 2,
        "items": 3,
        "train": 2,
        "val": 2,
        "test": 2,
        "items_without_training_positives": 1,
    }


def test_history_matrices() -> None:
    p = prepare.prepare(hand_dataset(), Params(min_rating=4.0, min_user_positives=3, min_item_positives=2))
    assert p.history(include_val=False).toarray().tolist() == [[1, 0, 0], [0, 0, 1]]
    assert p.history(include_val=True).toarray().tolist() == [[1, 1, 0], [1, 0, 1]]


def test_parameters_that_leave_nothing_are_refused() -> None:
    with pytest.raises(PrepareError, match="at least 3"):
        prepare.prepare(hand_dataset(), Params(min_user_positives=2))
    with pytest.raises(PrepareError, match="no positives survive"):
        prepare.prepare(hand_dataset(), Params(min_user_positives=50))


@pytest.fixture(scope="module")
def synthetic_prepared(tmp_path_factory: pytest.TempPathFactory) -> prepare.Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic")
    return prepare.prepare(movielens.load("synthetic", raw), Params())


def test_save_and_load_round_trip(tmp_path: Path, synthetic_prepared: prepare.Prepared) -> None:
    manifest = prepare.save(synthetic_prepared, tmp_path / "out", "synthetic", {"ratings.csv": "ab"}, Params())
    loaded = prepare.load(tmp_path / "out")
    for name in prepare.ARRAYS:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(synthetic_prepared, name))
        assert getattr(loaded, name).dtype == np.int64
    assert loaded.titles == synthetic_prepared.titles
    assert loaded.genres == synthetic_prepared.genres
    assert manifest == json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["counts"] == prepare.counts(synthetic_prepared)
    assert manifest["sources"] == {"ratings.csv": "ab"}


def test_the_same_inputs_give_the_same_bytes(tmp_path: Path, synthetic_prepared: prepare.Prepared) -> None:
    prepare.save(synthetic_prepared, tmp_path / "a", "synthetic", {}, Params())
    prepare.save(synthetic_prepared, tmp_path / "b", "synthetic", {}, Params())
    names = sorted(f.name for f in (tmp_path / "a").iterdir())
    assert names == sorted(f.name for f in (tmp_path / "b").iterdir())
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_a_changed_file_is_refused_on_load(tmp_path: Path, synthetic_prepared: prepare.Prepared) -> None:
    prepare.save(synthetic_prepared, tmp_path / "out", "synthetic", {}, Params())
    other = synthetic_prepared.test_items.copy()
    other[0] = (other[0] + 1) % synthetic_prepared.n_items
    np.save(tmp_path / "out" / "test_items.npy", other)
    with pytest.raises(PrepareError, match=r"test_items\.npy: SHA-256 differs"):
        prepare.load(tmp_path / "out")


def test_saving_replaces_an_older_directory(tmp_path: Path, synthetic_prepared: prepare.Prepared) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "stale.npy").write_bytes(b"old")
    prepare.save(synthetic_prepared, tmp_path / "out", "synthetic", {}, Params())
    assert not (tmp_path / "out" / "stale.npy").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_a_title_with_a_tab_is_refused(tmp_path: Path, synthetic_prepared: prepare.Prepared) -> None:
    bad = prepare.Prepared(**{**synthetic_prepared.__dict__, "titles": ("A\tB", *synthetic_prepared.titles[1:])})
    with pytest.raises(PrepareError, match="tab, line break or other control character"):
        prepare.save(bad, tmp_path / "out", "synthetic", {}, Params())
    assert list(tmp_path.iterdir()) == []


# Pinned from the first run on 2026-11-07 (Params() defaults); a pure function of the pinned archives, so a change
# here means the pipeline changed.
REAL_COUNTS = {
    "ml-latest-small": {
        "users": 601,
        "items": 1955,
        "train": 40025,
        "val": 601,
        "test": 601,
        "items_without_training_positives": 0,
    },
    "ml-1m": {
        "users": 6034,
        "items": 3125,
        "train": 562308,
        "val": 6034,
        "test": 6034,
        "items_without_training_positives": 0,
    },
}


@pytest.mark.parametrize("name", sorted(REAL_COUNTS))
def test_the_real_datasets_prepare_to_their_pinned_counts(name: str) -> None:
    p = prepare.prepare(movielens.load(name, real_raw_dir(name)), Params())
    assert prepare.counts(p) == REAL_COUNTS[name]
    assert (p.val_times <= p.test_times).all()


def test_replacing_a_directory_leaves_only_the_new_one(tmp_path: Path, synthetic_prepared: prepare.Prepared) -> None:
    prepare.save(synthetic_prepared, tmp_path / "out", "first", {}, Params())
    prepare.save(synthetic_prepared, tmp_path / "out", "second", {}, Params())
    assert prepare.read_manifest(tmp_path / "out")["dataset"] == "second"
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_a_missing_or_corrupt_manifest_is_a_prepare_error(tmp_path: Path) -> None:
    with pytest.raises(PrepareError, match="missing"):
        prepare.load(tmp_path)
    (tmp_path / "manifest.json").write_text("[1, 2]")
    with pytest.raises(PrepareError, match="not a JSON object"):
        prepare.load(tmp_path)
    (tmp_path / "manifest.json").write_text('{"form')
    with pytest.raises(PrepareError, match="unreadable"):
        prepare.load(tmp_path)


def test_movies_without_genres_round_trip(tmp_path: Path) -> None:
    p = prepare.prepare(hand_dataset(), Params(min_rating=4.0, min_user_positives=3, min_item_positives=2))
    assert () in p.genres
    prepare.save(p, tmp_path / "out", "hand", {}, Params())
    assert prepare.load(tmp_path / "out").genres == p.genres


def test_a_title_with_a_line_break_is_refused(tmp_path: Path, synthetic_prepared: prepare.Prepared) -> None:
    bad = prepare.Prepared(**{**synthetic_prepared.__dict__, "titles": ("A\nB", *synthetic_prepared.titles[1:])})
    with pytest.raises(PrepareError, match="control character"):
        prepare.save(bad, tmp_path / "out", "synthetic", {}, Params())


def test_an_item_bound_below_one_is_refused() -> None:
    with pytest.raises(PrepareError, match="min_item_positives"):
        Params(min_item_positives=0).check()


def test_a_repeated_pair_in_a_history_is_refused() -> None:
    with pytest.raises(PrepareError, match="appears twice"):
        prepare.interaction_matrix(arr([0, 0]), arr([1, 1]), 1, 2)


@settings(max_examples=10, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_the_prepared_data_does_not_depend_on_the_row_order(seed: int) -> None:
    # MovieLens files happen to be sorted by user; nothing in the pipeline may rely on it.
    ratings = SYNTHETIC.ratings
    idx = np.random.default_rng(seed).permutation(len(ratings))
    shuffled = Dataset(
        "synthetic",
        Ratings(ratings.users[idx], ratings.items[idx], ratings.ratings[idx], ratings.times[idx]),
        SYNTHETIC.movies,
    )
    a = prepare.prepare(SYNTHETIC, Params())
    b = prepare.prepare(shuffled, Params())
    for name in prepare.ARRAYS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.titles == b.titles


def test_a_manifest_that_cannot_be_read_is_a_prepare_error(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").mkdir()  # reading it raises IsADirectoryError, an OSError
    with pytest.raises(PrepareError, match="unreadable"):
        prepare.load(tmp_path)


@pytest.mark.parametrize("char", ["\x00", "\x01", "\x1f", "\x7f"])
def test_any_control_character_in_a_title_is_refused(
    tmp_path: Path, synthetic_prepared: prepare.Prepared, char: str
) -> None:
    bad = prepare.Prepared(**{**synthetic_prepared.__dict__, "titles": (f"A{char}B", *synthetic_prepared.titles[1:])})
    with pytest.raises(PrepareError, match="control character"):
        prepare.save(bad, tmp_path / "out", "synthetic", {}, Params())


def test_load_checks_the_manifest_against_the_arrays_and_the_dataset(
    tmp_path: Path, synthetic_prepared: prepare.Prepared
) -> None:
    prepare.save(synthetic_prepared, tmp_path / "out", "synthetic", {"ratings.csv": "ab"}, Params())
    prepare.load(tmp_path / "out", "synthetic")
    with pytest.raises(PrepareError, match="holds 'synthetic', not 'ml-1m'"):
        prepare.load(tmp_path / "out", "ml-1m")
    path = tmp_path / "out" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"]["train"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(PrepareError, match="disagree with the arrays"):
        prepare.load(tmp_path / "out")
