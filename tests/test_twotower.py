"""The two-tower model's pieces: sequences, examples and histories, the genre mix, the loss, the towers, training."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from signalrec import movielens, prepare, synthetic
from signalrec import twotower as tt
from signalrec.prepare import Prepared


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=120, n_items=150)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


# --- sequences -----------------------------------------------------------------------------------------------------


def test_training_sequences_are_each_users_training_rows_in_order(prepared: Prepared) -> None:
    seq = tt.sequences(prepared, include_val=False)
    assert seq.n_users == prepared.n_users
    for u in range(prepared.n_users):
        want = prepared.train_items[prepared.train_users == u]
        assert seq.items[seq.starts[u] : seq.starts[u + 1]].tolist() == want.tolist()


def test_with_validation_each_users_validation_item_comes_last(prepared: Prepared) -> None:
    seq = tt.sequences(prepared, include_val=True)
    for u in range(prepared.n_users):
        run = seq.items[seq.starts[u] : seq.starts[u + 1]].tolist()
        assert run[-1] == prepared.val_items[u]
        assert run[:-1] == prepared.train_items[prepared.train_users == u].tolist()
        assert prepared.test_items[u] not in run


def test_sequences_refuse_rows_not_grouped_by_user(prepared: Prepared) -> None:
    shuffled = Prepared(
        **{
            **{name: getattr(prepared, name) for name in prepare.ARRAYS},
            "train_users": prepared.train_users[::-1].copy(),
            "train_items": prepared.train_items[::-1].copy(),
            "train_times": prepared.train_times[::-1].copy(),
        },
        titles=prepared.titles,
        genres=prepared.genres,
    )
    with pytest.raises(ValueError, match="not grouped by user"):
        tt.sequences(shuffled, include_val=False)


# --- examples and histories ----------------------------------------------------------------------------------------


@st.composite
def sequence_sets(draw: st.DrawFn) -> tuple[tt.Sequences, int]:
    lengths = draw(st.lists(st.integers(0, 12), min_size=1, max_size=8))
    items = np.array(draw(st.lists(st.integers(0, 30), min_size=sum(lengths), max_size=sum(lengths))), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return tt.Sequences(items, starts), draw(st.integers(1, 6))


@given(sequence_sets())
@settings(max_examples=200, deadline=None)
def test_each_example_sees_exactly_its_most_recent_earlier_positives(case: tuple[tt.Sequences, int]) -> None:
    seq, max_history = case
    pad = 99
    positions, start_of = tt.example_positions(seq)
    rows = tt.histories(seq.items, positions, start_of, max_history, pad)
    # Brute force: walk every user's run; every position after the first is an example.
    want_positions, want_rows = [], []
    for u in range(seq.n_users):
        lo, hi = int(seq.starts[u]), int(seq.starts[u + 1])
        for g in range(lo + 1, hi):
            earlier = seq.items[max(lo, g - max_history) : g].tolist()
            want_positions.append(g)
            want_rows.append([pad] * (max_history - len(earlier)) + earlier)
    assert positions.tolist() == want_positions
    assert rows.tolist() == want_rows


def test_a_users_first_positive_is_not_an_example() -> None:
    seq = tt.Sequences(np.array([5, 6, 7, 8], dtype=np.int64), np.array([0, 1, 4], dtype=np.int64))
    positions, start_of = tt.example_positions(seq)
    assert positions.tolist() == [2, 3]  # user 0 has one positive; user 1's first (position 1) has no history
    assert start_of.tolist() == [1, 1]


def test_no_example_has_its_target_or_anything_later_in_its_history(prepared: Prepared) -> None:
    seq = tt.sequences(prepared, include_val=True)
    positions, start_of = tt.example_positions(seq)
    rows = tt.histories(seq.items, positions, start_of, 50, -1)
    for g, row in zip(positions, rows, strict=True):
        later = set(seq.items[g : seq.starts[np.searchsorted(seq.starts, g, side="right")]].tolist())
        assert later.isdisjoint(row.tolist())


def test_scoring_histories_end_with_the_users_last_positive() -> None:
    seq = tt.Sequences(np.array([1, 2, 3, 4, 5], dtype=np.int64), np.array([0, 3, 5], dtype=np.int64))
    rows = tt.histories(seq.items, seq.starts[1:], seq.starts[:-1], 2, 0)
    assert rows.tolist() == [[2, 3], [4, 5]]


# --- genres --------------------------------------------------------------------------------------------------------


def test_genre_rows_average_their_genres() -> None:
    matrix, names = tt.genre_matrix((("Drama", "Comedy"), (), ("Drama",), ("Drama", "Drama")))
    assert names == ("Comedy", "Drama")
    assert matrix.tolist() == [[0.5, 0.5], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]


# --- the loss ------------------------------------------------------------------------------------------------------


def _manual_loss(u: np.ndarray, v: np.ndarray, targets: list[int], tau: float, log_q: np.ndarray | None) -> float:
    total = 0.0
    for r in range(len(targets)):
        logits = []
        for c in range(len(targets)):
            if c != r and targets[c] == targets[r]:
                continue
            logit = float(u[r] @ v[c]) / tau - (0.0 if log_q is None else float(log_q[c]))
            logits.append((c, logit))
        top = max(x for _, x in logits)
        log_sum = top + np.log(sum(np.exp(x - top) for _, x in logits))
        total += log_sum - dict(logits)[r]
    return total / len(targets)


@pytest.mark.parametrize("use_log_q", [False, True])
def test_the_loss_matches_a_written_out_softmax(use_log_q: bool) -> None:
    rng = np.random.default_rng(3)
    u = rng.normal(size=(5, 4))
    v = rng.normal(size=(5, 4))
    targets = [7, 2, 7, 9, 4]  # 7 twice: an accidental hit for rows 0 and 2
    log_q = np.log(np.array([0.3, 0.1, 0.3, 0.05, 0.2])) if use_log_q else None
    got = tt.in_batch_loss(
        torch.from_numpy(u),
        torch.from_numpy(v),
        torch.tensor(targets),
        0.5,
        None if log_q is None else torch.from_numpy(log_q),
    )
    assert float(got) == pytest.approx(_manual_loss(u, v, targets, 0.5, log_q), rel=1e-12)


def test_log_q_lowers_a_popular_columns_logit() -> None:
    # Two rows whose user vectors match both items equally: without logQ the loss is symmetric; with it, the popular
    # item's column is pushed down, so the row whose target is the rare item gains and the other loses.
    u = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    v = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1])
    plain = tt.in_batch_loss(u, v, targets, 1.0, None)
    assert float(plain) == pytest.approx(np.log(2))
    corrected = tt.in_batch_loss(u, v, targets, 1.0, torch.log(torch.tensor([0.9, 0.1])))
    # Row 0 (target popular): logits 1 - log .9, 1 - log .1; row 1 (target rare): the same two logits.
    a, b = 1 - np.log(0.9), 1 - np.log(0.1)
    want = ((np.logaddexp(a, b) - a) + (np.logaddexp(a, b) - b)) / 2
    assert float(corrected) == pytest.approx(want)


def test_an_accidental_hit_is_not_a_negative() -> None:
    # Rows 0 and 1 share their target, so each loses the other's column: its own logit 1 against one logit 0. Row 2
    # keeps both negatives: its logit 1 against two logits 0.
    u = torch.eye(3)
    v = torch.eye(3)
    loss = tt.in_batch_loss(u, v, torch.tensor([4, 4, 5]), 1.0, None)
    shared = float(np.log(np.e + 1) - 1)
    alone = float(np.log(np.e + 2) - 1)
    assert float(loss) == pytest.approx((2 * shared + alone) / 3)


# --- the towers ----------------------------------------------------------------------------------------------------


def _net(n_items: int = 6, seed: int = 0) -> tt.TwoTowerNet:
    torch.manual_seed(seed)
    genres, _ = tt.genre_matrix(tuple(("A",) if i % 2 else ("B", "C") for i in range(n_items)))
    return tt.TwoTowerNet(genres, dim=8, hidden=16)


def test_both_towers_output_unit_vectors() -> None:
    net = _net()
    items = net.items(torch.arange(6))
    users = net.users(torch.tensor([[0, 1, 6], [6, 6, 2]]))
    assert torch.allclose(items.norm(dim=1), torch.ones(6))
    assert torch.allclose(users.norm(dim=1), torch.ones(2))


def test_the_user_tower_ignores_padding_and_order() -> None:
    net = _net()
    pad = net.pad
    a = net.users(torch.tensor([[1, 2, 3]]))
    b = net.users(torch.tensor([[pad, pad, 3, 2, 1]]))
    assert torch.allclose(a, b, atol=1e-6)


def test_an_empty_history_is_a_finite_vector() -> None:
    net = _net()
    out = net.users(torch.tensor([[net.pad, net.pad]]))
    assert torch.isfinite(out).all()


def test_a_movie_without_genres_gets_no_genre_vector() -> None:
    torch.manual_seed(0)
    genres, _ = tt.genre_matrix((("A",), ()))
    net = tt.TwoTowerNet(genres, dim=4, hidden=8)
    with torch.no_grad():
        net.genre_table.weight.fill_(100.0)
        net.item_ids.weight.zero_()
        net.item_ids.weight[1, 0] = 1.0
    out = net.items(torch.tensor([1]))
    assert torch.allclose(out, F.normalize(net.item_mlp(net.item_ids.weight[1:2]), dim=-1))


# --- training ------------------------------------------------------------------------------------------------------


def _clusters(n_users: int = 60, per_cluster: int = 15, length: int = 12, seed: int = 0) -> tt.Sequences:
    """Users 0..n/2-1 like only movies 0..per_cluster-1, the rest only per_cluster..2*per_cluster-1."""
    rng = np.random.default_rng(seed)
    runs = []
    for u in range(n_users):
        base = 0 if u < n_users // 2 else per_cluster
        runs.append(base + rng.choice(per_cluster, size=length, replace=False))
    items = np.concatenate(runs).astype(np.int64)
    starts = np.arange(0, length * n_users + 1, length, dtype=np.int64)
    return tt.Sequences(items, starts)


def _no_genres(n_items: int) -> np.ndarray:
    return np.zeros((n_items, 0), dtype=np.float32)


def test_training_learns_planted_clusters() -> None:
    seq = _clusters()
    config = tt.Config(dim=8, hidden=16, epochs=30, batch_size=32, lr=1e-2, temperature=0.1, seed=1)
    net, stats = tt.train(seq, _no_genres(30), config)
    assert stats[-1].loss < stats[0].loss
    emb = tt.embed(net, seq, config.max_history)
    scores = emb.score(np.arange(seq.n_users))
    top = np.argsort(-scores, axis=1)[:, :10]
    own = np.where(np.arange(seq.n_users)[:, None] < 30, top < 15, top >= 15)
    assert own.mean() > 0.95


def test_training_is_deterministic_for_a_seed() -> None:
    seq = _clusters()
    config = tt.Config(dim=8, hidden=16, epochs=2, batch_size=16, seed=5)
    a, stats_a = tt.train(seq, _no_genres(30), config)
    b, stats_b = tt.train(seq, _no_genres(30), config)
    c, _ = tt.train(seq, _no_genres(30), tt.Config(dim=8, hidden=16, epochs=2, batch_size=16, seed=6))
    for (name, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(x, y), name
    assert [s.loss for s in stats_a] == [s.loss for s in stats_b]
    assert not all(torch.equal(x, z) for x, z in zip(a.state_dict().values(), c.state_dict().values(), strict=True))


def test_the_padding_row_stays_zero_through_training() -> None:
    seq = _clusters()
    net, _ = tt.train(seq, _no_genres(30), tt.Config(dim=8, hidden=16, epochs=2, batch_size=16))
    assert not net.history_table.weight[net.pad].any()


def test_on_epoch_sees_every_epoch() -> None:
    seen: list[int] = []
    tt.train(
        _clusters(),
        _no_genres(30),
        tt.Config(dim=4, hidden=4, epochs=3, batch_size=64),
        on_epoch=lambda s, _: seen.append(s.epoch),
    )
    assert seen == [1, 2, 3]


def test_log_q_is_each_targets_share() -> None:
    log_q = tt.target_log_q(np.array([0, 0, 0, 2], dtype=np.int64), 4)
    assert np.exp(log_q).tolist() == pytest.approx([0.75, 0.0, 0.25, 0.0])


@pytest.mark.parametrize(
    "change",
    [
        {"dim": 0},
        {"hidden": -1},
        {"epochs": 0},
        {"batch_size": 1},
        {"max_history": 0},
        {"dim": True},
        {"temperature": 0.0},
        {"temperature": float("nan")},
        {"lr": float("inf")},
        {"lr": -1.0},
    ],
)
def test_bad_configs_are_refused(change: dict[str, object]) -> None:
    config = tt.Config(**change)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        config.check()


def test_training_refuses_movies_outside_the_genre_matrix() -> None:
    seq = tt.Sequences(np.array([0, 1, 9], dtype=np.int64), np.array([0, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="outside the genre matrix"):
        tt.train(seq, _no_genres(5), tt.Config(epochs=1, batch_size=2))


def test_training_needs_two_examples() -> None:
    seq = tt.Sequences(np.array([0, 1], dtype=np.int64), np.array([0, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="fewer than 2"):
        tt.train(seq, _no_genres(2), tt.Config(epochs=1, batch_size=2))
