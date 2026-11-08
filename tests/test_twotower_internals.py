"""What training actually feeds the loss, and what each tower computes, checked from outside.

Review round 1 (day 2) listed mutants that every other test survived, because those tests recompute results through
the same ``train`` and ``embed``: a target leaking into its own history, the logQ flag inverted, the temperature or
the accidental-hit mask ignored in training, a batch of one trained, scoring without each user's last positive, genres
ignored, the user tower reading the item table. Each test here kills at least one of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from signalrec import twotower as tt


def _sequences() -> tt.Sequences:
    # Five users, runs of different lengths, items repeated across users (so a target's movie alone can't tell where
    # its history should end).
    runs = [[3, 1, 4, 1 + 4, 9], [2, 6, 5], [3, 5, 8, 9, 7, 9 - 6, 2], [0, 1], [4, 6, 8, 0, 2, 7]]
    items = np.array([i for run in runs for i in run], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum([len(r) for r in runs])]).astype(np.int64)
    return tt.Sequences(items, starts)


def _genres(n_items: int = 10) -> np.ndarray:
    names = ("A", "B", "C")
    return tt.genre_matrix(tuple(tuple(names[: 1 + i % 3]) if i % 4 else () for i in range(n_items)))[0]


@pytest.mark.parametrize("logq", [True, False])
def test_each_batch_feeds_the_loss_its_examples_and_their_own_histories(
    logq: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    seq = _sequences()
    positions, start_of = tt.example_positions(seq)
    n_examples = len(positions)  # 18: with a batch of 4, the last batch holds 2
    config = tt.Config(dim=4, hidden=4, epochs=2, batch_size=4, temperature=0.37, logq=logq, max_history=3)
    calls: list[tuple[str, Any]] = []
    real_histories, real_loss = tt.histories, tt.in_batch_loss

    def histories_spy(items: Any, ends: Any, starts: Any, max_history: int, pad: int) -> Any:
        calls.append(("histories", (ends.copy(), starts.copy(), max_history)))
        return real_histories(items, ends, starts, max_history, pad)

    def loss_spy(users: Any, items: Any, targets: Any, temperature: float, log_q: Any) -> Any:
        calls.append(("loss", (targets.clone(), temperature, None if log_q is None else log_q.clone())))
        return real_loss(users, items, targets, temperature, log_q)

    monkeypatch.setattr(tt, "histories", histories_spy)
    monkeypatch.setattr(tt, "in_batch_loss", loss_spy)
    tt.train(seq, _genres(), config)

    log_q_all = tt.target_log_q(seq.items[positions], 10)
    seen_per_epoch: list[list[int]] = [[], []]
    batches = list(zip(calls[0::2], calls[1::2], strict=True))
    assert len(batches) == 2 * ((n_examples + 3) // 4)
    for b, ((kind_h, (ends, starts, max_history)), (kind_l, (targets, temperature, log_q))) in enumerate(batches):
        assert (kind_h, kind_l) == ("histories", "loss")
        assert max_history == 3
        assert len(targets) >= 2
        # The history of each example ends just before the example itself: its target is the movie at `ends`.
        assert targets.tolist() == seq.items[ends].tolist()
        assert set(ends.tolist()) <= set(positions.tolist())
        assert starts.tolist() == start_of[np.searchsorted(positions, ends)].tolist()
        assert temperature == 0.37
        if logq:
            assert log_q is not None
            assert log_q.tolist() == pytest.approx(log_q_all[targets.numpy()].tolist())
        else:
            assert log_q is None
        seen_per_epoch[b // (len(batches) // 2)].extend(ends.tolist())
    for seen in seen_per_epoch:  # every example once per epoch
        assert sorted(seen) == positions.tolist()


def test_a_final_batch_of_one_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = _sequences()
    n_examples = len(tt.example_positions(seq)[0])
    batch_size = n_examples - 1  # one full batch, then a batch of one
    sizes: list[int] = []
    real_loss = tt.in_batch_loss

    def loss_spy(users: Any, items: Any, targets: Any, temperature: float, log_q: Any) -> Any:
        sizes.append(len(targets))
        return real_loss(users, items, targets, temperature, log_q)

    monkeypatch.setattr(tt, "in_batch_loss", loss_spy)
    tt.train(seq, _genres(), tt.Config(dim=4, hidden=4, epochs=1, batch_size=batch_size))
    assert sizes == [batch_size]


def test_scoring_uses_each_users_last_positives_including_the_last() -> None:
    seq = _sequences()
    torch.manual_seed(0)
    net = tt.TwoTowerNet(_genres(), dim=6, hidden=5)
    emb = tt.embed(net, seq, max_history=3)
    for u in range(seq.n_users):
        run = seq.items[seq.starts[u] : seq.starts[u + 1]].tolist()
        recent = run[-3:]
        hist = torch.tensor([[net.pad] * (3 - len(recent)) + recent])
        with torch.no_grad():
            want = net.users(hist).numpy()[0]
        assert np.allclose(emb.users[u], want, atol=1e-6)
    with torch.no_grad():
        assert np.allclose(emb.items, net.items(torch.arange(10)).numpy(), atol=1e-6)


def test_the_item_tower_is_id_plus_genre_mean_through_its_mlp() -> None:
    torch.manual_seed(1)
    genres = _genres()
    net = tt.TwoTowerNet(genres, dim=6, hidden=5)
    with torch.no_grad():
        got = net.items(torch.arange(10))
        mixed = torch.from_numpy(genres) @ net.genre_table.weight
        x = net.item_ids.weight + mixed
        want = F.normalize(x + net.item_mlp.mlp(x), dim=-1)
    assert torch.allclose(got, want, atol=1e-6)
    assert mixed.abs().sum() > 0  # so a tower that ignored genres would differ


def test_the_user_tower_reads_its_own_table_not_the_item_table() -> None:
    torch.manual_seed(2)
    net = tt.TwoTowerNet(_genres(), dim=6, hidden=5)
    history = torch.tensor([[net.pad, 1, 4, 4], [0, 2, 3, 9]])
    with torch.no_grad():
        got = net.users(history)
        rows = []
        for h in history.tolist():
            kept = [i for i in h if i != net.pad]
            mean = net.history_table.weight[kept].mean(dim=0)
            rows.append(F.normalize(mean + net.user_mlp.mlp(mean), dim=-1))
        want = torch.stack(rows)
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(net.history_table.weight[:10], net.item_ids.weight)
