"""The two-tower retriever (DESIGN.md, "Day 2: two-tower retriever").

- **Item tower:** a movie-id embedding plus the mean of the movie's genre embeddings, a residual MLP, L2-normalised.
- **User tower:** the mean of the user's history embeddings (a table separate from the item tower's), a residual MLP,
  L2-normalised. There is no user-id embedding, so the tower works for any history, including one that grew a minute
  ago and one the model never saw.
- **Loss:** a softmax over the batch with in-batch negatives and temperature τ. A popular movie turns up as another
  example's negative more often than a rare one, so each logit is corrected by subtracting ``log q(i)``, the movie's
  share of the training targets (Yi et al., 2019). A movie that is the target of two examples in one batch is not a
  negative for either ("accidental hits" are masked).
- **Examples:** every training positive that has at least one earlier positive; its history is the user's earlier
  positives in the split's (time, movie id) order, the most recent ``max_history`` of them. No positive is ever in its
  own history, nor anything after it.

Training runs on the CPU with PyTorch; scoring exports both towers' outputs to NumPy (``Embeddings``), which is what
evaluation and, later, serving use.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from signalrec.prepare import Prepared

IntArray = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]


MIN_TEMPERATURE = 1e-3
MAX_LR = 1.0


class TrainingError(ValueError):
    """The data can't be trained on (too few examples, a movie index outside the catalogue)."""


@dataclass(frozen=True)
class Config:
    dim: int = 64
    hidden: int = 128
    temperature: float = 0.1
    logq: bool = True
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 3e-3
    max_history: int = 50
    seed: int = 0

    def check(self) -> None:
        for name in ("dim", "hidden", "epochs", "batch_size", "max_history"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2: in-batch negatives need another example")
        for name in ("temperature", "lr"):
            value = getattr(self, name)
            # Not a bool: True is an int, and "temperature": true is not 1.0 (review round 2). A NumPy float is a
            # float and fine (round 3: round 2's type() test refused it). An int too large for a float is refused,
            # not an OverflowError from isfinite (round 3).
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
            try:
                number = float(value)
            except OverflowError:
                number = math.inf
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
        # Round 3: a learning rate of 1e30 trained on NaN losses, as a tiny temperature had. Adam's step is about lr per
        # weight, and the embeddings are O(0.1), so anything above 1 is a typo, not a setting.
        if self.lr > MAX_LR:
            raise ValueError(f"lr must be at most {MAX_LR}, got {self.lr!r}")
        # Logits are cosines divided by τ, in float32. Round 1 found NaN losses at τ = 1e-39 and bounded 1/τ by
        # float32's maximum; round 2 found that bound too loose (3e-39 is a denormal and still NaN, 1e-38 gives infinite
        # losses). A sane floor instead: logits at most 2,000 apart, far past any useful temperature (0.05-0.2 here).
        if self.temperature < MIN_TEMPERATURE:
            raise ValueError(f"temperature must be at least {MIN_TEMPERATURE}, got {self.temperature!r}")
        if type(self.logq) is not bool:
            raise ValueError(f"logq must be True or False, got {self.logq!r}")
        # torch.manual_seed takes 0 <= seed < 2^64 (review round 1: 2^64 crashed after the whole grid had trained).
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError(f"seed must be an integer in [0, 2^64), got {self.seed!r}")

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class Sequences:
    """Every user's positives in (time, movie id) order, concatenated.

    User u's positives are ``items[starts[u]:starts[u + 1]]``.
    """

    items: IntArray
    starts: IntArray

    @property
    def n_users(self) -> int:
        return len(self.starts) - 1


def sequences(p: Prepared, include_val: bool) -> Sequences:
    """The training positives (and each user's validation item, last, if ``include_val``) as ``Sequences``.

    ``Prepared`` keeps training rows in (user, time, item) order and the validation item comes after all of a user's
    training rows in that order, so appending it keeps every sequence in time order.
    """
    users = p.train_users
    if len(users) and (np.diff(users) < 0).any():
        raise ValueError("training rows are not grouped by user in ascending order")
    counts = np.bincount(users, minlength=p.n_users).astype(np.int64)
    if include_val:
        counts += 1
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if not include_val:
        return Sequences(p.train_items.astype(np.int64), starts)
    items = np.empty(starts[-1], dtype=np.int64)
    val_positions = starts[1:] - 1
    is_val = np.zeros(len(items), dtype=bool)
    is_val[val_positions] = True
    items[is_val] = p.val_items
    items[~is_val] = p.train_items
    return Sequences(items, starts)


def example_positions(seq: Sequences) -> tuple[IntArray, IntArray]:
    """For every positive with at least one earlier positive: its position in ``seq.items`` and its user's start."""
    lengths = np.diff(seq.starts)
    start_of = np.repeat(seq.starts[:-1], lengths)
    positions = np.arange(len(seq.items), dtype=np.int64)
    keep = positions > start_of
    return positions[keep], start_of[keep]


def histories(items: IntArray, ends: IntArray, starts: IntArray, max_history: int, pad: int) -> IntArray:
    """Row r holds ``items[max(starts[r], ends[r] - max_history):ends[r]]``, oldest first, left-padded with ``pad``.

    ``ends[r]`` is exclusive: for a training example it is the example's own position, so the history stops just
    before it; for scoring a user it is one past their last positive.
    """
    offsets = np.arange(max_history, 0, -1, dtype=np.int64)
    idx = ends[:, None] - offsets[None, :]
    valid = idx >= starts[:, None]
    out: IntArray = np.where(valid, items[np.maximum(idx, 0)], pad).astype(np.int64)
    return out


def genre_matrix(genres: tuple[tuple[str, ...], ...]) -> tuple[Float32Array, tuple[str, ...]]:
    """Each movie's row averages its genres (1/n on each of its n genres); a movie with none has a zero row."""
    names = tuple(sorted({g for movie in genres for g in movie}))
    column = {name: c for c, name in enumerate(names)}
    matrix = np.zeros((len(genres), len(names)), dtype=np.float32)
    for row, movie in enumerate(genres):
        unique = set(movie)
        for g in unique:
            matrix[row, column[g]] = 1.0 / len(unique)
    return matrix, names


class _Residual(nn.Module):
    """x + MLP(x): the towers start close to their embeddings and learn a correction."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x + self.mlp(x)
        return out


class TwoTowerNet(nn.Module):
    def __init__(self, genres: Float32Array, dim: int, hidden: int) -> None:
        super().__init__()
        n_items, n_genres = genres.shape
        self.n_items: int = n_items
        self.item_ids = nn.Embedding(n_items, dim)
        self.genre_table = nn.Embedding(max(n_genres, 1), dim)
        self.register_buffer("genres", torch.from_numpy(np.ascontiguousarray(genres)))
        self.item_mlp = _Residual(dim, hidden)
        # Row n_items is the padding: always zero, never updated.
        self.history_table = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.user_mlp = _Residual(dim, hidden)
        for table in (self.item_ids, self.genre_table, self.history_table):
            nn.init.normal_(table.weight, std=0.1)
        with torch.no_grad():
            self.history_table.weight[n_items].zero_()

    @property
    def pad(self) -> int:
        return self.n_items

    def items(self, idx: torch.Tensor) -> torch.Tensor:
        genres: torch.Tensor = self.genres  # type: ignore[assignment]
        mixed = genres[idx] @ self.genre_table.weight if genres.shape[1] else 0.0
        x = self.item_ids(idx) + mixed
        return F.normalize(self.item_mlp(x), dim=-1)

    def users(self, history: torch.Tensor) -> torch.Tensor:
        present = (history != self.pad).to(torch.float32)
        total = self.history_table(history).sum(dim=1)
        mean = total / present.sum(dim=1, keepdim=True).clamp(min=1.0)
        return F.normalize(self.user_mlp(mean), dim=-1)


def in_batch_loss(
    users: torch.Tensor,
    items: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor | None,
) -> torch.Tensor:
    """Softmax cross-entropy where row r's positive is column r and every other column is a negative.

    ``log_q[c]`` (if given) is subtracted from column c's logit. A column holding the same movie as row r's target
    (another example of that movie in the batch) is masked out of row r: it is not a negative.
    """
    logits = users @ items.T / temperature
    if log_q is not None:
        logits = logits - log_q[None, :]
    n = len(targets)
    same = targets[None, :] == targets[:, None]
    diagonal = torch.eye(n, dtype=torch.bool)
    logits = logits.masked_fill(same & ~diagonal, float("-inf"))
    return F.cross_entropy(logits, torch.arange(n))


@dataclass(frozen=True)
class Embeddings:
    """Both towers' outputs as NumPy arrays: one row per user (from their latest history) and one per movie.

    Scores are dot products (cosine similarities: both sides are unit vectors) of the float32 vectors, computed in
    float64. Two movies can tie (identical vectors, or different ones at the same angle to the user); evaluation breaks
    a tie by movie index, as for every model.
    """

    users: Float32Array
    items: Float32Array
    name: str = "two-tower"

    def score(self, users: IntArray) -> Float64Array:
        out: Float64Array = self.users[users].astype(np.float64) @ self.items.T.astype(np.float64)
        return out


def embed(net: TwoTowerNet, seq: Sequences, max_history: int, batch: int = 4096) -> Embeddings:
    """Every movie's vector, and every user's from the last ``max_history`` positives of their sequence."""
    net.eval()
    with torch.no_grad():
        items = net.items(torch.arange(net.n_items)).numpy().astype(np.float32)
        ends = seq.starts[1:]
        starts = seq.starts[:-1]
        parts = []
        for lo in range(0, seq.n_users, batch):
            hist = histories(seq.items, ends[lo : lo + batch], starts[lo : lo + batch], max_history, net.pad)
            parts.append(net.users(torch.from_numpy(hist)).numpy().astype(np.float32))
        users = np.concatenate(parts) if parts else np.empty((0, items.shape[1]), dtype=np.float32)
    net.train()
    return Embeddings(users, items)


@dataclass(frozen=True)
class EpochStats:
    epoch: int
    loss: float
    seconds: float


def target_log_q(targets: IntArray, n_items: int) -> Float32Array:
    """log of each movie's share of the training targets (-inf for a movie that is never a target)."""
    counts = np.bincount(targets, minlength=n_items).astype(np.float64)
    with np.errstate(divide="ignore"):
        out: Float32Array = np.log(counts / counts.sum()).astype(np.float32)
    return out


def train(
    seq: Sequences,
    genres: Float32Array,
    config: Config,
    on_epoch: Callable[[EpochStats, TwoTowerNet], None] | None = None,
) -> tuple[TwoTowerNet, list[EpochStats]]:
    """Train on every example of ``seq`` for ``config.epochs`` epochs; ``on_epoch`` sees the model after each one.

    Deterministic for a seed on one machine: the initial weights come from ``torch.manual_seed(seed)`` and the
    shuffles from ``numpy.random.default_rng(seed)``.
    """
    config.check()
    if seq.items.size and (int(seq.items.min()) < 0 or int(seq.items.max()) >= genres.shape[0]):
        raise TrainingError("a sequence holds a movie index outside the genre matrix")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    net = TwoTowerNet(genres, config.dim, config.hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    positions, start_of = example_positions(seq)
    if len(positions) < 2:
        raise TrainingError("fewer than 2 training examples: in-batch negatives need a batch of at least 2")
    targets_all = seq.items[positions]
    log_q_all = torch.from_numpy(target_log_q(targets_all, net.n_items)) if config.logq else None
    stats: list[EpochStats] = []
    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(len(positions))
        total, batches = 0.0, 0
        for lo in range(0, len(order), config.batch_size):
            chosen = order[lo : lo + config.batch_size]
            if len(chosen) < 2:
                continue  # a batch of one has no negatives
            hist = histories(seq.items, positions[chosen], start_of[chosen], config.max_history, net.pad)
            targets = torch.from_numpy(targets_all[chosen])
            log_q = None if log_q_all is None else log_q_all[targets]
            loss = in_batch_loss(
                net.users(torch.from_numpy(hist)), net.items(targets), targets, config.temperature, log_q
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.item())
            batches += 1
        stat = EpochStats(epoch, total / max(batches, 1), time.perf_counter() - t0)
        stats.append(stat)
        if on_epoch is not None:
            on_epoch(stat, net)
    return net, stats
