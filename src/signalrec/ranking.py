"""The ranking stage (DESIGN.md, "Day 3: FAISS, ranking, the feature store").

Retrieval returns each user's best ``n`` unseen movies by one inner product. The ranker re-orders those candidates with
features the retriever can't use cheaply, computed from a ``UserFeatures`` and the popularity array only, so offline
(``features.offline``) and online (``featurestore.FeatureStore``) inputs give the same feature vectors:

0. **retrieval score**: the candidate's inner product with the user vector.
1. **log retrieval rank**: ``log(1 + position)`` in the retrieved list.
2. **log popularity**: ``log(1 + positives)``.
3. **genre affinity**: the user's share of positives in each genre, averaged over the candidate's genres.
4. **year distance**: ``|year - the user's mean year| / 10``; 0 when either is unknown.
5. **year unknown**: 1 when the candidate or the user has no year, else 0.
6. **log user positives**: ``log(1 + positives)``, the user's activity.
7. **similarity to the last positive**: the candidate's item vector against the user's latest positive's.

The model scores each candidate from its standardised features: a linear function, or a one-hidden-layer MLP. It is
trained **listwise**: a softmax over each user's candidate list, the held-out movie as the label, so it learns to put
that movie first among the movies retrieval proposed. Trained with PyTorch, then exported to NumPy (``RankerWeights``):
serving runs two small matrix products and no torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from signalrec.features import NO_YEAR, Catalogue, IntArray, UserFeatures

Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]

FEATURES = (
    "retrieval score",
    "log retrieval rank",
    "log popularity",
    "genre affinity",
    "year distance",
    "year unknown",
    "log user positives",
    "similarity to the last positive",
)


def genre_mix(catalogue: Catalogue) -> Float64Array:
    """Each movie's row: 1 / (its genre count) on each of its genres; a movie with none has a zero row."""
    out = np.zeros((catalogue.n_items, len(catalogue.genre_names)), dtype=np.float64)
    for movie, genres in enumerate(catalogue.movie_genres):
        for g in genres:
            out[movie, g] = 1.0 / len(genres)
    return out


@dataclass(frozen=True)
class Context:
    """What feature computation needs besides the user: the catalogue, its genre mix, and the item vectors."""

    catalogue: Catalogue
    mix: Float64Array
    item_vectors: Float32Array

    @classmethod
    def build(cls, catalogue: Catalogue, item_vectors: Float32Array) -> Context:
        if item_vectors.shape[0] != catalogue.n_items:
            raise ValueError(f"{item_vectors.shape[0]} item vectors for {catalogue.n_items} movies")
        return cls(catalogue, genre_mix(catalogue), item_vectors)


def candidate_features(
    user: UserFeatures,
    candidates: IntArray,
    retrieval_scores: npt.NDArray[np.floating],
    popularity: IntArray,
    context: Context,
) -> Float64Array:
    """One row per candidate (in retrieved order), one column per name in ``FEATURES``."""
    if len(candidates) != len(retrieval_scores):
        raise ValueError(f"{len(candidates)} candidates, {len(retrieval_scores)} scores")
    n = len(candidates)
    out = np.zeros((n, len(FEATURES)), dtype=np.float64)
    if n == 0:
        return out
    out[:, 0] = np.asarray(retrieval_scores, dtype=np.float64)
    out[:, 1] = np.log1p(np.arange(n, dtype=np.float64))
    out[:, 2] = np.log1p(popularity[candidates].astype(np.float64))
    total = int(user.genre_counts.sum())
    if total:
        out[:, 3] = context.mix[candidates] @ (user.genre_counts.astype(np.float64) / total)
    years = context.catalogue.years[candidates]
    known = years != NO_YEAR
    if user.year_count:
        mean_year = user.year_sum / user.year_count
        out[:, 4] = np.where(known, np.abs(years - mean_year) / 10.0, 0.0)
        out[:, 5] = (~known).astype(np.float64)
    else:
        out[:, 5] = 1.0
    out[:, 6] = math.log1p(user.positives)
    if len(user.recent):
        last = context.item_vectors[int(user.recent[-1])].astype(np.float64)
        out[:, 7] = context.item_vectors[candidates].astype(np.float64) @ last
    return out


@dataclass(frozen=True)
class RankerWeights:
    """A trained ranker in NumPy: standardisation, then a linear layer (``hidden`` 0) or Linear → ReLU → Linear."""

    mean: Float64Array
    scale: Float64Array
    layers: tuple[tuple[Float64Array, Float64Array], ...]  # (weight (out, in), bias (out,)) per layer

    def score(self, features: Float64Array) -> Float64Array:
        x = (features - self.mean) / self.scale
        for i, (weight, bias) in enumerate(self.layers):
            x = x @ weight.T + bias
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        out: Float64Array = x[:, 0]
        return out

    def rerank(self, candidates: IntArray, features: Float64Array) -> IntArray:
        """The candidates by descending ranker score; ties keep the retrieved order."""
        order = np.argsort(-self.score(features), kind="stable")
        out: IntArray = candidates[order]
        return out


class RankerNet(nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        if hidden == 0:
            self.body: nn.Module = nn.Linear(n_features, 1)
        else:
            self.body = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.body(x).squeeze(-1)
        return out


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy over each list's real candidates (``mask``), the label's position as the class."""
    return F.cross_entropy(scores.masked_fill(~mask, float("-inf")), labels)


@dataclass(frozen=True)
class Lists:
    """Users' candidate lists, padded to one length: features (users, n, F), mask (users, n), label position per user
    (-1 when the held-out movie wasn't retrieved)."""

    features: Float64Array
    mask: npt.NDArray[np.bool_]
    labels: IntArray

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, rows: IntArray) -> Lists:
        return Lists(self.features[rows], self.mask[rows], self.labels[rows])


def standardiser(lists: Lists) -> tuple[Float64Array, Float64Array]:
    """Mean and scale of each feature over the real candidates; a constant feature gets scale 1."""
    rows = lists.features[lists.mask]
    mean = rows.mean(axis=0)
    scale = rows.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def export(net: RankerNet, mean: Float64Array, scale: Float64Array) -> RankerWeights:
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    layers = tuple(
        (m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in linears
    )
    return RankerWeights(mean, scale, layers)


@dataclass(frozen=True)
class RankerConfig:
    hidden: int = 16
    epochs: int = 10
    batch_lists: int = 256
    lr: float = 1e-2
    seed: int = 0

    def check(self) -> None:
        if type(self.hidden) is not int or self.hidden < 0:
            raise ValueError(f"hidden must be an integer >= 0, got {self.hidden!r}")
        for name in ("epochs", "batch_lists"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if isinstance(self.lr, bool) or not isinstance(self.lr, int | float) or not 0 < self.lr <= 1:
            raise ValueError(f"lr must be a number in (0, 1], got {self.lr!r}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError(f"seed must be an integer in [0, 2^64), got {self.seed!r}")


def train(
    lists: Lists, config: RankerConfig, on_epoch: list[RankerWeights] | None = None
) -> tuple[RankerWeights, list[float]]:
    """Train on the lists whose held-out movie was retrieved; ``on_epoch`` (if given) collects every epoch's export.

    Deterministic for a seed on one machine: weights from ``torch.manual_seed``, list order from NumPy's generator.
    """
    config.check()
    usable = np.flatnonzero(lists.labels >= 0)
    if len(usable) == 0:
        raise ValueError("no list contains its held-out movie: nothing to learn from")
    mean, scale = standardiser(lists.subset(usable))
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    net = RankerNet(lists.features.shape[2], config.hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    x_all = torch.from_numpy(((lists.features - mean) / scale).astype(np.float32))
    mask_all = torch.from_numpy(lists.mask)
    labels_all = torch.from_numpy(lists.labels)
    losses = []
    for _ in range(config.epochs):
        order = usable[rng.permutation(len(usable))]
        total, batches = 0.0, 0
        for lo in range(0, len(order), config.batch_lists):
            rows = torch.from_numpy(order[lo : lo + config.batch_lists])
            loss = listwise_loss(net(x_all[rows]), labels_all[rows], mask_all[rows])
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.item())
            batches += 1
        losses.append(total / batches)
        if on_epoch is not None:
            on_epoch.append(export(net, mean, scale))
    return export(net, mean, scale), losses


def ranks_after(weights: RankerWeights | None, lists: Lists) -> IntArray:
    """Each list's held-out movie's 1-based rank after re-ranking (retrieved order when ``weights`` is None); a movie
    that wasn't retrieved gets ``n + 1``, outside every cut-off up to the list length."""
    n = lists.features.shape[1]
    out = np.full(len(lists), n + 1, dtype=np.int64)
    for row in np.flatnonzero(lists.labels >= 0):
        real = int(lists.mask[row].sum())
        label = int(lists.labels[row])
        if weights is None:
            out[row] = label + 1
            continue
        scores = weights.score(lists.features[row, :real])
        # Rank = 1 + higher scores + equal scores earlier in the retrieved order (the stable sort's tie rule).
        target = scores[label]
        out[row] = 1 + int((scores > target).sum()) + int((scores[:label] == target).sum())
    return out
