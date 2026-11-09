"""ranking.py: features by hand, the NumPy export against PyTorch, the rank rules and the listwise loss."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from signalrec import features as fe
from signalrec import ranking as rk

# Three genres, five movies: movie 3 has no genre, movie 4 no year.
CATALOGUE = fe.Catalogue(
    ("Comedy", "Drama", "War"),
    ((0,), (0, 1), (1, 2), (), (2,)),
    np.array([1990, 2000, 1980, 1995, fe.NO_YEAR], dtype=np.int64),
)
VECTORS = np.array([[1, 0], [0, 1], [0.6, 0.8], [-1, 0], [0.8, -0.6]], dtype=np.float32)
POPULARITY = np.array([10, 0, 3, 1, 7], dtype=np.int64)


def test_candidate_features_by_hand() -> None:
    # The user liked movies 0 then 1: Comedy 2, Drama 1; years 1990 and 2000 (mean 1995); last positive is movie 1.
    user = fe.user_features(np.array([0, 1], dtype=np.int64), CATALOGUE)
    context = rk.Context.build(CATALOGUE, VECTORS)
    candidates = np.array([2, 4, 3], dtype=np.int64)
    got = rk.candidate_features(user, candidates, np.array([0.5, 0.25, -0.125]), POPULARITY, context)
    share = np.array([2, 1, 0]) / 3
    want = np.array(
        [
            # score, log rank, log popularity, genre affinity, year distance, year unknown, log positives, last sim
            [0.5, 0.0, math.log(4), (share[1] + share[2]) / 2, 15 / 10, 0, math.log(3), 0.8],
            [0.25, math.log(2), math.log(8), share[2], 0, 1, math.log(3), -0.6],
            [-0.125, math.log(3), math.log(2), 0, 0, 0, math.log(3), 0],
        ]
    )
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
    assert got.shape == (3, len(rk.FEATURES))


def test_features_of_a_user_with_nothing() -> None:
    nobody = fe.user_features(np.empty(0, dtype=np.int64), CATALOGUE)
    context = rk.Context.build(CATALOGUE, VECTORS)
    got = rk.candidate_features(nobody, np.array([0, 1], dtype=np.int64), np.array([0.1, 0.0]), POPULARITY, context)
    assert got[:, 3].tolist() == [0, 0]  # no genre counts
    assert got[:, 4].tolist() == [0, 0] and got[:, 5].tolist() == [1, 1]  # no years: distance 0, unknown 1
    assert got[:, 6].tolist() == [0, 0] and got[:, 7].tolist() == [0, 0]
    assert rk.candidate_features(nobody, np.empty(0, dtype=np.int64), np.empty(0), POPULARITY, context).shape == (0, 8)


def test_mismatched_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="item vectors"):
        rk.Context.build(CATALOGUE, VECTORS[:3])
    user = fe.user_features(np.array([0], dtype=np.int64), CATALOGUE)
    with pytest.raises(ValueError, match="candidates"):
        rk.candidate_features(user, np.array([0, 1]), np.array([0.1]), POPULARITY, rk.Context.build(CATALOGUE, VECTORS))


def _random_lists(rng: np.random.Generator, users: int = 40, n: int = 12) -> rk.Lists:
    features = rng.normal(size=(users, n, len(rk.FEATURES)))
    lengths = rng.integers(1, n + 1, size=users)
    mask = np.arange(n)[None, :] < lengths[:, None]
    features[~mask] = 0.0
    labels = np.array([int(rng.integers(-1, length)) for length in lengths], dtype=np.int64)
    return rk.Lists(features, mask, labels)


@pytest.mark.parametrize("hidden", [0, 8])
def test_the_numpy_export_scores_like_the_torch_model(hidden: int) -> None:
    torch.manual_seed(1)
    net = rk.RankerNet(len(rk.FEATURES), hidden)
    rng = np.random.default_rng(2)
    mean, scale = rng.normal(size=8), rng.uniform(0.5, 2, size=8)
    weights = rk.export(net, mean, scale)
    x = rng.normal(size=(30, 8))
    with torch.no_grad():
        want = net(torch.from_numpy(((x - mean) / scale).astype(np.float32))).numpy()
    np.testing.assert_allclose(weights.score(x), want, rtol=1e-5, atol=1e-6)
    assert len(weights.layers) == (1 if hidden == 0 else 2)


def test_listwise_loss_is_a_softmax_over_the_real_candidates() -> None:
    scores = torch.tensor([[2.0, 1.0, 5.0], [0.5, -1.0, 3.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    labels = torch.tensor([1, 2])
    row0 = -(1.0 - math.log(math.exp(2.0) + math.exp(1.0)))  # the padded 5.0 doesn't count
    row1 = -(3.0 - math.log(math.exp(0.5) + math.exp(-1.0) + math.exp(3.0)))
    assert float(rk.listwise_loss(scores, labels, mask)) == pytest.approx((row0 + row1) / 2, rel=1e-6)


class _Fixed(rk.RankerWeights):
    """A ranker whose score is one feature column, to test the rank rules."""

    column: int = 0

    def __init__(self, column: int) -> None:
        super().__init__(np.zeros(8), np.ones(8), ())
        object.__setattr__(self, "column", column)

    def score(self, features: rk.Float64Array) -> rk.Float64Array:
        out: rk.Float64Array = features[:, self.column]
        return out


def test_ranks_after_reranking() -> None:
    features = np.zeros((3, 4, 8))
    features[0, :, 0] = [0.1, 0.9, 0.5, 0.0]  # label 2 (0.5): one higher → rank 2
    features[1, :, 0] = [0.3, 0.3, 0.3, 0.3]  # all tied, label 2: ties keep the retrieved order → rank 3
    features[2, :2, 0] = [1.0, 2.0]  # a list of 2, the held-out movie not retrieved
    mask = np.array([[True] * 4, [True] * 4, [True, True, False, False]])
    lists = rk.Lists(features, mask, np.array([2, 2, -1], dtype=np.int64))
    assert rk.ranks_after(None, lists).tolist() == [3, 3, 5]  # retrieved order: label + 1; missing: n + 1
    assert rk.ranks_after(_Fixed(0), lists).tolist() == [2, 3, 5]
    # The rank agrees with re-ordering the candidates by the stable sort.
    for row in range(2):
        order = _Fixed(0).rerank(np.arange(4, dtype=np.int64), features[row])
        assert int(np.flatnonzero(order == 2)[0]) + 1 == rk.ranks_after(_Fixed(0), lists)[row]


def test_padding_never_outranks_a_real_candidate() -> None:
    # A list of 2 padded to 4: the padding's features would score highest, but only real candidates are ranked.
    features = np.zeros((1, 4, 8))
    features[0, :, 0] = [0.2, 0.1, 9.0, 9.0]
    lists = rk.Lists(features, np.array([[True, True, False, False]]), np.array([1], dtype=np.int64))
    assert rk.ranks_after(_Fixed(0), lists).tolist() == [2]


def test_training_is_deterministic_and_learns() -> None:
    rng = np.random.default_rng(3)
    lists = _random_lists(rng, users=200)
    # Make feature 0 informative: the held-out movie's is the largest in its list.
    for u in np.flatnonzero(lists.labels >= 0):
        lists.features[u, lists.labels[u], 0] = lists.features[u, lists.mask[u], 0].max() + 1.0
    config = rk.RankerConfig(hidden=8, epochs=15, batch_lists=32, lr=0.03, seed=4)
    a, losses = rk.train(lists, config)
    b, _ = rk.train(lists, config)
    for (wa, ba), (wb, bb) in zip(a.layers, b.layers, strict=True):
        assert np.array_equal(wa, wb) and np.array_equal(ba, bb)
    assert losses[-1] < losses[0]
    before = rk.ranks_after(None, lists)
    after = rk.ranks_after(a, lists)
    retrieved = lists.labels >= 0
    assert after[retrieved].mean() < before[retrieved].mean()
    assert (after[~retrieved] == lists.features.shape[1] + 1).all()


def test_the_standardiser_uses_the_real_candidates_only() -> None:
    features = np.zeros((2, 3, 8))
    features[:, :, 0] = [[1.0, 3.0, 100.0], [5.0, 100.0, 100.0]]
    mask = np.array([[True, True, False], [True, False, False]])
    mean, scale = rk.standardiser(rk.Lists(features, mask, np.zeros(2, dtype=np.int64)))
    assert mean[0] == pytest.approx(3.0)
    assert scale[0] == pytest.approx(np.std([1.0, 3.0, 5.0]))
    assert scale[1] == 1.0  # a constant feature


def test_training_with_no_retrieved_labels_is_refused() -> None:
    lists = _random_lists(np.random.default_rng(5))
    lists = rk.Lists(lists.features, lists.mask, np.full(len(lists), -1, dtype=np.int64))
    with pytest.raises(ValueError, match="nothing to learn"):
        rk.train(lists, rk.RankerConfig(epochs=1))


def test_on_epoch_collects_every_epochs_export() -> None:
    lists = _random_lists(np.random.default_rng(6))
    snapshots: list[rk.RankerWeights] = []
    final, losses = rk.train(lists, rk.RankerConfig(hidden=0, epochs=3, seed=1), snapshots)
    assert len(snapshots) == len(losses) == 3
    assert np.array_equal(snapshots[-1].layers[0][0], final.layers[0][0])
    assert not np.array_equal(snapshots[0].layers[0][0], snapshots[-1].layers[0][0])


@pytest.mark.parametrize(
    "changes",
    [{"hidden": -1}, {"hidden": 1.5}, {"epochs": 0}, {"batch_lists": 0}, {"lr": 0}, {"lr": 2.0}, {"lr": True}],
)
def test_bad_ranker_configs(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        rk.RankerConfig(**changes).check()  # type: ignore[arg-type]
