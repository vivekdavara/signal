"""experiment_ranking.py on the synthetic dataset: each protocol rule recomputed independently."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signalrec import cli, movielens, prepare, synthetic
from signalrec import evaluate as ev
from signalrec import experiment_ranking as er
from signalrec import ranking as rk
from signalrec import twotower as tt
from signalrec.prepare import Prepared

TT = tt.Config(dim=16, hidden=16, batch_size=64, lr=1e-2, epochs=3, max_history=5)
HIDDEN = (0, 4)
EPOCHS = 3
N = 30


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=150, n_items=160)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


@pytest.fixture(scope="module")
def result(prepared: Prepared) -> dict[str, Any]:
    return er.run(prepared, TT, HIDDEN, EPOCHS, N)


def _embeddings(p: Prepared, include_val: bool) -> tt.Embeddings:
    seq = tt.sequences(p, include_val)
    net, _ = tt.train(seq, tt.genre_matrix(p.genres)[0], TT)
    return tt.embed(net, seq, TT.max_history)


@pytest.mark.parametrize("include_val", [False, True])
def test_the_retrieved_order_is_the_full_ranking_cut_at_n(prepared: Prepared, include_val: bool) -> None:
    # The lists' label + 1 must be the held-out movie's rank in evaluate's full ranking (the same model, its history
    # excluded), and every movie ranked below n must count as not retrieved. Validation lists hold the validation
    # movie and use the training-only model; test lists the test movie and the training + validation model.
    lists = er.build_lists(prepared, TT, include_val, N).lists
    history = prepared.history(include_val)
    targets = prepared.test_items if include_val else prepared.val_items
    full = ev.evaluate(_embeddings(prepared, include_val), history, targets, "x").ranks
    got = rk.ranks_after(None, lists)
    assert got.tolist() == np.where(full <= N, full, N + 1).tolist()
    assert 0 < (full <= N).sum() < prepared.n_users  # both cases occur


def test_lists_exclude_the_history_and_hold_n_candidates(prepared: Prepared) -> None:
    lists = er.build_lists(prepared, TT, include_val=False, n=N).lists
    assert lists.features.shape == (prepared.n_users, N, len(rk.FEATURES))
    assert lists.mask.all()  # 160 movies, at most ~40 seen: every list is full
    # The log-rank feature is the position, so the features are in retrieved order.
    assert np.allclose(lists.features[:, :, 1], np.log1p(np.arange(N))[None, :])


def test_the_split_is_disjoint_and_covers_every_user() -> None:
    fit, select = er.split_users(150)
    assert len(select) == 30 and len(fit) == 120
    assert sorted([*fit.tolist(), *select.tolist()]) == list(range(150))
    again = er.split_users(150)
    assert fit.tolist() == again[0].tolist()
    with pytest.raises(ValueError, match="can't be split"):
        er.split_users(1)


def test_the_choice_is_the_best_candidate_on_the_selection_users(prepared: Prepared, result: dict[str, Any]) -> None:
    lists = er.build_lists(prepared, TT, include_val=False, n=N).lists
    fit_users, select_users = er.split_users(prepared.n_users)
    candidates = []
    for h in HIDDEN:
        snapshots: list[rk.RankerWeights] = []
        rk.train(lists.subset(fit_users), rk.RankerConfig(hidden=h, epochs=EPOCHS, seed=0), snapshots)
        for epoch, weights in enumerate(snapshots, start=1):
            value = float(ev.ndcg_at(rk.ranks_after(weights, lists.subset(select_users)), 10).mean())
            candidates.append((value, h, epoch))
    best = max(candidates, key=lambda c: c[0])  # max keeps the first of equal values
    assert result["chosen"] == {"hidden": best[1], "epochs": best[2]}
    reported = [
        (c["ndcg@10"], row["hidden"], c["epoch"]) for row in result["validation"]["rankers"] for c in row["curve"]
    ]
    assert reported == candidates
    assert len({c[0] for c in candidates}) > 1  # the candidates differ, so the choice means something


def test_test_numbers_come_from_the_chosen_weights_on_the_test_lists(
    prepared: Prepared, result: dict[str, Any]
) -> None:
    saved = result["ranker weights"]
    weights = rk.RankerWeights(
        np.array(saved["mean"]),
        np.array(saved["scale"]),
        tuple((np.array(w), np.array(b)) for w, b in saved["layers"]),
    )
    lists = er.build_lists(prepared, TT, include_val=True, n=N).lists
    ranked = er.metrics(rk.ranks_after(weights, lists), N)
    retrieved = er.metrics(rk.ranks_after(None, lists), N)
    for name in ("ndcg@10", "recall@20", f"recall@{N}"):  # N = 30: no cut-off above it is reported
        assert result["test"]["re-ranked"][name]["value"] == float(ranked[name].mean())
        assert result["test"]["retrieved order"][name]["value"] == float(retrieved[name].mean())
        diff = result["re-ranked minus retrieved order (paired, test)"][name]
        assert diff["value"] == pytest.approx(float(ranked[name].mean() - retrieved[name].mean()))
        want = ev.paired_bootstrap_ci(ranked[name], retrieved[name], seed=er.SEED)
        assert tuple(diff["ci95"]) == want
    # Re-ranking can't change which movies were retrieved.
    assert result["test"]["re-ranked"][f"recall@{N}"] == result["test"]["retrieved order"][f"recall@{N}"]


def test_the_ranker_trains_on_the_fit_users_validation_labels_only(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[rk.Lists] = []
    real_train = rk.train

    def spy(lists: rk.Lists, *args: Any, **kwargs: Any) -> Any:
        seen.append(lists)
        return real_train(lists, *args, **kwargs)

    monkeypatch.setattr(rk, "train", spy)
    er.run(prepared, TT, (0,), 1, N)
    # The ranker, then the ablation's one retraining per feature: all on the fit users' validation lists.
    assert len(seen) == 1 + len(rk.FEATURES)
    fit_users, _ = er.split_users(prepared.n_users)
    val_lists = er.build_lists(prepared, TT, include_val=False, n=N).lists
    for lists in seen:
        assert lists.labels.tolist() == val_lists.labels[fit_users].tolist()
    assert np.array_equal(seen[0].features, val_lists.features[fit_users])
    for i, lists in enumerate(seen[1:]):
        assert np.array_equal(lists.features, er.without(val_lists.subset(fit_users), i).features)


def test_runs_are_reproducible(prepared: Prepared, result: dict[str, Any]) -> None:
    again = er.run(prepared, TT, HIDDEN, EPOCHS, N)

    def strip(r: dict[str, Any]) -> str:
        r = json.loads(json.dumps(r))
        r["validation"].pop("seconds")
        r["test"].pop("seconds")
        for row in r["validation"]["rankers"]:
            row.pop("train_seconds")
        return json.dumps(r, sort_keys=True)

    assert strip(again) == strip(result)


@pytest.mark.parametrize(
    ("hidden", "epochs", "n", "message"),
    [
        ((), 2, 10, "no hidden sizes"),
        ((4, 4), 2, 10, "repeat"),
        ((0,), 0, 10, "epochs"),
        ((0,), 2, 0, "candidate"),
        ((0,), 2, 9, "candidate"),  # fewer candidates than the choice's cut-off
    ],
)
def test_bad_inputs_are_refused_before_training(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch, hidden: Any, epochs: int, n: int, message: str
) -> None:
    def no_training(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("trained before refusing")

    monkeypatch.setattr(tt, "train", no_training)
    with pytest.raises(ValueError, match=message):
        er.run(prepared, TT, hidden, epochs, n)


def test_the_ranking_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "80", "--items", "120"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    source = tmp_path / "twotower.json"
    two_tower = ["twotower", "synthetic", "--data-dir", str(tmp_path), "--grid", "0.1:on", "--histories", "5"]
    assert (
        cli.main([*two_tower, "--epochs", "1", "--seeds", "0", "--knn-grid", "5:0", "--quiet", "--out", str(source)])
        == 0
    )
    out = tmp_path / "ranking.json"
    args = ["ranking", "synthetic", "--data-dir", str(tmp_path), "--from", str(source), "--hidden", "0,4"]
    assert cli.main([*args, "--epochs", "2", "--candidates", "20", "--quiet", "--out", str(out)]) == 0
    assert "test ndcg@10: retrieved order" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["two-tower configuration"]["max_history"] == 5
    assert result["candidates"] == 20
    assert [row["hidden"] for row in result["validation"]["rankers"]] == [0, 4]


@pytest.mark.parametrize("flags", [["--hidden", "-1"], ["--hidden", "4,4"], ["--hidden", "x"], ["--candidates", "0"]])
def test_bad_ranking_flags_are_usage_errors(flags: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["ranking", "synthetic", *flags])
    assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_cut_offs_above_the_candidate_count_are_not_reported() -> None:
    # A movie that wasn't retrieved ranks n + 1. With 20 candidates that is 21, inside Recall@50: so no cut-off above
    # n may be reported (it counted every miss as a hit before this fix).
    ranks = np.array([1, 21, 21, 5], dtype=np.int64)  # two users' movies not retrieved
    got = er.metrics(ranks, 20)
    assert set(got) == {"recall@10", "ndcg@10", "recall@20", "ndcg@20"}
    assert got["recall@20"].tolist() == [1, 0, 0, 1]
    assert set(er.metrics(ranks, 200)) == {
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
        "recall@200",
    }


def test_the_feature_ablation_chooses_each_ranker_like_the_full_one(prepared: Prepared, result: dict[str, Any]) -> None:
    # Review round 2: each ablated ranker gets the same choice (every hidden size and epoch, the best on the selection
    # users) as the full one; recomputed here for two features.
    rows = result["feature ablation (selection users, post hoc)"]
    assert [row["without"] for row in rows] == list(rk.FEATURES)
    lists = er.build_lists(prepared, TT, include_val=False, n=N).lists
    fit_users, select_users = er.split_users(prepared.n_users)
    for i in (0, 7):  # the retrieval score and the last-positive similarity
        fit, select = er.without(lists.subset(fit_users), i), er.without(lists.subset(select_users), i)
        candidates = []
        for h in HIDDEN:
            snapshots: list[rk.RankerWeights] = []
            rk.train(fit, rk.RankerConfig(hidden=h, epochs=EPOCHS, seed=0), snapshots)
            for epoch, weights in enumerate(snapshots, start=1):
                candidates.append((float(ev.ndcg_at(rk.ranks_after(weights, select), 10).mean()), h, epoch))
        best = max(candidates, key=lambda c: c[0])
        assert rows[i]["ndcg@10"] == best[0]
        assert rows[i]["chosen"] == {"hidden": best[1], "epochs": best[2]}
    assert len({row["ndcg@10"] for row in rows}) > 1


def test_without_makes_one_feature_constant() -> None:
    lists = rk.Lists(np.ones((2, 3, 8)), np.ones((2, 3), dtype=bool), np.zeros(2, dtype=np.int64))
    out = er.without(lists, 4)
    assert (out.features[:, :, 4] == 0).all()
    assert (np.delete(out.features, 4, axis=2) == 1).all()
    assert (lists.features == 1).all()  # the input is untouched
