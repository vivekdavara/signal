"""The two-tower experiment's protocol on the synthetic dataset, each rule checked by an independent recomputation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signalrec import cli, experiment, movielens, prepare, synthetic
from signalrec import evaluate as ev
from signalrec import experiment_twotower as ext
from signalrec import twotower as tt
from signalrec.baselines import ItemKNN, Popularity
from signalrec.prepare import Prepared

GRID = ((0.1, True), (0.5, False))
KNN_GRID = ((5, 0.0), (20, 0.0))
BASE = tt.Config(dim=16, hidden=16, batch_size=64, lr=1e-2)
EPOCHS = 4
SEEDS = (3, 4)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=120, n_items=150)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


@pytest.fixture(scope="module")
def result(prepared: Prepared) -> dict[str, Any]:
    return ext.run(prepared, GRID, EPOCHS, SEEDS, KNN_GRID, BASE)


def _val_ndcg(p: Prepared, config: tt.Config) -> ev.FloatArray:
    seq = tt.sequences(p, include_val=False)
    net, _ = tt.train(seq, tt.genre_matrix(p.genres)[0], config)
    history = p.history(include_val=False)
    evaluation = ev.evaluate(tt.embed(net, seq, config.max_history), history, p.val_items, "val")
    return ev.ndcg_at(evaluation.ranks, 10)


def test_the_choice_is_the_best_epoch_of_the_best_configuration(result: dict[str, Any]) -> None:
    rows = result["validation"]["two-tower"]
    assert [(r["temperature"], r["logq"]) for r in rows] == list(GRID)
    candidates = [(r["temperature"], r["logq"], c["epoch"], c["ndcg@10"]) for r in rows for c in r["curve"]]
    assert len(candidates) == len(GRID) * EPOCHS
    best = max(candidates, key=lambda c: c[3])  # max keeps the first of equal values: earlier config, then epoch
    assert result["chosen"]["two-tower"] == {"temperature": best[0], "logq": best[1], "epochs": best[2]}
    for row in rows:
        top = max(row["curve"], key=lambda c: c["ndcg@10"])
        assert row["best epoch"] == top["epoch"]
        assert row["best"]["ndcg@10"] == top["ndcg@10"]


def test_the_curve_scores_validation_with_training_history_only(prepared: Prepared, result: dict[str, Any]) -> None:
    # Retrained independently for each epoch count: the curve's value at epoch e is a model trained e epochs on the
    # training positives and scored on the validation items (not the test items, not with validation in the history).
    row = result["validation"]["two-tower"][0]
    temperature, logq = GRID[0]
    for point in (row["curve"][0], row["curve"][-1]):
        config = replace(BASE, temperature=temperature, logq=logq, epochs=point["epoch"], seed=SEEDS[0])
        assert float(_val_ndcg(prepared, config).mean()) == point["ndcg@10"]


def test_the_comparison_with_item_knn_is_paired_on_validation(prepared: Prepared, result: dict[str, Any]) -> None:
    chosen = result["chosen"]["two-tower"]
    config = replace(
        BASE, temperature=chosen["temperature"], logq=chosen["logq"], epochs=chosen["epochs"], seed=SEEDS[0]
    )
    two_tower = _val_ndcg(prepared, config)
    tuning = experiment.tune_item_knn(prepared, KNN_GRID)
    knn = tuning.per_user[tuning.chosen]
    cell = result["two-tower minus item-knn, validation ndcg@10 (paired)"]
    assert cell["value"] == pytest.approx(float(two_tower.mean() - knn.mean()))
    assert tuple(cell["ci95"]) == ev.paired_bootstrap_ci(two_tower, knn, seed=experiment.SEED)
    assert cell["two-tower beats item-knn (interval above 0)"] == (cell["ci95"][0] > 0)
    assert cell["value"] != 0  # so a swapped subtraction can't pass


def test_each_configurations_interval_is_chosen_minus_it(prepared: Prepared, result: dict[str, Any]) -> None:
    rows = result["validation"]["two-tower"]
    per_user = []
    for row, (temperature, logq) in zip(rows, GRID, strict=True):
        config = replace(BASE, temperature=temperature, logq=logq, epochs=row["best epoch"], seed=SEEDS[0])
        per_user.append(_val_ndcg(prepared, config))
    chosen = result["chosen"]["two-tower"]
    index = GRID.index((chosen["temperature"], chosen["logq"]))
    assert len({r["best"]["ndcg@10"] for r in rows}) > 1  # otherwise a mirrored interval could equal the right one
    for row, values in zip(rows, per_user, strict=True):
        assert tuple(row[ext.DIAGNOSTIC]) == ev.paired_bootstrap_ci(per_user[index], values, seed=experiment.SEED)
    listed = result[
        "validation configurations indistinguishable from the chosen one (paired ndcg@10 interval reaches 0)"
    ]
    want = [
        {"temperature": r["temperature"], "logq": r["logq"]}
        for i, r in enumerate(rows)
        if i != index and r[ext.DIAGNOSTIC][0] <= 0
    ]
    assert listed == want


def test_the_test_run_retrains_on_training_plus_validation(prepared: Prepared, result: dict[str, Any]) -> None:
    chosen = result["chosen"]["two-tower"]
    config = replace(
        BASE, temperature=chosen["temperature"], logq=chosen["logq"], epochs=chosen["epochs"], seed=SEEDS[0]
    )
    seq = tt.sequences(prepared, include_val=True)
    net, _ = tt.train(seq, tt.genre_matrix(prepared.genres)[0], config)
    history = prepared.history(include_val=True)
    want = ev.evaluate(tt.embed(net, seq, config.max_history), history, prepared.test_items, "test").summary()
    reported = result["test"]["two-tower"]
    assert {name: reported[name]["value"] for name in ("recall@10", "ndcg@10", "recall@50")} == {
        name: want[name] for name in ("recall@10", "ndcg@10", "recall@50")
    }
    assert reported["seed"] == SEEDS[0]
    assert result["test, two-tower by seed"]["runs"][0]["ndcg@10"] == want["ndcg@10"]


def test_every_seed_is_run_and_summarised(result: dict[str, Any]) -> None:
    by_seed = result["test, two-tower by seed"]
    assert [r["seed"] for r in by_seed["runs"]] == list(SEEDS)
    values = [r["ndcg@10"] for r in by_seed["runs"]]
    assert by_seed["ndcg@10 min"] == min(values)
    assert by_seed["ndcg@10 max"] == max(values)
    assert by_seed["ndcg@10 mean"] == pytest.approx(sum(values) / len(values))


def test_the_baselines_are_refit_on_training_plus_validation(prepared: Prepared, result: dict[str, Any]) -> None:
    knn = result["test"]["item-knn"]
    model = ItemKNN(k=knn["k"], shrink=knn["shrink"])
    history = prepared.history(include_val=True)
    model.fit(history)
    want = ev.evaluate(model, history, prepared.test_items, "test").summary()
    assert knn["ndcg@10"]["value"] == want["ndcg@10"]
    diff = result["two-tower minus item-knn (paired, test)"]["ndcg@10"]
    assert diff["value"] == pytest.approx(result["test"]["two-tower"]["ndcg@10"]["value"] - want["ndcg@10"])
    low, high = diff["ci95"]
    assert low <= diff["value"] <= high


def test_runs_are_reproducible(prepared: Prepared, result: dict[str, Any]) -> None:
    again = ext.run(prepared, GRID, EPOCHS, SEEDS, KNN_GRID, BASE)

    def strip(r: dict[str, Any]) -> str:
        r = json.loads(json.dumps(r))
        for row in r["validation"]["two-tower"]:
            for point in row["curve"]:
                point.pop("train_seconds")
                point.pop("eval_seconds")
        r["test"]["two-tower"].pop("train_seconds")
        return json.dumps(r, sort_keys=True)

    assert strip(again) == strip(result)


def test_an_empty_grid_or_no_seeds_is_refused(prepared: Prepared) -> None:
    with pytest.raises(ValueError, match="grid"):
        ext.run(prepared, (), EPOCHS, SEEDS, KNN_GRID, BASE)
    with pytest.raises(ValueError, match="seeds"):
        ext.run(prepared, GRID, EPOCHS, (), KNN_GRID, BASE)


def test_the_twotower_command_writes_json_and_a_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "80", "--items", "120"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    out = tmp_path / "twotower.json"
    args = [
        "twotower",
        "synthetic",
        "--data-dir",
        str(tmp_path),
        "--grid",
        "0.1:on",
        "--epochs",
        "2",
        "--seeds",
        "0",
        "--knn-grid",
        "5:0",
        "--out",
        str(out),
    ]
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert "| two-tower (τ=0.1, logQ on, " in captured.out
    assert "epoch 2" in captured.err  # progress goes to stderr
    result = json.loads(out.read_text())
    assert result["command"] == "signalrec " + " ".join(args)
    assert result["environment"]["torch"]


@pytest.mark.parametrize(
    "flags",
    [
        ["--grid", "0.1"],
        ["--grid", "0.1:yes"],
        ["--grid", "0:on"],
        ["--grid", "nan:on"],
        ["--grid", "-1:off"],
        ["--seeds", "0,0"],
        ["--seeds", "-1"],
        ["--seeds", "a"],
        ["--epochs", "0"],
    ],
)
def test_bad_twotower_flags_are_usage_errors(flags: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["twotower", "synthetic", *flags])
    assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_popularity_is_refit_on_training_plus_validation(prepared: Prepared, result: dict[str, Any]) -> None:
    # Review round 1: popularity fit on training only, and its paired difference with the sign swapped, both survived.
    history = prepared.history(include_val=True)
    popularity = Popularity()
    popularity.fit(history)
    want = ev.evaluate(popularity, history, prepared.test_items, "test").summary()
    assert result["test"]["popularity"]["ndcg@50"]["value"] == want["ndcg@50"]
    for metric in ("ndcg@10", "ndcg@50", "recall@50"):
        diff = result["two-tower minus popularity (paired, test)"][metric]
        assert diff["value"] == pytest.approx(result["test"]["two-tower"][metric]["value"] - want[metric])
    assert result["two-tower minus popularity (paired, test)"]["ndcg@50"]["value"] != 0


def test_ties_go_to_the_earlier_configuration_then_epoch(prepared: Prepared, monkeypatch: pytest.MonkeyPatch) -> None:
    # Every evaluation returns the same ranks, so every candidate ties: the choice must be the first configuration's
    # first epoch, and every other candidate (interval [0, 0]) is listed as indistinguishable. Review round 1's three
    # tie-rule mutants (>= for >, the last configuration, < 0 for <= 0) survived a fixture without exact ties.
    def same_ranks(model: Any, history: Any, targets: Any, split: str, *args: Any, **kwargs: Any) -> ev.Evaluation:
        n_users, n_items = history.shape
        return ev.Evaluation(
            "fake", split, np.full(n_users, 3, dtype=np.int64), n_items, np.zeros((n_users, 50), dtype=np.int64)
        )

    monkeypatch.setattr(ev, "evaluate", same_ranks)
    tied = ext.run(prepared, GRID, 3, (0,), KNN_GRID, BASE)
    assert tied["chosen"]["two-tower"] == {"temperature": GRID[0][0], "logq": GRID[0][1], "epochs": 1}
    assert [row["best epoch"] for row in tied["validation"]["two-tower"]] == [1, 1]
    reaches = "indistinguishable from the chosen one (paired ndcg@10 interval reaches 0)"
    configs = tied[f"validation configurations {reaches}"]
    assert configs == [{"temperature": GRID[1][0], "logq": GRID[1][1]}]
    candidates = tied[f"validation candidates {reaches}"]
    assert candidates["of"] == candidates["count"] == 2 * 3 - 1
