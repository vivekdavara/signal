"""The baseline runner and the baselines command on the synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalrec import cli, experiment, movielens, prepare, synthetic
from signalrec import evaluate as ev
from signalrec.baselines import ItemKNN, Popularity
from signalrec.prepare import Prepared

GRID = ((5, 0.0), (20, 0.0), (20, 10.0))


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=150, n_items=200)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


def test_the_choice_is_the_best_validation_row(prepared: Prepared) -> None:
    result = experiment.run(prepared, GRID)
    rows = result["validation"]["item-knn"]
    assert [(r["k"], r["shrink"]) for r in rows] == list(GRID)
    best = max(rows, key=lambda r: r[experiment.PRIMARY])
    assert result["chosen"]["item-knn"] == {"k": best["k"], "shrink": best["shrink"]}


def test_test_metrics_carry_intervals_around_them(prepared: Prepared) -> None:
    result = experiment.run(prepared, GRID)
    for model in ("popularity", "item-knn"):
        for metric in ("recall@10", "ndcg@10", "recall@50"):
            cell = result["test"][model][metric]
            low, high = cell["ci95"]
            assert low <= cell["value"] <= high
        assert "ci95" not in result["test"][model]["coverage@10"]
    diff = result["item-knn minus popularity (paired)"]["ndcg@10"]
    knn = result["test"]["item-knn"]["ndcg@10"]["value"]
    pop = result["test"]["popularity"]["ndcg@10"]["value"]
    assert diff["value"] == pytest.approx(knn - pop)


def test_runs_are_reproducible(prepared: Prepared) -> None:
    a = experiment.run(prepared, GRID)
    b = experiment.run(prepared, GRID)
    for r in (a, b):
        for row in r["validation"]["item-knn"]:
            row.pop("fit_seconds")
        for model in ("popularity", "item-knn"):
            r["test"][model].pop("fit_seconds")
            r["test"][model].pop("eval_seconds")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_the_test_split_uses_the_validation_item_as_history(prepared: Prepared) -> None:
    # If the validation item weren't in the test-time history it could be recommended (and it is often a strong
    # candidate); with it excluded, no user's top list may contain it.
    result = experiment._timed_eval(Popularity(), prepared, "test")
    for user, top in enumerate(result.evaluation.top):
        assert prepared.val_items[user] not in top.tolist()


def test_the_baselines_command_writes_json_and_a_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "80", "--items", "120"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    out = tmp_path / "baselines.json"
    args = ["baselines", "synthetic", "--data-dir", str(tmp_path), "--grid", "5:0,10:10", "--out", str(out)]
    assert cli.main(args) == 0
    printed = capsys.readouterr().out
    assert "| item-kNN (k=" in printed
    result = json.loads(out.read_text())
    assert result["command"] == "signalrec " + " ".join(args)
    assert result["prepared"]["counts"]["users"] == result["users"]
    assert [(r["k"], r["shrink"]) for r in result["validation"]["item-knn"]] == [(5, 0.0), (10, 10.0)]


def test_each_validation_row_is_compared_with_the_chosen_one(prepared: Prepared) -> None:
    result = experiment.run(prepared, GRID)
    rows = result["validation"]["item-knn"]
    chosen = result["chosen"]["item-knn"]
    for row in rows:
        low, high = row["chosen minus this, ndcg@10, paired ci95"]
        assert low <= high
        if (row["k"], row["shrink"]) == (chosen["k"], chosen["shrink"]):
            assert low == high == 0.0
        else:
            # The chosen row has the highest mean, so the difference's mean is >= 0 and the interval's top is too.
            assert high >= 0.0
    listed = result["validation rows indistinguishable from the chosen one (paired ndcg@10 interval reaches 0)"]
    assert chosen not in listed
    assert all(
        r["chosen minus this, ndcg@10, paired ci95"][0] <= 0
        for r in rows
        if {"k": r["k"], "shrink": r["shrink"]} in listed
    )


def test_tuning_scores_the_validation_items_with_training_history(prepared: Prepared) -> None:
    # Review round 1: tuning on the test items (else p.test_items) survived every test.
    result = experiment.run(prepared, GRID)
    history = prepared.history(include_val=False)
    for row in result["validation"]["item-knn"]:
        model = ItemKNN(k=row["k"], shrink=row["shrink"])
        model.fit(history)
        want = ev.evaluate(model, history, prepared.val_items, "val").summary()
        assert {name: row[name] for name in want} == want
    popularity = Popularity()
    popularity.fit(history)
    assert result["validation"]["popularity"] == ev.evaluate(popularity, history, prepared.val_items, "val").summary()


def test_a_tie_on_validation_goes_to_the_first_grid_row(prepared: Prepared) -> None:
    # k beyond the catalogue keeps every neighbour, so these two rows score identically.
    result = experiment.run(prepared, ((5000, 0.0), (9000, 0.0)))
    rows = result["validation"]["item-knn"]
    assert rows[0]["ndcg@10"] == rows[1]["ndcg@10"]
    assert result["chosen"]["item-knn"] == {"k": 5000, "shrink": 0.0}
    # An identical row's paired interval is [0, 0], which reaches 0: indistinguishable.
    key = "validation rows indistinguishable from the chosen one (paired ndcg@10 interval reaches 0)"
    assert result[key] == [{"k": 9000, "shrink": 0.0}]


def test_each_rows_interval_is_for_chosen_minus_that_row(prepared: Prepared) -> None:
    # Recomputed independently: each row's per-user validation NDCG@10, then the paired interval (chosen - row). An
    # interval for (row - chosen) would be the mirror image, so the comparison is exact, not "brackets the mean".
    result = experiment.run(prepared, GRID)
    history = prepared.history(include_val=False)
    per_user = {}
    for k, shrink in GRID:
        model = ItemKNN(k=k, shrink=shrink)
        model.fit(history)
        per_user[(k, shrink)] = ev.ndcg_at(ev.evaluate(model, history, prepared.val_items, "val").ranks, 10)
    chosen = result["chosen"]["item-knn"]
    best = per_user[(chosen["k"], chosen["shrink"])]
    rows = result["validation"]["item-knn"]
    assert len({r["ndcg@10"] for r in rows}) > 1  # otherwise a mirrored interval could equal the right one
    for row in rows:
        want = ev.paired_bootstrap_ci(best, per_user[(row["k"], row["shrink"])], seed=experiment.SEED)
        assert tuple(row[experiment.DIAGNOSTIC]) == want


def test_paired_differences_bracket_their_values(prepared: Prepared) -> None:
    result = experiment.run(prepared, GRID)
    for cell in result["item-knn minus popularity (paired)"].values():
        low, high = cell["ci95"]
        assert low <= cell["value"] <= high
    assert result["item-knn minus popularity (paired)"]["ndcg@10"]["value"] > 0  # so a sign swap can't pass
