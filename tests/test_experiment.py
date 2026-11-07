"""The baseline runner and the baselines command on the synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalrec import cli, experiment, movielens, prepare, synthetic
from signalrec.baselines import Popularity
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
