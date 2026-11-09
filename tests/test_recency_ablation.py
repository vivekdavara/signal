"""scripts/recency_ablation.py: recent-history matrices and item-kNN scored from them."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from signalrec import cli, movielens, prepare, synthetic
from signalrec import evaluate as ev
from signalrec import twotower as tt
from signalrec.baselines import ItemKNN
from signalrec.prepare import Prepared

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recency_ablation.py"
sys.path.insert(0, str(SCRIPT.parent))
import recency_ablation  # noqa: E402


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=80, n_items=120)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


@pytest.mark.parametrize("n", [1, 3, 0])
def test_last_n_keeps_each_users_most_recent_training_positives(prepared: Prepared, n: int) -> None:
    matrix = recency_ablation.last_n_matrix(prepared, n)
    for u in range(prepared.n_users):
        mine = prepared.train_items[prepared.train_users == u]  # already in (time, item) order
        want = mine if n == 0 else mine[-n:]
        assert sorted(matrix[u].indices.tolist()) == sorted(want.tolist())


def test_recent_knn_from_every_positive_is_item_knn(prepared: Prepared) -> None:
    history = prepared.history(include_val=False)
    knn = ItemKNN(k=5, shrink=0.0)
    knn.fit(history)
    scorer = recency_ablation.RecentKNN(knn, recency_ablation.last_n_matrix(prepared, 0))
    users = np.arange(prepared.n_users, dtype=np.int64)
    assert np.array_equal(scorer.score(users), knn.score(users))
    a = ev.evaluate(scorer, history, prepared.val_items, "val").summary()
    b = ev.evaluate(knn, history, prepared.val_items, "val").summary()
    assert a == b


def test_recent_knn_scores_only_from_the_recent_positives(prepared: Prepared) -> None:
    history = prepared.history(include_val=False)
    knn = ItemKNN(k=5, shrink=0.0)
    knn.fit(history)
    recent = recency_ablation.last_n_matrix(prepared, 1)
    scorer = recency_ablation.RecentKNN(knn, recent)
    assert knn.similarity is not None
    for u in range(prepared.n_users):
        (last,) = recent[u].indices.tolist()
        row = knn.similarity[[last]].toarray()[0].astype(np.float64)
        assert np.array_equal(scorer.score(np.array([u]))[0], row)


def test_the_script_scores_validation_with_training_history(tmp_path: Path) -> None:
    # Review round 2: main() had no test, so reading the test items (or validation in the history) survived. Each row
    # is recomputed independently on the validation items.
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "60", "--items", "90"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    source = tmp_path / "twotower.json"
    args = ["twotower", "synthetic", "--data-dir", str(tmp_path), "--grid", "0.1:on", "--epochs", "2", "--seeds", "0"]
    assert cli.main([*args, "--knn-grid", "5:0", "--quiet", "--out", str(source)]) == 0
    out = tmp_path / "recency.json"
    assert (
        recency_ablation.main(["synthetic", "--data-dir", str(tmp_path), "--from", str(source), "--out", str(out)]) == 0
    )
    rows = {row["model"]: row for row in json.loads(out.read_text())["rows"]}

    p = prepare.load(tmp_path / "processed" / "synthetic", "synthetic")
    history = p.history(include_val=False)
    knn = ItemKNN(k=5, shrink=0.0)
    knn.fit(history)
    assert rows["item-knn, last all"]["ndcg@10"] == ev.evaluate(knn, history, p.val_items, "val").summary()["ndcg@10"]
    config = cli._reported_config(source, "synthetic")
    seq = tt.sequences(p, include_val=False)
    net, _ = tt.train(seq, tt.genre_matrix(p.genres)[0], replace(config, max_history=5))
    want = ev.evaluate(tt.embed(net, seq, 5), history, p.val_items, "val").summary()["ndcg@10"]
    assert rows["two-tower, last 5"]["ndcg@10"] == want
    # The reference rows' intervals are against themselves: [0, 0].
    key = "minus the reported history length, ndcg@10, paired ci95"
    assert rows["item-knn, last all"][key] == [0.0, 0.0]
    assert rows[f"two-tower, last {config.max_history}"][key] == [0.0, 0.0]


def test_the_reference_is_the_reported_history_length(tmp_path: Path) -> None:
    # Review round 3: round 2's fix (the reported max_history always a row, and the reference) had no test; and the
    # prepared-data check was untested here.
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "50", "--items", "80"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    source = tmp_path / "twotower.json"
    args = ["twotower", "synthetic", "--data-dir", str(tmp_path), "--grid", "0.1:on", "--epochs", "1", "--seeds", "0"]
    assert cli.main([*args, "--knn-grid", "5:0", "--quiet", "--out", str(source)]) == 0
    result = json.loads(source.read_text())
    result["fixed"]["max_history"] = 7
    source.write_text(json.dumps(result))
    out = tmp_path / "recency.json"
    argv = ["synthetic", "--data-dir", str(tmp_path), "--from", str(source), "--out", str(out)]
    assert recency_ablation.main(argv) == 0
    rows = {row["model"]: row for row in json.loads(out.read_text())["rows"]}
    key = "minus the reported history length, ndcg@10, paired ci95"
    assert rows["two-tower, last 7"][key] == [0.0, 0.0]
    assert any(r[key] != [0.0, 0.0] for name, r in rows.items() if name.startswith("two-tower"))
    result["prepared"]["counts"]["users"] += 1
    source.write_text(json.dumps(result))
    with pytest.raises(prepare.PrepareError, match="other prepared data"):
        recency_ablation.main(argv)
