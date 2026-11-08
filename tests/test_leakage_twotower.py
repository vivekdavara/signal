"""The two-tower as-of check: buckets, the events before each cut, what each model trains on, and the command."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signalrec import cli, movielens, prepare, synthetic
from signalrec import evaluate as ev
from signalrec import leakage_twotower as lt
from signalrec import twotower as tt
from signalrec.prepare import Prepared

CONFIG = tt.Config(dim=16, hidden=16, batch_size=64, lr=1e-2, epochs=2, seed=3)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=100, n_items=140)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


def test_buckets_partition_users_in_test_time_order(prepared: Prepared) -> None:
    buckets = lt.bucket_users(prepared.test_times, 7)
    assert len(buckets) == 7
    flat = np.concatenate(buckets)
    assert sorted(flat.tolist()) == list(range(prepared.n_users))
    sizes = [len(b) for b in buckets]
    assert max(sizes) - min(sizes) <= 1
    for earlier, later in itertools.pairwise(buckets):
        assert prepared.test_times[earlier].max() <= prepared.test_times[later].min()


def test_more_buckets_than_users_leaves_none_empty() -> None:
    buckets = lt.bucket_users(np.array([5, 1, 3], dtype=np.int64), 10)
    assert [b.tolist() for b in buckets] == [[1], [2], [0]]
    with pytest.raises(ValueError):
        lt.bucket_users(np.array([1], dtype=np.int64), 0)


def test_events_before_a_cut_are_every_earlier_positive_in_order(prepared: Prepared) -> None:
    cut = int(np.median(prepared.test_times))
    seq = lt.events_before(prepared, cut)
    for u in range(prepared.n_users):
        rows = [(int(t), int(i)) for t, i in zip(prepared.train_times, prepared.train_items, strict=True)]
        mine = [r for r, owner in zip(rows, prepared.train_users, strict=True) if owner == u]
        mine += [(int(prepared.val_times[u]), int(prepared.val_items[u]))]
        mine += [(int(prepared.test_times[u]), int(prepared.test_items[u]))]
        want = [item for t, item in sorted(mine) if t < cut]
        assert seq.items[seq.starts[u] : seq.starts[u + 1]].tolist() == want


def test_each_model_trains_on_exactly_what_it_should(prepared: Prepared, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tt.Sequences] = []
    real_train = tt.train

    def spy(seq: tt.Sequences, *args: Any, **kwargs: Any) -> Any:
        seen.append(seq)
        return real_train(seq, *args, **kwargs)

    monkeypatch.setattr(tt, "train", spy)  # leakage_twotower calls tt.train through the module
    result = lt.run(prepared, CONFIG, buckets=3)
    full = tt.sequences(prepared, include_val=True)
    assert seen[0].items.tolist() == full.items.tolist()  # the leaky model: all training + validation
    cuts = [int(prepared.test_times[users].min()) for users in lt.bucket_users(prepared.test_times, 3)]
    trained = [cut for cut, row in zip(cuts, result["buckets"], strict=True) if row["trained"]]
    assert len(seen) == 1 + len(trained)
    for seq, cut in zip(seen[1:], trained, strict=True):
        want = lt.events_before(prepared, cut)
        assert seq.items.tolist() == want.items.tolist()
        assert seq.starts.tolist() == want.starts.tolist()


def test_the_leaky_side_is_the_reported_test_run(prepared: Prepared) -> None:
    result = lt.run(prepared, CONFIG, buckets=2)
    seq = tt.sequences(prepared, include_val=True)
    net, _ = tt.train(seq, tt.genre_matrix(prepared.genres)[0], CONFIG)
    history = prepared.history(include_val=True)
    want = ev.evaluate(tt.embed(net, seq, CONFIG.max_history), history, prepared.test_items, "test").summary()
    cell = result[lt.COMPARISON]["ndcg@10"]
    assert cell[lt.LEAKY] == want["ndcg@10"]
    assert cell["difference"] == pytest.approx(cell[lt.LEAKY] - cell[lt.AS_OF])
    low, high = cell["difference ci95"]
    assert low <= cell["difference"] <= high


def test_a_bucket_with_nothing_before_its_cut_uses_the_untrained_model() -> None:
    # Every positive in the same second: nothing is strictly before any cut, so no model can be trained as of it.
    n = 3
    users = np.arange(n, dtype=np.int64)
    same = np.full(n, 100, dtype=np.int64)
    p = Prepared(
        train_users=users,
        train_items=users % 4,
        train_times=same,
        val_items=(users + 1) % 4,
        val_times=same,
        test_items=(users + 2) % 4,
        test_times=same,
        user_raw_ids=users + 1,
        item_raw_ids=np.arange(1, 5, dtype=np.int64),
        titles=("a", "b", "c", "d"),
        genres=(("X",), ("Y",), (), ("X", "Y")),
    )
    config = tt.Config(dim=4, hidden=4, batch_size=2, epochs=1, seed=9)
    result = lt.run(p, config, buckets=1)
    (row,) = result["buckets"]
    assert row["positives before the cut"] == 0
    assert row["trained"] is False
    # The as-of side is the initial weights for this seed, reproducibly.
    again = lt.run(p, config, buckets=1)
    assert again[lt.COMPARISON] == result[lt.COMPARISON]


def _two_tower_results(path: Path, dataset: str) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "chosen": {"two-tower": {"temperature": 0.1, "logq": True, "epochs": 2}},
                "fixed": {"dim": 16, "hidden": 16, "batch_size": 64, "lr": 0.01, "max_history": 50},
                "test": {"two-tower": {"seed": 3}},
            }
        )
    )


def test_the_command_reads_the_reported_configuration(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "80", "--items", "120"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    source = tmp_path / "twotower.json"
    _two_tower_results(source, "synthetic")
    out = tmp_path / "leakage-twotower.json"
    args = ["leakage-twotower", "synthetic", "--data-dir", str(tmp_path), "--from", str(source)]
    assert cli.main([*args, "--buckets", "3", "--out", str(out)]) == 0
    assert "two-tower ndcg@10: leaky" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["configuration"] == {**CONFIG.as_dict(), "temperature": 0.1, "logq": True}
    assert len(result["buckets"]) == 3


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"dataset": "synthetic"}),
        json.dumps({"dataset": "ml-1m"}),
    ],
)
def test_a_bad_results_file_is_a_one_line_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    source = tmp_path / "twotower.json"
    source.write_text(content)
    code = cli.main(["leakage-twotower", "synthetic", "--data-dir", str(tmp_path), "--from", str(source)])
    assert code == 1
    assert capsys.readouterr().err.startswith("signalrec: ")


def test_a_missing_results_file_is_a_one_line_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["leakage-twotower", "synthetic", "--data-dir", str(tmp_path), "--from", str(tmp_path / "no")])
    assert code == 1
    assert "signalrec twotower" in capsys.readouterr().err
