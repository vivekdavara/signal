"""The two-tower as-of check: buckets, the events before each cut, what each model trains on, and the command."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

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


def test_events_before_a_cut_are_every_earlier_training_or_validation_positive(prepared: Prepared) -> None:
    cut = int(np.median(prepared.test_times))
    seq = lt.events_before(prepared, cut)
    for u in range(prepared.n_users):
        rows = [(int(t), int(i)) for t, i in zip(prepared.train_times, prepared.train_items, strict=True)]
        mine = [r for r, owner in zip(rows, prepared.train_users, strict=True) if owner == u]
        mine += [(int(prepared.val_times[u]), int(prepared.val_items[u]))]
        # Not the test positive: the leaky model never trains on it (review round 2).
        want = [item for t, item in sorted(mine) if t < cut]
        assert seq.items[seq.starts[u] : seq.starts[u + 1]].tolist() == want


def test_keeping_users_keeps_all_of_their_positives_and_only_theirs(prepared: Prepared) -> None:
    cut = int(np.quantile(prepared.test_times, 0.3))
    keep = np.array([0, 5], dtype=np.int64)
    strict, kept = lt.events_before(prepared, cut), lt.events_before(prepared, cut, keep_users=keep)
    full = tt.sequences(prepared, include_val=True)
    for u in range(prepared.n_users):
        got = kept.items[kept.starts[u] : kept.starts[u + 1]].tolist()
        source = full if u in keep else strict
        assert got == source.items[source.starts[u] : source.starts[u + 1]].tolist()


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
    # Then, per bucket: the strict as-of model, and the one that keeps the bucket's own users' histories.
    expected = []
    for users, row in zip(lt.bucket_users(prepared.test_times, 3), result["buckets"], strict=True):
        cut = int(prepared.test_times[users].min())
        if row["trained"]:
            expected.append(lt.events_before(prepared, cut))
        if row["trained, own histories kept"]:
            expected.append(lt.events_before(prepared, cut, keep_users=users))
    assert len(seen) == 1 + len(expected)
    for seq, want in zip(seen[1:], expected, strict=True):
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


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("data")
    assert cli.main(["synthetic", "--data-dir", str(d), "--users", "80", "--items", "120"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(d)]) == 0
    return d


def _two_tower_results(data_dir: Path, **changes: Any) -> dict[str, Any]:
    manifest = prepare.load(data_dir / "processed" / "synthetic", "synthetic").manifest
    assert manifest is not None
    result: dict[str, Any] = {
        "dataset": "synthetic",
        "prepared": {key: manifest[key] for key in ("params", "counts", "sources")},
        "chosen": {"two-tower": {"temperature": 0.1, "logq": True, "epochs": 2}},
        "fixed": {"dim": 16, "hidden": 16, "batch_size": 64, "lr": 0.01, "max_history": 50},
        "test": {"two-tower": {"seed": 3}},
    }
    result.update(changes)
    return result


def _leakage(data_dir: Path, source: Path, *flags: str) -> int:
    return cli.main(["leakage-twotower", "synthetic", "--data-dir", str(data_dir), "--from", str(source), *flags])


def test_the_command_reads_the_reported_configuration(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "twotower.json"
    source.write_text(json.dumps(_two_tower_results(data_dir)))
    out = tmp_path / "leakage-twotower.json"
    assert _leakage(data_dir, source, "--buckets", "3", "--out", str(out)) == 0
    assert "two-tower ndcg@10: leaky" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["configuration"] == {**CONFIG.as_dict(), "temperature": 0.1, "logq": True}
    assert len(result["buckets"]) == 3


def _bad_files(data_dir: Path) -> list[tuple[str, str]]:
    good = _two_tower_results(data_dir)
    zero_dim = _two_tower_results(data_dir, fixed={**good["fixed"], "dim": 0})
    return [
        ("not json", "not a two-tower results file"),
        (json.dumps({"dataset": "synthetic"}), "not a two-tower results file"),
        # A complete file for another dataset: refused for the dataset, not for a missing key (review round 2: the
        # old case lacked "chosen", so dropping the dataset check survived).
        (json.dumps({**good, "dataset": "ml-1m"}), "holds 'ml-1m'"),
        (json.dumps(zero_dim), "dim must be an integer >= 1"),
    ]


@pytest.mark.parametrize("case", range(4))
def test_a_bad_results_file_is_a_one_line_error(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], case: int
) -> None:
    content, message = _bad_files(data_dir)[case]
    source = tmp_path / "twotower.json"
    source.write_text(content)
    assert _leakage(data_dir, source) == 1
    err = capsys.readouterr().err
    assert err.startswith("signalrec: ")
    assert message in err


def test_a_missing_results_file_is_a_one_line_error(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _leakage(data_dir, data_dir / "no-such-file.json") == 1
    assert "signalrec twotower" in capsys.readouterr().err


def test_more_buckets_than_users_is_an_error(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "twotower.json"
    source.write_text(json.dumps(_two_tower_results(data_dir)))
    assert _leakage(data_dir, source, "--buckets", "1000000000") == 1
    assert "more buckets than" in capsys.readouterr().err


def test_an_untrainable_dataset_is_a_one_line_error(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Review round 2: dropping TrainingError from main's except survived.
    source = tmp_path / "twotower.json"
    source.write_text(json.dumps(_two_tower_results(data_dir)))

    def untrainable(*args: Any, **kwargs: Any) -> Any:
        raise tt.TrainingError("fewer than 2 training examples")

    monkeypatch.setattr(lt, "run", untrainable)
    assert _leakage(data_dir, source) == 1
    assert capsys.readouterr().err == "signalrec: fewer than 2 training examples\n"


# --- what the as-of side is, recomputed (review round 2's survivors) -------------------------------------------------


def test_each_buckets_numbers_are_its_own_models(prepared: Prepared) -> None:
    result = lt.run(prepared, CONFIG, buckets=2)
    genres = tt.genre_matrix(prepared.genres)[0]
    test_seq = tt.sequences(prepared, include_val=True)
    history = prepared.history(include_val=True)
    leaky_net, _ = tt.train(test_seq, genres, CONFIG)
    leaky_scores = tt.embed(leaky_net, test_seq, CONFIG.max_history)
    for users, row in zip(lt.bucket_users(prepared.test_times, 2), result["buckets"], strict=True):
        cut = int(prepared.test_times[users].min())
        assert row["cut (earliest test time)"] == datetime.fromtimestamp(cut, UTC).isoformat()
        net, _ = tt.train(lt.events_before(prepared, cut), genres, CONFIG)
        # As-of user vectors come from the users' full training + validation history, as in the reported test run.
        as_of = ev.ranks(
            tt.embed(net, test_seq, CONFIG.max_history).score(users), history[users], prepared.test_items[users]
        )
        leaky = ev.ranks(leaky_scores.score(users), history[users], prepared.test_items[users])
        assert row["ndcg@10"]["as of the cut"] == float(ev.ndcg_at(as_of, 10).mean())
        assert row["ndcg@10"]["leaky"] == float(ev.ndcg_at(leaky, 10).mean())
        own_net, _ = tt.train(lt.events_before(prepared, cut, keep_users=users), genres, CONFIG)
        own_scores = tt.embed(own_net, test_seq, CONFIG.max_history).score(users)
        own = ev.ranks(own_scores, history[users], prepared.test_items[users])
        assert row["ndcg@10"]["as of the cut, own histories kept"] == float(ev.ndcg_at(own, 10).mean())
    cell = result[lt.COMPARISON]["ndcg@10"]
    assert cell[lt.LEAKY] != cell[lt.AS_OF]  # so swapping the two can't pass
    own = result[lt.COMPARISON_OWN]["ndcg@10"]
    assert own[lt.LEAKY] == cell[lt.LEAKY]
    assert own["difference"] == pytest.approx(own[lt.LEAKY] - own[lt.AS_OF_OWN])
    assert own[lt.AS_OF_OWN] != cell[lt.AS_OF]


def _handmade(val_times: list[int]) -> Prepared:
    n = len(val_times)
    users = np.arange(n, dtype=np.int64)
    return Prepared(
        train_users=users,
        train_items=users % 4,
        train_times=np.full(n, 10, dtype=np.int64),
        val_items=(users + 1) % 4,
        val_times=np.array(val_times, dtype=np.int64),
        test_items=(users + 2) % 4,
        test_times=np.array([30, 50, 50][:n], dtype=np.int64),
        user_raw_ids=users + 1,
        item_raw_ids=np.arange(1, 5, dtype=np.int64),
        titles=("a", "b", "c", "d"),
        genres=(("X",), ("Y",), (), ("X", "Y")),
    )


def test_exactly_two_examples_before_the_cut_are_trained_on() -> None:
    # Cut 30 (user 0's test time): users 0 and 1 have training (10) and validation (20) before it, user 2 only training
    # (its validation is at 40): 1 + 1 + 0 = 2 examples, the smallest batch that has a negative.
    p = _handmade([20, 20, 40])
    config = tt.Config(dim=4, hidden=4, batch_size=2, epochs=1, seed=9)
    first = lt.run(p, config, buckets=3)["buckets"][0]
    assert first["training examples"] == 2
    assert first["trained"] is True


def test_the_untrained_as_of_model_is_the_seeds_initial_weights() -> None:
    # Every positive in the same second (30), so nothing is strictly before the only cut and no model is trained.
    same = np.full(3, 30, dtype=np.int64)
    p = replace(_handmade([30, 30, 30]), train_times=same, test_times=same)
    config = tt.Config(dim=4, hidden=4, batch_size=2, epochs=1, seed=9)
    torch.manual_seed(12345)  # another global state: the result must not depend on it (review round 2)
    result = lt.run(p, config, buckets=1)
    (row,) = result["buckets"]
    assert row["trained"] is False
    torch.manual_seed(config.seed)
    net = tt.TwoTowerNet(tt.genre_matrix(p.genres)[0], config.dim, config.hidden)
    test_seq = tt.sequences(p, include_val=True)
    scores = tt.embed(net, test_seq, config.max_history).score(np.arange(3, dtype=np.int64))
    ranks = ev.ranks(scores, p.history(include_val=True), p.test_items)
    assert result[lt.COMPARISON]["ndcg@10"][lt.AS_OF] == float(ev.ndcg_at(ranks, 10).mean())


def test_users_with_equal_test_times_are_bucketed_by_index() -> None:
    assert [b.tolist() for b in lt.bucket_users(np.array([5, 5, 5], dtype=np.int64), 3)] == [[0], [1], [2]]
