"""The whole chain on SYNTHETIC data through the command line: prepare → twotower → ranking → export → load-store →
scripts/serving_check.py, which must find the request path equal to ranking.json. Needs Redis."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest
import redis

from signalrec import cli
from signalrec import tower as tw
from signalrec.featurestore import FeatureStore

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import serving_check

from tests.serving.conftest import URL


def run(*argv: str) -> None:
    assert cli.main(list(argv)) == 0, argv


@pytest.mark.usefixtures("client")
def test_serving_reproduces_the_offline_test_metrics(tmp_path: Path, client: redis.Redis) -> None:
    data = str(tmp_path / "data")
    results = tmp_path / "results"
    namespace = f"test-{uuid.uuid4().hex}"
    run("synthetic", "--data-dir", data, "--users", "70", "--items", "100")
    run("prepare", "synthetic", "--data-dir", data)
    two = str(results / "twotower.json")
    run(
        *("twotower", "synthetic", "--data-dir", data, "--epochs", "2", "--grid", "0.1:on", "--histories", "2,4"),
        *("--seeds", "0", "--knn-grid", "2:0", "--quiet", "--out", two),
    )
    ranking = str(results / "ranking.json")
    run(
        *("ranking", "synthetic", "--data-dir", data, "--from", two, "--hidden", "0,4", "--epochs", "3"),
        *("--candidates", "20", "--quiet", "--out", ranking),
    )
    model_dir = str(tmp_path / "model")
    run("export", "synthetic", "--data-dir", data, "--from", two, "--ranking", ranking, "--out", model_dir)
    model = tw.load(Path(model_dir))
    assert model.info["ranker chosen"] == json.loads(Path(ranking).read_text())["chosen"]
    try:
        run(
            "load-store",
            "synthetic",
            "--data-dir",
            data,
            "--model",
            model_dir,
            "--redis",
            URL,
            "--namespace",
            namespace,
        )
        out = tmp_path / "check.json"
        code = serving_check.main(
            [
                *("synthetic", "--data-dir", data, "--model", model_dir, "--ranking", ranking),
                *("--redis", URL, "--namespace", namespace, "--out", str(out)),
            ]
        )
        check = json.loads(out.read_text())
        assert code == 0, check
        assert check["runs"]["protocol filter, ranked"]["equal"] is True
        assert check["runs"]["protocol filter, retrieved order"]["equal"] is True
    finally:
        FeatureStore(client, model.catalogue, namespace=namespace).clear()


def test_export_refuses_a_ranking_from_another_configuration(tmp_path: Path) -> None:
    data = str(tmp_path / "data")
    run("synthetic", "--data-dir", data, "--users", "40", "--items", "60")
    run("prepare", "synthetic", "--data-dir", data)
    two = tmp_path / "twotower.json"
    run(
        *("twotower", "synthetic", "--data-dir", data, "--epochs", "1", "--grid", "0.1:on", "--histories", "2"),
        *("--seeds", "0", "--knn-grid", "2:0", "--quiet", "--out", str(two)),
    )
    ranking = tmp_path / "ranking.json"
    run(
        *("ranking", "synthetic", "--data-dir", data, "--from", str(two), "--hidden", "0", "--epochs", "1"),
        *("--candidates", "10", "--quiet", "--out", str(ranking)),
    )
    edited = json.loads(ranking.read_text())
    edited["two-tower configuration"]["max_history"] = 9
    ranking.write_text(json.dumps(edited))
    code = cli.main(["export", "synthetic", "--data-dir", data, "--from", str(two), "--ranking", str(ranking)])
    assert code == 1


@pytest.mark.parametrize("bad", ["", "a b", "x:1.5", "x:-0.1", "x:half"])
def test_bad_experiment_flags_are_usage_errors(bad: str) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["serve", "ml-1m", "--experiment", bad])
    assert exc.value.code == 2
