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


def test_load_store_refuses_a_model_from_other_parameters(tmp_path: Path) -> None:
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
    model_dir = tmp_path / "model"
    run(
        "export",
        "synthetic",
        "--data-dir",
        data,
        "--from",
        str(two),
        "--ranking",
        str(ranking),
        "--out",
        str(model_dir),
    )
    meta = json.loads((model_dir / "model.json").read_text())
    meta["info"]["prepared"]["params"]["min_rating"] = 3.0
    meta["sha256"] = tw._meta_hash(meta)  # a consistent model.json, exported from other parameters
    (model_dir / "model.json").write_text(json.dumps(meta))
    code = cli.main(["load-store", "synthetic", "--data-dir", data, "--model", str(model_dir), "--namespace", "unused"])
    assert code == 1


@pytest.mark.parametrize("bad", ["-1", "3600", "nan", "inf", "soon", ""])
def test_bad_popular_ttls_are_usage_errors(bad: str) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["serve", "ml-1m", "--popular-ttl", bad])
    assert exc.value.code == 2


@pytest.mark.parametrize(("flag", "ttl"), [([], 1.0), (["--popular-ttl", "0"], 0.0), (["--popular-ttl", "2.5"], 2.5)])
def test_serve_passes_the_popular_ttl(
    tmp_path: Path, model: tw.Model, monkeypatch: pytest.MonkeyPatch, flag: list[str], ttl: float
) -> None:
    import uvicorn

    tw.save(model, tmp_path / "model")
    apps: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_: apps.append(app))
    run("serve", "synthetic", "--model", str(tmp_path / "model"), "--redis", URL, *flag)
    (app,) = apps
    assert app.state.recommender.popular_ttl == ttl  # type: ignore[attr-defined]


def test_serve_with_workers_hands_uvicorn_a_factory_and_the_settings(
    tmp_path: Path, model: tw.Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    import uvicorn

    tw.save(model, tmp_path / "model")
    # serve sets os.environ for its workers: give it a copy, so nothing leaks into later tests.
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k not in cli.SERVE_ENV.values()})
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    run(
        *("serve", "synthetic", "--model", str(tmp_path / "model"), "--redis", URL, "--namespace", "ns-1"),
        *("--workers", "3", "--popular-ttl", "0.25", "--experiment", "ranker-v1:0.3"),
    )
    ((target, kw),) = calls
    assert target == "signalrec.cli:app_from_env" and kw["factory"] is True and kw["workers"] == 3
    # What each worker would build from those settings.
    app = cli.app_from_env()
    rec = app.state.recommender
    assert rec.popular_ttl == 0.25 and rec.store.namespace == "ns-1"
    assert rec.model.npz_sha256 == tw.load(tmp_path / "model").npz_sha256
    assert os.environ[cli.SERVE_ENV["experiment"]] == "ranker-v1:0.3"
    assert cli.SERVE_ENV["model"] in os.environ and Path(os.environ[cli.SERVE_ENV["model"]]).is_absolute()


def test_app_from_env_without_settings_says_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k not in cli.SERVE_ENV.values()})
    with pytest.raises(RuntimeError, match="SIGNALREC_SERVE_MODEL"):
        cli.app_from_env()


def test_serve_with_two_workers_answers(tmp_path: Path, model: tw.Model, fresh: FeatureStore) -> None:
    """The real thing: two uvicorn worker processes behind one port, each with its own model and Redis client."""
    import socket
    import subprocess
    import time

    import httpx

    tw.save(model, tmp_path / "model")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    command = [sys.executable, "-m", "signalrec.cli", "serve", "synthetic", "--model", str(tmp_path / "model")]
    command += ["--redis", URL, "--namespace", fresh.namespace, "--workers", "2", "--port", str(port)]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, proc.stderr.read().decode() if proc.stderr else ""
            assert time.monotonic() < deadline, "no /health within 60 s"
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        user = int(model.user_raw_ids.max()) + 5
        movie = int(model.item_raw_ids[3])
        r = httpx.post(f"http://127.0.0.1:{port}/events", json={"user": user, "movie": movie, "rating": 5}, timeout=5)
        assert r.json() == {"applied": True, "positive": True, "new_user": True}
        for _ in range(20):  # whichever worker answers, it reads the same store
            body = httpx.get(f"http://127.0.0.1:{port}/recommendations/{user}", timeout=5).json()
            assert body["source"] == "two-tower" and movie not in [i["movie"] for i in body["items"]]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.parametrize("workers", ["1", "3"])
@pytest.mark.parametrize(("flag", "value"), [("--namespace", "bad:ns"), ("--redis", "nonsense://x")])
def test_serve_with_a_bad_setting_fails_once_with_a_message(
    tmp_path: Path,
    model: tw.Model,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workers: str,
    flag: str,
    value: str,
) -> None:
    import os

    import uvicorn

    tw.save(model, tmp_path / "model")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    calls: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append(a))
    argv = ["serve", "synthetic", "--model", str(tmp_path / "model"), "--redis", URL, "--workers", workers]
    assert cli.main([*argv, flag, value]) == 2
    assert calls == []  # no server, no workers
    assert capsys.readouterr().err.startswith("signalrec: ")
