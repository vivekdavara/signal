"""Review D4 round 2: ``signalrec load-store`` and the raw ratings it reads."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import redis

from signalrec import cli
from signalrec.features import Catalogue
from signalrec.featurestore import FeatureStore
from signalrec.movielens import FORMATS
from tests.serving.conftest import URL


def run(*argv: str) -> None:
    assert cli.main(list(argv)) == 0, argv


def test_load_store_refuses_raw_ratings_that_are_not_the_prepared_sources(
    tmp_path: Path, client: redis.Redis, capsys: pytest.CaptureFixture[str]
) -> None:
    """Defect (cli.py:311-317, _load_store): round 1 compares the prepared manifest's params, counts and sources with
    the model's, but then reads ``data/raw/<dataset>`` again (movielens.load) for the rated sets and never checks those
    files against the manifest's ``sources`` SHA-256s. A ratings file changed since ``prepare`` (re-downloaded, edited,
    another release) fills every user's rated set (what serving excludes) from other data, silently: the store holds
    what the model wasn't evaluated with, the case the round-1 check exists for.

    Input: the synthetic chain (prepare, twotower, ranking, export), then the raw ratings file's last line deleted,
    then load-store into a fresh test namespace.
    Expected: exit code 1 (the raw ratings don't match the SHA-256 in the prepared manifest).
    Actual: exit code 0, the store loaded from the changed file.
    """
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
    ratings = Path(data) / "raw" / "synthetic" / FORMATS["synthetic"].ratings_file
    lines = ratings.read_bytes().splitlines(keepends=True)
    ratings.write_bytes(b"".join(lines[:-1]))  # still well-formed: whole CRLF lines
    namespace = f"test-{uuid.uuid4().hex}"
    try:
        code = cli.main(
            [
                "load-store",
                "synthetic",
                "--data-dir",
                data,
                "--model",
                str(model_dir),
                "--namespace",
                namespace,
                "--redis",
                URL,
            ]
        )
    finally:
        FeatureStore(client, _catalogue_stub(), namespace=namespace).clear()
    assert code == 1, capsys.readouterr().err


def _catalogue_stub() -> Catalogue:
    # clear() only needs the namespace.
    import numpy as np

    return Catalogue((), (), np.zeros(0, dtype=np.int64))
