"""Review findings (day 1): CLI errors that escape as tracebacks instead of a one-line message and exit 1."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from signalrec import cli, download
from signalrec import download as dl


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = tmp_path_factory.mktemp("data")
    assert cli.main(["synthetic", "--data-dir", str(data), "--users", "60", "--items", "80"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(data)]) == 0
    return data


@pytest.mark.parametrize("grid", ["10", "x:1", "0:0", "10:-1"])
def test_a_bad_grid_is_a_one_line_error(
    grid: str, data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Defect: cli._grid (line 68) and the ItemKNN parameter checks raise a bare ValueError, which main (line 112)
    doesn't catch, so a typo in --grid ends in a traceback.

    Input: ``signalrec baselines synthetic --grid <grid>`` for "10" (no shrink), "x:1", "0:0" (k < 1), "10:-1".
    Expected: an argparse error (SystemExit 2) or exit code 1 with a "signalrec: " message, like the other errors.
    Actual: ValueError propagates out of cli.main.
    """
    args = ["baselines", "synthetic", "--data-dir", str(data_dir), "--grid", grid, "--out", str(tmp_path / "b.json")]
    try:
        code = cli.main(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
    assert code in (1, 2)
    assert "Traceback" not in capsys.readouterr().err


def test_a_corrupt_manifest_is_a_one_line_error(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Defect: prepare.read_manifest (prepare.py:229) lets json.JSONDecodeError through, so a truncated
    manifest.json (the file whose job is to catch a damaged processed directory) ends ``signalrec baselines`` in a
    traceback instead of the one-line error and exit 1 that a damaged .npy file gets.

    Input: a copy of a processed directory whose manifest.json is cut to its first 10 bytes.
    Expected: exit code 1 and "signalrec: ..." on stderr.
    Actual: JSONDecodeError propagates out of cli.main.
    """
    copy = tmp_path / "data"
    shutil.copytree(data_dir / "processed", copy / "processed")
    manifest = copy / "processed" / "synthetic" / "manifest.json"
    manifest.write_bytes(manifest.read_bytes()[:10])
    args = ["baselines", "synthetic", "--data-dir", str(copy), "--grid", "5:0", "--out", str(tmp_path / "b.json")]
    assert cli.main(args) == 1
    assert capsys.readouterr().err.startswith("signalrec: ")


def test_a_failed_fetch_is_a_download_error(tmp_path: Path) -> None:
    """Defect: download.fetch (line 98) lets the opener's URLError/OSError through, although DownloadError is
    documented as "the archive couldn't be fetched, ..." and it's the only download error the CLI catches.

    Input: an archive whose URL can't be opened (a file:// URL of a missing file: no network).
    Expected: DownloadError.
    Actual: urllib.error.URLError.
    """
    archive = dl.Archive("ml-x", (tmp_path / "missing.zip").as_uri(), "0" * 64, 1, (("ml-x/a", "a"),))
    with pytest.raises(dl.DownloadError):
        dl.fetch(archive, tmp_path / "zips")


def test_download_without_a_connection_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Defect (CLI side of the above): ``signalrec download`` with GroupLens unreachable raises URLError out of
    cli.main instead of printing "signalrec: ..." and returning 1.

    Input: ml-latest-small's pin with its URL pointed at a missing file:// path.
    Expected: exit code 1 and a one-line message on stderr.
    Actual: URLError propagates.
    """
    real = download.ARCHIVES["ml-latest-small"]
    unreachable = dl.Archive(real.name, (tmp_path / "missing.zip").as_uri(), real.sha256, real.size, real.members)
    monkeypatch.setitem(download.ARCHIVES, "ml-latest-small", unreachable)
    assert cli.main(["download", "ml-latest-small", "--data-dir", str(tmp_path / "data")]) == 1
    assert capsys.readouterr().err.startswith("signalrec: ")
