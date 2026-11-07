"""Review round 2 (day 1): fetch's OSError -> DownloadError mapping misses http.client's own exceptions."""

from __future__ import annotations

import hashlib
import http.client
from pathlib import Path
from types import TracebackType
from typing import IO

import pytest

from signalrec import download as dl
from signalrec.download import Archive, DownloadError


class DroppedChunkedResponse:
    """A response whose body ends early under chunked transfer encoding: http.client raises IncompleteRead."""

    def __init__(self) -> None:
        self.calls = 0

    def read(self, n: int = -1) -> bytes:
        self.calls += 1
        if self.calls == 1:
            return b"PK\x03\x04partial"
        raise http.client.IncompleteRead(b"", 100)

    def __enter__(self) -> DroppedChunkedResponse:
        return self

    def __exit__(self, kind: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        return None


def test_a_connection_dropped_mid_body_is_a_download_error(tmp_path: Path) -> None:
    """Defect: download.fetch (line 107) converts only OSError ("URLError, timeouts, a dropped connection") to
    DownloadError. http.client.HTTPException is not an OSError: IncompleteRead (a chunked body cut short),
    BadStatusLine and LineTooLong escape as themselves, and cli.main (line 171) catches none of them, so
    ``signalrec download`` ends in a traceback, the case round 1's fix meant to turn into one line.

    Input: an opener whose response returns one chunk and then raises http.client.IncompleteRead.
    Expected: DownloadError naming the URL, no .part file left. Actual: http.client.IncompleteRead propagates.
    """
    archive = Archive("ml-x", "https://example.invalid/ml-x.zip", hashlib.sha256(b"x").hexdigest(), 1000, ())

    def opener(url: str) -> IO[bytes]:
        response: IO[bytes] = DroppedChunkedResponse()  # type: ignore[assignment]
        return response

    try:
        dl.fetch(archive, tmp_path, opener)
    except DownloadError as exc:
        assert archive.url in str(exc)
    except http.client.HTTPException as exc:
        pytest.fail(f"{type(exc).__name__} escaped fetch instead of a DownloadError")
    else:
        pytest.fail("fetch accepted a truncated body")
    assert sorted(p.name for p in tmp_path.iterdir()) == []
