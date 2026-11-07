"""Download the MovieLens archives by pinned hash and extract only the files the pipeline reads.

The archives are never committed (see DESIGN.md, "Data"). A download is accepted only if its SHA-256 and size match the
pins below; extraction reads the expected member names and nothing else, into a temporary directory that is renamed
into place, so a hostile or corrupt archive can't write outside ``data/raw/<name>/`` and a crash can't leave a
half-extracted dataset that looks complete.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

CHUNK = 1 << 16
# The largest member any pinned archive holds is ml-1m/ratings.dat (24,594,131 bytes); anything declaring more than
# this is refused before a byte is written.
MAX_MEMBER_BYTES = 64 << 20
MARKER = ".source-sha256"


class DownloadError(Exception):
    """The archive couldn't be fetched, didn't match its pin, or lacked an expected member."""


@dataclass(frozen=True)
class Archive:
    name: str
    url: str
    sha256: str
    size: int
    # Member name inside the zip -> file name under data/raw/<name>/.
    members: tuple[tuple[str, str], ...]


# SHA-256 taken on 2026-11-07 from the files GroupLens served; their MD5s matched GroupLens's published .md5 files
# (31a303aabbc519bd33d025e44d6c2570 and c4d9eecfca2ab87c1945afe126590906).
ARCHIVES: dict[str, Archive] = {
    "ml-latest-small": Archive(
        name="ml-latest-small",
        url="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        sha256="696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436",
        size=978_202,
        members=(
            ("ml-latest-small/ratings.csv", "ratings.csv"),
            ("ml-latest-small/movies.csv", "movies.csv"),
            ("ml-latest-small/README.txt", "README.txt"),
        ),
    ),
    "ml-1m": Archive(
        name="ml-1m",
        url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        sha256="a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
        size=5_917_549,
        members=(
            ("ml-1m/ratings.dat", "ratings.dat"),
            ("ml-1m/movies.dat", "movies.dat"),
            ("ml-1m/README", "README"),
        ),
    ),
}

Opener = Callable[[str], IO[bytes]]


def _open_url(url: str) -> IO[bytes]:
    # Only ever called with a URL from ARCHIVES (tests use file:// URLs of archives they built).
    response: IO[bytes] = urllib.request.urlopen(url, timeout=60)
    return response


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(archive: Archive, zip_dir: Path, opener: Opener = _open_url) -> Path:
    """Return the path of the archive's zip under ``zip_dir``, downloading it unless a matching copy is there."""
    zip_dir.mkdir(parents=True, exist_ok=True)
    target = zip_dir / f"{archive.name}.zip"
    if target.exists() and target.stat().st_size == archive.size and sha256_of(target) == archive.sha256:
        return target
    fd, tmp_name = tempfile.mkstemp(prefix=f".{archive.name}-", suffix=".part", dir=zip_dir)
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb") as out, opener(archive.url) as response:
                while chunk := response.read(CHUNK):
                    size += len(chunk)
                    if size > archive.size:
                        raise DownloadError(f"{archive.url}: more than the pinned {archive.size} bytes; stopped")
                    digest.update(chunk)
                    out.write(chunk)
        except OSError as exc:  # URLError, timeouts, a dropped connection, a full disk
            raise DownloadError(f"{archive.url}: {exc}") from exc
        if size != archive.size:
            raise DownloadError(f"{archive.url}: got {size} bytes, pinned {archive.size}")
        if digest.hexdigest() != archive.sha256:
            raise DownloadError(f"{archive.url}: SHA-256 {digest.hexdigest()}, pinned {archive.sha256}")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def _complete(archive: Archive, directory: Path) -> bool:
    marker = directory / MARKER
    return (
        marker.is_file()
        and marker.read_text().strip() == archive.sha256
        and all((directory / out).is_file() for _, out in archive.members)
    )


def extract(archive: Archive, zip_path: Path, raw_dir: Path) -> Path:
    """Extract the archive's expected members to ``raw_dir/<name>/`` and return that directory."""
    target = raw_dir / archive.name
    if _complete(archive, target):
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{archive.name}-", dir=raw_dir))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member, out_name in archive.members:
                try:
                    info = zf.getinfo(member)
                except KeyError:
                    raise DownloadError(f"{zip_path}: no member {member!r}") from None
                if info.file_size > MAX_MEMBER_BYTES:
                    raise DownloadError(f"{zip_path}: {member!r} declares {info.file_size} bytes")
                # ZipFile checks the CRC when the member is read to its end, and stops at the declared size.
                with zf.open(info) as src, (tmp / out_name).open("wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
        (tmp / MARKER).write_text(archive.sha256 + "\n")
        if target.exists():
            # Only an incomplete earlier extraction (or one from another archive version) gets here.
            shutil.rmtree(target)
        os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def download(name: str, data_dir: Path, opener: Opener = _open_url) -> Path:
    """Fetch and extract a pinned dataset; return ``data_dir/raw/<name>``."""
    try:
        archive = ARCHIVES[name]
    except KeyError:
        raise DownloadError(f"unknown dataset {name!r}; known: {', '.join(sorted(ARCHIVES))}") from None
    raw = data_dir / "raw" / name
    if _complete(archive, raw):
        return raw
    zip_path = fetch(archive, data_dir / "zips", opener)
    try:
        return extract(archive, zip_path, data_dir / "raw")
    except (zipfile.BadZipFile, zlib.error) as exc:
        raise DownloadError(f"{zip_path}: {exc}") from exc
