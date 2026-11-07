"""The downloader against archives built here and served from file:// URLs: no network."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import IO

import pytest

from signalrec import download as dl
from signalrec.download import Archive, DownloadError

RATINGS = b"userId,movieId,rating,timestamp\n1,1,4.0,964982703\n"
MOVIES = b"movieId,title,genres\n1,Toy Story (1995),Animation\n"


def make_zip(path: Path, members: dict[str, bytes], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path.read_bytes()


def archive_for(path: Path, name: str = "ml-x") -> Archive:
    data = path.read_bytes()
    return Archive(
        name=name,
        url=path.as_uri(),
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        members=((f"{name}/ratings.csv", "ratings.csv"), (f"{name}/movies.csv", "movies.csv")),
    )


@pytest.fixture
def served(tmp_path: Path) -> Archive:
    src = tmp_path / "served"
    src.mkdir()
    make_zip(src / "ml-x.zip", {"ml-x/ratings.csv": RATINGS, "ml-x/movies.csv": MOVIES, "ml-x/links.csv": b"x"})
    return archive_for(src / "ml-x.zip")


def install(monkeypatch: pytest.MonkeyPatch, archive: Archive) -> None:
    monkeypatch.setitem(dl.ARCHIVES, archive.name, archive)


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith("."))


def test_download_extracts_exactly_the_expected_members(
    tmp_path: Path, served: Archive, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, served)
    data = tmp_path / "data"
    raw = dl.download("ml-x", data)
    assert raw == data / "raw" / "ml-x"
    assert sorted(p.name for p in raw.iterdir()) == [dl.MARKER, "movies.csv", "ratings.csv"]
    assert (raw / "ratings.csv").read_bytes() == RATINGS
    assert (raw / "movies.csv").read_bytes() == MOVIES
    assert (raw / dl.MARKER).read_text() == served.sha256 + "\n"
    assert leftovers(data / "raw") == []
    assert leftovers(data / "zips") == []


def test_a_second_download_opens_nothing(tmp_path: Path, served: Archive, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, served)
    dl.download("ml-x", tmp_path)

    def refuse(url: str) -> IO[bytes]:
        raise AssertionError(f"opened {url}")

    assert dl.download("ml-x", tmp_path, opener=refuse) == tmp_path / "raw" / "ml-x"


def test_a_complete_extraction_is_kept_even_without_the_zip(
    tmp_path: Path, served: Archive, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, served)
    dl.download("ml-x", tmp_path)
    (tmp_path / "zips" / "ml-x.zip").unlink()

    def refuse(url: str) -> IO[bytes]:
        raise AssertionError(f"opened {url}")

    dl.download("ml-x", tmp_path, opener=refuse)


def test_a_hash_mismatch_fails_and_leaves_nothing(tmp_path: Path, served: Archive) -> None:
    wrong = Archive(served.name, served.url, "0" * 64, served.size, served.members)
    with pytest.raises(DownloadError, match=f"SHA-256 {served.sha256}, pinned {'0' * 64}"):
        dl.fetch(wrong, tmp_path / "zips")
    assert list((tmp_path / "zips").iterdir()) == []


def test_a_bad_cached_zip_is_downloaded_again(tmp_path: Path, served: Archive) -> None:
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "ml-x.zip").write_bytes(b"not the archive")
    path = dl.fetch(served, zips)
    assert dl.sha256_of(path) == served.sha256


def test_a_response_longer_than_the_pin_is_cut_off(tmp_path: Path, served: Archive) -> None:
    reads = 0

    class Endless(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
            nonlocal reads
            reads += 1
            buffer[:] = b"\0" * len(buffer)
            return len(buffer)

    def opener(url: str) -> IO[bytes]:
        return io.BufferedReader(Endless())

    with pytest.raises(DownloadError, match="more than the pinned"):
        dl.fetch(served, tmp_path / "zips", opener=opener)
    assert reads <= served.size // dl.CHUNK + 2
    assert list((tmp_path / "zips").iterdir()) == []


def test_a_short_response_fails(tmp_path: Path, served: Archive) -> None:
    def opener(url: str) -> IO[bytes]:
        return io.BytesIO(b"short")

    with pytest.raises(DownloadError, match="got 5 bytes"):
        dl.fetch(served, tmp_path / "zips", opener=opener)


def test_hostile_members_are_never_written(tmp_path: Path) -> None:
    src = tmp_path / "served"
    src.mkdir()
    make_zip(
        src / "ml-x.zip",
        {
            "../escaped.csv": b"evil",
            "/abs.csv": b"evil",
            "ml-x/../../up.csv": b"evil",
            "ml-x/ratings.csv": RATINGS,
            "ml-x/movies.csv": MOVIES,
        },
    )
    archive = archive_for(src / "ml-x.zip")
    raw = tmp_path / "work" / "raw"
    dl.extract(archive, src / "ml-x.zip", raw)
    written = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert written == [
        "served/ml-x.zip",
        f"work/raw/ml-x/{dl.MARKER}",
        "work/raw/ml-x/movies.csv",
        "work/raw/ml-x/ratings.csv",
    ]


def test_a_missing_member_fails_before_anything_is_in_place(tmp_path: Path) -> None:
    src = tmp_path / "served"
    src.mkdir()
    make_zip(src / "ml-x.zip", {"ml-x/ratings.csv": RATINGS})
    raw = tmp_path / "raw"
    with pytest.raises(DownloadError, match=re.escape("no member 'ml-x/movies.csv'")):
        dl.extract(archive_for(src / "ml-x.zip"), src / "ml-x.zip", raw)
    assert list(raw.iterdir()) == []


def test_a_corrupt_member_fails_on_its_crc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "served"
    src.mkdir()
    data = bytearray(make_zip(src / "ml-x.zip", {"ml-x/ratings.csv": RATINGS, "ml-x/movies.csv": MOVIES}, 0))
    at = data.index(b"Toy Story")
    data[at] ^= 0x20
    (src / "ml-x.zip").write_bytes(bytes(data))
    archive = archive_for(src / "ml-x.zip")  # pinned to the corrupt bytes, so only the CRC can catch it
    install(monkeypatch, archive)
    with pytest.raises(DownloadError, match="CRC"):
        dl.download("ml-x", tmp_path / "data", opener=lambda url: (src / "ml-x.zip").open("rb"))
    assert list((tmp_path / "data" / "raw").iterdir()) == []


def test_a_member_declaring_too_many_bytes_is_refused(
    tmp_path: Path, served: Archive, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = dl.fetch(served, tmp_path / "zips")
    monkeypatch.setattr(dl, "MAX_MEMBER_BYTES", len(RATINGS) - 1)
    with pytest.raises(DownloadError, match="declares"):
        dl.extract(served, zip_path, tmp_path / "raw")
    assert list((tmp_path / "raw").iterdir()) == []


def test_an_incomplete_extraction_is_replaced(tmp_path: Path, served: Archive) -> None:
    zip_path = dl.fetch(served, tmp_path / "zips")
    stale = tmp_path / "raw" / "ml-x"
    stale.mkdir(parents=True)
    (stale / "ratings.csv").write_bytes(b"half")
    dl.extract(served, zip_path, tmp_path / "raw")
    assert (stale / "ratings.csv").read_bytes() == RATINGS
    assert (stale / dl.MARKER).is_file()


def test_an_unknown_dataset_names_the_known_ones(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="known: ml-1m, ml-latest-small"):
        dl.download("ml-100k", tmp_path)


def test_the_pins_are_well_formed() -> None:
    for name, archive in dl.ARCHIVES.items():
        assert archive.name == name
        assert re.fullmatch(r"[0-9a-f]{64}", archive.sha256)
        assert archive.url == f"https://files.grouplens.org/datasets/movielens/{name}.zip"
        outs = [out for _, out in archive.members]
        assert len(set(outs)) == len(outs)
        assert all(member.startswith(f"{name}/") and "/" not in out for member, out in archive.members)


def test_a_marker_from_another_archive_means_extract_again(tmp_path: Path, served: Archive) -> None:
    zip_path = dl.fetch(served, tmp_path / "zips")
    target = dl.extract(served, zip_path, tmp_path / "raw")
    (target / dl.MARKER).write_text("f" * 64 + "\n")
    (target / "ratings.csv").write_bytes(b"from another version")
    dl.extract(served, zip_path, tmp_path / "raw")
    assert (target / "ratings.csv").read_bytes() == RATINGS


def test_a_missing_member_file_means_extract_again(tmp_path: Path, served: Archive) -> None:
    zip_path = dl.fetch(served, tmp_path / "zips")
    target = dl.extract(served, zip_path, tmp_path / "raw")
    (target / "movies.csv").unlink()
    dl.extract(served, zip_path, tmp_path / "raw")
    assert (target / "movies.csv").read_bytes() == MOVIES


def test_a_cached_zip_of_the_right_size_but_wrong_bytes_is_fetched_again(tmp_path: Path, served: Archive) -> None:
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "ml-x.zip").write_bytes(b"\0" * served.size)
    assert dl.sha256_of(dl.fetch(served, zips)) == served.sha256


def test_a_member_whose_deflate_stream_is_corrupt_is_a_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 0xFF bytes over the compressed data make zlib fail ("invalid block type") before any CRC check could; the
    # archive is pinned to the corrupt bytes, so only extraction can notice.
    src = tmp_path / "served"
    src.mkdir()
    data = bytearray(make_zip(src / "ml-x.zip", {"ml-x/ratings.csv": RATINGS * 50, "ml-x/movies.csv": MOVIES}))
    info = zipfile.ZipFile(src / "ml-x.zip").getinfo("ml-x/ratings.csv")
    start = info.header_offset + 30 + len(info.filename)
    data[start : start + info.compress_size] = b"\xff" * info.compress_size
    (src / "ml-x.zip").write_bytes(bytes(data))
    install(monkeypatch, archive_for(src / "ml-x.zip"))
    with pytest.raises(DownloadError, match="invalid block type"):
        dl.download("ml-x", tmp_path / "data")
    assert list((tmp_path / "data" / "raw").iterdir()) == []


def test_a_failed_fetch_names_the_url(tmp_path: Path) -> None:
    archive = Archive("ml-x", (tmp_path / "missing.zip").as_uri(), "0" * 64, 1, (("ml-x/a", "a"),))
    with pytest.raises(DownloadError, match=re.escape(archive.url)):
        dl.fetch(archive, tmp_path / "zips")
