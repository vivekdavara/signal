"""Strict parsers for the two MovieLens formats the pipeline reads.

ml-latest-small is CSV with a header; ml-1m is ``::``-separated without one. Every line must match its format exactly;
a violation raises :class:`ParseError` naming the file and line. Nothing is coerced: " 4.0", "4e0", "nan", "+1" and an
extra field are all errors, not ratings.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]


class ParseError(ValueError):
    """A MovieLens file didn't match its format."""


@dataclass(frozen=True)
class Format:
    ratings_file: str
    movies_file: str
    encoding: str
    newline: str
    ratings_header: str | None
    # Groups: user, movie, rating, timestamp.
    ratings_line: re.Pattern[str]
    allowed_ratings: frozenset[float]


HALF_STARS = frozenset(x / 2 for x in range(1, 11))
WHOLE_STARS = frozenset(float(x) for x in range(1, 6))

FORMATS: dict[str, Format] = {
    "ml-latest-small": Format(
        ratings_file="ratings.csv",
        movies_file="movies.csv",
        encoding="utf-8",
        # Every line of both files ends in CRLF (checked 2026-11-07; GroupLens's README doesn't say).
        newline="\r\n",
        ratings_header="userId,movieId,rating,timestamp",
        ratings_line=re.compile(r"([1-9][0-9]{0,14}),([1-9][0-9]{0,14}),([0-5]\.[05]),(0|[1-9][0-9]{0,14})"),
        allowed_ratings=HALF_STARS,
    ),
    "ml-1m": Format(
        ratings_file="ratings.dat",
        movies_file="movies.dat",
        # The ml-1m README doesn't name an encoding. movies.dat holds 62 bytes above 0x7f, each alone (0xe9 for
        # "é" in "Café"), which is Latin-1 and not UTF-8 (checked 2026-11-07); ratings.dat is ASCII. LF endings.
        encoding="latin-1",
        newline="\n",
        ratings_header=None,
        ratings_line=re.compile(r"([1-9][0-9]{0,14})::([1-9][0-9]{0,14})::([1-5])::(0|[1-9][0-9]{0,14})"),
        allowed_ratings=WHOLE_STARS,
    ),
}

# The patterns allow at most 15 digits for an id or timestamp, so every value fits in int64.


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    genres: tuple[str, ...]


@dataclass(frozen=True)
class Ratings:
    """One row per rating, in file order. Ids are MovieLens's raw ids."""

    users: IntArray
    items: IntArray
    ratings: FloatArray
    times: IntArray

    def __len__(self) -> int:
        return len(self.users)


@dataclass(frozen=True)
class Dataset:
    name: str
    ratings: Ratings
    movies: dict[int, Movie]


def _lines(path: Path, fmt: Format) -> list[str]:
    """The file's lines without their endings; each must end in exactly ``fmt.newline``."""
    try:
        text = path.read_bytes().decode(fmt.encoding)
    except UnicodeDecodeError as exc:
        raise ParseError(f"{path}: not {fmt.encoding}: {exc}") from None
    if not text:
        return []
    if not text.endswith(fmt.newline):
        raise ParseError(f"{path}: the last line doesn't end in {fmt.newline!r} (truncated file?)")
    lines = text.split(fmt.newline)[:-1]
    for n, line in enumerate(lines):
        if "\r" in line or "\n" in line:
            raise ParseError(f"{path}:{n + 1}: a line ending other than {fmt.newline!r}")
    return lines


def parse_ratings(path: Path, fmt: Format) -> Ratings:
    lines = _lines(path, fmt)
    first = 0
    if fmt.ratings_header is not None:
        if not lines or lines[0] != fmt.ratings_header:
            got = lines[0] if lines else "an empty file"
            raise ParseError(f"{path}:1: expected the header {fmt.ratings_header!r}, got {got!r}")
        first = 1
    body = lines[first:]
    match = fmt.ratings_line.fullmatch
    groups: list[tuple[str, ...]] = []
    for row, line in enumerate(body):
        m = match(line)
        if m is None:
            raise ParseError(f"{path}:{row + first + 1}: not a rating line: {line!r}")
        groups.append(m.groups())
    # Every line matched its pattern, so the columns convert in bulk.
    columns = list(zip(*groups, strict=True)) if groups else [(), (), (), ()]
    users = np.array(columns[0], dtype=np.str_).astype(np.int64)
    items = np.array(columns[1], dtype=np.str_).astype(np.int64)
    ratings = np.array(columns[2], dtype=np.str_).astype(np.float32)
    times = np.array(columns[3], dtype=np.str_).astype(np.int64)
    allowed = np.array(sorted(fmt.allowed_ratings), dtype=np.float32)
    outside = ~np.isin(ratings, allowed)
    if outside.any():
        row = int(np.flatnonzero(outside)[0])
        raise ParseError(f"{path}:{row + first + 1}: rating {columns[2][row]} is not one of this dataset's ratings")
    _refuse_duplicates(path, users, items, first)
    return Ratings(users, items, ratings, times)


def _refuse_duplicates(path: Path, users: IntArray, items: IntArray, first: int) -> None:
    if len(users) < 2:
        return
    order = np.lexsort((items, users))
    same = (users[order[1:]] == users[order[:-1]]) & (items[order[1:]] == items[order[:-1]])
    if same.any():
        k = int(np.flatnonzero(same)[0])
        a, b = sorted((int(order[k]), int(order[k + 1])))
        raise ParseError(
            f"{path}:{b + first + 1}: user {users[b]} rated movie {items[b]} again (first on line {a + first + 1})"
        )


def _movie(path: Path, line_no: int, raw_id: str, title: str, genres: str) -> Movie:
    if not re.fullmatch(r"[1-9][0-9]{0,14}", raw_id):
        raise ParseError(f"{path}:{line_no}: bad movie id {raw_id!r}")
    # Titles are display text: 11 in ml-latest-small end in a space ("96 Minutes (2011) "), so they're stripped.
    title = title.strip()
    if not title:
        raise ParseError(f"{path}:{line_no}: an empty title")
    names = () if genres == "(no genres listed)" else tuple(genres.split("|"))
    if any(not g or g != g.strip() for g in names):
        raise ParseError(f"{path}:{line_no}: bad genres {genres!r}")
    return Movie(int(raw_id), title, names)


def parse_movies(path: Path, fmt: Format) -> dict[int, Movie]:
    movies: dict[int, Movie] = {}
    if fmt.ratings_header is None:
        rows = [(n + 1, line.split("::")) for n, line in enumerate(_lines(path, fmt))]
    else:
        lines = _lines(path, fmt)
        if not lines or lines[0] != "movieId,title,genres":
            raise ParseError(f"{path}:1: expected the header 'movieId,title,genres'")
        # csv handles the quoted titles ("American President, The (1995)"); the line numbers stay right because
        # no title spans lines (checked: each row is one record).
        rows = [(n + 2, row) for n, row in enumerate(csv.reader(lines[1:], strict=True))]
    for line_no, fields in rows:
        if len(fields) != 3:
            raise ParseError(f"{path}:{line_no}: expected 3 fields, got {len(fields)}")
        movie = _movie(path, line_no, *fields)
        if movie.movie_id in movies:
            raise ParseError(f"{path}:{line_no}: movie {movie.movie_id} listed twice")
        movies[movie.movie_id] = movie
    return movies


def load(name: str, raw_dir: Path) -> Dataset:
    """Parse ``raw_dir`` (``data/raw/<name>``) and check that every rated movie is in the movie list."""
    try:
        fmt = FORMATS[name]
    except KeyError:
        raise ParseError(f"no parser for {name!r}; known: {', '.join(sorted(FORMATS))}") from None
    ratings = parse_ratings(raw_dir / fmt.ratings_file, fmt)
    movies = parse_movies(raw_dir / fmt.movies_file, fmt)
    rated = np.unique(ratings.items)
    known = np.fromiter(movies, dtype=np.int64, count=len(movies))
    unknown = np.setdiff1d(rated, known)
    if len(unknown):
        raise ParseError(f"{raw_dir}: {len(unknown)} rated movies are not in {fmt.movies_file}, e.g. {unknown[0]}")
    return Dataset(name, ratings, movies)
