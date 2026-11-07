"""The MovieLens parsers on hand-made files, and on the real files when they've been downloaded."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from signalrec import movielens
from signalrec.movielens import FORMATS, ParseError
from tests.conftest import real_raw_dir

SMALL = FORMATS["ml-latest-small"]
ONE_M = FORMATS["ml-1m"]


def write(path: Path, lines: list[str], newline: str, encoding: str = "utf-8") -> Path:
    path.write_bytes("".join(line + newline for line in lines).encode(encoding))
    return path


def small_ratings(tmp_path: Path, rows: list[str]) -> Path:
    return write(tmp_path / "ratings.csv", ["userId,movieId,rating,timestamp", *rows], "\r\n")


def test_csv_ratings_parse_to_typed_arrays(tmp_path: Path) -> None:
    path = small_ratings(tmp_path, ["1,31,2.5,1260759144", "1,1029,3.0,1260759179", "7,1,0.5,0"])
    r = movielens.parse_ratings(path, SMALL)
    assert r.users.tolist() == [1, 1, 7]
    assert r.items.tolist() == [31, 1029, 1]
    assert r.ratings.tolist() == [2.5, 3.0, 0.5]
    assert r.times.tolist() == [1260759144, 1260759179, 0]
    assert (r.users.dtype, r.ratings.dtype, r.times.dtype) == (
        np.dtype(np.int64),
        np.dtype(np.float32),
        np.dtype(np.int64),
    )


def test_dat_ratings_parse(tmp_path: Path) -> None:
    path = write(tmp_path / "ratings.dat", ["1::1193::5::978300760", "2::661::3::978302109"], "\n")
    r = movielens.parse_ratings(path, ONE_M)
    assert (r.users.tolist(), r.items.tolist(), r.ratings.tolist()) == ([1, 2], [1193, 661], [5.0, 3.0])


def test_a_header_only_file_is_empty(tmp_path: Path) -> None:
    r = movielens.parse_ratings(small_ratings(tmp_path, []), SMALL)
    assert len(r) == 0
    assert r.users.dtype == np.int64


@pytest.mark.parametrize(
    "row",
    [
        "1,31,2.5",  # a missing field
        "1,31,2.5,1260759144,9",  # an extra field
        "1,31, 2.5,1260759144",  # whitespace
        "1,31,2.5e0,1260759144",
        "1,31,nan,1260759144",
        "1,31,2.25,1260759144",  # not a half star
        "1,31,5.5,1260759144",
        "1,31,0.0,1260759144",
        "0,31,2.5,1260759144",  # ids start at 1
        "-1,31,2.5,1260759144",
        "+1,31,2.5,1260759144",
        "01,31,2.5,1260759144",
        "1,31,2.5,-5",
        "1,31,2.5,1.5",
        "1234567890123456,31,2.5,1",  # 16 digits
        "1,31,2.5,1260759144\n",  # a bare LF inside a CRLF file
        '"1",31,2.5,1260759144',
    ],
)
def test_malformed_csv_rows_are_refused_with_their_line(tmp_path: Path, row: str) -> None:
    path = small_ratings(tmp_path, ["1,1,4.0,1", row])
    with pytest.raises(ParseError, match=r"ratings\.csv:3: "):
        movielens.parse_ratings(path, SMALL)


@pytest.mark.parametrize("row", ["1::1193::4.5::978300760", "1::1193::0::978300760", "1::1193::6::978300760"])
def test_ml_1m_ratings_are_whole_stars(tmp_path: Path, row: str) -> None:
    path = write(tmp_path / "ratings.dat", [row], "\n")
    with pytest.raises(ParseError, match=r"ratings\.dat:1: "):
        movielens.parse_ratings(path, ONE_M)


def test_the_header_must_match_exactly(tmp_path: Path) -> None:
    path = write(tmp_path / "ratings.csv", ["userId,movieId,rating,timestamp ", "1,1,4.0,1"], "\r\n")
    with pytest.raises(ParseError, match="expected the header"):
        movielens.parse_ratings(path, SMALL)


def test_the_line_ending_must_be_the_formats(tmp_path: Path) -> None:
    lf = write(tmp_path / "ratings.csv", ["userId,movieId,rating,timestamp", "1,1,4.0,1"], "\n")
    with pytest.raises(ParseError, match="doesn't end in"):
        movielens.parse_ratings(lf, SMALL)
    crlf = write(tmp_path / "ratings.dat", ["1::1::4::1", "2::1::4::1"], "\r\n")
    with pytest.raises(ParseError, match=r"ratings\.dat:1: a line ending other than"):
        movielens.parse_ratings(crlf, ONE_M)


def test_a_truncated_last_line_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ratings.dat"
    path.write_bytes(b"1::1::4::1\n2::1::4")
    with pytest.raises(ParseError, match="truncated"):
        movielens.parse_ratings(path, ONE_M)


def test_a_repeated_pair_names_both_lines(tmp_path: Path) -> None:
    path = small_ratings(tmp_path, ["1,5,4.0,1", "2,5,4.0,1", "1,6,4.0,1", "1,5,3.0,9"])
    with pytest.raises(ParseError, match=re.escape("ratings.csv:5: user 1 rated movie 5 again (first on line 2)")):
        movielens.parse_ratings(path, SMALL)


def test_csv_movies_handle_quotes_genres_and_padding(tmp_path: Path) -> None:
    path = write(
        tmp_path / "movies.csv",
        [
            "movieId,title,genres",
            '11,"American President, The (1995)",Comedy|Drama|Romance',
            "1,96 Minutes (2011) ,Drama|Thriller",
            "182727,A Christmas Story Live! (2017),(no genres listed)",
            "5,Café (2010),Drama",
        ],
        "\r\n",
    )
    movies = movielens.parse_movies(path, SMALL)
    assert movies[11] == movielens.Movie(11, "American President, The (1995)", ("Comedy", "Drama", "Romance"))
    assert movies[1].title == "96 Minutes (2011)"
    assert movies[182727].genres == ()
    assert movies[5].title == "Café (2010)"


def test_dat_movies_are_latin_1(tmp_path: Path) -> None:
    path = write(tmp_path / "movies.dat", ["1::Café (1999)::Comedy|Children's"], "\n", encoding="latin-1")
    assert movielens.parse_movies(path, ONE_M)[1] == movielens.Movie(1, "Café (1999)", ("Comedy", "Children's"))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("1::Title::Comedy::Extra", "expected 3 fields, got 4"),
        ("x::Title::Comedy", "bad movie id"),
        ("1:: ::Comedy", "an empty title"),
        ("1::Title::Comedy||Drama", "bad genres"),
    ],
)
def test_malformed_movies_are_refused(tmp_path: Path, line: str, message: str) -> None:
    path = write(tmp_path / "movies.dat", [line], "\n")
    with pytest.raises(ParseError, match=message):
        movielens.parse_movies(path, ONE_M)


def test_a_movie_listed_twice_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "movies.dat", ["1::A (1999)::Comedy", "1::B (1999)::Drama"], "\n")
    with pytest.raises(ParseError, match="movie 1 listed twice"):
        movielens.parse_movies(path, ONE_M)


def test_utf_8_files_refuse_other_encodings(tmp_path: Path) -> None:
    path = tmp_path / "movies.csv"
    path.write_bytes(b"movieId,title,genres\r\n1,Caf\xe9 (2010),Drama\r\n")
    with pytest.raises(ParseError, match="not utf-8"):
        movielens.parse_movies(path, SMALL)


def test_every_rated_movie_must_be_listed(tmp_path: Path) -> None:
    write(tmp_path / "ratings.dat", ["1::1::4::1", "1::2::4::1"], "\n")
    write(tmp_path / "movies.dat", ["1::A (1999)::Comedy"], "\n")
    with pytest.raises(ParseError, match=re.escape("1 rated movies are not in movies.dat, e.g. 2")):
        movielens.load("ml-1m", tmp_path)


# The counts GroupLens's READMEs state (ml-1m: "1,000,209 anonymous ratings of approximately 3,900 movies made by
# 6,040 MovieLens users"; ml-latest-small: "100836 ratings ... across 9742 movies ... by 610 users").
REAL = {
    "ml-latest-small": (100_836, 610, 9_724, 9_742),
    "ml-1m": (1_000_209, 6_040, 3_706, 3_883),
}


@pytest.mark.parametrize("name", sorted(REAL))
def test_the_real_files_parse_to_the_published_counts(name: str) -> None:
    dataset = movielens.load(name, real_raw_dir(name))
    ratings, users, rated, listed = REAL[name]
    assert len(dataset.ratings) == ratings
    assert len(np.unique(dataset.ratings.users)) == users
    assert len(np.unique(dataset.ratings.items)) == rated
    assert len(dataset.movies) == listed
