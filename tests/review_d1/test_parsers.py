"""Review findings (day 1): the movies.csv parser's strictness gaps."""

from __future__ import annotations

from pathlib import Path

import pytest

from signalrec import movielens
from signalrec.movielens import FORMATS, ParseError

SMALL = FORMATS["ml-latest-small"]


def movies_csv(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "movies.csv"
    path.write_bytes(("".join(line + "\r\n" for line in ["movieId,title,genres", *rows])).encode())
    return path


def test_an_unbalanced_quote_does_not_swallow_the_next_line(tmp_path: Path) -> None:
    """Defect: movielens.parse_movies feeds the lines to csv.reader, which continues an unclosed quoted field onto
    the next line, so one malformed line silently merges with the next.

    Input: line 2 ``1,"Foo (1999),Drama`` (a quote that never closes) followed by line 3 ``2,Bar (1999)",Comedy``.
    Expected: a ParseError (the module promises every line matches its format exactly, one record per line).
    Actual: parsing succeeds; movie 1 gets the title 'Foo (1999),Drama2,Bar (1999)' and genre Comedy, and movie 2
    disappears from the catalogue without an error.
    """
    path = movies_csv(tmp_path, ['1,"Foo (1999),Drama', '2,Bar (1999)",Comedy', "3,Baz (2000),Drama"])
    with pytest.raises(ParseError, match=r"movies\.csv:2"):
        movielens.parse_movies(path, SMALL)


def test_a_line_after_a_merged_record_keeps_its_own_line_number(tmp_path: Path) -> None:
    """Defect: parse_movies numbers csv *records*, not file lines, so after a record spans two lines every later
    error names the wrong line.

    Input: an unclosed quote on line 2 that closes on line 3, then a bad movie id on line 4 (``x,Baz (2000),Drama``).
    Expected: either a ParseError naming line 2 (the bad quote) or one naming line 4 (the bad id).
    Actual: "movies.csv:3: bad movie id 'x'" (line 3 is a valid-looking line).
    """
    path = movies_csv(tmp_path, ['1,"Foo (1999),Drama', '2,Bar (1999)",Comedy', "x,Baz (2000),Drama"])
    with pytest.raises(ParseError, match=r"movies\.csv:[24]:"):
        movielens.parse_movies(path, SMALL)


def test_a_dat_title_ending_in_a_colon_is_not_misread(tmp_path: Path) -> None:
    """Defect: parse_movies splits ml-1m lines with ``line.split("::")`` (movielens.py:174), so a title ending in ':'
    loses its colon to the genres field, and since genre names aren't checked against anything, the line is accepted.

    Input: movies.dat line ``1::Foo (1999):::Comedy`` (title "Foo (1999):", genre Comedy).
    Expected: Movie(1, "Foo (1999):", ("Comedy",)), or a ParseError for the ambiguous line.
    Actual: Movie(1, "Foo (1999)", (":Comedy",)): a silently wrong title and genre.
    """
    path = tmp_path / "movies.dat"
    path.write_bytes(b"1::Foo (1999):::Comedy\n")
    try:
        movies = movielens.parse_movies(path, FORMATS["ml-1m"])
    except ParseError:
        return
    assert movies[1] == movielens.Movie(1, "Foo (1999):", ("Comedy",))


def test_malformed_csv_quoting_raises_a_parse_error_with_the_line(tmp_path: Path) -> None:
    """Defect: a csv.Error from the strict reader escapes parse_movies unconverted.

    Input: line 2 ``1,"Foo" (1999),Drama`` (text after a closing quote).
    Expected: ParseError naming movies.csv:2 (the module's contract; the CLI catches only ParseError, so this also
    turns a data error into a traceback).
    Actual: _csv.Error: ',' expected after '"', with no file or line.
    """
    path = movies_csv(tmp_path, ['1,"Foo" (1999),Drama'])
    with pytest.raises(ParseError, match=r"movies\.csv:2"):
        movielens.parse_movies(path, SMALL)
