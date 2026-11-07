"""The ``signalrec`` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from signalrec import download, movielens

DEFAULT_DATA = Path("data")


def _download(args: argparse.Namespace) -> int:
    raw = download.download(args.dataset, args.data_dir)
    dataset = movielens.load(args.dataset, raw)
    users = len(set(dataset.ratings.users.tolist()))
    rated = len(set(dataset.ratings.items.tolist()))
    print(f"{raw}: {len(dataset.ratings)} ratings, {users} users, {rated} rated movies, {len(dataset.movies)} movies")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signalrec", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch a MovieLens archive by pinned hash, extract it and check it parses")
    p.add_argument("dataset", choices=sorted(download.ARCHIVES))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.set_defaults(run=_download)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.run(args)
    except (download.DownloadError, movielens.ParseError) as exc:
        print(f"signalrec: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
