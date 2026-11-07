"""The ``signalrec`` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from signalrec import download, movielens, prepare, synthetic

DEFAULT_DATA = Path("data")


def _download(args: argparse.Namespace) -> int:
    raw = download.download(args.dataset, args.data_dir)
    dataset = movielens.load(args.dataset, raw)
    users = len(set(dataset.ratings.users.tolist()))
    rated = len(set(dataset.ratings.items.tolist()))
    print(f"{raw}: {len(dataset.ratings)} ratings, {users} users, {rated} rated movies, {len(dataset.movies)} movies")
    return 0


def _synthetic(args: argparse.Namespace) -> int:
    raw = synthetic.write(args.data_dir / "raw" / "synthetic", n_users=args.users, n_items=args.items, seed=args.seed)
    print(f"{raw}: synthetic ratings ({args.users} users, {args.items} movies, seed {args.seed})")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    raw = args.data_dir / "raw" / args.dataset
    fmt = movielens.FORMATS[args.dataset]
    dataset = movielens.load(args.dataset, raw)
    params = prepare.Params(args.min_rating, args.min_user_positives, args.min_item_positives)
    prepared = prepare.prepare(dataset, params)
    sources = {name: prepare.sha256_of(raw / name) for name in (fmt.ratings_file, fmt.movies_file)}
    out = args.data_dir / "processed" / args.dataset
    manifest = prepare.save(prepared, out, args.dataset, sources, params)
    print(f"{out}: " + ", ".join(f"{k} {v}" for k, v in manifest["counts"].items()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signalrec", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch a MovieLens archive by pinned hash, extract it and check it parses")
    p.add_argument("dataset", choices=sorted(download.ARCHIVES))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.set_defaults(run=_download)

    p = sub.add_parser("synthetic", help="write a SYNTHETIC dataset in ml-latest-small's format (tests and CI)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--items", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(run=_synthetic)

    p = sub.add_parser("prepare", help="positives, k-core, leave-last-out split; writes data/processed/<dataset>")
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--min-rating", type=float, default=prepare.Params.min_rating)
    p.add_argument("--min-user-positives", type=int, default=prepare.Params.min_user_positives)
    p.add_argument("--min-item-positives", type=int, default=prepare.Params.min_item_positives)
    p.set_defaults(run=_prepare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.run(args)
    except (download.DownloadError, movielens.ParseError, prepare.PrepareError, FileNotFoundError) as exc:
        print(f"signalrec: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
