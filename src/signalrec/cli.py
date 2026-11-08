"""The ``signalrec`` command line."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from signalrec import (
    download,
    experiment,
    experiment_twotower,
    leakage,
    leakage_twotower,
    movielens,
    prepare,
    recommend,
    synthetic,
)
from signalrec import twotower as tt
from signalrec.baselines import ItemKNN, Popularity

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


def _baselines(args: argparse.Namespace) -> int:
    processed = args.data_dir / "processed" / args.dataset
    prepared = prepare.load(processed, args.dataset)
    manifest = prepared.manifest
    if manifest is None:  # load always attaches the manifest it checked; not an assert, which -O would remove
        raise prepare.PrepareError(f"{processed}: loaded without its manifest")
    grid = experiment.DEFAULT_GRID if args.grid is None else tuple(args.grid)
    result = {
        "command": "signalrec " + shlex.join(args.argv),
        "dataset": args.dataset,
        "prepared": {"params": manifest["params"], "counts": manifest["counts"], "sources": manifest["sources"]},
        **experiment.run(prepared, grid),
    }
    out = args.out or Path("results") / args.dataset / "baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(experiment.table(result))
    print(f"wrote {out}")
    return 0


def _twotower(args: argparse.Namespace) -> int:
    processed = args.data_dir / "processed" / args.dataset
    prepared = prepare.load(processed, args.dataset)
    manifest = prepared.manifest
    if manifest is None:  # as in _baselines
        raise prepare.PrepareError(f"{processed}: loaded without its manifest")

    def progress(line: str) -> None:
        if not args.quiet:
            print(line, file=sys.stderr, flush=True)

    result = {
        "command": "signalrec " + shlex.join(args.argv),
        "dataset": args.dataset,
        "prepared": {"params": manifest["params"], "counts": manifest["counts"], "sources": manifest["sources"]},
        **experiment_twotower.run(
            prepared,
            grid=experiment_twotower.DEFAULT_GRID if args.grid is None else args.grid,
            epochs=args.epochs,
            seeds=args.seeds,
            knn_grid=experiment.DEFAULT_GRID if args.knn_grid is None else args.knn_grid,
            progress=progress,
        ),
    }
    out = args.out or Path("results") / args.dataset / "twotower.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(experiment_twotower.table(result))
    versus = result["two-tower minus item-knn, validation ndcg@10 (paired)"]
    low, high = versus["ci95"]
    print(f"validation ndcg@10, two-tower minus item-kNN: {versus['value']:+.4f} [{low:+.4f}, {high:+.4f}]")
    print(f"wrote {out}")
    return 0


def _leakage(args: argparse.Namespace) -> int:
    processed = args.data_dir / "processed" / args.dataset
    result = {
        "command": "signalrec " + shlex.join(args.argv),
        "dataset": args.dataset,
        **leakage.run(prepare.load(processed, args.dataset)),
    }
    out = args.out or Path("results") / args.dataset / "leakage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    later = result["training positives later than the user's test item"]
    print(f"training positives later than the user's test item: mean share {later['mean share']:.3f}")
    for metric, cell in result["popularity, global minus as of test time (paired)"].items():
        low, high = cell["difference ci95"]
        print(
            f"popularity {metric}: global {cell['global (leaky)']:.4f}, as of test time {cell['as of test time']:.4f}, "
            f"difference {cell['difference']:+.4f} [{low:+.4f}, {high:+.4f}]"
        )
    print(f"wrote {out}")
    return 0


def _reported_config(path: Path, dataset: str) -> tt.Config:
    """The configuration a two-tower results file reports on test: the validation choice, the fixed settings, the
    first seed."""
    try:
        result = json.loads(path.read_text())
        if result["dataset"] != dataset:
            raise prepare.PrepareError(f"{path} holds {result['dataset']!r}, not {dataset!r}")
        chosen, fixed = result["chosen"]["two-tower"], result["fixed"]
        config = tt.Config(
            **fixed,
            temperature=chosen["temperature"],
            logq=chosen["logq"],
            epochs=chosen["epochs"],
            seed=result["test"]["two-tower"]["seed"],
        )
        config.check()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, prepare.PrepareError):
            raise
        raise prepare.PrepareError(f"{path}: not a two-tower results file ({exc!r}); run signalrec twotower") from None
    return config


def _leakage_twotower(args: argparse.Namespace) -> int:
    processed = args.data_dir / "processed" / args.dataset
    source = args.source or Path("results") / args.dataset / "twotower.json"
    config = _reported_config(source, args.dataset)
    result = {
        "command": "signalrec " + shlex.join(args.argv),
        "dataset": args.dataset,
        "configuration from": str(source),
        **leakage_twotower.run(prepare.load(processed, args.dataset), config, args.buckets),
    }
    out = args.out or Path("results") / args.dataset / "leakage-twotower.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for metric, cell in result[leakage_twotower.COMPARISON].items():
        low, high = cell["difference ci95"]
        print(
            f"two-tower {metric}: leaky {cell[leakage_twotower.LEAKY]:.4f}, "
            f"as of the cut {cell[leakage_twotower.AS_OF]:.4f}, "
            f"difference {cell['difference']:+.4f} [{low:+.4f}, {high:+.4f}]"
        )
    print(f"wrote {out}")
    return 0


def _recommend(args: argparse.Namespace) -> int:
    prepared = prepare.load(args.data_dir / "processed" / args.dataset, args.dataset)
    model = Popularity() if args.model == "popularity" else ItemKNN(k=args.neighbours, shrink=args.shrink)
    try:
        print(recommend.describe(prepared, model, args.user, args.k))
    except KeyError as exc:
        print(f"signalrec: {exc.args[0]}", file=sys.stderr)
        return 1
    return 0


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r}: expected an integer >= 1")
    return value


def _shrink(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r}: expected a finite number >= 0")
    return value


def _grid(spec: str) -> list[tuple[int, float]]:
    """Parse k:shrink pairs: 10:0,50:10 -> [(10, 0.0), (50, 10.0)]. A bad pair is an argparse usage error."""
    pairs = []
    for part in spec.split(","):
        k_text, sep, shrink_text = part.partition(":")
        try:
            if not sep:
                raise argparse.ArgumentTypeError("no ':'")
            pairs.append((_positive_int(k_text), _shrink(shrink_text)))
        except argparse.ArgumentTypeError:
            raise argparse.ArgumentTypeError(f"{part!r}: expected k:shrink with k >= 1 and shrink >= 0") from None
    return pairs


def _temperature_grid(spec: str) -> list[tuple[float, bool]]:
    """Parse temperature:logq pairs: 0.05:on,0.1:off -> [(0.05, True), (0.1, False)]."""
    pairs = []
    for part in spec.split(","):
        t_text, sep, q_text = part.partition(":")
        try:
            temperature = float(t_text)
        except ValueError:
            temperature = -1.0
        if not sep or q_text not in ("on", "off") or not math.isfinite(temperature) or temperature <= 0:
            raise argparse.ArgumentTypeError(f"{part!r}: expected temperature:on|off with a temperature > 0")
        try:
            tt.Config(temperature=temperature).check()  # e.g. so small that 1/temperature overflows float32
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{part!r}: {exc}") from None
        pairs.append((temperature, q_text == "on"))
    if len(set(pairs)) != len(pairs):  # 0.1:on,0.10:on is the same model twice
        raise argparse.ArgumentTypeError(f"{spec!r}: a configuration appears twice")
    return pairs


def _seeds(spec: str) -> list[int]:
    try:
        seeds = [int(part) for part in spec.split(",")]
    except ValueError:
        seeds = [-1]
    # torch.manual_seed takes 0 <= seed < 2^64.
    if any(not 0 <= seed < 2**64 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"{spec!r}: expected distinct integers in [0, 2^64), e.g. 0,1,2")
    return seeds


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

    p = sub.add_parser("baselines", help="tune item-kNN on validation, report both baselines on test")
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, help="default: results/<dataset>/baselines.json")
    p.add_argument(
        "--grid", type=_grid, help="item-kNN k:shrink pairs, e.g. 10:0,50:10 (default: 10 k values x 3 shrinks)"
    )
    p.set_defaults(run=_baselines)

    p = sub.add_parser("twotower", help="train the two-tower retriever by the protocol, report it beside the baselines")
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, help="default: results/<dataset>/twotower.json")
    p.add_argument(
        "--grid", type=_temperature_grid, help="temperature:logq pairs, e.g. 0.05:on,0.1:off (default: 3 x on/off)"
    )
    p.add_argument("--epochs", type=_positive_int, default=experiment_twotower.DEFAULT_EPOCHS)
    p.add_argument("--seeds", type=_seeds, default=list(experiment_twotower.DEFAULT_SEEDS), help="default: 0,1,2")
    p.add_argument("--knn-grid", type=_grid, help="item-kNN's k:shrink pairs to compare with (default: the full grid)")
    p.add_argument("--quiet", action="store_true", help="no per-epoch progress on stderr")
    p.set_defaults(run=_twotower)

    p = sub.add_parser("leakage", help="measure the split's calendar-time leakage and what it is worth to popularity")
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, help="default: results/<dataset>/leakage.json")
    p.set_defaults(run=_leakage)

    p = sub.add_parser(
        "leakage-twotower", help="retrain the reported two-tower model as of test-time buckets: what the leak is worth"
    )
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--from", dest="source", type=Path, help="default: results/<dataset>/twotower.json")
    p.add_argument("--buckets", type=_positive_int, default=10)
    p.add_argument("--out", type=Path, help="default: results/<dataset>/leakage-twotower.json")
    p.set_defaults(run=_leakage_twotower)

    p = sub.add_parser("recommend", help="one user's recent likes and a model's top k, by title")
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--user", type=int, required=True, help="the MovieLens user id")
    p.add_argument("--model", choices=("item-knn", "popularity"), default="item-knn")
    p.add_argument("--neighbours", type=_positive_int, default=3, help="item-kNN's k (ml-1m's validation choice is 3)")
    p.add_argument("--shrink", type=_shrink, default=10.0)
    p.add_argument("--k", type=_positive_int, default=10)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.set_defaults(run=_recommend)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    try:
        code: int = args.run(args)
    except (
        download.DownloadError,
        movielens.ParseError,
        prepare.PrepareError,
        tt.TrainingError,
        FileNotFoundError,
    ) as exc:
        print(f"signalrec: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
