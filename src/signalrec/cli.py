"""The ``signalrec`` command line."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from signalrec import (
    download,
    experiment,
    experiment_ranking,
    experiment_twotower,
    leakage,
    leakage_twotower,
    movielens,
    prepare,
    recommend,
    synthetic,
)
from signalrec import tower as tw
from signalrec import twotower as tt
from signalrec.baselines import ItemKNN, Popularity

DEFAULT_DATA = Path("data")
# Built from the data, so never committed (ml-1m's licence forbids redistribution); .gitignore has models/.
DEFAULT_MODELS = Path("models")
DEFAULT_REDIS = "redis://127.0.0.1:6383/0"


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
            histories=args.histories,
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


FIXED_SETTINGS = frozenset({"dim", "hidden", "batch_size", "lr"})


def _reported_config(path: Path, dataset: str, manifest: dict[str, Any] | None = None) -> tt.Config:
    """The configuration a two-tower results file reports on test: the validation choice, the fixed settings, the
    first seed.

    With ``manifest`` (the processed data about to be used), the file must also have been produced from exactly that
    data: the same parameters, counts and sources. Every fixed setting must be present (review round 2: missing ones
    silently took today's defaults). The history length is chosen on validation from day 3 and fixed in a day-2 file,
    so it must be in exactly one of the two.
    """
    try:
        result = json.loads(path.read_text())
        if result["dataset"] != dataset:
            raise prepare.PrepareError(f"{path} holds {result['dataset']!r}, not {dataset!r}")
        if manifest is not None:
            on_disk = {key: manifest[key] for key in ("params", "counts", "sources")}
            if result["prepared"] != on_disk:
                raise prepare.PrepareError(f"{path} was produced from other prepared data than {dataset}'s on disk")
        chosen, fixed = result["chosen"]["two-tower"], result["fixed"]
        if not isinstance(fixed, dict) or not isinstance(chosen, dict):
            raise prepare.PrepareError(f"{path}: fixed settings {fixed!r}, chosen {chosen!r}: expected objects")
        fixed = dict(fixed)
        if ("max_history" in fixed) == ("max_history" in chosen):
            raise prepare.PrepareError(f"{path}: max_history must be either fixed (day 2) or chosen (day 3), not both")
        max_history = fixed.pop("max_history") if "max_history" in fixed else chosen["max_history"]
        if set(fixed) != FIXED_SETTINGS:
            raise prepare.PrepareError(
                f"{path}: fixed settings {sorted(fixed)}, expected exactly {sorted(FIXED_SETTINGS)} (and max_history)"
            )
        config = tt.Config(
            **fixed,
            max_history=max_history,
            temperature=chosen["temperature"],
            logq=chosen["logq"],
            epochs=chosen["epochs"],
            seed=result["test"]["two-tower"]["seed"],
        )
        config.check()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, prepare.PrepareError):
            raise
        raise prepare.PrepareError(f"{path}: not a two-tower results file ({exc!r}); run signalrec twotower") from None
    return config


def _leakage_twotower(args: argparse.Namespace) -> int:
    processed = args.data_dir / "processed" / args.dataset
    source = args.source or Path("results") / args.dataset / "twotower.json"
    prepared = prepare.load(processed, args.dataset)
    if prepared.manifest is None:  # as in _baselines
        raise prepare.PrepareError(f"{processed}: loaded without its manifest")
    config = _reported_config(source, args.dataset, prepared.manifest)
    if args.buckets > prepared.n_users:
        raise prepare.PrepareError(f"--buckets {args.buckets}: more buckets than the {prepared.n_users} users")
    result = {
        "command": "signalrec " + shlex.join(args.argv),
        "dataset": args.dataset,
        "configuration from": str(source),
        **leakage_twotower.run(prepared, config, args.buckets),
    }
    out = args.out or Path("results") / args.dataset / "leakage-twotower.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for block, label, name in (
        (leakage_twotower.COMPARISON, leakage_twotower.AS_OF, "as of the cut"),
        (leakage_twotower.COMPARISON_OWN, leakage_twotower.AS_OF_OWN, "as of the cut, own histories kept"),
    ):
        for metric, cell in result[block].items():
            low, high = cell["difference ci95"]
            print(
                f"two-tower {metric}: leaky {cell[leakage_twotower.LEAKY]:.4f}, {name} {cell[label]:.4f}, "
                f"difference {cell['difference']:+.4f} [{low:+.4f}, {high:+.4f}]"
            )
    print(f"wrote {out}")
    return 0


def _ranking(args: argparse.Namespace) -> int:
    processed = args.data_dir / "processed" / args.dataset
    source = args.source or Path("results") / args.dataset / "twotower.json"
    prepared = prepare.load(processed, args.dataset)
    if prepared.manifest is None:  # as in _baselines
        raise prepare.PrepareError(f"{processed}: loaded without its manifest")
    config = _reported_config(source, args.dataset, prepared.manifest)

    def progress(line: str) -> None:
        if not args.quiet:
            print(line, file=sys.stderr, flush=True)

    result = {
        "command": "signalrec " + shlex.join(args.argv),
        "dataset": args.dataset,
        "configuration from": str(source),
        **experiment_ranking.run(
            prepared, config, hidden=args.hidden, epochs=args.epochs, n=args.candidates, progress=progress
        ),
    }
    out = args.out or Path("results") / args.dataset / "ranking.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    n = result["candidates"]
    test = result["test"]
    print(f"held-out movie in the top {n}: {test['held-out movie retrieved (share of users)']:.4f} of test users")
    for name in [m for m in ("ndcg@10", "recall@10", "recall@50") if m in test["re-ranked"]]:
        diff = result["re-ranked minus retrieved order (paired, test)"][name]
        low, high = diff["ci95"]
        print(
            f"test {name}: retrieved order {test['retrieved order'][name]['value']:.4f}, re-ranked "
            f"{test['re-ranked'][name]['value']:.4f}, difference {diff['value']:+.4f} [{low:+.4f}, {high:+.4f}]"
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


def _export(args: argparse.Namespace) -> int:
    from signalrec import export  # torch is imported here only

    processed = args.data_dir / "processed" / args.dataset
    source = args.source or Path("results") / args.dataset / "twotower.json"
    ranking = args.ranking or Path("results") / args.dataset / "ranking.json"
    prepared = prepare.load(processed, args.dataset)
    if prepared.manifest is None:  # as in _baselines
        raise prepare.PrepareError(f"{processed}: loaded without its manifest")
    config = _reported_config(source, args.dataset, prepared.manifest)
    ranker = export.ranker_from_results(ranking, args.dataset, config)
    ranked = json.loads(ranking.read_text())
    if (ranked.get("users"), ranked.get("items")) != (prepared.n_users, prepared.n_items):
        raise prepare.PrepareError(f"{ranking} ranked another dataset's users or movies than {processed}'s")
    info = {
        "dataset": args.dataset,
        "command": "signalrec " + shlex.join(args.argv),
        "prepared": {key: prepared.manifest[key] for key in ("params", "counts", "sources")},
        "min_rating": prepared.manifest["params"]["min_rating"],
        "two-tower configuration from": str(source),
        "ranker from": str(ranking),
        "ranker chosen": ranked.get("chosen"),
        "trained on": "training + validation positives (as the test runs)",
    }
    model = export.build_model(prepared, config, ranker, info)
    out = args.out or DEFAULT_MODELS / args.dataset
    meta = tw.save(model, out)
    print(f"wrote {out} (model.npz SHA-256 {meta['npz_sha256'][:12]}...)")
    return 0


def _load_store(args: argparse.Namespace) -> int:
    from signalrec import export
    from signalrec.featurestore import FeatureStore

    model = tw.load(args.model or DEFAULT_MODELS / args.dataset)
    prepared = prepare.load(args.data_dir / "processed" / args.dataset, args.dataset)
    if not (prepared.item_raw_ids.tolist() == model.item_raw_ids.tolist()) or not (
        prepared.user_raw_ids.tolist() == model.user_raw_ids.tolist()
    ):
        raise prepare.PrepareError("the model was exported from other prepared data")
    ratings = movielens.load(args.dataset, args.data_dir / "raw" / args.dataset).ratings
    store = FeatureStore.connect(model.catalogue, args.redis, args.namespace)
    if args.fresh:
        print(f"deleted {store.clear()} keys under {args.namespace}:")
    positives, rated = export.load_store(store, prepared, ratings)
    print(f"positives: {positives.applied} of {positives.events} applied in {positives.seconds:.1f} s")
    print(f"ratings into the rated sets: {rated.applied} of {rated.events} new in {rated.seconds:.1f} s")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from signalrec.api import create_app
    from signalrec.featurestore import FeatureStore
    from signalrec.serving import Recommender

    model = tw.load(args.model or DEFAULT_MODELS / args.dataset)
    store = FeatureStore.connect(model.catalogue, args.redis, args.namespace)
    threads = None if args.search_threads == 0 else args.search_threads
    app = create_app(Recommender(model, store, search_threads=threads), args.experiment)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    return 0


def _experiment(text: str) -> Any:
    from signalrec import abtest

    name, colon, share = text.partition(":")
    try:
        return abtest.Experiment(name, float(share) if colon else 0.5)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r}: {exc}") from None


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} is negative")
    return value


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


def _histories(spec: str) -> list[int]:
    """Parse distinct history lengths: 1,5,50 -> [1, 5, 50]."""
    try:
        lengths = [_positive_int(part) for part in spec.split(",")]
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(f"{spec!r}: expected integers >= 1, e.g. 1,5,50") from None
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError(f"{spec!r}: a history length appears twice")
    return lengths


def _candidates(text: str) -> int:
    """At least the cut-off the ranker is chosen by: run refuses fewer, after loading everything (review round 2)."""
    value = _positive_int(text)
    if value < experiment_ranking.PRIMARY_K:
        raise argparse.ArgumentTypeError(f"{text!r}: expected at least {experiment_ranking.PRIMARY_K} candidates")
    return value


def _hidden_sizes(spec: str) -> list[int]:
    """Parse distinct ranker hidden sizes, 0 for a linear ranker: 0,16 -> [0, 16]."""
    try:
        sizes = [int(part) for part in spec.split(",")]
    except ValueError:
        sizes = [-1]
    if any(size < 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError(f"{spec!r}: expected distinct integers >= 0, e.g. 0,16,64")
    return sizes


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
        "--grid",
        type=_temperature_grid,
        help="temperature:logq pairs, e.g. 0.05:on,0.1:off (default: 0.05, 0.1 and 0.2 with logQ on)",
    )
    p.add_argument(
        "--histories",
        type=_histories,
        default=list(experiment_twotower.DEFAULT_HISTORIES),
        help="history lengths crossed with the grid (default: 1,3,5,10,20,50)",
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

    p = sub.add_parser(
        "ranking", help="retrieve the reported two-tower model's top candidates, train a ranker on validation, test it"
    )
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--from", dest="source", type=Path, help="default: results/<dataset>/twotower.json")
    p.add_argument(
        "--hidden", type=_hidden_sizes, default=list(experiment_ranking.DEFAULT_HIDDEN), help="default: 0,16,64"
    )
    p.add_argument("--epochs", type=_positive_int, default=experiment_ranking.DEFAULT_EPOCHS)
    p.add_argument(
        "--candidates",
        type=_candidates,
        default=experiment_ranking.N_CANDIDATES,
        help=f"at least {experiment_ranking.PRIMARY_K}, the cut-off the ranker is chosen by (default 200)",
    )
    p.add_argument("--out", type=Path, help="default: results/<dataset>/ranking.json")
    p.add_argument("--quiet", action="store_true", help="no progress on stderr")
    p.set_defaults(run=_ranking)

    p = sub.add_parser("recommend", help="one user's recent likes and a model's top k, by title")
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--user", type=int, required=True, help="the MovieLens user id")
    p.add_argument("--model", choices=("item-knn", "popularity"), default="item-knn")
    p.add_argument("--neighbours", type=_positive_int, default=3, help="item-kNN's k (ml-1m's validation choice is 3)")
    p.add_argument("--shrink", type=_shrink, default=10.0)
    p.add_argument("--k", type=_positive_int, default=10)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.set_defaults(run=_recommend)

    p = sub.add_parser(
        "export", help="the served model: the reported two-tower retrained on training + validation, and the ranker"
    )
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--from", dest="source", type=Path, help="default: results/<dataset>/twotower.json")
    p.add_argument("--ranking", type=Path, help="default: results/<dataset>/ranking.json")
    p.add_argument("--out", type=Path, help="default: models/<dataset>")
    p.set_defaults(run=_export)

    p = sub.add_parser("load-store", help="replay the positives and earlier ratings into the Redis feature store")
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--model", type=Path, help="default: models/<dataset>")
    p.add_argument("--redis", default=DEFAULT_REDIS, help=f"default: {DEFAULT_REDIS}")
    p.add_argument("--namespace", default="signal")
    p.add_argument("--fresh", action="store_true", help="delete the namespace's keys first")
    p.set_defaults(run=_load_store)

    p = sub.add_parser("serve", help="the HTTP API on 127.0.0.1:8116")
    p.add_argument("dataset", choices=sorted(movielens.FORMATS))
    p.add_argument("--model", type=Path, help="default: models/<dataset>")
    p.add_argument("--redis", default=DEFAULT_REDIS, help=f"default: {DEFAULT_REDIS}")
    p.add_argument("--namespace", default="signal")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=_positive_int, default=8116)
    p.add_argument(
        "--search-threads",
        type=_non_negative_int,
        default=1,
        help="FAISS's OpenMP threads (default 1, see serving.SEARCH_THREADS; 0 leaves FAISS's own default)",
    )
    p.add_argument(
        "--experiment",
        type=_experiment,
        help="name[:treatment share], e.g. ranker-v1:0.5: the control arm skips the ranker",
    )
    p.set_defaults(run=_serve)
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
        tw.ModelError,
        FileNotFoundError,
    ) as exc:
        print(f"signalrec: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
