"""A simulated A/B test of the ranker (DESIGN.md, "Day 4: serving and experiments"). **The click model is SYNTHETIC.**

Control serves the retrieved order (the ranker skipped), treatment the re-ranked one, both through the request path
with the serving filter, top 10. For every real ml-1m user, the rank of the movie they watched next (their held-out
test movie) in each arm's list goes through ``abtest.ClickModel`` (synthetic: a position-based model, see abtest.py)
to give that user's probability of converting in each arm. Then:

1. **The truth under the model**: each arm's conversion rate (the mean over users) and the difference.
2. **One experiment** over the dataset's users, split by ``abtest.assign`` (experiment ``ranker-v1``, 50/50): the
   measured difference, its 95% interval and p-value.
3. **Power analysis**: the users per arm a test needs to detect the true difference (α = 0.05, power 0.8).
4. **Checking the statistics** (Monte Carlo, ``--experiments`` each): A/A tests at the dataset's size must be
   significant about 5% of the time; tests of the computed size must detect the lift about 80% of the time; the
   intervals must cover the true difference about 95% of the time. Each share has its own Monte Carlo interval.
5. The same for two other click models, to show how much the answer rests on the synthetic assumptions.

Needs the model and the store, like serving_check.py.

    python scripts/ab_simulate.py ml-1m
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from signalrec import abtest, prepare
from signalrec import tower as tw
from signalrec.featurestore import FeatureStore
from signalrec.serving import Recommender

SEED = 0
MODELS = (abtest.ClickModel(0.9, 0.02), abtest.ClickModel(0.5, 0.02), abtest.ClickModel(0.9, 0.1))


def target_ranks(rec: Recommender, test_items: np.ndarray, k: int, rank: bool) -> np.ndarray:
    out = np.zeros(len(test_items), dtype=np.int64)  # 0: not in the list shown
    for u, target in enumerate(test_items.tolist()):
        hit = np.flatnonzero(rec.recommend(u, k=k, rank=rank).movies == target)
        if len(hit):
            out[u] = int(hit[0]) + 1
    return out


def share_interval(share: float, n: int) -> list[float]:
    """A normal-approximation 95% Monte Carlo interval for a share measured over n experiments."""
    half = 1.96 * math.sqrt(share * (1 - share) / n)
    return [share - half, share + half]


def analyse(
    model: abtest.ClickModel,
    ranks_control: np.ndarray,
    ranks_treatment: np.ndarray,
    raw_users: np.ndarray,
    experiments: int,
) -> dict[str, Any]:
    p_c, p_t = model.conversion(ranks_control), model.conversion(ranks_treatment)
    true_c, true_t = float(p_c.mean()), float(p_t.mean())
    rng = np.random.default_rng(SEED)
    one = abtest.run(abtest.Experiment("ranker-v1", 0.5), raw_users, p_c, p_t, rng)
    # Arms that put every next movie in the same place have no difference to detect: no test size is enough.
    needed = abtest.users_per_arm(true_c, true_t) if true_t != true_c else None
    half = len(raw_users) // 2
    checks: dict[str, dict[str, Any]] = {
        "A/A at the dataset's size": abtest.simulate(p_c, p_c, half, experiments, rng),
        "A/B at the dataset's size": abtest.simulate(p_c, p_t, half, experiments, rng),
    }
    if needed is not None:
        checks["A/B at the computed size"] = abtest.simulate(p_c, p_t, needed, experiments, rng)
    for row in checks.values():
        for key in ("share significant", "share of intervals covering the true difference"):
            row[f"{key}, Monte Carlo ci95"] = share_interval(row[key], experiments)
    return {
        "click model (SYNTHETIC)": {"p_target": model.p_target, "p_other": model.p_other, "k": model.k},
        "true conversion": {"control": true_c, "treatment": true_t},
        "true difference": true_t - true_c,
        "true relative lift": (true_t - true_c) / true_c if true_c else None,
        "one experiment (ranker-v1, 50/50, the dataset's users)": {
            "control": one.control,
            "treatment": one.treatment,
            "users": {"control": one.n_control, "treatment": one.n_treatment},
            "difference": one.difference,
            "ci95": list(one.ci95),
            "p_value": one.p_value,
        },
        "users per arm for 80% power at alpha 0.05": needed,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, help="default: models/<dataset>")
    parser.add_argument("--redis", default="redis://127.0.0.1:6383/0")
    parser.add_argument("--namespace", default="signal")
    parser.add_argument("--experiments", type=int, default=1000)
    parser.add_argument("--out", type=Path, help="default: results/<dataset>/abtest.json")
    args = parser.parse_args(argv)

    model = tw.load(args.model or Path("models") / args.dataset)
    prepared = prepare.load(args.data_dir / "processed" / args.dataset, args.dataset)
    rec = Recommender(model, FeatureStore.connect(model.catalogue, args.redis, args.namespace))
    k = MODELS[0].k
    t0 = time.perf_counter()
    control = target_ranks(rec, prepared.test_items, k, rank=False)
    treatment = target_ranks(rec, prepared.test_items, k, rank=True)
    serve_seconds = time.perf_counter() - t0
    raw_users = model.user_raw_ids.astype(np.int64)
    rows = [analyse(m, control, treatment, raw_users, args.experiments) for m in MODELS]
    result = {
        "command": "python " + " ".join(["scripts/ab_simulate.py", *(sys.argv[1:] if argv is None else argv)]),
        "dataset": args.dataset,
        "SYNTHETIC": "the click model is synthetic; the users, their lists and the movie each watched next are real",
        "arms": {"control": "retrieved order (ranker skipped)", "treatment": "re-ranked"},
        "users": len(raw_users),
        "next movie in the top 10 (share of users)": {
            "control": float((control > 0).mean()),
            "treatment": float((treatment > 0).mean()),
        },
        "serving seconds (both arms, every user)": serve_seconds,
        "click models": rows,
        "seed": SEED,
    }
    out = args.out or Path("results") / args.dataset / "abtest.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        cm = row["click model (SYNTHETIC)"]
        one = row["one experiment (ranker-v1, 50/50, the dataset's users)"]
        aa = row["checks"]["A/A at the dataset's size"]
        truth = row["true conversion"]
        needed = row["users per arm for 80% power at alpha 0.05"]
        if needed is None:
            print(f"click model p_target={cm['p_target']} p_other={cm['p_other']}: the arms don't differ")
            continue
        sized = row["checks"]["A/B at the computed size"]
        print(
            f"click model p_target={cm['p_target']} p_other={cm['p_other']}: true {truth['control']:.4f} -> "
            f"{truth['treatment']:.4f} ({row['true difference']:+.4f}, {row['true relative lift'] or 0:+.1%}); "
            f"one experiment {one['difference']:+.4f} [{one['ci95'][0]:+.4f}, {one['ci95'][1]:+.4f}] "
            f"p={one['p_value']:.3f}; needs {needed:,} per arm; "
            f"A/A significant {aa['share significant']:.3f}, power at that size {sized['share significant']:.3f}, "
            f"coverage {sized['share of intervals covering the true difference']:.3f}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
