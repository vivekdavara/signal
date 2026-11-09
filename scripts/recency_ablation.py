"""Where does the two-tower model's lead over item-kNN come from? A validation-only diagnostic, run after the choice.

The two-tower user vector averages the **last 50** positives; item-kNN sums over **all** of them. If recency is the
difference, item-kNN scored from only a user's most recent positives should close some of the gap, and the two-tower
model given longer histories should lose some of its lead. Both are measured on the validation items with the training
positives as history (the same users, so every difference is paired), and nothing here feeds back into a choice: the
configuration comes from the two-tower results file, and the test items are never read.

    python scripts/recency_ablation.py ml-1m --out results/ml-1m/recency-ablation.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from signalrec import evaluate as ev
from signalrec import prepare
from signalrec import twotower as tt
from signalrec.baselines import ItemKNN
from signalrec.cli import _reported_config
from signalrec.prepare import Prepared

SEED = 20261107
KNN_LENGTHS = (1, 5, 10, 20, 50, 100, 200, 0)  # 0: every positive
# No "every positive" for the two-tower model: a batch of 1,024 histories as long as the longest (over 1,000 on ml-1m)
# would gather hundreds of MB of embeddings per step. 200 covers every positive of most users (the JSON says how many).
TT_LENGTHS = (1, 5, 10, 20, 50, 100, 200)


def last_n_matrix(p: Prepared, n: int) -> sp.csr_matrix:
    """Each user's last ``n`` training positives (all of them for n = 0) as a binary user × item matrix."""
    seq = tt.sequences(p, include_val=False)
    users, items = [], []
    for u in range(p.n_users):
        run = seq.items[seq.starts[u] : seq.starts[u + 1]]
        kept = run if n == 0 else run[-n:]
        users.append(np.full(len(kept), u, dtype=np.int64))
        items.append(kept)
    return prepare.interaction_matrix(np.concatenate(users), np.concatenate(items), p.n_users, p.n_items)


class RecentKNN:
    """Item-kNN fit on the whole training history, scoring each user from only their recent positives."""

    name = "item-knn (recent)"

    def __init__(self, fitted: ItemKNN, recent: sp.csr_matrix) -> None:
        if fitted.similarity is None:
            raise ValueError("fit first")
        self.similarity = fitted.similarity
        self.recent = sp.csr_matrix(recent, dtype=np.int64)

    def score(self, users: ev.IntArray) -> ev.FloatArray:
        out: ev.FloatArray = np.asarray((self.recent[users] @ self.similarity).toarray(), dtype=np.float64)
        return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--from", dest="source", type=Path, help="default: results/<dataset>/twotower.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source or Path("results") / args.dataset / "twotower.json"
    reported = json.loads(source.read_text())
    p = prepare.load(args.data_dir / "processed" / args.dataset, args.dataset)
    config = _reported_config(source, args.dataset, p.manifest)
    knn_params = reported["validation"]["item-knn"]
    history = p.history(include_val=False)
    seq = tt.sequences(p, include_val=False)
    genres, _ = tt.genre_matrix(p.genres)

    # No int()/float() coercion: "k": true or 2.7 must be refused, not run as 1 or 2 (review round 3). ItemKNN checks
    # that k is an integer; the shrink must be a number and not a bool.
    k, shrink = knn_params["k"], knn_params["shrink"]
    if isinstance(shrink, bool) or not isinstance(shrink, int | float):
        raise ValueError(f"{source}: item-kNN shrink {shrink!r} is not a number")
    knn = ItemKNN(k=k, shrink=float(shrink))
    knn.fit(history)
    longest = int(np.diff(seq.starts).max())
    per_user: dict[str, ev.FloatArray] = {}
    rows: list[dict[str, Any]] = []
    for n in KNN_LENGTHS:
        scorer = RecentKNN(knn, last_n_matrix(p, n))
        evaluation = ev.evaluate(scorer, history, p.val_items, "val")
        label = f"item-knn, last {n if n else 'all'}"
        per_user[label] = ev.ndcg_at(evaluation.ranks, 10)
        rows.append({"model": label, **{k: v for k, v in evaluation.summary().items() if "@10" in k or "@50" in k}})
        print(f"{label}: ndcg@10 {rows[-1]['ndcg@10']:.4f}", file=sys.stderr, flush=True)
    # The reported history length is always one of the rows: it is the reference (review round 2: the reference was
    # hard-coded as 50, wrong for a results file with another max_history).
    tt_lengths = sorted({*TT_LENGTHS, config.max_history})
    for n in tt_lengths:
        net, _ = tt.train(seq, genres, replace(config, max_history=n))
        evaluation = ev.evaluate(tt.embed(net, seq, n), history, p.val_items, "val")
        label = f"two-tower, last {n}"
        per_user[label] = ev.ndcg_at(evaluation.ranks, 10)
        rows.append({"model": label, **{k: v for k, v in evaluation.summary().items() if "@10" in k or "@50" in k}})
        print(f"{label}: ndcg@10 {rows[-1]['ndcg@10']:.4f}", file=sys.stderr, flush=True)

    reference = {
        "item-knn": per_user["item-knn, last all"],
        "two-tower": per_user[f"two-tower, last {config.max_history}"],
    }
    for row in rows:
        family = "item-knn" if row["model"].startswith("item-knn") else "two-tower"
        low, high = ev.paired_bootstrap_ci(per_user[row["model"]], reference[family], seed=SEED)
        row["minus the reported history length, ndcg@10, paired ci95"] = [low, high]
    result = {
        "command": "python scripts/recency_ablation.py " + shlex.join(argv),
        "dataset": args.dataset,
        "split": "validation (training positives as history); post hoc, no choice depends on it",
        "two-tower configuration": config.as_dict(),
        "item-knn": {"k": knn.k, "shrink": knn.shrink},
        "longest training history": longest,
        "users with more than 200 training positives": int((np.diff(seq.starts) > 200).sum()),
        "rows": rows,
        "bootstrap": {"resamples": 1000, "seed": SEED},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        low, high = row["minus the reported history length, ndcg@10, paired ci95"]
        print(f"| {row['model']} | {row['ndcg@10']:.4f} | {row['recall@50']:.4f} | [{low:+.4f}, {high:+.4f}] |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
