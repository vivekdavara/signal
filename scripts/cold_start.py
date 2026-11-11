"""New users: how good are the lists for users the model never trained on, after 0, 1, 2, 3 positives?

    .venv/bin/python scripts/cold_start.py ml-1m --out results/ml-1m/cold-start.json

Serving (day 5) answers a user with no positives with the most popular movies, and switches to the two-tower model at
their first positive, because the model trained only on positives that had at least one earlier positive: its vector
for an empty history was never trained. This measures both choices on users the model has never seen:

1. **Hold users out.** A fixed share of users (SHA-256 of "cold-start:<MovieLens id>", 20% by default) is removed
   from training entirely: the reported two-tower configuration (``twotower.json``) is retrained on the *other*
   users' training + validation positives, and popularity counts only theirs. The held-out users' positives train no
   weight and add to no count. They did take part in two earlier choices: the configuration in ``twotower.json``
   (history length, temperature, epochs) was chosen on validation over all users, and the 5-core filter that fixed
   the catalogue counted their positives. Both are small, and both favour the held-out users (review of day 5).
2. **Replay a new user.** Each held-out user arrives with only their last ``h`` positives before the test movie
   (h = 0, 1, 2, 3, then all of them), as events would bring them. The two-tower list (the NumPy user tower serving
   uses, exact retrieval) and the popularity list each exclude those ``h`` movies, the only ones such a user's store
   would hold, and the test movie's exact rank gives NDCG@10 and Recall@10 (``evaluate.py``).
3. **Compare.** Per ``h``: two-tower minus popularity on the held-out users (paired), and the held-out users against
   the trained-on users given the same ``h`` (unpaired: did training on a user help them?).

Retrieval order only: the ranker was trained on lists of every user, held-out ones included, so it is left out. Test
movies, post hoc: nothing is chosen from these numbers. The model reads the last ``max_history`` positives (3 for
ml-1m's reported configuration), so h = 3 and "all" give the same vector and differ only in what is excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from signalrec import evaluate as ev
from signalrec import features as fe
from signalrec import prepare
from signalrec import twotower as tt
from signalrec.cli import _reported_config
from signalrec.tower import UserTower, export_user_tower

SEED = 20261107  # experiment.SEED
SALT = "cold-start"
SHARE = 0.2
HISTORIES: tuple[int | None, ...] = (0, 1, 2, 3, None)  # None: every positive before the test movie
BoolArray = npt.NDArray[np.bool_]
Float32Array = npt.NDArray[np.float32]


def held_out(user_raw_ids: ev.IntArray, share: float, salt: str = SALT) -> BoolArray:
    """True for the users whose SHA-256("<salt>:<id>") falls in the first ``share`` of the hash space."""
    if not 0 < share < 1:
        raise ValueError(f"share must be in (0, 1), got {share!r}")
    cut = int(share * 2**64)
    return np.array(
        [int.from_bytes(hashlib.sha256(f"{salt}:{int(u)}".encode()).digest()[:8], "big") < cut for u in user_raw_ids],
        dtype=bool,
    )


def without_users(p: prepare.Prepared, drop: BoolArray) -> prepare.Prepared:
    """``p`` with the ``drop`` users removed entirely and the rest renumbered in order; the catalogue is unchanged."""
    if drop.shape != (p.n_users,):
        raise ValueError(f"a mask of {drop.shape} for {p.n_users} users")
    keep = ~drop
    new_index = np.cumsum(keep) - 1
    rows = keep[p.train_users]
    return replace(
        p,
        train_users=new_index[p.train_users[rows]].astype(p.train_users.dtype),
        train_items=p.train_items[rows],
        train_times=p.train_times[rows],
        val_items=p.val_items[keep],
        val_times=p.val_times[keep],
        test_items=p.test_items[keep],
        test_times=p.test_times[keep],
        user_raw_ids=p.user_raw_ids[keep],
        manifest=None,
    )


def known(sequences: list[ev.IntArray], h: int | None) -> list[ev.IntArray]:
    """Each user's last ``h`` positives (all of them for None): what a new user's store would hold."""
    if h is not None and h < 0:
        raise ValueError(f"h must be >= 0, got {h}")
    return [s if h is None else s[len(s) - min(h, len(s)) :] for s in sequences]


def history_matrix(histories: list[ev.IntArray], n_items: int) -> sp.csr_matrix:
    users = np.repeat(np.arange(len(histories), dtype=np.int64), [len(h) for h in histories])
    items = np.concatenate(histories) if histories else np.empty(0, dtype=np.int64)
    return prepare.interaction_matrix(users, items.astype(np.int64), len(histories), n_items)


def two_tower_ranks(
    tower: UserTower, items: Float32Array, histories: list[ev.IntArray], targets: ev.IntArray
) -> ev.IntArray:
    """The test movie's rank by the served user tower's inner products, the ``histories`` excluded."""
    vectors = np.stack([tower.vector(h.astype(np.int64)) for h in histories]) if histories else np.empty((0, tower.dim))
    scores = (vectors.astype(np.float32) @ items.T).astype(np.float32)
    return ev.ranks(scores, history_matrix(histories, len(items)), targets)


def popularity_ranks(popularity: ev.IntArray, histories: list[ev.IntArray], targets: ev.IntArray) -> ev.IntArray:
    """The test movie's rank by popularity (ties to the smaller index, as serving), the ``histories`` excluded."""
    scores = np.broadcast_to(popularity.astype(np.float64), (len(histories), len(popularity)))
    return ev.ranks(scores, history_matrix(histories, len(popularity)), targets)


def unpaired_ci(a: ev.FloatArray, b: ev.FloatArray, n_resamples: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """A percentile bootstrap interval for mean(a) - mean(b), a and b different users resampled independently."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_resamples, len(a)))
    ib = rng.integers(0, len(b), size=(n_resamples, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return float(low), float(high)


def _cell(values: ev.FloatArray) -> dict[str, Any]:
    return {"value": float(values.mean()), "ci95": list(ev.bootstrap_ci(values, seed=SEED))}


def _label(h: int | None) -> str:
    return "all" if h is None else str(h)


def measure(p: prepare.Prepared, config: tt.Config, share: float, salt: str = SALT) -> dict[str, Any]:
    drop = held_out(p.user_raw_ids, share, salt)
    if drop.all() or not drop.any():
        raise ValueError(f"holding out {int(drop.sum())} of {p.n_users} users leaves nothing to compare")
    trained = without_users(p, drop)
    seq = tt.sequences(trained, include_val=True)
    net, _ = tt.train(seq, tt.genre_matrix(trained.genres)[0], config)
    emb = tt.embed(net, seq, config.max_history)
    tower = export_user_tower(net, config.max_history)
    popularity = np.asarray(trained.history(include_val=True).sum(axis=0), dtype=np.int64).ravel()

    everyone = fe.user_sequences(p, include_val=True)  # train + validation positives, in the split's order
    groups = {"held out": np.flatnonzero(drop), "trained on": np.flatnonzero(~drop)}
    rows: dict[str, Any] = {}
    for h in HISTORIES:
        row: dict[str, Any] = {}
        metrics: dict[tuple[str, str], dict[str, ev.FloatArray]] = {}
        for group, users in groups.items():
            histories = known([everyone[u] for u in users], h)
            targets = p.test_items[users].astype(np.int64)
            for model, ranks in (
                ("two-tower", two_tower_ranks(tower, emb.items, histories, targets)),
                ("popularity", popularity_ranks(popularity, histories, targets)),
            ):
                metrics[(group, model)] = {"ndcg@10": ev.ndcg_at(ranks, 10), "recall@10": ev.recall_at(ranks, 10)}
            row[group] = {
                f"{model} {metric}": _cell(metrics[(group, model)][metric])
                for model in ("two-tower", "popularity")
                for metric in ("ndcg@10", "recall@10")
            }
        tower_h, pop_h = metrics[("held out", "two-tower")], metrics[("held out", "popularity")]
        tower_t = metrics[("trained on", "two-tower")]
        row["held out: two-tower minus popularity, ndcg@10 (paired)"] = {
            "value": float(tower_h["ndcg@10"].mean() - pop_h["ndcg@10"].mean()),
            "ci95": list(ev.paired_bootstrap_ci(tower_h["ndcg@10"], pop_h["ndcg@10"], seed=SEED)),
        }
        row["two-tower ndcg@10: held out minus trained on (unpaired)"] = {
            "value": float(tower_h["ndcg@10"].mean() - tower_t["ndcg@10"].mean()),
            "ci95": list(unpaired_ci(tower_h["ndcg@10"], tower_t["ndcg@10"])),
        }
        rows[_label(h)] = row
    return {
        "users": {"held out": int(drop.sum()), "trained on": int((~drop).sum())},
        "hold-out": {"share": share, "salt": salt, "rule": "SHA-256('<salt>:<MovieLens id>')[:8] < share * 2^64"},
        "configuration": config.as_dict(),
        "by known positives": rows,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source", type=Path, help="default: results/<dataset>/twotower.json")
    parser.add_argument("--share", type=float, default=SHARE, help="the share of users held out (default 0.2)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source or Path("results") / args.dataset / "twotower.json"
    p = prepare.load(args.data_dir / "processed" / args.dataset, args.dataset)
    config = _reported_config(source, args.dataset, p.manifest)
    try:
        result = measure(p, config, args.share)
    except ValueError as exc:
        print(f"cold_start.py: {exc}", file=sys.stderr)
        return 1
    result = {
        "command": "python scripts/cold_start.py " + shlex.join(argv),
        "dataset": args.dataset,
        "split": "test, post hoc; retrieval order only (the ranker saw every user's lists)",
        "configuration from": str(source),
        **result,
        "bootstrap": {"resamples": 1000, "seed": SEED},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    users = result["users"]
    print(f"held out {users['held out']} users, trained on {users['trained on']}; NDCG@10 by known positives:")
    for h, row in result["by known positives"].items():
        cells = []
        for group in ("held out", "trained on"):
            t, q = row[group]["two-tower ndcg@10"]["value"], row[group]["popularity ndcg@10"]["value"]
            cells.append(f"{group}: two-tower {t:.4f}, popularity {q:.4f}")
        d = row["held out: two-tower minus popularity, ndcg@10 (paired)"]
        g = row["two-tower ndcg@10: held out minus trained on (unpaired)"]
        print(f"  h={h:>3}  " + "; ".join(cells))
        print(
            f"         held out two-tower - popularity {d['value']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]; "
            f"two-tower held out - trained on {g['value']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
