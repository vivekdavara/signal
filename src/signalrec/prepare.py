"""From parsed ratings to the train/validation/test arrays every model reads (DESIGN.md, "From ratings to implicit
feedback").

positives (rating >= min_rating) → iterated k-core → per-user leave-last-out split by (time, movie id) → dense indices.
The result is saved as one ``.npy`` per array plus ``items.tsv`` and ``manifest.json`` (with every file's SHA-256), and
loading checks those hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from signalrec.movielens import Dataset, Movie

# Tabs and other control characters can't be stored in items.tsv (the parsers refuse them too).
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

FORMAT_VERSION = 1
ARRAYS = (
    "train_users",
    "train_items",
    "train_times",
    "val_items",
    "val_times",
    "test_items",
    "test_times",
    "user_raw_ids",
    "item_raw_ids",
)


class PrepareError(ValueError):
    """The parameters leave nothing to evaluate, or a processed directory doesn't match its manifest."""


@dataclass(frozen=True)
class Params:
    min_rating: float = 4.0
    min_user_positives: int = 5
    min_item_positives: int = 5

    def check(self) -> None:
        # Leave-last-out needs a validation item, a test item and at least one training item per user.
        if self.min_user_positives < 3:
            raise PrepareError("min_user_positives must be at least 3 (train, validation and test)")
        if self.min_item_positives < 1:
            raise PrepareError("min_item_positives must be at least 1")


@dataclass(frozen=True)
class Prepared:
    """Dense indices: users 0..n_users-1 and items 0..n_items-1, each in raw-id order.

    Training rows are sorted by (user, time, item). ``val_items[u]`` and ``test_items[u]`` are user u's held-out items.
    """

    train_users: IntArray
    train_items: IntArray
    train_times: IntArray
    val_items: IntArray
    val_times: IntArray
    test_items: IntArray
    test_times: IntArray
    user_raw_ids: IntArray
    item_raw_ids: IntArray
    titles: tuple[str, ...]
    genres: tuple[tuple[str, ...], ...]

    @property
    def n_users(self) -> int:
        return len(self.user_raw_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_raw_ids)

    def history(self, include_val: bool) -> sp.csr_matrix:
        """The binary user × item matrix of training positives, plus each validation item if ``include_val``."""
        users, items = self.train_users, self.train_items
        if include_val:
            users = np.concatenate([users, np.arange(self.n_users, dtype=np.int64)])
            items = np.concatenate([items, self.val_items])
        return interaction_matrix(users, items, self.n_users, self.n_items)


def interaction_matrix(users: IntArray, items: IntArray, n_users: int, n_items: int) -> sp.csr_matrix:
    data = np.ones(len(users), dtype=np.float32)
    matrix = sp.csr_matrix((data, (users, items)), shape=(n_users, n_items), dtype=np.float32)
    matrix.sum_duplicates()
    if matrix.nnz != len(users):
        raise PrepareError("a (user, item) pair appears twice in the history")
    return matrix


def k_core(users: IntArray, items: IntArray, min_user: int, min_item: int) -> BoolArray:
    """Rows to keep so that every kept user has >= min_user rows and every kept item >= min_item, at once.

    Repeats until nothing changes: dropping an item can push a user under the bound and the other way round. Each
    pass either removes a row or stops, so it ends after at most len(users) passes.
    """
    keep = np.ones(len(users), dtype=bool)
    while True:
        _, user_inverse, user_counts = np.unique(users[keep], return_inverse=True, return_counts=True)
        _, item_inverse, item_counts = np.unique(items[keep], return_inverse=True, return_counts=True)
        ok = (user_counts[user_inverse] >= min_user) & (item_counts[item_inverse] >= min_item)
        if ok.all():
            return keep
        kept = np.flatnonzero(keep)
        keep[kept[~ok]] = False


def leave_last_out(users: IntArray, items: IntArray, times: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """Split each user's rows by (time, item): returns the row indices of training, validation and test.

    Validation and test hold one row per user, in ascending user order; training rows are in (user, time, item) order.
    Every user needs at least three rows.
    """
    order = np.lexsort((items, times, users))
    sorted_users = users[order]
    last = np.flatnonzero(np.append(sorted_users[1:] != sorted_users[:-1], True)) if len(order) else order
    counts = np.diff(np.concatenate([[-1], last]))
    if (counts < 3).any():
        raise PrepareError("every user needs at least 3 positives for leave-last-out")
    held = np.zeros(len(order), dtype=bool)
    held[last] = True
    held[last - 1] = True
    return order[~held], order[last - 1], order[last]


def prepare(dataset: Dataset, params: Params) -> Prepared:
    params.check()
    r = dataset.ratings
    positive = r.ratings >= params.min_rating
    users, items, times = r.users[positive], r.items[positive], r.times[positive]
    keep = k_core(users, items, params.min_user_positives, params.min_item_positives)
    users, items, times = users[keep], items[keep], times[keep]
    if len(users) == 0:
        raise PrepareError("no positives survive the k-core; lower the bounds or min_rating")

    user_raw_ids = np.unique(users)
    item_raw_ids = np.unique(items)
    u = np.searchsorted(user_raw_ids, users).astype(np.int64)
    i = np.searchsorted(item_raw_ids, items).astype(np.int64)
    train, val, test = leave_last_out(u, i, times)
    movies: list[Movie] = [dataset.movies[int(raw)] for raw in item_raw_ids]
    return Prepared(
        train_users=u[train],
        train_items=i[train],
        train_times=times[train],
        val_items=i[val],
        val_times=times[val],
        test_items=i[test],
        test_times=times[test],
        user_raw_ids=user_raw_ids.astype(np.int64),
        item_raw_ids=item_raw_ids.astype(np.int64),
        titles=tuple(m.title for m in movies),
        genres=tuple(m.genres for m in movies),
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(p: Prepared) -> dict[str, int]:
    train_per_item = np.bincount(p.train_items, minlength=p.n_items)
    return {
        "users": p.n_users,
        "items": p.n_items,
        "train": len(p.train_users),
        "val": len(p.val_items),
        "test": len(p.test_items),
        "items_without_training_positives": int((train_per_item == 0).sum()),
    }


def save(p: Prepared, out_dir: Path, dataset: str, sources: dict[str, str], params: Params) -> dict[str, Any]:
    """Write ``out_dir`` and return the manifest.

    Everything is written to a temporary directory first. With no ``out_dir`` yet, one rename puts it in place. To
    replace an existing ``out_dir`` (POSIX can't swap two directories atomically) the old one is renamed aside, the new
    one renamed in, and only then the old one deleted; if the second rename fails, the old one is renamed back. A
    crash between the two renames leaves the old directory beside ``out_dir`` as ``.<name>-old-*``, never a mix.

    ``sources`` maps each raw input file's name to its SHA-256, so a manifest names exactly what it was built from.
    """
    for text in (*p.titles, *(g for genres in p.genres for g in genres)):
        if _CONTROL.search(text):
            raise PrepareError(f"{text!r} holds a tab, line break or other control character")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        for name in ARRAYS:
            np.save(tmp / f"{name}.npy", getattr(p, name), allow_pickle=False)
        with (tmp / "items.tsv").open("w", encoding="utf-8", newline="\n") as f:
            f.write("index\traw_id\ttitle\tgenres\n")
            for index, (raw, title, genres) in enumerate(zip(p.item_raw_ids, p.titles, p.genres, strict=True)):
                f.write(f"{index}\t{raw}\t{title}\t{'|'.join(genres)}\n")
        files = sorted(f.name for f in tmp.iterdir())
        manifest: dict[str, Any] = {
            "format": FORMAT_VERSION,
            "dataset": dataset,
            "sources": dict(sorted(sources.items())),
            "params": {
                "min_rating": params.min_rating,
                "min_user_positives": params.min_user_positives,
                "min_item_positives": params.min_item_positives,
            },
            "counts": counts(p),
            "files": {name: sha256_of(tmp / name) for name in files},
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _replace_dir(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


def _replace_dir(new: Path, target: Path) -> None:
    if not target.exists():
        os.replace(new, target)
        return
    old = Path(tempfile.mkdtemp(prefix=f".{target.name}-old-", dir=target.parent))
    os.replace(target, old)  # onto the empty directory mkdtemp made
    try:
        os.replace(new, target)
    except BaseException:
        os.replace(old, target)
        raise
    shutil.rmtree(old)


EXPECTED_FILES = frozenset({*(f"{name}.npy" for name in ARRAYS), "items.tsv"})


def read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError:
        raise PrepareError(f"{path} is missing (run signalrec prepare first)") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrepareError(f"{path}: unreadable: {exc}") from None
    if not isinstance(manifest, dict):
        raise PrepareError(f"{path}: not a JSON object")
    return manifest


def load(directory: Path) -> Prepared:
    """Load a processed directory, refusing any file whose SHA-256 differs from the manifest's."""
    manifest = read_manifest(directory)
    if manifest.get("format") != FORMAT_VERSION:
        raise PrepareError(f"{directory}: format {manifest.get('format')!r}, expected {FORMAT_VERSION}")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != EXPECTED_FILES:
        listed = sorted(files) if isinstance(files, dict) else files
        raise PrepareError(f"{directory}: the manifest lists {listed}, expected {sorted(EXPECTED_FILES)}")
    for name, digest in files.items():
        if sha256_of(directory / name) != digest:
            raise PrepareError(f"{directory / name}: SHA-256 differs from manifest.json")
    arrays = {name: np.load(directory / f"{name}.npy", allow_pickle=False) for name in ARRAYS}
    titles: list[str] = []
    genres: list[tuple[str, ...]] = []
    with (directory / "items.tsv").open(encoding="utf-8", newline="\n") as f:
        next(f)
        for line in f:
            _, _, title, genre_field = line.rstrip("\n").split("\t")
            titles.append(title)
            genres.append(tuple(genre_field.split("|")) if genre_field else ())
    return Prepared(**arrays, titles=tuple(titles), genres=tuple(genres))
