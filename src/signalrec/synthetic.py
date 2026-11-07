"""A small synthetic dataset in ml-latest-small's exact file format, for tests and CI. **Synthetic**: no real ratings.

Users and movies get random latent factors; each user rates movies drawn with probability rising with their affinity
and a popularity bias, and the rating rises with the affinity. So there is structure for item-kNN to find beyond
popularity, and every number measured on it is about this generator, not about people. Timestamps include runs of
equal seconds so that the split's tie-breaking is exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GENRES = ("Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Western")


def write(
    raw_dir: Path,
    n_users: int = 300,
    n_items: int = 400,
    dim: int = 8,
    ratings_per_user: tuple[int, int] = (15, 60),
    seed: int = 7,
) -> Path:
    """Write ``ratings.csv`` and ``movies.csv`` (CRLF, as GroupLens ships them) to ``raw_dir``; return it."""
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(size=(n_users, dim))
    item_factors = rng.normal(size=(n_items, dim))
    popularity = rng.normal(scale=1.0, size=n_items)
    raw_dir.mkdir(parents=True, exist_ok=True)

    lines = ["userId,movieId,rating,timestamp"]
    for user in range(n_users):
        affinity = item_factors @ user_factors[user] / np.sqrt(dim)
        logits = 1.5 * affinity + popularity
        p = np.exp(logits - logits.max())
        p /= p.sum()
        count = int(rng.integers(ratings_per_user[0], ratings_per_user[1] + 1))
        chosen = rng.choice(n_items, size=count, replace=False, p=p)
        stars = np.clip(np.round((3.2 + 1.2 * affinity[chosen] + rng.normal(scale=0.6, size=count)) * 2) / 2, 0.5, 5)
        # Gaps of 0 (a tie), a few seconds, or days.
        gaps = rng.choice([0, 1, 30, 86_400], size=count, p=[0.2, 0.3, 0.3, 0.2])
        times = 946_684_800 + int(rng.integers(0, 10**8)) + np.cumsum(gaps)
        for item, star, t in zip(chosen, stars, times, strict=True):
            lines.append(f"{user + 1},{item + 1},{star:.1f},{t}")
    (raw_dir / "ratings.csv").write_bytes(("\r\n".join(lines) + "\r\n").encode())

    movies = ["movieId,title,genres"]
    for item in range(n_items):
        top = np.argsort(-item_factors[item][: len(GENRES)])[: 1 + item % 3]
        genres = "|".join(GENRES[g] for g in sorted(top))
        movies.append(f"{item + 1},Synthetic Movie {item + 1} ({1980 + item % 40}),{genres}")
    (raw_dir / "movies.csv").write_bytes(("\r\n".join(movies) + "\r\n").encode())
    return raw_dir
