# Changelog

## 1.0.0 (2026-11-11)

The first release: five days of work, each in the README's results and review sections.

- **Day 1 (2026-11-07): data, protocol, baselines.** Pinned MovieLens downloads (ml-latest-small, ml-1m), strict
  parsers, positives at rating ≥ 4, a 5-core, a leave-last-out split by (time, movie id), byte-deterministic
  processed data with a hashed manifest. Full-ranking evaluation with exact ranks and paired bootstrap intervals.
  Popularity and item-kNN, chosen on validation. The split's cross-user time leakage measured.
- **Day 2 (2026-11-08): the two-tower retriever.** In-batch negatives with logQ correction, a user tower over the
  history with no user id, epochs chosen on validation; what the leak is worth to it.
- **Day 3 (2026-11-09): history length, retrieval, ranking, the feature store.** The history length chosen on
  validation (3); FAISS exact, HNSW and IVF with the user's movies excluded inside the search; a listwise ranker over
  eight features; a Redis feature store (a Lua script per event: atomic, idempotent) whose features equal the offline
  ones bit for bit. The same-second order measured (31.4% of ml-1m users).
- **Day 4 (2026-11-10): serving and experiments.** The user tower in NumPy; FastAPI (`GET /recommendations/{user}`,
  `POST /events`, `/health`); a script requiring the request path to reproduce the offline test metrics (equal to
  1e-12); in-process, closed-loop and open-loop latency; an A/B simulator with a power analysis that checks its own
  statistics.
- **Day 5 (2026-11-11): new users, v1.0.0.** A registry gives MovieLens ids the model doesn't know a dense index on
  their first event (`FeatureStore.register`: atomic, idempotent, numbered from the model's users up); a user with no
  positives gets the most popular movies they haven't rated, and the two-tower model from their first positive;
  `scripts/cold_start.py` measures both on users held out of training; the popular list is kept for a second
  (`--popular-ttl`), measured by `scripts/bench_new_users.py`; `signalrec serve --workers N` runs N server
  processes (1,152 → 1,841 requests/s from 1 to 4 on the same laptop as the clients). The day's review fixed five
  defects (registry ordinals, a damaged registry refused, event id bounds, untimed exclusions, an unpaced benchmark).
  README results summary and reading order, WALKTHROUGH day 5, this changelog.
