# Signal: design

Signal recommends movies. It is built the way production recommenders are: a cheap **retrieval** stage narrows a
catalogue to a few hundred candidates, a **ranking** stage orders them with richer features, and a **filter** stage
applies business rules, behind an online API that reads fresh user features from a feature store. Everything is
measured offline first, against baselines that a fancier model has to beat.

This file is the specification for all five days. Each day's section of the README says what was built and measured.

## Goals and non-goals

- **Goals.** A reproducible data pipeline from the public MovieLens files; an evaluation protocol that is hard to fool
  (full ranking, temporal hold-out, tuned on validation, reported on test, with confidence intervals); baselines;
  a two-tower retriever; approximate nearest-neighbour retrieval; a ranker; online features; a serving API with
  measured latency; an A/B test simulator that shows what an online experiment would need to detect a difference.
- **Non-goals.** State-of-the-art accuracy, GPUs, distributed training, a UI, real users. The models are small enough
  to train on a laptop CPU in minutes.

## Data

### Sources

| Dataset | Ratings | Users | Movies | Use |
|---|---|---|---|---|
| `ml-latest-small` | 100,836 | 610 | 9,742 | development, CI |
| `ml-1m` | 1,000,209 | 6,040 | 3,706 (rated) | the reported results |

Both come from GroupLens (Harper and Konstan, 2015). **Neither is committed.** `signalrec download <name>` fetches the
zip from `files.grouplens.org`. The ml-1m licence forbids redistribution without permission and ml-latest-small allows
it only under its own terms, so the repository holds code, not data, and CI downloads from GroupLens itself and
publishes nothing derived from the data as an artifact. Research use only; no endorsement by GroupLens is implied.

### Download

- Each archive is pinned by **SHA-256** (taken once by hand and cross-checked against GroupLens's published MD5) and
  by size. The file is streamed to a temporary name in the target directory, hashed, and renamed into place only if
  both match; a mismatch deletes it and fails with both hashes.
- Extraction reads only the **expected member names**, byte for byte (`ml-1m/ratings.dat`, ...), and writes them
  under `data/raw/<name>/`. Nothing else in the archive is touched, so a hostile or corrupted zip cannot write outside
  the directory (no "zip slip"), and an archive missing a member fails before anything is written. Members go to a
  temporary directory that is renamed into place, so a crash never leaves a half-extracted dataset that looks done.
- Re-running is a no-op when the zip's hash still matches and the extracted files exist.

### Parsing

Two formats: ml-latest-small is CSV with a header (`userId,movieId,rating,timestamp`; titles quoted when they hold
commas); ml-1m is `::`-separated with no header (`UserID::MovieID::Rating::Timestamp`), Latin-1. The parsers are
strict: every row must have exactly the expected fields, ids are positive integers, ratings are in the dataset's
allowed set (half stars 0.5-5.0 for ml-latest-small, whole stars 1-5 for ml-1m), timestamps are non-negative
integers, and a (user, movie) pair may appear only once. A violation names the file and line. Movies carry a title
and a list of genres; every rated movie must exist in the movie file.

### From ratings to implicit feedback

Recommendation here is a top-N problem on implicit feedback: *which movies will this user like next?*

1. **Positives.** A rating of at least 4.0 is a positive; lower ratings are dropped (`--min-rating`, default 4.0).
   A recommender that predicts *watching* would keep every rating; predicting *liking* matches what the product
   wants and makes the task harder for popularity. The cost: a movie a user rated 2 can be recommended back to them
   offline. The serving filter (day 4) removes every rated movie, not just the positives.
2. **k-core.** Users and movies with fewer than 5 positives are dropped, repeatedly, until both conditions hold at
   once (dropping a movie can push a user under 5, and the other way round). The loop always terminates: each pass
   removes at least one row or stops.
3. **Split: per-user leave-last-out by time.** Each user's positives are sorted by (timestamp, movie id); the last is
   the **test** item, the one before it the **validation** item, the rest are **training**. Movie id breaks timestamp
   ties so the split is a pure function of the data: ml-1m users often rated dozens of movies in the same second.
4. **Indexing.** Users and movies get dense indices 0..n-1 in raw-id order; the raw ids are stored next to them.

The k-core runs before the split, so a movie can survive the k-core and still have no training positives (all of its
positives were someone's last or second-to-last). Such movies stay in the catalogue: they are the cold-start case.

**Known leakage.** A per-user split is not a global time split: one user's training data can be later in calendar
time than another user's test item, so a model can learn from "the future" (Ji et al., 2023). The popularity baseline
is the most exposed to this. A global time split would remove it but leaves few users with both history and a test
item in these datasets. Instead the leak is **measured** (`signalrec leakage`): the share of training positives later
than each user's test item, and popularity counted only from positives strictly before each user's test time against
the usual popularity, compared with a paired bootstrap. The README reports both.

For the two-tower model (day 2, `signalrec leakage-twotower`) a model can't be retrained per user, so users are cut
into 10 buckets of equal size by test time, and for each bucket the reported configuration is retrained on the
training + validation positives before the bucket's earliest test time (a subset of the leaky model's data), then
compared paired with the reported model. That also drops the bucket's own users' recent positives, which are their
own past, so a second model per bucket keeps them. Neither is a strict bound (`leakage_twotower.py` says which way
each errs); the README reports both.

### Processed format

`data/processed/<name>/` holds one `.npy` file per array (`train_users`, `train_items`, `train_times`, the same for
`val` and `test`, `user_raw_ids`, `item_raw_ids`), `items.tsv` (index, raw id, title, genres) and `manifest.json`
(the source archive's SHA-256, the parameters, the counts, and the SHA-256 of every file). `np.save` is
byte-deterministic, so the same inputs and parameters give the same bytes, and the manifest's hashes prove it.
Loading requires the manifest to list exactly those files and every hash to match. Titles and genres may not hold
control characters (the parsers refuse them too). Replacing an existing directory renames the old one aside, the
new one in, then deletes the old one; if the second rename fails the old one is put back. (POSIX has no atomic swap
of two directories: a crash between the renames leaves the old one beside it under a hidden name.)

## Evaluation protocol

- **Full ranking.** Every model scores every movie for every evaluated user; no sampled negatives (sampled metrics can
  reorder models: Krichene and Rendle, 2020).
- **History is excluded.** A user's history items are removed from their ranking before the cut-off.
- **Validation for choices, test for numbers.** Hyperparameters are chosen on the validation item with the training
  positives as history. Test numbers come from the model refit on training + validation (the validation item is part
  of the history at test time: it happened earlier), with that combined history excluded.
- **Metrics at K = 10, 20, 50.** With one held-out item per user, Recall@K is the fraction of users whose item is in
  the top K, and NDCG@K is `1 / log2(rank + 1)` when it is (rank from 1), else 0; both are averaged over users.
  Catalogue coverage@K is the share of movies that appear in at least one user's top K.
- **Uncertainty.** Each test metric gets a 95% bootstrap interval over users; two models are compared with a paired
  bootstrap on the same resampled users.
- **Ties.** Scores are ranked by (score descending, movie index ascending), so ties never depend on the sort
  algorithm.

## Baselines (day 1)

- **Popularity.** Score = number of training positives. Personalised only by the history filter.
- **Item-kNN** (item-based top-N, Deshpande and Karypis, 2004). On the binary user × movie matrix X of the history:
  `sim(i, j) = co(i, j) / (sqrt(n_i * n_j) + shrink)` where `co` counts users who liked both, `n` counts users who
  liked each; the diagonal is zero; each movie keeps its `k` most similar neighbours. A user's score for movie i is the
  sum of `sim(j, i)` over their history items j. The similarity is computed in column blocks (`X^T X[:, block]`), so
  memory is O(movies × block) rather than O(movies²). `k` and `shrink` are tuned on validation.
- **Exact ties.** "Ties to the smaller index" must not be decided by floating-point rounding (review round 1 found
  that it was). With no shrink a similarity is computed as `sqrt(co² / (n_i n_j))`, so equal rationals give the same
  double; every similarity is rounded to a multiple of 2^-32 and kept as an integer; a score is an exact integer sum.
  Equal similarities are therefore equal and no score depends on summation order. Not guaranteed, because no finite
  representation can: two sums of *different* square roots that are equal as reals may differ by a unit.

## Day 2: two-tower retriever

- **Item tower:** a movie-id embedding plus the mean of its genre embeddings, then a small MLP, L2-normalised.
- **User tower:** the mean of the user's history embeddings (a separate table from the item tower's), a small MLP,
  L2-normalised. No user-id embedding: the tower works for any history, including one that grew a minute ago from the
  online feature store, and for users the model never saw.
- **Loss:** softmax over the batch with in-batch negatives and temperature τ; every other positive in the batch is a
  negative. Popular movies show up as negatives more often than uniform sampling would, so their logits are corrected
  by subtracting `log q(i)`, the movie's sampling probability (Yi et al., 2019, "logQ correction").
- **Training examples:** every training positive except each user's first (which has no earlier positive), with the
  user's earlier positives as its history (by the same time order as the split), capped at the 50 most recent; no
  future item ever enters its own history.
- Same protocol and metrics as the baselines; the model must beat item-kNN on validation to be worth serving.
- **Choosing hyperparameters under noise.** Day 1 showed that the validation maximum can be noise: item-kNN's k=3 was
  indistinguishable from 7 of 29 grid rows on ml-1m (19 of 29 on ml-latest-small) and k=10 did no worse on test.
  So day 2 reports the paired validation interval of every candidate against the best, as day 1 now does, and
  "beats item-kNN" means the paired interval of (two-tower - item-kNN) NDCG@10 on validation excludes zero, not just
  a higher mean. Training seeds are a source of noise too: the chosen configuration is trained with three seeds and
  the spread reported.

### As built (day 2)

- **Towers** (`twotower.py`). Item: `id embedding + genre_mix @ genre_table`, where a movie's genre row averages its
  genres (a movie with none gets no genre vector), then `x + MLP(x)` (Linear 64→128, ReLU, Linear 128→64) and L2
  normalisation. User: the mean of the history table's rows over the non-padding positions (row `n_items` is padding:
  zero, never updated), then its own residual MLP and L2 normalisation. Embeddings start at N(0, 0.1²). The residual
  form means each tower starts as its embeddings and learns a correction.
- **Examples.** A user's first positive has no earlier positive, so it is never a target; every later one is, with
  the up to 50 positives just before it (left-padded). ml-1m: 556,274 examples when tuning (562,308 training
  positives minus one per user), 562,308 for the test run (the validation item adds one per user). Histories are
  sliced from one concatenated array per batch, so no example table is materialised.
- **Loss** (`in_batch_loss`). `logits = U Vᵀ / τ − log q(column's movie)`, cross-entropy with the diagonal as the
  label. `q` is each movie's exact share of the training targets (one pass over the data, not Yi et al.'s streaming
  estimate, which is for data that doesn't fit). A column holding the same movie as the row's target is masked to
  −∞ for that row: an accidental hit is not a negative.
- **Training.** Adam, learning rate 3e-3, batch 1,024, shuffled each epoch by `numpy.random.default_rng(seed)`; the
  weights start from `torch.manual_seed(seed)`. Same seed, same machine → same weights (a test). A final batch of one
  example is skipped (no negatives).
- **Scoring.** The towers are exported to NumPy (`Embeddings`): one vector per movie and one per user from the last 50
  positives of their history at that point (training for validation, training + validation for test). Scores are
  float64 dot products of the float32 vectors; evaluation is exactly the baselines' (`evaluate.Scorer`).
- **Selection** (`experiment_twotower.py`). Fixed: dim 64, hidden 128, batch 1,024, lr 3e-3, 50-item histories.
  Grid: τ ∈ {0.05, 0.1, 0.2} × logQ on/off, 20 epochs each, validation scored after every epoch. The candidates
  are the 120 (configuration, epoch) pairs; the choice is the best validation NDCG@10 (ties to the earlier
  configuration, then epoch). Item-kNN is tuned on its own day-1 grid in the same run and compared paired on
  validation. Test: the chosen configuration retrained on training + validation for the chosen number of epochs with
  seeds 0, 1 and 2; seed 0 (the seed the grid used) is the reported row, the other two give the spread.

## Day 3: FAISS, ranking, the feature store

- **First, the history length.** Day 2's post-hoc recency ablation found that the two-tower model does much better on
  validation with 5-item histories than with DESIGN's fixed 50 (NDCG@10 0.1233 against 0.0799, one seed). Day 3 makes
  the history length a dimension of the validation grid, so the protocol chooses it, and the retriever FAISS indexes
  is the chosen one.

- **Retrieval.** Item embeddings in FAISS: `IndexFlatIP` (exact) as the reference, then an approximate index (IVF or
  HNSW). Measured: recall of the approximate top 200 against the exact top 200, and query latency.
- **Ranking.** A small model scores the retrieved candidates with features retrieval cannot use cheaply: the retrieval
  score, movie popularity, the user's genre affinity against the movie's genres, the movie's release year against the
  user's typical years, the user's activity. Trained on the validation targets' candidate lists (label: is this the
  held-out movie), evaluated on test, so it never trains on what it is scored on.
- **Feature store (Redis on 6383).** Online features: the user's recent history (the last 50 positives), their genre
  counts, and movie popularity counters. A test computes the features offline from the processed data and online
  from Redis after replaying the same events, and requires them to be equal: training-serving skew is a bug class
  with a test.

### As built (day 3)

- **History length in the grid** (`experiment_twotower.py`). The configurations are the (temperature, logQ) pairs
  crossed with `--histories` (default 1, 3, 5, 10, 20, 50), pairs outer; the chosen length is reported under
  `"chosen"`, no longer `"fixed"`. The default grid keeps logQ on (τ ∈ {0.05, 0.1, 0.2}): on day 2 every logQ-off
  configuration lost about 0.03 validation NDCG@10 with an interval excluding zero, a validation result carried
  forward. A results file holds `max_history` in exactly one of the two places, so day 2's files
  (`twotower-day2.json`) still read.
- **Retrieval** (`retrieval.py`). `Retriever` over `IndexFlatIP` (exact), `IndexHNSWFlat` or `IndexIVFFlat`, inner
  product. The user's movies are excluded inside the search by an `IDSelectorNot(IDSelectorBatch(seen))`, one query
  per call (the serving path; a selector belongs to one query), so a heavy user still gets n unseen candidates.
  Builds run on one OpenMP thread so the same vectors build the same index. FAISS's HNSW search list stops at
  `efSearch` itself (it does not take the maximum with k, which this line used to claim; review of day 3), so the
  Retriever passes `max(ef_search, k)`, where k = n + 16 is what it asks for. The 16 extra candidates are rescored in
  float64 and sorted by (score, index) before the cut; if the last fetched is not at least 1e-5 below the n-th, a
  tie group may run past them, and the search repeats with twice as many (review round 2).
- **Two OpenMP runtimes.** The macOS PyTorch and faiss-cpu wheels each bundle libomp, and they coexist in one order
  only (measured): faiss loaded first makes torch's first parallel region segfault; torch first aborts the process
  (OMP Error #15) unless `KMP_DUPLICATE_LIB_OK=TRUE`; torch first with the flag works for one-query searches, while a
  5-query batch search hung. So on macOS `retrieval.py` sets the flag and imports torch (if installed) before faiss,
  and the Retriever searches one query per call. `tests/test_openmp_order.py` trains and then retrieves in a fresh
  interpreter for both import orders of the caller (the faiss-first one segfaulted before this, as
  `scripts/bench_retrieval.py` did). Intel calls the flag unsupported; the tests check FAISS against a NumPy brute
  force in that setting. Linux wheels use libgomp.
- **Features** (`features.py`). Per user: the last 50 positives in the split's (time, movie id) order, every positive
  (the history filter), positives per genre, and the sum and count of the liked movies' release years (from the
  title's trailing "(1995)"; some ml-latest-small titles have none). Per movie: positives. All integers, so offline
  and online compare exactly. The replay order is (time, movie, user), which restricted to one user is the split's
  order.
- **Feature store** (`featurestore.py`, Redis on 6383). Keys `<ns>:u:<user>:recent` (list), `:seen` (set), `:genres`
  and `:stats` (hashes), and `<ns>:popularity` (hash). One Lua script applies an event: `SADD` to `seen` first, and
  nothing else if the movie was already there, so the script is atomic and a redelivered event is a no-op (a (user,
  movie) pair is a positive at most once). A read is one MULTI/EXEC of four commands, so it never sees half an event;
  values the store didn't write (a non-integer count, a movie outside the catalogue, a key of another type) are
  errors, not features. Events are appended in arrival order: online, an event that arrives late becomes the most
  recent one (a limitation).
- **The skew test.** The events are replayed into Redis, and every user's features and the popularity array must equal
  the offline ones exactly; then the ranker's feature vectors from either source must be equal bit for bit.
- **Ranking** (`ranking.py`, `experiment_ranking.py`). Eight features per candidate, computed from a `UserFeatures` and
  the popularity array only: retrieval score, log(1 + position), log(1 + popularity), genre affinity (the user's genre
  shares averaged over the candidate's genres), |year − the user's mean year| / 10, year unknown, log(1 + the user's
  positives), and the candidate's item vector against the user's last positive's. The model is linear or Linear →
  ReLU → Linear on standardised features, trained listwise (a softmax over each list, the held-out movie as the
  label; lists without it are skipped), Adam, 256 lists per batch, exported to NumPy. Protocol: the reported two-tower
  configuration trained on training retrieves the top 200 unseen movies per user (exact index) for the validation
  lists; the ranker trains on a seeded 80% of the users and (hidden size, epochs) is chosen on the other 20% by
  NDCG@10; the test lists come from the model retrained on training + validation, and a movie not retrieved ranks
  201. The ranker never sees a test label, and the movies it re-orders on test were retrieved by a model that never
  saw them.

## Day 4: serving and experiments

- **API (FastAPI on 8116).** `GET /recommendations/{user}?k=10`: history from Redis → user tower (exported to numpy,
  no torch on the request path) → FAISS top 200 → ranker → filter (every movie the user has rated, and any excluded
  genres) → top k. `POST /events` records an interaction and updates the features. `/health`.
- **Latency benchmark:** p50/p95/p99 per stage and end to end, open loop at fixed rates and closed loop, warm-up
  excluded, with the command next to every number.
- **A/B test simulator.** Users are assigned to arms by a hash of (experiment, user); a simulated user model decides
  clicks; the simulator reports the measured lift with a confidence interval and how many users a real test would
  need to detect it (power analysis). Its user model is synthetic and labelled as such.
- **What day 3 leaves for day 4.**
  - The user tower must run in NumPy: its history table, MLP weights and normalisation exported next to the item
    vectors, with a test that NumPy and PyTorch give the same user vector for the same history. The reported model
    reads the last 3 positives, which the store's 50-item recent list covers.
  - The filter removes every *rated* movie, but the store's seen set holds positives only (rating >= 4), because the
    processed data keeps only positives. Serving needs a rated set, filled from the raw ratings by the replay and by
    `POST /events` for any rating; retrieval then excludes that set, and the ranker's features stay positives-only.
  - `POST /events` appends in arrival order (a late event becomes the most recent), and goes through the same Lua
    script, so a retried request is harmless.
  - On macOS the serving process loads torch through `retrieval.py` (the only safe order with faiss) but never
    calls it; the user tower runs in NumPy. A serving process without torch installed has one runtime.

## Day 5

README with results, WALKTHROUGH.md, `v1.0.0`.

## Testing

- Parsers and the downloader against small hand-made files and archives (including hostile zips), no network.
- The pipeline on a synthetic MovieLens-shaped dataset (generated, labelled synthetic): k-core fixed point, split
  invariants (one test and one validation item per user, each the latest, training strictly earlier or tied with a
  smaller id), determinism (same bytes twice).
- Metrics against hand-computed cases and a brute-force implementation (Hypothesis).
- Item-kNN against a direct dense computation on random matrices; popularity by hand.
- CI runs everything on every push and then the whole pipeline on the real `ml-latest-small`, downloaded from
  GroupLens and cached by its SHA-256.

## Ports

API 8116, Redis 6383.
