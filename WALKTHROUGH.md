# Walkthrough

Questions an interviewer is likely to ask about Signal, with short answers grounded in the code. Day 1 covers the
data, the evaluation protocol and the baselines; later days add sections.

## The data (day 1)

**Which data, and why isn't it in the repo?**
MovieLens ml-1m (1,000,209 ratings) for the results and ml-latest-small (100,836) for development and CI. The ml-1m
licence forbids redistribution without permission, so `signalrec download` fetches the zip from GroupLens and checks
it against a pinned SHA-256 and size (`download.ARCHIVES`; the pins were cross-checked against GroupLens's published
MD5s). CI downloads ml-latest-small the same way and uploads nothing derived from it.

**What does the downloader protect against?**
A swapped or truncated file (the hash and size), a server that sends forever (`fetch` stops reading past the pinned
size), and a hostile or corrupt archive: `extract` reads only the expected member names (so `../escaped.csv` is never
opened, let alone written), caps each member's declared size, lets `zipfile` check the CRC, and writes into a temporary
directory that is renamed into place behind a marker file. A crash mid-extraction leaves no directory that looks
complete. `tests/test_download.py` builds each of those archives.

**Why such strict parsers?**
Because the files had surprises, and a lenient parser would have hidden them: ml-latest-small uses CRLF line endings,
ml-1m's `movies.dat` is Latin-1 (a lone 0xe9 byte is "é"), and 11 ml-latest-small titles end in a space. Each was a
parse error first, then a documented decision in `movielens.FORMATS` or `_movie`. A rating line must match a regex
exactly (`Format.ratings_line`), ratings must be in the dataset's star set, and a repeated (user, movie) pair is an
error that names both lines.

**What is a positive, and why?**
A rating of 4 or more (`prepare.Params.min_rating`). The product wants movies a user will *like*. Lower ratings are
dropped from the signal; the day-4 serving filter still removes every movie a user has rated.

**What is the k-core and why iterate it?**
Users and movies with fewer than 5 positives are dropped. Dropping a movie can push a user below 5 and vice versa, so
`prepare.k_core` repeats until both bounds hold at once. The test compares it with a brute-force search over every
choice of users and movies: the k-core is the union of all valid choices, and the code must return exactly that.

**How do you split, and what's wrong with it?**
Per-user leave-last-out (`prepare.leave_last_out`): each user's positives sorted by (time, movie id); the last is the
test item, the one before it validation, the rest training. Movie id breaks ties so the split is a pure function of
the data; ml-1m users often rated dozens of movies in the same second. What's wrong: one user's training data can be
later in calendar time than another user's test item, so models see "the future" (DESIGN.md, "Known leakage").
Popularity is the most exposed. A global time split removes it but leaves few users with both history and a test item.

**How do you know the processed data is what you think it is?**
`prepare.save` writes one `.npy` per array (byte-deterministic), `items.tsv`, and `manifest.json` with every file's
SHA-256, the source files' hashes, the parameters and the counts. `prepare.load` refuses a file whose hash differs.
A test writes the same data twice and compares bytes; another pins the real datasets' counts.

## The evaluation protocol (day 1)

**Why full ranking instead of sampled negatives?**
Ranking the held-out movie against 100 random negatives is cheaper but can reorder models (Krichene and Rendle, 2020).
Here every model scores all ~3,000 movies for every user, minus the user's history. Hence the small absolute numbers:
NDCG@10 of 0.03 means the held-out movie is usually not in the top 10 of 3,000.

**How is the rank computed, and why not just sort?**
`evaluate.ranks` counts: 1 + the non-history movies scored higher + those scored equal with a smaller index. That is
exactly the rank a sort by (score descending, index ascending) gives, without depending on which sort is used.
Popularity produces many ties, so this matters. `top_k` uses a stable sort for the same order; a property test checks
that the target is in `top_k[:K]` exactly when its rank is ≤ K.

**What are Recall@K and NDCG@K with one held-out item?**
Recall@K is 1 if the item is in the top K, else 0, averaged over users. NDCG@K is `1/log2(rank+1)` inside the cut-off
(the ideal DCG is 1). Coverage@K is the share of the catalogue that appears in anyone's top K.

**Why validation for choices and test for numbers?**
Choosing item-kNN's k on the test set would overstate it. `experiment.run` chooses (k, shrink) by validation NDCG@10
with the training positives as history, then refits both models on training + validation and evaluates the test item,
with the validation item excluded as history (it happened earlier). A test checks no user's top list contains their
validation item at test time.

**What do the intervals mean?**
95% percentile bootstrap over users (`evaluate.bootstrap_ci`): resample the users with replacement 1,000 times and take
the 2.5th and 97.5th percentiles of the mean. Comparing two models uses a paired bootstrap on the same resampled
users (`paired_bootstrap_ci`), which is much tighter than comparing two separate intervals because the per-user
difficulty cancels.

## The baselines (day 1)

**Explain item-kNN.**
`sim(j, i) = co(j, i) / (sqrt(n_j n_i) + shrink)`, cosine on the binary like matrix with a shrink that discounts
similarities built on few users; each movie keeps its k most similar movies; a user's score for i sums `sim(j, i)` over
the movies j they liked (`baselines.neighbours`, `ItemKNN.score`). The similarity is built in column blocks of
`X^T X`, so memory is O(items × block), not O(items²); the test checks it against a direct item-by-item computation of
the definition for random matrices, k, shrink and block sizes.

**Which k won, and what does that tell you?**
On ml-1m, validation NDCG@10 peaks at k=3, so the protocol reports k=3: NDCG@10 0.0314 against popularity's 0.0193
(paired difference +0.0122, interval [0.0085, 0.0157]). A post-hoc test run with k=10 scored 0.0322 at NDCG@10 and
0.2158 at Recall@50 (k=3: 0.1845): NDCG@10 is flat across small k, larger k recovers more of the long tail, and
validation's small preference for k=3 didn't carry to test. I didn't switch to k=10, because that would be choosing on
test; the better fix is a selection rule that accounts for validation noise.

**Why is popularity so strong, and what's wrong with it?**
Many users like the same well-known movies, so popularity gets about 4% of users' next liked movie into the top 10.
But its lists cover 3.7% of the catalogue against item-kNN's 21%, and it is the model most helped by the split's
calendar leakage.

**How did you test the tie-breaking in item-kNN?**
With 64 movies liked by the same users, every similarity ties, and each movie's 10 neighbours must be the 10 smallest
indices. A smaller case wouldn't do: NumPy sorts fewer than 16 elements by insertion sort, which is stable anyway, so
an unstable-sort mutant survived a 4-movie test and is killed by the 64-movie one.

**What runs in CI?**
Lint, mypy --strict and the tests (on synthetic and hand-made data); then the real ml-latest-small, downloaded by its
pinned hash, through every real-data test, the pipeline and the baselines, compared with the committed results
(`scripts/compare_results.py`): counts and the chosen k exactly, metrics within 0.005.

## What review and measurement changed (day 1)

**How do you make "ties go to the smaller index" true for item-kNN?**
Two traps. Equal similarities computed differently (3/√27 and 1/√3) differ in a double's last bit, and equal scores
summed in different orders can differ too. So with no shrink a similarity is `sqrt(co²/(n_i n_j))`: IEEE division
and square root are correctly rounded, so equal rationals give identical doubles. Then each similarity is rounded to
a multiple of 2^-32 and stored as an int64, and a score is an integer sum: order can't matter. What can't be fixed:
1/√12 + 1/√12 and 1/√3 are equal as reals but each rounds separately; deciding whether two sums of square roots are
equal needs exact algebraic arithmetic. `tests/review_d1/test_precision.py` pins that case as a bound.

**Why integers and not float64?**
float64 makes the error smaller but keeps both traps: the reviewer's score case is exactly one ulp apart in float64
too, in the same direction (checked by summing the terms in Python). With
integers, the guarantees are exact and the cost is a resolution of 2^-32 (about 2.3e-10) in what "equal similarity"
means. A score stays exact in float64 for histories up to 2^21 movies; `fit` checks it.

**What was the most serious thing the review found?**
Not a defect but a gap: replacing the validation items with the test items in tuning (`experiment._timed_eval`)
survived every test. The protocol's most important rule was untested. The test that kills it re-evaluates every grid
row independently on the validation items and requires the same numbers.

**Is replacing the processed directory atomic?**
No, and DESIGN.md says so. POSIX can't swap two directories atomically. `_replace_dir` renames the old one aside,
the new one in, and deletes the old one; if the second rename fails, the old one goes back. A crash between the
renames leaves the old directory beside the target under a hidden name, never a half-written one in its place.

**Why does the leakage barely help popularity?**
For the average ml-1m user, 44% of training positives are later than their test item, but popularity counted only
from earlier positives is within noise at NDCG@10 (+0.0002 [-0.0006, +0.0010]) and only slightly worse at
Recall@50 (+0.0043 [+0.0013, +0.0073]). My reading, not something I measured: the head of the popularity list is
stable over ml-1m's few years, so later likes mostly reshuffle positions 10-50. `leakage.popularity_as_of` computes it in one sweep over users sorted by test time.

**Why run more than one review round?**
Because fixes have bugs. Round 1 mapped `OSError` to a one-line error, but a dropped HTTP body raises
`http.client.IncompleteRead`, which isn't an `OSError`; it replaced "delete then rename" with "rename aside, rename
in", and left an empty directory when the first rename failed; it validated `--grid` and left `recommend`'s own
arguments unvalidated. Round 2 was asked only "what did each fix break?" and found 8 defects, most of them of that
kind. Each finding arrives as a failing test, so a fix is done when its test passes and the mutant it guards is
re-applied and fails.

**How do you know a test isn't vacuous?**
Mutation: change the code the test guards (an operator, a constant, a swapped argument), run the test, restore the
file from a copy and compare bytes. One example of a test that looked right and wasn't: "the interval brackets the
mean difference" passed with the arguments of the paired bootstrap swapped, because on 150 users the interval is
wide enough to hold both d and -d. The test now recomputes the interval independently and compares exactly.

## The two-tower retriever (day 2)

**Describe the model.**
Two networks that map into the same 64-dimensional space (`twotower.TwoTowerNet`). The item tower takes a movie-id
embedding plus the average of its genre embeddings, then `x + MLP(x)` and L2 normalisation. The user tower averages
the embeddings of the user's last 50 liked movies (from its own table, not the item tower's), then its own residual
MLP and normalisation. The score is the dot product, i.e. a cosine. Retrieval only needs the user vector at request
time; the movie vectors are precomputed, which is what makes FAISS (day 3) possible.

**Why no user-id embedding?**
A user is their history. The tower works for a user the model never saw and for a history that changed a minute ago
(the day-3 feature store keeps the last 50 positives in Redis), with no retraining. The cost: nothing about a user
outside those 50 movies.

**What is the loss?**
A softmax over the batch (`in_batch_loss`): for 1,024 (history, movie) examples, each user vector is scored against
all 1,024 movies, its own movie is the label and the other 1,023 are negatives. It is cheap (one matrix product) and
needs no sampler. Two corrections: **logQ**, subtracting `log q(movie)` from each column's logit, because a popular
movie appears as somebody's negative far more often than a rare one, so the uncorrected loss learns to push popular
movies down; and **accidental hits**: when the same movie is the target of two examples in a batch, it is masked out
as a negative for both.

**Does logQ matter?**
More than anything else in the grid. On ml-1m validation at τ=0.1, NDCG@10 is 0.0799 with it and 0.0496 without, and
the same gap holds at every temperature. Without it the model's top-10 lists cover almost the whole catalogue
(coverage@10 0.997 against 0.62): it has learned that popular movies are "usually wrong", which is the sampling bias,
not the users. `results/ml-1m/twotower-day2.json` has every configuration's learning curve.

**How are training examples built, and how do you know nothing leaks?**
Every training positive except each user's first is a target; its history is the up to 50 positives just before it
in the split's (time, movie id) order (`example_positions`, `histories`). A batch is sliced from one concatenated array
by index arithmetic, so no 556,274 × 50 table is built. Tests: a Hypothesis test compares every example's history with
a brute-force walk; another checks no history contains its target or anything after it; and a spy on `histories` and
`in_batch_loss` during real training checks that each batch's targets are exactly the movies at its histories' ends.
That last one was added because review round 1 showed that shifting the history by one position (the target leaking
into its own history) survived every other test.

**How did you choose the hyperparameters without fooling yourself?**
The same protocol as the baselines. Six configurations (τ ∈ {0.05, 0.1, 0.2} × logQ on/off), 20 epochs each, scored on
validation after every epoch; the 120 (configuration, epoch) candidates are compared by validation NDCG@10 and the
best is chosen. Every candidate gets the paired interval of (chosen - it), so the JSON says how firm the choice is.
Then the chosen configuration is retrained on training + validation for the chosen number of epochs and scored once on
test. "Beats item-kNN" is a paired validation interval above zero, not a higher mean.

**How noisy is training?**
The chosen configuration was trained with seeds 0, 1 and 2 on the test split: NDCG@10 0.0713, 0.0706 and 0.0693.
The seed spread (0.002) is much smaller than the gap to item-kNN (0.040), and smaller than the bootstrap interval
over users (about ±0.005).

**Why does it beat item-kNN by so much?**
Partly recency. The user tower sees the last 50 positives; item-kNN sums over all of them. Post hoc, on validation
only (`scripts/recency_ablation.py`), item-kNN scored from just the last 5 positives rises from 0.0386 to 0.0599, half
its gap to the two-tower model. But with both given the same 5-item history the two-tower model still leads by two to
one (0.1233 against 0.0599), so the learned representation is worth something on its own. The same ablation found
that DESIGN's fixed 50-item history is far from the best (5 items: 0.1233 against 0.0799); I didn't switch, because
that would be acting on one seed of a post-hoc look; day 3 puts the history length in the grid.

**Does the split's time leak help the model more than it helped popularity?**
Probably more, but this check only brackets it. `leakage_twotower.py` retrains the model per test-time bucket on only
what existed before the bucket's earliest test time: the score halves (0.0713 to 0.0354), but that also removes each
user's *own* recent positives, which are their past, so it overstates the leak. A second model keeps the bucket's own
users' histories and scores 0.0712, but it keeps bucket-mates' positives from after a user's test item, a leak of its
own and not a small one (review round 3: in the first bucket, 468 of 43,520 examples predate the cut). So the leak is
worth somewhere between nothing measurable and half the model's NDCG@10. Halving the buckets' size (20 instead of
10) narrows it from below: the own-histories estimate's Recall@50 gap becomes +0.022 [+0.011, +0.034], so the leak
is worth something deeper in the list, as it was for popularity, and little at the top. The
last bucket is a built-in check that the code does what it says: with its own histories kept it trains on exactly the
reported model's data and scores exactly the same.

**What was the most important thing the reviews found?**
That my tests couldn't see inside training. Every test recomputed results through the same `train` and `embed` it
was checking, so a target leaking into its own history, an inverted logQ flag, an ignored temperature or an unmasked
accidental hit all passed. The fix was to watch from outside: a spy on `histories` and `in_batch_loss` records what
each batch actually feeds the loss, and each tower is recomputed by hand from its weights. Round 2's best catch was a
reasoning error rather than a code bug: the as-of model trained on other users' test positives, which the model it
was compared with never sees, so "the as-of model knows less" was false.

**Why is the evaluation code unchanged?**
`evaluate.evaluate` takes any `Scorer` (a name and a `score(users)` method). The trained towers are exported to NumPy
(`twotower.Embeddings`) and scored as float64 dot products, then ranked by exactly the same exact-rank code as the
baselines, with the same tie rule and the same history exclusion. Nothing model-specific touches the metric.

## Retrieval, ranking and the feature store (day 3)

**Why did the history length change, and isn't that tuning on the answer?**
Day 2 fixed the user tower's input at the last 50 positives; its recency ablation (validation only, post hoc) found 5
far better. Day 3 makes the length a grid dimension (`--histories`, default 1, 3, 5, 10, 20, 50) crossed with the
temperature, so the same validation rule chooses it and test is still touched once, by the choice. The ablation only
said which values to put in the grid. logQ stays on in the default grid because day 2's validation already showed
every logQ-off configuration losing with an interval excluding zero.

**What did the history length do?**
On ml-1m validation NDCG@10 falls steadily with length at every temperature: 0.1357 at 3 positives, 0.1261 at 5,
0.1110 at 10, 0.0799 at 50 (τ=0.1). The chosen model (τ=0.1, 3 positives, 8 epochs) scores 0.1220 [0.1155, 0.1286]
on test, against 0.0713 for day 2's 50-positive model and 0.0314 for item-kNN. MovieLens users rate in sessions, and
the last few movies say much more about the next one than the mean of fifty does.

**That is a big jump for reading less. Did you check it?**
Yes, and it is partly an artefact. The split orders positives by (timestamp, movie id), and 31.4% of ml-1m users rated
their test movie in the same second as their validation movie, so the "next" one was picked by movie id.
`scripts/same_second.py` splits test users by that: the 3-item model's gain over day 2 is +0.1042 on the tied users
and +0.0262 [+0.0191, +0.0328] on the untied ones. So the shorter history really is better, but about two thirds of
the headline gain comes from the tied third. I kept the protocol's number (the split was fixed before any model) and
report the untied one beside it (NDCG@10 0.0841; still +0.0570 over item-kNN). A cleaner fix, for later, would be to
treat the movies of the test second as a set.

**Does the ranker help?**
A little, and only on the big dataset. On ml-1m test it lifts NDCG@10 from 0.1220 to 0.1298 (+0.0078 [+0.0050,
+0.0105]) and Recall@10 by +0.0138. On its validation selection users the gain's interval is [+0.0007, +0.0143],
barely above zero, and in the run before the review's float64 rescoring it reached zero; on ml-latest-small it doesn't
help NDCG@10 at all. Retrieval already uses most of what the features know: removing any one feature (and choosing
the ranker again the same way) changes selection NDCG@10 by an amount whose interval reaches zero; the largest is the
similarity to the last positive (-0.0054). My first ablation retrained only the full ranker's setting, which flatters
the full ranker because its setting was chosen on those users; it made two features look significant. The ceiling is
retrieval: the test movie is in the top 200 for 69.2% of users.

**How does retrieval exclude what the user has already seen?**
Inside the FAISS search: `Retriever.search` passes an `IDSelectorNot(IDSelectorBatch(seen))`, so the index returns
the best n unseen movies. Filtering the top n afterwards would leave a heavy user (ml-1m has users with over 1,000
positives) with far fewer than n. A selector belongs to one query, so search is one query per call, which is the
serving path anyway. `test_a_heavy_user_still_gets_n_candidates` excludes the 300 best movies and still gets 50.

**Why is the exact index the reference, and how do you know it is exact?**
`IndexFlatIP` scores every movie. The tests compare it with a float64 brute force (ties to the smaller id), and
`test_the_retrieved_order_is_the_full_ranking_cut_at_n` shows that the retrieved order's position of the held-out
movie is exactly its rank in `evaluate`'s full ranking. So the ranker's "retrieved order" row is the two-tower model's
own metric, not a different number.

**What goes wrong when PyTorch and FAISS share a process on a Mac?**
Both wheels bundle their own libomp, and the second one to start aborts the process (OMP Error #15). My first FAISS
commit went in on that failing check, because I had piped `make check` into `tail` and the pipe's status was tail's;
the next commit fixed it and says so. `retrieval.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` on macOS before importing faiss;
Intel calls it unsupported, so the tests run FAISS against a NumPy brute force after torch has trained in the same
process. Later that day the retrieval benchmark segfaulted inside torch: it had imported faiss first, and only
torch-first works (a batch search also hung). `retrieval.py` now imports torch before faiss, and
`tests/test_openmp_order.py` runs both import orders in fresh interpreters.

**What is in the feature store, and why those?**
Per user: the last 50 positives (the user tower's input and the "last positive" ranker feature), every positive (the
history filter), positives per genre and the sum and count of the liked movies' years (ranker features); per movie,
positives (popularity). Everything the ranker reads about a user or a movie's popularity comes from these; genres and
years of movies are static catalogue data, not features.

**How is an event applied, and what if it arrives twice?**
One Lua script (`featurestore._RECORD`): `SADD` the movie to the user's seen set; if it was already there, return
without touching anything else; otherwise append to the recent list and trim it, increment the genre and year
counters and the movie's popularity. A Lua script runs atomically in Redis, so a reader never sees half an event, and
the `SADD` guard makes a redelivered event a no-op (`test_a_redelivered_event_changes_nothing`). Reads are one
MULTI/EXEC of four commands, so a user's features come from one moment.

**How do you know online features equal offline ones?**
`test_online_features_equal_offline_for_every_user` replays the processed data's events into Redis and compares every
user's features and the popularity array exactly (integers, no tolerance);
`test_ranker_features_from_redis_equal_offline_ones_bit_for_bit` carries it through to the ranker's feature vectors.
The features are computed by one function from a `UserFeatures`, whichever side produced it, so skew can only come
from the two sources disagreeing, which is what the test checks. `scripts/bench_featurestore.py` repeats the check on
all of ml-1m.

**What can still skew?**
Order. Redis appends in arrival order, the offline lists are in event-time order; a late event online becomes the most
recent one. The replay sorts, so the test can't see it. Day 4's `POST /events` has to either accept that or keep a
sorted set by time.

**How is the ranker trained without touching test?**
Its training lists are validation lists: the two-tower model trained on training positives only retrieves 200 unseen
movies per user, and the label is where the validation movie sits in that list. 80% of users (a seeded permutation)
train it, the other 20% choose the hidden size and the epoch. On test, the two-tower model retrained on training +
validation retrieves new lists, features come from training + validation, and the chosen ranker re-orders them.
`test_the_ranker_trains_on_the_fit_users_validation_labels_only` spies on what training receives.

**Why listwise?**
The ranker's only job is to order 200 candidates for one user, and the label is one movie among them, so a softmax
over the list is exactly that objective (`ranking.listwise_loss`). Lists whose held-out movie wasn't retrieved have no
label and are skipped: the ranker can't fix a retrieval miss, which is why Recall@200 is its ceiling.

## Serving and experiments (day 4)

**What happens on one request?**
`GET /recommendations/{user}` (`api.py`) maps the MovieLens id to the model's dense index and calls
`Recommender.recommend` (`serving.py`), six timed stages: one MULTI/EXEC reads the user's features and rated set from
Redis; the NumPy user tower turns their last 3 positives into a vector; FAISS's exact index returns the best 200 movies
with every rated movie (and any excluded genre) excluded inside the search; one HMGET reads those 200 movies'
popularity; the ranker scores the 200 on eight features and re-orders them; a final check raises if anything excluded
slipped through, and the top k go back with each stage's time in the `Server-Timing` header.

**Why is there no PyTorch on the request path?**
The user tower is 64×128 and 128×64 matrices and a table lookup; NumPy does it in about 12 µs. `tower.UserTower` is
`TwoTowerNet.users` written out in float32, and `tests/test_tower.py` checks every synthetic user's vector against
PyTorch's within 1e-6 and that retrieval returns the same movies from either. (torch is still imported on macOS,
before faiss, for the OpenMP reason of day 3, but never called.)

**How do you know serving computes what you evaluated?**
`scripts/serving_check.py` serves every ml-1m user through the request path, Redis included, with the protocol's
filter (positives only), reads the rank of their test movie off the full 200-candidate list, and requires
`ranking.json`'s test metrics back. They are equal to 1e-12 for both the re-ranked list (NDCG@10 0.1298) and the
retrieved order (0.1220). That only holds if the exported model, the replayed store, the NumPy tower and the online
features all match the offline ones; a test runs the same chain end to end on synthetic data through the CLI.

**Why exclude inside the search instead of filtering afterwards?**
A heavy user has rated hundreds of movies; filtering after retrieval would leave fewer than 200 candidates, or fewer
than k results. FAISS's `IDSelectorNot` skips them during the scan, so the index returns the best 200 *eligible*
movies. Genre exclusions go into the same set. The filter stage after ranking only checks, and raises on a violation:
a silent drop there would hide a bug in the exclusion.

**What is the rated set, and why "strictly before the test time"?**
The processed data keeps only positives (rating ≥ 4), but a user who rated a movie 2 stars shouldn't be shown it. So
the store has a rated set per user: positives join it in the event script, lower ratings by one SADD. `load-store`
fills it from the raw ratings dated strictly before each user's test movie: the store is the state just before the
test event, so the test movie itself is never filtered and evaluation through the serving path is still possible.

**The serving filter raises NDCG@10 from 0.1298 to 0.1415. Is that a leak?**
No future information: every rating in the set was made before the test movie. It removes competitors: movies the user
rated but didn't like, which the protocol (positives only) kept in the ranking. +0.0117 [+0.0103, +0.0131] paired. It
means the offline protocol understates what the deployed filter gets, and it's reported separately, not as the
model's metric.

**What does `POST /events` guarantee?**
A rating at or above the model's `min_rating` goes through the same Lua script as the replay: atomic (a reader sees all
of it or none) and idempotent (SADD to `seen` first; a repeat changes nothing), so a client retrying after a timeout is
safe; the response says whether it applied. A lower rating is one SADD to the rated set. Bodies are strict (`1.0`,
`true`, `"1"` and extra fields are 422s), unknown users or movies 404, Redis down 503, values Redis holds that the store
didn't write 500. Arrival order is kept, not event time: a late event becomes the most recent (the day-3 limitation).

**How does the A/B assignment work?**
SHA-256 of "<experiment>:<user>", the first 8 bytes mod 10,000 is the bucket; buckets below the treatment share get the
treatment. Stateless, so every server agrees and a user keeps their arm; another name splits independently (a
chi-square test checks both). With `serve --experiment ranker-v1:0.5` the control arm skips the ranker and the
response names the arm.

**What does the A/B simulator say?**
Its click model is synthetic (position-based; the clicked movie is the one the real user watched next), everything
else is real: both arms' lists for all 6,034 users and the next movies. Under the default model the ranker lifts
conversion 0.1963 → 0.2026 (+3.2% relative). One test over the 6,034 users measures +0.0169 [-0.0033, +0.0372],
p = 0.10: not significant. Detecting the true lift at 80% power needs 63,199 users per arm (126,398 in all, 21 times
ml-1m's users); the other two click models need 168k and 223k per arm. So an offline gain this size can't be confirmed online at this scale.

**Why trust those statistics?**
Because the simulator checks them where the truth is known: 1,000 A/A tests are significant 5.6% of the time (target
5%), tests of the computed size detect the lift 78.8% of the time (target 80%), and 95.4% of intervals cover the true
difference (target 95%), each within its Monte Carlo interval. The unit tests check the formulas by hand (the 10% →
12% textbook case needs 3,839 per arm) and the click model by brute force.

**Why an open loop and a closed loop?**
A closed loop of c clients sends the next request only when the last answer arrives, so when the server slows down
the clients slow down with it and the queueing never shows in the latencies (coordinated omission): it measures
capacity. An open loop sends on a fixed schedule whatever the server does and measures each latency from the
request's scheduled time, so time spent waiting counts: that is the latency a user sees at a given load. Above
capacity an open loop's queue grows for the whole run, so latencies climb into seconds; that is the right answer, not
a broken benchmark.

**How do you know the load generator isn't the bottleneck?**
It reports its own lateness (sent − scheduled). The first version, one asyncio + httpx process, was 0.18 ms late at
400 requests/s and 46 ms late at 600/s, where it got only 185/s out, so those numbers were about the client. The HTTP
runs now split one schedule over four spawned client processes (perf_counter is system-wide, so they share the start
time), and the lateness is in every row.

**What did you get wrong on day 4?**
I saw the open loop collapse at 600 requests/s, guessed that FAISS's 8 OpenMP threads per search were oversubscribing
the cores, set one thread, saw a short run hold, and committed it as the fix. Then alternating runs of both settings
disagreed with each other, so I checked the mechanism directly: a one-query search never starts an OpenMP worker (the
process stays at one OS thread) and costs the same CPU time either way (83-89 µs). The "before and after" had been
another session's builds sharing the laptop (load average 6 to 9). The setting was removed in a commit that says so,
and every benchmark now records the load average.

**What did measuring change?**
redis-py was parsing replies in pure Python: a 200-field HMGET took 251-343 µs against 95 µs for a PING; with hiredis
(its C parser, the `redis[hiredis]` extra, now a dependency) 162 µs. The two Redis reads are most of the request, so
that was the one optimisation worth making.

**So how fast is it?**
In-process, a request takes 749 µs at p50 and 1,329 µs at p99 (every ml-1m user, hiredis), 72% of it the two Redis
reads. Over HTTP with one uvicorn process, about 1,150 requests/s is the capacity (closed loop), and an open loop up to
1,000 requests/s is served at 2.7 ms p50 and 4.2 ms p99 from the scheduled time. More throughput needs more processes:
the GIL serialises the thread pool, so at 16 requests in flight the pipeline itself waits (8.5 ms at p50 against 0.8 ms
alone). `results/ml-1m/serving-bench.json`, taken with the load average recorded at 2.
