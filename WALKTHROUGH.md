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
