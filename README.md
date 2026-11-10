# Signal

A movie recommender with online serving, built in five days on MovieLens: baselines (popularity, item-kNN), a
two-tower retriever trained with in-batch negatives, FAISS retrieval, a ranking stage, a Redis feature store for
online features, and a FastAPI service (retrieve → rank → filter) with a latency benchmark and an A/B test simulator.

**Status: day 4 of 5** (day 1: design, data pipeline, evaluation protocol, baselines; day 2: the two-tower retriever,
chosen and reported by the same protocol, and what the split's time leak is worth to it; day 3: the history length
chosen on validation, FAISS retrieval, a ranking stage, and a Redis feature store whose features equal the offline
ones; day 4: the serving API, which reproduces the offline test metrics exactly, a latency benchmark, and an A/B test
simulator). [DESIGN.md](DESIGN.md) is
the specification for all five days. CI runs on every push (see "Day 1 notes").

## Architecture

```mermaid
flowchart LR
  subgraph offline["Offline (days 1-3)"]
    GL[(GroupLens<br/>MovieLens zips)] -->|pinned SHA-256| DL[download]
    DL --> PR[prepare<br/>positives, 5-core,<br/>leave-last-out]
    PR --> NPY[(data/processed<br/>.npy + manifest)]
    NPY --> BL[baselines<br/>popularity, item-kNN]
    NPY --> TT[two-tower<br/>training]
    TT --> IDX[(FAISS index)]
    TT --> RK[ranker<br/>training]
    BL --> EV[evaluation<br/>full ranking,<br/>bootstrap CIs]
    TT --> EV
  end
  subgraph online["Online (days 3-4)"]
    EV2[POST /events] --> FS
    C([client]) --> API[FastAPI :8116]
    API --> FS[(Redis :6383<br/>feature store)]
    API --> UT[user tower<br/>numpy]
    UT --> IDX
    IDX --> RANK[rank]
    RANK --> FIL[filter]
    FIL --> C
  end
```

Built: the whole offline side and the whole online side. A request reads the user's features and rated set from Redis
(one MULTI/EXEC), runs the user tower in NumPy, retrieves 200 eligible movies from the exact FAISS index (rated movies
and excluded genres excluded inside the search), reads their popularity (one HMGET), ranks them, checks the filter and
returns the top k.

## How to run

```bash
make venv                                  # Python 3.11 virtualenv with the package and dev tools
make check                                 # ruff, ruff format --check, mypy --strict, pytest
.venv/bin/signalrec download ml-1m         # from GroupLens, checked against its pinned SHA-256
.venv/bin/signalrec prepare ml-1m          # data/processed/ml-1m/
.venv/bin/signalrec baselines ml-1m        # results/ml-1m/baselines.json and a table
.venv/bin/signalrec leakage ml-1m          # results/ml-1m/leakage.json
.venv/bin/signalrec recommend ml-1m --user 1   # one user's recent likes and item-kNN's top 10, by title
.venv/bin/signalrec twotower ml-1m         # results/ml-1m/twotower.json (day 3's grid: about 30 minutes here)
.venv/bin/signalrec leakage-twotower ml-1m # results/ml-1m/leakage-twotower.json (needs twotower.json first)
.venv/bin/python scripts/recency_ablation.py ml-1m --out results/ml-1m/recency-ablation.json
.venv/bin/signalrec ranking ml-1m          # results/ml-1m/ranking.json: FAISS top 200 -> ranker (needs twotower.json)
.venv/bin/python scripts/bench_retrieval.py ml-1m --out results/ml-1m/retrieval.json
redis-server --port 6383 --daemonize yes --save "" --appendonly no   # the feature store's Redis (tests skip without)
.venv/bin/python scripts/bench_featurestore.py ml-1m --out results/ml-1m/featurestore.json
# Day 4: the served model, the store, the API, and the measurements behind "Results (day 4)"
.venv/bin/signalrec export ml-1m           # models/ml-1m/ (gitignored): two-tower on training + validation + ranker
.venv/bin/signalrec load-store ml-1m --fresh   # positives and earlier ratings into Redis (about 15 s)
.venv/bin/python scripts/serving_check.py ml-1m   # the request path must give ranking.json's test metrics back
.venv/bin/signalrec serve ml-1m &          # http://127.0.0.1:8116 (--experiment ranker-v1:0.5 for an A/B split)
curl -s 'http://127.0.0.1:8116/recommendations/1?k=5'
curl -s -X POST http://127.0.0.1:8116/events -H 'content-type: application/json' \
  -d '{"user": 1, "movie": 364, "rating": 5}'
kill %1
.venv/bin/python scripts/bench_serving.py ml-1m --rates 100,200,400,600,800,1000 --concurrency 1,4,16 \
  --out results/ml-1m/serving-bench.json  # about 3 minutes; on a quiet machine
.venv/bin/python scripts/ab_simulate.py ml-1m   # results/ml-1m/abtest.json (the click model is SYNTHETIC)
redis-cli -p 6383 shutdown nosave
```

`make results` reruns every command behind the numbers below, day 2's into the `-day2` files.

PyTorch runs on the CPU (`pip install torch` pulls in the right wheel on a Mac; on Linux, install the CPU-only wheel
from `https://download.pytorch.org/whl/cpu` first, as CI does).

For example, ml-1m user 1's last five likes are Disney and Pixar animation (The Hunchback of Notre Dame, Antz, A
Bug's Life, Hercules, Mulan); item-kNN (k=3) puts Sleeping Beauty and Pinocchio in its top 10 next to broad favourites
such as The Shawshank Redemption; popularity's top 10 has no animation at all. The held-out test movie is Pocahontas,
which neither list contains.

`make data` downloads and prepares both datasets; `make results` regenerates every number in the results below
(the benchmark wants a quiet machine). `ml-latest-small` works the same way as ml-1m. `signalrec synthetic` writes a
**synthetic** dataset in ml-latest-small's file format (`signalrec prepare synthetic`, `signalrec baselines
synthetic`); the tests use it, and no number from it is a claim about real users.

## Data

| Dataset | Ratings | Users | Movies | After the pipeline (users × movies, training positives) |
|---|---|---|---|---|
| ml-latest-small | 100,836 | 610 | 9,742 | 601 × 1,955, 40,025 |
| ml-1m | 1,000,209 | 6,040 | 3,883 listed, 3,706 rated | 6,034 × 3,125, 562,308 |

Counts after the pipeline from `signalrec prepare <dataset>` (also pinned by a test). **No MovieLens data is in this
repository**: the ml-1m licence forbids redistribution without permission, so the code downloads it from GroupLens,
and CI does the same for ml-latest-small without uploading anything derived from it. Research use only; GroupLens
doesn't endorse this project. Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM TiiS 5, 4: 19:1-19:19. <https://doi.org/10.1145/2827872>

The pipeline (DESIGN.md, "From ratings to implicit feedback"): a rating of 4 or more is a positive; users and movies
with fewer than 5 positives are dropped until both bounds hold at once; each user's last positive by (time, movie id)
is the test item and the one before it the validation item.

## Evaluation protocol

Every model ranks **every** movie for every user (no sampled negatives), with the user's history removed. Choices are
made on the validation item with the training positives as history; test numbers come from the model refit on
training + validation. With one held-out item, Recall@K is the share of users whose item is in their top K and
NDCG@K averages `1/log2(rank+1)` inside the cut-off. Ranks are exact counts with ties broken by movie index, so they
don't depend on a sort. Intervals are 95% percentile bootstraps over users (1,000 resamples, seed 20261107); model
differences use a paired bootstrap on the same resampled users.

## Results (day 1)

Measured on 2026-11-07 (Apple M3, macOS 26.6, Python 3.11.12, NumPy 2.4.6, SciPy 1.17.1).

**ml-1m**, test, 6,034 users (`signalrec baselines ml-1m`, `results/ml-1m/baselines.json`):

| Model | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Coverage@10 |
|---|---|---|---|---|---|
| popularity | 0.0389 [0.0343, 0.0441] | 0.0193 [0.0166, 0.0221] | 0.1294 [0.1210, 0.1387] | 0.0385 [0.0355, 0.0417] | 0.037 |
| item-kNN (k=3, shrink=10) | 0.0630 [0.0568, 0.0689] | 0.0314 [0.0280, 0.0349] | 0.1845 [0.1743, 0.1946] | 0.0581 [0.0544, 0.0618] | 0.213 |

Item-kNN minus popularity (paired): NDCG@10 +0.0122 [0.0085, 0.0157], Recall@50 +0.0550 [0.0444, 0.0653].

**ml-latest-small**, test, 601 users (`signalrec baselines ml-latest-small`, `results/ml-latest-small/baselines.json`):

| Model | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Coverage@10 |
|---|---|---|---|---|---|
| popularity | 0.0449 [0.0283, 0.0616] | 0.0235 [0.0146, 0.0328] | 0.1414 [0.1131, 0.1697] | 0.0441 [0.0339, 0.0542] | 0.031 |
| item-kNN (k=2, shrink=0) | 0.0765 [0.0549, 0.0982] | 0.0360 [0.0252, 0.0470] | 0.1780 [0.1464, 0.2063] | 0.0584 [0.0462, 0.0703] | 0.384 |

Item-kNN minus popularity (paired): NDCG@10 +0.0125 [-0.0003, 0.0256]. On 601 users the interval includes zero:
this dataset can't separate the two baselines at NDCG@10, which is why ml-1m carries the reported results.

What the numbers say:

- **NDCG@10 is flat across small neighbourhoods; Recall@50 wants more.** Validation NDCG@10 on ml-1m peaks at k=3
  (0.0386), with k=10 at 0.0359 and k=100 at 0.0325, so the protocol picks k=3. After the protocol's run I also
  scored k=10 (shrink 0) on test, labelled post hoc (`signalrec baselines ml-1m --grid 10:0 --out
  results/ml-1m/post-hoc-item-knn-k10.json`): NDCG@10 0.0322 [0.0286, 0.0355], no worse than k=3's 0.0314, and
  Recall@50 0.2158 against k=3's 0.1845. So validation's preference for k=3 at NDCG@10 didn't carry over to test
  (the gap is inside the intervals), while the Recall@50 cost of tuning for NDCG@10 is real. The reported number
  stays k=3's: switching to k=10 after seeing test would be tuning on test. The JSON now says how firm each choice
  is: for every grid row, the paired validation interval of (chosen - row) NDCG@10. On ml-1m, k=3 is
  indistinguishable from 7 of the other 29 rows (k=2 to 5) and only just ahead of k=10 ([0.0000, 0.0054]); on
  ml-latest-small it is indistinguishable from 19 of 29, so that dataset's choice is noise.
- **Popularity is a strong floor and a narrow one.** Its top 10 lists cover 3.7% of the catalogue; item-kNN's cover
  21%. A day-2 model has to beat item-kNN's 0.0314, not popularity's 0.0193.
- **These are low absolute numbers by design**: one held-out item ranked against about 3,000 movies, with no sampled
  negatives. Papers that rank against 100 sampled negatives report much larger numbers; they aren't comparable.

The grid (`experiment.DEFAULT_GRID`) is k ∈ {1, 2, 3, 5, 10, 25, 50, 100, 200, 400} × shrink ∈ {0, 10, 50}; it starts
at 1 because the first runs, with the grid starting at 10, then 5, then 2, chose its smallest value each time. Every
validation row is in the JSON. The baselines JSON also records fit and evaluation times from runs taken while other
jobs loaded the machine; use the benchmark above instead.

### How much does the split leak? (`signalrec leakage <dataset>`, `results/<dataset>/leakage.json`)

A per-user split lets a model train on positives that happened after a user's test item. On ml-1m the leak is
large in exposure: for the average user, **44.2%** of all training positives are later than their test item (52% on
ml-latest-small), and 99.98% of users have at least one. What is it worth? Popularity is the model most exposed, so
`signalrec leakage` ranks each test item under popularity counted only from positives strictly *before* that user's
test time and compares it, paired, with the usual popularity:

| ml-1m, test, 6,034 users | Global popularity (leaky) | As of each test time | Difference [95% CI] |
|---|---|---|---|
| Recall@10 | 0.0389 | 0.0384 | +0.0005 [-0.0013, +0.0025] |
| NDCG@10 | 0.0193 | 0.0190 | +0.0002 [-0.0006, +0.0010] |
| Recall@50 | 0.1294 | 0.1251 | +0.0043 [+0.0013, +0.0073] |
| NDCG@50 | 0.0385 | 0.0375 | +0.0010 [+0.0003, +0.0018] |

So at the top of the list the leak is worth nothing measurable to popularity, and deeper in the list it is real but
small (3% of Recall@50). The two counts differ in one more way: the as-of count includes other users' earlier test
items, which the global count never sees. On ml-latest-small every difference's interval includes zero.

### Pipeline cost (`python scripts/bench_pipeline.py ml-1m --repeats 5 --out results/ml-1m/bench.json`)

ml-1m, median of 5 runs, load average 2.1-2.3, no sleep during any run (each run's load and sleep are in the JSON).
The power source wasn't recorded: the Mac was on AC at 15:52 and on battery by 17:21 (Low Power Mode off).

| Stage | Median | Min |
|---|---|---|
| parse (`movielens.load`, 1,000,209 ratings) | 1.423 s | 1.422 s |
| prepare (positives, 5-core, split) | 0.243 s | 0.240 s |
| item-kNN fit (k=3, shrink=10, block 512) | 0.664 s | 0.662 s |
| item-kNN evaluate (6,034 users × 3,125 movies, full ranking) | 0.217 s | 0.215 s |
| popularity evaluate (same) | 0.476 s | 0.475 s |

The similarity build's peak traced memory by column block (`tracemalloc`, which sees NumPy's arrays): 27.0 MiB at
128 columns, 90.9 MiB at 512, 297.4 MiB at 2,048 and 451.5 MiB with all 3,125 movies in one block: linear in the
block, as DESIGN.md says, so a catalogue too big for one block still fits by shrinking it. Popularity evaluates more
slowly than item-kNN because of `top_k`'s stable sort (35.4 ms against 13.2 ms per 512 users, timed separately):
item-kNN's scores are mostly equal zeros (74 distinct values in user 0's row against popularity's 682), which the
sort runs through quickly. Neither matters at this size.

## Results (day 2): the two-tower retriever

Measured on 2026-11-08 (Apple M3, macOS 26.6, Python 3.11.12, PyTorch 2.15.0 on the CPU with 4 threads, NumPy
2.4.6), on AC power, with a review agent running alongside, so the recorded seconds are not benchmarks.

**ml-1m**, test, 6,034 users (`signalrec twotower ml-1m` at day 2's commit `c7ca9bb`,
`results/ml-1m/twotower-day2.json`; the baseline rows are the same models as day 1's, refit in the same run):

| Model | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Coverage@10 |
|---|---|---|---|---|---|
| popularity | 0.0389 [0.0343, 0.0441] | 0.0193 [0.0166, 0.0221] | 0.1294 [0.1210, 0.1387] | 0.0385 [0.0355, 0.0417] | 0.037 |
| item-kNN (k=3, shrink=10) | 0.0630 [0.0568, 0.0689] | 0.0314 [0.0280, 0.0349] | 0.1845 [0.1743, 0.1946] | 0.0581 [0.0544, 0.0618] | 0.213 |
| two-tower (τ=0.1, logQ on, 17 epochs, seed 0) | 0.1361 [0.1279, 0.1448] | 0.0713 [0.0663, 0.0766] | 0.3533 [0.3421, 0.3651] | 0.1186 [0.1134, 0.1238] | 0.626 |

Two-tower minus item-kNN (paired, test): NDCG@10 +0.0398 [0.0345, 0.0453], Recall@50 +0.1689 [0.1551, 0.1836].
The rule DESIGN.md set before any training, a paired *validation* interval above zero, holds: +0.0414 [+0.0352,
+0.0469]. Seeds 0, 1 and 2 of the same configuration score NDCG@10 0.0713, 0.0706 and 0.0693 on test: the spread
over seeds (0.002) is a twentieth of the gap to item-kNN.

**ml-latest-small**, test, 601 users (`signalrec twotower ml-latest-small`): two-tower (τ=0.1, logQ on, 10 epochs)
NDCG@10 0.0523 [0.0392, 0.0674] against item-kNN's 0.0360; paired +0.0163 [+0.0009, +0.0319], just above zero.
Seeds give 0.0523, 0.0455 and 0.0526: on 601 users the seed spread is about half the gap.

What the validation grid says (every configuration's learning curve and every candidate's paired interval are in the
JSON):

| ml-1m validation | τ=0.05 | τ=0.1 | τ=0.2 |
|---|---|---|---|
| logQ on: best NDCG@10 (epoch) | 0.0777 (5) | **0.0799 (17)** | 0.0717 (12) |
| logQ off: best NDCG@10 (epoch) | 0.0440 (11) | 0.0496 (11) | 0.0436 (15) |
| logQ on: coverage@10 at that epoch | 0.71 | 0.62 | 0.48 |
| logQ off: coverage@10 at that epoch | 1.00 | 1.00 | 0.99 |

- **logQ correction matters more than anything else in the grid.** Without it, every temperature loses about 0.03
  NDCG@10 (40%; still above item-kNN's 0.0386 on validation), and the top-10 lists cover the entire catalogue:
  in-batch negatives are popularity-weighted, so the uncorrected model learns to push popular movies down, and
  recommends the long tail to everyone.
- **The choice is firm in configuration and soft in epochs.** τ=0.05 with logQ is indistinguishable from the chosen
  τ=0.1 (paired interval [-0.0025, +0.0072]); everything without logQ, and τ=0.2, is clearly worse. 27 of the other
  119 (configuration, epoch) candidates are indistinguishable from the chosen one, all of them logQ-on at τ 0.05 or
  0.1 from epoch 5 on: past epoch 5 the curve is flat within noise.

### Where does the lead come from? (`python scripts/recency_ablation.py ml-1m`, `results/ml-1m/recency-ablation.json`)

The user tower averages the last 50 positives; item-kNN sums over all of them. Post hoc, on validation only (the
test items are never read), item-kNN was scored from each user's last *n* positives (similarities still from all the
data) and the reported two-tower configuration was retrained with *n*-item histories (same τ, epochs and seed):

| History | item-kNN NDCG@10 | two-tower NDCG@10 | two-tower Recall@50 |
|---|---|---|---|
| last 1 | 0.0411 | 0.1126 | 0.3941 |
| last 5 | **0.0599** | **0.1233** | **0.4899** |
| last 10 | 0.0554 | 0.1090 | 0.4652 |
| last 20 | 0.0481 | 0.0889 | 0.4211 |
| last 50 (two-tower as reported) | 0.0402 | 0.0799 | 0.3870 |
| last 200 | 0.0386 | 0.0744 | 0.3742 |
| all (item-kNN as reported) | 0.0386 | - | - |

Recency is worth a lot to both: scoring item-kNN from the last 5 positives alone lifts it from 0.0386 to 0.0599
(paired +0.0213 [+0.0171, +0.0255]), half of its gap to the reported two-tower model. But at *equal* history length
the two-tower model still leads by about two to one (0.1233 against 0.0599 at five), so the lead is not only recency.
And the reported model's 50-item history, a fixed setting from DESIGN.md, is far from the best: five items score
0.1233 against 0.0799 on validation. It wasn't switched today, because this is one seed of a post-hoc look; day 3 makes
the history length a grid dimension so the protocol chooses it. (724 of 6,034 users have more than 200 training
positives, up to 1,415; the two-tower model wasn't run on whole histories because a batch of 1,024 such histories
would gather hundreds of MB per step.)

### What is the leak worth to the two-tower model? (`signalrec leakage-twotower <dataset>`)

Day 1 found the split's calendar-time leak worth almost nothing to popularity. A learned model could use it more. A
model can't be retrained per user, so users are cut into 10 buckets of about 603 by test time, and for each bucket the
reported configuration (same τ, epochs, seed) is retrained on the training + validation positives before the bucket's
earliest test time, then compared paired with the reported model on test (`results/<dataset>/leakage-twotower.json`).

The first version of this check halved the model, and the reason was not the leak. The cut drops everything after
it, including each bucket's own users' positives between the cut and their test item, which are their own past. Even
the last bucket, trained on 97% of the examples, fell from 0.0335 to 0.0186. So each bucket also trains a second model
that keeps its own users' training and validation positives whatever their time:

| ml-1m, test NDCG@10 (6,034 users) | Value | Leaky minus it [95% CI] |
|---|---|---|
| reported model (trained on all training + validation) | 0.0713 | |
| as of each bucket's cut, everything after it dropped | 0.0354 | +0.0359 [+0.0310, +0.0412] |
| as of the cut, the bucket's own histories kept | 0.0712 | +0.0001 [-0.0048, +0.0054] |

So the two estimates bracket the leak's worth, roughly: between nothing measurable and half the model. Review round 3
showed that the second one leaks too, and not a little: it keeps a bucket-mate's positives later than a user's own
test item, and in ml-1m's first bucket only 468 of its 43,520 training examples are from before the cut. What the
pair can't say how much of the first estimate's drop was the leak and how much was users' own recent past. A
check on the code: the last bucket's own-histories model trains on exactly the reported model's 562,308 examples and scores
exactly its 0.0335.

Smaller buckets narrow both errors. With 20 buckets of about 302 users (`signalrec leakage-twotower ml-1m --buckets
20 --out results/ml-1m/leakage-twotower-20-buckets.json`, 19:27-19:45), the strict model scores 0.0368 (+0.0345
[+0.0296, +0.0396]) and the own-histories model 0.0692 (+0.0020 [-0.0028, +0.0069]); at Recall@50 the own-histories
difference, +0.0220 [+0.0109, +0.0336], now excludes zero (10 buckets: +0.0071 [-0.0035, +0.0177]). So as the
within-bucket leak shrinks, the lower end of the bracket rises: the leak is worth something to the two-tower model
deeper in the list, as it was to popularity (day 1: Recall@50 +0.0043), and little at the top. On **ml-latest-small** (601 users over 22 years, 60 per bucket) a gap remains even with own
histories kept: NDCG@10 0.0523 against 0.0308, +0.0215 [+0.0094, +0.0356]. With so few users per bucket, an early
bucket's model learns from very little, so that gap mixes the leak with plain data volume; this run can't separate
them.

## Results (day 3): history length, retrieval, ranking

Measured on 2026-11-09 (Apple M3, macOS 26.6, Python 3.11.12, PyTorch 2.15.0, faiss-cpu 1.15.1, Redis 8.8.0), on
battery, with a review agent or another run alongside the training runs, so their recorded seconds are not
benchmarks. Every number is in the JSON named next to it.

### The history length, chosen on validation (`signalrec twotower <dataset>`, `results/<dataset>/twotower.json`)

Day 2 fixed the user tower's input at the last 50 positives and its recency ablation found 5 better; day 3 puts the
length in the grid (1, 3, 5, 10, 20, 50 × τ 0.05, 0.1, 0.2, logQ on, 20 epochs each: 360 candidates). On ml-1m
validation (best NDCG@10 per configuration; the chosen one in bold; every row's paired interval is in the JSON):

| ml-1m validation | history 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| τ=0.05 | 0.1127 | 0.1328 | 0.1230 | 0.1078 | 0.0901 | 0.0777 |
| τ=0.1 | 0.1137 | **0.1357** | 0.1261 | 0.1110 | 0.0939 | 0.0799 |
| τ=0.2 | 0.1099 | 0.1205 | 0.1135 | 0.0968 | 0.0835 | 0.0717 |

Chosen: τ=0.1, 3 positives, 8 epochs. Only τ=0.05 with 3 positives is indistinguishable (paired [-0.0008, +0.0063]);
every length of 10 or more loses with an interval starting above +0.018. 13 of the other 359 candidates are
indistinguishable. Test, 6,034 users:

| Model | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Coverage@10 |
|---|---|---|---|---|---|
| popularity | 0.0389 [0.0343, 0.0441] | 0.0193 [0.0166, 0.0221] | 0.1294 [0.1210, 0.1387] | 0.0385 [0.0355, 0.0417] | 0.037 |
| item-kNN (k=3, shrink=10) | 0.0630 [0.0568, 0.0689] | 0.0314 [0.0280, 0.0349] | 0.1845 [0.1743, 0.1946] | 0.0581 [0.0544, 0.0618] | 0.213 |
| two-tower, day 2 (history 50) | 0.1361 [0.1279, 0.1448] | 0.0713 [0.0663, 0.0766] | 0.3533 [0.3421, 0.3651] | 0.1186 [0.1134, 0.1238] | 0.626 |
| two-tower, day 3 (history 3, τ=0.1, 8 epochs) | 0.2176 [0.2067, 0.2285] | 0.1220 [0.1155, 0.1286] | 0.4581 [0.4458, 0.4717] | 0.1747 [0.1683, 0.1808] | 0.632 |

Seeds 0, 1, 2: 0.1220, 0.1250, 0.1239. Two-tower minus item-kNN (paired): validation +0.0971 [+0.0903, +0.1048],
test +0.0906 [+0.0839, +0.0975]. **ml-latest-small** chose τ=0.1, 3 positives, 14 epochs: test NDCG@10 0.0609
[0.0466, 0.0768] (day 2: 0.0523; item-kNN 0.0360), and 9 of the other 17 configurations are indistinguishable on
its 601 users. Day 2's grid run through today's code (`--grid` of day 2's six pairs, `--histories 50`) reproduces
`twotower-day2.json` exactly on ml-latest-small: every learning-curve point, the test metrics and all three seeds.

### How much of that is the same-second order? (`python scripts/same_second.py ml-1m`, `results/ml-1m/same-second.json`)

A jump from 0.0713 to 0.1220 by reading fewer positives needed a check. The split orders a user's positives by
(timestamp, movie id), and ml-1m users rated dozens of movies per second: for **31.4%** of users (1,897) the test
movie has the same timestamp as the validation movie, so which one is "next" was decided by movie id, not by the
user. A model reading the last three positives can learn from that order. Post hoc, on test, NDCG@10 by group:

| Users | day 3 (history 3) | day 2 (history 50) | item-kNN | popularity | day 3 − day 2 (paired) |
|---|---|---|---|---|---|
| tied (1,897) | 0.2046 | 0.1004 | 0.0409 | 0.0182 | +0.1042 [+0.0901, +0.1175] |
| untied (4,137) | 0.0841 | 0.0579 | 0.0271 | 0.0198 | +0.0262 [+0.0191, +0.0328] |
| all (6,034) | 0.1220 | 0.0713 | 0.0314 | 0.0193 | +0.0507 [+0.0435, +0.0573] |

The shorter history is better for untied users too, and the two-tower model's lead over item-kNN survives there
(+0.0570 [+0.0492, +0.0643]); but about two thirds of the overall gain (1,897 × 0.1042 of 6,034 × 0.0507) comes from
the tied third, where every model scores higher and the order inside the second is partly an artefact of the split.
The reported number stays the protocol's (the split was fixed on day 1, before any model); the untied row is the
better guide to what the model would do on a user's genuinely next movie. Day 2's leakage numbers
(`leakage-twotower*.json`) are about day 2's model, whose configuration each file records.

### Retrieval → ranking (`signalrec ranking <dataset>`, `results/<dataset>/ranking.json`)

The day-3 two-tower model retrieves 200 unseen movies per user with the exact FAISS index (the retrieved order is
exactly its full-ranking order: a test); a ranker re-orders them. On ml-1m the validation movie is among the 200 for
72.7% of users and the test movie for 69.2%: the ranker's ceiling. Chosen on the 1,207 selection users (20% of the
validation users; the other 80% train it): a 64-unit hidden layer, 20 epochs (linear 0.1392, 16 units 0.1399, 64 units
0.1419 against the retrieved order's 0.1344; chosen minus retrieved +0.0075 [+0.0007, +0.0143]). Test:

| ml-1m test | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Recall@200 |
|---|---|---|---|---|---|
| retrieved order | 0.2176 [0.2067, 0.2285] | 0.1220 [0.1155, 0.1286] | 0.4581 [0.4458, 0.4717] | 0.1747 [0.1683, 0.1808] | 0.6924 |
| re-ranked | 0.2314 [0.2206, 0.2423] | 0.1298 [0.1231, 0.1364] | 0.4708 [0.4576, 0.4838] | 0.1822 [0.1757, 0.1884] | 0.6924 |
| re-ranked − retrieved (paired) | +0.0138 [+0.0083, +0.0192] | +0.0078 [+0.0050, +0.0105] | +0.0128 [+0.0063, +0.0191] | +0.0076 [+0.0051, +0.0100] | 0 |

A small, real gain: 6% relative on NDCG@10. The validation evidence for it is thin: the run before the review's
retrieval fix (float32 retrieval scores, otherwise the same; `b41e283`) chose 15 epochs and its selection interval
was [-0.0004, +0.0145], reaching zero, with a test gain of +0.0072 [+0.0042, +0.0102]. Which features carry it (post
hoc, selection users; each feature removed and the ranker chosen again exactly as the full one was): **none is
distinguishable from zero on its own**. The largest drops are the similarity to the last positive (-0.0054 [-0.0122,
+0.0012]) and the retrieval score (-0.0032 [-0.0084, +0.0020]); "year unknown" is constant on ml-1m (every title has a
year). The first version retrained only the full ranker's chosen setting, which review round 2 showed biases every drop
upward (the full ranker is the best of 60 settings on these users); it had reported those two as distinguishable
(-0.0095 and -0.0055), and the symmetric comparison doesn't support that. On **ml-latest-small** the ranker doesn't help NDCG@10
(-0.0023 [-0.0122, +0.0100]; Recall@50 +0.0316 [+0.0116, +0.0532]): 481 training lists, 223 of them with the held-out
movie.

### FAISS: approximate against exact (`python scripts/bench_retrieval.py ml-1m`, `results/ml-1m/retrieval.json`)

The day-3 model retrained on training + validation; every user's 200 unseen candidates from each index against the
protocol's exact top 200 (float64, ties to the smaller index); one query per call on one FAISS thread, 200 untimed
warm-up queries, inverted-CDF percentiles. 16:20-16:23 at the final code, on battery, no sleep, nothing else
running (the 16:03 run before review round 2's fix gave the same recalls and latencies within a few µs, one p99
within 50 µs). "efSearch 200" searches with 216: the Retriever asks for n + 16 and passes max(efSearch, 216).

| ml-1m, 3,125 movies, 6,034 queries | recall of the exact top 200 | test movie in the 200 | p50 | p99 |
|---|---|---|---|---|
| flat (exact) | 1.0000 | 0.6924 | 93 µs | 143 µs |
| HNSW, M=32, efSearch 200 | 0.9966 | 0.6934 | 140 µs | 245 µs |
| HNSW, M=32, efSearch 400 | 0.9998 | 0.6922 | 239 µs | 307 µs |
| IVF, 56 cells, nprobe 8 | 0.6380 | 0.5870 | 50 µs | 97 µs |
| IVF, 56 cells, nprobe 16 | 0.8397 | 0.6546 | 57 µs | 101 µs |

At 3,125 movies the exact index is both exact and faster than HNSW: scoring every movie is 200,000 multiply-adds, less
than walking the graph. IVF is faster still but loses a third of the exact list at nprobe 8. So serving uses the exact
index. A **SYNTHETIC** catalogue of 200,000 vectors (real movie vectors plus noise, labelled so in the JSON; 2,000 real
user vectors, no exclusion) shows where that changes: flat 937 µs at p50, HNSW (efSearch 200) 261 µs at recall 0.9492
after a 32 s single-threaded build, IVF (512 cells, nprobe 32) 129 µs at recall 0.9608 after 0.5 s.

### The feature store (`python scripts/bench_featurestore.py ml-1m`, `results/ml-1m/featurestore.json`)

Local Redis 8.8 on 6383 (loopback), 16:05-16:06, on battery: replaying all 568,342 training + validation positives
through the Lua script, 1,000 per round trip, takes 8.47 s (**67,128 events/s**, best of 3); reading one user back
(one MULTI/EXEC of four commands) takes **276 µs** at p50 and 659 µs at p99 over all 6,034 users; and **0** users and 0
movies differ from the offline features. Reading the whole popularity hash takes 4.8 ms, too slow for every request:
day 4 reads only the candidates' counts or caches it.

## Results (day 4): serving, latency, A/B testing

Measured on 2026-11-10 (Apple M3, 4 performance + 4 efficiency cores, macOS 26.6, Python 3.11.12, faiss-cpu 1.15.1,
redis-py 8.1.0 with hiredis 3.5.0, Redis 8.8.0 on loopback, FastAPI 0.162.4, uvicorn 0.54.0), on AC power. Every number
is in the JSON named next to it.

### Does serving compute what was evaluated? (`python scripts/serving_check.py ml-1m`, `results/ml-1m/serving-check.json`)

`signalrec export ml-1m` (20 s) retrained the reported two-tower model on training + validation and bundled it with
`ranking.json`'s ranker; `signalrec load-store ml-1m --fresh` replayed 568,342 positives (9.8 s) and 973,546 earlier
ratings (4.3 s; 408,571 of them not already positives) into Redis, 17 MB. Then every one of the 6,034 users went
through the request path (Redis reads, NumPy user tower, FAISS, ranker), about 6.5 s per pass:

| ml-1m test, through the request path | Recall@10 | NDCG@10 | Recall@50 | Recall@200 | equal to `ranking.json` |
|---|---|---|---|---|---|
| protocol filter (positives), re-ranked | 0.2314 | 0.1298 | 0.4708 | 0.6924 | yes, to 1e-12 |
| protocol filter, retrieved order | 0.2176 | 0.1220 | 0.4581 | 0.6924 | yes, to 1e-12 |
| serving filter (every rated movie), re-ranked | 0.2504 | 0.1415 | 0.4954 | 0.7058 | (not comparable) |

So there is no training-serving skew to measure: the served lists give the offline test numbers exactly. The serving
filter, which also removes movies the user rated below 4 (all rated before the test movie), adds +0.0117 [+0.0103,
+0.0131] NDCG@10 (paired): it removes competitors the protocol kept, so the offline protocol slightly understates the
deployed system.

### Latency (`python scripts/bench_serving.py ml-1m --rates 100,200,400,600,800,1000 --concurrency 1,4,16`, `results/ml-1m/serving-bench.json`)

16:22-16:25, on AC, no sleep, load average 2.0 before and 2.1 after, nothing else above 15% CPU (checked before and
after). Client and server on the same laptop over loopback; one uvicorn process with sync endpoints on its thread pool;
k = 10 from 200 candidates; the exact FAISS index. **In-process**, every user once after 1,000 untimed requests:

| stage (µs) | p50 | p95 | p99 | max |
|---|---|---|---|---|
| features (Redis, one MULTI/EXEC) | 263 | 535 | 729 | 1,569 |
| user tower (NumPy) | 12 | 12 | 12 | 23 |
| retrieve (FAISS flat, 200, exclusions in the search) | 113 | 182 | 231 | 397 |
| popularity (Redis, one 200-field HMGET) | 274 | 283 | 292 | 332 |
| rank (8 features, 64-unit MLP) | 56 | 58 | 60 | 68 |
| filter | 31 | 41 | 44 | 47 |
| **total, with hiredis** | **749** | **1,079** | **1,329** | 2,327 |
| total, redis-py's Python parser | 947 | 1,618 | 2,093 | 4,031 |

The two Redis reads are 72% of the request. hiredis, redis-py's C parser (now a dependency), is day 4's one
optimisation: p50 947 → 749 µs, p99 2,093 → 1,329 µs; with the Python parser the two reads take 372 and 360 µs at p50.

**Over HTTP**, four client processes sharing one schedule; open-loop latencies count from each request's scheduled
time, after a 2 s paced untimed lead-in; 10 s timed per row:

| closed loop, concurrency | throughput (req/s) | p50 | p99 | server pipeline p50 |
|---|---|---|---|---|
| 1 | 575 | 1,650 µs | 2,690 µs | 794 µs |
| 4 | 1,109 | 3,560 µs | 4,600 µs | 2,032 µs |
| 16 | 1,148 | 13,749 µs | 20,502 µs | 8,460 µs |

| open loop (req/s) | served (req/s) | failed | p50 | p95 | p99 | client lateness p99 | server pipeline p50 |
|---|---|---|---|---|---|---|---|
| 100 | 100 | 0 | 4,599 µs | 5,419 µs | 5,953 µs | 1,130 µs | 1,668 µs |
| 200 | 200 | 0 | 3,611 µs | 4,192 µs | 4,557 µs | 1,111 µs | 1,187 µs |
| 400 | 400 | 0 | 2,747 µs | 3,153 µs | 4,177 µs | 1,118 µs | 795 µs |
| 600 | 600 | 0 | 2,282 µs | 2,758 µs | 3,183 µs | 664 µs | 788 µs |
| 800 | 800 | 0 | 2,392 µs | 3,079 µs | 8,339 µs | 436 µs | 835 µs |
| 1,000 | 1,000 | 0 | 2,705 µs | 3,561 µs | 4,166 µs | 285 µs | 1,096 µs |

Capacity is about 1,150 requests/s: one process, its thread pool's requests taking turns on the GIL, so more
concurrency only adds waiting (the server's own pipeline time grows from 0.8 ms at one request in flight to 8.5 ms at
16). Up to 1,000 requests/s the open loop keeps up with p99 under 9 ms. Latency *falls* as the rate rises up to 600/s:
at 100/s the server's pipeline time is twice what it is at 400/s (1,668 against 795 µs at p50) and the client wakes
about a millisecond late, both signs of a machine that idles between requests, not of load. With one request in flight
0.86 ms of each is outside the pipeline (1,650 µs end to end, 794 µs inside): FastAPI, HTTP, the thread hand-off and
the client; the open loop's medians also carry the client's own scheduling.

How these numbers were got matters as much as they do. The first runs of the day were wrong twice. One client process
(asyncio + httpx) couldn't keep its schedule above about 400/s (46 ms late at 600/s); the load is now split over four.
Then runs taken while another session's builds and tests shared the laptop (load average 6 to 9) collapsed at 400-1,000
requests/s, inconsistently; I blamed FAISS's OpenMP threads, and a direct check showed a one-query search never uses
them (DESIGN.md, "As built (day 4)"). Every result file now records the load average before and after.

### A/B test simulation (`python scripts/ab_simulate.py ml-1m`, `results/ml-1m/abtest.json`): the click model is SYNTHETIC

Control serves the retrieved order and treatment the re-ranked one (serving filter, top 10): the next movie is in the
top 10 for 23.7% and 25.0% of users. A synthetic position-based click model turns each real user's two lists into
conversion probabilities (abtest.py). 1,000 simulated tests per check, seed 0:

| click model (synthetic) | true conversion, control → treatment | true lift | one test over the 6,034 users (hash split 2,985 / 3,049) | users per arm for 80% power |
|---|---|---|---|---|
| p_target 0.9, p_other 0.02 | 0.1963 → 0.2026 | +0.0063 (+3.2%) | +0.0169 [-0.0033, +0.0372], p = 0.10 | 63,199 |
| p_target 0.5, p_other 0.02 | 0.1467 → 0.1502 | +0.0034 (+2.3%) | +0.0008 [-0.0174, +0.0189], p = 0.94 | 168,217 |
| p_target 0.9, p_other 0.1 | 0.4455 → 0.4496 | +0.0042 (+0.9%) | +0.0025 [-0.0226, +0.0275], p = 0.85 | 223,380 |

The statistics, checked where the truth is known (Monte Carlo 95% intervals in brackets): A/A tests at the dataset's
size were significant 5.6% [4.2, 7.0], 5.3% and 5.0% of the time (target 5%); tests of the computed size detected the
lift 78.8% [76.3, 81.3], 79.0% and 81.4% of the time (target 80%); their intervals covered the true difference 95.4%,
94.3% and 95.7% of the time (target 95%). At ml-1m's own size the test has 10.3% power under the first model. So the
ranker's offline gain is real but small, and confirming it online would take 126,000 to 447,000 users in all, 21 to 74 times ml-1m's
6,034; how many depends mostly on the synthetic model, which is the point of showing three.

## Day 1 notes

Facts about the real files that the strict parsers found (DESIGN.md said nothing about them before):

- ml-latest-small uses **CRLF** line endings in both files; GroupLens's README doesn't say so. ml-1m uses LF.
- ml-1m's `movies.dat` is **Latin-1**, not UTF-8: 62 bytes above 0x7f, each alone (0xe9 for "é").
- **11 titles** in ml-latest-small end in a space ("96 Minutes (2011) "); titles are stripped.
- The real files parse to GroupLens's published counts (a test, run in CI on ml-latest-small).

CI: the workflow has two jobs (lint, types and tests; then the whole pipeline on the real ml-latest-small, with the
baselines and the leakage compared with the committed results as whole documents). The first run on GitHub
(Ubuntu 24.04, x86-64, Python 3.11.18) passed both jobs: 212 tests passed and 4 skipped without data, then the real
ml-latest-small matched the committed baselines and leakage results with 0 differences on another CPU architecture.
Before the repository existed, every step was rehearsed in a fresh clone with a fresh virtualenv on macOS.

## Review of day 1

One review agent read the code at `914dcdf` (16:20-16:34, about 14 minutes, its own worktree, a correctness brief:
find where the code disagrees with DESIGN.md or itself, prove each defect with a failing test). Its tests are in
`tests/review_d1/`, each file's docstrings stating the defect; all pass now.

| Finding | Fix |
|---|---|
| **Ties decided by rounding.** 3/√27 and 1/√3 are equal but differ in a double's last bit, so item-kNN kept the wrong neighbour; float32 score sums put a tied item 2nd. | Similarities as `sqrt(co²/(n_i n_j))` with no shrink, rounded to 2^-32 and kept as integers; scores are exact integer sums (`6c8847f`). The second case is only bounded: sums of *different* square roots equal as reals can still differ by a unit, and the test pins that. |
| An unclosed quote in movies.csv swallowed the next line (a movie vanished silently); a `csv.Error` escaped with no file or line. | One CSV record per line; errors name the line (`e6766e5`). |
| ml-1m `Foo (1999):::Comedy` read as the genre `:Comedy`. | The last `::` ends the title; a title holding `::` is refused (`e6766e5`). |
| A genre with a tab saved fine and broke `load`. | Control characters refused by the parsers and by `save` (`e6766e5`, `005a9e5`). |
| `load` checked only the files the manifest listed. | The manifest must list exactly the expected files (`005a9e5`). |
| Replacing a processed directory deleted the old one first. | Old renamed aside, put back on failure (`005a9e5`; the reviewer's injection failed the put-back too, so its setup was narrowed, not its assertion). |
| A bad `--grid`, an offline download and a corrupt manifest ended in tracebacks. | argparse validates `--grid`; `OSError` from a fetch and a bad manifest are one-line errors (`247cf6a`, `005a9e5`). |

It also listed about 30 mutants no test killed. The worst: **swapping the validation items for the test items in
tuning survived every test**, so the protocol's central rule was untested. Each listed survivor now fails a test
(`d10cb54`, `03223f5`; re-applied from a copy and restored byte for byte), except `extract`'s `is_dir()` check,
which was unreachable and is removed.

A second round (one agent at `f2c825f`, 16:46-16:58, asked "what did each round-1 fix break?"; tests in
`tests/review_d1r2/`) found 8 more defects, most of them in round 1's fixes:

| Finding | Fix |
|---|---|
| `shrink < 0` let NaN through; `rint(NaN · 2^32)` as int64 is INT64_MIN, so every similarity was dropped. | Shrink must be finite and ≥ 0, also checked when an `ItemKNN` is built (`d5ed2b5`). |
| Round 1 mapped `OSError` to `DownloadError`, but `http.client.IncompleteRead` isn't an `OSError`. | Both mapped (`d5ed2b5`). |
| Round 1's replacement left an empty `.<name>-old-*` directory when the first rename failed. | Removed; a failed delete of the old copy after the swap no longer fails the save (`86f3e56`). |
| `"format": true` passed the version check (`True == 1`). | Must be an `int` (`86f3e56`). |
| A manifest without params, counts or sources was a `KeyError`; another dataset's directory was evaluated and mislabelled; counts weren't checked. | Fields checked, counts must equal the arrays', `load(directory, dataset)` (`86f3e56`). |
| `recommend --neighbours 0`, `--shrink -1` tracebacks; `--shrink nan`, `--k -1`, `--k 0` exited 0 with wrong lists. | argparse types shared with `--grid` (`7b2f7f2`). |

Its 13 surviving mutants (including the leakage report's two columns swapped, which would flip the sign of the
reported leak) each fail a test now (`d5ed2b5`, `86f3e56`, `7b2f7f2`, `c5df773`). It also pointed out that the
brute-force item-kNN test had come to mirror the implementation's formula, so a second test checks the plain float
definition independently (`694c87b`).

A third round (one agent at `c5df773`, 17:04-17:08, on round 2's fixes; `tests/review_d1r3/`) found 3: manifest
counts of the wrong type passed (`60.0 == 60`, `False == 0`; counts must be ints); `prepare` over a symlink or a
file at the target still ended in a traceback (now refused up front); and `baselines` read the manifest twice, so a
concurrent `prepare` could put new parameters beside old metrics (`load` now attaches the manifest it checked).
All fixed in `2e9f886`, its 3 survivors killed in `296bbf0`. A fourth round (17:14-17:16) looked at round 3's fixes
and at the leakage comparison I had added to CI an hour earlier: round 3's fixes held, the new comparison didn't
(NaN passed, intervals unchecked, extra keys ignored, missing keys a `KeyError`, `601.0` passing as a count).
`compare_results.py` now compares whole documents recursively instead of hand-picked fields (`e4b4fb0`).
Findings per round: 11, 8, 3, 5; every round's were mostly in code written since the round before.

My own checks found what the reviewer didn't look for: the real files' CRLF and Latin-1 (above), the grid choosing
its own edge three times, and two claims of mine that measurement contradicted: the median ml-1m history (56
training positives, not the 70 I first wrote; caught before it was committed) and the reason popularity evaluates
more slowly (the sort's work on varied scores, not ties; corrected in `f2c825f`).

## Review of day 2

Two review rounds, each one agent in its own worktree with a correctness brief, each finding a failing test (now
passing) in `tests/review_d2/` and `tests/review_d2r2/`, plus a list of mutants that no test killed.

**Round 1** (at `fd4b0f4`, about 9 minutes) found no leak, loss or selection bug, and 7 defects at the edges, fixed in
`9899a33`: `--seeds` took 2^64, which `torch.manual_seed` refuses after the whole grid had trained; `--grid` took a
repeated configuration; `run` didn't check the item-kNN grid (an empty one failed after all the training) or later
seeds; a temperature of 1e-39 trained on NaN losses; DESIGN asked for every candidate's paired interval and only each
configuration's best epoch had one; and DESIGN said every training positive is an example, which isn't true of each
user's first (no history), so DESIGN changed. Its 14 surviving mutants mattered more: **a target leaking into its own
training history**, the logQ flag inverted, the temperature or the accidental-hit mask ignored in training, genres
ignored, the user tower reading the item table, scoring without each user's last positive. Every other test
recomputed results through the same `train` and `embed`, so it couldn't see inside them.
`tests/test_twotower_internals.py` now watches from outside (spies on what each batch feeds the loss; both towers
recomputed by hand), and all 18 listed mutants fail a test (`01f22e2`; applied from a copy, restored, compared byte
for byte).

**Round 2** (at `01f22e2`, about 14 minutes) was asked what round 1's fixes broke, and reviewed the two pieces no round
had seen (`leakage_twotower.py`, `scripts/recency_ablation.py`). 5 defects, fixed in `7faa1a9` and `b9d7045`:

| Finding | Fix |
|---|---|
| Round 1's temperature bound (1/τ a finite float32) was too loose: 3e-39 is a denormal and still trained on NaN. | A floor, τ ≥ 1e-3, for `--grid` too. |
| `temperature=True` and `lr=True` passed as 1.0, so a results file saying `"temperature": true` became a configuration. | Must be an int or float, not a bool. |
| `ItemKNN` took `k=5.0`; `run` failed with a `TypeError` after the whole two-tower grid. | k and block must be integers. |
| `leakage-twotower` filled fixed settings missing from the results file with today's defaults, and took a file produced from other prepared data. | Every fixed setting required; the file's params, counts and sources must equal the manifest's. |
| The recency ablation's reference row was hard-coded as "last 50". | The reported length is always a row and the reference; `main()` tested. |

Its most useful note was a hunch, which the code confirmed: the as-of model trained on other users' *test* positives
before the cut, which the leaky model never sees, so "the as-of model knows less" wasn't true and the docstring's
bound didn't hold. The as-of model now trains on training + validation positives before the cut only, a subset of the
leaky model's data. 24 surviving mutants, including **the ablation reading the test items** (`main()` had no test);
each now fails a test (`7faa1a9`, `b9d7045`).

**Round 3** (at `eca3bd9`, about 10 minutes) checked round 2's fixes and the second as-of model. 5 defects: an
integer too large for a float raised `OverflowError` from `Config.check` (and a traceback from `leakage-twotower`);
round 2's type rule refused NumPy floats; the learning rate had no bound (`lr=1e30` trained on NaN, now at most 1);
the recency ablation coerced the results file's item-kNN `k` (`true` ran as 1, `2.7` as 2). And its hunch changed a
conclusion: the own-histories model leaks too, and not a little (above), so "the leak is worth nothing" became a
bracket. 17 surviving mutants, all now failing a test. Findings per round: 7, 5, 5, each mostly in code written
since the round before.

## Review of day 3

One review agent in its own worktree at `3afe5de` (everything up to the retrieval benchmark), with a correctness
brief: failing tests in `tests/review_d3/`, and a mutation run. I stopped it after about 30 minutes, once its tests and
mutation log were written but before its summary (it was waiting on work of its own); the findings below are from its
tests, each docstring stating the defect, and its log.

- **Retrieval** (`2901a79`). My docstring said FAISS searches HNSW with `max(efSearch, n)`; FAISS 1.15 doesn't (about
  0.65 of the exact top 200 at efSearch 16): the Retriever now sets it. FAISS's flat index returned tied movies
  largest index first, against the protocol's smaller-index rule; and in float32 a movie just below the cut can come
  out just above it (one of 149 users' held-out movie, tied with a twin at the 30th place, was pushed to 31st). The
  index is now asked for 16 extra candidates, which are rescored in float64 and sorted by (score, index) before the
  cut.
- **Ranking** (`4ccfca9`). A NaN score ranked first in `ranks_after` and last in `rerank`; non-finite scores and
  features are now errors.
- **Feature store** (`c84603a`). A namespace with a glob character made `clear()` delete other namespaces' keys;
  negative or beyond-int64 values read back as features; an event with user `3.0` or `True` was written under keys
  `user(3)` never reads.
- **Metrics above n**: the same bug I had found and fixed in `fc5c122`; its tests are kept, one adapted to the fix.
- **Mutants**: 32 applied, 18 survived. The worst: **the validation lists' popularity counting the validation
  positives**, which are the ranker's labels; the test's "share retrieved" read from the validation lists; the
  selection users' paired difference with its sign swapped; the retrieval benchmark's held-out share computed for the
  validation movie (always 0). `tests/review_d3/test_survivors_d3.py` has a test per mutant, and re-applied to
  today's code (`.scratch/mutate_d3.py`, restoring from git HEAD) all 16 still applicable are killed. The snapshot
  read survived a concurrent writer (Redis runs a client's pipelined commands from one buffer together), so a spy
  checks that it asks for MULTI/EXEC.

Then measurement found one more: **the retrieval benchmark segfaulted** inside torch, because it imported faiss
before torch (DESIGN.md, "Two OpenMP runtimes"; `a4be56b`).

**Round 2** (one agent at `b9cb71e`, about 7 minutes, asked what round 1's fixes broke; `tests/review_d3r2/`): the
float64 rescoring kept a tie group's smaller indices only if it fit in the 16-candidate margin (50 tied movies, the cut
inside them: indices 10-14 kept instead of 0-4); the search now widens until the cut is clear (`823e08c`). The HNSW
docs said `max(efSearch, n)` where the code uses n + 16 (docs fixed, its test adapted to the documented value);
`FeatureStore.user(3.0)` still read an empty user, as round 1 had fixed only writes (`1ef51b5`); `--candidates 5`
was a traceback, not a usage error (`4f66dc3`). Its 3 surviving mutants (rescoring in float32, `>=` for the int64
bound, the bool refusal dropped) are killed. Its suspicion that the feature ablation was biased (only the full ranker
was chosen on the selection users) was right: chosen alike (`73d3c67`), no feature's drop excludes zero. Re-run after
the fixes, the ranker's own results are identical and the benchmark's recalls unchanged.

## Review of day 4

One review agent in its own worktree at `c4b266d` (all of day 4's code), with a correctness brief: failing tests in
`tests/review_d4/` and a mutation run; about 18 minutes. Ten defects, all fixed (`fcf2007`):

- **Skew**: a model reading more recent positives than the store keeps (`--histories 100` against the store's 50)
  was served from half its history without a word; the Recommender refuses it.
- **Atomicity**: the event script's new `SADD rated` came last, and a Lua script isn't rolled back on an error, so a
  rated key of another type kept the rest of the event and made every retry a no-op. Every key's type is checked
  before the first write.
- **Model files**: only `model.npz` was hashed, while `model.json` carries the catalogue, the titles and
  `max_history`; `max_history` "4", 4.9 or `true` was coerced; and the arrays were read in a second read of the file,
  not from the bytes hashed. Format 2 hashes model.json's own fields, types are strict, arrays come from the hashed
  bytes.
- **API**: `/recommendations/1_0` served user 10 and `k=+5`, `5.0`, `" 5"` were accepted (FastAPI's lax integers);
  a NaN or Infinity rating made the 422 body unwritable, a 500; `/health` named the dataset, never the model.
- **A/B**: the click model divided by the target slot's no-click factor, 0/0 = NaN with `p_other = 1`.

Mutants: 58, all importable; 41 killed, 17 survived, 16 meaningful (one equivalent), each killed by
`tests/review_d4/test_survivors_d4.py`, among them seen positives dropped from the default exclusion, share 0 putting
bucket 0 in the treatment, the arms' sizes swapped in the standard error, `>=` as `>` for a positive rating, and in
the benchmark the achieved rate divided over the lead-in, `sent = due` and early sends. Two of its suspicions were
right and are fixed (`af6ce04`): failed open-loop requests were left out of the latency percentiles (they count now,
at the time they failed), and `load-store` didn't check the prepared data's parameters against the model's.

**Round 2** (one agent at `af6ce04`, about 11 minutes, asked what round 1's fixes broke; `tests/review_d4r2/`), five
defects, four of them in round 1's fixes (`b4a7a94`): leading zeros were still a second spelling of an id (`01`,
`k=05`); a hash of the right type could still fail the event script halfway (a non-integer count, or one at 2^63 - 1
overflowing an HINCRBY), so every count the script increments is checked before the first write; the replays raised a
bare Redis error (a traceback from `load-store`); `load-store` re-read the raw ratings without checking them against
the manifest's hashes; and counting a failure at the time it failed let instant errors pull the percentiles down, so
failures now rank after every success. Its three untested suspicions were real too (`70fcfa3`): an int key in a
model's `info` hashed differently after the JSON round trip, a NaN in `info` crashed export, and two 422s had a
different body shape. Mutants: 14, 8 killed, 5 meaningful survivors killed by `test_survivors_d4r2.py`, one
equivalent. After both rounds the serving check is still exact and the exported model's hash unchanged.

## Design decisions

- **Implicit feedback at rating ≥ 4.** The product wants movies a user will like, not merely watch. Low ratings are
  dropped from the training signal; the day-4 serving filter still removes every rated movie.
- **Leave-last-out by time, with movie id breaking ties.** It mirrors "what will this user like next" and makes the
  split a pure function of the data (ml-1m users often rated dozens of movies in one second). Known cost: a per-user
  split leaks calendar time across users (DESIGN.md, "Known leakage").
- **Full ranking, exact ranks.** Sampled metrics can reorder models; counting ranks (rather than sorting) makes ties
  explicit and the result independent of the sort.
- **Validation for choices, test for numbers, intervals on everything.** The test set is touched once per model.
- **Byte-deterministic processed data with a hashed manifest.** Re-running gives the same bytes; loading refuses a
  file that changed.
- **Pinned downloads and member-name extraction.** A swapped or truncated archive fails the hash; a hostile archive
  can't write outside its directory because only the expected member names are ever read.
- **A user tower over the history, no user id** (day 2). Any history works, including one that changed a minute ago
  or belongs to a user the model never saw; that is what lets day 3 serve fresh histories from Redis.
- **In-batch negatives with logQ correction** (day 2). One matrix product per batch, no sampler; logQ undoes the
  popularity bias that in-batch sampling adds (measured: worth +0.030 validation NDCG@10 at τ=0.1).
- **Epochs chosen on validation, like every other choice** (day 2). Validation is scored after every epoch, so the
  epoch count is one more hyperparameter with a paired interval, and the test run retrains for exactly that many.
- **Evaluation through exported embeddings** (day 2). The towers become two NumPy matrices and go through the same
  exact-rank code as the baselines, which is also what serving will use: no PyTorch on the request path.
- **The history length is a hyperparameter like any other** (day 3). A post-hoc ablation suggested it; the protocol
  chose it, on validation, from a grid that includes day 2's value.
- **Exclude seen movies inside the index search** (day 3), with a FAISS ID selector, rather than filtering afterwards:
  a user with 1,000 positives would otherwise get a handful of the 200 candidates the ranker expects.
- **One function computes the ranker's features from either source** (day 3). Offline features and Redis features
  are both a `UserFeatures` plus a popularity array, so skew can only come from the sources disagreeing, and a test
  compares those exactly (integers) and then the feature vectors bit for bit.
- **A Lua script per event, guarded by the seen set** (day 3). Atomic (no reader sees half an event) and idempotent
  (a redelivered event changes nothing), which an at-least-once event stream needs.
- **A listwise ranker trained on validation lists, chosen on a held-out fifth of them** (day 3). It never sees a test
  label, and the test lists are retrieved by a model that never saw the test movies.
- **The served model is the evaluated one, and a script proves it** (day 4). `export` retrains the reported
  configuration exactly as the test run did and takes the ranker from `ranking.json`; `serving_check.py` requires the
  request path, Redis included, to give the offline test metrics back to 1e-12. Skew would show up as a mismatch.
- **NumPy on the request path, PyTorch only offline** (day 4). The user tower is two small matrices; exporting it
  keeps the serving process free of autograd and lets a test pin NumPy's vectors to PyTorch's.
- **Exclusions inside the search, a check after ranking** (day 4). Rated movies and excluded genres are excluded by
  FAISS's selector, so k is always met when k eligible movies exist; the filter stage raises instead of dropping, so an
  exclusion bug can't hide.
- **Stateless hash assignment** (day 4). Every server computes the same arm from (experiment, user); no table to keep
  in sync, and a renamed experiment reshuffles independently.
- **A simulator that checks its own statistics** (day 4). The click model is synthetic, so the lift it reports is too;
  but the A/A rate, power and coverage are checked where the truth is known, which is what makes its sample-size
  answer worth reading.

## Limitations

- The per-user split's cross-user time leakage is measured, not removed: for popularity (day 1, worth almost nothing)
  and for the two-tower model (day 2, bracketed between nothing measurable and half the model's NDCG@10; above).
- The two-tower choice is the best of 120 validation candidates and item-kNN's the best of 30, on the same
  validation users that are then used to compare them, so the validation comparison slightly favours the larger
  search (review round 1 raised this). The test comparison is not affected: test is touched once per model.
- In-batch negatives include the same user's other positives when two of their examples share a batch (only the same
  *movie* is masked). That is the standard formulation and DESIGN's; it slightly penalises a user's own taste.
- Nothing makes training identical across machines (float reductions can differ by CPU and thread count; not
  measured, since only one machine was available), so CI runs the two-tower commands on real data without comparing
  their numbers. On one machine a seed gives the same weights (a test).
- Item-kNN's similarity is cosine on binary likes; no recency weighting, which the k results suggest would help.
- ml-latest-small is too small to separate the baselines at NDCG@10; it is the CI dataset, not the results dataset.
- The split breaks same-second ties by movie id, and 31.4% of ml-1m users rated their test movie in the same second
  as their validation movie; about two thirds of the 3-item history's gain over day 2 comes from those users
  (`scripts/same_second.py`). Reported as measured, with the untied numbers beside it; not fixed.
- The feature store appends events in arrival order; offline lists are in event-time order. A late event online
  becomes the most recent one. The replay sorts, so the skew test can't see it.
- On macOS, PyTorch and FAISS each bundle libomp and run side by side only with torch loaded first and under
  `KMP_DUPLICATE_LIB_OK` (DESIGN.md, "As built (day 3)"), which Intel calls unsupported; faiss-first segfaults and a
  batch search hung, so `retrieval.py` imports torch first and searches one query at a time. The tests check FAISS's
  answers in that setting.
- The ranker's gain on ml-1m (+0.0078 NDCG@10 on test) is barely established on validation: its selection-users
  interval is [+0.0007, +0.0143], and reached zero in the run before the review's float64 rescoring. No single feature
  is shown to matter (the ablation's intervals all reach zero on 1,207 selection users).
- Serving (day 4) knows only the prepared users: the store is keyed by the model's dense user index, so a new
  MovieLens id is a 404, although the user tower itself works for any history.
- One uvicorn process: its thread pool's requests take turns on the GIL, so throughput tops out near one core's worth
  (about 1,100 requests/s here) and latency grows with concurrency; more cores need more processes. Client and server
  shared one laptop in every measurement.
- The features read and the popularity read are two round trips, so a request can rank with features one event
  apart; both are counts that move by one per event.
- The A/B simulator's click model is synthetic: its lift and sample sizes are about that model. The users, their lists
  and their next movies are real.
- Similarities are kept to 2^-32, so a shrink in the billions leaves them a few grid units apart (ties take over),
  and above about 2^33 ≈ 8.6e9 times a pair's co-count rounds it to zero (checked: a co-count-1 pair survives a shrink
  of 8e9 and vanishes at 9e9). Nothing refuses such a shrink; the useful range is 0 to about 100.

## Layout

```
src/signalrec/
  download.py     pinned download and extraction
  movielens.py    strict parsers for both formats
  prepare.py      positives, k-core, split, indexing, the on-disk format
  evaluate.py     exact ranks, Recall/NDCG/coverage, bootstrap intervals
  baselines.py    popularity, item-kNN
  experiment.py   the protocol: tune on validation, report on test
  twotower.py     the two-tower model: sequences, histories, towers, loss, training, exported embeddings
  experiment_twotower.py   the two-tower model by the same protocol
  leakage.py      the split's calendar-time leakage, measured
  leakage_twotower.py      ... and what it is worth to the two-tower model
  retrieval.py    FAISS indexes (exact, HNSW, IVF) with the user's movies excluded in the search
  features.py     offline features (recent positives, seen set, genre and year counts, popularity), event replay order
  featurestore.py the same features in Redis: a Lua script per event, snapshot reads
  ranking.py      the ranker's features, model, listwise training and NumPy export
  experiment_ranking.py    retrieve -> rank by the protocol, and the post-hoc feature ablation
  recommend.py    one user's recommendations by title
  tower.py        the user tower in NumPy, and the model files serving reads (hash-checked)
  export.py       the served model (two-tower on training + validation + the chosen ranker); loading the store
  serving.py      the request path: features → user tower → retrieval → popularity → ranker → filter, timed
  api.py          FastAPI: GET /recommendations/{user}, POST /events, /health
  abtest.py       hash assignment, the synthetic click model, two-proportion statistics, power, the simulator
  synthetic.py    a synthetic dataset for tests and CI
  cli.py          signalrec download | synthetic | prepare | baselines | leakage | recommend | twotower |
                  leakage-twotower | ranking | export | load-store | serve
scripts/compare_results.py   CI's check against the committed results
scripts/bench_pipeline.py    stage timings and the similarity build's memory by block size
scripts/recency_ablation.py  validation-only: history length for item-kNN and the two-tower model
scripts/same_second.py       test, post hoc: each model's NDCG@10 for users tied and untied in the test's second
scripts/bench_retrieval.py   approximate vs exact top 200, per-query latency (real and a synthetic large catalogue)
scripts/bench_featurestore.py  replay throughput, one user's read latency, the skew check on a whole dataset
scripts/serving_check.py     every user through the request path against ranking.json's test metrics
scripts/bench_serving.py     per-stage latency in-process; HTTP closed loop and open loop at fixed rates
scripts/ab_simulate.py       a simulated A/B test of the ranker: lift, power analysis, the statistics checked
results/<dataset>/{baselines,leakage,twotower,twotower-day2,leakage-twotower,ranking}.json,
results/ml-1m/{recency-ablation,same-second,retrieval,featurestore}.json,
results/ml-1m/{serving-check,serving-bench,abtest}.json
```

## License

MIT for the code. MovieLens is not in this repository: it is downloaded by script, under GroupLens's own terms.
