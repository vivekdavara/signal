PY := .venv/bin/python

.PHONY: venv lint test check

venv:
	python3.11 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e '.[dev]'

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

test:
	$(PY) -m pytest

check: lint test

SIGNALREC := .venv/bin/signalrec
DATASETS := ml-latest-small ml-1m

.PHONY: data results

# Downloads both MovieLens archives from GroupLens (checked against their pinned SHA-256) and prepares them.
data:
	for d in $(DATASETS); do $(SIGNALREC) download $$d && $(SIGNALREC) prepare $$d || exit 1; done

# Every number in the README's results: the baselines and the leakage on both datasets, then the ml-1m benchmark
# (about 2 minutes on an M3; take the benchmark on a quiet machine); then day 2's two-tower grid, what the leak is worth
# to it (10 and 20 buckets) and the recency ablation (about 55 minutes on an M3), all into the day-2 files; then day
# 3's grid (history length crossed with temperature, about 30 minutes), the ranker on its candidates, and the retrieval
# and feature-store benchmarks (take those on a quiet machine; the feature store needs Redis on 6383).
DAY2_GRID := 0.05:on,0.05:off,0.1:on,0.1:off,0.2:on,0.2:off
# Day 4: the served model, the store (Redis on 6383), the serving check, the latency benchmark (FAISS on its default 8
# threads, then on one: about 3 minutes each, on a quiet machine) and the A/B simulation.
BENCH_SERVING := --rates 100,200,400,600,800,1000 --concurrency 1,4,16

results: data
	for d in $(DATASETS); do $(SIGNALREC) baselines $$d && $(SIGNALREC) leakage $$d || exit 1; done
	$(SIGNALREC) baselines ml-1m --grid 10:0 --out results/ml-1m/post-hoc-item-knn-k10.json
	$(PY) scripts/bench_pipeline.py ml-1m --repeats 5 --out results/ml-1m/bench.json
	for d in $(DATASETS); do \
	  $(SIGNALREC) twotower $$d --grid $(DAY2_GRID) --histories 50 --quiet --out results/$$d/twotower-day2.json && \
	  $(SIGNALREC) leakage-twotower $$d --from results/$$d/twotower-day2.json || exit 1; done
	$(SIGNALREC) leakage-twotower ml-1m --from results/ml-1m/twotower-day2.json --buckets 20 \
	  --out results/ml-1m/leakage-twotower-20-buckets.json
	$(PY) scripts/recency_ablation.py ml-1m --from results/ml-1m/twotower-day2.json \
	  --out results/ml-1m/recency-ablation.json
	for d in $(DATASETS); do $(SIGNALREC) twotower $$d --quiet && $(SIGNALREC) ranking $$d --quiet || exit 1; done
	$(PY) scripts/bench_retrieval.py ml-1m --out results/ml-1m/retrieval.json
	$(PY) scripts/bench_featurestore.py ml-1m --out results/ml-1m/featurestore.json
	$(SIGNALREC) export ml-1m
	$(SIGNALREC) load-store ml-1m --fresh
	$(PY) scripts/serving_check.py ml-1m
	$(PY) scripts/bench_serving.py ml-1m $(BENCH_SERVING) --search-threads 0 \
	  --out results/ml-1m/serving-bench-faiss-8-threads.json
	$(PY) scripts/bench_serving.py ml-1m $(BENCH_SERVING) --out results/ml-1m/serving-bench.json
	$(PY) scripts/ab_simulate.py ml-1m
