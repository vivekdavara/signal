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
# (about 2 minutes on an M3; take the benchmark on a quiet machine).
results: data
	for d in $(DATASETS); do $(SIGNALREC) baselines $$d && $(SIGNALREC) leakage $$d || exit 1; done
	$(SIGNALREC) baselines ml-1m --grid 10:0 --out results/ml-1m/post-hoc-item-knn-k10.json
	$(PY) scripts/bench_pipeline.py ml-1m --repeats 5 --out results/ml-1m/bench.json
