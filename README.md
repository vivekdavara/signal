# Signal

A movie recommender with online serving, built in five days on MovieLens: baselines (popularity, item-kNN), a
two-tower retriever trained with in-batch negatives, FAISS retrieval, a ranking stage, a Redis feature store for
online features, and a FastAPI service (retrieve → rank → filter) with a latency benchmark and an A/B test simulator.

Status: day 1 of 5 (design, data pipeline, baselines). See [DESIGN.md](DESIGN.md).

## How to run

```bash
make venv     # Python 3.11 virtualenv with the package and dev tools
make check    # ruff, ruff format --check, mypy --strict, pytest
```

## License

MIT for the code. MovieLens is not in this repository: it is downloaded by script, under GroupLens's own terms.
