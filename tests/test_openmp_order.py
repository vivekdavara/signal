"""macOS: PyTorch and FAISS in one process, whichever the caller imports first (each case in a fresh interpreter).

Importing faiss before torch made torch's first parallel region segfault (scripts/bench_retrieval.py did, on day 3);
retrieval.py now imports torch before faiss. Each case trains a tiny two-tower model and then retrieves with it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

PROGRAM = """
import numpy as np
{imports}
p_items = 300
seq = twotower.Sequences(np.random.default_rng(0).integers(0, p_items, 4000).astype(np.int64),
                         np.arange(0, 4001, 40, dtype=np.int64))
genres = np.zeros((p_items, 0), dtype=np.float32)
config = twotower.Config(dim=16, hidden=16, epochs=2, batch_size=128, max_history=5)
net, _ = twotower.train(seq, genres, config)
emb = twotower.embed(net, seq, 5)
r = retrieval.Retriever(emb.items, retrieval.IndexSpec("hnsw", m=8))
for u in range(seq.n_users):
    ids, _ = r.search(emb.users[u], 50, seq.items[seq.starts[u]:seq.starts[u + 1]])
    assert len(ids) == 50
print("ok")
"""


@pytest.mark.parametrize(
    "imports",
    [
        "from signalrec import retrieval\nfrom signalrec import twotower",  # faiss's module first
        "from signalrec import twotower\nfrom signalrec import retrieval",
    ],
    ids=["retrieval-first", "twotower-first"],
)
def test_training_then_retrieval_in_one_process(imports: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", PROGRAM.format(imports=imports)], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
    assert result.stdout.strip() == "ok"
