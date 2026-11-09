"""Review D2 round 3: ``scripts/recency_ablation.py`` reads item-kNN's settings from the results file unchecked."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from signalrec import cli, prepare

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "recency_ablation.py"
sys.path.insert(0, str(SCRIPT.parent))
import recency_ablation  # noqa: E402


@pytest.mark.parametrize("k", [True, 2.7])
def test_a_non_integer_item_knn_k_in_the_results_file_is_refused(tmp_path: Path, k: object) -> None:
    """``ItemKNN(k=int(knn_params["k"]))`` coerces whatever the file says, so ``"k": true`` runs item-kNN with k = 1
    and ``"k": 2.7`` with k = 2, the corruption round 2 refused for ``"temperature": true`` and ItemKNN's own k."""
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "60", "--items", "90"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    source = tmp_path / "twotower.json"
    args = ["twotower", "synthetic", "--data-dir", str(tmp_path), "--grid", "0.1:on", "--epochs", "1", "--seeds", "0"]
    assert cli.main([*args, "--knn-grid", "5:0", "--quiet", "--out", str(source)]) == 0
    result = json.loads(source.read_text())
    result["validation"]["item-knn"]["k"] = k
    source.write_text(json.dumps(result))
    out = tmp_path / "recency.json"
    argv = ["synthetic", "--data-dir", str(tmp_path), "--from", str(source), "--out", str(out)]
    try:
        code = recency_ablation.main(argv)
    except (ValueError, TypeError, prepare.PrepareError):
        return
    assert code != 0, json.loads(out.read_text())["item-knn"]
