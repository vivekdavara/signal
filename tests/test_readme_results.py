"""The README's results summary and day-5 tables say what the committed result files say.

Every number in them was copied from a results file; this fails when a file is regenerated and the README isn't (or
the other way round). It checks the formatting the README uses, row by row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
RESULTS = ROOT / "results" / "ml-1m"


def load(name: str) -> Any:
    return json.loads((RESULTS / name).read_text())


def section(heading: str) -> str:
    """The README from ``heading`` to the next heading of any level."""
    start = README.index(heading)
    following = [i for i in (README.find("\n#", start + len(heading)),) if i != -1]
    return README[start : following[0] if following else len(README)]


def row(label: str, within: str | None = None) -> str:
    """The table row starting with ``| label |`` (in ``within``, the whole README by default)."""
    text = README if within is None else within
    lines = [line for line in text.splitlines() if line.startswith(f"| {label} |")]
    assert len(lines) == 1, f"{len(lines)} rows labelled {label!r}"
    return lines[0]


def interval(cell: dict[str, Any], signed: bool = False) -> str:
    low, high = cell["ci95"]
    f = "{:+.4f}" if signed else "{:.4f}"
    return f"[{f.format(low)}, {f.format(high)}]"


def test_the_summary_baselines_and_retriever() -> None:
    test = load("twotower.json")["test"]
    pop, knn, two = test["popularity"]["ndcg@10"], test["item-knn"]["ndcg@10"], test["two-tower"]["ndcg@10"]
    assert f"{pop['value']:.4f} {interval(pop)}" in row("Popularity, NDCG@10")
    assert f"{knn['value']:.4f} {interval(knn)}" in row("Item-kNN (chosen on validation), NDCG@10")
    line = row("Two-tower retriever (history 3, chosen on validation), NDCG@10")
    assert f"**{two['value']:.4f}** {interval(two)}" in line
    assert f"Recall@10 {test['two-tower']['recall@10']['value']:.4f}" in line
    # The same numbers as baselines.json's, which is the day-1 file.
    baselines = load("baselines.json")["test"]
    assert baselines["popularity"]["ndcg@10"]["value"] == pop["value"]
    assert baselines["item-knn"]["ndcg@10"]["value"] == knn["value"]


def test_the_summary_untied_users_and_ranker() -> None:
    untied = load("same-second.json")["groups"]["untied"]
    diff = untied["paired differences, ndcg@10"]["two-tower (day 3) minus item-knn"]
    line = row("... of which users not tied in the test's second (4,137 of 6,034)")
    assert untied["users"] == 4137
    assert f"{untied['ndcg@10']['two-tower (day 3)']['value']:.4f}" in line
    assert f"(item-kNN {untied['ndcg@10']['item-knn']['value']:.4f})" in line
    assert f"{diff['value']:+.4f} {interval(diff, signed=True)}" in line
    ranking = load("ranking.json")
    gain = ranking["re-ranked minus retrieved order (paired, test)"]["ndcg@10"]
    line = row("Two-tower top 200 → listwise ranker, NDCG@10")
    assert f"**{ranking['test']['re-ranked']['ndcg@10']['value']:.4f}**" in line
    assert f"{gain['value']:+.4f} {interval(gain, signed=True)}" in line


def test_the_summary_serving_numbers() -> None:
    stages = load("serving-bench.json")["stages, in-process"]["hiredis"]["total (µs)"]
    line = row("Latency in-process (every user, hiredis)")
    assert f"p50 {stages['p50']:,.0f} µs, p99 {stages['p99']:,.0f} µs" in line
    check = load("serving-check.json")
    assert all(run.get("equal", True) for run in check["runs"].values())


@pytest.mark.parametrize("h", ["0", "1", "2", "3", "all"])
def test_the_cold_start_table(h: str) -> None:
    result = load("cold-start.json")
    r = result["by known positives"][h]
    held, trained = r["held out"], r["trained on"]
    paired = r["held out: two-tower minus popularity, ndcg@10 (paired)"]
    unpaired = r["two-tower ndcg@10: held out minus trained on (unpaired)"]
    line = row(h, section("### Users the model never saw"))
    cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
    assert cells[1] == f"{held['two-tower ndcg@10']['value']:.4f}"
    assert cells[2] == f"{held['popularity ndcg@10']['value']:.4f}"
    assert cells[3].replace("**", "") == f"{paired['value']:+.4f} {interval(paired, signed=True)}"
    assert cells[4] == f"{held['two-tower recall@10']['value']:.4f}"
    assert cells[5] == f"{held['popularity recall@10']['value']:.4f}"
    assert cells[6] == f"{trained['two-tower ndcg@10']['value']:.4f}"
    assert cells[7] == f"{unpaired['value']:+.4f} {interval(unpaired, signed=True)}"
    assert result["users"] == {"held out": 1231, "trained on": 4803}


def test_the_cold_start_summary_rows() -> None:
    rows = load("cold-start.json")["by known positives"]
    one, zero = rows["1"], rows["0"]
    for r, label in (
        (one, "A user the model never saw, after 1 positive (two-tower vs popularity)"),
        (zero, "... with no positives (untrained empty history vs popularity)"),
    ):
        d = r["held out: two-tower minus popularity, ndcg@10 (paired)"]
        line = row(label)
        held = r["held out"]
        assert f"{held['two-tower ndcg@10']['value']:.4f} vs {held['popularity ndcg@10']['value']:.4f}" in line
        assert f"{d['value']:+.4f} {interval(d, signed=True)}" in line
    for r in rows.values():  # "no measurable difference (every interval reaches 0)"
        low, high = r["two-tower ndcg@10: held out minus trained on (unpaired)"]["ci95"]
        assert low < 0 < high


def test_the_new_user_latency_table() -> None:
    runs = load("new-users-bench.json")["runs"]
    labels = {
        "known users": "known users (the day-4 path)",
        "new users, popular list read every time (popular_ttl 0)": (
            "new users, popular list read every time (`popular_ttl` 0)"
        ),
        "new users, popular list kept 1 s": "new users, popular list kept 1 s",
        "nobody, popular list read every time (popular_ttl 0)": "nobody, read every time",
        "nobody, popular list kept 1 s": "nobody, kept 1 s",
    }
    for name, label in labels.items():
        total, popularity = runs[name]["total (µs)"], runs[name]["stages (µs)"]["popularity"]
        cells = [c.strip().strip("*") for c in row(label).strip("|").split("|")]
        assert cells[1:4] == [f"{total[p]:,.0f}" for p in ("p50", "p95", "p99")]
        assert cells[4].split(" ")[0] == f"{popularity['p50']:,.0f}"
    kept, read = (
        runs["new users, popular list kept 1 s"],
        runs["new users, popular list read every time (popular_ttl 0)"],
    )
    line = row("A new user's request, popular list kept 1 s vs read each time")
    assert f"p50 {kept['total (µs)']['p50']:,.0f} µs vs {read['total (µs)']['p50']:,.0f} µs" in line
