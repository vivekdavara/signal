"""abtest.py: hash assignment, the (synthetic) click model by brute force, the two-proportion statistics by hand, and
the simulator's own checks (A/A false positives, power, coverage) within Monte Carlo bounds."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from signalrec import abtest

# --- assignment ---------------------------------------------------------------------------------------------------


def test_assignment_is_a_pure_function_of_name_and_user() -> None:
    e = abtest.Experiment("ranker-v1", 0.5)
    first = [abtest.assign(e, u) for u in range(500)]
    assert first == [abtest.assign(abtest.Experiment("ranker-v1", 0.5), u) for u in range(500)]
    assert set(first) == {abtest.CONTROL, abtest.TREATMENT}


@pytest.mark.parametrize("share", [0.1, 0.5, 0.9])
def test_the_treatment_share_is_met(share: float) -> None:
    n = 40_000
    e = abtest.Experiment("balance", share)
    treated = sum(abtest.assign(e, u) == abtest.TREATMENT for u in range(n))
    # Within 4.5 standard deviations of a fair coin of that weight.
    assert abs(treated - share * n) < 4.5 * math.sqrt(n * share * (1 - share))


def test_shares_zero_and_one_put_everyone_in_one_arm() -> None:
    assert {abtest.assign(abtest.Experiment("x", 0), u) for u in range(2000)} == {abtest.CONTROL}
    assert {abtest.assign(abtest.Experiment("x", 1), u) for u in range(2000)} == {abtest.TREATMENT}


def test_two_experiments_split_independently() -> None:
    a, b = abtest.Experiment("first"), abtest.Experiment("second")
    table = np.zeros((2, 2), dtype=np.int64)
    for u in range(20_000):
        table[abtest.assign(a, u) == abtest.TREATMENT, abtest.assign(b, u) == abtest.TREATMENT] += 1
    _, p, _, _ = stats.chi2_contingency(table)
    assert p > 1e-3
    # The same name twice is the same split, not an independent one.
    assert all(abtest.assign(a, u) == abtest.assign(abtest.Experiment("first"), u) for u in range(1000))


def test_buckets_are_uniform() -> None:
    counts = np.bincount([abtest.bucket("uniform", u) % 20 for u in range(40_000)], minlength=20)
    _, p = stats.chisquare(counts)
    assert p > 1e-3


@pytest.mark.parametrize(
    "bad",
    [
        {"name": ""},
        {"name": "a b"},
        {"name": "x" * 65},
        {"treatment_share": 1.5},
        {"treatment_share": -0.1},
        {"treatment_share": True},
    ],
)
def test_bad_experiments_are_refused(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        abtest.Experiment(**{"name": "ok", "treatment_share": 0.5, **bad})  # type: ignore[arg-type]


def test_a_user_must_be_an_integer() -> None:
    for bad in (1.0, True, "1"):
        with pytest.raises(ValueError):
            abtest.bucket("x", bad)  # type: ignore[arg-type]


# --- the click model (synthetic) ------------------------------------------------------------------------------------


def test_conversion_by_brute_force() -> None:
    model = abtest.ClickModel(p_target=0.7, p_other=0.05, k=5)
    exam = [1 / math.log2(r + 1) for r in range(1, 6)]
    for rank in (0, 1, 2, 5, 6, 99):
        no_click = 1.0
        for r in range(1, 6):
            p = model.p_target if r == rank else model.p_other
            no_click *= 1 - exam[r - 1] * p
        got = model.conversion(np.array([rank], dtype=np.int64))[0]
        assert got == pytest.approx(1 - no_click, abs=1e-15)


def test_conversion_falls_as_the_next_movie_moves_down() -> None:
    model = abtest.ClickModel()
    probs = model.conversion(np.array([1, 2, 3, 5, 10, 0, 11], dtype=np.int64))
    assert all(probs[i] > probs[i + 1] for i in range(4))
    assert probs[4] > probs[5] == probs[6]  # not shown: the same, whatever the rank beyond k


def test_bad_click_models_are_refused() -> None:
    for bad in ({"p_target": 1.1}, {"p_other": -0.1}, {"k": 0}, {"p_target": True}):
        with pytest.raises(ValueError):
            abtest.ClickModel(**bad)


# --- the statistics -------------------------------------------------------------------------------------------------


def test_compare_by_hand() -> None:
    c = abtest.compare(500, 1000, 560, 1000)
    assert c.control == 0.5 and c.treatment == 0.56
    assert c.difference == pytest.approx(0.06)
    se = math.sqrt(0.5 * 0.5 / 1000 + 0.56 * 0.44 / 1000)
    assert c.ci95[0] == pytest.approx(0.06 - 1.959964 * se, abs=1e-6)
    assert c.ci95[1] == pytest.approx(0.06 + 1.959964 * se, abs=1e-6)
    pooled = 1060 / 2000
    z = 0.06 / math.sqrt(pooled * (1 - pooled) * (2 / 1000))
    assert c.p_value == pytest.approx(2 * (1 - stats.norm.cdf(z)))
    assert c.significant  # z is about 2.69


def test_compare_edges() -> None:
    same = abtest.compare(0, 10, 0, 10)
    assert same.difference == 0 and same.p_value == 1.0 and not same.significant
    for bad in ((11, 10, 0, 10), (0, 0, 0, 10), (-1, 10, 0, 10)):
        with pytest.raises(ValueError):
            abtest.compare(*bad)


def test_users_per_arm_matches_the_textbook_formula() -> None:
    # 10% → 12%: (1.96 + 0.8416)^2 × (0.09 + 0.1056) / 0.02^2 = 3,838.5.
    assert abtest.users_per_arm(0.10, 0.12) == 3839
    assert abtest.users_per_arm(0.12, 0.10) == 3839  # two-sided: the direction doesn't matter
    # Half the difference needs about four times the users (a little less here: 11% has less variance than 12%).
    assert abtest.users_per_arm(0.10, 0.11) == math.ceil((1.959964 + 0.841621) ** 2 * (0.09 + 0.11 * 0.89) / 0.01**2)
    with pytest.raises(ValueError):
        abtest.users_per_arm(0.1, 0.1)


# --- the simulator checks itself ----------------------------------------------------------------------------------


def _population(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    control = rng.uniform(0.05, 0.4, size=n)
    return control, np.clip(control + rng.normal(0.03, 0.02, size=n), 0, 1)


def test_an_aa_test_is_significant_about_five_percent_of_the_time() -> None:
    control, _ = _population()
    row = abtest.simulate(control, control, 2000, 2000, np.random.default_rng(2))
    assert row["true difference"] == 0
    assert 0.03 < row["share significant"] < 0.07  # 0.05 ± 4.1 Monte Carlo standard errors


def test_the_computed_size_has_about_eighty_percent_power_and_honest_intervals() -> None:
    control, treatment = _population()
    n = abtest.users_per_arm(float(control.mean()), float(treatment.mean()))
    row = abtest.simulate(control, treatment, n, 2000, np.random.default_rng(3))
    assert 0.76 < row["share significant"] < 0.84
    assert 0.935 < row["share of intervals covering the true difference"] < 0.965
    assert row["mean measured difference"] == pytest.approx(row["true difference"], abs=0.002)
    # A quarter of the users: about the power the formula gives for half the standard error, much less.
    small = abtest.simulate(control, treatment, n // 4, 2000, np.random.default_rng(4))
    assert 0.25 < small["share significant"] < 0.37  # Φ(2.8 / 2 − 1.96) ≈ 0.29


def test_one_experiment_splits_by_the_hash() -> None:
    users = np.arange(1, 3001, dtype=np.int64)
    p = np.full(3000, 0.3)
    e = abtest.Experiment("one", 0.5)
    c = abtest.run(e, users, p, p, np.random.default_rng(0))
    assert c.n_treatment == sum(abtest.assign(e, int(u)) == abtest.TREATMENT for u in users)
    assert c.n_control + c.n_treatment == 3000
    sure = abtest.run(e, users, np.zeros(3000), np.ones(3000), np.random.default_rng(0))
    assert sure.control == 0 and sure.treatment == 1


def test_simulate_refuses_mismatched_populations() -> None:
    with pytest.raises(ValueError):
        abtest.simulate(np.ones(3), np.ones(4), 10, 1, np.random.default_rng(0))
