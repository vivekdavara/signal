"""A/B tests: assignment, and a simulator that measures a lift and asks how many users a real test would need
(DESIGN.md, "Day 4: serving and experiments").

**Assignment.** A user's arm is a pure function of (experiment name, user id): the first 8 bytes of
SHA-256("<name>:<user>") as an integer, modulo 10,000, is the user's bucket, and buckets below
``treatment_share`` × 10,000 get the treatment. No state, so every server agrees and a user keeps their arm across
requests; a different name gives an independent split, so one experiment's arms don't line up with another's.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

CONTROL = "control"
TREATMENT = "treatment"
BUCKETS = 10_000
_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")


@dataclass(frozen=True)
class Experiment:
    name: str
    treatment_share: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError(f"an experiment name is 1-64 of letters, digits, '_', '.', '-'; got {self.name!r}")
        share = self.treatment_share
        if isinstance(share, bool) or not isinstance(share, int | float) or not 0 <= share <= 1:
            raise ValueError(f"treatment_share must be a number in [0, 1], got {share!r}")


def bucket(experiment: str, user: int) -> int:
    """The user's bucket in [0, BUCKETS) for this experiment."""
    if isinstance(user, bool) or not isinstance(user, int):
        raise ValueError(f"user must be an integer, got {user!r}")
    digest = hashlib.sha256(f"{experiment}:{user}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % BUCKETS


def assign(experiment: Experiment, user: int) -> str:
    return TREATMENT if bucket(experiment.name, user) < round(experiment.treatment_share * BUCKETS) else CONTROL


# --- the simulator ------------------------------------------------------------------------------------------------
#
# **The user model is SYNTHETIC**, a position-based click model: a user is shown the top ``k`` movies; slot r (1-based)
# is looked at with probability 1 / log2(r + 1), and a movie looked at is clicked with probability ``p_target`` if it is
# the movie the user went on to watch (their held-out test movie, a real outcome) and ``p_other`` otherwise. A session
# converts if anything is clicked. So two arms differ only in where they put the movie each real user watched next,
# and every number the simulator prints is about this model, not about how people click.
#
# The statistics, on the other hand, are the real ones a test would use: each arm's conversion rate, the difference
# with a 95% interval (normal approximation for two independent proportions), a two-sided p-value, and the users per
# arm for 80% power. Because the model is known, the simulator can check those statistics: the interval must cover
# the true difference about 95% of the time, an A/A test must be "significant" about 5% of the time, and a test of the
# computed size must detect the lift about 80% of the time.


@dataclass(frozen=True)
class ClickModel:
    p_target: float = 0.9
    p_other: float = 0.02
    k: int = 10

    def __post_init__(self) -> None:
        for name in ("p_target", "p_other"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a probability, got {value!r}")
        if type(self.k) is not int or self.k < 1:
            raise ValueError(f"k must be an integer >= 1, got {self.k!r}")

    def examined(self) -> npt.NDArray[np.float64]:
        out: npt.NDArray[np.float64] = 1.0 / np.log2(np.arange(2, self.k + 2, dtype=np.float64))
        return out

    def conversion(self, target_ranks: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Each user's probability of clicking anything, from the 1-based rank of their next movie in the list shown
        (a rank above ``k``, or 0, means it wasn't shown)."""
        ranks = np.asarray(target_ranks, dtype=np.int64)
        exam = self.examined()
        no_click_other = 1.0 - exam * self.p_other  # per slot, when the slot holds another movie
        all_other = float(np.prod(no_click_other))
        shown = (ranks >= 1) & (ranks <= self.k)
        slot = np.clip(ranks, 1, self.k) - 1
        with_target = all_other / no_click_other[slot] * (1.0 - exam[slot] * self.p_target)
        out: npt.NDArray[np.float64] = 1.0 - np.where(shown, with_target, all_other)
        return out


@dataclass(frozen=True)
class Comparison:
    """Treatment minus control: conversion rates, their difference, its 95% interval, and a two-sided p-value."""

    control: float
    treatment: float
    n_control: int
    n_treatment: int
    difference: float
    ci95: tuple[float, float]
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def compare(conversions_control: int, n_control: int, conversions_treatment: int, n_treatment: int) -> Comparison:
    """The usual two-proportion comparison: the interval uses each arm's own variance, the test the pooled one."""
    for count, n in ((conversions_control, n_control), (conversions_treatment, n_treatment)):
        if n < 1 or not 0 <= count <= n:
            raise ValueError(f"{count} conversions of {n} users")
    pc, pt = conversions_control / n_control, conversions_treatment / n_treatment
    diff = pt - pc
    se = math.sqrt(pc * (1 - pc) / n_control + pt * (1 - pt) / n_treatment)
    z = float(stats.norm.ppf(0.975))
    pooled = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    p_value = 1.0 if se_pooled == 0 else float(2 * stats.norm.sf(abs(diff) / se_pooled))
    return Comparison(pc, pt, n_control, n_treatment, diff, (diff - z * se, diff + z * se), p_value)


def users_per_arm(p_control: float, p_treatment: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per arm for a two-sided test at ``alpha`` to detect ``p_treatment - p_control`` with ``power``
    (the normal approximation, unpooled variances)."""
    delta = p_treatment - p_control
    if delta == 0:
        raise ValueError("no difference to detect")
    z = float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
    variance = p_control * (1 - p_control) + p_treatment * (1 - p_treatment)
    return math.ceil(z * z * variance / (delta * delta))


def run(
    experiment: Experiment,
    user_ids: npt.NDArray[np.int64],
    p_control: npt.NDArray[np.float64],
    p_treatment: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> Comparison:
    """One experiment: each user is assigned by ``assign``, then converts with their arm's probability."""
    treated = np.array([assign(experiment, int(u)) == TREATMENT for u in user_ids.tolist()], dtype=bool)
    converted = rng.random(len(user_ids)) < np.where(treated, p_treatment, p_control)
    n_t = int(treated.sum())
    return compare(int(converted[~treated].sum()), len(user_ids) - n_t, int(converted[treated].sum()), n_t)


def simulate(
    p_control: npt.NDArray[np.float64],
    p_treatment: npt.NDArray[np.float64],
    n_per_arm: int,
    experiments: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """``experiments`` independent tests with ``n_per_arm`` users in each arm, each user drawn with replacement from
    the population (so the population's mean is the true rate). Returns how often the test was significant, how often
    the interval covered the true difference, and the mean measured difference.

    Users are split by a random draw, not by ``assign``'s hash: the hash is a uniform split (the tests check it), and
    hashing millions of simulated users would dominate the run."""
    if len(p_control) != len(p_treatment) or len(p_control) == 0:
        raise ValueError("the arms need the same, non-empty population")
    true_difference = float(p_treatment.mean() - p_control.mean())
    significant = covered = 0
    measured = 0.0
    for _ in range(experiments):
        c = int((rng.random(n_per_arm) < p_control[rng.integers(0, len(p_control), n_per_arm)]).sum())
        t = int((rng.random(n_per_arm) < p_treatment[rng.integers(0, len(p_treatment), n_per_arm)]).sum())
        result = compare(c, n_per_arm, t, n_per_arm)
        significant += result.significant
        covered += result.ci95[0] <= true_difference <= result.ci95[1]
        measured += result.difference
    return {
        "experiments": experiments,
        "users per arm": n_per_arm,
        "true difference": true_difference,
        "share significant": significant / experiments,
        "share of intervals covering the true difference": covered / experiments,
        "mean measured difference": measured / experiments,
    }
