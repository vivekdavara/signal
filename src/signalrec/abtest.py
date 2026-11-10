"""A/B tests: assignment, and a simulator that measures a lift and asks how many users a real test would need
(DESIGN.md, "Day 4: serving and experiments").

**Assignment.** A user's arm is a pure function of (experiment name, user id): the first 8 bytes of
SHA-256("<name>:<user>") as an integer, modulo 10,000, is the user's bucket, and buckets below
``treatment_share`` × 10,000 get the treatment. No state, so every server agrees and a user keeps their arm across
requests; a different name gives an independent split, so one experiment's arms don't line up with another's.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

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
