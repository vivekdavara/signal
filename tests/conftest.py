from __future__ import annotations

import os
from pathlib import Path

import pytest

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def real_raw_dir(name: str) -> Path:
    """``data/raw/<name>`` if it has been downloaded; otherwise skip, or fail when CI says the data must be there.

    ``SIGNALREC_REQUIRE_DATA`` is a comma-separated list of dataset names (CI's data job sets ``ml-latest-small``).
    """
    raw = RAW / name
    if (raw / ".source-sha256").is_file():
        return raw
    if name in os.environ.get("SIGNALREC_REQUIRE_DATA", "").split(","):
        pytest.fail(f"{raw} is missing and SIGNALREC_REQUIRE_DATA names {name}")
    pytest.skip(f"{raw} not downloaded (signalrec download {name})")
