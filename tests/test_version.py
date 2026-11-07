import importlib.metadata

import signalrec


def test_version_matches_the_package_metadata() -> None:
    assert signalrec.__version__ == importlib.metadata.version("signalrec")
