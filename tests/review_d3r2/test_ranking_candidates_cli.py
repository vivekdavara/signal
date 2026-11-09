"""Review D3 round 2: ``run`` now refuses fewer than 10 candidates; the CLI still accepts them."""

from __future__ import annotations

import argparse

import pytest

from signalrec import cli, experiment_ranking


def test_the_cli_refuses_the_candidate_counts_run_refuses() -> None:
    """Defect (low severity): ``experiment_ranking.run`` now raises ValueError for n < PRIMARY_K (10), but
    ``signalrec ranking --candidates`` is still parsed with ``_positive_int``, and ``cli.main`` doesn't catch
    ValueError: ``--candidates 5`` is accepted, the dataset and the two-tower configuration are loaded, and then the
    command dies with a traceback instead of a usage error.

    Input: ``signalrec ranking ml-1m --candidates 5`` through the parser alone.
    Expected: argparse refuses it (SystemExit 2), as it refuses 0. Actual: parsed, candidates = 5.
    """
    assert experiment_ranking.PRIMARY_K == 10
    with pytest.raises(ValueError):
        experiment_ranking._check_inputs((0,), 1, 5, 0)  # what run() does with it
    with pytest.raises(SystemExit) as refused:
        args: argparse.Namespace = cli.build_parser().parse_args(["ranking", "ml-1m", "--candidates", "5"])
        print(args.candidates)
    assert refused.value.code == 2
