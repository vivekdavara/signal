"""Review D5, round 2: defects in round 1's fixes and in serve --workers. Each test fails on bb4a08d."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from signalrec.featurestore import FeatureStore, FeatureStoreError

ROOT = Path(__file__).resolve().parents[2]
BASE = 6040


def _keys(store: FeatureStore) -> tuple[str, str, str]:
    registry, count, base = store._registry_keys
    return registry, count, base


def test_register_returns_an_ordinal_registered_refuses(fresh: FeatureStore) -> None:
    """featurestore.py:127-130 (_REGISTER's HGET path) and :227-228 (_dense): the script returns an existing entry's
    ordinal without checking it against the counter, and register() passes count=None to _dense, whose docstring
    says "the script checked the count when it returned one". It didn't. Input: registry {7: 5}, count 3, base
    6040 (a counter that went back, e.g. a stale restore: HLEN 1 <= 3 passes the new check). Expected: register(7)
    refuses, as registered(7) does ("ordinal 5, outside 1..3"). Actual: register(7) returns (6044, False), and the
    next two new users get ordinals 4 and 5, so one of them shares user 7's index 6044."""
    registry, count, base = _keys(fresh)
    fresh.client.hset(registry, "7", "5")
    fresh.client.set(count, "3")
    fresh.client.set(base, str(BASE))
    with pytest.raises(FeatureStoreError):
        fresh.registered(7, BASE)  # the read path refuses this state
    with pytest.raises(FeatureStoreError):
        fresh.register(7, BASE)


def test_registered_serves_a_registry_register_refuses(fresh: FeatureStore) -> None:
    """featurestore.py:205-225 (registered) vs :124 (the script): the read checks the user's ordinal against the
    counter but not the counter against HLEN, the check register()'s script makes first (BADREGISTRY "more entries
    than its count"). Input: registry {7: 1, 8: 2}, count 1, base 6040. Expected: registered(7) refuses the
    registry, as register(7) does. Actual: registered(7) returns 6040, so GET serves user 7 from a registry that
    every event refuses."""
    registry, count, base = _keys(fresh)
    fresh.client.hset(registry, mapping={"7": "1", "8": "2"})
    fresh.client.set(count, "1")
    fresh.client.set(base, str(BASE))
    with pytest.raises(FeatureStoreError):
        fresh.register(7, BASE)  # the script refuses: 2 entries, count 1
    with pytest.raises(FeatureStoreError):
        fresh.registered(7, BASE)


def test_the_last_user_allowed_can_never_send_another_event(fresh: FeatureStore) -> None:
    """featurestore.py:116-119 (_REGISTER's BADCOUNT check, before the HGET at :127): a count of 999999999999999 (15
    digits) passes, so INCR allocates ordinal 10^15 and writes count = 10^15 (16 digits). From then on every
    register() call, including an existing user's second event, fails BADCOUNT, while registered() still serves
    them. Input: count 999999999999999, base 6040, no entries; register(7) twice. Expected: the second call
    returns (the same index, False). Actual: FeatureStoreError "BADCOUNT ... is not a count below 10^15"."""
    _, count, base = _keys(fresh)
    fresh.client.set(count, "999999999999999")
    fresh.client.set(base, str(BASE))
    dense, created = fresh.register(7, BASE)
    assert created and dense == BASE + 10**15 - 1
    assert fresh.registered(7, BASE) == dense
    assert fresh.register(7, BASE) == (dense, False)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_serve_workers_with_a_bad_namespace_respawns_forever() -> None:
    """cli.py:376-389 (_serve, N workers): the parent checks only the model ("Fail here, once"); the store is first
    built in each worker's app_from_env, where FeatureStore's ValueError (or redis.from_url's, for a bad --redis)
    escapes uvicorn's factory call (it turns only TypeError into STARTUP_FAILURE). A worker that exits 1, not 3,
    is respawned by uvicorn's supervisor every 0.5 s, so the command never ends. Input: serve ml-1m --workers 2
    --namespace 'bad:ns'. Expected: it exits non-zero (as --workers 1 does, at once). Actual: still running after
    30 s, a worker traceback every half second."""
    if not (ROOT / "models" / "ml-1m" / "model.json").is_file():
        pytest.skip("no exported ml-1m model")
    command = [sys.executable, "-m", "signalrec.cli", "serve", "ml-1m", "--port", str(_free_port())]
    command += ["--redis", "redis://127.0.0.1:6383/0", "--namespace", "bad:ns", "--workers", "2"]
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    proc = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True
    )
    try:
        _, err = proc.communicate(timeout=30)
        exited = True
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        _, err = proc.communicate(timeout=10)
        exited = False
    failures = err.decode(errors="replace").count("namespace must be")
    assert exited and proc.returncode != 0, f"still running after 30 s; {failures} worker failures logged"
