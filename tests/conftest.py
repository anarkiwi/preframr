"""Shared fixtures: resolve a Hubbard "Monty" (.sid, .dump) pair for the BACC
recover/render integration tests. The pair is heavy + tune-specific, so tests
that need it skip cleanly when it is not present locally (CI mounts /scratch)."""

import os

import pytest

_CANDIDATE_SIDS = (
    "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid",
)
_CANDIDATE_DUMPS = (
    "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.1.dump.parquet",
)


@pytest.fixture(scope="session")
def monty_pair():
    sid = next((p for p in _CANDIDATE_SIDS if os.path.exists(p)), None)
    dump = next((p for p in _CANDIDATE_DUMPS if os.path.exists(p)), None)
    if not sid or not dump:
        pytest.skip("Monty (.sid, .dump) pair not available locally")
    return sid, dump
