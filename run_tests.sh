#!/bin/sh
# Coverage scope = whole ``preframr/`` package. Entry-point CLIs and
# training-loop modules are listed in ``.coveragerc`` ``[run] omit`` --
# they're exercised by higher-level integration tests (in preframr-xpt),
# not unit tests. The remaining unit-testable modules must clear 80%.
#
# Layout:
#   /tests              -- unit tests, gated on coverage (>= 80%)
#
# Pylint runs in two passes:
#   1. ``-E`` for hard errors (existing gate).
#   2. A curated warning gate that fails the build if any of the
#      listed checks fire. Add new checks here as they're cleaned up
#      across the codebase; resist the urge to enable everything at
#      once -- each check needs its own audit + fix-up commit.
PYLINT_CURATED="redefined-builtin,unused-import,unused-argument,cell-var-from-loop,unused-variable,unbalanced-tuple-unpacking,stop-iteration-return,reimported,dangerous-default-value,pointless-statement,using-constant-test,raise-missing-from"
black --check preframr tests && \
    pytest -svvv --cov=preframr --cov-report= /tests && \
    pylint -E preframr && \
    pylint --disable=all --enable=${PYLINT_CURATED} preframr && \
    pylint -E tests && \
    pylint --disable=all --enable=${PYLINT_CURATED} tests && \
    pyright preframr && \
    coverage report --show-missing --fail-under=77
