#!/bin/sh
# Coverage scope = whole ``preframr/`` package. Entry-point CLIs and
# training-loop modules are listed in ``.coveragerc`` ``[run] omit`` --
# they're exercised by the integration tests, not unit tests. The
# remaining unit-testable modules must collectively clear 80%.
black --check preframr tests && \
    pytest -svvv --cov=preframr --cov-report=term-missing --cov-fail-under=80 /tests && \
    pylint -E preframr
