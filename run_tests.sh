#!/bin/sh
black --check preframr tests && pytest -svvv --cov=preframr.reglogparser --cov=preframr.seq_mapper --cov=preframr.sidwav --cov=preframr.stftokenize --cov=preframr.train_worker --cov-report=term-missing /tests && pylint -E preframr
