#!/bin/sh
black --check preframr tests && pytest -svvv /tests && pylint -E preframr
