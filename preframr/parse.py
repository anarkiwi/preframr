#!/usr/bin/env python
"""Thin CLI shim around preframr_tokens.parse_runner.parse_corpus."""

import argparse

from preframr_tokens import parse_corpus

from preframr.args import add_args, apply_macro_flags_to_args
from preframr.utils import get_logger


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    apply_macro_flags_to_args(args)
    parse_corpus(args, get_logger("INFO"))


if __name__ == "__main__":
    main()
