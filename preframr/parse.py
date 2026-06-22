#!/usr/bin/env python
"""Thin CLI shim around preframr.corpus.parse_corpus: builds (and caches) each tune's BACC token block array from a (.sid, subtune) manifest (sid-only codec recovery) so a later train run reads ready-made ``.blocks.npy`` files."""

import argparse

from preframr.args import add_args
from preframr.corpus import parse_corpus
from preframr.utils import get_logger


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    parse_corpus(args, get_logger("INFO"))


if __name__ == "__main__":
    main()
