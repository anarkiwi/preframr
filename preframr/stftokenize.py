#!/usr/bin/env python

import argparse
from preframr.args import add_args
from preframr.regdataset import RegDataset
from preframr.utils import get_logger


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    dataset = RegDataset(args, logger=logger)
    dataset.preload()


if __name__ == "__main__":
    main()
