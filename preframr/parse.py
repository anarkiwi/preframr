#!/usr/bin/env python

import argparse
import concurrent.futures
import multiprocessing
from torchtune.utils import get_logger
from tqdm import tqdm
from preframr.args import add_args
from preframr.regdataset import glob_dumps
from preframr.reglogparser import RegLogParser


def write_df(log_parser, name):
    base_name = name.replace(".dump.zst", "")
    for i, df in enumerate(log_parser.parse(name, max_perm=99)):
        pq_name = base_name + f".{i}.parquet"
        df.to_parquet(pq_name, engine="pyarrow", compression="zstd")


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    logger = get_logger("INFO")
    log_parser = RegLogParser(logger)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count()
    ) as executor:
        for name in tqdm(glob_dumps(args.reglogs, args.max_files, args.min_dump_size)):
            executor.submit(write_df(log_parser, name))


if __name__ == "__main__":
    main()
