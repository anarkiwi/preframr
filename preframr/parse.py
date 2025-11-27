#!/usr/bin/env python

import argparse
import concurrent.futures
import multiprocessing
from torchtune.utils import get_logger
from tqdm import tqdm
from preframr.args import add_args
from preframr.regdataset import glob_dumps
from preframr.reglogparser import RegLogParser


def write_df(args, name):
    logger = get_logger("INFO")
    log_parser = RegLogParser(args, logger)
    base_name = name.replace(".dump.zst", "")
    for i, df in enumerate(log_parser.parse(name, max_perm=99)):
        pq_name = base_name + f".{i}.parquet"
        df.to_parquet(pq_name, engine="pyarrow", compression="zstd")


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count()
    ) as executor:
        futures = []
        for name in glob_dumps(args.reglogs, args.max_files, args.min_dump_size):
            futures.append(executor.submit(write_df, args, name))
        for future in tqdm(concurrent.futures.as_completed(futures)):
            assert not future.exception(), future.exception()


if __name__ == "__main__":
    main()
