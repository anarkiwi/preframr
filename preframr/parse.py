#!/usr/bin/env python

import argparse
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from preframr.args import add_args
from preframr.regdataset import glob_dumps
from preframr.reglogparser import RegLogParser
from preframr.utils import get_logger


def write_df(args, name):
    logger = get_logger("INFO")
    log_parser = RegLogParser(args, logger)
    base_name = name.replace(".dump.parquet", "")
    try:
        for i, df in enumerate(log_parser.parse(name, max_perm=99, require_pq=False)):
            pq_name = base_name + f".{i}.parquet"
            df.to_parquet(pq_name, engine="pyarrow", compression="zstd")
    except Exception as err:
        raise ValueError(f"{name}: {err}")


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    with logging_redirect_tqdm():
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=multiprocessing.cpu_count()
        ) as executor:
            futures = []
            for name in glob_dumps(
                args.reglogs, args.max_files, args.min_dump_size, require_pq=False
            ):
                futures.append(executor.submit(write_df, args, name))
            with tqdm(total=len(futures)) as t:
                for future in concurrent.futures.as_completed(futures):
                    assert not future.exception(), future.exception()
                    t.update(1)


if __name__ == "__main__":
    main()
