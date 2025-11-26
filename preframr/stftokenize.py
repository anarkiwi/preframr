#!/usr/bin/env python

import argparse
import concurrent.futures
import io
import multiprocessing
import pandas as pd
from preframr.args import add_args
from preframr.regdataset import glob_dumps
from preframr.regtokenizer import RegTokenizer


def get_tokens(name):
    tokenizer = RegTokenizer(args=None, tokens=None)
    df = pd.read_parquet(name)
    token_df = tokenizer._filter_tokens(df)
    return token_df.to_parquet()


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    max_files = args.max_files
    min_dump_size = args.min_dump_size
    reglogs = args.reglogs
    reglogs = "/scratch/preframr/training-dumps//MUSICIANS/G/Goto80/*parquet"
    args.tkmodel = "/tmp/tk"
    names = glob_dumps(reglogs, max_files, min_dump_size)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count()
    ) as executor:
        results = []
        for name in names:
            results.append(executor.submit(get_tokens, name))

    tokenizer = RegTokenizer(args=args, tokens=None)
    tokenizer.tokens = tokenizer._make_tokens(
        [
            pd.read_parquet(io.BytesIO(result.result()))
            for result in concurrent.futures.as_completed(results)
        ]
    )
    dfs = [pd.read_parquet(name) for name in names]
    dfs = tokenizer.merge_tokens(tokenizer.tokens, dfs)
    tokenizer.train_tokenizer(dfs)


if __name__ == "__main__":
    main()
