#!/usr/bin/env python

import argparse
import concurrent.futures
import io
import multiprocessing
import pandas as pd
from tqdm import tqdm
import numpy as np
from preframr.args import add_args
from preframr.regdataset import glob_dumps
from preframr.regtokenizer import RegTokenizer


def get_tokens(name):
    tokenizer = RegTokenizer(args=None, tokens=None)
    df = pd.read_parquet(name)
    token_df = tokenizer._filter_tokens(df)
    return token_df.to_parquet()


def merge_tokens(args, names, results):
    tokenizer = RegTokenizer(args=args, tokens=None)
    tokenizer.tokens = tokenizer._make_tokens(
        [
            pd.read_parquet(io.BytesIO(result.result()))
            for result in concurrent.futures.as_completed(results)
        ]
    )
    for name in tqdm(names):
        df = pd.read_parquet(name)
        try:
            irq = df["irq"].iloc[0]
        except KeyError:
            continue
        if irq < args.min_irq or irq > args.max_irq:
            continue
        if len(df) < args.seq_len:
            continue
        vol = sorted(np.bitwise_and(df[df["reg"] == 24]["val"], 15).unique().tolist())
        if len(vol) >= 8:
            continue
        df = tokenizer.merge_token_df(tokenizer.tokens, df)
        if df is not None:
            yield df


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    names = glob_dumps(args.reglogs, args.max_files, args.min_dump_size)

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
            for result in tqdm(concurrent.futures.as_completed(results))
        ]
    )
    tokenizer.train_tokenizer(merge_tokens(args, names, results))


if __name__ == "__main__":
    main()
