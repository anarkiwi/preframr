#!/usr/bin/env python

import argparse
import concurrent.futures
import io
import multiprocessing
import pandas as pd
from tqdm import tqdm
from preframr.args import add_args
from preframr.regdataset import glob_dumps
from preframr.regtokenizer import RegTokenizer


def get_tokens(name):
    tokenizer = RegTokenizer(args=None, tokens=None)
    df = pd.read_parquet(name)
    tokenizer.accumulate_tokens(df, "placeholder")
    return tokenizer.frame_tokens[0].to_parquet()


def merge_tokens(args, names, tokenizer):
    for name in tqdm(names):
        df = pd.read_parquet(name)
        df = tokenizer.merge_token_df(tokenizer.tokens, df)
        if df is not None:
            yield df


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    names = glob_dumps(
        args.reglogs, args.max_files, args.min_dump_size, args.require_pq
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count()
    ) as executor:
        results = []
        for name in names:
            results.append(executor.submit(get_tokens, name))

    tokenizer = RegTokenizer(args=args, tokens=None)
    for result in tqdm(concurrent.futures.as_completed(results)):
        tokenizer.accumulate_tokens(
            pd.read_parquet(io.BytesIO(result.result())), "placeholder"
        )
    tokenizer.tokens = tokenizer.make_tokens()
    if args.token_csv:
        tokenizer.tokens.to_csv(args.token_csv, index=False)
    tokenizer.train_tokenizer(merge_tokens(args, names, tokenizer))
    tokenizer.tkmodel.save(args.tkmodel)


if __name__ == "__main__":
    main()
