import concurrent.futures
import logging
import glob
import io
import os
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import zstandard as zstd
from preframr.reglogparser import RegLogParser, remove_voice_reg, state_df
from preframr.regtokenizer import RegTokenizer
from preframr.seq_mapper import SeqMapper, SeqMeta
from preframr.stfconstants import DUMP_SUFFIX, PARSED_SUFFIX


def glob_dumps(reglogs, max_files, min_dump_size, require_pq, seed=0):
    random.seed(seed)
    dump_files = []
    for r in reglogs.split(","):
        max_globbed = max_files - len(dump_files)
        if max_globbed <= 0:
            break
        globbed = [
            f
            for f in glob.glob(r, recursive=True)
            if (os.path.getsize(f) >= min_dump_size)
            and (not require_pq or glob.glob(f.replace(DUMP_SUFFIX, PARSED_SUFFIX)))
        ]
        random.shuffle(globbed)
        dump_files.extend(globbed[:max_globbed])
    random.seed()
    return dump_files


def parser_worker(args, logger, dump_file, max_perm):
    reg_log_parser = RegLogParser(args, logger)
    dfs = [
        df.to_parquet()
        for df in reg_log_parser.parse(
            dump_file, max_perm=max_perm, require_pq=args.require_pq
        )
    ]
    return dump_file, dfs


def get_prompt(args, dataset, logger):
    seq, seq_meta = dataset.getseq(args.start_seq)
    if args.start_n is None:
        start = random.randint(0, len(seq))
    else:
        start = args.start_n
    logger.info("starting at %u / %u, irq %u", start, len(seq), seq_meta.irq)
    n = args.max_seq_len - args.prompt_seq_len
    if n <= 0:
        raise ValueError("max seq length too short")
    prompt = seq[start:][: args.prompt_seq_len].unsqueeze(0).long()
    prompt_compare = seq[start:][: args.max_seq_len]
    preamble_df, _reg_widths = remove_voice_reg(
        state_df(dataset.tokenizer.decode(seq[:start].numpy()), dataset, seq_meta.irq),
        dataset.reg_widths,
    )
    reg_start = {
        r: preamble_df[preamble_df["reg"] == r]["val"].iat[-1]
        for r in preamble_df["reg"].unique()
        if r >= 0
    }
    return seq_meta.irq, n, prompt, prompt_compare, reg_start


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, args, logger=logging):
        self.args = args
        self.logger = logger
        self.n_vocab = 0
        self.n_words = 0
        self.reg_widths = {}
        self.seq_mapper = SeqMapper(args.seq_len)
        self.tokenizer = RegTokenizer(args, tokens=None)

    def load_dfs(self, reglogs, max_perm=99, encode=True):
        dump_files = glob_dumps(
            reglogs,
            self.args.max_files,
            self.args.min_dump_size,
            self.args.require_pq,
            seed=0,
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=2,
        ) as executor:
            futures = [
                executor.submit(
                    parser_worker, self.args, self.logger, dump_file, max_perm
                )
                for dump_file in dump_files
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                dump_file, df_strs = future.result()
                for df_str in df_strs:
                    df = pd.read_parquet(io.BytesIO(df_str))
                    seq = None
                    if self.tokenizer.tokens is not None:
                        df = self.tokenizer.merge_token_df(self.tokenizer.tokens, df)
                        if encode:
                            seq = self.tokenizer.encode(
                                df["n"].astype(np.int16).to_numpy()
                            ).astype(np.int16)
                            if len(seq) < self.args.seq_len:
                                self.logger.info(
                                    "rejecting sequence from %s too short %u",
                                    dump_file,
                                    len(seq),
                                )
                                break
                    irq = df["irq"].iloc[0]
                    yield dump_file, df, seq, irq

    def make_tokens(self, reglogs):
        for df_file, df, _seq, _irq in self.load_dfs(
            reglogs, max_perm=self.args.max_perm
        ):
            self.tokenizer.accumulate_tokens(df, df_file)
        tokens = self.tokenizer.make_tokens()
        self.tokenizer.tokens = tokens
        assert self.tokenizer.tokens[tokens["val"].isna()].empty, tokens[
            tokens["val"].isna()
        ]
        # assert self.tokenizer.tokens[tokens["val"] < 0].empty

    def preload(self, tokens=None, tkmodel=None):
        if tokens is not None:
            self.tokenizer.load(tkmodel, tokens)
            return
        self.logger.info("making tokens")
        self.make_tokens(self.args.reglogs)
        if self.args.token_csv:
            self.logger.info("writing tokens to %s", self.args.token_csv)
            self.tokenizer.tokens.to_csv(self.args.token_csv, index=False)

        def worker():
            df_files = []
            dataset_csv = self.args.dataset_csv
            if dataset_csv:
                self.logger.info("writing dataset to %s", dataset_csv)
            else:
                dataset_csv = "/dev/null"
            df_map_csv = self.args.df_map_csv
            with zstd.open(dataset_csv, "w") as f:
                for i, (df_file, df, _seq, _irq) in enumerate(
                    self.load_dfs(
                        self.args.reglogs, max_perm=self.args.max_perm, encode=False
                    )
                ):
                    df_files.append(df_file)
                    df["i"] = int(i)
                    df.to_csv(f, index=False, header=(i == 0))
                    yield df
            if df_map_csv:
                with open(df_map_csv, "w") as f:
                    f.write("dump_file\n")
                    f.write("\n".join(df_files))

        if self.args.tkvocab:
            self.tokenizer.train_tokenizer(worker())
        else:
            for _df in worker():
                continue

    def load(self):
        self.logger.info("loading data")
        assert self.tokenizer.tokens is not None
        reglogs = self.args.reglogs
        if self.args.reglog:
            reglogs = self.args.reglog
        self.n_vocab = len(self.tokenizer.tokens["n"])
        if self.args.tkvocab:
            self.n_vocab = self.args.tkvocab
        self.n_words = 0
        n_seq = 0
        n_words = 0
        reg_max = {}
        for _df_file, df, seq, irq in self.load_dfs(
            reglogs, max_perm=self.args.max_perm
        ):
            seq_meta = SeqMeta(irq=irq)
            self.seq_mapper.add(seq, seq_meta)
            reg_max = self.tokenizer.get_reg_max(df, reg_max)
            self.n_words += seq.size
            n_words += df.size
            n_seq += 1
        self.seq_mapper.finalize()
        self.reg_widths = self.tokenizer.get_reg_width_from_max(reg_max)
        self.logger.info(
            f"n_vocab: {self.n_vocab}, n_words {n_words}, n_encoded_words {self.n_words}, reg widths {sorted(self.reg_widths.items())}, {n_seq} sequences"
        )
        self.seq_mapper.shuffle()

    def __len__(self):
        return len(self.seq_mapper)

    def __getitem__(self, index):
        return self.seq_mapper[index]

    def getseq(self, i):
        seq, seq_meta = self.seq_mapper.seqs[i]
        return torch.from_numpy(seq), seq_meta


def get_loader(args, dataset):
    dataset.load()
    if args.shuffle:
        length = args.shuffle / 1.0
        sampler = torch.utils.data.RandomSampler(
            dataset, num_samples=int(length * len(dataset))
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        pin_memory=True,
        batch_size=args.batch_size,
    )
