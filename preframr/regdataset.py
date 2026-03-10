import concurrent.futures
import copy
import logging
import glob
import io
import multiprocessing
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
        globbed = []
        for f in glob.iglob(r, recursive=True):
            if os.path.getsize(f) >= min_dump_size and (
                not require_pq or glob.glob(f.replace(DUMP_SUFFIX, PARSED_SUFFIX))
            ):
                globbed.append(f)
                if len(globbed) >= max_globbed:
                    break
        random.shuffle(globbed)
        dump_files.extend(globbed[:max_globbed])
    random.seed()
    return dump_files


def parser_worker(args, logger, dump_file, max_perm):
    reg_log_parser = RegLogParser(args, logger)
    dfs = [
        df
        for df in reg_log_parser.parse(
            dump_file, max_perm=max_perm, require_pq=args.require_pq
        )
    ]
    return dump_file, dfs


def get_prompt(args, dataset, logger):
    seq, seq_meta = dataset.getseq(args.start_seq)
    if args.start_n is None:
        # Don't predict past where we can compare accuracy.
        start = random.randint(0, len(seq) - args.max_seq_len)
    else:
        start = args.start_n
    logger.info(
        "starting at seq %u (%s), %u / %u, irq %u",
        args.start_seq,
        seq_meta.df_file,
        start,
        len(seq),
        seq_meta.irq,
    )
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
        self.tokenizer = RegTokenizer(args, tokens=None, logger=logger)

    def load_dfs(self, reglogs=None, dump_files=None, max_perm=99, encode=True):
        if not dump_files:
            if not reglogs:
                raise ValueError("need reglogs or dump_files")
            dump_files = glob_dumps(
                reglogs,
                int(self.args.max_files * 1.25),
                self.args.min_dump_size,
                self.args.require_pq,
                seed=0,
            )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=4,
        ) as executor:
            futures = [
                executor.submit(
                    parser_worker, self.args, self.logger, dump_file, max_perm
                )
                for dump_file in dump_files
            ]
            dump_files = set()
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=min(len(futures), self.args.max_files),
            ):
                if len(dump_files) == self.args.max_files:
                    break
                dump_file, dfs = future.result()
                for i, df in enumerate(dfs):
                    seq = None
                    if self.tokenizer.tokens is not None:
                        df = self.tokenizer.merge_token_df(self.tokenizer.tokens, df)
                        if encode:
                            n = df["n"].astype(np.int16).to_numpy()
                            seq = self.tokenizer.encode(n).astype(np.int16)
                            if len(seq) < self.args.seq_len:
                                self.logger.info(
                                    "rejecting sequence from %s too short %u",
                                    dump_file,
                                    len(seq),
                                )
                                break
                    dump_files.add(dump_file)
                    irq = df["irq"].iloc[0]
                    yield dump_file, i, df, seq, irq
            executor.shutdown(wait=True, cancel_futures=True)

    def make_tokens(self, reglogs):
        df_files = []
        for df_file, _i, df, _seq, _irq in self.load_dfs(
            reglogs=reglogs, max_perm=self.args.max_perm
        ):
            self.tokenizer.accumulate_tokens(df, df_file)
            try:
                if df_files[-1] == df_file:
                    continue
            except IndexError:
                pass
            df_files.append(df_file)
        tokens = self.tokenizer.make_tokens()
        self.tokenizer.tokens = tokens
        assert self.tokenizer.tokens[tokens["val"].isna()].empty, tokens[
            tokens["val"].isna()
        ]
        return df_files

    def preload(self, tokens=None, tkmodel=None):
        if tokens is not None:
            self.tokenizer.load(tkmodel, tokens)
            return
        self.logger.info("preload making tokens")
        df_files = self.make_tokens(self.args.reglogs)
        if self.args.token_csv:
            self.logger.info("writing tokens to %s", self.args.token_csv)
            self.tokenizer.tokens.to_csv(self.args.token_csv, index=False)
        dataset_csv = self.args.dataset_csv
        df_map_csv = self.args.df_map_csv

        if not self.args.tkvocab and not dataset_csv:
            if df_map_csv:
                df_map = pd.DataFrame(df_files, columns=["dump_file"])
                df_map.to_csv(df_map_csv, index=False)
            return

        def worker():
            dataset_csv = self.args.dataset_csv

            def worker_gen():
                for i, (df_file, _i, df, _seq, _irq) in enumerate(
                    self.load_dfs(
                        dump_files=df_files,
                        max_perm=self.args.max_perm,
                        encode=False,
                    )
                ):
                    yield df_file, df, i

            if dataset_csv:
                self.logger.info("writing dataset to %s", dataset_csv)
                with zstd.open(dataset_csv, "w") as f:
                    for df_file, df, i in worker_gen():
                        df["i"] = int(i)
                        df.to_csv(f, index=False, header=(i == 0))
                        yield df_file, df, i
            else:
                for df_file, df, i in worker_gen():
                    yield df_file, df, i

            if df_map_csv:
                self.logger.info("writing dataset map to %s", df_map_csv)
                df_map = pd.DataFrame(df_files, columns=["dump_file"])
                df_map.to_csv(df_map_csv, index=False)

        if self.args.tkvocab:
            self.tokenizer.train_tokenizer(worker())
        else:
            for _df in worker():
                continue

    def load(self):
        assert self.tokenizer.tokens is not None
        dump_files = None
        reglogs = None
        if self.args.reglog:
            self.logger.info(f"loading data from {self.args.reglog}")
            reglogs = self.args.reglog
        elif os.path.exists(self.args.df_map_csv):
            df_map_df = pd.read_csv(self.args.df_map_csv)
            dump_files = df_map_df["dump_file"].drop_duplicates().tolist()
            self.logger.info(
                f"loading data from {self.args.df_map_csv} - {len(dump_files)} files"
            )
        elif self.args.reglogs:
            self.logger.info(f"loading data from {self.args.reglogs}")
            reglogs = self.args.reglogs
        self.n_vocab = len(self.tokenizer.tokens["n"])
        if self.args.tkvocab:
            self.n_vocab = self.args.tkvocab
        self.n_words = 0
        n_seq = 0
        n_words = 0
        reg_max = {}
        for df_file, i, df, seq, irq in self.load_dfs(
            reglogs=reglogs,
            dump_files=dump_files,
            max_perm=self.args.max_perm,
            encode=True,
        ):
            seq_meta = SeqMeta(irq=irq, df_file=df_file, i=i)
            self.seq_mapper.add(seq, seq_meta)
            reg_max = self.tokenizer.get_reg_max(df, reg_max)
            self.n_words += len(seq)
            n_words += len(df)
            n_seq += 1
        self.seq_mapper.finalize()
        self.reg_widths = self.tokenizer.get_reg_width_from_max(reg_max)
        n_frac = 0
        if n_words:
            n_frac = round(self.n_words / n_words, 2)
        self.logger.info(
            f"n_vocab: {self.n_vocab}, n_words {n_words}, n_encoded_words {self.n_words} ({n_frac}), reg widths {sorted(self.reg_widths.items())}, {n_seq} sequences"
        )

    def __len__(self):
        return len(self.seq_mapper)

    def __getitem__(self, index):
        return self.seq_mapper[index]

    def getseq(self, i):
        seq, seq_meta = self.seq_mapper.getseq(i)
        return torch.from_numpy(np.copy(seq)), seq_meta


class LowMemoryRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            yield torch.randint(low=0, high=len(self.data_source), size=(1,)).item()

    def __len__(self):
        return self.num_samples


def _get_loader(args, dataset):

    if args.shuffle:
        length = args.shuffle / 1.0
        sampler = torch.utils.data.RandomSampler(
            # sampler = LowMemoryRandomSampler(
            dataset,
            num_samples=int(length * len(dataset)),
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=2,
    )


def get_loader(args, dataset, seq_mapper=False):
    dataset.load()
    if seq_mapper:
        return _get_loader(args, copy.deepcopy(dataset.seq_mapper))
    return get_loader(args, dataset)
