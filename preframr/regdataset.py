import logging
import glob
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import zstandard as zstd
from preframr.reglogparser import RegLogParser
from preframr.regtokenizer import RegTokenizer
from preframr.seq_mapper import SeqMapper, SeqMeta


def glob_dumps(reglogs, max_files, min_dump_size, seed=0):
    random.seed(seed)
    dump_files = []
    for r in reglogs.split(","):
        max_globbed = max_files - len(dump_files)
        if max_globbed <= 0:
            break
        globbed = [
            f
            for f in glob.glob(r, recursive=True)
            if os.path.getsize(f) >= min_dump_size
        ]
        random.shuffle(globbed)
        dump_files.extend(globbed[:max_globbed])
    random.seed()
    return dump_files


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, args, logger=logging):
        self.args = args
        self.logger = logger
        self.n_vocab = 0
        self.n_words = 0
        self.reg_widths = {}
        self.reg_log_parser = RegLogParser(args, logger)
        self.seq_mapper = SeqMapper(args.seq_len)
        self.tokenizer = RegTokenizer(args, tokens=None)

    def load_dfs(self, reglogs, max_perm=99, shuffle=0):
        dump_files = glob_dumps(reglogs, self.args.max_files, self.args.min_dump_size)
        for dump_file in dump_files:
            for i, df in enumerate(
                self.reg_log_parser.parse(dump_file, max_perm=max_perm)
            ):
                seq = None
                if self.tokenizer.tokens is not None:
                    df = self.tokenizer.merge_token_df(self.tokenizer.tokens, df)
                    seq = self.tokenizer.encode(df["n"].astype(np.int64).to_numpy())
                    if len(seq) < self.args.seq_len:
                        self.logger.info(
                            "rejecting sequence from %s too short %u",
                            dump_file,
                            len(seq),
                        )
                        continue
                irq = df["irq"].iloc[0]
                self.logger.info("loaded %s, irq %u, augment %u", dump_file, irq, i)
                yield dump_file, df, seq

    def make_tokens(self, reglogs):
        df_files = []
        dfs = []
        for df_file, df, _seq in self.load_dfs(reglogs, max_perm=self.args.max_perm):
            df_files.append(df_file)
            dfs.append(df)
        self.tokenizer.tokens = self.tokenizer._make_tokens(dfs)
        dfs = self.tokenizer.merge_tokens(self.tokenizer.tokens, dfs)
        assert self.tokenizer.tokens[self.tokenizer.tokens["val"].isna()].empty
        assert self.tokenizer.tokens[self.tokenizer.tokens["val"] < 0].empty
        return df_files, dfs

    def preload(self, tokens=None, tkmodel=None):
        if tokens is not None and tkmodel is not None:
            self.tokenizer.load(tkmodel, tokens)
            return
        df_files, dfs = self.make_tokens(self.args.reglogs)
        if self.args.token_csv:
            self.logger.info("writing %s", self.args.token_csv)
            self.tokenizer.tokens.to_csv(self.args.token_csv)
        if self.args.tkvocab:
            self.tokenizer.train_tokenizer(dfs)
        if self.args.df_map_csv:
            self.logger.info(f"writing {self.args.df_map_csv}")
            pd.DataFrame(df_files, columns=["dump_file"]).to_csv(self.args.df_map_csv)
        if self.args.dataset_csv:
            self.logger.info(f"writing {self.args.dataset_csv}")
            with zstd.open(self.args.dataset_csv, "w") as f:
                for i in tqdm(range(len(df_files)), ascii=True):
                    dfs[i]["i"] = int(i)
                    dfs[i].to_csv(f, index=False, header=(i == 0))

    def load(self):
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
        for df_file, df, seq in self.load_dfs(reglogs, max_perm=self.args.max_perm):
            reg_max = self.tokenizer.get_reg_max(df, reg_max)
            seq_meta = SeqMeta(irq=int(df["irq"].iat[0]))
            self.seq_mapper.add(seq, seq_meta)
            self.n_words += len(seq)
            n_words += len(df)
            n_seq += 1
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
