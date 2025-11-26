import logging
import glob
import os
import random
from tqdm import tqdm
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
        results = []
        unsorted_dump_files = dump_files
        random.shuffle(unsorted_dump_files)
        for dump_file in unsorted_dump_files:
            for i, df in enumerate(
                self.reg_log_parser.parse(dump_file, max_perm=max_perm)
            ):
                if self.tokenizer.tokens is not None:
                    df = self.tokenizer.merge_token_df(self.tokenizer.tokens, df)
                irq = df["irq"].iloc[0]
                self.logger.info("loaded %s, irq %u, augment %u", dump_file, irq, i)
                results.append((dump_file, df))
        if shuffle is not None:
            random.seed(shuffle)
            random.shuffle(results)
            random.seed()
        else:
            results = sorted(results, key=lambda x: x[0])
        df_files = [result[0] for result in results]
        dfs = [result[1] for result in results]
        if self.tokenizer.tokens is None:
            self.tokenizer.tokens = self.tokenizer._make_tokens(dfs)
            dfs = self.tokenizer.merge_tokens(self.tokenizer.tokens, dfs)
            if self.args.token_csv:
                self.logger.info("writing %s", self.args.token_csv)
                self.tokenizer.tokens.to_csv(self.args.token_csv)
        return df_files, dfs

    def load(self, tokens=None, tkmodel=None):
        self.tokenizer.load(tkmodel, tokens)
        if self.args.reglog:
            df_files, dfs = self.load_dfs(
                self.args.reglog,
                max_perm=self.args.max_perm,
            )
        else:
            df_files, dfs = self.load_dfs(
                self.args.reglogs, max_perm=self.args.max_perm
            )
            if self.args.tkvocab and tkmodel is None:
                self.tokenizer.train_tokenizer(dfs)
        self.logger.info("getting reg widths")
        self.reg_widths = self.tokenizer.get_reg_widths(dfs)
        self.n_vocab = len(self.tokenizer.tokens["n"])
        self.n_words = sum((len(df) for df in dfs))
        assert self.tokenizer.tokens[self.tokenizer.tokens["val"].isna()].empty
        assert self.tokenizer.tokens[self.tokenizer.tokens["val"] < 0].empty
        self.logger.info(
            f"n_vocab: {self.n_vocab}, n_words {self.n_words}, reg widths {sorted(self.reg_widths.items())}"
        )
        if self.args.tkvocab:
            self.n_vocab = self.args.tkvocab
        self.n_words = 0
        self.logger.info("mapping sequences")
        final_dfs = []
        final_df_files = []
        for df_file, df in tqdm(zip(df_files, dfs), ascii=True):
            seq = self.tokenizer.validate_encoding(df_file, df["n"].to_numpy())
            seq_meta = SeqMeta(irq=int(df["irq"].iat[0]))
            try:
                self.seq_mapper.add(seq, seq_meta)
                self.n_words += len(seq)
                final_df_files.append(df_file)
                final_dfs.append(df)
            except ValueError:
                self.logger.info(
                    "rejecting sequence from %s too short %u", df_file, len(seq)
                )
        self.logger.info(
            f"n_encoded_words {self.n_words}, {len(dfs)} sequences",
        )
        if not self.args.reglog:
            if self.args.df_map_csv:
                self.logger.info(f"writing {self.args.df_map_csv}")
                pd.DataFrame(final_df_files, columns=["dump_file"]).to_csv(
                    self.args.df_map_csv
                )
            if self.args.dataset_csv:
                self.logger.info(f"writing {self.args.dataset_csv}")
                with zstd.open(self.args.dataset_csv, "w") as f:
                    for i in tqdm(range(len(final_dfs)), ascii=True):
                        final_dfs[i]["i"] = int(i)
                        final_dfs[i].to_csv(f, index=False, header=(i == 0))

    def __len__(self):
        return len(self.seq_mapper)

    def __getitem__(self, index):
        return self.seq_mapper[index]

    def getseq(self, i):
        return torch.from_numpy(self.seq_mapper.seqs[i]), self.seq_mapper.seq_metas[i]


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
