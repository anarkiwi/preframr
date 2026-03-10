import concurrent.futures
import difflib
import logging
import multiprocessing
import string
import zstandard as zstd
from tqdm import tqdm
from tokenizers import Tokenizer
import multiprocessing
import numpy as np
import pandas as pd
from preframr.stfconstants import (
    DUMP_SUFFIX,
    FRAME_REG,
    UNICODE_BASE,
    UNI_SUFFIX,
    TOKEN_KEYS,
)
from preframr.train_worker import train_worker

DIFF_PDTYPE = pd.UInt16Dtype()
VAL_PDTYPE = pd.Int32Dtype()
TOKEN_PDTYPE = pd.Int64Dtype()  # Same as torch
UNK_TOKEN = "<unk>"
END_OF_WORD_SUFFIX = "</w>"
SPLITCHS = [ord(i) for i in string.punctuation]
SPLITTERS = len(SPLITCHS)


class RegTokenizer:
    def __init__(self, args, tokens, logger=logging):
        self.args = args
        self.logger = logger
        self.tokens = tokens
        self.tkmodel = None
        self.frame_tokens = []
        self.splitters = SPLITTERS
        self.splitchs = SPLITCHS

    def load(self, tkmodel, tokens):
        self.tokens = tokens
        if tkmodel:
            self.logger.info("loading tokenizer model")
            self.tkmodel = Tokenizer.from_str(tkmodel)

    def encode_unicode(self, tokens, dtype=np.uint16):
        t = np.array(tokens, dtype=dtype)
        m = t >= self.splitters
        t[m] += UNICODE_BASE
        for c in range(self.splitters):
            t = np.where(t == c, self.splitchs[c], t)
        encoded = "".join([chr(i) for i in t])
        assert len(encoded) == t.shape[0]
        return encoded

    def decode_unicode(self, encoded_tokens, dtype=np.uint16):
        encoded = [ord(i) for i in encoded_tokens]
        t = np.array(encoded, dtype=dtype)
        assert len(encoded_tokens) == len(encoded)
        assert len(encoded_tokens) == t.shape[0]
        for c in range(self.splitters):
            t = np.where(t == self.splitchs[c], c, t)
        m = t >= self.splitters
        t[m] -= UNICODE_BASE
        return t

    def encode(self, tokens, dtype=np.uint16):
        if self.tkmodel:
            encoded = self.tkmodel.encode(self.encode_unicode(tokens, dtype=dtype))
            return np.array(encoded.ids, dtype=dtype)
        return tokens

    def decode(self, encoded_tokens, dtype=np.uint16):
        if self.tkmodel:
            return self.decode_unicode(self.tkmodel.decode(encoded_tokens), dtype=dtype)
        return encoded_tokens

    def train_tokenizer(self, dfs):
        frame_tokens = 1
        if self.tokens is not None and len(self.tokens):
            frame_tokens = len(self.tokens[self.tokens["reg"] == FRAME_REG])
        self.logger.info(
            f"feeding {self.args.tokenizer} tokenizer with {frame_tokens} frame tokens"
        )
        self.splitters = min(self.splitters, frame_tokens)

        def write_uni(t):
            df_file, df, i = t
            uni_file = df_file.replace(DUMP_SUFFIX, f".{i}{UNI_SUFFIX}")
            orig_seq = df["n"].to_numpy()
            encoded = self.encode_unicode(orig_seq)
            with zstd.open(uni_file, "w") as f:
                f.write(encoded)
            return uni_file

        uni_files = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
            futures = [p.submit(write_uni, df) for df in dfs]
            for future in concurrent.futures.as_completed(futures):
                uni_files.append(future.result())

        self.logger.info("running tokenizer")
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=train_worker,
            args=(
                self.args.tokenizer,
                self.args.tkvocab,
                self.args.tkmodel,
                uni_files,
            ),
        )
        p.start()
        p.join()
        assert p.exitcode == 0, p.exitcode

        self.tkmodel = Tokenizer.from_file(self.args.tkmodel)
        assert self.tkmodel.get_vocab_size() == self.args.tkvocab, (
            self.tkmodel.get_vocab_size(),
            self.args.tkvocab,
        )

    def token_metadata(self):
        if self.tkmodel:
            metadata = []
            for t in range(self.args.tkvocab):
                decoded = [self.tokens.iloc[x] for x in self.decode([t])]
                metadata.append(
                    ",".join((f"{x.op} {x.reg} {x.subreg} {x.val}" for x in decoded))
                )
            return metadata
        metadata = [
            f"{x.op} {x.reg} {x.subreg} {x.val}" for x in self.tokens.itertuples()
        ]
        return metadata

    def crunch_tokens(self):
        frame_tokens = pd.concat(self.frame_tokens, ignore_index=True)
        frame_tokens["count"] = frame_tokens.groupby(TOKEN_KEYS)["count"].transform(
            "sum"
        )
        self.frame_tokens = [
            frame_tokens.drop_duplicates(TOKEN_KEYS)
            .sort_values(TOKEN_KEYS)
            .reset_index(drop=True)
        ]

    def accumulate_tokens(self, df, df_file):
        frame_tokens = df[TOKEN_KEYS].copy().reset_index(drop=True)
        frame_tokens = frame_tokens.join(
            frame_tokens.value_counts(), on=TOKEN_KEYS
        ).drop_duplicates(TOKEN_KEYS)
        assert frame_tokens["reg"].max() < 256, df_file
        self.frame_tokens.append(frame_tokens)
        if len(self.frame_tokens) > 64:
            self.crunch_tokens()

    def make_tokens(self):
        self.logger.info("making tokens")
        self.crunch_tokens()
        tokens = self.frame_tokens[0]
        tokens["n"] = tokens.index
        tokens = tokens.sort_values(["n"])
        tokens = tokens.astype({"val": VAL_PDTYPE, "n": TOKEN_PDTYPE})
        assert tokens["reg"].max() < 256
        return tokens

    def _merged_and_missing(self, tokens, df):
        m = df["reg"] == FRAME_REG
        irq = df[m]["diff"].iloc[0]
        df.loc[m, "diff"] = 0
        df = df.merge(tokens, on=TOKEN_KEYS, how="left")
        df.loc[m, "diff"] = irq
        missing_tokens = (
            df[df["n"].isna()].drop_duplicates().sort_values(["reg", "val"])
        )
        return df, missing_tokens

    def merge_tokens(self, tokens, dfs):
        self.logger.info("merging tokens")
        merged_dfs = []
        for df in tqdm(dfs, ascii=True):
            merged_df = self.merge_token_df(tokens, df)
            if merged_df is None:
                return None
            merged_dfs.append(merged_df)
        return merged_dfs

    def merge_token_df(self, tokens, df):
        orig_cols, orig_dtypes = df.columns, df.dtypes
        df, missing_tokens = self._merged_and_missing(tokens, df)
        if not missing_tokens.empty:
            for missing_token in missing_tokens.itertuples():
                reg = missing_token.reg
                val = missing_token.val
                reg_tokens = tokens[tokens["reg"] == reg]
                if reg_tokens.empty:
                    self.logger.error("no possible token for reg %u val %u", reg, val)
                    assert False
                compare_tokens = reg_tokens.copy()
                compare_tokens["diff_val"] = (compare_tokens["val"] - val).abs()
                best_token = compare_tokens[
                    compare_tokens["diff_val"] == compare_tokens["diff_val"].min()
                ].iloc[0]
                best_val = best_token.val
                self.logger.info(
                    "substitute reg %u val %u with val %u", reg, val, best_val
                )
                df.loc[((df["reg"] == reg) & (df["val"] == val)), "val"] = best_val
            df = df[orig_cols].astype(orig_dtypes)
            df, missing_tokens = self._merged_and_missing(tokens, df)
            assert missing_tokens.empty, missing_tokens
            return df
        return df

    def validate_encoding(self, df_file, seq):
        if not self.args.tkvocab:
            return seq
        orig_seq = seq.copy()
        seq = self.encode(orig_seq, dtype=np.int64)
        decoded_seq = self.decode(seq, dtype=np.int64)
        if not np.array_equal(orig_seq, decoded_seq):
            for i, (orig, decoded) in enumerate(zip(orig_seq, decoded_seq)):
                if orig == decoded:
                    continue
                a = [str(i) for i in orig_seq]
                b = [str(i) for i in decoded_seq]
                d = "\n".join(difflib.context_diff(a, b))
                print(d)
                assert False, (
                    df_file,
                    i,
                    orig,
                    decoded,
                    self.tokens.iloc[int(orig)],
                )
        return seq

    def get_reg_max(self, df, reg_max):
        df_max = df.groupby("reg")["val"].max().to_dict()
        for reg, val_max in df_max.items():
            reg_max[reg] = max(val_max, reg_max.get(reg, 0))
        return reg_max

    def get_reg_width_from_max(self, reg_max):
        reg_widths = {}
        for reg, val in reg_max.items():
            for width in range(1, 8):
                if val < 2 ** (8 * width):
                    reg_widths[int(reg)] = width
                    break
            assert reg_widths[int(reg)]
        return reg_widths
