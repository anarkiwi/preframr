import difflib
import logging
from tqdm import tqdm
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import numpy as np
import pandas as pd
from preframr.stfconstants import (
    FRAME_REG,
    UNICODE_BASE,
    TOKEN_KEYS,
)

DIFF_PDTYPE = pd.UInt16Dtype()
VAL_PDTYPE = pd.Int32Dtype()
TOKEN_PDTYPE = pd.Int64Dtype()  # Same as torch
UNK_TOKEN = "<unk>"
END_OF_WORD_SUFFIX = "</w>"


class RegTokenizer:
    def __init__(self, args, tokens, logger=logging):
        self.args = args
        self.logger = logger
        self.tokens = tokens
        self.tkmodel = None
        self.frame_tokens = []

    def load(self, tkmodel, tokens):
        self.tokens = tokens
        if tkmodel:
            self.tkmodel = Tokenizer.from_str(tkmodel)

    def encode_unicode(self, tokens, dtype=np.uint16):
        t = np.array(tokens, dtype=dtype)
        t = np.where(t == 0, np.nan, t)
        t += UNICODE_BASE
        t = np.nan_to_num(t).astype(dtype)
        t = np.where(t == 0, 32, t)
        return "".join([chr(i) for i in t])

    def decode_unicode(self, encoded_tokens, dtype=np.uint16):
        t = np.array([ord(i) for i in encoded_tokens])
        t = np.where(t == 32, np.nan, t)
        t -= UNICODE_BASE
        t = np.nan_to_num(t).astype(dtype)
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

    def train_tokenizer(self, dfs, tokenizer="unigram"):
        def encode_dfs():
            for df in tqdm(dfs):
                orig_seq = df["n"].to_numpy()
                encoded = self.encode_unicode(orig_seq)
                yield encoded

        self.tkmodel, trainer = self.get_tk(tokenizer=tokenizer)
        self.tkmodel.train_from_iterator(encode_dfs(), trainer=trainer)
        assert self.tkmodel.get_vocab_size() == self.args.tkvocab, (
            self.tkmodel.get_vocab_size(),
            self.args.tkvocab,
        )

    def crunch_tokens(self):
        self.frame_tokens = [
            pd.concat(self.frame_tokens)
            .drop_duplicates()
            .sort_values(TOKEN_KEYS)
            .copy()
            .reset_index(drop=True)
        ]

    def accumulate_tokens(self, df, df_file):
        frame_tokens = df[TOKEN_KEYS].drop_duplicates().copy().reset_index(drop=True)
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

    def get_tk(self, tokenizer="unigram"):
        if tokenizer == "unigram":
            tk = Tokenizer(models.Unigram())
            tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement=" ")
            tk.decoder = decoders.Metaspace(replacement=" ")
            tk.normalizer = None
            trainer = trainers.UnigramTrainer(
                vocab_size=self.args.tkvocab,
                show_progress=True,
                special_tokens=[UNK_TOKEN],
                initial_alphabet=[],
                unk_token=UNK_TOKEN,
            )
            return tk, trainer
        if tokenizer == "bpe":
            tk = Tokenizer(
                models.BPE(
                    dropout=None,
                    unk_token=UNK_TOKEN,
                    end_of_word_suffix=END_OF_WORD_SUFFIX,
                    fuse_unk=False,
                    byte_fallback=False,
                    ignore_merges=False,
                    vocab={},
                    merges=[],
                )
            )
            tk.normalizer = None
            tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            tk.decoder = decoders.BPEDecoder(suffix=END_OF_WORD_SUFFIX)
            trainer = trainers.BpeTrainer(
                vocab_size=self.args.tkvocab,
                min_frequency=2,
                special_tokens=[UNK_TOKEN],
                limit_alphabet=self.args.tkvocab,
                initial_alphabet=[],
                end_of_word_suffix=END_OF_WORD_SUFFIX,
                show_progress=True,
            )
            return tk, trainer
        raise ValueError

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
