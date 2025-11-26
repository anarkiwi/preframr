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
VAL_PDTYPE = pd.UInt32Dtype()
TOKEN_PDTYPE = pd.Int64Dtype()  # Same as torch
UNK_TOKEN = "<unk>"
END_OF_WORD_SUFFIX = "</w>"


class RegTokenizer:
    def __init__(self, args, tokens, logger=logging):
        self.args = args
        self.logger = logger
        self.tokens = tokens
        self.tkmodel = None

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
        encoded_dfs = []
        for df in dfs:
            orig_seq = df["n"].to_numpy()
            encoded = self.encode_unicode(orig_seq)
            encoded_dfs.append(encoded)
        self.tkmodel, trainer = self.get_tk(tokenizer=tokenizer)
        self.tkmodel.train_from_iterator(encoded_dfs, trainer=trainer)
        assert self.tkmodel.get_vocab_size() == self.args.tkvocab, (
            self.tkmodel.get_vocab_size(),
            self.args.tkvocab,
        )

    def _make_tokens(self, dfs):
        self.logger.info("making tokens")
        tokens = [df[TOKEN_KEYS].drop_duplicates() for df in dfs]
        tokens = pd.concat(tokens, copy=False)
        tokens = tokens.drop_duplicates().sort_values(TOKEN_KEYS).reset_index(drop=True)
        tokens.loc[tokens["reg"] == FRAME_REG, ["val", "diff"]] = 0
        tokens = tokens.drop_duplicates().sort_values(TOKEN_KEYS).reset_index(drop=True)
        tokens["n"] = tokens.index
        tokens = tokens.sort_values(["n"])
        tokens = tokens.astype(
            {"val": VAL_PDTYPE, "diff": DIFF_PDTYPE, "n": TOKEN_PDTYPE}
        )
        assert len(tokens[tokens["reg"] == FRAME_REG]) <= 1
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
            orig_cols, orig_dtypes = df.columns, df.dtypes
            df, missing_tokens = self._merged_and_missing(tokens, df)
            if not missing_tokens.empty:
                for missing_token in missing_tokens.itertuples():
                    reg = missing_token.reg
                    val = missing_token.val
                    reg_tokens = tokens[tokens["reg"] == reg]
                    if reg_tokens.empty:
                        self.logger.error(
                            "no possible token for reg %u val %u", reg, val
                        )
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
                assert missing_tokens.empty
            merged_dfs.append(df)
        return merged_dfs

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

    def get_reg_widths(self, dfs):
        reg_widths = {}
        unique_regs = set()
        for df in tqdm(dfs, ascii=True):
            unique_regs.update(list(df["reg"].unique()))
        for reg in unique_regs:
            reg_max = 0
            for df in dfs:
                reg_df = df[df["reg"] == reg]
                if reg_df.empty:
                    continue
                reg_max = max(reg_df["val"].max(), reg_max)
            for width in range(1, 8):
                if reg_max < 2 ** (8 * width):
                    reg_widths[int(reg)] = width
                    break
            assert reg_widths[int(reg)]
        return reg_widths
