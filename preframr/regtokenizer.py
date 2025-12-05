import difflib
import logging
import random
from tqdm import tqdm
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import numpy as np
import pandas as pd
from preframr.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    UNICODE_BASE,
    VOICES,
    VOICE_REG,
    VOICE_REG_SIZE,
    TOKEN_KEYS,
)

DIFF_PDTYPE = pd.UInt16Dtype()
VAL_PDTYPE = pd.UInt32Dtype()
TOKEN_PDTYPE = pd.Int64Dtype()  # Same as torch
UNK_TOKEN = "<unk>"
END_OF_WORD_SUFFIX = "</w>"


def state_df(states, dataset, irq):
    tokens = dataset.tokenizer.tokens.copy()
    tokens.loc[tokens["reg"] == FRAME_REG, "diff"] = irq
    df = pd.DataFrame(states, columns=["n"]).merge(tokens, on="n", how="left")
    return df


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
    prompt = seq[start:][: args.prompt_seq_len].unsqueeze(0)
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


def remove_voice_reg(orig_df, reg_widths):
    voice_regs = len(orig_df[orig_df["reg"] == VOICE_REG])
    if voice_regs:
        df = orig_df.copy()
        df["vr"] = pd.NA
        df.loc[df["reg"].isin({FRAME_REG, VOICE_REG}), "vr"] = df["val"]
        df.loc[df["reg"] == DELAY_REG, "vr"] = 0
        df["vr"] = df["vr"].astype(pd.UInt8Dtype()).ffill().fillna(0)
        df = df[df["reg"] != VOICE_REG]
        df["vr"] = df["vr"].astype(pd.Int64Dtype()) * VOICE_REG_SIZE
        df.loc[df["reg"] >= VOICE_REG_SIZE, "vr"] = 0
        df["reg"] += df["vr"]
        df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for i in range(VOICE_REG_SIZE):
                if i in reg_widths:
                    reg_widths[v_offset + i] = reg_widths[i]
        return df, reg_widths
    return orig_df, reg_widths


class RegTokenizer:
    def __init__(self, args, tokens, logger=logging):
        self.args = args
        self.logger = logger
        self.tokens = tokens
        self.tkmodel = None

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

    def _filter_tokens(self, df):
        return df[TOKEN_KEYS].drop_duplicates().copy().reset_index(drop=True)

    def _make_tokens(self, dfs):
        self.logger.info("making tokens")
        tokens = [self._filter_tokens(df) for df in dfs]
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
            assert missing_tokens.empty
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
        for reg in df["reg"].unique():
            val_max = df[df["reg"] == reg]["val"].max()
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

    def get_reg_widths(self, dfs):
        reg_max = {}
        for df in tqdm(dfs, ascii=True):
            reg_max = self.get_reg_max(df, reg_max)
        return self.get_reg_width_from_max(reg_max)
