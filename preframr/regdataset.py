import concurrent.futures
import difflib
import itertools
import json
import logging
import glob
import os
import random
import shutil
import tempfile
import torch
from tokenizers import CharBPETokenizer
import numpy as np
import pandas as pd
from preframr.stfconstants import (
    CTRL_REG,
    DELAY_REG,
    FRAME_REG,
    RESET_REG,
    NOOP_REG,
    VOICE_REG,
    VOICES,
    VOICE_REG_SIZE,
    UNICODE_BASE,
    FILTER_REG,
    MAX_REG,
    FC_LO_REG,
)

TOKEN_KEYS = ["reg", "val", "diff"]
MODEL_PDTYPE = pd.Int32Dtype()
REG_PDTYPE = pd.Int8Dtype()
VAL_PDTYPE = pd.UInt32Dtype()
TOKEN_PDTYPE = pd.UInt16Dtype()


def wrapbits(x, reglen):
    base = (x << 1) & (2**reglen - 1)
    lsb = (x >> (reglen - 1)) & 1
    return base ^ lsb


FILTER_SHIFT_DF = pd.DataFrame(
    [{"reg": FILTER_REG, "val": i, "y": wrapbits(i, 3)} for i in range(2**3)],
    dtype=MODEL_PDTYPE,
)


class SeqMapper:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.seq_map = None
        self.seqs = []
        self.len = 0

    def add(self, seq):
        if len(seq) <= self.seq_len:
            raise ValueError("sequence too short %u" % len(seq))
        self.seqs.append(torch.LongTensor(seq))
        self.len = 0
        seq_map = []
        for seq in self.seqs:
            seq_map.append(self.len)
            self.len += len(seq) - self.seq_len
        self.seq_map = np.array(seq_map, dtype=np.uint64)

    def __len__(self):
        return self.len

    def slice_n(self, seq, n):
        return seq[int(n) : int(n) + self.seq_len]

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        seq_i = np.clip(
            np.searchsorted(self.seq_map, index, side="right") - 1, a_min=0, a_max=None
        )
        seq = self.seqs[seq_i]
        seq_index = index - self.seq_map[seq_i]
        return (self.slice_n(seq, seq_index), self.slice_n(seq, seq_index + 1))


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, args, logger=logging):
        self.args = args
        self.logger = logger
        self.dfs = None
        self.df_files = []
        self.tokens = None
        self.n_vocab = 0
        self.n_words = 0
        self.reg_widths = {}
        self.tk = None
        self.seq_mapper = SeqMapper(args.seq_len)

    def _ctrl_match(self, df):
        return (df["reg"] == 4) | (df["reg"] == 11) | (df["reg"] == 18)

    def _read_df(self, name):
        df = pd.read_csv(
            name,
            sep=" ",
            names=["clock", "irq_diff", "nmi_diff", "chipno", "reg", "val"],
            dtype={
                "clock": MODEL_PDTYPE,
                "irq_diff": MODEL_PDTYPE,
                "nmi_diff": MODEL_PDTYPE,
                "chipno": REG_PDTYPE,
                "reg": REG_PDTYPE,
                "val": VAL_PDTYPE,
            },
        )
        # assert df["reg"].min() >= 0
        df["irq"] = df["clock"].astype(MODEL_PDTYPE) - df["irq_diff"]
        # keep only chipno 0
        df = df[df["chipno"] == 0]
        df = df[df["reg"] <= MAX_REG]
        df = df[["clock", "irq", "reg", "val"]]
        return df

    def _make_tokens(self, dfs):
        tokens = [df[TOKEN_KEYS].drop_duplicates() for df in dfs]
        tokens = pd.concat(tokens).drop_duplicates().sort_values(TOKEN_KEYS)
        tokens.reset_index(drop=True, inplace=True)
        tokens["n"] = tokens.index
        tokens = tokens.sort_values(["n"])
        tokens = tokens.astype(
            {"val": VAL_PDTYPE, "diff": MODEL_PDTYPE, "n": TOKEN_PDTYPE}
        )
        return tokens

    def _maskreg(self, df, reg, valmask):
        mask = df["reg"] == reg
        df.loc[mask, ["val"]] = df[mask]["val"] & valmask

    def highbitmask(self, bits):
        return 255 - (2**bits - 1)

    def _maskregbits(self, df, reg, bits):
        self._maskreg(df, reg, self.highbitmask(bits))

    def _squeeze_changes(self, df):
        diff_cols = df.reg.unique()
        reg_df = (
            df.pivot(columns="reg", values="val").astype(MODEL_PDTYPE).ffill().fillna(0)
        )
        reg_df = reg_df.loc[
            (reg_df[diff_cols].shift(fill_value=0) != reg_df[diff_cols]).any(axis=1)
        ]
        df = reg_df.join(df)[["clock", "irq", "reg", "val"]]
        return df.reset_index(drop=True)

    def _combine_val(self, reg_df, reg, reg_range, dtype=MODEL_PDTYPE):
        origcols = reg_df.columns
        for i in range(reg_range):
            reg_df[str(i)] = reg_df[reg_df["reg"] == (reg + i)]["val"]
            reg_df[str(i)] = reg_df[str(i)].ffill().fillna(0)
            reg_df[str(i)] = np.left_shift(reg_df[str(i)].values, int(8 * i))
        reg_df.loc[:, "val"] = 0
        reg_df.loc[:, "reg"] = reg
        for i in range(reg_range):
            reg_df["val"] = reg_df["val"].astype(dtype) + reg_df[str(i)]
        return reg_df[origcols]

    def _combine_reg(self, orig_df, reg, diffmax=512, bits=0):
        df = orig_df.copy()
        df["dclock"] = df["clock"].floordiv(diffmax)
        cond = (df["reg"] == reg) | (df["reg"] == (reg + 1))
        reg_df = df[cond].copy()
        non_reg_df = df[~cond].copy()
        reg_df = self._combine_val(reg_df, reg, 2).drop_duplicates(
            ["dclock"], keep="last"
        )
        if bits:
            reg_df["val"] = np.left_shift(np.right_shift(reg_df["val"], bits), bits)
        df = pd.concat([non_reg_df, reg_df]).sort_values(["clock"], ascending=True)
        df = df[orig_df.columns].reset_index(drop=True).astype(orig_df.dtypes)
        return df

    def _combine_vreg(self, df, reg, reg_range=VOICE_REG_SIZE, dtype=MODEL_PDTYPE):
        origcols = df.columns
        df["val"] = df["val"].astype(dtype)
        cond = (df["reg"] >= reg) & (df["reg"] < (reg + reg_range))
        reg_df = df[cond].copy()
        df = df[~cond]
        reg_df = self._combine_val(reg_df, reg, reg_range, dtype=dtype)
        reg_df = reg_df[origcols]
        df = pd.concat([df, reg_df]).sort_values(["clock"]).reset_index(drop=True)
        return df

    def _combine_regs(self, df, diffmax=512, regs=(0,)):
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for reg in regs:
                df = self._combine_reg(df, reg + v_offset, diffmax)
        return df

    def _combine_vregs(self, df):
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            df = self._combine_vreg(df, v_offset)
        return df

    def _downsample_diff(self, df_diff, diffq):
        return (df_diff["diff"].floordiv(diffq).clip(lower=1) * diffq).astype(
            MODEL_PDTYPE
        )

    def _quantize_diff(self, df):
        for diffq_pow in (2, 3, 4, 5):
            diffq = self.args.diffq**diffq_pow
            mask = df["diff"] > diffq
            df.loc[mask, ["diff"]] = self._downsample_diff(df, diffq)
        df["diff"] = self._downsample_diff(df, self.args.diffq)
        return df

    def _quantize_longdiff(self, df, diffmin, diffmax):
        df["diff"] = df["clock"].diff().shift(-1).fillna(0).astype(MODEL_PDTYPE)
        # add delay rows
        m = df["diff"] >= diffmax
        long_df = df[m].copy()
        df.loc[m, "diff"] = diffmin
        long_df["reg"] = DELAY_REG
        long_df["val"] = 0
        long_df["clock"] += diffmin
        df = pd.concat([df, long_df]).sort_values(["clock"]).reset_index(drop=True)
        # move delay to DELAY_REG
        df["delaymarker"] = (
            (df["reg"] == DELAY_REG)
            .astype(MODEL_PDTYPE)
            .diff(periods=1)
            .astype(MODEL_PDTYPE)
            .cumsum()
            .cumsum()
            .shift(1)
            .fillna(0)
        )
        df["markerdelay"] = df.groupby("delaymarker")["diff"].transform("sum")
        df["markercount"] = df.groupby("delaymarker")["diff"].transform("count")
        df.loc[df["reg"] != DELAY_REG, ["diff"]] = 0
        df["diff"] = df["markerdelay"] - (df["markercount"] * diffmin)
        df.loc[df["reg"] != DELAY_REG, ["diff"]] = diffmin
        df = df.drop(["clock", "delaymarker", "markerdelay", "markercount"], axis=1)
        return df

    def _add_ctrl_reg(self, orig_df):
        df = orig_df.copy()
        for v in range(VOICES):
            ctrl = (VOICE_REG_SIZE * v) + 4
            m = df["reg"] == ctrl
            col = f"v{v}"
            df.loc[m, col] = df["val"]
            df[col] = np.left_shift(df[col].values, v * 8)
            df[col] = df[col].ffill().fillna(0)
        m = self._ctrl_match(df)
        df.loc[m, "val"] = df[m]["v0"] + df[m]["v1"] + df[m]["v2"]
        df.loc[m, "reg"] = CTRL_REG
        return df[orig_df.columns]

    def _rotate_filter(self, df, r):
        m = df["reg"] == FILTER_REG
        df.loc[m, "fres"] = df[m]["val"].values & 0xF0
        df.loc[m, "val"] -= df[m]["fres"]
        for _ in range(r):
            df = df.merge(FILTER_SHIFT_DF, how="left", on=["reg", "val"])
            m = df["reg"] == FILTER_REG
            df.loc[m, "val"] = df[m]["y"]
            df = df.drop(["y"], axis=1)
        m = df["reg"] == FILTER_REG
        df.loc[m, "val"] += df[m]["fres"]
        return df

    def _rotate_voice_augment(self, orig_df, augment=True):
        if not augment:
            yield orig_df
            return
        for r in range(VOICES):
            df = orig_df.copy()
            m = (df["reg"] < VOICE_REG_SIZE * VOICES) & (df["reg"] >= 0)
            df.loc[m, "reg"] = (df[m]["reg"] + (VOICE_REG_SIZE * r)).mod(
                VOICE_REG_SIZE * VOICES
            )
            df = self._rotate_filter(df, r)
            df = df[orig_df.columns]
            yield df

    def _add_frame_reg(self, df, diffmax):
        irqshift = df["irq"].shift().astype(MODEL_PDTYPE)
        irqdiff = df["irq"] - irqshift
        m = irqdiff > diffmax
        try:
            irq = int(irqdiff[m].value_counts().nlargest(1).index[0])
        except IndexError:
            irq = 0
        irq_df = df[m].copy()
        irq_df["clock"] = irq_df["clock"] - 1
        irq_df.loc[:, ["reg"]] = FRAME_REG
        irq_df.loc[:, ["val"]] = 0
        df = pd.concat([df, irq_df]).sort_values(["clock"])[["clock", "reg", "val"]]
        df = df.reset_index(drop=True)
        return irq, df

    def _drop_subdiff(self, df, irq):
        irqdiff = (df["diff"] - irq).abs()
        return df[(df["reg"] != DELAY_REG) | ((irqdiff >= irq) & (df["diff"] > irq))]

    def _rand_voice_order(self, orig_df):
        df = orig_df.copy()
        permutations = sorted(itertools.permutations(range(VOICES)))
        m = df["reg"] == FRAME_REG
        rng = np.random.default_rng()
        df.loc[m, "rand_slice"] = rng.integers(
            low=0, high=len(permutations), size=len(df[m])
        )
        df["rand_slice"] = df["rand_slice"].ffill().bfill().astype(MODEL_PDTYPE)
        df["f"] = m.astype(MODEL_PDTYPE)
        df["f"] = df["f"].cumsum()
        df["v"] = df["reg"].floordiv(VOICE_REG_SIZE)
        df.loc[df["v"] < 0, "v"] = int(-1)
        df["r"] = df["reg"]
        df.loc[df["v"] < 0, "r"] = int(-1)
        perm_dfs = []
        for i, permutation in enumerate(permutations):
            perm_df = df[df["rand_slice"] == i].copy()
            perm_df["new_v"] = perm_df["v"]
            for j, v in enumerate(permutation):
                perm_df.loc[perm_df["v"] == v, "new_v"] = j
            perm_df = perm_df.sort_values(["f", "new_v", "r"], ascending=True)
            perm_df = perm_df.reset_index(drop=True)
            perm_df["i"] = perm_df.index
            perm_dfs.append(perm_df)
        df = pd.concat(perm_dfs).sort_values(["f", "i"], ascending=True)
        df = df[orig_df.columns].reset_index(drop=True)
        return df

    def derange_voiceorder(self, max_perm=99):
        voices = list(range(VOICES))
        permutations = [voices]
        for p in sorted(itertools.permutations(voices)):
            if all(i != p[j] for j, i in enumerate(voices)):
                permutations.append(p)
        return permutations[:max_perm]

    def _norm_reg_order(self, orig_df, max_perm=99):
        if max_perm == 0:
            yield orig_df
        for order in self.derange_voiceorder(max_perm):
            df = orig_df.copy()
            df["i"] = df.index
            df["f"] = df["reg"] == FRAME_REG
            df["f"] = df["f"].astype(MODEL_PDTYPE).cumsum()
            df["v"] = df["reg"].floordiv(VOICE_REG_SIZE)
            df["r"] = df["v"]
            df["reg_order"] = df["reg"]
            df.loc[self._ctrl_match(df), "reg_order"] = int(99)
            df.loc[df["r"] < 0, "r"] = int(999)
            df.loc[df["reg"] == FRAME_REG, "r"] = int(9999)
            for i, j in zip(order, range(VOICES)):
                df.loc[df["v"] == i, "r"] = j
            df = df.sort_values(["f", "r", "reg_order", "i"], ascending=True)
            df = df[orig_df.columns].reset_index(drop=True)
            yield df

    def _split_reg(self, orig_df, reg):
        df = orig_df.copy().reset_index(drop=True)
        df["f"] = df["reg"] == FRAME_REG
        df["f"] = df["f"].cumsum().astype(MODEL_PDTYPE)
        df["fs"] = df.index
        df["prev_f"] = df["f"].shift(1).fillna(-1)
        df.loc[df["f"] == df["prev_f"], "fs"] = pd.NA
        df["fs"] = df["fs"].ffill()
        df["reg_order"] = df.index - df["fs"]
        m = df["reg"] == reg
        reg_df = df[m].copy()
        reg_df["val"] = reg_df["val"].floordiv(256)
        reg_df.loc[:, "reg"] += 1
        df.loc[m, "val"] -= reg_df["val"] * 256
        df = pd.concat([df, reg_df]).sort_values(["f", "reg_order"], ascending=True)
        df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        return df

    def _norm_voice_reg_order(self, orig_df, diffmax=512):
        df = orig_df.copy()
        assert df[df["reg"] == VOICE_REG].empty
        m = df["reg"] == FRAME_REG
        dmax = df[m]["diff"].max()
        df.loc[m, "diff"] = diffmax * 2
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            df = self._combine_reg(df, reg=v_offset, bits=1)
            df = self._combine_reg(df, reg=(v_offset + 2), bits=4)
        df = self._combine_reg(df, 21, bits=1)
        df.loc[df["reg"] == FRAME_REG, "diff"] = dmax
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for reg in (0, 2):
                reg = reg + v_offset
                df = self._split_reg(df, reg)
        df = self._split_reg(df, 21)
        return df

    def _add_voice_reg(self, orig_df):
        df = orig_df.copy()
        df["v"] = pd.NA
        df.loc[(df["reg"] >= 0) & (df["reg"] < (VOICES * VOICE_REG_SIZE)), "v"] = df[
            "reg"
        ].floordiv(VOICE_REG_SIZE)
        df.loc[df["v"] >= 0, "reg"] = df["reg"].mod(VOICE_REG_SIZE)
        df.loc[df["reg"] == FRAME_REG, "v"] = 0
        df["v"] = df["v"].astype(MODEL_PDTYPE).ffill()
        df["i"] = df.index * 2

        last_v = df.copy()
        last_v["last_v"] = df["v"].shift().fillna(0).astype(MODEL_PDTYPE)
        last_v = last_v[
            (last_v["reg"] != FRAME_REG) & (last_v["v"] != last_v["last_v"])
        ]
        last_v["i"] -= 1
        last_v["reg"] = VOICE_REG
        last_v["val"] = last_v["v"]

        df = pd.concat([last_v, df]).sort_values(["i"]).reset_index(drop=True)
        df = df[orig_df.columns].astype(orig_df.dtypes)
        return df

    def _consolidate_delays(self, orig_df, irq):
        df = orig_df.copy()
        val = (df["diff"] / float(irq)).round(0).astype(VAL_PDTYPE)
        df.loc[df["reg"] == DELAY_REG, "val"] = val
        df = df[~((df["reg"] == DELAY_REG) & (df["val"] == 0))]
        df["i"] = df.index * 10
        m = df["reg"] == DELAY_REG
        df_delay = df[m].copy()
        df_frames = df_delay.copy()
        df_nodelay = df[~m].copy()
        df_delay["val"] -= 1
        df_delay["diff"] = df_delay["val"] * irq
        df_frames.loc[:, "reg"] = FRAME_REG
        df_frames.loc[:, "val"] = 0
        df_frames.loc[:, "diff"] = irq
        pre_df_frames = df_frames.copy()
        pre_df_frames["i"] -= 5
        # post_df_frames = df_frames.copy()
        # post_df_frames["i"] += 5
        df = pd.concat([df_delay, df_nodelay, pre_df_frames]).sort_values(
            ["i"], ascending=True
        )
        df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        return df

    def _reduce_val_res(self, df, reg, bits):
        m = df["reg"] == reg
        df.loc[m, "val"] = np.left_shift(np.right_shift(df[m]["val"], bits), bits)
        return df

    def _downsample_df(self, df, diffmin=8, diffmax=512, max_perm=99):
        df = self._squeeze_changes(df)
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for reg, bits in ((v_offset, 2), ((v_offset + 2), 4)):
                df = self._combine_reg(df, reg=reg, bits=bits)
                # df = self._split_reg(df, reg)
        df = self._combine_reg(df, 21, bits=2)
        # df = self._split_reg(df, 21)
        if df.empty:
            return
        irq, df = self._add_frame_reg(df, diffmax)
        df = self._quantize_longdiff(df, diffmin, diffmax)
        df = self._quantize_diff(df)
        # TODO: handle short delays
        df = self._drop_subdiff(df, irq)
        df = self._consolidate_delays(df, irq)
        for xdf in self._norm_reg_order(df, max_perm=max_perm):
            xdf = self._add_voice_reg(xdf)
            xdf.loc[xdf["reg"] < 0, "diff"] = 8
            xdf["irq"] = irq
            xdf = xdf[TOKEN_KEYS + ["irq"]].astype(
                {
                    "reg": REG_PDTYPE,
                    "val": VAL_PDTYPE,
                    "diff": MODEL_PDTYPE,
                    "irq": MODEL_PDTYPE,
                }
            )
            if xdf.iloc[-1]["reg"] == FRAME_REG:
                xdf = xdf.head(len(xdf) - 1)
            if xdf.iloc[0]["reg"] == FRAME_REG:
                xdf = xdf.tail(len(xdf) - 1)
            yield xdf

    def get_reg_widths(self, dfs):
        reg_widths = {}
        unique_regs = set()
        for df in dfs:
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

    def encode_unicode(self, tokens):
        t = np.array(tokens, dtype=np.uint16)
        t = np.where(t == 0, np.nan, t)
        t += UNICODE_BASE
        t = np.nan_to_num(t).astype(np.uint16)
        t = np.where(t == 0, 32, t)
        return "".join([chr(i) for i in t])

    def decode_unicode(self, encoded_tokens):
        t = np.array([ord(i) for i in encoded_tokens])
        t = np.where(t == 32, np.nan, t)
        t -= UNICODE_BASE
        t = np.nan_to_num(t).astype(np.uint16)
        return t

    def encode(self, tokens, tk=None):
        if self.args.tkvocab:
            if tk is None:
                if self.tk is None:
                    self.tk = self.get_tk(self.args.tkmodel)
                tk = self.tk
            encoded = tk.encode(self.encode_unicode(tokens))
            return np.array(encoded.ids, dtype=np.int16)
        return tokens

    def decode(self, encoded_tokens, tk=None):
        if self.args.tkvocab:
            if tk is None:
                if self.tk is None:
                    self.tk = self.get_tk(self.args.tkmodel)
                tk = self.tk
            return self.decode_unicode(tk.decode(encoded_tokens))
        return encoded_tokens

    def train_tokenizer(self, dfs, min_frequency=2):
        encoded_dfs = []
        for df in dfs:
            orig_seq = df["n"].to_numpy()
            encoded = self.encode_unicode(orig_seq)
            encoded_dfs.append(encoded)
        tk = self.get_tk()
        tk.train_from_iterator(
            encoded_dfs,
            vocab_size=self.args.tkvocab,
            min_frequency=min_frequency,
            limit_alphabet=self.args.tkvocab,
        )
        assert tk.get_vocab_size() == self.args.tkvocab, (
            tk.get_vocab_size(),
            self.args.tkvocab,
        )
        tk.save(self.args.tkmodel)
        del tk

    def load_df(self, name, max_perm=99):
        dfs = []
        for i, df in enumerate(
            self._downsample_df(self._read_df(name), max_perm=max_perm)
        ):
            irq = df["irq"][0]
            if irq < self.args.min_irq or irq > self.args.max_irq:
                self.logger.info("skipped %s, irq %u (outside IRQ range)", name, irq)
                break
            if len(df) < self.args.seq_len:
                self.logger.info("skipped %s, length %u (too short)", name, len(df))
                break
            vol = sorted(
                np.bitwise_and(df[df["reg"] == 24]["val"], 15).unique().tolist()
            )
            if len(vol) >= 8:
                self.logger.info(
                    "skipped %s, too many (%u) vol changes %s", name, len(vol), vol
                )
                break
            self.logger.info("loaded %s, irq %u, augment %u", name, irq, i)
            dfs.append(df)
        return dfs

    def load_dfs(self, dump_files, max_perm=99):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(
                self.load_df,
                sorted(dump_files),
                [max_perm for _ in range(len(dump_files))],
            )
        dfs = []
        df_files = []
        for name, file_dfs in zip(sorted(dump_files), results):
            for file_df in file_dfs:
                df_files.append(name)
                dfs.append(file_df)
        return df_files, dfs

    def _merged_and_missing(self, tokens, df):
        df = df.merge(tokens, on=TOKEN_KEYS, how="left")
        missing_tokens = (
            df[df["n"].isna()].drop_duplicates().sort_values(["reg", "val"])
        )
        return df, missing_tokens

    def merge_tokens(self, tokens, dfs):
        merged_dfs = []
        for df in dfs:
            orig_df = df.copy()
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
                df = df[orig_df.columns].astype(orig_df.dtypes)
                df, missing_tokens = self._merged_and_missing(tokens, df)
                assert missing_tokens.empty
            merged_dfs.append(df)
        return merged_dfs

    def glob_dumps(self, reglogs, max_files):
        random.seed(0)
        dump_files = []
        for reglogs in reglogs.split(","):
            globbed = list(glob.glob(reglogs))
            while len(dump_files) < max_files and globbed:
                file = random.choice(globbed)
                globbed.remove(file)
                dump_files.append(file)
        random.seed()
        return dump_files

    def load(self, train=True):
        if self.args.reglog:
            self.tokens = pd.read_csv(
                self.args.token_csv, dtype=MODEL_PDTYPE, index_col=0
            )
            df_files, self.dfs = self.load_dfs(
                [self.args.reglog],
                max_perm=self.args.max_perm,
            )
            self.dfs = self.merge_tokens(self.tokens, self.dfs)
        else:
            dump_files = self.glob_dumps(self.args.reglogs, self.args.max_files)
            df_files, self.dfs = self.load_dfs(dump_files, max_perm=self.args.max_perm)
            _token_df_files, token_dfs = self.load_dfs(
                self.glob_dumps(
                    self.args.token_reglogs,
                    self.args.max_files,
                ),
                max_perm=self.args.max_perm,
            )
            self.tokens = self._make_tokens(self.dfs + token_dfs)
            self.dfs = self.merge_tokens(self.tokens, self.dfs)
            token_dfs = self.merge_tokens(self.tokens, token_dfs)
            if train:
                if self.args.token_csv:
                    self.logger.info("writing %s", self.args.token_csv)
                    self.tokens.to_csv(self.args.token_csv)
                if self.args.tkvocab:
                    self.train_tokenizer(self.dfs + token_dfs)
        self.reg_widths = self.get_reg_widths(self.dfs)
        self.n_vocab = len(self.tokens["n"])
        self.n_words = sum([len(df) for df in self.dfs])
        assert self.tokens[self.tokens["val"].isna()].empty
        assert self.tokens[self.tokens["val"] < 0].empty
        self.logger.info(
            f"n_vocab: {self.n_vocab}, n_words {self.n_words}, reg widths {sorted(self.reg_widths.items())}"
        )
        dfs = self.dfs
        self.dfs = []
        if self.args.tkvocab:
            self.n_vocab = self.args.tkvocab
        self.n_words = 0
        for df_file, df in zip(df_files, dfs):
            seq = self.encode(df["n"])
            decoded_seq = self.decode(seq)
            orig_seq = df["n"].to_numpy()
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
            try:
                self.seq_mapper.add(seq)
                self.n_words += len(seq)
                self.df_files.append(df_file)
                self.dfs.append(df)
            except ValueError:
                self.logger.info(
                    "rejecting sequence from %s too short %u", df_file, len(seq)
                )
        self.logger.info(
            f"n_encoded_words {self.n_words}, {len(dfs)} sequences",
        )
        if train:
            if self.args.df_map_csv:
                pd.DataFrame(self.df_files, columns=["dump_file"]).to_csv(
                    self.args.df_map_csv
                )
            if self.args.dataset_csv:
                for i in range(len(self.dfs)):
                    self.dfs[i]["i"] = int(i)
                pd.DataFrame(pd.concat(self.dfs)).to_csv(self.args.dataset_csv)

    def get_tk(self, tkmodel=None):
        vocab = None
        merges = None
        if tkmodel:
            self.logger.info("reading tokenizer from %s", tkmodel)
            with open(tkmodel) as f:
                model = json.load(f)["model"]
            vocab = model["vocab"]
            merges = [tuple(x) for x in model["merges"]]
        return CharBPETokenizer(
            split_on_whitespace_only=True,
            bert_normalizer=False,
            vocab=vocab,
            merges=merges,
        )

    def __len__(self):
        return len(self.seq_mapper)

    def __getitem__(self, index):
        return self.seq_mapper[index]

    def getseq(self, i):
        return self.seq_mapper.seqs[i]


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
        num_workers=1,
    )
