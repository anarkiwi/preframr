import concurrent.futures
import difflib
import itertools
import logging
import glob
import os
import random
import shutil
import tempfile
import time
from tqdm import tqdm
import torch
from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers
import numpy as np
import pandas as pd
import zstandard as zstd
from preframr.stfconstants import (
    DELAY_REG,
    FILTER_REG,
    FRAME_REG,
    IMPLIED_FRAME_REG,
    MAX_REG,
    MIDI_N_TO_F,
    MODE_VOL_REG,
    PAL_CLOCK,
    UNICODE_BASE,
    VOICES,
    VOICE_REG,
    VOICE_REG_SIZE,
)

TOKEN_KEYS = ["reg", "val", "diff"]
MODEL_PDTYPE = pd.Int32Dtype()
REG_PDTYPE = pd.Int8Dtype()
VAL_PDTYPE = pd.UInt32Dtype()
TOKEN_PDTYPE = pd.Int64Dtype()  # Same as torch
DIFF_PDTYPE = pd.UInt16Dtype()
IRQ_PDTYPE = pd.UInt16Dtype()
MIN_DIFF = 32
FRAME_DTYPES = {
    "reg": REG_PDTYPE,
    "val": VAL_PDTYPE,
    "diff": DIFF_PDTYPE,
    "irq": IRQ_PDTYPE,
}
UNK_TOKEN = "<unk>"
END_OF_WORD_SUFFIX = "</w>"


def wrapbits(x, reglen):
    base = (x << 1) & (2**reglen - 1)
    lsb = (x >> (reglen - 1)) & 1
    return base ^ lsb


FILTER_SHIFT_DF = pd.DataFrame(
    [{"reg": FILTER_REG, "val": i, "y": wrapbits(i, 3)} for i in range(2**3)],
    dtype=MODEL_PDTYPE,
)


def remove_voice_reg(orig_df, reg_widths):
    voice_regs = len(orig_df[orig_df["reg"] == VOICE_REG])
    if voice_regs:
        df = orig_df.copy()
        df["vr"] = pd.NA
        df.loc[df["reg"] == VOICE_REG, "vr"] = df["val"]
        df.loc[df["reg"].isin({FRAME_REG, DELAY_REG}), "vr"] = 0
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


def state_df(states, dataset, irq):
    tokens = dataset.tokens.copy()
    tokens.loc[tokens["reg"] == FRAME_REG, "diff"] = irq
    df = pd.DataFrame(states, columns=["n"]).merge(tokens, on="n", how="left")
    return df


def get_prompt(args, dataset, logger):
    seq = dataset.getseq(args.start_seq)
    if args.start_n is None:
        start = random.randint(0, len(seq))
    else:
        start = args.start_n
    logger.info("starting at %u / %u", start, len(seq))
    n = args.max_seq_len - args.prompt_seq_len
    if n <= 0:
        raise ValueError("max seq length too short")
    prompt = seq[start:][: args.prompt_seq_len].unsqueeze(0)
    prompt_compare = seq[start:][: args.max_seq_len]
    irq = int(dataset.dfs[args.start_seq]["irq"].iat[start])
    preamble_df, _reg_widths = remove_voice_reg(
        state_df(dataset.decode(seq[:start].numpy()), dataset, irq),
        dataset.reg_widths,
    )
    reg_start = {
        r: preamble_df[preamble_df["reg"] == r]["val"].iat[-1]
        for r in preamble_df["reg"].unique()
        if r >= 0
    }
    return irq, n, prompt, prompt_compare, reg_start


class SeqMapper:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.seq_map = None
        self.seqs = []
        self.len = 0

    def add(self, seq):
        if len(seq) <= self.seq_len:
            raise ValueError(f"sequence too short ({len(seq)}")
        # TODO: move to memmap array.
        if isinstance(seq, np.ndarray):
            self.seqs.append(torch.from_numpy(seq))
        else:
            self.seqs.append(torch.LongTensor(seq))
        self.len = 0
        seq_map = []
        for s in self.seqs:
            seq_map.append(self.len)
            self.len += len(s) - self.seq_len
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

    def _freq_match(self, df):
        return (df["reg"] == 0) | (df["reg"] == 7) | (df["reg"] == 14)

    def _frame_match(self, df):
        return (df["reg"] == FRAME_REG) | (df["reg"] == DELAY_REG)

    def _frame_reg(self, df):
        return self._frame_match(df).cumsum()

    def _read_df(self, name):
        try:
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
        except Exception as e:
            raise ValueError(f"cannot read {name}: {e}")
        # assert df["reg"].min() >= 0
        df["irq"] = df["clock"].astype(MODEL_PDTYPE) - df["irq_diff"]
        # keep only chipno 0
        df = df[df["chipno"] == 0]
        df = df[df["reg"] <= MAX_REG]
        df = df[["clock", "irq", "reg", "val"]]
        return df

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
        df = pd.concat([non_reg_df, reg_df], copy=False).sort_values(
            ["clock"], ascending=True
        )
        df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        return df

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

    def _rotate_voice_augment(self, orig_df, max_perm):
        if not max_perm:
            yield orig_df
            return
        for r in range(min(VOICES, max_perm)):
            df = orig_df.copy()
            m = (df["reg"] < VOICE_REG_SIZE * VOICES) & (df["reg"] >= 0)
            df.loc[m, "reg"] = (df[m]["reg"] + (VOICE_REG_SIZE * r)).mod(
                VOICE_REG_SIZE * VOICES
            )
            df = self._rotate_filter(df, r)
            df = df[orig_df.columns]
            yield df

    def _add_frame_reg(self, orig_df, diffmax):
        df = orig_df.copy()
        df["irqdiff"] = df["irq"].diff().fillna(0).astype(MODEL_PDTYPE)
        df["diff"] = MIN_DIFF
        df["i"] = df.index * 10
        m = df["irqdiff"] > diffmax
        try:
            irq = int(df["irqdiff"][m].value_counts().nlargest(1).index[0])
        except IndexError:
            irq = 0
        irq_df = df[m].copy()
        irq_df["i"] -= 1
        irq_df["reg"] = FRAME_REG
        irq_df["diff"] = irq_df["irqdiff"]
        irq_df["val"] = (irq_df["diff"] / irq).astype(MODEL_PDTYPE)
        irq_df.loc[irq_df["val"] > 1, "reg"] = DELAY_REG
        irq_df.loc[irq_df["reg"] == FRAME_REG, "val"] = 0
        irq_df["diff"] = irq
        df = (
            pd.concat([df, irq_df], copy=False)
            .sort_values(["i"])[["reg", "val", "diff", "i"]]
            .astype(MODEL_PDTYPE)
            .reset_index(drop=True)
        )
        return irq, df[["reg", "val", "diff"]]

    def _drop_implied_frame_reg(self, orig_df):
        df = orig_df.copy()
        df["f"] = self._frame_reg(df)
        fr = df.groupby("f")["reg"].unique()
        df = df.merge(fr, on="f", how="left", suffixes=("", "_y")).rename(
            columns={"reg_y": "fr"}
        )
        fr_df = df[df["reg"] == FRAME_REG].reset_index(drop=True)
        fr_df["fr"] = fr_df["fr"].apply(
            lambda x: set(x) - {DELAY_REG, FRAME_REG, 4, 11, 18}
        )
        fr_df["fr_d"] = fr_df["fr"].shift()
        fr_df.at[0, "fr_d"] = {}
        fr_df["fr_s"] = fr_df.apply(lambda row: row["fr"].issubset(row["fr_d"]), axis=1)
        df = df.merge(fr_df[["f", "fr_s"]], on="f", how="left")
        df = df[~((df["reg"] == FRAME_REG) & df["fr_s"])]
        return df[orig_df.columns].reset_index(drop=True)

    def derange_voiceorder(self, max_perm=99):
        voices = list(range(VOICES))
        permutations = [voices]
        for p in sorted(itertools.permutations(voices)):
            if all(i != p[j] for j, i in enumerate(voices)):
                permutations.append(p)
        return permutations[:max_perm]

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
        df = pd.concat([df, reg_df], copy=False).sort_values(
            ["f", "reg_order"], ascending=True
        )
        df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        return df

    def _reduce_val_res(self, df, reg, bits):
        m = df["reg"] == reg
        df.loc[m, "val"] = np.left_shift(np.right_shift(df[m]["val"], bits), bits)
        return df

    def _quantize_freq_to_cents(self, df, cents=50, clock=PAL_CLOCK):
        f = MIDI_N_TO_F[0]
        sid_clock = (18 * 2**24) / clock
        max_sid_f = 65535 / sid_clock
        rq_map = {i: 0 for i in range(65536)}

        while True:
            l = f * (2 ** ((-cents / 2) / 1200))
            h = f * (2 ** ((cents / 2) / 1200))
            lr = round(sid_clock * l)
            lh = round(sid_clock * h)
            r = round(sid_clock * f)
            for i in range(lh - lr):
                rq_map[i + lr] = r
            f *= 2 ** (cents / 1200)
            if f > max_sid_f:
                break

        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            cond = df["reg"] == v_offset
            df.loc[cond, "val"] = df[cond]["val"].map(rq_map)

        return df

    def _squeeze_frames(self, orig_df):
        df = orig_df.copy()
        df["f"] = self._frame_reg(df)
        cm = self._ctrl_match(df).astype(int)
        df["c"] = cm * cm.cumsum()
        df = df.drop_duplicates(["f", "c", "reg"], keep="last")
        return df[orig_df.columns].reset_index(drop=True)

    def _norm_df(self, orig_df):
        norm_df = orig_df.copy().reset_index(drop=True)
        norm_df["f"] = self._frame_reg(norm_df)
        norm_df["v"] = norm_df["reg"].floordiv(VOICE_REG_SIZE).astype(int)
        norm_df["n"] = norm_df.index * 10
        norm_df.loc[norm_df["f"].diff() != 0, "v"] = 0
        norm_df["vd"] = norm_df["v"].diff().astype(MODEL_PDTYPE).fillna(0)
        return norm_df

    def _norm_pr_order(self, orig_df):
        norm_df = self._norm_df(orig_df)

        df = norm_df.copy()
        df = df.sort_values(["f", "v", "reg", "n"])
        df["m"] = df[self._freq_match(df)]["val"]
        df["m"] = df["m"].ffill()
        df.loc[~df["v"].isin(set(range(VOICES))), "m"] = pd.NA
        df.loc[df["reg"] < 0, "m"] = df[df["reg"] < 0]["reg"]

        df = df.sort_values(["f", "m", "v", "reg", "n"])
        df = df[orig_df.columns].reset_index(drop=True)
        return df

    def _add_voice_reg(self, orig_df):
        norm_df = self._norm_df(orig_df)
        m = (norm_df["reg"] >= 0) & (norm_df["v"].isin(set(range(VOICES))))

        norm_df.loc[m, "reg"] = norm_df[m]["reg"] % VOICE_REG_SIZE

        df = norm_df[(norm_df["vd"] != 0) & m].copy()
        df["n"] -= 1
        df["val"] = df["v"]
        df["reg"] = VOICE_REG

        df = pd.concat([norm_df, df]).sort_values(["n"])
        df = df[orig_df.columns].reset_index(drop=True)
        return df

    def _simplify_ctrl(self, orig_df):
        df = orig_df.copy()
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            ctrl_reg = v_offset + 4
            # if no triangle, turn off ring
            df.loc[(df["reg"] == ctrl_reg) & (df["val"] & 0b00010000 == 0), "val"] = (
                df["val"] & 0b11111011
            )
            # if no waveform, turn off sync
            df.loc[(df["reg"] == ctrl_reg) & (df["val"] & 0b11110000 == 0), "val"] = (
                df["val"] & 0b11111101
            )
        return df

    def _combine_freq_ctrl(self, orig_df):
        norm_df = self._norm_df(orig_df.copy())
        for v in range(VOICES):
            col = f"v{v}"
            v_offset = v * VOICE_REG_SIZE
            ctrl_reg = v_offset + 4
            v_df = norm_df.copy()
            v_df[col] = pd.NA
            m = v_df["reg"] == ctrl_reg
            v_df.loc[m, col] = v_df[m]["val"] & 0b11110000
            v_df[col] = v_df[col].astype(MODEL_PDTYPE).ffill().fillna(0)
            v_df = (
                v_df[["f", col]]
                .sort_values(["f"])
                .drop_duplicates(["f"], keep="last")
                .reset_index(drop=True)
            )
            norm_df = norm_df.merge(v_df, on="f")
        for v in range(VOICES):
            col = f"v{v}"
            v_offset = v * VOICE_REG_SIZE
            f_reg = v_offset
            m = norm_df["reg"] == f_reg
            norm_df.loc[m, "val"] = (
                np.left_shift(norm_df[m]["val"], 8) + norm_df[m][col]
            )
        df = norm_df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        return df

    def _downsample_df(self, df, diffmax=512, max_perm=99):
        df = self._squeeze_changes(df)
        df = self._simplify_ctrl(df)
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for reg, bits in ((v_offset, 0), ((v_offset + 2), 4)):
                df = self._combine_reg(df, reg=reg, bits=bits)
        df = self._quantize_freq_to_cents(df)
        df = self._combine_reg(df, 21, bits=2)
        df = self._squeeze_changes(df)
        if df.empty:
            return
        irq, df = self._add_frame_reg(df, diffmax)
        irq = min(2 ** (IRQ_PDTYPE.itemsize * 8) - 1, irq)
        df = self._squeeze_frames(df)
        for a_xdf in self._rotate_voice_augment(df, max_perm):
            xdf = self._norm_pr_order(a_xdf)
            xdf["irq"] = irq
            xdf = xdf[FRAME_DTYPES.keys()].astype(FRAME_DTYPES)
            while self._frame_reg(xdf.iloc[-1]):
                xdf = xdf.head(len(xdf) - 1)
            while self._frame_reg(xdf.iloc[0]) or (
                xdf.iloc[0]["reg"] == MODE_VOL_REG and xdf.iloc[0]["val"] == 15
            ):
                xdf = xdf.tail(len(xdf) - 1)
            # xdf = self._combine_freq_ctrl(xdf)
            xdf = self._add_voice_reg(xdf)
            xdf = xdf.reset_index(drop=True)
            yield xdf

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

    def encode(self, tokens, tk=None, dtype=np.uint16):
        if self.args.tkvocab:
            if tk is None:
                if self.tk is None:
                    self.tk, _ = self.get_tk(self.args.tkmodel)
                tk = self.tk
            encoded = tk.encode(self.encode_unicode(tokens, dtype=dtype))
            return np.array(encoded.ids, dtype=dtype)
        return tokens

    def decode(self, encoded_tokens, tk=None, dtype=np.uint16):
        if self.args.tkvocab:
            if tk is None:
                if self.tk is None:
                    self.tk, _ = self.get_tk(self.args.tkmodel)
                tk = self.tk
            return self.decode_unicode(tk.decode(encoded_tokens), dtype=dtype)
        return encoded_tokens

    def train_tokenizer(self, dfs, min_frequency=2, tokenizer="unigram"):
        encoded_dfs = []
        for df in dfs:
            orig_seq = df["n"].to_numpy()
            encoded = self.encode_unicode(orig_seq)
            encoded_dfs.append(encoded)
        tk, trainer = self.get_tk(tokenizer=tokenizer)
        tk.train_from_iterator(encoded_dfs, trainer=trainer)
        assert tk.get_vocab_size() == self.args.tkvocab, (
            tk.get_vocab_size(),
            self.args.tkvocab,
        )
        tk.save(self.args.tkmodel)
        del tk

    def load_df(self, name, df_dir, max_perm=99, min_space=0):
        dfs = []
        try:
            for i, df in enumerate(
                self._downsample_df(self._read_df(name), max_perm=max_perm)
            ):
                try:
                    irq = df["irq"].iloc[0]
                except KeyError:
                    self.logger.info("skipped %s, no irq", name)
                    break
                if irq < self.args.min_irq or irq > self.args.max_irq:
                    self.logger.info(
                        "skipped %s, irq %u (outside IRQ range)", name, irq
                    )
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
                if min_space:
                    while True:
                        usage = shutil.disk_usage(df_dir)
                        prop = usage.free / usage.total
                        if prop > min_space:
                            break
                        time.sleep(1)
                df_base = os.path.splitext(os.path.basename(name))[0]
                df_name = os.path.join(df_dir, f"{hash(name)}-{df_base}.{i}.zst")
                df.to_parquet(df_name, engine="pyarrow", compression="zstd")
                dfs.append(df_name)
        except Exception as e:
            raise ValueError(f"cannot read {name}: {e}")
        return name, dfs

    def load_dfs(
        self, dump_files, max_perm=99, max_workers=16, min_space=0.2, shuffle=0
    ):
        results = []
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
            unsorted_dump_files = dump_files
            random.shuffle(unsorted_dump_files)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as preload_executor:
                preload_futures = [
                    preload_executor.submit(
                        self.load_df,
                        dump_file,
                        tmpdir,
                        max_perm=max_perm,
                        min_space=min_space,
                    )
                    for dump_file in unsorted_dump_files
                ]

                def load_df(name, file_dfs):
                    if not file_dfs:
                        return []
                    dfs = []
                    for file_df in file_dfs:
                        try:
                            dfs.append(pd.read_parquet(file_df))
                        except Exception as e:
                            raise ValueError(f"cannot read {file_df}: {e}")
                    irq = dfs[0]["irq"].iloc[0]
                    for i, file_df in enumerate(file_dfs):
                        os.unlink(file_df)
                        self.logger.info("loaded %s, irq %u, augment %u", name, irq, i)
                    return dfs

                for preload_future in tqdm(
                    concurrent.futures.as_completed(preload_futures),
                    total=len(preload_futures),
                    ascii=True,
                ):
                    assert not preload_future.exception(), preload_future.exception()
                    name, file_dfs = preload_future.result()
                    dfs = load_df(name, file_dfs)
                    for df in dfs:
                        results.append((name, df))
        if shuffle is not None:
            random.seed(shuffle)
            random.shuffle(results)
            random.seed()
        else:
            results = sorted(results, key=lambda x: x[0])
        df_files = [result[0] for result in results]
        dfs = [result[1] for result in results]
        return df_files, dfs

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

    def glob_dumps(self, reglogs, max_files, min_dump_size):
        random.seed(0)
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

    def load(self, tokens=None):
        if self.args.reglog:
            df_files, self.dfs = self.load_dfs(
                [self.args.reglog],
                max_perm=self.args.max_perm,
            )
            if tokens is None:
                tokens = self._make_tokens(self.dfs)
            self.tokens = tokens
            self.dfs = self.merge_tokens(self.tokens, self.dfs)
        else:
            dump_files = self.glob_dumps(
                self.args.reglogs, self.args.max_files, self.args.min_dump_size
            )
            df_files, self.dfs = self.load_dfs(dump_files, max_perm=self.args.max_perm)
            _token_df_files, token_dfs = self.load_dfs(
                self.glob_dumps(
                    self.args.token_reglogs,
                    self.args.max_files,
                    self.args.min_dump_size,
                ),
                max_perm=self.args.max_perm,
            )
            self.tokens = self._make_tokens(self.dfs + token_dfs)
            self.dfs = self.merge_tokens(self.tokens, self.dfs)
            token_dfs = self.merge_tokens(self.tokens, token_dfs)
            if self.args.token_csv:
                self.logger.info("writing %s", self.args.token_csv)
                self.tokens.to_csv(self.args.token_csv)
            if self.args.tkvocab:
                self.train_tokenizer(self.dfs + token_dfs)
        self.logger.info("getting reg widths")
        self.reg_widths = self.get_reg_widths(self.dfs)
        self.n_vocab = len(self.tokens["n"])
        self.n_words = sum((len(df) for df in self.dfs))
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
        self.logger.info("mapping sequences")
        for df_file, df in tqdm(zip(df_files, dfs), ascii=True):
            seq = self.validate_encoding(df_file, df["n"].to_numpy())
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
        if not self.args.reglog:
            if self.args.df_map_csv:
                self.logger.info(f"writing {self.args.df_map_csv}")
                pd.DataFrame(self.df_files, columns=["dump_file"]).to_csv(
                    self.args.df_map_csv
                )
            if self.args.dataset_csv:
                self.logger.info(f"writing {self.args.dataset_csv}")
                with zstd.open(self.args.dataset_csv, "w") as f:
                    for i in tqdm(range(len(self.dfs)), ascii=True):
                        self.dfs[i]["i"] = int(i)
                        self.dfs[i].to_csv(f, index=False, header=(i == 0))

    def get_tk(self, tkmodel=None, min_frequency=2, tokenizer="unigram"):
        if tkmodel:
            self.logger.info("reading tokenizer from %s", tkmodel)
            return Tokenizer.from_file(tkmodel), None
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
                min_frequency=min_frequency,
                special_tokens=[UNK_TOKEN],
                limit_alphabet=self.args.tkvocab,
                initial_alphabet=[],
                end_of_word_suffix=END_OF_WORD_SUFFIX,
                show_progress=True,
            )
            return tk, trainer
        raise ValueError

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
    )
