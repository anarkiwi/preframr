import glob
import itertools
import logging
import numpy as np
import pandas as pd
from preframr.stfconstants import (
    DELAY_REG,
    DIFF_PDTYPE,
    FILTER_REG,
    FRAME_REG,
    MAX_REG,
    MIDI_N_TO_F,
    MODE_VOL_REG,
    MODEL_PDTYPE,
    PAL_CLOCK,
    REG_PDTYPE,
    VAL_PDTYPE,
    VOICES,
    VOICE_REG,
    VOICE_REG_SIZE,
)

REG_PDTYPE = pd.Int8Dtype()
IRQ_PDTYPE = pd.UInt16Dtype()
MIN_DIFF = 32
FRAME_DTYPES = {
    "reg": REG_PDTYPE,
    "val": VAL_PDTYPE,
    "diff": DIFF_PDTYPE,
    "irq": IRQ_PDTYPE,
}


def wrapbits(x, reglen):
    base = (x << 1) & (2**reglen - 1)
    lsb = (x >> (reglen - 1)) & 1
    return base ^ lsb


FILTER_SHIFT_DF = pd.DataFrame(
    [{"reg": FILTER_REG, "val": i, "y": wrapbits(i, 3)} for i in range(2**3)],
    dtype=MODEL_PDTYPE,
)


class RegLogParser:
    def __init__(self, args, logger=logging):
        self.args = args
        self.logger = logger

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
        irq_df.loc[irq_df["reg"] == DELAY_REG, "diff"] = 0
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

        ordreg = "m"
        df[ordreg] = df[self._freq_match(df)]["val"]
        df[ordreg] = df[ordreg].ffill()
        df.loc[~df["v"].isin(set(range(VOICES))), ordreg] = pd.NA
        df.loc[df["reg"] < 0, ordreg] = df[df["reg"] < 0]["reg"]

        df = df.sort_values(["f", ordreg, "v", "reg", "n"])
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

        df = pd.concat([norm_df, df]).sort_values(["n"]).reset_index(drop=True)
        df["nr"] = df["reg"].shift(-1)
        df["nval"] = df["val"].shift(-1)
        df["pr"] = df["reg"].shift(1)
        df.loc[((df["reg"] == FRAME_REG) & (df["nr"] == VOICE_REG)), "val"] = df["nval"]
        df = df[~((df["reg"] == VOICE_REG) & (df["pr"] == FRAME_REG))]
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

    def _filter(self, df, name):
        try:
            irq = df["irq"].iloc[0]
        except KeyError:
            self.logger.info(df)
            self.logger.info("skipped %s, no irq", name)
            return False
        if irq < self.args.min_irq or irq > self.args.max_irq:
            self.logger.info("skipped %s, irq %u (outside IRQ range)", name, irq)
            return False
        if len(df[df["reg"] == FRAME_REG]) == 0:
            self.logger.info("skipped %s, no frames", name)
            return False
        if len(df) < self.args.seq_len:
            self.logger.info("skipped %s, length %u (too short)", name, len(df))
            return False
        vol = sorted(np.bitwise_and(df[df["reg"] == 24]["val"], 15).unique().tolist())
        if len(vol) >= 8:
            self.logger.info(
                "skipped %s, too many (%u) vol changes %s", name, len(vol), vol
            )
            return False
        return True

    def parse(self, name, diffmax=512, max_perm=99):
        parquet_glob = glob.glob(name.replace(".dump.zst", ".*parquet"))
        if parquet_glob:
            for parquet_name in sorted(parquet_glob):
                df = pd.read_parquet(parquet_name)
                if self._filter(df, parquet_name):
                    self.logger.info("returning pre-parsed %s", parquet_name)
                    yield df
            return
        df = self._read_df(name)
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
            while not xdf.empty and self._frame_reg(xdf.iloc[-1]):
                xdf = xdf.head(len(xdf) - 1)
            while not xdf.empty and (
                self._frame_reg(xdf.iloc[0])
                or (xdf.iloc[0]["reg"] == MODE_VOL_REG and xdf.iloc[0]["val"] == 15)
            ):
                xdf = xdf.tail(len(xdf) - 1)
            if xdf.empty:
                continue
            # xdf = self._combine_freq_ctrl(xdf)
            xdf = self._add_voice_reg(xdf)
            xdf = xdf.reset_index(drop=True)
            if self._filter(xdf, name):
                yield xdf
