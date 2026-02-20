from collections import defaultdict
import glob
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
from preframr.reg_mappers import FreqMapper
from preframr.stfconstants import (
    DELAY_REG,
    DIFF_OP,
    DIFF_PDTYPE,
    DUMP_SUFFIX,
    FC_LO_REG,
    FILTER_REG,
    FLIP_OP,
    FRAME_REG,
    MAX_REG,
    MIN_DIFF,
    MODE_VOL_REG,
    MODEL_PDTYPE,
    PARSED_SUFFIX,
    REG_PDTYPE,
    REPEAT_OP,
    SET_OP,
    VAL_PDTYPE,
    VOICES,
    VOICE_REG,
    VOICE_REG_SIZE,
)

OP_PDTYPE = pd.UInt8Dtype()
REG_PDTYPE = pd.Int8Dtype()
IRQ_PDTYPE = pd.UInt16Dtype()
FRAME_DTYPES = {
    "reg": REG_PDTYPE,
    "val": VAL_PDTYPE,
    "diff": DIFF_PDTYPE,
    "irq": IRQ_PDTYPE,
    "op": OP_PDTYPE,
}

PY_MTIME = Path(__file__).resolve().stat().st_mtime


def wrapbits(x, reglen):
    base = (x << 1) & (2**reglen - 1)
    lsb = (x >> (reglen - 1)) & 1
    return base ^ lsb


FILTER_SHIFT_DF = pd.DataFrame(
    [{"reg": FILTER_REG, "val": i, "y": wrapbits(i, 3)} for i in range(2**3)],
    dtype=MODEL_PDTYPE,
)


def state_df(states, dataset, irq):
    tokens = dataset.tokenizer.tokens.copy()
    tokens["diff"] = MIN_DIFF
    tokens.loc[tokens["reg"] < -MAX_REG, "diff"] = 0
    tokens.loc[tokens["reg"] == FRAME_REG, "diff"] = irq
    df = pd.DataFrame(states, columns=["n"]).merge(tokens, on="n", how="left")
    return df


def remove_voice_reg(orig_df, reg_widths):
    voice_regs = len(orig_df[orig_df["reg"] == VOICE_REG])
    if voice_regs:
        df = orig_df.copy()
        df["vr"] = pd.NA
        df.loc[df["reg"].isin({FRAME_REG, VOICE_REG}), "vr"] = df["val"] & 255
        df.loc[df["reg"] == DELAY_REG, "vr"] = 0
        df["vr"] = df["vr"].astype(pd.UInt8Dtype()).ffill().fillna(0)
        df = df[df["reg"] != VOICE_REG]
        df["vr"] = df["vr"].astype(pd.Int64Dtype()) * VOICE_REG_SIZE
        df.loc[df["reg"] >= VOICE_REG_SIZE, "vr"] = 0
        df.loc[df["reg"] >= 0, "reg"] += df["vr"]

        df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for i in range(VOICE_REG_SIZE):
                if i in reg_widths:
                    reg_widths[v_offset + i] = reg_widths[i]
        return df, reg_widths
    return orig_df, reg_widths


def reset_diffs(orig_df, irq, sidq):
    df = orig_df.copy().reset_index(drop=True)
    frame_cond = df["reg"] == FRAME_REG

    if irq is None:
        irq = df[frame_cond]["diff"].iat[0]

    df.loc[df["reg"] == DELAY_REG, "diff"] = df["val"] * irq
    df["delay"] = df["diff"] * sidq

    df["f"] = (frame_cond).cumsum()
    df["fd"] = df["diff"]
    df.loc[df["reg"] < 0, "fd"] = pd.NA

    df["fd"] = df.groupby(["f"])["fd"].transform("sum") * sidq
    df.loc[frame_cond, "delay"] = df[frame_cond]["delay"] - df[frame_cond][
        "fd"
    ].shift().fillna(0)
    return df


def expand_ops(orig_df, strict):
    df = orig_df.copy()
    last_val = defaultdict(int)
    last_repeat = defaultdict(int)
    last_flip = defaultdict(int)
    last_diff = {}
    for reg in df["reg"].unique():
        reg_df = df[(df["reg"] == reg) & (df["op"] == SET_OP)]["diff"]
        if len(reg_df):
            last_diff[reg] = reg_df.iloc[0]
        else:
            last_diff[reg] = MIN_DIFF

    sid_writes = []

    df["f"] = (
        df["reg"]
        .isin({DELAY_REG, FRAME_REG})
        .astype(MODEL_PDTYPE)
        .cumsum()
        .astype(MODEL_PDTYPE)
    )
    for f, f_df in df.groupby("f"):
        f_sid_writes = []
        for row in f_df.itertuples():
            if row.reg < 0:
                if row.reg not in {DELAY_REG, FRAME_REG}:
                    assert False, f"unknown reg {row.reg}, {row}"
                f_sid_writes.append((row.reg, row.val, row.diff))
                continue

            if row.op == SET_OP:
                last_val[row.reg] = row.val
            elif row.op == DIFF_OP:
                last_val[row.reg] += row.val
            elif row.op == REPEAT_OP:
                if row.val == 0:
                    last_val[row.reg] += last_repeat[row.reg]
                    del last_repeat[row.reg]
                else:
                    if strict:
                        assert row.reg not in last_repeat
                    last_repeat[row.reg] = row.val
                    continue
            elif row.op == FLIP_OP:
                if row.val == 0:
                    last_val[row.reg] += last_flip[row.reg]
                    del last_flip[row.reg]
                else:
                    if strict:
                        assert row.reg not in last_flip
                    last_flip[row.reg] = row.val
                    continue
            else:
                assert False, f"unknown op {row.op}, {row}"

            f_sid_writes.append((row.reg, last_val[row.reg], row.diff))

        for reg, val in last_repeat.items():
            last_val[reg] += val
            f_sid_writes.append((reg, last_val[reg], last_diff[reg]))
        for reg, val in list(last_flip.items()):
            last_val[reg] += val
            last_flip[reg] = -val
            f_sid_writes.append((reg, last_val[reg], last_diff[reg]))
        sid_writes.append(
            pd.DataFrame(
                f_sid_writes, dtype=MODEL_PDTYPE, columns=["reg", "val", "diff"]
            ).sort_values("reg")
        )

    df = pd.concat(sid_writes, ignore_index=True)
    return df


def prepare_df_for_audio(orig_df, reg_widths, irq, sidq, strict=False):
    df = orig_df.copy()
    df, reg_widths = remove_voice_reg(df, reg_widths)
    df = expand_ops(df, strict)
    df = reset_diffs(df, irq, sidq)
    return df, reg_widths


class RegLogParser:
    def __init__(self, args, logger=logging):
        self.args = args
        self.logger = logger
        self.freq_mapper = FreqMapper()

    def _vreg_match(self, vreg):
        return {(v * VOICE_REG_SIZE) + vreg for v in range(VOICES)}

    def _freq_match(self, df):
        return df["reg"].isin(self._vreg_match(0))

    def _pcm_match(self, df):
        return df["reg"].isin(self._vreg_match(2))

    def _ctrl_match(self, df):
        return df["reg"].isin(self._vreg_match(4))

    def _adsr_match(self, df):
        return df["reg"].isin(self._vreg_match(5) | self._vreg_match(6))

    def _ad_match(self, df):
        return df["reg"].isin(self._vreg_match(5))

    def _sr_match(self, df):
        return df["reg"].isin(self._vreg_match(6))

    def _filter_match(self, df):
        return df["reg"] == FC_LO_REG

    def _frame_match(self, df):
        return (df["reg"] == FRAME_REG) | (df["reg"] == DELAY_REG)

    def _frame_reg(self, orig_df):
        df = orig_df[["reg", "val"]].copy()
        df.loc[(df["reg"] == FRAME_REG), "val"] = 1
        m = self._frame_match(df)
        df.loc[~m, "val"] = 0
        return df["val"].cumsum()

    def _read_df(self, name):
        try:
            df = pd.read_parquet(name)
        except Exception as e:
            raise ValueError(f"cannot read {name}: {e}")

        # assert df["reg"].min() >= 0
        # keep only chipno 0
        df = df[df["chipno"] == 0]
        df = df[df["reg"] <= MAX_REG]
        df = df[["clock", "irq", "reg", "val"]]
        df["val"] = df["val"].astype(VAL_PDTYPE)
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
        reg_df["val"] = 0
        reg_df["reg"] = reg
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
            reg_df["val"] = np.right_shift(reg_df["val"], bits)
        df = pd.concat([non_reg_df, reg_df]).sort_values(["clock"], ascending=True)
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
        if orig_df.empty:
            return
        if not max_perm:
            yield orig_df
            return
        for r in range(min(VOICES, max_perm)):
            df = orig_df.copy()
            m = df["reg"].abs() < VOICE_REG_SIZE * VOICES
            df["rreg"] = (df[m]["reg"].abs() + (VOICE_REG_SIZE * r)).mod(
                VOICE_REG_SIZE * VOICES
            )
            df.loc[m, "reg"] = df["rreg"]
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
        irq_df.loc[(irq_df["reg"] == DELAY_REG) & (irq_df["val"] > 255), "val"] = 255
        irq_df.loc[(irq_df["reg"] == DELAY_REG) & (irq_df["val"] > 50), "val"] = (
            irq_df["val"] / 5
        ).astype(MODEL_PDTYPE) * 5
        irq_df.loc[irq_df["reg"] == FRAME_REG, "val"] = 0
        irq_df["diff"] = irq
        irq_df.loc[irq_df["reg"] == DELAY_REG, "diff"] = 0
        df = (
            pd.concat([df, irq_df])
            .sort_values(["i"])[["reg", "val", "diff", "i"]]
            .astype(MODEL_PDTYPE)
            .reset_index(drop=True)
        )
        return irq, df[["reg", "val", "diff"]]

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

    def _last_reg_val_frame(self, orig_df, regs):
        assert not len(orig_df[orig_df["reg"] == VOICE_REG])
        pivot_df = self._norm_df(orig_df.copy())
        pivot_df = (
            pivot_df.pivot(columns="reg", values="val", index=["f", "n", "v"])
            .astype(MODEL_PDTYPE)
            .ffill()
            .fillna(0)
        )
        for reg in regs:
            norm_df = pivot_df.copy()
            if reg < VOICE_REG_SIZE:
                vregs = [v * VOICE_REG_SIZE + reg for v in range(VOICES)]
            else:
                vregs = [reg]
            vregs = [vreg for vreg in vregs if vreg in norm_df.columns]
            norm_df = (
                norm_df[vregs]
                .reset_index()[["f"] + vregs]
                .drop_duplicates("f", keep="last")
            )
            norm_df = (
                pd.melt(norm_df, id_vars=["f"], var_name="v", value_name="val")
                .astype(MODEL_PDTYPE)
                .sort_values("f")
            ).reset_index(drop=True)
            norm_df["v"] = norm_df["v"].floordiv(VOICE_REG_SIZE)
            diff_df = norm_df.copy()
            diff_df["pval"] = diff_df["val"]
            diff_df["f"] += 1
            diff_df = diff_df[["pval", "v", "f"]]
            norm_df = norm_df.merge(diff_df, how="left", on=["v", "f"])
            norm_df = norm_df.fillna(0).astype(MODEL_PDTYPE).sort_values(["f", "v"])
            yield norm_df

    def _reduce_val_res(self, df, reg, bits):
        m = df["reg"] == reg
        df.loc[m, "val"] = np.left_shift(np.right_shift(df[m]["val"], bits), bits)
        return df

    def _quantize_freq_to_cents(self, df):
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            cond = df["reg"] == v_offset
            df.loc[cond, "val"] = df[cond]["val"].map(self.freq_mapper.fi_map)
        return df

    def _squeeze_frames(self, orig_df):
        df = orig_df.copy()
        df["f"] = self._frame_reg(df)
        cm = self._ctrl_match(df).astype(MODEL_PDTYPE)
        df["c"] = cm * cm.cumsum()
        df = df.drop_duplicates(["f", "c", "reg"], keep="last")
        return df[orig_df.columns].reset_index(drop=True)

    def _norm_df(self, orig_df):
        norm_df = orig_df.copy().reset_index(drop=True)
        norm_df["f"] = self._frame_reg(norm_df)
        norm_df["v"] = (
            norm_df["reg"].abs().floordiv(VOICE_REG_SIZE).astype(MODEL_PDTYPE)
        )
        norm_df["n"] = (norm_df.index + 1) * 10
        norm_df.loc[norm_df["f"].diff() != 0, "v"] = 0
        norm_df["vd"] = norm_df["v"].diff().astype(MODEL_PDTYPE).fillna(0)
        return norm_df

    def _norm_pr_order(self, orig_df):
        norm_df = self._norm_df(orig_df)
        df = norm_df.copy()
        df["areg"] = df["reg"].abs()
        df = df.sort_values(["f", "v", "areg", "n"])
        df = df[orig_df.columns].reset_index(drop=True)
        return df

    def _add_voice_reg(self, orig_df):
        norm_df = self._norm_df(orig_df)
        m = (norm_df["reg"] >= 0) & (norm_df["v"].isin(set(range(VOICES))))
        first_v = norm_df[m]
        if first_v.empty:
            return orig_df
        first_v = first_v.iloc[0]
        norm_df.loc[m, "reg"] = norm_df[m]["reg"] % VOICE_REG_SIZE
        df = norm_df[((norm_df["vd"] != 0) | (norm_df["n"] == first_v["n"])) & m].copy()
        df["n"] -= 1
        df["val"] = df["v"]
        df["reg"] = VOICE_REG
        df["op"] = SET_OP

        df = pd.concat([norm_df, df]).sort_values(["n"]).reset_index(drop=True)
        df["nr"] = df["reg"].shift(-1)
        df["nval"] = df["val"].shift(-1)
        df["pr"] = df["reg"].shift(1)
        df.loc[((df["reg"] == FRAME_REG) & (df["nr"] == VOICE_REG)), "val"] = df["nval"]
        df = df[~((df["reg"] == VOICE_REG) & (df["pr"].fillna(VOICE_REG) == FRAME_REG))]
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

    def _simplify_pcm(self, orig_df):
        df = orig_df.copy()
        df["n"] = df.index * 10
        df["v"] = df["reg"].floordiv(VOICE_REG_SIZE).astype(pd.UInt8Dtype())
        dfs = [df[df["v"] >= VOICES].copy()]

        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            pcm_reg = v_offset + 2
            ctrl_reg = v_offset + 4
            v_df = df[df["v"] == v].copy()
            # set PCM field
            v_df["pcm"] = pd.NA
            m = v_df["reg"] == pcm_reg
            v_df.loc[m, "pcm"] = v_df[m]["val"]
            v_df["pcm"] = v_df["pcm"].astype(MODEL_PDTYPE).ffill()
            # set p flag for when pulse enabled.
            v_df["p"] = pd.NA
            m = (v_df["reg"] == ctrl_reg) & (v_df["val"] & 0b01000000 == 0b01000000)
            v_df.loc[m, "p"] = 1
            m = (v_df["reg"] == ctrl_reg) & (v_df["val"] & 0b01000000 == 0)
            v_df.loc[m, "p"] = 0
            v_df["p"] = v_df["p"].astype(pd.UInt8Dtype()).ffill()
            # set PCM set, to 0 where pulse not enabled
            # forces PCM to be reset when pulse waveform selected.
            v_df.loc[((v_df["reg"] == pcm_reg) & (v_df["p"] == 0)), "val"] = 0
            # add PCM set when pulse enabled.
            p_df = v_df[
                (v_df["reg"] == ctrl_reg) & (v_df["val"] & 0b01000000 == 0b01000000)
            ].copy()
            p_df["reg"] = pcm_reg
            p_df["val"] = p_df["pcm"]
            p_df["n"] = p_df["n"] - 1
            v_df = pd.concat([v_df, p_df])
            dfs.append(v_df)

        df = pd.concat(dfs).sort_values("n")
        return df[orig_df.columns].reset_index(drop=True)

    def _add_change_reg(
        self, df, change_df, minchange=256, opcodes=[DIFF_OP, FLIP_OP, REPEAT_OP]
    ):
        change_dfs = []
        change_df["val"] -= change_df["pval"]
        change_df = change_df.drop("pval", axis=1)
        for reg in change_df["reg"].unique():
            v_df = change_df[change_df["reg"] == reg].copy()
            v_df = v_df.sort_values(["n", "val"])
            # Only one change per reg per frame.
            v_df["cpf"] = v_df.groupby("f").transform("size")
            v_df = v_df[
                (
                    (v_df["val"].abs() <= minchange)
                    | (v_df["val"].shift(1) == v_df["val"])
                )
                & (v_df["cpf"] == 1)
            ]
            df = df[~df["n"].isin(v_df["n"])]
            m_df = v_df[v_df["f"].diff().fillna(1) > 1].copy()
            m_df["op"] = DIFF_OP
            v_df = v_df[~v_df["n"].isin(m_df["n"])]
            change_dfs.append(m_df)
            v_df["aval"] = v_df["val"].abs()
            v_df["cf"] = (
                (
                    (v_df["f"].diff().fillna(1) > 1)
                    .astype(MODEL_PDTYPE)
                    .cumsum()
                    .astype(MODEL_PDTYPE)
                )
                * 255
            ) + 1
            v_df[["repeat", "flip", "begin", "end"]] = 0
            for f, c in (("repeat", "val"), ("flip", "aval")):
                cols = ["cf", c]
                v_df.loc[
                    (v_df[cols] == v_df[cols].shift(1)).all(axis=1)
                    | (v_df[cols] == v_df[cols].shift(-1)).all(axis=1),
                    f,
                ] = (
                    v_df["cf"] * v_df[c]
                )
            f = "repeat"
            m = v_df[f] != 0
            v_df.loc[m & (v_df[f] != v_df[f].shift(1)), "begin"] = 1
            v_df.loc[m & (v_df[f] != v_df[f].shift(-1)), "end"] = 1
            v_df.loc[(v_df["begin"] == 1) & (v_df["end"] == 1), ["begin", "end", f]] = 0
            v_df.loc[v_df["repeat"] != 0, "flip"] = 0
            f = "flip"
            m = v_df[f] != 0
            v_df.loc[m & (v_df[f] != v_df[f].shift(1)), "begin"] = 1
            v_df.loc[m & (v_df[f] != v_df[f].shift(-1)), "end"] = 1
            # v_df.loc[
            #    ((v_df["begin"] == 1) & (v_df["end"].shift(-1) == 1))
            #    | ((v_df["end"] == 1) & (v_df["begin"].shift(1) == 1)),
            #    ["repeat", "flip", "begin", "end"],
            # ] = 0
            assert not len(v_df[(v_df["repeat"] != 0) & (v_df["flip"] != 0)])
            v_df.loc[
                (v_df["begin"] == 1) & (v_df["end"] == 1), ["begin", "end", "flip"]
            ] = 0

            for f, op in (("repeat", REPEAT_OP), ("flip", FLIP_OP)):
                if op in opcodes:
                    d_df = v_df[v_df[f] != 0].copy()
                    v_df = v_df[v_df[f] == 0]
                    if d_df.empty:
                        continue
                    d_df = d_df[(d_df["begin"] == 1) | (d_df["end"] == 1)]
                    if d_df.empty:
                        continue
                    assert d_df["begin"].iloc[0] == 1, d_df
                    assert d_df["end"].iloc[-1] == 1, d_df
                    d_df.loc[d_df["end"] == 1, "val"] = 0
                    d_df["op"] = op
                    change_dfs.append(d_df.copy())

            v_df["op"] = DIFF_OP
            change_dfs.append(v_df)

        df = df.drop("pval", axis=1)
        return df, change_dfs

    def _add_change_regs(self, orig_df, opcodes=[DIFF_OP, FLIP_OP, REPEAT_OP]):
        df = self._norm_df(orig_df)
        df["op"] = SET_OP

        freq_df, pcm_df, ctrl_df, filter_df = self._last_reg_val_frame(
            orig_df, [0, 2, 4, FC_LO_REG]
        )
        freq_df["reg"] = freq_df["v"] * VOICE_REG_SIZE
        pcm_df["reg"] = pcm_df["v"] * VOICE_REG_SIZE + 2
        ctrl_df["reg"] = ctrl_df["v"] * VOICE_REG_SIZE + 4
        filter_df["reg"] = FC_LO_REG

        all_change_dfs = []
        for xdf, matcher, minchange in (
            (freq_df, self._freq_match, 14 * 2),
            (pcm_df, self._pcm_match, 64),
            (ctrl_df, self._ctrl_match, 1),
            (filter_df, self._filter_match, 64),
        ):
            df = df.merge(xdf[["reg", "f", "pval"]], how="left", on=["f", "reg"])
            xdf = df[matcher(df)].copy()
            df, change_dfs = self._add_change_reg(
                df, xdf, minchange=minchange, opcodes=opcodes
            )
            all_change_dfs.extend(change_dfs)

        df = pd.concat([df] + all_change_dfs).sort_values(["n"]).reset_index(drop=True)
        df = df[list(orig_df.columns) + ["op"]].reset_index(drop=True)
        return df

    def _filter_irq(self, df, name):
        try:
            irq = df["irq"].iloc[0]
        except KeyError:
            self.logger.info(df)
            self.logger.info("skipped %s, no irq", name)
            return False
        if irq < self.args.min_irq or irq > self.args.max_irq:
            self.logger.info("skipped %s, irq %u (outside IRQ range)", name, irq)
            return False
        return True

    def _filter(self, df, name):
        if not self._filter_irq(df, name):
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

    def _combine_regs(self, df):
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for reg, bits in ((v_offset, 0), ((v_offset + 2), 4)):
                df = self._combine_reg(df, reg=reg, bits=bits)
        df = self._combine_reg(df, FC_LO_REG, bits=2)
        return df

    def parse(self, name, diffmax=512, max_perm=99, require_pq=False):
        parquet_glob = glob.glob(name.replace(DUMP_SUFFIX, PARSED_SUFFIX))
        if parquet_glob:
            for parquet_name in sorted(parquet_glob):
                # assert (
                #     Path(parquet_name).stat().st_mtime > PY_MTIME
                # ), f"pre-parsed {parquet_name} out of date"
                pf = ParquetFile(parquet_name)
                sample_rows = next(pf.iter_batches(batch_size=1))
                df = pa.Table.from_batches([sample_rows]).to_pandas()
                if self._filter_irq(df, parquet_name):
                    df = pd.read_parquet(parquet_name)
                    if self._filter(df, parquet_name):
                        yield df
            return
        if require_pq:
            return
        df = self._read_df(name)
        df = self._squeeze_changes(df)
        df = self._combine_regs(df)
        df = self._quantize_freq_to_cents(df)
        df = self._simplify_ctrl(df)
        df = self._simplify_pcm(df)
        df = self._squeeze_changes(df)
        if df.empty:
            return
        irq, df = self._add_frame_reg(df, diffmax)
        df = self._add_change_regs(df, opcodes=[DIFF_OP, REPEAT_OP])
        delay_val = df[df["reg"] == DELAY_REG]["val"]
        if len(delay_val):
            delay_max = delay_val.max()
            assert delay_max < 256, delay_max
        irq = min(2 ** (IRQ_PDTYPE.itemsize * 8) - 1, irq)
        df = self._squeeze_frames(df)
        df["irq"] = irq
        while not df.empty and self._frame_match(df.iloc[-1]):
            df = df.head(len(df) - 1)
        while not df.empty and (
            self._frame_match(df.iloc[0])
            or (df.iloc[0]["reg"] == MODE_VOL_REG and df.iloc[0]["val"] == 15)
        ):
            df = df.tail(len(df) - 1)

        for xdf in self._rotate_voice_augment(df, max_perm=max_perm):
            xdf = xdf[FRAME_DTYPES.keys()].astype(FRAME_DTYPES)
            xdf = self._norm_pr_order(xdf)
            xdf = self._add_voice_reg(xdf)
            xdf = xdf.reset_index(drop=True)
            if not self._filter(xdf, name):
                break
            empty_val = xdf[xdf["val"].isna()]
            assert empty_val.empty, (name, empty_val)
            yield xdf
