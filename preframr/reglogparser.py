from collections import defaultdict
import itertools
import glob
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
from preframr import macros
from preframr.macros import (
    DECODERS,
    DecodeState,
    _FastRow,
    _deserialize_gate_palette,
    _df_arrays_and_frames,
)
from preframr.reg_mappers import FreqMapper
from preframr.stfconstants import (
    DELAY_REG,
    DIFF_OP,
    DIFF_PDTYPE,
    DUMP_SUFFIX,
    FC_LO_REG,
    FILTER_BITS,
    FILTER_REG,
    FLIP_OP,
    FRAME_REG,
    MAX_REG,
    META_FREQ_BITS,
    MIN_DIFF,
    MODE_VOL_REG,
    MODEL_PDTYPE,
    PAD_REG,
    PARSED_SUFFIX,
    PCM_BITS,
    REG_PDTYPE,
    REPEAT_OP,
    SET_OP,
    TOKEN_KEYS,
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
pd.set_option("future.no_silent_downcasting", True)


def wrapbits(x, reglen):
    base = (x << 1) & (2**reglen - 1)
    lsb = (x >> (reglen - 1)) & 1
    return base ^ lsb


FILTER_SHIFT_DF = pd.DataFrame(
    [{"reg": FILTER_REG, "val": i, "y": wrapbits(i, 3)} for i in range(2**3)],
    dtype=MODEL_PDTYPE,
)


def prepare_df_for_audio(orig_df, reg_widths, irq, sidq, strict=False, prompt_len=None):
    if not prompt_len:
        prompt_len = len(orig_df)
    df = orig_df.copy()
    df["description"] = 0
    if prompt_len < len(df):
        df.loc[prompt_len:, "description"] = 1
        assert len(df[df["description"] == 1])
    loader = RegLogParser()
    df, reg_widths = loader._remove_voice_reg(df, reg_widths)
    df = loader._expand_ops(df, strict)
    df = loader._reset_diffs(df, irq, sidq)
    return df, reg_widths


class RegLogParser:
    def __init__(self, args=None, logger=logging):
        self.args = args
        self.logger = logger
        self.valid_voiceorders = self._valid_voiceorders()
        self.freq_mapper = None
        if self.args:
            self.freq_mapper = FreqMapper(cents=self.args.cents)

    def _valid_voiceorders(self):
        voiceorders = set()
        for r in range(1, VOICES + 1):
            for perm in itertools.permutations(range(VOICES), r):
                voiceorder = 0
                for i, v in enumerate(perm):
                    voiceorder += v + 1 << (2 * i)
                voiceorders.add(voiceorder)
        assert len(voiceorders) == 15, (len(voiceorders), sorted(voiceorders))
        return voiceorders

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

        df = df[df["reg"] <= MAX_REG]
        df["val"] = df["val"].astype(VAL_PDTYPE)
        chips = df["chipno"].nunique()
        df = df[["clock", "irq", "reg", "val"]]
        if chips > 1:
            return df[df["clock"] < 0]
        return df

    def _maskreg(self, df, reg, valmask):
        mask = df["reg"] == reg
        df.loc[mask, ["val"]] = df[mask]["val"] & valmask

    def highbitmask(self, bits):
        return 255 - (2**bits - 1)

    def _maskregbits(self, df, reg, bits):
        self._maskreg(df, reg, self.highbitmask(bits))

    def _state_df(self, states, dataset, irq):
        tokens = dataset.tokenizer.tokens.copy()
        tokens["diff"] = MIN_DIFF
        tokens.loc[tokens["reg"] < -MAX_REG, "diff"] = 0
        # PAD_REG row (synthetic, vocab idx 0): zero cycle cost so a
        # leaked pad row at inference is a no-op in the audio render.
        tokens.loc[tokens["reg"] == PAD_REG, "diff"] = 0
        tokens.loc[tokens["reg"] == FRAME_REG, "diff"] = irq
        df = pd.DataFrame(states, columns=["n"]).merge(tokens, on="n", how="left")
        # Tokens have no ``description`` field but downstream walkers
        # (``_simulate_palette``, ``_FastRow``) cast ``description`` to
        # ``int`` per row. Default to 0 so a concat with a
        # ``description``-bearing prompt df doesn't produce NaN rows
        # that crash validate_gate_replays / validate_back_refs in the
        # predict path's safety net.
        if "description" not in df.columns:
            df["description"] = 0
        return df

    def _reset_diffs(self, orig_df, irq, sidq):
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

    def _squeeze_changes(self, df):
        # Drop a row when its (reg, val) repeats the previous write to the
        # same reg -- i.e., the SID register's running value is unchanged.
        # The old implementation pivoted to a wide form (one column per reg)
        # then ffill+shift to detect any change; that's O(n*k) and on
        # multi-million-row dumps takes 10s+ in pandas overhead.
        # ``groupby('reg')['val'].shift()`` gives each row the previous val
        # written to the same reg in O(n), and the row is kept iff its val
        # differs from that or there's no prior write.
        prev = df.groupby("reg")["val"].shift()
        mask = prev.isna() | (prev != df["val"])
        return df.loc[mask, ["clock", "irq", "reg", "val"]].reset_index(drop=True)

    def _combine_val(self, reg_df, reg, reg_range, dtype=MODEL_PDTYPE, lobits=8):
        origcols = reg_df.columns
        for i in range(reg_range):
            reg_df[str(i)] = reg_df[reg_df["reg"] == (reg + i)]["val"]
            reg_df[str(i)] = reg_df[str(i)].ffill().fillna(0)
            reg_df[str(i)] = np.left_shift(reg_df[str(i)].values, int(lobits * i))
        reg_df["val"] = 0
        reg_df["reg"] = reg
        for i in range(reg_range):
            reg_df["val"] = reg_df["val"].astype(dtype) + reg_df[str(i)]
        return reg_df[origcols]

    def _combine_reg(self, orig_df, reg, diffmax=512, bits=0, lobits=8):
        # Sort only the small (reg, reg+1) subset so ``_combine_val``'s
        # ffill sees temporal order; the rest of the df doesn't need to
        # be clock-sorted at this stage. ``_combine_regs`` does one big
        # sort at the end. This avoids 7 full-df sort_values on
        # multi-million-row dumps.
        cond = (orig_df["reg"] == reg) | (orig_df["reg"] == (reg + 1))
        reg_df = orig_df[cond].sort_values("clock", kind="stable").copy()
        non_reg_df = orig_df[~cond]
        reg_df["dclock"] = reg_df["clock"].floordiv(diffmax)
        reg_df = self._combine_val(reg_df, reg, 2, lobits=lobits)
        reg_df = reg_df.drop_duplicates(["dclock"], keep="last")
        if bits:
            reg_df["val"] = np.left_shift(np.right_shift(reg_df["val"], bits), bits)
        df = pd.concat([non_reg_df, reg_df[orig_df.columns]], ignore_index=True)
        df = df.astype(orig_df.dtypes)
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

    def _add_frame_reg(self, orig_df, diffmax, min_irq_prop=0.95):
        df = orig_df.copy()
        df["irqdiff"] = df["irq"].diff().fillna(0).astype(MODEL_PDTYPE)
        df["diff"] = MIN_DIFF
        df["i"] = df.index * 10
        m = df["irqdiff"] > diffmax
        try:
            irq_counts = df["irqdiff"][m].value_counts()
            largest_irqs_sum = sum([k * v for k, v in irq_counts.items()])
            irq = int(irq_counts.nlargest(1).index[0])
        except IndexError:
            irq = 0
        if largest_irqs_sum / df["clock"].max() < min_irq_prop:
            irq = 0
        irq_df = df[m].copy()
        irq_df["i"] -= 1
        irq_df["reg"] = FRAME_REG
        irq_df["diff"] = irq_df["irqdiff"]
        irq_df["val"] = (irq_df["diff"] / irq).astype(MODEL_PDTYPE)
        irq_df["diff"] = irq
        irq_df.loc[irq_df["val"] > 1, "reg"] = DELAY_REG
        irq_df.loc[irq_df["reg"] == DELAY_REG, "diff"] = 0
        irq_df.loc[irq_df["reg"] == FRAME_REG, "val"] = 0
        df = (
            pd.concat([df, irq_df], ignore_index=True)
            .sort_values(["i"])[["reg", "val", "diff", "i"]]
            .astype(MODEL_PDTYPE)
            .reset_index(drop=True)
        )
        return irq, df[["reg", "val", "diff"]]

    def _cap_delay(self, df, q=5):
        m = df["reg"] == DELAY_REG
        df.loc[m & (df["val"] > 255), "val"] = 255
        df.loc[m & (df["val"] > q**2), "val"] = (df["val"] / q).astype(MODEL_PDTYPE) * q
        return df

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
        df = pd.concat([df, reg_df], copy=False, ignore_index=True).sort_values(
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
        m = self._freq_match(df)
        df.loc[m, "val"] = df[m]["val"].map(self.freq_mapper.fi_map)
        return df

    def _norm_df(self, orig_df):
        norm_df = orig_df.copy().reset_index(drop=True)
        norm_df["f"] = self._frame_reg(norm_df)
        norm_df["v"] = (
            norm_df["reg"].abs().floordiv(VOICE_REG_SIZE).astype(MODEL_PDTYPE)
        )
        norm_df.loc[norm_df["v"] < 0, "v"] = 0
        norm_df["n"] = (norm_df.index + 1) * 10
        norm_df.loc[norm_df["f"].diff() != 0, "v"] = 0
        norm_df["vd"] = norm_df["v"].diff().astype(MODEL_PDTYPE).fillna(0)
        return norm_df

    def _norm_pr_order(self, orig_df, v_only=False):
        """Sort rows within each frame by strict numeric voice order.

        Within a frame: voice 0's rows first (reg-ascending, op-ascending,
        original-index tiebreak), then voice 1's rows, then voice 2's.

        The previous frequency/control-state-aware variant (via
        ``_get_vmeta``) clustered rows whose voice was holding similar
        pitches together, on the speculation that the LM would learn
        cross-voice harmonic structure from the locality. That signal
        was unmeasured and made downstream walks non-deterministic
        relative to the encoder's pre-norm walk -- the same logical
        frame could sort differently depending on prior frames' state.
        Strict numeric is deterministic, makes loop / repeat detection
        cheap, and lets ``InstrumentProgramPass`` mirror the order with
        a single ``sort_values`` so its Phase 2 walk matches downstream.

        ``v_only`` is retained for caller compatibility but is now a
        no-op (the v-only path was always strict numeric).
        """
        del v_only
        df = self._norm_df(orig_df.copy())
        df.loc[df["reg"] < 0, "v"] = df["reg"]
        df = df.sort_values(["f", "v", "reg", "op", "n"])
        df = df[orig_df.columns].reset_index(drop=True)
        if orig_df.attrs:
            df.attrs.update(orig_df.attrs)
        return df

    def _add_voice_reg(self, orig_df, zero_voice_reg=True):
        norm_df = self._norm_df(orig_df)
        m = (norm_df["reg"] >= 0) & (norm_df["v"].isin(set(range(VOICES))))
        first_v = norm_df[m]
        if first_v.empty:
            return orig_df
        first_v = first_v.iloc[0]
        norm_df["f"] = (norm_df["reg"] == FRAME_REG).astype(MODEL_PDTYPE).cumsum()
        df = norm_df[((norm_df["vd"] != 0) | (norm_df["n"] == first_v["n"])) & m].copy()
        norm_df.loc[m, "reg"] = norm_df[m]["reg"] % VOICE_REG_SIZE
        df["n"] -= 1
        df["val"] = df["v"]
        df["reg"] = VOICE_REG
        df["op"] = SET_OP
        df = (
            pd.concat([norm_df, df], ignore_index=True)
            .sort_values(["n"])
            .reset_index(drop=True)
        )
        df["nr"] = df["reg"].shift(-1).astype(REG_PDTYPE)
        df["nval"] = df["val"].shift(-1).astype(VAL_PDTYPE)
        df["pr"] = df["reg"].shift(1).astype(REG_PDTYPE)
        df.loc[((df["reg"] == FRAME_REG) & (df["nr"] == VOICE_REG)), ["v", "val"]] = df[
            "nval"
        ]
        df = df[~((df["reg"] == VOICE_REG) & (df["pr"].fillna(VOICE_REG) == FRAME_REG))]

        df["fn"] = 0
        m = df["reg"].isin({FRAME_REG, VOICE_REG})
        df.loc[m, "fn"] = 1
        df["fn"] = df.groupby("f")["fn"].cumsum()
        df["fn"] -= 1
        df["sv"] = 0
        df.loc[m, "sv"] = np.left_shift(df[m]["v"] + 1, df[m]["fn"] * 2)
        df["svc"] = df.groupby("f")["sv"].cumsum()
        df["svt"] = df.groupby("f")["svc"].transform("max")

        if zero_voice_reg:
            df.loc[df["reg"] == VOICE_REG, "val"] = 0
        else:
            for v in range(VOICES):
                fm = (df["v"] == v) & (df["op"] == SET_OP) & (df["reg"] == 0)
                df.loc[fm, f"{v}freqmeta"] = np.right_shift(
                    df[fm]["val"], self.freq_mapper.bits - META_FREQ_BITS
                )
                freqmeta = df[fm][f"{v}freqmeta"]
                assert (
                    len(freqmeta) == 0 or freqmeta.max() < 2**META_FREQ_BITS
                ), freqmeta.max()
                cm = (df["v"] == v) & (df["op"] == SET_OP) & (df["reg"] == 4)
                df.loc[cm, f"{v}ctrlmeta"] = df[cm]["val"] & 0b11110000
            df = df.ffill().fillna(0)
            m = df["reg"] == VOICE_REG
            df.loc[m, "v"] = df[m]["val"]
            for v in range(VOICES):
                m = (df["reg"] == VOICE_REG) & (df["v"] == v)
                df.loc[m, "val"] = df[m][f"{v}ctrlmeta"] + df[m][f"{v}freqmeta"]

        m = df["reg"] == FRAME_REG
        df.loc[m, "val"] = df[m]["svt"]
        invalid_val = set(df[m]["val"].unique()) - self.valid_voiceorders
        assert not invalid_val, invalid_val
        df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        # ``pd.concat`` of multiple dfs above drops attrs; restore the
        # encoder-published palettes explicitly.
        if orig_df.attrs:
            df.attrs.update(orig_df.attrs)
        return df

    def _remove_voice_reg(self, orig_df, reg_widths):
        voice_regs = len(orig_df[orig_df["reg"] == VOICE_REG])
        if voice_regs:
            df = orig_df.copy()
            df["v"] = pd.NA
            df["fn"] = 0
            df["f"] = 0
            df.loc[df["reg"] == FRAME_REG, "f"] = 1
            df["f"] = df["f"].cumsum()
            m = df["reg"].isin({FRAME_REG, VOICE_REG})
            df.loc[m, "fn"] = 1
            df["fn"] = df.groupby("f")["fn"].cumsum()
            df["fn"] -= 1
            df.loc[df["reg"] == FRAME_REG, "sval"] = (
                df[df["reg"] == FRAME_REG]["val"] & 2**6 - 1
            )
            df["sval"] = df["sval"].ffill().fillna(0)
            df.loc[m, "v"] = (
                np.right_shift(df[m]["sval"], df["fn"] * 2) & 2**2 - 1
            ) - 1
            df["v"] = df["v"].ffill().fillna(0)
            df.loc[df["v"] < 0, "v"] = 0
            m = (df["reg"] >= 0) & (df["reg"] < VOICE_REG_SIZE)
            df.loc[m, "reg"] = df[m]["reg"] + (df[m]["v"] * VOICE_REG_SIZE)
            df = df[df["reg"] != VOICE_REG]
            df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
            if orig_df.attrs:
                df.attrs.update(orig_df.attrs)
            for v in range(VOICES):
                v_offset = v * VOICE_REG_SIZE
                for i in range(VOICE_REG_SIZE):
                    if i in reg_widths:
                        reg_widths[v_offset + i] = reg_widths[i]
            return df, reg_widths
        return orig_df, reg_widths

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
        # Numpy-vectorised rewrite. The previous implementation built a
        # per-voice df slice with nullable-int ``pcm`` / ``p`` columns,
        # ffill'd on those nullable columns, then masked-and-assigned --
        # 1.5-2s on multi-million-row dumps. The same logic works on
        # numpy arrays (with float-NaN for the ffill placeholder) at
        # ~0.1s. Output is byte-identical (regression tests cover it).
        df = orig_df.copy()
        df["n"] = df.index.astype(np.int64) * 10
        regs = df["reg"].to_numpy()
        df["v"] = pd.Series(regs // VOICE_REG_SIZE).astype(pd.UInt8Dtype())

        out_dfs = [df[df["v"] >= VOICES]]
        vals = df["val"].to_numpy()
        n_arr = df["n"].to_numpy()
        v_arr = regs // VOICE_REG_SIZE

        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            pcm_reg = v_offset + 2
            ctrl_reg = v_offset + 4
            v_mask = v_arr == v
            if not v_mask.any():
                continue
            v_idx = np.where(v_mask)[0]
            v_regs = regs[v_idx]
            v_vals = vals[v_idx]

            # Running pcm value: val at pcm_reg writes, ffill elsewhere.
            # Leading NaN (before any pcm write) is preserved as NaN -- the
            # old code stored ``pd.NA`` and didn't fillna(0), so leading
            # rows whose p flag isn't yet defined keep their original val
            # (the override condition compares as False against NaN).
            pcm_col = np.where(v_regs == pcm_reg, v_vals.astype(np.float64), np.nan)
            pcm_running = pd.Series(pcm_col).ffill().to_numpy()

            # Running p flag: 1 if last ctrl write had bit 6 set, 0
            # otherwise. NaN before any ctrl write -- preserved.
            ctrl_mask = v_regs == ctrl_reg
            bit6_set = (v_vals & 0b01000000) == 0b01000000
            p_col = np.full(len(v_idx), np.nan, dtype=np.float64)
            p_col[ctrl_mask & bit6_set] = 1.0
            p_col[ctrl_mask & ~bit6_set] = 0.0
            p_running = pd.Series(p_col).ffill().to_numpy()

            # Build the per-voice df from the original rows. Override the
            # val for pcm_reg writes that occur while pulse waveform is
            # disabled. ``p_running == 0`` is False where p_running is
            # NaN, matching the old code.
            v_df = df.iloc[v_idx].copy()
            override = (v_regs == pcm_reg) & (p_running == 0)
            if override.any():
                new_vals = v_vals.copy()
                new_vals[override] = 0
                v_df["val"] = new_vals

            # Synthesize a pcm-reg row just before each ctrl-pulse-on
            # write, carrying the running pcm value. NaN pcm (no prior
            # pcm write) maps to 0 -- semantically equivalent to the
            # SID's uninitialised pcm state. The old code stored NA
            # here and relied on a later pipeline stage to coerce it,
            # but that stage no longer accepts NA val cells (the
            # itertuples->numpy refactor in EndTerminatorPass casts
            # ``int(vals[i])`` directly).
            synth = ctrl_mask & bit6_set
            if synth.any():
                p_df = df.iloc[v_idx[synth]].copy()
                synth_pcm = np.nan_to_num(pcm_running[synth], nan=0.0).astype(np.int64)
                p_df["reg"] = pcm_reg
                p_df["val"] = synth_pcm
                p_df["n"] = p_df["n"] - 1
                v_df = pd.concat([v_df, p_df], ignore_index=True)
            out_dfs.append(v_df)

        df = pd.concat(out_dfs, ignore_index=True).sort_values("n")
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
            v_df["aval"] = v_df["val"].abs()
            v_df = v_df[
                (v_df["aval"] > 0) & (v_df["aval"] <= minchange) & (v_df["cpf"] == 1)
            ]
            df = df[~df["n"].isin(v_df["n"])]
            m_df = v_df[v_df["f"].diff().fillna(1) > 1].copy()
            m_df["op"] = DIFF_OP
            v_df = v_df[~v_df["n"].isin(m_df["n"])]
            change_dfs.append(m_df)
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
            # filter repeat/flip ranges.
            for f, of in (("repeat", "flip"), ("flip", "repeat")):
                m = v_df[f] != 0
                v_df.loc[m & (v_df[f] != v_df[f].shift(1)), "begin"] = v_df["n"]
                v_df.loc[m & (v_df[f] != v_df[f].shift(-1)), "end"] = v_df["n"]
                # reject spans of length 0 and 1
                for shift in (0, 1):
                    v_df.loc[
                        ((v_df["end"] != 0) & (v_df["begin"].shift(shift) != 0))
                        | ((v_df["begin"] != 0) & (v_df["end"].shift(-shift) != 0)),
                        ["begin", "end", f],
                    ] = 0
                v_df.loc[v_df[f] != 0, of] = 0

            for f, op in (("repeat", REPEAT_OP), ("flip", FLIP_OP)):
                if op in opcodes:
                    d_df = v_df[
                        (v_df[f] != 0) & ((v_df["begin"] != 0) | (v_df["end"] != 0))
                    ].copy()
                    v_df = v_df[v_df[f] == 0]
                    if d_df.empty:
                        continue
                    assert d_df["begin"].iloc[0] != 0, d_df
                    assert d_df["end"].iloc[-1] != 0, d_df
                    d_df.loc[d_df["end"] != 0, "val"] = 0
                    d_df["op"] = op
                    change_dfs.append(d_df.copy())

            # default to diff
            v_df["op"] = DIFF_OP
            change_dfs.append(v_df)

        df = df.drop("pval", axis=1)
        return df, change_dfs

    def _add_change_regs(
        self, orig_df, opcodes=[DIFF_OP, FLIP_OP, REPEAT_OP], cents=50
    ):
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
            # restrict frequency change to two octaves
            (freq_df, self._freq_match, int((2 * 12) * 100 / cents)),
            (pcm_df, self._pcm_match, 64),
            (filter_df, self._filter_match, 512),
        ):
            df = df.merge(xdf[["reg", "f", "pval"]], how="left", on=["f", "reg"])
            xdf = df[matcher(df)].copy()
            df, change_dfs = self._add_change_reg(
                df, xdf, minchange=minchange, opcodes=opcodes
            )
            all_change_dfs.extend(change_dfs)

        df = (
            pd.concat([df] + all_change_dfs, ignore_index=True)
            .sort_values(["n"])
            .reset_index(drop=True)
        )
        df = df[list(orig_df.columns) + ["op"]].reset_index(drop=True)
        return df

    def _expand_ops(self, orig_df, strict):
        # Materialize any LOOP_BACK / DO_LOOP rows into literal frames before
        # per-row dispatch. No-op if the stream contains neither.
        df = macros.expand_loops(orig_df.copy())
        last_diff = {}
        for reg in df["reg"].unique():
            reg_df = df[(df["reg"] == reg) & (df["op"] == SET_OP)]["diff"]
            if len(reg_df):
                last_diff[reg] = reg_df.iloc[0]
            else:
                last_diff[reg] = MIN_DIFF

        frame_diff = df[df["reg"] == FRAME_REG]["diff"].iloc[0]
        cap = (
            getattr(self.args, "gate_palette_cap", None)
            if self.args is not None
            else None
        )
        state = DecodeState(
            frame_diff,
            last_diff=last_diff,
            strict=strict,
            gate_palette_cap=cap,
            frozen_instrument_palette=orig_df.attrs.get("instrument_palette"),
            frozen_gate_palette=_deserialize_gate_palette(
                orig_df.attrs.get("gate_palette")
            ),
        )
        # Build the result row by row in a flat Python list and let
        # pandas materialise the DataFrame ONCE at the end. The previous
        # design constructed a one-row-per-frame DataFrame, sort_values
        # 'd it, and pd.concat'd the lot at the end -- on Skybox that's
        # 20K micro-DataFrames + a 20K-way concat, which dominated
        # preload (80s of 90s wall) at 83% of total preload time.
        # Single-array path drops it to ~1-3s. The per-frame reg sort
        # is preserved by tagging each row with its add_frame() call
        # index (NOT frame_idx -- multiple add_frame calls share a
        # frame_idx during DELAY_REG unrolling) and doing one stable
        # sort by (call_idx, reg) at the end.
        all_rows = []  # list[(reg, val, diff, description, call_idx)]
        call_idx = [0]

        df["f"] = (
            df["reg"]
            .isin({DELAY_REG, FRAME_REG})
            .astype(MODEL_PDTYPE)
            .cumsum()
            .astype(MODEL_PDTYPE)
        )

        # frame_idx counts logical frame slots (each FRAME_REG or DELAY_REG
        # row = 1 slot), matching ``LoopPass``'s back-ref distance/length
        # semantics and ``materialize_*_outside``'s slice coordinates. The
        # extra audio frames produced by a DELAY_REG val>1 share the same
        # slot index, so palette ``def_frame`` values stay in slot coords.
        out_frame_idx = [0]

        def add_frame(writes, advance=True):
            ci = call_idx[0]
            for w in writes:
                # Tuple may be (reg, val, diff) or (reg, val, diff,
                # description). Tick-driven writes (REPEAT/FLIP/PWM
                # bursts) often emit 3-tuples; description=NA there
                # is intentional so the final ffill carries forward
                # the most-recent explicit description value.
                desc = w[3] if len(w) > 3 else pd.NA
                all_rows.append((w[0], w[1], w[2], desc, ci))
            state.observe_frame(writes, frame_idx=out_frame_idx[0])
            call_idx[0] += 1
            if advance:
                out_frame_idx[0] += 1

        arrs, frame_starts = _df_arrays_and_frames(df)
        regs = arrs["reg"]
        vals = arrs["val"]
        ops = arrs["op"]
        subregs = arrs["subreg"]
        diffs = arrs["diff"]
        descs = arrs["description"]
        indices = arrs["Index"]
        n_total = len(df)
        n_frames = len(frame_starts)
        for fi in range(n_frames):
            start = int(frame_starts[fi])
            end = int(frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
            f_sid_writes = []
            marker_reg = int(regs[start])
            marker_val = int(vals[start])
            marker_diff = int(diffs[start])
            marker_desc = int(descs[start])
            if marker_reg == FRAME_REG:
                f_sid_writes.append((marker_reg, marker_val, marker_diff, marker_desc))
            elif marker_reg == DELAY_REG:
                for _i in range(marker_val - 1):
                    delay_sid_writes = [(FRAME_REG, 0, frame_diff, marker_desc)]
                    delay_sid_writes.extend(state.tick_frame())
                    add_frame(delay_sid_writes, advance=False)
                f_sid_writes.append((FRAME_REG, 0, frame_diff, marker_desc))
            else:
                assert False, f"unknown reg {marker_reg}"
            for i in range(start + 1, end):
                reg = int(regs[i])
                assert reg >= 0, (i, reg)
                op = int(ops[i])
                decoder = DECODERS.get(op)
                assert decoder is not None, f"unknown op {op} reg {reg}"
                row = _FastRow(
                    reg=reg,
                    val=int(vals[i]),
                    op=op,
                    subreg=int(subregs[i]),
                    diff=int(diffs[i]),
                    description=int(descs[i]),
                    Index=int(indices[i]),
                )
                writes = decoder.expand(row, state)
                if writes:
                    f_sid_writes.extend(writes)

            f_sid_writes.extend(state.tick_frame())
            add_frame(f_sid_writes)

        if not all_rows:
            return pd.DataFrame(
                columns=["reg", "val", "diff", "description"], dtype=MODEL_PDTYPE
            )
        df = pd.DataFrame(
            all_rows,
            columns=["reg", "val", "diff", "description", "__c"],
            dtype=MODEL_PDTYPE,
        )
        # One stable sort over the whole array reproduces what the old
        # code did with N per-frame ``sort_values("reg")`` calls.
        # Primary key = call_idx (one per add_frame call, including
        # the multiple calls a single DELAY_REG unrolling makes) so
        # tick-spawned frames stay in chronological order. Secondary
        # key = reg with stable ties so insertion order is preserved
        # within a call.
        df = (
            df.sort_values(["__c", "reg"], kind="stable")
            .drop(columns="__c")
            .reset_index(drop=True)
        )
        df["description"] = df["description"].ffill().fillna(0)
        return df

    def _filter_irq(self, df, name):
        try:
            irq = df["irq"].iloc[0]
        except (IndexError, KeyError):
            self.logger.info(df)
            self.logger.info("skipped %s, no irq", name)
            return False
        if irq < self.args.min_irq or irq > self.args.max_irq:
            self.logger.info("skipped %s, irq %u (outside IRQ range)", name, irq)
            return False
        return True

    def _filter(self, df, name):
        if len(df[df["reg"] == FRAME_REG]) == 0:
            self.logger.info("skipped %s, no frames", name)
            return False
        # Minimum useful song length. Was ``seq_len * 2`` (16384) which
        # rejected the bulk of HVSC -- BlockMapper pads short blocks
        # already, so the gate just needs to ensure the song has enough
        # frames to be musically meaningful for next-token training.
        min_song = getattr(self.args, "min_song_tokens", 256)
        if len(df) < min_song:
            self.logger.info("skipped %s, length %u (< %u)", name, len(df), min_song)
            return False
        vol = sorted(np.bitwise_and(df[df["reg"] == 24]["val"], 15).unique().tolist())
        if len(vol) >= 8:
            self.logger.info(
                "skipped %s, too many (%u) vol changes %s", name, len(vol), vol
            )
            return False
        c_df = self._norm_df(df)
        ctrl_mask = self._ctrl_match(df)
        # Only count actual ctrl-reg SET writes; macros (GATE_REPLAY_OP,
        # PLAY_INSTRUMENT_OP) reuse ctrl_reg as the row's ``reg`` field
        # for voice rotation, but they don't represent independent ctrl
        # changes that the LM must learn. The pre-encoding df lacks an
        # ``op`` column, so apply the SET-only filter only when present.
        if "op" in df.columns:
            ctrl_mask = ctrl_mask & (df["op"] == SET_OP)
        c_df = c_df[ctrl_mask]
        c_df["ccount"] = c_df.groupby(["f", "v"])["reg"].transform("size")
        c_df = c_df[c_df["f"] > 16]
        if len(c_df):
            c_max = c_df["ccount"].max()
            if c_max > 6:
                self.logger.info(
                    "skipped %s, too many (%u) control reg changes per frame",
                    name,
                    c_max,
                )
                return False
        return True

    def _combine_regs(self, df):
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for reg, bits in ((v_offset, 0), ((v_offset + 2), PCM_BITS)):
                df = self._combine_reg(df, reg=reg, bits=bits)
        df = self._combine_reg(df, FC_LO_REG, bits=FILTER_BITS)
        # _combine_reg defers clock-sorting; restore it once for the
        # whole df at the end so downstream stages see clock-ordered
        # rows.
        return df.sort_values("clock", kind="stable").reset_index(drop=True)

    def _consolidate_frames(self, orig_df):
        df = self._norm_df(orig_df.copy())
        m = (
            (df["reg"] == FRAME_REG)
            & (df["reg"].shift(-1) != FRAME_REG)
            & (df["reg"].shift(1) == FRAME_REG)
        )
        df.loc[m, "val"] = 1
        df.loc[m, "reg"] = DELAY_REG
        for i in (-1, 1):
            while True:
                m = (df["reg"] == DELAY_REG) & (df["reg"].shift(i) == FRAME_REG)
                df.loc[m, "val"] += 1
                m = (df["reg"] == FRAME_REG) & (df["reg"].shift(-i) == DELAY_REG)
                if len(df[m]) == 0:
                    break
                df = df[~m]
        while True:
            m = (df["reg"] == DELAY_REG) & (df["reg"].shift(1) == DELAY_REG)
            df.loc[m, "val"] += df["val"].shift(1)
            m = (df["reg"] == DELAY_REG) & (df["reg"].shift(-1) == DELAY_REG)
            if len(df[m]) == 0:
                break
            df = df[~m]
        m = df["reg"] == DELAY_REG
        df.loc[m, "val"] = df[m]["val"] - 1
        u_df = df[m].copy()
        u_df["n"] += 1
        u_df["reg"] = FRAME_REG
        u_df["val"] = 0
        df = pd.concat([df, u_df], ignore_index=False).sort_values("n")
        return df[orig_df.columns].reset_index(drop=True)

    def _squeeze_frame_regs(self, orig_df, regs=[0, 2, 21]):
        df = self._norm_df(orig_df.copy())
        df["dreg"] = pd.NA
        for reg in regs:
            if reg < VOICE_REG_SIZE:
                for v in range(VOICES):
                    dreg = v * VOICE_REG_SIZE + reg
                    df.loc[df["reg"] == dreg, "dreg"] = int(dreg)
            else:
                df.loc[df["reg"] == reg, "dreg"] = df["reg"]
        df = df[~df.duplicated(["f", "dreg"], keep="last") | df["dreg"].isna()]
        df = df[orig_df.columns].reset_index(drop=True)
        return df

    def _add_subreg(self, orig_df):
        df = orig_df.copy()
        sub_dfs = []
        df["subreg"] = int(-1)
        df["n"] = df.index * 10
        for reg in (4, 5, 6, 23, 24):
            m = df["reg"] == reg
            sub_df = df[m].copy()
            sub_df["subreg"] = 1
            sub_df["val"] = np.right_shift(sub_df["val"] & 0b11110000, 4)
            sub_df["n"] += 1
            df.loc[m, "val"] = df[m]["val"] & 0b0001111
            df.loc[m, "subreg"] = 0
            sub_dfs.append(sub_df)
        df = pd.concat([df] + sub_dfs, ignore_index=True).sort_values("n")
        df = df[list(orig_df.columns) + ["subreg"]]
        return df

    def parse(self, name, diffmax=512, max_perm=99, require_pq=False, reparse=False):
        if not reparse:
            parquet_glob = glob.glob(name.replace(DUMP_SUFFIX, PARSED_SUFFIX))
            if parquet_glob:
                for parquet_name in sorted(parquet_glob)[:max_perm]:
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
        irq, df = self._add_frame_reg(df, diffmax=2048)
        if not self._filter(df, name):
            return
        df = self._squeeze_frame_regs(df)
        df = self._add_change_regs(
            df, opcodes=[DIFF_OP, FLIP_OP, REPEAT_OP], cents=self.args.cents
        )
        df = self._consolidate_frames(df)
        df = self._cap_delay(df)
        delay_val = df[df["reg"] == DELAY_REG]["val"]
        if len(delay_val):
            delay_max = delay_val.max()
            assert delay_max < 256, delay_max
        irq = min(2 ** (IRQ_PDTYPE.itemsize * 8) - 1, irq)
        df["irq"] = irq
        while not df.empty and self._frame_match(df.iloc[-1]):
            df = df.head(len(df) - 1)
        while not df.empty and (
            (df.iloc[0]["reg"] == MODE_VOL_REG and df.iloc[0]["val"] == 15)
        ):
            df = df.tail(len(df) - 1)

        if not self._frame_match(df.iloc[0]):
            first_frame = df[df["reg"] == FRAME_REG].head(1)
            df = pd.concat([first_frame, df], ignore_index=True).reset_index(drop=True)

        for xdf in self._rotate_voice_augment(df, max_perm=max_perm):
            xdf = xdf[FRAME_DTYPES.keys()].astype(FRAME_DTYPES)
            xdf = macros.run_passes(xdf, args=self.args)
            xdf = self._norm_pr_order(xdf, v_only=False)
            # ``_filter`` measures the row count of the post-voice-reg
            # form; check against a temporary _add_voice_reg view BEFORE
            # LoopPass collapses bodies into 1-row BACK_REF tokens.
            # Otherwise short songs drop below the threshold post-loop
            # even when their pre-encoding form was fine.
            xdf_voice_preview = self._add_voice_reg(xdf, zero_voice_reg=True)
            if not self._filter(xdf_voice_preview, name):
                break
            # Post-norm but pre-voice-reg passes (LoopPass): regs are
            # absolute so DECODERS dispatch on the right voice for the
            # fuzzy-match state walk.
            xdf = macros.run_post_norm_pre_voice_passes(xdf, args=self.args)
            xdf = self._add_voice_reg(xdf, zero_voice_reg=True)
            # xdf = self._add_subreg(xdf)
            xdf = xdf.reset_index(drop=True)
            for k in TOKEN_KEYS:
                if k not in xdf.columns:
                    xdf[k] = int(-1)
            empty_val = xdf[xdf["val"].isna()]
            assert empty_val.empty, (name, empty_val)
            yield xdf
