from datetime import timedelta
from tqdm import tqdm
from scipy.io import wavfile
from pyresidfp import SoundInterfaceDevice
from pyresidfp.sound_interface_device import ChipModel
import pandas as pd
import numpy as np
from preframr.stfconstants import (
    CTRL_REG,
    DELAY_REG,
    FRAME_REG,
    RESET_REG,
    VOICE_REG,
    VOICES,
    VOICE_REG_SIZE,
    MODE_VOL_REG,
    MAX_REG,
)


def sidq(sid=None):
    if sid is None:
        sid = SoundInterfaceDevice()
    return sid.clock_frequency / 1e6 / 1e6


def write_reg(sid, reg, val, reg_widths):
    width = reg_widths.get(reg, 1)
    for i in range(width):
        sid.write_register(reg + i, val & 255)
        val >>= 8


def write_samples(orig_df, name, reg_widths, reg_start=None, irq=None):
    df = orig_df.copy()
    sid = SoundInterfaceDevice(model=ChipModel.MOS8580)
    if reg_start is None:
        reg_start = {MODE_VOL_REG: 15}
        for v in range(3):
            offset = v * VOICE_REG_SIZE
            # max sustain all voices
            reg_start[6 + offset] = 240
            # 50% pwm
            reg_start[3 + offset] = 16
    for reg, val in sorted(reg_start.items()):
        write_reg(sid, reg, val, reg_widths)
    frame_cond = df["reg"] == FRAME_REG
    if irq is None:
        irq = df[frame_cond]["diff"].iat[0]
    df.loc[df["reg"] == DELAY_REG, "diff"] = df["val"] * irq
    df["delay"] = df["diff"] * sidq(sid)

    df["f"] = (frame_cond).cumsum()
    df["fd"] = df["diff"]
    df.loc[df["reg"] < 0, "fd"] = pd.NA
    df["fd"] = df.groupby(["f"])["fd"].transform("sum") * sidq(sid)
    df.loc[frame_cond, "delay"] = df[frame_cond]["delay"] - df[frame_cond][
        "fd"
    ].shift().fillna(0)
    total_secs = df["delay"].sum() + 1

    raw_samples = np.zeros(int(sid.sampling_frequency * total_secs), dtype=np.int16)
    voice = None
    sp = 0

    for row in tqdm(df.itertuples(), total=len(df), ascii=True):
        if row.reg < 0:
            if row.reg == CTRL_REG:
                val = row.val
                for reg in (4, 11, 18):
                    sid.write_register(reg, val & 255)
                    val >>= 8
            elif row.reg == RESET_REG:
                for reg in range(MAX_REG + 1):
                    sid.write_register(reg, 0)
            elif row.reg == VOICE_REG:
                voice = row.val
            elif row.reg == FRAME_REG:
                voice = 0
        else:
            reg = row.reg
            if voice is not None and reg < VOICE_REG_SIZE:
                reg = (voice * VOICE_REG_SIZE) + reg
            write_reg(sid, reg, row.val, reg_widths)
        samples = sid.clock(timedelta(seconds=row.delay))
        raw_samples[sp : sp + len(samples)] = samples
        sp += len(samples)
    raw_samples = raw_samples[:sp]
    wavfile.write(name, int(sid.sampling_frequency), raw_samples)
