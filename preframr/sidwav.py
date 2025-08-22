from datetime import timedelta
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


def sidq():
    sid = SoundInterfaceDevice()
    return sid.clock_frequency / 1e6 / 1e6


def write_samples(df, name, reg_widths):
    sid = SoundInterfaceDevice(model=ChipModel.MOS6581)
    # max vol
    sid.write_register(MODE_VOL_REG, 15)
    for v in range(3):
        offset = v * VOICE_REG_SIZE
        # max sustain all voices
        sid.write_register(6 + offset, 240)
        # 50% pwm
        sid.write_register(3 + offset, 16)
    raw_samples = []
    df["delay"] = df["diff"] * sidq()
    voice = None
    irq = df[df["reg"] == FRAME_REG]
    if not irq.empty:
        df["irq"] = irq["diff"].iloc[0]
    frame_duration = 0

    for row in df.itertuples():
        delay = row.delay
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
                delay -= frame_duration
                frame_duration = 0
            elif row.reg == DELAY_REG:
                delay = row.val * row.irq * sidq()
                if delay == 0:
                    continue
        else:
            val = row.val
            reg = row.reg
            width = reg_widths.get(reg, 1)
            if voice is not None and reg < VOICE_REG_SIZE:
                reg = (voice * VOICE_REG_SIZE) + reg
            for i in range(width):
                sid.write_register(reg + i, val & 255)
                val >>= 8
            frame_duration += delay
        if delay <= 0:
            continue
        raw_samples.extend(sid.clock(timedelta(seconds=delay)))
    wavfile.write(
        name,
        int(sid.sampling_frequency),
        np.array(raw_samples, dtype=np.float32) / 2.0**15,
    )
