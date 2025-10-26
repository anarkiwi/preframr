from collections import defaultdict
from datetime import timedelta
import time
from tqdm import tqdm
from scipy.io import wavfile
from pyresidfp import SoundInterfaceDevice
from pyresidfp.sound_interface_device import ChipModel
import pandas as pd
import mido
import numpy as np
from preframr.stfconstants import (
    CTRL_REG,
    DELAY_REG,
    FRAME_REG,
    RESET_REG,
    VOICE_REG,
    VOICE_REG_SIZE,
    MODE_VOL_REG,
    MAX_REG,
)

ID_REG = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 12,
    11: 13,
    12: 14,
    13: 15,
    14: 16,
    15: 17,
    16: 19,
    17: 20,
    18: 21,
    19: 22,
    20: 23,
    21: 24,
    22: 4,
    23: 11,
    24: 18,
}

ELEKTRON_MANID = 0x2D
ASID_START = 0x4C
ASID_STOP = 0x4D
ASID_UPDATE = 0x4E


def default_sid():
    return SoundInterfaceDevice(model=ChipModel.MOS8580)


class AsidProxy:
    def __init__(self, sid, asid, update_cmd=ASID_UPDATE, sysex_delay=0.0001):
        self.sid = sid
        self.asid = asid
        self.port = None
        self.update_cmd = update_cmd
        self.sysex_delay = sysex_delay
        self.pending_delay = 0
        self.pending_frame = False
        self._resetreg()
        if self.asid is not None:
            output_names = set(mido.get_output_names())
            print(f"available mido ports: {output_names}")
            if self.asid not in output_names:
                for output_asid in sorted(output_names):
                    if output_asid.startswith(self.asid):
                        self.asid = output_asid
                        break
            self.port = mido.open_output(self.asid)

    def __enter__(self):
        self.stop()
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def cue_frame(self):
        self.pending_frame = True

    def _resetreg(self):
        self.regs = defaultdict(int)
        self.pending_regs = defaultdict(int)

    @property
    def clock_frequency(self):
        return self.sid.clock_frequency

    @property
    def sampling_frequency(self):
        return self.sid.sampling_frequency

    def write_register(self, reg, val):
        self.last_reg = reg
        self.pending_regs[reg] = val
        self.sid.write_register(reg, val)

    def clock(self, delta):
        self.update(delta.total_seconds())
        return self.sid.clock(delta)

    def _sysex(self, data):
        if self.asid:
            msg = mido.Message("sysex", data=[ELEKTRON_MANID] + data)
            self.port.send(msg)

    def start(self):
        self._sysex([ASID_START])
        self._resetreg()

    def stop(self):
        self._sysex([ASID_STOP])
        self._resetreg()

    def update(self, seconds):
        if not self.pending_frame:
            self.pending_delay += seconds
            return
        masks = [0, 0, 0, 0]
        msbs = [0, 0, 0, 0]
        vals = []
        sysex_delay = 0

        for reg_id, reg in sorted(ID_REG.items()):
            new_val = self.pending_regs.get(reg, None)
            if new_val is None:
                continue
            if new_val == self.regs[reg]:
                continue
            self.regs[reg] = new_val
            meta_byte = int(reg_id / 7)
            meta_bit = reg_id % 7
            masks[meta_byte] |= 2**meta_bit
            if new_val & 0x80:
                msbs[meta_byte] |= 2**meta_bit
            vals.append(new_val & 0x7F)
        if vals:
            update_message = [self.update_cmd] + masks + msbs + vals
            self._sysex(update_message)
            sysex_delay = len(update_message) * self.sysex_delay
        if self.port:
            time.sleep(seconds + self.pending_delay - sysex_delay)

        self.pending_delay = 0
        self.pending_frame = False
        self.pending_regs = defaultdict(int)


def sidq(sid=None):
    if sid is None:
        sid = default_sid()
    return sid.clock_frequency / 1e6 / 1e6


def write_reg(sid, reg, val, reg_widths):
    width = reg_widths.get(reg, 1)
    for i in range(width):
        sid.write_register(reg + i, val & 255)
        val >>= 8


def write_samples(
    orig_df,
    name,
    reg_widths,
    reg_start=None,
    irq=None,
    sid=None,
    asid=None,
    sysex_delay=0.0001,
):
    df = orig_df.copy()
    if sid is None:
        sid = default_sid()

    with AsidProxy(sid=sid, asid=asid, sysex_delay=sysex_delay) as proxy:
        if reg_start is None:
            reg_start = {MODE_VOL_REG: 15}
            for v in range(3):
                offset = v * VOICE_REG_SIZE
                # max sustain all voices
                reg_start[6 + offset] = 240
                # 50% pwm
                reg_start[3 + offset] = 16
        for reg, val in sorted(reg_start.items()):
            write_reg(proxy, reg, val, reg_widths)
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
                        proxy.write_register(reg, val & 255)
                        val >>= 8
                elif row.reg == RESET_REG:
                    for reg in range(MAX_REG + 1):
                        proxy.write_register(reg, 0)
                elif row.reg == VOICE_REG:
                    voice = row.val
                elif row.reg == FRAME_REG:
                    voice = 0
                    proxy.cue_frame()
            else:
                reg = row.reg
                if voice is not None and reg < VOICE_REG_SIZE:
                    reg = (voice * VOICE_REG_SIZE) + reg
                write_reg(proxy, reg, row.val, reg_widths)
            samples = proxy.clock(timedelta(seconds=row.delay))
            raw_samples[sp : sp + len(samples)] = samples
            sp += len(samples)
        raw_samples = raw_samples[:sp]
        wavfile.write(name, int(sid.sampling_frequency), raw_samples)
