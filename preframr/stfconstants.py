import pandas as pd

CTRL_REG = -2
DELAY_REG = -1
DIFF_PDTYPE = pd.UInt16Dtype()
FC_LO_REG = 21
FILTER_REG = 23
FRAME_REG = -99
IMPLIED_FRAME_REG = False
MAX_REG = 24
MODE_VOL_REG = 24
MODEL_PDTYPE = pd.Int32Dtype()
NOOP_REG = -4
PAD_ID = -999
REG_PDTYPE = pd.Int8Dtype()
RESET_REG = -3
TOKEN_KEYS = ["reg", "val", "diff"]
TOKEN_PDTYPE = pd.Int64Dtype()  # Same as torch
UNICODE_BASE = 0x300
VAL_PDTYPE = pd.UInt32Dtype()
VOICES = 3
VOICE_REG = -5
VOICE_REG_SIZE = 7

PAL_CLOCK = 17734475
TUNING_REF_HZ = 440
MIDI_N_TO_F = {n: (2 ** ((n - 69) / 12)) * TUNING_REF_HZ for n in range(128)}
