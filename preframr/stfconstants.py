import pandas as pd


FRAME_REG = -128
DELAY_REG = -127
VOICE_REG = -126
FC_LO_REG = 21
FILTER_REG = 23

DIFF_PDTYPE = pd.UInt16Dtype()
IMPLIED_FRAME_REG = False
MAX_REG = 24
MIN_DIFF = 32
MODE_VOL_REG = 24
MODEL_PDTYPE = pd.Int32Dtype()
PAD_ID = -999
REG_PDTYPE = pd.Int8Dtype()
TOKEN_KEYS = ["reg", "val"]
TOKEN_PDTYPE = pd.Int64Dtype()  # Same as torch
UNICODE_BASE = 0x300
VAL_PDTYPE = pd.Int32Dtype()
VOICES = 3
VOICE_REG_SIZE = 7

PAL_CLOCK = 17734475
TUNING_REF_HZ = 440
MIDI_N_TO_F = {n: (2 ** ((n - 69) / 12)) * TUNING_REF_HZ for n in range(128)}
