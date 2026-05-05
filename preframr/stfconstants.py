import pandas as pd

FRAME_REG = -128
DELAY_REG = -127
VOICE_REG = -126
FC_LO_REG = 21
FILTER_REG = 23

SET_OP = 0
DIFF_OP = 1
REPEAT_OP = 2
FLIP_OP = 3
PWM_OP = 4
TRANSPOSE_OP = 5
FLIP2_OP = 7
INTERVAL_OP = 8
END_REPEAT_OP = 9
END_FLIP_OP = 10
FILTER_SWEEP_OP = 11
BACK_REF_OP = 15
DO_LOOP_OP = 16
SUBREG_FLUSH_OP = 18
GATE_REPLAY_OP = 19
PLAY_INSTRUMENT_OP = 20
# GATE_TOGGLE_OP (=6), FILTER_ROUTE_OP (=12), MASTER_VOL_OP (=13),
# FILTER_MODE_OP (=14) were retired when SubregPass took over the byte
# splitting they hand-rolled. Their op codes are deliberately not reused
# so any rare third-party stream that contains them fails loudly.
# 17 reserved for PATTERN_REF_OP

# Sentinel reg used by BACK_REF and DO_LOOP rows. Distinct from FRAME_REG
# (-128), DELAY_REG (-127), VOICE_REG (-126), and outside any valid SID
# register range, so loop ops fail loudly if they ever reach _expand_ops
# without being materialized by expand_loops first.
LOOP_OP_REG = -125

DIFF_PDTYPE = pd.UInt16Dtype()
IMPLIED_FRAME_REG = False
MAX_REG = 24
MIN_DIFF = 32
MODE_VOL_REG = 24
MODEL_PDTYPE = pd.Int32Dtype()
PAD_ID = 0
REG_PDTYPE = pd.Int8Dtype()
TOKEN_KEYS = ["op", "reg", "subreg", "val"]
TOKEN_PDTYPE = pd.Int64Dtype()  # Same as torch
UNICODE_BASE = 0x300
VAL_PDTYPE = pd.Int32Dtype()
VOICES = 3
VOICE_REG_SIZE = 7
META_FREQ_BITS = 4
FILTER_BITS = 5
PCM_BITS = 5

PAL_CLOCK = 17734475
TUNING_REF_HZ = 440
MIDI_N_TO_F = {n: (2 ** ((n - 69) / 12)) * TUNING_REF_HZ for n in range(128)}

DUMP_SUFFIX = r".dump.parquet"
UNI_SUFFIX = r".uni.zst"
PARSED_SUFFIX = r".[0-9]*.parquet"
