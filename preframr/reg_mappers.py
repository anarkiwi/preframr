from preframr.stfconstants import (
    CENTS,
    MIDI_N_TO_F,
    PAL_CLOCK,
)


class FreqMapper:
    def __init__(self, cents=CENTS, clock=PAL_CLOCK):
        f = MIDI_N_TO_F[0]
        sid_clock = (18 * 2**24) / clock
        max_sid_f = 65535 / sid_clock
        self.rq_map = {i: 0 for i in range(65536)}
        self.fi_map = {i: 0 for i in range(65536)}
        self.if_map = {}
        n = 0

        while True:
            l = f * (2 ** ((-cents / 2) / 1200))
            h = f * (2 ** ((cents / 2) / 1200))
            lr = round(sid_clock * l)
            lh = round(sid_clock * h)
            r = round(sid_clock * f)
            self.if_map[n] = r
            for i in range(lh - lr):
                self.rq_map[i + lr] = r
                self.fi_map[i + lr] = n
            f *= 2 ** (cents / 1200)
            n += 1
            if f > max_sid_f:
                break
