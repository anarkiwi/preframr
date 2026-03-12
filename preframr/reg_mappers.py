from preframr.stfconstants import (
    MIDI_N_TO_F,
    PAL_CLOCK,
)


class FreqMapper:
    def __init__(self, cents=10, clock=PAL_CLOCK):
        f = MIDI_N_TO_F[0]
        sid_clock = (18 * 2**24) / clock
        max_sid_f = 65535 / sid_clock
        self.rq_map = {i: 0 for i in range(65536)}
        self.fi_map = {i: 0 for i in range(65536)}
        self.if_map = {}
        n = 0
        self.bits = 0

        while True:
            l = f * (2 ** ((-cents / 2) / 1200))
            h = f * (2 ** ((cents / 2) / 1200))
            lr = round(sid_clock * l)
            lh = round(sid_clock * h)
            r = round(sid_clock * f)
            if n < 65536:
                self.if_map[n] = r
            for i in range(lh - lr):
                j = i + lr
                if j >= 65536:
                    break
                self.rq_map[j] = r
                self.fi_map[j] = n
            f *= 2 ** (cents / 1200)
            n += 1
            if l > max_sid_f:
                break

        self.bits = int(len(self.if_map)).bit_length()
