"""BaccTokenizer: the PAD-shift adapter over the fixed BACC alphabet."""

import unittest

from preframr_tokens import VOCAB, BaccProgram, NoteOn

from preframr.tokenizer import PAD_ID, BaccTokenizer

_SEED_KEYS = (
    "notenum",
    "instrnr",
    "lnthcc",
    "lenleft",
    "sfl",
    "sfh",
    "porta",
    "vctrl",
    "pdly",
    "pdir",
)


def _synthetic_program():
    """A minimal valid BaccProgram (covers the serialize header + one note)."""
    seed = {k: [0, 0, 0] for k in _SEED_KEYS}
    seed["init_speed"] = 1
    seed["resetspd"] = 1
    return BaccProgram(
        driver="hubbard_monty",
        nframes=16,
        boot=[0] * 25,
        instruments=[[0] * 8 for _ in range(64)],
        score=[
            NoteOn(frame=0, voice=0, note=60, instr=1, lnth=2, porta=0),
            NoteOn(frame=4, voice=0, note=62, instr=1, lnth=2, porta=0),
        ],
        seed=seed,
        tables={"static_img": [0] * 256},
    )


class TestBaccTokenizer(unittest.TestCase):
    def test_vocab_shape(self):
        tk = BaccTokenizer()
        self.assertEqual(tk.n_vocab, VOCAB + 1)
        self.assertEqual(tk.n_words, VOCAB + 1)
        self.assertEqual(len(tk.tokens), VOCAB + 1)
        self.assertEqual(len(tk.token_metadata()), VOCAB + 1)
        self.assertIsNone(tk.tkmodel)
        self.assertEqual(tk.tokens[PAD_ID], "PAD")

    def test_encode_shifts_into_model_space(self):
        tk = BaccTokenizer()
        ids = tk.encode(_synthetic_program())
        self.assertTrue(all(1 <= i <= VOCAB for i in ids))

    def test_round_trip(self):
        tk = BaccTokenizer()
        prog = _synthetic_program()
        ids = tk.encode(prog)
        prog2 = tk.decode(ids, driver="hubbard_monty")
        self.assertEqual(tk.encode(prog2), ids)

    def test_decode_drops_pad_and_out_of_range(self):
        tk = BaccTokenizer()
        ids = tk.encode(_synthetic_program())
        padded = [PAD_ID] + ids + [PAD_ID, VOCAB + 5]
        self.assertEqual(tk.encode(tk.decode(padded, driver="hubbard_monty")), ids)


if __name__ == "__main__":
    unittest.main()
