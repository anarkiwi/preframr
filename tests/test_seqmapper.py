import tempfile
import os
import unittest
import numpy as np
from preframr.seq_mapper import SeqMapper, SeqMeta


class TestSeqMapper(unittest.TestCase):
    def test_seq_mapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df_file = os.path.join(tmpdir, "test.parquet")
            s = SeqMapper(2)
            for i, seq in enumerate(
                ([1, 2, 3, 4], [8, 9, 10, 11, 12, 13, 14], [99, 100, 101])
            ):
                seq = np.array(seq, dtype=np.int16)
                df_file = os.path.join(tmpdir, f"test{i}.dump.parquet")
                seq_meta = SeqMeta(irq=1, df_file=df_file, i=i)
                s.add(seq, seq_meta)
            s.finalize()
            self.assertEqual(len(s), 8)
            results = [tuple(x.tolist() for x in s[i]) for i in range(len(s))]
            self.assertEqual(
                results,
                [
                    ([1, 2], [2, 3]),
                    ([2, 3], [3, 4]),
                    ([8, 9], [9, 10]),
                    ([9, 10], [10, 11]),
                    ([10, 11], [11, 12]),
                    ([11, 12], [12, 13]),
                    ([12, 13], [13, 14]),
                    ([99, 100], [100, 101]),
                ],
            )
