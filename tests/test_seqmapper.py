import unittest
import numpy as np
from preframr.seq_mapper import SeqMapper


class TestSeqMapper(unittest.TestCase):
    def test_seq_mapper(self):
        s = SeqMapper(2)
        s.add(np.array([1, 2, 3, 4], dtype=np.int64), 0)
        s.add(np.array([8, 9, 10, 11, 12, 13, 14], dtype=np.int64), 0)
        s.add(np.array([99, 100, 101], dtype=np.int64), 0)
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
