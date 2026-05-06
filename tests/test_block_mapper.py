import tempfile
import os
import unittest
import numpy as np

from preframr.block_mapper import BlockMapper, SeqMeta


class TestBlockMapper(unittest.TestCase):
    def _write_blocks(self, tmpdir, name, blocks):
        path = os.path.join(tmpdir, name)
        np.save(path, np.array(blocks, dtype=np.int16))
        return path

    def test_empty_unfinalized(self):
        bm = BlockMapper(seq_len=4)
        # ``__init__`` leaves the mapper in a finalised-empty state so
        # an unloaded BlockMapper behaves like an empty dataset.
        self.assertEqual(len(bm), 0)
        with self.assertRaises(IndexError):
            bm[0]

    def test_add_and_iterate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Each block is length seq_len + 1 = 5.
            blocks_a = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]
            blocks_b = [[20, 21, 22, 23, 24]]
            path_a = self._write_blocks(tmpdir, "a.blocks.npy", blocks_a)
            path_b = self._write_blocks(tmpdir, "b.blocks.npy", blocks_b)

            bm = BlockMapper(seq_len=4, mmap=False)
            bm.add(path_a, SeqMeta(irq=1, df_file="a.dump.parquet", i=0))
            bm.add(path_b, SeqMeta(irq=1, df_file="b.dump.parquet", i=0))
            bm.finalize()

            self.assertEqual(len(bm), 3)
            # ``__getitem__`` returns (input, target) split out of each
            # length-5 block: input = block[:-1], target = block[1:].
            x0, y0 = bm[0]
            self.assertEqual(x0.tolist(), [1, 2, 3, 4])
            self.assertEqual(y0.tolist(), [2, 3, 4, 5])
            x2, y2 = bm[2]
            self.assertEqual(x2.tolist(), [20, 21, 22, 23])
            self.assertEqual(y2.tolist(), [21, 22, 23, 24])

    def test_get_block_by_rotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            blocks = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]
            path = self._write_blocks(tmpdir, "a.blocks.npy", blocks)
            bm = BlockMapper(seq_len=4, mmap=False)
            bm.add(path, SeqMeta(irq=1, df_file="a.dump.parquet", i=0))
            bm.finalize()
            self.assertEqual(
                bm.get_block(rotation_i=0, block_j=0).tolist(), [1, 2, 3, 4, 5]
            )
            self.assertEqual(
                bm.get_block(rotation_i=0, block_j=1).tolist(), [10, 11, 12, 13, 14]
            )

    def test_finalize_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_blocks(tmpdir, "a.blocks.npy", [[1, 2, 3, 4, 5]])
            bm = BlockMapper(seq_len=4, mmap=False)
            bm.add(path, SeqMeta(irq=1, df_file="a.dump.parquet", i=0))
            with self.assertRaises(ValueError):
                bm[0]


if __name__ == "__main__":
    unittest.main()
