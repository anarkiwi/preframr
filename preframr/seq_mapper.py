from dataclasses import dataclass
import random
import numpy as np
import torch

from preframr.stfconstants import DUMP_SUFFIX


@dataclass
class SeqMeta:
    irq: int
    df_file: str
    i: int
    l: int = None
    npy_path: int = None


class SeqMapper(torch.utils.data.Dataset):
    def __init__(self, seq_len, mmap=True):
        self.seq_len = seq_len
        self.mmap = mmap
        self.seq_map = None
        self.seqs = {}
        self.seq_metas = []
        self.len = 0
        self.finalized = False

    def load(self):
        return

    def add(self, seq, seq_meta):
        if len(seq) <= self.seq_len:
            raise ValueError(f"sequence too short ({len(seq)}")
        assert isinstance(seq, np.ndarray), type(seq)
        assert seq.dtype == np.int16
        seq_meta.npy_path = seq_meta.df_file.replace(DUMP_SUFFIX, f".{seq_meta.i}.npy")
        seq_meta.l = len(seq)
        np.save(seq_meta.npy_path, seq)
        self.seq_metas.append(seq_meta)
        self.finalized = False

    def finalize(self):
        self._rebuild_map()
        self.finalized = True

    def _rebuild_map(self):
        self.len = 0
        self.seqs = {}
        seq_map = []
        for seq_meta in self.seq_metas:
            seq_map.append(self.len)
            self.len += seq_meta.l - self.seq_len
        self.seq_map = np.array(seq_map, dtype=np.uint64)

    def shuffle(self, seed=0):
        random.seed(seed)
        random.shuffle(self.seq_metas)
        random.seed()
        self._rebuild_map()
        self.finalized = False

    def __len__(self):
        return self.len

    def getseq(self, seq_i):
        try:
            seq = self.seqs[seq_i]
        except KeyError:
            seq_meta = self.seq_metas[seq_i]
            if self.mmap:
                self.seqs[seq_i] = np.load(seq_meta.npy_path, mmap_mode="r")
            else:
                self.seqs[seq_i] = np.load(seq_meta.npy_path)
            seq = self.seqs[seq_i]
        return (seq, self.seq_metas[seq_i])

    def slice_n(self, seq, n):
        return torch.from_numpy(seq.astype(np.int64)[int(n) : int(n) + self.seq_len])

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        if not self.finalized:
            raise ValueError

        seq_i = np.clip(
            np.searchsorted(self.seq_map, index, side="right") - 1, a_min=0, a_max=None
        )
        seq, _seq_meta = self.getseq(seq_i)
        seq_index = index - self.seq_map[seq_i]
        return (self.slice_n(seq, seq_index), self.slice_n(seq, seq_index + 1))


class BlockMapper(torch.utils.data.Dataset):
    """Block-array variant of ``SeqMapper`` for self-contained blocks.

    Reads ``<dump>.<i>.blocks.npy`` files (2D shape ``(num_blocks, N+1)``;
    the trailing slot makes shifted-target slicing trivial). One block =
    one training sample; the LM sees it as ``input[:-1]`` and predicts
    ``input[1:]``. Each block is self-contained (no undefined op refs)
    by construction of ``iter_self_contained_row_blocks`` at parse time.
    """

    def __init__(self, seq_len, mmap=True):
        self.seq_len = seq_len
        self.mmap = mmap
        self.block_metas = []  # list[(path, seq_meta, num_blocks)]
        self.loaded = {}
        self.cum_counts = None
        self.total = 0
        self.finalized = False

    def add(self, blocks_path, seq_meta):
        arr = np.load(blocks_path, mmap_mode="r" if self.mmap else None)
        num = int(arr.shape[0])
        if num <= 0:
            return
        self.block_metas.append((blocks_path, seq_meta, num))
        self.finalized = False

    def finalize(self):
        cum = 0
        cums = []
        for _path, _meta, n in self.block_metas:
            cums.append(cum)
            cum += n
        self.cum_counts = np.array(cums, dtype=np.int64)
        self.total = cum
        self.finalized = True

    def shuffle(self, seed=0):
        random.seed(seed)
        random.shuffle(self.block_metas)
        random.seed()
        self.finalize()

    def __len__(self):
        return self.total

    def _get_arr(self, path):
        if path not in self.loaded:
            self.loaded[path] = np.load(
                path, mmap_mode="r" if self.mmap else None
            )
        return self.loaded[path]

    def __getitem__(self, index):
        if not self.finalized:
            raise ValueError("call finalize() first")
        if index >= self.total:
            raise IndexError
        which = int(np.searchsorted(self.cum_counts, index, side="right") - 1)
        which = max(which, 0)
        block_i = int(index - self.cum_counts[which])
        path, _meta, _n = self.block_metas[which]
        block = self._get_arr(path)[block_i]
        # Shifted targets: model predicts token t+1 from token t. Block is
        # length seq_len+1; slice into (input, target) of length seq_len.
        x = torch.from_numpy(block.astype(np.int64)[:-1].copy())
        y = torch.from_numpy(block.astype(np.int64)[1:].copy())
        return x, y

    def get_block(self, rotation_i=0, block_j=0):
        """Read a single block by (rotation, block) -- used by predict."""
        path, _meta, _n = self.block_metas[rotation_i]
        return self._get_arr(path)[block_j]
