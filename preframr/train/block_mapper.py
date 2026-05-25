import random
import numpy as np
import torch


class BlockMapper(torch.utils.data.Dataset):
    """Per-rotation block storage for both training and inference."""

    def __init__(self, seq_len, mmap=True):
        self.seq_len = seq_len
        self.mmap = mmap
        self.block_metas = []
        self.loaded = {}
        self.cum_counts = np.zeros(0, dtype=np.int64)
        self.total = 0
        self.finalized = True

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
            self.loaded[path] = np.load(path, mmap_mode="r" if self.mmap else None)
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
        x = torch.from_numpy(block.astype(np.int64)[:-1].copy())
        y = torch.from_numpy(block.astype(np.int64)[1:].copy())
        return x, y

    def get_block(self, rotation_i=0, block_j=0):
        """Read a single block by (rotation, block) -- used by predict."""
        path, _meta, _n = self.block_metas[rotation_i]
        return self._get_arr(path)[block_j]
