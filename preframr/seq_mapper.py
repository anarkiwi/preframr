from dataclasses import dataclass
import random
import numpy as np
import torch


@dataclass
class SeqMeta:
    irq: int
    df_file: str
    i: int
    l: int = None
    npy_path: int = None


class BlockMapper(torch.utils.data.Dataset):
    """Per-rotation block storage for both training and inference.

    Reads ``<dump>.<i>.blocks.npy`` files (2D shape ``(num_blocks, N+1)``;
    the trailing slot makes shifted-target slicing trivial). One block =
    one training sample; the LM sees it as ``input[:-1]`` and predicts
    ``input[1:]``. Each block is self-contained (no undefined op refs)
    by construction of ``iter_self_contained_row_blocks`` at parse time.
    Inference (``RegDataset.getseq``) returns one whole block as the
    prompt source so it sees exactly what training saw.
    """

    def __init__(self, seq_len, mmap=True):
        self.seq_len = seq_len
        self.mmap = mmap
        self.block_metas = []  # list[(path, seq_meta, num_blocks)]
        self.loaded = {}
        # Start in a "finalized empty" state so an unloaded BlockMapper
        # behaves like an empty sequence (len=0, __getitem__ raises
        # IndexError) rather than ValueError("call finalize() first").
        # ``add`` resets ``finalized`` so callers must re-finalize before
        # iteration.
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
        # Shifted targets: model predicts token t+1 from token t. Block is
        # length seq_len+1; slice into (input, target) of length seq_len.
        x = torch.from_numpy(block.astype(np.int64)[:-1].copy())
        y = torch.from_numpy(block.astype(np.int64)[1:].copy())
        return x, y

    def get_block(self, rotation_i=0, block_j=0):
        """Read a single block by (rotation, block) -- used by predict."""
        path, _meta, _n = self.block_metas[rotation_i]
        return self._get_arr(path)[block_j]
