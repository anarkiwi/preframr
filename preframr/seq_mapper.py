from dataclasses import dataclass
import random
import numpy as np
import torch


@dataclass
class SeqMeta:
    irq: int


class SeqMapper:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.seq_map = None
        self.seqs = []
        self.seq_metas = []
        self.len = 0

    def add(self, seq, seq_meta):
        if len(seq) <= self.seq_len:
            raise ValueError(f"sequence too short ({len(seq)}")
        assert isinstance(seq, np.ndarray)
        assert seq.dtype == np.int64
        self.seqs.append((seq, seq_meta))
        self._rebuild_map()

    def _rebuild_map(self):
        self.len = 0
        seq_map = []
        for seq, _seq_meta in self.seqs:
            seq_map.append(self.len)
            self.len += len(seq) - self.seq_len
        self.seq_map = np.array(seq_map, dtype=np.uint64)

    def shuffle(self, seed=0):
        random.seed(seed)
        random.shuffle(self.seqs)
        random.seed()
        self._rebuild_map()

    def __len__(self):
        return self.len

    def slice_n(self, seq, n):
        return torch.from_numpy(seq[int(n) : int(n) + self.seq_len])

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        seq_i = np.clip(
            np.searchsorted(self.seq_map, index, side="right") - 1, a_min=0, a_max=None
        )
        seq = self.seqs[seq_i][0]
        seq_index = index - self.seq_map[seq_i]
        return (self.slice_n(seq, seq_index), self.slice_n(seq, seq_index + 1))
