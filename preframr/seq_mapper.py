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
    def __init__(self, seq_len, mmap=False):
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
        seq_meta.npy_path = seq_meta.df_file.replace(DUMP_SUFFIX, f"{seq_meta.i}.npy")
        seq_meta.l = len(seq)
        np.save(seq_meta.npy_path, seq.astype(np.int64))
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
        return torch.from_numpy(seq[int(n) : int(n) + self.seq_len])

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
