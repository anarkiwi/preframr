"""Thin torch.utils.data.Dataset adapter around preframr.corpus.Corpus + preframr.train.block_mapper.BlockMapper. Corpus owns the torch-free state (BaccTokenizer, n_vocab) + the (.sid,.dump)->block-array orchestration; this module routes Corpus's yielded (kind, blocks_path, seq_meta) tuples into per-subset BlockMappers and exposes the torch Dataset protocol."""

import logging
from collections import OrderedDict

import torch

from preframr.corpus import Corpus
from preframr.train.block_mapper import BlockMapper

LEGACY_EVAL_SUBSET_NAME = "val"


def get_prompt(args, dataset, logger):
    """First ``prompt_seq_len`` model ids of the selected block + its tail truth; BACC tokens are self-contained ids so a prompt is just a slice (the old register-preamble reconstruction is gone). Returns ``(irq, n, prompt, prompt_compare, seq_meta)``."""
    seq, seq_meta = dataset.getseq(args.start_seq, block_j=args.start_block)
    logger.info(
        "starting at seq %u block %u (%s), %u tokens, irq %s",
        args.start_seq,
        args.start_block,
        seq_meta.df_file,
        len(seq),
        seq_meta.irq,
    )
    n = args.max_seq_len - args.prompt_seq_len
    if n <= 0:
        raise ValueError("max seq length too short")
    prompt = seq[: args.prompt_seq_len].unsqueeze(0).long()
    prompt_compare = seq[: args.max_seq_len]
    return seq_meta.irq, n, prompt, prompt_compare, seq_meta


class RegDataset(torch.utils.data.Dataset):
    """torch Dataset adapter around Corpus + BlockMapper."""

    def __init__(self, args, logger=logging):
        self.corpus = Corpus(args, logger)
        self.block_mapper = BlockMapper(args.seq_len)
        self.val_block_mappers = OrderedDict()
        self._empty_val_mapper = None

    @property
    def args(self):
        return self.corpus.args

    @property
    def logger(self):
        return self.corpus.logger

    @property
    def tokenizer(self):
        return self.corpus.tokenizer

    @property
    def reg_widths(self):
        return self.corpus.reg_widths

    @reg_widths.setter
    def reg_widths(self, value):
        self.corpus.reg_widths = value

    @property
    def n_vocab(self):
        return self.corpus.n_vocab

    @property
    def n_words(self):
        return self.corpus.n_words

    @property
    def val_block_mapper(self):
        """Back-compat: legacy single-subset accessor."""
        if LEGACY_EVAL_SUBSET_NAME in self.val_block_mappers:
            return self.val_block_mappers[LEGACY_EVAL_SUBSET_NAME]
        if self.val_block_mappers:
            return next(iter(self.val_block_mappers.values()))
        if self._empty_val_mapper is None:
            self._empty_val_mapper = BlockMapper(self.corpus.args.seq_len)
        return self._empty_val_mapper

    def _val_subset_for(self, name):
        if name not in self.val_block_mappers:
            self.val_block_mappers[name] = BlockMapper(self.corpus.args.seq_len)
        return self.val_block_mappers[name]

    def _route(self, kind):
        if kind == "train":
            return self.block_mapper
        return self._val_subset_for(kind)

    def _finalize_mappers(self):
        self.block_mapper.finalize()
        for mapper in self.val_block_mappers.values():
            mapper.finalize()

    def preload(self, tokens=None, tkmodel=None):
        self.corpus.preload(tokens=tokens, tkmodel=tkmodel)

    def load(self):
        for kind, blocks_path, seq_meta in self.corpus.iter_block_seqs():
            self._route(kind).add(blocks_path, seq_meta)
        self._finalize_mappers()

    def predict_load(self):
        for kind, blocks_path, seq_meta in self.corpus.iter_predict_block_seqs():
            self._route(kind).add(blocks_path, seq_meta)
        self._finalize_mappers()

    def __len__(self):
        return len(self.block_mapper)

    def __getitem__(self, index):
        return self.block_mapper[index]

    def getseq(self, rotation_i, block_j=0):
        """Return ``(block, seq_meta)`` for one block of one rotation."""
        import numpy as np  # pylint: disable=import-outside-toplevel

        predict_set = getattr(self.corpus.args, "predict_set", "train")
        if predict_set == "train":
            mapper = self.block_mapper
        elif predict_set in self.val_block_mappers:
            mapper = self.val_block_mappers[predict_set]
        else:
            mapper = self.val_block_mapper
        block = mapper.get_block(rotation_i=rotation_i, block_j=block_j)
        _path, seq_meta, _n = mapper.block_metas[rotation_i]
        return torch.from_numpy(np.copy(block)), seq_meta


class LowMemoryRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            yield torch.randint(low=0, high=len(self.data_source), size=(1,)).item()

    def __len__(self):
        return self.num_samples


def _get_loader(args, dataset):
    if args.shuffle:
        length = args.shuffle / 1.0
        sampler = torch.utils.data.RandomSampler(
            dataset,
            num_samples=int(length * len(dataset)),
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=2,
    )


def get_loader(args, dataset):
    dataset.load()
    return _get_loader(args, dataset)


def get_val_loader(args, dataset):
    """Return validation DataLoaders + the parallel subset-name list."""
    if not getattr(args, "eval_manifest", ""):
        return None, []
    loaders = []
    names = []
    for name, mapper in dataset.val_block_mappers.items():
        if len(mapper) == 0:
            continue
        loaders.append(
            torch.utils.data.DataLoader(
                mapper,
                sampler=torch.utils.data.SequentialSampler(mapper),
                pin_memory=True,
                batch_size=args.batch_size,
                num_workers=2,
            )
        )
        names.append(name)
    if not loaders:
        return None, []
    if len(loaders) == 1:
        return loaders[0], names
    return loaders, names
