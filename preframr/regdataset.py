import concurrent.futures
import copy
import logging
import glob
import io
import multiprocessing
import os
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import zstandard as zstd
from preframr.macros import (
    iter_self_contained_row_blocks,
    materialize_back_refs_outside,
    materialize_gate_palette_outside,
    materialize_instrument_palette_outside,
    self_contain_slice,
)
from preframr.reglogparser import RegLogParser
from preframr.regtokenizer import RegTokenizer
from preframr.seq_mapper import BlockMapper, SeqMapper, SeqMeta
from preframr.stfconstants import (
    DELAY_REG,
    DUMP_SUFFIX,
    FRAME_REG,
    PARSED_SUFFIX,
)


def glob_dumps(reglogs, max_files, min_dump_size, require_pq, seed=0):
    random.seed(seed)
    dump_files = []
    for r in reglogs.split(","):
        max_globbed = max_files - len(dump_files)
        if max_globbed <= 0:
            break
        pre_globbed = glob.glob(r, recursive=True)
        random.shuffle(pre_globbed)
        globbed = []
        for f in pre_globbed:
            if os.path.getsize(f) >= min_dump_size and (
                not require_pq or glob.glob(f.replace(DUMP_SUFFIX, PARSED_SUFFIX))
            ):
                globbed.append(f)
                if len(globbed) >= max_globbed:
                    break
        dump_files.extend(globbed[:max_globbed])
    random.seed()
    return dump_files


def materialize_block_array(
    tokenizer, raw_df, seq_len, parser, reg_widths, frames_per_block=None
):
    """Materialise the encoded ``raw_df`` (post-voice-reg, post-LoopPass)
    into a fixed-size 2D array of self-contained blocks.

    Block storage size is ``seq_len + 1`` so ``BlockMapper`` can split each
    row into ``input = block[:-1]`` and shifted target ``target = block[1:]``.
    Longer blocks are truncated; shorter ones pad with zero (which encodes
    the unk/pad slot in the tokenizer's vocabulary).

    Internally calls ``parser._remove_voice_reg`` to strip the voice-
    rotation markers so the simulator inside ``self_contain_slice`` sees
    absolute regs (the same form the predict path passes to
    ``_self_contained_prompt_df``). After block extraction, the absolute-
    reg block is round-tripped through ``parser._add_voice_reg`` to
    restore the LM's view; without that, every block would lose voice
    rotation and the tokenizer would fail to merge.

    The materialiser shares ``iter_self_contained_row_blocks`` with the
    predict path so a block produced at parse time decodes identically
    to one produced on-the-fly during inference.
    """
    block_size = seq_len + 1
    if frames_per_block is None:
        # Default: a generous frame budget that almost always tokenizes
        # under ``block_size``. Bin-search refinement is a future
        # optimisation; for now a 2-tokens-per-frame estimate works for
        # the post-bundle-decoupling encoding.
        frames_per_block = max(1, seq_len // 2)

    abs_df, _ = parser._remove_voice_reg(raw_df.copy(), reg_widths)

    blocks = []
    for block_df in iter_self_contained_row_blocks(
        abs_df, frames_per_block, args=parser.args
    ):
        if block_df.empty:
            continue
        # Re-add voice rotation so the block matches the LM's input form.
        # _add_voice_reg recomputes per-frame VOICE_REG markers from the
        # block's absolute regs alone, so a block produced this way is
        # byte-identical to the corresponding slice of the full-song
        # voice-reg'd form.
        try:
            voiced = parser._add_voice_reg(block_df.copy(), zero_voice_reg=True)
        except Exception:
            continue
        merged = tokenizer.merge_token_df(tokenizer.tokens, voiced.copy())
        if merged is None or "n" not in merged.columns:
            continue
        n = merged["n"].astype(np.int16).to_numpy()
        seq = tokenizer.encode(n).astype(np.int16)
        if len(seq) >= block_size:
            blocks.append(seq[:block_size])
        else:
            padded = np.zeros(block_size, dtype=np.int16)
            padded[: len(seq)] = seq
            blocks.append(padded)
    if not blocks:
        return np.zeros((0, block_size), dtype=np.int16)
    return np.stack(blocks)


def parser_worker(args, logger, dump_file, max_perm):
    reg_log_parser = RegLogParser(args, logger)
    dfs = [
        df
        for df in reg_log_parser.parse(
            dump_file, max_perm=max_perm, require_pq=args.require_pq
        )
    ]
    return dump_file, dfs


def get_prompt(args, dataset, logger):
    seq, seq_meta = dataset.getseq(args.start_seq)
    if args.start_n is None:
        # Don't predict past where we can compare accuracy.
        start = random.randint(0, len(seq) - args.max_seq_len)
    else:
        start = args.start_n
    logger.info(
        "starting at seq %u (%s), %u / %u, irq %u",
        args.start_seq,
        seq_meta.df_file,
        start,
        len(seq),
        seq_meta.irq,
    )
    n = args.max_seq_len - args.prompt_seq_len
    if n <= 0:
        raise ValueError("max seq length too short")
    prompt = seq[start:][: args.prompt_seq_len].unsqueeze(0).long()
    prompt_compare = seq[start:][: args.max_seq_len]
    loader = RegLogParser()
    preamble_df, _reg_widths = loader._remove_voice_reg(
        loader._state_df(
            dataset.tokenizer.decode(seq[:start].numpy()), dataset, seq_meta.irq
        ),
        dataset.reg_widths,
    )
    reg_start = {
        r: preamble_df[preamble_df["reg"] == r]["val"].iat[-1]
        for r in preamble_df["reg"].unique()
        if r >= 0
    }
    # Slice the prompt's frame range out of the full sequence and
    # materialise any BACK_REF / GATE_REPLAY whose target lies before the
    # slice so the prompt's df is self-contained at decode time.
    prompt_df_self_contained = _self_contained_prompt_df(
        loader, dataset, seq, start, args.prompt_seq_len, seq_meta.irq
    )
    return (
        seq_meta.irq,
        n,
        prompt,
        prompt_compare,
        reg_start,
        prompt_df_self_contained,
    )


def _self_contained_prompt_df(loader, dataset, seq, start, prompt_seq_len, irq):
    """Return a row-level prompt df where BACK_REF / GATE_REPLAY rows whose
    targets fall before the prompt have been materialised into literal
    frames. Decoders can then expand the df without the preamble in scope.

    Coordinates are *logical frame slots* (each FRAME_REG / DELAY_REG row is
    one slot), matching ``materialize_back_refs_outside``."""
    full_states = dataset.tokenizer.decode(seq.numpy())
    full_df = loader._state_df(full_states, dataset, irq)
    full_df, _ = loader._remove_voice_reg(full_df, dataset.reg_widths)
    if "op" not in full_df.columns:
        # Tokenizer produced a df without op metadata (no macros possible);
        # nothing to materialise.
        return full_df.iloc[start : start + prompt_seq_len].reset_index(drop=True)
    is_marker = full_df["reg"].isin({FRAME_REG, DELAY_REG})
    slice_lo = int(is_marker.iloc[:start].sum())
    slice_hi = slice_lo + int(
        is_marker.iloc[start : start + prompt_seq_len].sum()
    )
    # Same materialisation chain the training-time block iterator uses
    # (parse-time block_array generation funnels through
    # ``self_contain_slice`` too), so a prompt produced here matches the
    # corresponding block byte-for-byte. Passing args triggers the
    # re-encode path so palette indices in the prompt are slice-local
    # and never reference slots defined before slice_lo.
    return self_contain_slice(full_df, slice_lo, slice_hi, args=dataset.args)


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, args, logger=logging):
        self.args = args
        self.logger = logger
        self.n_vocab = 0
        self.n_words = 0
        self.reg_widths = {}
        self.seq_mapper = SeqMapper(args.seq_len)
        # Self-contained block storage; populated alongside seq_mapper
        # during load() from per-rotation ``.blocks.npy`` files written
        # at parse time. When non-empty, ``__getitem__`` reads from
        # block_mapper so each batch sample is a self-contained block.
        self.block_mapper = BlockMapper(args.seq_len)
        self.tokenizer = RegTokenizer(args, tokens=None, logger=logger)

    def load_dfs(self, reglogs=None, dump_files=None, max_perm=99, encode=True):
        if not dump_files:
            if not reglogs:
                raise ValueError("need reglogs or dump_files")
            dump_files = glob_dumps(
                reglogs,
                int(self.args.max_files * 1.25),
                self.args.min_dump_size,
                self.args.require_pq,
                seed=0,
            )
        output_dumps = set()
        max_workers = min(multiprocessing.cpu_count(), len(dump_files))
        max_files = min(self.args.max_files, len(dump_files))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(
                    parser_worker, self.args, self.logger, dump_file, max_perm
                )
                for dump_file in dump_files[:max_workers]
            ]
            dump_files = dump_files[max_workers:]
            done = False
            with tqdm(total=max_files) as pbar:
                while futures and len(output_dumps) < max_files:
                    new_futures = []
                    for future in concurrent.futures.as_completed(futures):
                        dump_file, dfs = future.result()
                        if dump_files:
                            new_futures.append(
                                executor.submit(
                                    parser_worker,
                                    self.args,
                                    self.logger,
                                    dump_files[0],
                                    max_perm,
                                )
                            )
                            dump_files = dump_files[1:]
                        for i, df in enumerate(dfs):
                            seq = None
                            # Preserve the pre-tokenize encoded form for
                            # block materialisation; merge_token_df mutates
                            # df by adding an ``n`` column that confuses
                            # subsequent re-merges of materialised rows.
                            raw_df = df.copy()
                            if self.tokenizer.tokens is not None:
                                df = self.tokenizer.merge_token_df(
                                    self.tokenizer.tokens, df
                                )
                                if encode:
                                    n = df["n"].astype(np.int16).to_numpy()
                                    seq = self.tokenizer.encode(n).astype(np.int16)
                                    if len(seq) < self.args.seq_len:
                                        self.logger.info(
                                            "rejecting sequence from %s too short %u",
                                            dump_file,
                                            len(seq),
                                        )
                                        break
                                    # Self-contained block file generation
                                    # for the BlockMapper training data path.
                                    # ``materialize_block_array`` calls
                                    # ``self_contain_slice`` per slice, which
                                    # re-encodes via ``run_passes`` so each
                                    # block is decodable standalone -- palette
                                    # indices are slice-local and any in-slice
                                    # PLAY_INSTRUMENT_OP / GATE_REPLAY_OP
                                    # references resolve within the block.
                                    # Default on; pass ``--no-write-blocks`` to
                                    # skip and fall back to the sliding-window
                                    # SeqMapper training path.
                                    if getattr(self.args, "write_blocks", True):
                                        try:
                                            block_parser = RegLogParser(self.args)
                                            blocks_arr = materialize_block_array(
                                                self.tokenizer,
                                                raw_df,
                                                self.args.seq_len,
                                                block_parser,
                                                self.reg_widths,
                                            )
                                            blocks_path = dump_file.replace(
                                                DUMP_SUFFIX, f".{i}.blocks.npy"
                                            )
                                            np.save(blocks_path, blocks_arr)
                                        except Exception as e:
                                            self.logger.info(
                                                "block materialisation failed for "
                                                "%s rotation %u: %s",
                                                dump_file,
                                                i,
                                                e,
                                            )
                            output_dumps.add(dump_file)
                            irq = df["irq"].iloc[0]
                            yield dump_file, i, df, seq, irq
                        pbar.n = len(output_dumps)
                        pbar.refresh()
                        if len(output_dumps) == max_files:
                            break
                    futures = new_futures
            executor.shutdown(wait=True, cancel_futures=True)

    def make_tokens(self, reglogs):
        df_files = []
        for df_file, _i, df, _seq, _irq in self.load_dfs(
            reglogs=reglogs, max_perm=self.args.max_perm
        ):
            self.tokenizer.accumulate_tokens(df, df_file)
            try:
                if df_files[-1] == df_file:
                    continue
            except IndexError:
                pass
            df_files.append(df_file)
        tokens = self.tokenizer.make_tokens()
        self.tokenizer.tokens = tokens
        assert self.tokenizer.tokens[tokens["val"].isna()].empty, tokens[
            tokens["val"].isna()
        ]
        return df_files

    def preload(self, tokens=None, tkmodel=None):
        if tokens is not None:
            self.tokenizer.load(tkmodel, tokens)
            return
        self.logger.info("preload making tokens")
        df_files = self.make_tokens(self.args.reglogs)
        if self.args.token_csv:
            self.logger.info("writing tokens to %s", self.args.token_csv)
            self.tokenizer.tokens.to_csv(self.args.token_csv, index=False)
        dataset_csv = self.args.dataset_csv
        df_map_csv = self.args.df_map_csv

        if not self.args.tkvocab and not dataset_csv:
            if df_map_csv:
                df_map = pd.DataFrame(df_files, columns=["dump_file"])
                df_map.to_csv(df_map_csv, index=False)
            return

        def worker():
            dataset_csv = self.args.dataset_csv

            def worker_gen():
                for i, (df_file, _i, df, _seq, _irq) in enumerate(
                    self.load_dfs(
                        dump_files=df_files,
                        max_perm=self.args.max_perm,
                        encode=False,
                    )
                ):
                    yield df_file, df, i

            if dataset_csv:
                self.logger.info("writing dataset to %s", dataset_csv)
                with zstd.open(dataset_csv, "w") as f:
                    for df_file, df, i in worker_gen():
                        df["i"] = int(i)
                        df.to_csv(f, index=False, header=(i == 0))
                        yield df_file, df, i
            else:
                for df_file, df, i in worker_gen():
                    yield df_file, df, i

            if df_map_csv:
                self.logger.info("writing dataset map to %s", df_map_csv)
                df_map = pd.DataFrame(df_files, columns=["dump_file"])
                df_map.to_csv(df_map_csv, index=False)

        if self.args.tkvocab:
            self.tokenizer.train_tokenizer(worker())
        else:
            for _df in worker():
                continue

    def load(self):
        assert self.tokenizer.tokens is not None
        dump_files = None
        reglogs = None
        if self.args.reglog:
            self.logger.info(f"loading data from {self.args.reglog}")
            reglogs = self.args.reglog
        elif os.path.exists(self.args.df_map_csv):
            df_map_df = pd.read_csv(self.args.df_map_csv)
            dump_files = df_map_df["dump_file"].drop_duplicates().tolist()
            self.logger.info(
                f"loading data from {self.args.df_map_csv} - {len(dump_files)} files"
            )
        elif self.args.reglogs:
            self.logger.info(f"loading data from {self.args.reglogs}")
            reglogs = self.args.reglogs
        self.n_vocab = len(self.tokenizer.tokens["n"])
        if self.args.tkvocab:
            self.n_vocab = self.args.tkvocab
        self.n_words = 0
        n_seq = 0
        n_words = 0
        reg_max = {}
        for df_file, i, df, seq, irq in self.load_dfs(
            reglogs=reglogs,
            dump_files=dump_files,
            max_perm=self.args.max_perm,
            encode=True,
        ):
            seq_meta = SeqMeta(irq=irq, df_file=df_file, i=i)
            self.seq_mapper.add(seq, seq_meta)
            reg_max = self.tokenizer.get_reg_max(df, reg_max)
            self.n_words += len(seq)
            n_words += len(df)
            n_seq += 1
            # If load_dfs wrote a .blocks.npy alongside the .npy, register
            # it with block_mapper. Missing files are non-fatal -- the
            # block path is opt-in and falls back to seq_mapper.
            blocks_path = df_file.replace(DUMP_SUFFIX, f".{i}.blocks.npy")
            if os.path.exists(blocks_path):
                try:
                    self.block_mapper.add(blocks_path, seq_meta)
                except Exception as e:
                    self.logger.info(
                        "block_mapper add failed for %s: %s", blocks_path, e
                    )
        self.seq_mapper.finalize()
        self.block_mapper.finalize()
        self.reg_widths = self.tokenizer.get_reg_width_from_max(reg_max)
        n_frac = 0
        if n_words:
            n_frac = round(self.n_words / n_words, 2)
        self.logger.info(
            f"n_vocab: {self.n_vocab}, n_words {n_words}, n_encoded_words {self.n_words} ({n_frac}), reg widths {sorted(self.reg_widths.items())}, {n_seq} sequences"
        )

    def __len__(self):
        # Prefer block_mapper when populated -- each block is a single
        # self-contained training sample. Falls back to the flat seq
        # sliding-window when blocks haven't been written (e.g., loading
        # an older dataset that pre-dates the block path).
        if len(self.block_mapper) > 0:
            return len(self.block_mapper)
        return len(self.seq_mapper)

    def __getitem__(self, index):
        if len(self.block_mapper) > 0:
            return self.block_mapper[index]
        return self.seq_mapper[index]

    def getseq(self, i):
        seq, seq_meta = self.seq_mapper.getseq(i)
        return torch.from_numpy(np.copy(seq)), seq_meta


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
        # sampler = LowMemoryRandomSampler(
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


def get_loader(args, dataset, seq_mapper=False):
    dataset.load()
    if seq_mapper:
        return _get_loader(args, copy.deepcopy(dataset.seq_mapper))
    return get_loader(args, dataset)
