import concurrent.futures
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
from preframr.seq_mapper import BlockMapper, SeqMeta
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


def iter_voiced_blocks(
    raw_df, seq_len, parser, reg_widths, frames_per_block=None, stride=None
):
    """Yield each self-contained block as a post-voice-reg row df --
    the same form ``materialize_block_array`` would later turn into
    int-encoded rows. Used by ``RegDataset.make_tokens`` to build the
    token alphabet from blocks (so the alphabet covers exactly the
    (op, reg, subreg, val) tuples training will see), and by
    ``materialize_block_array`` to write the .blocks.npy training data.

    The two callers MUST consume the same block stream so the alphabet
    matches the encoded blocks; routing both through this generator
    eliminates the alphabet/training-data mismatch that previously
    caused ``merge_token_df`` to fail on macro-op rows whose val
    payload (back-ref distance, palette slot, etc.) is block-local.
    """
    if frames_per_block is None:
        frames_per_block = max(1, seq_len // 2)
    abs_df, _ = parser._remove_voice_reg(raw_df.copy(), reg_widths)
    for block_df in iter_self_contained_row_blocks(
        abs_df, frames_per_block, args=parser.args, stride=stride
    ):
        if block_df.empty:
            continue
        # Re-add voice rotation so the block matches the LM's input
        # form. _add_voice_reg can fail on degenerate blocks (e.g.
        # entirely empty after self_contain materialisation); skip
        # those rather than abort the whole song.
        try:
            voiced = parser._add_voice_reg(block_df.copy(), zero_voice_reg=True)
        except Exception:
            continue
        yield voiced


def materialize_block_array(
    tokenizer,
    raw_df,
    seq_len,
    parser,
    reg_widths,
    frames_per_block=None,
    stride=None,
):
    """Materialise the encoded ``raw_df`` (post-voice-reg, post-LoopPass)
    into a fixed-size 2D array of self-contained blocks.

    Block storage size is ``seq_len + 1`` so ``BlockMapper`` can split each
    row into ``input = block[:-1]`` and shifted target ``target = block[1:]``.
    Longer blocks are truncated; shorter ones pad with zero (which encodes
    the unk/pad slot in the tokenizer's vocabulary).

    Walks blocks via ``iter_voiced_blocks`` (shared with
    ``make_tokens``); each block goes through ``merge_token_df`` and
    is then int-encoded. A merge failure is fatal -- it indicates the
    alphabet didn't cover this block's tokens, which means the training
    data is silently corrupted. Earlier behaviour swallowed the
    exception; we now let it propagate so the bug surfaces at parse
    time instead of mid-training.
    """
    block_size = seq_len + 1
    blocks = []
    for voiced in iter_voiced_blocks(
        raw_df,
        seq_len,
        parser,
        reg_widths,
        frames_per_block=frames_per_block,
        stride=stride,
    ):
        merged = tokenizer.merge_token_df(tokenizer.tokens, voiced.copy())
        if merged is None or "n" not in merged.columns:
            raise RuntimeError(
                "merge_token_df returned no 'n' column; alphabet does "
                "not cover block tokens"
            )
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
    block_parser = RegLogParser(args, logger)
    stride = getattr(args, "block_stride", None)
    seq_len = args.seq_len
    # Materialise blocks here in the parser's worker process so the
    # expensive per-block ``run_passes`` cost runs in parallel across
    # songs rather than serially in the parent's make_tokens loop.
    # Empty reg_widths is fine -- ``_remove_voice_reg`` only consults it
    # when explicitly populated (post-load() reg_max scan), and the
    # block materialisation that uses it doesn't need per-reg widths
    # for the rotation removal step.
    out = []
    for df in reg_log_parser.parse(
        dump_file, max_perm=max_perm, require_pq=args.require_pq
    ):
        blocks = list(iter_voiced_blocks(df, seq_len, block_parser, {}, stride=stride))
        out.append((df, blocks))
    return dump_file, out


def get_prompt(args, dataset, logger):
    seq, seq_meta = dataset.getseq(args.start_seq, block_j=args.start_block)
    # ``seq`` is one self-contained block of length ``seq_len + 1``.
    # Prompt = first ``prompt_seq_len`` tokens; predict = next
    # ``max_seq_len - prompt_seq_len`` tokens; compare against the
    # block's ground truth. No mid-block offset: each block is the
    # natural training-aligned unit.
    start = 0
    logger.info(
        "starting at seq %u block %u (%s), %u / %u, irq %u",
        args.start_seq,
        args.start_block,
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
    slice_hi = slice_lo + int(is_marker.iloc[start : start + prompt_seq_len].sum())
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
        # Self-contained block storage; populated during load() from
        # per-rotation ``.blocks.npy`` files written at parse time by
        # ``make_tokens``. ``__getitem__`` reads from block_mapper so
        # each batch sample is a self-contained block, and ``getseq``
        # serves single blocks for inference. ``val_block_mapper``
        # carries blocks from ``--eval-reglogs`` (held-out songs);
        # the trainer's ``validation_step`` reads from it.
        self.block_mapper = BlockMapper(args.seq_len)
        self.val_block_mapper = BlockMapper(args.seq_len)
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
                        dump_file, dfs_with_blocks = future.result()
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
                        for i, (df, blocks) in enumerate(dfs_with_blocks):
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
                                    min_seq = getattr(self.args, "min_song_tokens", 256)
                                    if len(seq) < min_seq:
                                        self.logger.info(
                                            "rejecting sequence from %s too short %u (< %u)",
                                            dump_file,
                                            len(seq),
                                            min_seq,
                                        )
                                        break
                                    # ``.blocks.npy`` files are written
                                    # by ``RegDataset.make_tokens`` from
                                    # the cached materialised blocks, so
                                    # ``load_dfs`` doesn't need to
                                    # repeat the per-block run_passes
                                    # cost. ``load()`` discovers the
                                    # written files later.
                            output_dumps.add(dump_file)
                            irq = df["irq"].iloc[0]
                            yield dump_file, i, df, seq, irq, blocks
                        pbar.n = len(output_dumps)
                        pbar.refresh()
                        if len(output_dumps) == max_files:
                            break
                    futures = new_futures
            executor.shutdown(wait=True, cancel_futures=True)

    def make_tokens(self, reglogs, eval_reglogs=""):
        """Parse each song, materialise its self-contained blocks, build
        the token alphabet from those blocks, and write the encoded
        blocks to ``.blocks.npy`` per (dump_file, rotation).

        ``eval_reglogs`` (optional) is a separate glob whose songs feed
        the alphabet AND have their blocks written, but are tagged as
        ``"val"`` so ``load()`` registers them with ``val_block_mapper``
        rather than the training mapper. Pass empty for no held-out
        set (memorise-back regime).

        One pass through ``load_dfs`` per set: each parsed rotation has
        its block stream cached in memory, then after the alphabet is
        finalised the cached blocks are encoded and saved. This avoids
        the previous design which materialised blocks twice -- once for
        alphabet building, once again for training-data writing.

        Memory: roughly ``num_songs x num_rotations x num_blocks_per_song
        x bytes_per_block``. For an integration test (4 songs, 1
        rotation, ~120 blocks of 512 rows) that's ~10 MB. For a
        full-corpus run (8000 songs, 3 rotations) the cache should fit
        in tens of GB; if that's a problem in production, the next step
        is per-song streaming with a disk-backed cache, but that's not
        needed for the corpora we're using today.
        """
        train_files = []
        val_files = []
        # (df_file, rotation_i, kind) -> list[voiced block df]
        cached_blocks = {}

        def walk(reglogs_glob, kind, files_out):
            for df_file, i, df, _seq, _irq, blocks in self.load_dfs(
                reglogs=reglogs_glob, max_perm=self.args.max_perm
            ):
                # Blocks were materialised in parallel inside
                # parser_worker (so the per-block run_passes cost is
                # amortised across the parser pool, not paid serially
                # here). Accumulate alphabet from BOTH the full-song
                # df and its blocks: the full-song df carries DELAY /
                # FRAME / VOICE marker tokens that
                # expand_to_literal_form destructures into FRAME_REGs
                # only; blocks contribute the macro tokens
                # (BACK_REF / PATTERN_REPLAY / GATE_REPLAY / ...) whose
                # val/subreg is block-local. Union covers both shapes,
                # and including eval here means tokens that only
                # appear on held-out songs are still in the vocab.
                self.tokenizer.accumulate_tokens(df, df_file)
                cached_blocks[(df_file, i, kind)] = blocks
                for voiced in blocks:
                    self.tokenizer.accumulate_tokens(voiced, df_file)
                try:
                    if files_out[-1] == df_file:
                        continue
                except IndexError:
                    pass
                files_out.append(df_file)

        walk(reglogs, "train", train_files)
        if eval_reglogs:
            walk(eval_reglogs, "val", val_files)

        tokens = self.tokenizer.make_tokens()
        self.tokenizer.tokens = tokens
        assert self.tokenizer.tokens[tokens["val"].isna()].empty, tokens[
            tokens["val"].isna()
        ]
        # Encode the cached blocks now that the alphabet is finalised
        # and write per-(dump_file, rotation) .blocks.npy files. This
        # is the single source of truth for BlockMapper training data;
        # ``load_dfs`` no longer re-materialises blocks downstream.
        if getattr(self.args, "write_blocks", True):
            self._encode_and_save_cached_blocks(cached_blocks)
        return train_files, val_files

    def _encode_and_save_cached_blocks(self, cached_blocks):
        """Encode each cached voiced-block df via the now-finalised
        tokenizer and write ``.blocks.npy``. Failures (alphabet doesn't
        cover a row's (op, reg, subreg, val)) propagate; this is the
        catch point for any bug in the alphabet-building pipeline.

        ``cached_blocks`` keys are ``(df_file, rotation_i, kind)``.
        ``kind`` only affects which mapper later registers the file --
        the on-disk path is the same for train and val.
        """
        block_size = self.args.seq_len + 1
        for (df_file, i, _kind), blocks in cached_blocks.items():
            block_arrs = []
            for voiced in blocks:
                merged = self.tokenizer.merge_token_df(
                    self.tokenizer.tokens, voiced.copy()
                )
                if merged is None or "n" not in merged.columns:
                    raise RuntimeError(
                        f"merge_token_df returned no 'n' column for "
                        f"{df_file} rotation {i}"
                    )
                n = merged["n"].astype(np.int16).to_numpy()
                seq = self.tokenizer.encode(n).astype(np.int16)
                if len(seq) >= block_size:
                    block_arrs.append(seq[:block_size])
                else:
                    padded = np.zeros(block_size, dtype=np.int16)
                    padded[: len(seq)] = seq
                    block_arrs.append(padded)
            if not block_arrs:
                continue
            blocks_path = df_file.replace(DUMP_SUFFIX, f".{i}.blocks.npy")
            np.save(blocks_path, np.stack(block_arrs))

    def preload(self, tokens=None, tkmodel=None):
        if tokens is not None:
            self.tokenizer.load(tkmodel, tokens)
            return
        self.logger.info("preload making tokens")
        eval_reglogs = getattr(self.args, "eval_reglogs", "") or ""
        train_files, val_files = self.make_tokens(
            self.args.reglogs, eval_reglogs=eval_reglogs
        )
        df_files = train_files + val_files
        if self.args.token_csv:
            self.logger.info("writing tokens to %s", self.args.token_csv)
            self.tokenizer.tokens.to_csv(self.args.token_csv, index=False)
        dataset_csv = self.args.dataset_csv
        df_map_csv = self.args.df_map_csv

        # df_map.csv carries a ``kind`` column so ``load()`` can route
        # each block file to the right mapper without re-globbing.
        def _df_map_frame():
            return pd.DataFrame(
                [(p, "train") for p in train_files] + [(p, "val") for p in val_files],
                columns=["dump_file", "kind"],
            )

        if not self.args.tkvocab and not dataset_csv:
            if df_map_csv:
                _df_map_frame().to_csv(df_map_csv, index=False)
            return

        def worker():
            dataset_csv = self.args.dataset_csv

            def worker_gen():
                for i, (df_file, _i, df, _seq, _irq, _blocks) in enumerate(
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
                _df_map_frame().to_csv(df_map_csv, index=False)

        if self.args.tkvocab:
            self.tokenizer.train_tokenizer(worker())
        else:
            for _df in worker():
                continue

    def load(self):
        assert self.tokenizer.tokens is not None
        dump_files = None
        reglogs = None
        kind_by_dump = {}
        if self.args.reglog:
            self.logger.info(f"loading data from {self.args.reglog}")
            reglogs = self.args.reglog
        elif os.path.exists(self.args.df_map_csv):
            df_map_df = pd.read_csv(self.args.df_map_csv)
            # Older df_map.csv files (pre-eval-reglogs) lack a ``kind``
            # column; default everything to "train" then.
            if "kind" not in df_map_df.columns:
                df_map_df = df_map_df.assign(kind="train")
            df_map_df = df_map_df.drop_duplicates("dump_file")
            dump_files = df_map_df["dump_file"].tolist()
            kind_by_dump = dict(zip(df_map_df["dump_file"], df_map_df["kind"]))
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
        for df_file, i, df, seq, irq, _blocks in self.load_dfs(
            reglogs=reglogs,
            dump_files=dump_files,
            max_perm=self.args.max_perm,
            encode=True,
        ):
            seq_meta = SeqMeta(irq=irq, df_file=df_file, i=i)
            reg_max = self.tokenizer.get_reg_max(df, reg_max)
            self.n_words += len(seq) if seq is not None else 0
            n_words += len(df)
            n_seq += 1
            blocks_path = df_file.replace(DUMP_SUFFIX, f".{i}.blocks.npy")
            if os.path.exists(blocks_path):
                target = (
                    self.val_block_mapper
                    if kind_by_dump.get(df_file) == "val"
                    else self.block_mapper
                )
                target.add(blocks_path, seq_meta)
        self.block_mapper.finalize()
        self.val_block_mapper.finalize()
        self.reg_widths = self.tokenizer.get_reg_width_from_max(reg_max)
        n_frac = 0
        if n_words:
            n_frac = round(self.n_words / n_words, 2)
        self.logger.info(
            f"n_vocab: {self.n_vocab}, n_words {n_words}, n_encoded_words {self.n_words} ({n_frac}), reg widths {sorted(self.reg_widths.items())}, {n_seq} sequences"
        )

    def __len__(self):
        # Training and inference both read from BlockMapper -- each
        # block is self-contained (palettes, back-refs, loop bodies
        # all resolve within the block by construction).
        return len(self.block_mapper)

    def __getitem__(self, index):
        return self.block_mapper[index]

    def getseq(self, rotation_i, block_j=0):
        """Return ``(block, seq_meta)`` for one block of one rotation.

        Block 0 of rotation 0 is the natural prompt for memorise-back
        tests; pass ``block_j > 0`` to start from a later block.
        ``args.predict_set`` selects between train and val mappers so
        generalisation runs can prompt from held-out songs.
        """
        mapper = (
            self.val_block_mapper
            if getattr(self.args, "predict_set", "train") == "val"
            else self.block_mapper
        )
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


def get_loader(args, dataset):
    dataset.load()
    return _get_loader(args, dataset)


def get_val_loader(args, dataset):
    """Return a validation DataLoader, or ``None`` if no eval data.

    The val mapper is populated by ``RegDataset.load`` (called from
    ``get_loader``); this helper just wraps it in a sequential
    DataLoader. Returns ``None`` when ``--eval-reglogs`` was empty so
    callers can ``trainer.fit(..., val_dataloaders=None)`` without
    branching themselves.
    """
    if not getattr(args, "eval_reglogs", "") or len(dataset.val_block_mapper) == 0:
        return None
    return torch.utils.data.DataLoader(
        dataset.val_block_mapper,
        sampler=torch.utils.data.SequentialSampler(dataset.val_block_mapper),
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=2,
    )
