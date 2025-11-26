import concurrent.futures
import difflib
import logging
import glob
import os
import random
import shutil
import tempfile
import time
from tqdm import tqdm
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import numpy as np
import pandas as pd
import zstandard as zstd
from preframr.reglogparser import RegLogParser
from preframr.seq_mapper import SeqMapper, SeqMeta
from preframr.stfconstants import (
    DELAY_REG,
    FILTER_REG,
    FRAME_REG,
    UNICODE_BASE,
    VOICES,
    VOICE_REG,
    VOICE_REG_SIZE,
)

TOKEN_KEYS = ["reg", "val", "diff"]
MODEL_PDTYPE = pd.Int32Dtype()
REG_PDTYPE = pd.Int8Dtype()
VAL_PDTYPE = pd.UInt32Dtype()
TOKEN_PDTYPE = pd.Int64Dtype()  # Same as torch
DIFF_PDTYPE = pd.UInt16Dtype()
IRQ_PDTYPE = pd.UInt16Dtype()
MIN_DIFF = 32
FRAME_DTYPES = {
    "reg": REG_PDTYPE,
    "val": VAL_PDTYPE,
    "diff": DIFF_PDTYPE,
    "irq": IRQ_PDTYPE,
}
UNK_TOKEN = "<unk>"
END_OF_WORD_SUFFIX = "</w>"


def wrapbits(x, reglen):
    base = (x << 1) & (2**reglen - 1)
    lsb = (x >> (reglen - 1)) & 1
    return base ^ lsb


FILTER_SHIFT_DF = pd.DataFrame(
    [{"reg": FILTER_REG, "val": i, "y": wrapbits(i, 3)} for i in range(2**3)],
    dtype=MODEL_PDTYPE,
)


def remove_voice_reg(orig_df, reg_widths):
    voice_regs = len(orig_df[orig_df["reg"] == VOICE_REG])
    if voice_regs:
        df = orig_df.copy()
        df["vr"] = pd.NA
        df.loc[df["reg"] == VOICE_REG, "vr"] = df["val"]
        df.loc[df["reg"].isin({FRAME_REG, DELAY_REG}), "vr"] = 0
        df["vr"] = df["vr"].astype(pd.UInt8Dtype()).ffill().fillna(0)
        df = df[df["reg"] != VOICE_REG]
        df["vr"] = df["vr"].astype(pd.Int64Dtype()) * VOICE_REG_SIZE
        df.loc[df["reg"] >= VOICE_REG_SIZE, "vr"] = 0
        df["reg"] += df["vr"]
        df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for i in range(VOICE_REG_SIZE):
                if i in reg_widths:
                    reg_widths[v_offset + i] = reg_widths[i]
        return df, reg_widths
    return orig_df, reg_widths


def state_df(states, dataset, irq):
    tokens = dataset.tokens.copy()
    tokens.loc[tokens["reg"] == FRAME_REG, "diff"] = irq
    df = pd.DataFrame(states, columns=["n"]).merge(tokens, on="n", how="left")
    return df


def get_prompt(args, dataset, logger):
    seq, seq_meta = dataset.getseq(args.start_seq)
    if args.start_n is None:
        start = random.randint(0, len(seq))
    else:
        start = args.start_n
    logger.info("starting at %u / %u, irq %u", start, len(seq), seq_meta.irq)
    n = args.max_seq_len - args.prompt_seq_len
    if n <= 0:
        raise ValueError("max seq length too short")
    prompt = seq[start:][: args.prompt_seq_len].unsqueeze(0)
    prompt_compare = seq[start:][: args.max_seq_len]
    preamble_df, _reg_widths = remove_voice_reg(
        state_df(dataset.decode(seq[:start].numpy()), dataset, seq_meta.irq),
        dataset.reg_widths,
    )
    reg_start = {
        r: preamble_df[preamble_df["reg"] == r]["val"].iat[-1]
        for r in preamble_df["reg"].unique()
        if r >= 0
    }
    return seq_meta.irq, n, prompt, prompt_compare, reg_start


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, args, logger=logging):
        self.args = args
        self.logger = logger
        self.n_vocab = 0
        self.n_words = 0
        self.reg_widths = {}
        self.tokens = None
        self.tkmodel = None
        self.reg_log_parser = RegLogParser(args, logger)
        self.seq_mapper = SeqMapper(args.seq_len)

    def encode_unicode(self, tokens, dtype=np.uint16):
        t = np.array(tokens, dtype=dtype)
        t = np.where(t == 0, np.nan, t)
        t += UNICODE_BASE
        t = np.nan_to_num(t).astype(dtype)
        t = np.where(t == 0, 32, t)
        return "".join([chr(i) for i in t])

    def decode_unicode(self, encoded_tokens, dtype=np.uint16):
        t = np.array([ord(i) for i in encoded_tokens])
        t = np.where(t == 32, np.nan, t)
        t -= UNICODE_BASE
        t = np.nan_to_num(t).astype(dtype)
        return t

    def encode(self, tokens, dtype=np.uint16):
        if self.tkmodel:
            encoded = self.tkmodel.encode(self.encode_unicode(tokens, dtype=dtype))
            return np.array(encoded.ids, dtype=dtype)
        return tokens

    def decode(self, encoded_tokens, dtype=np.uint16):
        if self.tkmodel:
            return self.decode_unicode(self.tkmodel.decode(encoded_tokens), dtype=dtype)
        return encoded_tokens

    def train_tokenizer(self, dfs, tokenizer="unigram"):
        encoded_dfs = []
        for df in dfs:
            orig_seq = df["n"].to_numpy()
            encoded = self.encode_unicode(orig_seq)
            encoded_dfs.append(encoded)
        self.tkmodel, trainer = self.get_tk(tokenizer=tokenizer)
        self.tkmodel.train_from_iterator(encoded_dfs, trainer=trainer)
        assert self.tkmodel.get_vocab_size() == self.args.tkvocab, (
            self.tkmodel.get_vocab_size(),
            self.args.tkvocab,
        )

    def load_df(self, name, df_dir, max_perm=99, min_space=0):
        dfs = []
        try:
            for i, df in enumerate(
                self.reg_log_parser._downsample_df(
                    self.reg_log_parser._read_df(name), max_perm=max_perm
                )
            ):
                try:
                    irq = df["irq"].iloc[0]
                except KeyError:
                    self.logger.info("skipped %s, no irq", name)
                    break
                if irq < self.args.min_irq or irq > self.args.max_irq:
                    self.logger.info(
                        "skipped %s, irq %u (outside IRQ range)", name, irq
                    )
                    break
                if len(df) < self.args.seq_len:
                    self.logger.info("skipped %s, length %u (too short)", name, len(df))
                    break
                vol = sorted(
                    np.bitwise_and(df[df["reg"] == 24]["val"], 15).unique().tolist()
                )
                if len(vol) >= 8:
                    self.logger.info(
                        "skipped %s, too many (%u) vol changes %s", name, len(vol), vol
                    )
                    break
                if min_space:
                    while True:
                        usage = shutil.disk_usage(df_dir)
                        prop = usage.free / usage.total
                        if prop > min_space:
                            break
                        time.sleep(1)
                df_base = os.path.splitext(os.path.basename(name))[0]
                df_name = os.path.join(df_dir, f"{hash(name)}-{df_base}.{i}.zst")
                df.to_parquet(df_name, engine="pyarrow", compression="zstd")
                dfs.append(df_name)
        except Exception as e:
            raise ValueError(f"cannot read {name}: {e}")
        return name, dfs

    def load_dfs(
        self, dump_files, max_perm=99, max_workers=16, min_space=0.2, shuffle=0
    ):
        results = []
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
            unsorted_dump_files = dump_files
            random.shuffle(unsorted_dump_files)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as preload_executor:
                preload_futures = [
                    preload_executor.submit(
                        self.load_df,
                        dump_file,
                        tmpdir,
                        max_perm=max_perm,
                        min_space=min_space,
                    )
                    for dump_file in unsorted_dump_files
                ]

                def load_df(name, file_dfs):
                    if not file_dfs:
                        return []
                    dfs = []
                    for file_df in file_dfs:
                        try:
                            dfs.append(pd.read_parquet(file_df))
                        except Exception as e:
                            raise ValueError(f"cannot read {file_df}: {e}")
                    irq = dfs[0]["irq"].iloc[0]
                    for i, file_df in enumerate(file_dfs):
                        os.unlink(file_df)
                        self.logger.info("loaded %s, irq %u, augment %u", name, irq, i)
                    return dfs

                for preload_future in tqdm(
                    concurrent.futures.as_completed(preload_futures),
                    total=len(preload_futures),
                    ascii=True,
                ):
                    assert not preload_future.exception(), preload_future.exception()
                    name, file_dfs = preload_future.result()
                    dfs = load_df(name, file_dfs)
                    for df in dfs:
                        results.append((name, df))
        if shuffle is not None:
            random.seed(shuffle)
            random.shuffle(results)
            random.seed()
        else:
            results = sorted(results, key=lambda x: x[0])
        df_files = [result[0] for result in results]
        dfs = [result[1] for result in results]
        return df_files, dfs

    def _merged_and_missing(self, tokens, df):
        m = df["reg"] == FRAME_REG
        irq = df[m]["diff"].iloc[0]
        df.loc[m, "diff"] = 0
        df = df.merge(tokens, on=TOKEN_KEYS, how="left")
        df.loc[m, "diff"] = irq
        missing_tokens = (
            df[df["n"].isna()].drop_duplicates().sort_values(["reg", "val"])
        )
        return df, missing_tokens

    def merge_tokens(self, tokens, dfs):
        self.logger.info("merging tokens")
        merged_dfs = []
        for df in tqdm(dfs, ascii=True):
            orig_cols, orig_dtypes = df.columns, df.dtypes
            df, missing_tokens = self._merged_and_missing(tokens, df)
            if not missing_tokens.empty:
                for missing_token in missing_tokens.itertuples():
                    reg = missing_token.reg
                    val = missing_token.val
                    reg_tokens = tokens[tokens["reg"] == reg]
                    if reg_tokens.empty:
                        self.logger.error(
                            "no possible token for reg %u val %u", reg, val
                        )
                        assert False
                    compare_tokens = reg_tokens.copy()
                    compare_tokens["diff_val"] = (compare_tokens["val"] - val).abs()
                    best_token = compare_tokens[
                        compare_tokens["diff_val"] == compare_tokens["diff_val"].min()
                    ].iloc[0]
                    best_val = best_token.val
                    self.logger.info(
                        "substitute reg %u val %u with val %u", reg, val, best_val
                    )
                    df.loc[((df["reg"] == reg) & (df["val"] == val)), "val"] = best_val
                df = df[orig_cols].astype(orig_dtypes)
                df, missing_tokens = self._merged_and_missing(tokens, df)
                assert missing_tokens.empty
            merged_dfs.append(df)
        return merged_dfs

    def glob_dumps(self, reglogs, max_files, min_dump_size):
        random.seed(0)
        dump_files = []
        for r in reglogs.split(","):
            max_globbed = max_files - len(dump_files)
            if max_globbed <= 0:
                break
            globbed = [
                f
                for f in glob.glob(r, recursive=True)
                if os.path.getsize(f) >= min_dump_size
            ]
            random.shuffle(globbed)
            dump_files.extend(globbed[:max_globbed])
        random.seed()
        return dump_files

    def validate_encoding(self, df_file, seq):
        if not self.args.tkvocab:
            return seq
        orig_seq = seq.copy()
        seq = self.encode(orig_seq, dtype=np.int64)
        decoded_seq = self.decode(seq, dtype=np.int64)
        if not np.array_equal(orig_seq, decoded_seq):
            for i, (orig, decoded) in enumerate(zip(orig_seq, decoded_seq)):
                if orig == decoded:
                    continue
                a = [str(i) for i in orig_seq]
                b = [str(i) for i in decoded_seq]
                d = "\n".join(difflib.context_diff(a, b))
                print(d)
                assert False, (
                    df_file,
                    i,
                    orig,
                    decoded,
                    self.tokens.iloc[int(orig)],
                )
        return seq

    def load(self, tokens=None, tkmodel=None):
        self.tkmodel = tkmodel
        if tkmodel:
            self.tkmodel = Tokenizer.from_str(tkmodel)

        if self.args.reglog:
            df_files, dfs = self.load_dfs(
                [self.args.reglog],
                max_perm=self.args.max_perm,
            )
            if tokens is None:
                tokens = self.reg_log_parser._make_tokens(dfs)
            self.tokens = tokens
            dfs = self.merge_tokens(self.tokens, dfs)
        else:
            dump_files = self.glob_dumps(
                self.args.reglogs, self.args.max_files, self.args.min_dump_size
            )
            df_files, dfs = self.load_dfs(dump_files, max_perm=self.args.max_perm)
            _token_df_files, token_dfs = self.load_dfs(
                self.glob_dumps(
                    self.args.token_reglogs,
                    self.args.max_files,
                    self.args.min_dump_size,
                ),
                max_perm=self.args.max_perm,
            )
            self.tokens = self.reg_log_parser._make_tokens(dfs + token_dfs)
            dfs = self.merge_tokens(self.tokens, dfs)
            token_dfs = self.merge_tokens(self.tokens, token_dfs)
            if self.args.token_csv:
                self.logger.info("writing %s", self.args.token_csv)
                self.tokens.to_csv(self.args.token_csv)
            if self.args.tkvocab:
                self.train_tokenizer(dfs + token_dfs)
        self.logger.info("getting reg widths")
        self.reg_widths = self.reg_log_parser.get_reg_widths(dfs)
        self.n_vocab = len(self.tokens["n"])
        self.n_words = sum((len(df) for df in dfs))
        assert self.tokens[self.tokens["val"].isna()].empty
        assert self.tokens[self.tokens["val"] < 0].empty
        self.logger.info(
            f"n_vocab: {self.n_vocab}, n_words {self.n_words}, reg widths {sorted(self.reg_widths.items())}"
        )
        if self.args.tkvocab:
            self.n_vocab = self.args.tkvocab
        self.n_words = 0
        self.logger.info("mapping sequences")
        final_dfs = []
        final_df_files = []
        for df_file, df in tqdm(zip(df_files, dfs), ascii=True):
            seq = self.validate_encoding(df_file, df["n"].to_numpy())
            seq_meta = SeqMeta(irq=int(df["irq"].iat[0]))
            try:
                self.seq_mapper.add(seq, seq_meta)
                self.n_words += len(seq)
                final_df_files.append(df_file)
                final_dfs.append(df)
            except ValueError:
                self.logger.info(
                    "rejecting sequence from %s too short %u", df_file, len(seq)
                )
        self.logger.info(
            f"n_encoded_words {self.n_words}, {len(dfs)} sequences",
        )
        if not self.args.reglog:
            if self.args.df_map_csv:
                self.logger.info(f"writing {self.args.df_map_csv}")
                pd.DataFrame(final_df_files, columns=["dump_file"]).to_csv(
                    self.args.df_map_csv
                )
            if self.args.dataset_csv:
                self.logger.info(f"writing {self.args.dataset_csv}")
                with zstd.open(self.args.dataset_csv, "w") as f:
                    for i in tqdm(range(len(final_dfs)), ascii=True):
                        final_dfs[i]["i"] = int(i)
                        final_dfs[i].to_csv(f, index=False, header=(i == 0))

    def get_tk(self, tokenizer="unigram"):
        if tokenizer == "unigram":
            tk = Tokenizer(models.Unigram())
            tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement=" ")
            tk.decoder = decoders.Metaspace(replacement=" ")
            tk.normalizer = None
            trainer = trainers.UnigramTrainer(
                vocab_size=self.args.tkvocab,
                show_progress=True,
                special_tokens=[UNK_TOKEN],
                initial_alphabet=[],
                unk_token=UNK_TOKEN,
            )
            return tk, trainer
        if tokenizer == "bpe":
            tk = Tokenizer(
                models.BPE(
                    dropout=None,
                    unk_token=UNK_TOKEN,
                    end_of_word_suffix=END_OF_WORD_SUFFIX,
                    fuse_unk=False,
                    byte_fallback=False,
                    ignore_merges=False,
                    vocab={},
                    merges=[],
                )
            )
            tk.normalizer = None
            tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            tk.decoder = decoders.BPEDecoder(suffix=END_OF_WORD_SUFFIX)
            trainer = trainers.BpeTrainer(
                vocab_size=self.args.tkvocab,
                min_frequency=2,
                special_tokens=[UNK_TOKEN],
                limit_alphabet=self.args.tkvocab,
                initial_alphabet=[],
                end_of_word_suffix=END_OF_WORD_SUFFIX,
                show_progress=True,
            )
            return tk, trainer
        raise ValueError

    def __len__(self):
        return len(self.seq_mapper)

    def __getitem__(self, index):
        return self.seq_mapper[index]

    def getseq(self, i):
        return torch.from_numpy(self.seq_mapper.seqs[i]), self.seq_mapper.seq_metas[i]


def get_loader(args, dataset):
    dataset.load()
    if args.shuffle:
        length = args.shuffle / 1.0
        sampler = torch.utils.data.RandomSampler(
            dataset, num_samples=int(length * len(dataset))
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        pin_memory=True,
        batch_size=args.batch_size,
    )
