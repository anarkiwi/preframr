import concurrent.futures
import zstandard as zstd
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)

UNK_TOKEN = "<unk>"
END_OF_WORD_SUFFIX = "</w>"


def get_tk(tkvocab, tokenizer="bpe", initial_alphabet=[]):
    if tokenizer == "unigram":
        tk = Tokenizer(models.Unigram())
        tk.pre_tokenizer = pre_tokenizers.Punctuation()
        tk.decoder = decoders.Metaspace(replacement=" ")
        tk.normalizer = None
        trainer = trainers.UnigramTrainer(
            vocab_size=tkvocab,
            show_progress=True,
            special_tokens=[UNK_TOKEN],
            initial_alphabet=initial_alphabet,
            unk_token=UNK_TOKEN,
            shrinking_factor=0.95,
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
        tk.pre_tokenizer = pre_tokenizers.Punctuation()
        tk.decoder = decoders.BPEDecoder()
        trainer = trainers.BpeTrainer(
            vocab_size=tkvocab,
            min_frequency=2,
            special_tokens=[UNK_TOKEN],
            limit_alphabet=tkvocab,
            initial_alphabet=initial_alphabet,
            show_progress=True,
        )
        return tk, trainer
    raise ValueError


def train_worker(tokenizer, tkvocab, args_tkmodel, uni_files, initial_alphabet=[]):
    def read_uni(uni_file):
        with zstd.open(uni_file, "r") as f:
            return f.read()

    def reader():
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
            futures = [p.submit(read_uni, uni_file) for uni_file in uni_files]
            for future in concurrent.futures.as_completed(futures):
                yield future.result()

    tkmodel, trainer = get_tk(
        tkvocab, tokenizer=tokenizer, initial_alphabet=initial_alphabet
    )
    if trainer is not None:
        tkmodel.train_from_iterator(reader(), trainer=trainer)
    else:
        tkmodel.train_from_iterator(reader(), vocab_size=tkvocab, show_progress=True)
    tkmodel.save(args_tkmodel)
