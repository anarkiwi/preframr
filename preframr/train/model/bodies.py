"""Torchtune body factories: gemma / llama2 / llama3_2 / mistral / phi3 / qwen2 wrappers + `MODEL_GETTERS` dispatch dict + `MODEL_PRECISION` + `OPTIMIZER`. Pure torchtune; no pytorch-lightning."""

import pyarrow
import torch
from schedulefree import AdamWScheduleFree

pyarrow.PyExtensionType = pyarrow.ExtensionType
# pylint: disable=wrong-import-position
from torchtune.models.gemma._component_builders import gemma
from torchtune.models.llama2._component_builders import llama2
from torchtune.models.llama3_2._component_builders import llama3_2
from torchtune.models.mistral._component_builders import mistral
from torchtune.models.phi3._component_builders import phi3
from torchtune.models.qwen2._component_builders import qwen2

MODEL_PRECISION = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
OPTIMIZER = AdamWScheduleFree


def get_gemma(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed
    head_dim = args.embed // args.heads
    kv_heads = args.kv_heads if args.kv_heads else args.heads

    return gemma(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=kv_heads,
        embed_dim=args.embed,
        head_dim=head_dim,
        intermediate_dim=intermediate,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_gemma2(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed
    head_dim = args.embed // args.heads
    kv_heads = args.kv_heads if args.kv_heads else args.heads

    return gemma(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=kv_heads,
        embed_dim=args.embed,
        head_dim=head_dim,
        intermediate_dim=intermediate,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_llama2(n_vocab, args):
    return llama2(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_llama3_2(n_vocab, args):
    return llama3_2(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
        scale_factor=args.rope_scale,
        tie_word_embeddings=args.tie_word_embeddings,
    )


def get_mistral(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed

    return mistral(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        intermediate_dim=intermediate,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_phi3(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed

    return phi3(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        intermediate_dim=intermediate,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


def get_qwen2(n_vocab, args):
    intermediate = args.intermediate if args.intermediate else args.embed

    return qwen2(
        vocab_size=n_vocab,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.embed,
        intermediate_dim=intermediate,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
        norm_eps=args.norm_eps,
        rope_base=args.rope_base,
    )


MODEL_GETTERS = {
    "gemma": get_gemma,
    "gemma2": get_gemma2,
    "llama2": get_llama2,
    "mistral": get_mistral,
    "phi3": get_phi3,
    "qwen2": get_qwen2,
    "llama3_2": get_llama3_2,
}
