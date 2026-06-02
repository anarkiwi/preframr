"""Per-vocab-id tier classification (`structural` / `mid` / `content` / `zero`) used by the loss + heads to weight or partition per-token predictions. Constants `_LOSS_TIER_ORDER`, `_N_LOSS_TIERS`, `_LOSS_TIER_TO_ID`, `_CONTENT_TIER_ID` are the canonical ordering. The (op, reg, subreg) → tier switch lives in `preframr_tokens.tier_classify`; this module is the thin torch-side adapter."""

import torch

from preframr_tokens.stfconstants import LOSS_TIER_NAMES
from preframr_tokens import (
    CONTENT_TIER,
    build_vocab_tier_ids,
    build_vocab_tier_map,
    is_freq_onset_atom,
    op_name_by_id,
    vocab_id_tier,
)

_LOSS_TIER_ORDER = LOSS_TIER_NAMES
_N_LOSS_TIERS = len(_LOSS_TIER_ORDER)
_LOSS_TIER_TO_ID = {name: i for i, name in enumerate(_LOSS_TIER_ORDER)}
_CONTENT_TIER_ID = _LOSS_TIER_TO_ID[CONTENT_TIER]
_STRUCTURAL_TIER_ID = _LOSS_TIER_TO_ID["structural"]


def _build_vocab_onset_weight(args, n_vocab, tokens, tkmodel):
    """Per-vocab-id CE weight that up-weights FREQ V0-onset vids by --onset-loss-weight, to
    force capacity onto the rare melodic onset. A vid counts as onset if its first decoded
    base atom is a freq onset. W==1.0 (default) returns ones (no-op)."""
    weight = float(getattr(args, "onset_loss_weight", 1.0))
    weights = torch.ones(n_vocab, dtype=torch.float32)
    if weight == 1.0 or tokens is None or len(tokens) == 0:
        return weights
    from preframr_tokens import RegTokenizer  # pylint: disable=import-outside-toplevel

    if tkmodel is not None and not isinstance(tkmodel, str):
        tkmodel = tkmodel.to_str()
    rt = RegTokenizer(args, tokens=tokens)
    rt.load(tkmodel, tokens)
    n_base = len(tokens)
    for vid in range(n_vocab):
        base_ids = rt.decode([vid]) if rt.tkmodel else [vid]
        for bid in base_ids:
            bid = int(bid)
            if bid >= n_base:
                continue
            row = tokens.iloc[bid]
            if is_freq_onset_atom(row["op"], row["reg"], row["subreg"]):
                weights[vid] = weight
            break
    return weights


def _build_vocab_class_weight(args, n_vocab, tokens, tkmodel):
    """Per-vocab-id class weight: structural/mid/content/zero tiers."""
    weights = torch.ones(n_vocab, dtype=torch.float32)
    if tokens is None or len(tokens) == 0:
        return weights
    if not getattr(args, "token_class_loss", False):
        return weights
    from preframr_tokens import (  # pylint: disable=import-outside-toplevel
        RegTokenizer,
    )

    if tkmodel is not None and not isinstance(tkmodel, str):
        tkmodel = tkmodel.to_str()
    rt = RegTokenizer(args, tokens=tokens)
    rt.load(tkmodel, tokens)
    tier_w = {
        "structural": float(args.token_class_weight_structural),
        "mid": float(args.token_class_weight_mid),
        "content": float(args.token_class_weight_content),
        "zero": float(args.token_class_weight_zero),
    }
    for vid in range(n_vocab):
        tier = vocab_id_tier(vid, rt, tokens)
        weights[vid] = tier_w[tier]
    return weights


def _build_vocab_tier_id(args, n_vocab, tokens, tkmodel):
    """Per-vocab-id tier index in ``_LOSS_TIER_ORDER`` for the learnable-loss path. Always built (default tier-id is 'content') so the buffer is available even when ``--learnable-class-loss`` is off."""
    if tokens is None or len(tokens) == 0:
        return torch.full((n_vocab,), _CONTENT_TIER_ID, dtype=torch.long)
    from preframr_tokens import (  # pylint: disable=import-outside-toplevel
        RegTokenizer,
    )

    if tkmodel is not None and not isinstance(tkmodel, str):
        tkmodel = tkmodel.to_str()
    rt = RegTokenizer(args, tokens=tokens)
    rt.load(tkmodel, tokens)
    return torch.from_numpy(
        build_vocab_tier_ids(rt, tokens, n_vocab, tier_order=_LOSS_TIER_ORDER)
    ).long()


def build_tier_map(args, n_vocab, tokens, tkmodel):
    """Return ``{vocab_id: tier_name}`` for the active pipeline; consumed by `GeneralizationGate`."""
    if tokens is None or len(tokens) == 0:
        return {vid: CONTENT_TIER for vid in range(n_vocab)}
    from preframr_tokens import (  # pylint: disable=import-outside-toplevel
        RegTokenizer,
    )

    if tkmodel is not None and not isinstance(tkmodel, str):
        tkmodel = tkmodel.to_str()
    rt = RegTokenizer(args, tokens=tokens)
    rt.load(tkmodel, tokens)
    return build_vocab_tier_map(rt, tokens, n_vocab)


def build_op_map(args, n_vocab, tokens, tkmodel):
    """Return ``{vocab_id: op_class_name}`` — the op of each vid's first decoded base atom. Lets the
    GeneralizationGate report per-op-class accuracy, i.e. WHICH pattern-compressing token actually learns
    on the (byte-exact) data: the DIFF delta vs the BACK_REF distance vs the STAMP_REF / WAVETABLE
    codebook id, separately from the structural/content tier split. Mirrors ``build_tier_map``.
    """
    names = op_name_by_id()
    if tokens is None or len(tokens) == 0:
        return {vid: "SET" for vid in range(n_vocab)}
    from preframr_tokens import RegTokenizer  # pylint: disable=import-outside-toplevel

    if tkmodel is not None and not isinstance(tkmodel, str):
        tkmodel = tkmodel.to_str()
    rt = RegTokenizer(args, tokens=tokens)
    rt.load(tkmodel, tokens)
    n_base = len(tokens)
    out = {}
    for vid in range(n_vocab):
        base_ids = rt.decode([vid]) if rt.tkmodel else [vid]
        label = "_unknown"
        for bid in base_ids:
            bid = int(bid)
            if bid >= n_base:
                continue
            op = int(tokens.iloc[bid]["op"])
            label = names.get(op, f"OP{op}")
            break
        out[vid] = label
    return out


def _build_tier_vocab_partition(args, n_vocab, tokens, tkmodel):
    """For each tier name, return (vocab_ids_in_tier, full_to_local_map) tensors. full_to_local_map[v] = local-within-tier id if v is in tier, else -1."""
    tier_ids = _build_vocab_tier_id(args, n_vocab, tokens, tkmodel)
    partition = {}
    for tier_name, tier_id in _LOSS_TIER_TO_ID.items():
        in_tier = (tier_ids == tier_id).nonzero(as_tuple=True)[0]
        full_to_local = torch.full((n_vocab,), -1, dtype=torch.long)
        full_to_local[in_tier] = torch.arange(in_tier.numel(), dtype=torch.long)
        partition[tier_name] = (in_tier, full_to_local)
    return partition
