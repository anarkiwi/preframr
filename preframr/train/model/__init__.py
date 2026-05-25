"""Back-compat shim: every previously-public symbol of preframr.train.model is re-exported here so consumer imports (`from preframr.train.model import Model`, etc.) keep working unchanged. New code should import from the focused submodules (`preframr.train.model.lightning`, `.heads`, `.losses`, `.tier_map`, `.bodies`, `.factory`)."""

# pylint: disable=unused-import
from preframr.train.model.bodies import (
    MODEL_GETTERS,
    MODEL_PRECISION,
    OPTIMIZER,
    get_gemma,
    get_gemma2,
    get_llama2,
    get_llama3_2,
    get_mistral,
    get_phi3,
    get_qwen2,
)
from preframr.train.model.factory import (
    SchedulerFreeModelCheckpoint,
    cpu_compile,
    cuda_compile,
    get_device,
    get_model,
)
from preframr.train.model.heads import (
    MoSHead,
    PerTierHeads,
    _mos_log_mixture,
    per_tier_unified_log_p,
)
from preframr.train.model.lightning import Model
from preframr.train.model.losses import (
    _build_vocab_frame_weight,
    _chunked_list_cross_entropy,
    _cross_entropy_chunk,
    _cross_entropy_logit_chunk,
    _infonce_per_tensor,
    chunked_cross_entropy,
    content_contrastive_loss,
)
from preframr.train.model.tier_map import (
    _CONTENT_TIER_ID,
    _LOSS_TIER_ORDER,
    _LOSS_TIER_TO_ID,
    _N_LOSS_TIERS,
    _build_tier_vocab_partition,
    _build_vocab_class_weight,
    _build_vocab_tier_id,
    build_tier_map,
)
