"""Back-compat shim: the public symbols of preframr.train.model are re-exported here so consumer imports (`from preframr.train.model import Model`, etc.) keep working. New code should import from the focused submodules (`preframr.train.model.lightning`, `.losses`, `.bodies`, `.factory`)."""

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
from preframr.train.model.lightning import Model
from preframr.train.model.losses import (
    _chunked_list_cross_entropy,
    _cross_entropy_chunk,
    _cross_entropy_logit_chunk,
    chunked_cross_entropy,
)
