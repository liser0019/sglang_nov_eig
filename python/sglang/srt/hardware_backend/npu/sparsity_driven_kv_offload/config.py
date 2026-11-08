"""Configuration and validation for sparsity-driven KV offload."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    get_dsa_index_head_dim,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    attention_backends,
    get_disagg,
    get_exec,
    get_memory,
    get_schedule,
    get_spec,
)
from sglang.srt.utils.device import is_npu

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def is_sparsity_driven_kv_offload_enabled(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
) -> bool:
    if not envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get():
        return False

    prefill_attention_backend, decode_attention_backend = attention_backends()
    if not (
        is_npu()
        and prefill_attention_backend == "ascend"
        and decode_attention_backend == "ascend"
        and use_mla_backend
        and is_deepseek_dsa(model_config.hf_config)
    ):
        raise ValueError(
            "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD requires an NPU "
            "DSA-family MLA model "
            "(for example DeepSeek V3.2 or GLM-5.x) using the Ascend MLA "
            "attention backend."
        )
    if get_schedule().max_running_requests is None:
        raise ValueError(
            "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD requires max_running_requests "
            "to be set to bound the per-process host KV allocation."
        )
    return True


def get_sparsity_driven_kv_offload_sparse_context_len(
    *,
    model_config: ModelConfig,
) -> int:
    """Return the per-request on-device sparse KV window size."""
    sparse_context_len = int(get_dsa_index_topk(model_config.hf_config))
    if sparse_context_len <= 0:
        raise ValueError(
            "Sparsity-driven KV offload requires a positive DSA index_topk, "
            f"got {sparse_context_len}."
        )
    return sparse_context_len


def get_sparsity_driven_kv_offload_index_head_dim(
    *,
    model_config: ModelConfig,
) -> int:
    index_head_dim = getattr(model_config, "index_head_dim", None)
    if index_head_dim is None:
        index_head_dim = get_dsa_index_head_dim(model_config.hf_config)
    index_head_dim = int(index_head_dim)
    if index_head_dim <= 0:
        raise ValueError(
            "Sparsity-driven KV offload requires a positive DSA index_head_dim, "
            f"got {index_head_dim}."
        )
    return index_head_dim


def get_sparsity_driven_kv_offload_cell_size(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
    num_layers: int,
    element_size: int,
    start_layer: int = 0,
    is_draft_worker: bool = False,
) -> Optional[int]:
    if not is_sparsity_driven_kv_offload_enabled(
        model_config=model_config,
        use_mla_backend=use_mla_backend,
    ):
        return None

    index_head_dim = get_sparsity_driven_kv_offload_index_head_dim(
        model_config=model_config
    )
    from sglang.srt.mem_cache.kv_cache_configurator import (
        _should_elide_dsa_index_k,
    )

    if _should_elide_dsa_index_k(is_draft_worker=is_draft_worker):
        num_indexer_layers = sum(
            not dsa_layer_skips_topk(model_config.hf_config, layer_id)
            for layer_id in range(start_layer, start_layer + num_layers)
        )
    else:
        num_indexer_layers = num_layers
    return index_head_dim * num_indexer_layers * element_size


def get_simt_ops_path(*, enabled: bool, dcp_size: int) -> str:
    """Validate the initial resident-KV integration before allocating the pool."""
    path = envs.SGLANG_NPU_SPARSE_KV_SIMT_OPS_PATH.get()
    if not path:
        return ""
    if not enabled:
        raise ValueError(
            "A5 SIMT sparse KV requires SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1"
        )

    import torch

    from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
    from sglang.srt.model_executor.cuda_graph_config import Backend

    device_name = torch.npu.get_device_name(torch.npu.current_device())
    soc_name = device_name.replace(" ", "").lower().removeprefix("ascend")
    if not soc_name.startswith("950") or soc_name.startswith("950dt"):
        raise ValueError(
            f"A5 SIMT sparse KV requires a 950 device with ACL Host mapping, got {device_name}"
        )
    if get_disagg().disaggregation_mode != "null" or get_spec().speculative_algorithm:
        raise ValueError(
            "A5 SIMT sparse KV currently requires colocated, non-speculative decode"
        )
    if dcp_size > 1 or is_dsa_enable_prefill_cp():
        raise ValueError("A5 SIMT sparse KV currently requires context parallelism disabled")
    if not get_memory().disable_radix_cache:
        raise ValueError("A5 SIMT sparse KV currently requires --disable-radix-cache")
    graph = get_exec().graph.cuda_graph_config
    if (
        graph.decode.backend != Backend.DISABLED
        or graph.prefill.backend != Backend.DISABLED
    ):
        raise ValueError("A5 SIMT sparse KV currently requires eager prefill and decode")
    return path
