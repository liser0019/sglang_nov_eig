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
    is_draft_worker: bool = False,
) -> bool:
    if not use_sparse_offload_pool(is_draft_worker):
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
        is_draft_worker=is_draft_worker,
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


def use_sparse_offload_pool(is_draft_worker: bool) -> bool:
    # EAGLE's draft workers address their paged KV by physical locations.
    # Only the target runner uses the request-major SIMT Host cache.
    return envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get() and not (
        is_draft_worker and envs.SGLANG_NPU_SPARSE_KV_SIMT_OPS_PATH.get()
    )


SIMT_KV_PAGE_SIZE = 128


def simt_plan_storage(topk: int, verify_width: int) -> tuple[int, int]:
    """Return resident capacity and GM scratch elements per query row."""
    page = SIMT_KV_PAGE_SIZE
    capacity = (2 * topk + page - 1) // page * page
    hash_capacity = max(32, 1 << (2 * topk - 1).bit_length())
    scratch = 3 * hash_capacity + capacity + topk
    if verify_width > 1:
        sorted_scratch = (
            64 + ((4 * topk + capacity + 7) & ~7) + max(3 * capacity, 64)
        )
        scratch = max(scratch, sorted_scratch)
    # The native tiler tests 32-byte aligned scratch against the same budget.
    return capacity, scratch if ((scratch * 4 + 31) & ~31) > 112 * 1024 else 0


def get_simt_max_rows(request_slots: int, verify_width: int) -> int:
    from sglang.srt.layers.dp_attention import (
        get_cuda_graph_max_batch_size,
        get_eager_max_batch_size,
    )

    # The request pool includes dummy slot 0. Batch padding can need more
    # query rows than real request slots, especially for small TP batches.
    requests = max(1, request_slots - 1)
    padded_requests = max(
        request_slots,
        get_eager_max_batch_size(requests),
        get_cuda_graph_max_batch_size(requests),
    )
    return padded_requests * verify_width


def get_simt_resident_budget(model_config, num_layers, request_slots) -> int:
    width = get_simt_verify_width()
    topk = get_sparsity_driven_kv_offload_sparse_context_len(model_config=model_config)
    capacity, scratch = simt_plan_storage(topk, width)
    rows = get_simt_max_rows(request_slots, width)
    kv_bytes = (model_config.kv_lora_rank + model_config.qk_rope_head_dim) * 2
    # Per-layer resident KV, reverse map, LRU, row mode and request generation.
    per_layer = rows * (capacity * (kv_bytes + 8) + 12) + 16
    shared = rows * (4 * (4 * topk + scratch + capacity // SIMT_KV_PAGE_SIZE) + 32)
    host_pages = (model_config.context_len + SIMT_KV_PAGE_SIZE - 1) // SIMT_KV_PAGE_SIZE
    return num_layers * per_layer + shared + request_slots * (host_pages * 4 + 8)


def get_simt_verify_width() -> int:
    spec = get_spec()
    if not spec.speculative_algorithm:
        return 1
    width = spec.speculative_num_draft_tokens
    if (
        spec.speculative_algorithm not in ("EAGLE", "NEXTN")
        or spec.speculative_eagle_topk != 1
        or spec.speculative_adaptive
        or width not in (2, 3, 4)
    ):
        raise ValueError(
            "A5 SIMT sparse KV supports fixed-width chain MTP (EAGLE/NEXTN "
            "with eagle_topk=1 and two to four target query rows per request)"
        )
    return width


def get_simt_ops_path(
    *, enabled: bool, dcp_size: int, is_draft_worker: bool
) -> str:
    """Validate the initial resident-KV integration before allocating the pool."""
    path = envs.SGLANG_NPU_SPARSE_KV_SIMT_OPS_PATH.get()
    if not path or is_draft_worker:
        return ""
    if not enabled:
        raise ValueError(
            "A5 SIMT sparse KV requires SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1"
        )

    import torch

    from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
        is_mla_preprocess_enabled,
    )
    from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
    from sglang.srt.model_executor.cuda_graph_config import Backend

    device_name = torch.npu.get_device_name(torch.npu.current_device())
    soc_name = device_name.replace(" ", "").lower().removeprefix("ascend")
    if not soc_name.startswith("950") or soc_name.startswith("950dt"):
        raise ValueError(
            f"A5 SIMT sparse KV requires a 950 device with ACL Host mapping, got {device_name}"
        )
    if get_disagg().disaggregation_mode != "null":
        raise ValueError("A5 SIMT sparse KV currently requires colocated P/D")
    get_simt_verify_width()
    if is_mla_preprocess_enabled():
        raise ValueError("A5 SIMT sparse KV requires SGLANG_NPU_USE_MLAPO=0")
    if dcp_size > 1 or is_dsa_enable_prefill_cp():
        raise ValueError("A5 SIMT sparse KV currently requires context parallelism disabled")
    memory = get_memory()
    if (
        memory.enable_unified_memory
        or memory.enable_hisparse
        or memory.enable_hierarchical_cache
    ):
        raise ValueError("A5 SIMT sparse KV requires the native Ascend offload pool")
    if not memory.disable_radix_cache:
        raise ValueError("A5 SIMT sparse KV currently requires --disable-radix-cache")
    graph = get_exec().graph.cuda_graph_config
    if (
        graph.decode.backend not in (Backend.DISABLED, Backend.FULL)
        or graph.prefill.backend != Backend.DISABLED
        or get_exec().graph.enable_torch_compile
    ):
        raise ValueError(
            "A5 SIMT sparse KV supports eager prefill and eager or full NPU "
            "decode graph, with torch.compile disabled"
        )
    return path
