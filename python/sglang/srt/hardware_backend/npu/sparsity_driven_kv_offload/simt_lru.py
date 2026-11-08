"""A5 resident MLA cache backed by the native sparse KV Plan and Transfer."""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    SIMT_KV_PAGE_SIZE,
    simt_plan_storage,
)


def load_simt_ops(package_path: str) -> None:
    # Load the selected-operator CANN build beside the full legacy wheel,
    # which provides the Host allocator and the existing KV writeback op.
    package = Path(package_path).resolve()
    vendor = package / "_cann/vendors/sgl_kernel_npu"
    configured = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    if vendor not in {Path(path).resolve() for path in configured if path}:
        raise RuntimeError(
            f"Add {vendor} to ASCEND_CUSTOM_OPP_PATH before starting SGLang"
        )
    library = package / "lib/libsgl_kernel_npu.so"
    torch.ops.load_library(str(library))
    if not all(
        hasattr(torch.ops.npu, name)
        for name in ("sgl_sparse_kv_plan", "sgl_sparse_kv_transfer")
    ):
        raise RuntimeError(f"Sparse KV Plan and Transfer are missing from {library}")
    if not all(
        hasattr(torch.ops.npu, name)
        for name in ("shm_allocator_create_and_register", "unidex_copy")
    ):
        raise RuntimeError("Install the A5 sgl-kernel-npu wheel with sparse KV I/O")


@dataclass
class _LayerState:
    last_req_ids: torch.Tensor
    slot_to_token: torch.Tensor
    lru_slots: torch.Tensor
    row_modes: torch.Tensor
    host_bases: torch.Tensor
    nope: torch.Tensor
    rope: torch.Tensor


class SparseKVSimtLRU:
    def __init__(self, manager, verify_width: int):
        self.size = manager.size
        self.max_rows = self.size * verify_width
        self.verify_width = verify_width
        self.topk = manager.sparse_context_len
        self.block_size = SIMT_KV_PAGE_SIZE
        self.capacity, scratch = simt_plan_storage(self.topk, verify_width)
        self.max_context_len = manager.max_context_len
        host_pages = self.max_context_len // self.block_size
        self.host_num_blocks = self.size * host_pages
        self.nope_bytes = manager.kv_lora_rank * torch.bfloat16.itemsize
        self.rope_bytes = manager.qk_rope_head_dim * torch.bfloat16.itemsize
        if self.host_num_blocks > torch.iinfo(torch.int32).max:
            raise ValueError("A5 SIMT sparse KV host block table exceeds int32")
        if self.nope_bytes % 32 or self.rope_bytes % 32:
            raise ValueError("A5 SIMT sparse KV requires 32-byte aligned MLA components")

        device = manager.device
        self.req_ids = torch.arange(self.size, dtype=torch.int64, device=device)
        self.host_blocks = torch.arange(
            self.host_num_blocks, dtype=torch.int32, device=device
        ).view(self.size, host_pages)
        self.query_lens = torch.ones(self.max_rows, dtype=torch.int32, device=device)
        self.cu_query_lens = torch.arange(
            1, self.max_rows + 1, dtype=torch.int32, device=device
        )
        self.query_rows = torch.arange(self.max_rows, dtype=torch.int64, device=device)
        self.verify_offsets = self.cu_query_lens[:verify_width] - 1
        self.slot_lens = torch.full_like(self.query_lens, self.capacity)
        self.topk_positions = torch.arange(self.topk, device=device)
        self.active_rows = torch.zeros(1, dtype=torch.int32, device=device)
        self.resident_rows = torch.empty_like(self.query_lens)
        pages = self.capacity // self.block_size
        self.resident_blocks = torch.empty(
            (self.max_rows, pages), dtype=torch.int32, device=device
        )
        self.current_slots = torch.empty(
            (self.max_rows, self.topk), dtype=torch.int32, device=device
        )
        self.topk_buffer = torch.empty_like(self.current_slots)
        self.miss_count = torch.empty_like(self.query_lens)
        self.miss_tokens = torch.empty_like(self.current_slots)
        self.miss_slots = torch.empty_like(self.current_slots)
        workspace_size = max(1, self.max_rows * scratch)
        self.workspace = torch.empty(workspace_size, dtype=torch.int32, device=device)
        initial_lru = torch.arange(self.capacity, dtype=torch.int32, device=device)

        self.layers = []
        for base in manager.dev_ptr_list:
            self.layers.append(
                _LayerState(
                    last_req_ids=torch.full(
                        (self.max_rows,), -1, dtype=torch.int64, device=device
                    ),
                    slot_to_token=torch.full(
                        (self.max_rows, self.capacity), -1, dtype=torch.int32, device=device
                    ),
                    lru_slots=initial_lru.repeat(self.max_rows, 1),
                    row_modes=torch.zeros_like(self.query_lens),
                    host_bases=torch.tensor(
                        [base, base + self.nope_bytes], dtype=torch.int64, device=device
                    ),
                    nope=torch.zeros(
                        (self.max_rows, self.capacity, 1, manager.kv_lora_rank),
                        dtype=manager.store_dtype,
                        device=device,
                    ),
                    rope=torch.zeros(
                        (self.max_rows, self.capacity, 1, manager.qk_rope_head_dim),
                        dtype=manager.store_dtype,
                        device=device,
                    ),
                )
            )

    def reset_requests(self, request_slots: torch.Tensor) -> None:
        # Batch rows may move, and the scheduler can reuse a request-pool slot.
        # Add a generation to its ID so neither case can produce a stale hit.
        increments = torch.full_like(request_slots, self.size)
        self.req_ids.index_add_(0, request_slots, increments)

    def _query_rows(self, forward_batch, verify: bool):
        width = self.verify_width if verify else 1
        request_slots = forward_batch.req_pool_indices.to(torch.int32)
        rows = request_slots.numel() * width
        seq_lens = forward_batch.seq_lens[: request_slots.numel()].to(torch.int32)
        if verify:
            # TARGET_VERIFY seq_lens is the committed prefix. Each row adds
            # one candidate and can see exactly up to its own position.
            positions = (seq_lens[:, None] + self.verify_offsets).reshape(-1)
            stable = seq_lens.repeat_interleave(width)
            request_slots = request_slots.repeat_interleave(width)
        else:
            positions = seq_lens - 1
            stable = positions
        valid = (
            (request_slots > 0)
            & (request_slots < self.size)
            & (forward_batch.out_cache_loc[:rows] > 0)
            & (positions >= 0)
            & (positions < self.max_context_len)
        )
        return request_slots.contiguous(), positions, stable.contiguous(), valid

    def _write_kv(self, host_kv, host_base, nope, rope, req_slots, positions, valid):
        rows = req_slots.numel()
        dst_indices = req_slots.to(torch.int64) * self.max_context_len + positions
        unidex_copy_inplace(
            torch.cat((nope[:rows], rope[:rows]), dim=-1).contiguous(),
            host_kv,
            self.query_rows[:rows],
            dst_indices.contiguous(),
            valid.contiguous(),
            1,
            2,
            block_dim=48,
            dst_ptr=host_base,
        )

    def plan_and_load(
        self, layer_idx, forward_batch, topk, nope, rope, host_kv, host_base
    ):
        verify = forward_batch.forward_mode.is_target_verify()
        if verify and forward_batch.spec_info.draft_token_num != self.verify_width:
            raise ValueError("SIMT target verification requires the configured query width")
        req_slots, positions, stable, valid = self._query_rows(forward_batch, verify)
        rows = req_slots.numel()
        index_rows, width = topk.shape
        if not (0 < width <= self.topk and 0 < rows <= self.max_rows):
            raise ValueError(f"Invalid SIMT rows={rows} or TopK shape={tuple(topk.shape)}")
        if index_rows >= rows and width == self.topk:
            plan_topk = topk[:rows].to(torch.int32).contiguous()
        else:
            plan_topk = self.topk_buffer[:rows]
            plan_topk.fill_(-1)
            kept = min(rows, index_rows)
            plan_topk[:kept, :width].copy_(topk[:kept])
        self._write_kv(host_kv, host_base, nope, rope, req_slots, positions, valid)
        state = self.layers[layer_idx]
        seq_lens = torch.where(valid, positions + 1, 0).contiguous()
        self.active_rows.fill_(rows)
        torch.ops.npu.sgl_sparse_kv_plan(
            self.req_ids,
            plan_topk,
            stable,
            seq_lens,
            req_slots,
            self.host_blocks,
            self.active_rows,
            state.last_req_ids,
            state.slot_to_token,
            state.lru_slots,
            self.current_slots,
            self.miss_count,
            self.miss_tokens,
            self.miss_slots,
            self.workspace,
            state.row_modes,
            self.resident_rows,
            self.resident_blocks,
            self.topk,
            self.capacity,
            self.max_context_len,
            self.block_size,
            self.host_num_blocks,
            int(verify),
        )
        torch.ops.npu.sgl_sparse_kv_transfer(
            self.miss_count,
            self.miss_tokens,
            self.miss_slots,
            req_slots,
            self.host_blocks,
            state.host_bases,
            self.active_rows,
            state.nope,
            state.rope,
            self.resident_rows,
            self.topk,
            self.capacity,
            self.block_size,
            self.host_num_blocks,
            self.nope_bytes,
            self.rope_bytes,
        )
        return state.nope[:rows], state.rope[:rows], self.current_slots[:rows]
