"""A5 resident MLA cache backed by the native sparse KV Plan and Transfer."""

import os
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import torch


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
    def __init__(self, manager):
        self.size = manager.size
        self.topk = manager.sparse_context_len
        self.capacity = 2 * self.topk
        self.max_context_len = manager.max_context_len
        self.block_size = gcd(128, self.max_context_len, self.capacity)
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
        self.query_lens = torch.ones(self.size, dtype=torch.int32, device=device)
        self.slot_lens = torch.full_like(self.query_lens, self.capacity)
        self.topk_positions = torch.arange(self.topk, device=device)
        self.active_rows = torch.zeros(1, dtype=torch.int32, device=device)
        self.resident_rows = torch.empty_like(self.query_lens)
        pages = self.capacity // self.block_size
        self.resident_blocks = torch.empty(
            (self.size, pages), dtype=torch.int32, device=device
        )
        self.current_slots = torch.empty(
            (self.size, self.topk), dtype=torch.int32, device=device
        )
        self.topk_buffer = torch.empty_like(self.current_slots)
        self.miss_count = torch.empty_like(self.query_lens)
        self.miss_tokens = torch.empty_like(self.current_slots)
        self.miss_slots = torch.empty_like(self.current_slots)
        hash_capacity = max(32, 1 << (2 * self.topk - 1).bit_length())
        row_bytes = (3 * hash_capacity + self.capacity + self.topk) * 4
        aligned_row_bytes = (row_bytes + 31) // 32 * 32
        workspace_size = (
            1 if aligned_row_bytes <= 112 * 1024 else self.size * (row_bytes // 4)
        )
        self.workspace = torch.empty(workspace_size, dtype=torch.int32, device=device)
        initial_lru = torch.arange(self.capacity, dtype=torch.int32, device=device)

        self.layers = []
        for base in manager.dev_ptr_list:
            self.layers.append(
                _LayerState(
                    last_req_ids=torch.full_like(self.req_ids, -1),
                    slot_to_token=torch.full(
                        (self.size, self.capacity), -1, dtype=torch.int32, device=device
                    ),
                    lru_slots=initial_lru.repeat(self.size, 1),
                    row_modes=torch.zeros_like(self.query_lens),
                    host_bases=torch.tensor(
                        [base, base + self.nope_bytes], dtype=torch.int64, device=device
                    ),
                    nope=torch.zeros(
                        (self.size, self.capacity, 1, manager.kv_lora_rank),
                        dtype=manager.store_dtype,
                        device=device,
                    ),
                    rope=torch.zeros(
                        (self.size, self.capacity, 1, manager.qk_rope_head_dim),
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

    def plan_and_load(self, layer_idx, forward_batch, topk):
        rows, width = topk.shape
        if not (0 < width <= self.topk and 0 < rows <= self.size) or (
            rows != forward_batch.batch_size
        ):
            raise ValueError(
                f"Expected {forward_batch.batch_size} SIMT rows and 1..{self.topk} TopK entries, "
                f"got {tuple(topk.shape)}"
            )
        if width == self.topk:
            plan_topk = topk.to(torch.int32).contiguous()
        else:
            plan_topk = self.topk_buffer[:rows]
            plan_topk[:, :width].copy_(topk)
            plan_topk[:, width:].fill_(-1)
        state = self.layers[layer_idx]
        req_slots = forward_batch.req_pool_indices.to(torch.int32).contiguous()
        # Pool slot 0 is reserved. Make any padded query invisible to Plan.
        seq_lens = torch.where(
            (req_slots > 0) & (forward_batch.out_cache_loc >= 0),
            forward_batch.seq_lens.to(torch.int32),
            0,
        ).contiguous()
        self.active_rows.fill_(rows)
        torch.ops.npu.sgl_sparse_kv_plan(
            self.req_ids,
            plan_topk,
            (seq_lens - 1).clamp_min_(0),
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
            0,  # ordinary Decode, with one query per request
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
