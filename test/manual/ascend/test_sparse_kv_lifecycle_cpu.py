"""Dependency-free regression checks for sparse KV lifecycle and row sizing.

Run this file directly with Python. It executes the production Python methods
in isolation; it does not emulate NPU kernels or validate graph execution.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(
    os.getenv("SGLANG_CPU_AUDIT_SOURCE_ROOT", Path(__file__).resolve().parents[3])
)
REFERENCE = Path(os.getenv("SGLANG_CPU_AUDIT_REFERENCE_ROOT", ROOT))
OFFLOAD = "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload"


def load_definitions(path, names, namespace=None, owner=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if owner is not None:
        nodes = next(
            node.body for node in nodes if getattr(node, "name", None) == owner
        )
    selected = [node for node in nodes if getattr(node, "name", None) in names]
    assert len(selected) == len(names), (path, names)
    module = ast.Module(body=selected, type_ignores=[])
    result = dict(namespace or {})
    exec(compile(module, str(path), "exec", dont_inherit=False), result)
    return result


class Selection(list):
    def __iadd__(self, amount):
        return [value + amount for value in self]


class Generations:
    def __init__(self, size):
        self.values = [0] * size

    def __getitem__(self, rows):
        return Selection(self.values[row] for row in rows)

    def __setitem__(self, rows, values):
        for row, value in zip(rows, values):
            self.values[row] = value


class RequestKV:
    def __init__(self, row=None):
        self.req_pool_idx = row
        self.kv_allocated_len = int(row is not None)

    @property
    def holds_kv(self):
        return self.req_pool_idx is not None


class TestRequestLifecycle(unittest.TestCase):
    def setUp(self):
        methods = load_definitions(
            REFERENCE / "python/sglang/srt/mem_cache/memory_pool.py",
            {"alloc", "alloc_rows", "free_rows"},
            owner="ReqToTokenPool",
        )
        pool_type = type(
            "Pool",
            (),
            {name: methods[name] for name in ("alloc", "alloc_rows", "free_rows")},
        )
        self.pool = pool_type()
        self.pool.free_slots = [1, 2, 3]
        self.pool.req_generation = Generations(4)
        self.pool._aux_cache = None
        self.prior_hooks = []
        self.pool._on_alloc_rows = lambda rows: self.prior_hooks.append(list(rows))
        manager_methods = load_definitions(
            ROOT / OFFLOAD / "manager.py",
            {"_install_req_alloc_hook", "init_req"},
            owner="SparseKVCacheManager",
        )
        self.manager_type = type(
            "Manager",
            (),
            {
                name: manager_methods[name]
                for name in ("_install_req_alloc_hook", "init_req")
            },
        )
        self.resets = []
        self.manager = self.manager_type()
        self.manager.reset_requests = lambda rows: self.resets.append(list(rows))
        self.manager.host_kv_ctx_len = {}
        self.manager._install_req_alloc_hook(self.pool)

    def request(self, row=None):
        # Current Req has a kv member, without the old top-level fields.
        return SimpleNamespace(kv=RequestKV(row), origin_input_ids=[10, 20])

    def test_new_request_uses_current_req_interface(self):
        req = self.request()
        self.assertEqual(self.pool.alloc([req]), [3])
        self.assertEqual(req.kv.req_pool_idx, 3)
        self.assertEqual(self.resets, [[3]])

    def test_existing_request_does_not_reset_chunked_prefill(self):
        req = self.request()
        self.pool.alloc([req])
        req.kv.kv_allocated_len = 5
        self.pool.alloc([req])
        self.assertEqual(self.resets, [[3]])

    def test_direct_row_allocation_and_existing_hook(self):
        self.assertEqual(self.pool.alloc_rows(2), [2, 3])
        self.assertEqual(self.resets, [[2, 3]])
        self.assertEqual(self.prior_hooks, [[2, 3]])

    def test_failed_or_empty_allocation_does_not_reset(self):
        self.assertIsNone(self.pool.alloc_rows(4))
        self.assertEqual(self.pool.alloc_rows(0), [])
        self.assertEqual(self.resets, [])

    def test_reused_row_gets_another_reset(self):
        rows = self.pool.alloc_rows(1)
        self.pool.free_rows(rows)
        self.assertEqual(self.pool.alloc_rows(1), rows)
        self.assertEqual(self.resets, [rows, rows])
        self.assertEqual(self.pool.req_generation.values[3], 2)

    def test_reinstall_does_not_keep_previous_manager(self):
        replacement = self.manager_type()
        replacement.reset_requests = lambda rows: self.resets.append(["new", *rows])
        replacement._install_req_alloc_hook(self.pool)
        self.pool.alloc_rows(1)
        self.assertEqual(self.resets, [["new", 3]])

    def test_explicit_request_init_uses_kv_member(self):
        self.manager.init_req(self.request(row=2))
        self.assertEqual(self.manager.host_kv_ctx_len, {2: 2})
        self.assertEqual(self.resets, [[2]])


class TestPaddedRows(unittest.TestCase):
    def setUp(self):
        self.dp_module = ModuleType("sglang.srt.layers.dp_attention")
        self.dp_module.get_eager_max_batch_size = lambda size: (size + 7) // 8 * 8
        self.dp_module.get_cuda_graph_max_batch_size = lambda size: (size + 7) // 8 * 8
        self.module_patch = patch.dict(
            "sys.modules", {self.dp_module.__name__: self.dp_module}
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        functions = load_definitions(
            ROOT / OFFLOAD / "config.py", {"get_simt_max_rows"}
        )
        self.max_rows = functions["get_simt_max_rows"]

    def test_small_tp_graph_bucket_fits_resident_rows(self):
        execution = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(bs=[1, 2, 4, 8])
                )
            ),
            overlap=SimpleNamespace(enable_two_batch_overlap=False),
        )
        config = SimpleNamespace(from_server_args=lambda args: None)
        functions = load_definitions(
            REFERENCE
            / "python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py",
            {"get_batch_sizes_to_capture"},
            {
                "get_exec": lambda: execution,
                "get_flags": lambda: SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=False)
                ),
                "DllmConfig": config,
                "can_defer_cp_decode_attn_tp": lambda: False,
                "get_cuda_graph_batch_size_alignment": lambda **kwargs: 8,
                "ceil_align": lambda size, multiple: (
                    (size + multiple - 1) // multiple * multiple
                ),
            },
        )
        runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(size=1), server_args=None
        )
        buckets, _ = functions["get_batch_sizes_to_capture"](runner, 4)
        self.assertEqual(max(buckets), 8)
        # Two request slots (one real plus padding) must serve eight padded requests.
        self.assertGreaterEqual(self.max_rows(2, 4), max(buckets) * 4)

    def test_normal_batch_preserves_reserved_dummy_capacity(self):
        self.assertEqual(self.max_rows(33, 4), 33 * 4)

    def test_eager_padding_is_also_reserved(self):
        self.dp_module.get_cuda_graph_max_batch_size = lambda size: size
        self.assertEqual(self.max_rows(2, 1), 8)

    def test_budget_includes_the_padded_resident_rows(self):
        functions = load_definitions(
            ROOT / OFFLOAD / "config.py",
            {"get_simt_max_rows", "get_simt_resident_budget", "simt_plan_storage"},
            {
                "SIMT_KV_PAGE_SIZE": 128,
                "get_simt_verify_width": lambda: 4,
                "get_sparsity_driven_kv_offload_sparse_context_len": (
                    lambda **kwargs: 2048
                ),
            },
        )
        model = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, context_len=128)
        resident_data_bytes = self.max_rows(2, 4) * 4096 * (512 + 64) * 2
        self.assertGreater(
            functions["get_simt_resident_budget"](model, 1, 2), resident_data_bytes
        )


if __name__ == "__main__":
    unittest.main()
