"""CPU checks of the real SIMT attention adapter with an SFA call spy.

Run directly with a CPU PyTorch installation. No NPU kernels are executed.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(
    os.getenv("SGLANG_CPU_AUDIT_SOURCE_ROOT", Path(__file__).resolve().parents[3])
)


@unittest.skipIf(torch is None, "requires CPU PyTorch")
class TestResidentSFAArguments(unittest.TestCase):
    def setUp(self):
        path = ROOT / (
            "python/sglang/srt/hardware_backend/npu/"
            "sparsity_driven_kv_offload/attention.py"
        )
        function = next(
            node for node in ast.parse(path.read_text()).body
            if getattr(node, "name", None) == "_forward_simt_decode"
        )
        self.sfa = Mock(return_value=torch.ones((4, 1, 1)))
        namespace = {
            "torch": torch,
            "torch_npu": SimpleNamespace(npu_sparse_flash_attention=self.sfa),
            "_get_sparse_kv_manager": lambda backend: backend.sparse_kv_manager,
            "normalize_batch_topk_indices": lambda value: value,
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        self.forward = namespace["_forward_simt_decode"]
        self.slots = torch.full((4, 16), -1, dtype=torch.int32)
        # Include holes, a fully empty query, a high physical slot and a full row.
        self.slots[0, [1, 5, 9]] = torch.tensor([0, 4, 5], dtype=torch.int32)
        self.slots[2, 6] = 127
        self.slots[3] = torch.arange(16, dtype=torch.int32)
        lru = SimpleNamespace(
            plan_and_load=lambda *args: (
                torch.zeros((4, 128, 1, 1)),
                torch.zeros((4, 128, 1, 1)),
                self.slots,
            ),
            topk_positions=torch.arange(16),
            block_size=128,
            cu_query_lens=torch.arange(1, 5, dtype=torch.int32),
            slot_lens=torch.full((4,), 128, dtype=torch.int32),
            resident_blocks=torch.arange(4, dtype=torch.int32).view(4, 1),
        )
        self.backend = SimpleNamespace(
            sparse_kv_manager=SimpleNamespace(
                simt_lru=lru, start_layer=0, host_kv_buffer=[None], dev_ptr_list=[0]
            ),
            kv_lora_rank=1,
            qk_rope_head_dim=1,
        )
        self.layer = SimpleNamespace(layer_id=0, scaling=1.0, tp_q_head_num=1)

    def run_forward(self):
        # One extra query tests the existing graph-padding output path.
        query = torch.zeros((5, 1, 1))
        return self.forward(
            self.backend, self.layer, None, query, query, query, query, self.slots
        )

    def test_packed_counts_and_empty_rows(self):
        output = self.run_forward()
        args, kwargs = self.sfa.call_args
        lengths = kwargs["actual_seq_lengths_kv"]
        self.assertEqual(lengths.tolist(), [3, 1, 1, 16])
        self.assertEqual(lengths.dtype, torch.int32)
        self.assertTrue(lengths.is_contiguous())
        indices = args[3][:, 0]
        self.assertEqual(indices[0].tolist(), [0, 4, 5] + [-1] * 13)
        self.assertEqual(indices[1].tolist(), [0] + [-1] * 15)
        self.assertEqual(indices[2].tolist(), [127] + [-1] * 15)
        self.assertEqual(indices[3].tolist(), list(range(16)))
        self.assertEqual(output[:, 0].tolist(), [1, 0, 1, 1, 0])

    def test_lengths_follow_the_current_plan(self):
        self.run_forward()
        self.slots.fill_(-1)
        self.slots[0, :2] = torch.tensor([7, 9], dtype=torch.int32)
        self.run_forward()
        self.assertEqual(
            self.sfa.call_args.kwargs["actual_seq_lengths_kv"].tolist(),
            [2, 1, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
