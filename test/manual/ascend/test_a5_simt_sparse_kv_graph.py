"""A5 smoke test for writeback, joint/fallback Plan, Transfer and SFA replay.

Use the A5 kernel wheel with sparse KV I/O, the Plan/Transfer overlay and
ASCEND_CUSTOM_OPP_PATH, as for the SGLang server.
"""

import os
import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import create_shm_tensor, free_shm

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.attention import (
    _forward_simt_decode,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.simt_lru import (
    SparseKVSimtLRU,
    load_simt_ops,
)
from sglang.srt.model_executor.forward_mode import ForwardMode


@unittest.skipUnless(
    os.getenv("SGLANG_NPU_SPARSE_KV_SIMT_OPS_PATH"),
    "requires the A5 SIMT CANN overlay",
)
class TestA5SimtSparseKVGraph(unittest.TestCase):
    def test_mtp3_replay_after_reject(self):
        device = torch.device("npu")
        torch.npu.set_device(int(os.getenv("SGLANG_TEST_NPU_DEVICE", "0")))
        load_simt_ops(os.environ["SGLANG_NPU_SPARSE_KV_SIMT_OPS_PATH"])
        host, _, dva = create_shm_tensor(
            (2, 128, 1, 576), torch.bfloat16, torch.npu.current_device()
        )
        self.addCleanup(self._free_host)
        host[1, 0, :, :512].fill_(2)
        host[1, 4, :, :512].fill_(6)
        host[1, [0, 4], :, 512:] = 0
        manager = SimpleNamespace(
            size=2,
            max_context_len=128,
            sparse_context_len=16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            device=device,
            store_dtype=torch.bfloat16,
            dev_ptr_list=[dva],
            host_kv_buffer=[host],
            start_layer=0,
        )
        manager.simt_lru = SparseKVSimtLRU(manager, verify_width=4)
        backend = SimpleNamespace(
            sparse_kv_manager=manager, kv_lora_rank=512, qk_rope_head_dim=64
        )
        layer = SimpleNamespace(layer_id=0, tp_q_head_num=16, scaling=1.0)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=4),
            req_pool_indices=torch.tensor([1, 0], device=device),
            seq_lens=torch.zeros(2, dtype=torch.int64, device=device),
            out_cache_loc=torch.tensor([1, 2, 3, 4, 0, 0, 0, 0], device=device),
        )
        query = torch.zeros((8, 16, 512), dtype=torch.bfloat16, device=device)
        q_rope = torch.zeros((8, 16, 64), dtype=torch.bfloat16, device=device)
        nope = torch.zeros((8, 1, 512), dtype=torch.bfloat16, device=device)
        rope = torch.zeros((8, 1, 64), dtype=torch.bfloat16, device=device)
        topk = torch.full((8, 16), -1, dtype=torch.int32, device=device)

        def set_step(prefix, previous, first_value, *, unsorted):
            batch.seq_lens[0] = prefix
            for row in range(4):
                nope[row].fill_(first_value + row)
                tokens = [0, previous, prefix + row]
                if unsorted:
                    # An invalid future position in the middle tests that the
                    # SFA indices retain all remaining valid slots.
                    tokens = [tokens[2], 127, tokens[0], tokens[1]]
                topk[row].fill_(-1)
                values = torch.tensor(tokens, device=device, dtype=torch.int32)
                if unsorted:
                    topk[row, : len(tokens)] = values
                else:
                    topk[row, -len(tokens) :] = values

        def forward():
            return _forward_simt_decode(
                backend, layer, batch, query, q_rope, nope, rope, topk
            )

        set_step(5, 4, 10, unsorted=False)
        self._check(forward(), previous_value=6, first_value=10)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            output = forward()
        graph.replay()
        self._check(output, previous_value=6, first_value=10)

        # Only position 5 was accepted. The old resident KV for positions
        # 6..8 must not be reused after replay writes the replacement values.
        set_step(6, 5, 30, unsorted=True)
        graph.replay()
        self._check(output, previous_value=10, first_value=30)

    def _check(self, output, previous_value, first_value):
        expected = torch.zeros_like(output.cpu())
        for row in range(4):
            expected[row].fill_((2 + previous_value + first_value + row) / 3)
        torch.testing.assert_close(output.cpu(), expected, atol=0.125, rtol=0)

    def _free_host(self):
        torch.npu.synchronize()
        free_shm(torch.npu.current_device())


if __name__ == "__main__":
    unittest.main()
