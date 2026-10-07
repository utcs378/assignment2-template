"""Part 3.2: fused QKV tensor-parallel functional test.

Run: PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/test_fused_qkv.py
"""

import datetime
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from picotron.process_group_manager import setup_process_group_manager
from picotron.tensor_parallel.tensor_parallel import FusedQKVColumnParallelLinear


RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", RANK))
DEVICE = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")


def test_fused_qkv_column_parallel():
    assert WORLD_SIZE == 2, "Run this test with exactly two ranks."
    setup_process_group_manager(tp_size=WORLD_SIZE, cp_size=1, pp_size=1, dp_size=1)
    # SmolLM-1.7B geometry: hidden=2048, 32 Q heads, 32 KV heads, head_dim=64.
    hidden_size, num_heads, num_kv_heads, head_dim = 2_048, 32, 32, 64
    projection = FusedQKVColumnParallelLinear(hidden_size, num_heads, num_kv_heads, head_dim, bias=False).to(DEVICE)
    assert projection.heads_per_partition == 16
    assert projection.kv_heads_per_partition == 16
    assert tuple(projection.qkv_split_sizes_per_partition) == (1_024, 1_024, 1_024)

    q_weight = torch.arange(2_048 * hidden_size, device=DEVICE, dtype=torch.float32).view(2_048, hidden_size) / 10
    k_weight = torch.arange(2_048 * hidden_size, device=DEVICE, dtype=torch.float32).view(2_048, hidden_size) / 7
    v_weight = torch.arange(2_048 * hidden_size, device=DEVICE, dtype=torch.float32).view(2_048, hidden_size) / 5
    local_weight = torch.cat((q_weight.chunk(2)[RANK], k_weight.chunk(2)[RANK], v_weight.chunk(2)[RANK]))
    with torch.no_grad():
        projection.weight.copy_(local_weight)

    x = torch.randn(2, 128, hidden_size, device=DEVICE, requires_grad=True)
    output = projection(x)
    torch.testing.assert_close(output, F.linear(x, local_weight), rtol=1e-5, atol=1e-5)
    output.sum().backward()
    # Large full-width float32 reductions accumulate in a different order across
    # ranks than the local oracle; retain a tight relative tolerance for that sum.
    torch.testing.assert_close(
        x.grad[0, 0],
        q_weight.sum(0) + k_weight.sum(0) + v_weight.sum(0),
        rtol=2e-6,
        atol=1e-3,
    )


if __name__ == "__main__":
    dist.init_process_group(
        backend="nccl" if DEVICE.type == "cuda" else "gloo",
        rank=RANK, world_size=WORLD_SIZE, timeout=datetime.timedelta(minutes=2),
    )
    try:
        test_fused_qkv_column_parallel()
        if RANK == 0:
            print("Fused-QKV test passed")
    finally:
        dist.destroy_process_group()
