"""Part 3.4: zigzag context-parallel correctness and balance test.

Run: PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/test_zigzag_context_parallel.py
"""

import datetime
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F


# Some course checkouts import FlashAttention at module import time even when
# FLASH_ATTEN=0.  Zigzag tests use only the PyTorch/ring path, so provide a
# deliberately unusable placeholder when that optional package is absent.
try:
    import flash_attn  # noqa: F401
except ModuleNotFoundError:
    flash_attn = types.ModuleType("flash_attn")
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.flash_attn_func = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("FlashAttention was called despite FLASH_ATTEN=0")
    )
    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.flash_attn_interface"] = interface

import picotron.process_group_manager as pgm
from picotron.context_parallel.context_parallel import get_zigzag_indices, ring_attention
from picotron.process_group_manager import setup_process_group_manager


RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", RANK))
DEVICE = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")


def test_zigzag_indices_and_balance():
    assert WORLD_SIZE == 2, "Run this test with exactly two ranks."
    setup_process_group_manager(tp_size=1, cp_size=WORLD_SIZE, pp_size=1, dp_size=1)
    sequence_length = 128
    indices = get_zigzag_indices(sequence_length, RANK, WORLD_SIZE, device=DEVICE)
    expected = torch.cat((
        torch.arange(0, 32, device=DEVICE), torch.arange(96, 128, device=DEVICE)
    )) if RANK == 0 else torch.arange(32, 96, device=DEVICE)
    assert torch.equal(indices, expected)

    local_work = (indices + 1).sum(dtype=torch.int64)
    all_work = [torch.zeros_like(local_work) for _ in range(WORLD_SIZE)]
    dist.all_gather(all_work, local_work, group=pgm.process_group_manager.cp_group)
    assert len({int(work.item()) for work in all_work}) == 1, f"imbalanced causal work: {all_work}"
    for seq_len in (4, 8, 12, 20):
        work = [int((get_zigzag_indices(seq_len, rank, WORLD_SIZE, DEVICE) + 1).sum()) for rank in range(WORLD_SIZE)]
        assert len(set(work)) == 1, f"zigzag did not balance seq_len={seq_len}: {work}"

    if DEVICE.type == "cuda":
        torch.manual_seed(71)
        full_q = torch.randn(2, 32, sequence_length, 64, device=DEVICE)
        full_k, full_v = torch.randn_like(full_q), torch.randn_like(full_q)
        actual = ring_attention(full_q[:, :, indices], full_k[:, :, indices], full_v[:, :, indices], 1.0 / 8.0, True, cp_zigzag_en=True)
        reference = F.scaled_dot_product_attention(full_q, full_k, full_v, is_causal=True)
        torch.testing.assert_close(actual, reference[:, :, indices], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    dist.init_process_group(
        backend="nccl" if DEVICE.type == "cuda" else "gloo",
        rank=RANK, world_size=WORLD_SIZE, timeout=datetime.timedelta(minutes=2),
    )
    try:
        test_zigzag_indices_and_balance()
        if RANK == 0:
            print("Zigzag context-parallel test passed")
    finally:
        dist.destroy_process_group()
