"""Part 3.5: headwise context-parallel layout and attention parity test.

Run: PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/test_headwise_context_parallel.py
"""

import datetime
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F


# See test_zigzag_context_parallel.py: headwise tests do not execute a
# FlashAttention path, but older course checkouts import it unconditionally.
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
from picotron.context_parallel.context_parallel import HeadwiseContextParallel, head_to_sequence, sequence_to_head
from picotron.process_group_manager import setup_process_group_manager


RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", RANK))
DEVICE = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")


def test_headwise_layout_and_attention():
    assert WORLD_SIZE == 2, "Run this test with exactly two ranks."
    setup_process_group_manager(tp_size=1, cp_size=WORLD_SIZE, pp_size=1, dp_size=1)
    # Match the CP=2, mbs=2, seq_len=128 SmolLM test configuration.
    batch, heads, local_seq, dim = 2, 32, 64, 64
    x = (torch.arange(batch * heads * local_seq * dim, device=DEVICE, dtype=torch.float32).view(batch, heads, local_seq, dim) + 100 * RANK).requires_grad_()
    head_shard = sequence_to_head(x, pgm.process_group_manager.cp_group, WORLD_SIZE)
    restored = head_to_sequence(head_shard, pgm.process_group_manager.cp_group, WORLD_SIZE)
    torch.testing.assert_close(restored, x)
    restored.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x)

    torch.manual_seed(1234 + RANK)
    q = torch.randn(batch, heads, local_seq, dim, device=DEVICE, requires_grad=True)
    k = torch.randn(batch, heads, local_seq, dim, device=DEVICE, requires_grad=True)
    v = torch.randn(batch, heads, local_seq, dim, device=DEVICE, requires_grad=True)
    output = HeadwiseContextParallel.apply(q, k, v, True)

    def full_sequence(tensor):
        pieces = [torch.empty_like(tensor) for _ in range(WORLD_SIZE)]
        dist.all_gather(pieces, tensor.detach(), group=pgm.process_group_manager.cp_group)
        return torch.cat(pieces, dim=2).requires_grad_(True)

    q_ref, k_ref, v_ref = full_sequence(q), full_sequence(k), full_sequence(v)
    reference = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    torch.testing.assert_close(output, reference.chunk(WORLD_SIZE, dim=2)[RANK], rtol=1e-5, atol=1e-5)
    output.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(q.grad, q_ref.grad.chunk(WORLD_SIZE, dim=2)[RANK], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k.grad, k_ref.grad.chunk(WORLD_SIZE, dim=2)[RANK], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v.grad, v_ref.grad.chunk(WORLD_SIZE, dim=2)[RANK], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    dist.init_process_group(
        backend="nccl" if DEVICE.type == "cuda" else "gloo",
        rank=RANK, world_size=WORLD_SIZE, timeout=datetime.timedelta(minutes=2),
    )
    try:
        test_headwise_layout_and_attention()
        if RANK == 0:
            print("Headwise context-parallel test passed")
    finally:
        dist.destroy_process_group()
