"""Part 3.1: distributed vocabulary-padding functional test.

Run: PYTHONPATH=. torchrun --standalone --nproc-per-node=2 tests/test_vocab_padding.py
"""

import datetime
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from picotron.process_group_manager import setup_process_group_manager
from picotron.tensor_parallel.tensor_parallel import VocabParallelEmbeddingPadding


RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", RANK))
DEVICE = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")


def test_vocab_padding():
    assert WORLD_SIZE == 2, "Run this test with exactly two ranks."
    setup_process_group_manager(tp_size=WORLD_SIZE, cp_size=1, pp_size=1, dp_size=1)
    # SmolLM-1.7B uses hidden_size=2048 and vocab_size=49152.  Add one
    # token so the Part 3 padding path is exercised for TP=2.
    vocab_size, hidden_size = 49_153, 2_048
    embedding = VocabParallelEmbeddingPadding(vocab_size, hidden_size).to(DEVICE)

    assert embedding.padded_num_embeddings == 49_154
    per_rank_vocab = 24_577
    assert (embedding.vocab_start_index, embedding.vocab_end_index) == (RANK * per_rank_vocab, (RANK + 1) * per_rank_vocab)
    assert embedding.weight.shape == (per_rank_vocab, hidden_size)

    # Avoid materializing a second 49K x 2048 reference table: every real
    # token i has the easily reconstructed row [i*H, ..., (i+1)*H-1].
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(
            embedding.vocab_start_index * hidden_size,
            embedding.vocab_end_index * hidden_size,
            device=DEVICE,
            dtype=torch.float32,
        ).view(per_rank_vocab, hidden_size))

    token_ids = torch.tensor([[0, 24_576, 49_152, 49_153]], device=DEVICE)
    output = embedding(token_ids)
    expected = torch.stack([
        torch.arange(token * hidden_size, (token + 1) * hidden_size, device=DEVICE, dtype=torch.float32)
        if token < vocab_size else torch.zeros(hidden_size, device=DEVICE)
        for token in token_ids[0].tolist()
    ]).unsqueeze(0)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    output.sum().backward()

    for token in token_ids[0].tolist():
        if embedding.vocab_start_index <= token < vocab_size and token < embedding.vocab_end_index:
            torch.testing.assert_close(embedding.weight.grad[token - embedding.vocab_start_index], torch.ones(hidden_size, device=DEVICE))
    # The final padded ID must not contribute a gradient on its owning rank.
    if RANK == 1:
        torch.testing.assert_close(embedding.weight.grad[-1], torch.zeros(hidden_size, device=DEVICE))


if __name__ == "__main__":
    dist.init_process_group(
        backend="nccl" if DEVICE.type == "cuda" else "gloo",
        rank=RANK, world_size=WORLD_SIZE, timeout=datetime.timedelta(minutes=2),
    )
    try:
        test_vocab_padding()
        if RANK == 0:
            print("Vocab-padding test passed")
    finally:
        dist.destroy_process_group()
