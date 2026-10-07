# Assignment 2: Basic Profiling and Parallelism

## Preparation

**Get the skeleton code.** Create a private repository from the [template](https://github.com/utcs378/assignment2-template) (**Use this template > Create a new repository**, select **Private**).

**Lease your GPU nodes.** Clone your repository to your own machine, then run `resource_request/gpulease.py` from the repository root to bring up your group's nodes. For this assignment your group gets **two** AWS `g4dn.xlarge` nodes, each with one NVIDIA T4, on a private network where they can reach each other:

```bash
python3 resource_request/gpulease.py login <your-token>  # once per machine
python3 resource_request/gpulease.py start               # brings both nodes up
python3 resource_request/gpulease.py status              # ssh command per node, budget
```

Read `resource_request/README.md` before your first session: it covers the 80 GPU-hour budget and how to connect. Run `gpulease.py stop` as soon as you are done: the budget is counted **per node**, so with both nodes up it lasts 40 hours of wall-clock time.

Once you are connected, clone your repository onto **both** nodes. `stop`, the end of your lease, and the assignment deadline all destroy the nodes **and their disks**: every session starts from a fresh copy of the course image, so you will clone again each time. **Commit and push at the end of every session** or you will lose your work.

Get a HuggingFace token: go to the HuggingFace Access Tokens settings page to generate a token, used to download models from HuggingFace.
Set the token as an environment variable:

```bash
export HF_TOKEN=xxx
```

Replace `hf_xxxxxxxxxxxxxxxxxxxxx` with your actual Hugging Face token.
The export command only applies to the current terminal session. To avoid setting it every time, add it to your shell configuration file.

For Bash:

```bash
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

## Environment Preparation

The nodes boot from a course image that already has Miniforge3 and an `a2` conda environment with all dependencies installed (PyTorch 2.1.0, Transformers 4.47.0, Datasets 2.19.1, etc.). The `roneneldan/TinyStories` dataset and the SmolLM tokenizer are also pre-downloaded to `~/.cache/huggingface`.

At the start of each session, on **both** nodes, activate the environment and install your clone of the repository into it:

```bash
conda activate a2
cd <your-repository>
pip install -e . --no-deps
```

The `HF_TOKEN` step above is also lost on every `stop`, so repeat it too.

## Quick Test

### 1. Generate a config file

Outputs to the `tmp` directory in JSON format by default:

```bash
python create_config.py \
  --out_dir tmp \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 1 \
  --seq_len 128 \
  --hf_token $HF_TOKEN \
  --optimizer_type sgd
```

This creates `tmp/llama-1B/config.json`.

### 2. Run a single-GPU training test

```bash
torchrun --nproc_per_node 1 train.py --config tmp/llama-1B/config.json
```

## Multi-Node Training Guide

This guide explains how to run multi-node training.

**Prerequisite**: Complete Part 2: Parallelism Implementation before following this guide. The multi-node setup will only run successfully after the required basic parallelism implementation is complete.

**Before you start:** Make sure the code and `config.json` used for training are identical on both machines. Use Git or `rsync` to synchronize them, and repeat this step after making any changes.

**Your Cluster Info**

`gpulease.py` sets up the cluster for you. `status` lists your two nodes as `node0` (the master, `node_rank=0`) and `node1` (`node_rank=1`), with an ssh command for each. The ssh commands use `-A` to forward your ssh agent, so from either node you can `ssh node0` or `ssh node1` to hop to the other. This only works if your key is loaded in the agent on **your own machine** before you connect:

```bash
# on your own machine, once per terminal (the first line is needed on Linux/WSL if `ssh-add` says it cannot connect to an agent)
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/gpulease_<group>
```

On each node, `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, and that node's own `NODE_RANK` are exported in every login shell, and the peers are listed in `/etc/gpulease/peers` and in `/etc/hosts` as `node0` and `node1`. These are filled in a few minutes after the nodes boot. If you log in before then, you will see `gpulease: cluster info (NODE_RANK, MASTER_ADDR, ...) is not available yet`. Wait a minute and log in again. Check them on each node:

```bash
echo $NODE_RANK $NNODES $MASTER_ADDR $MASTER_PORT   # node0: 0 2 <node0 IP> 29500, node1: 1 2 <node0 IP> 29500
cat /etc/gpulease/peers
```

If they are empty in a terminal that was opened before they were ready (including an editor's built-in terminal), load them into that terminal with:

```bash
source /etc/profile.d/gpulease-cluster.sh
```

NCCL also needs the name of the network interface to use. Run the following on EACH node:

```bash
ip route get 8.8.8.8
```

Example output:

```
8.8.8.8 via 172.31.0.1 dev ens5 src 172.31.5.20 uid 1000
    cache
```

- `dev ens5` — the network interface name used for that route. Use this value for `NCCL_SOCKET_IFNAME` on that specific machine.
- `src 172.31.5.20` — this machine's private IP address.

Verify the nodes can reach each other by name:

```bash
# From node1:
ping -c 3 node0

# From node0:
ping -c 3 node1
```

If this fails, run `python3 resource_request/gpulease.py status` from your own machine to check both nodes are up, and ask on the course forum if they are — `torchrun` will hang or fail otherwise.

### 1. Generate a config file

```bash
python create_config.py \
  --out_dir DP \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 2 \
  --seq_len 64 \
  --hf_token $HF_TOKEN \
  --dp_engine "bucket" \
  --dp 2
```

This creates `DP/llama-1B/config.json`.

### 2. Launch Training on Both Nodes

Because the cluster variables are already exported, run the same command on both nodes, replacing `ens5` with the interface name you found above:

```bash
NCCL_SOCKET_IFNAME=ens5 NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=1 --nnodes=$NNODES --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  train.py --config DP/llama-1B/config.json
```

`--nnodes=$NNODES`
- Total number of machines participating in training (2).

`--node_rank=$NODE_RANK`
- Unique index for each machine: 0 for the master (node0), 1 for the second node (node1).

`--nproc_per_node=1`
- Number of GPUs (processes) to launch on this machine — 1, since each node has a single GPU.

`--master_addr=$MASTER_ADDR`
- The IP address of the master node (node0). Must be the SAME value on both nodes.

`--master_port=$MASTER_PORT`
- Port used for the rendezvous handshake between nodes. Must match on both nodes and be free (see Troubleshooting).

`NCCL_SOCKET_IFNAME`
- The network interface NCCL should use for inter-node communication. Set to each node's own interface name (may differ per machine).

`NCCL_IB_DISABLE=1`
- Disables InfiniBand support in NCCL. Include this if your machines don't have InfiniBand hardware, to avoid NCCL trying to use a nonexistent IB device.

For more about `torchrun`: please refer to [https://docs.pytorch.org/docs/2.10/elastic/run.html](https://docs.pytorch.org/docs/2.10/elastic/run.html)

### Troubleshooting

**`torchrun: error: argument --node-rank/--node_rank: invalid int value: ''`**

- `NODE_RANK` is not set in this shell. Run `source /etc/profile.d/gpulease-cluster.sh`. If that prints `not available yet`, wait a minute and try again. See [Your Cluster Info](#multi-node-training-guide).

**`torchrun` prints nothing and seems stuck**

- node0 waits until every node has started `torchrun`. Start the same command on node1, too.

**`ssh node1` from node0 says `Permission denied (publickey)`**

- Your key is not in your local ssh agent, or you connected without `-A`. On your own machine, run `eval "$(ssh-agent -s)"` and `ssh-add ~/.ssh/gpulease_<group>`, then reconnect with `ssh -A`.

**NCCL prints `NET/OFI ... aws-ofi-nccl initialization failed` warnings**

- Expected. That plugin is for EFA networking, which these instances do not have. NCCL falls back to TCP sockets (`Using network Socket`), which is what you want.

**Address already in use (`$MASTER_PORT`)**

- Another process is already using that port. Either kill it (find it with `lsof -i :$MASTER_PORT`) or pass a different `--master_port` on BOTH nodes, e.g. `--master_port=29501`.

**config.json differs between nodes**

- Both nodes must use an identical config file. If not using shared storage, re-copy the file after any change to `create_config.py`'s arguments.

## Part 1: Baseline & Memory Profile

In this part, you will run training on a single GPU without any distributed parallelism and analyze the memory footprint of a training step.

Set all parallelism degrees to 1: `DP = TP = PP = CP = 1`

**Note on precision:** the generated config says `"dtype": "bfloat16"`, but the T4 GPUs on these nodes do not support bfloat16, so `train.py` falls back to float32 (see the `dtype` line in `train.py`). Parameters, gradients, and optimizer states are therefore stored in fp32, at 4 bytes per value. Use `b = 4` when you compare your measurements against your derivations.

### 1.1 Profile Training Memory

**Generate a config file**

```bash
python create_config.py \
  --out_dir part1 \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 1 \
  --seq_len 128 \
  --hf_token "$HF_TOKEN" \
  --optimizer_type adamw
```

**Run a single-GPU training test**

```bash
torchrun --nproc_per_node 1 train.py --config part1/llama-1B/config.json
```

Instrument the training script to report the following memory components:

* Number of model parameters
* Parameter memory
* Gradient memory
* Optimizer-state memory
* Activation memory
* Peak GPU memory

Answer:

1. Which component consumes the most GPU memory?
2. Why does the measured peak GPU memory differ from the sum of parameters, gradients, optimizer states, and estimated activations?
3. At which point during a training step does GPU memory usage reach its peak?

### 1.2 Optimizer-State Memory

Measure and compare memory usage with three optimizers: SGD, SGD with momentum, and AdamW. Keep all other settings the same so the comparison reflects the optimizer choice.

Create a configuration for SGD:

```bash
python create_config.py \
  --out_dir part1 \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 1 \
  --seq_len 128 \
  --hf_token "$HF_TOKEN" \
  --optimizer_type sgd
```

Repeat the test with SGD with momentum and AdamW by changing `--optimizer_type` to the appropriate value.

Keep the model, micro-batch size, and sequence length unchanged.
Report:

| Optimizer  | Optimizer-State Memory  | Peak GPU Memory  |
| :---- | :---- | :---- |
| SGD  |  |  |
| SGD + Momentum  |  |  |
| AdamW  |  |  |

Answer:

1. Which optimizer has the largest memory overhead? Derive the optimizer-state memory for each optimizer in terms of the number of parameters `N` and bytes per value `b`.
2. Does the difference in optimizer-state memory fully explain the observed difference in peak GPU memory? Why or why not?

### 1.3 Activation Memory Scaling

Study how activation memory changes with batch size and sequence length.

Config:

```bash
python create_config.py \
  --out_dir part1 \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 1 \
  --seq_len 128 \
  --hf_token "$HF_TOKEN" \
  --optimizer_type sgd
```

**Experiment A: Micro-batch Size**

Keep the sequence length fixed at 128 and run:

`micro-batch size = 1, 2, 4`

Report the activation memory and peak GPU memory for each configuration.

**Experiment B: Sequence Length**

Keep the micro-batch size fixed at 1 and run:

`sequence length = 128, 256, 512`

Report the activation memory and peak GPU memory for each configuration.

Plot:

* Activation memory vs. micro-batch size
* Activation memory vs. sequence length

Answer:

1. How does activation memory scale with micro-batch size?
2. How does activation memory scale with sequence length?

## Part 2: Basic Parallelism Implementation

In this part, you will complete the missing operations required to make the parallelism implementations work correctly.

You are given files in which specific lines have been removed and replaced with **`[Part 2] TODO`** blocks. Everything surrounding each blank—including variable initialization, docstrings, control flow, and edge-case handling—has already been implemented correctly and should not be modified.

Your task is to fill in only the missing operation(s) between:

```python
# [Part 2] TODO: ...

# END OF YOUR CODE
```

After completing each implementation, verify its correctness using **2 NVIDIA T4 GPUs**.

For instructions on running across multiple nodes, see the [**Multi-Node Training Guide**](#multi-node-training-guide).

### 2.1 Data Parallelism

Complete the `sync_gradient` function in:

`picotron/data_parallel/bucket.py`

#### 2.1.1 Inspect the DP Implementation

Identify the two variants implemented: `DataParallelNaive` and `DataParallelBucket`.

Answer:

* At which point in the training step does gradient synchronization happen?
* Which collective communication operation is used, and what does it compute?
* How does gradient accumulation interact with gradient synchronization?

#### 2.1.2 Data-Parallel Performance and Communication Scaling

Run `DP = 2`, using both `DataParallelNaive` and `DataParallelBucket` at each DP degree

Config:

```bash
python create_config.py \
  --out_dir DP \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 1 \
  --seq_len 128 \
  --hf_token "$HF_TOKEN" \
  --optimizer_type sgd \
  --dp 2 \
  --dp_engine bucket
```

Set `dp_engine` for different values: `naive` / `bucket`

Measure and report the following metrics:

* Average training step time
* Throughput (tokens/second)
* Throughput per GPU (tokens/second/GPU)
* Peak GPU memory per GPU
* Parameter memory per GPU
* Data-parallel communication time

| Metric  | DataParallelNaive | DataParallelBucket |
| :---- | :---- | :---- |
| Average training step time  |   |   |
| Throughput (tokens/second)  |  |  |
| Throughput per GPU (tokens/second/GPU)  |  |  |
| Peak GPU memory per GPU  |  |  |
| Parameter memory per GPU  |  |  |
| Data-parallel communication time  |  |  |

### 2.2 Tensor Parallelism

Complete the missing communication operations in:

`picotron/tensor_parallel/tp_communications.py`

Then run the workload with `TP = 2`.

Config:

```bash
python create_config.py \
  --out_dir TP \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 1 \
  --seq_len 128 \
  --hf_token "$HF_TOKEN" \
  --optimizer_type sgd \
  --tp 2
```

#### 2.2.1 Understand the Tensor-Parallel Partitioning

Inspect tensor-parallel implementation and complete the following table.

For each layer, identify:

1. the tensor-parallel layer type; and
2. the dimension along which the corresponding weight or embedding tensor is partitioned across GPUs.

| Layer | TP Type | Partitioned Dimension |
| :---- | :---- | :---- |
| `q_proj` |   |   |
| `k_proj` |   |   |
| `v_proj` |   |   |
| `out_proj` |   |   |
| `up_proj` |   |   |
| `gate_proj` |   |   |
| `down_proj` |   |   |
| token embedding |   |   |
| `final_proj` |   |   |

Answer: Why some linear layers use **column parallelism** while others use **row parallelism**.

#### 2.2.2 Trace Tensor-Parallel Communication

Trace the forward and backward execution of the following tensor-parallel components:

1. `ColumnParallelLinear`
2. `RowParallelLinear`
3. `VocabParallelEmbedding`
4. the final output projection

For each component, identify:

* how the input, weight, and output tensors are partitioned;
* whether communication occurs during the forward pass;
* whether communication occurs during the backward pass; and
* the communication collective used, such as `all_reduce`, `all_gather`, or `reduce_scatter`.

|  | Forward Communication | Backward Communication |
| :---- | :---- | :---- |
| Layer |  |  |
| `ColumnParallelLinear`  |  |  |
| `RowParallelLinear`  |  |  |
| `VocabParallelEmbedding`  |  |  |
| Final output projection |  |  |

#### 2.2.3 Tensor-Parallel Performance

Run the training with `TP = 2`.

Measure and report the following metrics:

* Average training step time
* Throughput (tokens/second)
* Throughput per GPU (tokens/second/GPU)
* Peak GPU memory per GPU
* Parameter memory per GPU
* Tensor-parallel communication time

| Average training step time  |  |
| :---- | :---- |
| Throughput (tokens/second)  |  |
| Throughput per GPU (tokens/second/GPU)  |  |
| Peak GPU memory per GPU  |  |
| Parameter memory per GPU  |  |
| Data-parallel communication time  |  |

Then, based on your measurements, analyze the benefits and costs of tensor parallelism.

### 2.3 Context Parallelism

Complete the missing communication operations in:

`picotron/context_parallel/cp_communications.py`

Then run the workload with `CP = 2`.

Config:

```bash
python create_config.py \
  --out_dir CP \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 1 \
  --seq_len 128 \
  --hf_token "$HF_TOKEN" \
  --optimizer_type sgd \
  --cp 2
```

#### 2.3.1 Inspect Context Partitioning

Inspect:

* `picotron/context_parallel/context_parallel.py`
* `picotron/context_parallel/cp_communications.py`

Answer the following questions:

1. Along which tensor dimension is the input sequence partitioned across CP ranks?
2. Which tokens are assigned to each CP rank?
3. Why must the global sequence length be divisible by the context-parallel degree?
4. Under causal attention, is the attention workload evenly balanced across CP ranks? Explain why or why not.

#### 2.3.2 Trace Ring Attention

Trace one complete forward pass of the ring attention implementation.

Answer:

1. Does Q move between GPUs?
2. Which tensors are communicated around the ring?
3. How many ring steps are required for `CP = p`?
4. Why is it insufficient for a GPU to compute attention using only its local K and V?
5. Which point-to-point communication primitives are used by Picotron?

#### 2.3.3 Context-Parallel Memory and Performance

Run the training with `CP = 2`.

Measure and report the following metrics:

* Average training step time
* Throughput (tokens/second)
* Throughput per GPU (tokens/second/GPU)
* Peak GPU memory per GPU
* Parameter memory per GPU
* Context-parallel communication time

| Average training step time  |  |
| :---- | :---- |
| Throughput (tokens/second)  |  |
| Throughput per GPU (tokens/second/GPU)  |  |
| Peak GPU memory per GPU  |  |
| Parameter memory per GPU  |  |
| Context-parallel communication time  |  |

Then, based on your measurements, analyze the benefits and costs of context parallelism.

### 2.4 Pipeline Parallelism

Complete the missing implementations in:

* `picotron/pipeline_parallel/pp_communications.py` — `pipeline_communicate()`
* `picotron/pipeline_parallel/pipeline_parallel.py` — `distribute_layers()`

Config:

```bash
python create_config.py \
  --out_dir PP \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 8 \
  --mbs 1 \
  --seq_len 128 \
  --hf_token "$HF_TOKEN" \
  --optimizer_type sgd \
  --pp 2
```

After completing the implementation, verify correctness by running training.

#### 2.4.1 Inspect Pipeline Stage Placement

Inspect:

* `picotron/pipeline_parallel/pipeline_parallel.py`
* `picotron/pipeline_parallel/pp_communications.py`

Answer the following questions:

1. How are Transformer layers distributed across pipeline stages?
2. How does Picotron distribute layers when the number of Transformer layers is not divisible by the pipeline-parallel degree?
3. Which pipeline stage contains the token embedding layer?
4. Which pipeline stage contains the final normalization and output projection?
5. Are all pipeline stages guaranteed to contain the same number of parameters? Explain why or why not.

#### 2.4.3 Pipeline-Parallel Memory and Performance

Run the training with `PP = 2`.

Evaluate both pipeline schedules:

* `--pp_engine afab`
* `--pp_engine 1f1b`

Config example:

```bash
python create_config.py \
  --out_dir PP \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 8 \
  --mbs 1 \
  --seq_len 128 \
  --hf_token "$HF_TOKEN" \
  --optimizer_type sgd \
  --pp 2 \
  --pp_engine afab
```

Measure and report the following metrics:

* Average training step time
* Throughput (tokens/second)
* Throughput per GPU (tokens/second/GPU)
* Peak GPU memory per GPU
* Parameter memory per GPU
* Pipeline-parallel communication time

Report the results separately for `AFAB` and `1F1B`.

| Metric  | AFAB  | 1F1B  |
| :---- | :---- | :---- |
| Average training step time  |   |   |
| Throughput (tokens/second)  |  |  |
| Throughput per GPU (tokens/second/GPU)  |  |  |
| Peak GPU memory per GPU  |  |  |
| Parameter memory per GPU  |  |  |
| Pipeline-parallel communication time   |  |  |

Then, based on your measurements, analyze the **benefits and costs of pipeline parallelism** and compare the performance and memory behavior of the `AFAB` and `1F1B` schedules.

## Part 3: Feature Implementation

### 3.1 Implementation: Vocab Padding for VocabParallelEmbedding

The original `VocabParallelEmbedding` assumes that the vocabulary size is evenly divisible by the tensor-parallel (TP) degree. In practice, model vocabulary sizes do not always satisfy this requirement.

Your task is to extend vocabulary parallelism to support arbitrary vocabulary sizes by padding the vocabulary before partitioning it across TP ranks.

For a vocabulary of size `V` and TP degree `N`, compute the smallest padded vocabulary size `V_pad >= V` such that:

```
V_pad % N == 0
```

Then divide the padded vocabulary evenly across TP ranks. Each rank should own a contiguous vocabulary range of equal size.

For example, if:

```
vocab_size = 10
tp_size    = 4
```

the vocabulary should be padded to:

```
padded_vocab_size = 12
```

and partitioned as:

```
Rank 0: [0, 3)
Rank 1: [3, 6)
Rank 2: [6, 9)
Rank 3: [9, 12)
```

Note that token IDs `10` and `11` are padding entries and do not correspond to real vocabulary tokens.

**Test config:**

```bash
python create_config.py \
  --out_dir TP \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 2 \
  --seq_len 128 \
  --hf_token "$HF_TOKEN" \
  --optimizer_type sgd \
  --tp 2 \
  --vocab_padding_en
```

### 3.2 Implementation: Fusing Q/K/V Column-Parallel Communication

In the original attention implementation, Q, K, and V are computed using three separate linear projections:

```
Q = X Wq
K = X Wk
V = X Wv
```

This requires three separate matrix multiplications. In this task, you will fuse the three projections into a single linear operation:

```
[Q | K | V] = X Wqkv
```

and then split the fused output back into Q, K, and V before computing attention.

**Step 1: Implement `FuseQKVAttention`**

First, implement the `FuseQKVAttention` class and use it to replace the original `Attention` class when `fuse_qkv_en=True`.

You can first test this implementation on a **single GPU** with Tensor Parallelism **disabled**. The fused implementation should produce the same output shape and training behavior as the original attention implementation.

**Step 2: Support Tensor Parallelism with `FusedQKVColumnParallelLinear`**

Next, extend fused QKV projection to work with Tensor Parallelism.
A normal `ColumnParallelLinear` partitions its output dimension into equal contiguous chunks. However, directly applying this strategy to a fused tensor `[Q | K | V]` can assign a rank only part of Q, K, or V rather than a valid set of complete attention heads.
Instead, implement **`FusedQKVColumnParallelLinear`** so that each TP rank receives its own local Q, K, and V head partitions:

```
Rank i: [Q_i | K_i | V_i]
```

**Test config:**

```bash
python create_config.py \
  --out_dir TP \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 2 \
  --seq_len 128 \
  --hf_token "$HF_TOKEN" \
  --optimizer_type sgd \
  --tp 2 \
  --fuse_qkv_en
```

### 3.3 Implementation: Supporting Non-Divisible Sequence Lengths

Picotron currently requires the global sequence length to be divisible by the Context Parallelism (CP) degree:

```
sequence_length % CP == 0
```

This allows each CP rank to receive an equal-sized sequence partition and simplifies the communication pattern used by Ring Attention.

In practice, sequence lengths are not always divisible by the CP degree. Your task is to extend Picotron so that Context Parallelism can support arbitrary sequence lengths by padding the sequence before the dataloader prepares each batch.

When `cp_seq_padding_en` is enabled, adjust the configured sequence length to the smallest value greater than or equal to the original sequence length that is divisible by the CP world size.

For example, if:

```
sequence_length = 10
cp_size = 4
```

the sequence length should be padded to:

```
padded_sequence_length = 12
```

so that each CP rank receives an equal-sized partition.

**Test config:**

```bash
python create_config.py \
  --out_dir CP \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 2 \
  --seq_len 129 \
  --hf_token $HF_TOKEN \
  --cp 2 \
  --cp_seq_padding_en
```

### 3.4 Implementation: Load-Balanced Ring Attention for Causal Attention (Zigzag)

Plain Ring Attention partitions the input sequence into contiguous, equal-sized chunks across N ranks. For example, with 4 GPUs:

```
b0 | b1 | b2 | b3
```

Under causal attention, this partitioning leads to significant workload imbalance. Queries in earlier blocks can attend to only a small portion of the sequence, while queries in later blocks can attend to many more preceding tokens. As a result, later ranks perform substantially more valid attention computation than earlier ranks.

Your task is to implement a load-balanced sequence partitioning scheme for causal Ring Attention.

Instead of splitting the sequence into N blocks, split it into 2N equal-sized blocks and assign one block from the beginning and one symmetric block from the end to each rank.

For example, with 4 GPUs:

```
b0, b7 | b1, b6 | b2, b5 | b3, b4
```

This pairing combines low-workload early blocks with high-workload later blocks, making the causal attention workload approximately balanced across all ranks.

Your implementation should preserve the original token order and causal attention semantics while correctly handling the non-contiguous blocks assigned to each rank.

**Test config:**

```bash
python create_config.py \
  --out_dir tmp \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 2 \
  --seq_len 128 \
  --hf_token $HF_TOKEN \
  --cp 2 \
  --cp_zigzag_en
```

### 3.5 Headwise Context Parallelism

Headwise Context Parallelism is based on the context parallel attention method introduced by DeepSpeed Ulysses. ([https://arxiv.org/abs/2309.14509](https://arxiv.org/abs/2309.14509))

Instead of performing attention while keeping the sequence dimension partitioned across GPUs, Headwise Context Parallelism **redistributes the tensors so that attention heads are partitioned instead**.

Suppose the context-parallel group contains N ranks, the sequence length is S, and there are H attention heads. Initially, the sequence dimension is partitioned across ranks:

```
[B, H, S/N, D] (Batch, Head, Sequence/N, hidden_Dim)
```

Each rank therefore holds all H heads, but only S/N tokens. Before computing attention, an **all-to-all** redistributes the tensor from the sequence dimension to the head dimension:

```
[B, H, S/N, D] -- AlltoAll --> [B, H/N, S, D]
```

Now, each rank holds only H/N heads, but has the **complete sequence** for those heads. Since attention heads can be computed independently, each rank can compute standard full-sequence attention for its assigned heads.

After attention, another all-to-all performs the inverse transformation:

```
[B, H/N, S, D] -- AlltoAll --> [B, H, S/N, D]
```

Thus, Headwise Context Parallelism can be summarized as:

Headwise Context Parallelism therefore consists of three steps:

1. **Redistribute S->H:** redistribute Q, K, and V from sequence-partitioned to head-partitioned.
2. **Attention:** compute full-sequence attention independently for the assigned heads.
3. **Redistribute H->S:** redistribute the attention output back to the original sequence partition.

In this assignment, you will implement these two layout transformations and use them to perform Headwise Context Parallel Attention.

**Step 1: Implement `sequence_to_head` and `head_to_sequence` in `context_parallel.py`**

An `all_to_all` function is provided in `cp_communications.py`. Use this function to implement both the **`sequence_to_head`** and **`head_to_sequence`** redistributions. Specific directions are given in the TODO blocks in those functions.

**Step 2: Implement the `apply` function in class `HeadwiseContextParallel` in `context_parallel.py`**

Use FlashAttention to compute attention after Q, K, and V have been transformed into the head-partitioned layout. Specific directions are given in the TODO blocks in those functions.

**Test config:**

```bash
python create_config.py \
  --out_dir tmp \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 2 \
  --seq_len 128 \
  --hf_token $HF_TOKEN \
  --cp 2 \
  --cp_mode headwise
```

### 3.6 Sequence Parallelism -> extra credit

Sequence Parallelism is an extension of Tensor Parallelism introduced in *Reducing Activation Recomputation in Large Transformer Models*. ([https://arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198))

A Transformer layer consists primarily of the **attention** and **MLP** sublayers, together with **normalization and residual operations** around them. In Tensor Parallelism, the large linear operations in attention and the MLP are partitioned across the tensor-parallel ranks. However, normalization and residual operations are not tensor-parallelized.

In standard Tensor Parallelism, the outputs of the tensor-parallel attention and MLP are **AllReduced**, so that every rank obtains the same complete activation. The following normalization and residual operations are then performed redundantly on the same activation on every rank. This results in both **duplicated computation and duplicated activation memory**.

Sequence Parallelism removes this duplication by partitioning these intermediate activations along the sequence dimension. Suppose the tensor-parallel group contains N ranks, the sequence length is S, and the hidden dimension is D. Instead of every rank storing and computing on the complete activation, Sequence Parallelism partitions the activation across the N tensor-parallel ranks through a ReduceScatter operation:

```
[B, S, D] -- ReduceScatter --> [B, S/N, D]
```

Each rank now stores and processes only S/N tokens. Since operations such as normalization and residual addition can be computed independently for each token, each rank can perform these operations directly on its local sequence shard.

However, the next tensor-parallel layer again requires the complete sequence on each rank. Before entering the next tensor-parallel computation, the sequence shards are therefore reconstructed using an **AllGather**:

```
[B, S/N, D] -- AllGather --> [B, S, D]
```

Sequence Parallelism therefore consists of three main actions:

1. **Gather sequence shards:** before a column-parallel linear layer, gather the full sequence using AllGather.
2. **Tensor-parallel computation:** perform the usual column-parallel and row-parallel linear computations.
3. **Reduce and repartition:** after a row-parallel linear layer, use ReduceScatter to sum the partial outputs and restore the sequence partition.

In this assignment, you will add Sequence Parallelism to the existing Tensor Parallelism implementation. The essential communication functions are provided in `tp_communications.py`.

##### Step 1: Modify `ColumnParallelLinear.forward` in `tensor_parallel.py`

When Sequence Parallelism is enabled, use `GatherFromSequenceParallelRegion` to gather the sequence shards before applying the local column-parallel linear transformation.

##### Step 2: Modify `RowParallelLinear.forward` in `tensor_parallel.py`

When Sequence Parallelism is enabled, apply the local row-parallel linear transformation and use `ReduceScatterToSequenceParallelRegion` instead of `ReduceFromModelParallelRegion`.

##### Step 3: Modify `Llama.forward` in `model.py`

The vocabulary-parallel embedding initially produces the complete sequence activation of shape `[B, S, D]`. When Sequence Parallelism is enabled, use `ScatterToSequenceParallelRegion` immediately after the embedding layer:

```
[B, S, D] -- Scatter --> [B, S/N, D]
```

The decoder layers will then keep their normalization and residual activations partitioned along the sequence dimension. Their column- and row-parallel linear layers perform the required AllGather and ReduceScatter operations.

**Test config:**

```bash
python create_config.py \
  --out_dir tmp \
  --exp_name llama-1B \
  --model_name HuggingFaceTB/SmolLM-1.7B \
  --num_hidden_layers 8 \
  --grad_acc_steps 1 \
  --mbs 2 \
  --seq_len 128 \
  --hf_token $HF_TOKEN \
  --tp 2 \
  --sequence_parallel
```

### Unit tests

We added distributed unit tests for the applicable Part 3 features: vocabulary
padding, fused QKV, zigzag attention, headwise context parallelism, and
sequence parallelism.  Non-divisible sequence lengths are excluded.

On the two-node GPU lease, run the same command on both nodes from the
`assignment2` repository:

```bash
FLASH_ATTEN=0 bash tests/run_part3_tests.sh
```

To run one test, use its dedicated launcher on both nodes, for example:

```bash
FLASH_ATTEN=0 bash tests/run_fused_qkv_test.sh
```

## Submission

Submit the following:

* **Your source files** — pack your completed codebase into `submission.tar.gz`:

  ```bash
  tar czf submission.tar.gz <your completed source files>
  ```

* **A report (PDF)** containing:
  * Your group name, and the name and EID of every group member.
  * A detailed account of your approach, with an explanation of your implementation for each part.
  * Your answers to every question in this document. Keep each original question and add your answer directly below it, following the same order as this guide.

Package `submission.tar.gz` and the report together into the final archive:

```bash
tar czf assignment2_{Group_Name}.tar.gz submission.tar.gz assignment2_report.pdf
```

Submission is to be done through Canvas. Only one person per group is required to submit.


## Grading

Total: 100% (+10% extra credit)

**Part 1 — Baseline & Memory Profile: 20%**
* 1.1 Profile Training Memory (memory instrumentation + answers): 8%
* 1.2 Optimizer-State Memory (SGD / SGD+Momentum / AdamW comparison + answers): 6%
* 1.3 Activation Memory Scaling (micro-batch and sequence-length sweeps, plots + answers): 6%

**Part 2 — Basic Parallelism Implementation: 45%**
* 2.1 Data Parallelism ( correctness, DP=2 naive vs. bucket benchmarks + answers): 10%
* 2.2 Tensor Parallelism ( correctness, TP=2 benchmarks + answers): 12%
* 2.3 Context Parallelism (correctness, Ring Attention trace, CP=2 benchmarks + answers): 12%
* 2.4 Pipeline Parallelism (correctness, AFAB vs. 1F1B benchmarks + answers): 11%

**Part 3 — Feature Implementation: 35%**
* 3.1 Vocab Padding for VocabParallelEmbedding: 6%
* 3.2 Fused QKV Column-Parallel Communication: 8%
* 3.3 Non-Divisible Sequence Lengths (CP padding): 6%
* 3.4 Load-Balanced Ring Attention (Zigzag): 7%
* 3.5 Headwise Context Parallelism: 8%
* 3.6 Sequence Parallelism — extra credit: +10%


# Acknowledgments
This assignment builds on https://github.com/huggingface/picotron
