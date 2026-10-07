#!/usr/bin/env bash
# Run Assignment 2's Part 3 tests against a Picotron checkout.
#
# Usage:
#   Local (two GPUs): FLASH_ATTEN=0 CUDA_VISIBLE_DEVICES=0,1 bash tests/run_part3_tests.sh
#   gpu-lease (one GPU per node): run this command on both node0 and node1.
#   One test: bash tests/run_part3_tests.sh test_vocab_padding.py
#
# The caller is responsible for activating the desired Python environment.
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# The Picotron Python package lives directly under this assignment repository.
# Override this only when testing a different Picotron checkout.
PICOTRON_ROOT=${PICOTRON_ROOT:-"$ROOT_DIR"}
if [[ ! -d "$PICOTRON_ROOT/picotron" ]]; then
    echo "Picotron source package not found at: $PICOTRON_ROOT/picotron" >&2
    echo "Set PICOTRON_ROOT to the Picotron checkout to test." >&2
    exit 2
fi
PICOTRON_ROOT=$(cd "$PICOTRON_ROOT" && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}

# The course GPU lease exports these variables for every login shell.  Source
# them explicitly too, which handles terminals opened before the node's
# cluster metadata was ready.  This is a no-op on local machines.
if [[ -f /etc/profile.d/gpulease-cluster.sh ]]; then
    # shellcheck source=/etc/profile.d/gpulease-cluster.sh
    source /etc/profile.d/gpulease-cluster.sh
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
if [[ -z ${NPROC_PER_NODE+x} ]]; then
    # The lease gives each of two nodes one T4; locally retain the original
    # two-process launch for a machine with two visible GPUs.
    NPROC_PER_NODE=$([[ "$NNODES" -gt 1 ]] && echo 1 || echo 2)
fi
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29600}

# Select the routed network interface on multi-node leases unless the caller
# deliberately supplied one.  g4dn.xlarge has no EFA, so use TCP sockets.
if [[ "$NNODES" -gt 1 && -z ${NCCL_SOCKET_IFNAME:-} ]]; then
    NCCL_SOCKET_IFNAME=$(ip route get 8.8.8.8 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit }}')
    export NCCL_SOCKET_IFNAME
fi
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if [[ "$WORLD_SIZE" -ne 2 ]]; then
    echo "Part 3 tests require exactly 2 ranks; got NNODES=$NNODES and NPROC_PER_NODE=$NPROC_PER_NODE." >&2
    exit 2
fi

export PYTHONPATH="${PICOTRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export FLASH_ATTEN=${FLASH_ATTEN:-0}

run_distributed_test() {
    local port=$1
    local test_file=$2
    echo "==> $test_file"
    "$TORCHRUN_BIN" \
        --nnodes="$NNODES" \
        --nproc-per-node="$NPROC_PER_NODE" \
        --node-rank="$NODE_RANK" \
        --master-addr="$MASTER_ADDR" \
        --master-port="$port" \
        "tests/$test_file"
}

declare -a TEST_FILES=(
    test_vocab_padding.py
    test_fused_qkv.py
    test_zigzag_context_parallel.py
    test_headwise_context_parallel.py
    test_sequence_parallel.py
)

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [test file]" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    requested_test=${1#tests/}
    found=false
    for test_file in "${TEST_FILES[@]}"; do
        if [[ "$test_file" == "$requested_test" ]]; then
            TEST_FILES=("$test_file")
            found=true
            break
        fi
    done
    if [[ "$found" != true ]]; then
        echo "Unknown Part 3 test: $1" >&2
        exit 2
    fi
fi

for test_index in "${!TEST_FILES[@]}"; do
    test_file=${TEST_FILES[$test_index]}
    case "$test_file" in
        test_vocab_padding.py) port=$MASTER_PORT_BASE ;;
        test_fused_qkv.py) port=$((MASTER_PORT_BASE + 1)) ;;
        test_zigzag_context_parallel.py) port=$((MASTER_PORT_BASE + 2)) ;;
        test_headwise_context_parallel.py) port=$((MASTER_PORT_BASE + 3)) ;;
        test_sequence_parallel.py) port=$((MASTER_PORT_BASE + 4)) ;;
    esac
    run_distributed_test "$port" "$test_file"
done

if [[ $# -eq 0 ]]; then
    echo "All Part 3 tests passed."
fi
