#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# Complete perturbed contexts -> clean targets, one process per visible GPU.
# Set NPROC_PER_NODE to select the process count; --config overrides the default.
exec python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE:-gpu}" \
  --module lvebcsp.train.train_jepa \
  --config configs/crystal_jepa_ddp.yaml "$@"
