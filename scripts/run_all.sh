#!/usr/bin/env bash
# Run a configured training experiment under multiple fixed seeds.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_all.sh <config.yaml> [seed ...]" >&2
  exit 2
fi

config="$1"
shift
seeds=("${@:-42 2026 3407}")

for seed in "${seeds[@]}"; do
  echo "==> Training ${config} with seed=${seed}"
  python scripts/train.py --config "${config}" --seed "${seed}"
done
