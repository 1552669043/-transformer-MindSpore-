#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-pet_fast_$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "outputs/${RUN_NAME}/run_logs"

CONFIGS=(
  configs/setr_tiny_pet_fast.json
  configs/segformer_tiny_pet_fast.json
  configs/segformer_edge_pet_fast.json
  configs/mask2former_tiny_pet_fast.json
  configs/oneformer_tiny_pet_fast.json
)

for cfg in "${CONFIGS[@]}"; do
  exp="$(python -c "import json; print(json.load(open('$cfg', encoding='utf-8'))['experiment_name'])")"
  out_dir="outputs/${RUN_NAME}/outputs/${exp}"
  log_file="outputs/${RUN_NAME}/run_logs/${exp}.log"
  echo "[$(date '+%F %T')] start ${exp}" | tee -a "$log_file"
  python train.py --config "$cfg" --device-target Ascend --output-dir "$out_dir" 2>&1 | tee -a "$log_file"
  echo "[$(date '+%F %T')] done ${exp}" | tee -a "$log_file"
done

echo "All pet experiments finished: outputs/${RUN_NAME}"
