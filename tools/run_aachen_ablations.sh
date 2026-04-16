#!/usr/bin/env bash
# Run all Aachen day-time ablations sequentially.
# Usage: bash tools/run_aachen_ablations.sh [--device cpu]
#
# Ablation ladder:
#   A  mean_only       lambdas=[0, 1, 0, 0, 0]   cosine-to-mean only
#   B  mean_static     lambdas=[0, 1, 0.6, 0, 0] mean + staticness
#   C  plm_no_static   lambdas=[1, 1.2, 0, 0, 0] manifold + mean, no static
#   D  plm_full        lambdas=[1, 1.2, 0.6, 0.2, 0] full PLM

set -e
cd "$(dirname "$0")/.."

DEVICE="cuda"
if [[ "$1" == "--device" ]]; then DEVICE="$2"; fi

ABLATIONS=(
  "mean_only:configs/aachen_v1_1_day_mean_only.yaml:outputs/aachen_ablation_mean_only"
  "mean_static:configs/aachen_v1_1_day_mean_static.yaml:outputs/aachen_ablation_mean_static"
  "plm_no_static:configs/aachen_v1_1_day_plm_no_static.yaml:outputs/aachen_ablation_plm_no_static"
  "plm_full:configs/aachen_v1_1_day.yaml:outputs/aachen_ablation_plm_full"
)

for entry in "${ABLATIONS[@]}"; do
  IFS=':' read -r name cfg out_dir <<< "$entry"
  echo ""
  echo "========================================"
  echo "  Running ablation: $name"
  echo "========================================"

  # Override device in the config if needed (sed on a temp copy)
  tmp_cfg="/tmp/plm_ablation_${name}.yaml"
  sed "s/device:.*/device: ${DEVICE}/" "$cfg" > "$tmp_cfg"

  python -m plm_match.pipelines.hloc_localize \
    --config "$tmp_cfg" \
    --dataset_root /home/cvdp/phd/object_matching/datasets/aachen_v1_1 \
    --out_dir "$out_dir"

  echo "  -> done: $out_dir/metrics.json"
done

echo ""
echo "========================================"
echo "  All ablations done. Summary:"
echo "========================================"
for entry in "${ABLATIONS[@]}"; do
  IFS=':' read -r name cfg out_dir <<< "$entry"
  if [ -f "$out_dir/metrics.json" ]; then
    echo ""
    echo "--- $name ---"
    python -c "
import json, sys
m = json.load(open('$out_dir/metrics.json'))
keys = ['success_rate','med_rot_err_deg','med_trans_err_m','n_queries','n_success']
for k in keys:
    if k in m: print(f'  {k}: {m[k]}')
" 2>/dev/null || cat "$out_dir/metrics.json"
  fi
done
