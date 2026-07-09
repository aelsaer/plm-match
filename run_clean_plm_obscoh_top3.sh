#!/usr/bin/env bash
set -euo pipefail

echo "run_clean_plm_obscoh_top3.sh is deprecated."
echo "Running the clean observation-coherent top-1 experiment instead:"
echo "  bash run_clean_plm_obscoh_top1.sh"

exec bash run_clean_plm_obscoh_top1.sh
