#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for Aachen LOO-1000 using the same best PLMLoc settings
# as scripts/run_aachen_loo500_plmloc_best.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOO_NUM_QUERIES="${LOO_NUM_QUERIES:-1000}" exec "$SCRIPT_DIR/run_aachen_loo500_plmloc_best.sh" "$@"
