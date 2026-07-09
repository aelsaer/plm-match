# LOO-Aachen Experiment Status

Comparable split: `outputs/loo_aachen_benchmark/loo_aachen_500`

All rows below use the same LOO-Aachen-500 split unless marked otherwise.
Runtime is `mean_query_time_s` from each `metrics.json`; for HLoc this is the
cached localization-time row, not the original feature extraction + matching
wall time.

## Main LOO-Aachen-500 Table

| Method | N | Pose success | 0.25m/2deg | 0.5m/5deg | 5m/10deg | Median trans | Median rot | Runtime/q | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HLoc SP+SG | 500 | 100.0 | 96.4 | 99.2 | 99.6 | 0.034m | 0.09deg | 0.23s | `hloc_sp_sg/metrics.json` |
| PLM + SG + SimpleGraph | 500 | 98.4 | 73.6 | 83.6 | 96.0 | 0.099m | 0.30deg | 2.29s | `plm_sg_verifier_simple_graph/metrics.json` |
| PLM-SPMemory-PPCA + SG + SimpleGraph | 500 | 99.2 | 69.4 | 84.2 | 96.8 | 0.115m | 0.33deg | 2.61s | `plm_sp_ppca_sg_simplegraph/metrics.json` |
| PLM + SG + CorrGraph | 500 | 99.0 | 62.8 | 76.6 | 94.8 | 0.149m | 0.50deg | 6.20s | `plm_sg_verifier_corr_graph/metrics.json` |
| PLM-SPMemory-PPCA r4 obs8 | 500 | 93.6 | 51.8 | 63.6 | 78.6 | 0.209m | 0.62deg | 2.00s | `plm_local_memory_sp_ppca_rank4_obs8/metrics.json` |
| PLM-LocalMemory EUPE-1024 aspect | 500 | 95.2 | 36.0 | 49.2 | 69.0 | 0.453m | 1.27deg | 1.92s | `plm_local_memory_eupe_1024_aspect/metrics.json` |
| PLM-LocalMemory EUPE+SP | 500 | 95.8 | 35.4 | 50.4 | 69.4 | 0.433m | 1.36deg | 1.68s | `plm_local_memory/metrics.json` |
| PLM-LocalMemory EUPE-256 square | 500 | 93.2 | 35.4 | 48.8 | 69.6 | 0.449m | 1.30deg | 1.81s | `plm_local_memory_eupe_256_square/metrics.json` |
| PLM-LocalMemory DINOv3-1024 aspect | 500 | 94.6 | 33.6 | 49.8 | 68.0 | 0.447m | 1.28deg | 1.95s | `plm_local_memory_dinov3_1024_aspect/metrics.json` |
| PLM-SPMemory-PPCA + SimpleGraph | 500 | 92.0 | 33.4 | 53.8 | 77.0 | 0.369m | 1.18deg | 1.92s | `plm_local_memory_sp_ppca_simplegraph/metrics.json` |
| PLM-LocalMemory DINOv3-512 | 500 | 94.4 | 32.6 | 44.8 | 66.2 | 0.562m | 1.83deg | 1.93s | `plm_local_memory_dinov3_512/metrics.json` |
| PLM-LocalMemory SP only | 500 | 91.2 | 30.2 | 44.8 | 63.0 | 0.533m | 1.53deg | 1.93s | `plm_local_memory_no_eupe/metrics.json` |
| PLM-LocalMemory + SimpleGraph | 500 | 91.4 | 29.8 | 43.8 | 68.2 | 0.574m | 1.62deg | 1.64s | `plm_local_memory_graph/metrics.json` |
| PLM-LocalMemory + CorrGraph | 500 | 94.6 | 27.2 | 41.8 | 69.0 | 0.659m | 1.61deg | 5.67s | `plm_local_memory_corr_graph/metrics.json` |
| PLM-Memory-Graph | 500 | 97.2 | 1.6 | 4.6 | 43.2 | 3.840m | 10.63deg | 1.37s | `plm_memory_graph/metrics.json` |
| PLM-Memory GFTT+EUPE+Graph | 500 | 97.8 | 1.4 | 5.6 | 42.8 | 4.056m | 11.83deg | 1.17s | `plm_memory/metrics.json` |
| PLM-PPCA EUPE-1024 r4 obs8 | 500 | 91.8 | 0.8 | 7.6 | 52.4 | 2.323m | 7.35deg | 1.08s | `plm_ppca_eupe1024_rank4_obs8/metrics.json` |
| PLM-Direct EUPE | 500 | 86.2 | 0.2 | 2.6 | 40.4 | 3.762m | 9.22deg | 0.81s | `plm_direct/metrics.json` |

## Chat-Only Or Deleted Rows

These were pasted in the chat, but the corresponding `metrics.json` is not
currently present in the LOO-500 folder.

| Method | N | Pose success | 0.25m/2deg | 0.5m/5deg | 5m/10deg | Median trans | Median rot | Runtime/q | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PLM + SG verifier, no graph | 500 | 100.0 | 52.8 | 58.6 | 92.6 | 0.148m | 0.46deg | 3.50s | pasted earlier; folder not currently present |

## Older LOO-Aachen-50 Pilots

These are useful for history, but should not be compared directly to LOO-500.

| Method / output | N | Pose success | Median trans | Median rot | Runtime/q |
|---|---:|---:|---:|---:|---:|
| `aachen_day_refactor_loo_50` | 50 | 88.0 | 1.741m | 5.64deg | 28.26s |
| `aachen_day_refactor_loo_gftt_50` | 50 | 100.0 | 0.741m | 2.23deg | 42.63s |
| `aachen_day_refactor_loo_gftt_xfeat_primary_50` | 50 | 92.0 | 0.725m | 2.07deg | 59.53s |
| `aachen_day_refactor_loo_superpoint_50_v2` first pasted run | 50 | 96.0 | 0.301m | 0.93deg | 3.42s |
| `aachen_day_refactor_loo_superpoint_50_v2` later pasted run | 50 | 98.0 | 0.225m | 0.57deg | 3.86s |

## Non-Comparable Aachen Query Runs

These are the old Table-I style Aachen query runs. They report PnP pose output
success on the real Aachen query set, not true pose accuracy thresholds.

| Method | N | Pose output success | Runtime/q | Source |
|---|---:|---:|---:|---|
| PLM residual | 824 | 89.3 | 10.47s | `outputs/table_i/plm_residual/aachen/metrics.json` |
| PLM mean | 824 | 50.8 | 0.92s | `outputs/table_i/plm_mean/aachen/metrics.json` |
| Frozen backbone flat | 824 | 48.3 | 0.85s | `outputs/table_i/frozen_backbone_flat/aachen/metrics.json` |
| PLM full refactor | 824 | 27.8 | 15.94s | `outputs/table_i/plm_full_refactor/aachen/metrics.json` |
| PLM PPCA | 824 | 25.5 | 5.86s | `outputs/table_i/plm_ppca/aachen/metrics.json` |
| SuperPoint + LightGlue | 824 | 0.0 | 6.05s | run failed / not meaningful |
| ALIKED + LightGlue | 824 | 0.0 | 3.49s | run failed / not meaningful |

## Current Read

- Best PLM without external pairwise verifier: `PLM-SPMemory-PPCA r4 obs8`.
- Best PLM overall at strict 0.25m/2deg: `PLM + SG + SimpleGraph`.
- Best PLM overall at 0.5m/5deg and 5m/10deg: `PLM-SPMemory-PPCA + SG + SimpleGraph`.
- SimpleGraph helps strongly with SG verification, but hurts the map-only PPCA row.
- EUPE/DINO high-level priors help modestly, but the decisive gain is SuperPoint local landmark memory.
- EUPE PPCA alone is not useful for accurate localization.
- FuseLoc has no completed comparable row yet because the official runner was killed/OOM/CUDA unstable.
