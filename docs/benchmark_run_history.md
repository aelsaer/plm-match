# Large-Scale Benchmark Run History

Last updated: 2026-07-12

This document records the large-scale PLMLoc runs submitted to the Visual
Localization benchmark. Parameters are taken from preserved `command.txt` and
`run_summary.json` files when available. Official scores are copied from the
benchmark portal records supplied by the experiment author. A dash means that
an official score has not been recorded here.

See [Dataset Workflows and Map Provenance](dataset_workflows.md) for the exact
map-construction, feature-attachment, retrieval, localization, and submission
pipeline used by each dataset family. In particular, feature labels do not by
themselves imply a feature-native triangulated map.

Metric triplets are reported in this order:

```text
0.25 m / 2 deg, 0.5 m / 5 deg, 5 m / 10 deg
```

## Selector Semantics

- `diverse_desc`: fixed-budget descriptor farthest-point sampling. Only
  `point_memory_max_obs` controls its budget. Adaptive-cover parameters are not
  used.
- `adaptive_cover`: variable-budget weighted coverage based on marginal gain.
- `adaptive_cover_farthest`: adaptive cover followed by farthest-mode rescue.
- `adaptive_cover_v2`: radius coverage controlled by `s_min`, with an optional
  quality gate and hard `K_min` floor in the latest implementation.

The old Aachen wrapper included strings such as `k16_16_gain0005` in filenames
for `diverse_desc` runs. Those strings were naming-only and did not affect fixed
farthest selection. The wrapper now emits `obs8` or `obs16` for fixed selectors.

## Artifact Roots

```text
External runs: /media/photogrammetry/A26C3DDF6C3DAF431/plm-match-runs
Local runs:    /home/photogrammetry/andreas/plm-match/outputs
```

## Aachen Day-Night v1.1

The official split contains 824 day and 191 night queries (1,015 total).
Combined values use these counts as weights.

### Fixed Farthest Runs

| Status | Retrieval | Local feature | Selector | Budget | PnP | Pose guided | Submission file | Day | Night | Combined |
|---|---|---|---|---:|---|---|---|---|---|---|
| Submitted | MixVPR-5 | Official Aachen geometry + nearest-attached ALIKED-r1024 | `diverse_desc` | 8 | 16/16, min 10 | off | `plmloc_aliked_mixvpr_aachen5_obs8_pnp16_min10_v1_1_full_official_format/Aachen_v1_1_eval_plmloc_aliked_mixvpr_aachen5_obs8_pnp16_min10_full1015.txt` | 83.0 / 90.3 / 94.3 | 65.4 / 84.8 / 93.2 | 79.7 / 89.3 / 94.1 |
| Pending portal result | MixVPR-5 | Official Aachen geometry + nearest-attached ALIKED-r1024 | `diverse_desc` | 16 | 16/16, min 10 | off | `adaptive_cover/Aachen_v1_1_eval_PLMLoc_aliked_mixvpr5_farthest16_diverse_desc_r1024_k16_16_gain0005_pnp16_min10.txt` | - | - | - |
| Pending portal result | MixVPR-5 | Official Aachen geometry + nearest-attached ALIKED-r1024 | `diverse_desc` | 8 | 16/16, min 10 | off | `adaptive_cover/Aachen_v1_1_eval_PLMLoc_aliked_mixvpr5_farthest8_diverse_desc_r1024_obs8_pnp16_min10.txt` | - | - | - |
| Pending portal result | MegaLoc-5 | Official Aachen geometry + nearest-attached ALIKED-r1024 | `diverse_desc` | 8 | 16/16, min 10 | off | `adaptive_cover/Aachen_v1_1_eval_PLMLoc_aliked_megaloc5_farthest8_diverse_desc_r1024_obs8_pnp16_min10.txt` | - | - | - |

The first row is the original `PLMLoc-aliked` portal entry. Its preserved run
localized 1,008/1,015 queries at 0.133 s/query. Provenance:

See
[Aachen PLMLoc-ALIKED MixVPR-5 Fixed-8 Reproduction](aachen_aliked_mixvpr5_fixed8.md)
for the exact split, extraction, attachment, retrieval, localization, packaging,
and validation commands.

```text
aachen_submission_plmloc/results/mixvpr5_v1_1/
  point_memory_aliked_obs8_pnp16_min10/command.txt
aachen_submission_plmloc/results/mixvpr5_v1_1/
  point_memory_aliked_obs8_pnp16_min10/run_summary.json
```

The pending MegaLoc farthest-8 run localized 1,012/1,015 queries locally. The
pending MixVPR farthest-16 run localized 1,007/1,015. These are coverage values,
not official accuracy.

### Adaptive Selector Chronology

| Portal/submission label | Retrieval | Selector parameters | PnP / guidance | Day | Night |
|---|---|---|---|---|---|
| `PLMLoc-aliked-adaptive cover` | NetVLAD (legacy wrapper) | adaptive cover, K 1-32, gain 0.005 | 12/12, min 12 | 82.3 / 89.1 / 92.6 | 62.3 / 81.7 / 91.1 |
| `PLMLoc-aliked-adaptive cover - pnp12-12min` | MixVPR-10 | adaptive cover, K 1-32, gain 0.005 | 12/12, min 12 | 81.6 / 88.1 / 92.4 | 60.7 / 80.1 / 91.1 |
| `...mixvpr10_adaptive_cover_k1_32_gain0005_pnp16_min10_poseguided` | MixVPR-10 | adaptive cover, K 1-32, gain 0.005 | 16/16, min 10, PG on | 82.0 / 89.1 / 92.2 | 62.3 / 81.7 / 91.6 |
| `...mixvpr10_adaptive_cover_k4_32_gain0005_pnp16_min10_poseguided_pgmin12` | MixVPR-10 | adaptive cover, K 4-32, gain 0.005 | 16/16, min 10, PG min 12 | 81.9 / 88.3 / 92.6 | 60.2 / 80.1 / 91.1 |
| `...mixvpr10_adaptive_cover_k1_32_gain0001_pnp16_min10_poseguided_pgmin12` | MixVPR-10 | adaptive cover, K 1-32, gain 0.001 | 16/16, min 10, PG min 12 | 81.9 / 89.0 / 92.5 | 60.7 / 80.6 / 89.5 |
| `...mixvpr10_adaptive_cover_farthest_k1_32_gain0005_pnp16_min10_poseguided` | MixVPR-10 | adaptive + farthest rescue, K 1-32 | 16/16, min 10, PG on | 82.0 / 88.6 / 92.5 | 61.3 / 81.7 / 92.1 |
| `...mixvpr10_adaptive_cover_rawcos_k1_32_gain0005_pnp16_min10_poseguided` | MixVPR-10 | adaptive raw cosine, K 1-32 | 16/16, min 10, PG on | 82.4 / 89.0 / 93.2 | 60.7 / 81.2 / 91.6 |
| `...mixvpr10_adaptive_cover_farthest_rawcos_k1_32_gain0005_pnp16_min10` | MixVPR-10 | adaptive raw cosine + farthest | 16/16, min 10 | 82.5 / 89.0 / 92.7 | 60.7 / 80.1 / 90.6 |
| `...superpoint_mixvpr10_adaptive_cover_rawcos_k1_32_gain0005_pnp16_min10` | MixVPR-10 | SP, adaptive raw cosine, K 1-32 | 16/16, min 10 | 80.8 / 86.0 / 90.9 | 53.4 / 69.1 / 77.5 |
| `...mixvpr5_adaptive_cover_farthest_rawcos_thr095_k8_r1024_k1_8_gain0005` | MixVPR-5 | farthest rescue 0.95, K 1-8 | 16/16, min 10 | 83.0 / 89.9 / 94.8 | 64.4 / 83.2 / 93.7 |
| `...mixvpr5_adaptive_cover_rawcos_k8_r1024_k1_8_gain0005` | MixVPR-5 | adaptive raw cosine, K 1-8 | 16/16, min 10 | 82.9 / 89.3 / 94.8 | 64.4 / 82.2 / 93.2 |
| `...mixvpr5_adaptive_cover_rawcos_k12_r1024_k1_12_gain0005` | MixVPR-5 | adaptive raw cosine, K 1-12 | 16/16, min 10 | 82.6 / 89.8 / 94.2 | 67.0 / 82.2 / 92.1 |
| `...mixvpr5_adaptive_cover_farthest_rawcos_thr095_k12_r1024_k1_12_gain0005` | MixVPR-5 | farthest rescue 0.95, K 1-12 | 16/16, min 10 | 82.4 / 89.8 / 94.7 | 64.9 / 81.7 / 92.1 |
| `...megaloc5_adaptive_cover_farthest_rawcos_thr095_k8_r1024_k1_8_gain0005` | MegaLoc-5 | farthest rescue 0.95, K 1-8 | 16/16, min 10 | 83.7 / 91.9 / 97.0 | 65.4 / 84.8 / 99.0 |
| `...megaloc5_adaptive_cover_v2_s080_g030_k12_r1024_k1_12_gain0005` | MegaLoc-5 | v2, s=0.80, gate=0.30, K 1-12 | 16/16, min 10 | 84.2 / 91.9 / 97.7 | 64.9 / 85.9 / 97.9 |
| `...mixvpr5_adaptive_cover_v2_s080_g030_k12_r1024_k1_12_gain0005` | MixVPR-5 | v2 uniform legacy index, s=0.80, K 1-12 | 16/16, min 10 | 82.9 / 89.8 / 95.0 | 63.4 / 83.2 / 92.7 |
| `...mixvpr5_adaptive_cover_v2_weighted_s080_g030_k12_r1024_k1_12_gain0005` | MixVPR-5 | v2 weighted, s=0.80, gate=0.30, K 1-12 | 16/16, min 10 | 82.4 / 89.4 / 94.8 | 63.9 / 84.3 / 93.2 |
| `...mixvpr5_adaptive_cover_v2_keypointscores_s080_g030_k12_r1024_k1_12` | MixVPR-5 | v2 with actual ALIKED keypoint scores | 16/16, min 10 | 82.9 / 89.6 / 94.3 | 61.8 / 83.2 / 92.1 |
| `...mixvpr5_adaptive_cover_v2_floor8_s085_k16_r1024_k8_16_gain0005` | MixVPR-5 | v2 descriptor-only, s=0.85, hard K 8-16 | 16/16, min 10 | 82.8 / 89.9 / 94.5 | 65.4 / 84.3 / 93.7 |

For the floor-8 run, the selector retained 791,726/867,398 observations,
`mean_K=2.520`, and `fraction_below_hard_floor_unexpected=0`. The low mean is
caused by short landmark tracks: only 5.36% of landmarks have at least eight
available observations.

### Legacy Retrieval and Memory Ablations

The following preserved submissions use SuperPoint point memory unless their
name states otherwise. Their official scores are not all mapped unambiguously
to portal labels, so filenames are retained as the authoritative history.

| Submission file | Parameters encoded by the run |
|---|---|
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_full1015.txt` | MixVPR-5, SP, diverse-16 legacy default |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen10_full1015.txt` | MixVPR-10, SP, diverse-16 legacy default |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen50_full1015.txt` | MixVPR-50, SP, diverse-16 legacy default |
| `Aachen_v1_1_eval_plmloc_netvlad_aachen5_full1015.txt` | NetVLAD-5, SP, diverse-16 |
| `Aachen_v1_1_eval_plmloc_netvlad_aachen50_full1015.txt` | NetVLAD-50, SP, diverse-16 |
| `Aachen_v1_1_eval_plmloc_salad_aachen5_full1015.txt` | SALAD-5, SP, diverse-16 |
| `Aachen_v1_1_eval_plmloc_salad_aachen50_full1015.txt` | SALAD-50, SP, diverse-16 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs8_diverse_full1015.txt` | MixVPR-5, SP, diverse-8 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs12_diverse_full1015.txt` | MixVPR-5, SP, diverse-12 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs14_diverse_full1015.txt` | MixVPR-5, SP, diverse-14 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs16_diverse_full1015.txt` | MixVPR-5, SP, diverse-16 |
| `Aachen_v1_1_eval_plmloc_salad_aachen5_obs8_diverse_full1015.txt` | SALAD-5, SP, diverse-8 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs8_min10_full1015.txt` | MixVPR-5, SP, diverse-8, PnP 12/min10 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs8_pnp16_min10_full1015.txt` | MixVPR-5, SP, diverse-8, PnP 16/min10 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs8_clusters8x15_full1015.txt` | MixVPR-5, SP, diverse-8, clusters 8x15 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs8_clusters8x15_pnp16_min10_full1015.txt` | previous row with PnP 16/min10 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen10_obs8_adapt_5_7_10_full1015.txt` | MixVPR-10, SP, diverse-8, adaptive retrieval schedule 5/7/10 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs8_active_rank4000_full1015.txt` | MixVPR-5, SP, diverse-8, active rank pool 4000 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs8_desc_center_l2_full1015.txt` | MixVPR-5, SP, diverse-8, descriptor-center L2 attachment |
| `Aachen_v1_1_eval_plmloc_megaloc_aachen10_obs8_diverse_full1015.txt` | MegaLoc-10, SP, diverse-8, PnP 16/min10 |
| `Aachen_v1_1_eval_plmloc_megaloc_aachen10_obs16_diverse_full1015.txt` | MegaLoc-10, SP, diverse-16, PnP 12/min12 |
| `Aachen_v1_1_eval_plmloc_mixvpr_aachen5_obs16_illumdiv_err2_full1015.txt` | MixVPR-5, SP, illumination-diverse-16, COLMAP point error <=2 |

## RobotCar Seasons v2

The official split contains 1,443 day and 429 night queries (1,872 total).

| Status | Retrieval | Map/local feature | Selector | Budget | PnP | Submission file | Day | Night | Combined |
|---|---|---|---|---:|---|---|---|---|---|
| Submitted | MixVPR-5 | ALIKED+LightGlue SfM, ALIKED-r1024 | retrieval-conditioned `image_obs` | n/a | 16/16, min 10 | `RobotCar_eval_PLMLoc_aliked_mixvpr5_obs8_pnp16_min10_v2_test.txt` | 65.3 / 95.0 / 100.0 | 38.0 / 70.9 / 82.3 | 59.0 / 89.5 / 95.9 |
| Submitted | MixVPR-5 | Converted RobotCar reference + nearest-attached ALIKED-r1024 | adaptive v2 descriptor-only | K 8-16, s=0.85 | 16/16, min 10 | `adaptive_cover/RobotCar_eval_PLMLoc_aliked_top5_adaptive_cover_v2_floor8_s085_k16_r1024_k8_16_gain0005_pnp16_min10_v2_test.txt` | 63.8 / 94.7 / 100.0 | 27.5 / 64.3 / 83.4 | 55.5 / 87.7 / 96.2 |
| Submitted | MegaLoc-5 | Converted RobotCar reference + nearest-attached ALIKED-r1024 | fixed `diverse_desc` | 8 | 16/16, min 10 | `adaptive_cover/RobotCar_eval_PLMLoc_plmloc_aliked_megaloc5_farthest8_diverse_desc_r1024_obs8_pnp16_min10_v2_test.txt` | 63.3 / 94.7 / 100.0 | 35.9 / 80.9 / 98.6 | 57.0 / 91.5 / 99.7 |
| Portal score not recorded here | MegaLoc-5 | Converted RobotCar reference + nearest-attached ALIKED-r1024 | adaptive + farthest rescue 0.95 | K 1-8 | 16/16, min 10 | `adaptive_cover/RobotCar_eval_PLMLoc_plmloc_aliked_megaloc5_adaptive_cover_farthest_rawcos_thr095_k8_r1024_k1_8_gain0005_pnp16_min10_v2_test.txt` | - | - | - |

Preserved legacy RobotCar submissions:

| Submission file | Parameters |
|---|---|
| `RobotCar_eval_PLMLoc_mixvpr5_v2_test.txt` | SP map, MixVPR-5, legacy diverse point memory |
| `RobotCar_eval_PLMLoc_mixvpr50_v2_test.txt` | SP map, MixVPR-50, legacy diverse point memory |

The fixed MegaLoc farthest-8 run produced all 1,872 pose lines. Its exact-zero
portal entry observed during one upload was therefore not caused by missing
query names: the submission names matched `robotcar_v2_test.txt` exactly. The
subsequent recorded portal result is the non-zero row shown above.

The first native-ALIKED row has a legacy `obs8` token in its filename, but its
preserved `command.txt` sets `landmark_match_mode=image_obs`. Therefore the
observation cap and point-memory selector were inactive; this result must not
be described as a fixed diverse-8 point-memory run.

## Extended CMU Seasons

Extended CMU is reported by urban, suburban, and park conditions, not as a
day/night weighted aggregate.

| Status | Retrieval | Map/local feature | Memory | PnP | Submission file | Urban | Suburban | Park |
|---|---|---|---|---|---|---|---|---|
| Submitted | MixVPR-5 | native ALIKED+LightGlue SfM, covisibility 30, ALIKED-r1024 | point memory, max 8 | 16/16, min 10 | `cmu_extended_plmloc_aliked_lg_covis30_mixvpr5/CMU_eval_PLMLoc_aliked_lg_covis30_mixvpr5.txt` | 94.0 / 97.0 / 99.1 | 95.2 / 97.6 / 99.6 | 85.3 / 89.9 / 96.2 |
| Submitted | MixVPR-5 | native SP+SG SfM, SP-r1600, covisibility 30 | point memory, max 8 | 16/16, min 10 | `CMU_eval_PLMLoc_sp_sg_1600_covis30_mixvpr5_full56613.txt` | 93.4 / 96.2 / 98.7 | 88.1 / 90.8 / 97.0 | 77.0 / 80.1 / 88.9 |
| Submitted | MixVPR-5 | native SP+SG SfM, SP-r1600, covisibility 30 | point memory, max 16 | 16/16, min 10 | `CMU_eval_PLMLoc_sp_sg_1600_covis30_mixvpr5_obs16_full56613.txt` | 93.6 / 96.3 / 98.6 | 88.2 / 91.0 / 97.0 | 76.8 / 80.3 / 88.8 |
| Portal score not recorded here | MegaLoc-5 | Provided per-slice geometry + nearest-attached ALIKED-r1024 | adaptive raw cosine, K 1-12 | 16/16, min 10 | `adaptive_cover/CMU_eval_PLMLoc_plmloc_aliked_megaloc5_adaptive_cover_rawcos_k12_r1024_k1_12_gain0005_pnp16_min10.txt` | - | - | - |
| In progress | MixVPR-5 | Provided per-slice geometry + nearest-attached ALIKED-r1600 | adaptive v2 descriptor-only, K 8-16, s=0.85 | 16/16, min 10 | expected `CMU_eval_PLMLoc_aliked_top5_adaptive_cover_v2_floor8_s085_k16_r1600...txt` | - | - | - |

### CMU Park Ablations

All rows retain the original urban/suburban poses and replace only park slices.

| Submission file | Park parameters | Park result |
|---|---|---|
| `CMU_eval_PLMLoc_sp_sg_park_mixvpr10_obs16_pnp16_min10_full56613.txt` | MixVPR-10, obs16, PnP16/min10 | 75.5 / 78.9 / 87.6 |
| `CMU_eval_PLMLoc_sp_sg_park_mixvpr10_obs16_pnp20_min8_full56613.txt` | MixVPR-10, obs16, PnP20/min8 | 76.2 / 79.5 / 87.9 |
| `CMU_eval_PLMLoc_sp_sg_park_mixvpr20_obs16_pnp16_min10_full56613.txt` | MixVPR-20, obs16, PnP16/min10 | 73.4 / 76.7 / 85.1 |
| `CMU_eval_PLMLoc_sp_sg_park_mixvpr20_obs16_pnp20_min8_full56613.txt` | MixVPR-20, obs16, PnP20/min8 | 73.8 / 77.0 / 85.2 |

## Interpretation Notes

1. Fixed `diverse_desc` farthest-8 is the strongest verified simple selector
   among the recorded point-memory Aachen and RobotCar attachment runs. The
   native RobotCar ALIKED result at 65.3/95.0/100.0 day used `image_obs` and is
   not a selector ablation.
2. Adaptive v2's Aachen floor fix improved night results over unconstrained v2,
   but it did not transfer to RobotCar strict/medium night thresholds.
3. MegaLoc retrieval is substantially more spatially coherent than MixVPR on
   Aachen night in the saved retrieval diagnostics. Selector comparisons must
   therefore keep retrieval fixed.
4. ALIKED keypoint-score weighting is a negative ablation. The quality gate
   removed very few observations and the weighted representative choice reduced
   inliers.
5. Official benchmark accuracy and local pose coverage are separate quantities.
   Public query ground truth is not available for Aachen or RobotCar test.

## Updating This History

For each new submission, add:

1. Dataset and benchmark split.
2. Feature map source and local descriptor family.
3. Retrieval method and top-k.
4. Selector mode and all active selector parameters.
5. PnP thresholds, minimum inliers, and pose-guided state.
6. Exact submitted filename.
7. Official portal score once available.
8. Path to `command.txt` and `run_summary.json` when artifacts are retained.
