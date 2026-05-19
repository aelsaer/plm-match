# PLMLoc Ablation Study Plan

## 1. Datasets & Metrics

### Cambridge Landmarks (5 scenes)
**Metric:** median translation error (cm) / median rotation error (°)
**Scenes:** Great Court, King's College, Hospital, ShopFacade, St. Mary's Church
**Split:** official train/test
**Retrieval:** NetVLAD top-10 (LOO)

| Scene | #queries | Notes |
|---|---|---|
| Great Court | 760 | ✅ HLoc-map full ablation complete; largest outdoor scene |
| King's College | 343 | ✅ HLoc-map full ablation complete; accuracy sweep running |
| Hospital | 182 | ✅ HLoc-map full ablation complete |
| ShopFacade | 103 | ✅ ablations done |
| St. Mary's Church | 530 | ✅ HLoc-map full ablation complete |

---

### Aachen Day/Night v1.1
**Metric:** % queries at (0.25m, 2°) / (0.5m, 5°) / (5m, 10°) — day + night
**Split:** official (4,479 db / 824 day queries / 191 night queries)
**Retrieval:** NetVLAD top-50
**Dataset root:** `D:\private\pairs\aachen_v1_1` on Windows, `/mnt/d/private/pairs/aachen_v1_1` in WSL
**Status:** The full official Aachen v1.1 dataset is available on disk but still needs an official split/config/index run. SIFT index not yet built.

---

### 7-Scenes (chess scene, extendable)
**Metric:** % queries at (5 cm, 5°), (10 cm, 5°), (25 cm, 10°), plus median translation/rotation
**Map sources:** two required tracks:
1. **HLoc SP+SG SfM map** for direct comparison to HLoc.
2. **RGB-D memory** to show PLMLoc can build landmarks without SfM.
**Status:** Chess RGB-D full ablation suite is complete, including SP, point modes, SuperGlue/LightGlue lifted baselines, SIFT, and RootSIFT. Chess and Heads HLoc SP+SG SfM suites are also available locally.

---

### TUM RGB-D
**Metric:** % queries at (5 cm, 5°), (10 cm, 5°), (25 cm, 10°), plus median translation/rotation
**Sequences available locally:** `rgbd_dataset_freiburg1_desk`, `rgbd_dataset_freiburg1_room`
**Map source:** RGB-D memory from map-frame depth and ground-truth poses; query depth is not used.
**Status:** `fr1_desk` is prepared and full SP RGB-D ablations are complete under `outputs/tum_rgbd/fr1_desk_lifted/full_ablation_suite`. `fr1_room` is downloaded but still needs preparation/runs.

---

### Extended CMU Seasons
**Metric:** % queries at (0.25m, 2°) / (0.5m, 5°) / (5m, 10°)
**Slices:** 17 slices (suburban + urban)
**Status:** Dataset files are present, but this still needs a dedicated loader/prep path, split generation, retrieval, SP features, and PLM indexes.

---

### RobotCar Seasons v2
**Metric:** % queries at (0.25m, 2°) / (0.5m, 5°) / (5m, 10°)
**Conditions:** dawn, dusk, night, night-rain, overcast-summer, overcast-winter, rain, snow, sun (9 query conditions)
**Status:** Dataset files and `3D-models/overcast-reference.db` are present locally. Recommended paper path is still: run HLoc RobotCar first on a machine that can handle the SfM/model conversion, then use `tools/prepare_robotcar_hloc_outputs.py` to make PLMLoc consume HLoc's `sfm_superpoint+superglue`, HLoc features, and HLoc retrieval. `tools/prepare_robotcar_seasons.py` is only a fallback direct-NVM prep path, not the paper path.

---

## 2. Published Baselines to Beat

### Cambridge Landmarks — median (cm / °)

| Method | Court | King's | Hospital | Shop | St. Mary's | Avg |
|---|---|---|---|---|---|---|
| AS (SIFT) [Sattler] | 24/0.1 | 13/0.2 | 20/0.4 | 4/0.2 | 8/0.3 | 14/0.2 |
| hLoc (SP+SG) [Sarlin] | 16/0.1 | 12/0.2 | 15/0.3 | 4/0.2 | 7/0.2 | 11/0.2 |
| pixLoc | 30/0.1 | 14/0.2 | 16/0.3 | 5/0.2 | 10/0.3 | 15/0.2 |
| DSAC* (no depth) | 34/0.2 | 18/0.3 | 21/0.4 | 5/0.3 | 15/0.6 | 19/0.4 |
| ACE (ours, from paper) | 43/0.2 | 28/? | — | — | — | — |
| FuseLoc SIFT | 23.4/0.1 | 10.4/0.2 | 13.3/0.3 | 4.2/0.2 | 6.8/0.2 | 11.6/0.2 |
| FuseLoc SP | 28.0/0.1 | 10.7/0.2 | 15.1/0.3 | 4.1/0.2 | 7.2/0.2 | 13.0/0.2 |
| FuseLoc D2+SALAD | — | — | — | — | — | — |

**Target:** beat hLoc (SP+SG) on ShopFacade (4/0.2) and match or beat on other scenes.

---

### Aachen Day/Night v1.1 — % at thresholds

| Method | Day 0.25/2 | Day 0.5/5 | Day 5/10 | Night 0.25/2 | Night 0.5/5 | Night 5/10 |
|---|---|---|---|---|---|---|
| hLoc (SP+SG) | 83.4 | 93.4 | 99.7 | — | — | — |
| FuseLoc SIFT | 56.0 | 60.0 | 65.6 | — | — | — |
| FuseLoc SP | 67.4 | 77.7 | 85.6 | — | — | — |
| FuseLoc D2+SALAD (best) | 77.9 | 88.7 | 95.5 | — | — | — |

*(FuseLoc paper reports combined day+night; split out where available)*

---

### RobotCar Seasons v2 — % at thresholds

| Method | 0.25m/2° | 0.5m/5° | 5m/10° |
|---|---|---|---|
| hLoc (SP+SG) | 52.0 | 87.2 | 96.1 |
| FuseLoc SIFT | 24.9 | 39.2 | 43.6 |
| FuseLoc D2+SALAD (best) | 47.0 | 87.0 | 99.6 |

---

### Extended CMU Seasons — % at thresholds

| Method | 0.25m/2° | 0.5m/5° | 5m/10° |
|---|---|---|---|
| hLoc (SP+SG) | 90.7 | 93.9 | 96.0 |
| AS (SIFT) | 63.0 | 69.9 | 78.5 |
| FuseLoc D2+SALAD (best) | 88.5 | 93.6 | 98.0 |

---

### 7-Scenes — % at (5cm, 5°) per scene

| Method | Chess | Fire | Heads | Office | Pumpkin | RedKitchen | Stairs | Avg |
|---|---|---|---|---|---|---|---|---|
| DSAC* (full, w/ depth) | — | — | — | — | — | — | — | — |
| ACE (no depth) | — | — | — | — | — | — | — | — |
| hLoc (SP+SG) | — | — | — | — | — | — | — | — |

*(Fill from benchmark website once runs are done)*

---

## 3. PLM Variant Results (Cambridge ShopFacade, HLoc SP+SG map)

These runs use the HLoc-style SuperPoint+SuperGlue triangulated model rather than the official SIFT COLMAP model:

```
config         : configs/cambridge_shopfacade_hloc_sp_sg.yaml
attached_index : outputs/cambridge_shopfacade_official/sp_colmap_attach_hloc_sp_sg_r2
retrieval      : outputs/hloc_cambridge_sp_sg/ShopFacade/pairs-query-netvlad10.txt
features       : outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5
metric         : median translation cm / median rotation deg
leakage        : ok=true
```

HLoc SP+SG on the same triangulated model reports **4.3 / 0.2**.

| Run | Mode / setting | Median cm | Median deg | Mean time/query |
|---|---|---:|---:|---:|
| `plm_point_memory_support8_hloc_sp_sg_r2` | point memory + support/rank, max 8 obs/point | **4.66** | **0.225** | 3.36s |
| `plm_point_memory_support_full_hloc_sp_sg_r2` | point memory + support/rank, all obs | 4.70 | 0.232 | 4.57s |
| `main_sp_image_obs_hloc_sp_sg_r2_rank_only` | image obs, rank prior only | 4.72 | 0.233 | 0.57s |
| `main_sp_image_obs_hloc_sp_sg_r2_no_priors` | image obs, no support/rank priors | 4.76 | 0.240 | 0.52s |
| `main_sp_image_obs_hloc_sp_sg_r2` | image obs, default support/rank priors | 4.88 | 0.250 | 0.55s |
| `main_sp_image_obs_hloc_sp_sg_r2_alltracks` | image obs, all SP+SG tracks | 4.94 | 0.243 | 0.61s |
| `plm_point_memory8_hloc_sp_sg_r2` | point memory, no support/rank, max 8 obs/point | 5.04 | 0.233 | 3.19s |
| `main_sp_image_obs_hloc_sp_sg_r2_poseguided` | image obs + pose-guided refinement | 5.18 | 0.249 | 0.64s |
| `plm_point_mean_hloc_sp_sg_r2` | point mean/codebook descriptor | 5.28 | 0.265 | 0.94s |
| `main_sp_image_obs_hloc_sp_sg_r2_no_rank` | image obs, support priors only | 5.44 | 0.260 | 0.52s |

### ShopFacade takeaways
- Best PLM flavor is `point_memory_support` with `point_memory_max_obs=8`: **4.66 / 0.225**.
- Full point memory is slower and slightly worse than capped memory, so capping observations is a useful regularizer here.
- Practical fast variant is `image_obs rank_only`: **4.72 / 0.233** at **0.57s/query**.
- Pose-guided refinement hurts ShopFacade median error even though it increases inliers.
- Mean descriptor/codebook memory is weaker than multi-observation memory.
- Support priors without rank are harmful on this scene; rank-only is better than default support+rank.

## 3.1 PLM Variant Results (Cambridge King's College, HLoc SP+SG map)

These runs use the HLoc Cambridge SuperPoint+SuperGlue triangulated model rather than the official SIFT COLMAP model:

```
config         : configs/cambridge_kingscollege_lifted.yaml
attached_index : outputs/cambridge_kingscollege_lifted/sp_colmap_attach_hloc_index
retrieval      : /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/KingsCollege/pairs-query-netvlad10.txt
features       : /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/KingsCollege/feats-superpoint-n4096-r1024.h5
summary        : outputs/cambridge_kingscollege_lifted/full_ablation_suite/ablation_suite/summary_tables/ablation_summary.md
metric         : median translation cm / median rotation deg, plus 0.25m/2deg, 0.5m/5deg, 5m/10deg
leakage        : ok=true
```

Stock pure HLoc SP+SG on the same triangulated model, evaluated from `CambridgeLandmarks_Colmap_Retriangulated_1024px/KingsCollege/results.txt`, reports **11.59 / 0.205** with **73.8 / 91.0 / 100.0** at `0.25m/2deg`, `0.5m/5deg`, and `5m/10deg`.

| Run | Median cm | Median deg | 0.25m/2deg | 0.5m/5deg | 5m/10deg | Runtime/query |
|---|---:|---:|---:|---:|---:|---:|
| `pure_hloc_sp_sg` | **11.59** | 0.205 | 73.8 | **91.0** | 100.0 | — |
| `matcher_lg_lifted` | 11.76 | **0.202** | 73.8 | 90.4 | 100.0 | **0.913s** |
| `matcher_sg_lifted` | 12.38 | 0.206 | **74.1** | 90.4 | 100.0 | 1.877s |
| `main_sp_image_obs` | 14.53 | 0.244 | 70.3 | 89.2 | 100.0 | 0.987s |
| `plm_point_memory` | 15.45 | 0.248 | 67.9 | 89.5 | 100.0 | 3.494s |
| `plm_point_memory_support` | 15.51 | 0.242 | 68.2 | 89.5 | 100.0 | 3.696s |
| `plm_point_mean` | 15.78 | 0.254 | 66.8 | 90.7 | 100.0 | 0.952s |

### King's College takeaways
- Stock pure HLoc SP+SG is currently the best median result on King's: **11.59 / 0.205**.
- Among the lifted baselines, `matcher_lg_lifted` is the strongest overall completed row: **11.76 / 0.202** at **0.913s/query**.
- `matcher_sg_lifted` wins the strict success threshold slightly, but LightGlue is better on median error and runtime.
- Default PLMLoc `image_obs` is clearly stronger than the point-level PLM variants on this scene, but it does not yet match pure HLoc or the lifted LG/SG baselines.
- The KingsCollege accuracy sweep is still running; early completed sweep rows slightly improve the strict threshold over default `image_obs`, but the full sweep ranking is not final yet.

## 4. PLM Variant Results (7-Scenes Chess, RGB-D map)

These runs use the direct RGB-D PLMLoc memory rather than an SfM map:

```
config         : configs/7scenes_chess_official_rgbd.yaml
attached_index : outputs/7scenes_chess_official_rgbd/sp_rgbd_memory_r002
retrieval      : /mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/chess_top10.txt
features       : outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_{db,queries}.h5
summary        : outputs/7scenes_chess_official_rgbd/full_ablation_suite/ablation_suite/summary_tables/ablation_summary.md
metric         : 5cm/5deg, 10cm/5deg, 25cm/10deg, median translation/rotation
leakage        : ok=true
```

| Run | 5cm/5deg | 10cm/5deg | 25cm/10deg | Median m | Median deg | Runtime/query |
|---|---:|---:|---:|---:|---:|---:|
| `main_sp_image_obs` | **86.2** | **99.2** | **100.0** | **0.028** | **1.78** | 0.271 |
| `feature_sift_image_obs` | 83.1 | 98.8 | 99.9 | 0.029 | 1.85 | 0.307 |
| `feature_rootsift_image_obs` | 82.8 | 98.9 | 100.0 | 0.029 | 1.85 | **0.270** |
| `matcher_lg_lifted` | 82.7 | 98.0 | 100.0 | 0.031 | 1.83 | 0.335 |
| `matcher_sg_lifted` | 76.4 | 97.7 | 100.0 | 0.033 | 1.89 | 0.559 |
| `plm_point_mean` | 46.4 | 67.0 | 73.0 | 0.041 | 2.00 | 0.432 |
| `plm_point_memory` | 42.4 | 65.6 | 74.4 | 0.044 | 2.20 | 1.728 |
| `plm_point_memory_support` | 42.9 | 65.8 | 75.2 | 0.044 | 2.21 | 1.887 |

### Chess RGB-D takeaways
- The strongest RGB-D PLMLoc row is `main_sp_image_obs`: **86.2% at 5cm/5deg**, **2.84cm / 1.78deg** median.
- SIFT and RootSIFT are close to SuperPoint on this scene, supporting the feature-agnostic memory claim.
- LightGlue lifted is stronger and faster than SuperGlue lifted in this RGB-D setup.
- Point-level modes are much weaker than retrieved image-observation memory on RGB-D Chess. Do not claim point-memory superiority from this dataset; use it as evidence that preserving retrieved observation context matters.

## 5. Runtime Notes

Runtime should be reported in the same convention as FuseLoc Table 4: **exclude query local/global descriptor extraction**, because both hLoc and PLM need those front-end descriptors. In our summaries this corresponds to `mean_query_time_s` for localization/matching after descriptors are available. Convert to FPS as:

```
fps = 1 / mean_query_time_s
```

New run summaries also expose this directly as `query_fps` / `query_process_fps`. For paper tables, prefer FPS and keep seconds/query only as a diagnostic column.

### Representative FPS

| Dataset / run | Mean time/query | FPS | Note |
|---|---:|---:|---|
| ShopFacade `image_obs no_priors` | 0.52s | 1.93 | fastest ShopFacade HLoc-SP-map PLM run |
| ShopFacade `image_obs rank_only` | 0.57s | 1.74 | best fast ShopFacade PLM run |
| ShopFacade `point_memory_support8` | 3.36s | 0.30 | best ShopFacade accuracy, slower |

Interpretation: PLM fast variants should only be compared under matched descriptor-excluded timing. Do not claim a runtime win over hLoc unless the comparison is carefully matched.

## 6. Implementation Status

Core ablation machinery is implemented:

- `landmark_match_mode`: `image_obs`, `point_mean`, `point_memory`, `point_memory_support`
- point-memory caps and batching: `point_memory_max_obs`, `point_memory_batch_size`
- SIFT and RootSIFT ablations: SIFT parameters, `sift_match_test`, `sift_ratio`, `sift_descriptor_norm`
- attachment modes: `detected_nearest`, `colmap_uv_compute`
- generic COLMAP/RGB-D attachment wrappers
- RobotCar Seasons v2 HLoc-output adapter: train-query split, GT pose export, generated config around HLoc's SP+SG model/features/retrieval
- RobotCar direct-NVM fallback prep: NVM-to-COLMAP text conversion, train-query split, GT pose export, generated config
- HLoc COLMAP-attached baseline script
- ablation suite runner and summary script
- Cambridge median cm/degree report fields

Still requiring implementation or dataset setup:

- full official Aachen v1.1 config/split/index using `/mnt/d/private/pairs/aachen_v1_1`
- CMU Seasons loader/prep path
- RobotCar HLoc setup run: obtain `overcast-reference.db`, run HLoc RobotCar, then build PLM attachment/index from HLoc outputs
- multi-scene Cambridge automation for the 4 remaining scenes

### Commands to run these variants on a new dataset

Replace `$CFG`, `$SPLIT`, `$BASE`, `$RETRIEVAL`, `$SP_R3`, `$SP_R5`, `$DB_H5`, `$Q_H5` with dataset-specific paths.

```bash
PY="/home/andreas/anaconda3/envs/sam3/bin/python"
BASE_CMD="$PY -m plm_match.pipelines.lifted_nn_localize \
  --config $CFG --split_json $SPLIT \
  --method superpoint_h5 \
  --landmark_match_mode image_obs \
  --db_features_path $DB_H5 \
  --query_features_path $Q_H5 \
  --retrieval_file $RETRIEVAL \
  --topk 20 --query_topk 4096 \
  --support_weight 0.03 --point_support_weight 0.02 \
  --rank_weight 0.02 --attach_dist_weight 0.01 \
  --max_cluster_images 5 --max_cluster_seeds 10 \
  --pnp_first_thresh 8.0 --pnp_refine_thresh 4.0 \
  --min_final_inliers 12"

# Main (full method)
$BASE_CMD --attached_index $SP_R3 --out_dir $BASE/results/main_sp_r3 --pose_guided

# No pose-guided
$BASE_CMD --attached_index $SP_R3 --out_dir $BASE/results/no_pose_guided --no-pose_guided

# No support weight
$BASE_CMD --attached_index $SP_R3 --out_dir $BASE/results/no_support_weight \
  --pose_guided --support_weight 0.0

# No point support weight
$BASE_CMD --attached_index $SP_R3 --out_dir $BASE/results/no_point_support_weight \
  --pose_guided --point_support_weight 0.0

# No rank weight
$BASE_CMD --attached_index $SP_R3 --out_dir $BASE/results/no_rank_weight \
  --pose_guided --rank_weight 0.0

# No attach dist weight
$BASE_CMD --attached_index $SP_R3 --out_dir $BASE/results/no_attach_dist_weight \
  --pose_guided --attach_dist_weight 0.0

# Radius 5
$BASE_CMD --attached_index $SP_R5 --out_dir $BASE/results/radius5_same_settings --pose_guided

# TopK 30
$BASE_CMD --attached_index $SP_R3 --out_dir $BASE/results/topk30_same_settings \
  --pose_guided --topk 30
```

### Sensitivity Status Per Dataset

| Dataset | main | no_pg | no_sw | no_psw | no_rw | no_adw | r5 | topk30 |
|---|---|---|---|---|---|---|---|---|
| Cambridge ShopFacade | ✅ partial | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Cambridge (other 4) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 7-Scenes chess | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| TUM RGB-D | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| CMU Seasons | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| RobotCar Seasons | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## 6. PLMLoc Runs

### 7.1 Main Result — PLMLoc (SP, image_obs mode)

The primary method result. Run on all datasets.

```
--method superpoint_h5
--landmark_match_mode image_obs
--topk 10 (Cambridge/7Scenes)  /  20 (official Aachen/CMU/RobotCar)
--query_topk 4096
--support_weight 0.03
--point_support_weight 0.02
--rank_weight 0.02
```

**Status per dataset:**

| Dataset | Status |
|---|---|
| Cambridge ShopFacade | ✅ HLoc-SP+SG-map SP runs done; best fast `image_obs rank_only` is 4.72/0.233 |
| Cambridge King's College | ✅ HLoc-SP+SG-map full ablation complete; pure HLoc eval is 11.59/0.205; sweep running |
| Cambridge Great Court / Hospital / St. Mary's | ✅ HLoc-SP+SG-map full ablations complete |
| Aachen official Day/Night | ❌ dataset available at `/mnt/d/private/pairs/aachen_v1_1`; official config/split/index still needed |
| 7-Scenes chess | ✅ RGB-D full ablation suite complete |
| TUM RGB-D | ✅ fr1/desk full SP RGB-D ablation complete; fr1/room downloaded but not run |
| CMU Seasons | ❌ loader/prep path still needed |
| RobotCar Seasons | ❌ loader/prep path still needed |

---

### 6.2 PLM Representation Ablation

Tests what descriptor to use per 3D point. Runs on Cambridge ShopFacade first, then official Aachen once the official setup is ready.

| Mode | Flag | Description |
|---|---|---|
| `image_obs` | `--landmark_match_mode image_obs` | Match against all retrieved-image observations (current best) |
| `point_mean` | `--landmark_match_mode point_mean` | One mean descriptor per 3D point |
| `point_memory` | `--landmark_match_mode point_memory` | Max over all observation descriptors |
| `point_memory_support` | `--landmark_match_mode point_memory_support` | Max + support/rank priors |

**Status:** ablation suite runner supports all four. Cambridge ShopFacade HLoc-SP-map representation ablation is complete. 7-Scenes Chess RGB-D representation ablation is complete. Official Aachen still needs setup before any paper-facing Aachen representation table.

---

### 6.3 Feature Ablation (Descriptor)

Tests which local feature descriptor is used. Run on Cambridge ShopFacade (fastest).

| Descriptor | Index | Best config | Status |
|---|---|---|---|
| SuperPoint (H5), official SIFT COLMAP map | `sp_colmap_attach_r3_scaled` | default | ✅ run exists but is not the fair hLoc-style comparison |
| SuperPoint (H5), HLoc-SP+SG map | `sp_colmap_attach_hloc_sp_sg_r2` | rank_only / no_priors | ✅ 4.72/0.233 fast, 4.66/0.225 best point-memory |
| SIFT (detected_nearest, r6) | `sift_detected_r6` | l2_ratio=0.75 | ✅ 5.5/0.2 |
| RootSIFT (r6) | `rootsift_detected_r6` | l2_ratio=0.80 | ✅ 5.2/0.2 |

**Note:** SIFT/RootSIFT ablation uses PLM pipeline with SIFT descriptors, not a standalone SIFT baseline.

---

### 6.4 Pairwise Matcher Lifted Baselines (SuperGlue / LightGlue)

The direct pairwise-matcher competitors within the same lifted framework. These use query-DB image matches lifted through the COLMAP or RGB-D attached index, then the same PnP evaluation.

```
tools/run_hloc_colmap_attached_baseline.py
--method superpoint_h5
--matcher_conf superglue
--attached_index sp_colmap_attach_r3_scaled
```

The ablation suite supports:

```bash
--mode_groups matcher
--matcher_confs superglue,superpoint+lightglue
```

It writes separate results as `matcher_sg_lifted` and `matcher_lg_lifted`. `--max_queries` is supported for smoke tests.

**Status:** command generation is dry-run checked. Full SuperGlue/LightGlue validation runs still need to be executed per scene.

---

### 6.5 Sensitivity Analysis

Parameter sweep on Cambridge ShopFacade (fast), then validate best settings on official Aachen once the official setup is ready.

| Experiment | What changes | Variants |
|---|---|---|
| `topk` | retrieval candidates | 10, 20, 30 |
| `attach_radius` | SP attachment radius | r3, r5 |
| `support_weight` | support prior | 0.0 vs 0.03 |
| `rank_weight` | rank prior | 0.0 vs 0.02 |
| `pose_guided` | pose-guided hypotheses | on vs off |

---

## 7. Ablation Suite Invocations

### Parameter-driven accuracy sweeps

Use `tools/run_lifted_parameter_sweep.py` when a dataset/scene needs the same kind of tuning we did for ShopFacade. It runs named variants, keeps every command/result/leakage check, and writes:

```text
<base_dir>/parameter_sweeps/<sweep_name>/
  commands/
  results/
  summary/sweep_summary.{json,csv,md}
  summary/best_run.json
```

Presets:

| Preset | Purpose |
|---|---|
| `quick` | Small sanity sweep: image_obs variants, point_mean, point_memory, point_memory_support |
| `plm` | Representation-focused sweep with point-memory observation caps |
| `support` | Support/rank prior ablation |
| `accuracy` | Wider per-scene search: PLM modes, support/rank, topK, thresholds, pose-guided, PnP looseness |

For a 5-query smoke run, add `--max_queries 5 --preset quick`. For the full scene, remove `--max_queries` and use `--preset accuracy`.

Shell shortcuts:

```bash
scripts/run_final_ablation_suite.sh parameter-sweeps-local-smoke5
scripts/run_final_ablation_suite.sh parameter-sweeps-local
scripts/run_final_ablation_suite.sh shopfacade-hloc-sweep5
scripts/run_final_ablation_suite.sh shopfacade-hloc-sweep
scripts/run_final_ablation_suite.sh 7scenes-chess-sweep5
scripts/run_final_ablation_suite.sh 7scenes-chess-sweep
```

For full 7-Scenes Chess core ablations, including SIFT/RootSIFT feature rows:

```bash
scripts/run_final_ablation_suite.sh 7scenes-chess-full
```

This target builds `sift_rgbd_memory_r002` and `rootsift_rgbd_memory_r002` if missing, then runs `main,plm,matcher,feature` with SuperGlue and LightGlue matcher baselines.

For the fair HLoc comparison on 7-Scenes Chess, first build HLoc's own SuperPoint+SuperGlue SfM map, then run PLMLoc on that same map:

```bash
scripts/run_final_ablation_suite.sh 7scenes-chess-hloc-sfm-build
scripts/run_final_ablation_suite.sh 7scenes-chess-hloc-sfm-full
```

This second target builds `outputs/7scenes_chess_hloc_sp_sg_sfm/sp_colmap_attach_r2` from HLoc's `sfm_superpoint+superglue` model and runs `main,plm,matcher` using the shared HLoc SuperPoint H5 and official DenseVLAD retrieval.

For a new dataset/scene, provide the scene-specific paths:

```bash
$PY tools/run_lifted_parameter_sweep.py \
  --dataset_name <dataset_scene_name> \
  --config <config.yaml> \
  --dataset_root <dataset_root> \
  --split_json <split.json> \
  --base_dir <output_base> \
  --attached_index sp_r3=<attached_index_dir> \
  --retrieval_file <retrieval_pairs.txt> \
  --db_features_path <db_features.h5> \
  --query_features_path <query_features.h5> \
  --preset accuracy \
  --topk 10 \
  --metric_thresholds <dataset_thresholds>
```

Every accepted run must keep `leakage_check.json` with `ok=true`.

### Cambridge ShopFacade

Legacy official-SIFT-COLMAP-map invocation:

```bash
PY="/home/andreas/anaconda3/envs/sam3/bin/python"
CFG="configs/cambridge_shopfacade_official.yaml"
BASE="outputs/cambridge_shopfacade_official"
SPLIT="$BASE/split/split.json"
RETRIEVAL="$BASE/retrieval/pairs-loo-netvlad10.txt"

$PY tools/run_lifted_ablation_suite.py \
  --dataset_name cambridge_shopfacade \
  --config "$CFG" \
  --split_json "$SPLIT" \
  --base_dir "$BASE" \
  --attached_index_sp "$BASE/sp_colmap_attach_r3_scaled" \
  --attached_index_sift "$BASE/sift_detected_r6" \
  --retrieval_file "$RETRIEVAL" \
  --db_features_path "$BASE/sp_features/feats-superpoint-n4096-rmax1600_db.h5" \
  --query_features_path "$BASE/sp_features/feats-superpoint-n4096-rmax1600_queries.h5" \
  --mode_groups main,plm,matcher,feature,sensitivity \
  --topk 10 \
  --sift_match_test l2_ratio \
  --sift_ratio 0.75
```

For the fair hLoc-style ShopFacade comparison, use:

```bash
PY="/home/andreas/anaconda3/envs/sam3/bin/python"
CFG="configs/cambridge_shopfacade_hloc_sp_sg.yaml"
BASE="outputs/cambridge_shopfacade_official"
SPLIT="$BASE/split/split.json"
RETRIEVAL="outputs/hloc_cambridge_sp_sg/ShopFacade/pairs-query-netvlad10.txt"
SP_INDEX="$BASE/sp_colmap_attach_hloc_sp_sg_r2"
HLOC_FEATS="outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5"

$PY -m plm_match.pipelines.lifted_nn_localize \
  --config "$CFG" \
  --dataset_root "/mnt/d/private/pairs/cambridge_landmarks/ShopFacade" \
  --split_json "$SPLIT" \
  --attached_index "$SP_INDEX" \
  --retrieval_file "$RETRIEVAL" \
  --out_dir "$BASE/ablation_suite/results/main_sp_image_obs_hloc_sp_sg_r2_rank_only" \
  --method superpoint_h5 \
  --db_features_path "$HLOC_FEATS" \
  --query_features_path "$HLOC_FEATS" \
  --landmark_match_mode image_obs \
  --topk 10 \
  --query_topk 4096 \
  --ratio_margin 0.10 \
  --min_similarity 0.65 \
  --support_weight 0.0 \
  --point_support_weight 0.0 \
  --rank_weight 0.02 \
  --attach_dist_weight 0.0 \
  --max_cluster_images 5 \
  --max_cluster_seeds 10 \
  --pnp_first_thresh 8.0 \
  --pnp_refine_thresh 4.0 \
  --min_final_inliers 12 \
  --metric_thresholds 0.05/5,0.25/2,0.5/5
```

### Aachen Official Day/Night Dataset

The full dataset is available locally:

```bash
AACHEN_ROOT="/mnt/d/private/pairs/aachen_v1_1"
```

Before running the official Aachen table, build an official-dataset config/split/index from this root and use only the official day/night benchmark split.

### RobotCar Seasons v2

For paper experiments, RobotCar must follow HLoc's pipeline exactly up to the point where PLM starts:

1. HLoc converts `3D-models/all-merged/all.nvm` with `3D-models/overcast-reference.db` into a SIFT seed model.
2. HLoc extracts `superpoint_aachen` features (`feats-superpoint-n4096-r1024.h5`).
3. HLoc matches covisible DB pairs with SuperGlue and triangulates `sfm_superpoint+superglue`.
4. HLoc extracts NetVLAD and writes `pairs-query-netvlad20.txt`.
5. PLM consumes that HLoc SP+SG model, HLoc feature H5, and HLoc retrieval file.
6. Use `robotcar_v2_train.txt` for local ablation metrics because it contains GT poses; avoid `robotcar_v2_test.txt` for local metrics because it has no GT poses.

Required missing file: the local tree currently has no `datasets/RobotCar-Seasons/3D-models/overcast-reference.db`. Download/restore that file before running HLoc's RobotCar pipeline.

Run HLoc first:

```bash
PY="/home/andreas/anaconda3/envs/sam3/bin/python"
DATA="datasets/RobotCar-Seasons"
HLOC_ROOT="/home/phd/Hierarchical-Localization"
HLOC_OUT="/home/phd/plm-match/outputs/hloc_robotcar_seasons_v2"

cd "$HLOC_ROOT"
$PY -m hloc.pipelines.RobotCar.pipeline \
  --dataset "/home/phd/plm-match/$DATA" \
  --outputs "$HLOC_OUT" \
  --num_covis 20 \
  --num_loc 20
```

Then prepare PLM metadata around HLoc outputs:

```bash
cd /home/phd/plm-match
BASE="outputs/robotcar_hloc_plm"

$PY tools/prepare_robotcar_hloc_outputs.py \
  --dataset_root "$DATA" \
  --hloc_outputs "$HLOC_OUT" \
  --out_dir "$BASE" \
  --topk 20
```

This writes:

```text
$BASE/robotcar_hloc_plm.yaml
$BASE/split/split.json
$BASE/split/query_list_with_intrinsics.txt
$BASE/split/query_gt_pose_dir/
```

Build the PLM attachment on HLoc's SP+SG model:

```bash
CFG="$BASE/robotcar_hloc_plm.yaml"
SPLIT="$BASE/split/split.json"
RETRIEVAL="$HLOC_OUT/pairs-query-netvlad20.txt"
HLOC_FEATS="$HLOC_OUT/feats-superpoint-n4096-r1024.h5"

$PY tools/build_local_colmap_attachment.py \
  --config "$CFG" \
  --split_json "$SPLIT" \
  --dataset_root . \
  --out_dir "$BASE/sp_colmap_attach_r3" \
  --method superpoint_h5 \
  --db_features_path "$HLOC_FEATS" \
  --attach_radius_px 3 \
  --max_keypoints 4096 \
  --descriptor_dtype float16 \
  --min_colmap_track_len 3

$PY tools/check_split_leakage.py \
  --split_json "$SPLIT" \
  --attached_index "$BASE/sp_colmap_attach_r3" \
  --retrieval_file "$RETRIEVAL" \
  --out "$BASE/ablation_suite/results/main_sp_image_obs/leakage_check.json"

$PY -m plm_match.pipelines.lifted_nn_localize \
  --config "$CFG" \
  --dataset_root . \
  --split_json "$SPLIT" \
  --attached_index "$BASE/sp_colmap_attach_r3" \
  --retrieval_file "$RETRIEVAL" \
  --out_dir "$BASE/ablation_suite/results/main_sp_image_obs" \
  --method superpoint_h5 \
  --db_features_path "$HLOC_FEATS" \
  --query_features_path "$HLOC_FEATS" \
  --landmark_match_mode image_obs \
  --topk 20 \
  --query_topk 4096 \
  --ratio_margin 0.10 \
  --min_similarity 0.65 \
  --support_weight 0.03 \
  --point_support_weight 0.02 \
  --rank_weight 0.02 \
  --attach_dist_weight 0.0 \
  --max_cluster_images 5 \
  --max_cluster_seeds 10 \
  --pnp_first_thresh 8.0 \
  --pnp_refine_thresh 4.0 \
  --min_final_inliers 12 \
  --metric_thresholds 0.25/2,0.5/5,5/10
```

Fallback only: `tools/prepare_robotcar_seasons.py` can build a direct NVM-based PLM setup without HLoc's DB, but do not use that path for the paper table because it does not reproduce HLoc's SP+SG triangulated reference model.

Disk policy for RobotCar: keep `split/`, generated config, `prepare_summary.json`, `run_summary.json`, `metrics.json`, `leakage_check.json`, and `command.txt`. Keep HLoc features/retrieval/model and the PLM attachment index while ablations are still being run. Delete per-run `artifacts/` and obsolete probe outputs after summaries are archived.

---

## 8. Prerequisites Checklist

### Still to build / set up

- [x] **Cambridge HLoc-map full suites:** Great Court, King's, Hospital, ShopFacade, and St Mary's have HLoc-map PLMLoc outputs available where used in the paper tables.
- [ ] **Aachen official Day/Night:** config, official split, retrieval, SP features, SP index from `/mnt/d/private/pairs/aachen_v1_1`
- [ ] **Aachen official SIFT index:** `tools/build_local_colmap_attachment.py --method sift --sift_attach_mode detected_nearest --attach_radius_px 6`
- [x] **7-Scenes chess retrieval:** official DenseVLAD top-10 retrieval available and used
- [x] **7-Scenes chess HLoc SP+SG SfM:** HLoc SfM build complete
- [x] **7-Scenes chess HLoc-SfM PLMLoc attachment/suite:** HLoc-SfM attachment and ablation suite complete
- [x] **7-Scenes chess SIFT/RootSIFT indexes:** RGB-D attachment complete
- [x] **7-Scenes chess RGB-D full ablation suite:** SP, point modes, SG/LG, SIFT, RootSIFT complete
- [x] **TUM RGB-D converter:** `tools/prepare_tum_rgbd_split.py`
- [x] **TUM RGB-D fr1 desk:** converted, SP features/retrieval/memory built, full SP ablation suite complete
- [ ] **TUM RGB-D fr1 room:** convert, extract SP, build RGB-D memory, run PLMLoc ablations
- [ ] **CMU Seasons:** implement loader/prep path, build split, extract SP features, build SP + SIFT index, build config
- [x] **RobotCar direct-NVM fallback prep:** `tools/prepare_robotcar_seasons.py`
- [x] **RobotCar HLoc-output adapter:** `tools/prepare_robotcar_hloc_outputs.py`
- [ ] **RobotCar HLoc setup:** run HLoc RobotCar, build PLMLoc index from HLoc outputs

### Smoke test before full official Aachen runs

```bash
for mode in image_obs point_mean point_memory point_memory_support; do
  $PY -m plm_match.pipelines.lifted_nn_localize \
    --config $CFG --split_json $SPLIT \
    --attached_index $BASE/sp_colmap_attach_r3 \
    --retrieval_file $RETRIEVAL \
    --out_dir $BASE/debug_${mode}_q5 \
    --method superpoint_h5 \
    --db_features_path $BASE/sp_features/feats-superpoint-n4096-rmax1600_db.h5 \
    --query_features_path $BASE/sp_features/feats-superpoint-n4096-rmax1600_queries.h5 \
    --landmark_match_mode $mode \
    --topk 20 --query_topk 4096 \
    --point_memory_max_obs 4 --point_memory_batch_size 128 \
    --max_queries 5
done
```

---

## 9. Paper Table Template

### Cambridge Landmarks (median cm / °)

| Method | Map source | Court | King's | Hospital | Shop | St. Mary's | Avg |
|---|---|---|---|---|---|---|---|
| AS (SIFT) | SfM | 24/0.1 | 13/0.2 | 20/0.4 | 4/0.2 | 8/0.3 | 14/0.2 |
| hLoc (SP+SG) | SfM | 16/0.1 | 12/0.2 | 15/0.3 | 4/0.2 | 7/0.2 | 11/0.2 |
| HLoc-Lifted (SP+SG, ours) | SfM | — | 12.4/0.2 | — | — | — | — |
| HLoc-Lifted (SP+LG, ours) | SfM | — | 11.8/0.2 | — | — | — | — |
| **PLMLoc (SIFT)** | SfM | — | — | — | **5.5/0.2** | — | — |
| **PLMLoc (RootSIFT)** | SfM | — | — | — | **5.2/0.2** | — | — |
| **PLMLoc (SP, point_mean)** | SfM | — | 15.8/0.3 | — | **5.3/0.3** | — | — |
| **PLMLoc (SP, point_memory_support)** | SfM | — | 15.5/0.2 | — | **4.7/0.2** | — | — |
| **PLMLoc (SP, image_obs)** | SfM | — | 14.5/0.2 | — | **4.7/0.2** | — | — |

### Aachen Day/Night v1.1 (% success)

| Method | Day 0.25/2 | Day 0.5/5 | Day 5/10 | Night 0.25/2 | Night 0.5/5 | Night 5/10 |
|---|---|---|---|---|---|---|
| hLoc (SP+SG) | 83.4 | 93.4 | 99.7 | — | — | — |
| **PLMLoc (SP, image_obs)** | — | — | — | — | — | — |
| **PLMLoc (SP, point_mean)** | — | — | — | — | — | — |
| **PLMLoc (SP, point_memory)** | — | — | — | — | — | — |

### 7-Scenes — % at (5cm, 5°)

| Method | Map source | Chess | Fire | Heads | Office | Pumpkin | RedKitchen | Stairs | Avg |
|---|---|---|---|---|---|---|---|---|---|
| hLoc (SP+SG) | SfM | — | — | — | — | — | — | — | — |
| **PLMLoc (SP, image_obs)** | HLoc SP+SG SfM | — | — | — | — | — | — | — | — |
| **PLMLoc (SP, point_memory)** | HLoc SP+SG SfM | — | — | — | — | — | — | — | — |
| HLoc-Lifted (SP+SG, ours) | HLoc SP+SG SfM | — | — | — | — | — | — | — | — |
| HLoc-Lifted (SP+SG, ours) | RGB-D | 76.4 | — | — | — | — | — | — | — |
| HLoc-Lifted (SP+LG, ours) | RGB-D | 82.7 | — | — | — | — | — | — | — |
| **PLMLoc (SP, image_obs)** | RGB-D | **86.2** | — | — | — | — | — | — | — |
| **PLMLoc (SIFT, image_obs)** | RGB-D | 83.1 | — | — | — | — | — | — | — |
| **PLMLoc (RootSIFT, image_obs)** | RGB-D | 82.8 | — | — | — | — | — | — | — |
| **PLMLoc (SP, point_mean)** | RGB-D | 46.4 | — | — | — | — | — | — | — |
| **PLMLoc (SP, point_memory)** | RGB-D | 42.4 | — | — | — | — | — | — | — |

## 10. Priority Order

1. **Official Aachen setup** from `/mnt/d/private/pairs/aachen_v1_1` — config, split, retrieval, SP features, SP index
2. **Official Aachen smoke test** (5 queries, all 4 PLM modes) — verifies the full-dataset path
3. **Aachen official full run** (`main`, `plm`, `sensitivity`) — primary outdoor benchmark table
4. **TUM RGB-D fr1 room** — second small robotics-friendly RGB-D sanity sequence
5. **Aachen SIFT index + feature ablation** — complete feature comparison on outdoor SfM
6. **HLoc COLMAP-attached baseline validation runs** — SG-lifted comparison inside the same lifting/eval path
9. **CMU + RobotCar loaders/setup and runs** — large-scale outdoor generalization
