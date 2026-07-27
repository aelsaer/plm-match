# Table 7 MegaLoc5 macro values

Updated on 2026-07-27 from the saved VisualLocalization benchmark export at:

`/home/cvdp/.codex/attachments/4ad294fb-989b-4ef6-bc41-d186d8731419/pasted-text.txt`

Scope: paper-facing long-term visual localization rows, MegaLoc top-5 retrieval, HLoc-style MNN point-memory matching. Dataset macro values are unweighted averages over benchmark splits.

## Paper rows

| Method | Aachen macro | RobotCar macro | CMU macro | Overall macro |
|---|---:|---:|---:|---:|
| PLMLoc (SuperPoint+MNN) | 77.0 / 89.6 / 97.1 | 46.6 / 77.0 / 95.7 | 86.4 / 89.4 / 95.2 | 70.0 / 85.3 / 96.0 |
| PLMLoc (ALIKED+MNN) | 82.8 / 93.2 / 99.4 | 56.5 / 91.3 / 99.4 | 92.0 / 95.4 / 98.8 | 77.1 / 93.3 / 99.2 |

## Source rows

### SuperPoint+MNN

| Dataset | Source benchmark row | Split scores | Macro |
|---|---|---|---:|
| Aachen Day-Night v1.1 | `Aachen_v1_1_eval_PLMLoc_superpoint_native_sp_sg_fullmap_sfm_megaloc5_adaptive_cover_v2_s080_g030_k12` | day: 87.9 / 95.3 / 98.8; night: 66.0 / 83.8 / 95.3 | 77.0 / 89.6 / 97.1 |
| RobotCar Seasons v2 | `PLMLoc` | day: 64.7 / 94.7 / 100.0; night: 28.4 / 59.2 / 91.4 | 46.6 / 77.0 / 95.7 |
| CMU Seasons Extended | `CMU_eval_PLMLoc_superpoint_native_sp_sg_covis30_sp1600_megaloc5_farthest8_diverse_desc_obs8_pnp16_mi` | urban: 93.7 / 96.4 / 98.8; suburban: 88.1 / 90.9 / 97.0; park: 77.3 / 80.9 / 89.9 | 86.4 / 89.4 / 95.2 |

Note: the RobotCar SuperPoint+MNN value intentionally uses the submitted `PLMLoc` SP+SG native-SfM row requested for Table 7.

### ALIKED+MNN

| Dataset | Source benchmark row | Split scores | Macro |
|---|---|---|---:|
| Aachen Day-Night v1.1 | `Aachen_v1_1_eval_PLMLoc_aliked_native_fullmap_sfm_megaloc5_farthest8_diverse_desc_r1024_obs8_pnp16_m` | day: 88.5 / 95.8 / 99.3; night: 77.0 / 90.6 / 99.5 | 82.8 / 93.2 / 99.4 |
| RobotCar Seasons v2 | `RobotCar_eval_PLMLoc_aliked_native_lg_megaloc5_farthest8_diverse_desc_r1024_obs8_pnp16_min10_v2_test` | day: 65.0 / 95.0 / 100.0; night: 48.0 / 87.6 / 98.8 | 56.5 / 91.3 / 99.4 |
| CMU Seasons Extended | `CMU_eval_PLMLoc_aliked_native_lg_covis30_megaloc5_farthest8_diverse_desc_r1024_k1_12_gain0005_pnp16_` | urban: 94.1 / 97.3 / 99.3; suburban: 95.6 / 98.0 / 99.8; park: 86.2 / 90.8 / 97.4 | 92.0 / 95.4 / 98.8 |

