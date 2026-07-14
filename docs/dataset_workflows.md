# Dataset Workflows and Map Provenance

This document describes how the large-scale PLMLoc experiments are assembled
for Aachen Day-Night v1.1, RobotCar Seasons v2, and Extended CMU Seasons. It
separates four choices that must not be inferred from a method or submission
filename:

1. **Reference geometry**: the COLMAP/NVM cameras, registered images, 3D
   points, and tracks used by localization.
2. **Local feature family**: SuperPoint, ALIKED, or another detector and
   descriptor used to represent map observations and query keypoints.
3. **Global retrieval**: MixVPR, MegaLoc, NetVLAD, or SALAD pairs used to select
   database images.
4. **Point-memory selection**: fixed `diverse_desc`, `adaptive_cover`,
   `adaptive_cover_farthest`, or `adaptive_cover_v2`.

In particular, `FEATURE=aliked` controls local feature extraction. It does not
by itself imply that the 3D map was triangulated with ALIKED+LightGlue.

## Map Construction Terms

### Legacy or reference-model attachment

The cameras, poses, 3D coordinates, and tracks come from an existing reference
model. New local descriptors are attached to its existing 2D observations.
The large-scale wrappers use:

```text
tools/build_sp_colmap_attachment.py
  --attach_mode detected_nearest
  --colmap_feature_index_mode nearest
  --attach_radius_px 3
```

This can attach either SuperPoint or ALIKED descriptors. It does not
retriangulate the map, and LightGlue/SuperGlue matches do not define its 3D
points.

### Feature-native triangulation

The reference model supplies fixed cameras and map-image poses, but the 3D
points and tracks are triangulated again from feature-specific database
matches. The native workflow uses:

```text
tools/build_native_feature_sfm.py
```

For ALIKED, database images are matched with LightGlue and the output model is
normally named `sfm_aliked_lightglue`. Because the COLMAP point observations
come directly from the ALIKED feature indices, PLMLoc builds the memory with:

```text
--attach_mode index_aligned
--colmap_feature_index_mode index
```

The equivalent SuperPoint path can triangulate `sfm_superpoint+superglue` or a
SuperPoint+LightGlue model, depending on the selected matcher configuration.

## Provenance Summary

| Dataset and run family | Reference geometry used by PLMLoc | Local memory construction | Native feature SfM? |
|---|---|---|---|
| Aachen standard SP submissions | Official `3D-models/aachen_v_1_1` | SuperPoint descriptors attached to existing observations | No |
| Aachen standard ALIKED/adaptive submissions | Official `3D-models/aachen_v_1_1` | ALIKED descriptors attached with nearest detection | No |
| Aachen PixLoc night diagnostic | PixLoc `sfm_superpoint+superglue` | SuperPoint descriptors indexed against that model | Yes, externally supplied map |
| RobotCar generic adaptive wrappers | Converted `all-merged/all.nvm` reference model | Selected feature attached to existing observations | No |
| RobotCar native ALIKED submission family | Fixed-pose RobotCar reference followed by ALIKED+LightGlue retriangulation | Index-aligned ALIKED observations | Yes |
| RobotCar HLoc SP+SG map family | Fixed-pose RobotCar reference followed by SP+SG retriangulation | SuperPoint observations from the HLoc model | Yes |
| Extended CMU generic adaptive wrapper | Provided per-slice `sparse` model or converted slice NVM | Selected feature attached to existing observations | No |
| Extended CMU ALIKED+LG covis30 submission | Per-slice ALIKED+LightGlue retriangulation | Index-aligned ALIKED observations | Yes |
| Extended CMU SP+SG covis30 submission | Per-slice SP+SG retriangulation | SuperPoint observations from the native model | Yes |

## Aachen Day-Night v1.1

### Standard 1,015-query submission workflow

The standard runner is:

```text
scripts/run_aachen_adaptive_cover_submission.sh
  -> scripts/run_large_scale_adaptive_cover.sh
```

The official split contains 824 day and 191 night queries. The wrapper creates
the day/night lists from the preserved full split and requires the following
reference model:

```text
<aachen-root>/3D-models/aachen_v_1_1
```

Both Aachen configs point to that model:

```text
configs/aachen_v1_1_day_refactor.yaml
configs/aachen_v1_1_night_refactor.yaml
```

#### Local features

For `FEATURE=aliked`, the auxiliary HLoc stage is configured as
`aliked_lightglue`; for `FEATURE=superpoint`, it is configured as
`superpoint_superglue`. This stage supplies H5 keypoints and descriptors for
the database and queries. It does not replace the official reference model.

The PLMLoc attachment is built from the official model with:

```text
attach_mode: detected_nearest
COLMAP feature-index mode: nearest
attachment radius: 3 px by default
minimum reference track length: 3
maximum reference point error: 4 px
```

The preserved ALIKED attachment contains:

```text
database images:       4,328
reference landmarks:   314,169
attached observations: 867,398
```

Therefore the standard Aachen labels mean:

```text
SP:     official Aachen geometry + attached SuperPoint descriptors
ALIKED: official Aachen geometry + nearest-attached ALIKED descriptors
```

They do not mean native SP+SG or ALIKED+LightGlue triangulation.

#### Retrieval and localization

Retrieval pairs are independent of map construction. `AACHEN_PLM_RETRIEVAL_FILE`
can select MixVPR, MegaLoc, NetVLAD, or another compatible pair file. PLMLoc
then restricts candidate landmarks to the retrieved database images, selects
the configured observation memory, performs lifted descriptor matching, and
runs PnP.

The principal fixed-memory configuration has been:

```text
local feature:       ALIKED, resize_max=1024, max_keypoints=4096
retrieval:           MixVPR-5 or MegaLoc-5
selector:            diverse_desc
maximum observations: 8
PnP:                 16 px first/refine, minimum 10 final inliers
pose guidance:       off
```

`diverse_desc` is fixed-budget descriptor farthest-point sampling. Adaptive
cover parameters such as gain or `K_min` do not affect it.

#### Submission packaging

Day and night pose files are concatenated, directory prefixes are removed from
query names, and the final file is written under:

```text
<runs>/visual_localization_submissions/adaptive_cover/
```

Before upload, verify that the file contains 1,015 unique query names. Local
coverage and inlier counts are diagnostics only because official query poses
are hidden.

### PixLoc SP+SG-map diagnostic

The one feature-native Aachen exception in this workspace is:

```text
outputs/aachen_night_pixloc_hloc_plmloc/
```

Its config points to the externally supplied PixLoc map:

```text
pixloc_bundle/sfm_superpoint+superglue
```

This experiment used 98 night queries, HLoc SP+SG with NetVLAD-10, and PLMLoc
SuperPoint with MixVPR-10 and fixed diverse-16 memory. It produced 98/98 HLoc
poses and 94/98 PLMLoc poses. It is a coverage and pose-agreement diagnostic,
not one of the full 1,015-query benchmark submissions.

### Native Aachen experiment not yet completed

A feature-native Aachen comparison would need to use the official model only
as the fixed-pose reference, generate DB-DB covisibility pairs, triangulate a
new SP+SG or ALIKED+LightGlue map with `build_native_feature_sfm.py`, validate
COLMAP/H5 feature-index alignment, and localize both day and night against that
new model. None of the standard Aachen portal rows currently has this
provenance.

## RobotCar Seasons v2

### Dataset preparation

`tools/prepare_robotcar_seasons.py` converts:

```text
3D-models/all-merged/all.nvm
```

to a COLMAP reference model and creates map/query lists. The official v2 test
list contains 1,872 queries: 1,443 day and 429 night. Final benchmark query
names must come exactly from `robotcar_v2_test.txt`; submitting public-train
names produces an invalid zero-overlap submission.

### Generic attachment workflow

The current generic adaptive runners, including
`run_robotcar_megaloc5_adaptive_cover_submission.sh`, use the converted NVM
model directly. They extract ALIKED or SuperPoint descriptors and call
`build_sp_colmap_attachment.py` against that model. These runs are legacy
attachment runs even when their result name contains `aliked` or `megaloc`.

MegaLoc and MixVPR only determine which database images are retrieved. They do
not change the reference map.

### Native ALIKED+LightGlue workflow

The feature-native RobotCar path is built in two distinct stages:

1. Use the converted RobotCar model as a fixed-camera/fixed-pose reference.
2. Extract ALIKED database/query features, generate DB-DB covisibility pairs,
   match DB pairs with LightGlue, and triangulate `sfm_aliked_lightglue`.

The completed native map in the historical workspace registered all 20,862
map images and contained 6,125,748 ALIKED-triangulated points. The aligned PLM
index uses native feature indices, not nearest attachment.

The orchestration tool is:

```text
tools/run_native_aliked_plm_pipeline.py
```

For an existing native model, use `--skip_sfm` and point `--native_sfm_dir`,
`--features_dir`, and `--attached_index` at the same feature-native artifact
family. Mixing an index from the converted NVM model with a native ALIKED model
is invalid.

### Native SuperPoint+SuperGlue workflow

The HLoc RobotCar runner creates a feature-native SP+SG model by extracting
SuperPoint features, matching covisible DB images with SuperGlue, and
triangulating a new reference model. PLMLoc uses that geometry only when its
config/model and attachment index explicitly point to the HLoc model. Merely
running HLoc before the generic PLMLoc wrapper does not prove that PLMLoc used
the HLoc map; confirm the saved `model_path` and `attached_index`.

### Retrieval, localization, and submission

For the official hidden test set:

1. Extract only the official test-query local features while reusing compatible
   DB features.
2. Generate retrieval pairs against map images.
3. Localize with the matching native or attached index.
4. Convert predictions using the exact names in `robotcar_v2_test.txt`.
5. Verify 1,872 expected names, no unknown names, and report missing poses.

Missing test poses may be omitted by the pure PLMLoc submission, but they cap
benchmark recall. Filling them with another method changes the method and must
be reported as a fallback system.

## Extended CMU Seasons

### Official slice protocol

The workspace uses the 14 benchmark slices:

```text
slice2, slice3, slice4, slice5, slice6,
slice13, slice14, slice15, slice16, slice17,
slice18, slice19, slice20, slice21
```

Their query lists contain 56,613 official queries in total. Results are merged
across slices and reported by urban, suburban, and park conditions; they are
not reduced to a day/night weighted triplet.

### Generic adaptive attachment workflow

`scripts/run_cmu_adaptive_cover_submission.sh` dispatches each slice through
`run_large_scale_adaptive_cover.sh`. In slice layout it uses:

```text
<cmu-root>/<slice>/sparse
```

as the reference model. In NVM layout it converts the corresponding slice NVM.
It then extracts ALIKED or SuperPoint features, generates MixVPR pairs, and
attaches descriptors to the existing slice observations using nearest
attachment. This current adaptive wrapper does not build a native ALIKED+LG
map.

### Native ALIKED+LightGlue covis30 workflow

The submitted ALIKED+LG CMU result used a different map family. For every
slice, it:

1. Reduced the reference model to map images.
2. Generated 30-neighbor DB covisibility pairs.
3. Extracted ALIKED features.
4. Matched the DB pairs with LightGlue.
5. Triangulated a per-slice `sfm_aliked_lightglue` model.
6. Built an index-aligned ALIKED point memory.
7. Localized queries using MixVPR-5, maximum eight observations, and PnP
   16/16 with minimum 10 inliers.

That native workflow produced the recorded official result:

```text
urban:    94.0 / 97.0 / 99.1
suburban: 95.2 / 97.6 / 99.6
park:     85.3 / 89.9 / 96.2
```

### Native SP+SG covis30 workflow

The SP baseline likewise uses per-slice SuperPoint+SuperGlue triangulation,
not SuperPoint attachment to the provided sparse model. Its model family and
feature H5 files must remain aligned per slice. The submitted max-8 result was:

```text
urban:    93.4 / 96.2 / 98.7
suburban: 88.1 / 90.8 / 97.0
park:     77.0 / 80.1 / 88.9
```

### Slice completion and packaging

Each slice is complete only when its expected `hloc_results.txt` exists and
contains the slice query set. The final packager concatenates all selected
slices, strips directory prefixes as required by the benchmark, and writes a
summary containing expected, submitted, and missing counts. A valid full file
should contain 56,613 query poses unless the method failed to return some.

Do not concatenate results from different map families under one method label
without documenting the mixture. In particular, an adaptive result produced
from provided sparse models is not directly the same map setup as the native
ALIKED+LG covis30 result.

## Reproducibility Checklist

For every run, save or inspect the following before assigning a method label:

1. `config*.yaml`: authoritative `dataset.model_path`.
2. Native SfM summary, if present: registered images, 3D points, matcher, and
   reference model.
3. Attachment `summary.json`: `method`, `attach_mode`, feature-index mode,
   landmark count, and attached observation count.
4. Retrieval summary: method, top-k, number of queries, and pair count.
5. Localization `command.txt`: selector, observation budget, PnP thresholds,
   pose guidance, feature H5 paths, and attachment index.
6. Localization `run_summary.json`: query count, success rate, runtime, memory
   statistics, and inlier diagnostics.
7. Submission summary: expected names, submitted names, missing names, and
   duplicates.

Useful provenance checks are:

```bash
rg -n 'model_path|sfm_dir' <run-config.yaml>
rg -n -- '--attached_index|--db_features_path|--point_memory_obs_select' <result>/command.txt
python -m json.tool <attachment>/summary.json | less
python -m json.tool <native-sfm>/native_feature_sfm_summary.json | less
```

Treat `command.txt`, config paths, and summaries as authoritative. Directory
and submission filenames are descriptive labels and may contain legacy tokens
that were inactive during localization.

