# PLM-Match diagnosis and refactor plan

## Linked Aachen root

Real-data runs in this workspace should use the repo-local symlink:

- `/home/phd/plm-match/datasets/aachen_v1_1 -> /mnt/d/private/pairs/aachen_v1_1`

The Aachen configs and experiment manifests that are meant to run on the public
dataset should point to that linked root so sweeps do not silently use stale
machine-specific paths.

## What the provided run says

From the attached summary JSON for `query/day/nexus4/IMG_20140520_182846.jpg -> db/1410.jpg`:

- tentative matches visualized: **55**
- PnP inliers: **5**
- anchor distinctiveness mean: **0.395**
- anchor stability mean: **0.956**
- candidate group images: **1** (`db/1410.jpg` only)
- materialized landmarks: **408**
- runtime breakdown:
  - anchor extraction: **0.147 s**
  - retrieval: **0.050 s**
  - materialize selected: **3.995 s**
  - match query: **4.209 s**

Interpretation:

1. The bottleneck is **not retrieval**; it is candidate materialization and query scoring.
2. The failure is **not anchor instability**; anchors are stable but **not distinctive enough**.
3. With `candidate_group_images = 1`, the system is barely exploiting a persistent **multi-view** landmark memory. On a repetitive facade this collapses into many visually valid but geometrically wrong correspondences.

## The biggest mathematical fix

The residual in the paper penalizes distance only **orthogonal** to the landmark subspace. That is too weak on repetitive structures.

Use a **PPCA / low-rank Gaussian likelihood** instead:

- `z = q - mu_k`
- `a = U_k^T z`
- penalize both:
  - displacement *inside* the learned subspace with `sum_i a_i^2 / lambda_i`
  - displacement *outside* the subspace with `||z - U_k U_k^T z||^2 / sigma_perp^2`

This prevents descriptors that lie far away along the same broad subspace from all looking equally valid.

## The biggest system fix

Do **not** rely on DINO alone as the final local descriptor.

Recommended split:

1. **Foundation descriptor** (DINO / FM) for:
   - retrieval
   - anchor proposal
   - coarse candidate landmark shortlist

2. **True local descriptor head** for final verification:
   - XFeat patch head
   - or DISK / ALIKED / DoGHardNet + LightGlue on selected views
   - or a small learned/local patch descriptor stored per landmark observation

The local head only needs to run on the shortlisted candidate set, not globally.

## Aachen ablation plan

Run on Aachen with the same retrieval protocol as HLoc and compare:

1. DINO-only mean scoring
2. DINO-only PPCA scoring
3. DINO shortlist + fine local patch descriptor reranking
4. multi-view candidate group sizes: 1 / 5 / 10 / 20 images
5. final-layer DINO vs intermediate-layer DINO vs mixed-layer descriptor

Report:
- localized % at `(0.25m, 2°)`, `(0.5m, 5°)`, `(5m, 10°)`
- runtime per query
- average candidate count after each stage
- inlier count distribution

## Efficiency checklist

- cache candidate clusters from the top retrieved image set;
- store landmark means/bases in `fp16`;
- batch anchors and gather landmark tensors once per batch;
- replace Python loops with batched matrix operations;
- keep rank small (`r=4` or `r=8`);
- use multi-image covisible groups instead of a single image;
- materialize fine descriptors only for the shortlisted candidates.

## Implementation checklist

### Phase 0: Real-data wiring

- [x] Link Aachen into the repo at `/home/phd/plm-match/datasets/aachen_v1_1`.
- [x] Repoint the Aachen day configs and experiment manifests to the linked root.
- [x] Smoke-test the dataset adapter and map/query frame discovery on the linked root.
  - Verified with `configs/aachen_v1_1_day_precise.yaml`:
    - map frames: `4328`
    - day queries: `824`
    - first query: `query/day/nexus4/IMG_20140520_182846.jpg`
- [x] Record the exact Python/env command that will be used for PLM runs.
  - `source /home/andreas/anaconda3/etc/profile.d/conda.sh && conda activate sam3 && cd /home/phd/plm-match`

### Phase 1: Baseline on real Aachen before changing the matcher

- [x] Run a real-data Aachen **minitest** in `sam3` to validate the end-to-end PLM path before committing to full-day sweeps.
  - Command:
    - `source /home/andreas/anaconda3/etc/profile.d/conda.sh && conda activate sam3 && cd /home/phd/plm-match && TORCH_HOME=/tmp/torch-hub python -m plm_match.pipelines.hloc_localize --config configs/aachen_minitest.yaml --dataset_root /home/phd/plm-match/datasets/aachen_v1_1 --out_dir /tmp/plm_match_outputs/aachen_minitest_run --disable-query-cache`
  - Result on real Aachen subset (`10` queries / `75` map frames):
    - success rate: `0.10`
    - mean retrieval time: `0.020 s`
    - mean materialization time: `5.118 s`
    - mean scoring time: `0.003 s`
    - mean query time: `5.537 s`
    - landmarks in compact store: `48030`
  - This already confirms the original diagnosis on real data: retrieval is cheap, scoring is cheap, and candidate materialization dominates.
- [ ] Run `configs/aachen_v1_1_day_precise.yaml` unchanged on the linked dataset.
- [ ] Run `configs/aachen_v1_1_day.yaml` unchanged on the linked dataset.
- [ ] Save per-query debug JSON and extract:
  - materialization time
  - scoring time
  - candidate group size
  - materialized landmark count
  - tentative matches vs inliers
- [ ] Confirm the failure mode on real Aachen examples before refactoring further.

### Phase 2: Multi-view verification ablation with current scorer

- [x] Prepare a real-data grouped-verification sweep manifest for `hloc.verify_image_batch_size = 1 / 5 / 10 / 20`.
  - Manifest:
    - `experiments/aachen_minitest_group_verify_gpu.yaml`
  - Status:
    - prepared against the real Aachen minitest cache
    - intentionally left **unrun** here so results do not get mixed with your own later test session
- [ ] Sweep `hloc.verify_image_batch_size` over `1 / 5 / 10 / 20`.
- [ ] Keep the current scorer fixed while varying only grouped verification.
- [ ] Compare runtime, candidate counts, and inlier counts per query.
- [ ] Decide whether grouped verification alone already removes the worst facade failures.

### Phase 3: Integrate PPCA scoring into the real PLM pipeline

- [ ] Extend `Landmark` / compact store data to carry:
  - retained eigenvalues
  - `sigma_perp2`
  - support count
  - compact view-envelope statistics
- [ ] Compute those statistics offline during compact-store construction, not at query time.
- [ ] Add a matcher backend flag for `residual` vs `ppca`.
- [ ] Replace the residual-only score with PPCA scoring in the real matcher path.
- [ ] Keep output compatible with `Match3D2D`, current PnP, and debug accounting.

### Phase 4: Vectorize the hot path

- [ ] Replace the Python anchor-by-anchor scoring loop with batched tensor scoring.
- [ ] Preserve current thresholds:
  - `min_cosine_sim`
  - `min_score`
  - `ratio_margin`
  - uniqueness pruning
- [ ] Keep grouped verification and stage-wise early stopping intact.
- [ ] Re-measure materialization time vs scoring time after vectorization.

### Phase 5: Fine local descriptor reranking

- [ ] Do not store only one fine descriptor per landmark.
- [ ] Store fine descriptors per observation or per selected support view.
- [ ] Run the fine local head only on the shortlisted candidates after coarse gating.
- [ ] Compare:
  - DINO mean
  - DINO PPCA
  - DINO shortlist + fine reranking

### Phase 6: Aachen experiment manifest updates

- [ ] Add grouped-verification sweeps to `experiments/aachen_day_sweep.yaml`.
- [ ] Add scorer sweeps for `residual` vs `ppca`.
- [ ] Add rank sweeps only after the PPCA backend is live.
- [ ] Keep the current `rank=0` precise config as the strongest pre-PPCA baseline.

### Phase 7: Environment gaps to close

- [ ] PLM path: verify DINO can load in the chosen env and note whether it is CPU or CUDA.
- [ ] HLoc baseline path: install `h5py` and `pycolmap` in the env used for DISK-LightGlue.
- [ ] If Torch stays CPU-only in `sam3`, note that runtime comparisons must be labeled CPU.
