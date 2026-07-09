# LOO-Aachen Benchmark Protocol

This benchmark uses Aachen database images as held-out queries. Their COLMAP
poses are known, so this is local ground truth, not pseudo-GT.

## Fairness Rules

1. Use one fixed split for every method.
2. Remove held-out images from the reference/map image set.
3. Use the same NetVLAD retrieval file for HLoc and PLM variants.
4. FuseLoc must be run from the official repository:
   `https://github.com/sontung/descriptor-disambiguation`.
5. The old `oracle_pose` LOO mode is only a diagnostic; do not use it in the
   benchmark table.

## One-Command Suite

For the clean publication table, use `--publication_core`. This runs only:

```text
HLoc SP+SG
FuseLoc
PLM-LocalMemory
PLM + SG verifier
PLM + SG verifier + SimpleGraph
```

Recommended fresh 500-image publication run:

```bash
cd /home/phd/plm-match

TORCH_HOME=/tmp/torch-hub /home/andreas/anaconda3/envs/sam3/bin/python \
  tools/run_loo_benchmark_suite.py \
  --publication_core \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --out_root outputs/loo_aachen_publication_core \
  --num_queries 500 \
  --selection stride \
  --min_observations 100 \
  --topk 50 \
  --fuseloc_root /home/phd/descriptor-disambiguation \
  --fuseloc_python /home/phd/descriptor-disambiguation/.pixi/envs/default/bin/python
```

If FuseLoc is still not stable in the current environment, run the same clean
suite without it first:

```bash
TORCH_HOME=/tmp/torch-hub /home/andreas/anaconda3/envs/sam3/bin/python \
  tools/run_loo_benchmark_suite.py \
  --publication_core \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --out_root outputs/loo_aachen_publication_core_no_fuseloc \
  --num_queries 500 \
  --selection stride \
  --min_observations 100 \
  --topk 50 \
  --skip_fuseloc
```

Each run writes:

```text
outputs/<out_root>/loo_aachen_500/summary.md
```

The broader development suite below keeps extra ablations such as PLM-Direct,
PLM-Memory, map-only graph, and correspondence graph.

```bash
cd /home/phd/plm-match

TORCH_HOME=/tmp/torch-hub /home/andreas/anaconda3/envs/sam3/bin/python \
  tools/run_loo_benchmark_suite.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --out_root outputs/loo_aachen_benchmark \
  --num_queries 50 \
  --selection stride \
  --min_observations 100 \
  --fuseloc_root /tmp/descriptor-disambiguation
```

For the 500-image split:

```bash
TORCH_HOME=/tmp/torch-hub /home/andreas/anaconda3/envs/sam3/bin/python \
  tools/run_loo_benchmark_suite.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --out_root outputs/loo_aachen_benchmark \
  --num_queries 500 \
  --selection stride \
  --min_observations 100 \
  --fuseloc_root /tmp/descriptor-disambiguation
```

Each run writes:

```text
outputs/loo_aachen_benchmark/loo_aachen_<N>/summary.md
```

with:

```text
Method              0.25m/2deg   0.5m/5deg   5m/10deg   Median err   Runtime/query
HLoc SP+SG
FuseLoc
PLM-Direct
PLM-Memory
PLM + SG verifier
```

## Manual Steps

Prepare the split:

```bash
python tools/prepare_aachen_loo_split.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --out_dir outputs/loo_aachen_benchmark/loo_aachen_50/split \
  --num_queries 50 \
  --selection stride \
  --min_observations 100
```

Generate fair NetVLAD retrieval:

```bash
python tools/generate_loo_netvlad_retrieval.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --split_json outputs/loo_aachen_benchmark/loo_aachen_50/split/split.json \
  --out_dir outputs/loo_aachen_benchmark/loo_aachen_50/retrieval \
  --topk 50
```

Run HLoc:

```bash
python tools/run_hloc_loo.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --split_json outputs/loo_aachen_benchmark/loo_aachen_50/split/split.json \
  --retrieval_file outputs/loo_aachen_benchmark/loo_aachen_50/retrieval/pairs-loo-netvlad50.txt \
  --out_dir outputs/loo_aachen_benchmark/loo_aachen_50/hloc_sp_sg
```

Run official FuseLoc:

```bash
python tools/run_fuseloc_loo.py \
  --fuseloc_root /tmp/descriptor-disambiguation \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --split_json outputs/loo_aachen_benchmark/loo_aachen_50/split/split.json \
  --out_dir outputs/loo_aachen_benchmark/loo_aachen_50/fuseloc
```

Run PLM-Direct:

```bash
python tools/run_db_leave_one_out.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --split_json outputs/loo_aachen_benchmark/loo_aachen_50/split/split.json \
  --out_dir outputs/loo_aachen_benchmark/loo_aachen_50/plm_direct \
  --retrieval_mode file \
  --retrieval_file outputs/loo_aachen_benchmark/loo_aachen_50/retrieval/pairs-loo-netvlad50.txt \
  --override matching.pairwise_verifier.enabled=false \
  --override matching.coherence_radius_m=null \
  --override matching.db_pose_radius_m=null \
  --override landmarks.min_staticness=0.0 \
  --override landmarks.harris_weight=0.0 \
  --override anchors.harris_weight=0.0
```

Run PLM-Memory:

```bash
python tools/run_db_leave_one_out.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --split_json outputs/loo_aachen_benchmark/loo_aachen_50/split/split.json \
  --out_dir outputs/loo_aachen_benchmark/loo_aachen_50/plm_memory \
  --retrieval_mode file \
  --retrieval_file outputs/loo_aachen_benchmark/loo_aachen_50/retrieval/pairs-loo-netvlad50.txt \
  --override matching.pairwise_verifier.enabled=false
```

Generate real SuperGlue matches for the PLM verifier:

```bash
mkdir -p outputs/loo_aachen_benchmark/loo_aachen_50/plm_sg_verifier
cp outputs/loo_aachen_benchmark/loo_aachen_50/split/split.json \
   outputs/loo_aachen_benchmark/loo_aachen_50/plm_sg_verifier/split.json

python tools/generate_loo_superglue_matches.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --loo_dir outputs/loo_aachen_benchmark/loo_aachen_50/plm_sg_verifier \
  --retrieval_file outputs/loo_aachen_benchmark/loo_aachen_50/retrieval/pairs-loo-netvlad50.txt \
  --topk_db_images 5
```

Run PLM + SG verifier:

```bash
python tools/run_db_leave_one_out.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --split_json outputs/loo_aachen_benchmark/loo_aachen_50/split/split.json \
  --out_dir outputs/loo_aachen_benchmark/loo_aachen_50/plm_sg_verifier \
  --retrieval_mode file \
  --retrieval_file outputs/loo_aachen_benchmark/loo_aachen_50/retrieval/pairs-loo-netvlad50.txt
```

Summarize:

```bash
python tools/summarize_loo_table.py \
  --title LOO-Aachen-50 \
  --out outputs/loo_aachen_benchmark/loo_aachen_50/summary.md \
  "HLoc SP+SG=outputs/loo_aachen_benchmark/loo_aachen_50/hloc_sp_sg/metrics.json" \
  "FuseLoc=outputs/loo_aachen_benchmark/loo_aachen_50/fuseloc/metrics.json" \
  "PLM-Direct=outputs/loo_aachen_benchmark/loo_aachen_50/plm_direct/metrics.json" \
  "PLM-Memory=outputs/loo_aachen_benchmark/loo_aachen_50/plm_memory/metrics.json" \
  "PLM + SG verifier=outputs/loo_aachen_benchmark/loo_aachen_50/plm_sg_verifier/metrics.json"
```

## FuseLoc Dependencies

The runner imports the official FuseLoc code. It will fail explicitly if the
FuseLoc environment is not available. At minimum, the official environment needs
`faiss`, `pykdtree`, `poselib`, `pycolmap`, HLoc, and the global descriptor
dependencies/weights required by the selected FuseLoc descriptor pair.
