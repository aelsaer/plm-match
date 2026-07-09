#!/usr/bin/env bash
set -euo pipefail

SEQ="${1:-rgbd_dataset_freiburg1_desk}"
ROOT="${ROOT:-datasets/tum_rgbd/sequences/${SEQ}}"
CONFIG="${CONFIG:-configs/tum_online_plm_fineobs.yaml}"
MODE_TAG="${INTEGRATE_POSE_SOURCE:-pred}"
OUT_DIR="${OUT_DIR:-outputs/tum_rgbd/${SEQ}_plm_fineobs_${MODE_TAG}}"
TMP_CONFIG="$(mktemp /tmp/tum_plm_fineobs_XXXXXX.yaml)"
trap 'rm -f "$TMP_CONFIG"' EXIT

python - "$CONFIG" "$TMP_CONFIG" <<'PY'
import sys
import yaml

src, dst = sys.argv[1:3]
with open(src, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

import os

def env_int(name):
    val = os.environ.get(name)
    return None if val in (None, '') else int(val)

def env_str(name):
    val = os.environ.get(name)
    return None if val in (None, '') else str(val)

def env_bool(name):
    val = os.environ.get(name)
    if val in (None, ''):
        return None
    return str(val).lower() not in ('0', 'false', 'no', 'off')

seq = cfg.setdefault('sequence_dataset', {})
online = cfg.setdefault('online', {})
anchors = cfg.setdefault('anchors', {})
matching = cfg.setdefault('matching', {})
backbone = cfg.setdefault('backbone', {})
feature_cache = backbone.setdefault('feature_cache', {})
local_memory = matching.setdefault('local_memory', {})
vps = local_memory.setdefault('vps', {})

if (v := env_int('MAX_FRAMES')) is not None:
    seq['max_frames'] = v
if (v := env_int('STRIDE')) is not None:
    seq['stride'] = v
if (v := env_int('BOOTSTRAP')) is not None:
    online['bootstrap_count'] = v
if (v := env_bool('INTEGRATE_ON_SUCCESS')) is not None:
    online['integrate_on_success'] = v
if (v := env_str('INTEGRATE_POSE_SOURCE')) is not None:
    online['integrate_pose_source'] = v
if (v := env_int('MAX_AGE_FRAMES')) is not None:
    online['max_age_frames'] = v
if (v := env_int('MAX_ACTIVE_LANDMARKS')) is not None:
    online['max_active_landmarks'] = v
if (v := env_int('ANCHORS')) is not None:
    anchors['topk'] = v
if (v := env_int('KEYPOINT_MAX')) is not None:
    anchors['keypoint_max_corners'] = v
if (v := env_int('MAX_MATCHES')) is not None:
    matching['max_matches'] = v
if (v := env_int('LM_TOPK_OBS')) is not None:
    local_memory['topk_observations_per_anchor'] = v
if (v := env_bool('FEATURE_CACHE')) is not None:
    feature_cache['enabled'] = v
if (v := env_str('FEATURE_CACHE_DIR')) is not None:
    feature_cache['cache_dir'] = v
if (v := env_str('FEATURE_CACHE_DTYPE')) is not None:
    feature_cache['dtype'] = v
if (v := env_bool('FEATURE_CACHE_COMPRESS')) is not None:
    feature_cache['compress'] = v
if os.environ.get('LM_VPS', '0') not in ('', '0', 'false', 'False'):
    vps['enabled'] = True
if (v := env_int('LM_VPS_WORDS')) is not None:
    vps['num_words'] = v
if (v := env_int('LM_VPS_TOP_WORDS')) is not None:
    vps['top_words'] = v
if (v := env_int('LM_VPS_TARGET')) is not None:
    vps['target_correspondences'] = v
if (v := env_int('LM_VPS_TOPK_OBS')) is not None:
    vps['topk_observations_per_anchor'] = v
if (v := env_int('LM_VPS_MAX_BUCKET')) is not None:
    vps['max_bucket_size'] = v
if (v := env_str('DEVICE')) is not None:
    backbone['device'] = v
if (v := env_str('BACKBONE_INPUT_SIZE')) is not None:
    if 'x' in v.lower():
        parts = [int(x) for x in v.lower().split('x')]
    else:
        parts = [int(x) for x in v.split(',')]
    backbone['input_size'] = parts if len(parts) == 2 else [parts[0], parts[0]]

with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

python -m plm_match.pipelines.online_benchmark \
  --config "$TMP_CONFIG" \
  --dataset_root "$ROOT" \
  --out_dir "$OUT_DIR"
